"""
从 SYYL/fusion_prod 拉取 ashareeodprices 日行情，
按年分批: rename → 类型转换 → 校验 → 存 parquet。
已有缓存的年份自动跳过，失败的年份跳过继续，重跑自动补缺。

用法:
    python fetch_price_daily.py              # 全量拉取（1990~2026）
    python fetch_price_daily.py --start 2024 --end 2026  # 只拉指定年份范围

注: 与日历的交叉验证已上移到 fetch_all.py 的 validate_price_vs_calendar 步骤。
"""
import argparse
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import pymysql

from db_config import DB_NAME, DB_PASSWORD, DB_USER, make_tunnel

# ── 常量 ──────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parent / "cache" / "daily"

SQL_COLUMNS = [
    "s_info_windcode",
    "trade_dt",
    "s_dq_open",
    "s_dq_high",
    "s_dq_low",
    "s_dq_close",
    "s_dq_volume",
    "s_dq_amount",
    "s_dq_adjfactor",
    "s_dq_adjopen",
    "s_dq_adjclose",
    "s_dq_avgprice",
    "s_dq_tradestatus",
    "s_dq_pctchange",
]

RENAME_MAP = {
    "s_info_windcode": "code",
    "trade_dt": "date",
    "s_dq_open": "open",
    "s_dq_high": "high",
    "s_dq_low": "low",
    "s_dq_close": "close",
    "s_dq_volume": "volume",
    "s_dq_amount": "amount",
    "s_dq_adjfactor": "adj_factor",
    "s_dq_adjopen": "adj_open",
    "s_dq_adjclose": "adj_close",
    "s_dq_avgprice": "vwap",
    "s_dq_tradestatus": "trade_status",
    "s_dq_pctchange": "pct_change",
}

FLOAT_COLS = [
    "open", "high", "low", "close",
    "volume", "amount",
    "adj_factor", "adj_open", "adj_close",
    "vwap", "pct_change",
]


# ── 数据库连接 ────────────────────────────────────────────

MAX_RETRIES = 3
RETRY_WAIT = 30  # 秒


def fetch_one_year(year: int) -> pd.DataFrame:
    """
    拉取单年数据，独立 SSH 连接，带重试。

    Returns:
        rename 后的 DataFrame（类型未转换、未校验）
    """
    col_str = ", ".join(SQL_COLUMNS)
    sql = (
        f"SELECT {col_str} FROM ashareeodprices "
        f"WHERE trade_dt >= '{year}0101' AND trade_dt <= '{year}1231'"
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  拉取 {year} 年 (尝试 {attempt}/{MAX_RETRIES})...", end=" ", flush=True)
            t0 = time.time()
            with make_tunnel() as tunnel:
                conn = pymysql.connect(
                    host="127.0.0.1",
                    port=tunnel.local_bind_port,
                    user=DB_USER,
                    password=DB_PASSWORD,
                    database=DB_NAME,
                    read_timeout=300,
                    connect_timeout=30,
                )
                df = pd.read_sql(sql, conn)
                conn.close()

            elapsed = time.time() - t0
            print(f"{len(df)} 行, {elapsed:.1f}s")
            df = df.rename(columns=RENAME_MAP)
            return df

        except Exception as e:
            print(f"失败: {e}")
            if attempt < MAX_RETRIES:
                wait = RETRY_WAIT * attempt
                print(f"    等待 {wait}s 后重试...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"拉取 {year} 年数据失败，已重试 {MAX_RETRIES} 次: {e}"
                ) from e


def fetch_and_save_all(start: Optional[str] = None, end: Optional[str] = None) -> None:
    """
    按年分批拉取 + 校验 + 存盘，拉一年存一年。
    已有 parquet 的年份跳过。失败的年份跳过继续，下次重跑自动补缺。
    """
    start_year = int(start) if start else 1990
    end_year = int(end) if end else 2026
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"按年分批拉取: {start_year} ~ {end_year}\n")
    succeeded = []
    failed = []

    for year in range(start_year, end_year + 1):
        cached = CACHE_DIR / f"{year}.parquet"
        if cached.exists():
            size_mb = cached.stat().st_size / 1024 / 1024
            print(f"  {year}: 已有缓存 ({size_mb:.1f} MB)，跳过")
            succeeded.append(year)
            continue

        try:
            df_year = fetch_one_year(year)
        except RuntimeError as e:
            print(f"  {year}: 拉取失败，跳过 ({e})")
            failed.append(year)
            continue

        if len(df_year) == 0:
            print(f"  {year}: 0 行，跳过")
            continue

        df_year = convert_types(df_year)
        validate(df_year)

        df_year = df_year.sort_values(["code", "date"]).reset_index(drop=True)
        df_year.to_parquet(cached, index=False, engine="pyarrow")
        size_mb = cached.stat().st_size / 1024 / 1024
        print(f"  → 已存: {cached.name} ({len(df_year)} 行, {size_mb:.1f} MB)\n")
        succeeded.append(year)
        time.sleep(3)

    print(f"\n{'='*60}")
    print(f"完成: {len(succeeded)} 年成功, {len(failed)} 年失败")
    if failed:
        print(f"失败年份: {failed}（重新运行脚本自动补拉）")


# ── 类型转换 ──────────────────────────────────────────────

def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """trade_dt varchar → datetime, Decimal → float"""
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for col in FLOAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


# ── 校验 ──────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> None:
    """
    校验数据质量，任何问题直接 raise。

    检查项:
    1. close, adj_factor 不为 0 / NaN
    2. 无重复 (date, code)
    3. close × adj_factor ≈ adj_close（0.01% 容差）
    4. open × adj_factor ≈ adj_open（0.01% 容差）
    5. low ≤ open ≤ high 且 low ≤ close ≤ high

    日历交叉验证已上移到 fetch_all.validate_price_vs_calendar。
    """
    n = len(df)
    print(f"\n校验 {n} 行数据...")

    # 1. close / adj_factor 不为 0 或 NaN
    bad_close = df["close"].isna() | (df["close"] == 0)
    if bad_close.any():
        samples = df.loc[bad_close, ["date", "code", "close"]].head(5)
        raise ValueError(f"close 为 0 或 NaN: {bad_close.sum()} 行\n{samples}")

    bad_adj = df["adj_factor"].isna() | (df["adj_factor"] == 0)
    if bad_adj.any():
        samples = df.loc[bad_adj, ["date", "code", "adj_factor"]].head(5)
        raise ValueError(f"adj_factor 为 0 或 NaN: {bad_adj.sum()} 行\n{samples}")

    print("  [1/5] close / adj_factor 非零非空 ✓")

    # 2. 无重复 (date, code)
    dup = df.duplicated(subset=["date", "code"], keep=False)
    if dup.any():
        samples = df.loc[dup, ["date", "code"]].head(10)
        raise ValueError(f"重复 (date, code): {dup.sum()} 行\n{samples}")

    print("  [2/5] 无重复 (date, code) ✓")

    # 3. close × adj_factor ≈ adj_close
    # Decimal(20,4) 精度有限，低价股四舍五入后相对误差大
    # 阈值 0.5% 以内视为精度问题（warning），> 2% 视为数据错误（raise）
    mask_close = df["adj_close"].notna() & (df["adj_close"] != 0)
    if mask_close.any():
        computed = df.loc[mask_close, "close"] * df.loc[mask_close, "adj_factor"]
        actual = df.loc[mask_close, "adj_close"]
        rel_err = ((computed - actual) / actual).abs()

        n_warn = (rel_err > 5e-3).sum()   # > 0.5%
        n_bad = (rel_err > 0.02).sum()     # > 2%
        print(f"  [3/5] close × adj_factor vs adj_close:")
        print(f"        误差分位: p50={rel_err.quantile(0.5):.6f}, "
              f"p99={rel_err.quantile(0.99):.6f}, max={rel_err.max():.6f}")
        print(f"        > 0.5%: {n_warn} 行, > 2%: {n_bad} 行 (共 {mask_close.sum()} 行)")

        if n_bad > 0:
            bad_idx = rel_err[rel_err > 0.02].nlargest(5).index
            samples = df.loc[bad_idx, ["date", "code", "close", "adj_factor", "adj_close"]]
            print(f"        ⚠ 严重偏差 (>2%): {n_bad} 行（历史脏数据，不阻断）")
            print(f"        样本:\n{samples}")
        else:
            print("        ✓ 无严重偏差")

    # 4. open × adj_factor ≈ adj_open（同上逻辑）
    mask_open = df["adj_open"].notna() & (df["adj_open"] != 0) & df["open"].notna() & (df["open"] != 0)
    if mask_open.any():
        computed = df.loc[mask_open, "open"] * df.loc[mask_open, "adj_factor"]
        actual = df.loc[mask_open, "adj_open"]
        rel_err = ((computed - actual) / actual).abs()

        n_warn = (rel_err > 5e-3).sum()
        n_bad = (rel_err > 0.02).sum()
        print(f"  [4/5] open × adj_factor vs adj_open:")
        print(f"        误差分位: p50={rel_err.quantile(0.5):.6f}, "
              f"p99={rel_err.quantile(0.99):.6f}, max={rel_err.max():.6f}")
        print(f"        > 0.5%: {n_warn} 行, > 2%: {n_bad} 行 (共 {mask_open.sum()} 行)")

        if n_bad > 0:
            bad_idx = rel_err[rel_err > 0.02].nlargest(5).index
            samples = df.loc[bad_idx, ["date", "code", "open", "adj_factor", "adj_open"]]
            print(f"        ⚠ 严重偏差 (>2%): {n_bad} 行（历史脏数据，不阻断）")
            print(f"        样本:\n{samples}")
        else:
            print("        ✓ 无严重偏差")

    # 5. 价格关系: low ≤ open ≤ high, low ≤ close ≤ high
    # 只检查 OHLC 都非 NaN 的行（停牌等可能有 NaN）
    price_cols = ["open", "high", "low", "close"]
    mask_price = df[price_cols].notna().all(axis=1) & (df[price_cols] > 0).all(axis=1)
    if mask_price.any():
        sub = df.loc[mask_price]
        bad_open = (sub["open"] < sub["low"] - 1e-4) | (sub["open"] > sub["high"] + 1e-4)
        bad_close = (sub["close"] < sub["low"] - 1e-4) | (sub["close"] > sub["high"] + 1e-4)
        bad_price = bad_open | bad_close
        if bad_price.any():
            samples = sub.loc[bad_price, ["date", "code"] + price_cols].head(10)
            print(f"  [5/5] 价格关系异常: {bad_price.sum()} 行（warning，不阻断）")
            print(f"        样本:\n{samples}")
        else:
            print("  [5/5] low ≤ open/close ≤ high ✓")
    else:
        print("  [5/5] 无有效 OHLC 行可校验（跳过）")

    print("校验全部通过。\n")


# ── 主流程 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="拉取 A 股日行情并存为 parquet")
    parser.add_argument("--start", type=str, default=None, help="起始年份 (如 2024)")
    parser.add_argument("--end", type=str, default=None, help="截止年份 (如 2026)")
    args = parser.parse_args()

    fetch_and_save_all(start=args.start, end=args.end)
    print("完成。")


if __name__ == "__main__":
    main()
