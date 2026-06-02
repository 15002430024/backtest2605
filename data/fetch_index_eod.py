"""
从 SYYL/fusion_prod 拉取 aindexeodprices 指数日行情（基准收益用），
按指数分批: fetch → 类型转换 → 校验 → 存 parquet。
已有缓存的指数自动跳过，失败的跳过继续，重跑自动补缺。

用法:
    python fetch_index_eod.py                                # 拉 INDICES 全部基准
    python fetch_index_eod.py --indices 000001.SH,000300.SH  # 只拉指定指数

注:
  - 指数不需要复权，基准 NAV 直接用 close，日收益用 close 或现成的 pct_change。
  - 与日历的交叉验证已上移到 fetch_all.py 的 validate_index_eod_vs_calendar 步骤。
"""
import argparse
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pymysql

from db_config import DB_NAME, DB_PASSWORD, DB_USER, make_tunnel

# ── 常量 ──────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parent / "cache" / "index_eod"

# 待拉取基准指数（按需增删）。2026-05-31 探针确认 8 个均有数据，覆盖到 2026-05。
INDICES = [
    "000001.SH",   # 上证综指
    "399001.SZ",   # 深证成指
    "000016.SH",   # 上证50
    "000300.SH",   # 沪深300
    "000905.SH",   # 中证500
    "000852.SH",   # 中证1000
    "399006.SZ",   # 创业板指
    "000688.SH",   # 科创50
]

# 已用 DESCRIBE aindexeodprices 验证（无复权因子列）
SQL_COLUMNS = [
    "s_info_windcode",   # 指数代码
    "trade_dt",          # 交易日 varchar(8) "YYYYMMDD"
    "s_dq_preclose",     # 昨收
    "s_dq_open",
    "s_dq_high",
    "s_dq_low",
    "s_dq_close",
    "s_dq_change",       # 涨跌点数
    "s_dq_pctchange",    # 涨跌幅(%)
    "s_dq_volume",
    "s_dq_amount",
]

RENAME_MAP = {
    "s_info_windcode": "code",
    "trade_dt": "date",
    "s_dq_preclose": "pre_close",
    "s_dq_open": "open",
    "s_dq_high": "high",
    "s_dq_low": "low",
    "s_dq_close": "close",
    "s_dq_change": "change",
    "s_dq_pctchange": "pct_change",
    "s_dq_volume": "volume",
    "s_dq_amount": "amount",
}

FLOAT_COLS = [
    "pre_close", "open", "high", "low", "close",
    "change", "pct_change", "volume", "amount",
]


# ── 数据库连接 ────────────────────────────────────────────

MAX_RETRIES = 3
RETRY_WAIT = 30  # 秒


def fetch_one_index(index_code: str) -> pd.DataFrame:
    """
    拉取单个指数的全历史日行情，独立 SSH 连接，带重试。

    Returns:
        rename 后的 DataFrame（类型未转换、未校验）
    """
    col_str = ", ".join(SQL_COLUMNS)
    sql = (
        f"SELECT {col_str} FROM aindexeodprices "
        f"WHERE s_info_windcode = '{index_code}'"
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  拉取 {index_code} (尝试 {attempt}/{MAX_RETRIES})...", end=" ", flush=True)
            t0 = time.time()
            with make_tunnel() as tunnel:
                conn = pymysql.connect(
                    host="127.0.0.1",
                    port=tunnel.local_bind_port,
                    user=DB_USER,
                    password=DB_PASSWORD,
                    database=DB_NAME,
                    read_timeout=120,
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
                    f"拉取 {index_code} 指数行情失败，已重试 {MAX_RETRIES} 次: {e}"
                ) from e


def fetch_and_save_all(indices: Optional[List[str]] = None) -> None:
    """
    按指数分批拉取 + 校验 + 存盘，拉一个存一个。
    已有 parquet 的跳过。失败的跳过继续，下次重跑自动补缺。
    """
    target_indices = indices if indices else INDICES
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"按指数分批拉取基准行情: {len(target_indices)} 个指数\n")
    succeeded = []
    failed = []

    for index_code in target_indices:
        # 文件名: 把 . 换成 _ 防止与 parquet 扩展名混淆（与 index_members 一致）
        cached = CACHE_DIR / f"{index_code.replace('.', '_')}.parquet"
        if cached.exists():
            size_mb = cached.stat().st_size / 1024 / 1024
            print(f"  {index_code}: 已有缓存 ({size_mb:.1f} MB)，跳过")
            succeeded.append(index_code)
            continue

        try:
            df = fetch_one_index(index_code)
        except RuntimeError as e:
            print(f"  {index_code}: 拉取失败，跳过 ({e})")
            failed.append(index_code)
            continue

        if len(df) == 0:
            print(f"  {index_code}: 0 行，跳过")
            continue

        df = convert_types(df)
        validate(df)

        df = df.sort_values("date").reset_index(drop=True)
        df.to_parquet(cached, index=False, engine="pyarrow")
        size_mb = cached.stat().st_size / 1024 / 1024
        print(f"  → 已存: {cached.name} ({len(df)} 行, {size_mb:.1f} MB)\n")
        succeeded.append(index_code)
        time.sleep(3)

    print(f"\n{'='*60}")
    print(f"完成: {len(succeeded)} 成功, {len(failed)} 失败")
    if failed:
        print(f"失败指数: {failed}（重新运行脚本自动补拉）")


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
    1. code / date / close 不为 NaN
    2. 无重复 (code, date)
    3. close 为正

    日期与日历的交叉验证已上移到 fetch_all.validate_index_eod_vs_calendar。
    """
    n = len(df)
    code = df["code"].iloc[0]
    print(f"\n校验 {code} {n} 行数据...")

    # 1. code / date / close 非空
    for col in ["code", "date", "close"]:
        bad = df[col].isna()
        if bad.any():
            raise ValueError(f"{code} 列 {col} 含 NaN: {bad.sum()} 行")
    print("  [1/3] code / date / close 非空 ✓")

    # 2. 无重复 (code, date)
    dup = df.duplicated(subset=["code", "date"], keep=False)
    if dup.any():
        samples = df.loc[dup, ["code", "date"]].head(10)
        raise ValueError(f"{code} 重复 (code, date): {dup.sum()} 行\n{samples}")
    print("  [2/3] 无重复 (code, date) ✓")

    # 3. close 为正
    bad_close = df["close"] <= 0
    if bad_close.any():
        samples = df.loc[bad_close, ["date", "code", "close"]].head(5)
        raise ValueError(f"{code} close <= 0: {bad_close.sum()} 行\n{samples}")
    print("  [3/3] close > 0 ✓")

    print("校验全部通过。\n")


# ── 主流程 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="拉取基准指数日行情并存为 parquet")
    parser.add_argument(
        "--indices", type=str, default=None,
        help="逗号分隔的指数代码列表（如 000001.SH,000300.SH），默认拉 INDICES 全部"
    )
    args = parser.parse_args()

    indices = args.indices.split(",") if args.indices else None
    fetch_and_save_all(indices=indices)
    print("完成。")


if __name__ == "__main__":
    main()
