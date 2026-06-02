"""
从 SYYL/fusion_prod 拉取 ashareeodderivativeindicator 日衍生指标，
按年分批: rename → 类型转换 → 校验 → 存 parquet。
已有缓存的年份自动跳过，失败的年份跳过继续，重跑自动补缺。

字段重点:
- up_down_limit_status (limit_status):    -1=跌停, 0=正常, 1=涨停（Wind 已按板块规则消化 ST/主板/科创/创业/北交所）
- lowest_highest_status (extremum_status): -1=触当日最低, 0=普通, 1=触当日最高
- s_dq_turn / s_dq_freeturnover:           换手率 / 自由流通换手率
- s_val_mv / s_dq_mv:                      总市值 / 流通市值

用法:
    python fetch_derivative.py                          # 全量拉取（1990~2026）
    python fetch_derivative.py --start 2024 --end 2026  # 只拉指定年份范围

注: 与日历的交叉验证已上移到 fetch_all.py 的 validate_derivative_vs_calendar 步骤。
"""
import argparse
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import pymysql

from db_config import DB_NAME, DB_PASSWORD, DB_USER, make_tunnel

# ── 常量 ──────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parent / "cache" / "derivative"

SQL_COLUMNS = [
    "s_info_windcode",
    "trade_dt",
    "up_down_limit_status",
    "lowest_highest_status",
    "s_dq_turn",
    "s_dq_freeturnover",
    "s_val_mv",
    "s_dq_mv",
]

RENAME_MAP = {
    "s_info_windcode": "code",
    "trade_dt": "date",
    "up_down_limit_status": "limit_status",
    "lowest_highest_status": "extremum_status",
    "s_dq_turn": "turnover",
    "s_dq_freeturnover": "free_turnover",
    "s_val_mv": "total_mv",
    "s_dq_mv": "float_mv",
}

# decimal(20,4) → float64
FLOAT_COLS = ["turnover", "free_turnover", "total_mv", "float_mv"]
# decimal(2,0) → Int8 (nullable，保留 NaN)
INT_COLS = ["limit_status", "extremum_status"]


# ── 数据库连接 ────────────────────────────────────────────

MAX_RETRIES = 3
RETRY_WAIT = 30  # 秒


QUARTERS = [("0101", "0331"), ("0401", "0630"), ("0701", "0930"), ("1001", "1231")]


def _fetch_with_retry(sql: str, label: str) -> pd.DataFrame:
    """单次 SQL 拉取 + 重试（内部 helper，每次独立 SSH 连接）"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"    {label} (尝试 {attempt}/{MAX_RETRIES})...", end=" ", flush=True)
            t0 = time.time()
            with make_tunnel() as tunnel:
                conn = pymysql.connect(
                    host="127.0.0.1",
                    port=tunnel.local_bind_port,
                    user=DB_USER,
                    password=DB_PASSWORD,
                    database=DB_NAME,
                    read_timeout=600,
                    connect_timeout=30,
                )
                df = pd.read_sql(sql, conn)
                conn.close()
            elapsed = time.time() - t0
            print(f"{len(df)} 行, {elapsed:.1f}s")
            return df
        except Exception as e:
            print(f"失败: {e}")
            if attempt < MAX_RETRIES:
                wait = RETRY_WAIT * attempt
                print(f"      等待 {wait}s 后重试...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"拉取 {label} 失败，已重试 {MAX_RETRIES} 次: {e}"
                ) from e


def fetch_one_year(year: int) -> pd.DataFrame:
    """
    拉取单年衍生指标。该表单年 ~130 万行，单次查询会被服务器 ~5 min 后断连，
    所以内部按季度分 4 段拉，每段独立 SSH 连接 + 3 次重试，单年聚合后返回。

    Returns:
        rename 后的 DataFrame（类型未转换、未校验）
    """
    col_str = ", ".join(SQL_COLUMNS)
    print(f"  拉取 {year} 年（季度分批）...")
    parts = []
    for q_start, q_end in QUARTERS:
        sql = (
            f"SELECT {col_str} FROM ashareeodderivativeindicator "
            f"WHERE trade_dt >= '{year}{q_start}' AND trade_dt <= '{year}{q_end}'"
        )
        df_q = _fetch_with_retry(sql, f"{year}-{q_start[:2]}~{q_end[:2]}")
        parts.append(df_q)
        time.sleep(2)  # 季度间间隔，给服务器一点缓冲
    df = pd.concat(parts, ignore_index=True)
    df = df.rename(columns=RENAME_MAP)
    return df


def fetch_and_save_all(start: Optional[str] = None, end: Optional[str] = None) -> None:
    """
    按年分批拉取 + 校验 + 存盘，拉一年存一年。
    已有 parquet 的年份跳过。失败的年份跳过继续，下次重跑自动补缺。
    """
    start_year = int(start) if start else 1990
    end_year = int(end) if end else 2026
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"按年分批拉取衍生指标: {start_year} ~ {end_year}\n")
    succeeded, failed = [], []

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
    """trade_dt varchar → datetime, decimal(20,4) → float64, decimal(2,0) → Int8 (nullable)"""
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for col in FLOAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in INT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int8")
    return df


# ── 校验 ──────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> None:
    """
    校验数据质量，任何问题直接 raise。

    检查项:
    1. 无重复 (date, code)
    2. limit_status / extremum_status ∈ {-1, 0, 1} 或 NaN
    3. turnover / free_turnover / total_mv / float_mv 非负（NaN 允许）

    日历交叉验证已上移到 fetch_all.validate_price_vs_calendar。
    """
    n = len(df)
    print(f"\n校验 {n} 行数据...")

    # 1. 无重复 (date, code)
    dup = df.duplicated(subset=["date", "code"], keep=False)
    if dup.any():
        samples = df.loc[dup, ["date", "code"]].head(10)
        raise ValueError(f"重复 (date, code): {dup.sum()} 行\n{samples}")
    print("  [1/3] 无重复 (date, code) ✓")

    # 2. status 字段取值合法
    for col in INT_COLS:
        bad = df[col].notna() & ~df[col].isin([-1, 0, 1])
        if bad.any():
            bad_vals = df.loc[bad, col].unique()
            raise ValueError(f"{col} 含非法值 (合法: -1/0/1/NaN): {list(bad_vals)[:10]}")
        n_na = df[col].isna().sum()
        print(f"  [2/3] {col} ∈ {{-1, 0, 1}} (NaN={n_na} 行) ✓")

    # 3. 数值列非负（NaN 允许）
    for col in FLOAT_COLS:
        bad = df[col].notna() & (df[col] < 0)
        if bad.any():
            samples = df.loc[bad, ["date", "code", col]].head(10)
            raise ValueError(f"{col} 含负值: {bad.sum()} 行\n{samples}")
    print(f"  [3/3] turnover / free_turnover / total_mv / float_mv 非负 ✓")

    print("校验全部通过。\n")


# ── 主流程 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="拉取 A 股日衍生指标并存为 parquet")
    parser.add_argument("--start", type=str, default=None, help="起始年份 (如 2024)")
    parser.add_argument("--end", type=str, default=None, help="截止年份 (如 2026)")
    args = parser.parse_args()

    fetch_and_save_all(start=args.start, end=args.end)
    print("完成。")


if __name__ == "__main__":
    main()
