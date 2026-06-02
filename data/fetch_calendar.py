"""
从 SYYL/fusion_prod 拉取 asharecalendar 上交所交易日历，
rename → 类型转换 → 校验 → 存 parquet。
已有缓存自动跳过，--force 强制重拉。

用法:
    python fetch_calendar.py            # 已有缓存则跳过
    python fetch_calendar.py --force    # 强制重拉
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import pymysql

from db_config import DB_NAME, DB_PASSWORD, DB_USER, make_tunnel

# ── 常量 ──────────────────────────────────────────────────

CACHE_PATH = Path(__file__).parent / "cache" / "calendar" / "trade_dates.parquet"

SQL = (
    "SELECT trade_days, s_info_exchmarket "
    "FROM asharecalendar "
    "WHERE s_info_exchmarket = 'SSE' "
    "ORDER BY trade_days"
)

RENAME_MAP = {
    "trade_days": "date",
    "s_info_exchmarket": "exchange",
}


# ── 数据库连接 ────────────────────────────────────────────

MAX_RETRIES = 3
RETRY_WAIT = 30  # 秒


def fetch(sql: str) -> pd.DataFrame:
    """
    执行 SQL 拉取交易日历，独立 SSH 连接，带重试。

    Returns:
        rename 后的 DataFrame（类型未转换、未校验）
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"拉取交易日历 (尝试 {attempt}/{MAX_RETRIES})...", end=" ", flush=True)
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
                    f"拉取交易日历失败，已重试 {MAX_RETRIES} 次: {e}"
                ) from e


def fetch_and_save(force: bool = False) -> None:
    """
    拉取交易日历 + 类型转换 + 校验 + 存盘。
    已有 parquet 缓存时跳过；force=True 强制重拉。
    """
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if CACHE_PATH.exists() and not force:
        size_kb = CACHE_PATH.stat().st_size / 1024
        print(f"已有缓存 ({size_kb:.1f} KB)，跳过。使用 --force 强制重拉。")
        return

    df = fetch(SQL)

    if len(df) == 0:
        print("0 行，跳过")
        return

    df = convert_types(df)
    validate(df)

    df = df.sort_values("date").reset_index(drop=True)
    df.to_parquet(CACHE_PATH, index=False, engine="pyarrow")
    size_kb = CACHE_PATH.stat().st_size / 1024
    print(f"  → 已存: {CACHE_PATH.name} ({len(df)} 行, {size_kb:.1f} KB)\n")


# ── 类型转换 ──────────────────────────────────────────────

def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """trade_days varchar(8) "YYYYMMDD" → date datetime"""
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df


# ── 校验 ──────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> None:
    """
    校验数据质量，任何问题直接 raise。

    检查项:
    1. date 不为 NaN
    2. 无重复日期
    3. 日期严格升序
    4. 不含周末
    5. 每年交易日数量在 [235, 260]（首尾不完整年份自动跳过）
       范围覆盖 1991-1995 早期少节假日（最高 257）与 1999/2000/2002/2013
       春节长假年份（最低 237）的真实历史，作 sanity 量级检查用
    """
    n = len(df)
    print(f"\n校验 {n} 行数据...")

    # 1. date 不为 NaN
    bad = df["date"].isna()
    if bad.any():
        raise ValueError(f"date 含 NaN: {bad.sum()} 行")
    print("  [1/5] date 无 NaN ✓")

    # 2. 无重复日期
    dup = df["date"].duplicated(keep=False)
    if dup.any():
        samples = df.loc[dup, "date"].head(10).tolist()
        raise ValueError(f"日期重复: {dup.sum()} 行\n样本: {samples}")
    print("  [2/5] 无重复日期 ✓")

    # 3. 日期严格升序（步骤 2 已排重，单调递增即严格升序）
    if not df["date"].is_monotonic_increasing:
        diffs = df["date"].diff()
        violations = df[diffs < pd.Timedelta(0)]
        raise ValueError(f"日期非升序，前 5 个违反:\n{violations.head(5)}")
    print("  [3/5] 日期严格升序 ✓")

    # 4. 不含周末
    weekday = df["date"].dt.weekday  # 0=Mon, ..., 5=Sat, 6=Sun
    is_weekend = weekday >= 5
    if is_weekend.any():
        samples = df.loc[is_weekend, "date"].head(10).tolist()
        raise ValueError(f"含周末日期: {is_weekend.sum()} 行\n样本: {samples}")
    print("  [4/5] 无周末日期 ✓")

    # 5. 每年交易日数量在 [240, 250]，首尾不完整年份跳过
    yearly = df.groupby(df["date"].dt.year).size().sort_index()
    min_date, max_date = df["date"].min(), df["date"].max()
    first_year, last_year = min_date.year, max_date.year
    skip_years = set()
    if min_date > pd.Timestamp(f"{first_year}-01-15"):
        skip_years.add(first_year)
    if max_date < pd.Timestamp(f"{last_year}-12-15"):
        skip_years.add(last_year)
    full_years = yearly.drop(index=list(skip_years), errors="ignore")
    bad_years = full_years[(full_years < 235) | (full_years > 260)]

    print(f"  [5/5] 年度交易日: 共 {len(yearly)} 年, "
          f"完整 {len(full_years)} 年, 跳过 {sorted(skip_years) or '无'}")
    print(f"        完整年份范围: min={full_years.min()}, max={full_years.max()}")
    if len(bad_years) > 0:
        raise ValueError(
            f"以下年份交易日数量不在 [235, 260]:\n{bad_years.to_string()}"
        )
    print("        ✓ 完整年份全部在 [235, 260]")

    print("校验全部通过。\n")


# ── 主流程 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="拉取上交所交易日历并存为 parquet")
    parser.add_argument("--force", action="store_true", help="强制重拉（忽略已有缓存）")
    args = parser.parse_args()

    fetch_and_save(force=args.force)
    print("完成。")


if __name__ == "__main__":
    main()
