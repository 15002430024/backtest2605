"""
从 SYYL/fusion_prod 拉取 aindexmembers 指数成分股，
按指数分批: rename → 类型转换 → 校验 → 存 parquet。
已有缓存的指数自动跳过，失败的跳过继续，重跑自动补缺。

用法:
    python fetch_index_members.py                                # 拉 INDICES 全部
    python fetch_index_members.py --indices 000300.SH,000905.SH  # 只拉指定指数

注: entry_date / exit_date 的日历交叉验证已上移到 fetch_all.validate_index_dates_vs_calendar。
"""
import argparse
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pymysql

from db_config import DB_NAME, DB_PASSWORD, DB_USER, make_tunnel

# ── 常量 ──────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parent / "cache" / "index_members"

# 待拉取指数（按需增删）
INDICES = [
    "000016.SH",   # 上证50
    "000300.SH",   # 沪深300
    "000905.SH",   # 中证500
    "000906.SH",   # 中证800
    "000852.SH",   # 中证1000
    "932000.CSI",  # 中证2000（2023-08-11 起）
    "399303.SZ",   # 国证2000（2014-03-28 起）
    "399006.SZ",   # 创业板指
    "000688.SH",   # 科创50
]

# 已用 DESCRIBE aindexmembers 验证
# 输出只保留前 4 列；cur_sign / opdate 仅用于 convert_types 内部去重
SQL_COLUMNS = [
    "s_info_windcode",   # 指数代码
    "s_con_windcode",    # 成分股代码
    "s_con_indate",      # 纳入日期 varchar(8) "YYYYMMDD"
    "s_con_outdate",     # 剔除日期 varchar(8), NULL = 当前仍是成分股
    "cur_sign",          # 1=最新段（去重辅助，不进 parquet）
    "opdate",            # ETL 录入时间（去重辅助，不进 parquet）
]

RENAME_MAP = {
    "s_info_windcode": "index_code",
    "s_con_windcode": "code",
    "s_con_indate": "entry_date",
    "s_con_outdate": "exit_date",
    # cur_sign, opdate 不 rename，convert_types 用完即丢
}

DATE_COLS = ["entry_date", "exit_date"]


# ── 数据库连接 ────────────────────────────────────────────

MAX_RETRIES = 3
RETRY_WAIT = 30  # 秒


def fetch_one_index(index_code: str) -> pd.DataFrame:
    """
    拉取单个指数的成分股历史，独立 SSH 连接，带重试。

    Returns:
        rename 后的 DataFrame（类型未转换、未校验）
    """
    col_str = ", ".join(SQL_COLUMNS)
    sql = (
        f"SELECT {col_str} FROM aindexmembers "
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
                    f"拉取 {index_code} 失败，已重试 {MAX_RETRIES} 次: {e}"
                ) from e


def fetch_and_save_all(indices: Optional[List[str]] = None) -> None:
    """
    按指数分批拉取 + 校验 + 存盘，拉一个存一个。
    已有 parquet 的跳过。失败的跳过继续，下次重跑自动补缺。
    """
    target_indices = indices if indices else INDICES
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"按指数分批拉取: {len(target_indices)} 个指数\n")
    succeeded, failed = [], []

    for index_code in target_indices:
        # 文件名: 把 . 换成 _ 防止与 parquet 扩展名混淆
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

        df = df.sort_values(["index_code", "code", "entry_date"]).reset_index(drop=True)
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
    """
    entry_date / exit_date varchar(8) "YYYYMMDD" → datetime（exit_date 可空 → NaT）。
    顺带处理 Wind 数据冗余: 同 (index_code, code, entry_date) 多行时按 opdate 最新保留，
    并丢掉辅助列 cur_sign / opdate，输出只剩 4 列。
    """
    df["entry_date"] = pd.to_datetime(df["entry_date"], format="%Y%m%d")
    df["exit_date"] = pd.to_datetime(df["exit_date"], format="%Y%m%d")

    n_before = len(df)
    df = (
        df.sort_values("opdate")
          .drop_duplicates(subset=["index_code", "code", "entry_date"], keep="last")
          .drop(columns=["cur_sign", "opdate"])
          .reset_index(drop=True)
    )
    n_removed = n_before - len(df)
    if n_removed > 0:
        print(f"    Wind 数据修复: 同 (index_code, code, entry_date) 重复 {n_removed} 行，保留 opdate 最新")
    return df


# ── 校验 ──────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> None:
    """
    校验数据质量，任何问题直接 raise。

    检查项:
    1. index_code / code / entry_date 不为 NaN
    2. 无重复 (index_code, code, entry_date)
    3. exit_date 可空（仍在成分股内），若非空则 > entry_date

    entry_date / exit_date 的日历交叉验证已上移到
    fetch_all.validate_index_dates_vs_calendar。
    """
    n = len(df)
    print(f"\n校验 {n} 行数据...")

    # 1. 关键字段非空
    for col in ["index_code", "code", "entry_date"]:
        bad = df[col].isna()
        if bad.any():
            raise ValueError(f"{col} 含 NaN: {bad.sum()} 行")
    print("  [1/3] index_code / code / entry_date 非空 ✓")

    # 2. 无重复 (index_code, code, entry_date)
    dup = df.duplicated(subset=["index_code", "code", "entry_date"], keep=False)
    if dup.any():
        samples = df.loc[dup, ["index_code", "code", "entry_date"]].head(10)
        raise ValueError(f"重复 (index_code, code, entry_date): {dup.sum()} 行\n{samples}")
    print("  [2/3] 无重复 (index_code, code, entry_date) ✓")

    # 3. exit_date > entry_date（NaT 即"仍在"，跳过）
    mask_exit = df["exit_date"].notna()
    if mask_exit.any():
        sub = df.loc[mask_exit]
        bad = sub["exit_date"] <= sub["entry_date"]
        if bad.any():
            samples = sub.loc[bad, ["index_code", "code", "entry_date", "exit_date"]].head(10)
            raise ValueError(f"exit_date <= entry_date: {bad.sum()} 行\n{samples}")
    print(f"  [3/3] exit_date > entry_date ({mask_exit.sum()} 行有效 exit, "
          f"{(~mask_exit).sum()} 行仍在) ✓")

    print("校验全部通过。\n")


# ── 主流程 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="拉取 A 股指数成分股并存为 parquet")
    parser.add_argument(
        "--indices", type=str, default=None,
        help="逗号分隔的指数代码列表（如 000300.SH,000905.SH），默认拉 INDICES 全部"
    )
    args = parser.parse_args()

    indices = args.indices.split(",") if args.indices else None
    fetch_and_save_all(indices=indices)
    print("完成。")


if __name__ == "__main__":
    main()
