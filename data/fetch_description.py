"""
从 SYYL/fusion_prod 拉取 asharedescription 股票基本信息，
还原为「上市/退市日表」: [code, list_date, delist_date]，每只股票一行。

用途：现金引擎判退市用真实退市日（替代"窗口内最后有效成交日"的前视判定）。
delist_date 为 NaT = 截至数据日仍未退市。

数据源:
  asharedescription — 股票描述表（s_info_windcode / s_info_listdate / s_info_delistdate）

用法:
    python fetch_description.py          # 全量拉取 → cache/description/description.parquet
"""
import time
from pathlib import Path

import pandas as pd
import pymysql

from db_config import DB_NAME, DB_PASSWORD, DB_USER, make_tunnel

# ── 常量 ──────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parent / "cache" / "description"
CACHE_PATH = CACHE_DIR / "description.parquet"

SQL = (
    "SELECT s_info_windcode, s_info_listdate, s_info_delistdate "
    "FROM asharedescription"
)

RENAME_MAP = {
    "s_info_windcode": "code",
    "s_info_listdate": "list_date",
    "s_info_delistdate": "delist_date",
}

MAX_RETRIES = 3
RETRY_WAIT = 30  # 秒


# ── 拉取 ──────────────────────────────────────────────────

def fetch() -> pd.DataFrame:
    """连 fusion_prod（SSH 隧道）拉 asharedescription，带重试。返回 rename 后的 DataFrame。"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  拉取 asharedescription (尝试 {attempt}/{MAX_RETRIES})...", end=" ", flush=True)
            t0 = time.time()
            with make_tunnel() as tunnel:
                conn = pymysql.connect(
                    host="127.0.0.1", port=tunnel.local_bind_port,
                    user=DB_USER, password=DB_PASSWORD, database=DB_NAME,
                    read_timeout=120, connect_timeout=30,
                )
                df = pd.read_sql(SQL, conn)
                conn.close()
            print(f"{len(df)} 行, {time.time() - t0:.1f}s")
            return df.rename(columns=RENAME_MAP)
        except Exception as e:
            print(f"失败: {e}")
            if attempt < MAX_RETRIES:
                wait = RETRY_WAIT * attempt
                print(f"    等待 {wait}s 后重试...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"拉取 asharedescription 失败，已重试 {MAX_RETRIES} 次: {e}") from e


# ── 转换 ──────────────────────────────────────────────────

def _to_date(s: pd.Series) -> pd.Series:
    """varchar8 'YYYYMMDD' → datetime；空串 / '0' / '00000000' / NaN → NaT。"""
    s = s.astype("string").str.strip()
    s = s.where(~s.isin(["", "0", "00000000"]), other=pd.NA)
    return pd.to_datetime(s, format="%Y%m%d", errors="coerce")


def transform(raw: pd.DataFrame) -> pd.DataFrame:
    """rename 后的原始行 → [code, list_date, delist_date]，每 code 一行。"""
    df = raw.copy()
    df["list_date"] = _to_date(df["list_date"])
    df["delist_date"] = _to_date(df["delist_date"])
    df = df[["code", "list_date", "delist_date"]]
    return df.sort_values("code").reset_index(drop=True)


# ── 校验 ──────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> None:
    """质量校验，任何问题直接 raise（带样例，可定位）。"""
    n = len(df)
    print(f"\n校验 {n} 行 description...")

    bad = df["code"].isna()
    if bad.any():
        raise ValueError(f"code 含 NaN: {int(bad.sum())} 行")
    if df["code"].duplicated().any():
        dup = df.loc[df["code"].duplicated(keep=False), "code"].unique()[:10]
        raise ValueError(f"code 重复（asharedescription 应每股一行）: {list(dup)}")
    print(f"  [1/2] code 非空且唯一（{df['code'].nunique()} 只）✓")

    both = df["list_date"].notna() & df["delist_date"].notna()
    if both.any():
        bad = df.loc[both, "delist_date"] <= df.loc[both, "list_date"]
        if bad.any():
            samples = df.loc[both].loc[bad].head(10)
            raise ValueError(f"delist_date <= list_date: {int(bad.sum())} 行\n{samples}")
    n_delisted = int(df["delist_date"].notna().sum())
    print(f"  [2/2] delist_date > list_date（{n_delisted} 只已退市, {n - n_delisted} 只在市）✓")
    print("校验全部通过。\n")


# ── 主流程 ────────────────────────────────────────────────

def fetch_and_save() -> None:
    """全量拉取 + 转换 + 校验 + 存盘。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw = fetch()
    if len(raw) == 0:
        raise RuntimeError("asharedescription 拉取结果为空")
    df = transform(raw)
    validate(df)
    df.to_parquet(CACHE_PATH, index=False, engine="pyarrow")
    size_kb = CACHE_PATH.stat().st_size / 1024
    print(f"→ 已存: {CACHE_PATH.name}（{len(df)} 只, {size_kb:.1f} KB）")


def main():
    fetch_and_save()
    print("完成。")


if __name__ == "__main__":
    main()
