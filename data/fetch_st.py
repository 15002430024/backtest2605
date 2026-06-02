"""
从 JYDB（聚源，内网 SQL Server）拉取 A 股 ST（特别处理）历史，
还原为「ST 区间表」: [code, st_start, st_end]，每段连续 ST 一行。
时点口径、无前视：某交易日是否 ST 由"该日落在哪段 ST 区间内"决定。

数据源:
  LC_SpecialTrade  — 证券特别处理事件表（每次带帽/摘帽一行，1998 起，覆盖 ST 全历史）
  SecuMain         — InnerCode → SecuCode + SecuMarket（含已退市，用于拼 windcode）

判 ST 方法: 事件行的 SecurityAbbr 含 "ST" 即该事件起进入 ST 状态（与 barra st_ipo_data 口径一致），
            不解码 SpecialTradeType 的 15 种数字。

用法:
    python fetch_st.py          # 全量拉取 → cache/st/st_intervals.parquet

注: ST 制度 1998-04 才开始，本表即覆盖全历史，无 2016 前缺口。
"""
import time
from pathlib import Path

import pandas as pd

from db_config import make_jydb_conn

# ── 常量 ──────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parent / "cache" / "st"
CACHE_PATH = CACHE_DIR / "st_intervals.parquet"

# A 股市场码 → windcode 后缀（JYDB CT_SystemConst LB=201 已核验）
# 83=上交所, 90=深交所, 18=北交所；81=三板、cat=2=B股 已排除
MARKET_SUFFIX = {83: "SH", 90: "SZ", 18: "BJ"}

# 拉 A 股 ST 事件 + 代码映射。SecuCategory=1 取 A 股，SecuMarket 限三所。
SQL = """
SELECT sm.SecuCode, sm.SecuMarket, st.SpecialTradeTime, st.SecurityAbbr, st.ID
FROM dbo.LC_SpecialTrade st
JOIN dbo.SecuMain sm ON st.InnerCode = sm.InnerCode
WHERE sm.SecuCategory = 1
  AND sm.SecuMarket IN (83, 90, 18)
  AND st.SpecialTradeTime IS NOT NULL
"""

MAX_RETRIES = 3
RETRY_WAIT = 30  # 秒


# ── 拉取 ──────────────────────────────────────────────────

def fetch() -> pd.DataFrame:
    """连 JYDB 拉 A 股 ST 事件原始行，带重试。返回未转换的 DataFrame。"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  拉取 LC_SpecialTrade (尝试 {attempt}/{MAX_RETRIES})...", end=" ", flush=True)
            t0 = time.time()
            conn = make_jydb_conn()
            df = pd.read_sql(SQL, conn)
            conn.close()
            print(f"{len(df)} 行, {time.time() - t0:.1f}s")
            return df
        except Exception as e:
            print(f"失败: {e}")
            if attempt < MAX_RETRIES:
                wait = RETRY_WAIT * attempt
                print(f"    等待 {wait}s 后重试...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"拉取 LC_SpecialTrade 失败，已重试 {MAX_RETRIES} 次: {e}") from e


# ── 事件 → 区间 ────────────────────────────────────────────

def build_intervals(raw: pd.DataFrame) -> pd.DataFrame:
    """
    事件行还原为 ST 区间表。

    输入:
      raw — DataFrame[SecuCode, SecuMarket, SpecialTradeTime, SecurityAbbr, ID]
            每行一次特别处理（带帽/摘帽/变更）事件。

    输出:
      DataFrame[code, st_start, st_end]，每段连续 ST 一行。
      code     — windcode（如 000004.SZ）
      st_start — 进入 ST 的日期（datetime）
      st_end   — 退出 ST 的日期（datetime）；仍为 ST 则 NaT

    逻辑:
      1. 拼 windcode；事件含 "ST" 标记 is_st；按 (code, date) 排序。
      2. 同一 code 内，把"状态从非ST→ST"的事件起为区间开始，
         "状态从ST→非ST"的事件日为区间结束（左闭右开：摘帽日当天已非 ST）。
      3. 末尾仍 ST 的区间 st_end = NaT。
    """
    df = raw.copy()
    df["code"] = df["SecuCode"] + "." + df["SecuMarket"].map(MARKET_SUFFIX)
    df["date"] = pd.to_datetime(df["SpecialTradeTime"])
    df["is_st"] = df["SecurityAbbr"].str.contains("ST", case=False, na=False)

    # 同 (code, date) 多事件去重：按 ID 取最新一条（JYDB 同日多次登记）
    df = df.sort_values(["code", "date", "ID"]).drop_duplicates(
        subset=["code", "date"], keep="last"
    )

    intervals = []
    for code, g in df.groupby("code", sort=False):
        g = g.sort_values("date")
        open_start = None  # 当前打开的 ST 区间起点
        prev_is_st = False
        for date, is_st in zip(g["date"], g["is_st"]):
            if is_st and not prev_is_st:
                open_start = date              # 非ST→ST：开区间
            elif not is_st and prev_is_st:
                intervals.append((code, open_start, date))  # ST→非ST：闭区间（摘帽日为 end）
                open_start = None
            prev_is_st = is_st
        if prev_is_st:                          # 末尾仍 ST
            intervals.append((code, open_start, pd.NaT))

    out = pd.DataFrame(intervals, columns=["code", "st_start", "st_end"])
    return out.sort_values(["code", "st_start"]).reset_index(drop=True)


# ── 校验 ──────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> None:
    """校验区间表质量，任何问题直接 raise。"""
    n = len(df)
    print(f"\n校验 {n} 个 ST 区间...")

    # 1. code / st_start 非空
    for col in ["code", "st_start"]:
        bad = df[col].isna()
        if bad.any():
            raise ValueError(f"{col} 含 NaN: {bad.sum()} 行")
    print("  [1/3] code / st_start 非空 ✓")

    # 2. st_end > st_start（NaT 即"仍 ST"，跳过）
    mask = df["st_end"].notna()
    if mask.any():
        bad = df.loc[mask, "st_end"] <= df.loc[mask, "st_start"]
        if bad.any():
            samples = df.loc[mask].loc[bad].head(10)
            raise ValueError(f"st_end <= st_start: {bad.sum()} 行\n{samples}")
    print(f"  [2/3] st_end > st_start（{mask.sum()} 段已结束, {(~mask).sum()} 段仍 ST）✓")

    # 3. 同一 code 的区间不重叠（按 start 排序后，前一段 end <= 后一段 start）
    overlap_codes = []
    for code, g in df.groupby("code"):
        g = g.sort_values("st_start")
        ends = g["st_end"].fillna(pd.Timestamp.max)
        starts = g["st_start"]
        if (ends.values[:-1] > starts.values[1:]).any():
            overlap_codes.append(code)
    if overlap_codes:
        raise ValueError(f"以下 code 的 ST 区间重叠: {overlap_codes[:10]}")
    print(f"  [3/3] 同 code 区间不重叠（{df['code'].nunique()} 只股票）✓")

    print("校验全部通过。\n")


# ── 主流程 ────────────────────────────────────────────────

def fetch_and_save() -> None:
    """全量拉取 + 还原区间 + 校验 + 存盘。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw = fetch()
    if len(raw) == 0:
        raise RuntimeError("LC_SpecialTrade 拉取结果为空")
    df = build_intervals(raw)
    validate(df)
    df.to_parquet(CACHE_PATH, index=False, engine="pyarrow")
    size_kb = CACHE_PATH.stat().st_size / 1024
    span = f"{df['st_start'].min().date()} ~ {df['st_start'].max().date()}"
    print(f"→ 已存: {CACHE_PATH.name}（{len(df)} 段, {df['code'].nunique()} 只, "
          f"起始日跨度 {span}, {size_kb:.1f} KB）")


def main():
    fetch_and_save()
    print("完成。")


if __name__ == "__main__":
    main()
