"""
数据读取层：从 parquet 缓存读出标准 DataFrame（缓存目录结构只在这里出现）。

定位：fetch_*.py 是「写缓存」侧，本模块是「读缓存」侧——上层（factor/engine）只调这里的
load_* 函数，不再关心 cache 目录怎么分片、文件名怎么拼。缓存路径/字段/分片方式改了，只改本文件。

命名约定：load_=从缓存读数据。
"""
import os
from pathlib import Path

import pandas as pd

CACHE_ROOT = Path(__file__).resolve().parent / "cache"  # 缓存根（本文件在 data/，cache/ 同级）

# 价格质量校验：A 股单日涨跌幅有硬上限（最高北交所 ±30%），复权价单日比值超过 MAX_DAILY_RATIO
# 倍（或 <1/该值、或 adj_close≤0）在物理上不可能，几乎必为数据错（如北交所新三板期 close 卡 0.01
# 再跳的脏价）。默认 Fail-Fast 报错，逼着修数据；BT_PRICE_ON_BAD=drop 则剔除整只问题股后继续。
MAX_DAILY_RATIO = 10.0
_DEFAULT_ON_BAD = os.environ.get("BT_PRICE_ON_BAD", "raise")


class DataQualityError(ValueError):
    """缓存里检出物理不可能的价格数据（复权价单日跳变越界 / 非正价）。"""


def validate_adjclose_quality(px, on_bad=None, max_daily_ratio=MAX_DAILY_RATIO):
    """检出物理不可能的单日复权价跳变并按 on_bad 处置。

    px: long [code, date, adj_close, ...]，已按 [code, date] 排序。
    判据(逐股按日)：adj_close ≤ 0，或相邻交易日比值 > max_daily_ratio 或 < 1/max_daily_ratio。
    每股首日(无前值)不判，避开 IPO 首日无涨跌幅限制。真实重整复牌(如 +300%) 比值 <10 不会误伤。
    on_bad='raise'(默认) 列清单报错；'drop' 剔除整只问题股(脏股本就是垃圾票)并打印。
    """
    on_bad = on_bad or _DEFAULT_ON_BAD
    if on_bad not in ("raise", "drop"):
        raise ValueError(f"on_bad 非法: {on_bad!r}（仅 'raise' | 'drop'）")
    prev = px.groupby("code")["adj_close"].shift(1)
    ratio = px["adj_close"] / prev
    bad = ((px["adj_close"] <= 0)
           | (ratio > max_daily_ratio)
           | (ratio < 1.0 / max_daily_ratio)).fillna(False)
    if not bad.any():
        return px
    viol = px.loc[bad, ["code", "date", "adj_close"]].assign(prev=prev[bad], ratio=ratio[bad])
    head = "\n".join(
        f"  {r.code} {pd.Timestamp(r.date).date()} adj_close {r.prev:.4g}->{r.adj_close:.4g} (×{r.ratio:.0f})"
        for r in viol.sort_values("ratio", ascending=False).head(20).itertuples())
    summary = (f"价格质量校验：检出 {int(bad.sum())} 处物理不可能的单日复权价跳变"
               f"（>{max_daily_ratio:g} 倍或非正价），涉及 {viol['code'].nunique()} 只股。最离谱前 20：\n{head}")
    if on_bad == "raise":
        raise DataQualityError(
            summary + "\n\n这些几乎必是缓存里的脏价（修 fetch / 重拉缓存）。"
            "确认要带病回测则设环境变量 BT_PRICE_ON_BAD=drop（剔除整只问题股后继续）。")
    bad_codes = set(viol["code"].unique())
    print(f"[价格质量][drop] {summary}\n→ 剔除整只问题股 {len(bad_codes)} 只后继续。", flush=True)
    return px[~px["code"].isin(bad_codes)]


def load_calendar() -> pd.DatetimeIndex:
    """读交易日历缓存 → 升序去重 DatetimeIndex（单一真相源）。"""
    path = CACHE_ROOT / "calendar" / "trade_dates.parquet"
    if not path.exists():
        raise FileNotFoundError(f"交易日历缺失：{path}（先跑 data/fetch_calendar.py）")
    df = pd.read_parquet(path, columns=["date"])
    return pd.DatetimeIndex(pd.to_datetime(df["date"]).unique()).sort_values()


def load_price_df(codes, start, end, need_feasibility=False) -> pd.DataFrame:
    """逐年读日行情 → long price_df [date, code, adj_close(, limit_status, trade_status)]。

    codes: 可迭代股票代码 或 None(全市场)；start/end: 决定读哪些年份；
    need_feasibility=True 时额外 merge 衍生表 limit_status + daily trade_status（开可行性过滤用）。
    """
    code_set = None if codes is None else set(codes)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start > end:
        raise ValueError(f"load_price_df: start({start.date()}) > end({end.date()})，区间为空"
                         f"（codes={'全市场' if codes is None else len(code_set)}）")
    cols = ["code", "date", "adj_close"] + (["trade_status"] if need_feasibility else [])
    frames = []
    for year in range(start.year, end.year + 1):
        f = CACHE_ROOT / "daily" / f"{year}.parquet"
        if not f.exists():
            raise FileNotFoundError(f"日行情缺失：{f}（先跑 data/fetch_price_daily.py）")
        d = pd.read_parquet(f, columns=cols)
        if code_set is not None:
            d = d[d["code"].isin(code_set)]
        frames.append(d)
    px = pd.concat(frames, ignore_index=True)
    px["date"] = pd.to_datetime(px["date"])
    px = px[(px["date"] >= start) & (px["date"] <= end)]
    if px.empty:
        raise ValueError(f"price_df 为空：start={start.date()} end={end.date()} "
                         f"codes={'全市场' if codes is None else len(code_set)}")

    if need_feasibility:
        deriv = []
        for year in range(start.year, end.year + 1):
            f = CACHE_ROOT / "derivative" / f"{year}.parquet"
            if not f.exists():
                raise FileNotFoundError(f"衍生表缺失：{f}（开可行性过滤需 limit_status）")
            dd = pd.read_parquet(f, columns=["code", "date", "limit_status"])
            if code_set is not None:
                dd = dd[dd["code"].isin(code_set)]
            deriv.append(dd)
        deriv = pd.concat(deriv, ignore_index=True)
        deriv["date"] = pd.to_datetime(deriv["date"])
        px = px.merge(deriv, on=["code", "date"], how="left")

    dup = px.groupby(["date", "code"]).size()
    if (dup > 1).any():
        bad = dup[dup > 1].head(5)
        raise ValueError(f"price_df 存在重复 (date,code)（缓存应已干净，dup 即数据问题）：\n{bad}")
    px = px.sort_values(["code", "date"]).reset_index(drop=True)
    px = validate_adjclose_quality(px)              # 价格质量守卫：物理不可能的复权价跳变 → 报错/剔除
    return px


def _read_years(subdir, cols, codes, start, end) -> pd.DataFrame:
    """逐年读 cache/<subdir>/{year}.parquet 指定列 → 拼接、转 date、按 [start,end] 过滤的长表。"""
    code_set = None if codes is None else set(codes)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start > end:
        raise ValueError(f"_read_years({subdir}): start({start.date()}) > end({end.date()})，区间为空"
                         f"（codes={'全市场' if codes is None else len(code_set)}）")
    frames = []
    for year in range(start.year, end.year + 1):
        f = CACHE_ROOT / subdir / f"{year}.parquet"
        if not f.exists():
            raise FileNotFoundError(f"{subdir} 缓存缺失：{f}（先跑 data/fetch_all.py 补齐缓存）")
        d = pd.read_parquet(f, columns=cols)
        if code_set is not None:
            d = d[d["code"].isin(code_set)]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df[(df["date"] >= start) & (df["date"] <= end)]


def load_daily_df(codes, start, end,
                  fields=("vwap", "close", "adj_factor", "trade_status")) -> pd.DataFrame:
    """逐年读日行情指定字段 → long [date, code, *fields]。

    load_price_df 只给 adj_close；现金引擎要真实成交价(vwap)、真实收盘价(close)、复权因子、
    停牌状态(trade_status)，故单列一个字段可选的读法。codes=None 取全市场。
    """
    df = _read_years("daily", ["code", "date", *fields], codes, start, end)
    if df.empty:
        raise ValueError(f"daily_df 为空：start={pd.Timestamp(start).date()} end={pd.Timestamp(end).date()} "
                         f"codes={'全市场' if codes is None else len(set(codes))}")
    dup = df.groupby(["date", "code"]).size()
    if (dup > 1).any():
        raise ValueError(f"daily_df 存在重复 (date,code)（缓存应已干净）：\n{dup[dup > 1].head(5)}")
    return df.sort_values(["code", "date"]).reset_index(drop=True)


def load_derivative_df(codes, start, end, fields=("limit_status",)) -> pd.DataFrame:
    """逐年读衍生表指定字段 → long [date, code, *fields]（现金引擎判涨跌停用 limit_status：1涨停/-1跌停/0正常/NA）。"""
    df = _read_years("derivative", ["code", "date", *fields], codes, start, end)
    if df.empty:
        raise ValueError(f"derivative_df 为空：start={pd.Timestamp(start).date()} end={pd.Timestamp(end).date()}")
    return df.sort_values(["code", "date"]).reset_index(drop=True)


def load_index_eod(index_code: str) -> pd.DataFrame:
    """读单个指数 EOD → [date, close]（基准净值用）。文件名点换下划线。"""
    path = CACHE_ROOT / "index_eod" / f"{index_code.replace('.', '_')}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"指数 {index_code} EOD 未缓存。当前仅 000001.SH 已落盘，"
            f"请跑：python data/fetch_index_eod.py --indices {index_code}"
        )
    df = pd.read_parquet(path, columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_index_members(index_code: str) -> pd.DataFrame:
    """读单个指数成分股区间表 [index_code, code, entry_date, exit_date]（exit_date NaT=当前）。"""
    path = CACHE_ROOT / "index_members" / f"{index_code.replace('.', '_')}.parquet"
    if not path.exists():
        have = sorted(p.stem for p in (CACHE_ROOT / "index_members").glob("*.parquet"))
        raise FileNotFoundError(f"指数 {index_code} 成分股未缓存。已有：{have}")
    df = pd.read_parquet(path)
    for c in ("entry_date", "exit_date"):
        df[c] = pd.to_datetime(df[c])
    return df


def load_st_intervals() -> pd.DataFrame:
    """读 ST 区间表 [code, st_start, st_end]（st_end NaT=至今仍 ST）。"""
    path = CACHE_ROOT / "st" / "st_intervals.parquet"
    if not path.exists():
        raise FileNotFoundError(f"ST 区间表缺失：{path}")
    df = pd.read_parquet(path)
    for c in ("st_start", "st_end"):
        df[c] = pd.to_datetime(df[c])
    return df


def load_delist_dates() -> pd.Series:
    """读股票退市日 → Series(index=code, value=delist_date)。delist_date 为 NaT=未退市。

    现金引擎判退市用（真实退市日，替代"窗口内最后有效成交日"的前视判定）。
    """
    path = CACHE_ROOT / "description" / "description.parquet"
    if not path.exists():
        raise FileNotFoundError(f"股票描述表缺失：{path}（先跑 data/fetch_description.py）")
    df = pd.read_parquet(path, columns=["code", "delist_date"])
    df["delist_date"] = pd.to_datetime(df["delist_date"])
    return df.set_index("code")["delist_date"]
