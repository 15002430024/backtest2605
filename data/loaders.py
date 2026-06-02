"""
数据读取层：从 parquet 缓存读出标准 DataFrame（缓存目录结构只在这里出现）。

定位：fetch_*.py 是「写缓存」侧，本模块是「读缓存」侧——上层（factor/engine）只调这里的
load_* 函数，不再关心 cache 目录怎么分片、文件名怎么拼。缓存路径/字段/分片方式改了，只改本文件。

命名约定：load_=从缓存读数据。
"""
from pathlib import Path

import pandas as pd

CACHE_ROOT = Path(__file__).resolve().parent / "cache"  # 缓存根（本文件在 data/，cache/ 同级）


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
        raise ValueError(f"price_df 为空：start={start.date()} end={end.date()} codes={len(code_set) if code_set else '全市场'}")

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
    return px.sort_values(["code", "date"]).reset_index(drop=True)


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
