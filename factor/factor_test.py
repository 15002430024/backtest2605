"""
因子测试模块（分析层）

定位：引擎的"前一层"。传入因子 → 内部转成权重（仅喂引擎，不导出） → 调用 run_backtest
→ 输出 IC / 十分组净值 / 多空 / 超额。原引擎 engine/backtest.py 逻辑一行不改。

入口：run_factor_test(factor_wide, config) -> dict

所有组合都走 run_backtest：多空用 weight_mode='long_short'（多头和+1、空头和−1，标准100/100），
其余 long_only。因子按原样对齐到调仓日，不做任何 ffill；默认只使用有因子值的有效覆盖域，
并在 meta 报告缺失覆盖情况。若需要强验数，可设 strict_missing_factor=True。

命名约定：load_/make_/build_=造数据 · compute_=算指标 · assign_=分组 · backtest_=调引擎 · run_=入口。
"""

import numpy as np
import pandas as pd

from engine.backtest import run_backtest, calc_benchmark
from analysis.metrics import calc_metrics  # 分组绩效复用统一指标，不再自带 _nav_metrics
from data.loaders import (  # 缓存读取已下沉到数据层，因子层不再碰 cache 目录
    load_calendar, load_price_df, load_index_eod, load_index_members, load_st_intervals,
)
from data.panels import intervals_to_panel  # 区间→面板下沉到数据层，与现金引擎共用

_PPY = {"M": 12, "W": 52, "D": 252}  # 各调仓频率的年化周期数

DEFAULT_FACTOR_CONFIG = {
    "rebalance": "M",        # 'M'|'W'|'D'|list[调仓日]
    "universe": "all",       # 'all' | 指数代码'000905.SH' | 'user'
    "user_mask": None,       # universe=='user' 时传 DataFrame(调仓日×code) bool
    "exclude_st": False,     # 可选剔 ST（默认关；开则自动读 st_intervals.parquet）
    "direction": +1,         # +1 高因子看好 | -1 翻转
    "n_groups": 10,
    "benchmark": None,       # 超额对标的外部指数代码；必填，缺省 None 会在入口 raise（不再用池内等权兜底）
    "start_date": None,      # None → factor_wide.index.min()
    "end_date": None,        # None → factor_wide.index.max()
    "bt_config": None,       # 透传 run_backtest 摩擦 config（默认 None=理想态）
    "strict_missing_factor": False,  # True 时指定池内缺因子直接 raise；默认只报告并剔除
}


# ============================================================
# 调仓日 / 因子对齐 / 股票池
# ============================================================
def make_rebalance_dates(calendar, start, end, rebalance) -> pd.DatetimeIndex:
    """按频率生成调仓日，每个落在真实交易日上。

    'M'/'W' 在交易日历上 groupby 周期取 max → 该周/月真实存在的最后一个交易日
    （周五休市则自动落到周四）；'D' 全部；list 校验每个 ∈ 交易日。
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    cal = calendar[(calendar >= start) & (calendar <= end)]
    if isinstance(rebalance, (list, tuple, pd.DatetimeIndex)):
        reb = pd.DatetimeIndex(pd.to_datetime(list(rebalance)))
        bad = reb.difference(calendar)
        if len(bad):
            raise ValueError(f"rebalance 列表含非交易日：{list(bad[:10])}")
        reb = reb[(reb >= start) & (reb <= end)]
    elif rebalance == "D":
        reb = cal
    elif rebalance in ("M", "W"):
        s = cal.to_series()
        reb = pd.DatetimeIndex(s.groupby(cal.to_period(rebalance)).max().values)
    else:
        raise ValueError(f"非法 rebalance: {rebalance!r}（支持 'M'|'W'|'D'|list）")
    reb = pd.DatetimeIndex(reb).unique().sort_values()
    if len(reb) < 2:
        raise ValueError(f"调仓日不足 2 个（{len(reb)}），无法形成持有区间/IC：start={start.date()} end={end.date()} rebalance={rebalance}")
    return reb


def build_factor_panel(factor_wide, price_codes, rebalance_dates) -> pd.DataFrame:
    """因子按原样对齐到 (调仓日 × code)，不做任何 ffill。缺失保留 NaN。"""
    if factor_wide.empty:
        raise ValueError("factor_wide 为空")
    if factor_wide.index.has_duplicates:
        raise ValueError("factor_wide 的 index（日期）有重复，存在歧义")
    factor_wide = factor_wide.copy()
    factor_wide.index = pd.to_datetime(factor_wide.index)
    price_set = set(price_codes)
    n_missing = len(set(factor_wide.columns) - price_set)
    if n_missing:
        # 在回测区间行情里无数据的因子 code（区间外退市/未上市）→ 剔除（无价无法测），不 raise
        print(f"[build_factor_panel] {n_missing} 个因子 code 在回测区间无行情，已剔除（区间外退市/未上市股）")
    if not (set(factor_wide.columns) & price_set):
        raise ValueError("factor_wide 没有任何 code 在行情里出现——疑似代码格式不符（如缺 .SH/.SZ 交易所后缀）")
    cols = sorted(price_set)   # 对齐到行情全集：因子缺的股票 → NaN（按 universe 决定 raise/出池）
    return factor_wide.reindex(index=rebalance_dates).reindex(columns=cols)


def _true_locations_sample(mask: pd.DataFrame, n: int = 10) -> list:
    """布尔面板里 True 的前 n 个 (date, code) 样例，用于覆盖率报告。"""
    if mask is None or mask.empty:
        return []
    stacked = mask.stack()
    hits = stacked[stacked].index[:n]
    return [(str(pd.Timestamp(d).date()), c) for d, c in hits]


def _coverage_report(raw_pool: pd.DataFrame, effective_pool: pd.DataFrame,
                     missing_factor: pd.DataFrame, dropped_membership: pd.DataFrame = None) -> dict:
    """指定池/有效覆盖域的覆盖率报告；只报告，不参与组合计算。"""
    idx = effective_pool.index
    raw_count = raw_pool.sum(axis=1).reindex(idx).fillna(0)
    effective_count = effective_pool.sum(axis=1).reindex(idx).fillna(0)

    if dropped_membership is not None and not dropped_membership.empty:
        dropped_count = dropped_membership.sum(axis=1).reindex(idx).fillna(0)
        dropped_cells = int(dropped_membership.values.sum())
        dropped_codes = sorted(dropped_membership.columns[dropped_membership.any(axis=0)].tolist())
    else:
        dropped_count = pd.Series(0, index=idx)
        dropped_cells = 0
        dropped_codes = []

    reported_pool_count = raw_count + dropped_count
    coverage = effective_count / reported_pool_count.replace(0, np.nan)
    known_missing_cells = int(missing_factor.values.sum()) if missing_factor is not None else 0

    return {
        "pool_size_mean": float(reported_pool_count.mean()) if len(reported_pool_count) else np.nan,
        "effective_size_mean": float(effective_count.mean()) if len(effective_count) else np.nan,
        "effective_coverage_mean": float(coverage.mean()) if coverage.notna().any() else np.nan,
        "effective_coverage_min": float(coverage.min()) if coverage.notna().any() else np.nan,
        "known_missing_factor_cells": known_missing_cells,
        "known_missing_factor_sample": _true_locations_sample(missing_factor),
        "membership_not_in_factor_columns_cells": dropped_cells,
        "membership_not_in_factor_columns_code_count": len(dropped_codes),
        "membership_not_in_factor_columns_codes_sample": dropped_codes[:20],
        "membership_not_in_factor_columns_sample": _true_locations_sample(dropped_membership),
    }


def _format_coverage_warning(coverage: dict) -> str:
    """覆盖率问题的一行提示；无问题时返回空字符串。"""
    known = coverage["known_missing_factor_cells"]
    dropped = coverage["membership_not_in_factor_columns_cells"]
    if known == 0 and dropped == 0:
        return ""
    mean_cov = coverage["effective_coverage_mean"]
    min_cov = coverage["effective_coverage_min"]
    return (
        "[run_factor_test] 因子覆盖提示："
        f"有效覆盖率均值={mean_cov:.2%} 最低={min_cov:.2%}；"
        f"池内有行情但因子为 NaN={known} 处；"
        f"指定池成员不在因子/行情覆盖列={dropped} 处"
    )


def build_universe_mask(factor_panel, price_df, rebalance_dates, universe,
                        members_df=None, user_mask=None, exclude_st=False, st_intervals=None,
                        strict_missing_factor=False, return_coverage=False):
    """每个调仓日的可选股票池 = 在池内 & 当日有价格 [& 非ST]。

    默认口径：有效股票池 = 指定池/全市场 ∩ 有行情 ∩ 非 ST ∩ 有因子值。
    缺因子不 ffill、不赋值、不阻断，只从有效池剔除并报告覆盖率；
    strict_missing_factor=True 时才对指定池内缺因子直接 raise。
    """
    codes = factor_panel.columns
    # 当日真有价（挡退市/未上市/缺价：build_factor_panel 不 ffill，但仍要确认调仓日有价）
    avail = (price_df.pivot(index="date", columns="code", values="adj_close")
             .reindex(index=rebalance_dates).notna()
             .reindex(columns=codes).fillna(False))

    if exclude_st:
        if st_intervals is None:
            raise ValueError("exclude_st=True 但未提供 st_intervals")
        st_panel = intervals_to_panel(st_intervals, "st_start", "st_end", rebalance_dates, codes, end_inclusive=True)
        not_st = ~st_panel
    else:
        not_st = True

    dropped_membership = None
    if universe == "all":
        raw_pool = avail & not_st

    if universe == "user":
        if user_mask is None:
            raise ValueError("universe=='user' 但未提供 user_mask")
        um = user_mask.copy()
        um.index = pd.to_datetime(um.index)
        full_membership = um.reindex(index=rebalance_dates).fillna(False).astype(bool)
        dropped_cols = [c for c in full_membership.columns if c not in set(codes)]
        dropped_membership = full_membership[dropped_cols] if dropped_cols else None
        membership = full_membership.reindex(columns=codes).fillna(False)
        raw_pool = membership & avail & not_st
    elif universe != "all":  # 指数代码
        if members_df is None:
            raise ValueError(f"universe={universe!r} 但未提供 members_df")
        report_codes = sorted(set(codes) | set(members_df["code"].unique()))
        full_membership = intervals_to_panel(members_df, "entry_date", "exit_date", rebalance_dates, report_codes, end_inclusive=False)
        active_cols = set(full_membership.columns[full_membership.any(axis=0)])
        dropped_cols = sorted(active_cols - set(codes))
        dropped_membership = full_membership[dropped_cols] if dropped_cols else None
        membership = full_membership.reindex(columns=codes).fillna(False)
        raw_pool = membership & avail & not_st

    miss = raw_pool & factor_panel.isna()
    if strict_missing_factor and (miss.values.any() or (dropped_membership is not None and dropped_membership.values.any())):
        stacked = miss.stack()
        bad = stacked[stacked].index[:10].tolist()
        dropped_sample = _true_locations_sample(dropped_membership)
        raise ValueError(
            f"指定股票池内存在因子缺失：已知覆盖列 NaN={int(miss.values.sum())} 处，"
            f"不在因子/行情覆盖列={0 if dropped_membership is None else int(dropped_membership.values.sum())} 处。"
            f"NaN 样例 (date, code)：{[(str(d.date()), c) for d, c in bad]}；"
            f"缺列样例 (date, code)：{dropped_sample}"
        )
    effective_pool = raw_pool & factor_panel.notna()
    coverage = _coverage_report(raw_pool, effective_pool, miss, dropped_membership)
    if return_coverage:
        return effective_pool, coverage
    return effective_pool


# ============================================================
# IC
# ============================================================
def compute_forward_returns(price_df, rebalance_dates, calendar) -> pd.DataFrame:
    """每个调仓日 T→下一调仓日 的持有期收益（仅 IC 用）。沿日历轴对齐再 shift，绝不裸 shift。"""
    px = price_df.pivot(index="date", columns="code", values="adj_close").reindex(index=calendar)
    pr = px.reindex(index=rebalance_dates)
    fwd = pr.shift(-1) / pr - 1
    return fwd.iloc[:-1]  # 最后一个调仓日无未来，丢掉


def _row_corr(a, b):
    """逐行（每个调仓日）截面相关，pairwise 跳 NaN（a、b 已在非共有处置 NaN）。"""
    ad = a.sub(a.mean(axis=1), axis=0)
    bd = b.sub(b.mean(axis=1), axis=0)
    num = (ad * bd).sum(axis=1)
    den = np.sqrt((ad ** 2).sum(axis=1) * (bd ** 2).sum(axis=1))
    return num / den.where(den > 0)


def compute_ic(factor_panel, forward_returns, mask, rebalance) -> dict:
    """截面 Pearson IC + Spearman RankIC + 统计（向量化）。"""
    idx = forward_returns.index
    m = mask.reindex(index=idx)
    f = factor_panel.reindex(index=idx).where(m)
    r = forward_returns.where(m)
    both = f.notna() & r.notna()
    f, r = f.where(both), r.where(both)
    n_valid = both.sum(axis=1)
    enough = n_valid >= 3

    ic = _row_corr(f, r).where(enough)
    rankic = _row_corr(f.rank(axis=1), r.rank(axis=1)).where(enough)

    ppy = _PPY.get(rebalance if isinstance(rebalance, str) else "M", 12)
    if not isinstance(rebalance, str):
        # 自定义调仓日列表：gap 是日历日，故用 365.25/日历gap 得年化周期数
        # （与字符串路径 W=52/M=12 对齐；用 252/日历gap 会混单位，周频得 36、月频得 8.4）
        gaps = np.diff(idx.values).astype("timedelta64[D]").astype(float)
        if len(gaps):
            median_gap = np.median(gaps)
            # 相邻交易日（逐日调仓，中位日历 gap=1）按交易日年化=252，与字符串 'D' 同口径；
            # 否则用 365.25/gap（W=365.25/7≈52、M=365.25/30≈12，与字符串分支对齐）。
            # 逐日若按 365.25/gap 会得 365、与 'D' 的 252 差 sqrt(365/252)=1.2 倍。
            ppy = 252.0 if median_gap <= 1.0 else max(1.0, round(365.25 / median_gap))

    def stats(s):
        mean, std = s.mean(), s.std(ddof=1)
        n = int(s.notna().sum())
        ir = mean / std if std and std > 0 else np.nan
        return mean, ir, ir * np.sqrt(ppy) if pd.notna(ir) else np.nan, ir * np.sqrt(n) if pd.notna(ir) else np.nan, (s.dropna() > 0).mean()

    ic_mean, ic_ir, ic_ir_annual, ic_t, ic_win = stats(ic)
    rk_mean, rk_ir, _, rk_t, _ = stats(rankic)
    if ic.notna().sum() == 0:
        print("[compute_ic] warning: 所有期 IC 均为 NaN（因子可能无效或全常数）")
    return {
        "ic_series": ic, "rankic_series": rankic,
        "ic_mean": ic_mean, "ic_ir": ic_ir, "ic_ir_annual": ic_ir_annual,
        "ic_t": ic_t, "ic_winrate": ic_win,
        "rankic_mean": rk_mean, "rankic_ir": rk_ir, "rankic_t": rk_t,
        "ic_cum": ic.fillna(0).cumsum(),
    }


# ============================================================
# 分组 / 权重
# ============================================================
def assign_groups(factor_panel, mask, n_groups, direction) -> pd.DataFrame:
    """每期截面分 n_groups 组（等数量分组，向量化 rank）。

    组号 n_groups 永远=做多组（direction 只改 score 符号，组号语义不变：n=最高 score、1=最低）。
    某期有效股 < n_groups → raise（不 skip）。
    """
    f = factor_panel.where(mask)
    counts = f.notna().sum(axis=1)
    bad = counts[counts < n_groups]
    if len(bad):
        d0 = bad.index[0]
        raise ValueError(f"调仓日 {d0.date()} 有效股票 {int(bad.iloc[0])} 只 < n_groups={n_groups}，无法分组")
    ranks = f.rank(axis=1, method="first", ascending=(direction == 1))
    labels = np.ceil(ranks.div(counts, axis=0) * n_groups).clip(1, n_groups)
    return labels.where(f.notna())


def _renorm(w):
    """防御性重归一：每行除以 Σ|w|，使 Σ|w| 精确=1（保留符号）。"""
    gross = w.abs().sum(axis=1)
    return w.div(gross.where(gross > 0), axis=0)


def build_group_weights(group_labels, g, side, weighting, factor_panel, direction) -> pd.DataFrame:
    """第 g 组 → 合法 long-only weights_df（side=+1 做多 / −1 做空），仅内部喂引擎。"""
    member = (group_labels == g)
    if weighting == "equal":
        w = member.astype(float).div(member.sum(axis=1).where(lambda s: s > 0), axis=0) * side
    elif weighting == "factor":
        score = direction * factor_panel
        rank_basis = score if side == 1 else -score
        r = rank_basis.where(member).rank(axis=1, method="first")
        w = r.div(r.sum(axis=1).where(lambda s: s > 0), axis=0) * side
    else:
        raise ValueError(f"非法 weighting: {weighting!r}（'equal'|'factor'）")
    w = _renorm(w)
    long = w.stack().reset_index()  # stack 自动丢 NaN
    long.columns = ["date", "code", "weight"]
    long = long[long["weight"] != 0].reset_index(drop=True)
    if long["weight"].isna().any():
        raise RuntimeError(f"build_group_weights(g={g}) 产生 NaN 权重")
    if long.duplicated(["date", "code"]).any():
        raise RuntimeError(f"build_group_weights(g={g}) 产生重复 (date,code)")
    return long


def build_long_short_weights(group_labels, weighting, factor_panel, direction, n_groups) -> pd.DataFrame:
    """多空（100/100）：第 n_groups 组做多(和=+1) + 第 1 组做空(和=−1)，喂 long_short 模式。"""
    top = build_group_weights(group_labels, n_groups, +1, weighting, factor_panel, direction)
    bot = build_group_weights(group_labels, 1, -1, weighting, factor_panel, direction)
    ls = pd.concat([top, bot], ignore_index=True)
    # 自检：每调仓日 多头和≈+1、空头和≈−1（满足引擎 long_short 校验）；同股不会两组
    chk = ls.groupby("date")["weight"]
    long_sum = chk.apply(lambda x: x[x > 0].sum())
    short_sum = chk.apply(lambda x: x[x < 0].sum())
    if ((long_sum - 1).abs() > 1e-9).any() or ((short_sum + 1).abs() > 1e-9).any():
        raise RuntimeError("build_long_short_weights 多/空头和不满足 ±1（引擎 long_short 会 raise）")
    if ls.duplicated(["date", "code"]).any():
        raise RuntimeError("build_long_short_weights 顶/底组有同股重复")
    return ls


# ============================================================
# 调引擎（每次先按 codes 过滤 price_df）
# ============================================================
def _bt_end(calendar, end_date) -> pd.Timestamp:
    """区间内 ≤ end_date 的最后一个交易日（强制显式传给引擎，避免只跑到末调仓日漏末段收益）。"""
    end_date = pd.Timestamp(end_date)
    days = calendar[calendar <= end_date]
    if len(days) == 0:
        raise ValueError(f"end_date {end_date.date()} 之前没有交易日")
    return days.max()


def _run(weights_df, price_df, weight_mode, bt_end, bt_config) -> pd.Series:
    """喂引擎前按该组合 codes 过滤 price_df（避免全市场重算），显式传 end_date。"""
    psub = price_df[price_df["code"].isin(weights_df["code"].unique())]
    cfg = dict(bt_config or {})
    cfg["weight_mode"] = weight_mode
    return run_backtest(weights_df, psub, config=cfg, end_date=bt_end)["nav"]


def backtest_groups(group_labels, factor_panel, price_df, weighting, direction, bt_end, bt_config) -> dict:
    """1..n_groups 每组 long-only 过引擎拿日度净值（单独多 = 最高组）。"""
    n_groups = int(group_labels.max().max())
    out = {}
    for g in range(1, n_groups + 1):
        w = build_group_weights(group_labels, g, +1, weighting, factor_panel, direction)
        out[g] = _run(w, price_df, "long_only", bt_end, bt_config)
    return out


def backtest_short_only(group_labels, factor_panel, price_df, weighting, direction, bt_end) -> pd.Series:
    """第 1 组做空 → 引擎(long_only, 负权重, 理想态)拿单独空净值。"""
    w = build_group_weights(group_labels, 1, -1, weighting, factor_panel, direction)
    return _run(w, price_df, "long_only", bt_end, None)


def backtest_long_short(group_labels, factor_panel, price_df, weighting, direction, bt_end) -> pd.Series:
    """多空(100/100) → 引擎 long_short 模式（理想态）拿多空净值。"""
    n_groups = int(group_labels.max().max())
    w = build_long_short_weights(group_labels, weighting, factor_panel, direction, n_groups)
    return _run(w, price_df, "long_short", bt_end, None)


# ============================================================
# 超额
# ============================================================
def compute_excess(strategy_nav, benchmark_nav) -> pd.Series:
    """超额净值（主动收益复利），inner join + 首日 fillna(0)。"""
    common = strategy_nav.index.intersection(benchmark_nav.index)
    if len(common) == 0:
        raise ValueError("策略与基准净值无共同交易日，无法算超额")
    s = strategy_nav.reindex(common)
    b = benchmark_nav.reindex(common)
    diff = (s.pct_change() - b.pct_change()).fillna(0)
    return (1 + diff).cumprod()


# 分组绩效统一用 analysis.calc_metrics（已 import），原自带 _nav_metrics 删除（消副本）。
# 注：calc_metrics 单参 nav 即可，返回含 年化收益/夏普/最大回撤 等，分组表按需取键。


# ============================================================
# 入口
# ============================================================
def run_factor_test(factor_wide, config=None) -> dict:
    """因子测试总入口：传因子 → 内部转权重 → 调 run_backtest → 出 IC/十分组/多空/超额。

    factor_wide: 宽表 DataFrame(index=日期, columns=股票代码, 值=因子值)，你已填好。
    config: 见 DEFAULT_FACTOR_CONFIG。
    返回 dict：见模块文档 / 计划。
    """
    cfg = dict(DEFAULT_FACTOR_CONFIG)
    if config:
        unknown = set(config) - set(DEFAULT_FACTOR_CONFIG)
        if unknown:
            raise ValueError(f"run_factor_test config 含未知键 {sorted(unknown)}（拼错会静默回落默认）；"
                             f"合法键为 {sorted(DEFAULT_FACTOR_CONFIG)}")
        cfg.update(config)
    if not cfg["benchmark"]:
        raise ValueError(f"benchmark 必填：需传外部指数代码（如 '000300.SH'），当前为 {cfg['benchmark']!r}；不再用池内等权兜底")
    factor_wide = factor_wide.copy()
    factor_wide.index = pd.to_datetime(factor_wide.index)

    start = pd.Timestamp(cfg["start_date"]) if cfg["start_date"] else factor_wide.index.min()
    end = pd.Timestamp(cfg["end_date"]) if cfg["end_date"] else factor_wide.index.max()
    bt_config = cfg["bt_config"]
    need_feas = bool(bt_config and bt_config.get("enable_feasibility_filter"))

    calendar = load_calendar()
    rebalance_dates = make_rebalance_dates(calendar, start, end, cfg["rebalance"])
    price_df = load_price_df(list(factor_wide.columns), start, end, need_feasibility=need_feas)
    price_codes = price_df["code"].unique()

    factor_panel = build_factor_panel(factor_wide, price_codes, rebalance_dates)

    members_df = None
    if cfg["universe"] not in ("all", "user"):
        members_df = load_index_members(cfg["universe"])
    st_intervals = load_st_intervals() if cfg["exclude_st"] else None
    mask, coverage = build_universe_mask(
        factor_panel, price_df, rebalance_dates, cfg["universe"],
        members_df=members_df, user_mask=cfg["user_mask"],
        exclude_st=cfg["exclude_st"], st_intervals=st_intervals,
        strict_missing_factor=cfg["strict_missing_factor"], return_coverage=True,
    )
    coverage_warning = _format_coverage_warning(coverage)
    if coverage_warning:
        print(coverage_warning)

    # IC（不走引擎）
    fwd = compute_forward_returns(price_df, rebalance_dates, calendar)
    ic = compute_ic(factor_panel, fwd, mask, cfg["rebalance"])

    # 分组
    group_labels = assign_groups(factor_panel, mask, cfg["n_groups"], cfg["direction"])
    n_groups = cfg["n_groups"]
    bt_end = _bt_end(calendar, end)

    # 基准 = 外部指数（必填，入口已校验）
    bench_nav = calc_benchmark(load_index_eod(cfg["benchmark"]), start, bt_end)

    group_nav, long_short, long_only, short_only, excess = {}, {}, {}, {}, {}
    for weighting in ("equal", "factor"):
        navs = backtest_groups(group_labels, factor_panel, price_df, weighting, cfg["direction"], bt_end, bt_config)
        group_nav[weighting] = pd.DataFrame({g: navs[g] for g in range(1, n_groups + 1)})
        long_only[weighting] = navs[n_groups]
        short_only[weighting] = backtest_short_only(group_labels, factor_panel, price_df, weighting, cfg["direction"], bt_end)
        long_short[weighting] = backtest_long_short(group_labels, factor_panel, price_df, weighting, cfg["direction"], bt_end)
        ex = {f"第{g}组": compute_excess(navs[g], bench_nav) for g in range(1, n_groups + 1)}
        ex["多头"] = compute_excess(long_only[weighting], bench_nav)
        excess[weighting] = pd.DataFrame(ex)

    # 每条曲线的绩效指标（自包含 _nav_metrics）
    metrics = {}
    for weighting in ("equal", "factor"):
        m = {f"第{g}组": calc_metrics(group_nav[weighting][g].dropna()) for g in range(1, n_groups + 1)}
        m["多头"] = calc_metrics(long_only[weighting].dropna())
        m["空头"] = calc_metrics(short_only[weighting].dropna())
        m["多空"] = calc_metrics(long_short[weighting].dropna())
        metrics[weighting] = m
    metrics["基准"] = calc_metrics(bench_nav.dropna())

    friction = "理想态" if not bt_config else f"含摩擦 {bt_config}"
    meta = {
        "rebalance": cfg["rebalance"], "universe": cfg["universe"], "n_groups": n_groups,
        "direction": cfg["direction"], "exclude_st": cfg["exclude_st"],
        "benchmark": cfg["benchmark"],
        "config_分组与基准": friction,
        "config_多空与单独空": "理想态（引擎规定 long_short/负权重不可开可行性过滤）",
        "因子覆盖": coverage,
        "回测区间": f"{rebalance_dates[0].date()} ~ {bt_end.date()}",
    }

    # 出图/CSV 已拆到 factor/plot_factor.py：拿到 result 后调 plot_factor_report(result, out_dir)
    return {
        "ic": ic, "group_nav": group_nav, "long_short": long_short,
        "long_only": long_only, "short_only": short_only, "bench_nav": bench_nav,
        "excess": excess, "metrics": metrics, "meta": meta,
    }


# ============================================================
# 因子 → weights_df（策略层公开入口：喂权重引擎/现金引擎对标用）
# ============================================================
def _select_group(factor_panel, mask, selection, direction) -> pd.DataFrame:
    """选股 → group_labels（选中=组号，未选=NaN），供 build_group_weights 复用。

    selection: ('top_n', N) 取因子最高 N 只（合成单组=1）；
               ('top_group', n_groups) 等数量分 n_groups 组取顶组（=n_groups）。
    顶分位 q = 用 ('top_group', round(1/q))。direction: +1 高因子看好 / -1 翻转。
    """
    mode, n = selection
    if mode == "top_group":
        return assign_groups(factor_panel, mask, n, direction)  # 顶组=n_groups，build 时取 max
    if mode == "top_n":
        score = (direction * factor_panel).where(mask)          # 高分=看好；不在池/缺因子→NaN
        counts = score.notna().sum(axis=1)
        bad = counts[counts < n]
        if len(bad):
            d0 = bad.index[0]
            raise ValueError(f"factor_to_weights: 调仓日 {d0.date()} 有效股 {int(bad.iloc[0])} 只 < top_n={n}，无法选满")
        r = score.rank(axis=1, ascending=False, method="first")  # 最优=1
        selected = r <= n
        return selected.astype(float).where(selected)            # 选中=1.0 / 否则 NaN（合成单组）
    raise ValueError(f"非法 selection: {selection!r}（支持 ('top_n',N) | ('top_group',n_groups)）")


def factor_to_weights(factor_wide, rebalance="M", pool=None, exclude_st=False,
                      selection=("top_n", 200), weighting="equal", direction=1,
                      start=None, end=None) -> pd.DataFrame:
    """因子宽表 → 稀疏 weights_df [date, code, weight]（long_only，每调仓日 Σweight=1）。

    只含策略约束（票池/因子非空/可选剔ST/排序选股/加权/调仓频率），**不含当日涨跌停/停牌**
    （那归执行层）。产物可同时喂 run_backtest（权重引擎）与 run_cash_backtest（现金引擎）对标。

    入参:
      factor_wide: date×code 宽表。
      rebalance: 'M'|'W'|'D'|list → make_rebalance_dates。
      pool: None(全市场) / 指数代码 str / date×code bool 表（user 池）。
      exclude_st: 是否剔 ST（策略偏好，默认不剔；非执行禁令）。
      selection: ('top_n',N) | ('top_group',n_groups)（顶分位 q 用 top_group round(1/q)）。
      weighting: 'equal' | 'factor'（'factor' 为截面 rank 归一加权，复用 build_group_weights）。
      direction: +1 高因子看好 / -1 翻转。

    无未来时点约定:
      因子滞后一步——row 标的「成交日 D」用「D 的前一交易日」收盘因子算出。这样 D 当天用
      close/vwap/open 任意 exec_price 成交都不偷看未来（vwap/open 在收盘前，用当日收盘因子=偷看）。
    """
    factor_wide = factor_wide.copy()
    factor_wide.index = pd.to_datetime(factor_wide.index)
    start = pd.Timestamp(start) if start is not None else factor_wide.index.min()
    end = pd.Timestamp(end) if end is not None else factor_wide.index.max()

    calendar = load_calendar()
    signal_dates = make_rebalance_dates(calendar, start, end, rebalance)   # 因子观测日
    # 成交日 = 信号日的下一交易日（无未来：D 收盘因子 → D+1 成交）
    pos = calendar.get_indexer(signal_dates)
    nxt = pos + 1
    keep = nxt < len(calendar)
    signal_dates = signal_dates[keep]
    exec_dates = calendar[nxt[keep]]
    if len(signal_dates) < 2:
        raise ValueError(f"factor_to_weights: 有效调仓日不足 2（start={start.date()} end={end.date()} rebalance={rebalance}）")

    price_df = load_price_df(list(factor_wide.columns), start, end)
    factor_panel = build_factor_panel(factor_wide, price_df["code"].unique(), signal_dates)

    if pool is None:
        universe, members_df, user_mask = "all", None, None
    elif isinstance(pool, str):
        universe, members_df, user_mask = pool, load_index_members(pool), None
    else:
        universe, members_df, user_mask = "user", None, pool
    st_intervals = load_st_intervals() if exclude_st else None
    mask = build_universe_mask(
        factor_panel, price_df, signal_dates, universe,
        members_df=members_df, user_mask=user_mask,
        exclude_st=exclude_st, st_intervals=st_intervals,
    )

    group_labels = _select_group(factor_panel, mask, selection, direction)
    sel_group = int(group_labels.max().max())                  # top_n→1，top_group→n_groups（顶组）
    weights = build_group_weights(group_labels, sel_group, +1, weighting, factor_panel, direction)

    # 信号日 → 成交日 重标（无未来）
    weights["date"] = weights["date"].map(dict(zip(signal_dates, exec_dates)))
    return weights.sort_values(["date", "code"]).reset_index(drop=True)
