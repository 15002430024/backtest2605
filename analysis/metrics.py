"""
分析层：绩效指标 + 绘图数据准备（"算"的统一落点，被画图/导出/报告复用）

两个公开产物（都在本文件，因返回类型不同分两个函数）：
  calc_metrics     — NAV → 标量绩效指标 dict（KPI 条、指标表、factor 分组表都用它）
  build_report_data — run_backtest 输出 → ReportData（画图要的全部预计算序列）

设计原则：所有"加工"（resample/rolling/cumprod/cummax/pivot…）都在这一层算好，
plot.py 只接 ReportData 渲染，一行加工都不算。
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# 夏普用 rf=0（因子研究通行做法）；年化用 252 交易日
DEFAULT_PERIODS_PER_YEAR = 252

# pandas 2.2 起月末/年末别名 "M"/"Y" 弃用改为 "ME"/"YE"；按运行版本选别名，兼容 2.0.x 与 2.2+
_PD_GE_22 = tuple(int(p) for p in pd.__version__.split(".")[:2]) >= (2, 2)
_FREQ_MONTH_END = "ME" if _PD_GE_22 else "M"
_FREQ_YEAR_END = "YE" if _PD_GE_22 else "Y"


# ── 私有 helper：算一次，calc_metrics 和 build_report_data 共用 ──

def _daily_returns(nav: pd.Series) -> pd.Series:
    """NAV → 日收益率（去掉首日 NaN）。"""
    return nav.sort_index().pct_change().dropna()


def _drawdown(nav: pd.Series) -> pd.Series:
    """NAV → 回撤序列 = nav / 历史最高 - 1（≤ 0）。"""
    nav = nav.sort_index()
    return nav / nav.cummax() - 1.0


def _align_returns(nav: pd.Series, benchmark_nav: pd.Series):
    """策略与基准按日期交集对齐，返回 (策略日收益, 基准日收益)；无足够交集返回 (None, None)。

    基准在策略区间内若缺交易日（传入了周/月频或缺行基准），交集会变稀疏、pct_change 把跨多日
    收益当 1 日 → 下游按 252 年化时超额年化/IR/跟踪误差/Beta 全部虚高，故缺日直接 fail-fast。
    框架自身通路（calc_benchmark 返回每个交易日的指数 EOD）交集完整，不触发。
    """
    common = nav.index.intersection(benchmark_nav.index)
    if len(common) < 2:
        return None, None
    # 重叠区间内策略有、基准缺的交易日数：>0 即基准稀疏，跨日合并收益会被当 1 日年化
    lo, hi = common.min(), common.max()
    nav_in_range = nav.index[(nav.index >= lo) & (nav.index <= hi)]
    missing = len(nav_in_range) - len(common)
    if missing > 0:
        raise ValueError(
            f"基准在策略区间 [{lo.date()}, {hi.date()}] 内缺 {missing} 个交易日"
            f"（策略 {len(nav_in_range)} 天、基准只覆盖 {len(common)} 天）；跨日合并收益会被当 1 日年化、"
            f"超额指标失真。请传与策略同交易日历（每个交易日都有值）的基准。")
    s_ret = nav.loc[common].pct_change().dropna()
    b_ret = benchmark_nav.loc[common].pct_change().dropna()
    ci = s_ret.index.intersection(b_ret.index)
    return s_ret.loc[ci], b_ret.loc[ci]


def _period_returns(nav: pd.Series, freq: str) -> pd.Series:
    """NAV → 周期收益率（index=周期末时间戳）；首期相对起始 NAV 算，不丢首期。

    freq: pandas resample 频率字符串，"ME"=月末 / "YE"=年末。
    """
    nav = nav.sort_index()
    period_end = nav.resample(freq).last()
    ret = period_end.pct_change()
    ret.iloc[0] = period_end.iloc[0] / nav.iloc[0] - 1.0
    return ret.dropna()


# ── 标量指标 ──

def calc_metrics(
    nav: pd.Series,
    benchmark_nav: pd.Series = None,
    trade_records: list = None,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> dict:
    """
    净值 → 标量绩效指标 dict。

    输入:
      nav            — pd.Series, index=交易日, value=NAV（起点任意）
      benchmark_nav  — pd.Series 或 None；给了则额外算 超额年化/信息比率/Beta/跟踪误差
      trade_records  — list[dict] 或 None；给了则算 年均换手(双边)/调仓次数
      periods_per_year — 年化交易日数，默认 252

    输出: dict，字段见下方组装处。
    边界: nav < 2 点 → raise；最大回撤=0 → Calmar=inf；基准按日期交集对齐。
    """
    if len(nav) < 2:
        raise ValueError(f"calc_metrics: nav 至少需要 2 个点，实际 {len(nav)} 个")

    nav = nav.sort_index()
    ret = _daily_returns(nav)
    n_days = len(ret)

    total_return = nav.iloc[-1] / nav.iloc[0] - 1.0
    ann_return = (nav.iloc[-1] / nav.iloc[0]) ** (periods_per_year / n_days) - 1.0
    ann_vol = ret.std() * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    drawdown = _drawdown(nav)
    max_dd = drawdown.min()
    trough_date = drawdown.idxmin()
    # 峰日取「谷之前最后一次触及该轮高点」的日期：净值二次触顶（未创新高）再大跌时，
    # idxmax 会标到首次触顶日、把中间已收复段并进回撤区间，故取并列最大值的最后一次。
    pre_trough = nav.loc[:trough_date]
    peak_date = pre_trough[pre_trough == pre_trough.max()].index[-1]
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.inf

    win_rate = (ret > 0).mean()
    avg_win = ret[ret > 0].mean()
    avg_loss = ret[ret < 0].mean()
    profit_loss_ratio = abs(avg_win / avg_loss) if (avg_loss < 0 and not np.isnan(avg_win)) else np.nan

    metrics = {
        "总收益率": total_return,
        "年化收益": ann_return,
        "年化波动": ann_vol,
        "夏普": sharpe,
        "最大回撤": max_dd,
        "最大回撤起": peak_date,
        "最大回撤止": trough_date,
        "Calmar": calmar,
        "日胜率": win_rate,
        "盈亏比": profit_loss_ratio,
        "交易天数": n_days,
    }

    if trade_records:
        # turnover 来自 run_backtest 的 Σ|Δw|=买+卖**双边**口径；年均换手即年化双边换手。
        # 注意现金引擎 turnover_cap 是**单边各自卡**口径（买边总额≤cap 且卖边总额≤cap，两边独立），
        # 与此处双边 Σ|Δw| 口径不同，对比时勿混。
        total_turnover = sum(r["turnover"] for r in trade_records)
        years = n_days / periods_per_year
        metrics["年均换手(双边)"] = total_turnover / years if years > 0 else np.nan
        metrics["调仓次数"] = len(trade_records)

    if benchmark_nav is not None and len(benchmark_nav) >= 2:
        s_ret, b_ret = _align_returns(nav.sort_index(), benchmark_nav.sort_index())
        if s_ret is not None:
            # 逐日算术超额累乘口径（与 build_report_data、factor.compute_excess 统一）：
            # 日超额 α=r_s−r_b，超额净值 ∏(1+α)，年化用复利、回撤直接对超额净值求
            excess = s_ret - b_ret
            excess_nav = (1 + excess).cumprod()
            n_ex = len(excess)
            te = excess.std() * np.sqrt(periods_per_year)
            metrics["超额年化"] = excess_nav.iloc[-1] ** (periods_per_year / n_ex) - 1 if n_ex else np.nan
            # 回撤序列补 1.0 起点锚（excess_nav 首点已是 1+α₁，不含 1.0）：否则首段超额回撤从 1+α₁
            # 而非 1.0 量起、低估≈|首日超额|（与 factor.compute_excess 的 fillna(0) 锚点口径统一）。
            ex_anchored = np.concatenate([[1.0], excess_nav.values])
            metrics["超额最大回撤"] = (ex_anchored / np.maximum.accumulate(ex_anchored) - 1).min()
            metrics["跟踪误差"] = te
            metrics["信息比率"] = excess.mean() / excess.std() * np.sqrt(periods_per_year) if excess.std() > 0 else np.nan
            var_b = b_ret.var()
            metrics["Beta"] = s_ret.cov(b_ret) / var_b if var_b > 0 else np.nan

    return metrics


# ── 绘图数据（画图模块消费的全部契约）──

@dataclass
class ReportData:
    """
    画图模块（report/plot.py）消费的全部数据，已在本层算好，plot 不再加工。

    将来把 build_report_data 整体搬进引擎时，plot 一行不用改 —— 这个 dataclass
    的字段定义 = 引擎需要额外输出的绘图数据契约。

    字段（均已预计算）:
      metrics        标量指标 dict（= calc_metrics 输出，KPI 条/指标表用）
      nav_norm       归一化策略净值（起点 1.0）
      bench_norm     归一化基准净值（对齐到 nav 日期，起点 1.0）；无基准则 None
      drawdown       回撤序列（≤0，单位：比例）
      monthly_table  月度收益透视表（index=年, columns=1..12, 值=%）
      annual_strategy 年度策略收益（index=年, 值=%）
      annual_bench    年度基准收益（index=年, 值=%）；无基准则 None
      rolling_sharpe 滚动夏普序列
      rolling_vol    滚动年化波动序列（%）
      excess_cum     超额收益累计曲线（%，逐日算术超额 α=r_s−r_b 累乘 ∏(1+α)−1）；无基准则 None
      turnover_dates / turnover_pct / cost_cum_bps  换手率柱(%) + 累计成本(bps)；无调仓则空
      blocked_counts 被拦交易按原因计数 Series；无则空 Series
      daily_ret_pct  日收益率序列（%，画分布直方图）
      position_count 每日持仓数量序列；无 weights 则 None
      weights_top    持仓权重热力图数据（index=top-N 股, columns=日期）；无 weights 则 None
    """
    metrics: dict
    nav_norm: pd.Series
    drawdown: pd.Series
    monthly_table: pd.DataFrame
    annual_strategy: pd.Series
    rolling_sharpe: pd.Series
    rolling_vol: pd.Series
    daily_ret_pct: pd.Series
    blocked_counts: pd.Series
    turnover_dates: list = field(default_factory=list)
    turnover_pct: list = field(default_factory=list)
    cost_cum_bps: list = field(default_factory=list)
    bench_norm: pd.Series = None
    annual_bench: pd.Series = None
    excess_cum: pd.Series = None
    position_count: pd.Series = None
    weights_top: pd.DataFrame = None


def build_report_data(
    result: dict,
    benchmark_nav: pd.Series = None,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    rolling_window: int = DEFAULT_PERIODS_PER_YEAR,
    heatmap_top_n: int = 15,
) -> ReportData:
    """
    run_backtest 输出 → ReportData（画图要的全部预计算序列）。

    输入:
      result        — run_backtest 输出 dict（含 nav, weights, trade_records, blocked_trades）
      benchmark_nav — pd.Series 或 None
      periods_per_year / rolling_window / heatmap_top_n — 年化天数 / 滚动窗 / 热力图选股数

    输出: ReportData（见其 docstring）

    依赖: calc_metrics、本模块私有 helper。所有 resample/rolling/cumprod/pivot 都在这里做。
    """
    nav = result["nav"].sort_index()
    weights = result.get("weights")
    trade_records = result.get("trade_records")
    blocked_trades = result.get("blocked_trades")

    metrics = calc_metrics(nav, benchmark_nav, trade_records, periods_per_year)

    nav_norm = nav / nav.iloc[0]
    drawdown = _drawdown(nav_norm)
    ret = _daily_returns(nav)

    # 月度收益透视表（年×月，%）；首月相对起始 NAV，不丢首月
    monthly = _period_returns(nav, _FREQ_MONTH_END)
    mdf = pd.DataFrame({"年": monthly.index.year, "月": monthly.index.month,
                        "收益": monthly.values * 100})
    monthly_table = mdf.pivot(index="年", columns="月", values="收益").reindex(columns=range(1, 13))

    # 年度收益（%）；首年相对起始 NAV，不丢首年（单年回测也有值）
    annual_strategy = _period_returns(nav, _FREQ_YEAR_END) * 100
    annual_strategy.index = annual_strategy.index.year

    # 滚动夏普 / 波动
    roll = ret.rolling(rolling_window)
    rolling_sharpe = (roll.mean() * periods_per_year) / (roll.std() * np.sqrt(periods_per_year))
    rolling_vol = ret.rolling(rolling_window).std() * np.sqrt(periods_per_year) * 100

    daily_ret_pct = ret * 100

    # 被拦交易计数
    if blocked_trades:
        blocked_counts = pd.Series([b["reason"] for b in blocked_trades]).value_counts()
    else:
        blocked_counts = pd.Series(dtype=int)

    # 换手率 + 累计成本
    turnover_dates, turnover_pct, cost_cum_bps = [], [], []
    if trade_records:
        turnover_dates = [r["date"] for r in trade_records]
        turnover_pct = [r["turnover"] * 100 for r in trade_records]
        cost_cum_bps = list(np.cumsum([r["cost"] * 10000 for r in trade_records]))

    # 基准相关
    bench_norm = annual_bench = excess_cum = None
    if benchmark_nav is not None:
        b = benchmark_nav.reindex(nav.index).ffill()
        anchor = b.first_valid_index()  # 基准首日可能晚于策略首日，ffill 填不动开头缺口
        if anchor is None:
            raise ValueError(
                f"build_report_data: 基准与策略日期无交集，"
                f"基准 {benchmark_nav.index.min().date()}~{benchmark_nav.index.max().date()}，"
                f"策略 {nav.index.min().date()}~{nav.index.max().date()}")
        # 基准重锚到「策略在 anchor 日的净值水平」（而非恒 1.0）：基准首日晚于策略首日时，
        # 两线在 anchor 相交、之后缺口=真实超额；否则净值对比图把策略前段涨幅画成伪超额。
        # 常见情形（基准覆盖全程，anchor=策略首日）下 nav_norm.loc[anchor]=1.0，与原行为一致。
        bench_norm = b / b.loc[anchor] * nav_norm.loc[anchor]
        # 年度基准截断到策略区间再算（与 annual_strategy 同口径）：否则首年用基准整年 vs 策略半年，
        # 基准比策略覆盖长时凭空显示策略跑输。
        bench_in_range = benchmark_nav[(benchmark_nav.index >= nav.index.min()) &
                                       (benchmark_nav.index <= nav.index.max())]
        ab = _period_returns(bench_in_range, _FREQ_YEAR_END) * 100
        ab.index = ab.index.year
        annual_bench = ab.reindex(annual_strategy.index)
        s_ret, b_ret = _align_returns(nav, benchmark_nav.sort_index())
        if s_ret is not None:
            # 逐日算术超额累乘（α=r_s−r_b，∏(1+α)−1），与 calc_metrics 超额净值同口径
            excess_cum = ((1 + (s_ret - b_ret)).cumprod() - 1) * 100

    # 持仓相关
    position_count = weights_top = None
    if weights is not None and not weights.empty:
        position_count = (weights.abs() > 1e-9).sum(axis=1)
        top = weights.abs().mean().nlargest(heatmap_top_n).index
        weights_top = weights[top].T  # 行=股票, 列=日期

    return ReportData(
        metrics=metrics, nav_norm=nav_norm, drawdown=drawdown,
        monthly_table=monthly_table, annual_strategy=annual_strategy,
        rolling_sharpe=rolling_sharpe, rolling_vol=rolling_vol,
        daily_ret_pct=daily_ret_pct, blocked_counts=blocked_counts,
        turnover_dates=turnover_dates, turnover_pct=turnover_pct, cost_cum_bps=cost_cum_bps,
        bench_norm=bench_norm, annual_bench=annual_bench, excess_cum=excess_cum,
        position_count=position_count, weights_top=weights_top,
    )
