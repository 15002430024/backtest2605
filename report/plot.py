"""
报告层：机构研报风可视化（纯渲染，一行数据加工都不算）

公开函数：
  plot_dashboard — run_backtest 输出 → 一页仪表盘大拼图 + 关键单图（全中文）

所有"算"在 analysis/metrics.py 的 build_report_data 里完成；本文件子图只接
ReportData 的已算好字段 + ax 渲染。风格：白底、深蓝主色、暖灰基准、细网格、
去顶/右边框、顶部 KPI 条。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无显示环境出图
import matplotlib.pyplot as plt
from matplotlib import gridspec

from analysis.metrics import build_report_data
from analysis.plot_style import setup_chinese_font  # 跨平台中文字体，单一真相源

# ── 配色（机构研报风）──
C_STRATEGY = "#1f3a5f"   # 策略：深蓝
C_BENCH = "#9b8b70"      # 基准：暖灰
C_UP = "#c0392b"         # 涨/正：克制红
C_DOWN = "#27ae60"       # 跌/负：克制绿
C_DD = "#c0392b"         # 回撤阴影
C_GRID = "#cccccc"
C_KPI_BG = "#1f3a5f"


def _apply_style():
    """统一 rcParams（中文字体 + 研报风）。所有图共用，调一次。"""
    setup_chinese_font()  # 跨平台中文字体 + 负号（analysis.plot_style 单一真相源）
    matplotlib.rcParams["axes.edgecolor"] = "#666666"
    matplotlib.rcParams["axes.linewidth"] = 0.8
    matplotlib.rcParams["figure.facecolor"] = "white"
    matplotlib.rcParams["axes.facecolor"] = "white"


def _style_ax(ax):
    """单个 ax 的研报风：去顶/右边框，细网格。"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=C_GRID, alpha=0.3, linewidth=0.6)
    ax.tick_params(labelsize=8)


# ── 各子图（纯渲染：接 ReportData 已算好字段 + ax，不做任何加工）──

def _plot_nav(ax, rd):
    """图1：净值曲线（策略 vs 基准）+ 回撤阴影。"""
    ax.plot(rd.nav_norm.index, rd.nav_norm.values, color=C_STRATEGY, lw=1.6, label="策略")
    if rd.bench_norm is not None:
        ax.plot(rd.bench_norm.index, rd.bench_norm.values, color=C_BENCH, lw=1.3, ls="--", label="基准")
    ax2 = ax.twinx()
    ax2.fill_between(rd.drawdown.index, rd.drawdown.values, 0, color=C_DD, alpha=0.12)
    ax2.set_ylim(rd.drawdown.min() * 3, 0)
    ax2.set_yticks([])
    ax.set_title("净值曲线（策略 vs 基准）", fontsize=11, fontweight="bold", color=C_STRATEGY)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    _style_ax(ax)


def _plot_drawdown(ax, rd):
    """图2：回撤水下图。"""
    dd = rd.drawdown * 100
    ax.fill_between(dd.index, dd.values, 0, color=C_DD, alpha=0.35)
    ax.plot(dd.index, dd.values, color=C_DD, lw=0.8)
    ax.set_title("回撤曲线（%）", fontsize=11, fontweight="bold", color=C_STRATEGY)
    _style_ax(ax)


def _plot_monthly_heatmap(ax, rd):
    """图3：月度收益热力图（年×月）。"""
    pivot = rd.monthly_table
    if pivot is None or pivot.empty:
        ax.text(0.5, 0.5, "数据不足", ha="center", va="center"); ax.axis("off"); return
    data = pivot.values.astype(float)
    vmax = np.nanmax(np.abs(data)) if not np.all(np.isnan(data)) else 1.0
    ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(12)); ax.set_xticklabels([f"{m}月" for m in range(1, 13)], fontsize=7)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=7)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=6, color="black")
    ax.set_title("月度收益热力图（%）", fontsize=11, fontweight="bold", color=C_STRATEGY)


def _plot_annual_bar(ax, rd):
    """图4：年度收益柱（策略 vs 基准）。"""
    s = rd.annual_strategy
    x = np.arange(len(s))
    has_b = rd.annual_bench is not None
    w = 0.38 if has_b else 0.6
    ax.bar(x - (w / 2 if has_b else 0), s.values, w, color=C_STRATEGY, label="策略")
    if has_b:
        ax.bar(x + w / 2, rd.annual_bench.values, w, color=C_BENCH, label="基准")
        ax.legend(fontsize=8, frameon=False)
    ax.axhline(0, color="#666666", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(s.index, fontsize=7)
    ax.set_title("年度收益（%）", fontsize=11, fontweight="bold", color=C_STRATEGY)
    _style_ax(ax)


def _plot_rolling_sharpe(ax, rd):
    """图5：滚动夏普。"""
    rs = rd.rolling_sharpe
    ax.plot(rs.index, rs.values, color=C_STRATEGY, lw=1.2)
    ax.axhline(0, color="#666666", lw=0.8)
    ax.set_title("滚动夏普", fontsize=11, fontweight="bold", color=C_STRATEGY)
    _style_ax(ax)


def _plot_rolling_vol(ax, rd):
    """图6：滚动年化波动。"""
    rv = rd.rolling_vol
    ax.plot(rv.index, rv.values, color="#8e44ad", lw=1.2)
    ax.set_title("滚动年化波动（%）", fontsize=11, fontweight="bold", color=C_STRATEGY)
    _style_ax(ax)


def _plot_excess(ax, rd):
    """图7：超额收益累计曲线（策略净值/基准净值−1，相对超额）。"""
    if rd.excess_cum is None:
        ax.text(0.5, 0.5, "无基准", ha="center", va="center", fontsize=10); ax.axis("off"); return
    cum = rd.excess_cum
    ax.fill_between(cum.index, cum.values, 0, color=C_STRATEGY, alpha=0.15)
    ax.plot(cum.index, cum.values, color=C_STRATEGY, lw=1.4)
    ax.axhline(0, color="#666666", lw=0.8)
    ax.set_title("超额收益累计（%）", fontsize=11, fontweight="bold", color=C_STRATEGY)
    _style_ax(ax)


def _plot_turnover_cost(ax, rd):
    """图8：换手率 + 累计成本（双轴）。"""
    if not rd.turnover_dates:
        ax.text(0.5, 0.5, "无调仓记录", ha="center", va="center", fontsize=10); ax.axis("off"); return
    ax.bar(rd.turnover_dates, rd.turnover_pct, color=C_STRATEGY, alpha=0.6, width=8, label="换手率(%)")
    ax.set_ylabel("换手率（%）", fontsize=8, color=C_STRATEGY)
    ax2 = ax.twinx()
    ax2.plot(rd.turnover_dates, rd.cost_cum_bps, color=C_UP, lw=1.2, label="累计成本(bps)")
    ax2.set_ylabel("累计成本（bps）", fontsize=8, color=C_UP)
    ax2.spines["top"].set_visible(False)
    ax.set_title("换手率与累计成本", fontsize=11, fontweight="bold", color=C_STRATEGY)
    _style_ax(ax)


def _plot_blocked(ax, rd):
    """图9：被拦交易分析（按原因计数）。"""
    counts = rd.blocked_counts
    if counts is None or len(counts) == 0:
        ax.text(0.5, 0.5, "无被拦交易\n（未开可行性过滤或全部成交）",
                ha="center", va="center", fontsize=10, color="#888888"); ax.axis("off")
        ax.set_title("被拦交易分析", fontsize=11, fontweight="bold", color=C_STRATEGY)
        return
    colors = {"涨停": C_UP, "跌停": C_DOWN, "停牌": "#7f8c8d", "无数据": "#bdc3c7", "容量不足": "#e67e22"}
    bar_colors = [colors.get(r, C_STRATEGY) for r in counts.index]
    ax.bar(counts.index, counts.values, color=bar_colors)
    ax.set_title("被拦交易（按原因计数）", fontsize=11, fontweight="bold", color=C_STRATEGY)
    _style_ax(ax)


def _plot_return_dist(ax, rd):
    """图10：日收益分布直方图 + 正态参考。"""
    ret = rd.daily_ret_pct
    ax.hist(ret.values, bins=50, color=C_STRATEGY, alpha=0.6, density=True)
    mu, sigma = ret.mean(), ret.std()
    xs = np.linspace(ret.min(), ret.max(), 200)
    ax.plot(xs, (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2),
            color=C_UP, lw=1.4, label="正态参考")
    ax.axvline(mu, color="#666666", lw=0.8, ls="--")
    ax.set_title("日收益分布（%）", fontsize=11, fontweight="bold", color=C_STRATEGY)
    ax.legend(fontsize=8, frameon=False)
    _style_ax(ax)


def _plot_position_count(ax, rd):
    """图11：持仓数量曲线。"""
    pc = rd.position_count
    if pc is None or pc.empty:
        ax.text(0.5, 0.5, "无持仓数据", ha="center", va="center"); ax.axis("off"); return
    ax.fill_between(pc.index, pc.values, 0, color=C_STRATEGY, alpha=0.2)
    ax.plot(pc.index, pc.values, color=C_STRATEGY, lw=1.2)
    ax.set_title("持仓数量", fontsize=11, fontweight="bold", color=C_STRATEGY)
    _style_ax(ax)


def _plot_weights_heatmap(ax, rd):
    """图12：持仓权重热力图（top-N 股 × 时间）。"""
    sub = rd.weights_top
    if sub is None or sub.empty:
        ax.text(0.5, 0.5, "无持仓数据", ha="center", va="center"); ax.axis("off"); return
    vmax = np.nanmax(np.abs(sub.values)) if sub.size else 1.0
    ax.imshow(sub.values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(sub.index))); ax.set_yticklabels(sub.index, fontsize=6)
    n = sub.shape[1]
    step = max(1, n // 6)
    xticks = list(range(0, n, step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([pd.Timestamp(sub.columns[i]).strftime("%y-%m") for i in xticks], fontsize=6)
    ax.set_title(f"持仓权重热力图（前{len(sub.index)}大）", fontsize=11, fontweight="bold", color=C_STRATEGY)


def _plot_metrics_table(ax, metrics):
    """指标汇总表（图形化）。"""
    ax.axis("off")
    rows = [
        ("总收益率", f"{metrics['总收益率']*100:.2f}%"),
        ("年化收益", f"{metrics['年化收益']*100:.2f}%"),
        ("年化波动", f"{metrics['年化波动']*100:.2f}%"),
        ("夏普比率", f"{metrics['夏普']:.2f}"),
        ("最大回撤", f"{metrics['最大回撤']*100:.2f}%"),
        ("Calmar", f"{metrics['Calmar']:.2f}" if np.isfinite(metrics['Calmar']) else "∞"),
        ("日胜率", f"{metrics['日胜率']*100:.1f}%"),
        ("盈亏比", f"{metrics['盈亏比']:.2f}" if not np.isnan(metrics['盈亏比']) else "—"),
    ]
    for k in ("超额年化", "信息比率", "Beta", "年均换手"):
        if k in metrics:
            v = metrics[k]
            if k in ("超额年化", "年均换手"):
                rows.append((k, f"{v*100:.1f}%"))
            else:
                rows.append((k, f"{v:.2f}"))
    table = ax.table(cellText=[[k, v] for k, v in rows],
                     colLabels=["指标", "数值"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_facecolor(C_KPI_BG); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f5f5f5")
    ax.set_title("绩效指标汇总", fontsize=11, fontweight="bold", color=C_STRATEGY, pad=20)


def _kpi_banner(fig, metrics, title):
    """顶部 KPI 条：标题 + 4 个关键数字。"""
    fig.text(0.5, 0.975, title, ha="center", fontsize=16, fontweight="bold", color=C_KPI_BG)
    kpis = [
        ("年化收益", f"{metrics['年化收益']*100:.1f}%"),
        ("夏普", f"{metrics['夏普']:.2f}"),
        ("最大回撤", f"{metrics['最大回撤']*100:.1f}%"),
        ("Calmar", f"{metrics['Calmar']:.2f}" if np.isfinite(metrics['Calmar']) else "∞"),
    ]
    for i, (k, v) in enumerate(kpis):
        x = 0.18 + i * 0.21
        fig.text(x, 0.945, v, ha="center", fontsize=15, fontweight="bold", color=C_STRATEGY)
        fig.text(x, 0.925, k, ha="center", fontsize=9, color="#666666")


# ── 主入口 ──

def plot_dashboard(result: dict, benchmark_nav: pd.Series = None,
                   save_dir: str = ".", title: str = "回测分析报告",
                   periods_per_year: int = 252, rolling_window: int = 252) -> str:
    """
    一次性出仪表盘大拼图 + 关键单图，全部存 save_dir（中文文件名）。

    输入:
      result        — run_backtest 输出 dict
      benchmark_nav — pd.Series 或 None；None 时跳过基准相关图
      save_dir / title / periods_per_year / rolling_window — 输出目录/标题/年化天数/滚动窗

    输出: 仪表盘大图路径（str）。同时单独存 净值曲线/回撤曲线/月度收益热力图/超额收益.png

    依赖: analysis.metrics.build_report_data（所有数据加工在那里做，本函数只渲染）
    """
    _apply_style()
    rd = build_report_data(result, benchmark_nav, periods_per_year, rolling_window)
    metrics = rd.metrics

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # ── 仪表盘大拼图：4 行 × 3 列 ──
    fig = plt.figure(figsize=(18, 22))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.25,
                           top=0.90, bottom=0.04, left=0.06, right=0.96)
    _kpi_banner(fig, metrics, title)

    _plot_nav(fig.add_subplot(gs[0, :2]), rd)
    _plot_metrics_table(fig.add_subplot(gs[0, 2]), metrics)
    _plot_drawdown(fig.add_subplot(gs[1, 0]), rd)
    _plot_monthly_heatmap(fig.add_subplot(gs[1, 1]), rd)
    _plot_annual_bar(fig.add_subplot(gs[1, 2]), rd)
    _plot_rolling_sharpe(fig.add_subplot(gs[2, 0]), rd)
    _plot_rolling_vol(fig.add_subplot(gs[2, 1]), rd)
    _plot_excess(fig.add_subplot(gs[2, 2]), rd)
    _plot_turnover_cost(fig.add_subplot(gs[3, 0]), rd)
    _plot_blocked(fig.add_subplot(gs[3, 1]), rd)
    _plot_return_dist(fig.add_subplot(gs[3, 2]), rd)

    dashboard_path = save_path / "回测仪表盘.png"
    fig.savefig(dashboard_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ── 持仓类图（单独一张）──
    if rd.position_count is not None:
        fig2 = plt.figure(figsize=(16, 6))
        gs2 = gridspec.GridSpec(1, 2, figure=fig2, wspace=0.2)
        _plot_position_count(fig2.add_subplot(gs2[0, 0]), rd)
        _plot_weights_heatmap(fig2.add_subplot(gs2[0, 1]), rd)
        fig2.savefig(save_path / "持仓分析.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig2)

    # ── 关键单图各自单独存 ──
    _save_single(_plot_nav, save_path / "净值曲线.png", rd)
    _save_single(_plot_drawdown, save_path / "回撤曲线.png", rd)
    _save_single(_plot_monthly_heatmap, save_path / "月度收益热力图.png", rd)
    if rd.excess_cum is not None:
        _save_single(_plot_excess, save_path / "超额收益.png", rd)

    return str(dashboard_path)


def _save_single(plot_fn, path, rd):
    """把单个子图函数画到独立 figure 并存盘。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_fn(ax, rd)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
