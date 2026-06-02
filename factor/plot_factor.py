"""
因子研究「画」层：消费 run_factor_test 的 result → 出图 + CSV（全中文）。

定位：和引擎侧的 report/plot.py 对称。factor_test.py 只产数据、不画图；
拿到 result 后显式调 plot_factor_report(result, out_dir) 落盘。

公开入口：plot_factor_report(result, out_dir, weighting='equal') -> None

图：① 十分组净值曲线 ② 多空/单独多/单独空 ③ 分组年化收益柱 ④ IC 时间序列+累计IC
   ⑤ 多头超额曲线 ⑥ 滚动 IC/RankIC/ICIR（窗口=调仓期数）⑦ 滚动夏普/波动（窗口=252日）
CSV：十分组净值/超额净值（equal+factor 各一份）、IC 明细、IC 统计。

滚动夏普/波动不手写——复用 analysis.build_report_data（同一份滚动公式，不留副本）。
滚动 IC 窗口单位是「调仓期数」（月频 12 期=1 年），复用 factor_test._PPY，与 IC 年化口径一致。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无显示环境出图
import matplotlib.pyplot as plt

from factor.factor_test import _PPY  # 单一真相源：调仓频率→年化期数，滚动 IC 窗口复用
from analysis.metrics import build_report_data  # 滚动夏普/波动复用，不重写


def _setup_chinese_font():
    """中文字体 + 负号正常显示（所有图共用，调一次）。"""
    matplotlib.rcParams["font.sans-serif"] = ["STHeiti", "Songti SC", "Arial Unicode MS", "Heiti TC", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def plot_factor_report(result: dict, out_dir, weighting: str = "equal") -> None:
    """run_factor_test 的 result → 7 张图 + 4 类 CSV，落盘 out_dir（中文文件名）。

    输入:
      result    — run_factor_test 返回 dict（含 ic/group_nav/long_short/long_only/
                  short_only/excess/bench_nav/metrics/meta）
      out_dir   — 落盘目录（自动创建）
      weighting — 用哪套权重出图：'equal'(等权,默认) | 'factor'(因子加权)。
                  CSV 两套都导，图只画选中的这套。

    输出: 无返回。依赖 analysis.build_report_data（滚动夏普/波动）、factor_test._PPY（滚动IC窗口）。
    """
    if weighting not in result["group_nav"]:
        raise ValueError(f"weighting={weighting!r} 不在 result['group_nav'] 里（可选 {list(result['group_nav'])}）")
    _setup_chinese_font()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gnav = result["group_nav"][weighting]
    n_groups = gnav.shape[1]
    wname = "等权" if weighting == "equal" else "因子加权"

    # ① 十分组净值曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    for g in range(1, n_groups + 1):
        ax.plot(gnav.index, gnav[g], label=f"第{g}组")
    ax.set_title(f"十分组净值曲线（{wname}）"); ax.set_xlabel("日期"); ax.set_ylabel("净值")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "十分组净值曲线.png", dpi=150); plt.close(fig)

    # ② 多空 / 单独多 / 单独空
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(result["long_short"][weighting].index, result["long_short"][weighting], label="多空(100/100)")
    ax.plot(result["long_only"][weighting].index, result["long_only"][weighting], label="单独多")
    ax.plot(result["short_only"][weighting].index, result["short_only"][weighting], label="单独空")
    ax.set_title(f"多空 / 单独多 / 单独空 净值（{wname}）"); ax.set_xlabel("日期"); ax.set_ylabel("净值"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "多空净值.png", dpi=150); plt.close(fig)

    # ③ 分组年化收益柱（直接取 result 已算好的 metrics，不重算）
    annual = [result["metrics"][weighting][f"第{g}组"]["年化收益"] for g in range(1, n_groups + 1)]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([f"第{g}组" for g in range(1, n_groups + 1)], np.array(annual) * 100)
    ax.set_title(f"分组年化收益（{wname}，%）"); ax.set_ylabel("年化收益 %")
    fig.tight_layout(); fig.savefig(out_dir / "分组年化收益.png", dpi=150); plt.close(fig)

    # ④ IC 时间序列 + 累计 IC
    ic = result["ic"]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(ic["ic_series"].index, ic["ic_series"].values, width=15, alpha=0.5, label="每期 IC")
    ax1.set_ylabel("IC"); ax1.set_xlabel("日期")
    ax2 = ax1.twinx(); ax2.plot(ic["ic_cum"].index, ic["ic_cum"].values, color="red", label="累计 IC")
    ax2.set_ylabel("累计 IC")
    ax1.set_title(f"IC 时间序列（IC均值={ic['ic_mean']:.3f}, ICIR年化={ic['ic_ir_annual']:.2f}, t={ic['ic_t']:.2f}）")
    fig.tight_layout(); fig.savefig(out_dir / "IC时间序列.png", dpi=150); plt.close(fig)

    # ⑤ 多头超额曲线（对外部基准）
    ex = result["excess"][weighting]["多头"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ex.index, ex.values, label="多头超额")
    ax.set_title(f"多头超额净值（{wname}，对 {result['meta']['benchmark']}）")
    ax.set_xlabel("日期"); ax.set_ylabel("超额净值"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "多头超额曲线.png", dpi=150); plt.close(fig)

    # ⑥ 滚动 IC / RankIC / ICIR —— 窗口=调仓期数（非 252 天）。别拿 ic_cum 去 rolling。
    rb = result["meta"]["rebalance"]
    n = _PPY.get(rb, 12) if isinstance(rb, str) else 12   # 自定义调仓日列表退 12（与 compute_ic 同口径）
    ic_s, rk_s = ic["ic_series"], ic["rankic_series"]
    roll_ic, roll_rankic = ic_s.rolling(n).mean(), rk_s.rolling(n).mean()
    roll_icir = ic_s.rolling(n).mean() / ic_s.rolling(n).std()
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(roll_ic.index, roll_ic.values, label=f"滚动IC（{n}期）")
    ax1.plot(roll_rankic.index, roll_rankic.values, label=f"滚动RankIC（{n}期）")
    ax1.axhline(0, color="gray", lw=0.8)
    ax1.set_ylabel("滚动 IC / RankIC"); ax1.set_xlabel("日期")
    ax2 = ax1.twinx(); ax2.plot(roll_icir.index, roll_icir.values, color="red", lw=1.0, label=f"滚动ICIR（{n}期）")
    ax2.set_ylabel("滚动 ICIR")
    ax1.set_title(f"滚动 IC / RankIC / ICIR（窗口={n} 个调仓期）")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(out_dir / "滚动IC.png", dpi=150); plt.close(fig)

    # ⑦ 滚动夏普 / 滚动波动 —— 复用 build_report_data（窗口=252 日，日频 NAV）
    curves = {"多空": result["long_short"][weighting], "多头": result["long_only"][weighting]}
    rds = {name: build_report_data({"nav": nav.dropna()}) for name, nav in curves.items()}
    fig, (axs, axv) = plt.subplots(1, 2, figsize=(14, 6))
    for name, rd in rds.items():
        axs.plot(rd.rolling_sharpe.index, rd.rolling_sharpe.values, label=name)
        axv.plot(rd.rolling_vol.index, rd.rolling_vol.values, label=name)
    axs.axhline(0, color="gray", lw=0.8)
    axs.set_title(f"滚动夏普（252 日窗，{wname}）"); axs.set_xlabel("日期"); axs.set_ylabel("滚动夏普"); axs.legend()
    axv.set_title(f"滚动年化波动（252 日窗，{wname}，%）"); axv.set_xlabel("日期"); axv.set_ylabel("波动 %"); axv.legend()
    fig.tight_layout(); fig.savefig(out_dir / "滚动夏普与波动.png", dpi=150); plt.close(fig)

    # CSV（两套权重都导）
    for w in ("equal", "factor"):
        result["group_nav"][w].to_csv(out_dir / f"十分组净值_{w}.csv", encoding="utf-8-sig")
        result["excess"][w].to_csv(out_dir / f"超额净值_{w}.csv", encoding="utf-8-sig")
    pd.DataFrame({"每期IC": ic["ic_series"], "每期RankIC": ic["rankic_series"], "累计IC": ic["ic_cum"]}).to_csv(
        out_dir / "IC明细.csv", encoding="utf-8-sig")
    pd.Series({k: ic[k] for k in ("ic_mean", "ic_ir", "ic_ir_annual", "ic_t", "ic_winrate",
                                  "rankic_mean", "rankic_ir", "rankic_t")}).to_csv(
        out_dir / "IC统计.csv", encoding="utf-8-sig")
