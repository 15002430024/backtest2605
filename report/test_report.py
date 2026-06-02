"""
报告层验收测试
  指标正确性：构造已知 NAV，手算 总收益/最大回撤/起止/Calmar，断言一致
  出图落盘：跑真实引擎输出 → plot_dashboard，断言文件落盘、非空
运行: conda activate torch1010 && python test_report.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "engine"))
sys.path.insert(0, str(_ROOT / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # report/（import plot）
from backtest import run_backtest, calc_benchmark  # noqa: E402
from metrics import calc_metrics  # noqa: E402
from plot import plot_dashboard  # noqa: E402

TOL = 1e-9
OUT_DIR = Path(__file__).parent / "_test_output"


def _assert_close(a, e, label):
    if abs(a - e) > TOL:
        raise AssertionError(f"{label}: 实际 {a!r} != 预期 {e!r}")


# ============================================================
# 测试 1: 指标手算 —— NAV=[1.0,1.2,0.9,1.0]，最大回撤/总收益可手算
# ============================================================
def test_metrics_hand():
    dates = pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"])
    nav = pd.Series([1.0, 1.2, 0.9, 1.0], index=dates)
    m = calc_metrics(nav)
    # 总收益 = 1.0/1.0 - 1 = 0
    _assert_close(m["总收益率"], 0.0, "总收益率")
    # cummax=[1,1.2,1.2,1.2]，dd=[0,0,-0.25,-0.1667]，最大回撤=-0.25
    _assert_close(m["最大回撤"], 0.9 / 1.2 - 1.0, "最大回撤")
    # 回撤最深在 D2(0.9)，此前最高点 D1(1.2)
    if m["最大回撤止"] != dates[2]:
        raise AssertionError(f"最大回撤止应为 {dates[2].date()}，实际 {m['最大回撤止']}")
    if m["最大回撤起"] != dates[1]:
        raise AssertionError(f"最大回撤起应为 {dates[1].date()}，实际 {m['最大回撤起']}")


def test_metrics_monotonic():
    """单调上涨：最大回撤=0、Calmar=inf、总收益/年化为正。"""
    dates = pd.date_range("2023-01-03", periods=10, freq="D")
    nav = pd.Series(np.linspace(1.0, 1.5, 10), index=dates)
    m = calc_metrics(nav)
    _assert_close(m["最大回撤"], 0.0, "单调 最大回撤=0")
    if not np.isinf(m["Calmar"]):
        raise AssertionError("单调上涨 Calmar 应为 inf")
    if m["年化收益"] <= 0 or m["总收益率"] <= 0:
        raise AssertionError("单调上涨 年化/总收益应为正")


def test_metrics_benchmark():
    """有基准时算出超额/信息比率/Beta/跟踪误差字段。"""
    dates = pd.date_range("2023-01-03", periods=20, freq="D")
    nav = pd.Series(np.cumprod(1 + np.full(20, 0.01)), index=dates)
    bench = pd.Series(np.cumprod(1 + np.full(20, 0.005)), index=dates)
    m = calc_metrics(nav, benchmark_nav=bench)
    for k in ("超额年化", "信息比率", "Beta", "跟踪误差"):
        if k not in m:
            raise AssertionError(f"有基准应算出 {k}")
    if m["超额年化"] <= 0:
        raise AssertionError("策略每日跑赢基准，超额年化应为正")


def test_metrics_too_short():
    """nav 少于 2 点 → raise。"""
    try:
        calc_metrics(pd.Series([1.0], index=pd.to_datetime(["2023-01-03"])))
    except ValueError:
        return
    raise AssertionError("nav 1 点应 raise")


# ============================================================
# 测试 2: 出图落盘 —— 跑真实引擎 → plot_dashboard
# ============================================================
def _make_backtest_result():
    """用一段构造行情跑引擎，得到含 weights/trade_records 的真实输出。
    取 ~3 年（780 工作日）以便年度柱、滚动夏普/波动等图都有数据。"""
    n = 780
    dates = pd.date_range("2021-01-04", periods=n, freq="B")
    rng = np.arange(n)
    # 两只股票，确定性波动 + 缓慢上行趋势（不用随机，便于复现）
    a = 10 * (1 + 0.06 * np.sin(rng / 30)) * (1 + 0.0003 * rng)
    b = 20 * (1 + 0.05 * np.cos(rng / 25)) * (1 + 0.0002 * rng)
    price = pd.concat([
        pd.DataFrame({"date": dates, "code": "A", "adj_close": a}),
        pd.DataFrame({"date": dates, "code": "B", "adj_close": b}),
    ], ignore_index=True)
    # 每 20 天调一次仓，A/B 在 0.6/0.4 与 0.4/0.6 之间切
    rebs = []
    for i, d in enumerate(dates[::20]):
        wa, wb = (0.6, 0.4) if i % 2 == 0 else (0.4, 0.6)
        rebs += [(d, "A", wa), (d, "B", wb)]
    weights_df = pd.DataFrame(rebs, columns=["date", "code", "weight"])
    cfg = {"buy_cost": 0.0003, "sell_cost": 0.0013, "slippage": 0.001}
    result = run_backtest(weights_df, price, config=cfg, end_date=dates[-1])
    # 基准：A 当指数
    idx = pd.DataFrame({"date": dates, "close": a})
    bench = calc_benchmark(idx, dates[0], dates[-1])
    return result, bench


def test_dashboard_outputs():
    result, bench = _make_backtest_result()
    path = plot_dashboard(result, benchmark_nav=bench, save_dir=str(OUT_DIR),
                          title="测试回测报告")
    expect = ["回测仪表盘.png", "净值曲线.png", "回撤曲线.png",
              "月度收益热力图.png", "超额收益.png", "持仓分析.png"]
    for name in expect:
        f = OUT_DIR / name
        if not f.exists():
            raise AssertionError(f"缺图: {name}")
        if f.stat().st_size < 1000:
            raise AssertionError(f"{name} 文件过小（{f.stat().st_size}B），疑似空图")


def test_dashboard_no_benchmark():
    """缺基准时不报错，仍出仪表盘。"""
    result, _ = _make_backtest_result()
    out2 = OUT_DIR / "no_bench"
    path = plot_dashboard(result, benchmark_nav=None, save_dir=str(out2), title="无基准测试")
    if not (out2 / "回测仪表盘.png").exists():
        raise AssertionError("无基准时仪表盘应仍落盘")


def main():
    tests = [
        ("1 指标手算（回撤/总收益）", test_metrics_hand),
        ("2 指标单调上涨", test_metrics_monotonic),
        ("3 指标含基准", test_metrics_benchmark),
        ("4 nav过短raise", test_metrics_too_short),
        ("5 仪表盘出图落盘", test_dashboard_outputs),
        ("6 缺基准不报错", test_dashboard_no_benchmark),
    ]
    for name, fn in tests:
        fn()
        print(f"  ✓ 测试{name}")
    print(f"\n全部 {len(tests)} 个分析层测试通过")
    print(f"出图已落盘到: {OUT_DIR}")


if __name__ == "__main__":
    main()
