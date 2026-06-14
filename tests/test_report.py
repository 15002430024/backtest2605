"""
报告层验收测试
  指标正确性：构造已知 NAV，手算 总收益/最大回撤/起止/Calmar，断言一致
  出图落盘：跑真实引擎输出 → plot_dashboard，断言文件落盘、非空
运行: conda activate torch1010 && python tests/test_report.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from engine.backtest import run_backtest, calc_benchmark
from analysis.metrics import calc_metrics
from report.plot import plot_dashboard

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


# ---- 本轮指标层修复回归（M1/M3/M5/M6）----
def test_max_dd_peak_is_round_local_high():
    """M6：净值二次触顶再大跌，最大回撤起取本轮峰（末次触顶）而非首次触顶。"""
    nav = pd.Series([1.0, 0.9, 1.0, 0.5],
                    index=pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"]))
    m = calc_metrics(nav)
    _assert_close(m["最大回撤"], -0.5, "最大回撤幅度")
    if m["最大回撤起"] != pd.Timestamp("2023-01-05"):
        raise AssertionError(f"最大回撤起应为本轮峰 2023-01-05，实得 {m['最大回撤起'].date()}")


def test_excess_dd_has_unit_anchor():
    """M5：超额净值补 1.0 起点锚——首日超额 -5% 后走平，超额最大回撤应为 -5%（原报 0）。"""
    idx = pd.date_range("2023-01-03", periods=11, freq="D")
    s = pd.Series([1.0] + [0.95] * 10, index=idx)
    b = pd.Series([1.0] * 11, index=idx)
    m = calc_metrics(s, benchmark_nav=b)
    _assert_close(m["超额最大回撤"], -0.05, "超额最大回撤")


def test_align_returns_raises_on_sparse_benchmark():
    """M1：基准在策略区间内缺交易日 → raise（跨日合并收益当 1 日年化会失真）。"""
    idx = pd.bdate_range("2023-01-02", periods=60)
    nav = pd.Series(np.linspace(1.0, 1.2, 60), index=idx)
    bench_sparse = nav.iloc[::5]                       # 每 5 个交易日采样一次（稀疏）
    try:
        calc_metrics(nav, benchmark_nav=bench_sparse)
    except ValueError:
        return
    raise AssertionError("稀疏基准应 raise")


def test_annual_bench_truncated_to_strategy():
    """M3：年度基准截断到策略区间——同源数据策略半年 vs 基准应同口径，不显示凭空跑输。"""
    from analysis.metrics import build_report_data
    cal = pd.bdate_range("2020-01-02", "2021-12-31")
    bench = pd.Series(np.cumprod(1 + np.full(len(cal), 0.0005)), index=cal)
    strat = bench[cal >= "2020-07-01"]                # 同源、策略年中起步
    rd = build_report_data({"nav": strat}, benchmark_nav=bench)
    _assert_close(rd.annual_bench[2020], rd.annual_strategy[2020], "年度基准首年(同源应等于策略)")


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
        ("7 M6 回撤起取本轮峰", test_max_dd_peak_is_round_local_high),
        ("8 M5 超额回撤补锚", test_excess_dd_has_unit_anchor),
        ("9 M1 稀疏基准raise", test_align_returns_raises_on_sparse_benchmark),
        ("10 M3 年度基准截断", test_annual_bench_truncated_to_strategy),
    ]
    for name, fn in tests:
        fn()
        print(f"  ✓ 测试{name}")
    print(f"\n全部 {len(tests)} 个分析层测试通过")
    print(f"出图已落盘到: {OUT_DIR}")


if __name__ == "__main__":
    main()
