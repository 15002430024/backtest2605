"""
因子测试模块验收（纯函数 + python 直跑，对齐 tests/test_backtest.py 风格）

  python test_factor_test.py          # 合成数据，全 assert
  python test_factor_test.py --real    # 真实数据 sanity（动量因子，打印指标）

合成 case 不依赖缓存：直接把构造好的 price_df / factor 喂给纯函数与引擎。
"""

import sys

import numpy as np
import pandas as pd

from factor.factor_test import (
    build_factor_panel, build_universe_mask, compute_forward_returns, compute_ic,
    assign_groups, build_group_weights, build_long_short_weights,
    backtest_groups, backtest_short_only, backtest_long_short, _bt_end,
)
from engine.backtest import run_backtest


def _assert(cond, msg):
    if not cond:
        raise AssertionError("✗ " + msg)
    print("  ✓", msg)


# ============================================================
# 合成数据：股票 i 每日收益 = slope*i（高因子→高收益，单调）
# ============================================================
def make_synthetic(n_stocks=50, n_days=20, slope=0.0005):
    codes = [f"S{i:02d}" for i in range(n_stocks)]
    cal = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=n_days))
    rows = []
    for i, c in enumerate(codes):
        p = 10.0
        for d in cal:
            rows.append((d, c, p))
            p *= (1 + slope * i)
    price_df = pd.DataFrame(rows, columns=["date", "code", "adj_close"])
    factor = pd.DataFrame({c: float(i) for i, c in enumerate(codes)}, index=cal)
    return codes, cal, price_df, factor


def _prep(price_df, factor, cal):
    rebalance_dates = cal  # 日频调仓
    panel = build_factor_panel(factor, price_df["code"].unique(), rebalance_dates)
    mask = build_universe_mask(panel, price_df, rebalance_dates, "all")
    return rebalance_dates, panel, mask


def test_ic_and_grouping():
    print("[test_ic_and_grouping]")
    codes, cal, price_df, factor = make_synthetic()
    rebalance_dates, panel, mask = _prep(price_df, factor, cal)
    fwd = compute_forward_returns(price_df, rebalance_dates, cal)
    ic = compute_ic(panel, fwd, mask, "D")
    _assert(ic["ic_mean"] > 0.99, f"IC≈+1（单调因子），got {ic['ic_mean']:.4f}")
    _assert(ic["rankic_mean"] > 0.99, f"RankIC≈+1，got {ic['rankic_mean']:.4f}")
    labels = assign_groups(panel, mask, 10, +1)
    top = sorted(labels.iloc[0][labels.iloc[0] == 10].index)
    _assert(all(int(c[1:]) >= 45 for c in top), f"第10组=最高因子(S45-49)，got {top}")
    bot = sorted(labels.iloc[0][labels.iloc[0] == 1].index)
    _assert(all(int(c[1:]) <= 4 for c in bot), f"第1组=最低因子(S00-04)，got {bot}")


def test_ic_ppy_daily_list_matches_string():
    """M2：逐日调仓用 list(交易日) 与字符串 'D' 的 ICIR 年化应一致（原 list 走 365 虚高 1.2 倍）。"""
    print("[test_ic_ppy_daily_list_matches_string]")
    # 用带噪因子/收益（IC 非常数）使 ICIR 有限可比；交易日序列（中位 gap=1）
    cal = pd.bdate_range("2023-01-02", periods=40)
    rng = np.random.RandomState(7)
    cols = [f"S{i:02d}" for i in range(20)]
    panel = pd.DataFrame(rng.randn(40, 20), index=cal, columns=cols)
    fwd = pd.DataFrame(rng.randn(40, 20), index=cal, columns=cols)
    mask = pd.DataFrame(True, index=cal, columns=cols)
    ic_str = compute_ic(panel, fwd, mask, "D")["ic_ir_annual"]
    ic_list = compute_ic(panel, fwd, mask, list(cal))["ic_ir_annual"]
    _assert(np.isfinite(ic_str) and abs(ic_list - ic_str) < 1e-9,
            f"逐日 list 与 'D' 的 ICIR 年化应一致，got list={ic_list:.4f} vs D={ic_str:.4f}")


def test_decile_monotonic_and_long_short():
    print("[test_decile_monotonic_and_long_short]")
    codes, cal, price_df, factor = make_synthetic()
    rebalance_dates, panel, mask = _prep(price_df, factor, cal)
    labels = assign_groups(panel, mask, 10, +1)
    bt_end = _bt_end(cal, cal[-1])
    navs = backtest_groups(labels, panel, price_df, "equal", +1, bt_end, None)
    finals = [navs[g].iloc[-1] for g in range(1, 11)]
    _assert(all(finals[k] < finals[k + 1] for k in range(9)),
            f"十分组终值单调递增：{[round(x,4) for x in finals]}")
    ls = backtest_long_short(labels, panel, price_df, "equal", +1, bt_end)
    _assert(ls.iloc[-1] > 1.0, f"多空净值>1，got {ls.iloc[-1]:.4f}")
    so = backtest_short_only(labels, panel, price_df, "equal", +1, bt_end)
    _assert(navs[10].iloc[-1] > so.iloc[-1], f"单独多({navs[10].iloc[-1]:.3f}) > 单独空({so.iloc[-1]:.3f})")
    # 翻 direction：做多低因子、做空高因子 → 多空净值 < 1
    labels_d = assign_groups(panel, mask, 10, -1)
    ls_d = backtest_long_short(labels_d, panel, price_df, "equal", -1, bt_end)
    _assert(ls_d.iloc[-1] < 1.0, f"翻 direction 后多空净值<1，got {ls_d.iloc[-1]:.4f}")


def test_weight_legality():
    print("[test_weight_legality]")
    codes, cal, price_df, factor = make_synthetic()
    rebalance_dates, panel, mask = _prep(price_df, factor, cal)
    labels = assign_groups(panel, mask, 10, +1)
    for weighting in ("equal", "factor"):
        w = build_group_weights(labels, 10, +1, weighting, panel, +1)
        s = w.groupby("date")["weight"].apply(lambda x: x.abs().sum())
        _assert((s.sub(1).abs() < 1e-9).all(), f"{weighting} long_only 组每期 Σ|w|=1")
        _assert((w["weight"] > 0).all(), f"{weighting} 做多组权重全正")
        _assert(not w["weight"].isna().any(), f"{weighting} 无 NaN 权重")
        _assert(not w.duplicated(["date", "code"]).any(), f"{weighting} 无重复 (date,code)")
    ls = build_long_short_weights(labels, "factor", panel, +1, 10)
    ln = ls[ls.weight > 0].groupby("date")["weight"].sum()
    sh = ls[ls.weight < 0].groupby("date")["weight"].sum()
    _assert((ln.sub(1).abs() < 1e-9).all(), "多空多头和=+1")
    _assert((sh.add(1).abs() < 1e-9).all(), "多空空头和=−1")


def test_avail_excludes_missing_price():
    print("[test_avail_excludes_missing_price]")
    codes, cal, price_df, factor = make_synthetic()
    price_df = price_df.copy()
    price_df.loc[price_df["code"] == "S25", "adj_close"] = np.nan  # S25 有因子但无价
    rebalance_dates, panel, mask = _prep(price_df, factor, cal)
    _assert(not mask["S25"].any(), "无价股票 S25 被 avail mask 剔出池")
    labels = assign_groups(panel, mask, 10, +1)
    w = build_group_weights(labels, 10, +1, "equal", panel, +1)
    _assert("S25" not in set(w["code"]), "S25 不进任何组合权重")


def test_factor_weight_short_worst_most():
    print("[test_factor_weight_short_worst_most]")
    codes, cal, price_df, factor = make_synthetic()
    rebalance_dates, panel, mask = _prep(price_df, factor, cal)
    labels = assign_groups(panel, mask, 10, +1)
    w = build_group_weights(labels, 1, -1, "factor", panel, +1)  # 第1组(最差)做空
    wd = w[w["date"] == w["date"].iloc[0]].set_index("code")["weight"].abs()
    worst = min(wd.index, key=lambda c: int(c[1:]))   # 因子最低=最差
    _assert(wd[worst] >= wd.max() - 1e-12, f"因子加权下空头最差股({worst})权重最大：{wd.to_dict()}")
    _assert((w["weight"] < 0).all(), "空头权重全负")


def test_long_short_is_100_100():
    print("[test_long_short_is_100_100]")
    # A +1、B −1，单个持有日；A +2%、B −1% → 100/100 NAV=1.03（50/50 会是 1.015）
    price_df = pd.DataFrame([
        ("2024-01-01", "A", 10.0), ("2024-01-02", "A", 10.2),
        ("2024-01-01", "B", 10.0), ("2024-01-02", "B", 9.9),
    ], columns=["date", "code", "adj_close"])
    price_df["date"] = pd.to_datetime(price_df["date"])
    w = pd.DataFrame([("2024-01-01", "A", 1.0), ("2024-01-01", "B", -1.0)],
                     columns=["date", "code", "weight"])
    w["date"] = pd.to_datetime(w["date"])
    nav = run_backtest(w, price_df, config={"weight_mode": "long_short"}, end_date="2024-01-02")["nav"]
    _assert(abs(nav.iloc[-1] - 1.03) < 1e-9,
            f"多空 100/100：NAV=1.03（非 50/50 的 1.015），got {nav.iloc[-1]:.6f}")


def test_missing_factor_in_specified_pool_reports():
    print("[test_missing_factor_in_specified_pool_reports]")
    # user 池内某可交易股因子缺失 → 不 raise，剔出有效池，并记录覆盖率问题
    codes, cal, price_df, factor = make_synthetic()
    rebalance_dates = cal
    factor = factor.copy()
    factor.loc[:, "S10"] = np.nan  # S10 因子全缺
    panel = build_factor_panel(factor, price_df["code"].unique(), rebalance_dates)
    user_mask = pd.DataFrame(True, index=rebalance_dates, columns=list(panel.columns) + ["S999"])
    mask, coverage = build_universe_mask(panel, price_df, rebalance_dates, "user",
                                         user_mask=user_mask, return_coverage=True)
    _assert(not mask["S10"].any(), "指定池内因子 NaN → 剔出有效池")
    _assert(coverage["known_missing_factor_cells"] > 0, "覆盖率报告记录已知列因子缺失")
    _assert(coverage["membership_not_in_factor_columns_code_count"] == 1, "覆盖率报告记录指定池缺列股票")
    raised = False
    try:
        build_universe_mask(panel, price_df, rebalance_dates, "user",
                            user_mask=user_mask, strict_missing_factor=True)
    except ValueError as e:
        raised = "因子缺失" in str(e)
    _assert(raised, "strict_missing_factor=True 时仍可强校验 raise")


def run_synthetic():
    print("=" * 60, "\n合成数据验收\n" + "=" * 60)
    test_ic_and_grouping()
    test_ic_ppy_daily_list_matches_string()
    test_decile_monotonic_and_long_short()
    test_weight_legality()
    test_avail_excludes_missing_price()
    test_factor_weight_short_worst_most()
    test_long_short_is_100_100()
    test_missing_factor_in_specified_pool_reports()
    print("\n全部合成验收通过 ✅")


# ============================================================
# 真实数据 sanity（20 日动量/反转），打印不 assert
# ============================================================
def run_real():
    from factor.factor_test import run_factor_test
    from data.loaders import load_calendar, load_price_df
    print("=" * 60, "\n真实数据 sanity：20 日动量，全市场，月频，对 000001.SH 超额\n" + "=" * 60)
    start, end = "2022-01-01", "2022-12-31"
    load_start = "2021-09-01"                    # 20 日动量需前置 lookback
    cal = load_calendar()
    px = load_price_df(None, load_start, end)
    cal_slice = cal[(cal >= pd.Timestamp(load_start)) & (cal <= pd.Timestamp(end))]
    wide = px.pivot(index="date", columns="code", values="adj_close").reindex(index=cal_slice)
    wide = wide[[c for c in wide.columns if not c.endswith(".BJ")]]  # 剔北交所（adj_close 有脏数据 + 因子研究常排除）
    mom = wide / wide.shift(20) - 1.0           # 20 日动量（日历对齐后 shift，符合纪律）
    res = run_factor_test(mom, {"rebalance": "M", "universe": "all", "n_groups": 10,
                                "direction": +1, "benchmark": "000001.SH",
                                "start_date": start, "end_date": end})
    ic = res["ic"]
    print(f"IC均值={ic['ic_mean']:.4f}  RankIC均值={ic['rankic_mean']:.4f}  ICIR(年化)={ic['ic_ir_annual']:.2f}  t={ic['ic_t']:.2f}  IC胜率={ic['ic_winrate']:.2f}")
    m = res["metrics"]["equal"]
    print("十分组（年化 / 夏普 / 终值）：")
    for gi in range(1, 11):
        mm = m[f"第{gi}组"]
        print(f"  第{gi}组: {mm['年化收益']*100:6.2f}%  夏普={mm['夏普']:5.2f}  终值={res['group_nav']['equal'][gi].iloc[-1]:.3f}")
    lsm = m["多空"]
    print(f"多空: 年化={lsm['年化收益']*100:.2f}%  夏普={lsm['夏普']:.2f}  最大回撤={lsm['最大回撤']*100:.1f}%  终值={res['long_short']['equal'].iloc[-1]:.3f}")
    print("（动量在 A 股常表现为反转，低动量组可能跑赢；看是否有单调梯度即可）")


if __name__ == "__main__":
    if "--real" in sys.argv:
        run_real()
    else:
        run_synthetic()
