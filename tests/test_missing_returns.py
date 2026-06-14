"""
缺失/NaN 收益处理的「当前行为」回归测试（钉死症状，便于将来改 Fail-Fast 时对照）。

现状：run_backtest 主循环算 portfolio_return 与 update_weights 漂移时，对「持仓票当天在
price_df 没行 / adj_close 为 NaN」一律 `ret=0`、不报错（backtest.py:99-101 与 519-523）。
本测试断言这一现状——不 raise、把缺失那天当 0 推进 NAV。等加了「无法解释的缺失要 raise」后，
再在此补反向用例（无法解释→raise、停牌→放行 0）。

运行: conda activate torch1010 && python tests/test_missing_returns.py
"""
import numpy as np
import pandas as pd

from engine.backtest import run_backtest


def _assert(cond, msg):
    if not cond:
        raise AssertionError("✗ " + msg)
    print("  ✓", msg)


def _price(rows):
    df = pd.DataFrame(rows, columns=["date", "code", "adj_close"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def _hold(code):
    w = pd.DataFrame([("2023-01-03", code, 1.0)], columns=["date", "code", "weight"])
    w["date"] = pd.to_datetime(w["date"])
    return w


def test_missing_row_held_treated_as_zero():
    """持仓票某天在 price_df 没行 → 该天收益当 0、NAV 不动、不 raise（症状钉死）。"""
    print("[test_missing_row_held_treated_as_zero]")
    # A 全程有价（让 01-04 成为交易日）；B 全仓持有，但 01-04 没行（缺行），01-05 有
    price_df = _price([
        ("2023-01-03", "A", 10.0), ("2023-01-04", "A", 11.0), ("2023-01-05", "A", 12.0),
        ("2023-01-03", "B", 20.0),                            ("2023-01-05", "B", 22.0),
    ])
    res = run_backtest(_hold("B"), price_df, end_date="2023-01-05")  # 不应抛异常
    nav = res["nav"]
    d = pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"])
    _assert(abs(nav.loc[d[0]] - 1.0) < 1e-12, f"d0 收盘调仓后 NAV=1.0，got {nav.loc[d[0]]:.6f}")
    _assert(abs(nav.loc[d[1]] - 1.0) < 1e-12, f"d1 B 缺行→收益当 0→NAV 仍 1.0（症状），got {nav.loc[d[1]]:.6f}")
    # d2 B 又有行：pct_change 跨缺行算 22/20-1=+0.10（这天非缺失，正常推进）
    _assert(abs(nav.loc[d[2]] - 1.10) < 1e-12, f"d2 B=22→NAV=1.10，got {nav.loc[d[2]]:.6f}")
    _assert(nav.notna().all(), "全程 NAV 无 NaN（缺失被当 0，但已记进 missing_log）")
    ml = res["missing_log"]
    _assert(any(m["code"] == "B" and m["reason"] == "no_row" and m["date"] == d[1] for m in ml),
            f"missing_log 记下 d1 B 缺行(no_row)，got {ml}")


def test_nan_adjclose_held_treated_as_missing_row():
    """持仓票 adj_close=NaN → 等同缺行（no_row）：不 raise、当 0、NAV 不动、记 missing_log。

    发现2 修复后：calc_daily_returns 先丢弃 NaN 价行（避免 pct_change pad 把 NaN 价当 0 收益、
    复牌日算成跨段全程收益）。NaN 价的 (date,code) 在下游不再有行 → 走 no_row 分支（不是 nan_return）。
    NAV 逐点不变（缺失那天仍按 0 推进）。
    """
    print("[test_nan_adjclose_held_treated_as_missing_row]")
    price_df = _price([
        ("2023-01-03", "A", 10.0), ("2023-01-04", "A", 11.0), ("2023-01-05", "A", 12.0),
        ("2023-01-03", "B", 20.0), ("2023-01-04", "B", np.nan), ("2023-01-05", "B", 22.0),
    ])
    res = run_backtest(_hold("B"), price_df, end_date="2023-01-05")  # 不应抛异常
    nav = res["nav"]
    d = pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"])
    _assert(nav.notna().all(), "adj_close=NaN 未让 NAV 出现 NaN（被当 0，但已记进 missing_log）")
    _assert(abs(nav.loc[d[1]] - 1.0) < 1e-12, f"d1 NaN 价那天 NAV 不动（当 0），got {nav.loc[d[1]]:.6f}")
    # d2 B 复牌：22/20-1=+0.10（NaN 价行被丢，pct_change 跨缺行不再 pad 成 0），NAV=1.10
    _assert(abs(nav.loc[d[2]] - 1.10) < 1e-12, f"d2 B=22→NAV=1.10，got {nav.loc[d[2]]:.6f}")
    ml = res["missing_log"]
    _assert(any(m["code"] == "B" and m["reason"] == "no_row" and m["date"] == d[1] for m in ml),
            f"missing_log 记下 d1 B 缺价(no_row，NaN 价已等同缺行)，got {ml}")


def run_all():
    print("=" * 60, "\n缺失/NaN 收益「当前行为」回归（钉死症状）\n" + "=" * 60)
    test_missing_row_held_treated_as_zero()
    test_nan_adjclose_held_treated_as_missing_row()
    print("\n行为已钉死：缺行/NaN 价仍按 0 计、不 raise，但已记进 result['missing_log'] 暴露出来 ✅")


if __name__ == "__main__":
    run_all()
