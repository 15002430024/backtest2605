# -*- coding: utf-8 -*-
"""现金引擎对抗用例（计划 §C）：用合成 MarketData 钉死最容易错的分支——
涨跌停/停牌/ST 用现成字段拦、退市 vs 停牌区分、复权守恒、换手卡死、limit NA、空窗。

跑：conda activate torch1010 && cd backtest && python -m pytest engine/test_cash_engine.py -v
"""
import numpy as np
import pandas as pd
import pytest

from engine.cash_engine import (
    MarketData, BacktestConfig, CashBacktest, _infer_is_star, _round_down_lot,
)

D = ["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06", "2023-01-09"]  # 5 个交易日（值无关）
A, B, C, E = "000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"


def ts(d):
    return pd.Timestamp(d)


def _df(val, idx, cols, dtype):
    return pd.DataFrame(val, index=idx, columns=cols).astype(dtype)


def mk_market(codes, vwap=10.0, close=None, adj=1.0):
    """全部默认正常可交易的 MarketData；测试再 .loc 改单格（frozen 只挡重绑字段，不挡改表内值）。"""
    idx = pd.DatetimeIndex(pd.to_datetime(D))
    cols = pd.Index(codes)
    return MarketData(
        trade_price=_df(vwap, idx, cols, float),
        close_price=_df(close if close is not None else vwap, idx, cols, float),
        adj=_df(adj, idx, cols, float),
        trade_status=_df("交易", idx, cols, object),
        limit_status=_df(0.0, idx, cols, float),
        is_st=_df(False, idx, cols, bool),
        is_star=_infer_is_star(cols),
        calendar=idx,
    )


def mk_factor(market, rows):
    """rows: {日期str: {code: 因子值}}；未填的格 → NaN。"""
    f = pd.DataFrame(np.nan, index=market.calendar, columns=market.trade_price.columns)
    for d, kv in rows.items():
        for c, v in kv.items():
            f.loc[ts(d), c] = v
    return f


def run_bt(market, factor, n_holdings=1, start_date=D[1], turnover_cap=0.30, init=1e6):
    bench = pd.Series(1.0, index=market.calendar)            # 平基准，超额=策略收益
    cfg = BacktestConfig(initial_capital=init, n_holdings=n_holdings,
                         turnover_cap=turnover_cap, start_date=start_date)
    return CashBacktest(market, factor, None, bench, cfg).run()


def pos(res, d, code):
    """持仓股数，列不存在（从未持有过）→ 0。holdings 只含曾持有过的 code。"""
    h = res.holdings
    return float(h.loc[ts(d), code]) if code in h.columns else 0.0


def test_build_and_account_consistent():
    """建仓 + 账户自洽：现金 + 股票市值 == 总资产，每日成立。"""
    m = mk_market([A, B])
    f = mk_factor(m, {d: {A: 2, B: 1} for d in D})
    res = run_bt(m, f, n_holdings=1)
    assert res.holdings.loc[ts(D[1]), A] > 0
    acc = res.account
    assert ((acc["cash"] + acc["stock_value"] - acc["total"]).abs() < 1e-3).all()


def test_limit_up_blocks_buy():
    """涨停(=1)当天只拦买：想买的票买不进、现金不动。"""
    m = mk_market([A, B])
    m.limit_status.loc[ts(D[1]), A] = 1
    f = mk_factor(m, {D[0]: {A: 2, B: 1}})
    res = run_bt(m, f, n_holdings=1)
    assert pos(res, D[1], A) == 0
    assert abs(res.account.loc[ts(D[1]), "total"] - 1e6) < 1.0   # 没买，总资产=初始


def test_limit_down_blocks_sell():
    """跌停(=-1)当天只拦卖：想卖的持仓卖不掉、锁仓。"""
    m = mk_market([A, B])
    m.limit_status.loc[ts(D[2]), A] = -1
    f = mk_factor(m, {D[0]: {A: 2, B: 1}, D[1]: {A: 1, B: 2}})
    res = run_bt(m, f, n_holdings=1, turnover_cap=1.0)   # cap 放开，单纯验跌停拦卖
    assert pos(res, D[2], A) > 0                          # D1 建 A；D2 要卖 A 但 A 跌停 → 留


def test_suspend_blocks_sell():
    """停牌(trade_status=='停牌')双拦：该卖的持仓卖不掉、锁仓。"""
    m = mk_market([A, B])
    m.trade_status.loc[ts(D[2]), A] = "停牌"
    f = mk_factor(m, {D[0]: {A: 2, B: 1}, D[1]: {A: 1, B: 2}})
    res = run_bt(m, f, n_holdings=1, turnover_cap=1.0)
    assert pos(res, D[2], A) > 0


def test_st_blocks_buy():
    """ST 只拦买：ST 票不进买入候选。"""
    m = mk_market([A, B])
    m.is_st.loc[ts(D[1]), A] = True
    f = mk_factor(m, {D[0]: {A: 2, B: 1}})
    res = run_bt(m, f, n_holdings=1)
    assert pos(res, D[1], A) == 0


def test_st_allows_sell():
    """ST 不拦卖：已持有的票变 ST 仍能卖出。"""
    m = mk_market([A, B])
    m.is_st.loc[ts(D[2]), B] = True
    f = mk_factor(m, {D[0]: {A: 1, B: 2}, D[1]: {A: 2, B: 1}})
    res = run_bt(m, f, n_holdings=1, turnover_cap=1.0)   # D1 建 B；D2 目标 A、卖 B（ST 不挡卖）
    assert pos(res, D[2], B) == 0
    assert pos(res, D[2], A) > 0


def test_delist_settle_to_cash():
    """退市（vwap 从某日 NaN 到末尾不恢复）：按最后有效收盘价折现金、移出、记 missing_log。"""
    m = mk_market([A, B])
    for d in (D[2], D[3], D[4]):
        m.trade_price.loc[ts(d), A] = np.nan          # A 从 D2 起再无成交价
    f = mk_factor(m, {D[0]: {A: 2, B: np.nan}})        # 建 A，之后空窗（B 无因子）
    res = run_bt(m, f, n_holdings=1)
    assert res.holdings.loc[ts(D[2]), A] == 0
    ml = res.missing_log
    assert len(ml) == 1 and ml.iloc[0]["type"] == "delist" and ml.iloc[0]["ticker"] == A
    assert res.account.loc[ts(D[2]), "cash"] > res.account.loc[ts(D[1]), "cash"]


def test_suspend_recover_not_delisted():
    """停牌中间断、之后恢复 ≠ 退市：一直留持仓、停牌日用最后有效价估值。"""
    m = mk_market([A, B])
    m.trade_price.loc[ts(D[2]), A] = np.nan
    m.close_price.loc[ts(D[2]), A] = np.nan            # 停牌当天无收盘 → 用 last_valid
    m.trade_status.loc[ts(D[2]), A] = "停牌"
    f = mk_factor(m, {D[0]: {A: 2, B: np.nan}})
    res = run_bt(m, f, n_holdings=1)
    assert res.holdings.loc[ts(D[2]), A] > 0
    assert len(res.missing_log) == 0


def test_split_adjust_preserves_value():
    """复权：adj 1→2、原价减半（adj_close 连续），股数翻倍、总资产不因复权跳变。"""
    m = mk_market([A, B], vwap=10.0, close=10.0, adj=1.0)
    for d in (D[2], D[3], D[4]):                        # D2 起拆股：因子×2、原价÷2
        m.adj.loc[ts(d), A] = 2.0
        m.trade_price.loc[ts(d), A] = 5.0
        m.close_price.loc[ts(d), A] = 5.0
    f = mk_factor(m, {D[0]: {A: 2, B: np.nan}})         # 建 A、之后空窗只发生复权
    res = run_bt(m, f, n_holdings=1)
    sh1, sh2 = res.holdings.loc[ts(D[1]), A], res.holdings.loc[ts(D[2]), A]
    assert sh1 > 0 and abs(sh2 - 2 * sh1) < 1e-6
    t1, t2 = res.account.loc[ts(D[1]), "total"], res.account.loc[ts(D[2]), "total"]
    assert abs(t2 - t1) / t1 < 1e-6                     # 复权本身不改总资产


def test_limit_na_allows_trade():
    """limit_status 交易日为 NaN：按可正常买卖（与权重引擎一致），不误判涨跌停。"""
    m = mk_market([A, B])
    m.limit_status.loc[ts(D[1]), A] = np.nan
    f = mk_factor(m, {D[0]: {A: 2, B: 1}})
    res = run_bt(m, f, n_holdings=1)
    assert res.holdings.loc[ts(D[1]), A] > 0


def test_empty_factor_holds_position():
    """空窗（当日因子全缺）：持仓不动、不清成现金、不产生成交。"""
    m = mk_market([A, B])
    f = mk_factor(m, {D[0]: {A: 2, B: np.nan}})         # 建 A，D1 起因子全缺
    res = run_bt(m, f, n_holdings=1)
    assert res.holdings.loc[ts(D[2]), A] == res.holdings.loc[ts(D[1]), A] > 0
    assert len(res.trades[res.trades["date"] == ts(D[2])]) == 0


def test_turnover_cap_caps_oneway():
    """换手卡死：非建仓日单边换手 ≤ turnover_cap；建仓日豁免（满仓建到 2 只）。"""
    m = mk_market([A, B, C, E])
    f = mk_factor(m, {D[0]: {A: 4, B: 3, C: 2, E: 1},   # D1 建 {A,B}
                      D[1]: {A: 1, B: 2, C: 3, E: 4}})  # D2 目标 {C,E}，整组换手
    res = run_bt(m, f, n_holdings=2, turnover_cap=0.30)
    assert (res.holdings.loc[ts(D[1])] > 0).sum() == 2  # 建仓日不卡，建满 2 只
    prev_total = res.account.loc[ts(D[1]), "total"]
    td2 = res.trades[res.trades["date"] == ts(D[2])]
    realized_oneway = (td2["shares"] * td2["price"]).sum() / 2 / prev_total
    assert realized_oneway <= 0.30 + 1e-6


# ---- 最小申报规则：科创板 ≥200 后 1 股递增，其余 100 整数倍（真实交易所规则）----
STAR = "688001.SH"


def test_round_down_lot_and_is_star_unit():
    """单元：_infer_is_star 认 688/689；_round_down_lot 科创板≥200后1股递增、主板100整数倍。"""
    flags = _infer_is_star(pd.Index(["688001.SH", "689001.SH", "600000.SH", "300001.SZ"]))
    assert flags.tolist() == [True, True, False, False]
    assert _round_down_lot(199.9, True) == 0       # 不足 200 → 0
    assert _round_down_lot(200, True) == 200
    assert _round_down_lot(349.7, True) == 349     # ≥200 后 1 股递增（不取 200/100 整数倍）
    assert _round_down_lot(99, False) == 0         # 主板/创业板 100 整数倍
    assert _round_down_lot(350, False) == 300
    assert _round_down_lot(349.7, False) == 300


def test_star_min_200_step_1():
    """科创板买入按真实规则：≥200 起、1 股递增（不被强行取到 200/100 整数倍）。"""
    m = mk_market([STAR, B], vwap=2000.0)          # 高价让目标股数落在非整百
    f = mk_factor(m, {D[0]: {STAR: 2, B: 1}})
    res = run_bt(m, f, n_holdings=1, init=700000)  # 700000/2000≈350 股
    sh = pos(res, D[1], STAR)
    assert sh >= 200 and sh % 100 != 0             # 过最低 200 且非 100/200 整数倍 → 1 股粒度


def test_star_below_200_not_bought():
    """科创板目标不足 200 股 → 不买（最低申报 200）。"""
    m = mk_market([STAR, B], vwap=2000.0)
    f = mk_factor(m, {D[0]: {STAR: 2, B: 1}})
    res = run_bt(m, f, n_holdings=1, init=100000)  # 100000/2000=50 股 <200
    assert pos(res, D[1], STAR) == 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
