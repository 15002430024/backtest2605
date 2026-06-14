# -*- coding: utf-8 -*-
"""现金引擎（纯 weights_df 执行器）对抗用例：用合成 MarketData + target_panel 钉死最易错的分支——
非调仓日不动仓、清仓、涨跌停/停牌拦截并记 blocked_log、退市 vs 停牌、复权守恒、换手卡帽、
整手/现金吞没记账、入口权重校验。ST 已移出执行层（归策略层），不在此测。

跑：conda activate torch1010 && cd backtest && python -m pytest tests/test_cash_engine.py -v
"""
import numpy as np
import pandas as pd
import pytest

import engine.cash_engine as ce
from engine.cash_engine import (
    MarketData, BacktestConfig, CashBacktest, run_cash_backtest,
    _infer_is_star, _round_down_lot, MAIN_LOT,
)

D = ["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06", "2023-01-09"]  # 5 个交易日
A, B, C, E = "000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"


def ts(d):
    return pd.Timestamp(d)


def _df(val, idx, cols, dtype):
    return pd.DataFrame(val, index=idx, columns=cols).astype(dtype)


def mk_market(codes, vwap=10.0, close=None, adj=1.0, delist=None):
    """全部默认正常可交易的 MarketData（无 is_st，ST 已移出执行层）；测试再 .loc 改单格。

    delist: {code: 退市日str} 或 None（全 NaT=均未退市）。
    """
    idx = pd.DatetimeIndex(pd.to_datetime(D))
    cols = pd.Index(codes)
    delist_date = pd.Series(pd.NaT, index=cols, dtype="datetime64[ns]")
    if delist:
        for c, d in delist.items():
            delist_date[c] = pd.Timestamp(d)
    return MarketData(
        trade_price=_df(vwap, idx, cols, float),
        close_price=_df(close if close is not None else vwap, idx, cols, float),
        adj=_df(adj, idx, cols, float),
        trade_status=_df("交易", idx, cols, object),
        limit_status=_df(0.0, idx, cols, float),
        is_star=_infer_is_star(cols),
        delist_date=delist_date,
        calendar=idx,
    )


def mk_target(market, rows):
    """rows: {调仓日str: {code: weight}} → (target_panel 调仓日×cols 稠密 fillna(0), rebalance_dates)。"""
    reb = pd.DatetimeIndex([ts(d) for d in rows])
    panel = pd.DataFrame(0.0, index=reb, columns=market.trade_price.columns)
    for d, kv in rows.items():
        for c, w in kv.items():
            panel.loc[ts(d), c] = w
    return panel, reb


def run_bt(market, rows, turnover_cap=None, init=1e6, start_date=None, exec_price="vwap"):
    bench = pd.Series(1.0, index=market.calendar)            # 平基准，超额=策略收益
    panel, reb = mk_target(market, rows)
    cfg = BacktestConfig(initial_capital=init, turnover_cap=turnover_cap,
                         start_date=start_date, exec_price=exec_price)
    return CashBacktest(market, panel, reb, bench, cfg).run()


def pos(res, d, code):
    """持仓股数，列不存在（从未持有过）→ 0。"""
    h = res.holdings
    return float(h.loc[ts(d), code]) if code in h.columns else 0.0


def blocked(res, d=None, code=None, side=None, reason=None):
    """按条件过滤 blocked_log 行数。"""
    bl = res.blocked_log
    if d is not None:
        bl = bl[bl["date"] == ts(d)]
    for col, v in (("code", code), ("side", side), ("reason", reason)):
        if v is not None:
            bl = bl[bl[col] == v]
    return len(bl)


def test_build_and_account_consistent():
    """建仓 + 账户自洽：股票市值独立用 持仓股数×收盘价 重算，对账 account.stock_value（非恒真）。

    account.stock_value 由引擎按 total-cash 定义，直接断言 cash+stock_value==total 是恒真；
    这里用 holdings×close_price 独立重算市值再比，才真正校验估值口径（N14）。
    """
    m = mk_market([A, B], close=12.0)                  # 估值价 12，与成交价 10 不同，避免巧合
    res = run_bt(m, {D[1]: {A: 1.0}})
    assert pos(res, D[1], A) > 0
    acc, hold = res.account, res.holdings
    for d in [ts(x) for x in D[1:]]:
        recomputed_mv = float((hold.loc[d] * m.close_price.loc[d].reindex(hold.columns)).sum())
        assert abs(recomputed_mv - acc.loc[d, "stock_value"]) < 1e-3, f"{d.date()} 市值对账不平"
        assert abs(acc.loc[d, "cash"] + recomputed_mv - acc.loc[d, "total"]) < 1e-3


def test_nonrebalance_day_holds():
    """非调仓日不动仓：D1 建 A 后 D2（无调仓行）股数不变、不产生成交。"""
    m = mk_market([A, B])
    res = run_bt(m, {D[1]: {A: 1.0}})
    assert pos(res, D[2], A) == pos(res, D[1], A) > 0
    assert len(res.trades[res.trades["date"] == ts(D[2])]) == 0


def test_clear_on_drop():
    """清仓：D1 持 A，D2 目标改 B（A 不在目标）→ A 卖到 0、现金回笼（默认无卡帽）。"""
    m = mk_market([A, B])
    res = run_bt(m, {D[1]: {A: 1.0}, D[2]: {B: 1.0}})
    assert pos(res, D[1], A) > 0
    assert pos(res, D[2], A) == 0
    assert pos(res, D[2], B) > 0


def test_limit_up_blocks_buy():
    """涨停(=1)当天只拦买：想买的票买不进、现金不动、记 blocked_log(涨停)。"""
    m = mk_market([A, B])
    m.limit_status.loc[ts(D[1]), A] = 1
    res = run_bt(m, {D[1]: {A: 1.0}})
    assert pos(res, D[1], A) == 0
    assert abs(res.account.loc[ts(D[1]), "total"] - 1e6) < 1.0
    assert blocked(res, d=D[1], code=A, side="buy", reason="涨停") == 1


def test_limit_down_blocks_sell():
    """跌停(=-1)当天只拦卖：想卖的持仓卖不掉、锁仓、记 blocked_log(跌停)。"""
    m = mk_market([A, B])
    m.limit_status.loc[ts(D[2]), A] = -1
    res = run_bt(m, {D[1]: {A: 1.0}, D[2]: {B: 1.0}})
    assert pos(res, D[2], A) > 0                          # D1 建 A；D2 要卖 A 但跌停 → 留
    assert blocked(res, d=D[2], code=A, side="sell", reason="跌停") == 1


def test_suspend_blocks_sell():
    """停牌(trade_status=='停牌')双拦：该卖的持仓卖不掉、锁仓。"""
    m = mk_market([A, B])
    m.trade_status.loc[ts(D[2]), A] = "停牌"
    res = run_bt(m, {D[1]: {A: 1.0}, D[2]: {B: 1.0}})
    assert pos(res, D[2], A) > 0
    assert blocked(res, d=D[2], code=A, side="sell", reason="停牌") == 1


def test_delist_settle_to_cash():
    """退市（真实退市日 D2）：到 D2 按最后有效收盘价折现金、移出、记 missing_log。"""
    m = mk_market([A, B], delist={A: D[2]})            # A 真实退市日 = D2
    for d in (D[2], D[3], D[4]):
        m.trade_price.loc[ts(d), A] = np.nan          # 退市后再无成交价
    res = run_bt(m, {D[1]: {A: 1.0}})                  # 建 A，之后不再调仓
    assert pos(res, D[2], A) == 0
    ml = res.missing_log
    assert len(ml) == 1 and ml.iloc[0]["type"] == "delist" and ml.iloc[0]["ticker"] == A
    assert res.account.loc[ts(D[2]), "cash"] > res.account.loc[ts(D[1]), "cash"]


def test_suspend_recover_not_delisted():
    """停牌中间断、之后恢复 ≠ 退市：一直留持仓、停牌日用最后有效价估值。"""
    m = mk_market([A, B])
    m.trade_price.loc[ts(D[2]), A] = np.nan
    m.close_price.loc[ts(D[2]), A] = np.nan            # 停牌当天无收盘 → 用 last_valid
    m.trade_status.loc[ts(D[2]), A] = "停牌"
    res = run_bt(m, {D[1]: {A: 1.0}})
    assert pos(res, D[2], A) > 0
    assert len(res.missing_log) == 0


def test_split_adjust_preserves_value():
    """复权：adj 1→2、原价减半，股数翻倍、总资产不因复权跳变。"""
    m = mk_market([A, B], vwap=10.0, close=10.0, adj=1.0)
    for d in (D[2], D[3], D[4]):                        # D2 起拆股：原价÷2、adj×2
        m.adj.loc[ts(d), A] = 2.0
        m.trade_price.loc[ts(d), A] = 5.0
        m.close_price.loc[ts(d), A] = 5.0
    res = run_bt(m, {D[1]: {A: 1.0}})                   # 建 A、之后只发生复权
    sh1, sh2 = pos(res, D[1], A), pos(res, D[2], A)
    assert sh1 > 0 and abs(sh2 - 2 * sh1) < 1e-6
    t1, t2 = res.account.loc[ts(D[1]), "total"], res.account.loc[ts(D[2]), "total"]
    assert abs(t2 - t1) / t1 < 1e-6                     # 复权本身不改总资产


def test_limit_na_allows_trade():
    """limit_status 交易日为 NaN：按可正常买卖（与权重引擎一致），不误判涨跌停。"""
    m = mk_market([A, B])
    m.limit_status.loc[ts(D[1]), A] = np.nan
    res = run_bt(m, {D[1]: {A: 1.0}})
    assert pos(res, D[1], A) > 0


def test_turnover_cap_caps_oneway():
    """换手卡帽：非建仓日单边换手 ≤ turnover_cap；建仓日豁免（满仓建到 2 只）。"""
    m = mk_market([A, B, C, E])
    res = run_bt(m, {D[1]: {A: 0.5, B: 0.5}, D[2]: {C: 0.5, E: 0.5}}, turnover_cap=0.30)
    assert (res.holdings.loc[ts(D[1])] > 0).sum() == 2   # 建仓日不卡，建满 2 只
    prev_total = res.account.loc[ts(D[1]), "total"]
    td2 = res.trades[res.trades["date"] == ts(D[2])]
    realized_oneway = (td2["shares"] * td2["price"]).sum() / 2 / prev_total
    assert realized_oneway <= 0.30 + 1e-6


def test_turnover_cap_none_no_cap():
    """turnover_cap=None（默认）：整组换手不被限，D2 全换到新目标。"""
    m = mk_market([A, B, C, E])
    res = run_bt(m, {D[1]: {A: 0.5, B: 0.5}, D[2]: {C: 0.5, E: 0.5}}, turnover_cap=None)
    assert pos(res, D[2], A) == 0 and pos(res, D[2], B) == 0
    assert pos(res, D[2], C) > 0 and pos(res, D[2], E) > 0


# ---- 入口权重校验（run_cash_backtest 在 build_market_data 前 fail-fast，无需缓存）----
def _wdf(rows):
    out = []
    for d, kv in rows.items():
        for c, w in kv.items():
            out.append({"date": d, "code": c, "weight": w})
    return pd.DataFrame(out)


def test_negative_weight_raises():
    with pytest.raises(ValueError, match="long_only"):
        run_cash_backtest(_wdf({D[1]: {A: 1.0, B: -0.2}}), "000001.SH")


def test_nan_weight_raises():
    with pytest.raises(ValueError, match="NaN"):
        run_cash_backtest(_wdf({D[1]: {A: 1.0, B: np.nan}}), "000001.SH")


def test_sum_over_one_raises():
    with pytest.raises(ValueError, match="权重和"):
        run_cash_backtest(_wdf({D[1]: {A: 0.7, B: 0.6}}), "000001.SH")


# ---- 最小申报规则 + 整手/现金吞没记账 ----
STAR = "688001.SH"


def test_round_down_lot_and_is_star_unit():
    """单元：_infer_is_star 认 688/689；_round_down_lot 科创板≥200后1股递增、主板100整数倍。"""
    flags = _infer_is_star(pd.Index(["688001.SH", "689001.SH", "600000.SH", "300001.SZ"]))
    assert flags.tolist() == [True, True, False, False]
    assert _round_down_lot(199.9, True) == 0
    assert _round_down_lot(200, True) == 200
    assert _round_down_lot(349.7, True) == 349
    assert _round_down_lot(99, False) == 0
    assert _round_down_lot(350, False) == 300
    assert _round_down_lot(349.7, False) == 300


def test_star_min_200_step_1():
    """科创板买入按真实规则：≥200 起、1 股递增。"""
    m = mk_market([STAR, B], vwap=2000.0)
    res = run_bt(m, {D[1]: {STAR: 1.0}}, init=700000)  # 700000/2000≈350 股
    sh = pos(res, D[1], STAR)
    assert sh >= 200 and sh % 100 != 0


def test_star_below_200_records_blocked():
    """科创板目标不足 200 股 → 不买（最低申报 200），记 blocked_log(整手不足)。"""
    m = mk_market([STAR, B], vwap=2000.0)
    res = run_bt(m, {D[1]: {STAR: 1.0}}, init=100000)  # 100000/2000=50 股 <200
    assert pos(res, D[1], STAR) == 0
    assert blocked(res, d=D[1], code=STAR, reason="整手不足") == 1


# ---- 本轮正确性修复的新增对抗用例 ----

def test_split_across_missing_rows_preserves_value():
    """发现3：持仓票跨缺行段(停牌无行)发生除权，复牌日一次性补回，总资产不腰斩。"""
    m = mk_market([A, B], vwap=10.0, close=10.0, adj=1.0)
    for d in (D[2], D[3]):                              # D2/D3 全 NaN 模拟缺行（停牌无行）
        for panel in (m.trade_price, m.close_price, m.adj):
            panel.loc[ts(d), A] = np.nan
        m.trade_status.loc[ts(d), A] = "停牌"
    m.adj.loc[ts(D[4]), A] = 2.0                        # 复牌日 D4：adj 1→2、价减半
    m.trade_price.loc[ts(D[4]), A] = 5.0
    m.close_price.loc[ts(D[4]), A] = 5.0
    res = run_bt(m, {D[1]: {A: 1.0}})                   # A 无 delist_date → 不清算，挂到复牌
    sh1, sh4 = pos(res, D[1], A), pos(res, D[4], A)
    assert sh1 > 0 and abs(sh4 - 2 * sh1) < 1e-6, f"复牌日股数应翻倍：{sh1}→{sh4}"
    t1, t4 = res.account.loc[ts(D[1]), "total"], res.account.loc[ts(D[4]), "total"]
    assert abs(t4 - t1) / t1 < 1e-6, f"跨缺行除权不应改总资产：{t1}→{t4}"


def test_delist_judged_by_real_date_not_window_end():
    """发现4：退市只由真实退市日判，不看窗口末有无价格（消除前视/可复现性）。"""
    # 同样 D3 起价格缺失到窗口末：有 delist_date 则清算，无 delist_date 则当长停牌不清算
    m_del = mk_market([A, B], delist={A: D[3]})
    m_hold = mk_market([A, B])                          # A 无退市日
    for m in (m_del, m_hold):
        for d in (D[3], D[4]):
            m.trade_price.loc[ts(d), A] = np.nan
    r_del = run_bt(m_del, {D[1]: {A: 1.0}})
    r_hold = run_bt(m_hold, {D[1]: {A: 1.0}})
    assert pos(r_del, D[3], A) == 0 and len(r_del.missing_log) == 1   # 真实退市日 → 清算
    assert pos(r_hold, D[3], A) > 0 and len(r_hold.missing_log) == 0  # 无退市日 → 长停牌挂着


def test_direction_reversal_limit_down_blocks_actual_sell():
    """N5：权重判买、股数实际卖，跌停日复检 can_sell 拦下（不再照卖）。"""
    m = mk_market([A, B])
    # D1 建 A 50% (vwap=10 → 50000 股)；D2 vwap 涨到 13、跌停标记、目标权重升到 0.55
    res_ctrl = run_bt(m, {D[1]: {A: 0.5}})
    assert pos(res_ctrl, D[1], A) > 0
    m.trade_price.loc[ts(D[2]), A] = 13.0
    m.close_price.loc[ts(D[2]), A] = 13.0
    m.limit_status.loc[ts(D[2]), A] = -1               # 跌停（只拦卖）
    res = run_bt(m, {D[1]: {A: 0.5}, D[2]: {A: 0.55}}) # 权重升=判买；但价涨后 0.55 对应更少股数=实际卖
    td2 = res.trades[res.trades["date"] == ts(D[2])]
    assert (td2["side"] == "sell").sum() == 0, "跌停日实际卖单应被拦，不得成交"
    assert blocked(res, d=D[2], code=A, side="sell") >= 1


def test_turnover_cap_truncation_records_blocked():
    """N6：被换手限额截断的票（含清仓单）逐只记 blocked_log(换手限额)，不再静默落空。"""
    m = mk_market([A, B, C, E])
    # D1 建 {A:0.7,B:0.2}；D2 目标 {B:0.2,C:0.2}，cap=0.30 → A 清仓单(|Δ|≈0.7)超帽被截
    res = run_bt(m, {D[1]: {A: 0.7, B: 0.2}, D[2]: {B: 0.2, C: 0.2}}, turnover_cap=0.30)
    assert pos(res, D[2], A) > 0                         # A 被截、未清掉（锁仓）
    assert blocked(res, d=D[2], code=A, reason="换手限额") >= 1


def test_cap_exempt_anchors_to_first_rebalance_not_start_date():
    """N7：显式 start_date(非调仓日)+开 cap 时，首个真实建仓日仍享豁免、建满不被截半。"""
    m = mk_market([A, B])                               # 日历 D0..D4
    # start_date=D1(非调仓日，有前一日 D0)，首个调仓日=D2 目标 {A:0.5,B:0.5}，cap=0.30
    # 修复前豁免锚到 start_date(D1) → D2 被卡帽只建半；修复后锚到首个真实建仓日 D2 → 建满
    res = run_bt(m, {D[2]: {A: 0.5, B: 0.5}}, turnover_cap=0.30, start_date=D[1])
    assert (res.holdings.loc[ts(D[2])] > 0).sum() == 2, "建仓日应豁免卡帽、两只都建上"


def test_post_split_buy_is_round_lot():
    """发现6：复权致持仓非整百后加仓，买单数量仍为合法整手（主板 100 整数倍）。"""
    m = mk_market([A, B], vwap=10.0, close=10.0, adj=1.0)
    for d in (D[2], D[3], D[4]):                        # D2 起 adj 1→1.5、价同步缩，持仓变非整百
        m.adj.loc[ts(d), A] = 1.5
        m.trade_price.loc[ts(d), A] = 10.0 / 1.5
        m.close_price.loc[ts(d), A] = 10.0 / 1.5
    res = run_bt(m, {D[1]: {A: 0.5}, D[3]: {A: 0.9}})   # D3 加仓
    buys = res.trades[(res.trades["date"] == ts(D[3])) & (res.trades["side"] == "buy")]
    for sh in buys["shares"]:
        assert sh % MAIN_LOT == 0, f"主板买单应为 100 整数倍，实得 {sh}"


def test_mark_to_market_raises_on_unvaluable_holding():
    """C9：持仓票 close 全 NaN（vwap 有值能买入）→ 估值 fail-fast，不静默把本金当 0 吞掉。"""
    m = mk_market([A, B], vwap=10.0, close=10.0)
    m.close_price[A] = np.nan                          # A 估值价全程缺失
    with pytest.raises(ValueError, match="无有效收盘估值价"):
        run_bt(m, {D[1]: {A: 1.0}})


def test_first_rebalance_day_executes_via_entry(monkeypatch):
    """发现1：run_cash_backtest 把市场窗口前移一交易日，首个调仓日不再被静默跳过。

    构造全日历 [P, D0..D4]（P=首调仓日前一交易日），weights 首调仓日=D0。修复前 D0 会被
    _resolve_start_date 跳过、从 D1 才建仓；修复后 market_start=P、D0 在 index 1 被选中执行。
    """
    P = "2023-01-02"
    cal_full = pd.DatetimeIndex(pd.to_datetime([P] + D))

    def fake_market(codes, start, end, exec_price="vwap"):
        idx = cal_full[(cal_full >= pd.Timestamp(start)) & (cal_full <= pd.Timestamp(end))]
        cols = pd.Index(sorted(codes))
        return MarketData(
            trade_price=_df(10.0, idx, cols, float), close_price=_df(10.0, idx, cols, float),
            adj=_df(1.0, idx, cols, float), trade_status=_df("交易", idx, cols, object),
            limit_status=_df(0.0, idx, cols, float), is_star=_infer_is_star(cols),
            delist_date=pd.Series(pd.NaT, index=cols, dtype="datetime64[ns]"), calendar=idx,
        )

    monkeypatch.setattr(ce, "load_calendar", lambda: cal_full)
    monkeypatch.setattr(ce, "build_market_data", fake_market)
    monkeypatch.setattr(ce, "load_index_eod", lambda code: pd.DataFrame(
        {"date": cal_full, "close": 1.0}))
    monkeypatch.setattr(ce, "calc_benchmark",
                        lambda idxdf, s, e: pd.Series(1.0, index=cal_full))

    wdf = _wdf({D[0]: {A: 1.0}})                        # 首调仓日 = 全日历的 D0（前面还有 P）
    res = run_cash_backtest(wdf, "000001.SH")
    assert (res.trades["date"] == ts(D[0])).any(), "首个调仓日 D0 应真实成交，不得被跳过"
    assert pos(res, D[0], A) > 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
