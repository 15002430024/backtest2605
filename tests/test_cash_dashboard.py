# -*- coding: utf-8 -*-
"""现金回测完整仪表盘（plot_cash_dashboard）验收：钉死计划里三个会静默出错的对抗坑——
C1 仪表盘标量指标与引擎自算的 metrics_abs/excess 逐点相等、C3 喂热力图的是归一权重非股数、
C5 建仓锚点不在月/年表里画出伪 0% 上期；外加出图落盘冒烟。

合成 MarketData/run_bt 复用 test_cash_engine，不另造副本。
跑：conda activate torch1010 && cd backtest && python -m pytest tests/test_cash_dashboard.py -v
"""
import numpy as np
import pandas as pd
import pytest

from analysis.metrics import build_report_data
from engine.cash_engine import _ABS_KEYS, _EXCESS_KEYS
from report.plot import plot_cash_dashboard, _drop_pre_start_phantom
from test_cash_engine import mk_market, run_bt, D, A, B


def _cash_to_dict(res):
    """与 plot_cash_dashboard 内部同款适配（单一口径，测试用来核 C1）。"""
    return {
        "nav": res.strategy_nav,
        "weights": res.weights,
        "trade_records": res.trade_records,
        "blocked_trades": res.blocked_log.to_dict("records"),
    }


def test_dashboard_metrics_match_engine():
    """C1：build_report_data 喂适配 dict 算出的标量，与引擎 _assemble 自算的 metrics_abs/excess 逐点相等。"""
    m = mk_market([A, B], close=12.0)
    res = run_bt(m, {D[1]: {A: 0.6, B: 0.4}, D[3]: {A: 0.4, B: 0.6}})
    rd = build_report_data(_cash_to_dict(res), benchmark_nav=res.benchmark_nav)

    for k in _ABS_KEYS + _EXCESS_KEYS:
        engine_v = res.metrics_abs.get(k, res.metrics_excess.get(k))
        dash_v = rd.metrics[k]
        if isinstance(engine_v, pd.Timestamp):
            assert engine_v == dash_v, f"{k}: 引擎 {engine_v} ≠ 仪表盘 {dash_v}"
        else:
            assert np.isclose(engine_v, dash_v, equal_nan=True), f"{k}: 引擎 {engine_v} ≠ 仪表盘 {dash_v}"


def test_weights_are_normalized_not_shares():
    """C3：res.weights 是权重（每行 Σ≤1、|w|≤1），不是股数（量级 1e2~1e6）。"""
    m = mk_market([A, B], close=12.0)
    res = run_bt(m, {D[1]: {A: 0.6, B: 0.4}})
    w = res.weights
    assert w.abs().max().max() <= 1.0 + 1e-9, "权重越界，疑似把股数当权重"
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all(), "每行权重和 >1，含现金残余应 ≤1"
    # 建仓后某日股票市值占比应与 account 对得上
    d = res.account.index[-1]
    assert np.isclose(w.loc[d].sum(), res.account.loc[d, "stock_value"] / res.account.loc[d, "total"])


def _fake_rd_from_nav(nav):
    return build_report_data({"nav": nav}, benchmark_nav=None)


def test_drop_phantom_cross_year():
    """C5：锚点跨年（start 是年首交易日）→ 月/年表里的伪 2022 期被抹掉。"""
    idx = pd.to_datetime(["2022-12-30", "2023-01-03", "2023-01-04", "2023-02-01", "2023-02-02"])
    nav = pd.Series(np.linspace(1.0, 1.1, len(idx)), index=idx)
    rd = _fake_rd_from_nav(nav)
    assert 2022 in rd.annual_strategy.index and 2022 in rd.monthly_table.index, "前提：伪 2022 期应先存在"

    _drop_pre_start_phantom(rd, anchor=idx[0], start=idx[1])
    assert 2022 not in rd.annual_strategy.index, "跨年伪年柱未删"
    assert 2022 not in rd.monthly_table.index, "跨年伪月行未删"
    assert 2023 in rd.annual_strategy.index, "真实 2023 被误删"


def test_drop_phantom_cross_month_same_year():
    """C5：锚点跨月不跨年（start 是月首交易日）→ 抹掉伪 1 月格，保留该年其余月、不删年。"""
    idx = pd.to_datetime(["2023-01-31", "2023-02-01", "2023-02-02", "2023-03-01"])
    nav = pd.Series(np.linspace(1.0, 1.1, len(idx)), index=idx)
    rd = _fake_rd_from_nav(nav)

    _drop_pre_start_phantom(rd, anchor=idx[0], start=idx[1])
    assert 2023 in rd.monthly_table.index, "同年不应删整年"
    assert pd.isna(rd.monthly_table.loc[2023, 1]), "伪 1 月格未抹"
    assert not pd.isna(rd.monthly_table.loc[2023, 2]), "真实 2 月被误抹"


def test_drop_phantom_noop_same_month():
    """C5：常态（锚点与 start 同月）→ 周期表不动。"""
    idx = pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05", "2023-02-01"])
    nav = pd.Series(np.linspace(1.0, 1.1, len(idx)), index=idx)
    rd = _fake_rd_from_nav(nav)
    before = rd.monthly_table.copy()
    _drop_pre_start_phantom(rd, anchor=idx[0], start=idx[1])
    pd.testing.assert_frame_equal(rd.monthly_table, before)


def test_plot_cash_dashboard_outputs(tmp_path):
    """冒烟：完整仪表盘端到端跑通、大图落盘非空。"""
    m = mk_market([A, B], close=12.0)
    res = run_bt(m, {D[1]: {A: 0.6, B: 0.4}, D[3]: {A: 0.4, B: 0.6}})
    path = plot_cash_dashboard(res, save_dir=str(tmp_path), title="现金仪表盘测试")
    out = pd.Index([path])
    assert out[0].endswith("现金回测仪表盘.png")
    import os
    assert os.path.getsize(path) > 0, "大图为空"
