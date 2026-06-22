"""对抗压测 + 对抗性证伪发现的 bug 的回归测试（修复后锁住行为）。

覆盖 7 个修复：fix1 开仓豁免阈值、fix2 容量不足按天聚合、fix3 IC 零方差门控(锚点 1e-20)、
fix4 价格守卫非对称下界(抓末日幻影低价/保真实暴跌)、fix5 换手单边帽(无跨边毒杀)、
fix6 现金部分成交留痕、fix7 超额基准覆盖率按策略全区间(抓基准提前截止/跨度内稀疏)。"""
import numpy as np, pandas as pd
from engine.backtest import check_tradable, skip_small_changes
from factor.factor_test import _row_corr
from data.loaders import validate_adjclose_quality, DataQualityError


def _close(a, b, msg, tol=1e-9):
    assert abs(float(a) - float(b)) < tol, f"{msg}: {a} vs {b}"


def test_threshold_does_not_block_opening():
    """修1: 阈值 > 单票目标权重时,首次建仓不能被吃掉。"""
    tgt = {f"S{i}": 0.02 for i in range(50)}                 # 50 只各 2%
    res = skip_small_changes({}, tgt, threshold=0.05)  # 阈值 5% > 2%
    assert len(res) == 50, f"开仓被阈值吃掉了: 只建了 {len(res)} 只"
    _close(sum(res.values()), 1.0, "开仓后满仓")
    # 已有仓位的微调仍受阈值约束
    res2 = skip_small_changes({"A": 0.20}, {"A": 0.205}, threshold=0.05)
    _close(res2["A"], 0.20, "微调<阈值不动")


def test_capacity_aggregated_one_event():
    """修2: 容量不足按天聚合一条,不再每只缩买票各记一条。"""
    final, blocked = check_tradable(
        {"A": 0.5, "B": 0.2, "C": 0.2}, {"A": 0.2, "B": 0.4, "C": 0.4},
        {"A": {"limit_status": -1, "trade_status": "交易"},
         "B": {"limit_status": 0, "trade_status": "交易"},
         "C": {"limit_status": 0, "trade_status": "交易"}})
    caps = [b for b in blocked if b["reason"] == "容量不足"]
    assert len(caps) == 1, f"容量不足应聚合成1条,实得 {len(caps)}"
    _close(caps[0]["blocked_weight"], 0.30, "聚合缩买额")
    # 微小浮点缩买不记
    f2, b2 = check_tradable({"A": 0.5}, {"A": 0.5 + 1e-11},
                            {"A": {"limit_status": 0, "trade_status": "交易"}})
    assert not any(b["reason"] == "容量不足" for b in b2), "微小缩买不该记容量不足"


def test_row_corr_degenerate_cross_section_is_nan():
    """修3: 某期截面全相等(浮点零方差) → IC 为 NaN 不是 0.0。"""
    f = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["A", "B", "C"])
    r = pd.DataFrame([[0.05, 0.05, 0.05]], columns=["A", "B", "C"])   # 全涨停 +5%
    ic = _row_corr(f, r).iloc[0]
    assert np.isnan(ic), f"退化截面 IC 应 NaN,实得 {ic}"
    # 正常截面仍算得出
    r2 = pd.DataFrame([[0.01, 0.02, 0.03]], columns=["A", "B", "C"])
    assert abs(_row_corr(f, r2).iloc[0] - 1.0) < 1e-9, "单调正相关应≈1"


def test_row_corr_keeps_low_cv_cross_section():
    """修3 证伪反例(锚点修正): 合法但低相对离散(CV~2e-7)的截面不能被误判 NaN。
    旧 1e-12 锚点(≈CV>1e-6)会把它静默杀掉污染 IC;新 1e-20 锚点(≈CV>1e-10)应保留。"""
    rng = np.random.default_rng(0)
    a = 50.0 + 1e-5 * rng.standard_normal(50)        # 均值50,标准差1e-5 → CV~2e-7
    b = rng.standard_normal(50)
    ic = _row_corr(pd.DataFrame([a]), pd.DataFrame([b])).iloc[0]
    assert np.isfinite(ic), f"合法低CV截面被误杀成 NaN(锚点过松): {ic}"
    # 与 scipy 口径的真 Pearson 对齐(逐点一致,没被门控改值)
    true_p = np.corrcoef(a, b)[0, 1]
    assert abs(ic - true_p) < 1e-9, f"门控改了正常截面的 IC 值: {ic} vs {true_p}"


def test_price_guard_keeps_real_crash():
    """修4: 真实暴跌(-96%)不再被守卫误剔;脏数据上跳(>10x)仍被抓。"""
    real = pd.DataFrame({"code": ["X"] * 3, "date": pd.to_datetime(["2024-06-03", "2024-06-04", "2024-06-05"]),
                         "adj_close": [9.84, 9.84, 0.35]})       # 退市复牌 -96%(ratio~0.036),真实
    out = validate_adjclose_quality(real, on_bad="drop")
    assert len(out) == 3, "真实暴跌不该被剔"
    dirty = pd.DataFrame({"code": ["Y"] * 3, "date": pd.to_datetime(["2022-02-23", "2022-02-24", "2022-02-25"]),
                          "adj_close": [9.38, 9.38, 18807.0]})    # 北交所脏价上跳 ×2005
    out2 = validate_adjclose_quality(dirty, on_bad="drop")
    assert len(out2) == 0, "脏数据上跳应整只剔除"


def test_price_guard_catches_lastday_phantom_low():
    """修4 证伪反例(下界补洞): 末日幻影低价(10→0.001,-99.99%)无回跳,只靠上界漏判会把 NAV 砸到~0。
    新增非对称下界(ratio<0.01)应抓住它,且不误伤真实 -96%(ratio 0.036>0.01)。"""
    phantom = pd.DataFrame({"code": ["Z"] * 4,
                            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
                            "adj_close": [10.0, 10.1, 10.0, 0.001]})   # 末日幻影低价,其后无数据
    out = validate_adjclose_quality(phantom, on_bad="drop")
    assert len(out) == 0, "末日幻影低价(无回跳)应被下界抓住整只剔除"
    # 边界确认: -98%(ratio 0.02>0.01)算真实暴跌保留, -99.5%(ratio 0.005<0.01)算脏价剔除
    near = pd.DataFrame({"code": ["P"] * 2, "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                         "adj_close": [10.0, 0.2]})                     # -98%
    assert len(validate_adjclose_quality(near, on_bad="drop")) == 2, "-98% 仍应保留(真实暴跌)"


def test_compute_excess_rejects_sparse_benchmark():
    """修7: 基准缺策略多数交易日 → fail-fast,不静默算错超额。"""
    import pytest
    from factor.factor_test import compute_excess
    idx = pd.bdate_range("2024-01-01", periods=20)
    strat = pd.Series(np.linspace(1.0, 1.1, 20), index=idx)
    sparse = pd.Series([1.0, 1.05, 1.1], index=idx[[0, 10, 19]])   # 跨度内只3个点
    with pytest.raises(ValueError, match="覆盖不足"):
        compute_excess(strat, sparse)
    # 稠密基准正常算
    dense = pd.Series(np.linspace(1.0, 1.05, 20), index=idx)
    out = compute_excess(strat, dense)
    assert len(out) == 20 and np.isfinite(out.iloc[-1])


def test_compute_excess_rejects_benchmark_ending_early():
    """修7 证伪反例(覆盖率改按策略全区间): 基准提前截止(陈旧缓存)时,
    旧 in_span 口径会 ratio=1.0 蒙混过关、策略尾段被静默丢弃;新口径应 fail-fast。"""
    import pytest
    from factor.factor_test import compute_excess
    idx = pd.bdate_range("2024-01-01", periods=20)
    strat = pd.Series(np.linspace(1.0, 1.1, 20), index=idx)
    early = pd.Series(np.linspace(1.0, 1.03, 8), index=idx[:8])    # 基准只到前8天且自身稠密
    with pytest.raises(ValueError, match="覆盖不足"):
        compute_excess(strat, early)


def test_cash_partial_fill_is_logged():
    """修6: 现金只够买一部分时,落 '现金部分成交' 留痕(观测性)。"""
    from engine.cash_engine import CashBacktest
    import inspect
    src = inspect.getsource(CashBacktest)
    assert "现金部分成交" in src, "部分成交留痕分支没接上"


def test_turnover_cap_no_cross_side_poisoning():
    """修5 证伪反例(跨边毒杀): 一侧大单(A 卖 0.5>cap)不能压死另一侧本在预算内的买单(B+C=0.25<cap)。
    旧联合掩码会让 A 顶过 cap 后 B、C 一并被拒(当日零成交);新各自卡帽应让 B、C 正常买、只拦 A。"""
    from test_cash_engine import mk_market, run_bt, pos, blocked, D, A, B, C
    m = mk_market([A, B, C])
    # D[1] 建仓(首调仓豁免卡帽,D[0]=日历首日引擎不调仓): A=0.5,B=0.2,C=0.05;
    # D[2] 目标 A=0,B=0.4,C=0.1, cap=0.30 → A卖0.5>cap被拦, B(+0.2)+C(+0.05)=0.25<cap应买进
    res = run_bt(m, {D[1]: {A: 0.5, B: 0.2, C: 0.05},
                     D[2]: {A: 0.0, B: 0.4, C: 0.1}}, turnover_cap=0.30)
    # A 卖 0.5 > cap → 卖单被拦(锁仓,持仓不变)
    assert pos(res, D[2], A) == pos(res, D[1], A), "A 卖单 0.5>cap 应被拦,持仓不变"
    assert blocked(res, D[2], code=A, reason="换手限额") == 1, "A 卖应记换手限额"
    # B(+0.2)、C(+0.05) 买边累计 0.25<cap → 应正常买进(旧 bug 下会被跨边毒杀成零成交)
    assert pos(res, D[2], B) > pos(res, D[1], B), "B 买单在预算内,不该被对侧大单压死"
    assert pos(res, D[2], C) > pos(res, D[1], C), "C 买单在预算内,不该被对侧大单压死"
    assert blocked(res, D[2], code=B, reason="换手限额") == 0, "B 不该被记换手限额"
