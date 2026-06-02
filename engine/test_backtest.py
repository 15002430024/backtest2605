"""
阶段 2 回测引擎验收测试（对应 backtest_roadmap_v2.md 阶段 2 验收标准 + 多空扩展）

核心正确性（5 个）:
  1. 单股满仓     — NAV == 该股复权价 / 首日价
  2. 单股半仓     — NAV 相对 1.0 的偏离恰好是测试 1 的一半
  3. 调仓时序     — 调仓日当天用旧权重，新权重次日生效
  4. 权重漂移     — 非调仓日权重确实漂移，不保持 50/50
  5. 多空         — 多 A 空 B，A 涨 B 跌时 NAV 比纯多头涨得快

校验逻辑（3 个，确认能正确 raise）:
  6. weights_df 含 price_df 不存在的 code → raise
  7. 权重绝对值合计 > 1 → raise
  8. start_date / end_date 越界 → raise

统一行情（4 个交易日 d0~d3）:
  A: 10, 11, 12, 12   → 日收益 NaN, +0.10, +12/11-1, 0
  B: 20, 18, 18,  9   → 日收益 NaN, -0.10, 0,        -0.5
运行: conda activate torch1010 && python test_backtest.py
"""

import numpy as np
import pandas as pd

from engine.backtest import (
    run_backtest,
    update_weights,
    calc_trades,
    calc_cost,
    skip_small_changes,
    check_tradable,
    calc_benchmark,
)

# 4 个交易日（price_df 即交易日历，这几天就是交易日）
DATES = pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"])
D0, D1, D2, D3 = DATES

PRICE_DF = pd.DataFrame(
    {
        "date": list(DATES) * 2,
        "code": ["A"] * 4 + ["B"] * 4,
        "adj_close": [10.0, 11.0, 12.0, 12.0, 20.0, 18.0, 18.0, 9.0],
    }
)

TOL = 1e-9


def _weights(rows):
    """rows: list of (date, code, weight) → weights_df"""
    return pd.DataFrame(rows, columns=["date", "code", "weight"])


def _nav(*args, **kwargs):
    """run_backtest 现在返回 dict，取 nav 曲线。阶段2 测试只关心 nav。"""
    return run_backtest(*args, **kwargs)["nav"]


def _assert_close(actual, expected, label):
    diff = abs(actual - expected)
    if diff > TOL:
        raise AssertionError(
            f"{label}: 实际 {actual!r} != 预期 {expected!r}（差 {diff:.3e}）"
        )


# ============================================================
# 测试 1: 单股满仓 — NAV == A 复权价 / 首日价
# ============================================================
def test_single_full():
    # 单行 weights 的默认 end_date = 调仓日当天，需显式跑到 D3 看完整轨迹
    nav = _nav(_weights([(D0, "A", 1.0)]), PRICE_DF, end_date=D3)
    # A: 10,11,12,12 → 归一化 1.0, 1.1, 1.2, 1.2
    expected = {D0: 1.0, D1: 1.1, D2: 1.2, D3: 1.2}
    for d, exp in expected.items():
        _assert_close(nav[d], exp, f"测试1 单股满仓 {d.date()}")
    return nav


# ============================================================
# 测试 2: 单股半仓 — (NAV2 - 1) 恰好是 (NAV1 - 1) 的一半
# ============================================================
def test_single_half():
    nav_full = _nav(_weights([(D0, "A", 1.0)]), PRICE_DF, end_date=D3)
    nav_half = _nav(_weights([(D0, "A", 0.5)]), PRICE_DF, end_date=D3)
    # 半仓 + 现金: NAV = 0.5*(A/10) + 0.5 → 相对 1.0 的偏离恰好减半
    for d in DATES:
        _assert_close(
            nav_half[d] - 1.0,
            0.5 * (nav_full[d] - 1.0),
            f"测试2 单股半仓 {d.date()}（偏离应为满仓一半）",
        )
    # 另核对绝对值: d1=1.05, d2=1.1, d3=1.1
    _assert_close(nav_half[D1], 1.05, "测试2 d1 绝对值")
    _assert_close(nav_half[D2], 1.10, "测试2 d2 绝对值")
    return nav_half


# ============================================================
# 测试 3: 调仓时序 — 调仓日用旧权重，新权重次日生效
# ============================================================
def test_rebalance_timing():
    # d0 持 A 满仓，d2 调仓换成 B 满仓
    nav = _nav(_weights([(D0, "A", 1.0), (D2, "B", 1.0)]), PRICE_DF, end_date=D3)
    # d2 当天收益必须仍来自 A（旧权重）: 1.1 * (12/11) = 1.2
    # 若错误地当天就用 B（B 在 d2 收益为 0），NAV 会是 1.1 → 用此区分
    _assert_close(nav[D2], 1.2, "测试3 调仓日(d2)应仍用旧权重A → 1.2")
    # d3 切到 B 生效: B 从 18 跌到 9，收益 -0.5 → 1.2 * 0.5 = 0.6
    _assert_close(nav[D3], 0.6, "测试3 次日(d3)新权重B生效 → 0.6")
    return nav


# ============================================================
# 测试 4: 权重漂移 — 等权 50/50，一涨一跌后权重偏离 50/50
# ============================================================
def test_weight_drift():
    # 直接测漂移函数（引擎里非调仓日就调它）: A +10%, B -10%
    drifted = update_weights({"A": 0.5, "B": 0.5}, {"A": 0.1, "B": -0.1})
    # cash=0, vA=0.55, vB=0.45, total=1.0 → A=0.55, B=0.45
    _assert_close(drifted["A"], 0.55, "测试4 漂移后 A 权重")
    _assert_close(drifted["B"], 0.45, "测试4 漂移后 B 权重")
    if abs(drifted["A"] - 0.5) < TOL:
        raise AssertionError("测试4: 权重未漂移，仍是 50/50")
    if not (drifted["A"] > drifted["B"]):
        raise AssertionError("测试4: 涨的股票权重应大于跌的股票")
    return drifted


# ============================================================
# 测试 5: 多空 — 零投资 ±100%，单期收益 = r_多 − r_空
# ============================================================
def test_long_short():
    # 零投资：多 A +1.0、空 B −1.0（多头和=+1、空头和=−1），weight_mode="long_short"
    nav_ls = _nav(
        _weights([(D0, "A", 1.0), (D0, "B", -1.0)]), PRICE_DF,
        config={"weight_mode": "long_short"}, end_date=D3,
    )
    # D1 单期: 1 + (r_A − r_B) = 1 + (0.1 − (−0.1)) = 1.2（直接相减，不除以2）
    _assert_close(nav_ls[D1], 1.2, "测试5 多空 D1 = 1 + r_A − r_B")

    # 多空应快于纯多头 A 满仓（A 全多，多空额外赚 B 下跌的空头收益）
    nav_long = _nav(_weights([(D0, "A", 1.0)]), PRICE_DF, end_date=D3)
    for d in (D1, D2, D3):
        if not (nav_ls[d] > nav_long[d] + TOL):
            raise AssertionError(
                f"测试5 {d.date()}: 多空 NAV {nav_ls[d]:.6f} 应快于纯多头 {nav_long[d]:.6f}"
            )
    return nav_ls


# ============================================================
# 测试 5b: 零投资持仓的两腿恒等式（锁死漂移模型 = 两腿各自复利）
# ============================================================
def test_long_short_two_leg_identity():
    """持仓 ±100% 跨多天不调仓，验证 NAV_t = ∏(1+r_A) − ∏(1+r_B) + 1。
    这是持有(漂移)的正确口径，不是 ∏(1+r_A−r_B) 的逐日重配口径。"""
    nav = _nav(
        _weights([(D0, "A", 1.0), (D0, "B", -1.0)]), PRICE_DF,
        config={"weight_mode": "long_short"}, end_date=D3,
    )
    # 个股日收益：A 10→11→12→12，B 20→18→18→9
    rA = {D1: 0.1, D2: 12 / 11 - 1, D3: 0.0}
    rB = {D1: -0.1, D2: 0.0, D3: 9 / 18 - 1}
    cumA, cumB = 1.0, 1.0
    for d in (D1, D2, D3):
        cumA *= (1 + rA[d])
        cumB *= (1 + rB[d])
        expected = cumA - cumB + 1.0   # 多头市值 − 空头市值 + 现金($1 保证金)
        _assert_close(nav[d], expected, f"测试5b 两腿恒等式 {d.date()}")
    return nav


# ============================================================
# 测试 6~8: 校验逻辑必须正确 raise
# ============================================================
def _assert_raises(fn, keyword, label):
    try:
        fn()
    except ValueError as e:
        if keyword not in str(e):
            raise AssertionError(f"{label}: raise 了但信息不含 '{keyword}': {e}")
        return
    raise AssertionError(f"{label}: 预期 raise ValueError 但没有")


def test_validation_bad_code():
    # C 不在 price_df → raise
    _assert_raises(
        lambda: run_backtest(_weights([(D0, "C", 1.0)]), PRICE_DF),
        "找不到",
        "测试6 未知 code",
    )


def test_validation_abs_weight():
    # |0.6| + |0.6| = 1.2 > 1 → raise
    _assert_raises(
        lambda: run_backtest(_weights([(D0, "A", 0.6), (D0, "B", 0.6)]), PRICE_DF),
        "绝对值合计超过",
        "测试7 权重绝对值和 > 1",
    )


def test_validation_long_short():
    # long_short 模式：多头和≠+1 → raise（A 只 +0.5、B −1.0）
    _assert_raises(
        lambda: run_backtest(
            _weights([(D0, "A", 0.5), (D0, "B", -1.0)]), PRICE_DF,
            config={"weight_mode": "long_short"}, end_date=D3,
        ),
        "多头和",
        "long_short 多头和≠+1",
    )
    # 空头和≠−1 → raise（A +1.0、B 只 −0.5）
    _assert_raises(
        lambda: run_backtest(
            _weights([(D0, "A", 1.0), (D0, "B", -0.5)]), PRICE_DF,
            config={"weight_mode": "long_short"}, end_date=D3,
        ),
        "空头和",
        "long_short 空头和≠−1",
    )
    # 非法 weight_mode → raise
    _assert_raises(
        lambda: run_backtest(
            _weights([(D0, "A", 1.0)]), PRICE_DF,
            config={"weight_mode": "both"}, end_date=D3,
        ),
        "非法 weight_mode",
        "非法 weight_mode",
    )
    # long_short + 开过滤 → raise
    _assert_raises(
        lambda: run_backtest(
            _weights([(D0, "A", 1.0), (D0, "B", -1.0)]), PRICE_DF,
            config={"weight_mode": "long_short", "enable_feasibility_filter": True},
            end_date=D3,
        ),
        "不支持可行性过滤",
        "long_short + 开过滤",
    )


def test_validation_date_bounds():
    w = _weights([(D1, "A", 1.0)])
    # start_date 早于 weights 最早日期 → raise
    _assert_raises(
        lambda: run_backtest(w, PRICE_DF, start_date="2023-01-03"),
        "早于 weights_df 最早日期",
        "测试8a start_date 越界",
    )
    # end_date 晚于 price_df 最晚日期 → raise
    _assert_raises(
        lambda: run_backtest(w, PRICE_DF, end_date="2023-01-10"),
        "晚于 price_df 最晚日期",
        "测试8b end_date 越界",
    )


# ============================================================
# 阶段 3+5 测试
# ============================================================

# 扩展行情：在阶段2 的 A/B 基础上加 limit_status / trade_status（全正常）
def _price_with_status(limit_overrides=None, status_overrides=None):
    """
    构造带 limit_status / trade_status 的 price_df。
    limit_overrides / status_overrides: dict {(date, code): 值} 覆盖默认（0 / "交易"）。
    """
    df = PRICE_DF.copy()
    df["limit_status"] = 0
    df["trade_status"] = "交易"
    for (d, c), v in (limit_overrides or {}).items():
        df.loc[(df["date"] == d) & (df["code"] == c), "limit_status"] = v
    for (d, c), v in (status_overrides or {}).items():
        df.loc[(df["date"] == d) & (df["code"] == c), "trade_status"] = v
    return df


def test_calc_trades():
    t = calc_trades({"A": 0.5, "B": 0.2}, {"A": 0.2, "B": 0.4, "C": 0.1})
    _assert_close(t["A"], -0.3, "calc_trades A")
    _assert_close(t["B"], 0.2, "calc_trades B")
    _assert_close(t["C"], 0.1, "calc_trades C")
    if calc_trades({}, {}) != {} or calc_trades({"A": 0.5}, {"A": 0.5}) != {}:
        raise AssertionError("calc_trades 空/相同应返回 {}")


def test_calc_cost():
    cfg = {"buy_cost": 0.0003, "sell_cost": 0.0013, "slippage": 0.001}
    # 买 0.3×(0.0003+0.001)=0.00039；卖 0.3×(0.0013+0.001)=0.00069；合 0.00108
    _assert_close(calc_cost({"A": -0.3, "B": 0.2, "C": 0.1}, cfg), 0.00108, "calc_cost 手算")
    _assert_close(calc_cost({}, cfg), 0.0, "calc_cost 空")
    _assert_close(
        calc_cost({"A": 0.3}, {"buy_cost": 0, "sell_cost": 0, "slippage": 0}),
        0.0, "calc_cost 零费率",
    )


def test_skip_small_changes():
    # 阈值=0 原样返回 target
    r = skip_small_changes({"A": 0.5}, {"A": 0.3, "B": 0.2}, 0.0)
    _assert_close(r["A"], 0.3, "skip 阈值0 A"); _assert_close(r["B"], 0.2, "skip 阈值0 B")
    # 阈值0.05：A 变 0.01 跳过(保持0.5)，B 变 0.2 通过
    r = skip_small_changes({"A": 0.5, "B": 0.2}, {"A": 0.51, "B": 0.4}, 0.05)
    _assert_close(r["A"], 0.5, "skip 小变动跳过"); _assert_close(r["B"], 0.4, "skip 大变动通过")
    # 全跳过返回 old
    if skip_small_changes({"A": 0.5}, {"A": 0.51}, 0.05) != {"A": 0.5}:
        raise AssertionError("skip 全跳过应返回 old")


def test_check_tradable():
    NORMAL = {"limit_status": 0, "trade_status": "交易"}
    # 图2 例子：A 跌停锁仓 + B/C 超额缩买
    final, blocked = check_tradable(
        {"A": 0.5, "B": 0.2, "C": 0.2}, {"A": 0.2, "B": 0.4, "C": 0.4},
        {"A": {"limit_status": -1, "trade_status": "交易"}, "B": NORMAL, "C": NORMAL},
    )
    _assert_close(final["A"], 0.5, "check 图2 A 锁仓")
    _assert_close(final["B"], 0.25, "check 图2 B 缩买")
    _assert_close(final["C"], 0.25, "check 图2 C 缩买")
    reasons = sorted(b["reason"] for b in blocked)
    if reasons != ["容量不足", "容量不足", "跌停"]:
        raise AssertionError(f"check 图2 blocked 原因不对: {reasons}")

    # 涨停只拦买不拦卖
    f, b = check_tradable({"A": 0.3}, {"A": 0.5}, {"A": {"limit_status": 1, "trade_status": "交易"}})
    _assert_close(f["A"], 0.3, "涨停拦买");
    if not b or b[0]["reason"] != "涨停":
        raise AssertionError("涨停应记 blocked")
    f, b = check_tradable({"A": 0.5}, {"A": 0.3}, {"A": {"limit_status": 1, "trade_status": "交易"}})
    _assert_close(f["A"], 0.3, "涨停放行卖")
    if b:
        raise AssertionError("涨停卖出不应被拦")

    # 跌停只拦卖不拦买
    f, b = check_tradable({"A": 0.5}, {"A": 0.3}, {"A": {"limit_status": -1, "trade_status": "交易"}})
    _assert_close(f["A"], 0.5, "跌停拦卖锁仓")
    f, b = check_tradable({"A": 0.3}, {"A": 0.5}, {"A": {"limit_status": -1, "trade_status": "交易"}})
    _assert_close(f["A"], 0.5, "跌停放行买")

    # 停牌买卖都拦
    f, b = check_tradable({"A": 0.5}, {"A": 0.3}, {"A": {"limit_status": 0, "trade_status": "停牌"}})
    _assert_close(f["A"], 0.5, "停牌锁仓")

    # 清仓遇阻：target 空但 old 有持仓且跌停 → 锁仓 + blocked sell
    f, b = check_tradable({"A": 0.5}, {}, {"A": {"limit_status": -1, "trade_status": "交易"}})
    _assert_close(f["A"], 0.5, "清仓遇跌停锁仓")
    if not b or b[0]["intended_action"] != "sell":
        raise AssertionError("清仓遇阻应记 blocked sell")

    # ST 漏判反例：limit_status=1 即拦，不看涨幅
    f, b = check_tradable({"A": 0.2}, {"A": 0.5}, {"A": {"limit_status": 1, "trade_status": "交易"}})
    _assert_close(f["A"], 0.2, "ST limit_status=1 拦买")

    # 负权重报错
    raised = False
    try:
        check_tradable({"A": 0.5}, {"A": -0.2}, {"A": NORMAL})
    except ValueError:
        raised = True
    if not raised:
        raise AssertionError("负权重应 raise")

    # 都空
    if check_tradable({}, {}, {}) != ({}, []):
        raise AssertionError("都空应返回 ({}, [])")


def test_friction_le_ideal():
    """同一策略：有摩擦 NAV ≤ 无摩擦 NAV（逐点）。"""
    w = _weights([(D0, "A", 1.0), (D2, "B", 1.0)])
    ideal = run_backtest(w, PRICE_DF, end_date=D3)["nav"]
    cfg = {"buy_cost": 0.001, "sell_cost": 0.002, "slippage": 0.001}
    friction = run_backtest(w, PRICE_DF, config=cfg, end_date=D3)["nav"]
    for d in DATES:
        if friction[d] > ideal[d] + TOL:
            raise AssertionError(f"摩擦 {d.date()}: {friction[d]} > 理想 {ideal[d]}")
    # 至少有一天因调仓被扣（D2 调仓 → D2 起 friction < ideal）
    if not (friction[D2] < ideal[D2] - TOL):
        raise AssertionError("摩擦应在调仓日 D2 扣掉成本")


def test_drift_basis_regression():
    """关键回归：换手/成本以"漂移后权重"为基准，不是早上权重。
    构造目标 = 当天漂移后权重 → 换手/成本应为 0（若错用早上权重会算出虚假交易）。
    """
    # D0 建仓 A/B 各半；D1 调仓，目标设成"D1 漂移后的权重"
    # D1 当天 A +10%、B -10%，漂移后 A=0.55、B=0.45
    target_d1 = update_weights({"A": 0.5, "B": 0.5}, {"A": 0.1, "B": -0.1})
    rows = [(D0, "A", 0.5), (D0, "B", 0.5),
            (D1, "A", target_d1["A"]), (D1, "B", target_d1["B"])]
    cfg = {"buy_cost": 0.01, "sell_cost": 0.01, "slippage": 0.0}
    res = run_backtest(_weights(rows), PRICE_DF, config=cfg, end_date=D3)
    # 找 D1 的调仓记录
    rec_d1 = [r for r in res["trade_records"] if r["date"] == D1]
    if not rec_d1:
        raise AssertionError("应有 D1 调仓记录")
    _assert_close(rec_d1[0]["turnover"], 0.0, "漂移基准: D1 换手应为0")
    _assert_close(rec_d1[0]["cost"], 0.0, "漂移基准: D1 成本应为0（用漂移权重当基准）")


def test_feasibility_through_engine():
    """端到端：开过滤，调仓日某票涨停 → 买入被拦、blocked_trades 有记录。"""
    # D0 全现金（空仓），D1 想买 A 满仓，但 D1 当天 A 涨停 → 买不进
    w = _weights([(D1, "A", 1.0)])
    price = _price_with_status(limit_overrides={(D1, "A"): 1})
    cfg = {"enable_feasibility_filter": True}
    res = run_backtest(w, price, config=cfg, start_date=D1, end_date=D3)
    # D1 想从空仓买到 A=1.0，但涨停拦 → 仍空仓
    blk = [b for b in res["blocked_trades"] if b["date"] == D1]
    if not blk or blk[0]["reason"] != "涨停" or blk[0]["intended_action"] != "buy":
        raise AssertionError(f"涨停应记 blocked buy，实际: {blk}")
    # 买不进 → 当天空仓，NAV 不随 A 涨跌变化（保持 1.0 一线）
    _assert_close(res["nav"][D1], 1.0, "涨停买不进 → D1 仍空仓 NAV=1.0")

    # 开过滤但缺列 → raise
    _assert_raises(
        lambda: run_backtest(w, PRICE_DF, config={"enable_feasibility_filter": True},
                             start_date=D1, end_date=D3),
        "缺列",
        "开过滤缺 limit_status/trade_status 列",
    )


def test_calc_benchmark():
    idx = pd.DataFrame({"date": DATES, "close": [3000.0, 3300.0, 3600.0, 3600.0]})
    nav = calc_benchmark(idx, D0, D3)
    _assert_close(nav.iloc[0], 1.0, "基准起点1.0")
    _assert_close(nav.loc[D1], 1.1, "基准 D1 = 3300/3000")
    _assert_close(nav.loc[D2], 1.2, "基准 D2 = 3600/3000")
    # 起始日非交易日 → 用其后最近交易日
    nav2 = calc_benchmark(idx, pd.Timestamp("2023-01-01"), D3)
    _assert_close(nav2.iloc[0], 1.0, "基准非交易日起点仍1.0")
    if nav2.index[0] != D0:
        raise AssertionError("基准非交易日应用其后最近交易日 D0")


def test_weights_history():
    """run_backtest 输出的 weights：调仓日权重 = 该日 final_weights，且每日有记录。"""
    res = run_backtest(_weights([(D0, "A", 1.0), (D2, "B", 1.0)]), PRICE_DF, end_date=D3)
    w = res["weights"]
    # 每个交易日都有一行
    for d in DATES:
        if d not in w.index:
            raise AssertionError(f"weights 缺交易日 {d.date()}")
    # D0 调仓 A=1.0
    _assert_close(w.loc[D0, "A"], 1.0, "weights D0 A=1.0")
    # D2 调仓换 B：D2 生效后 B=1.0、A=0（次日生效语义下 D2 当日收益用旧权重，但权重已切到 B）
    _assert_close(w.loc[D2, "B"], 1.0, "weights D2 B=1.0")
    _assert_close(w.loc[D2, "A"], 0.0, "weights D2 A=0")
    # D1 非调仓日：A 持仓（从 D0 漂移而来，权重仍≈1）
    if w.loc[D1, "A"] <= 0:
        raise AssertionError("weights D1 应仍持有 A")


# ============================================================
# 主运行
# ============================================================
def main():
    tests = [
        ("1 单股满仓", test_single_full),
        ("2 单股半仓", test_single_half),
        ("3 调仓时序", test_rebalance_timing),
        ("4 权重漂移", test_weight_drift),
        ("5 多空±100%", test_long_short),
        ("5b 多空两腿恒等式", test_long_short_two_leg_identity),
        ("6 未知 code raise", test_validation_bad_code),
        ("7 权重绝对值和 raise", test_validation_abs_weight),
        ("8 long_short 校验 raise", test_validation_long_short),
        ("9 日期越界 raise", test_validation_date_bounds),
        ("10 calc_trades", test_calc_trades),
        ("11 calc_cost", test_calc_cost),
        ("12 skip_small_changes", test_skip_small_changes),
        ("13 check_tradable", test_check_tradable),
        ("14 有摩擦≤无摩擦", test_friction_le_ideal),
        ("15 漂移基准回归", test_drift_basis_regression),
        ("16 可行性过滤端到端", test_feasibility_through_engine),
        ("17 calc_benchmark", test_calc_benchmark),
        ("18 每日权重记录", test_weights_history),
    ]
    for name, fn in tests:
        fn()
        print(f"  ✓ 测试{name}")
    print(f"\n全部 {len(tests)} 个验收测试通过")


if __name__ == "__main__":
    main()
