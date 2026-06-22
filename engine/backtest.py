"""
回测引擎
weights_df + price_df → NAV 曲线

阶段 2（无摩擦）:
  calc_daily_returns  → 算日收益率
  update_weights      → 权重漂移
  run_backtest        → 主循环，输出结果包

阶段 3+5（交易摩擦 + 可行性过滤）:
  calc_trades         → 算每只股票的买卖量
  calc_cost           → 算调仓的总交易成本
  skip_small_changes  → 调仓阈值，变动太小不调
  check_tradable      → 涨跌停/停牌过滤（多头语义）
  calc_benchmark      → 指数收盘价 → 基准 NAV
"""

import pandas as pd
import numpy as np


# 默认配置：全 0 / False 时退化为阶段 2（无摩擦、无过滤）
DEFAULT_CONFIG = {
    "buy_cost": 0.0,                  # 买入手续费率
    "sell_cost": 0.0,                 # 卖出手续费率（含印花税）
    "slippage": 0.0,                  # 滑点（买卖对称，作附加成本扣）
    "rebalance_threshold": 0.0,       # 调仓阈值，权重变动 < 此值不调
    "enable_feasibility_filter": False,  # 是否开涨跌停/停牌过滤
    "weight_mode": "long_only",       # 权重口径："long_only"(Σ|w|≤1) / "long_short"(零投资±100%)
}


def _merge_config(config: dict) -> dict:
    """config=None 或缺键时用 DEFAULT_CONFIG 补齐，返回完整配置（不改原 dict）。

    未知键直接 raise（堵 buy_cost 拼成 buycost 之类静默回落零摩擦：拼错的键不被消费、
    实际跑的是默认零成本，回测无任何告警地 NAV 偏高）。
    """
    if config:
        unknown = set(config) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f"run_backtest config 含未知键 {sorted(unknown)}；合法键为 {sorted(DEFAULT_CONFIG)}")
    merged = dict(DEFAULT_CONFIG)
    if config:
        merged.update(config)
    return merged


# ============================================================
# 函数 1: calc_daily_returns
# ============================================================
def calc_daily_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    输入:
      price_df — DataFrame, columns: [date, code, adj_close]
                 date: datetime, code: str, adj_close: float

    输出:
      DataFrame, columns: [date, code, daily_return]
      daily_return = adj_close_t / adj_close_{t-1} - 1
      每只股票按 date 排序后，首日的 daily_return 为 NaN（保留）

    依赖: 无

    边界条件:
      - 每只股票的首个交易日 → daily_return = NaN，保留不处理
      - adj_close 为 NaN 的行先丢弃（NaN 价 ≡ 当天无有效价，等同缺行）：避免 pct_change 默认
        fill_method='pad' 把 NaN 价前向填充成"收益 0"、再把复牌日算成跨段全程收益。丢弃后这些
        (date,code) 在下游 run_backtest 主循环走 no_row 分支记录，NAV 逐点不变（缺失那天仍按 0 推进）。
    """
    df = price_df[["date", "code", "adj_close"]].copy()
    df = df.dropna(subset=["adj_close"])
    df = df.sort_values(["code", "date"])
    df["daily_return"] = df.groupby("code")["adj_close"].pct_change(fill_method=None)
    return df[["date", "code", "daily_return"]]


# ============================================================
# 函数 2: update_weights
# ============================================================
def update_weights(old_weights: dict, daily_returns: dict) -> dict:
    """
    输入:
      old_weights  — dict, {code: weight}，不含现金（现金 = 1 - sum(weights)）
      daily_returns — dict, {code: daily_return}

    输出:
      dict, {code: drifted_weight}，不含现金

    漂移公式:
      new_value_i = old_weight_i × (1 + return_i)
      cash_value  = old_cash × 1.0  （现金收益率 = 0）
      new_nav     = sum(new_values) + cash_value
      drifted_weight_i = new_value_i / new_nav

    依赖: 无

    边界条件:
      - daily_returns 中某只股票缺失或为 NaN → 视为收益率 0（停牌不动）
      - old_weights 为空 dict → 全现金，返回空 dict
    """
    if not old_weights:
        return {}

    cash = 1.0 - sum(old_weights.values())

    new_values = {}
    for code, w in old_weights.items():
        ret = daily_returns.get(code, 0.0)
        if np.isnan(ret):
            ret = 0.0
        new_values[code] = w * (1.0 + ret)

    total = sum(new_values.values()) + cash
    # 组合当日价值归零/转负（long_short 极端亏损时分母可过零）→ 漂移无定义，fail-fast，
    # 不让裸 ZeroDivisionError 或符号翻转的权重静默往下走。
    if total <= 1e-12:
        raise ValueError(
            f"update_weights: 组合当日价值 total={total:.6e} ≤ 0，权重漂移无定义"
            f"（long_short 两腿合计亏掉本金）；持仓 {len(old_weights)} 只、现金 {cash:.4f}")
    return {code: v / total for code, v in new_values.items()}


# ============================================================
# 阶段 3：摩擦层 — calc_trades / calc_cost / skip_small_changes
# ============================================================
def calc_trades(old_weights: dict, new_weights: dict) -> dict:
    """
    算每只股票的买卖量（调仓前后权重之差）。

    输入:
      old_weights — dict {code: weight}，调仓前权重
      new_weights — dict {code: weight}，调仓后权重

    输出:
      dict {code: trade}，trade = new - old；正 = 买入，负 = 卖出
      变动为 0 的不收录；两边都空或完全相同 → {}

    依赖: 无
    """
    codes = set(old_weights) | set(new_weights)
    trades = {}
    for code in codes:
        trade = new_weights.get(code, 0.0) - old_weights.get(code, 0.0)
        if trade != 0.0:
            trades[code] = trade
    return trades


def calc_cost(trades: dict, config: dict) -> float:
    """
    算本次调仓的总交易成本（占 NAV 的比例，≥ 0）。

    输入:
      trades — dict {code: trade}，calc_trades 的输出
      config — dict，含 buy_cost / sell_cost / slippage

    输出:
      float ≥ 0
      买入成本 = Σ(trade>0) trade × (buy_cost + slippage)
      卖出成本 = Σ(trade<0) |trade| × (sell_cost + slippage)
      trades 空或费率全 0 → 0.0

    依赖: 无
    """
    buy_rate = config["buy_cost"] + config["slippage"]
    sell_rate = config["sell_cost"] + config["slippage"]
    cost = 0.0
    for trade in trades.values():
        if trade > 0:
            cost += trade * buy_rate
        else:
            cost += -trade * sell_rate
    return cost


def skip_small_changes(old_weights: dict, target_weights: dict, threshold: float) -> dict:
    """
    调仓阈值过滤：权重变动太小的不调，避免白付手续费。

    输入:
      old_weights    — dict {code: weight}，当前（漂移后）权重
      target_weights — dict {code: weight}，策略目标权重
      threshold      — float，变动绝对值 < 此值则不调

    输出:
      dict {code: weight}，过滤后权重
      |target - old| >= threshold → 用 target；否则 → 保持 old
      不重新归一化、不塞满仓；现金由最终留下来的权重自然决定
      （跳过部分买卖后最终现金不一定等于 target 现金比例，这是对的，不补满）

    边界:
      threshold = 0 → 原样返回 target_weights
      全部 < threshold → 返回 old_weights（本次不调）
      old 为空（首次建仓）→ 变动 = target 本身，通常全部通过

    依赖: 无
    """
    codes = set(old_weights) | set(target_weights)
    result = {}
    for code in codes:
        old_w = old_weights.get(code, 0.0)
        target_w = target_weights.get(code, 0.0)
        # 开仓豁免阈值：old_w==0 时强制取 target（阈值本意是省微调手续费，不该阻止建仓；
        # 否则 threshold > 单票目标权重会让所有仓位永不建立、NAV 静默卡死 1.0）
        chosen = target_w if (old_w == 0.0 or abs(target_w - old_w) >= threshold) else old_w
        if chosen != 0.0:
            result[code] = chosen
    return result


# ============================================================
# 阶段 5：可行性过滤 — check_tradable（多头语义）
# ============================================================
def check_tradable(old_weights: dict, target_weights: dict, day_data: dict):
    """
    涨跌停 / 停牌过滤：把今天没法交易的过滤掉（A 股多头语义）。

    输入:
      old_weights    — dict {code: weight}，交易前（漂移后）权重，作为锁仓基准
      target_weights — dict {code: weight}，已过调仓阈值的目标权重
      day_data       — dict {code: {"limit_status": -1/0/1, "trade_status": str}}
                       limit_status 来自衍生表（Wind 板块自适应涨跌停标记）

    输出:
      (final_weights: dict, blocked_trades: list[dict])
      blocked_trades 每条 {code, reason, intended_action, blocked_weight}（date 由调用方补）
      reason ∈ {"涨停", "跌停", "停牌", "无数据", "容量不足"}

    不可交易规则（涨跌停单向拦、停牌/无数据双向拦）:
      涨停 limit_status==1  → 只拦买入，不拦卖出
      跌停 limit_status==-1 → 只拦卖出，不拦买入
      停牌 trade_status=="停牌" → 买卖都拦
      day_data 缺该 code（当天无行）→ 买卖都拦，reason="无数据"

    处理:
      想买买不进 → final = old（差额留现金）
      想卖卖不掉/停牌 → final = old（锁仓）
      锁仓导致 Σfinal > 1 → 能买的买入增量按各自想买量等比例缩小到刚好放下，
                            不重新分配；缩买少买的部分记 reason="容量不足"

    边界 / 保险:
      old 或 target 含任意负权重 → raise ValueError（带 code+权重值），落实"严格版只做多"
      old 和 target 都空 → 返回 ({}, [])；否则照常遍历 old ∪ target

    依赖: 无
    """
    # 做空保险：严格版只做多
    for label, weights in (("old_weights", old_weights), ("target_weights", target_weights)):
        for code, w in weights.items():
            if w < 0:
                raise ValueError(
                    f"check_tradable 不支持负权重（严格版只做多）：{label} 中 {code}={w}"
                )

    if not old_weights and not target_weights:
        return {}, []

    codes = set(old_weights) | set(target_weights)
    final = {}
    blocked = []
    # 想买的（trade>0）先收集起来，确认容量后再决定实买多少
    pending_buys = {}  # code -> {"old": old_w, "want": target_w, "delta": 买入增量}

    for code in codes:
        old_w = old_weights.get(code, 0.0)
        target_w = target_weights.get(code, 0.0)
        trade = target_w - old_w

        info = day_data.get(code)
        if info is None:
            reason = "无数据"
            limit_status, suspended = 0, True
        else:
            limit_status = info["limit_status"]
            suspended = info["trade_status"] == "停牌"
            reason = "停牌" if suspended else ("涨停" if limit_status == 1 else "跌停")

        if trade == 0.0:
            # 不动的仓位：直接保留（含 old==target!=0、或两边都 0 不会进来）
            if old_w != 0.0:
                final[code] = old_w
            continue

        if trade > 0:
            # 想买入
            can_buy = (not suspended) and (limit_status != 1)
            if can_buy:
                pending_buys[code] = {"old": old_w, "want": target_w, "delta": trade}
            else:
                # 买不进 → 锁旧权重（差额留现金）
                if old_w != 0.0:
                    final[code] = old_w
                blocked.append({
                    "code": code, "reason": reason,
                    "intended_action": "buy", "blocked_weight": trade,
                })
        else:
            # 想卖出（trade<0）
            can_sell = (not suspended) and (limit_status != -1)
            if can_sell:
                final[code] = target_w
            else:
                # 卖不掉 → 锁仓
                final[code] = old_w
                blocked.append({
                    "code": code, "reason": reason,
                    "intended_action": "sell", "blocked_weight": trade,
                })

    # 容量检查：已占空间 = final（锁仓/不动/买不进保留的）+ 想买股票自身的旧权重
    # （想买股票的旧权重本就占着仓位，只有"买入增量 delta"需要新空间）
    occupied = sum(final.values()) + sum(b["old"] for b in pending_buys.values())
    remaining = 1.0 - occupied
    want_total = sum(b["delta"] for b in pending_buys.values())

    if want_total <= remaining + 1e-12:
        # 容量够，全额买入
        for code, b in pending_buys.items():
            final[code] = b["want"]
    else:
        # 超额：按想买量等比例缩小到刚好放下。
        # [修] 容量不足按"当天一次事件"聚合记一条(blocked_weight=当天总缩买额)，不再每只待买票
        # 各记一条——后者会因满仓+高换手+1只锁仓就把笔数扇成几十条，严重夸大容量瓶颈。
        scale = remaining / want_total if want_total > 0 else 0.0
        total_short = 0.0
        for code, b in pending_buys.items():
            actual_delta = b["delta"] * scale
            final[code] = b["old"] + actual_delta
            total_short += b["delta"] - actual_delta
        if total_short > 1e-9:                      # 浮点/微小缩买不算被拦
            blocked.append({
                "code": f"<{len(pending_buys)}只>", "reason": "容量不足",
                "intended_action": "buy", "blocked_weight": total_short,
            })

    final = {c: w for c, w in final.items() if w != 0.0}
    return final, blocked


# ============================================================
# 函数 3: run_backtest
# ============================================================
def run_backtest(
    weights_df: pd.DataFrame,
    price_df: pd.DataFrame,
    config=None,
    start_date=None,
    end_date=None,
) -> dict:
    """
    输入:
      weights_df — DataFrame, columns: [date, code, weight]
                   weight 可正可负：正 = 做多，负 = 做空；约束是 Σ|weight| ≤ 1.0，
                   现金权重 = 1 - Σweight（多空对冲时现金会大于 1 - Σ多头）
      price_df   — DataFrame, columns: [date, code, adj_close]
                   开可行性过滤时额外需要 [limit_status, trade_status]
                   （limit_status 由调用方预先从衍生表 merge 进来）
      config     — dict 或 None，缺键用 DEFAULT_CONFIG 补齐；全默认时退化为阶段 2
      start_date — 默认 None → weights_df 最早日期
      end_date   — 默认 None → weights_df 最晚日期

    输出:
      dict {
        'nav':            pd.Series, index=交易日 datetime, value=NAV（初始值 1.0）
        'weights':        pd.DataFrame, index=交易日, columns=code, value=当日生效后权重（缺失=0）
        'trade_records':  list[dict]，每条 {date, trades, cost, turnover}
        'blocked_trades': list[dict]，每条 {date, code, reason, intended_action, blocked_weight}
        'missing_log':    list[dict]，每条 {date, code, weight, reason}；持仓票当天缺行(no_row)
                          或 NaN 收益(nan_return)被当 0 的记录（行为不变，仅暴露；多为退市/长停）
      }

    依赖: calc_daily_returns, update_weights, skip_small_changes, calc_trades,
          calc_cost, check_tradable

    校验规则:
      1. weights_df 的 code 必须全部在 price_df 中存在
      2. weights_df 的所有调仓日必须是 price_df 中的交易日，否则 raise
      3. 按 config["weight_mode"] 分叉：
         - long_only（默认）：每调仓日 Σ|weight| ≤ 1.0 + 1e-9
         - long_short：每调仓日 多头和≈+1、空头和≈−1（零投资 ±100%）
         - 非法值 → raise
      4. start_date 不能早于 weights_df 最早日期（不传则用 weights_df 最早日期）
      5. end_date 不能晚于 price_df 最晚日期（不传则用 weights_df 最晚日期）
      6. 开可行性过滤但 price_df 缺 limit_status / trade_status 列 → raise；
         long_short + 开可行性过滤 → raise（多空可行性未实现）

    主循环（收盘调仓）:
      每天：当日收益用旧权重算 → 净值更新 → 漂移得到交易前权重
      调仓日：以漂移权重为基准走流水线（阈值 → 过滤 → 交易量 → 成本），净值再扣成本
      非调仓日：当前权重 = 漂移权重
      全默认 config 时净值逐点等于阶段 2（golden test）
    """

    # =========================
    # 第一步：输入校验
    # =========================

    config = _merge_config(config)

    # 0. weights_df 结构检查：NaN 权重 / 重复 (date,code)（否则校验口径=按行求和、执行口径=dict
    #    去重只留末行，两者不一致会静默丢权重 / 误判超限）
    nan_w = weights_df[weights_df["weight"].isna()]
    if len(nan_w):
        r = nan_w.iloc[0]
        raise ValueError(f"weights_df 的 weight 含 NaN，首个 {r['code']}@{pd.Timestamp(r['date']).date()}（共 {len(nan_w)} 处）")
    dup = weights_df.duplicated(["date", "code"])
    if dup.any():
        r = weights_df[dup].iloc[0]
        raise ValueError(f"weights_df 存在重复 (date,code)，首个 {r['code']}@{pd.Timestamp(r['date']).date()}"
                         f"（共 {int(dup.sum())} 处）；校验按行求和、执行按 dict 去重，重复会静默丢权重")

    # 1. code 存在性检查
    weight_codes = set(weights_df["code"].unique())
    price_codes = set(price_df["code"].unique())
    missing_codes = weight_codes - price_codes
    if missing_codes:
        raise ValueError(
            f"weights_df 中以下 code 在 price_df 中找不到: {missing_codes}"
        )

    # 2. weights_df 的所有调仓日必须是 price_df 中的交易日
    trade_dates_set = set(price_df["date"].unique())
    weight_dates = set(weights_df["date"].unique())
    non_trade_dates = sorted(weight_dates - trade_dates_set)
    if non_trade_dates:
        raise ValueError(
            f"weights_df 中以下 {len(non_trade_dates)} 个调仓日不是 price_df 的交易日"
            f"（样本前 10 个）: {non_trade_dates[:10]}"
        )

    # 3. 权重结构校验，按 weight_mode 分叉
    weight_mode = config["weight_mode"]
    if weight_mode == "long_only":
        # 纯多头/含现金：每个调仓日 Σ|weight| ≤ 1.0 + 1e-9
        abs_weight_sums = weights_df.groupby("date")["weight"].apply(lambda x: x.abs().sum())
        over_days = abs_weight_sums[abs_weight_sums > 1.0 + 1e-9]
        if not over_days.empty:
            raise ValueError(
                f"long_only 模式下以下调仓日权重绝对值合计超过 1.0:\n{over_days}"
            )
    elif weight_mode == "long_short":
        # 零投资 ±100%：每个调仓日 多头和≈+1、空头和≈−1
        for date, group in weights_df.groupby("date"):
            w = group["weight"]
            long_sum = w[w > 0].sum()
            short_sum = w[w < 0].sum()
            if abs(long_sum - 1.0) > 1e-9 or abs(short_sum + 1.0) > 1e-9:
                raise ValueError(
                    f"long_short 模式要求多头和=+1、空头和=−1（零投资）；"
                    f"调仓日 {pd.Timestamp(date).date()} 实际 多头和={long_sum:.6f}、空头和={short_sum:.6f}"
                )
    else:
        raise ValueError(
            f"非法 weight_mode: {weight_mode!r}（只支持 'long_only' / 'long_short'）"
        )

    # 6. 开可行性过滤但 price_df 缺 limit_status / trade_status 列 → raise
    if config["enable_feasibility_filter"]:
        if weight_mode == "long_short":
            raise ValueError(
                "long_short 模式不支持可行性过滤（多空的涨跌停/停牌过滤未实现）；"
                "多空仅在理想态（enable_feasibility_filter=False）运行"
            )
        missing_cols = [c for c in ("limit_status", "trade_status") if c not in price_df.columns]
        if missing_cols:
            raise ValueError(
                f"enable_feasibility_filter=True 但 price_df 缺列: {missing_cols}"
            )

    # 4. 确定 start_date / end_date
    wdf_min_date = weights_df["date"].min()
    wdf_max_date = weights_df["date"].max()
    pdf_min_date = price_df["date"].min()
    pdf_max_date = price_df["date"].max()

    if start_date is None:
        start_date = wdf_min_date
    else:
        start_date = pd.Timestamp(start_date)

    if end_date is None:
        end_date = wdf_max_date
    else:
        end_date = pd.Timestamp(end_date)

    # start_date 不能早于 weights_df 最早日期
    if start_date < wdf_min_date:
        raise ValueError(
            f"start_date ({start_date}) 早于 weights_df 最早日期 ({wdf_min_date})"
        )

    # end_date 不能晚于 price_df 最晚日期
    if end_date > pdf_max_date:
        raise ValueError(
            f"end_date ({end_date}) 晚于 price_df 最晚日期 ({pdf_max_date})"
        )

    # start_date 不能早于 price_df 最早日期
    if start_date < pdf_min_date:
        raise ValueError(
            f"start_date ({start_date}) 早于 price_df 最早日期 ({pdf_min_date})"
        )

    # =========================
    # 第二步：准备数据
    # =========================

    # 算日收益率
    returns_df = calc_daily_returns(price_df)

    # 交易日序列：price_df 中 start_date 到 end_date 的所有唯一日期
    all_dates = sorted(price_df["date"].unique())
    trade_dates = [d for d in all_dates if start_date <= d <= end_date]

    # 日收益率整理成 {date: {code: return}} 便于循环中查找
    returns_lookup = {}
    for date, group in returns_df.groupby("date"):
        returns_lookup[date] = dict(zip(group["code"], group["daily_return"]))

    # weights_df 整理成 {date: {code: weight}}
    rebalance_lookup = {}
    for date, group in weights_df.groupby("date"):
        rebalance_lookup[date] = dict(zip(group["code"], group["weight"]))

    # 当天行情查询表 {date: {code: {limit_status, trade_status}}}，仅开过滤时构造
    feasibility_on = config["enable_feasibility_filter"]
    day_data_lookup = {}
    if feasibility_on:
        for date, group in price_df.groupby("date"):
            day_data_lookup[date] = {
                code: {"limit_status": ls, "trade_status": ts}
                for code, ls, ts in zip(
                    group["code"], group["limit_status"], group["trade_status"]
                )
            }

    # =========================
    # 第三步：逐日循环
    # =========================

    current_weights = {}  # 空 = 全现金
    nav = 1.0
    nav_series = {}
    weights_history = {}  # {date: {code: weight}}，记当日生效后权重（持仓类图用）
    trade_records = []    # 每次调仓的流水
    blocked_trades = []   # 被涨跌停/停牌挡下的记录
    missing_log = []      # 持仓票当天缺行/NaN 收益被当 0 的记录（每条 date/code/weight/reason）

    for t in trade_dates:
        # 1) 取当日收益率
        today_returns = returns_lookup.get(t, {})

        # 2) 当日组合收益 = Σ(早上权重_i × return_i)；持仓票缺行/NaN 收益仍按 0 计（行为不变），
        #    但记进 missing_log 暴露出来（缺行=no_row 多为退市/长停，NaN=nan_return）。不 raise。
        portfolio_return = 0.0
        for code, w in current_weights.items():
            if code not in today_returns:
                if w != 0.0:
                    missing_log.append({"date": t, "code": code, "weight": w, "reason": "no_row"})
                ret = 0.0
            else:
                ret = today_returns[code]
                if np.isnan(ret):
                    if w != 0.0:
                        missing_log.append({"date": t, "code": code, "weight": w, "reason": "nan_return"})
                    ret = 0.0
            portfolio_return += w * ret

        # 3) 更新 NAV
        nav = nav * (1.0 + portfolio_return)

        # 4) 漂移：早上权重经当天涨跌 → 收盘那一刻的交易前真实权重
        pre_trade_weights = update_weights(current_weights, today_returns)

        # 5) 调仓日走流水线（以漂移权重为基准）；非调仓日直接采用漂移权重
        if t in rebalance_lookup:
            target = rebalance_lookup[t]
            filtered = skip_small_changes(
                pre_trade_weights, target, config["rebalance_threshold"]
            )
            if feasibility_on:
                filtered, blocked = check_tradable(
                    pre_trade_weights, filtered, day_data_lookup.get(t, {})
                )
                for b in blocked:
                    b["date"] = t
                    blocked_trades.append(b)
            elif weight_mode == "long_only":
                # 关过滤路径：skip_small_changes 逐票独立取舍，可能跳过卖出(保留较大旧权重)+执行买入
                # → Σ|w|>1 = 负现金免息杠杆。fail-fast 暴露（开过滤时 check_tradable 容量分支已压回 1）。
                # 仅 long_only 适用（long_short 合计绝对敞口本就≈2，不是杠杆）。
                gross = sum(abs(w) for w in filtered.values())
                if gross > 1.0 + 1e-9:
                    top = sorted(filtered.items(), key=lambda kv: -abs(kv[1]))[:5]
                    raise ValueError(
                        f"调仓日 {pd.Timestamp(t).date()} 阈值过滤后 Σ|w|={gross:.6f} > 1（负现金杠杆）："
                        f"skip_small_changes 跳过卖出+执行买入所致，rebalance_threshold="
                        f"{config['rebalance_threshold']}；前5大持仓 {top}")

            trades = calc_trades(pre_trade_weights, filtered)
            cost = calc_cost(trades, config)
            nav = nav * (1.0 - cost)
            current_weights = filtered

            trade_records.append({
                "date": t,
                "trades": trades,
                "cost": cost,
                "turnover": sum(abs(v) for v in trades.values()),
            })
        else:
            current_weights = pre_trade_weights

        # 6) 记录 NAV + 当日生效后权重
        nav_series[t] = nav
        weights_history[t] = dict(current_weights)

    # 每日权重整理成 DataFrame（index=交易日, columns=code, 缺失=0）
    weights_df_out = pd.DataFrame.from_dict(weights_history, orient="index").sort_index()
    weights_df_out = weights_df_out.fillna(0.0)

    return {
        "nav": pd.Series(nav_series, name="nav"),
        "weights": weights_df_out,
        "trade_records": trade_records,
        "blocked_trades": blocked_trades,
        "missing_log": missing_log,
    }


# ============================================================
# 阶段 6：基准 — calc_benchmark
# ============================================================
def calc_benchmark(index_df: pd.DataFrame, start_date, end_date) -> pd.Series:
    """
    指数收盘价 → 基准 NAV 曲线（和策略 NAV 同口径，起点 1.0）。

    输入:
      index_df   — DataFrame, columns: [date, close]（来自 index_eod 缓存）
      start_date — 起始日；非交易日则用其后最近的交易日
      end_date   — 截止日

    输出:
      pd.Series, name="benchmark_nav", index=date, value=NAV（起点 1.0）
      nav = close / close[首日]

    边界:
      start_date 非交易日 → 用区间内首个 date >= start_date
      区间内为空 → raise ValueError（带 start/end）

    依赖: 无
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    df = index_df[["date", "close"]].sort_values("date")
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    if df.empty:
        raise ValueError(
            f"calc_benchmark: 区间 [{start.date()}, {end.date()}] 内无指数数据"
        )

    base = df["close"].iloc[0]
    nav = df["close"] / base
    return pd.Series(nav.values, index=df["date"].values, name="benchmark_nav")