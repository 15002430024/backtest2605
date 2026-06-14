# -*- coding: utf-8 -*-
"""现金账户逐日撮合回测引擎（阶段 7a）——纯 weights_df 执行器。

吃外部 weights_df（与权重引擎 engine/backtest.py 同一份），按调仓日驱动逐日撮合：维护每只
股票的持仓股数加一笔现金，调仓日按 exec_price（vwap/close/open）成交、扣双边成本(fee+slippage)、
非调仓日只随收盘价漂移，按真实 close 估值，逐日滚出净值与超额。

与权重引擎并列、互补：权重引擎答"因子有没有 alpha"（百分比），现金引擎答"一亿实盘实际能赚多少"
（真整手/真现金）。两者吃同一份 weights_df → 苹果对苹果对标，差额即现金约束代价。

职责切分（两类约束分家）：
- 策略约束（选什么/占多少/多久调）→ 上游 factor_to_weights，不在本引擎。
- 执行约束（当日涨跌停/停牌/有价、整手、现金、换手）→ 本引擎。停牌的票留在 target，
  填不进记 blocked_log，不在选股阶段假装它不存在。

数据全部来自框架缓存（data/cache，经 data/loaders 读出）：
- 成交价 = daily 的 exec_price 选定列、估值价 = daily.close、复权因子 = daily.adj_factor
- 停牌 = daily.trade_status=="停牌"；涨跌停 = derivative.limit_status(1涨停/-1跌停/0正常/NA)
- 退市日 = asharedescription（真实退市日，judge 退市无前视，不再用窗口内最后有效成交日）
ST 不在执行层拦（剔不剔归策略层 exclude_st）。
费率买卖分开：buy_fee(默认 3bp，不含印花)/sell_fee(默认 13bp，含印花)，与权重引擎 buy_cost/sell_cost 对齐。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from data.loaders import (
    load_calendar, load_daily_df, load_derivative_df, load_index_eod, load_delist_dates,
)
from engine.backtest import calc_benchmark
from analysis.metrics import calc_metrics

# 最小申报数量（交易所规则，非数据）：
# 科创板 688/689 → 最低 200 股、200 股以上以 1 股为单位递增（不是 200 整数倍）；
# 其余（主板/创业板）→ 100 股整数倍。
# 注：北交所（4/8 开头）有自己的规则，数据集已排除、此处按 100 整数倍兜底，不单独建模。
STAR_PREFIX = ("688", "689")
STAR_MIN_QTY = 200
MAIN_LOT = 100

SUSPEND_FLAG = "停牌"  # daily.trade_status 里只有这个值是停牌（XD/DR/XR/N 都是正常交易）

# calc_metrics 的输出键，按绝对/超额拆进 BacktestResult.metrics_abs / metrics_excess
_ABS_KEYS = ["总收益率", "年化收益", "年化波动", "夏普", "最大回撤",
             "最大回撤起", "最大回撤止", "Calmar", "日胜率", "盈亏比", "交易天数"]
_EXCESS_KEYS = ["超额年化", "超额最大回撤", "跟踪误差", "信息比率", "Beta"]


@dataclass(frozen=True)
class MarketData:
    """现金引擎的只读市场底座。所有宽表对齐到同一主轴：行=交易日(DatetimeIndex)，列=代码。"""
    trade_price: pd.DataFrame   # 成交价 = daily 的 exec_price 选定列（vwap/close/open 之一，真实价）
    close_price: pd.DataFrame   # 估值价 = daily.close（恒为真实收盘价，与 exec_price 无关）
    adj: pd.DataFrame           # 后复权因子 = daily.adj_factor，复权调整用
    trade_status: pd.DataFrame  # daily.trade_status，=="停牌" 判停牌
    limit_status: pd.DataFrame  # derivative.limit_status，1涨停/-1跌停/0正常/NaN
    is_star: pd.Series          # bool，是否科创板（688/689）；决定最小申报规则
    delist_date: pd.Series      # 每只票的真实退市日（Timestamp/NaT=未退市），判退市用（无前视）
    calendar: pd.DatetimeIndex  # 交易日主轴


@dataclass
class BacktestConfig:
    initial_capital: float = 1e8
    buy_fee: float = 0.0003                 # 买入费率（佣金，不含印花），3bp；与权重引擎 buy_cost 对齐
    sell_fee: float = 0.0013                # 卖出费率（佣金+印花），13bp；与权重引擎 sell_cost 对齐
    slippage: float = 0.0                   # 单边滑点/冲击附加成本，买卖各叠加（0=只算佣金印花）
    turnover_cap: Optional[float] = None    # 单边换手硬上限（None=不卡，默认；调仓日驱动无需减速阀，留作流动性约束）
    exec_price: str = "vwap"                # 成交价口径：'vwap'|'close'|'open'（估值始终用 close）
    start_date: Union[int, str, pd.Timestamp, None] = None  # None=自动取首个可建仓交易日

    def __post_init__(self):
        if self.turnover_cap is not None and self.turnover_cap <= 0:
            raise ValueError(f"turnover_cap 须 >0 或 None（不卡），当前 {self.turnover_cap}")
        for name, v in (("buy_fee", self.buy_fee), ("sell_fee", self.sell_fee), ("slippage", self.slippage)):
            if not (0 <= v < 1):
                raise ValueError(f"{name} 必须在 [0,1)，当前 {v}")
        if self.exec_price not in ("vwap", "close", "open"):
            raise ValueError(f"exec_price 须为 'vwap'|'close'|'open'，当前 {self.exec_price!r}")
        if self.initial_capital <= 0:
            raise ValueError(f"initial_capital 必须 >0，当前 {self.initial_capital}")


@dataclass
class BacktestResult:
    strategy_nav: pd.Series       # 策略净值 = 总资产/初始资金，首点为 1
    excess_nav: pd.Series         # 超额净值，日算术超额累乘，起点 1
    benchmark_nav: pd.Series      # 基准净值，回测起点归一到 1
    holdings: pd.DataFrame        # 每日持仓股数 date×code
    account: pd.DataFrame         # 每日 现金/股票市值/总资产
    trade_log: pd.DataFrame       # 每只票一条完整买卖往返
    trades: pd.DataFrame          # 成交流水
    metrics_abs: dict             # 绝对指标（calc_metrics 绝对部分）
    metrics_excess: dict          # 超额指标（calc_metrics 超额部分）
    missing_log: pd.DataFrame     # 退市清算留痕
    blocked_log: pd.DataFrame     # 目标未达成留痕（涨跌停/停牌/整手不足/现金耗尽/换手限额）；观测性，不改 NAV


# ===================== prep：从缓存装配市场底座 =====================
def _infer_is_star(tickers: pd.Index) -> pd.Series:
    """是否科创板（688/689）。科创板最小申报 200 股、之后 1 股递增；其余 100 整数倍。"""
    return pd.Series(tickers.str[:3].isin(STAR_PREFIX), index=tickers)


def _round_down_lot(qty: float, is_star: bool) -> float:
    """按交易所最小申报规则把股数向下取到合法值。
    科创板：≥200 股后以 1 股递增（floor 到整数股，不足 200 → 0）；其余：100 股整数倍。"""
    if is_star:
        return float(math.floor(qty)) if qty >= STAR_MIN_QTY else 0.0
    return math.floor(qty / MAIN_LOT) * MAIN_LOT


def build_market_data(codes, start, end, exec_price="vwap") -> MarketData:
    """从缓存长表拼装成对齐宽表底座。

    codes: 代码集合 或 None(全市场，取窗口内出现过的代码全集)；start/end: 回测区间端点。
    exec_price: 'vwap'|'close'|'open'，决定 trade_price 取哪个真实价列（估值价恒为 close）。
    任一必需源缺 → loaders 内 raise。
    """
    cal = load_calendar()
    cal = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    if len(cal) == 0:
        raise ValueError(f"build_market_data: [{pd.Timestamp(start).date()},{pd.Timestamp(end).date()}] 内无交易日")

    fields = ["close", "adj_factor", "trade_status"]
    if exec_price not in fields:
        fields.append(exec_price)                     # vwap/open；close 已在内
    daily = load_daily_df(codes, start, end, fields=tuple(fields))
    deriv = load_derivative_df(codes, start, end)     # [date,code,limit_status]
    code_union = pd.Index(sorted(daily["code"].unique()))

    def pivot(df, col, fill=np.nan):
        w = df.pivot(index="date", columns="code", values=col)
        return w.reindex(index=cal, columns=code_union).fillna(fill) if fill is not None \
            else w.reindex(index=cal, columns=code_union)

    trade_price = pivot(daily, exec_price)
    close_price = pivot(daily, "close")
    adj = pivot(daily, "adj_factor")
    trade_status = pivot(daily, "trade_status", fill=None)         # 缺格留 NaN，配合成交价缺当无数据
    limit_status = pivot(deriv, "limit_status", fill=None).astype("float64")  # NA→NaN，==1/==-1 自然为 False
    delist_date = load_delist_dates().reindex(code_union)         # 真实退市日（NaT=未退市），判退市用

    return MarketData(
        trade_price=trade_price, close_price=close_price, adj=adj,
        trade_status=trade_status, limit_status=limit_status,
        is_star=_infer_is_star(code_union), delist_date=delist_date, calendar=cal,
    )


def _weights_long_to_panel(weights_df, calendar, codes):
    """稀疏 weights_df [date,code,weight] → (target_panel: 调仓日×codes 稠密 fillna(0), rebalance_dates)。

    缺位填 0：某调仓日 weights 没出现的持仓票 = 该日目标 0 = 清仓（对齐权重引擎语义）。
    """
    w = weights_df.copy()
    w["date"] = pd.to_datetime(w["date"])
    reb = pd.DatetimeIndex(sorted(w["date"].unique()))
    bad_dates = reb.difference(pd.DatetimeIndex(calendar))
    if len(bad_dates):
        raise ValueError(f"_weights_long_to_panel: 调仓日不在交易日历内（前10）：{list(bad_dates[:10])}")
    bad_codes = sorted(set(w["code"].unique()) - set(codes))
    if bad_codes:
        raise ValueError(f"_weights_long_to_panel: weights_df 含 market 没有的 code（前10）：{bad_codes[:10]}")
    panel = w.pivot(index="date", columns="code", values="weight")   # 撞重复 (date,code) 这里自然 raise
    panel = panel.reindex(index=reb, columns=codes).fillna(0.0)
    return panel, reb


# ===================== 回测主体 =====================
class CashBacktest:
    """单组现金账户回测（纯执行器）。吃调仓日目标权重 target_panel，逐日撮合返回 BacktestResult。"""

    def __init__(self, market: MarketData, target_panel: pd.DataFrame,
                 rebalance_dates, benchmark_nav: pd.Series, config: BacktestConfig):
        self.m = market
        self.target_panel = target_panel            # 调仓日×cols 稠密目标权重（缺位=0=清仓）
        self.rebalance_set = set(pd.DatetimeIndex(rebalance_dates))
        self.benchmark = benchmark_nav
        self.cfg = config

        cal, cols = market.calendar, market.trade_price.columns
        if not target_panel.columns.equals(cols):
            raise ValueError("target_panel 列与 market 不一致（请经 run_cash_backtest 对齐）")
        if not target_panel.index.isin(cal).all():
            raise ValueError("target_panel 调仓日不全在 market 日历内")
        if benchmark_nav.index.intersection(cal).empty:
            raise ValueError("benchmark 与 market 日历无交集，无法算超额")

        # 复权因子按列前向填充：跨缺行段（停牌无行）也能用「最后有效 adj」算除权比例，
        # 不再因 t-1 缺行 adj=NaN 而漏掉横跨缺行的除权。
        self.adj_ffill = market.adj.ffill()
        self.dates = list(cal)
        self.start_date = self._resolve_start_date()
        # 该账户第一次真实下单日 = start_date 当日及之后的首个调仓日（建仓日卡帽豁免锚到它，
        # 而非 cfg.start_date——显式 start_date 非调仓日时两者不同）。
        future_reb = [d for d in self.dates if d >= self.start_date and d in self.rebalance_set]
        self._first_reb = future_reb[0] if future_reb else None

    def _resolve_start_date(self) -> pd.Timestamp:
        if self.cfg.start_date is not None:
            sd = pd.Timestamp(str(int(self.cfg.start_date))) if isinstance(self.cfg.start_date, (int, np.integer)) \
                else pd.Timestamp(self.cfg.start_date)
            if sd not in self.dates:
                raise ValueError(f"start_date {sd.date()} 不在交易日历内")
            if self.dates.index(sd) < 1:
                raise ValueError(f"start_date {sd.date()} 须有前一交易日（供 day-0 锚），不能是日历首日")
            return sd
        # 首个「是调仓日、当日成交价有值、且有前一交易日」的日子（不再依赖 factor）
        tp = self.m.trade_price
        for i in range(1, len(self.dates)):
            t = self.dates[i]
            if t in self.rebalance_set and tp.loc[t].notna().any():
                return t
        raise ValueError("找不到可建仓的交易日（无任一调仓日满足当日成交价有值且有前一交易日）")

    # ---------- run：多日循环（调仓日驱动） ----------
    def run(self) -> BacktestResult:
        cfg, cols = self.cfg, self.m.trade_price.columns
        self.cash = float(cfg.initial_capital)
        self.shares = pd.Series(dtype=float)             # 持仓股数，复权后可能非整数故 float
        self.last_valid_price = pd.Series(np.nan, index=cols, dtype=float)  # 追真实收盘价
        self.seg = {}                                    # 持仓段记账：code -> dict
        self.prev_total_asset = float(cfg.initial_capital)

        self.trades_rows, self.trade_log_rows = [], []
        self.missing_rows, self.blocked_rows = [], []
        holdings_rows, account_rows = {}, {}

        start_i = self.dates.index(self.start_date)
        for i in range(start_i, len(self.dates)):
            t, t_prev = self.dates[i], self.dates[i - 1]
            self._adjust_for_splits(t, t_prev)
            self._settle_delisted(t)
            if t in self.rebalance_set:                  # 调仓日：取目标 → 撮合
                target_w = self.target_panel.loc[t]
                can_buy, can_sell = self._mark_tradable(t)
                target_shares = self._alloc_turnover(t, target_w, can_buy, can_sell)
                self._execute(t, target_shares, can_buy, can_sell)
            # 非调仓日：不选股不下单，只随收盘价漂移（股数不动，市值变）
            total = self._mark_to_market(t)
            holdings_rows[t] = self.shares.copy()
            account_rows[t] = {"cash": self.cash,
                               "stock_value": total - self.cash, "total": total}
            self.prev_total_asset = total

        # 期末把还在手的持仓补一条 trade_log，不动账户（净值由收盘估值给出）
        last_t = self.dates[-1]
        for ticker in list(self.shares.index):
            self._close_trade_log(ticker, last_t, self.last_valid_price.get(ticker, np.nan),
                                  extra_shares=self.shares[ticker], close_type="eom")
        return self._assemble(holdings_rows, account_rows)

    # ---------- ① 复权调整 ----------
    def _adjust_for_splits(self, t, t_prev):
        """复权变动时按 adj_t/adj_{t-1} 缩放股数（不取整），last_valid_price 反向缩放，市值不变。

        adj 用前向填充版（self.adj_ffill）：持仓票长停牌缺行时，t_prev 的原始 adj 为 NaN，
        ffill 后取「缺行前最后有效 adj」做分母，复牌日一次性补回横跨缺行段的除权。
        """
        if self.shares.empty:
            return
        held = self.shares.index
        ratio = (self.adj_ffill.loc[t, held] / self.adj_ffill.loc[t_prev, held]) \
            .replace([np.inf, -np.inf], np.nan).fillna(1.0)
        moved = ratio[ratio != 1.0]
        if moved.empty:
            return
        self.shares.loc[moved.index] = self.shares.loc[moved.index] * moved
        self.last_valid_price.loc[moved.index] = self.last_valid_price.loc[moved.index] / moved

    # ---------- ② 退市清算 ----------
    def _settle_delisted(self, t):
        """到了真实退市日（delist_date <= t）按最后有效收盘价折现金、移出持仓。
        退市不占换手、不收费。停牌（trade_status=="停牌"）不会进这里——它没有 delist_date。

        用真实退市日（来自 asharedescription）判定，不用「窗口内最后有效成交日」——后者是
        前视（今天是否退市取决于未来有无价格），且会让同策略同数据因 end 不同给出不同历史段 NAV。
        清算价仍用最后有效收盘价、全额无费变现：这是已知的乐观假设（真实退市整理期常暴跌），
        但最后有效价已是退市整理期之后的最后真实成交价，比窗口口径保守。"""
        if self.shares.empty:
            return
        held = self.shares.index
        dld = self.m.delist_date.reindex(held)
        delisted = held[(dld <= t).fillna(False)]    # 真实退市日已到（NaT=未退市→False）
        for ticker in delisted:
            px = self.last_valid_price.get(ticker, np.nan)
            sh = float(self.shares[ticker])
            if not np.isfinite(px):
                px = 0.0
            proceeds = sh * px
            self.cash += proceeds
            self.missing_rows.append(
                {"date": t, "ticker": ticker, "type": "delist",
                 "price": px, "shares": sh, "proceeds": proceeds})
            self._close_trade_log(ticker, t, px, extra_shares=sh, close_type="delist")
            self.shares = self.shares.drop(ticker)

    # ---------- ③ 标可交易性（全用现成字段，ST 不在此拦） ----------
    def _mark_tradable(self, t):
        """涨停拦买、跌停拦卖、停牌/无成交价双拦。给出当日能买/能卖掩码。

        limit_status 在交易日为 NaN → ==1/==-1 均 False，按可正常买卖（与权重引擎 check_tradable 一致）。
        ST 不在此拦——剔不剔 ST 是策略选择（factor_to_weights 的 exclude_st），ST 本身可交易。
        """
        m = self.m
        suspended = m.trade_status.loc[t] == SUSPEND_FLAG
        no_data = m.trade_price.loc[t].isna()
        lu = m.limit_status.loc[t] == 1
        ld = m.limit_status.loc[t] == -1
        can_buy = (~suspended) & (~no_data) & (~lu)
        can_sell = (~suspended) & (~no_data) & (~ld)   # 涨停不挡卖；跌停挡卖
        return can_buy, can_sell

    # ---------- ④ 定目标股数与换手（target_w 由 run 传入） ----------
    def _alloc_turnover(self, t, target_w: pd.Series,
                        can_buy: pd.Series, can_sell: pd.Series) -> pd.Series:
        """目标权重 target_w(稠密) → 整手目标股数。Δ=目标−当前，不可交易方向置 0（记 blocked），
        turnover_cap 非 None 才卡单边换手；建仓日豁免卡帽。不在目标里的持仓票 → 目标 0（清仓）。"""
        cfg, m = self.cfg, self.m
        cols = m.trade_price.columns
        W = self.prev_total_asset

        tw_full = target_w.reindex(cols).fillna(0.0)
        cur_w = pd.Series(0.0, index=cols)
        if not self.shares.empty:
            val = self.shares * self.last_valid_price.reindex(self.shares.index)
            cur_w.loc[self.shares.index] = (val / W).fillna(0.0).values

        delta = tw_full - cur_w
        cb = can_buy.reindex(cols).fillna(False)
        cs = can_sell.reindex(cols).fillna(False)
        block_buy = (delta > 0) & (~cb)
        block_sell = (delta < 0) & (~cs)
        for ticker in cols[block_buy]:
            self._record_blocked(t, ticker, "buy")
        for ticker in cols[block_sell]:
            self._record_blocked(t, ticker, "sell")
        delta[block_buy | block_sell] = 0.0
        delta = delta[delta != 0.0]
        if delta.empty:
            return pd.Series(dtype=float)

        if cfg.turnover_cap is None or t == self._first_reb:
            admitted = delta.index                       # 不卡 / 建仓日豁免（锚到首个真实下单日）
        else:
            absd = delta.abs().sort_values(ascending=False, kind="mergesort")
            cum_oneway = absd.cumsum() / 2.0             # 双边变动累加除 2 得单边换手
            admitted = absd.index[cum_oneway <= cfg.turnover_cap + 1e-12]
            # 被换手限额截断的票（含本可装下的小单、清仓单）逐只留痕，不再静默落空
            for ticker in absd.index.difference(admitted):
                self._record_blocked(t, ticker, "buy" if delta[ticker] > 0 else "sell",
                                     reason="换手限额")

        price = m.trade_price.loc[t]
        star = m.is_star
        out = {}
        for ticker in admitted:
            p = price[ticker]
            tw = tw_full[ticker]
            if tw <= 0:
                out[ticker] = 0.0                         # 已掉出目标，清掉
            elif np.isfinite(p) and p > 0:
                lot = _round_down_lot(tw * W / p, star[ticker])
                if lot == 0 and self.shares.get(ticker, 0.0) <= 0:
                    self._record_blocked(t, ticker, "buy", reason="整手不足")  # 想买但不足一手
                out[ticker] = lot
            else:
                out[ticker] = self.shares.get(ticker, 0.0)
        return pd.Series(out, dtype=float)

    # ---------- ⑤ 先卖后买 ----------
    def _execute(self, t, target_shares: pd.Series, can_buy: pd.Series, can_sell: pd.Series):
        """先卖后买，用 trade_price 成交、买卖分费 (buy/sell_fee+slippage)、现金不为负，按持仓段记账。

        实际买卖方向以「股数空间」(desired vs 当前持仓)为准，并在此用 can_buy/can_sell 复检拦截：
        目标方向(权重空间，按 t-1 价口径)与实际方向(股数空间，按当日成交价口径)可反号，
        若不在此处按实际方向复检，跌停日会照卖、涨停日会照买。
        """
        if target_shares.empty:
            return
        price = self.m.trade_price.loc[t]
        star = self.m.is_star
        buy_cost = self.cfg.buy_fee + self.cfg.slippage
        sell_cost = self.cfg.sell_fee + self.cfg.slippage

        # 先卖：减仓和掉出目标的清仓（实际方向=卖，复检 can_sell）
        for ticker, desired in target_shares.items():
            cur = float(self.shares.get(ticker, 0.0))
            if desired < cur - 1e-9:
                if not bool(can_sell.get(ticker, False)):
                    self._record_blocked(t, ticker, "sell")  # 实际要卖但当日不可卖（跌停/停牌）→ 锁仓
                    continue
                qty = cur - desired
                p = price[ticker]
                self.cash += qty * p * (1 - sell_cost)
                self._record_trade(t, ticker, "sell", p, qty, qty * p * sell_cost)
                self._seg_sell(ticker, qty, p, qty * p * sell_cost)
                new_sh = cur - qty
                if new_sh <= 1e-9:
                    self._close_trade_log(ticker, t, p, close_type="sell")
                    self.shares = self.shares.drop(ticker, errors="ignore")
                else:
                    self.shares[ticker] = new_sh

        # 再买：预算为卖出后现金，按目标买入金额降序优先，保证现金不为负（实际方向=买，复检 can_buy）
        buys = []
        for ticker, desired in target_shares.items():
            cur = float(self.shares.get(ticker, 0.0))
            if desired > cur + 1e-9:
                buys.append((ticker, desired - cur))
        buys.sort(key=lambda x: x[1] * price[x[0]], reverse=True)
        for ticker, want in buys:
            if not bool(can_buy.get(ticker, False)):
                self._record_blocked(t, ticker, "buy")   # 实际要买但当日不可买（涨停/停牌）
                continue
            p = price[ticker]
            want_lot = _round_down_lot(want, star[ticker])   # 买单本身按最小申报规则取整
            affordable = _round_down_lot(self.cash / (p * (1 + buy_cost)), star[ticker])
            qty = min(want_lot, affordable)              # 两者都已取整，min 后仍合法
            if qty <= 0:
                self._record_blocked(t, ticker, "buy", reason="现金耗尽")  # 想买但买不到（钱不够一手）
                continue
            self.cash -= qty * p * (1 + buy_cost)
            self._record_trade(t, ticker, "buy", p, qty, qty * p * buy_cost)
            self._seg_buy(ticker, qty, p, qty * p * buy_cost, t)
            self.shares[ticker] = float(self.shares.get(ticker, 0.0)) + qty

    # ---------- ⑥ 收盘估值 ----------
    def _mark_to_market(self, t) -> float:
        """持仓按真实收盘价估值，停牌/缺价用最后有效价，并刷新 last_valid_price。"""
        close_t = self.m.close_price.loc[t]
        self.last_valid_price = self.last_valid_price.where(close_t.isna(), close_t)
        mv = 0.0
        if not self.shares.empty:
            px = self.last_valid_price.reindex(self.shares.index)
            # 持仓票无有效估值价（close 列自买入起一直缺失）→ fail-fast，不让 sum(skipna) 把它当 0、
            # 整笔本金从账上静默消失（违反 fail-fast；根因是 build_market_data 的 close 缺损）。
            bad = px[px.isna()]
            if len(bad):
                raise ValueError(
                    f"_mark_to_market: {pd.Timestamp(t).date()} 持仓票无有效收盘估值价（close 一直缺失）："
                    f"{list(bad.index[:5])}（共 {len(bad)} 只）；市值无法估算，请查 close 数据缺损")
            mv = float((self.shares * px).sum())
        return self.cash + mv

    # ---------- 段级记账辅助 ----------
    def _seg_buy(self, ticker, qty, price, fee, t):
        s = self.seg.get(ticker)
        if s is None:
            s = {"first_buy": t, "buy_gross": 0.0, "buy_fee": 0.0, "buy_sh": 0.0,
                 "sell_gross": 0.0, "sell_fee": 0.0, "sell_sh": 0.0}
            self.seg[ticker] = s
        s["buy_gross"] += qty * price
        s["buy_fee"] += fee
        s["buy_sh"] += qty

    def _seg_sell(self, ticker, qty, price, fee):
        s = self.seg.get(ticker)
        if s is None:                       # 卖之前一定先有买，进这里说明段记账状态丢了（bug）
            raise RuntimeError(f"_seg_sell: {ticker} 卖出时无持仓段记录（qty={qty}, price={price}）——段记账状态异常")
        s["sell_gross"] += qty * price
        s["sell_fee"] += fee
        s["sell_sh"] += qty

    def _record_trade(self, t, ticker, side, price, shares, fee):
        self.trades_rows.append(
            {"date": t, "ticker": ticker, "side": side,
             "price": price, "shares": shares, "fee": fee})

    def _blocked_reason(self, t, ticker, side) -> str:
        """按当日字段判某票为何买/卖不成（涨跌停/停牌/无数据）。"""
        m = self.m
        if m.trade_status.loc[t, ticker] == SUSPEND_FLAG:
            return "停牌"
        if not np.isfinite(m.trade_price.loc[t, ticker]):
            return "无数据"
        ls = m.limit_status.loc[t, ticker]
        if side == "buy" and ls == 1:
            return "涨停"
        if side == "sell" and ls == -1:
            return "跌停"
        return "受限"

    def _record_blocked(self, t, ticker, side, reason=None):
        """目标未达成留痕（观测性，不改 NAV）。reason 缺省时按当日字段判（涨跌停/停牌/无数据）。"""
        if reason is None:
            reason = self._blocked_reason(t, ticker, side)
        self.blocked_rows.append({"date": t, "code": ticker, "side": side, "reason": reason})

    def _close_trade_log(self, ticker, t, sell_price, extra_shares=0.0, close_type="sell"):
        """清仓时落一条完整买卖往返（仅记录，不动现金账户）。extra_shares 是没走普通卖出的
        剩余股数（退市/期末），按 sell_price 折进 trade_log 的 PnL、不收手续费。
        现金入账由调用方负责：退市在 _settle_delisted 自己 cash += proceeds，期末不动账户。"""
        s = self.seg.get(ticker)
        if s is None:
            return
        sell_gross, sell_fee, sell_sh = s["sell_gross"], s["sell_fee"], s["sell_sh"]
        if extra_shares > 0 and np.isfinite(sell_price):
            sell_gross += extra_shares * sell_price
            sell_sh += extra_shares
        buy_gross, buy_fee, buy_sh = s["buy_gross"], s["buy_fee"], s["buy_sh"]
        pnl = (sell_gross - sell_fee) - (buy_gross + buy_fee)
        dur = self.dates.index(t) - self.dates.index(s["first_buy"])
        self.trade_log_rows.append({
            "ticker": ticker, "buy_date": s["first_buy"], "sell_date": t,
            "buy_price": buy_gross / buy_sh if buy_sh else np.nan,
            "sell_price": sell_gross / sell_sh if sell_sh else np.nan,
            "shares": buy_sh, "pnl": pnl, "holding_days": dur,
            "close_type": close_type,
        })
        del self.seg[ticker]

    # ---------- 装配结果与指标 ----------
    def _assemble(self, holdings_rows, account_rows) -> BacktestResult:
        cap = self.cfg.initial_capital
        account = pd.DataFrame.from_dict(account_rows, orient="index")
        account.index.name = "date"
        holdings = pd.DataFrame.from_dict(holdings_rows, orient="index")
        holdings = holdings.reindex(index=account.index).fillna(0.0)   # 空仓日补 0，对齐日历
        holdings.index.name = "date"

        # 建仓日前一天放净值=1 锚点，让建仓手续费落进首日收益
        anchor = self.dates[self.dates.index(self.start_date) - 1]
        nav = account["total"] / cap
        nav = pd.concat([pd.Series([1.0], index=[anchor]), nav])
        nav.index.name = "date"

        # 基准截到回测区间、按回测首日归一到 1，与策略同起点
        b = self.benchmark.reindex(nav.index)
        if not b.notna().any():   # 基准在回测区间内全 NaN → 明确报错，不让 dropna().iloc[0] 抛裸 IndexError
            raise ValueError(f"基准在回测区间 [{nav.index[0].date()}, {nav.index[-1].date()}] 内全无数据，无法归一/算超额")
        base = b.iloc[0] if np.isfinite(b.iloc[0]) else b.dropna().iloc[0]
        bench_nav = b / base
        bench_nav.index.name = "date"

        excess_nav = self._excess_nav(nav, bench_nav)
        full = calc_metrics(nav, benchmark_nav=bench_nav)
        metrics_abs = {k: full[k] for k in _ABS_KEYS if k in full}
        metrics_excess = {k: full[k] for k in _EXCESS_KEYS if k in full}

        trades = pd.DataFrame(self.trades_rows,
                              columns=["date", "ticker", "side", "price", "shares", "fee"])
        trade_log = pd.DataFrame(self.trade_log_rows,
                                 columns=["ticker", "buy_date", "sell_date", "buy_price",
                                          "sell_price", "shares", "pnl", "holding_days", "close_type"])
        missing_log = pd.DataFrame(self.missing_rows,
                                   columns=["date", "ticker", "type", "price", "shares", "proceeds"])
        blocked_log = pd.DataFrame(self.blocked_rows,
                                   columns=["date", "code", "side", "reason"])
        return BacktestResult(nav, excess_nav, bench_nav, holdings, account, trade_log,
                              trades, metrics_abs, metrics_excess, missing_log, blocked_log)

    @staticmethod
    def _excess_nav(strategy_nav: pd.Series, benchmark_nav: pd.Series) -> pd.Series:
        """逐日算术超额累乘净值（plan v3 口径，与 calc_metrics 超额标量同源）。"""
        r_s = strategy_nav.pct_change(fill_method=None)
        r_b = benchmark_nav.pct_change(fill_method=None)
        alpha = (r_s - r_b).dropna()                   # 停牌/退市日 NaN，dropna 自然剔除
        excess_nav = (1 + alpha).cumprod()
        excess_nav = pd.concat([pd.Series([1.0], index=[strategy_nav.index[0]]), excess_nav])
        excess_nav.index.name = "date"
        return excess_nav


# ===================== 驱动 =====================
def run_cash_backtest(weights_df: pd.DataFrame, benchmark_code: str,
                      config: Optional[BacktestConfig] = None,
                      start=None, end=None) -> BacktestResult:
    """现金引擎总入口：外部 weights_df + 基准指数代码 → BacktestResult（调仓日驱动逐日撮合）。

    weights_df: [date, code, weight] 稀疏长表（与权重引擎同契约，你框架外或经 factor_to_weights 算好）；
                long_only（weight>0）、每调仓日 Σweight≤1。
    benchmark_code: 外部指数代码（如 '000852.SH'）。start/end: None 则取 weights_df 日期范围。
    """
    w = weights_df.copy()
    w["date"] = pd.to_datetime(w["date"])
    if w.empty:
        raise ValueError("run_cash_backtest: weights_df 为空")
    dup = w.duplicated(["date", "code"])
    if dup.any():
        r = w[dup].iloc[0]
        raise ValueError(f"run_cash_backtest: weights_df 存在重复 (date,code)，首个 {r['code']}@{r['date'].date()}"
                         f"（共 {int(dup.sum())} 处）；重复会让目标权重歧义")
    nan_w = w[w["weight"].isna()]
    if len(nan_w):
        r = nan_w.iloc[0]
        raise ValueError(f"run_cash_backtest: weight 含 NaN（{r['code']}@{r['date'].date()}）；NaN 会污染 target_panel")
    neg = w[w["weight"] < 0]
    if len(neg):
        r = neg.iloc[0]
        raise ValueError(f"run_cash_backtest: 现金侧只做 long_only，见负权重 {r['code']}@{r['date'].date()}={r['weight']}")
    daysum = w.groupby("date")["weight"].sum()
    over = daysum[daysum > 1.0 + 1e-9]
    if len(over):
        raise ValueError(f"run_cash_backtest: 调仓日权重和 >1，{over.index[0].date()} Σ={over.iloc[0]:.6f}")

    cfg = config if config is not None else BacktestConfig()
    start = pd.Timestamp(start) if start is not None else w["date"].min()
    end = pd.Timestamp(end) if end is not None else w["date"].max()
    src_lo, src_hi = w["date"].min(), w["date"].max()
    w = w[(w["date"] >= start) & (w["date"] <= end)]      # end 截断回测窗口：窗口外的调仓日不执行
    if w.empty:
        raise ValueError(f"run_cash_backtest: [{start.date()},{end.date()}] 内无调仓日（weights 范围 {src_lo.date()}~{src_hi.date()}）")
    codes = sorted(w["code"].unique())

    # 市场窗口前移一交易日：首个调仓日需要前一交易日做 day-0 净值锚，否则它会被 _resolve_start_date
    # 静默跳过。从全日历取首个调仓日的前一交易日作 market_start，benchmark 同步覆盖到锚日。
    first_reb = w["date"].min()
    cal_full = load_calendar()
    pos = cal_full.searchsorted(first_reb)
    if pos == 0 or cal_full[pos] != first_reb:
        raise ValueError(f"run_cash_backtest: 首个调仓日 {first_reb.date()} 不是交易日或无前一交易日（无法建 day-0 锚）")
    market_start = cal_full[pos - 1]

    market = build_market_data(codes, market_start, end, exec_price=cfg.exec_price)
    target_panel, rebalance_dates = _weights_long_to_panel(
        w, market.calendar, market.trade_price.columns)
    bench_nav = calc_benchmark(load_index_eod(benchmark_code), market_start, end)
    return CashBacktest(market, target_panel, rebalance_dates, bench_nav, cfg).run()
