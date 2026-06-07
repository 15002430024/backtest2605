# -*- coding: utf-8 -*-
"""现金账户逐日撮合回测引擎（阶段 7a）。

维护每只股票的持仓股数加一笔现金，每天用真实 vwap 成交、扣双边 14bp、按真实收盘价
估值，逐日滚出净值与超额。与权重引擎（engine/backtest.py）并列、互补：权重引擎答
"因子有没有 alpha"，现金引擎答"一亿实盘实际能赚多少"。

数据全部来自框架缓存（data/cache，经 data/loaders 读出）：
- 真实成交价 = daily.vwap，真实收盘价 = daily.close，复权因子 = daily.adj_factor
- 停牌 = daily.trade_status=="停牌"；涨跌停 = derivative.limit_status(1涨停/-1跌停/0正常/NA)
- ST = st_intervals 区间转当日布尔；票池 = index_members 区间转当日布尔
可交易性全用这些现成字段判，不再自算整板阈值或价格跨日比。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from data.loaders import (
    load_calendar, load_daily_df, load_derivative_df,
    load_st_intervals, load_index_members, load_index_eod,
)
from data.panels import intervals_to_panel
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
    trade_price: pd.DataFrame   # 真实成交价 = daily.vwap
    close_price: pd.DataFrame   # 真实收盘价 = daily.close
    adj: pd.DataFrame           # 后复权因子 = daily.adj_factor，复权调整用
    trade_status: pd.DataFrame  # daily.trade_status，=="停牌" 判停牌
    limit_status: pd.DataFrame  # derivative.limit_status，1涨停/-1跌停/0正常/NaN
    is_st: pd.DataFrame         # bool，当日是否 ST（st_intervals 转）
    is_star: pd.Series          # bool，是否科创板（688/689）；决定最小申报规则
    calendar: pd.DatetimeIndex  # 交易日主轴


@dataclass
class BacktestConfig:
    initial_capital: float = 1e8
    n_holdings: int = 200
    fee_rate: float = 0.0014           # 双边 14bp
    turnover_cap: float = 0.30         # 单边换手硬上限，建仓日豁免
    start_date: Union[int, str, pd.Timestamp, None] = None  # None=自动取首个可交易日

    def __post_init__(self):
        if self.n_holdings <= 0:
            raise ValueError(f"n_holdings 必须 >0，当前 {self.n_holdings}")
        if self.turnover_cap <= 0:
            raise ValueError(f"turnover_cap 必须 >0，当前 {self.turnover_cap}")
        if not (0 <= self.fee_rate < 1):
            raise ValueError(f"fee_rate 必须在 [0,1)，当前 {self.fee_rate}")
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


# ===================== prep：从缓存装配市场底座 =====================
def _infer_is_star(tickers: pd.Index) -> pd.Series:
    """是否科创板（688/689）。科创板最小申报 200 股、之后 1 股递增；其余 100 整数倍。
    注意这个边界和涨跌停板块（已改读 limit_status）不一样。"""
    return pd.Series(tickers.str[:3].isin(STAR_PREFIX), index=tickers)


def _round_down_lot(qty: float, is_star: bool) -> float:
    """按交易所最小申报规则把股数向下取到合法值。
    科创板：≥200 股后以 1 股递增（floor 到整数股，不足 200 → 0）；其余：100 股整数倍。"""
    if is_star:
        return float(math.floor(qty)) if qty >= STAR_MIN_QTY else 0.0
    return math.floor(qty / MAIN_LOT) * MAIN_LOT


def build_market_data(codes, start, end) -> MarketData:
    """从缓存长表拼装成对齐宽表底座（替代旧 prepare_market_data，数据源换成 data/cache）。

    codes: 代码集合 或 None(全市场，取窗口内出现过的代码全集，防生存者偏差)；
    start/end: 回测区间端点。任一必需源缺 → loaders 内 raise。
    """
    cal = load_calendar()
    cal = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    if len(cal) == 0:
        raise ValueError(f"build_market_data: [{pd.Timestamp(start).date()},{pd.Timestamp(end).date()}] 内无交易日")

    daily = load_daily_df(codes, start, end)          # [date,code,vwap,close,adj_factor,trade_status]
    deriv = load_derivative_df(codes, start, end)     # [date,code,limit_status]
    code_union = pd.Index(sorted(daily["code"].unique()))

    def pivot(df, col, fill=np.nan):
        w = df.pivot(index="date", columns="code", values=col)
        return w.reindex(index=cal, columns=code_union).fillna(fill) if fill is not None \
            else w.reindex(index=cal, columns=code_union)

    trade_price = pivot(daily, "vwap")
    close_price = pivot(daily, "close")
    adj = pivot(daily, "adj_factor")
    trade_status = pivot(daily, "trade_status", fill=None)         # 缺格留 NaN，配合 vwap 缺当无数据
    limit_status = pivot(deriv, "limit_status", fill=None).astype("float64")  # NA→NaN，==1/==-1 自然为 False
    is_st = intervals_to_panel(load_st_intervals(), "st_start", "st_end",
                               cal, code_union, end_inclusive=True)

    return MarketData(
        trade_price=trade_price, close_price=close_price, adj=adj,
        trade_status=trade_status, limit_status=limit_status, is_st=is_st,
        is_star=_infer_is_star(code_union), calendar=cal,
    )


def resolve_pool(pool_arg, market: MarketData) -> Optional[pd.DataFrame]:
    """票池入参归一成 date×code 当日布尔表（对齐 market 主轴）。

    pool_arg 三态：None(全市场，返回 None) / 指数代码 str / 现成 date×code bool 表。
    """
    cal, cols = market.calendar, market.trade_price.columns
    if pool_arg is None:
        return None
    if isinstance(pool_arg, str):
        members = load_index_members(pool_arg)
        return intervals_to_panel(members, "entry_date", "exit_date", cal, cols, end_inclusive=False)
    # 现成 bool 表
    pool = pool_arg.copy()
    pool.index = pd.to_datetime(pool.index)
    return pool.reindex(index=cal, columns=cols).fillna(False).astype(bool)


# ===================== 回测主体 =====================
class CashBacktest:
    """单组（票池×基准）现金账户回测。run() 跑完多日循环返回 BacktestResult。"""

    def __init__(self, market: MarketData, factor: pd.DataFrame,
                 pool_mask: Optional[pd.DataFrame], benchmark_nav: pd.Series,
                 config: BacktestConfig):
        self.m = market
        self.factor = factor
        self.pool = pool_mask
        self.benchmark = benchmark_nav
        self.cfg = config

        cal, cols = market.calendar, market.trade_price.columns
        if not (factor.index.equals(cal) and factor.columns.equals(cols)):
            raise ValueError("factor 与 market 主轴不一致（请经 run_cash_backtest 对齐，或自行 reindex 到 market）")
        if pool_mask is not None and not (
                pool_mask.index.equals(cal) and pool_mask.columns.equals(cols)):
            raise ValueError("pool_mask 与 market 主轴不一致")
        if benchmark_nav.index.intersection(cal).empty:
            raise ValueError("benchmark 与 market 日历无交集，无法算超额")

        # 每只票最后一个有效成交价(vwap)日期，用于退市判定
        self.last_trade_date = market.trade_price.apply(lambda c: c.last_valid_index())
        self.dates = list(cal)
        self.start_date = self._resolve_start_date()

    def _resolve_start_date(self) -> pd.Timestamp:
        if self.cfg.start_date is not None:
            sd = pd.Timestamp(str(int(self.cfg.start_date))) if isinstance(self.cfg.start_date, (int, np.integer)) \
                else pd.Timestamp(self.cfg.start_date)
            if sd not in self.dates:
                raise ValueError(f"start_date {sd.date()} 不在交易日历内")
            if self.dates.index(sd) < 1:
                raise ValueError(f"start_date {sd.date()} 须有前一交易日（供 t-1 因子/day-0 锚），不能是日历首日")
            return sd
        # 取第一个满足"前一日因子有值、当日有成交价"的交易日
        tp, fac = self.m.trade_price, self.factor
        for i in range(1, len(self.dates)):
            t, t_prev = self.dates[i], self.dates[i - 1]
            if fac.loc[t_prev].notna().any() and tp.loc[t].notna().any():
                return t
        raise ValueError("找不到可建仓的交易日（无任一日满足 t-1 因子有值且当日 vwap 有值）")

    # ---------- run：多日循环 ----------
    def run(self) -> BacktestResult:
        cfg, cols = self.cfg, self.m.trade_price.columns
        self.cash = float(cfg.initial_capital)
        self.shares = pd.Series(dtype=float)             # 持仓股数，复权后可能非整数故 float
        self.last_valid_price = pd.Series(np.nan, index=cols, dtype=float)  # 追真实收盘价
        self.seg = {}                                    # 持仓段记账：code -> dict
        self.prev_total_asset = float(cfg.initial_capital)

        self.trades_rows, self.trade_log_rows, self.missing_rows = [], [], []
        holdings_rows, account_rows = {}, {}

        start_i = self.dates.index(self.start_date)
        for i in range(start_i, len(self.dates)):
            t, t_prev = self.dates[i], self.dates[i - 1]
            self._adjust_for_splits(t, t_prev)
            self._settle_delisted(t)
            target = self._rank_signal(t, t_prev)
            can_buy, can_sell = self._mark_tradable(t)
            target_shares = self._alloc_turnover(t, target, can_buy, can_sell)
            self._execute(t, target_shares)
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
        """复权变动时按 adj_t/adj_{t-1} 缩放股数（不取整），last_valid_price 反向缩放，市值不变。"""
        if self.shares.empty:
            return
        held = self.shares.index
        ratio = (self.m.adj.loc[t, held] / self.m.adj.loc[t_prev, held]) \
            .replace([np.inf, -np.inf], np.nan).fillna(1.0)
        moved = ratio[ratio != 1.0]
        if moved.empty:
            return
        self.shares.loc[moved.index] = self.shares.loc[moved.index] * moved
        self.last_valid_price.loc[moved.index] = self.last_valid_price.loc[moved.index] / moved

    # ---------- ② 退市清算 ----------
    def _settle_delisted(self, t):
        """成交价(vwap)一直 NaN 到末日不再恢复即视退市，按最后有效收盘价折现金、移出持仓。
        退市不占换手、不收费。停牌（trade_status=="停牌"）不会进这里——它复牌后 vwap 会恢复。"""
        if self.shares.empty:
            return
        held = self.shares.index
        ltd = self.last_trade_date.reindex(held)
        delisted = held[~(ltd >= t).fillna(False)]   # 最后有效成交日早于 t（或从无成交）
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
            self._close_trade_log(ticker, t, px, extra_shares=sh,
                                  close_type="delist", extra_cash=True)
            self.shares = self.shares.drop(ticker)

    # ---------- ③ 选信号 ----------
    def _rank_signal(self, t, t_prev) -> pd.Index:
        """候选 = 票池 ∩ 因子(t-1)非空 ∩ 非ST(t) ∩ 非停牌(t) ∩ 当日有成交价，按因子降序取前 n_holdings。"""
        fac_prev = self.factor.loc[t_prev]
        has_px = self.m.trade_price.loc[t].notna()
        not_suspended = self.m.trade_status.loc[t] != SUSPEND_FLAG
        not_st = ~self.m.is_st.loc[t]
        cand = fac_prev.notna() & has_px & not_suspended & not_st
        if self.pool is not None:
            cand = cand & self.pool.loc[t]
        ranked = fac_prev[cand].sort_values(ascending=False, kind="mergesort")
        return ranked.index[: self.cfg.n_holdings]

    # ---------- ④ 标可交易性（全用现成字段） ----------
    def _mark_tradable(self, t):
        """涨跌停读 limit_status、停牌读 trade_status、ST 读 is_st，给出当日能买/能卖掩码。

        涨停(=1)只拦买、跌停(=-1)只拦卖、停牌/无成交价双拦、ST 只拦买不拦卖。
        limit_status 在交易日为 NaN → ==1/==-1 均 False，按可正常买卖（与权重引擎 check_tradable 一致）。
        """
        m = self.m
        suspended = m.trade_status.loc[t] == SUSPEND_FLAG
        no_data = m.trade_price.loc[t].isna()
        lu = m.limit_status.loc[t] == 1
        ld = m.limit_status.loc[t] == -1
        st = m.is_st.loc[t]
        can_buy = (~suspended) & (~no_data) & (~lu) & (~st)
        can_sell = (~suspended) & (~no_data) & (~ld)   # ST/涨停不挡卖；跌停挡卖
        return can_buy, can_sell

    # ---------- ⑤ 定目标与换手（排序卡死 30%，建仓日豁免） ----------
    def _alloc_turnover(self, t, target: pd.Index,
                        can_buy: pd.Series, can_sell: pd.Series) -> pd.Series:
        """目标权重恒 1/n_holdings；Δ=目标−当前，不可交易方向置 0，按 |Δ| 降序卡到 30% 单边换手。
        建仓日(t==start_date)不卡换手。返回纳入票的目标股数（整手向下取整）。"""
        cfg, m = self.cfg, self.m
        # 候选池为空（当日因子全缺）→ 持仓不动：空窗不是清仓信号，清了次日再买白吃手续费
        if len(target) == 0:
            return pd.Series(dtype=float)

        W = self.prev_total_asset
        n = cfg.n_holdings
        cols = m.trade_price.columns

        target_w = pd.Series(0.0, index=cols)
        target_w.loc[target] = 1.0 / n
        cur_w = pd.Series(0.0, index=cols)
        if not self.shares.empty:
            val = self.shares * self.last_valid_price.reindex(self.shares.index)
            cur_w.loc[self.shares.index] = (val / W).fillna(0.0).values

        delta = target_w - cur_w
        block_buy = (delta > 0) & (~can_buy.reindex(cols).fillna(False))
        block_sell = (delta < 0) & (~can_sell.reindex(cols).fillna(False))
        delta[block_buy | block_sell] = 0.0
        delta = delta[delta != 0.0]
        if delta.empty:
            return pd.Series(dtype=float)

        if t == self.start_date:
            admitted = delta.index                       # 建仓日不卡换手
        else:
            absd = delta.abs().sort_values(ascending=False, kind="mergesort")
            cum_oneway = absd.cumsum() / 2.0             # 双边变动累加除 2 得单边换手
            admitted = absd.index[cum_oneway <= cfg.turnover_cap + 1e-12]

        price = m.trade_price.loc[t]
        star = m.is_star
        out = {}
        for ticker in admitted:
            p = price[ticker]
            tw = target_w[ticker]
            if tw <= 0:
                out[ticker] = 0.0                         # 已掉出 top-N，清掉
            elif np.isfinite(p) and p > 0:
                out[ticker] = _round_down_lot(tw * W / p, star[ticker])
            else:
                out[ticker] = self.shares.get(ticker, 0.0)
        return pd.Series(out, dtype=float)

    # ---------- ⑥ 先卖后买 ----------
    def _execute(self, t, target_shares: pd.Series):
        """先卖后买，用真实 vwap 成交、双边扣费、现金不为负，同时按持仓段记账。"""
        if target_shares.empty:
            return
        price = self.m.trade_price.loc[t]
        star = self.m.is_star
        fee = self.cfg.fee_rate

        # 先卖：减仓和掉出 top-N 的清仓
        for ticker, desired in target_shares.items():
            cur = float(self.shares.get(ticker, 0.0))
            if desired < cur - 1e-9:
                qty = cur - desired
                p = price[ticker]
                self.cash += qty * p * (1 - fee)
                self._record_trade(t, ticker, "sell", p, qty, qty * p * fee)
                self._seg_sell(ticker, qty, p, qty * p * fee)
                new_sh = cur - qty
                if new_sh <= 1e-9:
                    self._close_trade_log(ticker, t, p, close_type="sell")
                    self.shares = self.shares.drop(ticker, errors="ignore")
                else:
                    self.shares[ticker] = new_sh

        # 再买：预算为卖出后现金，按目标买入金额降序优先，保证现金不为负
        buys = []
        for ticker, desired in target_shares.items():
            cur = float(self.shares.get(ticker, 0.0))
            if desired > cur + 1e-9:
                buys.append((ticker, desired - cur))
        buys.sort(key=lambda x: x[1] * price[x[0]], reverse=True)
        for ticker, want in buys:
            p = price[ticker]
            affordable = _round_down_lot(self.cash / (p * (1 + fee)), star[ticker])
            qty = min(want, affordable)                  # 两者都按最小申报规则取整，min 后仍合法
            if qty <= 0:
                continue
            self.cash -= qty * p * (1 + fee)
            self._record_trade(t, ticker, "buy", p, qty, qty * p * fee)
            self._seg_buy(ticker, qty, p, qty * p * fee, t)
            self.shares[ticker] = float(self.shares.get(ticker, 0.0)) + qty

    # ---------- ⑦ 收盘估值 ----------
    def _mark_to_market(self, t) -> float:
        """持仓按真实收盘价估值，停牌/缺价用最后有效价，并刷新 last_valid_price。"""
        close_t = self.m.close_price.loc[t]
        self.last_valid_price = self.last_valid_price.where(close_t.isna(), close_t)
        mv = 0.0
        if not self.shares.empty:
            px = self.last_valid_price.reindex(self.shares.index)
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
        if s is None:                       # 卖之前一定先有买，正常不会进这里
            return
        s["sell_gross"] += qty * price
        s["sell_fee"] += fee
        s["sell_sh"] += qty

    def _record_trade(self, t, ticker, side, price, shares, fee):
        self.trades_rows.append(
            {"date": t, "ticker": ticker, "side": side,
             "price": price, "shares": shares, "fee": fee})

    def _close_trade_log(self, ticker, t, sell_price, extra_shares=0.0,
                         close_type="sell", extra_cash=False):
        """清仓时落一条完整买卖往返。extra_shares 是没走普通卖出的剩余股数（退市/期末），
        按 sell_price 计入、不收手续费。退市真实入账，期末只补记录、不动账户。"""
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
        return BacktestResult(nav, excess_nav, bench_nav, holdings, account, trade_log,
                              trades, metrics_abs, metrics_excess, missing_log)

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
def run_cash_backtest(factor_wide: pd.DataFrame, pool, benchmark_code: str,
                      config: Optional[BacktestConfig] = None,
                      start=None, end=None) -> BacktestResult:
    """现金回测总入口：因子宽表 + 票池 + 基准指数代码 → BacktestResult。

    factor_wide: date×code 宽表（你框架外算好）；pool: None(全市场)/指数代码/bool 宽表；
    benchmark_code: 外部指数代码（如 '000852.SH'）；start/end: None 则取 factor_wide 时间范围。
    """
    factor_wide = factor_wide.copy()
    factor_wide.index = pd.to_datetime(factor_wide.index)
    start = pd.Timestamp(start) if start is not None else factor_wide.index.min()
    end = pd.Timestamp(end) if end is not None else factor_wide.index.max()

    market = build_market_data(None, start, end)        # 全市场底座（codes=None）
    factor = factor_wide.reindex(index=market.calendar, columns=market.trade_price.columns)
    pool_mask = resolve_pool(pool, market)
    bench_nav = calc_benchmark(load_index_eod(benchmark_code), start, end)
    cfg = config if config is not None else BacktestConfig()
    return CashBacktest(market, factor, pool_mask, bench_nav, cfg).run()
