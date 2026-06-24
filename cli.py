"""
回测框架统一命令行入口（薄编排层，配置驱动、可复现、能批量）。

  bt factor   --factor 因子.parquet --config cfg.yaml --out 结果目录 [--weighting equal|factor]
  bt backtest --weights 权重.parquet --config cfg.yaml --out 结果目录

只做「装配」：读数据文件 + 读 YAML 配置 + 调各层现成函数。不内置因子/策略计算
（因子宽表、权重表你自己在框架外算好存成文件），不做配置校验框架、不搞子命令插件化。

factor 子命令：因子宽表 + 配置 → run_factor_test → plot_factor_report（IC/分组/多空/超额 + 滚动图）
backtest 子命令：权重长表 + 配置 → load_price_df → run_backtest → plot_dashboard（机构研报仪表盘）

YAML 配置见 examples/ 下两个样例。批量 = 多个配置文件循环跑（shell for / 脚本均可）。
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

from data.loaders import load_calendar, load_price_df, load_index_eod, CACHE_ROOT
from engine.backtest import run_backtest, calc_benchmark, DEFAULT_CONFIG
from engine.cash_engine import run_cash_backtest, BacktestConfig
from factor.factor_test import run_factor_test, factor_to_weights, compute_factor_ic
from factor.plot_factor import plot_factor_report
from report.plot import plot_dashboard, plot_cash_dashboard

# cash 子命令 YAML 允许的键（白名单，拼错即 raise，不静默回落默认）：
# 引擎 BacktestConfig 字段 + end_date 截断 + --factor 路径的 factor_to_weights 策略键
_CASH_YAML_KEYS = {
    "initial_capital", "buy_fee", "sell_fee", "slippage", "turnover_cap",
    "exec_price", "start_date", "end_date",
    "rebalance", "selection", "weighting", "exclude_st", "direction", "exclude_bj",
}


def _read_yaml(path) -> dict:
    """读 YAML 配置 → dict（空文件→{}）。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在：{p}")
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def _read_factor_wide(path) -> pd.DataFrame:
    """读因子宽表（index=日期, columns=代码）。.parquet 保留索引；.csv 首列当日期索引。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"因子文件不存在：{p}")
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p, index_col=0, parse_dates=True)
    raise ValueError(f"因子文件格式不支持：{p.suffix}（仅 .parquet / .csv）：{p}")


def _max_cached_daily_year_end() -> pd.Timestamp:
    """已缓存日行情的最后年份的年末（用于把 end_date 钳进可读范围，避免 FileNotFoundError）。"""
    years = [int(p.stem) for p in (CACHE_ROOT / "daily").glob("*.parquet") if p.stem.isdigit()]
    if not years:
        raise FileNotFoundError(f"日行情缓存为空：{CACHE_ROOT / 'daily'}（先跑 data/fetch_price_daily.py）")
    return pd.Timestamp(f"{max(years)}-12-31")


def _read_weights_long(path) -> pd.DataFrame:
    """读权重长表 [date, code, weight]，date 转 datetime。.parquet / .csv 皆可。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"权重文件不存在：{p}")
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError(f"权重文件格式不支持：{p.suffix}（仅 .parquet / .csv）：{p}")
    need = {"date", "code", "weight"}
    if not need.issubset(df.columns):
        raise ValueError(f"权重文件缺列：需 {need}，实际 {list(df.columns)}：{p}")
    df["date"] = pd.to_datetime(df["date"])
    return df


def cmd_factor(args):
    """因子研究：因子宽表 + 配置 → run_factor_test → plot_factor_report。"""
    factor_wide = _read_factor_wide(args.factor)
    config = _read_yaml(args.config)
    res = run_factor_test(factor_wide, config)
    plot_factor_report(res, args.out, weighting=args.weighting)

    ic, m = res["ic"], res["meta"]
    print(f"[factor] {Path(args.factor).name} | 调仓={m['rebalance']} 池={m['universe']} 基准={m['benchmark']} 区间={m['回测区间']}")
    print(f"  IC均值={ic['ic_mean']:.4f}  RankIC={ic['rankic_mean']:.4f}  ICIR年化={ic['ic_ir_annual']:.2f}  t={ic['ic_t']:.2f}  IC胜率={ic['ic_winrate']:.2f}")
    print(f"  多空终值({args.weighting})={res['long_short'][args.weighting].iloc[-1]:.3f}  → 图+CSV 落到 {args.out}")


def cmd_backtest(args):
    """权重回测：权重长表 + 配置 → load_price_df → run_backtest → plot_dashboard。"""
    weights_df = _read_weights_long(args.weights)
    raw = _read_yaml(args.config)
    # 运行控制键 vs 引擎 config 键（其余原样透传 run_backtest）；未知键 raise（堵 buy_cost 拼错静默零摩擦）
    allowed = set(DEFAULT_CONFIG) | {"benchmark", "end_date"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"backtest 配置含未知键 {sorted(unknown)}；合法键为 {sorted(allowed)}")
    benchmark = raw.get("benchmark")
    end_date = raw.get("end_date")
    bt_config = {k: v for k, v in raw.items() if k not in ("benchmark", "end_date")}

    codes = weights_df["code"].unique().tolist()
    start = pd.Timestamp(weights_df["date"].min())  # 回测从首个调仓日起，无需前置历史
    end_req = pd.Timestamp(end_date) if end_date else pd.Timestamp(weights_df["date"].max())
    # end_req 超出已缓存年份会让 load_price_df 直接 FileNotFoundError；先钳到已缓存的最后年份，
    # 让"end_date 写到未来跑到最新"的自然用法可用。
    end_req = min(end_req, _max_cached_daily_year_end())
    need_feas = bool(bt_config.get("enable_feasibility_filter"))

    price_df = load_price_df(codes, start, end_req, need_feasibility=need_feas)
    # 再钳到实际有行情的最后一个交易日（end_req 落周末/超出当年数据末端）
    end = min(end_req, pd.Timestamp(price_df["date"].max()))
    result = run_backtest(weights_df, price_df, config=bt_config, end_date=end)

    bench_nav = calc_benchmark(load_index_eod(benchmark), start, end) if benchmark else None
    title = f"回测报告（{Path(args.weights).stem}）"
    dash = plot_dashboard(result, benchmark_nav=bench_nav, save_dir=args.out, title=title)

    nav = result["nav"]
    print(f"[backtest] {Path(args.weights).name} | 区间={start.date()}~{end.date()} 股票数={len(codes)} 调仓次数={len(result['trade_records'])} "
          f"基准={benchmark or '无'}")
    print(f"  终值={nav.iloc[-1]:.4f}  → 仪表盘 {dash}")
    ml = result["missing_log"]
    if ml:
        ml_codes = sorted({m["code"] for m in ml})
        print(f"  ⚠️ 持仓缺行/NaN 收益被当 0：{len(ml)} 格 / {len(ml_codes)} 只票（多为退市/长停，残值小）：{ml_codes[:8]}")


def _parse_pool(pool_arg):
    """--pool 解析：none/省略 → None(全市场)；存在的文件 → 读 bool 宽表；否则当指数代码字符串。

    若参数像文件路径（含 '/' 或 .parquet/.csv 后缀）但文件不存在 → raise"文件不存在"，
    不再静默当指数代码透传（否则下游报"指数 …parquet 成分股未缓存"误导排查方向）。
    """
    if pool_arg is None or str(pool_arg).lower() == "none":
        return None
    p = Path(pool_arg)
    if p.exists():
        return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, index_col=0, parse_dates=True)
    looks_like_path = ("/" in str(pool_arg)) or p.suffix.lower() in (".parquet", ".csv")
    if looks_like_path:
        raise FileNotFoundError(f"--pool 票池文件不存在：{pool_arg}（要传指数代码请用纯代码如 000852.SH，不带路径/后缀）")
    return pool_arg  # 指数代码（如 000852.SH），factor_to_weights 内部校验/报错


def cmd_cash(args):
    """现金回测：weights_df（或 因子→factor_to_weights）+ 基准 → run_cash_backtest → 简报 + CSV。"""
    if bool(args.weights) == bool(args.factor):
        raise ValueError("bt cash 需且仅需 --weights 或 --factor 之一")
    raw = _read_yaml(args.config) if args.config else {}
    unknown = set(raw) - _CASH_YAML_KEYS
    if unknown:
        raise ValueError(f"cash 配置含未知键 {sorted(unknown)}；合法键为 {sorted(_CASH_YAML_KEYS)}")
    cfg = BacktestConfig(
        initial_capital=raw.get("initial_capital", 1e8),
        buy_fee=raw.get("buy_fee", 0.0003),
        sell_fee=raw.get("sell_fee", 0.0013),
        slippage=raw.get("slippage", 0.0),
        turnover_cap=raw.get("turnover_cap", None),
        exec_price=raw.get("exec_price", "vwap"),
        start_date=raw.get("start_date"),
    )
    if args.weights:
        weights_df = _read_weights_long(args.weights)
        src = Path(args.weights).stem
        cash_end = raw.get("end_date")                   # weights 路径：end 截断回测窗口
        ic = None                                        # 无因子 → 无 IC
    else:
        sel = raw.get("selection", ["top_n", 200])       # YAML: [top_n, 200] / [top_group, 10]
        factor_wide = _read_factor_wide(args.factor)     # 读一次，factor_to_weights 与 IC 共用
        weights_df = factor_to_weights(
            factor_wide,
            rebalance=raw.get("rebalance", "M"),
            pool=_parse_pool(args.pool),
            exclude_st=raw.get("exclude_st", False),
            selection=(sel[0], sel[1]),
            weighting=raw.get("weighting", "equal"),
            direction=raw.get("direction", 1),
            exclude_bj=raw.get("exclude_bj", False),       # 剔北交所（.BJ）
            end=raw.get("end_date"),                      # end 已封顶因子窗口
        )
        # IC 与 factor_to_weights 同 rebalance/pool/exclude_st/exclude_bj/end（信号日口径、全票池）
        ic = compute_factor_ic(
            factor_wide,
            rebalance=raw.get("rebalance", "M"),
            pool=_parse_pool(args.pool),
            exclude_st=raw.get("exclude_st", False),
            exclude_bj=raw.get("exclude_bj", False),
            end=raw.get("end_date"),
        )
        src = Path(args.factor).stem
        cash_end = None                                  # 让最后一个滞后成交日(信号日+1)自然执行

    res = run_cash_backtest(weights_df, args.benchmark, config=cfg, end=cash_end)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.strategy_nav.to_csv(out / "策略净值.csv")
    res.benchmark_nav.to_csv(out / "基准净值.csv")
    res.excess_nav.to_csv(out / "超额净值.csv")
    res.account.to_csv(out / "每日账户.csv")
    res.holdings.to_csv(out / "每日持仓股数.csv")
    res.weights.to_csv(out / "每日权重.csv")
    res.trades.to_csv(out / "成交流水.csv", index=False)
    res.trade_log.to_csv(out / "买卖往返.csv", index=False)
    res.missing_log.to_csv(out / "退市清算.csv", index=False)
    res.blocked_log.to_csv(out / "成交受阻.csv", index=False)
    if ic is not None:                                    # --factor 路径才有 IC（复用 bt factor 同格式）
        pd.DataFrame({"每期IC": ic["ic_series"], "每期RankIC": ic["rankic_series"], "累计IC": ic["ic_cum"]}).to_csv(
            out / "IC明细.csv", encoding="utf-8-sig")
        pd.Series({k: ic[k] for k in ("ic_mean", "ic_ir", "ic_ir_annual", "ic_t", "ic_winrate",
                                      "rankic_mean", "rankic_ir", "rankic_t")}).to_csv(
            out / "IC统计.csv", encoding="utf-8-sig")
    fig = plot_cash_dashboard(res, args.out, title=f"现金回测仪表盘（{src}）", ic=ic)

    a, e = res.metrics_abs, res.metrics_excess
    held = int((res.holdings.iloc[-1] > 0).sum())
    print(f"[cash] {src} | 来源={'weights' if args.weights else 'factor'} 基准={args.benchmark} exec={cfg.exec_price}")
    print(f"  净值终值={res.strategy_nav.iloc[-1]:.4f}  年化={a['年化收益']:.2%}  夏普={a['夏普']:.2f}  最大回撤={a['最大回撤']:.2%}")
    print(f"  超额年化={e['超额年化']:.2%}  信息比率={e['信息比率']:.2f}  超额最大回撤={e['超额最大回撤']:.2%}")
    print(f"  末日持仓={held} 只  成交={len(res.trades)} 笔  退市清算={len(res.missing_log)} 只  受阻={len(res.blocked_log)} 次  → {fig}")
    if ic is not None:
        print(f"  IC均值={ic['ic_mean']:.4f}  ICIR年化={ic['ic_ir_annual']:.2f}  IC_t={ic['ic_t']:.2f}  "
              f"IC胜率={ic['ic_winrate']:.1%}  RankIC均值={ic['rankic_mean']:.4f}")


def main():
    parser = argparse.ArgumentParser(prog="bt", description="回测框架统一命令行入口（配置驱动）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("factor", help="因子研究：因子文件+config → IC/分组/多空/超额 + 滚动图")
    pf.add_argument("--factor", required=True, help="因子宽表文件 .parquet/.csv（index=日期, columns=代码）")
    pf.add_argument("--config", required=True, help="YAML 配置（键见 DEFAULT_FACTOR_CONFIG / examples）")
    pf.add_argument("--out", required=True, help="输出目录（图 + CSV）")
    pf.add_argument("--weighting", default="equal", choices=["equal", "factor"], help="出图用哪套权重（默认 equal）")
    pf.set_defaults(func=cmd_factor)

    pb = sub.add_parser("backtest", help="权重回测：权重文件+config → 机构研报仪表盘")
    pb.add_argument("--weights", required=True, help="权重长表文件 .parquet/.csv（列=date,code,weight）")
    pb.add_argument("--config", required=True, help="YAML 配置（引擎摩擦键 + benchmark/end_date）")
    pb.add_argument("--out", required=True, help="输出目录（仪表盘 + 单图）")
    pb.set_defaults(func=cmd_backtest)

    pc = sub.add_parser("cash", help="现金回测：weights_df 或 因子 → 调仓日驱动逐日撮合现金账户")
    pc.add_argument("--weights", default=None, help="权重长表 .parquet/.csv（列=date,code,weight）；与 --factor 二选一")
    pc.add_argument("--factor", default=None, help="因子宽表 .parquet/.csv；内部 factor_to_weights→权重；与 --weights 二选一")
    pc.add_argument("--pool", default=None, help="(仅 --factor 用)票池：none/省略=全市场 | 指数代码 | bool 宽表文件")
    pc.add_argument("--benchmark", required=True, help="基准指数代码（如 000852.SH）")
    pc.add_argument("--config", default=None, help="YAML：initial_capital/fee_rate/slippage/turnover_cap/exec_price/start_date/end_date；--factor 另加 rebalance/selection/weighting/exclude_st/direction")
    pc.add_argument("--out", required=True, help="输出目录（简报 + CSV）")
    pc.set_defaults(func=cmd_cash)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
