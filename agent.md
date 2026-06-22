# 回测框架 — 项目开发进度追踪

> 本文档由 AI Agent 自动维护，记录项目开发进度、设计决策和实现细节

## 📋 项目概述

- **项目名称**: 端到端回测框架
- **创建日期**: 2026-05-26
- **最后更新**: 2026-06-22 (拉取上游 2 提交：`data/loaders.py` 加北交所复权价质量守卫 `validate_adjclose_quality`，现金引擎 `turnover_cap` 改真·单边 + 准入跨边毒杀修复 + 现金部分成交留痕，权重引擎 `skip_small_changes` 开仓豁免 + 容量不足留痕聚合，新增剔北交所开关 `exclude_bj` 与 `tests/test_stressfix.py`。详见变更日志 2026-06-22。此前 2026-06-14 **可读性重构** + 正确性审计两轮闭环修 34 条)
- **⚠️ 新环境必跑一次**: `conda activate torch1010 && pip install -e .`（在 backtest 根目录），否则跨层导入 `from engine.backtest import` 等会 ModuleNotFoundError；装完即有 `bt` 命令
- **命令行入口**: `bt factor --factor 因子.parquet --config cfg.yaml --out 目录` / `bt backtest --weights 权重.parquet --config cfg.yaml --out 目录` / `bt cash {--weights 权重.parquet | --factor 因子.parquet} --benchmark 指数代码 --config cfg.yaml --out 目录`（cash 二选一：--weights 直接吃权重做对标 / --factor 内部 factor_to_weights→权重；配置样例见 `examples/`）
- **当前状态**: 阶段 2/3/4/5/6/7a 完成（权重引擎 + 现金引擎 + 指标 + 过滤 + 可视化）
- **当前进度**: 引擎层 阶段2(无摩擦)✅ + 阶段3+5(摩擦+可行性过滤)✅ + 阶段7a(现金账户逐日撮合)✅；数据层 阶段1 ✅
- **工作目录**: `/Users/shiyunshuo/Desktop/pythonproject/回测框架/backtest/`
- **测试环境**: conda 环境 `torch1010`
- **路线图文档**: `backtest_roadmap.md`（同目录）

---

## 🎯 设计目标

1. **端到端稳定性** — 从数据获取到绩效报告，全链路可靠，不在中间环节静默出错
2. **数据源可替换** — 当前用 SYYL/fusion_prod，后面可能换，只改适配器不动引擎
3. **交易摩擦可配置** — 手续费、滑点、涨跌停、停牌、调仓阈值全部参数化
4. **核心正确性可验证** — 每一层都有手动可验算的测试用例

---

## 🏗️ 架构设计（四层 + 适配器）

```
┌─────────────────────────────────────────────────────────┐
│                      策略层（用户写）                      │
│         输入: price_df → 输出: weights_df                 │
└────────────────────────┬────────────────────────────────┘
                         │ weights_df
┌────────────────────────▼────────────────────────────────┐
│                   回测引擎层（核心）                       │
│                                                         │
│  调仓执行管线:                                            │
│  target_weights                                         │
│    → ① 可行性过滤（涨跌停/停牌/成交量）                     │
│    → ② 调仓阈值检查（变动 < threshold 不调）               │
│    → ③ 交易量计算（trades = new_w - old_w）               │
│    → ④ 成本计算（买卖分别计费 + 滑点）                     │
│    → ⑤ 调仓日PnL拆分（持仓/卖出/买入三段）                 │
│    → ⑥ 执行（更新持仓，维护权重漂移）                      │
│                                                         │
│  逐日循环:                                               │
│    非调仓日: 按昨日持仓权重 × 当日收益率 → 组合收益          │
│    调仓日:   走管线 → 更新权重 → 组合收益                   │
│                                                         │
│  输出: NAV 序列 + 每日持仓权重 + 每次调仓交易明细           │
└────────────────────────┬────────────────────────────────┘
                         │ NAV, holdings, trades
┌────────────────────────▼────────────────────────────────┐
│                    分析层                                 │
│  指标计算: 年化收益/波动/夏普/最大回撤/Calmar/换手率         │
│  可视化:   NAV曲线/回撤曲线/月度热力图/持仓热力图            │
└─────────────────────────────────────────────────────────┘

数据翻译层（适配器模式，与引擎层解耦）:
┌─────────────────────────────────────────────────────────┐
│  DataAdapter (抽象基类)                                   │
│    ├── SyylAdapter     ← 当前: SYYL/fusion_prod 数据库    │
│    ├── CsvAdapter      ← 后备: 本地 CSV/Parquet 文件      │
│    └── XxxAdapter      ← 将来: 其他数据源                  │
│                                                         │
│  职责:                                                   │
│    get_price(codes, start, end) → 标准 price_df           │
│    get_calendar(start, end) → 交易日列表                  │
│    get_index_members(index_code, date) → 成分股列表       │
│    get_benchmark(index_code, start, end) → 基准 NAV      │
│                                                         │
│  数据校验（在适配器出口统一做）:                            │
│    - close/adj_factor 不能为 0/NaN                       │
│    - 不能有重复 (date, code)                              │
│    - 日期必须是交易日历子集                                │
│    - 涨跌幅不超过 ±22%（北交所/ST 放宽）                   │
└─────────────────────────────────────────────────────────┘
```

---

## 📐 标准数据格式

### price_df（行情数据）

| 列名 | 类型 | 说明 | 是否必须 | SYYL字段映射 |
|------|------|------|---------|-------------|
| date | datetime | 交易日 | 必须 | `trade_dt`（varchar→datetime） |
| code | str | 资产代码 | 必须 | `s_info_windcode` |
| close | float | 收盘价 > 0 | 必须 | `s_dq_close` |
| open | float | 开盘价 | 必须 | `s_dq_open` |
| high | float | 最高价 | 可选 | `s_dq_high` |
| low | float | 最低价 | 可选 | `s_dq_low` |
| volume | float | 成交量（手） | 可选 | `s_dq_volume` |
| amount | float | 成交金额（千元） | 可选 | `s_dq_amount` |
| adj_factor | float | 复权因子 > 0 | 必须 | `s_dq_adjfactor` |
| adj_open | float | 复权开盘价 | 必须 | `s_dq_adjopen`（数据库直接提供） |
| adj_close | float | 复权收盘价 | 必须 | `s_dq_adjclose`（数据库直接提供） |
| vwap | float | 成交均价 | 可选 | `s_dq_avgprice` |
| trade_status | str | 交易状态 | 可选 | `s_dq_tradestatus`（"交易" / 其他） |
| pct_change | float | 涨跌幅 (%) | 可选 | `s_dq_pctchange` |

### weights_df（调仓信号）

| 列名 | 类型 | 说明 | 是否必须 |
|------|------|------|---------|
| date | datetime | 调仓日期（必须是交易日） | 必须 |
| code | str | 资产代码 | 必须 |
| weight | float | 目标权重，同日合计 ≤ 1.0 | 必须 |

### 回测配置

```python
@dataclass
class BacktestConfig:
    start_date: str
    end_date: str
    buy_price: str = "open"           # 买入执行价: "open" / "close" / "vwap"
    sell_price: str = "close"          # 卖出执行价: "open" / "close" / "vwap"
    buy_cost: float = 0.0003          # 买入手续费率
    sell_cost: float = 0.0013         # 卖出手续费率（含印花税）
    slippage: float = 0.001           # 滑点
    rebalance_threshold: float = 0.0  # 调仓阈值（权重变动 < 此值不调）
    check_limit: bool = False         # 是否检查涨跌停
    check_suspension: bool = False    # 是否检查停牌
    initial_capital: float = 1.0      # 初始资金（归一化用 1.0）
```

---

## 🗄️ 数据源信息

### SYYL 数据库（当前数据源）

- **连接方式**: SSH 隧道 → MySQL（阿里云 RDS）
- **数据库**: `fusion_prod`（Wind 标准表结构，只读账号）
- **凭证**: 见 `backtest/.env`（已 gitignore），模板 `backtest/.env.example`

### 回测需要的核心表

| 表名 | 用途 | 数据量 | 更新频率 |
|------|------|--------|---------|
| `ashareeodprices` | A股日行情（OHLCV + 复权） | 2000万+ | 日更 |
| `asharecalendar` | 交易日历 | 3.1万 | 不定期（实际覆盖到2040年） |
| `aindexeodprices` | 指数日行情（基准收益用） | 2172万+ | 日更 |
| `aindexmembers` | 指数成分股 | 113万+ | 日内多次 |
| `ashareeodderivativeindicator` | 衍生指标（换手率等） | 2651万+ | 日更 |
| `asharedescription` | 股票基本信息（上市/退市日期） | 7787 | 基础数据 |

### 基准指数数据（aindexeodprices）— 2026-05-31 已探针确认

- **表结构**（无复权因子，指数本身不需要复权，基准 NAV 直接用 `s_dq_close`，日收益用 `s_dq_close` 或 `s_dq_pctchange`）：
  `s_info_windcode` / `trade_dt`(varchar8) / `s_dq_preclose` / `s_dq_open` / `s_dq_high` / `s_dq_low` / `s_dq_close` / `s_dq_change` / `s_dq_pctchange` / `s_dq_volume` / `s_dq_amount`
- **8 个常用基准全部存在，覆盖到 2026-05-29**（探针实测真值；注意 EOD 数据回填到指数发布日之前，如沪深300回填到2002、中证1000回填到2004）：

  | 代码 | 名称 | 行数 | 起始 | 截止 |
  |---|---|---|---|---|
  | 000001.SH | 上证综指 | 8650 | 1990-12-19 | 2026-05-29 |
  | 399001.SZ | 深证成指 | 8559 | 1991-04-03 | 2026-05-29 |
  | 000300.SH | 沪深300 | 5917 | 2002-01-04 | 2026-05-29 |
  | 000016.SH | 上证50 | 5440 | 2003-12-31 | 2026-05-29 |
  | 000905.SH | 中证500 | 5197 | 2004-12-31 | 2026-05-29 |
  | 000852.SH | 中证1000 | 5197 | 2004-12-31 | 2026-05-29 |
  | 399006.SZ | 创业板指 | 3883 | 2010-05-31 | 2026-05-29 |
  | 000688.SH | 科创50 | 1551 | 2019-12-31 | 2026-05-29 |

- **拉取脚本已写并接入** `fetch_index_eod.py`（见变更日志）；仅 000001.SH 已落盘，其余 7 个待拉。后续还需 (1) loader 读成 benchmark NAV，(2) 分析层做策略 vs 基准对比。

### ST 数据（JYDB.LC_SpecialTrade）— 2026-06-02

- **数据源是 JYDB（聚源），不是 fusion_prod**：内网 SQL Server 主机、库名和凭证都在 `.env` 的 `JYDB_*`，连接用 `db_config.make_jydb_conn()`（pymssql 直连，不走 SSH 隧道）。
- **表 `dbo.LC_SpecialTrade`**（证券特别处理事件表）+ `dbo.SecuMain`（InnerCode→SecuCode+SecuMarket）。`LC_ExgNameList` 在本实例不存在，`LC_NameChange` 无 ST 标记，故用 SpecialTrade。
- **判 ST 方法**：事件行 `SecurityAbbr` 含 "ST" 即该事件起为 ST 状态，按日期前向填充还原区间。不解码 `SpecialTradeType` 的 15 种数字。
- **市场码**（JYDB CT_SystemConst LB=201）：83→.SH、90→.SZ、18→.BJ；81=三板、SecuCategory=2=B股 已排除。
- **产物 `cache/st/st_intervals.parquet`**：列 `[code, st_start, st_end]`，每段连续 ST 一行，左闭右开（带帽日起、摘帽日止），st_end=NaT 表示仍 ST。1512 段/1079 只/1998-2026。
- **消费方式**（无前视）：判 D 日是否 ST = `(st_start <= D) & (st_end.fillna(max) > D)`。**回测引擎不剔 ST，仅供策略层选股时备用**（用户有专门的 ST 策略）。
- ⚠️ **barra 的 `st_ipo_data.parquet` 有前视偏差**（用 SecuMain 当前简称回填全历史，如 300419 实际 2024-10 才 ST 却被标 2016 起 ST）；本 ST 区间表用事件真实日期，无此问题。barra 受影响面未评估（用户暂不处理）。

### 字段映射（ashareeodprices → 标准 price_df）

```python
SYYL_PRICE_COLUMNS = {
    "s_info_windcode": "code",
    "trade_dt": "date",           # varchar(8) "20230103" → datetime
    "s_dq_open": "open",
    "s_dq_high": "high",
    "s_dq_low": "low",
    "s_dq_close": "close",
    "s_dq_volume": "volume",
    "s_dq_amount": "amount",
    "s_dq_adjfactor": "adj_factor",
    "s_dq_adjopen": "adj_open",
    "s_dq_adjclose": "adj_close",
    "s_dq_avgprice": "vwap",
    "s_dq_tradestatus": "trade_status",
    "s_dq_pctchange": "pct_change",
}
```

---

## 🔄 脚本运行顺序 / 前置依赖

| 顺序 | 脚本 | 产物 | 前置 |
|---|---|---|---|
| 0 | **`data/fetch_all.py`** | 编排 1→5 + 交叉校验 | 一键全跑入口 |
| 1 | `data/fetch_calendar.py` | `data/cache/calendar/trade_dates.parquet` | 无 |
| 2 | `data/fetch_price_daily.py` | `data/cache/daily/{year}.parquet` | 无（独立可跑）|
| 2 | `data/fetch_derivative.py` | `data/cache/derivative/{year}.parquet` | 无（独立可跑）；**白天跳板机限频，建议晚上跑** |
| 2 | `data/fetch_index_members.py` | `data/cache/index_members/{index}.parquet` | 无（独立可跑）|
| 2 | `data/fetch_index_eod.py` | `data/cache/index_eod/{index}.parquet` | 无（独立可跑）；基准指数日行情 |
| 2 | `data/fetch_st.py` | `data/cache/st/st_intervals.parquet` | 无（独立可跑）；**连 JYDB 不走隧道**；ST 区间表 |

**目录布局**:
```
backtest/
├── data/                       数据层（拉取 + 缓存）
│   ├── cache/                  所有 parquet 缓存
│   │   ├── calendar/
│   │   ├── daily/
│   │   ├── derivative/
│   │   ├── index_members/
│   │   ├── index_eod/
│   │   ├── st/                  ST 区间表（来自 JYDB）
│   │   └── description/         上市/退市日（asharedescription）
│   ├── fetch_calendar.py
│   ├── fetch_price_daily.py
│   ├── fetch_derivative.py
│   ├── fetch_index_members.py
│   ├── fetch_index_eod.py
│   ├── fetch_description.py
│   ├── fetch_st.py
│   ├── loaders.py              读缓存单一入口
│   └── fetch_all.py            数据层编排入口
├── engine/ analysis/ report/ factor/   四层 + cli.py 入口
├── tests/                      所有测试（pytest tests/）
├── scripts/                    一次性脚本（test_db_connection.py 诊断）
└── agent.md
```

**解耦设计**：5 个 fetch 脚本互不依赖，validate 内部只做"表内一致性"检查。日历交叉校验作为 fetch_all 末步集中跑：
- `validate_price_vs_calendar` — 日行情 date 列
- `validate_derivative_vs_calendar` — 衍生指标 date 列
- `validate_index_dates_vs_calendar` — 指数成分股 entry_date / exit_date（NaT 跳过）
- `validate_index_eod_vs_calendar` — 基准指数行情 date 列

**运行方式**：`cd data && python fetch_all.py`，或单跑 `cd data && python fetch_X.py`。

**`fetch_price_daily.py` → `fetch_calendar.CACHE_PATH`** 的硬依赖：
- 通过 `from fetch_calendar import CACHE_PATH as CALENDAR_CACHE_PATH` 引入路径
- `validate` 的 [6/6] 项调用 `_load_trade_dates()` 读 parquet 做交叉验证
- 日历 parquet 不存在 → `FileNotFoundError`，提示先运行 `python fetch_calendar.py`
- 后续要接入的脚本（基准指数、指数成分股等）同样应在 validate 中接日历交叉验证

---

## ⚠️ 关键设计决策 & 陷阱

### 1. 交易日历对齐（最重要）

- **所有数据必须对齐到交易日历**，非交易日的数据丢弃
- 交易日历的权威来源：`asharecalendar` 表（`s_info_exchmarket = 'SSE'` 取上交所日历）
- 该表实际覆盖到 2040-12-31（之前误读为 2026，实际 create_time 是 2026 但 trade_days 到 2040）
- weights_df 中的每个 date 必须是交易日，否则 raise
- price_df 按交易日历做 reindex，缺失日直接 raise 而不是 ffill（数据问题就该暴露）

### 2. 收益率计算

- 日收益率：`adj_close_t / adj_close_{t-1} - 1`（用复权收盘价，已消除除权除息影响）
- 调仓日收益拆分：
  - 持仓段：昨持仓 × (执行价 / 昨收复权价 - 1)
  - 交易段：执行后新持仓 × (收盘价 / 执行价 - 1)
  - 成本段：交易成本从 NAV 中扣除
- 非调仓日：权重漂移，`w_t = w_{t-1} × (1 + r_i) / (1 + r_portfolio)`

### 3. 数据源不稳定应对

- 适配器的 `get_price` 加重试（最多 3 次，间隔指数退避）
- 首次拉取后本地缓存为 Parquet（按日期分区）
- 回测引擎只认标准 price_df，不关心数据从哪来
- 换数据源 = 写新适配器 + 改一行实例化代码

### 4. 涨跌停判断

- 涨停：`close == high` 且 `pct_change >= 9.5`（10% 标准，给 0.5% 容差）
- 跌停：`close == low` 且 `pct_change <= -9.5`
- 涨停不可买入，跌停不可卖出
- 不能交易的权重回落到上一期持仓，差额按比例分配给其他可交易标的

### 5. 调仓阈值

- 从 GeneralBacktest 借鉴的关键设计
- 如果某只股票 |target_weight - current_weight| < threshold，就不调
- 避免微小调整产生不必要的交易成本
- 建议默认值 0.005（0.5%）

### 6. 买卖分价执行

- 从 GeneralBacktest 借鉴：卖出可以用 open（开盘出），买入可以用 close（收盘进）
- 比单一 rebalance_price 更灵活更真实
- 但调仓日 PnL 拆分逻辑会更复杂

---

## 📊 从 GeneralBacktest 借鉴的有价值设计

| 特性 | 说明 | 优先级 | 对应 roadmap 阶段 |
|------|------|--------|------------------|
| 买卖分价 | buy_price/sell_price 独立配置 | P0 | 阶段2（引擎核心） |
| 调仓阈值 | rebalance_threshold 避免微小调仓 | P0 | 阶段3（摩擦） |
| 调仓日PnL拆分 | 持仓/卖出/买入 三段收益 | P1 | 阶段2 |
| 总仓位控制 | position_ratio 动态现金比例 | P1 | 阶段3 |
| 现金约束回测 | 实际资金额 + 手数约束 | P2 | 阶段3之后 |
| T+0 日内回转 | 先卖后买 | P2 | 单独阶段 |
| 向量化计算 | 主流程不用 Python for 循环 | P1 | 贯穿全部 |
| 列名可配置 | adj_factor_col 等参数化 | P1 | 数据层 |

---

## 🔗 依赖关系

### 模块依赖图（规划）

```
data/
  ├── adapters/
  │     ├── base.py          # DataAdapter 抽象基类
  │     ├── syyl_adapter.py  # SYYL 数据库适配器
  │     └── csv_adapter.py   # CSV/Parquet 适配器
  ├── calendar.py            # 交易日历管理
  └── validators.py          # 数据校验

engine/
  ├── backtest.py            # 回测引擎主逻辑
  ├── rebalance.py           # 调仓执行管线
  └── config.py              # BacktestConfig

analysis/
  ├── metrics.py             # 绩效指标计算
  └── plotting.py            # 可视化
```

### 外部依赖

| 包名 | 用途 |
|------|------|
| pandas | 数据处理 |
| numpy | 数值计算 |
| matplotlib | 可视化 |
| pymysql / sqlalchemy | 数据库连接 |
| sshtunnel | SSH 隧道（SYYL 跳板机） |
| pyarrow | Parquet 缓存（可选） |

---

## ✅ 已实现功能

### 模块: 数据适配层

| 功能 | 状态 | 实现日期 | 说明 |
|------|------|----------|------|
| 架构设计 | ✅ 完成 | 2026-05-26 | 四层架构 + 适配器模式 |
| 数据源梳理 | ✅ 完成 | 2026-05-26 | SYYL 字段映射、核心表确认 |
| roadmap 细化 | ✅ 完成 | 2026-05-26 | 补充 GeneralBacktest 的借鉴点 |
| DB 连通性测试 | ✅ 完成 | 2026-05-26 | SSH 隧道 → MySQL 通，字段和数据字典一致 |
| fetch_price_daily.py | ✅ 完成 | 2026-05-26 | 全量拉取 + rename + 校验 + 按年存 parquet（含日历交叉验证） |
| fetch_calendar.py | ✅ 完成 | 2026-05-27 | 交易日历四步管线 fetch→transform→validate→save |
| fetch_index_members.py | ✅ 完成 | 2026-05-28 | 按指数分批拉 aindexmembers，含 Wind 重复行修复 + 日历交叉验证 |
| fetch_derivative.py | ✅ 完成 | 2026-05-31 | 按季度分批拉 ashareeodderivativeindicator，全量 1990-2026 共 37 年落盘（381 MB，0 失败）。已过日历交叉校验（非交易日行已查清=Wind 按自然日报市值，见 2026-05-31 晚变更日志） |
| fetch_index_eod.py | ✅ 完成 | 2026-05-31 | 按指数分批拉 aindexeodprices 基准日行情，脚手架/校验风格已与 index_members+price_daily 对齐，已接入 fetch_all（Step 5/6）。已烟测 000001.SH（8650行）。其余 7 个基准待拉 |
| fetch_st.py | ✅ 完成 | 2026-06-02 | 连 **JYDB**（聚源内网 SQL Server，非 fusion_prod）拉 LC_SpecialTrade→ST 区间表 cache/st/st_intervals.parquet。1512 段/1079 只/1998-2026，无前视。**回测引擎不剔 ST，仅供策略层备用** |
| **data/loaders.py** | ✅ 完成 | 2026-06-03 | **「读缓存」侧单一入口**（fetch_* 是写侧）：`load_calendar/load_price_df/load_index_eod/load_index_members/load_st_intervals` + `CACHE_ROOT`。从 factor_test.py 下沉而来，缓存目录结构从此只在 loaders+fetch 两处出现。上层（factor）只 `from data.loaders import`，不碰 cache 路径 |

### 模块: 回测引擎（阶段2）

| 功能 | 状态 | 实现日期 | 说明 |
|------|------|----------|------|
| engine/backtest.py | ✅ 完成 | 2026-05-28 | 无摩擦最小引擎：calc_daily_returns / update_weights / run_backtest |
| 调仓日交易日校验 | ✅ 完成 | 2026-05-29 | weights_df 所有调仓日必须 ∈ price_df 交易日，否则 raise |
| 多空支持（零投资±100%） | ✅ 完成 | 2026-06-01 | config["weight_mode"]：long_only(默认 Σ\|w\|≤1) / long_short(多头和+1、空头和−1)；单期收益 r_多−r_空，公式 Σ(wᵢrᵢ) 不变。**早期"Σ\|w\|≤1 用±50%"是错口径已废弃** |
| 阶段2 验收测试 | ✅ 完成 | 2026-05-29 | engine/test_backtest.py，8 个测试全过 |

### 模块: 回测引擎（阶段3+5 摩擦 + 可行性过滤）

| 功能 | 状态 | 实现日期 | 说明 |
|------|------|----------|------|
| run_backtest 输出改 dict | ✅ 完成 | 2026-06-01 | 返回 {nav, trade_records, blocked_trades}；config=None 退化为阶段2（golden test） |
| calc_trades / calc_cost | ✅ 完成 | 2026-06-01 | 算买卖量 / 算手续费+滑点成本 |
| skip_small_changes | ✅ 完成 | 2026-06-01 | 调仓阈值，变动<threshold 不调；不重新归一，现金自然决定 |
| check_tradable | ✅ 完成 | 2026-06-01 | 涨跌停(衍生表 limit_status)/停牌过滤；涨停只拦买、跌停只拦卖、停牌双拦；超额缩买；负权重 raise |
| 每日漂移 pipeline | ✅ 完成 | 2026-06-01 | 收盘调仓，交易前以"当天漂移后权重"为基准算换手/成本/阈值，净值多扣(1−cost) |
| calc_benchmark | ✅ 完成 | 2026-06-01 | 指数收盘价→基准NAV（起点1.0）；已对上证综指实测 |
| 阶段3+5 验收测试 | ✅ 完成 | 2026-06-01 | test_backtest.py 全过（含 golden、漂移基准回归、可行性端到端）；后续阶段3.5 扩到 18 个 |
| 每日权重记录 weights | ✅ 完成 | 2026-06-02 | run_backtest 输出加 weights（DataFrame index=交易日,columns=code），持仓类图前置；test 扩到 19 个 |

### 模块: 现金引擎（engine/cash_engine.py）—— 纯 weights_df 执行器（2026-06-08 重构）

定位：权重引擎的**真实执行孪生**。吃外部 weights_df（与权重引擎同一份），调仓日驱动逐日撮合，
非调仓日只随收盘价漂移。两类约束分家：**策略约束（选股/加权/调仓频率）归 factor_to_weights，
执行约束（涨跌停/停牌/整手/现金/换手）归本引擎**。同一份 weights_df 喂两引擎 → 对标，差额=现金约束代价。

| 功能 | 状态 | 实现日期 | 说明 |
|------|------|----------|------|
| run_cash_backtest（入口改吃 weights_df） | ✅ 完成 | 2026-06-08 | 签名 (weights_df, benchmark_code, config, start, end)；入口 fail-fast：负权重/NaN/单日Σ>1 raise；end 截断回测窗口。旧 factor 入口废弃 |
| factor_to_weights（策略层，factor_test.py） | ✅ 完成 | 2026-06-08 | 因子→稀疏 weights_df：复用 make_rebalance_dates/build_universe_mask/assign_groups/build_group_weights；selection=('top_n',N)/('top_group',n)；**因子滞后一步**（成交日前一交易日）保无未来 |
| CashBacktest（纯执行器） | ✅ 完成 | 2026-06-08 | 调仓日驱动：①复权②退市清算③[调仓日]标可交易→定目标股数→先卖后买④收盘估值。删 _rank_signal/1/n写死/每日重排；_resolve_start_date 改基于 weights 不依赖 factor |
| exec_price knob + slippage | ✅ 完成 | 2026-06-08 | exec_price='vwap'(默认)/'close'/'open' 决定成交价（估值恒用 close）；slippage 单边附加成本叠加 fee_rate。close 用作干净对标、vwap 真实，分层归因 nav_ideal≥friction≥realistic |
| 可交易性（ST 移出） | ✅ 完成 | 2026-06-08 | 涨停拦买/跌停拦卖/停牌·无价双拦（读 limit_status/trade_status）。**ST 不在执行层拦**（ST 可交易，剔不剔归策略层 exclude_st）；is_st 字段删除 |
| blocked_log（目标未达成留痕） | ✅ 完成 | 2026-06-08 | 涨停/跌停/停牌/无数据/整手不足/现金耗尽 合并记一表；观测性不改 NAV，是对标差异归因层。CLI 出 成交受阻.csv |
| turnover_cap 默认关 | ✅ 完成 | 2026-06-08 | 默认 None（无每日重排无需减速阀，对标权重引擎）；非 None 才卡单边换手、建仓日豁免 |
| 指标委托 + 超额 v3（沿用） | ✅ 完成 | 2026-06-07 | 绝对指标委托 analysis.calc_metrics；超额 plan v3 逐日算术累乘（未动） |
| engine/test_cash_engine.py | ✅ 完成 | 2026-06-08 | 18 用例：非调仓日不动仓/清仓/涨跌停·停牌拦截记blocked/退市vs停牌/复权守恒/换手卡帽开关/入口校验(负·NaN·Σ>1)/科创整手·吞没记账 |

### 模块: analysis/「算」层（被 report/factor 复用）

| 功能 | 状态 | 实现日期 | 说明 |
|------|------|----------|------|
| analysis/metrics.py · calc_metrics | ✅ 完成 | 2026-06-02 | 标量指标：年化/波动/夏普(rf=0,252)/最大回撤+起止/Calmar/胜率/盈亏比/换手；有基准加 超额年化/超额最大回撤/信息比率/Beta/跟踪误差。**2026-06-07 超额改 v3 口径：超额年化=逐日算术超额∏(1+α)复利年化(原线性 mean×252)，新增超额最大回撤，信息比不变** |
| analysis/metrics.py · build_report_data | ✅ 完成 | 2026-06-03 | run_backtest输出→ReportData（画图全部预计算序列：归一净值/回撤/月度透视/年度/滚动夏普波动/超额/换手成本/被拦计数/日收益/持仓数+权重热力）。**画图模块的数据消费契约**。2026-06-03 修 4 处：首期不丢/基准缺口不整条NaN/别名版本自适应。**2026-06-07 excess_cum 由比值(策略÷基准−1)改 v3 逐日算术累乘∏(1+α)−1（与 calc_metrics、factor.compute_excess 统一）** |
| analysis/metrics.py · 私有 helper | ✅ 完成 | 2026-06-03 | _daily_returns/_drawdown/_align_returns + _period_returns（周期收益不丢首期，月度/年度/基准年度三处复用）；calc_metrics 与 build_report_data 共用，算一次零重复 |

### 模块: report/「画」层（纯渲染，阶段6）

| 功能 | 状态 | 实现日期 | 说明 |
|------|------|----------|------|
| report/plot.py | ✅ 完成 | 2026-06-03 | plot_dashboard：机构研报风仪表盘 11 图+指标表 + 持仓分析 + 关键单图；**子图只吃 ReportData，一行加工都不算**（原 8 处加工已搬到 build_report_data）；全中文 |
| report/test_report.py | ✅ 完成 | 2026-06-03 | 6 个测试：指标手算/单调/含基准/过短raise/出图落盘/缺基准不报错 |

### 模块: factor/因子研究层（引擎上游）

| 功能 | 状态 | 实现日期 | 说明 |
|------|------|----------|------|
| factor/factor_test.py | ✅ 完成 | 2026-06-02 | run_factor_test(factor_wide,config) → IC/RankIC、十分组净值(走run_backtest)、单独多/空、多空(weight_mode='long_short' 100/100)、超额。不ffill+缺失raise、价格可用mask、因子加权带direction、ST可选剔。**分组绩效复用 analysis.calc_metrics（原自带 _nav_metrics 已删，消副本）**。**基准=外部指数（必填，缺省 raise）：bench_nav/metrics["基准"]/excess/meta 四处统一指 cfg["benchmark"]，2026-06-03 删除池内等权兜底（见变更日志）**。**ICIR 年化：自定义调仓日列表用 365.25/日历gap（与字符串 W=52/M=12 对齐，2026-06-03 修原 252/日历gap 混单位 bug）** |
| factor/plot_factor.py | ✅ 完成 | 2026-06-03 | **因子「画」层**（与 report/plot.py 对称），消费 run_factor_test 的 result：`plot_factor_report(result, out_dir, weighting='equal')` 出 7 图+6 CSV（全中文）。原 factor_test._dump_outputs 的 5 图整体搬入；**新增 ① 滚动IC/RankIC/ICIR（窗=调仓期数，复用 _PPY，与 IC 年化口径一致；≠ic_cum 累计）② 滚动夏普/波动（窗=252日，`build_report_data({"nav":那条})` 复用，不手写）**。factor_test.py 同步删 _dump_outputs/_setup_chinese_font/result_dir，退回纯产数据（见变更日志）。**2026-06-03 净值图带基准：图② 叠加基准指数虚线、图⑤ 由「多头超额单线」改「多头 vs 基准 绝对净值双线」(文件名→多头与基准净值.png)；基准重锚口径复用 build_report_data.bench_norm，② ⑤ ⑦ 共用一次 rd_long（见变更日志）** |
| factor/test_factor_test.py | ✅ 完成 | 2026-06-02 | 合成数据全 assert + --real 真实数据 sanity |
| analysis/test_factor_test.py | ✅ 完成 | 2026-06-02 | 合成单调因子全assert（IC≈+1/十分组单调/多空100·100=NAV1.03/因子加权空头最差股权重最大/avail剔退市/指定池缺因子raise）+ --real 动量sanity |

---

## 📜 变更日志

### [2026-06-22] - 拉取上游 2 提交：北交所复权价质量守卫 + 换手/开仓/容量多处正确性修复

**来源:** 从 `origin/main` 拉取（`git pull --ff-only`，快进无冲突），含 `5cdbc3b`(2026-06-17) 与 `a3ce439`(2026-06-22) 两提交。以下据 diff 记录，本地未重跑测试。

**5cdbc3b — 北交所复权价质量守卫（data/loaders.py）:**
- 新增 `validate_adjclose_quality(px, on_bad=None, max_daily_ratio=10.0)` + `DataQualityError`，挂在 `load_price_df` 返回前。逐股按日判：`adj_close≤0`，或相邻交易日 `adj_close` 比值 >10 倍或 <1/10。每股首日（无前值）不判，避开 IPO 首日无涨跌停限制；真实重整复牌（如 +300%）比值 <10 不误伤。
- 处置：默认 `BT_PRICE_ON_BAD=raise`，列最离谱前 20 条报错逼着修数据；设 `drop` 则剔除整只问题股后继续并打印。针对北交所新三板期 `close` 卡 0.01 再跳的脏价导致净值跳变。
- 顺带：删 `.env.example`（-16 行）、`.gitignore` +5 行。

**a3ce439 — commit message 写「图表 bug」，实为多处引擎正确性修复 + 剔北交所开关:**
- **backtest.py · skip_small_changes 开仓豁免**：`old_w==0` 时强制取 target。原逻辑当 `threshold > 单票目标权重` 会让所有仓位永不建立、NAV 静默卡死 1.0（静默 bug）。
- **backtest.py · check_tradable 容量不足留痕聚合**：原每只待买票各记一条 blocked，满仓 + 高换手 + 1 只锁仓会扇成几十条、夸大容量瓶颈；改为「当天一次事件」聚合记一条（`blocked_weight`=当天总缩买额）。
- **cash_engine.py · turnover_cap 改真·单边**：买 / 卖两边各自卡 cap（买边总额 ≤cap 且卖边总额 ≤cap，两边独立）。原 `Σ|Δw|/2` 在买卖不对称日（纯减仓 / 纯加仓 / 锁仓致一边为 0）会让活跃边实际单边换手达上限 2 倍，约束失效。
- **cash_engine.py · 准入按各单自己这边累计判**：买单只看 `cum_buy`、卖单只看 `cum_sell`，修原联合掩码 `(cum_buy≤cap)&(cum_sell≤cap)` 的「跨边毒杀」（一边越界后，另一边本在预算内的小单也被一并拒）。
- **cash_engine.py · 新增「现金部分成交」blocked 留痕**：买到了但钱不够一手（`qty<want_lot`）记一条，观测性不改 NAV。
- **剔北交所开关 exclude_bj**：cli.py + cash 通路 + `examples/factor_config.yaml` 样例，`exclude_bj=True` 剔除 `.BJ`。
- **metrics.py · 换手口径注释订正**：`turnover_cap` 是「单边各自卡」口径，与双边 `Σ|Δw|` 不同，对比勿混。
- **测试**：新增 `tests/test_stressfix.py`（143 行压力测试），`tests/test_backtest.py` 改 5 行。

### [2026-06-14] - 可读性重构（不改功能，71 测试当安全网，零行为变更）

**起因:** 用户反馈"测试脚本跟源码混在一块、可读性差、更像 AI 写的"。判断：代码结构本身没问题（函数粒度/行数都正常，coding-discipline 也反对给研究代码强行分层/包重构），真正的噪音是三样——测试与源码混放、审计 tag 注释、构建垃圾。**只改结构与注释，不动任何函数逻辑**。

**改动:**
- **测试抽出**：6 个 `test_*.py` 从 engine/factor/report/data 各目录移到 `backtest/tests/`（5 个真测试）+ `backtest/scripts/`（test_db_connection.py 是 DB 诊断脚本，非测试）。import 是绝对包路径（`from engine.X import`，包已 `pip install -e .`），移了照跑，无需改 import。两个 `_审计/验证脚本/` 引用测试助手的脚本改 `sys.path` 指向 tests/。
- **审计 tag 清理**：删源码里 18 处 `（发现X）/（Nx）/（Mx）/（Cx）` 这类对未来读者无意义的内部审计引用，保留其前面的"为什么这么写/防什么坑"决策性说明。测试文件里的 tag 保留（对 agent.md 变更日志的可追溯标记，且测试读得少）。
- **文档/路径校准**：CLAUDE.md/agent.md 的目录结构更新为 tests/ + scripts/ 布局；test 文件头的运行命令 `python engine/test_X.py` → `python tests/test_X.py`。.gitignore 原已覆盖构建垃圾（egg-info/pycache/.pytest_cache/data-cache）。
- **未动**：所有函数逻辑、docstring（run_backtest 等长 docstring 是 I/O+校验规则+输出结构的契约，删了反而丢信息，按 coding-discipline 公开函数必须留）、模块结构（未拆包、未拆函数）。

**验证（torch1010）:** `pytest tests/` **71 passed**（与重构前逐一相同）；e2e 对标逐点不变（首调仓日成交、两引擎相关 0.9920、终值 0.8718/0.8518，与重构前一致）；标准 runner（`python tests/test_X.py`）与 2 个归档脚本均跑通。**纯结构/注释改动，零行为变更。**

### [2026-06-14] - 审计第二轮：10 条待验真发现验真（9 真 1 已修）+ 修复

**起因:** 接上一轮（2026-06-13）留下的 10 条待验真发现（指标层 7 + 现金引擎 3），用多 agent 工作流对**当前已修改代码**对抗验真：9 条成立、1 条（C8 anchor 基准缺数据）已被上一轮 market_start 改动顺手修掉（机制在直接构造下仍能复现，但真实通路 run_cash_backtest 不可达）。验真存档 `_审计/验真结果_10条_机器可读.json`。9 条全修。

**指标层 analysis/metrics.py（M1/M3/M4/M5/M6）:**
- M1：`_align_returns` 基准在策略区间内缺交易日 → raise（传周/月频或缺行基准时，稀疏交集 pct_change 把跨多日收益当 1 日、按 252 年化致超额年化/IR/跟踪误差/Beta 虚高；框架 cli 通路 calc_benchmark 每个交易日都有值，不触发）。
- M3：`build_report_data` 年度基准先截断到策略区间再算（与 annual_strategy 同口径），修「基准比策略覆盖长 → 首年基准整年 vs 策略半年、凭空显示策略跑输」（实跑同源数据原差 14.6%）。
- M4：`bench_norm` 重锚到「策略在 anchor 日的净值水平」（非恒 1.0），基准晚于策略起步时两线在 anchor 相交、缺口=真实超额，不再把策略前段涨幅画成伪超额（仅净值对比图；常见情形 anchor=策略首日时与原行为一致）。
- M5：超额最大回撤的 `excess_nav` 补 1.0 起点锚（原首点即 1+α₁、不含 1.0，cummax 从 1+α₁ 起算低估首段回撤约 \|首日超额\|，极端报 0），与 factor.compute_excess 的 fillna(0) 口径统一。
- M6：最大回撤起 `peak_date` 取「谷前最后一次触及该轮高点」（原 idxmax 取首次触顶，净值二次触顶再跌时把已收复段并进回撤区间、起始日提前持续期虚长；回撤幅度本就正确）。

**因子层 factor/factor_test.py（M2/M7）:**
- M2：`compute_ic` 自定义调仓日 list 的 ppy——中位日历 gap≤1（相邻交易日=逐日调仓）按交易日年化 252，与字符串 'D' 同口径；原走 365.25/1=365、与 'D' 的 252 差 sqrt(365/252)=1.2 倍，同一份数据两种合法写法 ICIR 年化不一致。
- M7：`_bt_end` 删死参 `rebalance_dates`（函数体从不引用），签名 `_bt_end(calendar, end_date)`，同步改调用处与 test。

**现金引擎 engine/cash_engine.py（C9/C10 + C8 防御）:**
- C9：`_mark_to_market` 持仓票 close 全 NaN（自买入起估值价一直缺失）→ fail-fast，不让 `sum(skipna)` 把它当 0、整笔本金从账上静默消失（违反 fail-fast；根因是 close 列缺损）。
- C10：`_close_trade_log` 删死参 `extra_cash`（函数体不引用，退市现金入账实际在 _settle_delisted 自己），docstring 改为「仅记录、不动账户，现金由调用方负责」。
- C8（防御性加固，非当前 bug）：`_assemble` 基准在回测区间全 NaN → 明确报错，不让 `b.dropna().iloc[0]` 抛无上下文的裸 IndexError。

**验证（torch1010, py3.8.20）:**
- 全套 **71 passed**（24 条修复后的 65 + 本轮 6 个回归用例：M6 回撤起/M5 超额回撤锚/M1 稀疏基准 raise/M3 年度截断/M2 ICIR 一致/C9 估值 fail-fast）。
- 逐条 repro 实测：M6 回撤起=本轮峰、M5 超额回撤=-5%(原0)、M2 D与list ICIR比值=1.0000、M1 稀疏基准 raise、M3 同源策略=基准年度、C9/C8 fail-fast、C10 退市路径账户自洽。
- e2e 对标无回归：首调仓日仍真实成交、两引擎相关仍 0.9920、M1 fail-fast 不误伤（000001.SH 覆盖每个交易日）。

**至此审计两轮全部闭环**：34 条发现（原 24 + 本轮 9 成立 + 1 已修），代码已无已知正确性缺陷的待办。

### [2026-06-13] - 正确性审计修复（24 条已验真发现，多 agent 工作流审计 → 逐条修 + 测试）

**起因:** 用户要求评估框架实现质量。用多 agent 工作流做正确性审计：4 条原存疑发现对抗复核全成立，4 维度增量扫描报 28 条新发现、18 条已对抗验真（0 证伪），合计 24 条确认（高 2 / 中 12 / 低 10）+ 10 条待验真（指标层 7 + 现金引擎 3，留下一轮）。审计存档见 `回测框架/_审计/`（进度与发现.md + 机器可读.json + 验证脚本/）。本轮按"修正确性优先"修完全部 24 条。决策（用户拍板）：退市接真实退市日数据源、费率拆 buy/sell、skip_small_changes Σ>1 走 raise、修完重跑对标。

**⚠️ 两处预期的数值变化（旧现金引擎结果作废，需重出）:**
1. **现金引擎首期不再被丢**（发现1，high）：旧 `run_cash_backtest` 默认 start=首调仓日 → `_resolve_start_date` 从 i=1 扫 → 首个调仓日被静默跳过，从第二个调仓日才建仓。修复：market 窗口前移一交易日（从全日历取首调仓日前一交易日作 day-0 锚），首调仓日从此真实执行。**6/8 那次端到端对标数字含"少建仓一期"污染，已作废。**
2. **现金引擎费率默认值变**（发现5，medium）：旧 `fee_rate=0.0014` 每侧各收 14bp（往返 28bp，买侧含印花不符现实）→ 拆 `buy_fee=0.0003`/`sell_fee=0.0013`（往返 16bp，与权重引擎 buy_cost/sell_cost 对齐）。

**现金引擎 engine/cash_engine.py（发现1/3/4/5/6 + N5/N6/N7/N13）:**
- 发现3：`_adjust_for_splits` 改用 `adj.ffill()`（最后有效 adj 作分母），修复"持仓票长停牌缺行 + 跨段除权 → 市值静默腰斩"（真实缓存扫出 56 个触发实例，集中 2007-2013）。
- 发现4：退市判定改真实退市日——MarketData 加 `delist_date`（来自新 `asharedescription` 缓存），`_settle_delisted` 用 `delist_date<=t`，删除窗口内 `last_valid_index` 前视判定（旧逻辑"今天是否退市取决于未来有无价格"，同数据同策略改 end 给不同历史段 NAV，破坏可复现性）。
- 发现5：`BacktestConfig.fee_rate` → `buy_fee`/`sell_fee`，`_execute` 买卖分费。
- 发现6：`_execute` 买单 `want` 过 `_round_down_lot` 取整（复权后非整百持仓加仓不再出非法零股申报）。
- N5：`_execute` 按"股数空间实际方向"复检 can_buy/can_sell——目标方向(权重空间·t-1价口径)与实际方向(股数空间·当日价口径)可反号，旧逻辑跌停日照卖、涨停日照买且无留痕。
- N6：turnover_cap 截断的票（含清仓单/本可装下的小单）逐只记 `blocked_log(换手限额)`，不再静默落空。
- N7：建仓日卡帽豁免锚到"首个真实下单日"`_first_reb`（非 `cfg.start_date`），修显式 start_date 非调仓日时初始建仓被截半。
- N13：`run_cash_backtest` 入口加重复 (date,code) 检查（可定位、在 build_market_data 之前 fail-fast）；`_seg_sell` 静默 return → raise。

**权重引擎 engine/backtest.py（N1/N2/N3/N4/N11 + 发现2）:**
- N1：`run_backtest` 关过滤路径调仓后复检 Σ|w|（long_only），>1 → raise（skip_small_changes 跳过卖单+执行买单致负现金免息杠杆，旧逻辑静默按>100%敞口复利）。
- N2/N4：入口校验 weights_df 的 NaN 权重、重复 (date,code) → raise（校验按行求和、执行按 dict 去重，口径不一致会静默丢权重）。
- N3：`_merge_config` 未知键 → raise（堵 buy_cost 拼成 buycost 静默回落零摩擦）。
- N11：`update_weights` 漂移分母过零 → raise（long_short 极端亏损，替代裸 ZeroDivisionError/符号翻转）。
- 发现2：`calc_daily_returns` 先 `dropna(adj_close)` 再 `pct_change(fill_method=None)`——NaN 价 ≡ 缺行走 no_row 分支，修 pad 把 NaN 价当 0 收益的死分支（NAV 逐点不变，golden 不破；test_missing_returns 红测转绿）。

**数据层 + CLI + 指标:**
- 新增 `data/fetch_description.py`（拉 asharedescription 上市/退市日 → `cache/description/description.parquet`，8473 只/326 退市）+ `loaders.load_delist_dates` + 接入 `fetch_all`（Step 6/7）。
- N17/N18：`load_price_df`/`_read_years` 的 codes=[] 报错不再谎称"全市场"、start>end 报可定位错（不再裸 "No objects to concatenate"）。
- N8/N9：cli `cmd_backtest`/`cmd_cash`/`run_factor_test` 配置加键白名单，拼错键 → raise，不静默回落默认。
- N15：`cmd_backtest` end_date 超缓存年份先钳到已缓存年份（"跑到最新"自然可用），不再 FileNotFoundError。
- N16：`_parse_pool` 拼错文件路径 → raise"文件不存在"，不再静默当指数代码透传。
- 发现5：`calc_metrics` 换手键 `年均换手`→`年均换手(双边)` 标注双边口径（同步 report/plot.py），与现金引擎 turnover_cap 的单边口径区分。

**验证（torch1010, py3.8.20）:**
- 全套 `pytest engine/ report/ analysis/ factor/` **65 passed**（原 51 + 红测转绿 + 新增 13 用例：现金引擎 7 个对抗用例 + 权重引擎 6 个 N1/N2/N3/N4/N11/N12）。
- `test_build_and_account_consistent` 重写为真正的独立对账（shares×close 重算市值 vs stock_value），不再是恒真断言（N14）。
- 真实数据端到端对标（300 票×2022-06~2023-12 动量月频 top50 等权，费率拉平 buy 3bp/sell 13bp）：**首个调仓日 2022-07-01 真实成交 46 笔（发现1 验证）**、账户每日自洽、两引擎日收益相关 **0.9920**、退市 0/受阻 51 次（涨停33/跌停11/停牌6/现金耗尽1）。脚本 `_审计/验证脚本/e2e_对标_修复后.py`。

**待下一轮（凌晨限额后）:** 10 条待验真发现（指标层 `_align_returns` 跨日合并年化失真 / `compute_ic` 自定义日频 ppy 不一致 / `annual_bench` 不截断等；现金引擎 `_mark_to_market` skipna 本金消失 / anchor 缺数据等）需先验真再修。

### [2026-06-08] - 现金引擎重构为纯 weights_df 执行器（与权重引擎对标）

**起因:** 用户审出现金引擎职责混乱——它吃因子、内部把「选股 top-N + 写死 1/n 等权 + 每日重排 + 30%换手帽」全吞进去，还把停牌/ST 这种执行约束塞进了选股；而权重引擎是吃稀疏 weights_df 的纯执行器。两引擎输入契约不一致，无法用同一份 weights_df 对标。用户确认此回测**不再用于题目专用、是长期研究框架**，旧「因子是入参」思路可推倒。计划三段见 `~/.claude/plans/wild-splashing-kahan.md`。

**核心决策（用户拍板）:** ① 现金引擎收成**纯 weights_df 执行器**，与权重引擎完全对称；② 不做多空（现金侧只 long_only，见负权重 raise）；③ 因子→权重独立成 `factor_test.factor_to_weights`（不下沉成新包，复用现成函数）；④ 成交价做成 `exec_price` knob 默认 vwap，close 用作干净对标、vwap 真实，市场冲击单列 `slippage`（成交价≠成本）；⑤ 换手卡帽默认关（无每日重排无需减速阀，对标权重引擎）；⑥ ST 移出执行层归策略层。

**两类约束分家（设计核心）:** 策略约束（选什么/占多少/多久调）→ factor_to_weights；执行约束（当日涨跌停/停牌/有价、整手、现金、换手）→ 引擎。停牌的票留在 target、填不进记 `blocked_log`，不在选股阶段假装它消失。

**改动:**
- **`engine/cash_engine.py` 重写**：run_cash_backtest 改吃 weights_df（入口校验负/NaN/Σ>1，end 截断窗口）；新增 `_weights_long_to_panel`（稀疏长表→调仓日稠密目标 fillna(0)=清仓 + 调仓日集合）；CashBacktest 收成单一调仓日驱动路径（删 `_rank_signal`、`_alloc_turnover` 的 1/n 写死、每日重排循环、factor 入参；`_resolve_start_date` 改基于 weights 不依赖 factor；run 主循环调仓日才撮合、非调仓日只 `_mark_to_market` 漂移）；`build_market_data` 加 exec_price 选列；`_mark_tradable` 去掉 ST 拦买；`BacktestConfig` turnover_cap 默认 None + 加 exec_price/slippage + 删 n_holdings；`BacktestResult` 加 `blocked_log`；`_execute` 成交价用 exec_price、成本=fee+slippage、买不到记 blocked。删 `resolve_pool` / `is_st` / 三个未用 import（load_st_intervals/load_index_members/intervals_to_panel）。执行机制（复权/退市/seg记账/估值/整手）逐字保留。
- **`factor/factor_test.py` 加 `factor_to_weights` + `_select_group`**：复用 make_rebalance_dates/build_factor_panel/build_universe_mask/assign_groups/build_group_weights；selection=('top_n',N)/('top_group',n)；weighting equal/factor（factor=截面 rank 加权，沿用 build_group_weights）；**因子滞后一步**（成交日=信号日下一交易日，保 vwap/open 成交无未来）。run_factor_test 未动。
- **`cli.py` cmd_cash**：`--weights`/`--factor` 二选一 fail-fast；--factor 内部走 factor_to_weights（end 封顶因子窗口、cash 端不再传 end 让末次滞后成交自然执行）；多出 `成交受阻.csv`。
- **`examples/cash_config.yaml`** 更新（exec_price/slippage/selection/weighting/exclude_st，删 n_holdings）。
- **根目录 `cash_backtest.py` 归档** → `回测框架/_archive/`（无人 import，逻辑已并入）。
- **不动**：engine/backtest.py（权重引擎）、analysis/metrics.py（超额已统一）；未下沉策略层、未引入引擎基类。

**验证（torch1010, py3.8.20）:**
- `pytest engine/test_cash_engine.py` 18/18 全过（含非调仓日不动仓、清仓、涨跌停/停牌拦截记 blocked、退市vs停牌、复权守恒、换手卡帽开关、入口负/NaN/Σ>1 校验、科创整手+整手不足记账）。
- 全套（排除 DB 诊断）51 passed；唯一红 `test_missing_returns::test_nan_adjclose_held_no_raise` 是**既有**权重引擎 missing_log WIP（未触碰 backtest.py/该测试文件），非本次引入。
- **端到端对标（真实缓存，298 票×2022-06~2023-12 动量、月频 top50 等权）**：同一份 weights_df 喂权重引擎与现金引擎——日收益相关 0.9912（高度同向）、NAV 逐点最大差 0.0474（>0，整手/现金/手续费，符合"不逐点等"）；分层 nav 理想(权重@close)0.8843 ≥ 现金@close 0.8589 ≥ 现金@vwap 0.8332，纯摩擦 0.0254、执行参考价效应 0.0258 各自分离；blocked_log 51 次（涨停31/跌停11/停牌6/现金耗尽3）；账户每日 现金+股票市值==总资产。
- CLI 双路径烟测：`bt cash --weights`（end 截断，终值 0.8337）/ `bt cash --factor`（末次滞后成交执行，终值 0.8332）均出 10 张 CSV 含 成交受阻.csv + 简报图。

**待用户后续决策（plan 末，已按推荐默认实现，可改）:** B 因子加权当前=截面 rank（非原始因子值，若要值加权另写一支）；slippage 默认 0（体量大时开实证冲击）；turnover_cap 默认关（要当流动性约束再开，届时补"清仓豁免卡帽"）。

### [2026-06-07] - 修正科创板最小申报规则（cash_engine 整手）

**起因:** 用户审出 `_infer_lot_size` 的「688→200 整数倍」不是真实交易所规则——这是 plan v3 / 题面的简化口径，迁移时原样搬来。真实规则：科创板（688/689）单笔申报最低 200 股、200 股以上以 **1 股递增**（不是 200 整数倍）；主板/创业板 100 整数倍。

**改动（engine/cash_engine.py）:** `MarketData.lot_size`(单数字 Series) → `is_star`(bool Series)；取整逻辑收进 `_round_down_lot(qty, is_star)`（科创板 floor 到整数股、不足 200→0；其余 floor 到 100 整数倍）；`_infer_is_star` 认 688/689；`_alloc_turnover`/`_execute` 两处调用同步改；build_market_data 改填 is_star。主板/创业板行为与原来逐点一致，只有科创板变。北交所(4/8 开头)有自己的规则、数据集已排除，按 100 兜底不单独建模。

**验证:** test_cash_engine 加 3 例（_round_down_lot/_infer_is_star 单元 + 科创板≥200/1股递增 + 不足200不买），15/15 过；真实数据 688032.SH 实测持仓 2519 股（旧规则会压到 2400），1 股粒度生效。

**教训:** 硬编码的「市场规则常量」要对真实规则核，不只对计划/题面核——这类错不报错、只给个看着合理的少买股数（研究代码最危险的一类）。

### [2026-06-07] - 阶段 7a 现金引擎并入框架（根目录 cash_backtest.py → engine/cash_engine.py）

**起因:** 用户把之前写的「真实现金账户逐日撮合回测」单文件（自带读临时 BackTestData 五张宽表的数据层）改成框架专用模块。四条拍板：① 数据全切框架缓存（data/cache + loaders），旧 BackTestData 接口全弃用；② 可交易性用现成字段判（limit_status/trade_status/st_intervals），plan v3 的整板前缀阈值/价格跨日比作废；③ 超额用 plan v3 逐日算术累乘，且把框架原来的超额算法也一起改成这个；④ 因子做成入参 factor_wide。计划三段见根目录 `回测框架/现金引擎并入框架_实现计划.md`。

**关键数据口径（实测确认）:** 真实成交价=daily.vwap（直接用，不再÷adj）、真实收盘价=daily.close、复权=daily.adj_factor；停牌=trade_status=="停牌"（实测还有 XD/DR/XR/N，都是正常交易，不能误判）；涨跌停=derivative.limit_status（1涨停/-1跌停/0正常/NaN，交易日 NaN 按可交易，与权重引擎 check_tradable 一致）。

**改动:**
- **新增 `engine/cash_engine.py`**：MarketData（新底座，删 plan v3 的 twap_adj/close_adj/limit_pct）+ build_market_data + resolve_pool + BacktestConfig + BacktestResult + CashBacktest（8 步，日期由 int 改 Timestamp）+ run_cash_backtest。绝对指标委托 analysis.calc_metrics 消副本；超额净值用 v3 算术累乘。
- **新增 `data/panels.py`**：intervals_to_panel 从 factor_test 下沉（engine/factor 共用，避免 engine→factor 反向依赖）；factor_test 改 `from data.panels import`。
- **`data/loaders.py`** 加 load_daily_df / load_derivative_df（共用 _read_years），暴露 vwap/close/adj_factor/trade_status/limit_status（load_price_df 只给 adj_close，不动）。
- **`analysis/metrics.py` 超额统一 v3（全框架，含权重引擎报告）**：calc_metrics 超额年化由线性 `excess.mean()×252` 改复利 `∏(1+α)^(252/n)−1`、新增超额最大回撤，信息比不变；build_report_data excess_cum 由比值 `∏(1+rs)/∏(1+rb)−1` 改算术累乘 `∏(1+α)−1`（与 factor.compute_excess 本就一致，消掉框架内部口径分叉）。
- **`cli.py`** 加 `bt cash` 子命令；**`report/plot.py`** 加 plot_cash_report（现金简报两栏图，engine 层不进 matplotlib）；**`examples/cash_config.yaml`** 样例。
- **新增 `engine/test_cash_engine.py`** 12 个对抗用例。

**验证（torch1010, py3.8.20）:**
- `pytest engine/test_cash_engine.py` 12/12 全过（涨停拦买/跌停拦卖/停牌拦卖/ST拦买/ST可卖/退市清算/停牌复牌/复权守恒/limit NA/空窗/换手卡死/账户自洽）。
- 全框架 `pytest`（排除 data/test_db_connection.py 这个无关 DB 诊断脚本）45 passed，超额口径改动**未破坏** report/factor 既有断言。唯一红的 `test_missing_returns::test_nan_adjclose_held_no_raise` 是用户自己未提交的权重引擎 missing_log WIP（stash 我的改动后仍红），非本次引入，未触碰。
- 真实缓存端到端：`bt cash --factor (adj_close横截面) --pool none --benchmark 000001.SH` 跑通，账户每日 现金+股票市值==总资产，出简报+8 CSV。

**前置（数据缺口，提取即可，非阻塞）:** index_eod 当前仅 000001.SH 落盘；跑 zz1000/A500 基准需先 `python data/fetch_index_eod.py --indices 000852.SH`（A500=000510.SH 还需先 fetch_index_members + fetch_index_eod）。缺基准/成分时 loaders 已 fail-fast 提示。

### [2026-06-03] - 因子净值图带上基准指数（plot_factor 图②/图⑤）

**起因:** 用户反馈"factor 画图没带基准指数，尤其是净值"——基准数据(`result["bench_nav"]`)早算好了，只是净值图没画上去，看不出策略跑赢/跑输大盘。用户拍板只改两张图（图① 十分组明确不加；图⑥⑦ 与净值无关不动）。

**关键口径坑（已规避）:** 基准 `bench_nav` 按配置起始日归一到 1.0，策略各净值按首个调仓日归一——首日常不同，直接叠加基准会"抢跑"失真。复用 `build_report_data` 已有重锚口径（`reindex(策略.index).ffill()`→按首个有效点归一）把基准重锚到策略首日=1.0，**不另写对齐代码、不动 metrics.py**。

**改动（仅 `factor/plot_factor.py` 一个文件）:**
- 画 ② 前算一次共用 `rd_long = build_report_data({"nav": 单独多.dropna()}, benchmark_nav=bench_nav)`，取 `bench_norm`（基准重锚到多头首日=1.0）。单独多就是图②曲线之一，首日一致，② ⑤ 直接共用这条 `bench_norm`。
- **图②**：原三条（多空/单独多/单独空）后加一条灰色虚线 `基准(代码)`。
- **图⑤**：由 `多头超额单线` 整段改为 `多头 vs 基准 绝对净值双线`（`rd_long.nav_norm` + `bench_norm`，两线间距=超额）；**文件名 `多头超额曲线.png` → `多头与基准净值.png`**。超额明细仍在 `超额净值_*.csv`（line 125 未动），不丢数据。
- **图⑦**：多头复用 `rd_long`（已含 rolling_sharpe/vol），只新建多空那条，消一次重复 build。
- 模块 docstring 同步：图② 注"叠加基准指数线"、图⑤ 描述改"多头 vs 基准 净值（双线，间距=超额）"。

**验证（torch1010, py3.8.20）:**
- 合成 result（基准故意比策略早 15 天起）跑 `plot_factor_report`：13 个产物齐全，`多头与基准净值.png`/`多空净值.png` 出、旧 `多头超额曲线.png` 不再出、`超额净值_{equal,factor}.csv` 仍在；重锚核对——基准首日=策略首日、首点=1.000000（早出的 15 天被正确丢弃）。
- 肉眼看两张 PNG：图② 四线含灰虚线基准、图⑤ 多头实线+基准灰虚线，均从 1.0 起、间距即超额、中文无方框。
- 回归：engine 19 + report 6 + factor 合成 7 = 32 passed（无测试调 plot_factor_report，仅 cli.py 调，确认无 import 副作用）。

### [2026-06-03] - 引擎「持仓票缺行/NaN 收益」静默当 0：调查 + 记录-only 暴露（run_backtest 加 missing_log）

**问题:** run_backtest 主循环 portfolio_return（backtest.py:519-523）与 update_weights（:99-101）对持仓票当天「price_df 没行 / adj_close=NaN」一律 `ret=0`、不报错。源头：calc_daily_returns 如实保留 NaN（无锅），returns_lookup 只按现有行建（缺行→.get 落 0）；fetch_all 交叉校验只查 date⊆日历、不查每票每日的行在不在，看不见缺行。

**新增 `engine/test_missing_returns.py`（2 测试，钉死当前行为）:** 缺行/NaN 价 → 静默当 0、不 raise、NAV 照推（如缺行例 NAV=1.0→1.0→1.10 手算可核）。将来改 Fail-Fast 后在此补反向用例。

**实测探针（真实 3 年全市场十分位动量组合，4341 持仓/36 调仓，外部从 result[weights]+price 重建 missing_log，不改引擎）:**
- 命中 **75 个(日,票)静默格子、仅 4 只票**；全是 `no_row`（无 nan_return）；当日被冻结 |w| 之和**最大仅 0.28%**——影响极小。
- 分桶：**3 只退市**（002071.SZ/600240.SH/601558.SH，持有到摘牌、此后无行 → 冻结在最后价、下个调仓日按冻结价卖出，残值小）；**1 只长停**（000670.SZ 盈方微，2020-04 停牌重组、缺行 2.4 年、2022-08 复牌跳 +488%）——长停冻结 0 其实是对的。
- **关键数据事实:** 短停牌缓存**有行**（trade_status=停牌、adj_close carry，18261 行 100% 非空）→ 自然 0，不走静默路径；只有**多年长停 / 退市后**才彻底没行 → 走 .get→0。所以「无法解释的数据洞」桶**为空**，没查到真数据错误。

**修复（Fecus 选「记录-only」，已实现）:** run_backtest 返回值新增 `missing_log`（list[dict]：date/code/weight/reason，reason∈{no_row,nan_return}），仿 blocked_trades——**NAV 行为逐点不变（golden 19 全过）**，只把原本静默的缺失暴露出来，不 raise。检测点在主循环 portfolio_return（backtest.py:519+，只记 w≠0 持仓）；update_weights 不重复记（同一 today_returns/持仓，会双计）。cli.py `bt backtest` 跑完若 missing_log 非空打一行 ⚠️ 提示。

**验证:** 引擎原生 missing_log 与外部重建逐一对账一致（真实 3 年十分位：75 格/4 票/全 no_row，000670.SZ+三只退市）；engine/test_missing_returns.py 加断言 missing_log 记下 no_row/nan_return；report+factor 套件无回归。

**未做（如需再说）:** run_factor_test 内部各组子回测的 missing_log 未向上聚合（_run 只取 nav）；门控 raise（排除退市/长停后仍缺才 raise）需接退市判定数据源，留作后续。

### [2026-06-03] - 画图中文字体跨平台化 + 收口为单一真相源

**起因:** 出图中文字体要 Linux/mac/Win 三系统同一份代码都能用。原来字体设置在两处各写一份且都偏 mac：plot_factor `_setup_chinese_font`（STHeiti/Songti SC… 全 mac）、plot.py `_apply_style`（Arial Unicode MS/SimHei/PingFang SC）——既不跨平台、又是副本会漂移。

**改动:**
- 新增 `analysis/plot_style.py` · `setup_chinese_font()`：`font.sans-serif` 给一条跨平台 fallback 列表（matplotlib 按序取第一个已装的）——Win(微软雅黑/黑体) → mac(苹方/华文黑体/宋体/Arial Unicode) → Linux(思源黑体 Noto/Source Han、文泉驿) → DejaVu 兜底；`axes.unicode_minus=False`。放 analysis/（report+factor 都依赖它），**独立于 metrics.py 以免算层引入 matplotlib**。
- report/plot.py `_apply_style` 与 factor/plot_factor.py 都改调 `setup_chinese_font()`，删掉两处各自的字体列表（消副本，纪律1）。
- **Linux 注意**：三类 CJK 字体都没装会显示方框，装一个即可（`apt install fonts-noto-cjk` 或 `fonts-wqy-zenhei`），已写进函数 docstring。

**验证（torch1010, mac）:** 候选 13 个，本机按优先级实际选中 STHeiti（真 CJK，非 DejaVu 兜底）；report 6 测试全过；`bt factor` 重出图肉眼确认标题/图例/轴中文（滚动夏普/波动等）正常无方框。

### [2026-06-03] - 新增统一命令行入口 cli.py（bt factor / bt backtest，YAML 配置驱动）

**起因:** 用户问"要不要加统一使用入口"。澄清工作方式=**命令行跑脚本/批量**（非 notebook 交互），故加一个薄 CLI：配置驱动、可复现、能批量扫参。范围用户拍板=因子 + 权重回测两条；配置格式=YAML。

**实现（新增 `cli.py` 顶层入口模块，纯编排，不含因子/策略计算）:**
- `bt factor --factor 因子.parquet --config cfg.yaml --out 目录 [--weighting equal|factor]`：读因子宽表 + YAML 配置 → `run_factor_test` → `plot_factor_report`（IC/分组/多空/超额 + 滚动图 7 图 6 CSV）。
- `bt backtest --weights 权重.parquet --config cfg.yaml --out 目录`：读权重长表 → `load_price_df`（按权重 codes+区间）→ `run_backtest` → `plot_dashboard`（机构研报仪表盘）。YAML 里 benchmark/end_date 是运行控制键、其余透传引擎 config。**end_date 落周末/超数据 → 钳到有行情的最后交易日（与 factor 侧 _bt_end 同口径，区间打印出来透明可见）。**
- 输入文件 .parquet/.csv 皆可（因子=宽表 index=日期；权重=长表 date/code/weight）；因子/权重你自己在框架外算好存文件，CLI 只装配。
- 装包接线：`pyproject.toml` 加 `[project.scripts] bt = "cli:main"` + `py-modules=["cli"]` + 依赖加 pyyaml；`pip install -e .` 后 `bt` 进 PATH。
- 配置样例落 `examples/factor_config.yaml` + `examples/backtest_config.yaml`（带注释，可直接改）。

**不做（守"个人研究工具不过度设计"）:** 无插件化、无配置校验库、不内置因子/策略计算、不搞日志框架。

**验证（torch1010，从 /tmp 任意目录跑装好的 bt）:** 造真实小样例（动量因子 323×120 宽表、15 股月频等权权重，均 2022）——`bt factor` 出 7 PNG+6 CSV（IC=-0.150/ICIR年化=-4.40，小宇宙强反转合理）；`bt backtest` 出仪表盘+5 单图（区间 2022-01-28~2022-12-30 钳位生效、12 调仓、终值 1.049、基准 000001.SH 超额图正常），肉眼确认中文与 12 图全渲染。

### [2026-06-03] - 结构治理：可编辑安装包统一导入 + 缓存读取下沉 data/loaders.py

**起因（用户提的两个最实际的结构问题）:** ① factor_test.py / plot.py / test_report.py 等 5 个文件各自手改 `sys.path.insert`，从不同目录跑测试、拆文件、交给 AI 改容易因导入路径炸。② factor_test.py 直接读 calendar/price/index/ST 缓存，因子层不该关心缓存目录结构。

**用户拍板:** 导入机制选「可编辑安装包」（三选一：装包 / 设 PYTHONPATH / -m 跑）。

**改动 1 — 统一导入，去掉全部 sys.path.insert:**
- 新增 `pyproject.toml`（setuptools，packages=engine/analysis/report/factor/data）+ 五层各一个空 `__init__.py`。`pip install -e .` 后五层成顶层包。
- 6 个文件的跨层导入统一成 `from <层>.<模块> import`：`from engine.backtest import` / `from analysis.metrics import`（原裸 `from metrics`）/ `from report.plot import`（原裸 `from plot`）/ `from factor.factor_test import`（原裸 `from factor_test`）/ `from data.loaders import`。删光 10 处 sys.path.insert（factor_test/plot_factor/test_factor_test/plot/test_report）+ engine/test_backtest 的裸 `from backtest import` 改 `from engine.backtest import`。
- 顺带删掉因 sys.path 而留的 `import sys`/`from pathlib import Path` 冗余（各文件按实际是否还用判定）。

**改动 2 — 缓存读取下沉数据层:**
- 新增 `data/loaders.py`：把 factor_test.py 的 5 个 `load_calendar/load_price_df/load_index_eod/load_index_members/load_st_intervals` + `CACHE` 常量整体搬来（改名 `CACHE_ROOT`，本文件在 data/ 故 `parent/"cache"`）。`loaders.py` 是「读缓存」侧、`fetch_*.py` 是「写缓存」侧，缓存目录结构从此只在这两处出现。
- factor_test.py 删掉这 5 个函数 + `CACHE`，改 `from data.loaders import ...`；test_factor_test.run_real 的 `from factor_test import load_calendar,load_price_df,run_factor_test` 拆成 `from factor.factor_test import run_factor_test` + `from data.loaders import load_calendar,load_price_df`。
- engine/report/analysis 本就不读缓存（calc_benchmark 收 in-memory df），不受影响。

**验证（torch1010, pip install -e . 后）:** ① 从 `/tmp` 任意目录 import 六层全解析成功（位置无关达成）；② 引擎19 + 报告6 + 因子合成 三套全过；③ 真实 3 年月频动量端到端（含 data.loaders 读缓存 + factor→plot 全链路）：7 PNG+6 CSV 落盘、滚动IC/夏普边界精确，**IC均值/ICIR/多空终值与重构前逐位一致**（-0.0304/-0.92/0.692，纯结构重构零行为变化）。

**注意:** 新克隆/换环境必须先 `pip install -e .`（已写进 agent.md 顶部）；`*.egg-info/` 等装包产物已加 .gitignore。顶层包名 engine/data 较通用，个人 env 暂无碰撞。

### [2026-06-03] - 因子「画」层拆出 factor/plot_factor.py（彻底解耦 + 补滚动图）

**起因:** 用户要给 factor_test 结果画图（滚动IC + 标准套件），先评估"数据够不够直接消费"——结论：够，零复杂变换。标准套件 result 里已直接有；滚动指标只是 ① ic_series 一行 rolling ② 任意净值丢进 build_report_data 取 rolling_sharpe/vol。借此把混在 factor_test.py 里的出图逻辑拆成独立「画」层。

**架构（用户拍板"彻底解耦"）:** 新建 `factor/plot_factor.py` 作因子研究「画」层，和引擎侧 `report/plot.py` 对称。一处画图，无副本。

**改动:**
- **新增 `factor/plot_factor.py`** — 公开入口 `plot_factor_report(result, out_dir, weighting='equal')`，出 7 图 + 6 CSV（全中文文件名）：
  - 搬入原 _dump_outputs 5 图（十分组净值/多空净值/分组年化柱/IC时间序列+累计IC/多头超额）+ 4 类 CSV。分组年化柱改**直接取 result["metrics"] 已算值**，不再 calc_metrics 重算。
  - 新增 **滚动IC**：roll_ic/roll_rankic 用 `.rolling(n).mean()`、roll_icir 用 `mean/std`；窗口 `n=_PPY.get(rebalance,12)`（**调仓期数**，月频12期=1年，与 compute_ic 年化口径同源；自定义列表退12，按 isinstance 守住 _PPY.get 不吃 unhashable list）。
  - 新增 **滚动夏普/波动**：对 多空/多头 各 `build_report_data({"nav":那条})`，取 `rolling_sharpe`/`rolling_vol`（**窗口252日**，不手写公式=不留 metrics.py:226 第二份副本）。
- **改 `factor/factor_test.py`（删，不加）** — 删 `_dump_outputs`/`_setup_chinese_font` 整函数、`DEFAULT_FACTOR_CONFIG["result_dir"]` 键、run_factor_test 末尾 `if cfg["result_dir"]` 出图分支。run_factor_test 只 `return result`。`from metrics import calc_metrics` 保留（分组绩效仍用）。test_factor_test.py 不依赖 result_dir，无需改。

**两处易错点（已写进图与注释防再踩）:** ① 滚动IC ≠ ic_cum（cumsum 累计），窗口单位是期数不是252天；② 滚动夏普窗口才是252天，两者别混。

**验证（torch1010）:** ① 合成 7 测试全过（factor_test 删改无回归）；② 真实 3 年月频动量（2020-2022 全市场剔BJ，因子宽表 809×4989）端到端：7 PNG + 6 CSV 全落盘非空；滚动IC 35期→前11 NaN+24有值（边界精确）；滚动夏普 713日→461有值（=713−252）；ic_cum(-1.065)≠滚动IC末值(-0.063) 断言通过；两张新图肉眼确认渲染真线、中文正常。IC均值-0.030/ICIR年化-0.92/多空终值0.692（A股动量反转，量级合理）。

### [2026-06-03] - factor_test 基准口径统一为外部指数（用户对照源码 review 出 #1）

**背景:** 用户 review 发现 factor_test 的"基准"口径分裂——传了外部指数（如沪深300）时，`excess` 和 `meta["benchmark"]` 指外部指数，但 `result["bench_nav"]` 和 `metrics["基准"]` 仍是池内等权。同一份结果里"基准"指了两个东西，易误读。

**用户拍板:** 基准只认外部指数；不传外部指数直接 raise；彻底不要池内等权当基准。

**改动（factor_test.py + test_factor_test.py，纯口径收口，未碰 IC/分组/超额算法）:**
1. **入口 fail-fast** — `run_factor_test` 在 `cfg.update(config)` 后立即校验 `cfg["benchmark"]`，缺省 raise ValueError（带值定位），早于 load_calendar/load_price_df，不碰 DB。
2. **bench_nav 改外部指数** — 原 `bench_nav = backtest_benchmark(...)`(池内等权) 改为 `calc_benchmark(load_index_eod(cfg["benchmark"]), start, bt_end)`；删除 `bench_for_excess` 中间变量，excess 直接用 bench_nav。四处（bench_nav/metrics["基准"]/excess/meta）统一指外部指数。
3. **删死代码（纪律1/5）** — `backtest_benchmark` 和它唯一依赖 `build_equal_weights` 全删（删后全树零引用；`_run`/`run_backtest` 仍被分组/多空用，保留）。
4. **DEFAULT_FACTOR_CONFIG["benchmark"]** 注释由"None→对池内等权"改为"必填，缺省 None 入口 raise"。
5. **测试同步** — `run_real()` 的 `benchmark: None` 改 `"000001.SH"`（当前唯一已落盘指数）+ 打印文案改"对 000001.SH 超额"。

**验证（torch1010）:** 合成 7 测试全过（不碰 run_factor_test，回归无变）；新写 case 确认 `benchmark=None` 在入口即抛 ValueError 且未触达 DB；语法编译通过。

**后续:** review 提的问题 #2（自定义调仓日列表 ICIR 年化混单位）已在下一条修复。

### [2026-06-03] - compute_ic 自定义调仓日列表 ICIR 年化修单位（用户对照源码 review 出 #2）

**背景:** `compute_ic` 对自定义调仓日列表（`rebalance` 是 list 而非 'M'/'W'/'D'）算年化周期数 `ppy` 时用 `252/日历gap`——分子 252 是交易日/年、分母是日历日，混单位。周频列表 gap≈7 得 ppy≈36、月频 gap≈30.4 得≈8.4，而字符串路径 `_PPY` 是 W=52/M=12。同一频率两条路径不一致，`ic_ir_annual = ic_ir·√ppy` 自定义列表偏低约 17-18%。（`ic_t = ir·√n` 用期数，不受影响。）

**改动（仅 factor_test.py 一处，3 行）:** `252.0` → `365.25`（全程日历口径），加注释说明。选 `365.25/日历gap` 而非 `252/交易日gap`：`compute_ic` 签名无日历，走交易日口径要改签名+调用点；日历口径对真实的周/月列表结果即 52/12，与字符串路径对齐；自定义"日频"列表是退化情形（直接用 `"D"`），不值得为它加参数。

**验证（torch1010）:** 直接验 ppy——周频列表 52.0、月频列表 12.0，与字符串 W/M 完全一致；合成套件全过无回归（套件用 `"D"` 字符串路径，不触自定义分支）。

### [2026-06-03] - metrics.build_report_data 修 4 处报告序列问题（用户对照源码 review 出）

**背景:** build_report_data 的标量指标核心公式无误，但 4 处报告序列/基准对齐有坑，现有 test_report.py 6 测试未覆盖。

**改动（仅 metrics.py，外加 plot.py 一句 docstring）:**
1. **新增私有 helper `_period_returns(nav, freq)`** — 周期收益，首期相对起始 NAV 算（`首期末NAV/起始NAV−1`），不丢首期。月度/年度/基准年度三处复用（消重复 + 一处修首期，纪律1/2）。
2. **issue1 首期不再丢** — 原 `resample().last().pct_change().dropna()` 把首月/首年直接丢掉（**单年回测年度收益=空**、月度缺 1 月）。改走 `_period_returns`。
3. **issue2 基准缺口不再整条 NaN** — 原 `b/b.iloc[0]`，基准首日晚于策略首日时 `b.iloc[0]=NaN` 致 bench_norm 全 NaN。改按 `first_valid_index()` 归一；基准与策略**完全无交集时 raise**（带日期定位，Fail-fast）。
4. **issue3 累计超额改口径** — 由 `∏(1+s−b)−1`（每日超额复利）改为 **`策略累计净值/基准累计净值−1`（相对超额，用户选定）**。ReportData 字段注释 + plot.py 图7 docstring 同步。
5. **issue4 别名版本自适应** — `M/Y` 在 pandas 2.2 弃用、`ME/YE` 在 2.0.x 崩。加模块级 `_FREQ_MONTH_END/_FREQ_YEAR_END`（≥2.2 用 `ME/YE`，否则 `M/Y`）。**torch1010=pandas 2.0.3 走 M/Y 零警告，升级后自动切 ME/YE**。（注：用户报的 FutureWarning 来自更新的 pandas 环境，非 torch1010。）

**验证（torch1010, pandas 2.0.3 / py3.8.20）:** 现有 report 6 测试无回归；新增专项 5 项全过——单年年度=30%(非空)/首月非空、基准缺前2日 bench_norm 首点=1.0 开头NaN、超额口径=策略/基准−1、无交集 raise、无 FutureWarning。

**未做（建议后续）:** 这 4 个场景的永久回归用例尚未并入 report/test_report.py（专项验证脚本临时放 /tmp）；calc_metrics 标量指标核对无误未动。

### [2026-06-03] - 分析层重构：拆 analysis(算)/report(画)/factor(因子) 三层 + 消指标副本

**起因:** factor_test 和画图代码混在 analysis/ 一个目录；plot.py 每个子图自己做数据加工（resample/rolling/cumprod/pivot 等 8 处），违背"plot 只画图、加工在上游"。

**架构（用户定）:** 两桶——"算"(被复用) vs "画"(纯渲染)。落成三目录，依赖无环：
- `analysis/`「算」层：metrics.py，谁都不依赖，被 report/factor 复用。
- `report/`「画」层：plot.py + test_report.py，引擎下游。
- `factor/`：factor_test.py + test，引擎上游。
- 依赖方向：factor → engine ← report，绩效/绘图都 → analysis。

**改动:**
- metrics.py 加 `_daily_returns/_drawdown/_align_returns` 私有 helper（calc_metrics 与新函数共用，算一次）；新增 `ReportData` dataclass + `build_report_data()`：把原本散在 plot 里的 8 处加工全预计算成序列。**ReportData 字段 = 画图模块的数据消费契约**，将来 build_report_data 可整体搬进引擎，plot 一行不改。
- plot.py 重写为纯渲染：每个 `_plot_*` 子图只接 `ReportData + ax`，零加工；`plot_dashboard` 先调 build_report_data 再画。
- factor_test.py 删自带 `_nav_metrics`（与 calc_metrics 同逻辑的副本），分组绩效改调 `analysis.calc_metrics`（消副本，纪律1）。验证：_nav_metrics 的 `累计收益/卡玛` 键无人读，只用 年化收益/夏普/最大回撤（calc_metrics 都有），swap 安全。
- 文件移动：factor_test+test→factor/，plot+test_analysis(→test_report)→report/，metrics 留 analysis/。各文件 sys.path 相应调整（parent.parent 到 backtest 根的引用因层级不变而仍有效）。

**验证:** engine 19 + report 6 + factor 合成 三套全过；仪表盘出图肉眼对比重构前后一致（纯渲染未改行为）。

### [2026-06-02] - 因子测试模块（analysis/factor_test.py，引擎前一层）

**目标:** 给一套因子 → IC / 十分组 / 多空 / 超额，全部复用权重引擎 run_backtest，不改引擎。

**设计契约（与用户多轮敲定）:**
- 模块 = 引擎"前一层"：传因子 → 内部转权重(仅喂引擎、不导出) → run_backtest → 出收益率。单一入口 `run_factor_test`。
- 各组合走引擎：十分组/单独多/空 用 `long_only`；多空用 `long_short`（100/100，多头和+1/空头和−1，复用引擎已升级的 weight_mode）。
- **不 ffill、不替用户填因子**：指定指数/user 池内可交易股因子缺失 → raise（不同因子填法不同，用户在框架外自己填）；全市场 NaN=该期出池。
- IC 沿日历轴算未来持有期收益[T,T']（与引擎持仓窗口同口径、无前视）；输出 IC/RankIC + ICIR(每期&年化) + t + 累计IC。
- 股票池：全市场/指数成分/user；价格可用 mask 挡退市/未上市；ST 可选剔（共用 `intervals_to_panel` 读 st_intervals）。
- 因子加权：score=direction×factor，多头按 score、空头按 −score 的组内 rank 加权（稳健处理负因子）。
- 绩效用**自包含 `_nav_metrics`**（年化/夏普/最大回撤/卡玛/胜率）；按用户要求**不复用未 review 的 metrics.py/plot.py**。

**验收（test_factor_test.py，torch1010 全过）:**
- 合成单调因子：IC≈+1、十分组单调、多空 100/100（NAV=1.03 而非 50/50 的 1.015）、因子加权空头最差股权重最大、avail 剔退市股、指定池缺因子 raise。
- 真实 20 日动量 sanity：IC=−0.066 / ICIR年化 −2.05 / t=−1.97，十分组高动量组 −30%（A股动量反转），多空夏普 −1.38——量级合理。

**踩坑 / 发现:**
- ⚠️ **北交所(.BJ) adj_close 有脏数据**：838227.BJ 2022-02-25 单日 +200400%（9.38→18807）、870199.BJ +5163%。全市场回测被这些 spurious return 污染（根因是复权价值错，非对齐）。因子研究已排除北交所；**fetch_price_daily 的北交所复权价待核查（数据层 TODO）**。
- 多 agent workflow 把计划对照真实代码核验出并修掉 4 个 bug：calc_benchmark 三参、end_date 默认只到末调仓日(须显式传)、NaN/重复权重引擎静默放过(builder 自检+喂前防御)、price_df 去重自检。
- build_factor_panel 对"区间外退市/未上市"的因子 code 改为剔除+log（无价无法测，非 raise）；"可交易股因子缺失"才 raise。

### [2026-06-02] - 阶段4+6：分析层（绩效指标 + 机构研报风可视化仪表盘）

**目标:** 引擎能跑出净值后，建 analysis/ 把结果算成指标 + 画成图。对应 roadmap 阶段4（指标）+6（可视化）合并。

**引擎小改（持仓图前置）:**
- `run_backtest` 主循环每日记 `weights_history[t]=dict(current_weights)`，返回 dict 加 `weights` 键（DataFrame index=交易日, columns=code, 缺失=0）。非破坏性：现有测试读 `result['nav']` 不受影响，golden test 全过。test_backtest.py 加"每日权重记录"用例 → 19 个全过。

**新增 analysis/:**
- `metrics.py` `calc_metrics(nav, benchmark_nav=None, trade_records=None, periods_per_year=252)`：rf=0、252 年化。算总收益/年化/波动/夏普/最大回撤(+起止日)/Calmar/日胜率/盈亏比；有 trade_records 加年均换手；有基准加超额年化/信息比率/Beta/跟踪误差。日收益用 pct_change、回撤用 cummax，不造轮子。
- `plot.py` `plot_dashboard(result, benchmark_nav, save_dir, title)`：机构研报风（白底/深蓝主色#1f3a5f/暖灰基准/细网格/去顶右边框/顶部KPI条）。一页仪表盘 11 图+指标表（4×3 GridSpec）+ 持仓分析图（数量+权重热力）+ 关键单图（净值/回撤/月度热力/超额）单独存。中文字体复用 barra 配置。子图函数只画不算指标（指标在 metrics）。
- `test_analysis.py` 6 个测试全过。

**验证:** 指标手算核对（NAV=[1,1.2,0.9,1] 最大回撤=-25%、起止日正确）；构造 3 年行情跑引擎→出图，肉眼检查仪表盘中文正常、11 图全填充、研报风美观；缺基准不报错；引擎 19 个测试回归全过。

**口径:** rf=0（夏普=年化收益/年化波动）、年化 252。基准对比用指数 base（超额/信息比率/Beta）。

**不做:** 调仓日 PnL 三段归因（属现金引擎）、交互式图、不碰引擎收益/漂移逻辑（只加 weights 记录）。

### [2026-06-02] - ST 数据提取（JYDB.LC_SpecialTrade，无前视区间表）

**目标:** 把 ST 标识数据提取备用（回测引擎不剔 ST，留给策略层；用户有专门 ST 策略）。要无前视时点口径。

**数据源调查（探了 fusion_prod 和 JYDB 两个库）:**
- fusion_prod **没有**可做无前视 ST 的源：无 asharest、无名称变更历史表、量价表无名称列；s_info_name 只是当前名快照（前视）；bzsq_stock_st_risk 是事件流且只到 2024-09、95% 非 ST 风险预警。
- JYDB 有 **`LC_SpecialTrade`**（证券特别处理事件表），覆盖 1998-2026（ST 制度 1998 才开始，即全历史），InnerCode 维度，事件含简称。`LC_ExgNameList` 本实例不存在、`LC_NameChange` 无 ST 标记。

**实现:**
- `.env` / `.env.example` 加 `JYDB_*`；`db_config.py` 加 `make_jydb_conn()`（pymssql 直连内网 SQL Server）。
- `fetch_st.py`：拉 LC_SpecialTrade ⋈ SecuMain（A股 SecuCategory=1、市场 83/90/18），简称含 ST 判状态，事件→区间。校验 3 项（非空/end>start/同 code 不重叠）。产物 `cache/st/st_intervals.parquet`（1512 段/1079 只/24.5KB）。

**关键发现（交叉验证暴露的前视 bug）:**
- 用 barra `st_ipo_data.parquet` 交叉验证，差异极大（交集仅 ~12-64/200）。追查 300419：barra 全程标 ST浩丰（2016 起），但 LC_SpecialTrade 显示 2024-10-29 才带帽。
- 证实 barra 那份是**当前简称回填历史**（前视偏差），本 ST 区间表用事件真实日期才正确。交叉"对不上"恰恰证明本提取无前视。
- barra 受影响面未评估（用户选不顺带，先专注回测框架）。

**踩坑:** pymssql + pandas.read_sql 报 SQLAlchemy UserWarning（仅警告不影响）；JYDB 首次 TCP 偶发超时，重试即 0.0s 通。

### [2026-06-01] - 阶段3.5：修正多空权重口径（±50% → 零投资 ±100%）

**背景:** 早期多空设计用"多 50% + 空 50%、Σ|w|=1"，经文献核查（Detzel-Novy-Marx-Velikov 2023 JoF、Gray et al. 2024）确认是错口径——多空收益被砍半。标准是零投资组合：多头和 +100%、空头和 −100%，单期收益 = r_多 − r_空（直接相减，不除以2）。

**关键认知（为什么改动很小）:**
- 收益公式 `Σ(wᵢrᵢ)` 本就对：喂 ±100% 自动得 r_多 − r_空，逐日累乘 = 文献 ∏(1+r_多−r_空) 的逐期口径。**公式、漂移、累乘一行都不用改。**
- 真正挡路的只是约束 `Σ|w|≤1`（总敞口上限1，逼用 ±50%）。
- 代数验证：持仓(漂移)的正确口径是两腿恒等式 `NAV_t = ∏(1+r_多) − ∏(1+r_空) + 1`（cash=1−Σw 在 Σw=0 时=1，正是 $1 保证金账户）。

**改动（仅校验分叉 + 测试 + 文档）:**
- `DEFAULT_CONFIG` 加 `weight_mode`（默认 "long_only"）。入口校验规则3 分叉：long_only 维持 Σ|w|≤1；long_short 校验多头和≈+1、空头和≈−1；非法值 raise。
- long_short + enable_feasibility_filter → 入口 raise（多空可行性未实现，多空仅理想态）。
- 测试5 重写为 ±100% long_short（D1 NAV=1.2=1+r_A−r_B）；新增测试5b 两腿恒等式回归、测试8 long_short 校验（多头/空头和≠目标、非法 mode、+过滤 均 raise）。
- 接口：直接传负权重做空，不单独激活空头。

**验收:** test_backtest.py 共 18 个测试全过（含退化：默认 long_only 所有旧测试行为不变）。

**未改:** 收益公式 Σ(wᵢrᵢ)、update_weights 漂移、check_tradable、calc_* 全部不动。stage3+5 摩擦/过滤逻辑不碰。

### [2026-06-01] - 阶段3+5：交易摩擦 + 可行性过滤（引擎扩展完成）

**目标:** 在阶段2 权重引擎上加交易摩擦（手续费/滑点/调仓阈值）和可行性过滤（涨跌停/停牌），阶段3和5合并一步到位。全部加在 engine/backtest.py，不新建文件。

**新增 5 个函数 + run_backtest 改造:**
- `run_backtest` 签名加 `config` 参数，返回值从 pd.Series 改为 dict `{nav, trade_records, blocked_trades}`。`_merge_config` 用 DEFAULT_CONFIG 补齐缺键。
- `calc_trades(old, new)`：买卖量 = new−old，并集缺失补0，丢弃0，空/相同→{}。
- `calc_cost(trades, config)`：买入Σ×(buy_cost+slippage) + 卖出Σ|·|×(sell_cost+slippage)。
- `skip_small_changes(old, target, threshold)`：|变动|<阈值保持old，不重新归一，现金自然决定。
- `check_tradable(old, target, day_data)`：用衍生表 limit_status 判涨跌停（涨停=1只拦买、跌停=-1只拦卖、停牌双拦、无数据双拦）；买不进留现金、卖不掉锁仓、超额按想买量等比例缩买（少买记 reason=容量不足）；负权重 raise（严格版只做多）。
- `calc_benchmark(index_df, start, end)`：指数 close→NAV（起点1.0），非交易日起始用其后最近交易日。

**关键实现点（每日漂移 pipeline）:**
- 收盘调仓。每天：收益用早上权重算（不变）→ 更新NAV → `update_weights` 漂移得到交易前权重。调仓日以**漂移后权重**为基准走 skip→check→trades→cost，净值再 ×(1−cost)；非调仓日 current=漂移权重。
- 退化路径：config=None 时 NAV 逐点等于阶段2。漂移只改"成本/换手/阈值"的会计基准，不改 NAV。
- `day_data_lookup` 仅在 enable_feasibility_filter=True 时构造；开过滤但缺 limit_status/trade_status 列 → raise（校验规则6）。

**验收（torch1010）:** test_backtest.py 共 16 个测试全过：
- 8 个阶段2 golden test（config=None NAV 不变）
- calc_trades / calc_cost / skip_small_changes / check_tradable 单测（含图2 超额缩买例子精确核对）
- 有摩擦≤无摩擦逐点；**漂移基准回归**（目标=漂移后权重时换手/成本=0，证明用对了基准）
- 可行性过滤端到端（涨停拦买+blocked记录、缺列raise）
- calc_benchmark 对上证综指实测（起点1.0、2024终点1.1315、非交易日起始跳D0）

**两处合约/文档冲突已按"项目当前实现为准"改掉:**
- 不做买卖分价 / 不写 calc_rebalance_return（归现金引擎）。CLAUDE.md 第六/八/十二/十三节 + phase3_contracts_v2.md 已同步。
- 多空：引擎入口保留 Σ|weight|≤1（多空可跑），严格版靠 check_tradable 遇负权重 raise 防呆，不在入口禁。
- 涨跌停判据：v2 草稿的 close==high+9.5阈值 → 改用衍生表 limit_status（股票池含 ST/双创/北交所，固定阈值双向出错）。

### [2026-05-31 晚] - fetch_derivative 全量缓存完成（此前因带宽空缺）

**背景:** derivative 是 4 类缓存里唯一空目录（之前白天跳板机限频没拉）。周日 19:05 晚间 off-peak 实跑。

**结果:**
- 全量 1990-2026 共 **37 年成功，0 失败**，`cache/derivative/` 共 **381 MB**。行数从早期年份几百行增长到近年 ~190 万行/年（如 2024=196 万、2025=185 万、2026 当前=70 万）。
- 与 daily 缓存（1990-2026）年份范围完全对齐。

**实测速度（off-peak vs 白天）:**
- 今晚 Q1 2026 = 42 万行 / 18.1s ≈ **23,000 行/秒**，比 agent.md 此前记录的白天 ~1,000 行/秒快约 **23 倍**。全量约 40 分钟跑完（白天估算 15-20 小时）。
- 中段（~2012 起）跳板机限频累积，单季度耗时升到 100-200s，并多次出现 `Error reading SSH protocol banner` / `Could not establish session to SSH gateway`（踩坑 #3 已记录的瞬时现象）。`_fetch_with_retry` 的 3 次重试 + 指数退避全部自愈，无年度失败。

**校验 + 关键数据特征（非交易日行，已查清=数据特征不是错误）:**
- `validate_derivative_vs_calendar` 通过（warning 不阻断）。每年都报大量"非交易日日期"，规模占该年 26-34%（如 2024 共 66.5 万行 / 33.9%）。
- 深挖 2024（366 个唯一日期=覆盖每个自然日）：非交易日行里 `limit_status`/`extremum_status`/`turnover` **100% 全 NaN**，而 `total_mv`/`float_mv`（市值）**100% 有值**。落在工作日的那批非交易日正是法定节假日（元旦/春节/清明/五一/端午/中秋）。
- **结论:** `ashareeodderivativeindicator` 对**每个自然日**都有行——因为市值 = 股本 × 最近收盘价，每天都有定义，Wind 在周末/节假日照报市值；而涨跌停/换手/极值这类交易行为字段在非交易日本就是 NaN。这与 `ashareeodprices`（只在交易日有行，校验全清）的本质区别。
- **处理:** 维持现有设计"存原始、下游按日历 reindex 时自然过滤"——回测引擎对齐交易日历时这 ~30% 纯市值行自动掉出，且掉出的行交易字段本就 NaN，无信息损失。**未改 fetch 逻辑**（是否落盘前先按日历过滤掉这批行属另一设计选择，待 Fecus 定）。

**`Password is required for key` 噪声:** 每次连接 paramiko 都打这行 ERROR（尝试加载本地加密私钥 id_ed25519），但脚本走密码认证、查询正常成功，非致命，与 price/calendar/index 三脚本同源。

### [2026-05-31] - fetch_index_eod.py 基准指数日行情（脚手架对齐 + 接入 fetch_all）

**背景:** 要加基准对比（上证综指等）。先探针确认 aindexeodprices 有数据（8 个基准覆盖到 2026-05-29，见"基准指数数据"小节真值表），再写拉取脚本。

**新增:**
- `data/fetch_index_eod.py`：按指数分批拉 aindexeodprices → `cache/index_eod/{code}.parquet`，复用 `db_config.make_tunnel`。
  - 列：code/date/pre_close/open/high/low/close/change/pct_change/volume/amount（**无复权列**，指数不复权）。
  - validate 3 项（表内，[k/3] 风格）：code/date/close 无 NaN / 无重复(code,date) / close>0；日历交叉校验上移 fetch_all。
- `fetch_all.py`：import fetch_index_eod + 主流程 Step 5/6（交叉校验顺延为 Step 6/6）+ `validate_index_eod_vs_calendar` + 顶部 docstring 同步。

**风格对齐（本次重点，应"和其他脚本对齐便于阅读"要求）:**
- 段落注释头 `# ── 类型转换 ──` / `# ── 校验 ──` / `# ── 主流程 ──` 补齐，与 price_daily/index_members 一致。
- `FLOAT_COLS` 提为模块级常量；`convert_types` 用它 + `.astype("float64")`，与 price_daily 同写法。
- `fetch_and_save_all` 存盘打印改 `→ 已存: {name} (...)`、加 `time.sleep(3)`、收尾 `{'='*60}` + "完成: N 成功, N 失败" + "（重新运行脚本自动补拉）"，与 index_members 一致。
- `validate` 改 `[1/3]/[2/3]/[3/3]` + "校验 N 行数据..." 头 / "校验全部通过。" 尾。
- `main` 去掉 `--force`（与 index_members 对齐，全量清空走 fetch_all --force），结尾 `print("完成。")`。

**验证（仅静态，未跑 DB —— 按用户"先别提取"）:** `py_compile` 两文件通过；import fetch_index_eod / fetch_all 正常，validator 已挂、INDICES=8、函数齐全。

**已落盘:** 仅 000001.SH（早前烟测，8650 行，1990-12-19~2026-05-29）。**其余 7 个基准未拉**（用户要求先别提取）。跑全量：`python fetch_index_eod.py`（8 个，每个 1 SSH 连接，白天跳板机限频注意）。

**副本提醒（纪律1）:** fetch_one_X 的"SSH+重试"循环现已是第 5 份副本（price_daily/calendar/derivative/index_members/index_eod）。项目刻意走"各脚本自包含解耦"风格，本次按用户"先别提取/只对齐"要求**未抽公共函数**；若要治理，可抽 `db_config.fetch_with_retry(sql, read_timeout)` 一次性收口 5 处，需你确认。

### [2026-05-29] - 阶段2 验收测试 + 多空支持确认

**新增:**
- `engine/test_backtest.py`：阶段2 验收测试套件，8 个测试全过（torch1010）。统一行情 A[10,11,12,12]/B[20,18,18,9]，4 个交易日，所有预期 NAV 手算可核对：
  - 核心正确性 5 个：单股满仓(NAV=A/10)、单股半仓((NAV-1)恰好减半)、调仓时序(调仓日 d2 仍 1.2 用旧权重 A，次日 d3 切 B 到 0.6)、权重漂移(50/50→55/45)、多空(多A空B，NAV 1.1 > 纯多头 1.05)
  - 校验 3 个：未知 code / Σ|weight|>1 / start_date·end_date 越界 均正确 raise

**确认（无需改引擎）:**
- **多空已被现有引擎天然支持**：约束是 `Σ|weight| ≤ 1.0`（用 `.abs()`），`update_weights` 的 `cash = 1 - Σweight` 与漂移公式对负权重正确，主循环 `Σ(wᵢ·rᵢ)` 与符号无关。没有任何地方拒绝负权重，故未加多空代码路径（避免过度设计）。仅在 `run_backtest` docstring 注明"weight 可正可负，负=做空"。

**踩坑:**
- 单行 weights_df 时 `end_date` 默认 = 调仓日当天，回测只跑 1 天 → NAV 序列只有 D0。测试需显式传 `end_date` 才能看完整轨迹。这是引擎既定默认行为，非 bug。

### [2026-05-29] - engine/backtest.py 校验加入"调仓日必须是交易日"

**新增:**
- `run_backtest` 输入校验新增第 2 条规则：weights_df 的所有调仓日必须 ∈ price_df 的交易日集合（`set(price_df["date"].unique())`），不在则 raise，错误信息列出非交易日样本（前 10 个）。落实 agent.md 关键设计决策 1 中"weights_df 中的每个 date 必须是交易日，否则 raise"。
- 原 Σ|weight| / start_date / end_date 规则编号顺延（2→3、3→4、4→5），docstring 校验规则同步更新。

**自检:** 正常 case（调仓日为交易日）NAV 正常输出；含非交易日 `2023-01-06` 的 weights_df 正确 raise ValueError。

### [2026-05-28 后续] - 改回扁平 + `data/` 收纳

**理由:** 按 fetch 拆 package 后，未来加 `engine/` 时两个布局不对称（一个深一个浅）。改为整个数据层放一个 `data/` 下，扁平 + 集中 cache，与 `engine/` 平级。

**最终布局:**
```
backtest/
├── data/                       数据层
│   ├── cache/{calendar|daily|derivative|index_members}/
│   ├── fetch_calendar.py
│   ├── fetch_price_daily.py
│   ├── fetch_derivative.py
│   ├── fetch_index_members.py
│   └── fetch_all.py
├── engine/                     回测引擎（阶段 2）
└── test_db_connection.py
```

**改动:**
- 各 fetch.py 的 `CACHE_DIR` / `CACHE_PATH` 改回 `Path(__file__).parent / "cache" / X`
- fetch_all.py: imports 改回扁平 `import fetch_calendar` 等；`CACHE_ROOT` 统一指向 `data/cache/`
- 烟雾测试通过：3 个 validator 跑 OK，1993 脏数据 warning 仍正确

### [2026-05-28] - 目录重构 + 彻底解耦

**布局变化:** 每个 fetch 自成 package（代码 + data 同居）
```
backtest/
├── fetch_calendar/        { fetch.py, __init__.py, data/ }
├── fetch_price_daily/     { fetch.py, __init__.py, data/ }
├── fetch_derivative/      { fetch.py, __init__.py, data/ }
├── fetch_index_members/   { fetch.py, __init__.py, data/ }
├── fetch_all.py
└── test_db_connection.py
```
旧 `cache/` 目录已删除（parquets 全部迁到对应 `fetch_X/data/`，0 数据丢失）。

**解耦补完:**
- `fetch_index_members/fetch.py`：删 `from fetch_calendar import ...` + `_load_trade_dates` + validate [4/4]，validate 改为 3 项
- `fetch_all.py` 新增 `validate_derivative_vs_calendar` 和 `validate_index_dates_vs_calendar`，原 3 个 wrapper 复用私有 `_validate_dates_vs_calendar(data_dir, date_cols, label)` helper
- 所有 fetch_X 模块互不 import；fetch_all 用 `from fetch_X import fetch as fetch_X` 拿到模块

**烟雾测试（重构后）:**
- imports 正常
- validate_price: 仍正确捕获 1993.parquet 的 2 行 Wind 脏数据（warning）
- validate_derivative: 缓存为空，gracefully skip
- validate_index_dates: 9 个文件全过

### [2026-05-28] - fetch_all 编排 + 解耦日历交叉校验

**目标:** 让 4 个 fetch 脚本互相独立可跑，日历交叉校验集中到 fetch_all。

**改动:**
- 新增 `fetch_all.py`：
  - `fetch_all(force=False)`: 顺序跑 1→4，每步打印分隔条；--force 时清空各 cache 后重拉
  - `validate_price_vs_calendar()`: 加载 cache/daily/*.parquet 全部年份，date 列对照 calendar 缓存做 isin 检查
- `fetch_price_daily.py`：删 `from fetch_calendar import ...` + `lru_cache` + `_load_trade_dates` + validate [6/6] 块；validate 改为 5 项
- `fetch_derivative.py`：同上；validate 改为 3 项
- `fetch_index_members.py`：**保留**对日历的依赖（entry/exit 日期 cross-check 留在脚本内，语义和 daily 不同）

**烟雾测试发现的真实脏数据:**
- `1993.parquet` 含 2 行 `1993-01-03`（**周日**）的"交易"记录（000002.SZ 万科 A、000007.SZ）
- Wind 历史数据 bug，不是脚本问题
- 之前 fetch_price_daily 的 [6/6] 加上之前 1993.parquet 就已经落盘了，所以一直没暴露
- 现在 fetch_all step 5 集中校验时被发现 — **这是把校验抽出来的价值**
- 处理待定：用户选择 A=warning / B=重拉 / C=落盘前过滤

### [2026-05-28] - fetch_derivative 脚手架 + 实现（实跑暂缓）

**为什么需要这个表:**
- `ashareeodderivativeindicator.up_down_limit_status` 是 Wind 已按板块规则消化好的涨跌停标记（-1=跌停, 0=正常, 1=涨停），ST 5% / 主板 10% / 创业板/科创板/北交所 20% 全自动识别，省掉下游自己写分支
- 同表还有 `lowest_highest_status`、换手率、市值等

**实现:**
- `fetch_derivative.py` 结构对齐 `fetch_price_daily.py`，外加内部按季度分 4 段（详见下文踩坑）
- 字段：`limit_status`, `extremum_status`, `turnover`, `free_turnover`, `total_mv`, `float_mv`
- `convert_types`: decimal(20,4)→float64, decimal(2,0)→Int8 nullable
- `validate` 4 项：无重复 (date,code) / status ∈ {-1,0,1,NaN} / 数值非负 / 日期 ∈ 日历

**踩坑（实跑前的诊断，标记给晚上自己看）:**
1. **服务器 5 分钟 query timeout** — 单年 ~130 万行的 SELECT 在传输 5 分钟时被服务器主动断（`Lost connection to MySQL server during query`）。所以 `fetch_one_year` 改成内部按季度分 4 段，每段 ~32 万行，单段查询 < 5 分钟。外部 API 不变（仍是 1 parquet/年）
2. **白天跳板机带宽限制** — 实测 SELECT 8 列 165K 行 / 月 = **156 秒**，约 **1,000 行/秒**，比 fetch_price_daily 当时（~100K 行/秒）慢 100 倍。全量 2778 万行按白天速度需 ~15-20 小时
3. **跳板机连接限频** — 连续多次新建 SSH 连接会被跳板机临时拒绝（`Error reading SSH protocol banner` / `Could not connect to gateway`）。已经知道这是白天行为，晚上少
4. **表的索引设计** — DESCRIBE 显示有 `idx_tdt`（单列 trade_dt 索引）和复合索引，COUNT 单月 0.5 秒，索引完全没问题。慢的是传输不是查询

**下一步:**
- 晚上跑 `python fetch_derivative.py`，全量 1990-2026，按季度分批
- 若仍慢，按需切到按月（改 `QUARTERS` 常量即可）
- 完成后更新 agent.md 把状态从 🟡 待跑改为 ✅ 完成

### [2026-05-28] - fetch_index_members 实现 + 全量拉取

**实查的 aindexmembers schema:**
- 8 列：`object_id` / `s_info_windcode` / `s_con_windcode` / `s_con_indate` / `s_con_outdate` / `cur_sign` / `opdate` / `opmode`
- 987,448 行，2,614 个唯一指数
- 区间表（intervals），不是日快照。同一只股票剔出再纳入会产生多段
- "T 日成分股"必须用区间过滤：`entry_date <= T < exit_date (或 exit_date IS NULL)`

**已拉取的 9 个指数:**

| 代码 | 名称 | 行数 | 数据起 | 当前成分股 |
|---|---|---|---|---|
| 000016.SH | 上证50 | 279 | 2004-01-02 | 50 |
| 000300.SH | 沪深300 | 1,208 | 2005-04-08 | 301 |
| 000905.SH | 中证500 | 2,418 | 2007-01-15 | 500 |
| 000906.SH | 中证800 | 2,805 | 2007-01-15 | 801 |
| 000852.SH | 中证1000 | 3,339 | 2014-10-17 | 1,000 |
| 932000.CSI | 中证2000 | 3,069 | **2023-08-11** | 2,000 |
| 399303.SZ | 国证2000 | 5,595 | **2014-03-28** | 2,003 |
| 399006.SZ | 创业板指 | 431 | 2010-06-01 | 100 |
| 000688.SH | 科创50 | 128 | 2019-12-31 | 50 |

**实现:**
- `convert_types`: varchar(8) → datetime；含 Wind 数据冗余去重（按 opdate 最新保留）；输出保持 4 列
- `validate`: 4 项（NaN / 重复 / exit>entry / 日期 ∈ 日历）
- SQL 多拉 `cur_sign` + `opdate` 用于内部去重，不进 parquet

**踩坑（Wind 数据冗余）:**
- 国证2000 / 605077.SH / 2023-06-06 在 DB 里有 2 行：
  - 行1: `outdate=NULL, cur_sign=1, opdate=2023-06-05`（入选当天录入）
  - 行2: `outdate=20240614, cur_sign=0, opdate=2024-06-14`（剔除时插入新行，未清理行1）
- 9 个指数中只有国证2000 有这一例
- 处理：`convert_types` 内按 opdate 排序后 `drop_duplicates(keep="last")`，丢弃过时的"在册"行

### [2026-05-27] - fetch_index_members 脚手架

**新增:**
- `fetch_index_members.py` 脚手架，结构对齐 `fetch_price_daily.py`：
  - `INDICES` 占位常量（沪深300/中证500/1000/800），按需调整
  - `SQL_COLUMNS` / `RENAME_MAP` 用 Wind 命名约定（`s_info_windcode` / `s_con_windcode` / `s_con_indate` / `s_con_outdate`），实跑前需 `DESCRIBE aindexmembers` 核对
  - `fetch_one_index(index_code)`: 已实现（SSH 隧道 + 重试 + rename）
  - `fetch_and_save_all(indices)`: 已实现（cache skip + 失败跳过继续 + 拉一个存一个）
  - `convert_types` / `validate`: `NotImplementedError` 待实现
  - `_load_trade_dates()` 已接入，validate 实现时直接调用做日期交叉验证
  - `main` + argparse 支持 `--indices`

**待办:**
1. 核对 aindexmembers 实际列名（用 `test_db_connection.py` 跑 DESCRIBE 即可）
2. 实现 `convert_types`：entry_date / exit_date varchar → datetime（exit_date 可空）
3. 实现 `validate` 4 项检查：NaN / 重复 / exit>entry / 日期 ∈ 日历

### [2026-05-27] - fetch_price_daily 加入与日历的交叉验证

**新增:**
- 顶部 `from fetch_calendar import CACHE_PATH as CALENDAR_CACHE_PATH`（单一真相源）
- `_load_trade_dates()` 用 `@lru_cache(maxsize=1)` 懒加载日历缓存，缺失则报错提示先跑 `fetch_calendar.py`
- `validate` 加 [6/6] 项：所有日期 ∈ 交易日历，不在则 raise 并列出离群日期样本
- 原 [1/5]…[5/5] 全部改为 /6 编号
- 顶部 docstring 加"前置"块，明示日历缓存依赖
- agent.md 加"脚本运行顺序 / 前置依赖"小节，记录硬依赖

**自检:**
- 2026.parquet (505,303 行): 6/6 全过
- 1990.parquet (72 行, 边界): 6/6 全过

### [2026-05-27] - fetch_calendar 实现 convert_types / validate

**新增:**
- `convert_types`: `trade_days` varchar(8) → `date` datetime
- `validate` 5 项检查（fail-fast）：
  1. date 无 NaN
  2. 无重复
  3. 严格升序
  4. 不含周末
  5. 每年交易日 ∈ `[235, 260]`，首尾不完整年自动跳过
- 离线自检（已有缓存 12,263 行）全部通过

**踩坑:**
- 用户最初要求 `[240, 250]`，跑出 10 个真实离群年份：1991-1995（255/257/252/251，早期少法定假日）、1999/2000/2002/2013（237-239，春节调整）、2032（251，未来排期）。step 4 已确认周末不在表内，所以离群是节假日结构差异，不是数据错误。阈值放宽到 `[235, 260]` 作 sanity 量级检查。

### [2026-05-27] - fetch_calendar 脚手架对齐 fetch_price_daily

**修改:**
- `fetch_calendar.py` 脚手架重组（暂留 `convert_types`/`validate` 为 `NotImplementedError`）：
  - 顶部 docstring 加用法说明
  - `fetch` 在重试块末尾内联 rename，返回 rename 后的 DataFrame（与 `fetch_one_year` 对齐）
  - 原 `transform(raw, rename_map)` 改名 `convert_types(df)`，签名去掉 rename_map
  - `validate` 返回类型由 `DataFrame` 改为 `None`（raise on error 模式）
  - 删除独立的 `save` 函数，主流程直接 `df.to_parquet`
  - 新增 `fetch_and_save(force)` 主流程：cache skip + fetch → convert → validate → save
  - 新增 `main()` + argparse 支持 `--force`

### [2026-05-27] - 交易日历 ETL（初版）

**新增:**
- `fetch_calendar.py`: 交易日历拉取，四步管线模式
  - `fetch(sql)` → raw DataFrame
  - `transform(raw, rename_map)` → renamed + 类型转好的 DataFrame
  - `validate(df)` → 校验通过原样返回
  - `save(df, path)` → parquet
- 结果: 12,263 个交易日 (1990-12-19 ~ 2040-12-31), 110.3 KB

### [2026-05-27] - 阶段1 全量数据拉取完成

**结果:**
- 37 个 parquet 文件（1990-2026），共 622 MB，1850 万行全部通过校验

**踩坑记录（8 个问题）:**

1. **交叉校验精度问题** — `close × adj_factor ≈ adj_close` 最初设 0.01% 阈值，22万行报错。原因：`Decimal(20,4)` 只保留 4 位小数，低价股（几毛钱）四舍五入后相对误差大。实际 p50=0.003%, p99=0.07%。**处理**: 阈值改为 >2% 报错，0.5%~2% 只 warning

2. **1993 年历史脏数据** — 全量拉取后 12 行 >2% 严重偏差，全来自 1993-12-23，adj_factor 和 adj_close 对不上。是数据源遗留问题。**处理**: >2% 偏差从 raise 降级为 warning

3. **SSH 隧道不稳定** — 全量 1850 万行单次查询容易断连 `Lost connection to MySQL server during query`；连续多次连接会被跳板机拒绝 `Connection reset by peer`。**处理**: 改为按年分批拉取，每年独立 SSH 连接 + 3 次重试 + 指数退避 + 年间间隔 3s

4. **拉了数据没存盘就丢了** — 最初设计全量拉完 → 统一校验 → 统一存盘，中途报错前面数据全丢。**处理**: 改为拉一年存一年，每年独立走 fetch → convert → validate → save

5. **`s_dq_adjopen` 字段存在** — 数据字典没重点提，连通性测试时发现。数据库直接提供复权开盘价，调仓日 PnL 跨日计算不需要自己算 `open × adj_factor`

6. **交易日历覆盖到 2040 年** — 之前数据字典说"只到 2026-01-04"是误读（那是 `create_time`），`trade_days` 字段最大值 `20401231`，不需要兜底方案

7. **`s_dq_turn`（换手率）全是 None** — `ashareeodprices` 表没有换手率数据，需要的话去 `ashareeodderivativeindicator` 表拿

8. **Python 3.8 不支持 `str | None`** — torch1010 环境是 Python 3.8，需用 `Optional[str]`

### [2026-05-26] - 阶段1 启动 + 连通性测试

**新增:**
- `fetch_price_daily.py`: 全量拉取脚本
- `test_db_connection.py`: SSH 隧道连通性测试脚本
- 缓存目录 `cache/daily/`，按年存 parquet（如 `2023.parquet`）

### [2026-05-26] - 项目启动

**新增:**
- 创建 agent.md，记录架构设计、数据源信息、字段映射
- 分析 GeneralBacktest 框架，提取有价值的设计点
- 梳理 SYYL/fusion_prod 数据库中回测相关核心表
- 确定标准 price_df / weights_df 数据格式
- 识别关键风险点（交易日历缺失、数据源不稳定）

---

## 🚧 下一步 TODO

1. ~~**阶段0**: 画数据流图~~ ✅ 已完成（roadmap v2 + agent.md）
2. ~~**阶段1 部分**: 全量数据拉取~~ ✅ fetch_price_daily.py 已完成
3. ~~**阶段1 部分**: 交易日历拉取~~ ✅ fetch_calendar.py 已完成
4. **阶段1 剩余**: 基准指数数据拉取
4. **阶段2**: 最小回测引擎 — weights_df + price_df → NAV（无摩擦）
5. **阶段3**: 调仓执行管线 — 加入手续费/滑点/调仓阈值
