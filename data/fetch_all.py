"""
一键拉取全部回测数据。

逻辑:
    1. fetch_calendar       — 交易日历
    2. fetch_price_daily    — 日行情
    3. fetch_derivative     — 衍生指标（涨跌停状态等）
    4. fetch_index_members  — 指数成分股
    5. fetch_index_eod      — 基准指数日行情
    6. fetch_description    — 股票上市/退市日（现金引擎判退市用）
    7. 交叉校验:
       a. validate_price_vs_calendar       — 日行情 date 列 ⊆ 日历
       b. validate_derivative_vs_calendar  — 衍生指标 date 列 ⊆ 日历
       c. validate_index_dates_vs_calendar — 指数成分股 entry_date/exit_date ⊆ 日历
       d. validate_index_eod_vs_calendar   — 基准指数 date 列 ⊆ 日历

每一步内部已有"存在就跳过"的逻辑，重跑只会补拉缺失数据。
--force 时清空所有 cache 后从头拉。

用法:
    python fetch_all.py            # 增量补齐（推荐）
    python fetch_all.py --force    # ⚠ 清空所有 cache 后重拉（耗时极长）
"""
import argparse
import shutil
from pathlib import Path
from typing import List

import pandas as pd

import fetch_calendar
import fetch_derivative
import fetch_description
import fetch_index_eod
import fetch_index_members
import fetch_price_daily

CACHE_ROOT = Path(__file__).parent / "cache"
CALENDAR_DATA = CACHE_ROOT / "calendar"
PRICE_DATA = CACHE_ROOT / "daily"
DERIVATIVE_DATA = CACHE_ROOT / "derivative"
INDEX_DATA = CACHE_ROOT / "index_members"
INDEX_EOD_DATA = CACHE_ROOT / "index_eod"


def _load_calendar() -> pd.DatetimeIndex:
    """加载日历，未生成则 raise（fetch_all 在 step 1 之后才会调用，所以应该存在）"""
    cal_path = CALENDAR_DATA / "trade_dates.parquet"
    if not cal_path.exists():
        raise FileNotFoundError(f"日历缓存不存在: {cal_path}")
    cal = pd.read_parquet(cal_path, columns=["date"])
    return pd.DatetimeIndex(cal["date"].unique())


def _validate_dates_vs_calendar(
    data_dir: Path, date_cols: List[str], label: str
) -> None:
    """
    通用日历交叉校验 helper。
    遍历 data_dir 下所有 *.parquet，检查 date_cols 列 ⊆ 交易日历。
    不一致只打 warning 不阻断（Wind 历史脏数据已知，由下游使用方按需过滤）。
    """
    if not data_dir.exists() or not list(data_dir.glob("*.parquet")):
        print(f"  {label}: 缓存为空，跳过")
        return

    trade_dates = _load_calendar()
    files = sorted(data_dir.glob("*.parquet"))
    print(f"  {label}: 日历 {len(trade_dates)} 天 vs {len(files)} 个 parquet")

    bad = []
    for f in files:
        df = pd.read_parquet(f, columns=date_cols)
        total_bad = 0
        bad_dates_all = []
        for col in date_cols:
            mask = df[col].notna() & ~df[col].isin(trade_dates)  # NaT 跳过（如 exit_date）
            if mask.any():
                total_bad += int(mask.sum())
                bad_dates_all.extend(df.loc[mask, col].unique().tolist())
        if total_bad > 0:
            bad.append((f.name, total_bad, sorted(set(bad_dates_all))))

    if bad:
        print(f"  ⚠ {len(bad)} 个文件含非日历日期（不阻断）:")
        for fname, n, dates in bad:
            print(f"    {fname}: {n} 行，日期 {[str(pd.Timestamp(d).date()) for d in dates[:5]]}")
    else:
        print(f"  ✓ {label} 全部 ⊆ 日历")


def validate_price_vs_calendar() -> None:
    """日行情 date 列 ⊆ 日历"""
    _validate_dates_vs_calendar(PRICE_DATA, ["date"], "日行情")


def validate_derivative_vs_calendar() -> None:
    """衍生指标 date 列 ⊆ 日历"""
    _validate_dates_vs_calendar(DERIVATIVE_DATA, ["date"], "衍生指标")


def validate_index_dates_vs_calendar() -> None:
    """指数成分股 entry_date / exit_date ⊆ 日历（exit_date NaT 跳过）"""
    _validate_dates_vs_calendar(INDEX_DATA, ["entry_date", "exit_date"], "指数成分股")


def validate_index_eod_vs_calendar() -> None:
    """基准指数行情 date 列 ⊆ 日历"""
    _validate_dates_vs_calendar(INDEX_EOD_DATA, ["date"], "基准指数行情")


def fetch_all(force: bool = False) -> None:
    """全量数据拉取 orchestrator。force=True 时清空 cache 后重拉。"""
    if force:
        print("⚠️  --force 模式：清空所有 cache 后重拉\n")
        if CACHE_ROOT.exists():
            shutil.rmtree(CACHE_ROOT)
            print(f"  已清空: cache/")
        print()

    print("=" * 60)
    print("Step 1/6: 交易日历 (fetch_calendar)")
    print("=" * 60)
    fetch_calendar.fetch_and_save()

    print("\n" + "=" * 60)
    print("Step 2/6: 日行情 (fetch_price_daily)")
    print("=" * 60)
    fetch_price_daily.fetch_and_save_all()

    print("\n" + "=" * 60)
    print("Step 3/6: 衍生指标 (fetch_derivative)")
    print("=" * 60)
    fetch_derivative.fetch_and_save_all()

    print("\n" + "=" * 60)
    print("Step 4/6: 指数成分股 (fetch_index_members)")
    print("=" * 60)
    fetch_index_members.fetch_and_save_all()

    print("\n" + "=" * 60)
    print("Step 5/7: 基准指数日行情 (fetch_index_eod)")
    print("=" * 60)
    fetch_index_eod.fetch_and_save_all()

    print("\n" + "=" * 60)
    print("Step 6/7: 股票上市/退市日 (fetch_description)")
    print("=" * 60)
    fetch_description.fetch_and_save()

    print("\n" + "=" * 60)
    print("Step 7/7: 交叉校验 (日历)")
    print("=" * 60)
    validate_price_vs_calendar()
    validate_derivative_vs_calendar()
    validate_index_dates_vs_calendar()
    validate_index_eod_vs_calendar()

    print("\n✓ 全部完成")


def main():
    parser = argparse.ArgumentParser(description="一键拉取全部回测数据")
    parser.add_argument("--force", action="store_true", help="⚠ 清空所有 cache 后重拉")
    args = parser.parse_args()
    fetch_all(force=args.force)


if __name__ == "__main__":
    main()
