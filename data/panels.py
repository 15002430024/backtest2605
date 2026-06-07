"""
区间表 → (日期 × code) 布尔面板：指数成分股、ST 共用的纯变换层。

定位：放在数据层底部，engine 和 factor 都从这里取，避免 engine→factor 的反向依赖
（依赖方向：factor→engine、engine→data、factor→data，无环）。不读缓存、无副作用。
"""
import numpy as np
import pandas as pd

_FAR_FUTURE = pd.Timestamp("2100-01-01")  # end 为 NaT（至今仍在）时的右端代理


def intervals_to_panel(intervals, start_col, end_col, dates, codes, end_inclusive) -> pd.DataFrame:
    """区间表 → (日期 × code) 布尔面板。

    code 在日期 D 命中任一区间 [start, end] 则 True；end 为 NaT 视 +∞。
    end_inclusive=True → start≤D≤end（ST）；False → start≤D<end（指数成分，退出日当天不再算）。
    实现：按 code 遍历其区间（区间数千级，非 hot loop）。

    输入:
      intervals — DataFrame，至少含 [code, start_col, end_col]，start/end 为 datetime
      start_col / end_col — 区间起止列名（ST 用 st_start/st_end，成分用 entry_date/exit_date）
      dates — DatetimeIndex，面板的行轴（调仓日 或 全交易日历）
      codes — 面板的列轴（可迭代代码）
      end_inclusive — 右端是否闭区间
    输出: DataFrame(index=dates, columns=codes) bool，缺省 False
    """
    codes = list(codes)
    panel = pd.DataFrame(False, index=dates, columns=codes)
    rd = dates.values.astype("datetime64[ns]")
    code_pos = set(codes)
    iv = intervals[intervals["code"].isin(code_pos)]
    for code, grp in iv.groupby("code"):
        hit = np.zeros(len(dates), dtype=bool)
        for s, e in zip(grp[start_col].values, grp[end_col].values):
            e2 = _FAR_FUTURE.to_datetime64() if pd.isna(e) else e
            if end_inclusive:
                hit |= (rd >= s) & (rd <= e2)
            else:
                hit |= (rd >= s) & (rd < e2)
        panel[code] = hit
    return panel
