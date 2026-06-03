"""画图共用样式：中文字体的跨平台 fallback（同一份代码在 Linux / macOS / Windows 都能出中文）。

被 report/plot.py 与 factor/plot_factor.py 复用——字体列表只在这里写一份，不在两处各写还会漂移。
原理：matplotlib 的 `font.sans-serif` 是优先级列表，按序取第一个「系统已装」的字体；把三平台常见
中文字体都列上，在哪个系统就命中哪个。本模块独立于 metrics.py（算层不引入 matplotlib）。
"""
import matplotlib

# 三平台常见中文 sans-serif，按 Win → macOS → Linux → 兜底 排（matplotlib 取第一个已装的）
_CJK_FONTS = [
    "Microsoft YaHei", "SimHei",                                            # Windows（微软雅黑 / 黑体）
    "PingFang SC", "Heiti SC", "STHeiti", "Songti SC", "Arial Unicode MS",  # macOS（苹方 / 黑体 / 华文 / 宋体）
    "Noto Sans CJK SC", "Source Han Sans SC",                               # 通用/Linux（思源黑体两种命名）
    "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Droid Sans Fallback",      # Linux（文泉驿 / Android 回退）
    "DejaVu Sans",                                                          # 最后兜底（无中文，仅防崩）
]


def setup_chinese_font():
    """设全局中文字体 fallback + 负号正常显示（所有图共用，调一次即可）。

    Linux 上若上面三种 CJK 字体一个都没装会显示方框——装一个即可，例如
    Debian/Ubuntu：`apt install fonts-noto-cjk` 或 `apt install fonts-wqy-zenhei`。
    """
    matplotlib.rcParams["font.sans-serif"] = _CJK_FONTS
    matplotlib.rcParams["axes.unicode_minus"] = False
