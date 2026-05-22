#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境多参量联合绘图：
- 所有数据共享同一个时间轴
- 温度面板：5 个振动传感器温度 + 探测器本体 3 个温度（同一温度标尺、单 y 轴）
- 振动面板：加速度 / 速度 / 位移 / 频率均使用 xyz 矢量模长合成（每面板单 y 轴）
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

from analysis.environment.compressor import select_by_date_range as select_compressor_by_date_range
from analysis.environment.sensors import select_by_date_range_vibration


def _style_rc_params() -> None:
    """统一字体与坐标样式。"""
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "axes.linewidth": 1.2,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.fontsize": 12,
        "figure.dpi": 100,
    })


def _plot_series_single_y(
    ax,
    series_list: Sequence[Dict],
    y_label: str,
    show_grid: bool = False,
    log_y: bool = False,
) -> List:
    """单 y 轴绘制多条曲线。log_y=True 时仅绘制 y>0 的点（对数坐标）。"""
    handles = []
    for series in series_list:
        y = np.asarray(series["y"], dtype=float)
        if log_y:
            y = np.where(np.isfinite(y) & (y > 0), y, np.nan)
        line, = ax.plot(
            series["x"],
            y,
            color=series.get("color", "#2E86AB"),
            linestyle=series.get("linestyle", "-"),
            linewidth=series.get("linewidth", 1.2),
            alpha=series.get("alpha", 0.85),
            marker=series.get("marker", None),
            markersize=series.get("markersize", 4),
            markevery=series.get("markevery", None),
            label=series["label"],
        )
        handles.append(line)

    ax.set_ylabel(y_label)
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10)))
    else:
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
        ax.yaxis.get_major_formatter().set_scientific(False)
    ax.grid(show_grid)
    return handles


def _vector_magnitude_by_detector(
    data_dict: Dict[str, np.ndarray],
    component_cols: Sequence[str],
) -> List[Dict]:
    """
    按 detector_num 分组，对 xyz 分量做矢量模长（L2 范数）合成：
        |v| = sqrt(vx^2 + vy^2 + vz^2)
    缺省分量按 0 参与平方和；若某时刻三分量均为 NaN，则该点结果为 NaN。
    """
    if not data_dict or "datetime" not in data_dict:
        return []

    datetime_arr = data_dict["datetime"]
    if "detector_num" not in data_dict:
        return []

    det_arr = data_dict["detector_num"]
    comps = []
    for col in component_cols:
        comps.append(data_dict.get(col, np.full_like(datetime_arr, np.nan, dtype=float)))

    series_list = []
    sensor_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4749', '#9B5DE5', '#F15BB5']

    for i, det in enumerate(sorted(np.unique(det_arr))):
        mask_det = det_arr == det
        if not np.any(mask_det):
            continue

        t = datetime_arr[mask_det]
        vx = comps[0][mask_det]
        vy = comps[1][mask_det]
        vz = comps[2][mask_det]

        # 正确模长：sqrt(vx^2+vy^2+vz^2)。旧版误用 sqrt(mean(vx^2,vy^2,vz^2))，会系统性偏小。
        sq = np.stack([vx * vx, vy * vy, vz * vz], axis=0)
        mag = np.sqrt(np.nansum(sq, axis=0))
        all_nan = np.isnan(vx) & np.isnan(vy) & np.isnan(vz)
        mag[all_nan] = np.nan
        valid = np.isfinite(mag)
        if not np.any(valid):
            continue

        series_list.append({
            "x": t[valid],
            "y": mag[valid],
            "label": f"Sensor {int(det)}",
            "color": sensor_colors[i % len(sensor_colors)],
            "linestyle": "-",
            "linewidth": 1.2,
        })

    return series_list


def plot_environment_overview(
    vibration_data_dir: str,
    compressor_file_path: str,
    detector_nums: Sequence[int] = (1, 2, 3, 4, 5),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    downsample_factor: int = 1000,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    figsize: Tuple[float, float] = (8.27, 11.69),
) -> None:
    """绘制超长环境总览图。"""
    _style_rc_params()

    # 读取数据
    common_kwargs = dict(
        data_dir=vibration_data_dir,
        detector_num=list(detector_nums),
        start_date=start_date,
        end_date=end_date,
        start_time=start_time,
        end_time=end_time,
        downsample_factor=downsample_factor,
    )
    env_data = select_by_date_range_vibration(**common_kwargs)
    temp_data = env_data
    acc_data = env_data
    vel_data = env_data
    disp_data = env_data
    freq_data = env_data

    compressor_data = select_compressor_by_date_range(
        compressor_file_path,
        start_date=start_date,
        end_date=end_date,
        start_time=start_time,
        end_time=end_time,
    )

    # 统一时间轴格式
    fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True)
    # A4 竖版页面布局（适配 Word 单页）
    plt.subplots_adjust(left=0.12, right=0.97, top=0.98, bottom=0.07, hspace=0.16)

    locator = mdates.DayLocator(interval=1)
    formatter = mdates.DateFormatter("%m-%d")

    legend_handles = []
    legend_labels = []

    sensor_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4749', '#9B5DE5', '#F15BB5']
    compressor_color = '#C73E1D'
    compressor_cols = ['Compressor temp', 'Controller temp', 'Coldhead temp']
    compressor_markers = ['o', 's', '^']

    # 1) 温度面板：5 个传感器温度 + 探测器本身 3 个温度，不要 grid
    temp_series = []
    if temp_data and "detector_num" in temp_data:
        tdt = temp_data["datetime"]
        tval = temp_data["Temperature"]
        tdet = temp_data["detector_num"]
        for i, det in enumerate(sorted(np.unique(tdet))):
            m = (tdet == det) & (~np.isnan(tval))
            if np.any(m):
                temp_series.append({
                    "x": tdt[m],
                    "y": tval[m],
                    "label": f"Sensor {int(det)} (°C)",
                    "color": sensor_colors[i % len(sensor_colors)],
                    "linestyle": "-",
                    "linewidth": 1.2,
                })

    cdt = compressor_data.get("datetime", np.array([]))
    for i, col in enumerate(compressor_cols):
        if col in compressor_data:
            cv = compressor_data[col]
            m = ~np.isnan(cv)
            if np.any(m):
                short = col.replace(" temp", "")
                temp_series.append({
                    "x": cdt[m],
                    "y": cv[m],
                    "label": f"{short} (°C)",
                    "color": compressor_color,
                    "linestyle": "-",
                    "linewidth": 1.1,
                    "marker": compressor_markers[i % len(compressor_markers)],
                    "markevery": max(1, int(np.sum(m) // 200)),
                    "markersize": 5,
                })

    lines = _plot_series_single_y(axes[0], temp_series, "Temperature (°C)", show_grid=False)
    legend_handles.extend(lines)
    legend_labels.extend([s["label"] for s in temp_series])

    # 2) 加速度面板（xyz 矢量合成）
    acc_series = _vector_magnitude_by_detector(
        acc_data,
        ["Acceleration_x", "Acceleration_y", "Acceleration_z"],
    )
    lines = _plot_series_single_y(axes[1], acc_series, "Acceleration (g)", show_grid=False)
    legend_handles.extend(lines)
    legend_labels.extend([s["label"] for s in acc_series])

    # 3) 速度面板（xyz 矢量合成）
    vel_series = _vector_magnitude_by_detector(
        vel_data,
        ["Velocity_x", "Velocity_y", "Velocity_z"],
    )
    lines = _plot_series_single_y(axes[2], vel_series, "Velocity (mm/s)", show_grid=False, log_y=True)
    legend_handles.extend(lines)
    legend_labels.extend([s["label"] for s in vel_series])

    # 4) 位移面板（xyz 矢量合成）
    disp_series = _vector_magnitude_by_detector(
        disp_data,
        ["Displacement_x", "Displacement_y", "Displacement_z"],
    )
    lines = _plot_series_single_y(axes[3], disp_series, "Displacement (μm)", show_grid=False, log_y=True)
    legend_handles.extend(lines)
    legend_labels.extend([s["label"] for s in disp_series])

    # 5) 频率面板（xyz 矢量合成）
    freq_series = _vector_magnitude_by_detector(
        freq_data,
        ["Frequency_x", "Frequency_y", "Frequency_z"],
    )
    lines = _plot_series_single_y(axes[4], freq_series, "Frequency (Hz)", show_grid=False, log_y=True)
    legend_handles.extend(lines)
    legend_labels.extend([s["label"] for s in freq_series])

    # 公共 x 轴
    for ax in axes:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(top=False, right=False, labelsize=12)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.margins(x=0)
    axes[-1].set_xlabel("Time", fontsize=16)

    # 时间轴裁切到“加速度有数据”的最小/最大时间，去掉左右留白
    acc_time_series = [s["x"] for s in acc_series if len(s["x"]) > 0]
    if acc_time_series:
        x_min = min(np.min(x) for x in acc_time_series)
        x_max = max(np.max(x) for x in acc_time_series)
        for ax in axes:
            ax.set_xlim(x_min, x_max)

    # 图例只保留一份总图例（颜色绑定 Sensor x，不重复变量名）
    unique = OrderedDict()
    for h, l in zip(legend_handles, legend_labels):
        if "Sensor" in l:
            key = l.split()[0] + " " + l.split()[1]
        else:
            key = l
        if key not in unique:
            unique[key] = h

    axes[0].legend(
        list(unique.values()),
        list(unique.keys()),
        loc="upper right",
        fontsize=12,
        frameon=True,
        framealpha=0.9,
        edgecolor="gray",
        ncol=1,
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"图片已保存至: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    python_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(python_dir)

    vibration_data_dir = os.path.join(project_root, "data", "vibration", "hdf5")
    compressor_file_path = os.path.join(project_root, "data", "compressor", "txt", "EC1CP5.txt")
    output_path = os.path.join(project_root, "images", "presentation", "environment_overview.png")

    plot_environment_overview(
        vibration_data_dir=vibration_data_dir,
        compressor_file_path=compressor_file_path,
        detector_nums=(1, 2, 3, 4, 5),
        start_date="2025-05-28",
        end_date="2025-06-10",
        downsample_factor=1000,
        save_path=output_path,
        show_plot=True,
    )
