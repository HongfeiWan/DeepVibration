#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2号传感器：温度 & 振动位移 联合绘图脚本

功能：
- 复用 temperature/select.py 和 displacement/select.py 的读取/筛选函数
- 只取 detector 2 的数据
- 将温度和一个方向的位移（默认 Displacement_z，可改）画在同一张图上：
  - 上方子图：Temperature (°C) vs Time
  - 下方子图：Displacement_z (μm) vs Time
- 保持与项目其他脚本一致的科研绘图风格
"""

import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

# ----------------------------------------------------------------------
# 路径设置：把 DeepVibration/python 加入 sys.path，方便直接 python 运行本脚本
# 当前文件：.../python/data/sensor/vibration/temperature&displacement/temperature&sensor2.py
# 向上 4 层到 .../python
# ----------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
)
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

# 复用温度与位移的筛选函数
from data.sensor.vibration.temperature.select import (  # type: ignore
    select_by_date_range_vibration as select_temp_by_date_range,
)
from data.sensor.vibration.displacement.select import (  # type: ignore
    select_by_date_range_vibration as select_disp_by_date_range,
)


def _align_by_datetime(
    data_temp: Dict[str, np.ndarray],
    data_disp: Dict[str, np.ndarray],
    disp_axis: str = "Displacement_y",
) -> Dict[str, np.ndarray]:
    """
    将温度与位移按时间对齐（内连接），只保留两者都有数据的时间点。

    返回：
        dict: {'datetime', 'Temperature', disp_axis}
    """
    if "datetime" not in data_temp or "datetime" not in data_disp:
        raise ValueError("温度和位移数据中都必须包含 'datetime' 列")

    if "Temperature" not in data_temp:
        raise ValueError("温度数据中必须包含 'Temperature' 列")

    if disp_axis not in data_disp:
        raise ValueError(f"位移数据中必须包含 '{disp_axis}' 列")

    dt_temp = data_temp["datetime"]
    dt_disp = data_disp["datetime"]

    # 转成 pandas.Series 方便 merge，对齐精度按原来的 datetime 字符串
    import pandas as pd

    df_temp = pd.DataFrame({"datetime": pd.to_datetime(dt_temp), "Temperature": data_temp["Temperature"]})
    df_disp = pd.DataFrame({"datetime": pd.to_datetime(dt_disp), disp_axis: data_disp[disp_axis]})

    # 以时间为键做内连接
    df_merged = pd.merge(df_temp, df_disp, on="datetime", how="inner")
    if df_merged.empty:
        raise ValueError("温度与位移在时间上没有交集，无法对齐数据")

    df_merged = df_merged.sort_values("datetime").reset_index(drop=True)

    return {
        "datetime": df_merged["datetime"].values,
        "Temperature": df_merged["Temperature"].values.astype(float),
        disp_axis: df_merged[disp_axis].values.astype(float),
    }


def plot_temp_and_displacement_sensor2(
    data_temp: Dict[str, np.ndarray],
    data_disp: Dict[str, np.ndarray],
    disp_axis: str = "Displacement_y",
    save_path: Optional[str] = None,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """
    将 2 号传感器的温度与某一方向位移画在同一张图中，共用一个 x 轴，左右两个 y 轴：
    - 左 y 轴：Temperature (°C)
    - 右 y 轴：位移（默认 Displacement_y，μm）

    参数:
        data_temp: 温度数据字典（temperature/select.py 返回）
        data_disp: 位移数据字典（displacement/select.py 返回）
        disp_axis: 位移方向列名，默认 'Displacement_y'
        save_path: 图片保存路径；为 None 时不保存
        show_plot: 是否显示图片
        figsize: 图像大小
    """
    # 只保留 detector 2 的数据
    for d in (data_temp, data_disp):
        if "detector_num" in d:
            mask = d["detector_num"] == 2
            for k in list(d.keys()):
                d[k] = d[k][mask]

    # 对齐时间
    merged = _align_by_datetime(data_temp, data_disp, disp_axis=disp_axis)
    dt = merged["datetime"]
    temp = merged["Temperature"]
    disp = merged[disp_axis]

    # 过滤 NaN
    valid_mask = (~np.isnan(temp)) & (~np.isnan(disp))
    dt = dt[valid_mask]
    temp = temp[valid_mask]
    disp = disp[valid_mask]

    if dt.size == 0:
        raise ValueError("对齐并过滤 NaN 后没有有效数据点")

    # 科研绘图风格，与其他脚本保持一致
    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "axes.linewidth": 1.2,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.minor.width": 0.8,
            "ytick.minor.width": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.fontsize": 10,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "figure.dpi": 100,
        }
    )

    # 一个 axes，左右两个 y 轴
    fig, ax1 = plt.subplots(1, 1, figsize=figsize)
    ax2 = ax1.twinx()

    # 左 y 轴：温度
    ax1.plot(
        dt,
        temp,
        linewidth=1.5,
        alpha=0.85,
        color="#2E86AB",
        label="Temperature (Sensor 2)",
    )
    ax1.set_ylabel("Temperature (°C)")
    ax1.legend(loc="upper right", framealpha=0.9, edgecolor="gray", frameon=True, fancybox=False, shadow=False)
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax1.yaxis.get_major_formatter().set_scientific(False)
    ax1.grid(True, which="major", linestyle="-", linewidth=0.7, alpha=0.3, color="gray")
    ax1.grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.2, color="gray")
    for spine in ["top", "bottom", "left"]:
        ax1.spines[spine].set_visible(True)
        ax1.spines[spine].set_color("gray" if spine == "top" else "black")

    # 右 y 轴：位移
    ax2.plot(
        dt,
        disp,
        linewidth=1.5,
        alpha=0.85,
        color="#A23B72",
        label=f"{disp_axis} (Sensor 2)",
    )
    ax2.set_ylabel(f"{disp_axis} (μm)")
    ax1.set_xlabel("Time")
    ax2.legend(loc="upper left", framealpha=0.9, edgecolor="gray", frameon=True, fancybox=False, shadow=False)
    ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax2.yaxis.get_major_formatter().set_scientific(False)
    # 右轴只保留边框，不再单独画网格，避免太乱
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(True)
        ax2.spines[spine].set_color("gray")

    # x 轴时间格式自动处理
    time_span = dt[-1] - dt[0]
    if hasattr(time_span, "days"):
        days = time_span.days
        total_hours = time_span.total_seconds() / 3600
    else:
        days = time_span.astype("timedelta64[D]").astype(int)
        total_hours = time_span.astype("timedelta64[h]").astype(int)

    if days > 30:
        date_format = "%Y-%m-%d"
        locator = mdates.DayLocator(interval=max(1, days // 10))
        minor_locator = mdates.HourLocator(interval=6)
    elif days > 1:
        date_format = "%m-%d %H:%M"
        major_interval = max(1, int(total_hours / 10))
        locator = mdates.HourLocator(interval=major_interval)
        minor_locator = mdates.MinuteLocator(interval=30)
    else:
        date_format = "%H:%M"
        major_interval = max(1, int(total_hours / 8))
        locator = mdates.HourLocator(interval=major_interval)
        minor_locator = mdates.MinuteLocator(interval=15)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_minor_locator(minor_locator)

    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"图像已保存至: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    """
    示例：
    读取 2 号传感器在给定日期范围内的温度和位移数据，并画在同一张图上。
    """
    # 获取项目根目录
    # 从 python/data/sensor/vibration/temperature&displacement/temperature&sensor2.py 向上 5 层
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
    )
    data_dir = os.path.join(project_root, "data", "vibration", "hdf5")

    print("=" * 70)
    print("Sensor 2: Temperature & Displacement joint plot")
    print("=" * 70)

    # 日期和降采样参数可按需修改
    start_date = "2025-05-28"
    end_date = "2025-06-10"
    downsample_factor = 100

    try:
        print("\n读取 2 号传感器温度数据...")
        data_temp = select_temp_by_date_range(
            data_dir,
            detector_num=2,
            start_date=start_date,
            end_date=end_date,
            downsample_factor=downsample_factor,
        )

        print("\n读取 2 号传感器位移数据...")
        data_disp = select_disp_by_date_range(
            data_dir,
            detector_num=2,
            start_date=start_date,
            end_date=end_date,
            downsample_factor=downsample_factor,
        )

        if not data_temp or not data_disp:
            print("未能读取到温度或位移数据，无法绘图")
        else:
            out_dir = os.path.join(project_root, "imgaes", "temperature_displacement")
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, "sensor2_temp_displacement_y.png")

            print("\n开始绘制 2 号传感器温度与位移联合图...")
            plot_temp_and_displacement_sensor2(
                data_temp,
                data_disp,
                disp_axis="Displacement_y",
                save_path=save_path,
                show_plot=True,
                figsize=(12, 6),
            )

    except Exception as e:
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()

