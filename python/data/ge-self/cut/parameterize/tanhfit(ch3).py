#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CH3 波形的 tanh 拟合可视化脚本。

功能：
- 直接读取 HDF5 原始数据中的 CH3 通道；
- 对指定事件进行波形平滑后，使用 tanh 函数拟合前沿；
- 绘制波形与拟合曲线。
"""

import os
import sys
from typing import Optional, List
from datetime import datetime

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

current_dir = os.path.dirname(os.path.abspath(__file__))
cut_dir = os.path.dirname(current_dir)

def _resolve_ch0_3_file(ch0_3_file: Optional[str]) -> str:
    """当 ch0_3_file 为 None 时，自动查找 CH0-3 文件。"""
    if ch0_3_file is not None:
        return ch0_3_file
    ge_self_dir = os.path.dirname(cut_dir)
    data_dir = os.path.dirname(ge_self_dir)
    python_dir = os.path.dirname(data_dir)
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)
    from utils.visualize import get_h5_files
    h5_files = get_h5_files()
    if "CH0-3" not in h5_files or not h5_files["CH0-3"]:
        raise FileNotFoundError("在 data/hdf5/raw_pulse/CH0-3 目录中未找到 h5 文件")
    return h5_files["CH0-3"][0]

def _visualize_single_event(
    ch0_3_file: str,
    event_index: int,
    channel_idx: int = 3,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    smooth_window: int = 5,
    smooth_times: int = 25,) -> str:
    """
    可视化单个事件的 CH3 波形并进行 tanh 前沿拟合。

    参数：
        ch0_3_file: CH0-3 文件路径
        event_index: 事件索引
        channel_idx: 通道索引（默认 3，对应 CH3）
        save_path: 保存路径，None 时自动生成
        show_plot: 是否显示图像
        smooth_window: 滑动平均窗口宽度（奇数）
        smooth_times: 平滑次数

    返回：实际保存的图片路径。
    """
    with h5py.File(ch0_3_file, "r") as f:
        channel_data = f["channel_data"]
        time_samples, num_channels, num_events = channel_data.shape

        if channel_idx >= num_channels:
            raise ValueError(
                f"channel_idx={channel_idx} 超过通道数 {num_channels}"
            )
        if event_index >= num_events:
            raise IndexError(f"event_index={event_index} 超过事件数 {num_events}")

        waveform = channel_data[:, channel_idx, event_index].astype(np.float64)

    # 滑动平均平滑
    waveform_smooth = waveform.copy()
    if smooth_window is not None and smooth_window > 1:
        if smooth_window % 2 == 0:
            raise ValueError(f"smooth_window 必须为奇数，当前为 {smooth_window}")
        if waveform.size >= smooth_window:
            half = smooth_window // 2
            for _ in range(max(1, smooth_times)):
                tmp = waveform_smooth.copy()
                for i in range(half, waveform.size - half):
                    waveform_smooth[i] = float(np.mean(tmp[i - half : i + half + 1]))

    sampling_interval_ns = 4.0
    time_axis_us = np.arange(time_samples) * sampling_interval_ns / 1000.0

    # y 轴范围
    global_min = float(np.min(waveform_smooth))
    global_max = float(np.max(waveform_smooth))
    data_range = global_max - global_min
    if data_range > 0:
        margin = data_range * 0.15
        y_min, y_max = global_min - margin, global_max + margin
    else:
        center = (global_min + global_max) / 2.0
        margin = max(abs(center) * 0.1, 100.0)
        y_min, y_max = center - margin, center + margin

    # tanh 前沿拟合
    def _tanh_rise(x, p0, p1, p2, p3):
        return 0.5 * p0 * np.tanh(p1 * (x - p2)) + p3

    fit_curve = None
    try:
        baseline_window_us = 2.0
        samples_baseline = int(round(baseline_window_us * 1000.0 / sampling_interval_ns))
        samples_baseline = max(1, min(samples_baseline, waveform_smooth.size // 2))

        baseline_front = float(np.mean(waveform_smooth[:samples_baseline]))
        amp = waveform_smooth - baseline_front
        max_amp = float(np.max(amp))

        if max_amp > 0:
            idx_max = int(np.argmax(amp))
            t_max = float(time_axis_us[idx_max])
            # 拟合范围：从起点到峰值再往后 2000 个点
            n_samples = len(time_axis_us)
            idx_end = min(idx_max + 2000, n_samples - 1)
            mask = np.arange(n_samples) <= idx_end

            if np.count_nonzero(mask) >= 5:
                x_data = time_axis_us[mask]
                y_data = waveform_smooth[mask]

                p0_init = max_amp
                p3_init = baseline_front

                mid_level = 0.5 * max_amp
                idx_mid = int(np.argmax(amp >= mid_level))
                p2_init = float(time_axis_us[idx_mid]) if amp[idx_mid] >= mid_level else float(np.mean(x_data))

                level_5, level_95 = 0.05 * max_amp, 0.95 * max_amp
                idx_5 = np.argmax(amp >= level_5)
                idx_95 = np.argmax(amp >= level_95)
                if amp[idx_5] >= level_5 and amp[idx_95] >= level_95 and idx_95 > idx_5:
                    rise_time = max(float(time_axis_us[idx_95] - time_axis_us[idx_5]), 1e-6)
                    p1_init = np.log(19.0) / rise_time
                else:
                    p1_init = 1.0 / (x_data[-1] - x_data[0] + 1e-6)

                popt, _ = curve_fit(
                    _tanh_rise, x_data, y_data,
                    p0=[p0_init, p1_init, p2_init, p3_init],
                    maxfev=10000,
                )
                fit_curve = _tanh_rise(time_axis_us, *popt)
    except Exception:
        pass

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    ax.plot(time_axis_us, waveform_smooth, color="blue", linewidth=1, label="Waveform (smoothed)")
    if fit_curve is not None:
        ax.plot(time_axis_us, fit_curve, color="red", linewidth=3, linestyle="--", label="tanh fit")

    ax.set_xlabel("Time (µs)", fontsize=18, fontweight="bold")
    ax.set_ylabel("Amplitude (ADC)", fontsize=18, fontweight="bold")
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontweight("bold")
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontweight("bold")

    filename = os.path.basename(ch0_3_file)
    ax.set_title(f"CH3 Waveform (tanh fit)\n{filename}  |  Event #{event_index}", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path is None:
        ge_self_dir = os.path.dirname(cut_dir)
        data_dir = os.path.dirname(ge_self_dir)
        python_dir = os.path.dirname(data_dir)
        project_root = os.path.dirname(python_dir)
        output_dir = os.path.join(project_root, "images", "presentation")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(output_dir, f"ch3_tanh_event{event_index}_{timestamp}.png")

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"波形图已保存至: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return save_path

def visualize_ch3_tanh(
    ch0_3_file: Optional[str] = None,
    channel_idx: int = 3,
    max_events_to_plot: int = 8,
    show_plot: bool = True,) -> List[str]:
    """
    对 HDF5 原始数据中的 CH3 通道进行 tanh 拟合可视化。

    参数：
        ch0_3_file: CH0-3 文件路径，None 时自动查找
        channel_idx: 通道索引（默认 3，CH3）
        max_events_to_plot: 最多可视化事件数
        show_plot: 是否显示图像

    返回：保存的图片路径列表。
    """
    ch0_3_file = _resolve_ch0_3_file(ch0_3_file)

    with h5py.File(ch0_3_file, "r") as f:
        num_events = f["channel_data"].shape[2]

    n_plot = min(max_events_to_plot, num_events)
    print(f"将对前 {n_plot} 个事件进行 CH3 tanh 拟合可视化（共 {num_events} 个事件）")

    saved_paths: List[str] = []
    for i in range(n_plot):
        print(f"\n[{i + 1}/{n_plot}] 可视化 Event #{i}")
        path = _visualize_single_event(
            ch0_3_file=ch0_3_file,
            event_index=i,
            channel_idx=channel_idx,
            save_path=None,
            show_plot=show_plot,
        )
        saved_paths.append(path)

    return saved_paths

if __name__ == "__main__":
    try:
        paths = visualize_ch3_tanh(
            ch0_3_file=None,
            channel_idx=3,
            max_events_to_plot=8,
            show_plot=True,
        )
        print("\n保存的图片路径：")
        for p in paths:
            print("  ", p)
    except Exception as e:
        print(f"\nCH3 tanh 拟合可视化失败: {e}")
        import traceback
        traceback.print_exc()
