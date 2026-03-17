#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
针对指定 event（默认 68）的 tanh 拟合调试脚本。

输出详细的拟合参数、初值、范围等信息，便于定位拟合失败原因。
"""

import os
import sys
from typing import Optional

import h5py
import numpy as np
import matplotlib.pyplot as plt


current_dir = os.path.dirname(os.path.abspath(__file__))
cut_dir = os.path.dirname(current_dir)
sampling_interval_ns = 4.0

# 与正式快放拟合保持一致的实现
ge_self_dir = os.path.dirname(cut_dir)
data_dir = os.path.dirname(ge_self_dir)
python_dir = os.path.dirname(data_dir)
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)
from utils.fit import (  # type: ignore  # noqa: E402
    _compute_fast_fit_params,
    _tanh_rise,
    _smooth_waveform_for_fast_fit,
)


def _resolve_ch0_3_file(ch0_3_file: Optional[str]) -> str:
    if ch0_3_file is not None:
        return ch0_3_file
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)
    from utils.visualize import get_h5_files
    h5_files = get_h5_files()
    if "CH0-3" not in h5_files or not h5_files["CH0-3"]:
        raise FileNotFoundError("No HDF5 file found in data/hdf5/raw_pulse/CH0-3")
    return h5_files["CH0-3"][0]


def debug_fit_event(
    event_index: int = 68,
    ch0_3_file: Optional[str] = None,
    channel_idx: int = 3,
    smooth_window: int = 5,
    smooth_times: int = 25,
):
    """对指定 event 进行 tanh 拟合，并打印详细调试信息。"""
    ch0_3_file = _resolve_ch0_3_file(ch0_3_file)

    print("=" * 70)
    print(f"tanh 拟合调试 - Event #{event_index}")
    print("=" * 70)
    print(f"文件: {ch0_3_file}")
    print(f"通道: CH{channel_idx}")
    print()

    with h5py.File(ch0_3_file, "r") as f:
        channel_data = f["channel_data"]
        time_samples, num_channels, num_events = channel_data.shape
        print(f"数据形状: time_samples={time_samples}, num_channels={num_channels}, num_events={num_events}")
        if event_index >= num_events:
            print(f"错误: event_index={event_index} 超过 num_events={num_events}")
            return
        print()

        waveform = channel_data[:, channel_idx, event_index].astype(np.float64)

    time_axis_us = np.arange(time_samples) * sampling_interval_ns / 1000.0
    # 使用与 utils.fit 中相同的平滑方式（只是为了可视化更接近正式拟合）
    waveform_smooth = _smooth_waveform_for_fast_fit(
        waveform, smooth_window=smooth_window, smooth_times=smooth_times
    )

    print("--- 1. 波形概览 ---")
    print(f"波形范围: min={np.min(waveform_smooth):.1f}, max={np.max(waveform_smooth):.1f}")
    print(f"时间范围: {time_axis_us[0]:.3f} ~ {time_axis_us[-1]:.3f} µs")
    print()

    baseline_window_us = 2.0
    samples_baseline = int(round(baseline_window_us * 1000.0 / sampling_interval_ns))
    samples_baseline = max(1, min(samples_baseline, waveform_smooth.size // 2))
    baseline_front = float(np.mean(waveform_smooth[:samples_baseline]))
    amp = waveform_smooth - baseline_front
    max_amp = float(np.max(amp))

    print("--- 2. 基线与幅度 ---")
    print(f"samples_baseline = {samples_baseline} (前 {baseline_window_us} µs)")
    print(f"baseline_front = {baseline_front:.2f}")
    print(f"max_amp = {max_amp:.2f}")
    if max_amp <= 0:
        print("错误: max_amp <= 0，无法拟合")
        return
    print()

    idx_max = int(np.argmax(amp))
    t_max = float(time_axis_us[idx_max])
    n_samples = len(time_axis_us)
    idx_end = min(idx_max + 2000, n_samples - 1)
    mask = np.arange(n_samples) <= idx_end
    n_mask = int(np.count_nonzero(mask))

    print("--- 3. 峰值与拟合范围 ---")
    print(f"idx_max = {idx_max} (位置 {idx_max / n_samples * 100:.1f}% 处)")
    print(f"t_max = {t_max:.3f} µs")
    print(f"拟合范围: index 0 ~ {idx_end} (共 {idx_end + 1} 点)")
    print(f"mask 有效点数: {n_mask}")
    if idx_max < n_samples * 0.05:
        print("  [警告] 峰值在开头 5% 内，基线可能被污染")
    if idx_max > n_samples * 0.95:
        print("  [警告] 峰值在末尾 5% 内，可能是截断或慢上升")
    print()

    # 使用 utils.fit 中的统一快放拟合函数
    print("--- 4. 使用 utils.fit._compute_fast_fit_params 拟合 ---")
    params = _compute_fast_fit_params(
        waveform,
        sampling_interval_ns=sampling_interval_ns,
        baseline_window_us=baseline_window_us,
    )
    print(
        "tanh 参数: "
        f"p0={params['tanh_p0']:.4f}, "
        f"p1={params['tanh_p1']:.4f}, "
        f"p2={params['tanh_p2']:.4f}, "
        f"p3={params['tanh_p3']:.4f}"
    )
    print(f"tanh_rms = {params['tanh_rms']:.6g}")

    abnormal_value = 1e6
    fit_curve = None
    if params["tanh_rms"] < abnormal_value:
        fit_curve = _tanh_rise(
            time_axis_us,
            params["tanh_p0"],
            params["tanh_p1"],
            params["tanh_p2"],
            params["tanh_p3"],
        )
        print("拟合状态: 成功")
    else:
        print("拟合状态: 失败（命中异常值阈值）")

    # 绘图
    print()
    print("--- 5. 绘图 ---")
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

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(time_axis_us, waveform_smooth, "b-", lw=1, label="Waveform (smoothed)")
    if fit_curve is not None:
        ax.plot(time_axis_us, fit_curve, "r--", lw=2, label="tanh fit")

    # Mark fitting range and peak position
    ax.axvspan(time_axis_us[0], time_axis_us[idx_end], alpha=0.1, color="green", label="Fit range")
    ax.axvline(t_max, color="gray", ls=":", alpha=0.7, label=f"t_max @ idx={idx_max}")

    ax.set_xlabel("Time (µs)", fontsize=12)
    ax.set_ylabel("Amplitude (ADC)", fontsize=12)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(
        f"Event #{event_index}  tanh fit debug  " + ("SUCCESS" if fit_curve is not None else "FAIL"),
        fontsize=12,
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="调试指定 event 的 tanh 拟合")
    parser.add_argument("-e", "--event", type=int, default=5716, help="event 索引")
    parser.add_argument("-f", "--file", type=str, default=None, help="CH0-3 文件路径")
    parser.add_argument(
        "-c",
        "--channel",
        type=int,
        default=3,
        help="通道索引 (0=CH0, 3=CH3，默认 3)",
    )
    args = parser.parse_args()

    try:
        debug_fit_event(
            event_index=args.event,
            ch0_3_file=args.file,
            channel_idx=args.channel,
        )
    except Exception as e:
        print(f"运行失败: {e}")
        import traceback
        traceback.print_exc()
