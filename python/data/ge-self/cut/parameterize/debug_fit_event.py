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
from scipy.optimize import curve_fit


current_dir = os.path.dirname(os.path.abspath(__file__))
cut_dir = os.path.dirname(current_dir)
sampling_interval_ns = 4.0


def _resolve_ch0_3_file(ch0_3_file: Optional[str]) -> str:
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


def _tanh_rise(x, p0, p1, p2, p3):
    return 0.5 * p0 * np.tanh(p1 * (x - p2)) + p3


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
    waveform_smooth = waveform.copy()

    # # 平滑
    # if smooth_window is not None and smooth_window > 1 and waveform.size >= smooth_window:
    #     half = smooth_window // 2
    #     for _ in range(max(1, smooth_times)):
    #         tmp = waveform_smooth.copy()
    #         for i in range(half, waveform.size - half):
    #             waveform_smooth[i] = float(np.mean(tmp[i - half : i + half + 1]))

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

    x_data = time_axis_us[mask]
    y_data = waveform_smooth[mask]

    p0_init, p3_init = max_amp, baseline_front
    mid_level = 0.5 * max_amp
    idx_mid = int(np.argmax(amp >= mid_level))
    p2_init = float(time_axis_us[idx_mid]) if amp[idx_mid] >= mid_level else float(np.mean(x_data))
    level_5, level_95 = 0.05 * max_amp, 0.95 * max_amp
    idx_5, idx_95 = np.argmax(amp >= level_5), np.argmax(amp >= level_95)
    if amp[idx_5] >= level_5 and amp[idx_95] >= level_95 and idx_95 > idx_5:
        rise_time = max(float(time_axis_us[idx_95] - time_axis_us[idx_5]), 1e-6)
        p1_init = np.log(19.0) / rise_time
        print("--- 4. 初值（指定区间）---")
        print(f"p0_init = {p0_init:.2f}, p1_init = {p1_init:.4f}, p2_init = {p2_init:.3f}, p3_init = {p3_init:.2f}")
        print(f"rise_time (5%~95%) = {rise_time:.4f} µs")
    else:
        p1_init = 1.0 / (x_data[-1] - x_data[0] + 1e-6)
        print("--- 4. 初值（指定区间，退化）---")
        print(f"idx_5={idx_5}, idx_95={idx_95}, amp[idx_5]={amp[idx_5]:.2f}, amp[idx_95]={amp[idx_95]:.2f}")
        print(f"5%~95% 约束不满足，使用 p1_init = 1/时间跨度 = {p1_init:.6f}")
        print(f"p0_init={p0_init:.2f}, p1_init={p1_init:.6f}, p2_init={p2_init:.3f}, p3_init={p3_init:.2f}")
    print()

    # 第一次拟合：指定区间
    fit_curve = None
    popt = None

    print("--- 5. 第一次拟合（指定区间 0 ~ idx_max+2000）---")
    try:
        popt, pcov = curve_fit(
            _tanh_rise, x_data, y_data,
            p0=[p0_init, p1_init, p2_init, p3_init],
            maxfev=10000,
        )
        fit_curve = _tanh_rise(time_axis_us, *popt)
        print("  成功!")
        print(f"  popt = p0={popt[0]:.4f}, p1={popt[1]:.4f}, p2={popt[2]:.4f}, p3={popt[3]:.4f}")
    except Exception as e:
        print(f"  失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print()

        # 第二次拟合：全部点
        print("--- 6. 第二次拟合（全部点，无约束）---")
        x_data_full = time_axis_us
        y_data_full = waveform_smooth
        time_span = float(x_data_full[-1] - x_data_full[0]) + 1e-6
        p0_fb = float(np.max(y_data_full) - np.min(y_data_full))
        p3_fb = float(np.min(y_data_full))
        p2_fb = float(np.mean(x_data_full))
        p1_fb = 1.0 / time_span
        print(f"初值: p0={p0_fb:.2f}, p1={p1_fb:.6f}, p2={p2_fb:.3f}, p3={p3_fb:.2f}")
        # 全点拟合不易收敛：提高迭代上限，并加宽松边界避免发散
        bounds_low = [1e-6, 1e-6, float(x_data_full[0]) - 100, -np.inf]
        bounds_high = [np.inf, 1e2, float(x_data_full[-1]) + 100, np.inf]
        try:
            popt, pcov = curve_fit(
                _tanh_rise, x_data_full, y_data_full,
                p0=[p0_fb, p1_fb, p2_fb, p3_fb],
                bounds=(bounds_low, bounds_high),
                maxfev=100000,
            )
            fit_curve = _tanh_rise(time_axis_us, *popt)
            print("  成功!")
            print(f"  popt = p0={popt[0]:.4f}, p1={popt[1]:.4f}, p2={popt[2]:.4f}, p3={popt[3]:.4f}")
        except Exception as e2:
            print(f"  失败: {type(e2).__name__}: {e2}")
            import traceback
            traceback.print_exc()

    # 绘图
    print()
    print("--- 7. 绘图 ---")
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

    # 标出拟合区间与峰值位置
    ax.axvspan(time_axis_us[0], time_axis_us[idx_end], alpha=0.1, color="green", label="拟合区间")
    ax.axvline(t_max, color="gray", ls=":", alpha=0.7, label=f"t_max @ idx={idx_max}")

    ax.set_xlabel("Time (µs)", fontsize=12)
    ax.set_ylabel("Amplitude (ADC)", fontsize=12)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f"Event #{event_index}  tanh 拟合调试  " + ("成功" if fit_curve is not None else "失败"), fontsize=12)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="调试指定 event 的 tanh 拟合")
    parser.add_argument("-e", "--event", type=int, default=2647, help="event 索引")
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
