#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据随机触发（RT）的幅度筛选条件，从 CH5 原始脉冲数据中
筛选出符合 RT 条件的事件，并绘制其 CH5 原始波形。

筛选逻辑参考同目录下的 `distribution.py`：
- 先计算每个事件在 CH5 上的波形最大值
- 给定阈值 cut，满足 max_value > cut 的事件视为 RT 事件
"""

import os
import sys
from typing import Optional, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt

# 为了与 distribution.py 保持一致，这里重用 utils.visualize 中的工具函数
current_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.dirname(os.path.dirname(current_dir))
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from utils.visualize import get_h5_files  # noqa: E402


def select_rt_events(
    h5_file: Optional[str] = None,
    channel_idx: int = 0,
    cut: float = 6000.0,
    batch_size: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    按 RT 条件筛选事件：max_amplitude > cut

    参数:
        h5_file: HDF5 文件路径；为 None 时自动从 CH5 目录中选择第一个文件
        channel_idx: 通道索引（CH5 目录中只有通道 0）
        cut: RT 截断阈值，max_value > cut 的事件视为 RT 事件
        batch_size: 计算最大值时的批处理大小

    返回:
        rt_indices: 满足 RT 条件的事件索引数组
        max_values: 所有事件在指定通道上的最大值数组
    """
    # 自动选择 CH5 文件
    if h5_file is None:
        h5_files = get_h5_files()
        if "CH5" not in h5_files or not h5_files["CH5"]:
            raise FileNotFoundError("在 data/hdf5/raw_pulse/CH5 目录中未找到 h5 文件")
        h5_file = h5_files["CH5"][0]
        print(f"自动选择文件: {os.path.basename(h5_file)}")

    if not os.path.exists(h5_file):
        raise FileNotFoundError(f"文件不存在: {h5_file}")

    print("=" * 70)
    print(f"筛选 RT 事件的文件: {os.path.basename(h5_file)}")
    print(f"文件路径: {h5_file}")
    print(f"RT 阈值 cut = {cut:.2f}")
    print("=" * 70)

    with h5py.File(h5_file, "r") as f:
        if "channel_data" not in f:
            raise KeyError("文件中没有找到 channel_data 数据集")

        channel_data = f["channel_data"]
        time_samples, num_channels, num_events = channel_data.shape

        if channel_idx < 0 or channel_idx >= num_channels:
            raise IndexError(f"通道索引 {channel_idx} 超出范围 [0, {num_channels-1}]")

        print(f"数据维度: (时间采样点数={time_samples}, 通道数={num_channels}, 事件数={num_events})")
        print("\n正在计算所有事件的最大值用于 RT 筛选 ...")

        max_values = np.zeros(num_events, dtype=np.float64)

        for i in range(0, num_events, batch_size):
            end_idx = min(i + batch_size, num_events)
            batch = channel_data[:, channel_idx, i:end_idx]  # (time_samples, batch_size)
            max_values[i:end_idx] = np.max(batch, axis=0)
            if (i // batch_size + 1) % 10 == 0 or end_idx == num_events:
                print(f"  已处理 {end_idx}/{num_events} 个事件 ({end_idx / num_events * 100:.1f}%)")

    # RT 条件：max_values > cut
    rt_mask = max_values > cut
    rt_indices = np.where(rt_mask)[0]

    print("\nRT 筛选结果:")
    print(f"  总事件数: {num_events}")
    print(f"  满足 RT 条件 (max > {cut:.2f}) 的事件数: {len(rt_indices)}")
    if len(rt_indices) > 0:
        print(f"  第一个 RT 事件索引: {rt_indices[0]}，其最大值: {max_values[rt_indices[0]]:.2f}")

    return rt_indices, max_values


def select_and_plot_one_rt_event(
    h5_file: Optional[str] = None,
    channel_idx: int = 0,
    cut: float = 6000.0,
    time_unit: str = "us",
    prefer_max_peak: bool = True,
) -> None:
    """
    使用 RT 条件筛选事件，并从中选出一个事件画出 CH5 原始波形。

    参数:
        h5_file: HDF5 文件路径；为 None 时自动从 CH5 目录中选择第一个文件
        channel_idx: 通道索引（CH5 目录中只有通道 0）
        cut: RT 截断阈值
        time_unit: 波形横轴时间单位，见 utils.visualize.visualize_waveform
        prefer_max_peak: 如果为 True，则从 RT 事件中选择最大峰值的事件；
                         否则选择第一个 RT 事件
    """
    rt_indices, max_values = select_rt_events(
        h5_file=h5_file,
        channel_idx=channel_idx,
        cut=cut,
    )

    if len(rt_indices) == 0:
        print("\n没有找到满足 RT 条件的事件，无法绘制波形。")
        return

    # 选择要画的事件索引
    if prefer_max_peak:
        # 在所有 RT 事件中找到最大峰值对应的事件
        rt_max_vals = max_values[rt_indices]
        best_idx_in_rt = int(np.argmax(rt_max_vals))
        event_idx = int(rt_indices[best_idx_in_rt])
        print(
            f"\n选择最大峰值 RT 事件: 全部事件索引 = {event_idx}, "
            f"其最大值 = {max_values[event_idx]:.2f}"
        )
    else:
        # 选择第一个 RT 事件
        event_idx = int(rt_indices[0])
        print(
            f"\n选择第一个 RT 事件: 全部事件索引 = {event_idx}, "
            f"其最大值 = {max_values[event_idx]:.2f}"
        )

    # 自动确定文件路径（保持与 select_rt_events 中一致）
    if h5_file is None:
        h5_files = get_h5_files()
        h5_file = h5_files["CH5"][0]

    print(f"\n开始绘制该事件的 CH5 原始波形 ...")

    # ===== 绘图风格对齐 data/inhibit/select.py =====
    # 这里不再调用 visualize_waveform，而是手动使用 matplotlib 绘制，
    # 使得图片风格（坐标轴、网格、标题等）与 Inhibit 绘图脚本保持一致。
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f"文件不存在: {h5_file}")

    with h5py.File(h5_file, "r") as f:
        if "channel_data" not in f:
            raise KeyError("文件中没有找到 channel_data 数据集")

        channel_data = f["channel_data"]
        time_samples, num_channels, num_events = channel_data.shape

        if "time_data" in f:
            time_data = f["time_data"]
            event_time = float(time_data[event_idx])
        else:
            event_time = None

        waveform = channel_data[:, channel_idx, event_idx]

        # 时间轴：与 Inhibit 脚本相同，默认使用 μs
        sampling_interval_ns = 4.0  # 4ns per sample (V1725)
        time_axis_us = np.arange(time_samples) * sampling_interval_ns / 1000.0

        max_val = float(np.max(waveform))
        min_val = float(np.min(waveform))
        mean_val = float(np.mean(waveform))

        plt.figure(figsize=(10, 6))
        # 主波形：蓝色实线
        plt.plot(time_axis_us, waveform, "b-", linewidth=0.8, alpha=0.8, label="CH5 Waveform")

        # 标注最大值位置（RT 事件关注的是大幅度脉冲）
        max_idx = int(np.argmax(waveform))
        plt.plot(
            time_axis_us[max_idx],
            waveform[max_idx],
            "ro",
            markersize=6,
            label=f"Max: {max_val:.1f}",
        )

        plt.xlabel("Time (μs)", fontsize=12)
        plt.ylabel("Amplitude (ADC counts)", fontsize=12)

        title_lines = [
            f"RT Event Waveform on CH5 (event #{event_idx})",
            f"Max: {max_val:.1f}, Min: {min_val:.1f}, Mean: {mean_val:.1f}, Cut: {cut:.1f}",
        ]
        if event_time is not None:
            title_lines.append(f"Event Time: {event_time:.6f} s")

        plt.title("\n".join(title_lines), fontsize=11)

        # 与 Inhibit 绘图一致：网格、0 线、图例
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color="r", linestyle="--", linewidth=1, alpha=0.5)
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.show()


def plot_ch0_for_rt_events(
    h5_file_ch5: Optional[str] = None,
    h5_file_ch0_3: Optional[str] = None,
    ch0_idx: int = 0,
    cut: float = 6000.0,
    prefer_max_peak: bool = True,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    绘制「符合 RT 筛选条件」的事件在 CH0 上的波形（默认只绘制一个事件）。

    RT 条件仍然基于 CH5（随机触发通道）的最大幅度 max > cut，
    但画图时使用 CH0-3 文件中的 CH0 波形。

    参数:
        h5_file_ch5:   CH5 的 HDF5 文件路径，用于做 RT 选择；
                       为 None 时自动从 CH5 目录中选择第一个文件
        h5_file_ch0_3: CH0-3 的 HDF5 文件路径，用于读取 CH0 波形；
                       为 None 时自动从 CH0-3 目录中选择第一个文件
        ch0_idx:       在 CH0-3 文件中的通道索引（默认 0，即 CH0）
        cut:           RT 截断阈值（在 CH5 上的最大值阈值）
        prefer_max_peak: 如果为 True，则从 RT 事件中选择最大峰值的事件；
                         否则选择第一个 RT 事件
        figsize:       图片大小 (宽度, 高度)
    """
    # 先在 CH5 上做 RT 选择
    rt_indices, max_values = select_rt_events(
        h5_file=h5_file_ch5,
        channel_idx=0,
        cut=cut,
    )

    if len(rt_indices) == 0:
        print("\n没有找到满足 RT 条件的事件，无法绘制 CH0 波形。")
        return

    # 选择要绘制的事件索引（默认只绘制一个）
    if prefer_max_peak:
        # 在所有 RT 事件中找到最大峰值对应的事件
        rt_max_vals = max_values[rt_indices]
        best_idx_in_rt = int(np.argmax(rt_max_vals))
        event_idx = int(rt_indices[best_idx_in_rt])
        print(
            f"\n选择最大峰值 RT 事件: 全部事件索引 = {event_idx}, "
            f"其 CH5 最大值 = {max_values[event_idx]:.2f}"
        )
    else:
        # 选择第一个 RT 事件
        event_idx = int(rt_indices[0])
        print(
            f"\n选择第一个 RT 事件: 全部事件索引 = {event_idx}, "
            f"其 CH5 最大值 = {max_values[event_idx]:.2f}"
        )

    # 自动选择 CH0-3 文件
    if h5_file_ch0_3 is None:
        h5_files = get_h5_files()
        if "CH0-3" not in h5_files or not h5_files["CH0-3"]:
            raise FileNotFoundError("在 data/hdf5/raw_pulse/CH0-3 目录中未找到 h5 文件")
        h5_file_ch0_3 = h5_files["CH0-3"][0]
        print(f"自动选择 CH0-3 文件: {os.path.basename(h5_file_ch0_3)}")

    if not os.path.exists(h5_file_ch0_3):
        raise FileNotFoundError(f"文件不存在: {h5_file_ch0_3}")

    print(f"\n开始绘制该事件的 CH0 原始波形 ...")

    # ===== 绘图风格对齐 data/inhibit/select.py =====
    with h5py.File(h5_file_ch0_3, "r") as f:
        if "channel_data" not in f:
            raise KeyError("CH0-3 文件中没有找到 channel_data 数据集")

        channel_data = f["channel_data"]
        time_samples, num_channels, num_events = channel_data.shape

        if ch0_idx < 0 or ch0_idx >= num_channels:
            raise IndexError(f"CH0-3 文件中的通道索引 {ch0_idx} 超出范围 [0, {num_channels-1}]")

        if event_idx >= num_events:
            raise IndexError(f"事件索引 {event_idx} 超出 CH0-3 文件事件数范围 [0, {num_events-1}]")

        # 读取 CH0 波形数据
        waveform = channel_data[:, ch0_idx, event_idx]

        # 时间轴：与 Inhibit 脚本保持一致，使用 μs
        sampling_interval_ns = 4.0  # 4ns per sample (V1725)
        time_axis_us = np.arange(time_samples) * sampling_interval_ns / 1000.0

        # 统计信息
        w_min = float(np.min(waveform))
        w_max = float(np.max(waveform))
        w_mean = float(np.mean(waveform))

        # CH5 上对应事件的最大值（用于标题说明）
        rt_peak = float(max_values[event_idx])

        # 绘制单个波形图
        plt.figure(figsize=figsize)
        # 主波形：蓝色实线
        plt.plot(time_axis_us, waveform, "b-", linewidth=0.8, alpha=0.8, label="CH0 Waveform")

        # 标注最大值位置
        max_idx = int(np.argmax(waveform))
        plt.plot(
            time_axis_us[max_idx],
            waveform[max_idx],
            "ro",
            markersize=6,
            label=f"Max: {w_max:.1f}",
        )

        plt.xlabel("Time (μs)", fontsize=12)
        plt.ylabel("Amplitude (ADC counts)", fontsize=12)

        title_lines = [
            f"RT Event Waveform on CH0 (event #{event_idx})",
            f"CH0: Min: {w_min:.1f}, Max: {w_max:.1f}, Mean: {w_mean:.1f} | RT peak@CH5: {rt_peak:.1f}, Cut: {cut:.1f}",
        ]

        plt.title("\n".join(title_lines), fontsize=11)

        # 与 Inhibit 绘图一致：网格、0 线、图例
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color="r", linestyle="--", linewidth=1, alpha=0.5)
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    """
    直接运行本脚本时的示例用法：
    - 使用 cut=6000 的 RT 条件（与 distribution.py 中示例保持一致）
    - 从 CH5 的原始脉冲中筛选 RT 事件，并绘制一个代表性 RT 事件的 CH5 波形
    """
    print("=" * 70)
    print("基于 RT 条件筛选 CH5 事件并绘制原始波形，并查看对应的 CH0 信号")
    print("=" * 70)

    try:
        # 1) 在 CH5 上选择并绘制一个代表性的 RT 事件波形
        select_and_plot_one_rt_event(
            h5_file=None,    # 自动从 CH5 目录选第一个文件
            channel_idx=0,   # CH5 目录只有通道 0
            cut=6000.0,      # 参考 distribution.py 中的示例 cut 值
            time_unit="us",  # 画图时的时间单位，可按需修改 'ns'/'us'/'ms'/'s'
            prefer_max_peak=True,
        )

        # 2) 在 CH0-3 文件中绘制 RT 事件对应的 CH0 波形（默认只绘制一个）
        print("\n" + "=" * 70)
        print("绘制满足 RT 条件事件在 CH0 上的波形")
        print("=" * 70)
        plot_ch0_for_rt_events(
            h5_file_ch5=None,      # 与上面一致，自动选择 CH5 文件并做 RT 选择
            h5_file_ch0_3=None,    # 自动选择 CH0-3 文件
            ch0_idx=0,             # CH0 通道
            cut=6000.0,
            prefer_max_peak=True,  # 选择最大峰值的 RT 事件
        )
    except Exception as e:
        print(f"\n执行过程中出错: {e}")

