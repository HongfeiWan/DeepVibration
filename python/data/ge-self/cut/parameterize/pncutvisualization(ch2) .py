#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PN-cut 选中事件的 CH2 波形可视化脚本。

功能：
- 使用 overthreshold.py 中的 select_physical_events_no_overthreshold 先筛出不过阈值的 Physical 事件；
- 在这些事件中，计算两路主放（CH0 与 CH1）的最大值散点 (max_ch0, max_ch1)；
- 使用 lsmpncut.py 中的 fit_single_line_in_range 在 2000 < max_ch0 < 14000 范围内进行两步最小二乘拟合，
  得到参考直线 max_ch1 ≈ a * max_ch0 + b，并计算残差的标准差 σ；
- 选出所有事件中落在该直线的 ±1σ 带内的事件；
- 对这些事件，对应的 CH0 波形调用 visualization.py 中的可视化风格进行逐个参数化绘图。
"""

import os
import sys
from typing import Optional, Tuple, List
from datetime import datetime

import h5py
import numpy as np
import matplotlib.pyplot as plt
import importlib.util


# -----------------------------------------------------------------------------
# 导入 select_physical_events_no_overthreshold
# -----------------------------------------------------------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))          # .../cut/parameterize
cut_dir = os.path.dirname(current_dir)                            # .../cut

overthreshold_path = os.path.join(cut_dir, "overthreshold.py")
spec_over = importlib.util.spec_from_file_location("overthreshold_module", overthreshold_path)
overthreshold_module = importlib.util.module_from_spec(spec_over)
assert spec_over.loader is not None
spec_over.loader.exec_module(overthreshold_module)

select_physical_events_no_overthreshold = overthreshold_module.select_physical_events_no_overthreshold


# 导入 lsmpncut.py 中的 fit_single_line_in_range 函数
lsmpncut_path = os.path.join(cut_dir, "lsmpncut.py")
spec_ls = importlib.util.spec_from_file_location("lsmpncut_module", lsmpncut_path)
lsmpncut_module = importlib.util.module_from_spec(spec_ls)
assert spec_ls.loader is not None
spec_ls.loader.exec_module(lsmpncut_module)

fit_single_line_in_range = lsmpncut_module.fit_single_line_in_range


def _select_events_in_1sigma_band(
    ch0_3_file: Optional[str] = None,
    ch5_file: Optional[str] = None,
    rt_cut: float = 6000.0,
    ch0_threshold: float = 16382.0,
    ch0_idx: int = 0,
    ch1_idx: int = 1,
    x_min: float = 2000.0,
    x_max: float = 14000.0,
    sigma_factor: float = 1.0,
) -> Tuple[np.ndarray, str, str]:
    """
    在不过阈值 Physical 事件中，使用 lsmpncut 的 PN-cut 逻辑选出落在 ±1σ 线性带内的事件。

    逻辑：
    1. 先筛选不过阈值的 Physical 事件（既非 RT 也非 Inhibit）；
    2. 计算 CH0 和 CH1 的最大值 (max_ch0, max_ch1)；
    3. 使用 lsmpncut.fit_single_line_in_range 在 (x_min, x_max) 范围内进行两步最小二乘拟合，
       得到直线 max_ch1 ≈ a * max_ch0 + b；
    4. 在拟合范围内计算残差的标准差 σ；
    5. 选出所有事件中落在 ±sigma_factor*σ 带内的事件。

    返回：
        event_ranks : 在 select_physical_events_no_overthreshold 的 selected_indices 中的下标数组
        ch0_3_file  : 实际使用的 CH0-3 文件路径
        ch5_file    : 实际使用的 CH5 文件路径
    """
    print("=" * 70)
    print("PN-cut 可视化：使用 lsmpncut 逻辑选择 ±1σ 带内的事件")
    print("=" * 70)

    # 先筛选不过阈值 Physical 事件
    # 注意：CH5 文件通常只有一个通道（索引 0），RT 判定应始终使用 ch5_idx=0
    sel = select_physical_events_no_overthreshold(
        ch0_3_file=ch0_3_file,
        ch5_file=ch5_file,
        rt_cut=rt_cut,
        ch0_threshold=ch0_threshold,
        ch0_idx=ch0_idx,
        ch5_idx=0,
    )

    ch0_3_file_sel: str = sel["ch0_3_file"]
    ch5_file_sel: str = sel["ch5_file"]
    selected_indices: np.ndarray = sel["selected_indices"]
    final_physical_count: int = int(sel["final_physical_count"])

    if final_physical_count == 0 or selected_indices.size == 0:
        raise RuntimeError("未发现不过阈值的 Physical 信号，无法进行 PN-cut 可视化选择。")

    print(f"不过阈值 Physical 事件数: {final_physical_count}")

    # 读取这些事件的波形，并计算 CH0/CH1 的最大值（用于 PN-cut）
    with h5py.File(ch0_3_file_sel, "r") as f_ch0:
        channel_data = f_ch0["channel_data"]
        time_samples, num_channels, num_events = channel_data.shape

        if ch0_idx >= num_channels or ch1_idx >= num_channels:
            raise ValueError(
                f"通道索引超出范围：ch0_idx={ch0_idx}, ch1_idx={ch1_idx}, num_channels={num_channels}"
            )

        # 只取不过阈值 Physical 事件
        phys_ch0 = channel_data[:, ch0_idx, selected_indices].astype(np.float64)
        phys_ch1 = channel_data[:, ch1_idx, selected_indices].astype(np.float64)

    max_ch0 = phys_ch0.max(axis=0)
    max_ch1 = phys_ch1.max(axis=0)

    # 使用 lsmpncut 的 fit_single_line_in_range 在 (x_min, x_max) 范围内进行两步最小二乘拟合
    print(f"\n在 {x_min} < max_ch0 < {x_max} 范围内使用 lsmpncut 进行两步最小二乘拟合...")
    a, b = fit_single_line_in_range(max_ch0, max_ch1, x_min=x_min, x_max=x_max)

    # 在拟合范围内计算残差的标准差 σ
    mask_range = (max_ch0 > x_min) & (max_ch0 < x_max)
    x_fit = max_ch0[mask_range]
    y_fit = max_ch1[mask_range]
    y_fit_pred = a * x_fit + b
    residuals = y_fit - y_fit_pred
    sigma = residuals.std(ddof=1) if residuals.size > 1 else 0.0

    print(f"拟合直线: max_ch1 ≈ {a:.6f} * max_ch0 + {b:.3f}")
    print(f"拟合范围内点数: {x_fit.size}，残差标准差 σ = {sigma:.3f}")

    if sigma <= 0.0:
        raise RuntimeError("σ ≤ 0，无法定义 ±1σ 带。")

    # 对所有事件计算残差，并选出落在 ±sigma_factor*σ 带内的事件
    predicted_all = a * max_ch0 + b
    all_residuals = max_ch1 - predicted_all
    sigma_mask = np.abs(all_residuals) <= sigma_factor * sigma

    event_ranks = np.where(sigma_mask)[0]  # 在 selected_indices 中的下标

    print(f"\n落在 ±{sigma_factor}σ 带内的 Physical 事件数: {event_ranks.size}")

    if event_ranks.size == 0:
        raise RuntimeError("没有事件落在 ±σ 带内。")

    return event_ranks, ch0_3_file_sel, ch5_file_sel, selected_indices


def _visualize_single_event_by_index(
    ch0_3_file: str,
    event_index: int,
    channel_idx: int = 2,
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> str:
    """
    使用事件索引（全局事件号）来可视化单个通道的波形（本脚本用于 CH2）。

    逻辑：
        - 不再做参数化标注与基线放大，只绘制 PN-cut 选中事件对应通道的波形。

    参数：
        ch0_3_file: CH0-3 文件路径
        event_index: 全局事件索引
        channel_idx: 要展示的通道索引（本脚本默认 2，对应 CH2）
        save_path: 保存图片路径，None 时自动生成
        show_plot: 是否调用 plt.show()

    返回：
        实际保存的图片路径。
    """
    # 导入中值滤波函数
    python_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))  # .../python
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)
    from utils.filter import median_filter

    # 读取波形
    with h5py.File(ch0_3_file, "r") as f_ch0:
        channel_data = f_ch0["channel_data"]
        time_samples, num_channels, num_events = channel_data.shape

        if channel_idx >= num_channels:
            raise ValueError(
                f"channel_idx={channel_idx} 超过通道数 {num_channels}，无法读取该通道波形"
            )
        if event_index >= num_events:
            raise IndexError(f"event_index={event_index} 超过事件数 {num_events}")

        waveform = channel_data[:, channel_idx, event_index].astype(np.float64)

    # 对波形先做一次中值滤波
    waveform = median_filter(waveform, kernel_size=3)

    # 时间轴：4 ns 采样，单位 µs
    sampling_interval_ns = 4.0
    time_axis_us = np.arange(time_samples) * sampling_interval_ns / 1000.0

    # 统一 y 轴范围（两侧留一点 margin）
    global_min = float(np.min(waveform))
    global_max = float(np.max(waveform))
    data_range = global_max - global_min
    if data_range > 0:
        margin = data_range * 0.15
        y_min = global_min - margin
        y_max = global_max + margin
    else:
        center = (global_min + global_max) / 2.0
        margin = max(abs(center) * 0.1, 100.0)
        y_min = center - margin
        y_max = center + margin

    # 画图：仅展示波形
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
        }
    )
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        time_axis_us,
        waveform,
        color="C2",
        linewidth=1,
    )

    ax.set_xlabel("Time (µs)", fontsize=18, fontweight="bold")
    ax.set_ylabel("Amplitude (ADC)", fontsize=18, fontweight="bold")
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontweight("bold")
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontweight("bold")

    filename = os.path.basename(ch0_3_file)
    ax.set_title(
        f"CH2 Waveform after PN-cut\n"
        f"{filename}  |  Event #{event_index}",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()

    # 保存图片
    if save_path is None:
        ge_self_dir = os.path.dirname(cut_dir)
        data_dir = os.path.dirname(ge_self_dir)
        python_dir = os.path.dirname(data_dir)
        project_root = os.path.dirname(python_dir)

        output_dir = os.path.join(project_root, "images", "presentation")
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_png = (
            f"ch2_pncut_waveform_event{event_index}_{timestamp}.png"
        )
        save_path = os.path.join(output_dir, filename_png)

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"波形图已保存至: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return save_path


def visualize_pncut_waveforms(
    ch0_3_file: Optional[str] = None,
    ch5_file: Optional[str] = None,
    rt_cut: float = 6000.0,
    ch0_threshold: float = 16382.0,
    ch0_idx: int = 0,
    ch1_idx: int = 1,
    x_min: float = 2000.0,
    x_max: float = 14000.0,
    sigma_factor: float = 1.0,
    baseline_window_us: float = 2.0,
    display_channel_idx: int = 2,
    max_events_to_plot: int = 8,
    show_plot: bool = True,
) -> List[str]:
    """
    对符合 lsmpncut PN-cut ±1σ 带的事件，仅可视化指定通道（本脚本为 CH2）波形。

    参数：
        ch0_idx, ch1_idx: 通道索引，用于计算 max_ch0 和 max_ch1 进行 PN-cut
        x_min, x_max: PN-cut 拟合的范围（默认 2000 < max_ch0 < 14000）
        sigma_factor: σ 因子（默认 1.0，即 ±1σ）
        baseline_window_us: 前沿和后沿基线的时间窗口长度（微秒），默认 2.0 µs

    返回：
        保存的图片路径列表。
    """
    event_ranks, ch0_3_file_used, ch5_file_used, selected_indices = _select_events_in_1sigma_band(
        ch0_3_file=ch0_3_file,
        ch5_file=ch5_file,
        rt_cut=rt_cut,
        ch0_threshold=ch0_threshold,
        ch0_idx=ch0_idx,
        ch1_idx=ch1_idx,
        x_min=x_min,
        x_max=x_max,
        sigma_factor=sigma_factor,
    )

    # 只取前 max_events_to_plot 个事件做可视化
    n_plot = min(max_events_to_plot, event_ranks.size)
    print(f"\n将对前 {n_plot} 个事件进行 CH2 波形可视化。")

    saved_paths: List[str] = []
    for i in range(n_plot):
        rank = int(event_ranks[i])
        # 将 rank（在 selected_indices 中的下标）转换为全局事件索引
        event_index = int(selected_indices[rank])
        print(f"\n[{i+1}/{n_plot}] 可视化 event_rank = {rank} (全局 Event #{event_index})")
        path = _visualize_single_event_by_index(
            ch0_3_file=ch0_3_file_used,
            event_index=event_index,
            channel_idx=display_channel_idx,
            save_path=None,
            show_plot=show_plot,
        )
        saved_paths.append(path)

    return saved_paths


if __name__ == "__main__":
    try:
        # 示例：自动选择文件对，使用 lsmpncut 的 PN-cut 逻辑（2000 < max_ch0 < 14000），对前若干个事件做可视化
        paths = visualize_pncut_waveforms(
            ch0_3_file=None,
            ch5_file=None,
            rt_cut=6000.0,
            ch0_threshold=16382.0,
            ch0_idx=0,
            ch1_idx=1,
            x_min=2000.0,
            x_max=14000.0,
            sigma_factor=1.0,
            baseline_window_us=2.0,  # 该参数在本脚本中已不影响可视化，仅保留接口
            display_channel_idx=2,
            max_events_to_plot=2100,
            show_plot=True,
        )
        print("\n保存的图片路径：")
        for p in paths:
            print("  ", p)
    except Exception as e:
        print(f"\nPN-cut 可视化失败: {e}")
        import traceback

        traceback.print_exc()

