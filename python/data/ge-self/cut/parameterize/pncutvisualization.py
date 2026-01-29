#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PN-cut 选中事件的 CH0 波形可视化脚本。

功能：
- 使用 overthreshold.py 中的 select_physical_events_no_overthreshold 先筛出不过阈值的 Physical 事件；
- 在这些事件中，计算两路主放（例如 CH0 与 CH1）的最大值散点 (max_ch0, max_ch1)；
- 在给定的 (x, y) 范围内做一次线性拟合，得到参考直线 y = a x + b，并计算残差的标准差 σ；
- 选出同时满足：
    * 处于给定的 (x, y) 范围内；
    * 且在该直线的 ±1σ 带内
  的事件；
- 对这些事件，对应的 CH0 波形调用 visualization.py 中的可视化风格进行逐个参数化绘图。
"""

import os
import sys
from typing import Optional, Tuple, List

import h5py
import numpy as np


# -----------------------------------------------------------------------------
# 导入 select_physical_events_no_overthreshold
# -----------------------------------------------------------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))          # .../cut/parameterize
cut_dir = os.path.dirname(current_dir)                            # .../cut

overthreshold_path = os.path.join(cut_dir, "overthreshold.py")
import importlib.util

spec_over = importlib.util.spec_from_file_location("overthreshold_module", overthreshold_path)
overthreshold_module = importlib.util.module_from_spec(spec_over)
spec_over.loader.exec_module(overthreshold_module)

select_physical_events_no_overthreshold = overthreshold_module.select_physical_events_no_overthreshold


# 导入 visualization.py 中的单事件可视化函数
visualization_path = os.path.join(current_dir, "visualization.py")
spec_vis = importlib.util.spec_from_file_location("pulse_visualization_module", visualization_path)
pulse_vis_module = importlib.util.module_from_spec(spec_vis)
spec_vis.loader.exec_module(pulse_vis_module)

visualize_single_pulse_parameters = pulse_vis_module.visualize_single_pulse_parameters


def _select_events_in_1sigma_band(
    ch0_3_file: Optional[str] = None,
    ch5_file: Optional[str] = None,
    rt_cut: float = 6000.0,
    ch0_threshold: float = 16382.0,
    ch0_idx: int = 0,
    ch1_idx: int = 1,
    x_range: Tuple[float, float] = (1100.0, 1400.0),
    y_range: Tuple[float, float] = (1000.0, 2200.0),
    sigma_factor: float = 1.0,
) -> Tuple[np.ndarray, str, str]:
    """
    在不过阈值 Physical 事件中，选出 (max_ch0, max_ch1) 位于给定范围且落在 ±1σ 线性带内的事件。

    返回：
        event_ranks : 在 select_physical_events_no_overthreshold 的 selected_indices 中的下标数组
        ch0_3_file  : 实际使用的 CH0-3 文件路径
        ch5_file    : 实际使用的 CH5 文件路径
    """
    print("=" * 70)
    print("PN-cut 可视化：选择满足范围且在 ±1σ 带内的事件")
    print("=" * 70)

    # 先筛选不过阈值 Physical 事件
    sel = select_physical_events_no_overthreshold(
        ch0_3_file=ch0_3_file,
        ch5_file=ch5_file,
        rt_cut=rt_cut,
        ch0_threshold=ch0_threshold,
        ch0_idx=ch0_idx,
        ch5_idx=ch1_idx,  # 这里 ch5_idx 不重要，只用于 RT 判断
    )

    ch0_3_file_sel: str = sel["ch0_3_file"]
    ch5_file_sel: str = sel["ch5_file"]
    selected_indices: np.ndarray = sel["selected_indices"]
    final_physical_count: int = int(sel["final_physical_count"])

    if final_physical_count == 0 or selected_indices.size == 0:
        raise RuntimeError("未发现不过阈值的 Physical 信号，无法进行 PN-cut 可视化选择。")

    print(f"不过阈值 Physical 事件数: {final_physical_count}")

    # 读取这些事件的波形，并计算 CH0/CH1 的最大值
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

    x_min, x_max = x_range
    y_min, y_max = y_range

    region_mask = (
        (max_ch0 >= x_min)
        & (max_ch0 <= x_max)
        & (max_ch1 >= y_min)
        & (max_ch1 <= y_max)
    )

    if not np.any(region_mask):
        raise RuntimeError(
            f"在给定范围内未找到任何事件："
            f"x_range=({x_min}, {x_max}), y_range=({y_min}, {y_max})"
        )

    xw = max_ch0[region_mask]
    yw = max_ch1[region_mask]

    # 在线性范围内做一次最简单的线性拟合
    a, b = np.polyfit(xw, yw, deg=1)
    residuals = yw - (a * xw + b)
    sigma = residuals.std(ddof=1) if residuals.size > 1 else 0.0

    print(f"\n线性拟合: max_ch1 ≈ {a:.6f} * max_ch0 + {b:.3f}")
    print(f"在选定范围内的事件数: {xw.size}")
    print(f"残差标准差 σ = {sigma:.3f}")

    if sigma <= 0.0:
        print("警告：σ ≤ 0，无法构造 ±1σ 带，将只使用矩形范围筛选。")
        sigma_mask = np.ones_like(max_ch0, dtype=bool)
    else:
        all_residuals = max_ch1 - (a * max_ch0 + b)
        sigma_mask = np.abs(all_residuals) <= sigma_factor * sigma

    final_mask = region_mask & sigma_mask
    event_ranks = np.where(final_mask)[0]  # 在 selected_indices 中的下标

    print(f"\n最终满足“范围 + ±{sigma_factor}σ”条件的事件数: {event_ranks.size}")

    if event_ranks.size == 0:
        raise RuntimeError("没有事件同时满足矩形范围和 ±σ 带条件。")

    return event_ranks, ch0_3_file_sel, ch5_file_sel


def visualize_pncut_waveforms(
    ch0_3_file: Optional[str] = None,
    ch5_file: Optional[str] = None,
    rt_cut: float = 6000.0,
    ch0_threshold: float = 16382.0,
    ch0_idx: int = 0,
    ch1_idx: int = 1,
    x_range: Tuple[float, float] = (1100.0, 1400.0),
    y_range: Tuple[float, float] = (1000.0, 2200.0),
    sigma_factor: float = 1.0,
    max_events_to_plot: int = 8,
    show_plot: bool = True,
) -> List[str]:
    """
    对符合 “特定范围 + ±1σ 带” 的事件，使用 visualization.py 的风格可视化 CH0 波形。

    返回：
        保存的图片路径列表。
    """
    event_ranks, ch0_3_file_used, ch5_file_used = _select_events_in_1sigma_band(
        ch0_3_file=ch0_3_file,
        ch5_file=ch5_file,
        rt_cut=rt_cut,
        ch0_threshold=ch0_threshold,
        ch0_idx=ch0_idx,
        ch1_idx=ch1_idx,
        x_range=x_range,
        y_range=y_range,
        sigma_factor=sigma_factor,
    )

    # 只取前 max_events_to_plot 个事件做可视化
    n_plot = min(max_events_to_plot, event_ranks.size)
    print(f"\n将对前 {n_plot} 个事件进行 CH0 波形参数化可视化。")

    saved_paths: List[str] = []
    for i in range(n_plot):
        rank = int(event_ranks[i])
        print(f"\n[{i+1}/{n_plot}] 可视化 event_rank = {rank}")
        path = visualize_single_pulse_parameters(
            ch0_3_file=ch0_3_file_used,
            ch5_file=ch5_file_used,
            rt_cut=rt_cut,
            ch0_threshold=ch0_threshold,
            ch0_idx=ch0_idx,
            ch5_idx=ch1_idx,
            event_rank=rank,
            save_path=None,
            show_plot=show_plot,
        )
        saved_paths.append(path)

    return saved_paths


if __name__ == "__main__":
    try:
        # 示例：自动选择文件对，使用默认阈值和默认 PN 范围，对前若干个事件做可视化
        paths = visualize_pncut_waveforms(
            ch0_3_file=None,
            ch5_file=None,
            rt_cut=6000.0,
            ch0_threshold=16382.0,
            ch0_idx=0,
            ch1_idx=1,
            x_range=(1100.0, 1400.0),
            y_range=(1000.0, 2200.0),
            sigma_factor=1.0,
            max_events_to_plot=8,
            show_plot=True,
        )
        print("\n保存的图片路径：")
        for p in paths:
            print("  ", p)
    except Exception as e:
        print(f"\nPN-cut 可视化失败: {e}")
        import traceback

        traceback.print_exc()

