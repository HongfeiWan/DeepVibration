#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将未过阈值的锗自触发事件 CH0 波形的“逐个堆叠显示过程”做成 GIF 动画（约 60 fps）。

逻辑与 stack.py 中交互式堆叠类似：
1. 使用 overthreshold.py 中的 select_physical_events_no_overthreshold：
   - 先筛选：既非 RT 也非 Inhibit 的 Physical 事件，且 CH0 最大值 <= 阈值（不过阈值）。
2. 只读取这些事件的 CH0 波形。
3. 生成动画帧：
   - 第 1 帧：只显示第 1 个事件（alpha=1.0）。
   - 第 2 帧：第 1 个事件变为透明 (alpha≈0.2)，叠加第 2 个事件 (alpha=1.0)。
   - ...
   - 每一帧都在同一幅图上累积显示，最新事件最清晰，旧的事件较透明。
4. 以 ~60 fps 保存为 GIF（使用 PillowWriter）。
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from typing import Optional
import importlib.util
from datetime import datetime


# -----------------------------------------------------------------------------
# 导入 overthreshold.py 中的筛选函数
# -----------------------------------------------------------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))

overthreshold_path = os.path.join(current_dir, "overthreshold.py")
spec_over = importlib.util.spec_from_file_location("overthreshold_module", overthreshold_path)
overthreshold_module = importlib.util.module_from_spec(spec_over)
spec_over.loader.exec_module(overthreshold_module)

select_physical_events_no_overthreshold = overthreshold_module.select_physical_events_no_overthreshold


def create_stack_ch0_gif(
    ch0_3_file: Optional[str] = None,
    ch5_file: Optional[str] = None,
    rt_cut: float = 6000.0,
    ch0_threshold: float = 16382.0,
    ch0_idx: int = 0,
    ch5_idx: int = 0,
    max_events_for_gif: Optional[int] = 200,
    fps: int = 60,
    save_path: Optional[str] = None,
) -> str:
    """
    生成未过阈值 Physical 事件 CH0 波形堆叠过程的 GIF。

    参数:
        ch0_3_file: CH0-3 文件路径，如果为 None 则由 select_physical_events_no_overthreshold 自动选择。
        ch5_file: CH5 文件路径，如果为 None 则由 select_physical_events_no_overthreshold 自动选择。
        rt_cut: RT 截断阈值。
        ch0_threshold: CH0 最大值阈值（不过阈值条件）。
        ch0_idx: CH0 通道索引。
        ch5_idx: CH5 通道索引。
        max_events_for_gif: 为避免 GIF 过长，可限制最多使用的事件数量（None 表示使用全部）。
        fps: GIF 帧率（缺省 60 fps）。
        save_path: GIF 保存路径；None 时自动保存到项目根目录下 images/presentation。

    返回:
        实际保存的 GIF 文件路径。
    """
    print("=" * 70)
    print("生成 CH0 波形堆叠过程 GIF（未过阈值 Physical 事件）")
    print("=" * 70)

    # 1. 筛选不过阈值的 Physical 事件（与 stack.py 相同接口）
    selection_result = select_physical_events_no_overthreshold(
        ch0_3_file=ch0_3_file,
        ch5_file=ch5_file,
        rt_cut=rt_cut,
        ch0_threshold=ch0_threshold,
        ch0_idx=ch0_idx,
        ch5_idx=ch5_idx,
    )

    ch0_3_file = selection_result["ch0_3_file"]
    selected_indices = selection_result["selected_indices"]
    final_physical_count = selection_result["final_physical_count"]

    if final_physical_count == 0 or selected_indices.size == 0:
        raise RuntimeError("未发现不过阈值的 Physical 信号，无法生成 GIF。")

    print(
        f"\n共找到 {final_physical_count} 个不过阈值的 Physical 事件，"
        f"将用于 CH0 波形堆叠 GIF 生成。"
    )

    # 限制事件数量，避免 GIF 帧数过大
    if max_events_for_gif is not None and max_events_for_gif > 0:
        if selected_indices.size > max_events_for_gif:
            print(f"  仅使用前 {max_events_for_gif} 个事件生成 GIF（其余事件忽略）。")
            selected_indices = selected_indices[:max_events_for_gif]

    n_events = selected_indices.size

    # 2. 读取 CH0 波形数据
    with h5py.File(ch0_3_file, "r") as f_ch0:
        ch0_channel_data = f_ch0["channel_data"]
        time_samples, num_channels, num_events = ch0_channel_data.shape

        if ch0_idx >= num_channels:
            raise ValueError(
                f"ch0_idx={ch0_idx} 超过通道数 {num_channels}，无法读取 CH0 波形"
            )

        waveforms = ch0_channel_data[:, ch0_idx, selected_indices].astype(np.float64)

    # waveforms: (time_samples, n_events)
    time_samples = waveforms.shape[0]

    # 时间轴（采样间隔 4ns，单位 μs，与 overthreshold.py / stack.py 保持一致）
    sampling_interval_ns = 4.0
    time_axis_us = np.arange(time_samples) * sampling_interval_ns / 1000.0

    # 统一 y 轴范围（基于所有使用的事件）
    global_min = float(np.min(waveforms))
    global_max = float(np.max(waveforms))
    data_range = global_max - global_min
    if data_range > 0:
        margin = data_range * 0.1
        y_min = global_min - margin
        y_max = global_max + margin
    else:
        center = (global_min + global_max) / 2.0
        margin = max(abs(center) * 0.1, 100.0)
        y_min = center - margin
        y_max = center + margin

    # 3. 创建图形，用于动画
    fig, ax = plt.subplots(figsize=(10, 6))
    lines = []  # 存储 Line2D 对象

    ax.set_xlabel("Time (μs)", fontsize=12)
    ax.set_ylabel("Amplitude (ADC)", fontsize=12)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)

    title_text = ax.set_title(
        f"CH0 Waveform Stack (Non-overthreshold Physical Events)\n"
        f"Total events: {n_events}, Current: 1 / {n_events}",
        fontsize=13,
    )

    legend_handle = None

    # 初始化函数
    def init():
        nonlocal lines, legend_handle
        ax.cla()
        ax.set_xlabel("Time (μs)", fontsize=12)
        ax.set_ylabel("Amplitude (ADC)", fontsize=12)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        lines = []
        legend_handle = None

        wf0 = waveforms[:, 0]
        line0, = ax.plot(
            time_axis_us,
            wf0,
            color="C0",
            linewidth=1.0,
            alpha=1.0,
            label=f"Event #{int(selected_indices[0])}",
        )
        lines.append(line0)
        ax.set_title(
            f"CH0 Waveform Stack (Non-overthreshold Physical Events)\n"
            f"Total events: {n_events}, Current: 1 / {n_events}",
            fontsize=13,
        )
        legend_handle = ax.legend(
            [line0],
            [line0.get_label()],
            loc="upper right",
            fontsize=9,
        )
        return lines

    # 更新函数：第 i 帧对应第 i 个事件（从 0 开始）
    def update(frame_idx):
        nonlocal legend_handle

        # 保证 frame_idx 不越界
        if frame_idx >= n_events:
            frame_idx = n_events - 1

        # 旧线变透明
        for ln in lines:
            ln.set_alpha(0.2)

        # 新线
        wf = waveforms[:, frame_idx]
        line_new, = ax.plot(
            time_axis_us,
            wf,
            color="C0",
            linewidth=1.0,
            alpha=1.0,
            label=f"Event #{int(selected_indices[frame_idx])}",
        )
        lines.append(line_new)

        # 更新标题
        ax.set_title(
            f"CH0 Waveform Stack (Non-overthreshold Physical Events)\n"
            f"Total events: {n_events}, Current: {frame_idx + 1} / {n_events}",
            fontsize=13,
        )

        # 更新图例（只显示当前事件）
        if legend_handle is not None:
            legend_handle.remove()
        legend_handle = ax.legend(
            [line_new], [line_new.get_label()], loc="upper right", fontsize=9
        )

        return lines + [legend_handle]

    # 4. 创建动画
    interval_ms = 1000.0 / float(fps)
    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_events,
        interval=interval_ms,
        blit=False,
        repeat=False,
    )

    # 5. 生成保存路径
    if save_path is None:
        # 推断项目根目录：.../python/data/ge-self/cut -> python -> 项目根
        data_dir = os.path.dirname(os.path.dirname(current_dir))  # python/data
        python_dir = os.path.dirname(data_dir)                   # python
        project_root = os.path.dirname(python_dir)               # 项目根目录

        output_dir = os.path.join(project_root, "images", "presentation")
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stack_ch0_non_overthreshold_{n_events}events_{fps}fps_{timestamp}.gif"
        save_path = os.path.join(output_dir, filename)

    print(f"\n开始保存 GIF: {save_path}")
    writer = animation.PillowWriter(fps=fps)
    anim.save(save_path, writer=writer)
    plt.close(fig)
    print("GIF 保存完成。")

    return save_path


if __name__ == "__main__":
    try:
        gif_path = create_stack_ch0_gif(
            ch0_3_file=None,
            ch5_file=None,
            rt_cut=6000.0,
            ch0_threshold=16382.0,
            ch0_idx=0,
            ch5_idx=0,
            max_events_for_gif=10000,  # 防止事件太多导致 GIF 过长，可按需调整
            fps=60,
            save_path=None,          # 自动保存到 images/presentation
        )
        print(f"\nGIF 已保存到: {gif_path}")
    except Exception as e:
        print(f"生成 GIF 失败: {e}")
        import traceback

        traceback.print_exc()

