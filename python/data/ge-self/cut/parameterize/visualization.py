#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对未过阈值的锗自触发 CH0 波形做“参数化可视化”：

- 使用 cut/overthreshold.py 中的 select_physical_events_no_overthreshold 挑选一个不过阈值的 Physical 事件；
- 提取该事件的 CH0 波形，并计算/标注以下参数：
  * Ped  : 波形前沿基线，定义为采数时间窗口前 2 µs 的幅度平均值
  * Pedt : 波形尾部基线，定义为采数时间窗口最后 2 µs 的幅度平均值
  * Amax : 波形幅度最大值
  * Tmax : 达到最大值的时刻（单位 µs）
  * Ammin: 波形幅度最小值
  * Q    : CH0 波形的部分积分（对 Ped 扣除后的波形在全时间窗口上的积分，用于低能区能量刻度）

并在一幅图中高亮上述区域/特征点，并给出文字说明。
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
import importlib.util
from datetime import datetime


# -----------------------------------------------------------------------------
# 导入 overthreshold.py 中的筛选函数
# -----------------------------------------------------------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))          # .../cut/parameterize
cut_dir = os.path.dirname(current_dir)                            # .../cut

overthreshold_path = os.path.join(cut_dir, "overthreshold.py")
spec_over = importlib.util.spec_from_file_location("overthreshold_module", overthreshold_path)
overthreshold_module = importlib.util.module_from_spec(spec_over)
spec_over.loader.exec_module(overthreshold_module)

select_physical_events_no_overthreshold = overthreshold_module.select_physical_events_no_overthreshold


def compute_pulse_parameters(
    waveform: np.ndarray,
    sampling_interval_ns: float = 4.0,
) -> Dict[str, Any]:
    """
    对单条 CH0 波形做参数化，返回 Ped, Pedt, Amax, Tmax, Ammin, Q 等。

    参数:
        waveform: 波形数组，形状 (time_samples,)
        sampling_interval_ns: 采样间隔（ns），默认 4 ns

    返回:
        字典，包含:
            - ped: 前 2 µs 基线平均值
            - pedt: 尾部 2 µs 基线平均值
            - amax: 最大幅度
            - tmax_us: 达到最大幅度的时间 (µs)
            - idx_max: 最大幅度对应的采样点索引
            - ammin: 最小幅度
            - q: 部分积分（以 Pedt 为基线，在峰值两侧首次回到 Pedt 高度之间的积分）
            - q_t_left_us: Q 左边积分区间起点时间 (µs)
            - q_t_right_us: Q 右边积分区间终点时间 (µs)
    """
    wf = np.asarray(waveform, dtype=np.float64).ravel()
    n_samples = wf.shape[0]

    # 2 µs 对应的采样点数: 2e-6 / (4e-9) = 500
    samples_2us = int(round(2e-6 / (sampling_interval_ns * 1e-9)))
    samples_2us = max(1, min(samples_2us, n_samples // 2))

    ped = float(np.mean(wf[:samples_2us]))
    pedt = float(np.mean(wf[-samples_2us:]))

    idx_max = int(np.argmax(wf))
    amax = float(wf[idx_max])
    tmax_us = idx_max * sampling_interval_ns / 1000.0

    ammin = float(np.min(wf))

    # 计算 Q 的积分区间：以 Pedt 为基线，从峰值向两侧第一次回到 Pedt 高度的位置
    # 向左寻找
    left_idx = idx_max
    while left_idx > 0 and wf[left_idx] > pedt:
        left_idx -= 1
    # 向右寻找
    right_idx = idx_max
    while right_idx < n_samples - 1 and wf[right_idx] > pedt:
        right_idx += 1

    # 防御性处理，避免 left_idx >= right_idx 的极端情况
    if left_idx >= right_idx:
        left_idx = 0
        right_idx = n_samples - 1

    dt_s = sampling_interval_ns * 1e-9
    q = float(np.sum(wf[left_idx : right_idx + 1] - pedt) * dt_s)

    q_t_left_us = left_idx * sampling_interval_ns / 1000.0
    q_t_right_us = right_idx * sampling_interval_ns / 1000.0

    return {
        "ped": ped,
        "pedt": pedt,
        "amax": amax,
        "tmax_us": tmax_us,
        "idx_max": idx_max,
        "ammin": ammin,
        "q": q,
        "q_t_left_us": q_t_left_us,
        "q_t_right_us": q_t_right_us,
        "n_samples": n_samples,
        "sampling_interval_ns": sampling_interval_ns,
    }


def visualize_single_pulse_parameters(
    ch0_3_file: Optional[str] = None,
    ch5_file: Optional[str] = None,
    rt_cut: float = 6000.0,
    ch0_threshold: float = 16382.0,
    ch0_idx: int = 0,
    ch5_idx: int = 0,
    event_rank: int = 0,
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> str:
    """
    使用 select_physical_events_no_overthreshold 挑选一个“不过阈值”的 Physical 事件，
    对其 CH0 波形做参数化并可视化。

    参数:
        ch0_3_file: CH0-3 文件路径（None 时由筛选函数自动选择）
        ch5_file: CH5 文件路径（None 时由筛选函数自动选择）
        rt_cut: RT 截断阈值
        ch0_threshold: CH0 最大值阈值（不过阈值条件）
        ch0_idx: CH0 通道索引
        ch5_idx: CH5 通道索引
        event_rank: 在筛选出的事件中的“第几个”（0 为第一个）
        save_path: 保存图片路径，None 时自动生成
        show_plot: 是否调用 plt.show()

    返回:
        实际保存的图片路径。
    """
    print("=" * 70)
    print("参数化可视化：单条 CH0 波形（未过阈值 Physical 事件）")
    print("=" * 70)

    # 1. 先筛选不过阈值的 Physical 事件
    sel = select_physical_events_no_overthreshold(
        ch0_3_file=ch0_3_file,
        ch5_file=ch5_file,
        rt_cut=rt_cut,
        ch0_threshold=ch0_threshold,
        ch0_idx=ch0_idx,
        ch5_idx=ch5_idx,
    )

    ch0_3_file = sel["ch0_3_file"]
    selected_indices = sel["selected_indices"]
    final_physical_count = int(sel["final_physical_count"])

    if final_physical_count == 0 or selected_indices.size == 0:
        raise RuntimeError("未发现不过阈值的 Physical 信号，无法进行参数化可视化。")

    if event_rank < 0 or event_rank >= selected_indices.size:
        raise IndexError(
            f"event_rank={event_rank} 超出范围 [0, {selected_indices.size-1}]。"
        )

    event_index = int(selected_indices[event_rank])
    print(
        f"\n共找到 {final_physical_count} 个不过阈值 Physical 事件，"
        f"当前选择第 {event_rank} 个（全局 Event #{event_index}）。"
    )

    # 2. 打开 HDF5，取该事件的 CH0 波形
    with h5py.File(ch0_3_file, "r") as f_ch0:
        ch0_channel_data = f_ch0["channel_data"]
        time_samples, num_channels, num_events = ch0_channel_data.shape

        if ch0_idx >= num_channels:
            raise ValueError(
                f"ch0_idx={ch0_idx} 超过通道数 {num_channels}，无法读取 CH0 波形"
            )

        waveform = ch0_channel_data[:, ch0_idx, event_index].astype(np.float64)

    # 时间轴：4 ns 采样，单位 µs
    sampling_interval_ns = 4.0
    time_axis_us = np.arange(waveform.shape[0]) * sampling_interval_ns / 1000.0

    # 3. 计算参数
    params = compute_pulse_parameters(
        waveform, sampling_interval_ns=sampling_interval_ns
    )

    ped = params["ped"]
    pedt = params["pedt"]
    amax = params["amax"]
    tmax_us = params["tmax_us"]
    idx_max = params["idx_max"]
    ammin = params["ammin"]
    q = params["q"]

    # 2 µs 对应的采样点数，用于画 Ped/Pedt 区域
    samples_2us = int(round(2e-6 / (sampling_interval_ns * 1e-9)))
    samples_2us = max(1, min(samples_2us, waveform.shape[0] // 2))

    t_ped_start = time_axis_us[0]
    t_ped_end = time_axis_us[samples_2us - 1]
    t_pedt_start = time_axis_us[-samples_2us]
    t_pedt_end = time_axis_us[-1]

    # 统一 y 轴范围
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

    # 4. 画图
    # 设置全局字体为 Times New Roman
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
        }
    )
    fig, ax = plt.subplots(figsize=(10, 6))

    # 原始波形
    ax.plot(
        time_axis_us,
        waveform,
        color="C0",
        linewidth=1,
        label="CH0 waveform",
    )

    # Ped 区域（前 2 µs，仅高亮时间窗口，不再画水平虚线）
    ax.axvspan(
        t_ped_start,
        t_ped_end,
        color="tab:green",
        alpha=0.15,
        label="Ped region (first 2 µs)",
    )

    # Pedt 区域（最后 2 µs，仅高亮时间窗口，不再画水平虚线）
    ax.axvspan(
        t_pedt_start,
        t_pedt_end,
        color="tab:orange",
        alpha=0.15,
        label="Pedt region (last 2 µs)",
    )
    # 在 Pedt 高度画一条水平虚线直到 y 轴（整个时间范围），并标注 "Pedt"
    ax.plot(
        [time_axis_us[0], time_axis_us[-1]],
        [pedt, pedt],
        color="tab:orange",
        linestyle="--",
        linewidth=0.8,
        alpha=0.8,
    )
    ax.text(
        time_axis_us[-1],
        pedt,
        "Pedt",
        color="tab:orange",
        fontsize=18,
        va="bottom",
        ha="right",
    )

    # Amax / Tmax
    ax.scatter(
        [tmax_us],
        [amax],
        color="red",
        s=40,
        zorder=5,
        label=f"Amax = {amax:.1f} at Tmax = {tmax_us:.3f} µs",
    )
    # Amax 到坐标轴的虚线（垂直和水平）
    ax.plot(
        [tmax_us, tmax_us],
        [y_min, amax],
        color="red",
        linestyle="--",
        linewidth=0.8,
        alpha=0.8,
    )
    ax.plot(
        [time_axis_us[0], tmax_us],
        [amax, amax],
        color="red",
        linestyle="--",
        linewidth=0.8,
        alpha=0.8,
    )

    # Ammin
    idx_min = int(np.argmin(waveform))
    tmin_us = time_axis_us[idx_min]
    ax.scatter(
        [tmin_us],
        [ammin],
        color="purple",
        s=40,
        zorder=5,
        label=f"Ammin = {ammin:.1f} at {tmin_us:.3f} µs",
    )
    # Ammin 到坐标轴的虚线（垂直和水平）
    ax.plot(
        [tmin_us, tmin_us],
        [y_min, ammin],
        color="purple",
        linestyle="--",
        linewidth=0.8,
        alpha=0.8,
    )
    ax.plot(
        [time_axis_us[0], tmin_us],
        [ammin, ammin],
        color="purple",
        linestyle="--",
        linewidth=0.8,
        alpha=0.8,
    )

    # Q 的积分区域：以 Pedt 为基线，在峰值两侧首次回到 Pedt 高度之间
    q_t_left_us = params["q_t_left_us"]
    q_t_right_us = params["q_t_right_us"]
    q_mask = (time_axis_us >= q_t_left_us) & (time_axis_us <= q_t_right_us)
    ax.fill_between(
        time_axis_us,
        waveform,
        pedt,
        where=q_mask,
        color="tab:blue",
        alpha=0.10,
    )
    # 在积分区域上方标注 "Q"
    q_x_center = 0.5 * (q_t_left_us + q_t_right_us)
    q_y_label = pedt + 0.1 * (y_max - y_min)
    ax.text(
        q_x_center,
        q_y_label,
        "Q",
        color="tab:blue",
        fontsize=18,
        ha="center",
        va="bottom",
    )

    ax.set_xlabel("Time (µs)", fontsize=18, fontweight="bold")
    ax.set_ylabel("Amplitude (ADC)", fontsize=18, fontweight="bold")
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    # 坐标轴刻度数字加粗
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontweight("bold")
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontweight("bold")

    # 标题中写明文件和事件号
    filename = os.path.basename(ch0_3_file)
    ax.set_title(
        f"CH0 Pulse Parameterization\n"
        f"{filename}  |  Event #{event_index}",
        fontsize=13,
        fontweight="bold",
    )


    # 在坐标轴附近标出 Amax / Tmax / Amin / Tmin
    # Amax 在 y 轴附近
    ax.text(
        time_axis_us[0],
        amax,
        f"Amax",
        color="red",
        fontsize=18,
        va="bottom",
        ha="left",
    )
    # Tmax 在 x 轴附近
    ax.text(
        tmax_us,
        y_min,
        f"Tmax",
        color="red",
        fontsize=18,
        va="bottom",
        ha="left",
    )
    # Amin 在 y 轴附近
    ax.text(
        time_axis_us[0],
        ammin,
        f"Amin",
        color="purple",
        fontsize=18,
        va="top",
        ha="left",
    )
    # Tmin 在 x 轴附近
    ax.text(
        tmin_us,
        y_min,
        f"Tmin",
        color="purple",
        fontsize=18,
        va="bottom",
        ha="right",
    )

    plt.tight_layout()

    # 5. 保存图片路径
    if save_path is None:
        # 推断项目根目录：.../python/data/ge-self/cut/parameterize -> python -> 项目根
        ge_self_dir = os.path.dirname(cut_dir)     # .../ge-self
        data_dir = os.path.dirname(ge_self_dir)    # .../data
        python_dir = os.path.dirname(data_dir)     # .../python
        project_root = os.path.dirname(python_dir) # 项目根

        output_dir = os.path.join(project_root, "images", "presentation")
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_png = (
            f"ch0_pulse_parameterization_event{event_index}_{timestamp}.png"
        )
        save_path = os.path.join(output_dir, filename_png)

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\n参数化波形图已保存至: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return save_path


if __name__ == "__main__":
    try:
        visualize_single_pulse_parameters(
            ch0_3_file=None,
            ch5_file=None,
            rt_cut=6000.0,
            ch0_threshold=16382.0,
            ch0_idx=0,
            ch5_idx=0,
            event_rank=4,
            save_path=None,
            show_plot=True,
        )
    except Exception as e:
        print(f"参数化可视化失败: {e}")
        import traceback

        traceback.print_exc()
