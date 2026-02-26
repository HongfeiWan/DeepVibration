#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PN-cut 选中事件的 CH0 波形可视化脚本。

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
from mpl_toolkits.axes_grid1 import make_axes_locatable
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


# 导入 visualization.py 中的函数
visualization_path = os.path.join(current_dir, "visualization.py")
spec_vis = importlib.util.spec_from_file_location("pulse_visualization_module", visualization_path)
pulse_vis_module = importlib.util.module_from_spec(spec_vis)
assert spec_vis.loader is not None
spec_vis.loader.exec_module(pulse_vis_module)

compute_pulse_parameters = pulse_vis_module.compute_pulse_parameters


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
    ch0_idx: int = 0,
    baseline_window_us: float = 2.0,
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> str:
    """
    直接使用事件索引（全局事件号）来可视化 CH0 波形，避免重复筛选。

    参数：
        ch0_3_file: CH0-3 文件路径
        event_index: 全局事件索引
        ch0_idx: CH0 通道索引
        baseline_window_us: 前沿和后沿基线的时间窗口长度（微秒），默认 2.0 µs
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
        ch0_channel_data = f_ch0["channel_data"]
        time_samples, num_channels, num_events = ch0_channel_data.shape

        if ch0_idx >= num_channels:
            raise ValueError(
                f"ch0_idx={ch0_idx} 超过通道数 {num_channels}，无法读取 CH0 波形"
            )
        if event_index >= num_events:
            raise IndexError(f"event_index={event_index} 超过事件数 {num_events}")

        waveform = ch0_channel_data[:, ch0_idx, event_index].astype(np.float64)

    # 对波形先做一次中值滤波
    waveform = median_filter(waveform, kernel_size=3)

    # 时间轴：4 ns 采样，单位 µs
    sampling_interval_ns = 4.0
    time_axis_us = np.arange(waveform.shape[0]) * sampling_interval_ns / 1000.0

    # 计算参数
    params = compute_pulse_parameters(
        waveform, 
        sampling_interval_ns=sampling_interval_ns,
        baseline_window_us=baseline_window_us
    )

    ped = params["ped"]
    pedt = params["pedt"]
    amax = params["amax"]
    tmax_us = params["tmax_us"]
    idx_max = params["idx_max"]
    ammin = params["ammin"]
    q = params["q"]

    # baseline_window_us µs 对应的采样点数
    baseline_window_ns = baseline_window_us * 1000.0  # 转换为纳秒
    samples_baseline = int(round(baseline_window_ns / sampling_interval_ns))
    samples_baseline = max(1, min(samples_baseline, waveform.shape[0] // 2))

    t_ped_start = time_axis_us[0]
    t_ped_end = time_axis_us[samples_baseline - 1]
    t_pedt_start = time_axis_us[-samples_baseline]
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
    
    # 计算文字标签的 y 轴偏移量（基于 y 轴范围的一定比例）
    y_text_offset = (y_max - y_min) * 0.1

    # 画图
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
    )

    # Ped 区域
    ax.axvspan(
        t_ped_start,
        t_ped_end,
        color="tab:green",
        alpha=0.15,
    )
    # Ped 虚线：从 Ped 区域最尾部时间到最左侧 y 轴
    ax.plot(
        [time_axis_us[0], t_ped_end],
        [ped, ped],
        color="tab:green",
        linestyle="--",
        linewidth=0.8,
        alpha=0.8,
    )
    ax.text(
        time_axis_us[0],
        ped + y_text_offset,
        "Ped",
        color="tab:green",
        fontsize=18,
        va="bottom",
        ha="left",
    )

    # Pedt 区域
    ax.axvspan(
        t_pedt_start,
        t_pedt_end,
        color="tab:orange",
        alpha=0.15,
    )
    # Pedt 虚线
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
        pedt - y_text_offset,
        "Pedt",
        color="tab:orange",
        fontsize=18,
        va="bottom",
        ha="right",
    )

    # Ped 到 Pedt 之间的双箭头（放在最左侧）
    arrow_x = time_axis_us[0]  # 箭头位置在最左侧
    arrow_mid_y = (ped + pedt) / 2.0  # Ped 和 Pedt 的中间位置
    ax.annotate(
        "",
        xy=(arrow_x, pedt),  # 箭头终点（Pedt）
        xytext=(arrow_x, ped),  # 箭头起点（Ped）
        arrowprops=dict(
            arrowstyle="<->",
            color="black",
            lw=1.5,
            shrinkA=0,
            shrinkB=0,
        ),
    )
    # 标注文字 "pedt-ped"（放在箭头左侧，旋转与 y 轴平行）
    ax.text(
        arrow_x,  # 箭头
        arrow_mid_y,
        "pedt-ped",
        color="black",
        fontsize=18,
        va="center",
        ha="right",
        rotation=90,  # 旋转 90 度，与 y 轴平行
    )

    # Amax / Tmax
    ax.scatter(
        [tmax_us],
        [amax],
        color="red",
        s=40,
        zorder=5,
    )
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
    )
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

    # Q 的积分区域
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

    # 除了Q区域外，波形和pedt围合的区域用细虚线填充
    non_q_mask = ~q_mask  # Q区域之外的区域
    
    # 小于Tmax的部分：正灰色斜线
    before_tmax_mask = non_q_mask & (time_axis_us < tmax_us)
    ax.fill_between(
        time_axis_us,
        waveform,
        pedt,
        where=before_tmax_mask,  # 小于Tmax且不属于Q的区域
        color="gray",
        alpha=0.1,
        hatch="///",  # 正斜线填充
        edgecolor="gray",
        linewidth=0.5,
    )
    # 在小于Tmax的部分添加文字标签 "Qpre"
    if np.any(before_tmax_mask):
        before_tmax_indices = np.where(before_tmax_mask)[0]
        before_tmax_x_center = time_axis_us[before_tmax_indices[len(before_tmax_indices) // 2]]
        before_tmax_y_center = pedt 
        ax.text(
            before_tmax_x_center,
            before_tmax_y_center,
            "Qpre",
            color="black",
            fontsize=18,
            ha="center",
            va="center",
        )
    
    # 大于Tmax的部分：反灰色斜线
    after_tmax_mask = non_q_mask & (time_axis_us > tmax_us)
    ax.fill_between(
        time_axis_us,
        waveform,
        pedt,
        where=after_tmax_mask,  # 大于Tmax且不属于Q的区域
        color="gray",
        alpha=0.1,
        hatch="\\\\\\",  # 反斜线填充
        edgecolor="gray",
        linewidth=0.5,
    )
    # 在大于Tmax的部分添加文字标签 "Qprev"
    if np.any(after_tmax_mask):
        after_tmax_indices = np.where(after_tmax_mask)[0]
        after_tmax_x_center = time_axis_us[after_tmax_indices[len(after_tmax_indices) // 2]]
        after_tmax_y_center = pedt 
        ax.text(
            after_tmax_x_center,
            after_tmax_y_center,
            "Qprev",
            color="black",
            fontsize=18,
            ha="center",
            va="center",
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
        f"CH0 Pulse Parameterization\n"
        f"{filename}  |  Event #{event_index}",
        fontsize=13,
        fontweight="bold",
    )

    # 在坐标轴附近标出 Amax / Tmax / Amin / Tmin
    ax.text(
        time_axis_us[0],
        amax,
        f"Amax",
        color="red",
        fontsize=18,
        va="bottom",
        ha="left",
    )
    ax.text(
        tmax_us,
        y_min,
        f"Tmax",
        color="red",
        fontsize=18,
        va="bottom",
        ha="left",
    )
    ax.text(
        time_axis_us[0],
        ammin,
        f"Amin",
        color="purple",
        fontsize=18,
        va="top",
        ha="left",
    )
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

    # 新建单独窗口：仅显示基线区域（Ped 和 Pedt）的放大图
    fig_baseline, (ax_ped, ax_pedt) = plt.subplots(1, 2, figsize=(12, 5))
    fig_baseline.suptitle(
        f"Baseline Regions  |  {os.path.basename(ch0_3_file)}  |  Event #{event_index}",
        fontsize=13,
        fontweight="bold",
    )

    # 左图：Ped 前沿基线区域
    t_ped_zoom = time_axis_us[:samples_baseline]
    w_ped_zoom = waveform[:samples_baseline]
    ax_ped.plot(t_ped_zoom, w_ped_zoom, color="C0", linewidth=1)
    # Ped 区域线性拟合虚线
    k_ped, b_ped = np.polyfit(t_ped_zoom, w_ped_zoom, 1)
    ped_fit = k_ped * t_ped_zoom + b_ped
    ax_ped.plot(
        t_ped_zoom, ped_fit,
        color="tab:green",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
    )
    ax_ped.set_xlabel("Time (µs)", fontsize=14, fontweight="bold")
    ax_ped.set_ylabel("Amplitude (ADC)", fontsize=14, fontweight="bold")
    ax_ped.set_title(f"Ped Region (ped = {ped:.2f})", fontsize=12, fontweight="bold")
    ax_ped.grid(True, alpha=0.3)
    # Ped 基线值的分布（右侧直方图）
    div_ped = make_axes_locatable(ax_ped)
    ax_hist_ped = div_ped.append_axes("right", size="25%", pad=0.15, sharey=ax_ped)
    ax_hist_ped.tick_params(axis="y", labelleft=False)
    ax_hist_ped.hist(
        w_ped_zoom, bins=min(20, max(5, samples_baseline // 2)),
        orientation="horizontal", color="tab:green", alpha=0.6, edgecolor="tab:green",
    )
    ax_hist_ped.set_xlabel("Count", fontsize=11)
    ax_hist_ped.axhline(ped, color="tab:green", linestyle="--", linewidth=1, alpha=0.9)
    ax_hist_ped.set_title("Dist.", fontsize=10)
    std_ped = float(np.std(w_ped_zoom))
    ax_ped.text(
        0.02, 0.98,
        f"μ={ped:.2f}  σ={std_ped:.2f}\nmin={w_ped_zoom.min():.1f}  max={w_ped_zoom.max():.1f}",
        transform=ax_ped.transAxes, fontsize=10, va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # 右图：Pedt 后沿基线区域
    t_pedt_zoom = time_axis_us[-samples_baseline:]
    w_pedt_zoom = waveform[-samples_baseline:]
    ax_pedt.plot(t_pedt_zoom, w_pedt_zoom, color="C0", linewidth=1)
    # Pedt 区域线性拟合虚线
    k_pedt, b_pedt = np.polyfit(t_pedt_zoom, w_pedt_zoom, 1)
    pedt_fit = k_pedt * t_pedt_zoom + b_pedt
    ax_pedt.plot(
        t_pedt_zoom, pedt_fit,
        color="tab:orange",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
    )
    ax_pedt.set_xlabel("Time (µs)", fontsize=14, fontweight="bold")
    ax_pedt.set_ylabel("Amplitude (ADC)", fontsize=14, fontweight="bold")
    ax_pedt.set_title(f"Pedt Region (pedt = {pedt:.2f})", fontsize=12, fontweight="bold")
    ax_pedt.grid(True, alpha=0.3)
    # Pedt 基线值的分布（右侧直方图）
    div_pedt = make_axes_locatable(ax_pedt)
    ax_hist_pedt = div_pedt.append_axes("right", size="25%", pad=0.15, sharey=ax_pedt)
    ax_hist_pedt.tick_params(axis="y", labelleft=False)
    ax_hist_pedt.hist(
        w_pedt_zoom, bins=min(20, max(5, samples_baseline // 2)),
        orientation="horizontal", color="tab:orange", alpha=0.6, edgecolor="tab:orange",
    )
    ax_hist_pedt.set_xlabel("Count", fontsize=11)
    ax_hist_pedt.axhline(pedt, color="tab:orange", linestyle="--", linewidth=1, alpha=0.9)
    ax_hist_pedt.set_title("Dist.", fontsize=10)
    std_pedt = float(np.std(w_pedt_zoom))
    ax_pedt.text(
        0.02, 0.98,
        f"μ={pedt:.2f}  σ={std_pedt:.2f}\nmin={w_pedt_zoom.min():.1f}  max={w_pedt_zoom.max():.1f}",
        transform=ax_pedt.transAxes, fontsize=10, va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
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
            f"ch0_pulse_parameterization_event{event_index}_{timestamp}.png"
        )
        save_path = os.path.join(output_dir, filename_png)

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"参数化波形图已保存至: {save_path}")

    # 保存基线窗口图像（文件名加 _baseline 后缀）
    save_path_baseline = save_path.replace(".png", "_baseline.png")
    fig_baseline.savefig(
        save_path_baseline, dpi=300, bbox_inches="tight", facecolor="white"
    )
    print(f"基线区域图已保存至: {save_path_baseline}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig_baseline)

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
    max_events_to_plot: int = 8,
    show_plot: bool = True,
) -> List[str]:
    """
    对符合 lsmpncut PN-cut ±1σ 带的事件，使用 visualization.py 的风格可视化 CH0 波形。

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
    print(f"\n将对前 {n_plot} 个事件进行 CH0 波形参数化可视化。")

    saved_paths: List[str] = []
    for i in range(n_plot):
        rank = int(event_ranks[i])
        # 将 rank（在 selected_indices 中的下标）转换为全局事件索引
        event_index = int(selected_indices[rank])
        print(f"\n[{i+1}/{n_plot}] 可视化 event_rank = {rank} (全局 Event #{event_index})")
        path = _visualize_single_event_by_index(
            ch0_3_file=ch0_3_file_used,
            event_index=event_index,
            ch0_idx=ch0_idx,
            baseline_window_us=baseline_window_us,
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
            baseline_window_us=2.0,  # 前沿和后沿基线时间窗口（微秒）
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

