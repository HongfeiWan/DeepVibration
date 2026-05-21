#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PN-cut 选中事件的 CH0 波形参数化脚本。

参数化维度（8 维）：
- Amax: 波形幅度最大值
- μ(ped): 前沿基线均值
- μ(pedt): 后沿基线均值
- σ(ped): 前沿基线标准差
- σ(pedt): 后沿基线标准差
- Tmax: 达到最大值的时刻（µs）
- RMS_ped: 前沿基线线性拟合的 RMS 均方根误差
- RMS_pedt: 后沿基线线性拟合的 RMS 均方根误差

可视化仅展示与上述 8 维参数相关的内容。
"""

import os
import sys
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import importlib.util


# -----------------------------------------------------------------------------
# 导入依赖
# -----------------------------------------------------------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))
cut_dir = os.path.dirname(current_dir)

overthreshold_path = os.path.join(cut_dir, "overthreshold.py")
spec_over = importlib.util.spec_from_file_location("overthreshold_module", overthreshold_path)
overthreshold_module = importlib.util.module_from_spec(spec_over)
assert spec_over.loader is not None
spec_over.loader.exec_module(overthreshold_module)

select_physical_events_no_overthreshold = overthreshold_module.select_physical_events_no_overthreshold

visualization_path = os.path.join(current_dir, "visualization.py")
spec_vis = importlib.util.spec_from_file_location("pulse_visualization_module", visualization_path)
pulse_vis_module = importlib.util.module_from_spec(spec_vis)
assert spec_vis.loader is not None
spec_vis.loader.exec_module(pulse_vis_module)

compute_pulse_parameters = pulse_vis_module.compute_pulse_parameters

lsmpncut_path = os.path.join(cut_dir, "lsmpncut.py")
spec_ls = importlib.util.spec_from_file_location("lsmpncut_module", lsmpncut_path)
lsmpncut_module = importlib.util.module_from_spec(spec_ls)
assert spec_ls.loader is not None
spec_ls.loader.exec_module(lsmpncut_module)

fit_single_line_in_range = lsmpncut_module.fit_single_line_in_range


def compute_eight_params(
    waveform: np.ndarray,
    sampling_interval_ns: float = 4.0,
    baseline_window_us: float = 2.0,
) -> Dict[str, Any]:
    """
    计算 8 维参数：Amax, μ(ped), μ(pedt), σ(ped), σ(pedt), Tmax, RMS_ped, RMS_pedt。

    返回:
        字典，包含: amax, mu_ped, mu_pedt, sigma_ped, sigma_pedt, tmax_us, rms_ped, rms_pedt
    """
    base = compute_pulse_parameters(
        waveform,
        sampling_interval_ns=sampling_interval_ns,
        baseline_window_us=baseline_window_us,
    )

    baseline_window_ns = baseline_window_us * 1000.0
    samples_baseline = int(round(baseline_window_ns / sampling_interval_ns))
    n_samples = waveform.shape[0]
    samples_baseline = max(1, min(samples_baseline, n_samples // 2))

    w_ped = waveform[:samples_baseline]
    w_pedt = waveform[-samples_baseline:]

    mu_ped = float(np.mean(w_ped))
    mu_pedt = float(np.mean(w_pedt))
    sigma_ped = float(np.std(w_ped)) if w_ped.size > 1 else 0.0
    sigma_pedt = float(np.std(w_pedt)) if w_pedt.size > 1 else 0.0

    time_axis = np.arange(samples_baseline) * sampling_interval_ns / 1000.0
    k_ped, b_ped = np.polyfit(time_axis, w_ped, 1)
    ped_fit = k_ped * time_axis + b_ped
    rms_ped = float(np.sqrt(np.mean((w_ped - ped_fit) ** 2)))

    t_pedt = np.arange(samples_baseline) * sampling_interval_ns / 1000.0
    k_pedt, b_pedt = np.polyfit(t_pedt, w_pedt, 1)
    pedt_fit = k_pedt * t_pedt + b_pedt
    rms_pedt = float(np.sqrt(np.mean((w_pedt - pedt_fit) ** 2)))

    return {
        "amax": base["amax"],
        "mu_ped": mu_ped,
        "mu_pedt": mu_pedt,
        "sigma_ped": sigma_ped,
        "sigma_pedt": sigma_pedt,
        "tmax_us": base["tmax_us"],
        "rms_ped": rms_ped,
        "rms_pedt": rms_pedt,
    }


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
) -> Tuple[np.ndarray, str, str, np.ndarray]:
    """在不过阈值 Physical 事件中，选出落在 PN-cut ±1σ 线性带内的事件。"""
    print("=" * 70)
    print("PN-cut 可视化：使用 lsmpncut 逻辑选择 ±1σ 带内的事件")
    print("=" * 70)

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

    with h5py.File(ch0_3_file_sel, "r") as f_ch0:
        channel_data = f_ch0["channel_data"]
        time_samples, num_channels, num_events = channel_data.shape

        if ch0_idx >= num_channels or ch1_idx >= num_channels:
            raise ValueError(
                f"通道索引超出范围：ch0_idx={ch0_idx}, ch1_idx={ch1_idx}, num_channels={num_channels}"
            )

        phys_ch0 = channel_data[:, ch0_idx, selected_indices].astype(np.float64)
        phys_ch1 = channel_data[:, ch1_idx, selected_indices].astype(np.float64)

    max_ch0 = phys_ch0.max(axis=0)
    max_ch1 = phys_ch1.max(axis=0)

    print(f"\n在 {x_min} < max_ch0 < {x_max} 范围内使用 lsmpncut 进行两步最小二乘拟合...")
    a, b = fit_single_line_in_range(max_ch0, max_ch1, x_min=x_min, x_max=x_max)

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

    predicted_all = a * max_ch0 + b
    all_residuals = max_ch1 - predicted_all
    sigma_mask = np.abs(all_residuals) <= sigma_factor * sigma
    event_ranks = np.where(sigma_mask)[0]

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
    可视化 CH0 波形的 8 维参数：Amax, Tmax, Ped(μ,σ,RMS), Pedt(μ,σ,RMS)。
    """
    python_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)
    from utils.filter import median_filter

    with h5py.File(ch0_3_file, "r") as f_ch0:
        ch0_channel_data = f_ch0["channel_data"]
        time_samples, num_channels, num_events = ch0_channel_data.shape

        if ch0_idx >= num_channels:
            raise ValueError(f"ch0_idx={ch0_idx} 超过通道数 {num_channels}，无法读取 CH0 波形")
        if event_index >= num_events:
            raise IndexError(f"event_index={event_index} 超过事件数 {num_events}")

        waveform = ch0_channel_data[:, ch0_idx, event_index].astype(np.float64)

    waveform = median_filter(waveform, kernel_size=3)

    sampling_interval_ns = 4.0
    time_axis_us = np.arange(waveform.shape[0]) * sampling_interval_ns / 1000.0

    params = compute_eight_params(
        waveform,
        sampling_interval_ns=sampling_interval_ns,
        baseline_window_us=baseline_window_us,
    )

    amax = params["amax"]
    tmax_us = params["tmax_us"]
    mu_ped = params["mu_ped"]
    mu_pedt = params["mu_pedt"]
    sigma_ped = params["sigma_ped"]
    sigma_pedt = params["sigma_pedt"]
    rms_ped = params["rms_ped"]
    rms_pedt = params["rms_pedt"]

    baseline_window_ns = baseline_window_us * 1000.0
    samples_baseline = int(round(baseline_window_ns / sampling_interval_ns))
    samples_baseline = max(1, min(samples_baseline, waveform.shape[0] // 2))

    t_ped_start = time_axis_us[0]
    t_ped_end = time_axis_us[samples_baseline - 1]
    t_pedt_start = time_axis_us[-samples_baseline]
    t_pedt_end = time_axis_us[-1]

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

    y_text_offset = (y_max - y_min) * 0.1

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
    })

    # 主图：波形 + 7 维参数相关标注
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(time_axis_us, waveform, color="C0", linewidth=1)

    # Ped 区域
    ax.axvspan(t_ped_start, t_ped_end, color="tab:green", alpha=0.15)
    ax.plot(
        [time_axis_us[0], t_ped_end],
        [mu_ped, mu_ped],
        color="tab:green",
        linestyle="--",
        linewidth=0.8,
        alpha=0.8,
    )
    ax.text(
        time_axis_us[0], mu_ped + y_text_offset,
        f"μ(ped)={mu_ped:.1f} σ={sigma_ped:.2f}",
        color="tab:green", fontsize=12, va="bottom", ha="left",
    )

    # Pedt 区域
    ax.axvspan(t_pedt_start, t_pedt_end, color="tab:orange", alpha=0.15)
    ax.plot(
        [time_axis_us[0], time_axis_us[-1]],
        [mu_pedt, mu_pedt],
        color="tab:orange",
        linestyle="--",
        linewidth=0.8,
        alpha=0.8,
    )
    ax.text(
        time_axis_us[-1], mu_pedt - y_text_offset,
        f"μ(pedt)={mu_pedt:.1f} σ={sigma_pedt:.2f}",
        color="tab:orange", fontsize=12, va="top", ha="right",
    )

    # Amax / Tmax
    ax.scatter([tmax_us], [amax], color="red", s=40, zorder=5)
    ax.plot([tmax_us, tmax_us], [y_min, amax], color="red", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.plot([time_axis_us[0], tmax_us], [amax, amax], color="red", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.text(time_axis_us[0], amax, "Amax", color="red", fontsize=14, va="bottom", ha="left")
    ax.text(tmax_us, y_min, f"Tmax={tmax_us:.3f}µs", color="red", fontsize=14, va="top", ha="left")

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
        f"CH0 8-Parameter: Amax, μ(ped), μ(pedt), σ(ped), σ(pedt), Tmax, RMS_ped, RMS_pedt\n"
        f"{filename}  |  Event #{event_index}",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()

    # 基线放大图：Ped（含 RMS_ped）、Pedt
    fig_baseline, (ax_ped, ax_pedt) = plt.subplots(1, 2, figsize=(12, 5))
    fig_baseline.suptitle(
        f"Baseline: μ(ped), σ(ped), RMS_ped | μ(pedt), σ(pedt), RMS_pedt  |  {filename}  Event #{event_index}",
        fontsize=12,
        fontweight="bold",
    )

    t_ped_zoom = time_axis_us[:samples_baseline]
    w_ped_zoom = waveform[:samples_baseline]
    ax_ped.plot(t_ped_zoom, w_ped_zoom, color="C0", linewidth=1)
    k_ped, b_ped = np.polyfit(t_ped_zoom, w_ped_zoom, 1)
    ped_fit = k_ped * t_ped_zoom + b_ped
    ax_ped.plot(t_ped_zoom, ped_fit, color="tab:green", linestyle="--", linewidth=1.2, alpha=0.9)
    ax_ped.set_xlabel("Time (µs)", fontsize=14, fontweight="bold")
    ax_ped.set_ylabel("Amplitude (ADC)", fontsize=14, fontweight="bold")
    ax_ped.set_title(
        f"Ped: μ={mu_ped:.2f}  σ={sigma_ped:.2f}  RMS={rms_ped:.3f}",
        fontsize=12,
        fontweight="bold",
    )
    ax_ped.grid(True, alpha=0.3)

    div_ped = make_axes_locatable(ax_ped)
    ax_hist_ped = div_ped.append_axes("right", size="25%", pad=0.15, sharey=ax_ped)
    ax_hist_ped.tick_params(axis="y", labelleft=False)
    ax_hist_ped.hist(
        w_ped_zoom,
        bins=min(20, max(5, samples_baseline // 2)),
        orientation="horizontal",
        color="tab:green",
        alpha=0.6,
        edgecolor="tab:green",
    )
    ax_hist_ped.axhline(mu_ped, color="tab:green", linestyle="--", linewidth=1, alpha=0.9)

    t_pedt_zoom = time_axis_us[-samples_baseline:]
    w_pedt_zoom = waveform[-samples_baseline:]
    ax_pedt.plot(t_pedt_zoom, w_pedt_zoom, color="C0", linewidth=1)
    k_pedt, b_pedt = np.polyfit(t_pedt_zoom, w_pedt_zoom, 1)
    pedt_fit = k_pedt * t_pedt_zoom + b_pedt
    ax_pedt.plot(t_pedt_zoom, pedt_fit, color="tab:orange", linestyle="--", linewidth=1.2, alpha=0.9)
    ax_pedt.set_xlabel("Time (µs)", fontsize=14, fontweight="bold")
    ax_pedt.set_ylabel("Amplitude (ADC)", fontsize=14, fontweight="bold")
    ax_pedt.set_title(
        f"Pedt: μ={mu_pedt:.2f}  σ={sigma_pedt:.2f}  RMS={rms_pedt:.3f}",
        fontsize=12,
        fontweight="bold",
    )
    ax_pedt.grid(True, alpha=0.3)

    div_pedt = make_axes_locatable(ax_pedt)
    ax_hist_pedt = div_pedt.append_axes("right", size="25%", pad=0.15, sharey=ax_pedt)
    ax_hist_pedt.tick_params(axis="y", labelleft=False)
    ax_hist_pedt.hist(
        w_pedt_zoom,
        bins=min(20, max(5, samples_baseline // 2)),
        orientation="horizontal",
        color="tab:orange",
        alpha=0.6,
        edgecolor="tab:orange",
    )
    ax_hist_pedt.axhline(mu_pedt, color="tab:orange", linestyle="--", linewidth=1, alpha=0.9)

    plt.tight_layout()

    if save_path is None:
        ge_self_dir = os.path.dirname(cut_dir)
        data_dir = os.path.dirname(ge_self_dir)
        python_dir_parent = os.path.dirname(data_dir)
        project_root = os.path.dirname(python_dir_parent)
        output_dir = os.path.join(project_root, "images", "presentation")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_png = f"ch0_8param_event{event_index}_{timestamp}.png"
        save_path = os.path.join(output_dir, filename_png)

    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"8 参数波形图已保存至: {save_path}")

    save_path_baseline = save_path.replace(".png", "_baseline.png")
    fig_baseline.savefig(save_path_baseline, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"基线图已保存至: {save_path_baseline}")

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
    """对 PN-cut ±1σ 带内的事件，按 8 维参数可视化 CH0 波形。"""
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

    n_plot = min(max_events_to_plot, event_ranks.size)
    print(f"\n将对前 {n_plot} 个事件进行 CH0 8 参数可视化。")

    saved_paths: List[str] = []
    for i in range(n_plot):
        rank = int(event_ranks[i])
        event_index = int(selected_indices[rank])
        print(f"\n[{i + 1}/{n_plot}] 可视化 event_rank = {rank} (全局 Event #{event_index})")
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
            baseline_window_us=2.0,
            max_events_to_plot=2100,
            show_plot=True,
        )
        print("\n保存的图片路径：")
        for p in paths:
            print("  ", p)
    except Exception as e:
        print(f"\nPN-cut 8 参数可视化失败: {e}")
        import traceback
        traceback.print_exc()
