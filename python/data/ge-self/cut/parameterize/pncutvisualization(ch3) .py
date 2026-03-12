#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PN-cut 选中事件的 CH3 波形可视化脚本。

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
from multiprocessing import Pool, cpu_count

import h5py
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
from scipy.optimize import curve_fit


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
    sigma_factor: float = 1.0,) -> Tuple[np.ndarray, str, str]:
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

WAVEFORMS_PER_FIGURE = 9
sampling_interval_ns = 4.0

def _tanh_rise(x, p0, p1, p2, p3):
    return 0.5 * p0 * np.tanh(p1 * (x - p2)) + p3

def _fit_single_waveform(
    waveform: np.ndarray,
    smooth_window: int = 5,
    smooth_times: int = 25,) -> dict:
    """
    对单段波形做平滑与 tanh 拟合，供多进程调用。
    仅接受 1D 波形数组，便于 pickle 传递。
    """
    time_samples = waveform.shape[0]
    time_axis_us = np.arange(time_samples) * sampling_interval_ns / 1000.0

    waveform_smooth = waveform.copy()
    if smooth_window is not None and smooth_window > 1 and waveform.size >= smooth_window:
        half = smooth_window // 2
        for _ in range(max(1, smooth_times)):
            tmp = waveform_smooth.copy()
            for i in range(half, waveform.size - half):
                waveform_smooth[i] = float(np.mean(tmp[i - half : i + half + 1]))

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

    fit_curve = None
    try:
        baseline_window_us = 2.0
        samples_baseline = int(round(baseline_window_us * 1000.0 / sampling_interval_ns))
        samples_baseline = max(1, min(samples_baseline, waveform_smooth.size // 2))
        baseline_front = float(np.mean(waveform_smooth[:samples_baseline]))
        amp = waveform_smooth - baseline_front
        max_amp = float(np.max(amp))
        n_samples = len(time_axis_us)

        if max_amp > 0 and n_samples >= 5:
            idx_max = int(np.argmax(amp))
            idx_end = min(idx_max + 2000, n_samples - 1)
            mask = np.arange(n_samples) <= idx_end
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
            else:
                p1_init = 1.0 / (x_data[-1] - x_data[0] + 1e-6)

            popt = None
            if np.count_nonzero(mask) >= 5:
                try:
                    popt, _ = curve_fit(
                        _tanh_rise, x_data, y_data,
                        p0=[p0_init, p1_init, p2_init, p3_init],
                        maxfev=10000,
                    )
                except Exception:
                    pass

            if popt is None:
                x_data = time_axis_us
                y_data = waveform_smooth
                time_span = float(x_data[-1] - x_data[0]) + 1e-6
                p0_init = float(np.max(y_data) - np.min(y_data))
                p3_init = float(np.min(y_data))
                p2_init = float(np.mean(x_data))
                p1_init = 1.0 / time_span
                bounds_low = [1e-6, 1e-6, float(x_data[0]) - 100, -np.inf]
                bounds_high = [np.inf, 1e2, float(x_data[-1]) + 100, np.inf]
                try:
                    popt, _ = curve_fit(
                        _tanh_rise, x_data, y_data,
                        p0=[p0_init, p1_init, p2_init, p3_init],
                        bounds=(bounds_low, bounds_high),
                        maxfev=100000,
                    )
                except Exception:
                    pass

            if popt is not None:
                fit_curve = _tanh_rise(time_axis_us, *popt)
    except Exception:
        pass

    return {
        "time_axis_us": time_axis_us,
        "waveform_smooth": waveform_smooth,
        "fit_curve": fit_curve,
        "y_min": y_min,
        "y_max": y_max,
    }

def _visualize_batch(
    ch0_3_file: str,
    event_indices: List[int],
    channel_idx: int,
    batch_idx: int,
    show_plot: bool = True,
    pool: Optional[Pool] = None,
    smooth_window: int = 5,
    smooth_times: int = 25,
) -> Optional[str]:
    """在一幅图的 3x3 子图中显示至多 9 个波形。关闭窗口后返回，以便显示下一批。"""
    if not event_indices:
        return None

    batch_indices = event_indices[:WAVEFORMS_PER_FIGURE]
    with h5py.File(ch0_3_file, "r") as f:
        channel_data = f["channel_data"]
        waveforms = [channel_data[:, channel_idx, ev_idx].astype(np.float64) for ev_idx in batch_indices]

    if pool is not None:
        results = pool.starmap(
            _fit_single_waveform,
            [(w, smooth_window, smooth_times) for w in waveforms],
        )
    else:
        results = [_fit_single_waveform(w, smooth_window, smooth_times) for w in waveforms]

    n_rows, n_cols = 3, 3
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
    axes_flat = np.array(axes).flatten()

    for i, (data, event_index) in enumerate(zip(results, batch_indices)):
        ax = axes_flat[i]
        ax.plot(data["time_axis_us"], data["waveform_smooth"], color="blue", linewidth=1)
        if data["fit_curve"] is not None:
            ax.plot(data["time_axis_us"], data["fit_curve"], color="red", linewidth=1.5, linestyle="--")
        ax.set_ylim(data["y_min"], data["y_max"])
        ax.set_title(f"Event #{event_index}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time (µs)")
        ax.set_ylabel("ADC")

    n_axes = len(batch_indices)
    for j in range(n_axes, WAVEFORMS_PER_FIGURE):
        axes_flat[j].set_visible(False)

    filename = os.path.basename(ch0_3_file)
    fig.suptitle(f"CH3 Waveform (PN-cut)  Batch {batch_idx + 1}\n{filename}", fontsize=12, fontweight="bold")
    fig.tight_layout()

    save_path = None
    ge_self_dir = os.path.dirname(cut_dir)
    data_dir = os.path.dirname(ge_self_dir)
    python_dir = os.path.dirname(data_dir)
    project_root = os.path.dirname(python_dir)
    output_dir = os.path.join(project_root, "images", "presentation")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"ch3_pncut_batch{batch_idx + 1}_{timestamp}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"批次 {batch_idx + 1} 已保存: {save_path}")

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
    display_channel_idx: int = 3,
    show_plot: bool = True,
    n_workers: Optional[int] = None,
) -> List[str]:
    """
    对符合 lsmpncut PN-cut ±1σ 带的事件，每幅图显示 9 个 CH3 波形及 tanh 拟合。
    关闭当前图后显示下一批，直至全部事件显示完毕。
    每批内 9 个事件的拟合并行到 n_workers 个进程（默认用满所有 CPU）。
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() or 1)
    n_workers = min(n_workers, WAVEFORMS_PER_FIGURE)

    event_ranks, ch0_3_file_used, _, selected_indices = _select_events_in_1sigma_band(
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

    global_indices = [int(selected_indices[int(r)]) for r in event_ranks]
    total = len(global_indices)
    print(f"\n共 {total} 个 PN-cut 事件，每批 9 个，叉掉当前图后显示下一批。拟合并行进程数: {n_workers}")

    pool: Optional[Pool] = None
    if n_workers > 1:
        pool = Pool(processes=n_workers)

    try:
        saved_paths: List[str] = []
        for batch_idx in range(0, total, WAVEFORMS_PER_FIGURE):
            batch_indices = global_indices[batch_idx : batch_idx + WAVEFORMS_PER_FIGURE]
            print(f"\n显示批次 {batch_idx // WAVEFORMS_PER_FIGURE + 1}（事件 {batch_indices[0]}–{batch_indices[-1]}）")
            path = _visualize_batch(
                ch0_3_file=ch0_3_file_used,
                event_indices=batch_indices,
                channel_idx=display_channel_idx,
                batch_idx=batch_idx // WAVEFORMS_PER_FIGURE,
                show_plot=show_plot,
                pool=pool,
            )
            if path:
                saved_paths.append(path)
        return saved_paths
    finally:
        if pool is not None:
            pool.close()
            pool.join()

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
            display_channel_idx=3,
            show_plot=True,
        )
        print("\n保存的图片路径：")
        for p in paths:
            print("  ", p)
    except Exception as e:
        print(f"\nPN-cut 可视化失败: {e}")
        import traceback
        traceback.print_exc()

