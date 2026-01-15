#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gmmwavelet 工具脚本

在 gmmpncut 的基础上：
- 使用同样的物理事例筛选与 GMM 拟合流程，得到用于 GMM 的事件集合；
- 将被选中的事件按照 GMM 标签分成两类（component 0 和 component 1）；
- 对两类事件的 CH0 波形分别进行小波时频分析；
- 参考 python/data/wavelet/wavelet.py 的参数设置，对每一类事件统计平均功率谱；
- 使用所有 CPU 核心并行加速计算；
- 最终绘制两类事件的平均时频图和平均频率谱，对比它们的统计学差异。
"""

from __future__ import annotations

import os
import sys
from typing import Tuple, Dict, Any

import numpy as np
import h5py
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing as mp
from scipy import signal
import pywt


def _fit_single_line_two_step(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float = 2000.0,
    x_max: float = 14000.0,
) -> Tuple[float, float, float]:
    """
    直接复制 gmmpncut 中的“两步”最小二乘拟合函数，保持逻辑一致。
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    if x.shape != y.shape:
        raise ValueError(f"x 与 y 形状不一致: {x.shape} vs {y.shape}")

    mask_range = (x > x_min) & (x < x_max)
    n_range = int(mask_range.sum())
    if n_range < 2:
        raise ValueError(
            f"在区间 ({x_min}, {x_max}) 内有效点数不足 2 个，无法拟合直线 "
            f"(有效点数={n_range})"
        )

    x_sel = x[mask_range]
    y_sel = y[mask_range]

    a1, b1 = np.polyfit(x_sel, y_sel, deg=1)

    y_pred = a1 * x_sel + b1
    residuals = y_sel - y_pred
    sigma1 = residuals.std(ddof=1) if residuals.size > 1 else 0.0

    if sigma1 > 0.0:
        inlier_mask = np.abs(residuals) <= 3.0 * sigma1
        n_inliers = int(inlier_mask.sum())

        if 2 <= n_inliers < x_sel.size:
            x_in = x_sel[inlier_mask]
            y_in = y_sel[inlier_mask]
            a2, b2 = np.polyfit(x_in, y_in, deg=1)

            y_pred2 = a2 * x_in + b2
            residuals2 = y_in - y_pred2
            sigma2 = residuals2.std(ddof=1) if residuals2.size > 1 else 0.0
            return float(a2), float(b2), float(sigma2)

    return float(a1), float(b1), float(sigma1)


def _extract_gmm_event_sets() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    复用 gmmpncut 的数据选择和 GMM 拟合逻辑，
    返回：
        phys_waveforms: shape (time_samples, num_channels, num_phys_events)
        sel_indices:    被选中用于 GMM 的事件在 phys_waveforms 中的全局索引 (一维)
        labels:         每个被选中事件的 GMM 标签（0 或 1）
        max_ch0_all:    所有物理事例的 CH0 波形 (time_samples, num_phys_events)
    """
    from sklearn.mixture import GaussianMixture

    # 添加路径以便导入 utils 模块
    current_dir = os.path.dirname(os.path.abspath(__file__))
    python_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)

    from utils.visualize import get_h5_files

    print("=" * 70)
    print("gmmwavelet：基于 gmmpncut 的 GMM 事件集提取")
    print("=" * 70)

    # 1. 获取匹配的 CH0-3 和 CH5 文件
    h5_files = get_h5_files()
    if "CH0-3" not in h5_files or not h5_files["CH0-3"]:
        raise FileNotFoundError("在 data/hdf5/raw_pulse/CH0-3 目录中未找到 h5 文件")
    if "CH5" not in h5_files or not h5_files["CH5"]:
        raise FileNotFoundError("在 data/hdf5/raw_pulse/CH5 目录中未找到 h5 文件")

    ch0_3_files = h5_files["CH0-3"]
    ch5_files = h5_files["CH5"]
    ch0_3_dict = {os.path.basename(f): f for f in ch0_3_files}
    ch5_dict = {os.path.basename(f): f for f in ch5_files}

    matched = False
    ch0_3_file = None
    ch5_file = None
    for filename in ch0_3_dict.keys():
        if filename in ch5_dict:
            ch0_3_file = ch0_3_dict[filename]
            ch5_file = ch5_dict[filename]
            matched = True
            break

    if not matched:
        raise ValueError("未找到匹配的CH0-3和CH5文件对")

    print(f"使用文件: {os.path.basename(ch0_3_file)}")

    # 2. 筛选物理事例（既非RT也非Inhibit）
    rt_cut = 6000.0
    batch_size = 1000

    with h5py.File(ch0_3_file, "r") as f_ch0:
        ch0_channel_data = f_ch0["channel_data"]
        _, _, ch0_num_events = ch0_channel_data.shape
        ch0_min_values = np.zeros(ch0_num_events, dtype=np.float64)
        for i in range(0, ch0_num_events, batch_size):
            end_idx = min(i + batch_size, ch0_num_events)
            batch_data = ch0_channel_data[:, 0, i:end_idx]
            ch0_min_values[i:end_idx] = np.min(batch_data, axis=0)

    with h5py.File(ch5_file, "r") as f_ch5:
        ch5_channel_data = f_ch5["channel_data"]
        _, _, ch5_num_events = ch5_channel_data.shape
        ch5_max_values = np.zeros(ch5_num_events, dtype=np.float64)
        for i in range(0, ch5_num_events, batch_size):
            end_idx = min(i + batch_size, ch5_num_events)
            batch_data = ch5_channel_data[:, 0, i:end_idx]
            ch5_max_values[i:end_idx] = np.max(batch_data, axis=0)

        rt_mask = ch5_max_values > rt_cut
        inhibit_mask = ch0_min_values == 0
        neither_mask = ~rt_mask & ~inhibit_mask
        selected_indices = np.where(neither_mask)[0]

    physical_count = len(selected_indices)
    if physical_count == 0:
        raise RuntimeError("未发现物理事例，无法进行 GMM / 小波分析")

    print(f"找到 {physical_count} 个物理事例")

    # 3. 读取这些物理事例的所有通道波形
    print("\n正在读取物理事例波形数据...")
    batch_size = 1000
    waveforms_list = []

    with h5py.File(ch0_3_file, "r") as f:
        channel_data = f["channel_data"]
        time_samples, num_channels, _ = channel_data.shape

        if num_channels < 3:
            raise ValueError(
                f"文件只有 {num_channels} 个通道，需要至少 3 个通道（CH0, CH1, CH2）"
            )

        for i in range(0, len(selected_indices), batch_size):
            end_idx = min(i + batch_size, len(selected_indices))
            batch_indices = selected_indices[i:end_idx]
            batch_waveforms = channel_data[:, :, batch_indices]
            waveforms_list.append(batch_waveforms)
            if (i // batch_size + 1) % 10 == 0 or end_idx == len(selected_indices):
                print(f"  已读取 {end_idx}/{len(selected_indices)} 个事件")

    phys_waveforms = np.concatenate(waveforms_list, axis=2)
    print(f"物理事例波形数据形状: {phys_waveforms.shape}")

    # 4. 计算 CH1 和 CH2 的最大值并做 GMM 选择
    print("\n正在计算 CH1 和 CH2 的最大值并进行 GMM 拟合...")
    max_ch1 = phys_waveforms[:, 1, :].max(axis=0).astype(np.float64)
    max_ch2 = phys_waveforms[:, 2, :].max(axis=0).astype(np.float64)

    a, b, sigma = _fit_single_line_two_step(
        max_ch1,
        max_ch2,
        x_min=2000.0,
        x_max=14000.0,
    )

    print(f"  拟合得到直线: y = {a:.6f} * x + {b:.3f}")
    print(f"  残差标准差 σ (y 方向): {sigma:.3f}")

    mask_window = (
        (max_ch1 > 1100.0)
        & (max_ch1 < 1400.0)
        & (max_ch2 > 1000.0)
        & (max_ch2 < 2200.0)
    )
    if sigma > 0.0:
        residuals_all = max_ch2 - (a * max_ch1 + b)
        mask_band = np.abs(residuals_all) <= sigma
    else:
        mask_band = np.ones_like(max_ch1, dtype=bool)

    mask_sel = mask_window & mask_band
    x_sel = max_ch1[mask_sel]
    y_sel = max_ch2[mask_sel]

    n_sel = int(mask_sel.sum())
    print(f"  满足窗口和 ±1σ 条件的事件数: {n_sel}")
    if n_sel < 10:
        raise RuntimeError("选中事件数过少，无法稳定进行 GMM 拟合")

    sel_indices = np.where(mask_sel)[0]

    data = np.column_stack([x_sel, y_sel])

    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=0,
    )
    gmm.fit(data)

    labels = gmm.predict(data)

    print("GMM 拟合完成，两类事件数:")
    n_comp0 = int(np.sum(labels == 0))
    n_comp1 = int(np.sum(labels == 1))
    print(f"  component 0: {n_comp0} events")
    print(f"  component 1: {n_comp1} events")

    # 提取 CH0 全体波形 (time_samples, num_phys_events)
    max_ch0_all = phys_waveforms[:, 0, :]

    return phys_waveforms, sel_indices, labels, max_ch0_all


def _wavelet_for_single_event(
    ch0_wave: np.ndarray,
    sampling_interval_s: float,
    wavename: str,
    totalscal: int,
    min_freq: float,
    max_freq: float,
    detrend: bool,
) -> Dict[str, Any]:
    """
    对单个 event 的 CH0 波形做小波变换，返回功率谱等结果。
    参数设置参考 python/data/wavelet/wavelet.py。
    """
    ch0_wave = np.asarray(ch0_wave, dtype=np.float64)

    if detrend:
        ch0_wave = signal.detrend(ch0_wave)

    time_samples = ch0_wave.shape[0]
    sampling_rate = 1.0 / sampling_interval_s

    # 小波参数设置（参考 wavelet.py）
    if wavename.startswith("cmor"):
        params_str = wavename[4:]
        parts = params_str.split("-")
        if len(parts) == 2:
            bandwidth_param = float(parts[0])
            center_freq_param = float(parts[1])
            Fc = center_freq_param / (2 * np.pi)
        else:
            Fc = 1.0
    else:
        try:
            Fc = pywt.central_frequency(wavename)
        except Exception:
            Fc = 1.0

    max_scale = (Fc * sampling_rate) / min_freq
    min_scale = (Fc * sampling_rate) / max_freq
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), totalscal)
    frequencies = (Fc * sampling_rate) / scales

    coefficients, _ = pywt.cwt(
        ch0_wave,
        scales,
        wavename,
        sampling_period=sampling_interval_s,
    )

    if np.iscomplexobj(coefficients):
        power = np.abs(coefficients) ** 2
    else:
        power = coefficients ** 2

    mean_power_per_scale = np.mean(power, axis=1)
    dominant_freq = frequencies[np.argmax(mean_power_per_scale)]

    return {
        "power": power,
        "frequencies": frequencies,
        "scales": scales,
        "dominant_freq": dominant_freq,
    }


def analyze_gmm_wavelet(
    wavename: str = "cmor3-3",
    totalscal: int = 128,
    min_freq: float = 100e3,
    max_freq: float = 25e6,
    detrend: bool = True,
    max_events_per_class: int | None = None,
    show_plot: bool = True,
) -> Dict[str, Any]:
    """
    对 gmmpncut 选出的两类 GMM 事件的 CH0 分别做小波变换，并给出统计学结果。

    参数：
        wavename: 小波名称（默认 cmor3-3，与 python/data/ge-self/wavelet.py 一致）
        totalscal: 总尺度数
        min_freq, max_freq: 频率范围（Hz），默认 100kHz - 25MHz
        detrend: 是否对每个事件的 CH0 波形去趋势
        max_events_per_class: 每一类最多处理多少个事件（None 表示全部）
        show_plot: 是否显示最终统计图

    返回：
        包含两类事件的平均功率谱、主导频率等统计结果。
    """
    print("=" * 70)
    print("gmmwavelet：对 GMM 两类事件的 CH0 进行小波统计分析")
    print(f"目标频率范围: {min_freq/1e3:.1f}kHz - {max_freq/1e6:.2f}MHz")
    print("=" * 70)

    # 0. 基本采样参数（与 wavelet.py 一致）
    sampling_interval_ns = 4.0
    sampling_interval_s = sampling_interval_ns * 1e-9

    # 1. 先获取 GMM 事件集
    phys_waveforms, sel_indices, labels, max_ch0_all = _extract_gmm_event_sets()

    time_samples = phys_waveforms.shape[0]

    # 2. 根据 GMM 标签拆分两类事件在全局中的索引
    comp0_mask = labels == 0
    comp1_mask = labels == 1

    comp0_sel_indices = sel_indices[comp0_mask]
    comp1_sel_indices = sel_indices[comp1_mask]

    print("\nGMM 两类事件在全体物理事例中的索引数：")
    print(f"  component 0: {len(comp0_sel_indices)}")
    print(f"  component 1: {len(comp1_sel_indices)}")

    if max_events_per_class is not None and max_events_per_class > 0:
        if len(comp0_sel_indices) > max_events_per_class:
            comp0_sel_indices = comp0_sel_indices[:max_events_per_class]
        if len(comp1_sel_indices) > max_events_per_class:
            comp1_sel_indices = comp1_sel_indices[:max_events_per_class]

    # 3. 为两类事件分别提取 CH0 波形
    ch0_comp0 = max_ch0_all[:, comp0_sel_indices]  # shape: (time_samples, n0)
    ch0_comp1 = max_ch0_all[:, comp1_sel_indices]  # shape: (time_samples, n1)

    n0 = ch0_comp0.shape[1]
    n1 = ch0_comp1.shape[1]

    print("\n将对两类事件的 CH0 分别进行小波变换：")
    print(f"  component 0: {n0} events")
    print(f"  component 1: {n1} events")

    if n0 == 0 or n1 == 0:
        raise RuntimeError("某一类事件数为 0，无法进行比较分析")

    # 4. 并行计算每个事件的小波变换（使用所有 CPU）
    n_jobs = mp.cpu_count()
    print(f"\n使用 {n_jobs} 个 CPU 核心并行处理小波变换...")

    def _parallel_wavelet_for_set(ch0_set: np.ndarray, label: str):
        """
        对一组事件的 CH0 波形进行小波变换，使用增量统计模式避免 OOM。
        参考 python/data/ge-self/wavelet.py 的实现。
        """
        num_events = ch0_set.shape[1]
        print(f"\n[{label}] 开始对 {num_events} 个事件做小波变换（增量统计模式）...")

        # 先计算第一个事件以获取频率和尺度信息
        first_result = _wavelet_for_single_event(
            ch0_set[:, 0],
            sampling_interval_s,
            wavename,
            totalscal,
            min_freq,
            max_freq,
            detrend,
        )
        frequencies = first_result["frequencies"]
        scales = first_result["scales"]
        time_samples = ch0_set.shape[0]

        # 估算单个event的功率谱矩阵大小
        single_event_memory_mb = (totalscal * time_samples * 8) / (1024 * 1024)
        print(f"  [{label}] 单个event功率谱矩阵大小: {single_event_memory_mb:.2f} MB")

        # 小批次处理：每批处理少量events，立即累加到统计量，然后释放
        # 批次大小根据内存情况调整：每批最多保存 batch_size 个功率谱矩阵
        batch_size = 20  # 每批处理20个events（平衡内存和并行效率）
        n_batches = (num_events + batch_size - 1) // batch_size

        print(f"  [{label}] 总events数: {num_events}")
        print(
            f"  [{label}] 如果全部保存需要: {single_event_memory_mb * num_events / 1024:.2f} GB"
        )
        print(
            f"  [{label}] 使用增量统计模式，每批最多 {batch_size} 个events，峰值内存: ~{single_event_memory_mb * batch_size:.2f} MB"
        )
        print(f"  [{label}] 分批处理: {n_batches} 个批次")

        # 初始化累加器（增量统计，不保存所有中间结果）
        sum_power = None  # 累加和
        sum_power_sq = None  # 平方和（用于计算标准差）
        count = 0  # 已处理的event数量

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_events)
            batch_size_actual = end_idx - start_idx

            print(
                f"  [{label}] 处理批次 {batch_idx + 1}/{n_batches} (events {start_idx+1}-{end_idx})..."
            )

            # 准备该批次的参数（直接传递，不打包成元组）
            # 并行计算该批次
            n_jobs_batch = min(n_jobs, batch_size_actual)

            if n_jobs_batch > 1 and batch_size_actual > 1:
                batch_results = Parallel(
                    n_jobs=n_jobs_batch, backend="loky", verbose=0
                )(
                    delayed(_wavelet_for_single_event)(
                        ch0_set[:, i],
                        sampling_interval_s,
                        wavename,
                        totalscal,
                        min_freq,
                        max_freq,
                        detrend,
                    )
                    for i in range(start_idx, end_idx)
                )
            else:
                batch_results = [
                    _wavelet_for_single_event(
                        ch0_set[:, i],
                        sampling_interval_s,
                        wavename,
                        totalscal,
                        min_freq,
                        max_freq,
                        detrend,
                    )
                    for i in range(start_idx, end_idx)
                ]

            # 提取该批次的功率谱
            batch_power_list = [r["power"] for r in batch_results]

            # 转换为numpy数组并立即计算统计量（不长时间保存完整数组）
            batch_power_array = np.array(
                batch_power_list
            )  # shape: (batch_size, totalscal, time_samples)

            # 增量累加统计量（立即计算，避免保存完整批次数组）
            batch_sum = np.sum(batch_power_array, axis=0)  # shape: (totalscal, time_samples)
            batch_sum_sq = np.sum(batch_power_array ** 2, axis=0)

            # 立即释放批次数组和结果
            del batch_power_array, batch_power_list, batch_results

            if sum_power is None:
                # 初始化（使用第一个批次的形状）
                sum_power = batch_sum.copy()
                sum_power_sq = batch_sum_sq.copy()
            else:
                # 累加
                sum_power += batch_sum
                sum_power_sq += batch_sum_sq

            count += batch_size_actual

            # 释放其他临时变量
            del batch_sum, batch_sum_sq

        print(f"  [{label}] 所有event的小波变换完成")

        # 计算统计量（从累加和计算）
        mean_power = sum_power / count  # 平均值

        # 计算标准差: std = sqrt(mean(x^2) - mean(x)^2)
        mean_power_sq = sum_power_sq / count
        variance = mean_power_sq - mean_power ** 2
        # 避免负值（浮点数误差）
        variance = np.maximum(variance, 0)
        std_power = np.sqrt(variance)

        # 中位数：使用平均值近似（增量统计模式无法计算精确中位数）
        median_power = mean_power

        # 释放累加器（不再需要）
        del sum_power, sum_power_sq

        # 计算每个频率的平均功率（跨时间维度平均）
        mean_power_per_freq = np.mean(mean_power, axis=1)  # shape: (totalscal,)
        std_power_per_freq = np.mean(std_power, axis=1)

        # 计算主导频率
        dominant_idx = int(np.argmax(mean_power_per_freq))
        dominant_freq = frequencies[dominant_idx]

        print(
            f"  [{label}] 平均主导频率: {dominant_freq/1e6:.4f} MHz, "
            f"平均功率范围: {np.min(mean_power):.2e} - {np.max(mean_power):.2e}"
        )
        print(
            f"  [{label}] 注意: 使用平均值近似中位数（增量统计模式，无法计算精确中位数）"
        )

        return {
            "frequencies": frequencies,
            "scales": scales,
            "mean_power": mean_power,
            "std_power": std_power,
            "median_power": median_power,
            "mean_power_per_freq": mean_power_per_freq,
            "std_power_per_freq": std_power_per_freq,
            "dominant_freq": dominant_freq,
            "num_events": num_events,
        }

    stats0 = _parallel_wavelet_for_set(ch0_comp0, "component 0")
    stats1 = _parallel_wavelet_for_set(ch0_comp1, "component 1")

    # 5. 画统计学结果：两类事件的平均时频图 + 平均频谱对比
    if show_plot:
        freq0 = stats0["frequencies"]
        mean_power0 = stats0["mean_power"]
        freq1 = stats1["frequencies"]
        mean_power1 = stats1["mean_power"]

        # 步骤 1：时频图（对比两类，分别一张图）
        # 时间轴（微秒），与 wavelet.py 一致
        sampling_interval_ns = 4.0
        time_axis_us = np.arange(time_samples) * sampling_interval_ns / 1000.0

        # 只显示限定频率范围
        def _plot_tf(mean_power, freqs, label, num_events):
            freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
            freqs_disp = freqs[freq_mask]
            mp_disp = mean_power[freq_mask, :]

            fig, ax = plt.subplots(1, 1, figsize=(14, 8))
            log_power = np.log10(mp_disp + 1e-20)
            im = ax.contourf(
                time_axis_us,
                freqs_disp,
                log_power,
                levels=50,
                cmap="jet",
            )
            ax.set_xlabel("Time (μs)", fontsize=12)
            ax.set_ylabel("Frequency (Hz)", fontsize=12)
            ax.set_title(
                f"Average Wavelet Time-Frequency ({label})\n"
                f"{num_events} events averaged, Frequency: {min_freq/1e3:.1f}kHz - {max_freq/1e6:.2f}MHz",
                fontsize=13,
            )
            ax.set_ylim(min_freq, max_freq)
            plt.colorbar(im, ax=ax, label="Log10(Mean Power)")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        _plot_tf(mean_power0, freq0, "component 0", stats0["num_events"])
        _plot_tf(mean_power1, freq1, "component 1", stats1["num_events"])

        _plot_tf(mean_power0, freq0, "component 0")
        _plot_tf(mean_power1, freq1, "component 1")

        # 步骤 2：平均频谱对比
        freq_mask0 = (freq0 >= min_freq) & (freq0 <= max_freq)
        freq_mask1 = (freq1 >= min_freq) & (freq1 <= max_freq)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(
            freq0[freq_mask0],
            stats0["mean_power_per_freq"][freq_mask0],
            "b-",
            linewidth=2,
            label=f"component 0 (N={stats0['num_events']})",
        )
        ax.fill_between(
            freq0[freq_mask0],
            stats0["mean_power_per_freq"][freq_mask0]
            - stats0["std_power_per_freq"][freq_mask0],
            stats0["mean_power_per_freq"][freq_mask0]
            + stats0["std_power_per_freq"][freq_mask0],
            color="blue",
            alpha=0.3,
            label="±1 Std",
        )

        ax.plot(
            freq1[freq_mask1],
            stats1["mean_power_per_freq"][freq_mask1],
            "r-",
            linewidth=2,
            label=f"component 1 (N={stats1['num_events']})",
        )
        ax.fill_between(
            freq1[freq_mask1],
            stats1["mean_power_per_freq"][freq_mask1]
            - stats1["std_power_per_freq"][freq_mask1],
            stats1["mean_power_per_freq"][freq_mask1]
            + stats1["std_power_per_freq"][freq_mask1],
            color="red",
            alpha=0.3,
        )

        # 标记主导频率
        ax.axvline(
            stats0["dominant_freq"],
            color="b",
            linestyle="--",
            linewidth=2,
            label=f"Dominant 0: {stats0['dominant_freq']/1e6:.4f}MHz",
        )
        ax.axvline(
            stats1["dominant_freq"],
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Dominant 1: {stats1['dominant_freq']/1e6:.4f}MHz",
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(min_freq, max_freq)
        ax.set_xlabel("Frequency (Hz)", fontsize=12)
        ax.set_ylabel("Average Power", fontsize=12)
        ax.set_title(
            f"Average Wavelet Power Spectrum: component 0 vs component 1\n"
            f"Frequency: {min_freq/1e3:.1f}kHz - {max_freq/1e6:.2f}MHz",
            fontsize=13,
        )
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=11)
        plt.tight_layout()
        plt.show()

    return {
        "component_0": stats0,
        "component_1": stats1,
    }


if __name__ == "__main__":
    try:
        results = analyze_gmm_wavelet(
            wavename="cmor3-3",
            totalscal=128,
            min_freq=100e3,
            max_freq=25e6,
            detrend=True,
            max_events_per_class=None,  # 如需限流，可设定为一个正整数
            show_plot=True,
        )
        print("\n分析完成。")
        print(
            "component 0 主导频率: "
            f"{results['component_0']['dominant_freq']/1e6:.4f} MHz, "
            f"N={results['component_0']['num_events']}"
        )
        print(
            "component 1 主导频率: "
            f"{results['component_1']['dominant_freq']/1e6:.4f} MHz, "
            f"N={results['component_1']['num_events']}"
        )
    except Exception as e:
        print(f"\n分析过程中出错: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
