#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对指定事件的 CH3 波形做频域 FFT 分析，用于对比不同事件在频域上的差异。

当前默认对 5187 号和 5353 号两个事件的 CH3 进行频域分析，并绘制对比图。
"""

import os
import sys
from typing import Optional, Sequence

import h5py
import numpy as np
import matplotlib.pyplot as plt


# 与其他脚本保持一致的目录推断方式
current_dir = os.path.dirname(os.path.abspath(__file__))  # .../cut/parameterize
cut_dir = os.path.dirname(current_dir)                    # .../cut


def _resolve_ch0_3_file(ch0_3_file: Optional[str]) -> str:
    """
    当 ch0_3_file 为 None 时，自动查找 CH0-3 文件。
    逻辑与 `tanhfit(ch3).py` 中的实现保持一致。
    """
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


def _get_ch3_waveform(
    ch0_3_file: str,
    event_index: int,
    channel_idx: int = 3,
) -> np.ndarray:
    """从 HDF5 中读取指定事件的 CH3 波形。"""
    with h5py.File(ch0_3_file, "r") as f:
        channel_data = f["channel_data"]
        time_samples, num_channels, num_events = channel_data.shape

        if channel_idx >= num_channels:
            raise ValueError(f"channel_idx={channel_idx} 超过通道数 {num_channels}")
        if event_index >= num_events:
            raise IndexError(f"event_index={event_index} 超过事件数 {num_events}")

        waveform = channel_data[:, channel_idx, event_index].astype(np.float64)

    return waveform


def _get_waveform(
    ch0_3_file: str,
    event_index: int,
    channel_idx: int,
) -> np.ndarray:
    """从 HDF5 中读取指定通道的指定事件波形（可用于 CH0 / CH3 等）。"""
    with h5py.File(ch0_3_file, "r") as f:
        channel_data = f["channel_data"]
        time_samples, num_channels, num_events = channel_data.shape

        if channel_idx >= num_channels:
            raise ValueError(f"channel_idx={channel_idx} 超过通道数 {num_channels}")
        if event_index >= num_events:
            raise IndexError(f"event_index={event_index} 超过事件数 {num_events}")

        waveform = channel_data[:, channel_idx, event_index].astype(np.float64)

    return waveform


def _compute_fft(
    waveform: np.ndarray,
    sampling_interval_ns: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对波形做加窗 FFT，返回频率坐标和幅度谱（归一化到最大值为 1，便于比较）。

    - 窗长度：固定取前 120 µs 的数据；
    - 窗类型：Hann 窗；
    - 先去直流，再乘窗做 FFT。
    """
    wf = waveform.astype(np.float64)

    # 1) 固定 120 µs 窗长度
    target_us = 120.0
    n_120 = int(round(target_us * 1000.0 / sampling_interval_ns))
    n_120 = min(n_120, wf.size)
    wf = wf[:n_120]

    # 2) 去除直流分量，避免 0 频率处过大
    wf = wf - np.mean(wf)

    # 3) 乘 Hann 窗，抑制频谱泄漏
    if wf.size > 1:
        window = np.hanning(wf.size)
        wf = wf * window

    dt = sampling_interval_ns * 1e-9  # 转为秒
    n = wf.size

    freq = np.fft.rfftfreq(n, d=dt)          # 0 ~ Nyquist
    fft_vals = np.fft.rfft(wf)
    amp = np.abs(fft_vals)

    if amp.max() > 0:
        amp /= amp.max()

    return freq, amp


def _compute_non_dc_energy_ratio(waveform: np.ndarray) -> float:
    """
    计算波形的非直流能量占比：E_nonDC / E_total。

    使用时间域等效公式：
        E_total = sum(x^2)
        E_DC    = N * mean(x)^2
        ratio   = (E_total - E_DC) / E_total
    """
    x = waveform.astype(np.float64)
    n = x.size
    if n == 0:
        return 0.0

    total_energy = float(np.sum(x * x))
    if total_energy <= 0.0:
        return 0.0

    mean_val = float(np.mean(x))
    dc_energy = n * (mean_val ** 2)
    non_dc_energy = max(total_energy - dc_energy, 0.0)

    return non_dc_energy / total_energy


def _compute_second_diff_peak_to_mean(waveform: np.ndarray) -> float:
    """
    计算二阶差分的“峰均比”：
        d2 = diff(diff(waveform))
        使用绝对值后的最大值 / 平均值。
    """
    x = waveform.astype(np.float64)
    if x.size < 3:
        return 0.0
    d1 = np.diff(x)
    d2 = np.diff(d1)
    d2_abs = np.abs(d2)
    mean_val = float(np.mean(d2_abs))
    if mean_val <= 0.0:
        return 0.0
    peak_val = float(np.max(d2_abs))
    return peak_val / mean_val


def _compute_high_freq_energy_ratio(
    waveform: np.ndarray,
    sampling_interval_ns: float = 4.0,
    cutoff_mhz: float = 0.2,
) -> float:
    """
    计算频谱中高于 cutoff_mhz 的频率成分能量占比。

    实现方式：
    - 与 _compute_fft 保持一致的预处理（固定 120 µs 窗长、去直流、乘 Hann 窗）；
    - 使用功率谱 |FFT|^2；
    - 分子：freq >= cutoff_mhz 的功率和；
    - 分母：剔除直流分量后的总功率和（freq > 0）。
    """
    wf = waveform.astype(np.float64)

    # 固定 120 µs 窗长度
    target_us = 120.0
    n_120 = int(round(target_us * 1000.0 / sampling_interval_ns))
    n_120 = min(n_120, wf.size)
    wf = wf[:n_120]

    if wf.size == 0:
        return 0.0

    # 去直流
    wf = wf - np.mean(wf)

    # 乘 Hann 窗
    if wf.size > 1:
        window = np.hanning(wf.size)
        wf = wf * window

    dt = sampling_interval_ns * 1e-9
    n = wf.size
    freq = np.fft.rfftfreq(n, d=dt)
    fft_vals = np.fft.rfft(wf)
    power = np.abs(fft_vals) ** 2

    if power.size == 0:
        return 0.0

    # 频率大于 0（排除 DC）的总功率
    non_dc_mask = freq > 0.0
    total_power_non_dc = float(np.sum(power[non_dc_mask]))
    if total_power_non_dc <= 0.0:
        return 0.0

    # 高频（>= cutoff_mhz）的功率
    cutoff_hz = cutoff_mhz * 1e6
    high_mask = freq >= cutoff_hz
    high_power = float(np.sum(power[high_mask]))

    return high_power / total_power_non_dc


def analyze_fft_for_events(
    ch0_3_file: Optional[str] = None,
    event_indices: Sequence[int] = (5187, 5353),
    channel_idx: int = 3,
    sampling_interval_ns: float = 4.0,
    max_freq_mhz: Optional[float] = None,
    show_plot: bool = True,
) -> None:
    """
    对指定的一组事件（默认 5187 和 5353）在 CH3 上做 FFT，并进行频谱对比。

    参数：
        ch0_3_file        : CH0-3 文件路径，为 None 时自动查找。
        event_indices     : 需要分析的事件编号列表。
        channel_idx       : 通道索引（默认 3，即 CH3）。
        sampling_interval_ns : 采样间隔，单位 ns（默认 4.0，与其他脚本一致）。
        max_freq_mhz      : 频谱图显示的最大频率（MHz），None 表示显示到 Nyquist。
        show_plot         : 是否弹出图像窗口。
    """
    ch0_3_file_resolved = _resolve_ch0_3_file(ch0_3_file)
    print(f"使用 CH0-3 文件: {ch0_3_file_resolved}")
    print(f"待分析事件编号: {list(event_indices)}，CH3 通道索引: {channel_idx}")

    waveforms = {}
    for ev in event_indices:
        wf = _get_ch3_waveform(ch0_3_file_resolved, ev, channel_idx=channel_idx)
        waveforms[ev] = wf
        ratio = _compute_non_dc_energy_ratio(wf)
        peak_mean_ratio = _compute_second_diff_peak_to_mean(wf)
        high_freq_ratio = _compute_high_freq_energy_ratio(
            wf, sampling_interval_ns=sampling_interval_ns, cutoff_mhz=0.2
        )
        print(
            f"Event #{ev}: 采样点数 = {wf.size}, "
            f"最小值 = {wf.min():.1f}, 最大值 = {wf.max():.1f}, "
            f"非直流能量占比 = {ratio * 100:.3f}%, "
            f"二阶差分峰均比 = {peak_mean_ratio:.3f}, "
            f"freq > 0.2 MHz 能量占比 = {high_freq_ratio * 100:.3f}%"
        )

    # 计算 FFT（基于 CH3）
    fft_results = {}
    for ev, wf in waveforms.items():
        freq, amp = _compute_fft(wf, sampling_interval_ns=sampling_interval_ns)
        fft_results[ev] = (freq, amp)

    # 频率单位转为 MHz
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))

    # 第 1 行：CH3 时域波形
    dt = sampling_interval_ns * 1e-3  # ns -> µs
    for ev, wf in waveforms.items():
        t_us = np.arange(wf.size) * dt
        axes[0].plot(t_us, wf, label=f"Event #{ev}")

    axes[0].set_xlabel("Time (µs)")
    axes[0].set_ylabel("Amplitude (ADC)")
    axes[0].set_title("CH3")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 第 2 行：CH3 频域幅度谱
    for ev, (freq, amp) in fft_results.items():
        freq_mhz = freq * 1e-6
        if max_freq_mhz is not None:
            mask = freq_mhz <= max_freq_mhz
            freq_mhz = freq_mhz[mask]
            amp_plot = amp[mask]
        else:
            amp_plot = amp
        axes[1].plot(freq_mhz, amp_plot, label=f"Event #{ev}")

    axes[1].set_xlabel("Frequency (MHz)")
    axes[1].set_ylabel("Normalized |FFT|")
    axes[1].set_title("CH3 (FFT)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # 第 3 行：CH0 时域波形对比
    ch0_idx = 0
    dt = sampling_interval_ns * 1e-3  # ns -> µs
    for ev in event_indices:
        wf_ch0 = _get_waveform(ch0_3_file_resolved, ev, channel_idx=ch0_idx)
        t_us_ch0 = np.arange(wf_ch0.size) * dt
        axes[2].plot(t_us_ch0, wf_ch0, label=f"Event #{ev}")

    axes[2].set_xlabel("Time (µs)")
    axes[2].set_ylabel("Amplitude (ADC)")
    axes[2].set_title("CH0")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # 第 4 行：CH0 频域幅度谱
    fft_results_ch0 = {}
    for ev in event_indices:
        wf_ch0 = _get_waveform(ch0_3_file_resolved, ev, channel_idx=ch0_idx)
        freq_ch0, amp_ch0 = _compute_fft(wf_ch0, sampling_interval_ns=sampling_interval_ns)
        fft_results_ch0[ev] = (freq_ch0, amp_ch0)

    for ev, (freq_ch0, amp_ch0) in fft_results_ch0.items():
        freq_mhz_ch0 = freq_ch0 * 1e-6
        if max_freq_mhz is not None:
            mask = freq_mhz_ch0 <= max_freq_mhz
            freq_mhz_ch0 = freq_mhz_ch0[mask]
            amp_plot_ch0 = amp_ch0[mask]
        else:
            amp_plot_ch0 = amp_ch0
        axes[3].plot(freq_mhz_ch0, amp_plot_ch0, label=f"Event #{ev}")

    axes[3].set_xlabel("Frequency (MHz)")
    axes[3].set_ylabel("Normalized |FFT|")
    axes[3].set_title("CH0 (FFT)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    fig.tight_layout()

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    # 默认对 5187 号和 5353 号 CH3 事件做频域 FFT 对比
    analyze_fft_for_events(
        ch0_3_file=None,
        event_indices=(7688, 5353),
        channel_idx=3,
        sampling_interval_ns=4.0,
        max_freq_mhz=None,  # 如果只想看低频部分，可以改成例如 50.0
        show_plot=True,
    )

