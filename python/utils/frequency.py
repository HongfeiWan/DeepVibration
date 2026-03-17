import numpy as np


def _compute_fast_highfreq_energy_ratio(
    waveform: np.ndarray,
    sampling_interval_ns: float = 4.0,
    cutoff_mhz: float = 0.2,
) -> float:
    """
    计算快放（CH3）波形中频率高于 cutoff_mhz 的频率成分能量占比。
    实现方式参考 debug_fft_event.py 中的 _compute_high_freq_energy_ratio：
    - 固定 120 µs 窗长，对波形截断；
    - 去直流分量；
    - 乘 Hann 窗抑制频谱泄漏；
    - 使用功率谱 |FFT|^2；
    - 分母：freq > 0 的总功率（剔除 DC）；
    - 分子：freq >= cutoff_mhz 的功率。
    """
    wf = np.asarray(waveform, dtype=np.float64)
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


def _compute_spectral_centroid_mhz(
    waveform: np.ndarray,
    sampling_interval_ns: float = 4.0,
) -> float:
    """
    计算单个事件波形的频谱质心（spectral centroid），单位 MHz。

    预处理步骤与 _compute_fast_highfreq_energy_ratio 保持一致：
        - 固定 120 µs 窗长度；
        - 去直流；
        - 乘 Hann 窗；
        - 使用 rFFT 计算功率谱 |FFT|^2，在 freq > 0 区间上计算功率加权平均频率。
    """
    wf = np.asarray(waveform, dtype=np.float64)

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
        wf = wf * np.hanning(wf.size)

    dt = sampling_interval_ns * 1e-9  # ns -> s
    n = wf.size
    freq = np.fft.rfftfreq(n, d=dt)
    fft_vals = np.fft.rfft(wf)
    power = np.abs(fft_vals) ** 2

    if power.size == 0:
        return 0.0

    # 仅在非直流分量上计算质心
    non_dc_mask = freq > 0.0
    freq_non_dc = freq[non_dc_mask]
    power_non_dc = power[non_dc_mask]

    total_power = float(np.sum(power_non_dc))
    if total_power <= 0.0:
        return 0.0

    centroid_hz = float(np.sum(freq_non_dc * power_non_dc) / total_power)
    return centroid_hz * 1e-6  # Hz -> MHz

