from typing import Dict, Optional
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit


def _smooth_waveform_for_fast_fit(
    waveform: np.ndarray,
    smooth_window: int = 5,
    smooth_times: int = 35,
) -> np.ndarray:
    """
    针对快放信号前沿拟合所用的统一平滑函数。

    - 使用重复的滑动平均（由 `uniform_filter1d` 实现）进行去噪；
    - `smooth_window` 为滑动窗口长度（采样点数），必须为奇数；
    - `smooth_times` 为重复平滑次数。
    """
    waveform_smooth = np.asarray(waveform, dtype=np.float64).copy()
    if smooth_window is not None and smooth_window > 1 and waveform_smooth.size >= smooth_window:
        if smooth_window % 2 == 0:
            raise ValueError(f"smooth_window 必须为奇数，当前为 {smooth_window}")
        for _ in range(max(1, smooth_times)):
            waveform_smooth = uniform_filter1d(
                waveform_smooth,
                size=smooth_window,
                mode="nearest",
            )
    return waveform_smooth


def _tanh_rise(x: np.ndarray, p0: float, p1: float, p2: float, p3: float) -> np.ndarray:
    """
    通用的快放前沿 tanh 拟合模型：
        y(x) = 0.5 * p0 * tanh(p1 * (x - p2)) + p3
    其中：
    - p0: 快放幅值（峰 - 基线）；
    - p1: 上升速度相关系数；
    - p2: 上升沿中心时间（单位与 x 一致，这里为 µs）；
    - p3: 基线电平。
    """
    return 0.5 * p0 * np.tanh(p1 * (x - p2)) + p3


def _compute_fast_fit_params(
    waveform: np.ndarray,
    sampling_interval_ns: float = 4.0,
    baseline_window_us: float = 2.0,
) -> Dict[str, float]:
    """
    兼容旧接口的快放前沿 tanh 拟合函数。
    内部调用带返回“实际拟合点数”的实现，并丢弃该返回值。
    """
    params, _ = _compute_fast_fit_params_with_npoints(
        waveform,
        sampling_interval_ns=sampling_interval_ns,
        baseline_window_us=baseline_window_us,
    )
    return params


def _compute_fast_fit_params_with_npoints(
    waveform: np.ndarray,
    sampling_interval_ns: float = 4.0,
    baseline_window_us: float = 2.0,
) -> tuple[Dict[str, float], int]:
    """
    通用的快放前沿 tanh 拟合函数，同时返回实际用于拟合的点数。

    约定：
    - 输入为单道快放波形 `waveform`（任意长度的一维数组），采样间隔为 `sampling_interval_ns`；
    - 先在前 `baseline_window_us` 微秒内估计基线并做减基线处理；
    - 在平滑波形上，从起点到峰值再往后 2000 个点作为统一的拟合区间；
    - 只做一次局部拟合，不做多轮或全局 fallback；
    - 拟合质量（tanh_rms）仅在上述拟合区间上计算；
    - 若判断无有效快放或拟合失败，则返回各参数和 RMS 均为一个明显异常的大值，n_points 返回 0。
    """
    abnormal_value = 1e6

    waveform_smooth = _smooth_waveform_for_fast_fit(waveform)
    time_axis_us = np.arange(waveform_smooth.size) * sampling_interval_ns / 1000.0
    n_samples = len(time_axis_us)

    baseline_window_ns = baseline_window_us * 1000.0
    samples_baseline = int(round(baseline_window_ns / sampling_interval_ns))
    samples_baseline = max(1, min(samples_baseline, waveform_smooth.size // 2))

    baseline_front = float(np.mean(waveform_smooth[:samples_baseline]))
    amp = waveform_smooth - baseline_front
    max_amp = float(np.max(amp))

    # 无明显快放或波形过短，直接返回异常值，n_points=0
    if max_amp <= 0 or n_samples < 5:
        return (
            {
                "tanh_p0": abnormal_value,
                "tanh_p1": abnormal_value,
                "tanh_p2": abnormal_value,
                "tanh_p3": abnormal_value,
                "tanh_rms": abnormal_value,
            },
            0,
        )

    # 统一使用“起点到峰值再往后 2000 个点”的局部区间做拟合
    idx_max = int(np.argmax(amp))
    idx_end = min(idx_max + 2000, n_samples - 1)
    mask = np.arange(n_samples) <= idx_end
    x_data = time_axis_us[mask]
    y_data = waveform_smooth[mask]

    n_points = int(np.count_nonzero(mask))

    # 初值估计：幅值、基线、上升中心、上升速度
    p0_init, p3_init = max_amp, baseline_front
    mid_level = 0.5 * max_amp
    idx_mid = int(np.argmax(amp >= mid_level))
    p2_init = float(time_axis_us[idx_mid]) if amp[idx_mid] >= mid_level else float(np.mean(x_data))
    p1_init = 1.0 / (float(x_data[-1] - x_data[0]) + 1e-6)

    popt = None
    if n_points >= 5:
        try:
            popt, _ = curve_fit(
                _tanh_rise,
                x_data,
                y_data,
                p0=[p0_init, p1_init, p2_init, p3_init],
                maxfev=10000,
            )
        except Exception:
            popt = None

    if popt is not None:
        # 仅在拟合区间内评估残差 RMS
        fit_curve_seg = _tanh_rise(x_data, *popt)
        residuals = y_data - fit_curve_seg
        tanh_rms = float(np.sqrt(np.mean(residuals**2)))
        return (
            {
                "tanh_p0": float(popt[0]),
                "tanh_p1": float(popt[1]),
                "tanh_p2": float(popt[2]),
                "tanh_p3": float(popt[3]),
                "tanh_rms": tanh_rms,
            },
            n_points,
        )

    # 拟合失败：赋予明显异常的极大值，n_points=0
    return (
        {
            "tanh_p0": abnormal_value,
            "tanh_p1": abnormal_value,
            "tanh_p2": abnormal_value,
            "tanh_p3": abnormal_value,
            "tanh_rms": abnormal_value,
        },
        0,
    )


def debug_plot_fast_fit_ch3_from_first_hdf5(
    max_events: Optional[int] = None,
    sampling_interval_ns: float = 4.0,
    baseline_window_us: float = 2.0,) -> None:
    """
    测试/可视化函数：
    - 在 `data/hdf5/raw_pulse/CH0-3` 目录下找到第一个 hdf5 文件；
    - 假定文件内存在 `channel_data` 数据集，形状为 (时间采样点数, 通道数, 事件数)；
    - 从中取物理通道 3（索引 3）的波形，对每个 event 做快放 tanh 拟合；
    - 以 3x3 子图网格一次显示 9 个拟合结果；
    - 关闭当前窗口后自动显示下一批 9 个，直到所有 event 处理完毕或达到 max_events。

    如无特殊说明，采样时间轴按固定采样间隔 `sampling_interval_ns` 构造。
    """
    # 推断工程根目录（DeepVibration），然后定位到原始脉冲数据目录
    # __file__ = .../DeepVibration/python/utils/fit.py
    # parents[2] = .../DeepVibration
    root_dir = Path(__file__).resolve().parents[2]
    ch_dir = root_dir / "data" / "hdf5" / "raw_pulse" / "CH0-3"

    if not ch_dir.exists():
        raise FileNotFoundError(f"目录不存在：{ch_dir}")

    files = sorted(ch_dir.glob("*.h5*"))
    if not files:
        raise FileNotFoundError(f"目录 {ch_dir} 下未找到任何 hdf5 文件")

    h5_path = files[0]

    # 读取通道数据，按现有原始脉冲 HDF5 约定：
    # - `channel_data`: 形状为 (时间采样点数, 通道数, 事件数)
    with h5py.File(h5_path, "r") as f:
        if "channel_data" not in f:
            raise KeyError(f"文件 {h5_path} 中未找到 'channel_data' 数据集，可用键为：{list(f.keys())}")

        channel_data = np.asarray(f["channel_data"])

    if channel_data.ndim != 3:
        raise ValueError(f"channel_data 维度不支持：shape={channel_data.shape}，期望为 (time, channel, event)")

    time_samples, num_channels, num_events = channel_data.shape
    if num_channels <= 3:
        raise ValueError(f"channel_data 通道数={num_channels}，无法直接使用索引 3 作为 CH3")

    # 取物理通道 3（索引 3），并整理为 (n_events, n_samples)
    waveforms = channel_data[:, 3, :].astype(np.float64).T  # (num_events, time_samples)

    n_events_total = waveforms.shape[0]
    if max_events is not None:
        n_events_total = min(n_events_total, max_events)
        waveforms = waveforms[:n_events_total]

    if n_events_total == 0:
        raise ValueError("CH3 中没有任何 event 可用于拟合。")

    abnormal_value = 1e6

    idx_start = 0
    while idx_start < n_events_total:
        idx_end = min(idx_start + 9, n_events_total)
        batch = waveforms[idx_start:idx_end]
        n_batch = batch.shape[0]

        fig, axes = plt.subplots(3, 3, figsize=(12, 9))
        axes_flat = axes.ravel()

        for i in range(9):
            ax = axes_flat[i]
            if i >= n_batch:
                ax.axis("off")
                continue

            waveform = batch[i]
            n_samples = waveform.size
            t_us = np.arange(n_samples) * sampling_interval_ns / 1000.0

            params = _compute_fast_fit_params(
                waveform,
                sampling_interval_ns=sampling_interval_ns,
                baseline_window_us=baseline_window_us,
            )

            ax.plot(t_us, waveform, label="CH3", color="C0", linewidth=1.0)

            # 若拟合成功（未命中异常值），画出拟合曲线；失败也同样展示参数（一般为异常值）
            if params["tanh_rms"] < abnormal_value:
                fit_curve = _tanh_rise(
                    t_us,
                    params["tanh_p0"],
                    params["tanh_p1"],
                    params["tanh_p2"],
                    params["tanh_p3"],
                )
                ax.plot(t_us, fit_curve, label="tanh", color="C1", linewidth=1.0)

            title = (
                f"event {idx_start + i}\n"
                f"p0={params['tanh_p0']:.2f}, "
                f"p1={params['tanh_p1']:.2g}, "
                f"p2={params['tanh_p2']:.2f}µs, "
                f"p3={params['tanh_p3']:.2f}, "
                f"rms={params['tanh_rms']:.3g}"
            )
            ax.set_title(title, fontsize=8)

            ax.set_xlabel("t [µs]")
            ax.set_ylabel("[FADC units]")
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()  # 关闭窗口后继续下一批

        idx_start = idx_end

if __name__ == "__main__":
    debug_plot_fast_fit_ch3_from_first_hdf5()