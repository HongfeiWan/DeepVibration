#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对 ge-self 物理事例中，满足 lsmpncut 拟合直线 ±1σ 范围内的事件，
绘制 **能量-最大值时间** 的二维分布（Energy vs. Tmax）。

约定：
- 使用 CH0-3 HDF5 原始波形数据；
- 先按照 ge-self/select.py / lsmpncut.py 的方式筛选 Physical 事件（既非 RT 也非 Inhibit）；
- 在这些 Physical 事件中：
    * 计算 CH1 与 CH2 的最大值 (max_ch1, max_ch2)；
    * 在 2000 < max_ch1 < 14000 区间内，用 lsmpncut.fit_single_line_in_range 拟合直线
      max_ch2 ≈ a * max_ch1 + b；
    * 在同一区间内计算残差的标准差 σ，并选出 |max_ch2 - (a*max_ch1 + b)| <= σ 的事件；
- 对这些“1σ 区域内”的事件：
    * 定义能量 Energy = max(CH0)（CH0 的最大幅度）；
    * 定义 Tmax 为 CH0 达到最大幅度的时间（单位 µs）；
- 使用 2D 直方图 (hist2d) 绘制 Energy vs. Tmax 的二维分布。
"""

import os
import sys
from typing import Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt
import importlib.util


def _load_physical_events(
    rt_cut: float = 6000.0,
    ch0_idx: int = 0,
    ch5_idx: int = 0,
) -> Tuple[str, str, np.ndarray, np.ndarray]:
    """
    读取 ge-self 原始 HDF5 数据，筛选 Physical 事件，并返回：
        ch0_3_file, ch5_file, selected_indices, channel_data

    其中：
        - selected_indices: 满足“既非 RT 也非 Inhibit”的事件索引（全局事件号）；
        - channel_data: CH0-3 文件中的 channel_data 数组。
    """
    # 添加路径以便导入 utils.visualize
    current_dir = os.path.dirname(os.path.abspath(__file__))      # .../data/coincident
    python_dir = os.path.dirname(os.path.dirname(current_dir))    # .../python
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)

    from utils.visualize import get_h5_files

    print("=" * 70)
    print("加载 ge-self HDF5 数据并筛选 Physical 事件（既非 RT 也非 Inhibit）")
    print("=" * 70)

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

    if not matched or ch0_3_file is None or ch5_file is None:
        raise ValueError("未找到匹配的 CH0-3 和 CH5 文件对")

    print(f"使用文件: {os.path.basename(ch0_3_file)}")

    batch_size = 1000

    # 计算 CH0 最小值（用于 Inhibit 判定）
    with h5py.File(ch0_3_file, "r") as f_ch0:
        ch0_channel_data = f_ch0["channel_data"]
        _, _, ch0_num_events = ch0_channel_data.shape
        ch0_min_values = np.zeros(ch0_num_events, dtype=np.float64)
        for i in range(0, ch0_num_events, batch_size):
            end_idx = min(i + batch_size, ch0_num_events)
            batch_data = ch0_channel_data[:, ch0_idx, i:end_idx]
            ch0_min_values[i:end_idx] = np.min(batch_data, axis=0)

    # 计算 CH5 最大值（用于 RT 判定）
    with h5py.File(ch5_file, "r") as f_ch5:
        ch5_channel_data = f_ch5["channel_data"]
        _, _, ch5_num_events = ch5_channel_data.shape
        ch5_max_values = np.zeros(ch5_num_events, dtype=np.float64)
        for i in range(0, ch5_num_events, batch_size):
            end_idx = min(i + batch_size, ch5_num_events)
            batch_data = ch5_channel_data[:, ch5_idx, i:end_idx]
            ch5_max_values[i:end_idx] = np.max(batch_data, axis=0)

        rt_mask = ch5_max_values > rt_cut
        inhibit_mask = ch0_min_values == 0
        neither_mask = ~rt_mask & ~inhibit_mask
        selected_indices = np.where(neither_mask)[0]

    physical_count = selected_indices.size
    print(f"Physical 事件数（既非 RT 也非 Inhibit）: {physical_count}")
    if physical_count == 0:
        raise RuntimeError("未发现 Physical 事件，无法继续。")

    # 读取完整 CH0-3 波形
    with h5py.File(ch0_3_file, "r") as f_ch0:
        channel_data = f_ch0["channel_data"][:]  # (time_samples, n_channels, n_events)

    return ch0_3_file, ch5_file, selected_indices, channel_data


def plot_energy_vs_tmax_for_lsmpncut_1sigma(
    rt_cut: float = 6000.0,
    ch0_idx: int = 0,
    ch1_idx: int = 1,
    ch2_idx: int = 2,
    x_min: float = 2000.0,
    x_max: float = 14000.0,
    n_bins_energy: int = 100,
    n_bins_tmax: int = 100,
) -> None:
    """
    主函数：
    1. 加载 ge-self Physical 事件；
    2. 按 lsmpncut 风格在 2000 < max_ch1 < 14000 范围内拟合直线并算 σ；
    3. 选出落在 ±1σ 范围内的事件；
    4. 对这些事件计算：
         - Energy = max(CH0)
         - Tmax   = CH0 达到最大值的时间 (µs)
       并绘制 Energy vs. Tmax 的二维直方图。
    """
    # 动态加载 lsmpncut.py（因为目录名 ge-self 含有连字符，无法作为常规包导入）
    current_dir = os.path.dirname(os.path.abspath(__file__))      # .../python/data/coincident
    data_dir = os.path.dirname(current_dir)                       # .../python/data
    lsmpncut_path = os.path.join(data_dir, "ge-self", "cut", "lsmpncut.py")
    spec_ls = importlib.util.spec_from_file_location("lsmpncut_module", lsmpncut_path)
    lsmpncut_module = importlib.util.module_from_spec(spec_ls)
    assert spec_ls.loader is not None
    spec_ls.loader.exec_module(lsmpncut_module)
    fit_single_line_in_range = lsmpncut_module.fit_single_line_in_range

    # 1. 加载 Physical 事件
    ch0_3_file, ch5_file, selected_indices, channel_data = _load_physical_events(
        rt_cut=rt_cut,
        ch0_idx=ch0_idx,
        ch5_idx=0,
    )

    time_samples, n_channels, n_events = channel_data.shape
    print(f"\nCH0-3 channel_data 形状: {channel_data.shape}")

    if n_channels <= max(ch0_idx, ch1_idx, ch2_idx):
        raise ValueError(
            f"通道数不足：n_channels={n_channels}，需要至少包含 CH0/CH1/CH2。"
        )

    # 只取 Physical 事件的波形
    phys_waveforms = channel_data[:, :, selected_indices]  # (time_samples, n_channels, n_phys)
    n_phys = phys_waveforms.shape[2]
    print(f"Physical 事件数: {n_phys}")

    # 2. 计算 CH1/CH2 最大值
    max_ch1 = phys_waveforms[:, ch1_idx, :].max(axis=0).astype(np.float64)
    max_ch2 = phys_waveforms[:, ch2_idx, :].max(axis=0).astype(np.float64)

    # 3. 在 (x_min, x_max) 上用 lsmpncut 的“两步最小二乘”拟合直线
    print("\n在 2000 < CH1_max < 14000 范围内拟合直线 (lsmpncut)...")
    a, b = fit_single_line_in_range(max_ch1, max_ch2, x_min=x_min, x_max=x_max)

    # 计算范围内残差的 σ
    mask_range = (max_ch1 > x_min) & (max_ch1 < x_max)
    x_fit = max_ch1[mask_range]
    y_fit = max_ch2[mask_range]
    y_fit_pred = a * x_fit + b
    residuals = y_fit - y_fit_pred
    sigma = residuals.std(ddof=1) if residuals.size > 1 else 0.0

    print(f"拟合直线: max_ch2 ≈ {a:.6f} * max_ch1 + {b:.3f}")
    print(f"范围内点数: {x_fit.size}，σ = {sigma:.3f}")

    if sigma <= 0.0:
        raise RuntimeError("σ <= 0，无法定义 ±1σ 带。")

    # 4. 选出全体 Physical 事件中落在 ±1σ 带内的事件
    predicted_all = a * max_ch1 + b
    mask_1sigma = np.abs(max_ch2 - predicted_all) <= sigma
    kept_indices_in_phys = np.where(mask_1sigma)[0]
    n_kept = kept_indices_in_phys.size

    print(f"落在 ±1σ 带内的 Physical 事件数: {n_kept}")
    if n_kept == 0:
        raise RuntimeError("没有事件落在 ±1σ 带内。")

    # 5. 对这些事件计算 Energy (= max CH0) 与 Tmax
    sampling_interval_ns = 4.0
    sampling_interval_us = sampling_interval_ns / 1000.0
    time_axis_us = np.arange(time_samples) * sampling_interval_us

    # 取 CH0 波形
    ch0_phys = phys_waveforms[:, ch0_idx, :]  # (time_samples, n_phys)
    ch0_kept = ch0_phys[:, kept_indices_in_phys]  # (time_samples, n_kept)

    # Energy: max(CH0)；Tmax: argmax(CH0) * dt
    energy = ch0_kept.max(axis=0)
    idx_tmax = ch0_kept.argmax(axis=0)
    tmax_us = idx_tmax * sampling_interval_us

    # 6. 绘制二维散点图 Energy vs. Tmax
    print("\n绘制 Energy vs. Tmax 的二维散点图...")
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        energy,
        tmax_us,
        s=6,
        alpha=0.5,
        c="tab:blue",
        edgecolors="none",
    )

    ax.set_xlabel("Energy (CH0 peak, ADC counts)", fontsize=12)
    ax.set_ylabel("Tmax (µs)", fontsize=12)
    ax.set_title(
        "Energy vs. Tmax for ge-self events within ±1σ of lsmpncut line",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        plot_energy_vs_tmax_for_lsmpncut_1sigma(
            rt_cut=6000.0,
            ch0_idx=0,
            ch1_idx=1,
            ch2_idx=2,
            x_min=2000.0,
            x_max=14000.0,
            n_bins_energy=120,
            n_bins_tmax=120,
        )
    except Exception as e:
        print(f"maxenergy&maxtime 运行失败: {e}")
        import traceback

        traceback.print_exc()

