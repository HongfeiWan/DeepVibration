#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pncut 工具函数

用于根据两路主放最大值的线性关系，剔除在能量阈值附近偏离线性关系的噪声信号。

物理图像
--------
- 对每个事件，计算两路主放（例如 CH0 和 CH1）的最大值：max_ch0, max_ch1。
- 正常情况下，max_ch0 与 max_ch1 应服从线性关系：max_ch1 ≈ a * max_ch0 + b。
- 在能量阈值附近的噪声信号会偏离这种线性关系。
- 通过选取紧贴噪声分布的一条直线（cut line）进行筛除，保留符合线性关系的事例。
- 使用 ACT 事例计算 PNcut 在能量阈值附近的效率曲线。

说明
----
- 本文件只提供 **数组级工具函数**，不直接读写 HDF5 文件。
- 上层脚本可以自行根据返回的 max_ch0 / max_ch1 画出二维散点图，并叠加 cut line。
"""

from __future__ import annotations

from typing import Dict, Tuple, Union, Optional

import numpy as np

ArrayLike = Union[np.ndarray]


def _compute_max_in_window(
    waveforms: np.ndarray,
    ch_idx: int,
    max_window: Optional[Union[slice, Tuple[int, int]]] = None,
) -> np.ndarray:
    """
    在给定时间窗上计算某一通道每个事件的 **最大值**。

    参数:
        waveforms: 波形数组，形状为 (n_samples, n_channels, n_events)
        ch_idx:    需要提取的通道索引（如 CH0 -> 0, CH1 -> 1）
        max_window:
            - 若为 None，则在整个波形上求最大值；
            - 若为 slice，对应 waveforms[max_window, ch_idx, :] 的时间区间；
            - 若为 (start, stop) 元组，则自动转换为 slice(start, stop)。

    返回:
        amaxs: 形状为 (n_events,) 的最大值数组，每个事件一个最大值。
    """
    if not isinstance(waveforms, np.ndarray):
        waveforms = np.asarray(waveforms)

    if waveforms.ndim != 3:
        raise ValueError(
            f"waveforms 维度应为 3，形如 (n_samples, n_channels, n_events)，"
            f"当前形状为 {waveforms.shape}"
        )

    n_samples, n_channels, _ = waveforms.shape

    if not (0 <= ch_idx < n_channels):
        raise IndexError(f"通道索引 ch_idx={ch_idx} 超出范围 [0, {n_channels - 1}]")

    if max_window is None:
        # 整个波形上求最大值
        amaxs = waveforms[:, ch_idx, :].max(axis=0).astype(np.float64)
    else:
        if isinstance(max_window, tuple):
            if len(max_window) != 2:
                raise ValueError("max_window 元组必须为 (start, stop)")
            start, stop = max_window
            window_slice = slice(start, stop)
        elif isinstance(max_window, slice):
            window_slice = max_window
        else:
            raise TypeError("max_window 必须是 None、slice 或 (start, stop) 元组")

        # 规范化 slice 边界
        start, stop, step = window_slice.indices(n_samples)
        if step != 1:
            raise ValueError("当前实现仅支持连续的时间窗（step 必须为 1）")
        if start >= stop:
            raise ValueError(f"max_window 无效: start={start}, stop={stop}")

        # 取出该通道在时间窗内的波形: 形状 (n_win_samples, n_events)
        seg = waveforms[window_slice, ch_idx, :]  # type: ignore[arg-type]

        # 沿时间维度取最大值
        amaxs = seg.max(axis=0).astype(np.float64)

    return amaxs


def _fit_cut_line(
    max_ch0: np.ndarray,
    max_ch1: np.ndarray,
    method: str = "quantile",
    quantile: float = 0.05,
) -> Tuple[float, float]:
    """
    拟合紧贴噪声分布的 cut line（直线）。

    参数:
        max_ch0: CH0 的最大值数组
        max_ch1: CH1 的最大值数组
        method: 拟合方法
            - "quantile": 使用分位数方法，找到偏离主线性关系的噪声点，拟合紧贴噪声的直线
            - "linear": 直接线性拟合所有点，然后向下偏移
        quantile: 当 method="quantile" 时，用于识别噪声点的分位数阈值（默认 0.05）

    返回:
        (slope, intercept): 直线参数 y = slope * x + intercept
    """
    if method == "quantile":
        # 方法1：先对主线性关系做拟合，然后找偏离的点（噪声），再拟合噪声边界
        # 1. 对全部点做线性拟合（主线性关系）
        coeffs = np.polyfit(max_ch0, max_ch1, deg=1)
        main_slope, main_intercept = coeffs[0], coeffs[1]

        # 2. 计算每个点到主直线的距离（垂直距离）
        # 点到直线 ax + by + c = 0 的距离 = |ax0 + by0 + c| / sqrt(a^2 + b^2)
        # 对于 y = slope*x + intercept，转换为标准形式：slope*x - y + intercept = 0
        # 距离 = |slope*x0 - y0 + intercept| / sqrt(slope^2 + 1)
        distances = np.abs(main_slope * max_ch0 - max_ch1 + main_intercept) / np.sqrt(
            main_slope**2 + 1
        )

        # 3. 找到距离较大的点（可能是噪声），使用下分位数
        noise_threshold = np.quantile(distances, 1.0 - quantile)
        noise_mask = distances >= noise_threshold

        if noise_mask.sum() < 2:
            # 噪声点太少，回退到直接线性拟合
            print(
                f"警告：噪声点数量过少 ({noise_mask.sum()})，使用全部点进行线性拟合"
            )
            coeffs = np.polyfit(max_ch0, max_ch1, deg=1)
            return coeffs[0], coeffs[1]

        # 4. 对噪声点做线性拟合，得到紧贴噪声的直线
        noise_ch0 = max_ch0[noise_mask]
        noise_ch1 = max_ch1[noise_mask]
        coeffs = np.polyfit(noise_ch0, noise_ch1, deg=1)
        return coeffs[0], coeffs[1]

    elif method == "linear":
        # 方法2：直接线性拟合，然后向下偏移（简单但可能不够精确）
        coeffs = np.polyfit(max_ch0, max_ch1, deg=1)
        slope, intercept = coeffs[0], coeffs[1]

        # 计算残差的标准差，向下偏移 2-3σ
        predicted = slope * max_ch0 + intercept
        residuals = max_ch1 - predicted
        std_residual = residuals.std()
        intercept_cut = intercept - 2.5 * std_residual  # 向下偏移

        return slope, intercept_cut

    else:
        raise ValueError(f"不支持的拟合方法: {method}，支持: 'quantile', 'linear'")


def pncut_ch0_ch1_from_act(
    phys_waveforms: np.ndarray,
    energy: np.ndarray,
    max_window: Optional[Union[slice, Tuple[int, int]]] = None,
    ch0_idx: int = 0,
    ch1_idx: int = 1,
    cut_method: str = "quantile",
    quantile: float = 0.05,
    cut_side: str = "below",
    energy_bins: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    使用 ACT 物理事例，对两路主放（CH0/CH1）的最大值做 PNcut。

    典型用法：
      - phys_waveforms: 已经筛过 RT / Inhibit 等，主要包含 ACT 物理事例。
      - energy:         与 phys_waveforms 对应的能量数组（长度 = n_phys_events），
                        用于计算效率曲线。

    参数:
        phys_waveforms: 物理事例波形，形状 (n_samples, n_channels, n_events)
        energy:         每个事件的能量数组，形状 (n_events,)
        max_window:     计算最大值的时间窗（None 表示整个波形，或 slice / (start, stop)）
        ch0_idx:        CH0 的通道索引（默认 0）
        ch1_idx:        CH1 的通道索引（默认 1）
        cut_method:     拟合 cut line 的方法：'quantile'（默认）或 'linear'
        quantile:       当 cut_method='quantile' 时，用于识别噪声点的分位数（默认 0.05）
        cut_side:       保留哪一侧的点：
                        - 'below': 保留在 cut line 下方的点（默认，用于剔除上方噪声）
                        - 'above': 保留在 cut line 上方的点
        energy_bins:    用于计算效率曲线的能量区间（若为 None，自动生成）

    返回:
        keep_mask, stats 字典:
        - keep_mask: 形状为 (n_events,) 的 bool 数组，True 表示该事件通过 PNcut
        - stats: 包含以下键值的字典:
            * 'max_ch0', 'max_ch1'      : CH0/CH1 的最大值数组
            * 'energy'                  : 对应的能量数组（原样返回）
            * 'cut_slope', 'cut_intercept': Cut line 的参数（y = slope * x + intercept）
            * 'efficiency', 'energy_centers', 'energy_edges': 效率曲线数据
    """
    if not isinstance(phys_waveforms, np.ndarray):
        phys_waveforms = np.asarray(phys_waveforms)

    if phys_waveforms.ndim != 3:
        raise ValueError(
            f"phys_waveforms 维度应为 3，形如 (n_samples, n_channels, n_events)，"
            f"当前形状为 {phys_waveforms.shape}"
        )

    n_samples, n_channels, n_events = phys_waveforms.shape

    energy = np.asarray(energy, dtype=np.float64)
    if energy.ndim != 1 or energy.shape[0] != n_events:
        raise ValueError(
            f"energy 长度必须等于事件数 n_events={n_events}，"
            f"当前 energy 形状为 {energy.shape}"
        )

    # 1. 计算两路主放的最大值
    max_ch0 = _compute_max_in_window(phys_waveforms, ch0_idx, max_window)
    max_ch1 = _compute_max_in_window(phys_waveforms, ch1_idx, max_window)

    # 2. 拟合紧贴噪声分布的 cut line
    cut_slope, cut_intercept = _fit_cut_line(
        max_ch0, max_ch1, method=cut_method, quantile=quantile
    )

    # 3. 应用 PNcut：判断点是否在 cut line 的指定一侧
    predicted_ch1 = cut_slope * max_ch0 + cut_intercept

    if cut_side == "below":
        # 保留在 cut line 下方的点（max_ch1 < predicted_ch1）
        keep_mask = max_ch1 < predicted_ch1
    elif cut_side == "above":
        # 保留在 cut line 上方的点（max_ch1 > predicted_ch1）
        keep_mask = max_ch1 > predicted_ch1
    else:
        raise ValueError(f"cut_side 必须是 'below' 或 'above'，当前为 {cut_side}")

    # 4. 计算效率曲线（在不同能量区间下的通过率）
    if energy_bins is None:
        # 自动生成能量区间
        energy_min, energy_max = energy.min(), energy.max()
        n_bins = 20
        energy_bins = np.linspace(energy_min, energy_max, n_bins + 1)
    else:
        energy_bins = np.asarray(energy_bins)

    energy_centers = (energy_bins[:-1] + energy_bins[1:]) / 2.0
    efficiency = np.zeros(len(energy_centers))

    for i in range(len(energy_centers)):
        bin_mask = (energy >= energy_bins[i]) & (energy < energy_bins[i + 1])
        if bin_mask.sum() > 0:
            efficiency[i] = keep_mask[bin_mask].mean()
        else:
            efficiency[i] = np.nan

    stats: Dict[str, np.ndarray] = {
        "max_ch0": max_ch0,
        "max_ch1": max_ch1,
        "energy": energy,
        "cut_slope": np.array(cut_slope),
        "cut_intercept": np.array(cut_intercept),
        "efficiency": efficiency,
        "energy_centers": energy_centers,
        "energy_edges": energy_bins,
    }

    # 5. 打印简要统计信息
    print("=" * 70)
    print("PN Cut 统计信息（基于 ACT 物理事例两路主放最大值线性关系）")
    print("=" * 70)
    print(f"Cut line: max_ch1 = {cut_slope:.6f} * max_ch0 + {cut_intercept:.3f}")
    print(f"保留策略: {cut_side} cut line")
    print(f"\n事件总数: {n_events}")
    print(f"通过 PN Cut 的事例数: {int(keep_mask.sum())} " f"({keep_mask.mean() * 100:.2f}%)")
    print("=" * 70)

    return keep_mask, stats


__all__ = [
    "pncut_ch0_ch1_from_act",
    "_compute_max_in_window",
    "_fit_cut_line",
]


if __name__ == "__main__":
    """
    测试函数：绘制 max CH1 与 max CH2 的图形，并应用 PNcut。
    """
    import os
    import sys
    import h5py
    import matplotlib.pyplot as plt

    # 添加路径以便导入 utils 模块
    current_dir = os.path.dirname(os.path.abspath(__file__))
    python_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)

    from utils.visualize import get_h5_files

    print("=" * 70)
    print("PNcut 测试：绘制 max CH1 vs max CH2 图形")
    print("=" * 70)

    try:
        # 1. 自动获取文件并筛选物理事例（既非RT也非Inhibit）
        print("\n正在筛选物理事例（既非RT也非Inhibit）...")
        
        # 获取文件
        h5_files = get_h5_files()
        if "CH0-3" not in h5_files or not h5_files["CH0-3"]:
            raise FileNotFoundError("在 data/hdf5/raw_pulse/CH0-3 目录中未找到 h5 文件")
        if "CH5" not in h5_files or not h5_files["CH5"]:
            raise FileNotFoundError("在 data/hdf5/raw_pulse/CH5 目录中未找到 h5 文件")
        
        # 查找匹配的文件对
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
        
        # 筛选物理事例
        rt_cut = 6000.0
        batch_size = 1000
        
        with h5py.File(ch0_3_file, "r") as f_ch0:
            ch0_channel_data = f_ch0["channel_data"]
            ch0_time_samples, ch0_num_channels, ch0_num_events = ch0_channel_data.shape
            ch0_min_values = np.zeros(ch0_num_events, dtype=np.float64)
            for i in range(0, ch0_num_events, batch_size):
                end_idx = min(i + batch_size, ch0_num_events)
                batch_data = ch0_channel_data[:, 0, i:end_idx]
                ch0_min_values[i:end_idx] = np.min(batch_data, axis=0)
        
        with h5py.File(ch5_file, "r") as f_ch5:
            ch5_channel_data = f_ch5["channel_data"]
            ch5_time_samples, ch5_num_channels, ch5_num_events = ch5_channel_data.shape
            ch5_max_values = np.zeros(ch5_num_events, dtype=np.float64)
            for i in range(0, ch5_num_events, batch_size):
                end_idx = min(i + batch_size, ch5_num_events)
                batch_data = ch5_channel_data[:, 0, i:end_idx]
                ch5_max_values[i:end_idx] = np.max(batch_data, axis=0)
            
            # 判断信号类型
            rt_mask = ch5_max_values > rt_cut
            inhibit_mask = ch0_min_values == 0
            neither_mask = ~rt_mask & ~inhibit_mask
            selected_indices = np.where(neither_mask)[0]
        
        physical_count = len(selected_indices)

        if physical_count == 0:
            print("未发现物理事例，无法进行测试")
            sys.exit(1)

        print(f"找到 {physical_count} 个物理事例")

        # 2. 读取物理事例的波形数据
        print("\n正在读取波形数据...")
        batch_size = 1000
        waveforms_list = []

        with h5py.File(ch0_3_file, "r") as f:
            channel_data = f["channel_data"]
            time_samples, num_channels, total_events = channel_data.shape

            # 确保有 CH1 和 CH2（索引 1 和 2）
            if num_channels < 3:
                raise ValueError(f"文件只有 {num_channels} 个通道，需要至少 3 个通道（CH0, CH1, CH2）")

            # 批量读取选中的物理事例
            for i in range(0, len(selected_indices), batch_size):
                end_idx = min(i + batch_size, len(selected_indices))
                batch_indices = selected_indices[i:end_idx]
                batch_waveforms = channel_data[:, :, batch_indices]  # (time_samples, n_channels, batch_size)
                waveforms_list.append(batch_waveforms)
                if (i // batch_size + 1) % 10 == 0 or end_idx == len(selected_indices):
                    print(f"  已读取 {end_idx}/{len(selected_indices)} 个事件")

        # 合并所有批次
        phys_waveforms = np.concatenate(waveforms_list, axis=2)  # (time_samples, n_channels, n_events)
        print(f"波形数据形状: {phys_waveforms.shape}")

        # 3. 计算 CH1 和 CH2 的最大值
        print("\n正在计算 CH1 和 CH2 的最大值...")
        max_ch1 = _compute_max_in_window(phys_waveforms, ch_idx=1, max_window=None)
        max_ch2 = _compute_max_in_window(phys_waveforms, ch_idx=2, max_window=None)

        # 4. 绘制 max CH1 vs max CH2 散点图
        print("\n正在绘制图形...")
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        scatter = ax.scatter(
            max_ch1, max_ch2, s=3, alpha=0.4, c="blue", label="Physical Events"
        )

        ax.set_xlabel("CH0 Max (ADC counts)", fontsize=12)
        ax.set_ylabel("CH1 Max (ADC counts)", fontsize=12)
        ax.set_title(f"CH0 Max vs CH1 Max\nTotal: {len(max_ch1)} events", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 5. 绘制第二幅图：1000-2000 x,y 范围的局部放大图
        print("\n正在绘制第二幅图（1000-2000 x,y 范围）...")
        
        # 筛选在 1000-2000 范围内的点
        mask_range = (max_ch1 >= 1000) & (max_ch1 <= 2000) & (max_ch2 >= 1000) & (max_ch2 <= 2000)
        max_ch1_filtered = max_ch1[mask_range]
        max_ch2_filtered = max_ch2[mask_range]
        
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        
        scatter2 = ax2.scatter(
            max_ch1_filtered, max_ch2_filtered, s=3, alpha=0.4, c="blue", label="Physical Events (1000-2000 range)"
        )
        
        ax2.set_xlabel("CH0 Max (ADC counts)", fontsize=12)
        ax2.set_ylabel("CH1 Max (ADC counts)", fontsize=12)
        ax2.set_title(f"CH0 Max vs CH1 Max (1000-2000 range)\nTotal: {len(max_ch1_filtered)} events", fontsize=11)
        ax2.set_xlim(1000, 2000)
        ax2.set_ylim(1000, 2000)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

        print("\n测试完成！两幅图形已显示。")

    except Exception as e:
        print(f"\n测试过程中出错: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

