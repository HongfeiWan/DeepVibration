#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mincut 工具函数

用于根据自触发物理事例（ACT 事例）的 **最小值分布**，
对传入的事件波形在 CH0 / CH1 上做 “A_min ± nσ” 的最小值截断（Min Cut）。

物理图像
--------
- 对每个事件，在给定时间窗内（min_window）求 CH0 / CH1 的最小值 A_min。
- A_min 主要反映该段时间内的基线与噪声水平。
- 将 A_min 与能量（E）画成二维散点图 (E, A_min)，一般可看到 A_min 与能量无明显相关。
- 对 **ACT 物理事例** 的 A_min 分布做高斯拟合（用样本均值 μ 和标准差 σ 近似），
  然后对所有自触发事例应用：

      μ_ch0 - nσ_ch0 ≤ A_min_ch0 ≤ μ_ch0 + nσ_ch0
      μ_ch1 - nσ_ch1 ≤ A_min_ch1 ≤ μ_ch1 + nσ_ch1

  只有同时满足两条的事例被保留，从而得到 “基线 / 噪声合理” 的自触发样本。

说明
----
- 本文件只提供 **数组级工具函数**，不直接读写 HDF5 文件。
- 上层脚本可以自行根据返回的 A_min 与能量 E 画出 “最小值-能量二维散点图”。
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray]


def _compute_min_in_window(
    waveforms: np.ndarray,
    ch_idx: int,
    min_window: Union[slice, Tuple[int, int]],
) -> np.ndarray:
    """
    在给定时间窗上计算某一通道每个事件的 **最小值 A_min**。

    参数:
        waveforms: 波形数组，形状为 (n_samples, n_channels, n_events)
        ch_idx:    需要提取的通道索引（如 CH0 -> 0, CH1 -> 1）
        min_window:
            - 若为 slice，对应 waveforms[min_window, ch_idx, :] 的时间区间；
            - 若为 (start, stop) 元组，则自动转换为 slice(start, stop)。

    返回:
        amins: 形状为 (n_events,) 的最小值数组，每个事件一个 A_min。
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

    if isinstance(min_window, tuple):
        if len(min_window) != 2:
            raise ValueError("min_window 元组必须为 (start, stop)")
        start, stop = min_window
        window_slice = slice(start, stop)
    elif isinstance(min_window, slice):
        window_slice = min_window
    else:
        raise TypeError("min_window 必须是 slice 或 (start, stop) 元组")

    # 规范化 slice 边界
    start, stop, step = window_slice.indices(n_samples)
    if step != 1:
        raise ValueError("当前实现仅支持连续的时间窗（step 必须为 1）")
    if start >= stop:
        raise ValueError(f"min_window 无效: start={start}, stop={stop}")

    # 取出该通道在时间窗内的波形: 形状 (n_win_samples, n_events)
    seg = waveforms[window_slice, ch_idx, :]  # type: ignore[arg-type]

    # 沿时间维度取最小值，得到每个事件的 A_min
    amins = seg.min(axis=0).astype(np.float64)
    return amins


def mincut_ch0_ch1_from_act(
    phys_waveforms: np.ndarray,
    energy: np.ndarray,
    min_window: Union[slice, Tuple[int, int]] = slice(0, 200),
    ch0_idx: int = 0,
    ch1_idx: int = 1,
    nsigma: float = 3.0,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    使用 ACT 物理事例自身的最小值分布，对 CH0/CH1 做 Min Cut（默认 ±3σ）。

    典型用法：
      - phys_waveforms: 已经筛过 RT / Inhibit 等，主要包含 ACT 物理事例。
      - energy:         与 phys_waveforms 对应的能量数组（长度 = n_phys_events），
                        可用于画 A_min vs Energy 的二维散点图。

    参数:
        phys_waveforms: 物理事例波形，形状 (n_samples, n_channels, n_events)
        energy:         每个事件的能量数组，形状 (n_events,)
        min_window:     计算最小值的时间窗（slice 或 (start, stop)），
                        默认使用前 200 个采样点；可根据基线 / 触发位置调整。
        ch0_idx:        CH0 的通道索引（默认 0）
        ch1_idx:        CH1 的通道索引（默认 1）
        nsigma:         Min cut 的 σ 倍数，默认 3.0（±3σ）

    返回:
        keep_mask, stats 字典:
        - keep_mask: 形状为 (n_events,) 的 bool 数组，True 表示该事件通过 Min Cut
        - stats: 包含以下键值的字典（均为 numpy 数组 / 标量）:
            * 'amin_ch0', 'amin_ch1'  : CH0/CH1 的最小值数组 (A_min)
            * 'energy'                : 对应的能量数组（原样返回，便于作图）
            * 'mu_ch0', 'sigma_ch0'
            * 'mu_ch1', 'sigma_ch1'
            * 'lo_ch0', 'hi_ch0'      : CH0 的 [μ - nsigma·σ, μ + nsigma·σ]
            * 'lo_ch1', 'hi_ch1'      : CH1 的 [μ - nsigma·σ, μ + nsigma·σ]
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

    # 1. 计算 ACT 事例在 CH0 / CH1 的最小值 A_min 分布
    amin_ch0 = _compute_min_in_window(phys_waveforms, ch0_idx, min_window)
    amin_ch1 = _compute_min_in_window(phys_waveforms, ch1_idx, min_window)

    # 2. 用样本均值 / 标准差拟合 A_min 的高斯分布
    mu_ch0 = float(amin_ch0.mean())
    sigma_ch0 = float(amin_ch0.std(ddof=1))
    mu_ch1 = float(amin_ch1.mean())
    sigma_ch1 = float(amin_ch1.std(ddof=1))

    if sigma_ch0 <= 0 or sigma_ch1 <= 0:
        raise ValueError(
            f"ACT 最小值分布的标准差为 0（或数值异常），无法进行 mincut："
            f"sigma_ch0={sigma_ch0:.3g}, sigma_ch1={sigma_ch1:.3g}"
        )

    lo_ch0 = mu_ch0 - nsigma * sigma_ch0
    hi_ch0 = mu_ch0 + nsigma * sigma_ch0
    lo_ch1 = mu_ch1 - nsigma * sigma_ch1
    hi_ch1 = mu_ch1 + nsigma * sigma_ch1

    # 3. 对所有事件应用 Min Cut（CH0/CH1 同时满足）
    mask_ch0 = (amin_ch0 >= lo_ch0) & (amin_ch0 <= hi_ch0)
    mask_ch1 = (amin_ch1 >= lo_ch1) & (amin_ch1 <= hi_ch1)
    keep_mask = mask_ch0 & mask_ch1

    stats: Dict[str, np.ndarray] = {
        "amin_ch0": amin_ch0,
        "amin_ch1": amin_ch1,
        "energy": energy,
        "mu_ch0": np.array(mu_ch0),
        "sigma_ch0": np.array(sigma_ch0),
        "mu_ch1": np.array(mu_ch1),
        "sigma_ch1": np.array(sigma_ch1),
        "lo_ch0": np.array(lo_ch0),
        "hi_ch0": np.array(hi_ch0),
        "lo_ch1": np.array(lo_ch1),
        "hi_ch1": np.array(hi_ch1),
    }

    # 4. 打印简要统计信息（便于检查）
    print("=" * 70)
    print("Min Cut 统计信息（基于 ACT 物理事例最小值分布）")
    print("=" * 70)
    print(f"CH0 A_min: μ = {mu_ch0:.3f}, σ = {sigma_ch0:.3f}, 区间 = [{lo_ch0:.3f}, {hi_ch0:.3f}]")
    print(f"CH1 A_min: μ = {mu_ch1:.3f}, σ = {sigma_ch1:.3f}, 区间 = [{lo_ch1:.3f}, {hi_ch1:.3f}]")
    print(f"\n事件总数: {n_events}")
    print(f"通过 Min Cut 的事例数: {int(keep_mask.sum())} "
          f"({keep_mask.mean() * 100:.2f}%)")
    print("=" * 70)

    return keep_mask, stats


__all__ = [
    "mincut_ch0_ch1_from_act",
    "_compute_min_in_window",
]


if __name__ == "__main__":
    """
    使用真实 HDF5 事件数据做一个简单自测：
    - 自动在 data/hdf5/raw_pulse/CH0-3 下寻找 h5 文件
    - 读取其中 CH0/CH1 的波形作为 ACT 事例
    - 使用 mincut_ch0_ch1_from_act 计算最小值分布
    - 画出 CH0 / CH1 的最小值直方图，并标出 μ、μ±nσ
    """
    import os
    import sys
    import h5py
    import matplotlib.pyplot as plt

    # 为了导入 utils.visualize 中的工具，补充项目根下的 python 目录到 sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))          # .../python/data/ge-self/cut
    data_dir = os.path.dirname(current_dir)                           # .../python/data/ge-self
    python_dir = os.path.dirname(os.path.dirname(data_dir))           # .../python
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)

    from utils.visualize import get_h5_files  # type: ignore[import]  # noqa: E402

    # 1. 自动查找 CH0-3 的 HDF5 文件
    h5_files = get_h5_files()
    if "CH0-3" not in h5_files or not h5_files["CH0-3"]:
        raise FileNotFoundError("在 data/hdf5/raw_pulse/CH0-3 目录中未找到 h5 文件，无法进行自测")

    ch0_3_file = h5_files["CH0-3"][0]
    print("=" * 70)
    print("MinCut 自测：使用真实 HDF5 事件数据")
    print("=" * 70)
    print(f"使用文件（CH0-3）：{os.path.basename(ch0_3_file)}")
    print(f"完整路径：{ch0_3_file}")

    # 2. 读取 CH0/CH1 的波形数据，构造成 (n_samples, n_channels, n_events)
    with h5py.File(ch0_3_file, "r") as f:
        if "channel_data" not in f:
            raise KeyError("HDF5 文件中未找到数据集 'channel_data'")

        channel_data = f["channel_data"]
        n_samples, n_channels, n_events = channel_data.shape
        if n_channels < 2:
            raise ValueError(f"CH0-3 文件通道数为 {n_channels} (<2)，无法同时分析 CH0/CH1")

        # 为了避免内存过大，自测时可以限制最大事件数
        max_events = 20000
        use_events = min(n_events, max_events)
        print(f"总事件数: {n_events}，自测使用前 {use_events} 个事件")

        # 只取 CH0/CH1 两个通道，并裁剪事件数
        # 结果形状: (n_samples, 2, use_events)
        phys_waveforms = channel_data[:, 0:2, 0:use_events].astype(np.float64)

    # 这里能量只用于函数接口要求与后续可能的作图，本测试中用 0 占位
    energy = np.zeros(phys_waveforms.shape[2], dtype=np.float64)

    # 3. 计算 mincut 以及统计量
    keep_mask, stats = mincut_ch0_ch1_from_act(
        phys_waveforms,
        energy,
        min_window=(0, 200),
        ch0_idx=0,
        ch1_idx=1,
        nsigma=3.0,
    )

    amin_ch0 = stats["amin_ch0"]
    amin_ch1 = stats["amin_ch1"]
    mu_ch0 = float(stats["mu_ch0"])
    sigma_ch0 = float(stats["sigma_ch0"])
    mu_ch1 = float(stats["mu_ch1"])
    sigma_ch1 = float(stats["sigma_ch1"])
    lo_ch0 = float(stats["lo_ch0"])
    hi_ch0 = float(stats["hi_ch0"])
    lo_ch1 = float(stats["lo_ch1"])
    hi_ch1 = float(stats["hi_ch1"])

    # 设置 x 轴范围为 μ±6σ，并增加 bin 数量
    nsigma_display = 6.0  # 显示范围：μ±6σ
    bins_count = 400  # 增加 bin 数量

    # CH0 的范围
    xmin_ch0 = mu_ch0 - nsigma_display * sigma_ch0
    xmax_ch0 = mu_ch0 + nsigma_display * sigma_ch0

    # CH1 的范围
    xmin_ch1 = mu_ch1 - nsigma_display * sigma_ch1
    xmax_ch1 = mu_ch1 + nsigma_display * sigma_ch1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # CH0 minimum distribution
    ax0 = axes[0]
    ax0.hist(
        amin_ch0,
        bins=bins_count,
        range=(xmin_ch0, xmax_ch0),
        histtype="step",
        color="b",
        linewidth=0.8,
        alpha=0.8,
        label="CH0 A_min",
    )
    ax0.axvline(mu_ch0, color="r", linestyle="--", linewidth=1, alpha=0.8, label=r"$\mu$")
    ax0.axvline(lo_ch0, color="g", linestyle=":", linewidth=1, alpha=0.8, label=r"$\mu \pm 3\sigma$")
    ax0.axvline(hi_ch0, color="g", linestyle=":", linewidth=1, alpha=0.8)
    ax0.set_xlabel("A_min (CH0)", fontsize=9)
    ax0.set_ylabel("Counts", fontsize=9)
    ax0.set_title(f"CH0 Minimum Distribution\n(μ±6σ, bins={bins_count})", fontsize=10)
    ax0.set_xlim(xmin_ch0, xmax_ch0)
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=8)

    # CH1 minimum distribution
    ax1 = axes[1]
    ax1.hist(
        amin_ch1,
        bins=bins_count,
        range=(xmin_ch1, xmax_ch1),
        histtype="step",
        color="b",
        linewidth=0.8,
        alpha=0.8,
        label="CH1 A_min",
    )
    ax1.axvline(mu_ch1, color="r", linestyle="--", linewidth=1, alpha=0.8, label=r"$\mu$")
    ax1.axvline(lo_ch1, color="g", linestyle=":", linewidth=1, alpha=0.8, label=r"$\mu \pm 3\sigma$")
    ax1.axvline(hi_ch1, color="g", linestyle=":", linewidth=1, alpha=0.8)
    ax1.set_xlabel("A_min (CH1)", fontsize=9)
    ax1.set_ylabel("Counts", fontsize=9)
    ax1.set_title(f"CH1 Minimum Distribution\n(μ±6σ, bins={bins_count})", fontsize=10)
    ax1.set_xlim(xmin_ch1, xmax_ch1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    plt.suptitle(
        f"Minimum Value Distribution (Min Cut)\n"
        f"Total events: {use_events}, "
        f"Passed: {int(keep_mask.sum())} ({keep_mask.mean() * 100:.2f}%)",
        fontsize=12,
        y=0.995,
    )
    plt.tight_layout()
    plt.show()

