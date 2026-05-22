#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pedcut 工具函数

用于根据随机触发（RT）事例的基线分布，对自触发物理事例的 CH0 / CH1 进行
“基线 ±nσ” 的 Pedestal Cut，保留基线处于高斯中心 ±3σ 区间内的物理事例。

设计思路
--------
1. 认为“随机触发事例”的 CH0 / CH1 基线服从近似高斯分布，并能代表同一时间段内
   所有自触发物理事例的电子学基线波动。
2. 从 RT 事例中，在给定的时间窗（pedestal_window）上计算每个事件在 CH0 / CH1 的
   基线值（例如前若干个采样点的平均值）。
3. 对这两个通道的基线分布分别做高斯拟合：这里采用样本均值 μ 和样本标准差 σ
   作为高斯分布 N(μ, σ²) 的参数估计。
4. 对物理事例，同样在 pedestal_window 上计算 CH0 / CH1 的基线值，并要求：
      μ_ch0 - nσ_ch0 ≤ ped_ch0_phys ≤ μ_ch0 + nσ_ch0
      μ_ch1 - nσ_ch1 ≤ ped_ch1_phys ≤ μ_ch1 + nσ_ch1
   只有同时满足两条的物理事例才被保留。

本文件只提供纯“数组级”的工具函数，不直接读 / 写 HDF5 文件，方便在其它脚本中复用。
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np

ArrayLike = Union[np.ndarray]


def _compute_baseline(
    waveforms: np.ndarray,
    ch_idx: int,
    pedestal_window: Union[slice, Tuple[int, int]],
) -> np.ndarray:
    """
    在给定时间窗上计算某一通道的事件基线（对时间采样求平均）。

    参数:
        waveforms: 波形数组，形状为 (n_samples, n_channels, n_events)
        ch_idx:    需要提取的通道索引（如 CH0 -> 0, CH1 -> 1）
        pedestal_window:
            - 若为 slice，对应 waveforms[pedestal_window, ch_idx, :] 的时间区间；
            - 若为 (start, stop) 元组，则自动转换为 slice(start, stop)。

    返回:
        baselines: 形状为 (n_events,) 的基线数组，每个事件一个标量基线值。
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

    if isinstance(pedestal_window, tuple):
        if len(pedestal_window) != 2:
            raise ValueError("pedestal_window 元组必须为 (start, stop)")
        start, stop = pedestal_window
        pedestal_slice = slice(start, stop)
    elif isinstance(pedestal_window, slice):
        pedestal_slice = pedestal_window
    else:
        raise TypeError("pedestal_window 必须是 slice 或 (start, stop) 元组")

    # 规范化 slice 边界
    start, stop, step = pedestal_slice.indices(n_samples)
    if step != 1:
        raise ValueError("当前实现仅支持连续的时间窗（step 必须为 1）")
    if start >= stop:
        raise ValueError(f"pedestal_window 无效: start={start}, stop={stop}")

    # 取出该通道在基线时间窗内的波形: 形状 (n_ped_samples, n_events)
    ped_segment = waveforms[pedestal_slice, ch_idx, :]  # type: ignore[arg-type]

    # 沿时间维度求平均，得到每个事件的基线
    baselines = ped_segment.mean(axis=0).astype(np.float64)
    return baselines


def pedestal_cut_ch0_ch1_from_rt(
    phys_waveforms: np.ndarray,
    rt_waveforms: np.ndarray,
    pedestal_window: Union[slice, Tuple[int, int]] = slice(0, 200),
    ch0_idx: int = 0,
    ch1_idx: int = 1,
    nsigma: float = 3.0,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    使用随机触发（RT）事例的基线分布，对自触发物理事例的 CH0/CH1 做 pedestal cut。

    - 假设传入的波形均已对齐（相同的采样率与基线时间窗），形状一致：
          (n_samples, n_channels, n_events)
    - RT 事例用于估计基线的高斯分布参数（μ, σ）。
    - 物理事例中，只有当 CH0/CH1 的基线都落在 [μ - nsigma·σ, μ + nsigma·σ] 内时，才被保留。

    参数:
        phys_waveforms: 物理事例波形，形状 (n_samples, n_channels, n_phys_events)
        rt_waveforms:   随机触发事例波形，形状 (n_samples, n_channels, n_rt_events)
        pedestal_window:
            计算基线的时间窗（slice 或 (start, stop)），默认使用前 200 个采样点。
        ch0_idx:        CH0 的通道索引（默认 0）
        ch1_idx:        CH1 的通道索引（默认 1）
        nsigma:         Pedestal cut 的 σ 倍数，默认 3.0（±3σ）

    返回:
        keep_mask, stats 字典:
        - keep_mask: 形状为 (n_phys_events,) 的 bool 数组，True 表示该物理事例通过 pedcut
        - stats: 包含以下键值的字典（均为 numpy 数组 / 标量）:
            * 'rt_ped_ch0', 'rt_ped_ch1'   : RT 事例的 CH0/CH1 基线数组
            * 'phys_ped_ch0', 'phys_ped_ch1': 物理事例的 CH0/CH1 基线数组
            * 'mu_ch0', 'sigma_ch0'
            * 'mu_ch1', 'sigma_ch1'
            * 'lo_ch0', 'hi_ch0'           : CH0 的 [μ - nsigma·σ, μ + nsigma·σ]
            * 'lo_ch1', 'hi_ch1'           : CH1 的 [μ - nsigma·σ, μ + nsigma·σ]
    """
    # 1. 计算 RT 事例在 CH0 / CH1 的基线分布
    rt_ped_ch0 = _compute_baseline(rt_waveforms, ch0_idx, pedestal_window)
    rt_ped_ch1 = _compute_baseline(rt_waveforms, ch1_idx, pedestal_window)

    # 采用样本均值 / 标准差作为高斯分布参数估计
    mu_ch0 = float(rt_ped_ch0.mean())
    sigma_ch0 = float(rt_ped_ch0.std(ddof=1))  # 无偏估计
    mu_ch1 = float(rt_ped_ch1.mean())
    sigma_ch1 = float(rt_ped_ch1.std(ddof=1))

    # 避免极端情况：σ 过小导致几乎所有点被拒绝
    if sigma_ch0 <= 0 or sigma_ch1 <= 0:
        raise ValueError(
            f"RT 基线标准差为 0（或数值异常），无法进行 pedcut："
            f"sigma_ch0={sigma_ch0:.3g}, sigma_ch1={sigma_ch1:.3g}"
        )

    lo_ch0 = mu_ch0 - nsigma * sigma_ch0
    hi_ch0 = mu_ch0 + nsigma * sigma_ch0
    lo_ch1 = mu_ch1 - nsigma * sigma_ch1
    hi_ch1 = mu_ch1 + nsigma * sigma_ch1

    # 2. 计算物理事例在 CH0 / CH1 的基线
    phys_ped_ch0 = _compute_baseline(phys_waveforms, ch0_idx, pedestal_window)
    phys_ped_ch1 = _compute_baseline(phys_waveforms, ch1_idx, pedestal_window)

    # 3. 对物理事例应用 ±nsigma pedcut
    mask_ch0 = (phys_ped_ch0 >= lo_ch0) & (phys_ped_ch0 <= hi_ch0)
    mask_ch1 = (phys_ped_ch1 >= lo_ch1) & (phys_ped_ch1 <= hi_ch1)
    keep_mask = mask_ch0 & mask_ch1

    stats: Dict[str, np.ndarray] = {
        "rt_ped_ch0": rt_ped_ch0,
        "rt_ped_ch1": rt_ped_ch1,
        "phys_ped_ch0": phys_ped_ch0,
        "phys_ped_ch1": phys_ped_ch1,
        "mu_ch0": np.array(mu_ch0),
        "sigma_ch0": np.array(sigma_ch0),
        "mu_ch1": np.array(mu_ch1),
        "sigma_ch1": np.array(sigma_ch1),
        "lo_ch0": np.array(lo_ch0),
        "hi_ch0": np.array(hi_ch0),
        "lo_ch1": np.array(lo_ch1),
        "hi_ch1": np.array(hi_ch1),
    }

    print("=" * 70)
    print("Pedestal cut 统计信息（基于随机触发 RT 事例）")
    print("=" * 70)
    print(f"CH0: μ = {mu_ch0:.3f}, σ = {sigma_ch0:.3f}, 区间 = [{lo_ch0:.3f}, {hi_ch0:.3f}]")
    print(f"CH1: μ = {mu_ch1:.3f}, σ = {sigma_ch1:.3f}, 区间 = [{lo_ch1:.3f}, {hi_ch1:.3f}]")
    print(f"\n物理事例总数: {phys_ped_ch0.size}")
    print(f"通过 Pedestal Cut 的事例数: {int(keep_mask.sum())} "
          f"({keep_mask.mean() * 100:.2f}%)")
    print("=" * 70)

    return keep_mask, stats


__all__ = [
    "pedestal_cut_ch0_ch1_from_rt",
    "_compute_baseline",
]


if __name__ == "__main__":
    def _discover_project_root() -> str:
        here = os.path.abspath(__file__)
        cut_dir = os.path.dirname(here)
        ge_self_dir = os.path.dirname(cut_dir)
        data_dir = os.path.dirname(ge_self_dir)
        python_dir = os.path.dirname(data_dir)
        return os.path.dirname(python_dir)

    def _read_one_param_pair(args: Tuple[str, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ch0_path, ch1_path = args
        with h5py.File(ch0_path, "r") as f0, h5py.File(ch1_path, "r") as f1:
            for key in ("ch0ped_mean", "ch0pedt_mean"):
                if key not in f0:
                    raise KeyError(f"{os.path.basename(ch0_path)} 缺少数据集 {key}")
            for key in ("ch1ped_mean", "ch1pedt_mean"):
                if key not in f1:
                    raise KeyError(f"{os.path.basename(ch1_path)} 缺少数据集 {key}")
            ch0_front = np.asarray(f0["ch0ped_mean"][...], dtype=np.float64)
            ch0_tail = np.asarray(f0["ch0pedt_mean"][...], dtype=np.float64)
            ch1_front = np.asarray(f1["ch1ped_mean"][...], dtype=np.float64)
            ch1_tail = np.asarray(f1["ch1pedt_mean"][...], dtype=np.float64)
            n = min(ch0_front.size, ch0_tail.size, ch1_front.size, ch1_tail.size)
            return ch0_front[:n], ch0_tail[:n], ch1_front[:n], ch1_tail[:n]

    project_root = _discover_project_root()
    ch0_dir = os.path.join(project_root, "data", "hdf5", "raw_pulse", "CH0_parameters")
    ch1_dir = os.path.join(project_root, "data", "hdf5", "raw_pulse", "CH1_parameters")
    if not os.path.isdir(ch0_dir) or not os.path.isdir(ch1_dir):
        raise FileNotFoundError("CH0_parameters 或 CH1_parameters 目录不存在。")

    ch0_files = sorted([n for n in os.listdir(ch0_dir) if n.lower().endswith((".h5", ".hdf5"))])
    ch1_set = set(n for n in os.listdir(ch1_dir) if n.lower().endswith((".h5", ".hdf5")))
    pairs = [(os.path.join(ch0_dir, n), os.path.join(ch1_dir, n)) for n in ch0_files if n in ch1_set]
    if not pairs:
        raise RuntimeError("未找到可配对的 CH0/CH1 参数文件。")

    cpu_count = os.cpu_count() or 1
    print("=" * 70)
    print(f"使用 CH0/CH1 参数文件并行绘图，文件对数: {len(pairs)}，CPU 核心: {cpu_count}")
    print("=" * 70)

    ch0_front_all = []
    ch1_front_all = []
    with ProcessPoolExecutor(max_workers=cpu_count) as ex:
        fut_map = {ex.submit(_read_one_param_pair, pair): pair for pair in pairs}
        done = 0
        for fut in as_completed(fut_map):
            c0f, _c0t, c1f, _c1t = fut.result()
            ch0_front_all.append(c0f)
            ch1_front_all.append(c1f)
            done += 1
            if done % 20 == 0 or done == len(pairs):
                print(f"已完成 {done}/{len(pairs)} 文件对")

    front_ped_ch0 = np.concatenate(ch0_front_all)
    front_ped_ch1 = np.concatenate(ch1_front_all)

    def _mean_sigma(arr: np.ndarray) -> tuple[float, float]:
        return float(arr.mean()), float(arr.std(ddof=1))

    mu_f_ch0, sig_f_ch0 = _mean_sigma(front_ped_ch0)
    mu_f_ch1, sig_f_ch1 = _mean_sigma(front_ped_ch1)

    nsigma_display = 6.0
    nsigma_cut = 3.0
    bins_count = 400

    def _range_mu_sigma(mu: float, sigma: float, ns: float = 6.0) -> tuple[float, float]:
        if sigma <= 0:
            return mu - 1.0, mu + 1.0
        return mu - ns * sigma, mu + ns * sigma

    xmin_f_ch0, xmax_f_ch0 = _range_mu_sigma(mu_f_ch0, sig_f_ch0, nsigma_display)
    xmin_f_ch1, xmax_f_ch1 = _range_mu_sigma(mu_f_ch1, sig_f_ch1, nsigma_display)
    lo_f_ch0, hi_f_ch0 = mu_f_ch0 - nsigma_cut * sig_f_ch0, mu_f_ch0 + nsigma_cut * sig_f_ch0
    lo_f_ch1, hi_f_ch1 = mu_f_ch1 - nsigma_cut * sig_f_ch1, mu_f_ch1 + nsigma_cut * sig_f_ch1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    # 字体大小与 inhibit/select.py 保持同量级
    label_fs = 12
    legend_fs = 8

    ax = axes[0]
    ax.hist(front_ped_ch0, bins=bins_count, range=(xmin_f_ch0, xmax_f_ch0), histtype="step", color="C0", label="CH0 Ped",linewidth=2.0)
    ax.axvline(mu_f_ch0, color="C1", linestyle="--", label=r"$\mu$")
    ax.axvline(lo_f_ch0, color="g", linestyle=":", label=r"$\mu \pm 3\sigma$")
    ax.axvline(hi_f_ch0, color="g", linestyle=":")
    ax.set_xlabel("Baseline (CH0, ped)", fontsize=label_fs)
    ax.set_ylabel("Counts", fontsize=label_fs)
    ax.set_xlim(xmin_f_ch0, xmax_f_ch0)
    ax.legend(fontsize=legend_fs)

    ax = axes[1]
    ax.hist(front_ped_ch1, bins=bins_count, range=(xmin_f_ch1, xmax_f_ch1), histtype="step", color="C0", label="CH1 Ped",linewidth=2.0)
    ax.axvline(mu_f_ch1, color="C1", linestyle="--", label=r"$\mu$")
    ax.axvline(lo_f_ch1, color="g", linestyle=":", label=r"$\mu \pm 3\sigma$")
    ax.axvline(hi_f_ch1, color="g", linestyle=":")
    ax.set_xlabel("Baseline (CH1, ped)", fontsize=label_fs)
    ax.set_xlim(xmin_f_ch1, xmax_f_ch1)
    ax.legend(fontsize=legend_fs)

    plt.tight_layout()
    plt.show()

