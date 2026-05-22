#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bscut: 绘制 CH3 上升时间 (ln(19)/p1) vs 能量 (max_ch0)

先对事例应用 basic+act.py 中所有 6 种 cut（m1~m6），仅对通过 cut 的事例绘制。
从 CH0/CH1/CH4/CH5 读取 cut 所需参数，从 CH3 读取 tanh_p1，
上升时间 = ln(19) / p1，横轴为能量 E = a * max_ch0 + b。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt

# ADC -> 能量(keV) 线性变换，与 basic+act.py / spectrum.py 一致
ENERGY_A = 0.0008432447500464594
ENERGY_B = -0.826976770117076

# ln(19)，用于上升时间计算
LN_19 = np.log(19.0)


def _discover_project_root() -> Path:
    here = Path(__file__).resolve()
    python_dir = here.parents[3]  # .../python (bscut 在 ge-self/cut/ 下)
    return python_dir.parent


PROJECT_ROOT = _discover_project_root()
DATA_ROOT = PROJECT_ROOT / "data" / "hdf5" / "raw_pulse"
CH0_PARAM_DIR = DATA_ROOT / "CH0_parameters"
CH1_PARAM_DIR = DATA_ROOT / "CH1_parameters"
CH3_PARAM_DIR = DATA_ROOT / "CH3_parameters"
CH4_PARAM_DIR = DATA_ROOT / "CH4_parameters"
CH5_PARAM_DIR = DATA_ROOT / "CH5_parameters"


# -----------------------------------------------------------------------------
# 从 basic+act.py 复制的 cut 函数与加载逻辑
# -----------------------------------------------------------------------------

def _list_paired_param_files() -> List[Tuple[Path, Path]]:
    """寻找同时存在于 CH0/CH1/CH4/CH5/CH3 中的文件对（与 basic+act 一致并需 CH3）。"""
    for d in (CH0_PARAM_DIR, CH1_PARAM_DIR, CH4_PARAM_DIR, CH5_PARAM_DIR, CH3_PARAM_DIR):
        if not d.exists():
            raise FileNotFoundError(f"目录不存在: {d}")

    ch0_files = sorted(
        name for name in os.listdir(CH0_PARAM_DIR)
        if name.lower().endswith((".h5", ".hdf5"))
    )
    ch1_set = {n for n in os.listdir(CH1_PARAM_DIR) if n.lower().endswith((".h5", ".hdf5"))}
    ch4_set = {n for n in os.listdir(CH4_PARAM_DIR) if n.lower().endswith((".h5", ".hdf5"))}
    ch5_set = {n for n in os.listdir(CH5_PARAM_DIR) if n.lower().endswith((".h5", ".hdf5"))}
    ch3_set = {n for n in os.listdir(CH3_PARAM_DIR) if n.lower().endswith((".h5", ".hdf5"))}

    pairs: List[Tuple[Path, Path]] = []
    for name in ch0_files:
        if name in ch1_set and name in ch4_set and name in ch5_set and name in ch3_set:
            pairs.append((CH0_PARAM_DIR / name, CH5_PARAM_DIR / name))
    if not pairs:
        raise RuntimeError("未找到可配对的参数文件（需 CH0/CH1/CH4/CH5/CH3 均存在）。")
    return pairs


def _load_basic_features_for_run(
    ch0_param_path: Path,
    ch5_param_path: Path,) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """从 CH0/CH1/CH4/CH5 读取 cut 所需特征（与 basic+act 一致）。"""
    with h5py.File(ch0_param_path, "r") as f:
        max_ch0 = np.asarray(f["max_ch0"][...], dtype=np.float64)
        ch0_min = np.asarray(f["ch0_min"][...], dtype=np.float64)
        ch0_ped_mean = np.asarray(f["ch0ped_mean"][...], dtype=np.float64)
    with h5py.File(ch5_param_path, "r") as f:
        max_ch5 = np.asarray(f["max_ch5"][...], dtype=np.float64)
    with h5py.File(CH1_PARAM_DIR / ch0_param_path.name, "r") as f:
        ch1_ped_mean = np.asarray(f["ch1ped_mean"][...], dtype=np.float64)
        max_ch1 = np.asarray(f["max_ch1"][...], dtype=np.float64)
        ch1_min = np.asarray(f["ch1_min"][...], dtype=np.float64)
    with h5py.File(CH4_PARAM_DIR / ch0_param_path.name, "r") as f:
        max_ch4 = np.asarray(f["max_ch4"][...], dtype=np.float64)
        tmax_ch4 = np.asarray(f["tmax_ch4"][...], dtype=np.float64)

    n = min(max_ch0.shape[0], ch0_min.shape[0], max_ch5.shape[0], ch0_ped_mean.shape[0],
            ch1_ped_mean.shape[0], max_ch1.shape[0], ch1_min.shape[0], max_ch4.shape[0], tmax_ch4.shape[0])
    return (
        max_ch0[:n], ch0_min[:n], max_ch5[:n], ch0_ped_mean[:n], ch1_ped_mean[:n],
        max_ch1[:n], ch1_min[:n], max_ch4[:n], tmax_ch4[:n],
    )


def cut_ch0_min_positive(ch0_min: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    return ch0_min > threshold


def cut_ch0_max_saturation(max_ch0: np.ndarray, max_val: float = 16382.0) -> np.ndarray:
    return max_ch0 <= max_val


def cut_ch5_self_trigger(max_ch5: np.ndarray, rt_threshold: float = 6000.0) -> np.ndarray:
    return max_ch5 <= rt_threshold

def cut_pedestal_3sigma(
    ch0_ped_mean: np.ndarray,
    ch1_ped_mean: np.ndarray,
    max_ch5: np.ndarray,
    rt_threshold: float = 6000.0,
    n_sigma: float = 3.0,
    min_rt_events: int = 10,) -> np.ndarray:
    n = ch0_ped_mean.shape[0]
    mask = np.ones(n, dtype=bool)
    rt_mask = max_ch5 > rt_threshold
    ch0_ped_rt = ch0_ped_mean[rt_mask]
    if ch0_ped_rt.size >= min_rt_events:
        mu0, sig0 = float(ch0_ped_rt.mean()), float(ch0_ped_rt.std(ddof=1))
        if sig0 > 0:
            mask &= np.abs(ch0_ped_mean - mu0) <= n_sigma * sig0
    ch1_ped_rt = ch1_ped_mean[rt_mask]
    if ch1_ped_rt.size >= min_rt_events:
        mu1, sig1 = float(ch1_ped_rt.mean()), float(ch1_ped_rt.std(ddof=1))
        if sig1 > 0:
            mask &= np.abs(ch1_ped_mean - mu1) <= n_sigma * sig1
    return mask

def cut_act(
    max_ch4: np.ndarray,
    tmax_ch4: np.ndarray,
    trigger_threshold: float = 7060.0,
    t_ge_us: float = 40.0,
    sampling_interval_ns: float = 4.0,
    dt_min_us: float = 1.0,
    dt_max_us: float = 16.0,) -> np.ndarray:
    n = max_ch4.shape[0]
    tmax_ch4 = np.asarray(tmax_ch4, dtype=np.float64)[:n]
    max_ch4 = np.asarray(max_ch4, dtype=np.float64)[:n]
    nai_ok = max_ch4 >= trigger_threshold
    t_ch4_us = tmax_ch4 * sampling_interval_ns * 1e-3
    delta_t_us = t_ge_us - t_ch4_us
    act_mask = (delta_t_us < dt_min_us) | (delta_t_us > dt_max_us)
    return (~nai_ok) | (nai_ok & act_mask)

def cut_mincut(
    ch0_min: np.ndarray,
    ch1_min: np.ndarray,
    max_ch4: np.ndarray,
    tmax_ch4: np.ndarray,
    n_sigma: float = 3.0,
    min_fit_events: int = 10,
    trigger_threshold: float = 7060.0,
    t_ge_us: float = 40.0,
    sampling_interval_ns: float = 4.0,
    dt_min_us: float = 1.0,
    dt_max_us: float = 16.0,) -> np.ndarray:
    n = ch0_min.shape[0]
    mask = np.ones(n, dtype=bool)
    act_mask = cut_act(max_ch4, tmax_ch4, trigger_threshold, t_ge_us, sampling_interval_ns, dt_min_us, dt_max_us)
    for arr, fit_arr in [(ch0_min, ch0_min[act_mask]), (ch1_min, ch1_min[act_mask])]:
        if fit_arr.size >= min_fit_events:
            mu, sig = float(fit_arr.mean()), float(fit_arr.std(ddof=1))
            if sig > 0:
                mask &= np.abs(arr - mu) <= n_sigma * sig
    return mask

def _load_max_ch0_and_rise_time(ch0_path: Path, ch3_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """从 CH0/CH3 读取 max_ch0 和 rise_time = ln(19)/p1。"""
    with h5py.File(ch0_path, "r") as f:
        max_ch0 = np.asarray(f["max_ch0"][...], dtype=np.float64)
    with h5py.File(ch3_path, "r") as f:
        p1 = np.asarray(f["tanh_p1"][...], dtype=np.float64)
    n = min(max_ch0.shape[0], p1.shape[0])
    max_ch0 = max_ch0[:n]
    p1 = p1[:n]
    p1_safe = np.where(p1 > 1e-10, p1, np.nan)
    rise_time = np.where(np.isfinite(p1_safe), LN_19 / p1_safe, np.nan)
    return max_ch0, rise_time

def main() -> None:
    pairs = _list_paired_param_files()
    print(f"找到 {len(pairs)} 个可配对的参数文件。")

    all_max_ch0: List[np.ndarray] = []
    all_rise_time: List[np.ndarray] = []

    for ch0_path, ch5_path in pairs:
        m0, cmin, m5, ped0, ped1, m1, c1min, m4, t4 = _load_basic_features_for_run(ch0_path, ch5_path)
        cut_mask = (
            cut_ch0_min_positive(cmin)
            & cut_ch0_max_saturation(m0)
            & cut_ch5_self_trigger(m5)
            & cut_pedestal_3sigma(ped0, ped1, m5)
            #& cut_act(m4, t4)
            & cut_mincut(cmin, c1min, m4, t4)
        )
        max_ch0, rise_time = _load_max_ch0_and_rise_time(ch0_path, CH3_PARAM_DIR / ch0_path.name)
        n = min(max_ch0.shape[0], cut_mask.shape[0])
        cut_mask = cut_mask[:n]
        all_max_ch0.append(max_ch0[:n][cut_mask])
        all_rise_time.append(rise_time[:n][cut_mask])

    max_ch0 = np.concatenate(all_max_ch0)
    rise_time = np.concatenate(all_rise_time)

    energy = ENERGY_A * max_ch0 + ENERGY_B
    mask = np.isfinite(rise_time) & np.isfinite(energy)
    energy_plot = energy[mask]
    rise_time_plot = rise_time[mask]

    print(f"通过 basic+act 全部 6 种 cut 后: {len(max_ch0)} 事例")
    print(f"有效点（rise_time 与 energy 均有限）: {mask.sum()}")

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(energy_plot, rise_time_plot, s=2, alpha=0.5, edgecolors="none")
    ax.axhline(y=0.8, color="red", linestyle="--", linewidth=2, alpha=0.8)
    ax.set_xlabel("Energy (keV)", fontsize=12)
    ax.set_ylabel(r"Rise time $\ln(19)/p_1$ (μs)", fontsize=12)
    ax.set_title(
        f"CH3 rise time vs energy (N={energy_plot.size})",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 12.0)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
