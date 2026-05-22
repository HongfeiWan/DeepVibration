#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
combine_spectrum.py

功能：
1) 读取 30 参数 UMAP+HDBSCAN 的事件映射 HDF5，选择指定 cluster 的所有事件；
   同时将对应源文件中的 RT 事件也并入（RT 事件计数也算入谱中）。
2) 使用 basic+acv.py 的切割逻辑（CH0/CH5/CH1/CH4 参数文件 + 一系列 cut），得到通过所有 cuts
   的事件集合，并计算其能谱。
3) 将 (1) 与 (2) 两条能谱画在同一个图上，实现“能谱叠加”。
4) cluster 那条能谱仍然执行基本的 10–11 keV 高斯 + 线性本底拟合；
   basic+acv 那条在图之外额外绘制 CH0max vs CH1max 散点图（与 basic+acv.py 保持一致）。

默认行为尽量与 spectrum.py / basic+acv.py 一致：
- 能量换算 E(keV) 使用相同线性系数
- 能谱归一化使用相同的曝光参数：0.5 kg * 20 day
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Energy conversion: E = a * x + b (keV)
ENERGY_A = 0.0008432447500464594
ENERGY_B = -0.826976770117076

# Spectrum normalization: Rate [counts/(keV*kg*day)]
EXPOSURE_KG = 0.5
EXPOSURE_DAYS = 16.5

# 能谱 bin 固定范围，与 acvcombine_spectrum 一一对应，便于效率修正
E_MIN_BIN = 0.0
E_MAX_BIN = 12.0


def _project_root_from_script() -> Path:
    """
    推断 DeepVibration 项目根目录：
    当前文件位于 .../python/data/ge-self/cut/parameterize/umap-optimazation-2/combine_spectrum.py
    """

    script_dir = Path(__file__).resolve().parent
    parameterize_dir = script_dir.parent  # .../parameterize
    cut_dir = parameterize_dir.parent  # .../cut
    ge_self_dir = cut_dir.parent  # .../ge-self
    data_dir = ge_self_dir.parent  # .../data
    python_dir = data_dir.parent  # .../python
    return python_dir.parent  # .../DeepVibration


PROJECT_ROOT = _project_root_from_script()
DATA_ROOT = PROJECT_ROOT / "data" / "hdf5" / "raw_pulse"
CH0_PARAM_DIR = DATA_ROOT / "CH0_parameters"
CH1_PARAM_DIR = DATA_ROOT / "CH1_parameters"
CH2_PARAM_DIR = DATA_ROOT / "CH2_parameters"
CH3_PARAM_DIR = DATA_ROOT / "CH3_parameters"
CH4_PARAM_DIR = DATA_ROOT / "CH4_parameters"
CH5_PARAM_DIR = DATA_ROOT / "CH5_parameters"

# bscut 上升时间阈值 (samples): ln(19)/p1 < BS_RISE_TIME_THRESHOLD
LN_19 = np.log(19.0)
BS_RISE_TIME_THRESHOLD = 0.8


def _list_paired_param_files() -> List[Tuple[Path, Path]]:
    """
    基于 CH0_parameters 目录的文件名，寻找同时存在于 CH1_parameters 与 CH4_parameters 与 CH5_parameters 中的参数文件对。
    """

    if not CH0_PARAM_DIR.exists():
        raise FileNotFoundError(f"CH0_parameters 目录不存在: {CH0_PARAM_DIR}")
    if not CH1_PARAM_DIR.exists():
        raise FileNotFoundError(f"CH1_parameters 目录不存在: {CH1_PARAM_DIR}")
    if not CH4_PARAM_DIR.exists():
        raise FileNotFoundError(f"CH4_parameters 目录不存在: {CH4_PARAM_DIR}")
    if not CH5_PARAM_DIR.exists():
        raise FileNotFoundError(f"CH5_parameters 目录不存在: {CH5_PARAM_DIR}")

    ch0_files = sorted(
        name for name in os.listdir(CH0_PARAM_DIR) if name.lower().endswith((".h5", ".hdf5"))
    )
    if not ch0_files:
        raise FileNotFoundError(f"CH0_parameters 目录下未找到任何 h5 文件: {CH0_PARAM_DIR}")

    ch1_existing = {name for name in os.listdir(CH1_PARAM_DIR) if name.lower().endswith((".h5", ".hdf5"))}
    ch4_existing = {name for name in os.listdir(CH4_PARAM_DIR) if name.lower().endswith((".h5", ".hdf5"))}
    ch5_existing = {name for name in os.listdir(CH5_PARAM_DIR) if name.lower().endswith((".h5", ".hdf5"))}

    pairs: List[Tuple[Path, Path]] = []
    for name in ch0_files:
        if name in ch1_existing and name in ch4_existing and name in ch5_existing:
            pairs.append((CH0_PARAM_DIR / name, CH5_PARAM_DIR / name))

    if not pairs:
        raise RuntimeError(
            f"在 {CH0_PARAM_DIR}、{CH1_PARAM_DIR}、{CH4_PARAM_DIR} 与 {CH5_PARAM_DIR} 中未找到任何可配对的参数文件。"
        )

    return pairs

def _list_paired_param_files_with_ch3() -> List[Tuple[Path, Path]]:
    """与 _list_paired_param_files 相同，但额外要求 CH3_parameters 中存在对应文件。"""
    base_pairs = _list_paired_param_files()
    if not CH3_PARAM_DIR.exists():
        return []
    ch3_existing = {n for n in os.listdir(CH3_PARAM_DIR) if n.lower().endswith((".h5", ".hdf5"))}
    return [(ch0, ch5) for ch0, ch5 in base_pairs if ch0.name in ch3_existing]

def _list_paired_param_files_with_ch2_ch3() -> List[Tuple[Path, Path]]:
    """与 _list_paired_param_files 相同，但额外要求 CH2/CH3_parameters 中均存在对应文件。"""
    base_pairs = _list_paired_param_files()
    if not CH2_PARAM_DIR.exists() or not CH3_PARAM_DIR.exists():
        return []
    ch2_existing = {n for n in os.listdir(CH2_PARAM_DIR) if n.lower().endswith((".h5", ".hdf5"))}
    ch3_existing = {n for n in os.listdir(CH3_PARAM_DIR) if n.lower().endswith((".h5", ".hdf5"))}
    return [(ch0, ch5) for ch0, ch5 in base_pairs if (ch0.name in ch2_existing and ch0.name in ch3_existing)]

def _load_ch2_ch3_fit_quality_aligned(ch0_param_path: Path, n_events: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从 CH2/CH3 参数文件读取拟合质量相关量，并与 n_events 对齐：
        - ch2_n_fit_points
        - ch3_n_fit_points
        - ch2_tanh_p0
        - ch3_tanh_p0
    缺文件/缺数据集时返回默认值（n_fit_points=0, tanh_p0=NaN）。
    """
    ch2_n_fit_points = np.zeros(n_events, dtype=np.int32)
    ch3_n_fit_points = np.zeros(n_events, dtype=np.int32)
    ch2_tanh_p0 = np.full(n_events, np.nan, dtype=np.float64)
    ch3_tanh_p0 = np.full(n_events, np.nan, dtype=np.float64)

    ch2_path = CH2_PARAM_DIR / ch0_param_path.name
    if ch2_path.is_file():
        try:
            with h5py.File(ch2_path, "r") as f:
                if "n_fit_points" in f:
                    n2 = np.asarray(f["n_fit_points"][...], dtype=np.int32)
                    n = min(int(n2.shape[0]), n_events)
                    if n > 0:
                        ch2_n_fit_points[:n] = n2[:n]
                if "tanh_p0" in f:
                    p2 = np.asarray(f["tanh_p0"][...], dtype=np.float64)
                    n = min(int(p2.shape[0]), n_events)
                    if n > 0:
                        ch2_tanh_p0[:n] = p2[:n]
        except Exception:
            pass

    ch3_path = CH3_PARAM_DIR / ch0_param_path.name
    if ch3_path.is_file():
        try:
            with h5py.File(ch3_path, "r") as f:
                if "n_fit_points" in f:
                    n3 = np.asarray(f["n_fit_points"][...], dtype=np.int32)
                    n = min(int(n3.shape[0]), n_events)
                    if n > 0:
                        ch3_n_fit_points[:n] = n3[:n]
                if "tanh_p0" in f:
                    p3 = np.asarray(f["tanh_p0"][...], dtype=np.float64)
                    n = min(int(p3.shape[0]), n_events)
                    if n > 0:
                        ch3_tanh_p0[:n] = p3[:n]
        except Exception:
            pass

    return ch2_n_fit_points, ch3_n_fit_points, ch2_tanh_p0, ch3_tanh_p0

def _load_ch3_ped_min_aligned(ch0_param_path: Path, n_events: int) -> tuple[np.ndarray, np.ndarray]:
    """
    从 CH3 参数文件读取 ch3ped_mean / min_ch3，并与 n_events 对齐。
    缺文件/缺数据集时返回 NaN 数组，确保 cut_ch3ped_min 自动不通过。
    """
    ch3ped_mean = np.full(n_events, np.nan, dtype=np.float64)
    min_ch3 = np.full(n_events, np.nan, dtype=np.float64)
    ch3_path = CH3_PARAM_DIR / ch0_param_path.name
    if not ch3_path.is_file():
        return ch3ped_mean, min_ch3
    try:
        with h5py.File(ch3_path, "r") as f:
            if "ch3ped_mean" in f:
                x = np.asarray(f["ch3ped_mean"][...], dtype=np.float64)
                n = min(int(x.shape[0]), n_events)
                if n > 0:
                    ch3ped_mean[:n] = x[:n]
            if "min_ch3" in f:
                y = np.asarray(f["min_ch3"][...], dtype=np.float64)
                n = min(int(y.shape[0]), n_events)
                if n > 0:
                    min_ch3[:n] = y[:n]
    except Exception:
        pass
    return ch3ped_mean, min_ch3

def cut_fit_success(
    ch2_n_fit_points: np.ndarray,
    ch3_n_fit_points: np.ndarray,
    ch2_tanh_p0: np.ndarray,
    ch3_tanh_p0: np.ndarray,
    bad_val: float = 1e6,) -> np.ndarray:
    """
    过滤 fit_ch2_ch3_parallel.py 中拟合失败/未参与拟合的事件。

    判定规则（同时满足）：
    - CH2 与 CH3 的 n_fit_points 都 > 0；
    - CH2 与 CH3 的 tanh_p0 为有限值，且不等于异常值 bad_val（默认 1e6）。
    """
    ch2_n = np.asarray(ch2_n_fit_points, dtype=np.int32)
    ch3_n = np.asarray(ch3_n_fit_points, dtype=np.int32)
    ch2_p0 = np.asarray(ch2_tanh_p0, dtype=np.float64)
    ch3_p0 = np.asarray(ch3_tanh_p0, dtype=np.float64)
    n = min(ch2_n.shape[0], ch3_n.shape[0], ch2_p0.shape[0], ch3_p0.shape[0])
    ch2_n = ch2_n[:n]
    ch3_n = ch3_n[:n]
    ch2_p0 = ch2_p0[:n]
    ch3_p0 = ch3_p0[:n]
    ok_npts = (ch2_n > 0) & (ch3_n > 0)
    ok_ch2 = np.isfinite(ch2_p0) & (~np.isclose(ch2_p0, bad_val))
    ok_ch3 = np.isfinite(ch3_p0) & (~np.isclose(ch3_p0, bad_val))
    return ok_npts & ok_ch2 & ok_ch3

def _load_basic_features_for_run(
    ch0_param_path: Path,
    ch5_param_path: Path,) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,]:
    """
    从单个 run 的 CH0/CH5/CH1/CH4 参数文件中读取基准特征（basic+acv.py 保持一致）。
    """

    with h5py.File(ch0_param_path, "r") as f_ch0:
        if (
            "max_ch0" not in f_ch0
            or "ch0_min" not in f_ch0
            or "ch0ped_mean" not in f_ch0
            or "tmax_ch0" not in f_ch0
        ):
            raise KeyError(
                f"{ch0_param_path.name} 中缺少 max_ch0 / ch0_min / ch0ped_mean / tmax_ch0 数据集，"
                "请确认该文件由当前版本的 preprocessor.py 生成。"
            )
        max_ch0 = np.asarray(f_ch0["max_ch0"][...], dtype=np.float64)
        ch0_min = np.asarray(f_ch0["ch0_min"][...], dtype=np.float64)
        ch0_ped_mean = np.asarray(f_ch0["ch0ped_mean"][...], dtype=np.float64)
        tmax_ch0 = np.asarray(f_ch0["tmax_ch0"][...], dtype=np.float64)

    with h5py.File(ch5_param_path, "r") as f_ch5:
        if "max_ch5" not in f_ch5:
            raise KeyError(
                f"{ch5_param_path.name} 中缺少 max_ch5 数据集，"
                "请确认该文件由当前版本的 preprocessor.py 生成。"
            )
        max_ch5 = np.asarray(f_ch5["max_ch5"][...], dtype=np.float64)

    ch1_param_path = CH1_PARAM_DIR / ch0_param_path.name
    with h5py.File(ch1_param_path, "r") as f_ch1:
        if (
            "ch1ped_mean" not in f_ch1
            or "max_ch1" not in f_ch1
            or "ch1_min" not in f_ch1
            or "tmax_ch1" not in f_ch1
        ):
            raise KeyError(
                f"{ch1_param_path.name} 中缺少 ch1ped_mean / max_ch1 / ch1_min / tmax_ch1 之一，"
                "请确认该文件由当前版本的 preprocessor.py 生成。"
            )
        ch1_ped_mean = np.asarray(f_ch1["ch1ped_mean"][...], dtype=np.float64)
        max_ch1 = np.asarray(f_ch1["max_ch1"][...], dtype=np.float64)
        ch1_min = np.asarray(f_ch1["ch1_min"][...], dtype=np.float64)
        tmax_ch1 = np.asarray(f_ch1["tmax_ch1"][...], dtype=np.float64)

    ch4_param_path = CH4_PARAM_DIR / ch0_param_path.name
    with h5py.File(ch4_param_path, "r") as f_ch4:
        if "max_ch4" not in f_ch4 or "tmax_ch4" not in f_ch4:
            raise KeyError(
                f"{ch4_param_path.name} 中缺少 max_ch4 或 tmax_ch4 数据集，"
                "请确认该文件由当前版本的 preprocessor.py 生成。"
            )
        max_ch4 = np.asarray(f_ch4["max_ch4"][...], dtype=np.float64)
        tmax_ch4 = np.asarray(f_ch4["tmax_ch4"][...], dtype=np.float64)

    n0, nmin, n5, nped0, nped1, n1, nc1min, n4, nt4, nt0, nt1 = (
        max_ch0.shape[0],
        ch0_min.shape[0],
        max_ch5.shape[0],
        ch0_ped_mean.shape[0],
        ch1_ped_mean.shape[0],
        max_ch1.shape[0],
        ch1_min.shape[0],
        max_ch4.shape[0],
        tmax_ch4.shape[0],
        tmax_ch0.shape[0],
        tmax_ch1.shape[0],
    )
    n = min(n0, nmin, n5, nped0, nped1, n1, nc1min, n4, nt4, nt0, nt1)

    if not (n0 == nmin == n5 == nped0 == nped1 == n1 == nc1min == n4 == nt4 == nt0 == nt1):
        print(f"[警告] 事件数不一致，仅使用前 {n} 个事件。")

    max_ch0 = max_ch0[:n]
    ch0_min = ch0_min[:n]
    max_ch5 = max_ch5[:n]
    ch0_ped_mean = ch0_ped_mean[:n]
    ch1_ped_mean = ch1_ped_mean[:n]
    max_ch1 = max_ch1[:n]
    ch1_min = ch1_min[:n]
    max_ch4 = max_ch4[:n]
    tmax_ch4 = tmax_ch4[:n]
    tmax_ch0 = tmax_ch0[:n]
    tmax_ch1 = tmax_ch1[:n]

    return (
        max_ch0,
        ch0_min,
        max_ch5,
        ch0_ped_mean,
        ch1_ped_mean,
        max_ch1,
        ch1_min,
        max_ch4,
        tmax_ch4,
        tmax_ch0,
        tmax_ch1,
    )

def cut_ch0_min_positive(ch0_min: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """条件：ch0_min > threshold（排除抑制信号）。"""

    return ch0_min > threshold

def cut_ch0_max_saturation(max_ch0: np.ndarray, max_ch1: np.ndarray, max_val: float = 16382.0) -> np.ndarray:
    """条件：max_ch0 <= max_val 且 max_ch1 <= max_val（排除 CH0/CH1 饱和事例）。"""

    return (max_ch0 <= max_val) & (max_ch1 <= max_val)

def cut_ch5_self_trigger(max_ch5: np.ndarray, rt_threshold: float = 6000.0) -> np.ndarray:
    """条件：max_ch5 <= rt_threshold（排除随机触发）。"""

    return max_ch5 <= rt_threshold

def cut_pedestal_3sigma(
    ch0_ped_mean: np.ndarray,
    ch1_ped_mean: np.ndarray,
    max_ch5: np.ndarray,
    rt_threshold: float = 6000.0,
    n_sigma: float = 3.0,
    min_rt_events: int = 10,) -> np.ndarray:
    """
    前沿基线 cut：使用随机触发事例 (max_ch5 > rt_threshold) 的 CH0/CH1 pedestal 分别拟合高斯，
    保留 |ch0_ped - μ0| <= n_sigma*σ0 且 |ch1_ped - μ1| <= n_sigma*σ1 的事件。
    """

    n = ch0_ped_mean.shape[0]
    mask = np.ones(n, dtype=bool)
    rt_mask = max_ch5 > rt_threshold

    ch0_ped_rt = ch0_ped_mean[rt_mask]
    if ch0_ped_rt.size >= min_rt_events:
        ped_mu0 = float(ch0_ped_rt.mean())
        ped_sigma0 = float(ch0_ped_rt.std(ddof=1))
        if ped_sigma0 > 0.0:
            mask = mask & (np.abs(ch0_ped_mean - ped_mu0) <= n_sigma * ped_sigma0)

    ch1_ped_rt = ch1_ped_mean[rt_mask]
    if ch1_ped_rt.size >= min_rt_events:
        ped_mu1 = float(ch1_ped_rt.mean())
        ped_sigma1 = float(ch1_ped_rt.std(ddof=1))
        if ped_sigma1 > 0.0:
            mask = mask & (np.abs(ch1_ped_mean - ped_mu1) <= n_sigma * ped_sigma1)

    return mask

def cut_ch3ped_min(
    ch3ped_mean: np.ndarray,
    min_ch3: np.ndarray,
    *,
    sigma_yx: float = 15.0,) -> np.ndarray:
    """
    CH3 ped-min 带状 cut：
    - 要求点位于 y=x 周边 sigma_yx 内（默认 |min_ch3 - ch3ped_mean| <= 20）
    - 并显式要求 min_ch3 > 0。
    """
    x = np.asarray(ch3ped_mean, dtype=np.float64)
    y = np.asarray(min_ch3, dtype=np.float64)
    n = min(x.shape[0], y.shape[0])
    x = x[:n]
    y = y[:n]
    fin = np.isfinite(x) & np.isfinite(y)
    sig = float(sigma_yx)
    return fin & (np.abs(y - x) <= sig) & (y > 0.0)

def cut_acv(
    max_ch4: np.ndarray,
    tmax_ch4: np.ndarray,
    trigger_threshold: float = 7060.0,
    t_ge_us: float = 40.0,
    sampling_interval_ns: float = 4.0,
    dt_min_us: float = 1.0,
    dt_max_us: float = 16.0,) -> np.ndarray:
    """
    acv cut（与 basic+acv.py 保持一致）：
    - 对 NaI 过阈事件 (max_ch4 >= trigger_threshold)，选取 Δt 非 [dt_min_us, dt_max_us] μs 的事例；
    - 对 NaI 未过阈事件，一律保留。
    """

    n = max_ch4.shape[0]
    tmax_ch4 = np.asarray(tmax_ch4, dtype=np.float64)[:n]
    max_ch4 = np.asarray(max_ch4, dtype=np.float64)[:n]

    nai_ok = max_ch4 >= trigger_threshold
    t_ch4_us = tmax_ch4 * sampling_interval_ns * 1e-3
    delta_t_us = t_ge_us - t_ch4_us
    acv_mask = (delta_t_us < dt_min_us) | (delta_t_us > dt_max_us)
    return (~nai_ok) | (nai_ok & acv_mask)

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
    """
    mincut：在 acv 基础上，用 acv 事例拟合 CH0min/CH1min 分布，
    保留 CH0min、CH1min 均在中心值 ± n_sigma*σ 内的事件。
    """

    n = ch0_min.shape[0]
    mask = np.ones(n, dtype=bool)
    acv_mask = cut_acv(max_ch4, tmax_ch4, trigger_threshold, t_ge_us, sampling_interval_ns, dt_min_us, dt_max_us)

    fit_mask = acv_mask
    ch0_min_fit = ch0_min[fit_mask]
    if ch0_min_fit.size >= min_fit_events:
        mu0 = float(ch0_min_fit.mean())
        sigma0 = float(ch0_min_fit.std(ddof=1))
        if sigma0 > 0.0:
            mask = mask & (np.abs(ch0_min - mu0) <= n_sigma * sigma0)

    ch1_min_fit = ch1_min[fit_mask]
    if ch1_min_fit.size >= min_fit_events:
        mu1 = float(ch1_min_fit.mean())
        sigma1 = float(ch1_min_fit.std(ddof=1))
        if sigma1 > 0.0:
            mask = mask & (np.abs(ch1_min - mu1) <= n_sigma * sigma1)

    return mask

def cut_pncut(
    base_mask: np.ndarray,
    max_ch0: np.ndarray,
    max_ch1: np.ndarray,
    fit_ch0_min: float = 3000.0,
    fit_ch0_max: float = 12000.0,
    n_sigma: float = 1.0,
    min_fit_events: int = 10,) -> np.ndarray:
    """
    pncut（与 basic+acv.py 逻辑一致）：
    - 先在 base_mask 内，用 CH0max 落在指定区间的点拟合一条直线；
    - 计算所有事件相对直线的残差，并输出 |r| <= n_sigma * σ 的事件。

    注意：basic+acv.py 当前并未使用 pn_mask 来更新最终 mask，这里同样只计算不叠加。
    """

    n = max_ch0.shape[0]
    assert max_ch1.shape[0] == n and base_mask.shape[0] == n

    fit_mask = base_mask & (max_ch0 > fit_ch0_min) & (max_ch0 < fit_ch0_max)
    x_fit = max_ch0[fit_mask]
    y_fit = max_ch1[fit_mask]

    if x_fit.size < min_fit_events:
        return base_mask.copy()

    a, b = np.polyfit(x_fit, y_fit, deg=1)
    y_pred_fit = a * x_fit + b
    resid_fit = y_fit - y_pred_fit
    sigma = float(resid_fit.std(ddof=1))
    if sigma <= 0.0:
        return base_mask.copy()

    y_pred_all = a * max_ch0 + b
    resid_all = max_ch1 - y_pred_all
    band_mask = np.abs(resid_all) <= n_sigma * sigma
    return band_mask

def cut_ch0max_tmax(
    base_mask: np.ndarray,
    max_ch0: np.ndarray,
    tmax_ch0: np.ndarray,
    fit_x_min: float = 1160.0,
    fit_y_min: float = 10200.0,
    fit_y_max: float = 13500.0,
    n_sigma: float = 3.0,
    min_fit_events: int = 10,) -> np.ndarray:
    """CH0 max vs tmax(CH0) 相关带 cut（与 basic+acv.py 保持一致）。"""
    def _model(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
        return a * np.log(x - b) + c * x + d / x

    n = max_ch0.shape[0]
    assert tmax_ch0.shape[0] == n and base_mask.shape[0] == n
    max_ch0 = np.asarray(max_ch0, dtype=np.float64)
    tmax_ch0 = np.asarray(tmax_ch0, dtype=np.float64)

    fit_mask = base_mask & (max_ch0 > fit_x_min) & (tmax_ch0 > fit_y_min) & (tmax_ch0 < fit_y_max)
    xf = max_ch0[fit_mask]
    yf = tmax_ch0[fit_mask]
    if xf.size < max(4, min_fit_events):
        return base_mask.copy()

    xmin = float(np.min(xf))
    b_hi = xmin - 1e-9
    A0 = np.column_stack([np.log(xf), xf, 1.0 / xf])
    coef0, _, rank0, _ = np.linalg.lstsq(A0, yf, rcond=None)
    if rank0 < 3:
        return base_mask.copy()
    a0, c0, d0 = float(coef0[0]), float(coef0[1]), float(coef0[2])
    b0 = min(xmin * 0.5, xmin - 200.0)
    p0 = np.array([a0, b0, c0, d0], dtype=np.float64)
    bounds = ([-np.inf, -np.inf, -np.inf, -np.inf], [np.inf, b_hi, np.inf, np.inf])
    try:
        popt, _ = curve_fit(_model, xf, yf, p0=p0, bounds=bounds, maxfev=100000)
    except (ValueError, RuntimeError):
        return base_mask.copy()
    a, b, c, d = float(popt[0]), float(popt[1]), float(popt[2]), float(popt[3])
    if not np.isfinite([a, b, c, d]).all():
        return base_mask.copy()
    y_hat_fit = _model(xf, a, b, c, d)
    sigma = float(np.std(yf - y_hat_fit))
    if sigma <= 0.0:
        return base_mask.copy()

    valid = max_ch0 > b + 1e-9
    y_pred = np.empty(n, dtype=np.float64)
    y_pred[:] = np.nan
    y_pred[valid] = _model(max_ch0[valid], a, b, c, d)
    resid = tmax_ch0 - y_pred
    in_band = np.zeros(n, dtype=bool)
    in_band[valid] = np.abs(resid[valid]) <= n_sigma * sigma
    return in_band

def cut_ch1max_tmax(
    base_mask: np.ndarray,
    max_ch1: np.ndarray,
    tmax_ch1: np.ndarray,
    fit_x_min: float = 1350.0,
    fit_y_min: float = 11000.0,
    fit_y_max: float = 17500.0,
    n_sigma: float = 1.0,
    min_fit_events: int = 10,) -> np.ndarray:
    """CH1 max vs tmax(CH1) 相关带 cut（与 basic+acv.py 保持一致）。"""
    def _model(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
        return a * np.log(x - b) + c * x + d / x

    n = max_ch1.shape[0]
    assert tmax_ch1.shape[0] == n and base_mask.shape[0] == n
    max_ch1 = np.asarray(max_ch1, dtype=np.float64)
    tmax_ch1 = np.asarray(tmax_ch1, dtype=np.float64)

    fit_mask = base_mask & (max_ch1 > fit_x_min) & (tmax_ch1 > fit_y_min) & (tmax_ch1 < fit_y_max)
    xf = max_ch1[fit_mask]
    yf = tmax_ch1[fit_mask]
    if xf.size < max(4, min_fit_events):
        return base_mask.copy()

    xmin = float(np.min(xf))
    b_hi = xmin - 1e-9
    A0 = np.column_stack([np.log(xf), xf, 1.0 / xf])
    coef0, _, rank0, _ = np.linalg.lstsq(A0, yf, rcond=None)
    if rank0 < 3:
        return base_mask.copy()
    a0, c0, d0 = float(coef0[0]), float(coef0[1]), float(coef0[2])
    b0 = min(xmin * 0.5, xmin - 200.0)
    p0 = np.array([a0, b0, c0, d0], dtype=np.float64)
    bounds = ([-np.inf, -np.inf, -np.inf, -np.inf], [np.inf, b_hi, np.inf, np.inf])
    try:
        popt, _ = curve_fit(_model, xf, yf, p0=p0, bounds=bounds, maxfev=100000)
    except (ValueError, RuntimeError):
        return base_mask.copy()
    a, b, c, d = float(popt[0]), float(popt[1]), float(popt[2]), float(popt[3])
    if not np.isfinite([a, b, c, d]).all():
        return base_mask.copy()
    y_hat_fit = _model(xf, a, b, c, d)
    sigma = float(np.std(yf - y_hat_fit))
    if sigma <= 0.0:
        return base_mask.copy()

    valid = max_ch1 > b + 1e-9
    y_pred = np.empty(n, dtype=np.float64)
    y_pred[:] = np.nan
    y_pred[valid] = _model(max_ch1[valid], a, b, c, d)
    resid = tmax_ch1 - y_pred
    in_band = np.zeros(n, dtype=bool)
    in_band[valid] = np.abs(resid[valid]) <= n_sigma * sigma
    return in_band

def _compute_basic_acv_mask(
    ch2_n_fit_points: np.ndarray,
    ch3_n_fit_points: np.ndarray,
    ch2_tanh_p0: np.ndarray,
    ch3_tanh_p0: np.ndarray,
    max_ch0: np.ndarray,
    ch0_min: np.ndarray,
    max_ch5: np.ndarray,
    ch0_ped_mean: np.ndarray,
    ch1_ped_mean: np.ndarray,
    max_ch1: np.ndarray,
    ch1_min: np.ndarray,
    ch3ped_mean: np.ndarray,
    min_ch3: np.ndarray,
    max_ch4: np.ndarray,
    tmax_ch4: np.ndarray,
    tmax_ch0: np.ndarray,
    tmax_ch1: np.ndarray,
    log_prefix: str = "[basic+acv]",) -> np.ndarray:
    """按 basic+acv.py 顺序计算 m1~m8 并返回最终 mask。"""
    n_raw = max_ch0.shape[0]
    print(f"{log_prefix} 原始事件数: {n_raw}")

    m0 = cut_fit_success(ch2_n_fit_points, ch3_n_fit_points, ch2_tanh_p0, ch3_tanh_p0)
    print(f"{log_prefix} cut_fit_success 单独使用后: {int(m0.sum())} / {n_raw}")

    m1 = cut_ch0_min_positive(ch0_min)
    print(f"{log_prefix} cut_ch0_min_positive 单独使用后: {int(m1.sum())} / {n_raw}")

    m2 = cut_ch0_max_saturation(max_ch0, max_ch1)
    print(f"{log_prefix} cut_ch0_max_saturation 单独使用后: {int(m2.sum())} / {n_raw}")

    m3 = cut_ch5_self_trigger(max_ch5)
    print(f"{log_prefix} cut_ch5_self_trigger 单独使用后: {int(m3.sum())} / {n_raw}")

    m4 = cut_pedestal_3sigma(ch0_ped_mean, ch1_ped_mean, max_ch5)
    print(f"{log_prefix} cut_pedestal_3sigma 单独使用后: {int(m4.sum())} / {n_raw}")
    m4b = cut_ch3ped_min(ch3ped_mean, min_ch3)
    print(f"{log_prefix} cut_ch3ped_min 单独使用后: {int(m4b.sum())} / {n_raw}")

    m5 = cut_acv(max_ch4, tmax_ch4)
    print(f"{log_prefix} cut_acv 单独使用后: {int(m5.sum())} / {n_raw}")

    m6 = cut_mincut(ch0_min, ch1_min, max_ch4, tmax_ch4)
    print(f"{log_prefix} cut_mincut 单独使用后: {int(m6.sum())} / {n_raw}")

    # m7 = cut_ch0max_tmax(m1 & m2 & m3 & m4 & m5 & m6, max_ch0, tmax_ch0)
    # print(f"{log_prefix} cut_ch0max_tmax 单独使用后: {int((m1 & m2 & m3 & m4 & m5 & m6 & m7).sum())} / {n_raw}")

    # m8 = cut_ch1max_tmax(m1 & m2 & m3 & m4 & m5 & m6 & m7, max_ch1, tmax_ch1)
    # print(f"{log_prefix} cut_ch1max_tmax 单独使用后: {int((m1 & m2 & m3 & m4 & m5 & m6 & m7 & m8).sum())} / {n_raw}")

    return m0 & m1 & m2 & m3 & m4 & m4b & m5 & m6 

def _compute_basic_6mask(
    max_ch0: np.ndarray,
    max_ch1: np.ndarray,
    ch0_min: np.ndarray,
    max_ch5: np.ndarray,
    ch0_ped_mean: np.ndarray,
    ch1_ped_mean: np.ndarray,
    ch1_min: np.ndarray,
    max_ch4: np.ndarray,
    tmax_ch4: np.ndarray,
    log_prefix: str = "[basic_6mask]",) -> np.ndarray:
    """仅使用前六个 mask（m1~m6）。"""
    n_raw = max_ch0.shape[0]
    m1 = cut_ch0_min_positive(ch0_min)
    m2 = cut_ch0_max_saturation(max_ch0, max_ch1)
    m3 = cut_ch5_self_trigger(max_ch5)
    m4 = cut_pedestal_3sigma(ch0_ped_mean, ch1_ped_mean, max_ch5)
    m5 = cut_acv(max_ch4, tmax_ch4)
    m6 = cut_mincut(ch0_min, ch1_min, max_ch4, tmax_ch4)
    mask6 = m1 & m2 & m3 & m4 & m5 & m6
    print(f"{log_prefix} 仅 m1~m6 后剩余: {int(mask6.sum())} / {n_raw}")
    return mask6

def _load_basic_acv_pass_events() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取 basic+acv.py 的全部参数文件，对应 cut 后返回：
    - passed_max_ch0: (N,) 通过所有 cuts 的 CH0max
    - passed_max_ch1: (N,) 通过所有 cuts 的 CH1max
    - passed_energy:  (N,) 对应能量（keV）
    """

    pairs = _list_paired_param_files_with_ch2_ch3()
    print(f"[basic+acv] 找到 {len(pairs)} 个可配对的参数文件。")

    all_max_ch0: List[np.ndarray] = []
    all_ch0_min: List[np.ndarray] = []
    all_max_ch5: List[np.ndarray] = []
    all_ch0_ped_mean: List[np.ndarray] = []
    all_ch1_ped_mean: List[np.ndarray] = []
    all_max_ch1: List[np.ndarray] = []
    all_ch1_min: List[np.ndarray] = []
    all_max_ch4: List[np.ndarray] = []
    all_tmax_ch4: List[np.ndarray] = []
    all_tmax_ch0: List[np.ndarray] = []
    all_tmax_ch1: List[np.ndarray] = []
    all_ch2_n_fit_points: List[np.ndarray] = []
    all_ch3_n_fit_points: List[np.ndarray] = []
    all_ch2_tanh_p0: List[np.ndarray] = []
    all_ch3_tanh_p0: List[np.ndarray] = []
    all_ch3_ped_mean: List[np.ndarray] = []
    all_ch3_min: List[np.ndarray] = []

    for ch0_path, ch5_path in pairs:
        m0, cmin, m5, ped0, ped1, m1, c1min, m4, t4, tc0, tc1 = _load_basic_features_for_run(ch0_path, ch5_path)
        ch2_nfit, ch3_nfit, ch2_p0, ch3_p0 = _load_ch2_ch3_fit_quality_aligned(ch0_path, m0.shape[0])
        ch3_ped, ch3_min = _load_ch3_ped_min_aligned(ch0_path, m0.shape[0])
        all_max_ch0.append(m0)
        all_ch0_min.append(cmin)
        all_max_ch5.append(m5)
        all_ch0_ped_mean.append(ped0)
        all_ch1_ped_mean.append(ped1)
        all_max_ch1.append(m1)
        all_ch1_min.append(c1min)
        all_max_ch4.append(m4)
        all_tmax_ch4.append(t4)
        all_tmax_ch0.append(tc0)
        all_tmax_ch1.append(tc1)
        all_ch2_n_fit_points.append(ch2_nfit)
        all_ch3_n_fit_points.append(ch3_nfit)
        all_ch2_tanh_p0.append(ch2_p0)
        all_ch3_tanh_p0.append(ch3_p0)
        all_ch3_ped_mean.append(ch3_ped)
        all_ch3_min.append(ch3_min)

    max_ch0 = np.concatenate(all_max_ch0)
    ch0_min = np.concatenate(all_ch0_min)
    max_ch5 = np.concatenate(all_max_ch5)
    ch0_ped_mean = np.concatenate(all_ch0_ped_mean)
    ch1_ped_mean = np.concatenate(all_ch1_ped_mean)
    max_ch1 = np.concatenate(all_max_ch1)
    ch1_min = np.concatenate(all_ch1_min)
    max_ch4 = np.concatenate(all_max_ch4)
    tmax_ch4 = np.concatenate(all_tmax_ch4)
    tmax_ch0 = np.concatenate(all_tmax_ch0)
    tmax_ch1 = np.concatenate(all_tmax_ch1)
    ch2_n_fit_points = np.concatenate(all_ch2_n_fit_points)
    ch3_n_fit_points = np.concatenate(all_ch3_n_fit_points)
    ch2_tanh_p0 = np.concatenate(all_ch2_tanh_p0)
    ch3_tanh_p0 = np.concatenate(all_ch3_tanh_p0)
    ch3_ped_mean = np.concatenate(all_ch3_ped_mean)
    ch3_min = np.concatenate(all_ch3_min)

    mask = _compute_basic_acv_mask(
        ch2_n_fit_points=ch2_n_fit_points,
        ch3_n_fit_points=ch3_n_fit_points,
        ch2_tanh_p0=ch2_tanh_p0,
        ch3_tanh_p0=ch3_tanh_p0,
        max_ch0=max_ch0,
        ch0_min=ch0_min,
        max_ch5=max_ch5,
        ch0_ped_mean=ch0_ped_mean,
        ch1_ped_mean=ch1_ped_mean,
        max_ch1=max_ch1,
        ch1_min=ch1_min,
        ch3ped_mean=ch3_ped_mean,
        min_ch3=ch3_min,
        max_ch4=max_ch4,
        tmax_ch4=tmax_ch4,
        tmax_ch0=tmax_ch0,
        tmax_ch1=tmax_ch1,
        log_prefix="[basic+acv]",
    )

    passed_max_ch0 = max_ch0[mask]
    passed_max_ch1 = max_ch1[mask]
    passed_energy = ENERGY_A * passed_max_ch0 + ENERGY_B
    return passed_max_ch0, passed_max_ch1, passed_energy

def _load_basic_6mask_pass_events() -> np.ndarray:
    """读取参数并返回仅使用 6 个 mask（m1~m6）后的能量数组。"""
    pairs = _list_paired_param_files()
    all_max_ch0: List[np.ndarray] = []
    all_ch0_min: List[np.ndarray] = []
    all_max_ch5: List[np.ndarray] = []
    all_ch0_ped_mean: List[np.ndarray] = []
    all_ch1_ped_mean: List[np.ndarray] = []
    all_max_ch1: List[np.ndarray] = []
    all_ch1_min: List[np.ndarray] = []
    all_max_ch4: List[np.ndarray] = []
    all_tmax_ch4: List[np.ndarray] = []
    all_tmax_ch0: List[np.ndarray] = []
    all_tmax_ch1: List[np.ndarray] = []

    for ch0_path, ch5_path in pairs:
        m0, cmin, m5, ped0, ped1, m1, c1min, m4, t4, tc0, tc1 = _load_basic_features_for_run(ch0_path, ch5_path)
        all_max_ch0.append(m0)
        all_ch0_min.append(cmin)
        all_max_ch5.append(m5)
        all_ch0_ped_mean.append(ped0)
        all_ch1_ped_mean.append(ped1)
        all_max_ch1.append(m1)
        all_ch1_min.append(c1min)
        all_max_ch4.append(m4)
        all_tmax_ch4.append(t4)
        all_tmax_ch0.append(tc0)
        all_tmax_ch1.append(tc1)

    max_ch0 = np.concatenate(all_max_ch0)
    ch0_min = np.concatenate(all_ch0_min)
    max_ch5 = np.concatenate(all_max_ch5)
    ch0_ped_mean = np.concatenate(all_ch0_ped_mean)
    ch1_ped_mean = np.concatenate(all_ch1_ped_mean)
    ch1_min = np.concatenate(all_ch1_min)
    max_ch1 = np.concatenate(all_max_ch1)
    max_ch4 = np.concatenate(all_max_ch4)
    tmax_ch4 = np.concatenate(all_tmax_ch4)

    mask6 = _compute_basic_6mask(
        max_ch0=max_ch0,
        max_ch1=max_ch1,
        ch0_min=ch0_min,
        max_ch5=max_ch5,
        ch0_ped_mean=ch0_ped_mean,
        ch1_ped_mean=ch1_ped_mean,
        ch1_min=ch1_min,
        max_ch4=max_ch4,
        tmax_ch4=tmax_ch4,
        log_prefix="[basic_6mask]",
    )
    return ENERGY_A * max_ch0[mask6] + ENERGY_B

def _load_basic_acv_bs_pass_events(
    rise_time_threshold: float = BS_RISE_TIME_THRESHOLD,) -> Tuple[np.ndarray, np.ndarray]:
    """
    在 basic+acv 全部 cut 基础上，增加 bscut：rise_time = ln(19)/p1 < rise_time_threshold。
    返回 (passed_max_ch0, passed_energy) 用于能谱绘制。
    """
    pairs = _list_paired_param_files_with_ch2_ch3()
    if not pairs:
        print("[basic+acv+bs] 无 CH2/CH3 参数文件，跳过 basic+acv+bs 能谱。")
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    print(f"[basic+acv+bs] 找到 {len(pairs)} 个含 CH2/CH3 的可配对参数文件。")
    all_max_ch0: List[np.ndarray] = []
    all_ch0_min: List[np.ndarray] = []
    all_max_ch5: List[np.ndarray] = []
    all_ch0_ped_mean: List[np.ndarray] = []
    all_ch1_ped_mean: List[np.ndarray] = []
    all_max_ch1: List[np.ndarray] = []
    all_ch1_min: List[np.ndarray] = []
    all_max_ch4: List[np.ndarray] = []
    all_tmax_ch4: List[np.ndarray] = []
    all_tmax_ch0: List[np.ndarray] = []
    all_tmax_ch1: List[np.ndarray] = []
    all_p1: List[np.ndarray] = []
    all_ch2_n_fit_points: List[np.ndarray] = []
    all_ch3_n_fit_points: List[np.ndarray] = []
    all_ch2_tanh_p0: List[np.ndarray] = []
    all_ch3_tanh_p0: List[np.ndarray] = []
    all_ch3_ped_mean: List[np.ndarray] = []
    all_ch3_min: List[np.ndarray] = []

    for ch0_path, ch5_path in pairs:
        m0, cmin, m5, ped0, ped1, m1, c1min, m4, t4, tc0, tc1 = _load_basic_features_for_run(ch0_path, ch5_path)
        with h5py.File(CH3_PARAM_DIR / ch0_path.name, "r") as f:
            p1 = np.asarray(f["tanh_p1"][...], dtype=np.float64)
        ch2_nfit, ch3_nfit, ch2_p0, ch3_p0 = _load_ch2_ch3_fit_quality_aligned(ch0_path, m0.shape[0])
        ch3_ped, ch3_min = _load_ch3_ped_min_aligned(ch0_path, m0.shape[0])
        n = min(m0.shape[0], p1.shape[0], tc0.shape[0], tc1.shape[0], ch2_nfit.shape[0], ch3_nfit.shape[0], ch2_p0.shape[0], ch3_p0.shape[0], ch3_ped.shape[0], ch3_min.shape[0])
        all_max_ch0.append(m0[:n])
        all_ch0_min.append(cmin[:n])
        all_max_ch5.append(m5[:n])
        all_ch0_ped_mean.append(ped0[:n])
        all_ch1_ped_mean.append(ped1[:n])
        all_max_ch1.append(m1[:n])
        all_ch1_min.append(c1min[:n])
        all_max_ch4.append(m4[:n])
        all_tmax_ch4.append(t4[:n])
        all_tmax_ch0.append(tc0[:n])
        all_tmax_ch1.append(tc1[:n])
        all_p1.append(p1[:n])
        all_ch2_n_fit_points.append(ch2_nfit[:n])
        all_ch3_n_fit_points.append(ch3_nfit[:n])
        all_ch2_tanh_p0.append(ch2_p0[:n])
        all_ch3_tanh_p0.append(ch3_p0[:n])
        all_ch3_ped_mean.append(ch3_ped[:n])
        all_ch3_min.append(ch3_min[:n])

    max_ch0 = np.concatenate(all_max_ch0)
    ch0_min = np.concatenate(all_ch0_min)
    max_ch5 = np.concatenate(all_max_ch5)
    ch0_ped_mean = np.concatenate(all_ch0_ped_mean)
    ch1_ped_mean = np.concatenate(all_ch1_ped_mean)
    max_ch1 = np.concatenate(all_max_ch1)
    ch1_min = np.concatenate(all_ch1_min)
    max_ch4 = np.concatenate(all_max_ch4)
    tmax_ch4 = np.concatenate(all_tmax_ch4)
    tmax_ch0 = np.concatenate(all_tmax_ch0)
    tmax_ch1 = np.concatenate(all_tmax_ch1)
    p1 = np.concatenate(all_p1)
    ch2_n_fit_points = np.concatenate(all_ch2_n_fit_points)
    ch3_n_fit_points = np.concatenate(all_ch3_n_fit_points)
    ch2_tanh_p0 = np.concatenate(all_ch2_tanh_p0)
    ch3_tanh_p0 = np.concatenate(all_ch3_tanh_p0)
    ch3_ped_mean = np.concatenate(all_ch3_ped_mean)
    ch3_min = np.concatenate(all_ch3_min)

    n_align = min(max_ch0.shape[0], p1.shape[0], tmax_ch0.shape[0], tmax_ch1.shape[0])
    max_ch0 = max_ch0[:n_align]
    ch0_min = ch0_min[:n_align]
    max_ch5 = max_ch5[:n_align]
    ch0_ped_mean = ch0_ped_mean[:n_align]
    ch1_ped_mean = ch1_ped_mean[:n_align]
    max_ch1 = max_ch1[:n_align]
    ch1_min = ch1_min[:n_align]
    max_ch4 = max_ch4[:n_align]
    tmax_ch4 = tmax_ch4[:n_align]
    tmax_ch0 = tmax_ch0[:n_align]
    tmax_ch1 = tmax_ch1[:n_align]
    p1 = p1[:n_align]
    ch2_n_fit_points = ch2_n_fit_points[:n_align]
    ch3_n_fit_points = ch3_n_fit_points[:n_align]
    ch2_tanh_p0 = ch2_tanh_p0[:n_align]
    ch3_tanh_p0 = ch3_tanh_p0[:n_align]
    ch3_ped_mean = ch3_ped_mean[:n_align]
    ch3_min = ch3_min[:n_align]

    basic_mask = _compute_basic_acv_mask(
        ch2_n_fit_points=ch2_n_fit_points,
        ch3_n_fit_points=ch3_n_fit_points,
        ch2_tanh_p0=ch2_tanh_p0,
        ch3_tanh_p0=ch3_tanh_p0,
        max_ch0=max_ch0,
        ch0_min=ch0_min,
        max_ch5=max_ch5,
        ch0_ped_mean=ch0_ped_mean,
        ch1_ped_mean=ch1_ped_mean,
        max_ch1=max_ch1,
        ch1_min=ch1_min,
        ch3ped_mean=ch3_ped_mean,
        min_ch3=ch3_min,
        max_ch4=max_ch4,
        tmax_ch4=tmax_ch4,
        tmax_ch0=tmax_ch0,
        tmax_ch1=tmax_ch1,
        log_prefix="[basic+acv+bs]",
    )
    rise_time = np.where(p1 > 1e-10, LN_19 / p1, np.nan)
    bs_mask = np.isfinite(rise_time) & (rise_time < rise_time_threshold)
    mask = basic_mask & bs_mask

    passed_max_ch0 = max_ch0[mask]
    passed_energy = ENERGY_A * passed_max_ch0 + ENERGY_B
    print(f"[basic+acv+bs] 通过 basic+acv + rise_time<{rise_time_threshold} 后: {mask.sum()} 事例")
    return passed_max_ch0, passed_energy

def _load_event_mapping(path: Path) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"事件映射 HDF5 不存在: {path}")

    with h5py.File(path, "r") as f:
        file_paths_raw = f["file_paths"][...]
        event_file_indices = f["event_file_indices"][...]
        event_event_indices = f["event_event_indices"][...]
        labels = f["event_cluster_labels"][...]

    file_paths: List[str] = []
    for p in file_paths_raw:
        if isinstance(p, bytes):
            file_paths.append(p.decode("utf-8"))
        else:
            file_paths.append(str(p))

    return file_paths, event_file_indices, event_event_indices, labels

def _compute_max_ch0_for_clusters(
    file_paths: Sequence[str],
    event_file_indices: np.ndarray,
    event_event_indices: np.ndarray,
    labels: np.ndarray,
    target_clusters: Sequence[int],) -> np.ndarray:
    """
    对给定 cluster（可多个）中的所有事件，从 CH0max 源文件读取 max(ch0)，
    并将对应文件在 RTCH0max 中的 RT 事件并入。多 cluster 时返回叠加后的事件集合。
    """
    target_set = set(int(c) for c in target_clusters)
    mask = np.isin(labels.astype(int), list(target_set))
    if not np.any(mask):
        print(f"cluster={{{_format_cluster_label(target_clusters)}}}: 在映射文件中没有事件。")
        return np.array([], dtype=np.float64)

    indices = np.nonzero(mask)[0]
    cluster_str = _format_cluster_label(target_clusters)
    print(f"cluster={{{cluster_str}}}: 共有事件数 = {indices.size}")

    file_to_events: Dict[int, List[int]] = defaultdict(list)
    for i in indices:
        fi = int(event_file_indices[i])
        ev = int(event_event_indices[i])
        file_to_events[fi].append(ev)

    ch0max_dir = PROJECT_ROOT / "data" / "hdf5" / "raw_pulse" / "CH0max"
    ch0_param_dir = PROJECT_ROOT / "data" / "hdf5" / "raw_pulse" / "CH0_parameters"
    rtch0max_dir = PROJECT_ROOT / "data" / "hdf5" / "raw_pulse" / "RTCH0max"

    max_values: List[float] = []
    for fi, ev_list in file_to_events.items():
        if fi < 0 or fi >= len(file_paths):
            continue
        src_path = Path(file_paths[fi])

        ch0max_path = ch0max_dir / src_path.name
        ch0_param_path = ch0_param_dir / src_path.name
        max_source_path: Path | None = None
        if ch0max_path.exists():
            max_source_path = ch0max_path
        elif ch0_param_path.exists():
            max_source_path = ch0_param_path

        if max_source_path is None:
            print(
                f"警告: 未找到 max(ch0) 源文件，跳过: {ch0max_path} 或 {ch0_param_path}"
            )
            continue

        rtch0max_path = rtch0max_dir / src_path.name
        ev_arr = np.asarray(ev_list, dtype=np.int64)

        with h5py.File(max_source_path, "r") as f_max:
            if "max_ch0" not in f_max:
                print(
                    f"警告: max(ch0) 源文件中缺少 'max_ch0' 数据集，跳过: {max_source_path}"
                )
                continue

            dset = f_max["max_ch0"]
            if dset.ndim != 1:
                print(f"警告: 'max_ch0' 维度不是 1, shape={dset.shape}，跳过: {max_source_path}")
                continue

            n_events_in_file = dset.shape[0]
            valid_mask = (ev_arr >= 0) & (ev_arr < n_events_in_file)
            ev_arr = ev_arr[valid_mask]
            if ev_arr.size == 0:
                continue

            # h5py 花式索引要求索引严格递增；先标准化 cluster 事件索引
            union_indices = np.unique(ev_arr)
            if rtch0max_path.exists():
                with h5py.File(rtch0max_path, "r") as f_rt:
                    if "rt_event_indices" in f_rt:
                        rt_indices = np.asarray(f_rt["rt_event_indices"][...], dtype=np.int64)
                        rt_valid = (rt_indices >= 0) & (rt_indices < n_events_in_file)
                        rt_indices_valid = rt_indices[rt_valid]
                        if rt_indices_valid.size > 0:
                            union_indices = np.unique(np.concatenate([union_indices, rt_indices_valid]))

            max_vals_file = np.asarray(dset[union_indices], dtype=np.float64)
            max_values.extend(max_vals_file.tolist())

    max_values_arr = np.asarray(max_values, dtype=np.float64)
    print(f"cluster={{{cluster_str}}}: 成功计算 max(ch0) 的事件数 = {max_values_arr.size}")
    return max_values_arr

def _format_cluster_label(clusters: Sequence[int]) -> str:
    """将 cluster 列表格式化为标签字符串，如 '4' 或 '4,5,6'。"""
    return ",".join(map(str, sorted(clusters)))

def _compute_rates_from_energy(energy_values: np.ndarray, bin_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if energy_values.size == 0:
        bin_widths = np.diff(bin_edges)
        bin_centers = bin_edges[:-1] + bin_widths / 2.0
        rates = np.zeros_like(bin_centers, dtype=np.float64)
        return bin_centers, bin_widths, rates

    counts, _ = np.histogram(energy_values, bins=bin_edges)
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + bin_widths / 2.0
    denom = EXPOSURE_KG * bin_widths * EXPOSURE_DAYS
    denom[denom == 0] = np.inf
    rates = counts / denom
    return bin_centers, bin_widths, rates

def _default_hdf5_path_relative_to_project_root() -> Path:
    return PROJECT_ROOT / "data" / "hdf5" / "ge_30param_umap_hdbscan_eventmap.h5"

def _default_efficiency_curve_path() -> Path:
    """默认效率曲线路径：act/efficiency_curve.h5（相对于本脚本所在目录）。"""
    return Path(__file__).resolve().parent / "act" / "efficiency_curve.h5"

def _load_efficiency_curve(
    path: Path,) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
    """
    从 efficiency_curve.h5 加载效率曲线。
    返回 (energy_keV, efficiency, bin_edges_saved | None)，若文件不存在或格式不符则返回 None。
    bin_edges_saved 用于检查是否与当前 bin 一一对应。
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with h5py.File(path, "r") as f:
            if "energy_keV" not in f or "efficiency" not in f:
                return None
            energy_kev = np.asarray(f["energy_keV"][...], dtype=np.float64)
            efficiency = np.asarray(f["efficiency"][...], dtype=np.float64)
            bin_edges_saved = None
            if "bin_edges" in f:
                bin_edges_saved = np.asarray(f["bin_edges"][...], dtype=np.float64)
            if energy_kev.size == 0 or efficiency.size == 0:
                return None
            return energy_kev, efficiency, bin_edges_saved
    except Exception as exc:
        print(f"[效率修正] 加载 {path} 失败: {exc}")
        return None

def _compute_efficiency_corrected_rates(
    bin_centers: np.ndarray,
    bin_edges: np.ndarray,
    rates_cluster: np.ndarray,
    eff_energy: np.ndarray,
    efficiency: np.ndarray,
    bin_edges_saved: np.ndarray | None,) -> np.ndarray:
    """
    使用效率曲线对 cluster 计数率进行修正：真实计数率 = 观测计数率 / 效率。
    若 bin_edges 与保存的一致，则一一对应直接除法；否则在 bin_centers 处插值。
    """
    # 一一对应：bin 完全一致时直接除法，更准确
    if bin_edges_saved is not None and bin_edges_saved.size == bin_edges.size:
        if np.allclose(bin_edges_saved, bin_edges):
            eff_use = efficiency
        else:
            eff_use = np.interp(bin_centers, eff_energy, efficiency)
    elif eff_energy.size == bin_centers.size and np.allclose(eff_energy, bin_centers):
        eff_use = efficiency
    else:
        eff_use = np.interp(bin_centers, eff_energy, efficiency)
    min_eff = 1e-4
    rates_corrected = np.where(
        eff_use > min_eff,
        rates_cluster / eff_use,
        np.nan,
    )
    return rates_corrected

def plot_efficiency_vs_energy(
    bin_centers: np.ndarray,
    bin_widths: np.ndarray,
    rates_cluster: np.ndarray,
    rates_basic: np.ndarray,
    rates_basic_6mask: np.ndarray | None,
    cluster: int | str,
    e_min_kev: float = 0.0,
    e_max_kev: float = 0.6,) -> None:
    """
    绘制效率图：每个 bin 的 cluster 计数率 / basic+acv 计数率。
    另起一个 figure，横轴能量 (keV)，纵轴效率；仅显示 [e_min_kev, e_max_kev] 范围内的效率。
    """
    mask_range = (bin_centers >= e_min_kev) & (bin_centers <= e_max_kev)
    mask_valid = mask_range & (rates_basic > 0)
    if not np.any(mask_valid):
        print(f"[效率图] [{e_min_kev}, {e_max_kev}] keV 范围内无有效 bin（basic 计数率需 > 0），跳过。")
        return

    x_eff = bin_centers[mask_valid]
    rc = rates_cluster[mask_valid]
    rb = rates_basic[mask_valid]
    bw = bin_widths[mask_valid]
    eff = rc / rb

    # Poisson 误差传播: σ(eff)/eff = sqrt(1/counts_c + 1/counts_b)
    denom = EXPOSURE_KG * bw * EXPOSURE_DAYS
    counts_c = np.maximum(rc * denom, 0.5)  # 避免 1/0
    counts_b = np.maximum(rb * denom, 0.5)
    eff_err = eff * np.sqrt(1.0 / counts_c + 1.0 / counts_b)

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.errorbar(x_eff, eff, yerr=eff_err, fmt="o", color="C0", markersize=4, capsize=2, capthick=1, label=f"cluster={cluster} / basic+acv")
    if rates_basic_6mask is not None and rates_basic_6mask.size == rates_basic.size:
        mask_valid6 = mask_range & (rates_basic_6mask > 0)
        if np.any(mask_valid6):
            x6 = bin_centers[mask_valid6]
            rc6 = rates_cluster[mask_valid6]
            rb6 = rates_basic_6mask[mask_valid6]
            bw6 = bin_widths[mask_valid6]
            eff6 = rc6 / rb6
            denom6 = EXPOSURE_KG * bw6 * EXPOSURE_DAYS
            counts_c6 = np.maximum(rc6 * denom6, 0.5)
            counts_b6 = np.maximum(rb6 * denom6, 0.5)
            eff6_err = eff6 * np.sqrt(1.0 / counts_c6 + 1.0 / counts_b6)
            ax.errorbar(
                x6,
                eff6,
                yerr=eff6_err,
                fmt="s",
                color="C4",
                markersize=3,
                capsize=2,
                capthick=1,
                label=f"cluster={cluster} / basic_6mask",
            )
    ax.set_xlim(e_min_kev, e_max_kev)
    ax.set_xlabel("Energy (keV)", fontsize=12)
    ax.set_ylabel("Efficiency (cluster rate / basic+acv rate)", fontsize=12)
    ax.set_title(f"Efficiency in [{e_min_kev}, {e_max_kev}] keV", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()

def _plot_cluster_overlay_with_fit(
    bin_edges: np.ndarray,
    bin_centers: np.ndarray,
    bin_widths: np.ndarray,
    rates_cluster: np.ndarray,
    rates_basic: np.ndarray,
    rates_basic_6mask: np.ndarray | None,
    clusters: Sequence[int],
    efficiency_curve_path: Path | None = None,
    rates_basic_acv_bs: np.ndarray | None = None,) -> None:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # 使用 step 阶梯线绘制，代替填充 bar
    _step_x = np.append(bin_edges[:-1], bin_edges[-1])
    _step_y_cluster = np.append(rates_cluster, rates_cluster[-1])
    ax.step(
        _step_x,
        _step_y_cluster,
        where="post",
        color="C0",
        alpha=0.8,
        linewidth=0.6,
        label=f"UMAP+HDBSCAN cluster={_format_cluster_label(clusters)}",
    )
    _step_y_basic = np.append(rates_basic, rates_basic[-1])
    ax.step(
        _step_x,
        _step_y_basic,
        where="post",
        color="C1",
        alpha=0.8,
        linewidth=0.6,
        label="basic+noise cuts",
    )
    if rates_basic_6mask is not None and rates_basic_6mask.size == rates_basic.size:
        _step_y_basic6 = np.append(rates_basic_6mask, rates_basic_6mask[-1])
        ax.step(
            _step_x,
            _step_y_basic6,
            where="post",
            color="C4",
            alpha=0.8,
            linewidth=0.6,
            label="basic cuts",
        )
    # cluster 效率修正曲线：使用 act/efficiency_curve.h5（或命令行指定路径）
    if efficiency_curve_path is not None:
        loaded = _load_efficiency_curve(efficiency_curve_path)
        if loaded is None:
            print(f"[效率修正] 未能加载效率曲线: {efficiency_curve_path}")
        else:
            eff_energy, efficiency, bin_edges_saved = loaded
            rates_cluster_corr = _compute_efficiency_corrected_rates(
                bin_centers=bin_centers,
                bin_edges=bin_edges,
                rates_cluster=rates_cluster,
                eff_energy=eff_energy,
                efficiency=efficiency,
                bin_edges_saved=bin_edges_saved,
            )
            if np.isfinite(rates_cluster_corr).any():
                _step_y_corr = np.append(rates_cluster_corr, rates_cluster_corr[-1])
                ax.step(
                    _step_x,
                    _step_y_corr,
                    where="post",
                    color="C2",
                    alpha=0.9,
                    linewidth=0.7,
                    label=f"UMAP+HDBSCAN cluster={_format_cluster_label(clusters)} (eff-corrected)",
                )
            else:
                print("[效率修正] 修正后曲线全部为无效值，跳过绘制。")
    ax.set_yscale("log")
    ax.set_xlabel("Energy (keV)", fontsize=12)
    ax.set_ylabel(r"Rate [counts / (keV·kg·day)]", fontsize=12)
    ax.set_title("Energy spectrum overlay", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, 2)
    ax.set_ylim(1e-1, 1e7)
    fig.tight_layout()

    # ----------------------------
    # 10–11 keV 峰拟合：仅对 cluster 谱
    # ----------------------------
    e_min, e_max = 10.0, 11.0
    mask_roi = (bin_centers >= e_min) & (bin_centers <= e_max) & (rates_cluster > 0)
    if np.count_nonzero(mask_roi) >= 5:
        x_roi = bin_centers[mask_roi]
        y_roi = rates_cluster[mask_roi]

        peak_idx = np.argmax(y_roi)
        mu0 = x_roi[peak_idx]
        amp0 = y_roi[peak_idx] - np.min(y_roi)
        sigma0 = 0.05  # keV
        c0 = np.min(y_roi)
        d0 = 0.0

        def gauss_linear(x, A, mu, sigma, c, d):
            return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + c + d * x

        try:
            popt, _pcov = curve_fit(
                gauss_linear,
                x_roi,
                y_roi,
                p0=[amp0, mu0, sigma0, c0, d0],
                maxfev=10000,
            )
            A_fit, mu_fit, sigma_fit, c_fit, d_fit = popt
            fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * abs(sigma_fit)
            print(f"10–11 keV 峰拟合结果: mu = {mu_fit:.4f} keV, FWHM = {fwhm:.4f} keV")

            fig_fit, ax_fit = plt.subplots(1, 1, figsize=(8, 6))
            ax_fit.scatter(x_roi, y_roi, color="C0", label=f"Data (cluster={_format_cluster_label(clusters)})", zorder=3)

            x_fit = np.linspace(x_roi.min(), x_roi.max(), 400)
            y_fit = gauss_linear(x_fit, *popt)
            ax_fit.plot(x_fit, y_fit, color="C1", label="Gaussian + linear fit")

            ax_fit.set_xlabel("Energy (keV)", fontsize=12)
            ax_fit.set_ylabel(r"Rate [counts / (keV·kg·day)]", fontsize=12)
            ax_fit.set_title(
                f"Peak fit in [{e_min}, {e_max}] keV\n"
                f"mu = {mu_fit:.4f} keV, FWHM = {fwhm:.4f} keV",
                fontsize=13,
            )
            ax_fit.grid(True, alpha=0.3)
            ax_fit.legend()
            fig_fit.tight_layout()
        except Exception as exc:
            print(f"10–11 keV 拟合失败: {exc}")
    else:
        print("10–11 keV 区间内有效点太少，跳过拟合。")

    plt.show()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把 basic+acv.py 的能谱叠加到 spectrum.py 的能谱图上。"
    )
    parser.add_argument(
        "hdf5_path",
        nargs="?",
        help="事件映射 HDF5 路径；若不指定，则使用相对于项目根目录的默认文件。",
    )
    parser.add_argument(
        "--cluster",
        type=int,
        nargs="+",
        default=[0],
        metavar="C",
        help="要分析的 cluster label，可指定多个（如 --cluster 4 5 6），展示叠加后的能谱总形状。",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="能谱与效率曲线共用 bin 数（默认 1000，约 2 eV/bin，与 acvcombine_spectrum 一一对应）。",
    )
    parser.add_argument(
        "--efficiency-curve",
        type=str,
        default=None,
        help="效率曲线 HDF5 路径；若不指定，使用 act/efficiency_curve.h5。",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    # cluster spectrum (from spectrum.py)
    if args.hdf5_path:
        raw_path = Path(args.hdf5_path)
        if not raw_path.is_absolute():
            # 解释为“相对于脚本所在目录”
            hdf5_path = (Path(__file__).resolve().parent / raw_path).resolve()
        else:
            hdf5_path = raw_path
    else:
        hdf5_path = _default_hdf5_path_relative_to_project_root()

    print(f"[combine_spectrum] 使用事件映射 HDF5: {hdf5_path}")
    file_paths, event_file_indices, event_event_indices, labels = _load_event_mapping(hdf5_path)
    max_values_cluster = _compute_max_ch0_for_clusters(
        file_paths=file_paths,
        event_file_indices=event_file_indices,
        event_event_indices=event_event_indices,
        labels=labels,
        target_clusters=args.cluster,
    )
    energy_cluster = ENERGY_A * max_values_cluster + ENERGY_B

    # basic+acv spectrum (from basic+acv.py)
    passed_max_ch0, passed_max_ch1, energy_basic = _load_basic_acv_pass_events()
    energy_basic_6mask = np.array([], dtype=np.float64)

    # 固定 bin 范围 [0, 2] keV，与 acvcombine_spectrum 一一对应，便于效率修正
    energy_all = np.concatenate([energy_cluster, energy_basic]) if energy_basic.size > 0 else energy_cluster
    if energy_all.size == 0:
        raise RuntimeError("energy_all 为空，无法绘图（cluster 与 basic+acv 都没有事件通过/可用）。")

    bin_edges = np.linspace(E_MIN_BIN, E_MAX_BIN, args.bins + 1)
    ec_in = energy_cluster[(energy_cluster >= E_MIN_BIN) & (energy_cluster <= E_MAX_BIN)]
    eb_in = energy_basic[(energy_basic >= E_MIN_BIN) & (energy_basic <= E_MAX_BIN)]
    eb6_in = energy_basic_6mask[(energy_basic_6mask >= E_MIN_BIN) & (energy_basic_6mask <= E_MAX_BIN)]

    bin_centers, bin_widths, rates_cluster = _compute_rates_from_energy(ec_in, bin_edges)
    _, _, rates_basic = _compute_rates_from_energy(eb_in, bin_edges)
    if eb6_in.size > 0:
        _, _, rates_basic_6mask = _compute_rates_from_energy(eb6_in, bin_edges)
    else:
        rates_basic_6mask = None
    eff_path: Path | None = None
    if args.efficiency_curve:
        eff_path = Path(args.efficiency_curve)
        if not eff_path.is_absolute():
            eff_path = (Path(__file__).resolve().parent / eff_path).resolve()
    else:
        eff_path = _default_efficiency_curve_path()

    _plot_cluster_overlay_with_fit(
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        bin_widths=bin_widths,
        rates_cluster=rates_cluster,
        rates_basic=rates_basic,
        rates_basic_6mask=rates_basic_6mask,
        clusters=args.cluster,
        efficiency_curve_path=eff_path,
        rates_basic_acv_bs=None,
    )

    # 效率图使用与能谱叠加相同的 bin（一一对应）
    plot_efficiency_vs_energy(
        bin_centers=bin_centers,
        bin_widths=bin_widths,
        rates_cluster=rates_cluster,
        rates_basic=rates_basic,
        rates_basic_6mask=rates_basic_6mask,
        cluster=_format_cluster_label(args.cluster),
        e_min_kev=E_MIN_BIN,
        e_max_kev=E_MAX_BIN,
    )

    # basic+acv scatter plot (same as basic+acv.py)
    if passed_max_ch0.size > 0:
        plt.figure(figsize=(8, 6))
        plt.scatter(passed_max_ch0, passed_max_ch1, s=2, alpha=0.5, edgecolors="none")
        plt.xlabel("CH0 maximum amplitude (FADC)")
        plt.ylabel("CH1 maximum amplitude (FADC)")
        plt.title(f"CH0max vs CH1max (basic+acv cuts, N={passed_max_ch0.size})")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

