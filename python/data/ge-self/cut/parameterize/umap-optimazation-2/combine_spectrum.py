#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
combine_spectrum.py

功能：
1) 读取 30 参数 UMAP+HDBSCAN 的事件映射 HDF5，选择指定 cluster 的所有事件；
   同时将对应源文件中的 RT 事件也并入（RT 事件计数也算入谱中）。
2) 使用 basic+act.py 的切割逻辑（CH0/CH5/CH1/CH4 参数文件 + 一系列 cut），得到通过所有 cuts
   的事件集合，并计算其能谱。
3) 将 (1) 与 (2) 两条能谱画在同一个图上，实现“能谱叠加”。
4) cluster 那条能谱仍然执行基本的 10–11 keV 高斯 + 线性本底拟合；
   basic+act 那条在图之外额外绘制 CH0max vs CH1max 散点图（与 basic+act.py 保持一致）。

默认行为尽量与 spectrum.py / basic+act.py 一致：
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
EXPOSURE_DAYS = 20.0


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
CH4_PARAM_DIR = DATA_ROOT / "CH4_parameters"
CH5_PARAM_DIR = DATA_ROOT / "CH5_parameters"


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


def _load_basic_features_for_run(
    ch0_param_path: Path,
    ch5_param_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从单个 run 的 CH0/CH5/CH1/CH4 参数文件中读取基准特征（basic+act.py 保持一致）。
    """
    with h5py.File(ch0_param_path, "r") as f_ch0:
        if "max_ch0" not in f_ch0 or "ch0_min" not in f_ch0 or "ch0ped_mean" not in f_ch0:
            raise KeyError(
                f"{ch0_param_path.name} 中缺少 max_ch0 / ch0_min / ch0ped_mean 数据集，"
                "请确认该文件由当前版本的 preprocessor.py 生成。"
            )
        max_ch0 = np.asarray(f_ch0["max_ch0"][...], dtype=np.float64)
        ch0_min = np.asarray(f_ch0["ch0_min"][...], dtype=np.float64)
        ch0_ped_mean = np.asarray(f_ch0["ch0ped_mean"][...], dtype=np.float64)

    with h5py.File(ch5_param_path, "r") as f_ch5:
        if "max_ch5" not in f_ch5:
            raise KeyError(
                f"{ch5_param_path.name} 中缺少 max_ch5 数据集，"
                "请确认该文件由当前版本的 preprocessor.py 生成。"
            )
        max_ch5 = np.asarray(f_ch5["max_ch5"][...], dtype=np.float64)

    ch1_param_path = CH1_PARAM_DIR / ch0_param_path.name
    with h5py.File(ch1_param_path, "r") as f_ch1:
        if "ch1ped_mean" not in f_ch1 or "max_ch1" not in f_ch1 or "ch1_min" not in f_ch1:
            raise KeyError(
                f"{ch1_param_path.name} 中缺少 ch1ped_mean / max_ch1 / ch1_min 之一，"
                "请确认该文件由当前版本的 preprocessor.py 生成。"
            )
        ch1_ped_mean = np.asarray(f_ch1["ch1ped_mean"][...], dtype=np.float64)
        max_ch1 = np.asarray(f_ch1["max_ch1"][...], dtype=np.float64)
        ch1_min = np.asarray(f_ch1["ch1_min"][...], dtype=np.float64)

    ch4_param_path = CH4_PARAM_DIR / ch0_param_path.name
    with h5py.File(ch4_param_path, "r") as f_ch4:
        if "max_ch4" not in f_ch4 or "tmax_ch4" not in f_ch4:
            raise KeyError(
                f"{ch4_param_path.name} 中缺少 max_ch4 或 tmax_ch4 数据集，"
                "请确认该文件由当前版本的 preprocessor.py 生成。"
            )
        max_ch4 = np.asarray(f_ch4["max_ch4"][...], dtype=np.float64)
        tmax_ch4 = np.asarray(f_ch4["tmax_ch4"][...], dtype=np.float64)

    n0, nmin, n5, nped0, nped1, n1, nc1min, n4, nt4 = (
        max_ch0.shape[0],
        ch0_min.shape[0],
        max_ch5.shape[0],
        ch0_ped_mean.shape[0],
        ch1_ped_mean.shape[0],
        max_ch1.shape[0],
        ch1_min.shape[0],
        max_ch4.shape[0],
        tmax_ch4.shape[0],
    )
    n = min(n0, nmin, n5, nped0, nped1, n1, nc1min, n4, nt4)

    if not (n0 == nmin == n5 == nped0 == nped1 == n1 == nc1min == n4 == nt4):
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

    return max_ch0, ch0_min, max_ch5, ch0_ped_mean, ch1_ped_mean, max_ch1, ch1_min, max_ch4, tmax_ch4


def cut_ch0_min_positive(ch0_min: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """条件：ch0_min > threshold（排除抑制信号）。"""

    return ch0_min > threshold


def cut_ch0_max_saturation(max_ch0: np.ndarray, max_val: float = 16382.0) -> np.ndarray:
    """条件：max_ch0 <= max_val（排除饱和事例）。"""

    return max_ch0 <= max_val


def cut_ch5_self_trigger(max_ch5: np.ndarray, rt_threshold: float = 6000.0) -> np.ndarray:
    """条件：max_ch5 <= rt_threshold（排除随机触发）。"""

    return max_ch5 <= rt_threshold


def cut_pedestal_3sigma(
    ch0_ped_mean: np.ndarray,
    ch1_ped_mean: np.ndarray,
    max_ch5: np.ndarray,
    rt_threshold: float = 6000.0,
    n_sigma: float = 3.0,
    min_rt_events: int = 10,
) -> np.ndarray:
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

def cut_act(
    max_ch4: np.ndarray,
    tmax_ch4: np.ndarray,
    trigger_threshold: float = 7060.0,
    t_ge_us: float = 40.0,
    sampling_interval_ns: float = 4.0,
    dt_min_us: float = 1.0,
    dt_max_us: float = 16.0,
) -> np.ndarray:
    """
    ACT cut（与 basic+act.py 保持一致）：
    - 对 NaI 过阈事件 (max_ch4 >= trigger_threshold)，选取 Δt 非 [dt_min_us, dt_max_us] μs 的事例；
    - 对 NaI 未过阈事件，一律保留。
    """

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
    dt_max_us: float = 16.0,
) -> np.ndarray:
    """
    mincut：在 ACT 基础上，用 ACT 事例拟合 CH0min/CH1min 分布，
    保留 CH0min、CH1min 均在中心值 ± n_sigma*σ 内的事件。
    """

    n = ch0_min.shape[0]
    mask = np.ones(n, dtype=bool)
    act_mask = cut_act(max_ch4, tmax_ch4, trigger_threshold, t_ge_us, sampling_interval_ns, dt_min_us, dt_max_us)

    fit_mask = act_mask
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
    min_fit_events: int = 10,
) -> np.ndarray:
    """
    pncut（与 basic+act.py 逻辑一致）：
    - 先在 base_mask 内，用 CH0max 落在指定区间的点拟合一条直线；
    - 计算所有事件相对直线的残差，并输出 |r| <= n_sigma * σ 的事件。

    注意：basic+act.py 当前并未使用 pn_mask 来更新最终 mask，这里同样只计算不叠加。
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


def _load_basic_act_pass_events() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取 basic+act.py 的全部参数文件，对应 cut 后返回：
    - passed_max_ch0: (N,) 通过所有 cuts 的 CH0max
    - passed_max_ch1: (N,) 通过所有 cuts 的 CH1max
    - passed_energy:  (N,) 对应能量（keV）
    """

    pairs = _list_paired_param_files()
    print(f"[basic+act] 找到 {len(pairs)} 个可配对的参数文件。")

    all_max_ch0: List[np.ndarray] = []
    all_ch0_min: List[np.ndarray] = []
    all_max_ch5: List[np.ndarray] = []
    all_ch0_ped_mean: List[np.ndarray] = []
    all_ch1_ped_mean: List[np.ndarray] = []
    all_max_ch1: List[np.ndarray] = []
    all_ch1_min: List[np.ndarray] = []
    all_max_ch4: List[np.ndarray] = []
    all_tmax_ch4: List[np.ndarray] = []

    for ch0_path, ch5_path in pairs:
        m0, cmin, m5, ped0, ped1, m1, c1min, m4, t4 = _load_basic_features_for_run(ch0_path, ch5_path)
        all_max_ch0.append(m0)
        all_ch0_min.append(cmin)
        all_max_ch5.append(m5)
        all_ch0_ped_mean.append(ped0)
        all_ch1_ped_mean.append(ped1)
        all_max_ch1.append(m1)
        all_ch1_min.append(c1min)
        all_max_ch4.append(m4)
        all_tmax_ch4.append(t4)

    max_ch0 = np.concatenate(all_max_ch0)
    ch0_min = np.concatenate(all_ch0_min)
    max_ch5 = np.concatenate(all_max_ch5)
    ch0_ped_mean = np.concatenate(all_ch0_ped_mean)
    ch1_ped_mean = np.concatenate(all_ch1_ped_mean)
    max_ch1 = np.concatenate(all_max_ch1)
    ch1_min = np.concatenate(all_ch1_min)
    max_ch4 = np.concatenate(all_max_ch4)
    tmax_ch4 = np.concatenate(all_tmax_ch4)

    n_raw = max_ch0.shape[0]
    print(f"[basic+act] 原始事件数: {n_raw}")

    m1 = cut_ch0_min_positive(ch0_min)
    n1 = int(m1.sum())
    print(f"[basic+act] cut_ch0_min_positive 单独使用后: {n1} / {n_raw}")

    m2 = cut_ch0_max_saturation(max_ch0)
    n2 = int(m2.sum())
    print(f"[basic+act] cut_ch0_max_saturation 单独使用后: {n2} / {n_raw}")

    m3 = cut_ch5_self_trigger(max_ch5)
    n3 = int(m3.sum())
    print(f"[basic+act] cut_ch5_self_trigger 单独使用后: {n3} / {n_raw}")

    m4 = cut_pedestal_3sigma(ch0_ped_mean, ch1_ped_mean, max_ch5)
    n4 = int(m4.sum())
    print(f"[basic+act] cut_pedestal_3sigma 单独使用后: {n4} / {n_raw}")

    m5 = cut_act(max_ch4, tmax_ch4)
    n5 = int(m5.sum())
    print(f"[basic+act] cut_act 单独使用后: {n5} / {n_raw}")

    m6 = cut_mincut(ch0_min, ch1_min, max_ch5, max_ch4, tmax_ch4)
    n6 = int(m6.sum())
    print(f"[basic+act] cut_mincut 单独使用后: {n6} / {n_raw}")

    mask = m1 & m2 & m3 & m4 & m5 & m6
    
    pn_mask = cut_pncut(mask, max_ch0, max_ch1)
    n_final = int(mask.sum())
    n_pn = int(pn_mask.sum())
    print(f"[basic+act] 依次使用 cuts 后最终剩余(mask): {n_final} / {n_raw}")
    print(f"[basic+act] pn_mask 结果（注意：当前与 basic+act.py 一致，未叠加到最终 mask）: {n_pn} / {n_raw}")

    passed_max_ch0 = max_ch0[mask]
    passed_max_ch1 = max_ch1[mask]
    passed_energy = ENERGY_A * passed_max_ch0 + ENERGY_B
    return passed_max_ch0, passed_max_ch1, passed_energy


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


def _compute_max_ch0_for_cluster(
    file_paths: Sequence[str],
    event_file_indices: np.ndarray,
    event_event_indices: np.ndarray,
    labels: np.ndarray,
    target_cluster: int = 0,
) -> np.ndarray:
    """
    对给定 cluster 中的所有事件，从 CH0max 源文件读取 max(ch0)，并将对应文件在 RTCH0max 中的 RT 事件并入。
    """

    mask = labels.astype(int) == int(target_cluster)
    if not np.any(mask):
        print(f"cluster={target_cluster}: 在映射文件中没有事件。")
        return np.array([], dtype=np.float64)

    indices = np.nonzero(mask)[0]
    print(f"cluster={target_cluster}: 共有事件数 = {indices.size}")

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

            union_indices = ev_arr
            if rtch0max_path.exists():
                with h5py.File(rtch0max_path, "r") as f_rt:
                    if "rt_event_indices" in f_rt:
                        rt_indices = np.asarray(f_rt["rt_event_indices"][...], dtype=np.int64)
                        rt_valid = (rt_indices >= 0) & (rt_indices < n_events_in_file)
                        rt_indices_valid = rt_indices[rt_valid]
                        if rt_indices_valid.size > 0:
                            union_indices = np.unique(np.concatenate([ev_arr, rt_indices_valid]))

            max_vals_file = np.asarray(dset[union_indices], dtype=np.float64)
            max_values.extend(max_vals_file.tolist())

    max_values_arr = np.asarray(max_values, dtype=np.float64)
    print(f"cluster={target_cluster}: 成功计算 max(ch0) 的事件数 = {max_values_arr.size}")
    return max_values_arr


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


def _plot_cluster_overlay_with_fit(
    bin_edges: np.ndarray,
    bin_centers: np.ndarray,
    bin_widths: np.ndarray,
    rates_cluster: np.ndarray,
    rates_basic: np.ndarray,
    cluster: int,
) -> None:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.bar(
        bin_centers,
        rates_cluster,
        width=bin_widths,
        color="C0",
        alpha=0.6,
        align="center",
        label=f"UMAP+HDBSCAN cluster={cluster}",
    )
    ax.bar(
        bin_centers,
        rates_basic,
        width=bin_widths,
        color="C1",
        alpha=0.6,
        align="center",
        label="basic+act cuts (all cuts)",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Energy (keV)", fontsize=12)
    ax.set_ylabel(r"Rate [counts / (keV·kg·day)]", fontsize=12)
    # 尽量保持与 spectrum.py 的标题一致；仅通过 legend 区分叠加数据
    ax.set_title(f"Energy spectrum for cluster={cluster}", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()
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
            ax_fit.scatter(x_roi, y_roi, color="C0", label=f"Data (cluster={cluster})", zorder=3)

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
        description="把 basic+act.py 的能谱叠加到 spectrum.py 的能谱图上。"
    )
    parser.add_argument(
        "hdf5_path",
        nargs="?",
        help="事件映射 HDF5 路径；若不指定，则使用相对于项目根目录的默认文件。",
    )
    parser.add_argument(
        "--cluster",
        type=int,
        default=5,
        help="要分析的 cluster label（默认 5，与 spectrum.py 保持一致）。",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=500,
        help="能谱直方图 bin 数（默认 500）。",
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
    max_values_cluster = _compute_max_ch0_for_cluster(
        file_paths=file_paths,
        event_file_indices=event_file_indices,
        event_event_indices=event_event_indices,
        labels=labels,
        target_cluster=args.cluster,
    )
    energy_cluster = ENERGY_A * max_values_cluster + ENERGY_B

    # basic+act spectrum (from basic+act.py)
    passed_max_ch0, passed_max_ch1, energy_basic = _load_basic_act_pass_events()

    # Use common bin edges so the overlay is aligned.
    # 为了让 cluster 那条谱与 spectrum.py 完全一致：bin_edges 只由 cluster 决定；
    # basic+act 复用同一组 bin_edges 做计数，从而“叠加到 spectrum.py 的能谱图坐标轴上”。
    if energy_cluster.size > 0:
        bin_edges = np.histogram_bin_edges(energy_cluster, bins=args.bins)
    elif energy_basic.size > 0:
        bin_edges = np.histogram_bin_edges(energy_basic, bins=args.bins)
    else:
        raise RuntimeError("cluster 与 basic+act 都没有可用事件，无法绘图。")

    bin_centers, bin_widths, rates_cluster = _compute_rates_from_energy(energy_cluster, bin_edges)
    _, _, rates_basic = _compute_rates_from_energy(energy_basic, bin_edges)

    _plot_cluster_overlay_with_fit(
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        bin_widths=bin_widths,
        rates_cluster=rates_cluster,
        rates_basic=rates_basic,
        cluster=args.cluster,
    )

    # basic+act scatter plot (same as basic+act.py)
    if passed_max_ch0.size > 0:
        plt.figure(figsize=(8, 6))
        plt.scatter(passed_max_ch0, passed_max_ch1, s=2, alpha=0.5, edgecolors="none")
        plt.xlabel("CH0 maximum amplitude (FADC)")
        plt.ylabel("CH1 maximum amplitude (FADC)")
        plt.title(f"CH0max vs CH1max (basic+act cuts, N={passed_max_ch0.size})")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

