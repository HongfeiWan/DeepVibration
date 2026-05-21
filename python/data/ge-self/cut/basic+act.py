#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基础物理事例筛选 + 统计：

使用 CH0/CH5 的三个基准特征直接从参数文件中排除掉 inhibit 信号和 RT 信号，
对应于 30parameter&HDBSCAN.py 中的筛选条件：

    ch0_ch0_min > 0
    ch0_max_ch0 <= 16382
    ch5_max_ch5 <= 6000

其中三个特征分别来源于：
    - data/hdf5/raw_pulse/CH0_parameters  中的 dataset:
        max_ch0, ch0_min
    - data/hdf5/raw_pulse/CH5_parameters  中的 dataset:
        max_ch5
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def _discover_project_root() -> Path:
    """
    推断 DeepVibration 项目根目录：
    当前文件位于:
        .../python/data/ge-self/cut/basic+acv.py
    向上到 python，再上一层即项目根目录。
    """
    here = Path(__file__).resolve()
    # parents 索引：
    # 0: cut
    # 1: ge-self
    # 2: data
    # 3: python
    # 4: DeepVibration
    python_dir = here.parents[3]          # .../python
    return python_dir.parent              # .../DeepVibration

PROJECT_ROOT = _discover_project_root()
DATA_ROOT = PROJECT_ROOT / "data" / "hdf5" / "raw_pulse"
CH0_PARAM_DIR = DATA_ROOT / "CH0_parameters"
CH1_PARAM_DIR = DATA_ROOT / "CH1_parameters"
CH4_PARAM_DIR = DATA_ROOT / "CH4_parameters"
CH5_PARAM_DIR = DATA_ROOT / "CH5_parameters"

def _list_paired_param_files() -> List[Tuple[Path, Path]]:
    """
    基于 CH0_parameters 目录的文件名，寻找同时存在于 CH1_parameters 与 CH5_parameters 中的参数文件对。
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
        name for name in os.listdir(CH0_PARAM_DIR)
        if name.lower().endswith((".h5", ".hdf5"))
    )
    if not ch0_files:
        raise FileNotFoundError(f"CH0_parameters 目录下未找到任何 h5 文件: {CH0_PARAM_DIR}")

    ch1_existing = {
        name for name in os.listdir(CH1_PARAM_DIR)
        if name.lower().endswith((".h5", ".hdf5"))
    }
    ch4_existing = {
        name for name in os.listdir(CH4_PARAM_DIR)
        if name.lower().endswith((".h5", ".hdf5"))
    }
    ch5_existing = {
        name for name in os.listdir(CH5_PARAM_DIR)
        if name.lower().endswith((".h5", ".hdf5"))
    }

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
    从单个 run 的 CH0/CH5/CH1/CH4 参数文件中读取基准特征：
        - ch0_max_ch0    ← max_ch0
        - ch0_ch0_min    ← ch0_min
        - ch5_max_ch5    ← max_ch5
        - ch0_ped_mean   ← ch0ped_mean
        - ch1_ped_mean   ← ch1ped_mean
        - max_ch1        ← max_ch1
        - ch1_min        ← ch1_min
        - max_ch4        ← max_ch4
        - tmax_ch4       ← tmax_ch4
        - tmax_ch0       ← tmax_ch0
        - tmax_ch1       ← tmax_ch1

    返回:
        max_ch0, ch0_min, max_ch5, ch0_ped_mean, ch1_ped_mean, max_ch1, ch1_min, max_ch4, tmax_ch4, tmax_ch0, tmax_ch1
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
        print(
            f"[警告] 事件数不一致，仅使用前 {n} 个事件。"
        )
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

# -----------------------------------------------------------------------------
# 独立的 cut 函数：输入为对应参数数组，输出为 bool 掩码
# -----------------------------------------------------------------------------

def cut_ch0_min_positive(ch0_min: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    条件：ch0_min > threshold（排除抑制信号）。
    输入: ch0_min 数组
    输出: bool 掩码
    """
    return ch0_min > threshold

def cut_ch0_max_saturation(max_ch0: np.ndarray, max_val: float = 16382.0) -> np.ndarray:
    """
    条件：max_ch0 <= max_val（排除饱和事例）。
    输入: max_ch0 数组
    输出: bool 掩码
    """
    return max_ch0 <= max_val

def cut_ch5_self_trigger(max_ch5: np.ndarray, rt_threshold: float = 6000.0) -> np.ndarray:
    """
    条件：max_ch5 <= rt_threshold（排除随机触发）。
    输入: max_ch5 数组
    输出: bool 掩码
    """
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
    保留 |ch0_ped - μ0| <= n_sigma*σ0 且 |ch1_ped - μ1| <= n_sigma*σ1 的事件。若 RT 事例不足或 σ=0，对应通道返回全 True 掩码（不剔除）。
    输入: ch0_ped_mean, ch1_ped_mean, max_ch5
    输出: bool 掩码"""

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
    dt_max_us: float = 16.0,) -> np.ndarray:
    """
    ACT cut：
    - 对 NaI 过阈事件 (max_ch4 >= trigger_threshold)，选取 Δt 非 [dt_min_us, dt_max_us] μs 的事例（反符合）；
    - 对 NaI 未过阈事件，视为“非 ACV 约束”，一律保留。
    参考 acv.py：Δt = t_Ge - t_CH4，t_CH4(μs) = tmax_ch4 * sampling_interval_ns * 1e-3。
    输入: max_ch4, tmax_ch4
    输出: bool 掩码（True = 通过 ACT cut 的事例）
    """
    n = max_ch4.shape[0]
    tmax_ch4 = np.asarray(tmax_ch4, dtype=np.float64)[:n]
    max_ch4 = np.asarray(max_ch4, dtype=np.float64)[:n]
    # NaI 是否过阈
    nai_ok = max_ch4 >= trigger_threshold
    t_ch4_us = tmax_ch4 * sampling_interval_ns * 1e-3
    delta_t_us = t_ge_us - t_ch4_us
    # 对 NaI 过阈事件：ACT = Δt 非 [dt_min_us, dt_max_us] 范围
    act_mask = (delta_t_us < dt_min_us) | (delta_t_us > dt_max_us)
    # 最终通过条件：
    # - NaI 未过阈（nai_ok == False）：全部保留；
    # - NaI 过阈（nai_ok == True）：要求满足 act_mask。
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
    """
    mincut：在 ACT 基础上，用 ACT 事例拟合 CH0min/CH1min 分布，
    保留 CH0min、CH1min 均在中心值 ± n_sigma*σ 内的事件。
    若拟合样本不足或 σ=0，对应通道不剔除。
    输入: ch0_min, ch1_min, max_ch5, max_ch4, tmax_ch4
    输出: bool 掩码
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
    min_fit_events: int = 10,) -> np.ndarray:
    """
    pncut：
    - 先在传入 base_mask 选中的事件中，限定 CH0max 落在 (fit_ch0_min, fit_ch0_max) 区间，
      对对应的 (CH0max, CH1max) 做一次线性拟合 y = a * x + b；
    - 计算所有事件相对这条直线的残差 r = CH1max - (a * CH0max + b)；
    - 将 |r| <= n_sigma * σ 的事件（σ 为残差标准差）视为“主相关带”上的事件，输出 True。
    若拟合样本不足或 σ=0，则返回与 base_mask 相同的掩码。
    """
    n = max_ch0.shape[0]
    assert max_ch1.shape[0] == n and base_mask.shape[0] == n

    # 拟合样本：在 base_mask 内，且 CH0max 在指定范围内
    fit_mask = (
        base_mask
        & (max_ch0 > fit_ch0_min)
        & (max_ch0 < fit_ch0_max)
    )
    x_fit = max_ch0[fit_mask]
    y_fit = max_ch1[fit_mask]

    if x_fit.size < min_fit_events:
        # 样本太少，不做任何额外切除
        return base_mask.copy()

    # 一阶多项式拟合：y = a * x + b
    a, b = np.polyfit(x_fit, y_fit, deg=1)
    y_pred_fit = a * x_fit + b
    resid_fit = y_fit - y_pred_fit
    sigma = float(resid_fit.std(ddof=1))
    if sigma <= 0.0:
        return base_mask.copy()

    # 对所有事件计算残差，并要求在 ± n_sigma * σ 内
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
    """
    CH0 max vs tmax(CH0) 相关带 cut（与 maxch0-tmax 相同模型与拟合窗口）：
    在 base_mask 为 True 且 x∈(fit_x_min,∞)、y∈(fit_y_min, fit_y_max) 的子集上拟合
    y = a·ln(x−b) + c·x + d/x；σ 为拟合残差标准差；
    保留 |tmax_ch0 − ŷ(max_ch0)| <= n_sigma * σ 的事件（默认 3σ，与图中红色带一致）。
    若拟合样本不足、curve_fit 失败或 σ=0，则返回与 base_mask 相同的掩码（不额外剔除）。
    对 max_ch0 <= b 无法定义 ŷ 的位置，视为不通过本 cut（False）。
    输入: base_mask, max_ch0, tmax_ch0
    输出: bool 掩码
    """
    def _model(
        x: np.ndarray, a: float, b: float, c: float, d: float
    ) -> np.ndarray:
        """y = a·ln(x−b) + c·x + d/x，要求 x > b（与 maxch0-tmax 一致）。"""
        return a * np.log(x - b) + c * x + d / x

    n = max_ch0.shape[0]
    assert tmax_ch0.shape[0] == n and base_mask.shape[0] == n

    max_ch0 = np.asarray(max_ch0, dtype=np.float64)
    tmax_ch0 = np.asarray(tmax_ch0, dtype=np.float64)

    fit_mask = (
        base_mask
        & (max_ch0 > fit_x_min)
        & (tmax_ch0 > fit_y_min)
        & (tmax_ch0 < fit_y_max)
    )
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
    bounds = (
        [-np.inf, -np.inf, -np.inf, -np.inf],
        [np.inf, b_hi, np.inf, np.inf],
    )
    try:
        popt, _ = curve_fit(
            _model,
            xf,
            yf,
            p0=p0,
            bounds=bounds,
            maxfev=100000,
        )
    except (ValueError, RuntimeError):
        return base_mask.copy()
    a, b, c, d = float(popt[0]), float(popt[1]), float(popt[2]), float(popt[3])
    if not np.isfinite([a, b, c, d]).all():
        return base_mask.copy()
    y_hat_fit = _model(xf, a, b, c, d)
    sigma = float(np.std(yf - y_hat_fit))
    if sigma <= 0.0:
        return base_mask.copy()

    x_all = max_ch0
    valid = x_all > b + 1e-9
    y_pred = np.empty(n, dtype=np.float64)
    y_pred[:] = np.nan
    y_pred[valid] = _model(x_all[valid], a, b, c, d)
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
    """
    CH1 max vs tmax(CH1) 相关带 cut（与 maxch1-tmax 相同模型与拟合窗口）：
    在 base_mask 为 True 且 x∈(fit_x_min,∞)、y∈(fit_y_min, fit_y_max) 的子集上拟合
    y = a·ln(x−b) + c·x + d/x；σ 为拟合残差标准差；
    保留 |tmax_ch1 − ŷ(max_ch1)| <= n_sigma * σ 的事件（默认 1σ，与图中红色带一致）。
    若拟合样本不足、curve_fit 失败或 σ=0，则返回与 base_mask 相同的掩码（不额外剔除）。
    对 max_ch1 <= b 无法定义 ŷ 的位置，视为不通过本 cut（False）。
    输入: base_mask, max_ch1, tmax_ch1
    输出: bool 掩码
    """
    def _model(
        x: np.ndarray, a: float, b: float, c: float, d: float
    ) -> np.ndarray:
        return a * np.log(x - b) + c * x + d / x

    n = max_ch1.shape[0]
    assert tmax_ch1.shape[0] == n and base_mask.shape[0] == n

    max_ch1 = np.asarray(max_ch1, dtype=np.float64)
    tmax_ch1 = np.asarray(tmax_ch1, dtype=np.float64)

    fit_mask = (
        base_mask
        & (max_ch1 > fit_x_min)
        & (tmax_ch1 > fit_y_min)
        & (tmax_ch1 < fit_y_max)
    )
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
    bounds = (
        [-np.inf, -np.inf, -np.inf, -np.inf],
        [np.inf, b_hi, np.inf, np.inf],
    )
    try:
        popt, _ = curve_fit(
            _model,
            xf,
            yf,
            p0=p0,
            bounds=bounds,
            maxfev=100000,
        )
    except (ValueError, RuntimeError):
        return base_mask.copy()
    a, b, c, d = float(popt[0]), float(popt[1]), float(popt[2]), float(popt[3])
    if not np.isfinite([a, b, c, d]).all():
        return base_mask.copy()
    y_hat_fit = _model(xf, a, b, c, d)
    sigma = float(np.std(yf - y_hat_fit))
    if sigma <= 0.0:
        return base_mask.copy()

    x_all = max_ch1
    valid = x_all > b + 1e-9
    y_pred = np.empty(n, dtype=np.float64)
    y_pred[:] = np.nan
    y_pred[valid] = _model(x_all[valid], a, b, c, d)
    resid = tmax_ch1 - y_pred
    in_band = np.zeros(n, dtype=bool)
    in_band[valid] = np.abs(resid[valid]) <= n_sigma * sigma
    return in_band


# -----------------------------------------------------------------------------
# 临时测试函数（之后可删）
# -----------------------------------------------------------------------------
def _plot_ch0max_hist_passing_cuts(
    mask: np.ndarray,
    max_ch0: np.ndarray,
    max_ch1: np.ndarray,) -> None:
    """临时：绘制通过所有 cut 的事例的 CH0max 直方图和 CH0max vs CH1max 散点图。"""
    x = max_ch0[mask]
    y = max_ch1[mask]

    # ------------------------------------------------------------------
    # 图 1：按 spectrum.py 中的能量刻度与归一化画能谱
    # ADC -> 能量(keV) 线性变换: E = a * x + b
    # （系数与 spectrum.py 中保持一致）
    # ------------------------------------------------------------------
    a = 0.0008432447500464594
    b = -0.826976770117076
    energy_values = a * x + b

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})

    # 先用 numpy.histogram 得到计数和能量 bin 边界
    counts, bin_edges = np.histogram(energy_values, bins=500)
    bin_widths = np.diff(bin_edges)  # keV
    bin_centers = bin_edges[:-1] + bin_widths / 2.0

    # 归一化：rate = count / (0.5 kg * bin_width (keV) * 20 day)
    exposure_kg = 0.5
    exposure_days = 20.0
    denom = exposure_kg * bin_widths * exposure_days
    # 避免除以 0（理论上 bin_width 不会为 0，这里只是保险）
    denom[denom == 0] = np.inf
    rates = counts / denom

    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    ax1.bar(bin_centers, rates, width=bin_widths, color="C0", alpha=0.8, align="center")
    ax1.set_yscale("log")
    ax1.set_xlabel("Energy (keV)", fontsize=12)
    ax1.set_ylabel(r"Rate [counts / (keV·kg·day)]", fontsize=12)
    ax1.set_title(f"Energy spectrum of events passing all cuts (N={x.size})", fontsize=13)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()

    # 图 2：CH0max vs CH1max 散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=2, alpha=0.5, edgecolors="none")
    plt.xlabel("CH0 maximum amplitude (FADC)")
    plt.ylabel("CH1 maximum amplitude (FADC)")
    plt.title(f"CH0max vs CH1max (events passing all cuts, N={x.size})")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    pairs = _list_paired_param_files()
    print(f"找到 {len(pairs)} 个可配对的参数文件。")
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
        m0, cmin, m5, ped0, ped1, m1, c1min, m4, t4, tc0, tc1 = _load_basic_features_for_run(
            ch0_path, ch5_path
        )
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
    max_ch1 = np.concatenate(all_max_ch1)
    ch1_min = np.concatenate(all_ch1_min)
    max_ch4 = np.concatenate(all_max_ch4)
    tmax_ch4 = np.concatenate(all_tmax_ch4)
    tmax_ch0 = np.concatenate(all_tmax_ch0)
    tmax_ch1 = np.concatenate(all_tmax_ch1)

    n_raw = max_ch0.shape[0]
    print(f"\n原始事件数: {n_raw}")

    # 1. 单独使用各 cut
    m1 = cut_ch0_min_positive(ch0_min)
    n1 = int(m1.sum())
    print(f"cut_ch0_min_positive 单独使用后: {n1} / {n_raw}")

    m2 = cut_ch0_max_saturation(max_ch0)
    n2 = int(m2.sum())
    print(f"cut_ch0_max_saturation 单独使用后: {n2} / {n_raw}")

    m3 = cut_ch5_self_trigger(max_ch5)
    n3 = int(m3.sum())
    print(f"cut_ch5_self_trigger 单独使用后: {n3} / {n_raw}")

    m4 = cut_pedestal_3sigma(ch0_ped_mean, ch1_ped_mean, max_ch5)
    n4 = int(m4.sum())
    print(f"cut_pedestal_3sigma 单独使用后: {n4} / {n_raw}")

    m5 = cut_act(max_ch4, tmax_ch4)
    n5 = int(m5.sum())
    print(f"cut_act 单独使用后: {n5} / {n_raw}")

    m6 = cut_mincut(ch0_min, ch1_min, max_ch4, tmax_ch4)
    n6 = int(m6.sum())
    print(f"cut_mincut 单独使用后: {n6} / {n_raw}")

    m7 = cut_ch0max_tmax(m1 & m2 & m3 & m4 & m5 & m6, max_ch0, tmax_ch0)
    n7 = int((m1 & m2 & m3 & m4 & m5 & m6 & m7).sum())
    print(f"cut_ch0max_tmax 单独使用后: {n7} / {n_raw}")

    m8 = cut_pncut(m1 & m2 & m3 & m4 & m5 & m6 & m7, max_ch0, max_ch1)
    n8 = int((m1 & m2 & m3 & m4 & m5 & m6 & m7 & m8).sum())
    print(f"cut_pncut 单独使用后: {n8} / {n_raw}")

    m9 = cut_ch1max_tmax(m1 & m2 & m3 & m4 & m5 & m6 & m7 & m8, max_ch1, tmax_ch1)
    n9 = int((m1 & m2 & m3 & m4 & m5 & m6 & m7 & m8 & m9).sum())
    print(f"cut_ch1max_tmax 单独使用后: {n9} / {n_raw}")

    # 2. 依次使用九种 cut 后剩余（与上面 m1~m9 定义一致）
    mask = m1 & m2 & m3 & m4 & m5 & m6 & m7 & m9
    mask = m1 & m2 & m4 & m5 & m6 & m8 
    n_final = int(mask.sum())
    print(f"\n依次使用 cut 后最终剩余: {n_final} / {n_raw}")

    # 3. 绘制最终剩余事件的 CH0max 直方图 和 CH0max vs CH1max 散点图
    _plot_ch0max_hist_passing_cuts(mask, max_ch0, max_ch1)

