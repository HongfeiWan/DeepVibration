#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 CH3_parameters 中筛选 min_ch3 == 0 的事件，并绘制对应事件的 CH0/CH3 波形。

图像规范：
- 同一张 figure 中叠加 CH0 与 CH3；
- 左侧 y 轴为 CH0，右侧 y 轴为 CH3；
- x 轴为采样点索引（可切换为 us）。
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# 路径推断：
# __file__ = .../python/data/ge-self/cut/ch3zero.py
# current_dir = .../python/data/ge-self/cut
# data_dir = .../python/data
# python_dir = .../python
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.dirname(os.path.dirname(current_dir))
python_dir = os.path.dirname(data_dir)
project_root = os.path.dirname(python_dir)

ch3_param_dir_default = os.path.join(project_root, "data", "hdf5", "raw_pulse", "CH3_parameters")
ch0_3_dir_default = os.path.join(project_root, "data", "hdf5", "raw_pulse", "CH0-3")
ch0_param_dir_default = os.path.join(project_root, "data", "hdf5", "raw_pulse", "CH0_parameters")
ch1_param_dir_default = os.path.join(project_root, "data", "hdf5", "raw_pulse", "CH1_parameters")
ch4_param_dir_default = os.path.join(project_root, "data", "hdf5", "raw_pulse", "CH4_parameters")
ch5_param_dir_default = os.path.join(project_root, "data", "hdf5", "raw_pulse", "CH5_parameters")


def cut_ch0_min_positive(ch0_min: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    return ch0_min > threshold


def cut_ch0_max_saturation(max_ch0: np.ndarray, max_ch1: np.ndarray, max_val: float = 16382.0) -> np.ndarray:
    return (max_ch0 <= max_val) & (max_ch1 <= max_val)


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
    fit_mask = cut_act(max_ch4, tmax_ch4, trigger_threshold, t_ge_us, sampling_interval_ns, dt_min_us, dt_max_us)

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

def _resolve_source_file(param_path: str, source_file_attr: object, ch0_3_dir: str) -> str:
    """
    优先使用参数文件中的 source_file 属性；
    若缺失或无效，则回退为 CH0-3/同名文件。
    """
    source_file = ""
    if source_file_attr is not None:
        try:
            source_file = (
                source_file_attr.decode("utf-8")
                if isinstance(source_file_attr, bytes)
                else str(source_file_attr)
            )
        except Exception:
            source_file = str(source_file_attr)
    if source_file and os.path.exists(source_file):
        return source_file

    fallback = os.path.join(ch0_3_dir, os.path.basename(param_path))
    return fallback

def _find_zero_events_for_one_param_file(
    param_path: str,
    ch0_3_dir: str,
    ch0_param_dir: str,
    ch1_param_dir: str,
    ch4_param_dir: str,
    ch5_param_dir: str,
    apply_masks: bool = False,) -> Tuple[int, List[Tuple[str, int]]]:
    """
    返回该参数文件中：
    1) 原始 min_ch3 == 0 的事件数量（不加 mask）
    2) 若 apply_masks=False：直接返回所有 min_ch3 == 0 的事件索引
       若 apply_masks=True ：返回“快速 cuts”后仍满足 min_ch3 == 0 的事件索引
       （为保证速度，不包含基于 curve_fit 的 ch0/ch1 max-tmax 相关带 cuts）
    """
    zero_count_raw = 0
    out: List[Tuple[str, int]] = []
    try:
        file_name = os.path.basename(param_path)
        ch0_path = os.path.join(ch0_param_dir, file_name)
        ch1_path = os.path.join(ch1_param_dir, file_name)
        ch4_path = os.path.join(ch4_param_dir, file_name)
        ch5_path = os.path.join(ch5_param_dir, file_name)
        with h5py.File(param_path, "r") as f_param:
            if "min_ch3" not in f_param:
                return 0, out
            # 只做“快速候选”提取：直接在 CH3_parameters 中找 min_ch3==0
            # 这里不做 float64 强转，尽量减少开销
            min_ch3 = np.asarray(f_param["min_ch3"][...])
            zero_idx_all = np.flatnonzero(min_ch3 == 0)
            zero_count_raw = int(zero_idx_all.size)
            source_file = _resolve_source_file(param_path, f_param.attrs.get("source_file", None), ch0_3_dir)

        # 默认快速模式：不应用其它 mask，直接返回 min_ch3==0 的事件
        if not apply_masks:
            for idx in zero_idx_all:
                out.append((source_file, int(idx)))
            return zero_count_raw, out

        # apply_masks=True 时才做完整筛选：需要对应的 CH0/CH1/CH4/CH5 参数文件
        if not (
            os.path.exists(ch0_path)
            and os.path.exists(ch1_path)
            and os.path.exists(ch4_path)
            and os.path.exists(ch5_path)
        ):
            return zero_count_raw, out

        with h5py.File(ch0_path, "r") as f0:
            if (
                "max_ch0" not in f0
                or "ch0_min" not in f0
                or "ch0ped_mean" not in f0
            ):
                return zero_count_raw, out
            max_ch0 = np.asarray(f0["max_ch0"][...], dtype=np.float64)
            ch0_min = np.asarray(f0["ch0_min"][...], dtype=np.float64)
            ch0_ped_mean = np.asarray(f0["ch0ped_mean"][...], dtype=np.float64)

        with h5py.File(ch1_path, "r") as f1:
            if (
                "ch1ped_mean" not in f1
                or "ch1_min" not in f1
                or "max_ch1" not in f1
            ):
                return zero_count_raw, out
            ch1_ped_mean = np.asarray(f1["ch1ped_mean"][...], dtype=np.float64)
            ch1_min = np.asarray(f1["ch1_min"][...], dtype=np.float64)
            max_ch1 = np.asarray(f1["max_ch1"][...], dtype=np.float64)

        with h5py.File(ch4_path, "r") as f4:
            if "max_ch4" not in f4 or "tmax_ch4" not in f4:
                return zero_count_raw, out
            max_ch4 = np.asarray(f4["max_ch4"][...], dtype=np.float64)
            tmax_ch4 = np.asarray(f4["tmax_ch4"][...], dtype=np.float64)

        with h5py.File(ch5_path, "r") as f5:
            if "max_ch5" not in f5:
                return zero_count_raw, out
            max_ch5 = np.asarray(f5["max_ch5"][...], dtype=np.float64)

        n = min(
            min_ch3.shape[0],
            max_ch0.shape[0],
            ch0_min.shape[0],
            ch0_ped_mean.shape[0],
            ch1_ped_mean.shape[0],
            ch1_min.shape[0],
            max_ch1.shape[0],
            max_ch4.shape[0],
            tmax_ch4.shape[0],
            max_ch5.shape[0],
        )
        if n == 0:
            return zero_count_raw, out

        min_ch3 = min_ch3[:n]
        max_ch0 = max_ch0[:n]
        ch0_min = ch0_min[:n]
        ch0_ped_mean = ch0_ped_mean[:n]
        ch1_ped_mean = ch1_ped_mean[:n]
        ch1_min = ch1_min[:n]
        max_ch1 = max_ch1[:n]
        max_ch4 = max_ch4[:n]
        tmax_ch4 = tmax_ch4[:n]
        max_ch5 = max_ch5[:n]

        m1 = cut_ch0_min_positive(ch0_min)
        m2 = cut_ch0_max_saturation(max_ch0, max_ch1)
        m3 = cut_ch5_self_trigger(max_ch5)
        m4 = cut_pedestal_3sigma(ch0_ped_mean, ch1_ped_mean, max_ch5)
        m5 = cut_act(max_ch4, tmax_ch4)
        m6 = cut_mincut(ch0_min, ch1_min, max_ch4, tmax_ch4)
        base_mask = m1 & m2 & m3 & m4 & m5 & m6
        m_pn = cut_pncut(base_mask, max_ch0, max_ch1)
        final_mask = base_mask & m_pn

        zero_indices = np.where(final_mask & (min_ch3 == 0))[0]
        for idx in zero_indices:
            out.append((source_file, int(idx)))
    except Exception as e:
        print(f"[警告] 读取参数文件失败，跳过: {param_path}, 错误: {e}")
    return zero_count_raw, out

def _collect_zero_events(
    ch3_param_dir: str,
    ch0_3_dir: str,
    ch0_param_dir: str,
    ch1_param_dir: str,
    ch4_param_dir: str,
    ch5_param_dir: str,
    max_workers: Optional[int] = None,
    apply_masks: bool = False,) -> List[Tuple[str, int]]:
    """
    并行扫描 CH3_parameters 下所有 h5，收集 min_ch3 == 0 的事件。
    """
    if not os.path.isdir(ch3_param_dir):
        raise FileNotFoundError(f"CH3_parameters 目录不存在: {ch3_param_dir}")

    files = [
        os.path.join(ch3_param_dir, name)
        for name in sorted(os.listdir(ch3_param_dir))
        if name.lower().endswith(".h5")
    ]
    if not files:
        print(f"目录 {ch3_param_dir} 下未找到 h5 文件。")
        return []

    workers = max_workers if (max_workers is not None and max_workers > 0) else (os.cpu_count() or 1)
    workers = max(1, workers)
    if apply_masks:
        msg = "九个mask后 min_ch3 == 0"
    else:
        msg = "min_ch3 == 0（仅 CH3_parameters 快速筛选）"
    print(f"扫描 {len(files)} 个参数文件，使用 {workers} 个 CPU 核并行筛选“{msg}”事件。")

    matched: List[Tuple[str, int]] = []
    total_zero_raw = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _find_zero_events_for_one_param_file,
                p,
                ch0_3_dir,
                ch0_param_dir,
                ch1_param_dir,
                ch4_param_dir,
                ch5_param_dir,
                apply_masks,
            ): p
            for p in files
        }
        for fut in as_completed(futures):
            zero_count_raw, matched_one = fut.result()
            total_zero_raw += int(zero_count_raw)
            matched.extend(matched_one)

    print(f"所有文件中原始 min_ch3 == 0 的信号总数：{total_zero_raw}")
    if apply_masks:
        print(f"共找到 {len(matched)} 个“九个mask后 min_ch3 == 0”的事件。")
    else:
        print(f"共找到 {len(matched)} 个“min_ch3 == 0（快速模式）”事件。")
    return matched

def _plot_one_event(
    source_file: str,
    event_idx: int,
    use_time_us: bool = False,
    sampling_interval_ns: float = 4.0,) -> None:
    """
    绘制单个事件的 CH0/CH3 波形（双 y 轴）。
    """
    if not os.path.exists(source_file):
        print(f"[警告] 源文件不存在，跳过: {source_file}")
        return

    with h5py.File(source_file, "r") as f_src:
        if "channel_data" not in f_src:
            print(f"[警告] 源文件缺少 channel_data，跳过: {source_file}")
            return
        ch_data = f_src["channel_data"]
        if ch_data.ndim != 3:
            print(f"[警告] channel_data 维度异常，跳过: {source_file}, shape={ch_data.shape}")
            return
        d0, num_channels, d2 = ch_data.shape
        if num_channels <= 3:
            print(f"[警告] 源文件通道数不足（需要 CH0 和 CH3），跳过: {source_file}")
            return

        # 兼容两种常见布局：
        # 1) (time, ch, event)
        # 2) (event, ch, time)
        is_time_ch_event = d0 >= d2
        num_events = int(d2 if is_time_ch_event else d0)
        time_samples = int(d0 if is_time_ch_event else d2)
        if event_idx < 0 or event_idx >= num_events:
            print(f"[警告] event_idx 越界，跳过: {source_file}, event={event_idx}, num_events={num_events}")
            return

        if is_time_ch_event:
            wf_ch0 = np.asarray(ch_data[:, 0, event_idx], dtype=np.float64)
            wf_ch3 = np.asarray(ch_data[:, 3, event_idx], dtype=np.float64)
        else:
            wf_ch0 = np.asarray(ch_data[event_idx, 0, :], dtype=np.float64)
            wf_ch3 = np.asarray(ch_data[event_idx, 3, :], dtype=np.float64)

    ch0_min_val = float(np.min(wf_ch0)) if wf_ch0.size else float("nan")
    ch3_min_val = float(np.min(wf_ch3)) if wf_ch3.size else float("nan")
    print(
        f"[event] file={os.path.basename(source_file)} idx={event_idx} "
        f"min(CH0)={ch0_min_val:.6g} min(CH3)={ch3_min_val:.6g}"
    )

    if use_time_us:
        x = np.arange(time_samples, dtype=np.float64) * sampling_interval_ns / 1000.0
        xlabel = "Time [us]"
    else:
        x = np.arange(time_samples, dtype=np.int32)
        xlabel = "Sample index"

    fig, ax_left = plt.subplots(figsize=(10, 5.5))
    ax_right = ax_left.twinx()

    line0 = ax_left.plot(x, wf_ch0, color="C0", linewidth=1.0, label="CH0")[0]
    line3 = ax_right.plot(x, wf_ch3, color="C3", linewidth=1.0, label="CH3")[0]

    ax_left.set_xlabel(xlabel)
    ax_left.set_ylabel("CH0 amplitude", color="C0")
    ax_right.set_ylabel("CH3 amplitude", color="C3")
    ax_left.tick_params(axis="y", labelcolor="C0")
    ax_right.tick_params(axis="y", labelcolor="C3")

    title = (
        f"{os.path.basename(source_file)} | event={event_idx} | "
        f"min(CH3)={ch3_min_val:.6g}"
    )
    ax_left.set_title(title)
    ax_left.grid(True, alpha=0.25, linestyle="--")
    ax_left.legend([line0, line3], ["CH0", "CH3"], loc="upper right")

    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 CH3_parameters 中筛选 min_ch3==0 的事件并绘制 CH0/CH3 双 y 轴波形。"
    )
    parser.add_argument(
        "--ch3-param-dir",
        type=str,
        default=ch3_param_dir_default,
        help=f"CH3 参数目录，默认: {ch3_param_dir_default}",
    )
    parser.add_argument(
        "--ch0-3-dir",
        type=str,
        default=ch0_3_dir_default,
        help=f"CH0-3 原始波形目录，默认: {ch0_3_dir_default}",
    )
    parser.add_argument("--ch0-param-dir", type=str, default=ch0_param_dir_default, help=f"CH0 参数目录，默认: {ch0_param_dir_default}")
    parser.add_argument("--ch1-param-dir", type=str, default=ch1_param_dir_default, help=f"CH1 参数目录，默认: {ch1_param_dir_default}")
    parser.add_argument("--ch4-param-dir", type=str, default=ch4_param_dir_default, help=f"CH4 参数目录，默认: {ch4_param_dir_default}")
    parser.add_argument("--ch5-param-dir", type=str, default=ch5_param_dir_default, help=f"CH5 参数目录，默认: {ch5_param_dir_default}")
    parser.add_argument(
        "--apply-masks",
        action="store_true",
        help="在 min_ch3==0 的基础上，额外应用九个 mask（更慢，会读取 CH0/1/4/5 参数并拟合）。默认不启用以获得最快速度。",
    )
    parser.add_argument(
        "--max-plots",
        type=int,
        default=50,
        help="最多绘制多少个事件（默认 50，<=0 表示全部）。",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="只做扫描与筛选，不绘图（用于最快速批量查找）。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="并行扫描参数文件使用的进程数；0 表示自动使用 os.cpu_count()。",
    )
    parser.add_argument(
        "--time-us",
        action="store_true",
        help="x 轴使用时间（us），否则使用采样点索引。",
    )
    parser.add_argument(
        "--sampling-interval-ns",
        type=float,
        default=4.0,
        help="采样间隔（ns），仅在 --time-us 时生效，默认 4.0。",
    )
    args = parser.parse_args()

    workers = None if args.workers <= 0 else args.workers
    matched_events = _collect_zero_events(
        ch3_param_dir=args.ch3_param_dir,
        ch0_3_dir=args.ch0_3_dir,
        ch0_param_dir=args.ch0_param_dir,
        ch1_param_dir=args.ch1_param_dir,
        ch4_param_dir=args.ch4_param_dir,
        ch5_param_dir=args.ch5_param_dir,
        max_workers=workers,
        apply_masks=bool(args.apply_masks),
    )
    if not matched_events:
        if args.apply_masks:
            print("未找到经过九个 mask 后且 min_ch3 == 0 的事件。")
        else:
            print("未找到 min_ch3 == 0 的事件。")
        return

    if args.no_plot:
        print(f"已完成扫描（no-plot），匹配事件数：{len(matched_events)}")
        return

    if args.max_plots > 0:
        matched_events = matched_events[: args.max_plots]
    print(f"开始绘图，共 {len(matched_events)} 个事件。关闭当前图窗口后显示下一个。")

    for source_file, event_idx in matched_events:
        _plot_one_event(
            source_file=source_file,
            event_idx=event_idx,
            use_time_us=args.time_us,
            sampling_interval_ns=args.sampling_interval_ns,
        )

    print("绘图完成。")


if __name__ == "__main__":
    main()
