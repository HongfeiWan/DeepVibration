#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CH3 tanh 参数 vs max(CH0)，保留 combine_spectrum.py 中 m0~m6 的事例：
- m0: cut_fit_success
- m1: cut_ch0_min_positive（ch0_min>0）
- m2: cut_ch0_max_saturation（max_ch0≤16382）
- m3: cut_ch5_self_trigger（max_ch5≤6000）
- m4: cut_pedestal_3sigma
- m4b: cut_ch3ped_min
- m5: cut_acv
- m6: cut_mincut

作图：p0、p3 子图 y 轴为对数（仅绘制 y>0）。

数据来源（与 fit_ch2_ch3_parallel.py / preprocessor 一致）：
- CH0_parameters/<run>.h5 : max_ch0, ch0_min
- CH2_parameters/<run>.h5 : n_fit_points, tanh_p0
- CH1_parameters/<run>.h5 : ch1ped_mean, ch1_min
- CH3_parameters/<run>.h5 : n_fit_points, tanh_p0, tanh_p1, tanh_p2, tanh_p3, ch3ped_mean, min_ch3
- CH4_parameters/<run>.h5 : max_ch4, tmax_ch4
- CH5_parameters/<run>.h5 : max_ch5
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _discover_project_root() -> Path:
    here = Path(__file__).resolve()
    python_dir = here.parents[3]
    return python_dir.parent


PROJECT_ROOT = _discover_project_root()
DATA_ROOT = PROJECT_ROOT / "data" / "hdf5" / "raw_pulse"
CH0_PARAM_DIR = DATA_ROOT / "CH0_parameters"
CH1_PARAM_DIR = DATA_ROOT / "CH1_parameters"
CH2_PARAM_DIR = DATA_ROOT / "CH2_parameters"
CH3_PARAM_DIR = DATA_ROOT / "CH3_parameters"
CH4_PARAM_DIR = DATA_ROOT / "CH4_parameters"
CH5_PARAM_DIR = DATA_ROOT / "CH5_parameters"


def cut_fit_success(
    ch2_n_fit_points: np.ndarray,
    ch3_n_fit_points: np.ndarray,
    ch2_tanh_p0: np.ndarray,
    ch3_tanh_p0: np.ndarray,
    bad_val: float = 1e6,) -> np.ndarray:
    """
    与 basic+act.py 一致：拟合成功筛选。
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


def cut_ch0_min_positive(ch0_min: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """ch0_min > threshold（排除抑制信号），与 basic+act.py 一致。"""
    return ch0_min > threshold


def cut_ch0_max_saturation(max_ch0: np.ndarray, max_val: float = 16382.0) -> np.ndarray:
    """max_ch0 <= max_val（排除饱和事例），与 basic+act.py 一致。"""
    return max_ch0 <= max_val


def cut_ch5_self_trigger(max_ch5: np.ndarray, rt_threshold: float = 6000.0) -> np.ndarray:
    """max_ch5 <= rt_threshold（排除随机触发），与 basic+act.py 一致。"""
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
    sigma_yx: float = 20.0,) -> np.ndarray:
    """CH3 ped-min 带状 cut。"""
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
    """acv cut（与 combine_spectrum.py 一致）。"""
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
    """mincut：在 acv 基础上拟合 CH0min/CH1min。"""
    n = ch0_min.shape[0]
    mask = np.ones(n, dtype=bool)
    acv_mask = cut_acv(
        max_ch4,
        tmax_ch4,
        trigger_threshold,
        t_ge_us,
        sampling_interval_ns,
        dt_min_us,
        dt_max_us,
    )
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


def _list_runs_with_ch2_ch3() -> List[str]:
    if not CH0_PARAM_DIR.exists():
        raise FileNotFoundError(f"不存在: {CH0_PARAM_DIR}")
    if not CH2_PARAM_DIR.exists():
        raise FileNotFoundError(f"不存在: {CH2_PARAM_DIR}")
    if not CH3_PARAM_DIR.exists():
        raise FileNotFoundError(f"不存在: {CH3_PARAM_DIR}")
    if not CH4_PARAM_DIR.exists():
        raise FileNotFoundError(f"不存在: {CH4_PARAM_DIR}")
    if not CH5_PARAM_DIR.exists():
        raise FileNotFoundError(f"不存在: {CH5_PARAM_DIR}")
    if not CH1_PARAM_DIR.exists():
        raise FileNotFoundError(f"不存在: {CH1_PARAM_DIR}")

    ch0_files = sorted(
        n for n in os.listdir(CH0_PARAM_DIR) if n.lower().endswith((".h5", ".hdf5"))
    )
    ch2_set = {n for n in os.listdir(CH2_PARAM_DIR) if n.lower().endswith((".h5", ".hdf5"))}
    ch3_set = {n for n in os.listdir(CH3_PARAM_DIR) if n.lower().endswith((".h5", ".hdf5"))}
    ch4_set = {n for n in os.listdir(CH4_PARAM_DIR) if n.lower().endswith((".h5", ".hdf5"))}
    ch5_set = {n for n in os.listdir(CH5_PARAM_DIR) if n.lower().endswith((".h5", ".hdf5"))}
    ch1_set = {n for n in os.listdir(CH1_PARAM_DIR) if n.lower().endswith((".h5", ".hdf5"))}
    out = [n for n in ch0_files if n in ch1_set and n in ch2_set and n in ch3_set and n in ch4_set and n in ch5_set]
    if not out:
        raise RuntimeError("CH0/CH2/CH3_parameters 无同名 run 可配对。")
    return out


def _load_one_run(name: str) -> Tuple[np.ndarray, ...]:
    """返回对齐后的所有 cut 与作图所需数组。"""
    p0 = CH0_PARAM_DIR / name
    p1 = CH1_PARAM_DIR / name
    p2 = CH2_PARAM_DIR / name
    p3 = CH3_PARAM_DIR / name
    p4 = CH4_PARAM_DIR / name
    p5 = CH5_PARAM_DIR / name

    with h5py.File(p0, "r") as f:
        if "max_ch0" not in f or "ch0_min" not in f or "ch0ped_mean" not in f:
            raise KeyError(f"{p0} 缺少 max_ch0 或 ch0_min 或 ch0ped_mean")
        max_ch0 = np.asarray(f["max_ch0"][...], dtype=np.float64)
        ch0_min = np.asarray(f["ch0_min"][...], dtype=np.float64)
        ch0_ped_mean = np.asarray(f["ch0ped_mean"][...], dtype=np.float64)
    with h5py.File(p1, "r") as f:
        if "ch1ped_mean" not in f or "ch1_min" not in f:
            raise KeyError(f"{p1} 缺少 ch1ped_mean 或 ch1_min")
        ch1_ped_mean = np.asarray(f["ch1ped_mean"][...], dtype=np.float64)
        ch1_min = np.asarray(f["ch1_min"][...], dtype=np.float64)
    with h5py.File(p5, "r") as f:
        if "max_ch5" not in f:
            raise KeyError(f"{p5} 缺少 max_ch5")
        max_ch5 = np.asarray(f["max_ch5"][...], dtype=np.float64)
    with h5py.File(p4, "r") as f:
        if "max_ch4" not in f or "tmax_ch4" not in f:
            raise KeyError(f"{p4} 缺少 max_ch4 或 tmax_ch4")
        max_ch4 = np.asarray(f["max_ch4"][...], dtype=np.float64)
        tmax_ch4 = np.asarray(f["tmax_ch4"][...], dtype=np.float64)
    with h5py.File(p2, "r") as f:
        for k in ("n_fit_points", "tanh_p0"):
            if k not in f:
                raise KeyError(f"{p2} 缺少 {k}")
        ch2_n = np.asarray(f["n_fit_points"][...], dtype=np.int32)
        ch2_p0 = np.asarray(f["tanh_p0"][...], dtype=np.float64)
    with h5py.File(p3, "r") as f:
        for k in ("n_fit_points", "tanh_p0", "tanh_p1", "tanh_p2", "tanh_p3"):
            if k not in f:
                raise KeyError(f"{p3} 缺少 {k}")
        ch3_n = np.asarray(f["n_fit_points"][...], dtype=np.int32)
        ch3_p0 = np.asarray(f["tanh_p0"][...], dtype=np.float64)
        ch3_p1 = np.asarray(f["tanh_p1"][...], dtype=np.float64)
        ch3_p2 = np.asarray(f["tanh_p2"][...], dtype=np.float64)
        ch3_p3 = np.asarray(f["tanh_p3"][...], dtype=np.float64)
        ch3ped_mean = np.full(ch3_n.shape[0], np.nan, dtype=np.float64)
        min_ch3 = np.full(ch3_n.shape[0], np.nan, dtype=np.float64)
        if "ch3ped_mean" in f:
            ch3ped_mean = np.asarray(f["ch3ped_mean"][...], dtype=np.float64)
        if "min_ch3" in f:
            min_ch3 = np.asarray(f["min_ch3"][...], dtype=np.float64)

    n = min(
        max_ch0.shape[0],
        ch0_min.shape[0],
        ch0_ped_mean.shape[0],
        ch1_ped_mean.shape[0],
        ch1_min.shape[0],
        max_ch4.shape[0],
        tmax_ch4.shape[0],
        max_ch5.shape[0],
        ch2_n.shape[0],
        ch3_n.shape[0],
        ch2_p0.shape[0],
        ch3_p0.shape[0],
        ch3_p1.shape[0],
        ch3_p2.shape[0],
        ch3_p3.shape[0],
        ch3ped_mean.shape[0],
        min_ch3.shape[0],
    )
    return (
        max_ch0[:n],
        ch0_min[:n],
        ch0_ped_mean[:n],
        ch1_ped_mean[:n],
        ch1_min[:n],
        max_ch4[:n],
        tmax_ch4[:n],
        max_ch5[:n],
        ch2_n[:n],
        ch3_n[:n],
        ch2_p0[:n],
        ch3_p0[:n],
        ch3_p1[:n],
        ch3_p2[:n],
        ch3_p3[:n],
        ch3ped_mean[:n],
        min_ch3[:n],
    )


def load_all_events() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """全数据拼接；返回 max_ch0, ch3_p0, ch3_p1, ch3_p2, ch3_p3 及用于 cut 的数组。"""
    names = _list_runs_with_ch2_ch3()
    am0: List[np.ndarray] = []
    acmin: List[np.ndarray] = []
    aped0: List[np.ndarray] = []
    aped1: List[np.ndarray] = []
    ac1min: List[np.ndarray] = []
    am4: List[np.ndarray] = []
    at4: List[np.ndarray] = []
    am5: List[np.ndarray] = []
    a2n: List[np.ndarray] = []
    a3n: List[np.ndarray] = []
    a2p0: List[np.ndarray] = []
    a3p0: List[np.ndarray] = []
    a3p1: List[np.ndarray] = []
    a3p2: List[np.ndarray] = []
    a3p3: List[np.ndarray] = []

    a3ped: List[np.ndarray] = []
    a3min: List[np.ndarray] = []

    for name in names:
        (
            max_ch0,
            ch0_min,
            ch0_ped_mean,
            ch1_ped_mean,
            ch1_min,
            max_ch4,
            tmax_ch4,
            max_ch5,
            ch2_n,
            ch3_n,
            ch2_p0,
            ch3_p0,
            ch3_p1,
            ch3_p2,
            ch3_p3,
            ch3ped_mean,
            min_ch3,
        ) = (
            _load_one_run(name)
        )
        am0.append(max_ch0)
        acmin.append(ch0_min)
        aped0.append(ch0_ped_mean)
        aped1.append(ch1_ped_mean)
        ac1min.append(ch1_min)
        am4.append(max_ch4)
        at4.append(tmax_ch4)
        am5.append(max_ch5)
        a2n.append(ch2_n)
        a3n.append(ch3_n)
        a2p0.append(ch2_p0)
        a3p0.append(ch3_p0)
        a3p1.append(ch3_p1)
        a3p2.append(ch3_p2)
        a3p3.append(ch3_p3)
        a3ped.append(ch3ped_mean)
        a3min.append(min_ch3)

    max_ch0 = np.concatenate(am0)
    ch0_min = np.concatenate(acmin)
    ch0_ped_mean = np.concatenate(aped0)
    ch1_ped_mean = np.concatenate(aped1)
    ch1_min = np.concatenate(ac1min)
    max_ch4 = np.concatenate(am4)
    tmax_ch4 = np.concatenate(at4)
    max_ch5 = np.concatenate(am5)
    ch2_n = np.concatenate(a2n)
    ch3_n = np.concatenate(a3n)
    ch2_p0 = np.concatenate(a2p0)
    ch3_p0 = np.concatenate(a3p0)
    ch3_p1 = np.concatenate(a3p1)
    ch3_p2 = np.concatenate(a3p2)
    ch3_p3 = np.concatenate(a3p3)
    ch3ped_mean = np.concatenate(a3ped)
    min_ch3 = np.concatenate(a3min)

    m0 = cut_fit_success(ch2_n, ch3_n, ch2_p0, ch3_p0)
    m1 = cut_ch0_min_positive(ch0_min)
    m2 = cut_ch0_max_saturation(max_ch0)
    m3 = cut_ch5_self_trigger(max_ch5)
    m4 = cut_pedestal_3sigma(ch0_ped_mean, ch1_ped_mean, max_ch5)
    m4b = cut_ch3ped_min(ch3ped_mean, min_ch3)
    m5 = cut_acv(max_ch4, tmax_ch4)
    m6 = cut_mincut(ch0_min, ch1_min, max_ch4, tmax_ch4)

    m = m0 & m1 & m2 & m3 & m4 & m4b & m5 & m6
    return max_ch0, ch3_p0, ch3_p1, ch3_p2, ch3_p3, m


def plot_ch3_params_vs_max_ch0(
    max_ch0: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    mask: np.ndarray,
    *,
    save_path: Path | None = None,
    dpi: int = 120,) -> None:
    x = np.asarray(max_ch0[mask], dtype=np.float64)
    ys = [
        (np.asarray(p0[mask], dtype=np.float64), r"$p_0$ (CH3 $\tanh$)"),
        (np.asarray(p1[mask], dtype=np.float64), r"$p_1$"),
        (np.asarray(p2[mask], dtype=np.float64), r"$p_2$"),
        (np.asarray(p3[mask], dtype=np.float64), r"$p_3$"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=dpi)
    axes = axes.ravel()
    for i, (ax, (y, ylab)) in enumerate(zip(axes, ys)):
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=2, alpha=0.25, edgecolors="none")
        ax.set_xlabel(r"max(CH0) (ADC)")
        ax.set_ylabel(ylab)
        ax.set_xlim(0, 16382.0)
        if i == 0:
            ax.set_ylim(0, 200)
        elif i == 1:
            ax.set_ylim(0, 10)
            #pass
        elif i == 2:
            ax.set_ylim(0, 50)
        elif i == 3:
            ax.set_ylim(800, 1200)
        ax.grid(True, alpha=0.3)

    n_ok = int(mask.sum())
    fig.suptitle(
        "CH3 tanh parameters vs max(CH0), "
        "m0&m1&m2&m3&m4&m4b&m5&m6 applied "
        f"(N={n_ok})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"已保存: {save_path}")
    plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CH3 p0–p3 vs max(CH0)，应用 m0&m1&m2&m3&m4&m4b&m5&m6"
    )
    p.add_argument("--save", type=Path, default=None, help="保存图像路径")
    p.add_argument("--dpi", type=int, default=120)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    max_ch0, p0, p1, p2, p3, m = load_all_events()
    print(
        f"总事件 {max_ch0.shape[0]}，m0&m1&m2&m3&m4&m4b&m5&m6 通过 {int(m.sum())} "
        f"({100.0 * float(m.mean()):.2f}%)"
    )
    plot_ch3_params_vs_max_ch0(
        max_ch0, p0, p1, p2, p3, m, save_path=args.save, dpi=args.dpi
    )


if __name__ == "__main__":
    main()
