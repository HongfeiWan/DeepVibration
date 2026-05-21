#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
maxch1-tmax: 对通过 basic+act 六种 cut 且 cut_ch0max_tmax（m7）的事例，
绘制 max(CH1) vs tmax(CH1) 散点图（掩码计算方式与 basic+act.py 测试块一致：全量拼接后向量化）。

HDF5 结构见 preprocessor.py：datasets max_ch1, tmax_ch1（及其它 CH1 特征）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# 加载 basic+act 模块（文件名含 + 无法直接 import）
_BASIC_ACT_PATH = Path(__file__).resolve().parent / "basic+act.py"
_spec = importlib.util.spec_from_file_location("basic_act", _BASIC_ACT_PATH)
basic_act = importlib.util.module_from_spec(_spec)
sys.modules["basic_act"] = basic_act
_spec.loader.exec_module(basic_act)

# 与 basic+act.py 测试块一致：直接使用其 cut 与加载函数（全数组上向量化，比按文件多进程 mask 更快）
cut_ch0_min_positive = basic_act.cut_ch0_min_positive
cut_ch0_max_saturation = basic_act.cut_ch0_max_saturation
cut_ch5_self_trigger = basic_act.cut_ch5_self_trigger
cut_pedestal_3sigma = basic_act.cut_pedestal_3sigma
cut_act = basic_act.cut_act
cut_mincut = basic_act.cut_mincut
cut_ch0max_tmax = basic_act.cut_ch0max_tmax
_list_paired_param_files = basic_act._list_paired_param_files
_load_basic_features_for_run = basic_act._load_basic_features_for_run
CH1_PARAM_DIR = basic_act.CH1_PARAM_DIR

# 拟合子集：x=max(CH1), y=tmax(CH1)；模型 y = a·ln(x−b) + c·x + d/x
FIT_X_MIN = 1350.0
FIT_Y_MIN = 11000.0
FIT_Y_MAX = 17500.0


def _model_alog_x_minus_b_cx_d_over_x(
    x: np.ndarray, a: float, b: float, c: float, d: float
) -> np.ndarray:
    """y = a·ln(x−b) + c·x + d/x，要求 x > b。"""
    return a * np.log(x - b) + c * x + d / x


def _fit_alog_x_minus_b_cx_d_over_x(
    x: np.ndarray, y: np.ndarray) -> Tuple[Optional[np.ndarray], int, Optional[float]]:
    """
    在 FIT_* 窗口内对 y = a·ln(x−b) + c·x + d/x 做非线性最小二乘（curve_fit）。
    返回 (coeffs [a,b,c,d], n_points, 残差标准差 σ)，失败时返回 (None, n, None)。
    """
    mask = (x > FIT_X_MIN) & (y > FIT_Y_MIN) & (y < FIT_Y_MAX)
    xf = np.asarray(x[mask], dtype=np.float64)
    yf = np.asarray(y[mask], dtype=np.float64)
    n = int(xf.size)
    if n < 4:
        return None, n, None
    xmin = float(np.min(xf))
    b_hi = xmin - 1e-9
    A0 = np.column_stack([np.log(xf), xf, 1.0 / xf])
    coef0, _, rank0, _ = np.linalg.lstsq(A0, yf, rcond=None)
    if rank0 < 3:
        return None, n, None
    a0, c0, d0 = float(coef0[0]), float(coef0[1]), float(coef0[2])
    b0 = min(xmin * 0.5, xmin - 200.0)
    p0 = np.array([a0, b0, c0, d0], dtype=np.float64)
    bounds = (
        [-np.inf, -np.inf, -np.inf, -np.inf],
        [np.inf, b_hi, np.inf, np.inf],
    )
    try:
        popt, _ = curve_fit(
            _model_alog_x_minus_b_cx_d_over_x,
            xf,
            yf,
            p0=p0,
            bounds=bounds,
            maxfev=100000,
        )
    except (ValueError, RuntimeError):
        return None, n, None
    a, b, c, d = (float(popt[0]), float(popt[1]), float(popt[2]), float(popt[3]))
    if not np.isfinite([a, b, c, d]).all():
        return None, n, None
    y_hat = _model_alog_x_minus_b_cx_d_over_x(xf, a, b, c, d)
    sigma = float(np.std(yf - y_hat))
    return np.array([a, b, c, d], dtype=np.float64), n, sigma


def main() -> None:
    pairs = _list_paired_param_files()
    n_files = len(pairs)
    print(f"[maxch1-tmax] 找到 {n_files} 个可配对的参数文件，顺序加载并拼接（与 basic+act 测试块一致）。")

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
        m0, cmin, m5, ped0, ped1, mx1, c1min, m4, t4, tc0 = _load_basic_features_for_run(
            ch0_path, ch5_path
        )
        ch1_path = CH1_PARAM_DIR / ch0_path.name
        with h5py.File(ch1_path, "r") as f:
            if "tmax_ch1" not in f:
                raise KeyError(
                    f"{ch1_path.name} 中缺少 tmax_ch1，请确认由当前版本 preprocessor.py 生成。"
                )
            tx1 = np.asarray(f["tmax_ch1"][...], dtype=np.float64)[: m0.shape[0]]

        all_max_ch0.append(m0)
        all_ch0_min.append(cmin)
        all_max_ch5.append(m5)
        all_ch0_ped_mean.append(ped0)
        all_ch1_ped_mean.append(ped1)
        all_max_ch1.append(mx1)
        all_ch1_min.append(c1min)
        all_max_ch4.append(m4)
        all_tmax_ch4.append(t4)
        all_tmax_ch0.append(tc0)
        all_tmax_ch1.append(tx1)

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
    print(f"[maxch1-tmax] 原始事件数: {n_raw}")

    # 与 basic+act.py __main__ 中 m1–m7 一致（向量化）
    m1 = cut_ch0_min_positive(ch0_min)
    m2 = cut_ch0_max_saturation(max_ch0, max_ch1)
    m3 = cut_ch5_self_trigger(max_ch5)
    m4 = cut_pedestal_3sigma(ch0_ped_mean, ch1_ped_mean, max_ch5)
    m5 = cut_act(max_ch4, tmax_ch4)
    m6 = cut_mincut(ch0_min, ch1_min, max_ch4, tmax_ch4)
    m7 = cut_ch0max_tmax(m1 & m2 & m3 & m4 & m5 & m6, max_ch0, tmax_ch0)
    mask = m1 & m2 & m3 & m4 & m5 & m6 & m7

    n_pass = int(mask.sum())
    print(
        f"[maxch1-tmax] basic+act+m7 通过: {n_pass} / {n_raw} "
        f"（m7 与 basic+act 测试块中 cut_ch0max_tmax 定义一致）"
    )
    if n_pass == 0:
        print("[maxch1-tmax] 无通过事例，退出。")
        return

    m = max_ch1[mask]
    t = tmax_ch1[mask]
    print(f"[maxch1-tmax] 共 {m.size} 个 (max_ch1, tmax_ch1) 点，一次绘制。")

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_xlabel("max(CH1) (ADC)", fontsize=12)
    ax.set_ylabel("tmax(CH1) (sample index)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.scatter(m, t, s=2, alpha=0.5, edgecolors="none", label="data")

    coeffs, n_fit, sigma_fit = _fit_alog_x_minus_b_cx_d_over_x(m, t)
    if coeffs is not None:
        a, b, c, d = float(coeffs[0]), float(coeffs[1]), float(coeffs[2]), float(coeffs[3])
        print(
            f"[maxch1-tmax] 拟合 y=a·ln(x−b)+c·x+d/x（{n_fit} 点，"
            f"x>{FIT_X_MIN}, {FIT_Y_MIN}<y<{FIT_Y_MAX}）: "
            f"a={a:.6g}, b={b:.6g}, c={c:.6g}, d={d:.6g}, σ_res={sigma_fit:.6g}"
        )
        x_left = max(1050.0, b + 1e-6)
        if 16383.0 > x_left:
            x_line = np.linspace(x_left, 16383.0, 800)
            y_line = a * np.log(x_line - b) + c * x_line + d / x_line
            if sigma_fit is not None and sigma_fit > 0:
                ax.fill_between(
                    x_line,
                    y_line - 1.0 * sigma_fit,
                    y_line + 1.0 * sigma_fit,
                    color="red",
                    alpha=0.2,
                    zorder=4,
                    label=r"fit $\pm 1\sigma$",
                )
            ax.plot(
                x_line,
                y_line,
                color="crimson",
                lw=2,
                zorder=5,
                label=rf"y=aln(x-b)+cx+d/x",
            )
    else:
        print(
            f"[maxch1-tmax] 拟合跳过：窗口内点不足 4 个或 curve_fit 失败（n={n_fit}）。"
        )

    ax.set_title(
        f"CH1 max vs tmax (basic+act + cut_ch0max_tmax, N={m.size}, {n_files} 文件)",
        fontsize=13,
    )
    ax.set_xlim(0, 16382)
    ax.set_ylim(0, 30000)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
