#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
maxch2-tmax: 对通过 basic+act 六种 cut 且 cut_ch0max_tmax（m7）的事例，
从 CH0-3 原始波形中逐文件计算 CH2 的 max/tmax，并逐步增量绘制散点图。
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

# 与 basic+act.py 测试块一致：直接使用其 cut 与加载函数
cut_ch0_min_positive = basic_act.cut_ch0_min_positive
cut_ch0_max_saturation = basic_act.cut_ch0_max_saturation
cut_ch5_self_trigger = basic_act.cut_ch5_self_trigger
cut_pedestal_3sigma = basic_act.cut_pedestal_3sigma
cut_act = basic_act.cut_act
cut_mincut = basic_act.cut_mincut
cut_ch0max_tmax = basic_act.cut_ch0max_tmax
cut_ch1max_tmax = basic_act.cut_ch1max_tmax
_list_paired_param_files = basic_act._list_paired_param_files
_load_basic_features_for_run = basic_act._load_basic_features_for_run


def _discover_project_root() -> Path:
    here = Path(__file__).resolve()
    python_dir = here.parents[3]
    return python_dir.parent


PROJECT_ROOT = _discover_project_root()
DATA_ROOT = PROJECT_ROOT / "data" / "hdf5" / "raw_pulse"
CH0_3_DIR = DATA_ROOT / "CH0-3"
CH2_INDEX = 2

# 拟合子集：x=max(CH2), y=tmax(CH2)；模型 y = a·ln(x−b) + c·x + d/x（初值可与 CH1 区分后自调）
FIT_X_MIN = 1350.0
FIT_Y_MIN = 11000.0
FIT_Y_MAX = 17500.0


def _model_alog_x_minus_b_cx_d_over_x(
    x: np.ndarray, a: float, b: float, c: float, d: float
) -> np.ndarray:
    """y = a·ln(x−b) + c·x + d/x，要求 x > b。"""
    return a * np.log(x - b) + c * x + d / x


def _fit_alog_x_minus_b_cx_d_over_x(
    x: np.ndarray, y: np.ndarray
) -> Tuple[Optional[np.ndarray], int, Optional[float]]:
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


def _resolve_ch0_3_path(ch0_param_path: Path) -> Optional[Path]:
    candidates = [CH0_3_DIR / ch0_param_path.name, CH0_3_DIR / f"{ch0_param_path.stem}.h5"]
    for path in candidates:
        if path.exists():
            return path
    return None


def main() -> None:
    pairs = _list_paired_param_files()
    n_files = len(pairs)
    print(f"[maxch2-tmax] 找到 {n_files} 个可配对的参数文件，逐文件计算并绘制。")

    all_max: List[np.ndarray] = []
    all_tmax: List[np.ndarray] = []
    n_raw_total = 0
    n_pass_total = 0
    received = 0

    #plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_xlabel("max(CH2) (ADC)", fontsize=12)
    ax.set_ylabel("tmax(CH2) (sample index)", fontsize=12)
    ax.grid(True, alpha=0.3)
    scatter = ax.scatter([], [], s=2, alpha=0.5, edgecolors="none", label="data")
    fig.tight_layout()
    plt.show(block=False)

    for ch0_path, ch5_path in pairs:
        ch0_3_path = _resolve_ch0_3_path(ch0_path)
        if ch0_3_path is None:
            continue

        m0, cmin, m5, ped0, ped1, m1, c1min, m4, t4, tc0, tc1 = _load_basic_features_for_run(
            ch0_path, ch5_path
        )
        n_raw_total += int(m0.shape[0])
        m1_c = cut_ch0_min_positive(cmin)
        m2_c = cut_ch0_max_saturation(m0, m1)
        m3_c = cut_ch5_self_trigger(m5)
        m4_c = cut_pedestal_3sigma(ped0, ped1, m5)
        m5_c = cut_act(m4, t4)
        m6_c = cut_mincut(cmin, c1min, m4, t4)
        base6 = m1_c & m2_c & m3_c & m4_c & m5_c & m6_c
        m7_c = cut_ch0max_tmax(base6, m0, tc0)
        m8_c = cut_ch1max_tmax(base6 & m7_c, m1, tc1)
        passing = np.where(base6 & m7_c & m8_c)[0]
        n_pass_total += int(passing.size)
        if passing.size == 0:
            continue

        with h5py.File(ch0_3_path, "r") as f:
            dset = f["channel_data"]
            n_samples, n_channels, n_events = dset.shape
            if n_channels <= CH2_INDEX:
                continue
            valid = (passing >= 0) & (passing < n_events)
            passing = passing[valid]
            if passing.size == 0:
                continue
            wf = np.asarray(dset[:, CH2_INDEX, passing], dtype=np.float64)
            ch2_max = np.max(wf, axis=0)
            ch2_tmax = np.argmax(wf, axis=0).astype(np.float64)

        all_max.append(ch2_max)
        all_tmax.append(ch2_tmax)
        received += 1
        x_all = np.concatenate(all_max)
        y_all = np.concatenate(all_tmax)
        scatter.set_offsets(np.column_stack([x_all, y_all]))
        ax.set_title(f"CH2 max vs tmax, N={x_all.size}, {received}/{n_files} )", fontsize=13)
        xmin, xmax = float(np.min(x_all)), float(np.max(x_all))
        ymin, ymax = float(np.min(y_all)), float(np.max(y_all))
        pad_x = max((xmax - xmin) * 0.02, 1.0)
        pad_y = max((ymax - ymin) * 0.02, 1.0)
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    print(f"[maxch2-tmax] 原始事件数(累计): {n_raw_total}")
    print(f"[maxch2-tmax] basic+act+m7 通过(累计): {n_pass_total}")
    if not all_max:
        print("[maxch2-tmax] 无可绘制数据，退出。")
        plt.ioff()
        plt.show()
        return

    m = np.concatenate(all_max)
    t = np.concatenate(all_tmax)
    print(f"[maxch2-tmax] 共 {m.size} 个 (max_ch2, tmax_ch2) 点。")

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_xlabel("max(CH2) (ADC)", fontsize=12)
    ax.set_ylabel("tmax(CH2) (sample index)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.scatter(m, t, s=2, alpha=0.5, edgecolors="none", label="data")

    coeffs, n_fit, sigma_fit = _fit_alog_x_minus_b_cx_d_over_x(m, t)
    if coeffs is not None:
        a, b, c, d = float(coeffs[0]), float(coeffs[1]), float(coeffs[2]), float(coeffs[3])
        print(
            f"[maxch2-tmax] 拟合 y=a·ln(x−b)+c·x+d/x（{n_fit} 点，"
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
            f"[maxch2-tmax] 拟合跳过：窗口内点不足 4 个或 curve_fit 失败（n={n_fit}）。"
        )

    ax.set_title(f"CH2 max vs tmax (basic+act+m7, N={m.size}, {received}/{n_files} 文件)", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    plt.ioff()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
