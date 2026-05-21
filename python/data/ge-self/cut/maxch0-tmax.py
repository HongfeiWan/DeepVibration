#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
maxch0-ch0pedt: 对通过 basic+act 全部 cut 的事例（与 ch3max-q 相同筛选），
从 CH0_parameters 读取 max_ch0 与 tmax_ch0，绘制二维散点图。

HDF5 结构见 preprocessor.py：datasets max_ch0, tmax_ch0（及其它 CH0 特征）。
"""

from __future__ import annotations

import importlib.util
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _discover_project_root() -> Path:
    here = Path(__file__).resolve()
    python_dir = here.parents[3]
    return python_dir.parent


PROJECT_ROOT = _discover_project_root()
DATA_ROOT = PROJECT_ROOT / "data" / "hdf5" / "raw_pulse"
CH0_PARAM_DIR = DATA_ROOT / "CH0_parameters"

N_WORKERS = max(1, os.cpu_count() or 1)

# 拟合子集：x=max(CH0), y=tmax(CH0)；模型 y = a·ln(x−b) + c·x + d/x
FIT_X_MIN = 1160.0
FIT_Y_MIN = 10200.0
FIT_Y_MAX = 13500.0


def _get_first_existing_dataset(
    f: h5py.File,
    keys: List[str],
    *,
    groups: Optional[List[str]] = None,
) -> h5py.Dataset:
    """
    从 HDF5 文件中按候选路径顺序查找 dataset。

    - keys：候选 dataset 名（按顺序尝试）
    - groups：候选 group 路径前缀（默认为只在根目录查找）
      例如 groups=["", "features", "CH0"]，会尝试 "max_ch0" / "features/max_ch0" / "CH0/max_ch0"
    """
    groups = groups or [""]
    for g in groups:
        prefix = (g.strip("/") + "/") if g else ""
        for k in keys:
            name = f"{prefix}{k}"
            if name in f:
                obj = f[name]
                if isinstance(obj, h5py.Dataset):
                    return obj

    tried: List[str] = []
    for g in groups:
        prefix = (g.strip("/") + "/") if g else ""
        for k in keys:
            tried.append(f"{prefix}{k}")
    raise KeyError(f"未找到任何候选数据集: {tried}")


def _model_alog_x_minus_b_cx_d_over_x(
    x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """y = a·ln(x−b) + c·x + d/x，要求 x > b。"""
    return a * np.log(x - b) + c * x + d / x

def _fit_alog_x_minus_b_cx_d_over_x(
    x: np.ndarray, y: np.ndarray) -> Tuple[Optional[np.ndarray], int, Optional[float]]:
    """
    在 FIT_* 窗口内对 y = a·ln(x−b) + c·x + d/x 做非线性最小二乘（curve_fit）。
    返回 (coeffs [a,b,c,d], n_points, 残差标准差 σ)，失败时返回 (None, n, None)。
    需对所有数据点满足 x > b，故 b 上界为 min(x) − ε。
    """
    mask = (x > FIT_X_MIN) & (y > FIT_Y_MIN) & (y < FIT_Y_MAX)
    xf = np.asarray(x[mask], dtype=np.float64)
    yf = np.asarray(y[mask], dtype=np.float64)
    n = int(xf.size)
    if n < 4:
        return None, n, None
    xmin = float(np.min(xf))
    # b 必须严格小于 min(x)，否则存在 x−b≤0
    b_hi = xmin - 1e-9
    # 初值：先用无平移的线性基 [ln x, x, 1/x] 得到 a,c,d，b 取在可行域内
    A0 = np.column_stack([np.log(xf), xf, 1.0 / xf])
    coef0, _, rank0, _ = np.linalg.lstsq(A0, yf, rcond=None)
    if rank0 < 3:
        return None, n, None
    a0, c0, d0 = float(coef0[0]), float(coef0[1]), float(coef0[2])
    b0 = min(xmin * 0.5, xmin - 200.0)  # 保证 b0 < xmin
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

def _phase1_worker(args: Tuple[Path, Path]) -> Optional[Tuple[Path, np.ndarray]]:
    """单文件对：加载 CH0/CH5/CH1/CH4，应用 basic+act cut，返回 (ch0_path, passing)。"""
    ch0_path, ch5_path = args
    # basic_act._load_basic_features_for_run()（最新）返回 11 个数组：
    # max_ch0, ch0_min, max_ch5, ch0_ped_mean, ch1_ped_mean,
    # max_ch1, ch1_min, max_ch4, tmax_ch4, tmax_ch0, tmax_ch1
    m0, cmin, m5, ped0, ped1, m1, c1min, m4, t4, _tmax0, _tmax1 = (
        basic_act._load_basic_features_for_run(ch0_path, ch5_path)
    )
    m1_c = basic_act.cut_ch0_min_positive(cmin)
    m2_c = basic_act.cut_ch0_max_saturation(m0, m1)
    m3_c = basic_act.cut_ch5_self_trigger(m5)
    m4_c = basic_act.cut_pedestal_3sigma(ped0, ped1, m5)
    m5_c = basic_act.cut_act(m4, t4)
    m6_c = basic_act.cut_mincut(cmin, c1min, m4, t4)
    mask = m1_c & m2_c & m3_c & m4_c & m5_c & m6_c
    passing = np.where(mask)[0]
    if passing.size == 0:
        return None
    return (ch0_path, passing)

def _phase2_worker(
    args: Tuple[Path, np.ndarray],) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """单文件：按 passing 下标读取 max_ch0、tmax_ch0，返回 (max_arr, tmax_arr)。"""
    ch0_path, passing = args
    passing = np.asarray(passing, dtype=np.int64)
    with h5py.File(ch0_path, "r") as f:
        try:
            # CH0_parameters（preprocessor.py）最新默认写入：max_ch0 / tmax_ch0
            # 同时兼容：ch0_max_ch0 / ch0_tmax_ch0，或放在 features/CH0/ch0 等 group 下
            dset_max = _get_first_existing_dataset(
                f,
                keys=["max_ch0", "ch0_max_ch0"],
                groups=["", "features", "CH0", "ch0"],
            )
            dset_tmax = _get_first_existing_dataset(
                f,
                keys=["tmax_ch0", "ch0_tmax_ch0"],
                groups=["", "features", "CH0", "ch0"],
            )
        except KeyError:
            return None
        n_ev = int(dset_max.shape[0])
        valid = (passing >= 0) & (passing < n_ev)
        passing = passing[valid]
        if passing.size == 0:
            return None
        max_arr = np.asarray(dset_max[passing], dtype=np.float64)
        # tmax 与 max 通常同长度；若数据集长度略有差异，读取阶段仍按 passing 过滤即可
        tmax_arr = np.asarray(dset_tmax[passing], dtype=np.float64)
    return (max_arr, tmax_arr)

def _load_passing_events_per_file() -> List[Tuple[Path, np.ndarray]]:
    """与 ch3max-q 相同的 basic+act 全 cut，多进程返回 (CH0 参数文件路径, 通过事例下标)。"""
    pairs = basic_act._list_paired_param_files()
    print(f"[maxch0-tmax] 找到 {len(pairs)} 个可配对的参数文件，阶段1 使用 {N_WORKERS} 进程。")

    result: List[Tuple[Path, np.ndarray]] = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_phase1_worker, (ch0_path, ch5_path)): ch0_path for ch0_path, ch5_path in pairs}
        for fut in as_completed(futures):
            out = fut.result()
            if out is not None:
                result.append(out)
    # 按 ch0_path 排序，保持与原始顺序一致
    result.sort(key=lambda x: str(x[0]))
    return result

def main() -> None:
    file_tasks = _load_passing_events_per_file()
    if not file_tasks:
        print("[maxch0-tmax] 没有通过 cut 的事例可处理，退出。")
        return

    n_total = sum(len(idx) for _, idx in file_tasks)
    n_files = len(file_tasks)
    print(f"[maxch0-tmax] 共 {n_total} 个通过 cut 的事例，分布在 {n_files} 个文件中。")
    print(f"[maxch0-tmax] 阶段2 使用 {N_WORKERS} 进程批量读取 HDF5（仅通过 cut 的 event）...")

    all_max: List[np.ndarray] = []
    all_tmax: List[np.ndarray] = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_phase2_worker, (ch0_path, passing)): ch0_path for ch0_path, passing in file_tasks}
        for fut in as_completed(futures):
            out = fut.result()
            if out is not None:
                max_arr, tmax_arr = out
                all_max.append(max_arr)
                all_tmax.append(tmax_arr)
    received = len(all_max)
    if received < n_files:
        print(f"[maxch0-tmax] 阶段2 跳过 {n_files - received} 个文件（缺少 max_ch0/tmax_ch0）。")

    if not all_max:
        print("[maxch0-tmax] 未读取到任何数据，退出。")
        return

    m = np.concatenate(all_max)
    t = np.concatenate(all_tmax)
    print(f"[maxch0-tmax] 共 {m.size} 个 (max_ch0, tmax_ch0) 点，一次绘制。")

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_xlabel("max(CH0) (ADC)", fontsize=12)
    ax.set_ylabel("tmax(CH0) (sample index)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.scatter(m, t, s=2, alpha=0.5, edgecolors="none", label="data")

    coeffs, n_fit, sigma_fit = _fit_alog_x_minus_b_cx_d_over_x(m, t)
    if coeffs is not None:
        a, b, c, d = float(coeffs[0]), float(coeffs[1]), float(coeffs[2]), float(coeffs[3])
        print(
            f"[maxch0-tmax] 拟合 y=a·ln(x−b)+c·x+d/x（{n_fit} 点，"
            f"x>{FIT_X_MIN}, {FIT_Y_MIN}<y<{FIT_Y_MAX}）: "
            f"a={a:.6g}, b={b:.6g}, c={c:.6g}, d={d:.6g}, σ_res={sigma_fit:.6g}"
        )
        # 曲线定义域：x > b；横轴与原先一致画到 16383
        x_left = max(1050.0, b + 1e-6)
        if 16383.0 > x_left:
            x_line = np.linspace(x_left, 16383.0, 800)
            y_line = a * np.log(x_line - b) + c * x_line + d / x_line
            if sigma_fit is not None and sigma_fit > 0:
                ax.fill_between(
                    x_line,
                    y_line - 3.0 * sigma_fit,
                    y_line + 3.0 * sigma_fit,
                    color="red",
                    alpha=0.2,
                    zorder=4,
                    label=r"fit $\pm 3\sigma$",
                )
            ax.plot(
                x_line,
                y_line,
                color="crimson",
                lw=2,
                zorder=5,
                label=rf"y=aln(x-b)+cx+d/x"
            )
    else:
        print(
            f"[maxch0-tmax] 拟合跳过：窗口内点不足 4 个或 curve_fit 失败（n={n_fit}）。"
        )

    ax.set_title(
        f"CH0 max vs tmax (basic+act cuts, N={m.size}, {received}/{n_files} 文件)",
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
