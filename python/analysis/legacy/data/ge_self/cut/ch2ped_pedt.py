#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ch2ped-pedt: 对通过 basic+act 全部 cut 的事例，直接从 CH2_parameters 读取已计算好的 ped/pedt，
逐批绘制每个事件的：
- ch2ped_mean（前 N 点均值，N 由 backfill 写入时决定，当前默认 500）
- ch2pedt_mean（后 N 点均值，N 由 backfill 写入时决定，当前默认 500）

波形位置参考 python/data/preprocessor.py：
raw_pulse/CH0-3/*.h5 中 dataset "channel_data" 形状为 (n_samples, n_channels, n_events)，
其中 CH2 对应 channel index = 2。

本脚本不再读取原始波形计算 ped/pedt，而是依赖：
  data/hdf5/raw_pulse/CH2_parameters/*.h5
中的数据集：
  - ch2ped_mean
  - ch2pedt_mean
其补写逻辑见 python/data/fit_ch2_ch3_parallel.py 的 backfill。
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
CH2_PARAM_DIR = DATA_ROOT / "CH2_parameters"

CH2_PED_MEAN_KEY = "ch2ped_mean"
CH2_PEDT_MEAN_KEY = "ch2pedt_mean"

# 仅用于坐标轴标签展示（实际 N 由数据集的定义决定）
DEFAULT_PED_SAMPLES = 500
# 与 basic_act.cut_ch2ped_pedt_fit_range 同源
ADC_MAX = basic_act.CH2_PED_PEDT_ADC_MAX
INNER_X_MAX = basic_act.CH2_PED_PEDT_INNER_X_MAX

CH0_3_DIR = DATA_ROOT / "CH0-3"


def _ch2ped_pedt_sigmas_like_basic_act(
    ped: np.ndarray,
    pedt: np.ndarray,
) -> tuple[Optional[float], Optional[float], Optional[float], np.ndarray]:
    """
    与 basic_act.cut_ch2ped_pedt_fit_range（283–327 行）相同的 a、σ_ped、σ(y-x) 估计。
    返回 (a, sig_x, sig_yx, fin)。
    """
    x_min = 0.0
    x_max = float(basic_act.CH2_PED_PEDT_INNER_X_MAX)
    y_min = 0.0
    y_max = float(basic_act.CH2_PED_PEDT_ADC_MAX)
    x = np.asarray(ped, dtype=np.float64)
    y = np.asarray(pedt, dtype=np.float64)
    n = min(x.shape[0], y.shape[0])
    x = x[:n]
    y = y[:n]
    fin = np.isfinite(x) & np.isfinite(y)
    fit_m = fin & (x > x_min) & (x < x_max) & (y > y_min) & (y < y_max)
    a: Optional[float] = None
    sig_x: Optional[float] = None
    if int(np.count_nonzero(fit_m)) >= 2:
        xv = x[fit_m]
        a = float(np.mean(xv))
        sig_x = float(np.std(xv, ddof=1))
    sig_yx: Optional[float] = None
    if int(np.count_nonzero(fin)) >= 2:
        sig_yx = float(np.std(y[fin] - x[fin], ddof=1))
    return a, sig_x, sig_yx, fin


def _resolve_source_file_from_param(ch2_param_path: Path) -> str:
    try:
        with h5py.File(ch2_param_path, "r") as f:
            src = f.attrs.get("source_file", None)
            if src is None:
                raise KeyError("missing source_file")
            try:
                return src.decode("utf-8") if isinstance(src, (bytes, np.bytes_)) else str(src)
            except Exception:
                return str(src)
    except Exception:
        # fallback: CH0-3/<same name>
        return str((CH0_3_DIR / ch2_param_path.name).resolve())


def _extract_waveform_from_channel_data(
    ch_data: h5py.Dataset,
    event_idx: int,
    channel_idx: int,
) -> np.ndarray | None:
    if ch_data.ndim != 3:
        return None
    max_reasonable_waveform_len = 200000
    candidates = (
        lambda: ch_data[:, channel_idx, event_idx],  # (time, channel, event)
        lambda: ch_data[event_idx, channel_idx, :],  # (event, channel, time)
        lambda: ch_data[event_idx, :, channel_idx],  # (event, time, channel)
        lambda: ch_data[:, event_idx, channel_idx],  # (time, event, channel)
    )
    for getter in candidates:
        try:
            wf = np.asarray(getter(), dtype=np.float64)
        except Exception:
            continue
        if wf.ndim != 1 or wf.size == 0 or wf.size > max_reasonable_waveform_len:
            continue
        return wf
    return None


def _extract_ch0_ch3_waveforms_fast(
    ch_data: h5py.Dataset,
    event_idx: int,
    ch0_index: int = 0,
    ch3_index: int = 3,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if ch_data.ndim == 3:
        try:
            if ch_data.shape[1] > max(ch0_index, ch3_index):
                wf0 = np.asarray(ch_data[:, ch0_index, event_idx], dtype=np.float64)
                wf3 = np.asarray(ch_data[:, ch3_index, event_idx], dtype=np.float64)
                if wf0.ndim == 1 and wf3.ndim == 1 and wf0.size > 0 and wf3.size > 0:
                    return wf0, wf3
        except Exception:
            pass
    wf0 = _extract_waveform_from_channel_data(ch_data, event_idx, ch0_index)
    wf3 = _extract_waveform_from_channel_data(ch_data, event_idx, ch3_index)
    return wf0, wf3


def _load_passing_events_per_file() -> List[Tuple[Path, np.ndarray]]:
    pairs = basic_act._list_paired_param_files()
    print(f"[ch2ped-pedt] 找到 {len(pairs)} 个可配对的参数文件。")

    result: List[Tuple[Path, np.ndarray]] = []
    for ch0_path, ch5_path in pairs:
        ch2_param_path = CH2_PARAM_DIR / ch0_path.name
        if not ch2_param_path.exists():
            continue

        # basic_act._load_basic_features_for_run 的返回会随版本扩展，这里只取 cut 所需的前 9 项
        m0, cmin, m5, ped0, ped1, m1, c1min, m4, t4, *_ = basic_act._load_basic_features_for_run(
            ch0_path, ch5_path
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
            continue
        result.append((ch2_param_path, passing.astype(np.int64)))
    return result


def _phase2_worker(
    args: Tuple[Path, np.ndarray],
) -> Optional[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """单文件：按 passing 下标读取 ch2ped_mean、ch2pedt_mean，返回 (source_file, event_idx, ped, pedt)。"""
    ch2_param_path, passing = args
    passing = np.asarray(passing, dtype=np.int64)
    try:
        with h5py.File(ch2_param_path, "r") as f:
            if CH2_PED_MEAN_KEY not in f or CH2_PEDT_MEAN_KEY not in f:
                return None
            dset_ped = f[CH2_PED_MEAN_KEY]
            dset_pedt = f[CH2_PEDT_MEAN_KEY]
            src = f.attrs.get("source_file", None)
            try:
                source_file = src.decode("utf-8") if isinstance(src, (bytes, np.bytes_)) else str(src)
            except Exception:
                source_file = str(src) if src is not None else str((CH0_3_DIR / ch2_param_path.name).resolve())
            n_ev = int(min(dset_ped.shape[0], dset_pedt.shape[0]))
            if n_ev <= 0:
                return None
            valid = (passing >= 0) & (passing < n_ev)
            passing = passing[valid]
            if passing.size == 0:
                return None
            ped = np.asarray(dset_ped[passing], dtype=np.float64)
            pedt = np.asarray(dset_pedt[passing], dtype=np.float64)
        return (source_file, passing, ped, pedt)
    except Exception:
        return None


def main() -> None:
    file_tasks = _load_passing_events_per_file()
    if not file_tasks:
        print("[ch2ped-pedt] 没有通过 cut 的事例可处理，退出。")
        return

    n_total = int(sum(len(idx) for _, idx in file_tasks))
    print(f"[ch2ped-pedt] 共 {n_total} 个通过 cut 的事例，分布在 {len(file_tasks)} 个文件中。")
    workers = max(1, os.cpu_count() or 1)
    print(f"[ch2ped-pedt] 批量读取 CH2_parameters（仅通过 cut 的 event），进程数: {workers}")

    all_ped: List[np.ndarray] = []
    all_pedt: List[np.ndarray] = []
    all_event_idx: List[np.ndarray] = []
    all_source_file: List[np.ndarray] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_phase2_worker, (p, idx)): p for p, idx in file_tasks}
        done = 0
        for fut in as_completed(futures):
            out = fut.result()
            done += 1
            if out is None:
                continue
            source_file, ev_idx, ped_mean, pedt_mean = out
            if ped_mean.size == 0:
                continue
            all_ped.append(ped_mean)
            all_pedt.append(pedt_mean)
            all_event_idx.append(ev_idx)
            all_source_file.append(np.full(ev_idx.shape[0], source_file, dtype=object))
            if done % 20 == 0 or done == len(futures):
                n_acc = int(sum(x.size for x in all_ped))
                print(f"[ch2ped-pedt] 进度: {done}/{len(futures)} 文件, 已累计事件 {n_acc}")

    if not all_ped:
        print(
            "[ch2ped-pedt] 未读取到任何 CH2 ped/pedt 数据。"
            f"请确认 CH2_parameters 中存在 {CH2_PED_MEAN_KEY}/{CH2_PEDT_MEAN_KEY}。"
        )
        return

    x_vals = np.concatenate(all_ped)
    y_vals = np.concatenate(all_pedt)
    event_idx_vals = np.concatenate(all_event_idx)
    source_file_vals = np.concatenate(all_source_file).astype(object)
    print(f"[ch2ped-pedt] 完成：共绘制 {x_vals.size} 个点。")

    a_fit, sigma_fit, sigma_yx, fin_xy = _ch2ped_pedt_sigmas_like_basic_act(x_vals, y_vals)
    fit_mask = (
        np.isfinite(x_vals)
        & np.isfinite(y_vals)
        & (x_vals > 0.0)
        & (x_vals < INNER_X_MAX)
        & (y_vals > 0.0)
        & (y_vals < ADC_MAX)
    )
    n_viz = int(np.count_nonzero(fit_mask))
    n_fin = int(np.count_nonzero(fin_xy))
    print(
        f"[ch2ped-pedt] 内区 N={n_viz}；有限点 N={n_fin}；"
        f" a={a_fit}, σ_ped={sigma_fit}, σ(y-x)={sigma_yx}"
    )

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.scatter(x_vals, y_vals, s=2, alpha=0.5, edgecolors="none")

    # 竖线 / ±3σ：与 basic_act 中 mask_x3 几何一致；y 画在 [0, ADC_MAX]
    y_fit_lo, y_fit_hi = 0.0, ADC_MAX
    if a_fit is not None:
        if sigma_fit is not None and np.isfinite(sigma_fit) and sigma_fit > 0.0:
            left = a_fit - 3.0 * sigma_fit
            right = a_fit + 3.0 * sigma_fit
            ax.fill_betweenx(
                np.array([y_fit_lo, y_fit_hi]),
                left,
                right,
                color="C3",
                alpha=0.12,
                zorder=0,
                label=rf"fit ± 3σ (0<x<{INNER_X_MAX:g}, 0<y<{ADC_MAX:g})",
            )
        ax.plot(
            [a_fit, a_fit],
            [y_fit_lo, y_fit_hi],
            color="C3",
            linewidth=1.5,
            zorder=3,
            label=rf"fit: x={a_fit:.3g} (0<x<{INNER_X_MAX:g}, 0<y<{ADC_MAX:g})",
        )
    # y=x（绿）及 ±1σ：在 0<ped<ADC_MAX 且 0<pedt<ADC_MAX 内用着色条带（同红色 3σ 风格），不用虚线
    did_plot_yx = False
    if int(np.count_nonzero(fin_xy)) >= 1:
        xf = x_vals[fin_xy]
        xmin_f = float(np.min(xf))
        xmax_f = float(np.max(xf))
        if xmax_f > xmin_f:
            xs_line = np.linspace(0.0, ADC_MAX, 500)
            if sigma_yx is not None and sigma_yx > 0.0:
                x_band = np.linspace(0.0, ADC_MAX, 4000)
                y_lo = np.maximum(x_band - sigma_yx, 0.0)
                y_hi = np.minimum(x_band + sigma_yx, ADC_MAX)
                ax.fill_between(
                    x_band,
                    y_lo,
                    y_hi,
                    where=y_lo < y_hi,
                    color="green",
                    alpha=0.12,
                    zorder=0,
                    label=(
                        f"y=x±1σ (σ={sigma_yx:.4g}, "
                        f"0<ped,pedt<{ADC_MAX:g})"
                    ),
                )
            ax.plot(
                xs_line,
                xs_line,
                color="green",
                linewidth=1.5,
                zorder=4,
                label="y=x",
            )
            did_plot_yx = True
    if a_fit is not None or sigma_yx is not None or did_plot_yx:
        ax.legend(loc="best", fontsize=10)
    ax.set_xlabel(f"CH2 ped mean (~first {DEFAULT_PED_SAMPLES} samples) (ADC)", fontsize=12)
    ax.set_ylabel(f"CH2 pedt mean (~last {DEFAULT_PED_SAMPLES} samples) (ADC)", fontsize=12)
    title_parts = [f"N={x_vals.size}"]
    if a_fit is not None:
        title_parts.append(f"inner x={a_fit:.6g}")
    if sigma_yx is not None:
        title_parts.append(f"y=x σ(y-x)={sigma_yx:.6g}")
    ax.set_title(
        "CH2 ped vs pedt mean (basic+act cuts passed) | " + " | ".join(title_parts),
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

