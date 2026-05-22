#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ch3ped-min: 对通过 basic+acv 全部 cut 的事例，从 ch3_parameters 读取：
- ch3ped_mean（前 N 点均值）
- min_ch3（通道 3 波形最小值）
绘制 (ch3ped_mean, min_ch3) 的二维分布。

其中 min_ch3 的补写逻辑见 python/data/fit_ch2_ch3_parallel.py 的 _backfill_extrema_for_param_dir。
"""

from __future__ import annotations

import importlib.util
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import re
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np


# 加载 basic+acv 模块（文件名含 + 无法直接 import）
_BASIC_ACV_PATH = Path(__file__).resolve().parent / "basic+acv.py"
_spec = importlib.util.spec_from_file_location("basic_acv", _BASIC_ACV_PATH)
basic_acv = importlib.util.module_from_spec(_spec)
sys.modules["basic_acv"] = basic_acv
_spec.loader.exec_module(basic_acv)


def _discover_project_root() -> Path:
    here = Path(__file__).resolve()
    python_dir = here.parents[3]
    return python_dir.parent


PROJECT_ROOT = _discover_project_root()
DATA_ROOT = PROJECT_ROOT / "data" / "hdf5" / "raw_pulse"
CH3_PARAM_DIR = DATA_ROOT / "CH3_parameters"
CH0_3_DIR = DATA_ROOT / "CH0-3"

CH3_PED_MEAN_KEY = "ch3ped_mean"
# extrema backfill: min_key = f"min_ch{channel_idx}" → channel_idx=3
CH3_MIN_KEY = "min_ch3"

DEFAULT_PED_RANGE = (1160.0, 1200.0)
DEFAULT_MIN_RANGE = (1140.0, 1180.0)
# 红区：全体有限点 x 的 mean、σ_x，竖直带 [mean±h·σ_x]（与 tradition.cut_ch3ped_min 一致）
CH3PED_X_MEAN_BAND_HALF_SIGMA = 0.5
# 绿区：|y-x|≤sigma_band 上拟合 y=x+b 后 ±n·σ_res（与 tradition.cut_ch3ped_min 一致）
CH3PED_RESIDUAL_N_SIGMA = 6.0

# 与 parameterize/tradition/tradition.py 中绘图一致
_PLOT_TICK = 12
_PLOT_AXIS = 16
_PLOT_TITLE = 18
_PLOT_LEGEND = 12
_PLOT_SUBPLOT_TITLE = 14


def _apply_plotstyle_font() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
        }
    )

def _resolve_source_file_from_param(ch3_param_path: Path) -> str:
    try:
        with h5py.File(ch3_param_path, "r") as f:
            src = f.attrs.get("source_file", None)
            if src is None:
                raise KeyError("missing source_file")
            try:
                return src.decode("utf-8") if isinstance(src, (bytes, np.bytes_)) else str(src)
            except Exception:
                return str(src)
    except Exception:
        return str((CH0_3_DIR / ch3_param_path.name).resolve())

def _extract_waveform_from_channel_data(
    ch_data: h5py.Dataset,
    event_idx: int,
    channel_idx: int,) -> np.ndarray | None:
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
    ch3_index: int = 3,) -> tuple[np.ndarray | None, np.ndarray | None]:
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

def _plot_ch0_ch3_waveforms_9(
    sources: np.ndarray,
    event_indices: np.ndarray,
    *,
    title: str,
    sampling_interval_ns: float = 4.0,) -> None:
    n = int(min(9, sources.shape[0], event_indices.shape[0]))
    if n <= 0:
        print("[CH3ped-min] 未找到可绘制的事件。")
        return

    _apply_plotstyle_font()
    plt.rcParams.setdefault("axes.unicode_minus", False)
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.ravel()
    sampling_interval_ns = float(sampling_interval_ns)

    for i in range(9):
        ax = axes[i]
        if i >= n:
            ax.set_visible(False)
            continue
        sf = str(sources[i])
        ev = int(event_indices[i])
        p = Path(sf)
        if not p.is_absolute():
            p = (CH0_3_DIR / p).resolve()
        if not p.exists():
            ax.set_visible(False)
            continue

        try:
            with h5py.File(p, "r") as f:
                if "channel_data" not in f:
                    ax.set_visible(False)
                    continue
                ch_data = f["channel_data"]
                wf0, wf3 = _extract_ch0_ch3_waveforms_fast(ch_data=ch_data, event_idx=ev)
                if wf0 is None or wf3 is None:
                    ax.set_visible(False)
                    continue
        except Exception:
            ax.set_visible(False)
            continue

        time_us = np.arange(wf0.size) * sampling_interval_ns / 1000.0
        ax_r = ax.twinx()
        ax.plot(time_us, wf0, color="C0", linewidth=0.8, label="CH0")
        ax_r.plot(time_us, wf3, color="C3", linewidth=0.8, label="CH3")

        stem = p.stem
        m = re.search(r"(\d+)(?!.*\d)", stem)
        run_id = m.group(1) if m else stem
        ax.set_title(f"{run_id} | #{ev}", fontsize=_PLOT_SUBPLOT_TITLE)
        ax.set_xlabel("Time (us)", fontsize=_PLOT_AXIS)
        ax.set_ylabel("CH0 ADC", fontsize=_PLOT_AXIS)
        ax_r.set_ylabel("CH3 ADC", fontsize=_PLOT_AXIS)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", which="major", labelsize=_PLOT_TICK)
        ax_r.tick_params(axis="both", which="major", labelsize=_PLOT_TICK)

        lines_l, labels_l = ax.get_legend_handles_labels()
        lines_r, labels_r = ax_r.get_legend_handles_labels()
        ax.legend(lines_l + lines_r, labels_l + labels_r, fontsize=_PLOT_LEGEND, loc="best")

    fig.suptitle(title, fontsize=_PLOT_TITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def _load_passing_events_per_file() -> List[Tuple[Path, np.ndarray]]:
    """
    与 CH3ped-pedt.py 同源：对每个 run 用 basic+acv 的 cut 得到通过的 event indices，
    返回 (CH3_param_path, passing_indices) 列表。
    """
    pairs = basic_acv._list_paired_param_files()
    print(f"[CH3ped-min] 找到 {len(pairs)} 个可配对的参数文件。")

    result: List[Tuple[Path, np.ndarray]] = []
    for ch0_path, ch5_path in pairs:
        CH3_param_path = CH3_PARAM_DIR / ch0_path.name
        if not CH3_param_path.exists():
            continue

        # basic_acv._load_basic_features_for_run 的返回会随版本扩展，这里只取 cut 所需的前 9 项
        m0, cmin, m5, ped0, ped1, m1, c1min, m4, t4, *_ = basic_acv._load_basic_features_for_run(
            ch0_path, ch5_path
        )
        m1_c = basic_acv.cut_ch0_min_positive(cmin)
        m2_c = basic_acv.cut_ch0_max_saturation(m0, m1)
        m3_c = basic_acv.cut_ch5_self_trigger(m5)
        m4_c = basic_acv.cut_pedestal_3sigma(ped0, ped1, m5)
        m5_c = basic_acv.cut_acv(m4, t4)
        m6_c = basic_acv.cut_mincut(cmin, c1min, m4, t4)
        mask = m1_c & m2_c & m3_c & m4_c & m5_c & m6_c
        passing = np.where(mask)[0]
        if passing.size == 0:
            continue
        result.append((CH3_param_path, passing.astype(np.int64)))
    return result

def _phase2_worker(
    args: Tuple[Path, np.ndarray],) -> Optional[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """单文件：按 passing 下标读取 ch3ped_mean/min_ch3，返回 (source_file, event_idx, ped, ch3min)。"""
    CH3_param_path, passing = args
    passing = np.asarray(passing, dtype=np.int64)
    try:
        with h5py.File(CH3_param_path, "r") as f:
            if CH3_PED_MEAN_KEY not in f or CH3_MIN_KEY not in f:
                return None
            dset_ped = f[CH3_PED_MEAN_KEY]
            dset_min = f[CH3_MIN_KEY]
            src = f.attrs.get("source_file", None)
            try:
                source_file = src.decode("utf-8") if isinstance(src, (bytes, np.bytes_)) else str(src)
            except Exception:
                source_file = str(src) if src is not None else _resolve_source_file_from_param(CH3_param_path)
            n_ev = int(min(dset_ped.shape[0], dset_min.shape[0]))
            if n_ev <= 0:
                return None
            valid = (passing >= 0) & (passing < n_ev)
            passing = passing[valid]
            if passing.size == 0:
                return None
            ped = np.asarray(dset_ped[passing], dtype=np.float64)
            ch3min = np.asarray(dset_min[passing], dtype=np.float64)
        return (source_file, passing, ped, ch3min)
    except Exception:
        return None

def main() -> None:
    file_tasks = _load_passing_events_per_file()
    if not file_tasks:
        print("[CH3ped-min] 没有通过 cut 的事例可处理，退出。")
        return

    n_total = int(sum(len(idx) for _, idx in file_tasks))
    print(f"[CH3ped-min] 共 {n_total} 个通过 cut 的事例，分布在 {len(file_tasks)} 个文件中。")

    workers = max(1, os.cpu_count() or 1)
    print(f"[CH3ped-min] 批量读取 CH3_parameters（仅通过 cut 的 event），进程数: {workers}")

    all_ped: List[np.ndarray] = []
    all_min: List[np.ndarray] = []
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
            source_file, ev_idx, ped, ch3min = out
            if ped.size == 0:
                continue
            all_ped.append(ped)
            all_min.append(ch3min)
            all_event_idx.append(ev_idx)
            all_source_file.append(np.full(ev_idx.shape[0], source_file, dtype=object))
            if done % 20 == 0 or done == len(futures):
                n_acc = int(sum(x.size for x in all_ped))
                print(f"[CH3ped-min] 进度: {done}/{len(futures)} 文件, 已累计事件 {n_acc}")

    if not all_ped:
        print(
            "[CH3ped-min] 未读取到任何 CH3 ped/min 数据。"
            f"请确认 CH3_parameters 中存在 {CH3_PED_MEAN_KEY}/{CH3_MIN_KEY}。"
        )
        return

    x_vals = np.concatenate(all_ped)
    y_vals = np.concatenate(all_min)
    event_idx_vals = np.concatenate(all_event_idx)
    source_file_vals = np.concatenate(all_source_file).astype(object)
    fin = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals = x_vals[fin]
    y_vals = y_vals[fin]
    event_idx_vals = event_idx_vals[fin]
    source_file_vals = source_file_vals[fin]
    print(f"[CH3ped-min] 完成：共 {x_vals.size} 个有限值点。")

    _apply_plotstyle_font()
    plt.rcParams.setdefault("axes.unicode_minus", False)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.scatter(
        x_vals,
        y_vals,
        s=4,
        alpha=0.5,
        edgecolors="none",
        c="black",
        zorder=1,
        rasterized=True,
    )

    # 叠加 x 均值 ±3σ 红色区域
    x_mean_all = float(np.mean(x_vals))
    sigma_x = (
        float(np.std(x_vals, ddof=1)) if x_vals.size >= 2 else 0.0
    )
    if np.isfinite(sigma_x) and sigma_x > 0.0:
        h = CH3PED_X_MEAN_BAND_HALF_SIGMA * sigma_x
        ax.axvspan(
            x_mean_all - h,
            x_mean_all + h,
            color="red",
            alpha=0.08,
            zorder=0,
            label=rf"$x$: $\bar{{x}}\pm 1\sigma_x$ ($\sigma_x={sigma_x:.4g}$)",
        )
    ax.axvline(
        x_mean_all,
        color="red",
        linestyle="--",
        linewidth=1.2,
        zorder=2,
        label=rf"$\bar{{x}}={x_mean_all:.4g}$",
    )

    # 在 |y-x|≤sigma_band 内拟合 y=x+b；叠加 y=x+b 与残差 1σ 绿色带
    sigma_band = 20.0
    in_band = np.abs(y_vals - x_vals) <= sigma_band
    n_fit = int(np.sum(in_band))
    lo = float(min(x_vals.min(), y_vals.min()))
    hi = float(max(x_vals.max(), y_vals.max()))
    xs = np.linspace(lo, hi, 600)

    if n_fit >= 2:
        xf = x_vals[in_band]
        yf = y_vals[in_band]
        b_fit = float(np.mean(yf - xf))
        resid = yf - xf - b_fit
        sigma_1 = float(np.std(resid, ddof=1))
        if not np.isfinite(sigma_1) or sigma_1 < 0.0:
            sigma_1 = 0.0
        y_line = xs + b_fit
        n_res = float(CH3PED_RESIDUAL_N_SIGMA)
        ax.fill_between(
            xs,
            y_line - n_res * sigma_1,
            y_line + n_res * sigma_1,
            color="green",
            alpha=0.12,
            zorder=0,
            label=rf"±{n_res:g}σ ({n_res:g}σ={n_res * sigma_1:.4g})",
        )
        ax.plot(
            xs,
            y_line,
            color="green",
            linewidth=1.5,
            zorder=3,
            label=rf"$y=x+b$ ($b={b_fit:.4g}$)",
        )
        print(
            f"[CH3ped-min] y=x+b：|y-x|≤{sigma_band} 内 N={n_fit}, b={b_fit:.6g}, 残差σ={sigma_1:.6g}"
        )

    else:
        print(
            f"[CH3ped-min] |y-x|≤{sigma_band} 内有效点不足（{n_fit}），跳过 y=x+b 拟合。"
        )


    ax.set_xlabel("CH3 ped mean (ADC)", fontsize=_PLOT_AXIS)
    ax.set_ylabel("CH3 min (ADC)", fontsize=_PLOT_AXIS)
    ax.tick_params(axis="both", which="major", labelsize=_PLOT_TICK)
    ax.set_xlim(0,3000)
    ax.set_ylim(0,1750)
    ax.legend(loc="best", fontsize=_PLOT_LEGEND)
    fig.tight_layout()
    plt.show()

    # 从指定窗口内挑 9 个事件，绘制 CH0/CH3 波形九宫格
    # x0, x1 = DEFAULT_PED_RANGE
    # y0, y1 = DEFAULT_MIN_RANGE
    # in_box = (x_vals > x0) & (x_vals < x1) & (y_vals > y0) & (y_vals < y1)
    # idxs = np.where(in_box)[0]
    # print(f"[CH3ped-min] 窗口 ped∈({x0:g},{x1:g}) 且 min∈({y0:g},{y1:g}) 命中点数: {idxs.size}")
    # if idxs.size > 0:
    #     pick_n = min(9, int(idxs.size))
    #     rng = np.random.default_rng(42)
    #     picked = rng.choice(idxs, size=pick_n, replace=False) if idxs.size > pick_n else idxs
    #     _plot_ch0_ch3_waveforms_9(
    #         sources=source_file_vals[picked],
    #         event_indices=event_idx_vals[picked],
    #         title=f"CH3 ped∈({x0:g},{x1:g}), CH3 min∈({y0:g},{y1:g}) | shown {pick_n}",
    #     )


if __name__ == "__main__":
    main()

