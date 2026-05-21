#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ch2ped-min: 对通过 basic+act 全部 cut 的事例，从 CH2_parameters 读取：
- ch2ped_mean（前 N 点均值）
- min_ch2（通道 2 波形最小值）
绘制 (ped_mean, min_ch2) 的二维分布。
其中 min_ch2 的补写逻辑见 python/data/fit_ch2_ch3_parallel.py 的 _backfill_extrema_for_param_dir。
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
# extrema backfill: min_key = f"min_ch{channel_idx}" → channel_idx=2
CH2_MIN_KEY = "min_ch2"


def _load_passing_events_per_file() -> List[Tuple[Path, np.ndarray]]:
    """
    与 ch2ped-pedt.py 同源：对每个 run 用 basic+act 的 cut 得到通过的 event indices，
    返回 (ch2_param_path, passing_indices) 列表。
    """
    pairs = basic_act._list_paired_param_files()
    print(f"[ch2ped-min] 找到 {len(pairs)} 个可配对的参数文件。")

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
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """单文件：按 passing 下标读取 ch2ped_mean/min_ch2，返回 (ped, ch2min)。"""
    ch2_param_path, passing = args
    passing = np.asarray(passing, dtype=np.int64)
    try:
        with h5py.File(ch2_param_path, "r") as f:
            if CH2_PED_MEAN_KEY not in f or CH2_MIN_KEY not in f:
                return None
            dset_ped = f[CH2_PED_MEAN_KEY]
            dset_min = f[CH2_MIN_KEY]
            n_ev = int(min(dset_ped.shape[0], dset_min.shape[0]))
            if n_ev <= 0:
                return None
            valid = (passing >= 0) & (passing < n_ev)
            passing = passing[valid]
            if passing.size == 0:
                return None
            ped = np.asarray(dset_ped[passing], dtype=np.float64)
            ch2min = np.asarray(dset_min[passing], dtype=np.float64)
        return (ped, ch2min)
    except Exception:
        return None


def main() -> None:
    file_tasks = _load_passing_events_per_file()
    if not file_tasks:
        print("[ch2ped-min] 没有通过 cut 的事例可处理，退出。")
        return

    n_total = int(sum(len(idx) for _, idx in file_tasks))
    print(f"[ch2ped-min] 共 {n_total} 个通过 cut 的事例，分布在 {len(file_tasks)} 个文件中。")

    workers = max(1, os.cpu_count() or 1)
    print(f"[ch2ped-min] 批量读取 CH2_parameters（仅通过 cut 的 event），进程数: {workers}")

    all_ped: List[np.ndarray] = []
    all_min: List[np.ndarray] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_phase2_worker, (p, idx)): p for p, idx in file_tasks}
        done = 0
        for fut in as_completed(futures):
            out = fut.result()
            done += 1
            if out is None:
                continue
            ped, ch2min = out
            if ped.size == 0:
                continue
            all_ped.append(ped)
            all_min.append(ch2min)
            if done % 20 == 0 or done == len(futures):
                n_acc = int(sum(x.size for x in all_ped))
                print(f"[ch2ped-min] 进度: {done}/{len(futures)} 文件, 已累计事件 {n_acc}")

    if not all_ped:
        print(
            "[ch2ped-min] 未读取到任何 CH2 ped/min 数据。"
            f"请确认 CH2_parameters 中存在 {CH2_PED_MEAN_KEY}/{CH2_MIN_KEY}。"
        )
        return

    x_vals = np.concatenate(all_ped)
    y_vals = np.concatenate(all_min)
    fin = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals = x_vals[fin]
    y_vals = y_vals[fin]
    print(f"[ch2ped-min] 完成：共绘制 {x_vals.size} 个点（有限值）。")

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.scatter(x_vals, y_vals, s=2, alpha=0.5, edgecolors="none")
    ax.set_xlabel("CH2 ped mean (ADC)", fontsize=12)
    ax.set_ylabel("CH2 min (ADC)", fontsize=12)
    ax.set_title("CH2 ped mean vs CH2 min (basic+act cuts passed)", fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

