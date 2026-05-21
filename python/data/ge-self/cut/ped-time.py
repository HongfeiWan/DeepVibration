#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按文件绘制 CH0 pedestal 均值随时间变化图。

数据来源：
- 时间：复用 python/utils/time.py 的 read_hdf5_time_span
- pedestal：读取 data/hdf5/raw_pulse/CH0_parameters 下每个文件的 ch0ped_mean 并取均值
"""

from __future__ import annotations

import importlib.util
import os
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


def _discover_project_root() -> Path:
    here = Path(__file__).resolve()
    # .../python/data/ge-self/cut/ped-time -> .../DeepVibration
    return here.parent.parent.parent.parent.parent


def _load_time_module(project_root: Path):
    time_py = project_root / "python" / "utils" / "time.py"
    if not time_py.is_file():
        raise FileNotFoundError(f"未找到 time.py: {time_py}")

    spec = importlib.util.spec_from_file_location("time_utils_module", str(time_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {time_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_ped_stats(param_file: Path, dataset_name: str) -> Tuple[float, float]:
    with h5py.File(param_file, "r") as f:
        if dataset_name not in f:
            raise KeyError(f"{param_file.name} 中缺少数据集 {dataset_name}")
        ped = np.asarray(f[dataset_name][...], dtype=np.float64)
        if ped.size == 0:
            raise ValueError(f"{param_file.name} 的 {dataset_name} 为空")
        return float(np.mean(ped)), float(np.var(ped, ddof=1))


def _try_read_time_for_param(
    time_mod,
    project_root: Path,
    ch0_param_file: Path,
) -> Optional[Tuple[object, object]]:
    """
    读取该参数文件对应的起止时间：
    1) 直接读参数文件中的 time_data（若存在）
    2) 读 CH0-3 目录下同名 _processed.h5
    """
    # 方案1：参数文件本身
    try:
        return time_mod.read_hdf5_time_span(str(ch0_param_file))
    except Exception:
        pass

    # 方案2：映射到 CH0-3 的 processed 文件
    ch03_dir = project_root / "data" / "hdf5" / "raw_pulse" / "CH0-3"
    stem = ch0_param_file.stem
    candidates = [ch03_dir / f"{stem}.h5"]
    if stem.endswith("_processed"):
        candidates.append(ch03_dir / f"{stem[:-10]}_processed.h5")
    else:
        candidates.append(ch03_dir / f"{stem}_processed.h5")

    for candidate in candidates:
        if candidate.is_file():
            try:
                return time_mod.read_hdf5_time_span(str(candidate))
            except Exception:
                continue
    return None


if __name__ == "__main__":
    project_root = _discover_project_root()
    time_mod = _load_time_module(project_root)

    ch0_param_dir = project_root / "data" / "hdf5" / "raw_pulse" / "CH0_parameters"
    ch1_param_dir = project_root / "data" / "hdf5" / "raw_pulse" / "CH1_parameters"
    if not ch0_param_dir.is_dir() or not ch1_param_dir.is_dir():
        raise FileNotFoundError(f"目录不存在: {ch0_param_dir} 或 {ch1_param_dir}")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
        }
    )

    files = sorted(
        [
            p
            for p in ch0_param_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".h5", ".hdf5"}
        ]
    )
    if not files:
        raise RuntimeError(f"在 {ch0_param_dir} 未找到参数文件")

    ch1_map = {
        p.name: p
        for p in ch1_param_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".h5", ".hdf5"}
    }

    time_points: List[object] = []
    ped0_means: List[float] = []
    ped0_vars: List[float] = []
    ped1_means: List[float] = []
    ped1_vars: List[float] = []
    used_names: List[str] = []
    n_skip = 0

    for fp in files:
        try:
            fp1 = ch1_map.get(fp.name)
            if fp1 is None:
                n_skip += 1
                continue

            ped0_mean, ped0_var = _read_ped_stats(fp, "ch0ped_mean")
            ped1_mean, ped1_var = _read_ped_stats(fp1, "ch1ped_mean")
            span = _try_read_time_for_param(time_mod, project_root, fp)
            if span is None:
                n_skip += 1
                continue
            t_start, t_end = span
            t_mid = t_start + (t_end - t_start) / 2

            time_points.append(t_mid)
            ped0_means.append(ped0_mean)
            ped0_vars.append(ped0_var)
            ped1_means.append(ped1_mean)
            ped1_vars.append(ped1_var)
            used_names.append(fp.name)
        except Exception as e:
            n_skip += 1
            print(f"[跳过] {fp.name}: {e}")

    if not time_points:
        raise RuntimeError("没有成功读取到任何可用的 时间- ped 数据点")

    order = np.argsort(np.array(time_points, dtype="datetime64[ns]"))
    time_sorted = np.array(time_points, dtype="datetime64[ns]")[order]
    ped0_sorted = np.array(ped0_means, dtype=np.float64)[order]
    ped0_var_sorted = np.array(ped0_vars, dtype=np.float64)[order]
    ped1_sorted = np.array(ped1_means, dtype=np.float64)[order]
    ped1_var_sorted = np.array(ped1_vars, dtype=np.float64)[order]

    print("=" * 70)
    print(f"成功使用文件数: {len(time_sorted)}")
    print(f"跳过文件数: {n_skip}")
    print("=" * 70)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.errorbar(
        time_sorted,
        ped0_sorted,
        yerr=ped0_var_sorted,
        fmt="o",
        linestyle="none",
        color="C0",
        ecolor="C0",
        elinewidth=0.2,
        capsize=2,
        markersize=4,
        alpha=0.85,
        label="CH0 ped mean ± var",
    )
    ax.errorbar(
        time_sorted,
        ped1_sorted,
        yerr=ped1_var_sorted,
        fmt="o",
        linestyle="none",
        color="C1",
        ecolor="C1",
        elinewidth=0.2,
        capsize=2,
        markersize=4,
        alpha=0.85,
        label="CH1 ped mean ± var",
    )
    ax.set_xlabel("Time", fontsize=16)
    ax.set_ylabel("ped", fontsize=16)
    #ax.set_yscale("log")
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=12, loc="upper right")

    # 使用 AutoDateLocator + ConciseDateFormatter 自动生成简洁清晰的时间轴（与 unite.py 一致）
    locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

    ax.set_xlim(time_sorted[0], time_sorted[-1])
    ax.set_ylim(-2000,5000)
    fig.tight_layout()
    plt.show()
