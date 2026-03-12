#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对 raw_pulse/CH0max 中的所有 HDF5 文件：
1. 利用所有 CPU 并行读取每个文件的 max_ch0；
2. 汇总所有文件中事件的 CH0max 值；
3. 绘制整体分布直方图（x 轴为 CH0max，y 轴为 count）。

所有路径通过本脚本位置推导为相对路径。
"""

import argparse
import os
from pathlib import Path
from typing import Iterable, List

import h5py
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np


def _discover_ch0max_dir_relative_to_script() -> Path:
    """
    返回 CH0max 目录的绝对路径。

    目录结构假定为（与 data/ch0max.py 一致）：
        project_root/
          data/
            hdf5/
              raw_pulse/
                CH0max/
          python/
            data/
              ch0maxdistribution.py  (本脚本)
    """
    script_dir = Path(__file__).resolve().parent        # .../python/data
    python_dir = script_dir.parent                      # .../python
    project_root = python_dir.parent                    # 项目根
    ch0max_dir = project_root / "data" / "hdf5" / "raw_pulse" / "CH0max"
    return ch0max_dir


def _list_h5_files(folder: Path) -> List[Path]:
    """列出目录下所有 .h5 / .hdf5 文件（不递归）。"""
    if not folder.exists():
        raise FileNotFoundError(f"源目录不存在: {folder}")
    files: List[Path] = []
    for name in os.listdir(folder):
        if name.lower().endswith((".h5", ".hdf5")):
            files.append(folder / name)
    files.sort()
    return files

def _read_max_ch0_from_file(path: Path) -> np.ndarray:
    """
    从单个 CH0max 文件读取 max_ch0 数组。
    若格式不符合预期，则返回长度为 0 的数组。
    """
    try:
        with h5py.File(path, "r") as f:
            if "max_ch0" not in f:
                print(f"[警告] {path.name}: 缺少数据集 'max_ch0'，跳过。")
                return np.empty((0,), dtype=np.float64)
            dset = f["max_ch0"]
            if dset.ndim != 1:
                print(
                    f"[警告] {path.name}: 'max_ch0' 维度不是 1, shape={dset.shape}，跳过。"
                )
                return np.empty((0,), dtype=np.float64)
            data = np.asarray(dset[...], dtype=np.float64)
            return data
    except Exception as e:
        print(f"[错误] 读取 {path} 时失败: {e}")
        return np.empty((0,), dtype=np.float64)


def _worker_read_files(paths: Iterable[Path]) -> np.ndarray:
    """worker 进程：读取若干文件的 max_ch0 并在本进程内拼接。"""
    vals: List[np.ndarray] = []
    for p in paths:
        data = _read_max_ch0_from_file(p)
        if data.size > 0:
            vals.append(data)
    if not vals:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate(vals, axis=0)


def _chunk_files_for_workers(files: List[Path], num_workers: int) -> List[List[Path]]:
    """简单地按轮询方式把文件分到不同 worker。"""
    if num_workers <= 0:
        return [files]
    buckets: List[List[Path]] = [[] for _ in range(num_workers)]
    for i, f in enumerate(files):
        buckets[i % num_workers].append(f)
    return buckets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从 raw_pulse/CH0max 目录中读取所有 HDF5 文件的 max_ch0，"
            "汇总并绘制 CH0max 的分布直方图。"
        )
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=200,
        help="直方图的 bin 数（默认 200）。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并行 worker 数（默认=min(文件数, CPU 核数)）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ch0max_dir = _discover_ch0max_dir_relative_to_script()
    print(f"CH0max 目录: {ch0max_dir}")

    files = _list_h5_files(ch0max_dir)
    if not files:
        print("未在 CH0max 目录中找到任何 HDF5 文件。")
        return

    if args.workers is not None and args.workers > 0:
        num_workers = min(args.workers, len(files))
    else:
        num_workers = min(len(files), os.cpu_count() or 1)

    print(f"使用 worker 数: {num_workers} (bins={args.bins})")

    buckets = _chunk_files_for_workers(files, num_workers)

    # 多进程读取数据
    all_values_list: List[np.ndarray] = []
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_read_files, buckets)
        for arr in results:
            if arr.size > 0:
                all_values_list.append(arr)

    if not all_values_list:
        print("未从任何 CH0max 文件中读取到 max_ch0 数据。")
        return

    all_values = np.concatenate(all_values_list, axis=0)
    print(f"总事件数: {all_values.size}")
    print(f"CH0max 范围: min={all_values.min():.2f}, max={all_values.max():.2f}")

    # 绘制分布直方图
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.hist(all_values, bins=args.bins, color="C0", alpha=0.8)
    ax.set_xlabel("CH0max", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Distribution of CH0max (N={all_values.size})",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

