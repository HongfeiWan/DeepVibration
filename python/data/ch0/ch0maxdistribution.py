#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对 raw_pulse/CH0_parameters 中的所有 HDF5 文件：
1. 根据 CH0_parameters 目录下的子文件夹名自动找到对应的 HDF5 文件；
2. 读取其中的 max_ch0 数组；
3. 汇总所有文件中事件的 max_ch0 值；
4. 绘制整体分布直方图（x 轴为 max_ch0，y 轴为 count）。

所有路径通过本脚本位置推导为相对路径。
"""

import argparse
from pathlib import Path
from typing import List

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _discover_ch0_parameters_dir_relative_to_script() -> Path:
    """
    返回 CH0_parameters 目录的绝对路径。

    目录结构假定为：
        project_root/
          data/
            hdf5/
              raw_pulse/
                CH0_parameters/
          python/
            data/
              ch0/
                ch0maxdistribution.py  (本脚本)
    """
    script_dir = Path(__file__).resolve().parent        # .../python/data/ch0
    python_dir = script_dir.parent.parent               # .../python
    project_root = python_dir.parent                    # 项目根
    ch0_parameters_dir = project_root / "data" / "hdf5" / "raw_pulse" / "CH0_parameters"
    return ch0_parameters_dir


def _list_h5_files(folder: Path) -> List[Path]:
    """
    根据 CH0_parameters 目录下的子文件夹名查找对应的 HDF5 文件。

    约定：
    - CH0_parameters/
        run001/
            run001.h5   或 run001.hdf5
        run002/
            run002.h5   或 run002.hdf5
    若按文件夹名精确匹配的文件不存在，则回退为该子目录中找到的第一
    个 .h5 / .hdf5 文件。
    """
    if not folder.exists():
        raise FileNotFoundError(f"源目录不存在: {folder}")
    files: List[Path] = []
    for entry in sorted(folder.iterdir()):
        if entry.is_dir():
            dirname = entry.name
            # 优先尝试与文件夹同名的 .h5 / .hdf5
            candidates = [
                entry / f"{dirname}.h5",
                entry / f"{dirname}.hdf5",
            ]
            target = None
            for c in candidates:
                if c.exists():
                    target = c
                    break
            # 若没有严格同名文件，则在该目录下找第一个 .h5 / .hdf5
            if target is None:
                for child in sorted(entry.iterdir()):
                    if child.is_file() and child.name.lower().endswith((".h5", ".hdf5")):
                        target = child
                        break
            if target is not None:
                files.append(target)
        elif entry.is_file() and entry.name.lower().endswith((".h5", ".hdf5")):
            # 允许 CH0_parameters 根目录下直接放部分 HDF5 文件
            files.append(entry)
    files.sort()
    return files

def _read_max_ch0_from_file(path: Path) -> np.ndarray:
    """
    从单个 CH0_parameters 文件读取 max_ch0 数组。
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从 raw_pulse/CH0_parameters 目录中读取所有 HDF5 文件的 max_ch0，"
            "汇总并绘制 max_ch0 的分布直方图。"
        )
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=200,
        help="直方图的 bin 数（默认 200）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ch0_parameters_dir = _discover_ch0_parameters_dir_relative_to_script()
    print(f"CH0_parameters 目录: {ch0_parameters_dir}")

    files = _list_h5_files(ch0_parameters_dir)
    if not files:
        print("未在 CH0_parameters 目录中找到任何 HDF5 文件。")
        return

    # 顺序读取所有文件的 max_ch0
    all_values_list: List[np.ndarray] = []
    for path in files:
        data = _read_max_ch0_from_file(path)
        if data.size > 0:
            all_values_list.append(data)

    if not all_values_list:
        print("未从任何 CH0_parameters 文件中读取到 max_ch0 数据。")
        return

    all_values = np.concatenate(all_values_list, axis=0)
    print(f"总事件数: {all_values.size}")
    print(f"max_ch0 范围: min={all_values.min():.2f}, max={all_values.max():.2f}")

    # 绘制分布直方图：统一使用 Arial 字体
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"]})
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.hist(all_values, bins=args.bins, color="C0", alpha=0.8)

    # 轴标签 16pt
    ax.set_xlabel("max_ch0", fontsize=16)
    ax.set_ylabel("Count", fontsize=16)


    # 刻度值字体 12pt
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

