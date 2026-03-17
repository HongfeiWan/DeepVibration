#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CH4 通道 max_ch4 参数分布分析

从 data/hdf5/raw_pulse/CH4_parameters 目录中读取各子目录下同名 HDF5 文件，
提取其中的 max_ch4 数组，汇总并绘制其分布。
"""

from pathlib import Path
from typing import List, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _discover_ch4_parameters_dir_relative_to_script() -> Path:
    """
    返回 CH4_parameters 目录的绝对路径。

    目录结构假定为：
        project_root/
          data/
            hdf5/
              raw_pulse/
                CH4_parameters/
          python/
            data/
              ch4/
                distribution.py  (本脚本)
    """
    script_dir = Path(__file__).resolve().parent        # .../python/data/ch4
    python_dir = script_dir.parent.parent               # .../python
    project_root = python_dir.parent                    # 项目根
    ch4_parameters_dir = project_root / "data" / "hdf5" / "raw_pulse" / "CH4_parameters"
    return ch4_parameters_dir


def _list_h5_files(folder: Path) -> List[Path]:
    """
    根据 CH4_parameters 目录下的子文件夹名查找对应的 HDF5 文件。

    约定：
    - CH4_parameters/
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
            target: Optional[Path] = None
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
            # 允许 CH4_parameters 根目录下直接放部分 HDF5 文件
            files.append(entry)
    files.sort()
    return files


def _read_max_ch4_from_file(path: Path) -> np.ndarray:
    """
    从单个 CH4_parameters 文件读取 max_ch4 数组。
    若格式不符合预期，则返回长度为 0 的数组。
    """
    try:
        with h5py.File(path, "r") as f:
            if "max_ch4" not in f:
                print(f"[警告] {path.name}: 缺少数据集 'max_ch4'，跳过。")
                return np.empty((0,), dtype=np.float64)
            dset = f["max_ch4"]
            if dset.ndim != 1:
                print(
                    f"[警告] {path.name}: 'max_ch4' 维度不是 1, shape={dset.shape}，跳过。"
                )
                return np.empty((0,), dtype=np.float64)
            data = np.asarray(dset[...], dtype=np.float64)
            return data
    except Exception as e:
        print(f"[错误] 读取 {path} 时失败: {e}")
        return np.empty((0,), dtype=np.float64)


def main(bins: int = 100, trigger_threshold: Optional[float] = 7060.0) -> None:
    """
    读取 CH4_parameters 目录下所有 HDF5 文件中的 max_ch4，
    汇总并绘制分布直方图。
    """
    ch4_parameters_dir = _discover_ch4_parameters_dir_relative_to_script()
    print(f"CH4_parameters 目录: {ch4_parameters_dir}")

    files = _list_h5_files(ch4_parameters_dir)
    if not files:
        print("未在 CH4_parameters 目录中找到任何 HDF5 文件。")
        return

    all_values: List[np.ndarray] = []
    for path in files:
        data = _read_max_ch4_from_file(path)
        if data.size > 0:
            all_values.append(data)

    if not all_values:
        print("未从任何 CH4_parameters 文件中读取到 max_ch4 数据。")
        return

    max_values = np.concatenate(all_values)
    print(f"总事件数: {max_values.size}")
    print(f"max_ch4 范围: min={max_values.min():.2f}, max={max_values.max():.2f}")

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(max_values, bins=bins, edgecolor="black", alpha=0.7)

    # 触发阈值红色虚线
    if trigger_threshold is not None:
        ax.axvline(
            trigger_threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Trigger = {trigger_threshold:.0f} FADC",
        )
        ax.legend()

    ax.set_xlabel("max_ch4", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Distribution of max_ch4 (N={max_values.size})", fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(bins=500, trigger_threshold=7060.0)
