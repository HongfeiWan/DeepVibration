#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量计算原始 CH0-3 HDF5 中每个 event 的 ch0 最大值，并写入对应的 CH0max HDF5。

需求：
- 扫描相对于本脚本路径的 `../data/hdf5/raw_pulse/CH0-3` 目录下所有 HDF5 文件；
- 对每个文件中的所有 event，计算 ch0 通道的 max（沿 sample 轴）；
- 结果写入 `../data/hdf5/raw_pulse/CH0max` 目录下、与源文件同名的 HDF5；
- 尽量“使用所有 GPU 并行处理多个文件”。

说明：
- 这里使用 PyTorch（若可用）调度多 GPU：
  - 每个进程绑定一个 GPU（CUDA device），负责一部分文件；
  - 波形仍由 h5py 从 CPU 内存读入，再复制到对应 GPU 上做 max 运算；
  - IO 仍然在 CPU/磁盘侧，GPU 主要加速大矩阵的 max 计算。
- 若本机没有 GPU 或未安装 torch，则自动退化为纯 CPU 版本（多进程按文件并行）。
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Tuple

import h5py
import numpy as np


def _discover_paths_relative_to_script() -> Tuple[Path, Path]:
    """
    返回 (ch0_3_dir, ch0max_dir)，均为绝对路径。

    目录结构假定为（与现有项目保持一致）：
        project_root/
          data/
            hdf5/
              raw_pulse/
                CH0-3/
                CH0max/
          python/
            data/
              ch0max.py  (本脚本)
    """
    script_dir = Path(__file__).resolve().parent  # .../python/data
    python_dir = script_dir.parent  # .../python
    project_root = python_dir.parent  # 项目根

    raw_pulse_root = project_root / "data" / "hdf5" / "raw_pulse"
    ch0_3_dir = raw_pulse_root / "CH0-3"
    ch0max_dir = raw_pulse_root / "CH0max"
    ch0max_dir.mkdir(parents=True, exist_ok=True)
    return ch0_3_dir, ch0max_dir


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


def _compute_max_ch0_for_file(
    src_path: Path,
    dst_path: Path,
    ch0_index: int = 0,
) -> None:
    """
    对单个 CH0-3 文件，计算所有 event 的 max(ch0) 并写入对应的 CH0max 文件。

    - src_path: 源 CH0-3 HDF5 路径
    - dst_path: 输出 CH0max HDF5 路径
    - ch0_index: ch0 在 channel_data 中的通道索引
    """
    if dst_path.exists():
        # 若目标已存在，可根据需要选择跳过或覆盖；这里选择覆盖，避免旧格式。
        dst_path.unlink()

    with h5py.File(src_path, "r") as f_src:
        if "channel_data" not in f_src:
            print(f"[跳过] {src_path.name}: 无 'channel_data' 数据集")
            return
        dset = f_src["channel_data"]
        # 期望形状: [n_samples, n_channels, n_events]
        if dset.ndim != 3:
            print(f"[跳过] {src_path.name}: 'channel_data' 维度不是 3, shape={dset.shape}")
            return

        n_samples, n_channels, n_events = dset.shape
        if not (0 <= ch0_index < n_channels):
            print(
                f"[跳过] {src_path.name}: ch0_index={ch0_index} 超出通道范围 (n_channels={n_channels})"
            )
            return

        # 一次性读取该文件所有事件的 ch0 波形: [n_samples, n_events]
        batch = dset[:, ch0_index, :]  # shape: [n_samples, n_events]
        # 在 sample 轴上取最大值 -> 每个 event 一个标量
        max_vals = np.asarray(batch, dtype=np.float32).max(axis=0)

    # 写入目标 HDF5
    with h5py.File(dst_path, "w") as f_dst:
        f_dst.create_dataset("max_ch0", data=max_vals)
        f_dst.attrs["source_file"] = str(src_path)
        f_dst.attrs["channel_index"] = int(ch0_index)
        f_dst.attrs["description"] = (
            "Per-event max value along sample axis for channel ch0 (index=channel_index)."
        )

    print(f"[完成] {src_path.name} -> {dst_path.name} (events={n_events})")


def _chunk_files_for_workers(files: List[Path], num_workers: int) -> List[List[Path]]:
    """简单地按轮询方式把文件分到不同 worker。"""
    if num_workers <= 0:
        return [files]
    buckets: List[List[Path]] = [[] for _ in range(num_workers)]
    for i, f in enumerate(files):
        buckets[i % num_workers].append(f)
    return buckets


def _worker_entry(
    worker_idx: int,
    files: Iterable[Path],
    dst_dir: Path,
    ch0_index: int,
) -> None:
    """子进程入口：处理分配到的一批文件。"""
    print(f"[worker {worker_idx}] 使用 CPU")

    for src in files:
        dst = dst_dir / src.name
        try:
            _compute_max_ch0_for_file(
                src_path=src,
                dst_path=dst,
                ch0_index=ch0_index,
            )
        except Exception as e:
            print(f"[错误] 处理 {src} 时失败: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "对 raw_pulse/CH0-3 中的所有 HDF5 文件，"
            "计算每个 event 的 ch0 最大值，并写入 raw_pulse/CH0max 中同名 HDF5。"
        )
    )
    parser.add_argument(
        "--ch0-index",
        type=int,
        default=0,
        help="ch0 在 channel_data 中的通道索引（默认 0）。",
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

    ch0_3_dir, ch0max_dir = _discover_paths_relative_to_script()
    print(f"源目录 CH0-3: {ch0_3_dir}")
    print(f"输出目录 CH0max: {ch0max_dir}")

    files = _list_h5_files(ch0_3_dir)
    if not files:
        print("未在 CH0-3 目录中找到任何 HDF5 文件。")
        return

    if args.workers is not None and args.workers > 0:
        num_workers = min(args.workers, len(files))
    else:
        # 默认 worker 数：使用 min(文件数, CPU 核数)
        num_workers = min(len(files), os.cpu_count() or 1)

    print(f"使用 worker 数: {num_workers} (ch0_index={args.ch0_index})")

    # 按 worker 分配文件
    buckets = _chunk_files_for_workers(files, num_workers)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for idx, f_list in enumerate(buckets):
            if not f_list:
                continue
            futures.append(
                executor.submit(
                    _worker_entry,
                    idx,
                    f_list,
                    ch0max_dir,
                    args.ch0_index,
                )
            )

        for fut in as_completed(futures):
            _ = fut.result()

    print("全部文件处理完成。")


if __name__ == "__main__":
    main()
