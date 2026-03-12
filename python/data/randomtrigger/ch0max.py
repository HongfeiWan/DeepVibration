#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对 raw_pulse/CH5 中的所有 HDF5 原始波形文件：
1. 利用所有 CPU 并行处理，每个进程负责一部分文件；
2. 模仿 randomtrigger/select.py 的 RT 选择思路，在 CH5 上快速筛选出 RT 事件
   （基于 max(ch5) > cut 的条件，批量计算最大值）；
3. 对于每个 RT 事件，到 raw_pulse/CH0max 中同名 HDF5 文件里，用事件号读取对应的 CH0max 值；
4. 将这些 RT 事件的 event 号和 CH0max 值写入到同名 HDF5 文件，保存在 raw_pulse/RTCH0max 目录下。

所有路径均通过本脚本位置推导为相对路径。
"""

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import h5py
import numpy as np
import multiprocessing as mp


def _discover_paths_relative_to_script() -> Tuple[Path, Path, Path]:
    """
    返回 (ch5_dir, ch0max_dir, rtch0max_dir)，均为绝对路径。

    目录结构假定为（与现有项目保持一致）：
        project_root/
          data/
            hdf5/
              raw_pulse/
                CH5/
                CH0max/
                RTCH0max/
          python/
            data/
              randomtrigger/
                ch0max.py  (本脚本)
    """
    script_dir = Path(__file__).resolve().parent        # .../python/data/randomtrigger
    python_dir = script_dir.parent                      # .../python/data
    data_dir = python_dir.parent                        # .../python
    project_root = data_dir.parent                      # 项目根

    raw_pulse_root = project_root / "data" / "hdf5" / "raw_pulse"
    ch5_dir = raw_pulse_root / "CH5"
    ch0max_dir = raw_pulse_root / "CH0max"
    rtch0max_dir = raw_pulse_root / "RTCH0max"
    rtch0max_dir.mkdir(parents=True, exist_ok=True)
    return ch5_dir, ch0max_dir, rtch0max_dir


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


def _select_rt_indices_for_file(
    ch5_path: Path,
    cut: float,
    channel_idx: int = 0,
    batch_size: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在单个 CH5 文件上做 RT 选择，返回：
        rt_indices: 满足 max(ch5) > cut 的事件索引
        max_values: 所有事件在 CH5 上的最大值数组

    逻辑参考 randomtrigger/select.py 中的 select_rt_events，但简化为“只做选择、不画图”。
    """
    with h5py.File(ch5_path, "r") as f:
        if "channel_data" not in f:
            raise KeyError(f"{ch5_path} 中没有找到 channel_data 数据集")

        channel_data = f["channel_data"]
        time_samples, num_channels, num_events = channel_data.shape

        if channel_idx < 0 or channel_idx >= num_channels:
            raise IndexError(
                f"{ch5_path} 通道索引 {channel_idx} 超出范围 [0, {num_channels-1}]"
            )

        max_values = np.zeros(num_events, dtype=np.float64)

        for i in range(0, num_events, batch_size):
            end_idx = min(i + batch_size, num_events)
            batch = channel_data[:, channel_idx, i:end_idx]  # (time_samples, batch_size)
            max_values[i:end_idx] = np.max(batch, axis=0)

    rt_mask = max_values > cut
    rt_indices = np.where(rt_mask)[0]
    return rt_indices, max_values


def _process_single_ch5_file(
    ch5_path: Path,
    ch0max_dir: Path,
    rtch0max_dir: Path,
    cut: float,
    channel_idx: int,
    batch_size: int,
) -> None:
    """
    对单个 CH5 文件：
        1. 选出 RT 事件索引；
        2. 从对应的 CH0max 文件读取相同 event 号的 max_ch0；
        3. 把 (rt_event_indices, rt_max_ch0) 写入 RTCH0max 目录下的同名 HDF5。
    """
    basename = ch5_path.name
    print(f"[文件] 处理 CH5: {basename}")

    # 1. RT 选择
    try:
        rt_indices, max_values_ch5 = _select_rt_indices_for_file(
            ch5_path=ch5_path,
            cut=cut,
            channel_idx=channel_idx,
            batch_size=batch_size,
        )
    except Exception as e:
        print(f"[跳过] {basename}: 计算 RT 事件时出错: {e}")
        return

    if rt_indices.size == 0:
        print(f"[信息] {basename}: 没有满足 RT 条件的事件 (cut={cut})，不生成 RTCH0max 文件。")
        return

    # 2. 对应的 CH0max 文件
    ch0max_path = ch0max_dir / basename
    if not ch0max_path.exists():
        print(f"[跳过] {basename}: 未找到对应的 CH0max 文件: {ch0max_path}")
        return

    with h5py.File(ch0max_path, "r") as f_ch0max:
        if "max_ch0" not in f_ch0max:
            print(f"[跳过] {basename}: CH0max 文件中缺少 'max_ch0' 数据集。")
            return
        dset_max_ch0 = f_ch0max["max_ch0"]
        if dset_max_ch0.ndim != 1:
            print(
                f"[跳过] {basename}: 'max_ch0' 维度不是 1, shape={dset_max_ch0.shape}"
            )
            return

        n_events_ch0 = dset_max_ch0.shape[0]
        # 去掉越界的 RT 事件索引
        valid_mask = (rt_indices >= 0) & (rt_indices < n_events_ch0)
        rt_indices_valid = rt_indices[valid_mask]
        if rt_indices_valid.size == 0:
            print(
                f"[跳过] {basename}: RT 事件索引全部越界 (n_events_ch0={n_events_ch0})。"
            )
            return

        rt_max_ch0 = np.asarray(dset_max_ch0[rt_indices_valid], dtype=np.float64)

    # 3. 写入 RTCH0max 文件
    rtch0max_path = rtch0max_dir / basename
    if rtch0max_path.exists():
        rtch0max_path.unlink()

    with h5py.File(rtch0max_path, "w") as f_out:
        f_out.create_dataset("rt_event_indices", data=rt_indices_valid.astype(np.int64))
        f_out.create_dataset("rt_ch0max", data=rt_max_ch0.astype(np.float64))
        f_out.attrs["source_file_ch5"] = str(ch5_path)
        f_out.attrs["source_file_ch0max"] = str(ch0max_path)
        f_out.attrs["rt_cut"] = float(cut)
        f_out.attrs["channel_idx_ch5"] = int(channel_idx)
        f_out.attrs["description"] = (
            "Random trigger (RT) events: indices (in CH5/CH0-3) and corresponding CH0max values.\n"
            "rt_event_indices: indices of events in the original CH5/CH0-3 files.\n"
            "rt_ch0max: max_ch0 for those events from CH0max."
        )

    print(
        f"[完成] {basename}: RT 事件数 = {rt_indices_valid.size}, "
        f"输出文件: {rtch0max_path.name}"
    )


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
    ch0max_dir: Path,
    rtch0max_dir: Path,
    cut: float,
    channel_idx: int,
    batch_size: int,
) -> None:
    """子进程入口：处理分配到的一批 CH5 文件。"""
    files = list(files)
    print(f"[worker {worker_idx}] 使用 CPU，处理 {len(files)} 个文件")

    for ch5_path in files:
        try:
            _process_single_ch5_file(
                ch5_path=ch5_path,
                ch0max_dir=ch0max_dir,
                rtch0max_dir=rtch0max_dir,
                cut=cut,
                channel_idx=channel_idx,
                batch_size=batch_size,
            )
        except Exception as e:
            print(f"[错误] 处理 {ch5_path} 时失败: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "对 raw_pulse/CH5 中的所有 HDF5 原始波形文件："
            "筛选随机触发 (RT) 事件，并从 CH0max 中读取对应的 CH0max 值，"
            "写入 raw_pulse/RTCH0max 目录中的同名 HDF5。"
        )
    )
    parser.add_argument(
        "--cut",
        type=float,
        default=6000.0,
        help="RT 截断阈值：max(CH5) > cut 视为 RT 事件（默认 6000.0）。",
    )
    parser.add_argument(
        "--ch5-channel-idx",
        type=int,
        default=0,
        help="CH5 文件中的通道索引（默认 0）。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="计算 CH5 最大值时的批处理大小（默认 1000）。",
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

    ch5_dir, ch0max_dir, rtch0max_dir = _discover_paths_relative_to_script()
    print(f"源目录 CH5:      {ch5_dir}")
    print(f"源目录 CH0max:  {ch0max_dir}")
    print(f"输出目录 RTCH0max: {rtch0max_dir}")

    ch5_files = _list_h5_files(ch5_dir)
    if not ch5_files:
        print("未在 CH5 目录中找到任何 HDF5 文件。")
        return

    if args.workers is not None and args.workers > 0:
        num_workers = min(args.workers, len(ch5_files))
    else:
        # 默认 worker 数：使用 min(文件数, CPU 核数)
        num_workers = min(len(ch5_files), os.cpu_count() or 1)

    print(
        f"使用 worker 数: {num_workers} "
        f"(cut={args.cut}, ch5_channel_idx={args.ch5_channel_idx}, batch_size={args.batch_size})"
    )

    # 按 worker 分配文件
    buckets = _chunk_files_for_workers(ch5_files, num_workers)

    # 使用 multiprocessing.Pool 在 Linux / 其他环境下做多进程并行，
    # 避免对 concurrent.futures.ProcessPoolExecutor 的依赖。
    with mp.Pool(processes=num_workers) as pool:
        pool.starmap(
            _worker_entry,
            [
                (
                    idx,
                    buckets[idx],
                    ch0max_dir,
                    rtch0max_dir,
                    args.cut,
                    args.ch5_channel_idx,
                    args.batch_size,
                )
                for idx in range(len(buckets))
                if buckets[idx]
            ],
        )

    print("全部文件处理完成。")


if __name__ == "__main__":
    main()

