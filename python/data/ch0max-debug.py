#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用多 CPU 扫描原始 CH0-3 HDF5 中每个 event 的 ch0 最大值，
一旦发现某个 event 的 ch0 最大值超过给定阈值，则立刻画出该 event 的 ch0 波形并停止所有 worker。

需求：
- 扫描相对于本脚本路径的 `../data/hdf5/raw_pulse/CH0-3` 目录下所有 HDF5 文件；
- 对每个文件中的所有 event，计算 ch0 通道的 max（沿 sample 轴）；
- 若某个 event 的 ch0 max > 阈值（默认 16384），则绘图并停止全局搜索；
- 不再生成任何 CH0max 输出文件。
"""

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import h5py
import matplotlib.pyplot as plt
import multiprocessing as mp
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
    # 旧版脚本会使用 CH0max 目录，这里虽然不再写入，但保留返回值以兼容之前代码，方便调试。
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


def _compute_max_ch0_for_event(
    dset: h5py.Dataset,
    ch0_index: int,
    event_idx: int,
) -> float:
    """
    计算单个 event 的 ch0 最大值。
    dset 形状期望为 [n_samples, n_channels, n_events]。
    """
    # 直接按 event 维度切一列，避免一次性读入所有 event。
    waveform = np.asarray(dset[:, ch0_index, event_idx], dtype=np.float32)
    return float(waveform.max())


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
    ch0_index: int,
    threshold: float,
    stop_event: "mp.synchronize.Event",
) -> None:
    """
    子进程入口：处理分配到的一批文件。
    一旦发现 ch0 max 超过阈值的 event，则绘图、设置 stop_event 并返回。
    其他 worker 会检测到 stop_event 被设置后尽快退出。
    """
    print(f"[worker {worker_idx}] 使用 CPU，分配到 {len(list(files))} 个文件")

    # files 可能是任意可迭代，这里转成列表以便多次遍历长度等。
    files = list(files)

    for src in files:
        if stop_event.is_set():
            # 其他 worker 已经找到满足条件的 event，尽快退出。
            print(f"[worker {worker_idx}] 检测到 stop_event，提前结束。")
            return

        try:
            with h5py.File(src, "r") as f_src:
                if "channel_data" not in f_src:
                    print(f"[worker {worker_idx}] [跳过] {src.name}: 无 'channel_data' 数据集")
                    continue

                dset = f_src["channel_data"]
                if dset.ndim != 3:
                    print(
                        f"[worker {worker_idx}] [跳过] {src.name}: "
                        f"'channel_data' 维度不是 3, shape={dset.shape}"
                    )
                    continue

                n_samples, n_channels, n_events = dset.shape
                if not (0 <= ch0_index < n_channels):
                    print(
                        f"[worker {worker_idx}] [跳过] {src.name}: "
                        f"ch0_index={ch0_index} 超出通道范围 (n_channels={n_channels})"
                    )
                    continue

                for event_idx in range(n_events):
                    if stop_event.is_set():
                        print(
                            f"[worker {worker_idx}] 在文件 {src.name} 内检测到 stop_event，提前结束。"
                        )
                        return

                    max_val = _compute_max_ch0_for_event(dset, ch0_index, event_idx)
                    if max_val > threshold and not stop_event.is_set():
                        # 我们先设置 stop_event，再绘图，减少竞态。
                        stop_event.set()

                        waveform = np.asarray(
                            dset[:, ch0_index, event_idx], dtype=np.float32
                        )

                        print(
                            f"[worker {worker_idx}] 找到 ch0 max 超过阈值的 event: "
                            f"文件={src.name}, event_index={event_idx}, "
                            f"max={max_val}, 阈值={threshold}"
                        )

                        plt.figure()
                        plt.plot(waveform)
                        plt.xlabel("Sample")
                        plt.ylabel("Amplitude")
                        plt.title(
                            f"{src.name} event {event_idx} ch0, "
                            f"max={max_val:.1f} > {threshold}"
                        )
                        plt.grid(True)
                        plt.tight_layout()
                        plt.show()

                        return
        except Exception as e:
            print(f"[worker {worker_idx}] [错误] 处理 {src} 时失败: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "使用多 CPU 并行扫描 raw_pulse/CH0-3 中的所有 HDF5 文件，"
            "查找第一个 ch0 最大值超过阈值的 event，并立刻可视化其 ch0 波形。"
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
    parser.add_argument(
        "--threshold",
        type=float,
        default=16384.0,
        help="阈值：若某个 event 的 ch0 最大值超过该值，则可视化并停止搜索（默认 16384）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ch0_3_dir, ch0max_dir = _discover_paths_relative_to_script()
    print(f"源目录 CH0-3: {ch0_3_dir}")

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

    # 使用 multiprocessing 开启多个 worker 进程，并通过共享 Event 实现全局早停。
    stop_event = mp.Event()
    processes: list[mp.Process] = []

    for idx, f_list in enumerate(buckets):
        if not f_list:
            continue
        p = mp.Process(
            target=_worker_entry,
            args=(
                idx,
                f_list,
                args.ch0_index,
                args.threshold,
                stop_event,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    if stop_event.is_set():
        print("已找到满足条件的 event，已可视化 ch0，所有 worker 已停止。")
    else:
        print("全部文件处理完成，未找到 ch0 max 超过阈值的 event。")


if __name__ == "__main__":
    main()
