#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ch3max-q: 对通过 basic+act 全部 cut 的事例，从 CH0-3 读取 CH3 波形，
绘制横轴 max(CH3)、纵轴 q（CH3 全窗口积分值）的散点图。

调度策略：
- I/O 线程：预加载下一文件的 CH3 到内存，放入队列
- 计算：使用多线程对当前文件的 events 分块并行计算 max/q（NumPy 释放 GIL）
- 重叠：主线程计算文件 N 时，I/O 线程同时加载文件 N+1
- 绘图：主线程 hold on 增量绘制（Matplotlib GUI 需在主线程），set_offsets 增量更新
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import psutil

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
CH0_PARAM_DIR = DATA_ROOT / "CH0_parameters"
CH1_PARAM_DIR = DATA_ROOT / "CH1_parameters"
CH4_PARAM_DIR = DATA_ROOT / "CH4_parameters"
CH5_PARAM_DIR = DATA_ROOT / "CH5_parameters"
CH3_DIR = DATA_ROOT / "CH0-3"

RESERVE_MEMORY_BYTES = 2 * 1024**3
CH3_INDEX = 3

# 计算线程数：预留 1 核给 I/O/系统，其余用于计算
def _n_compute_workers() -> int:
    n = os.cpu_count() or 1
    return 96


def _resolve_ch0_3_path(ch0_param_path: Path) -> Path | None:
    name = ch0_param_path.name
    candidates = [CH3_DIR / name, CH3_DIR / (ch0_param_path.stem + ".h5")]
    for p in candidates:
        if p.exists():
            return p
    return None

def _load_passing_events_per_file() -> List[Tuple[Path, np.ndarray]]:
    pairs = basic_act._list_paired_param_files()
    print(f"[ch3max-q] 找到 {len(pairs)} 个可配对的参数文件。")

    result: List[Tuple[Path, np.ndarray]] = []
    for ch0_path, ch5_path in pairs:
        ch0_3_path = _resolve_ch0_3_path(ch0_path)
        if ch0_3_path is None:
            continue

        m0, cmin, m5, ped0, ped1, m1, c1min, m4, t4, _ = basic_act._load_basic_features_for_run(
            ch0_path, ch5_path
        )
        m1_c = basic_act.cut_ch0_min_positive(cmin)
        m2_c = basic_act.cut_ch0_max_saturation(m0)
        m3_c = basic_act.cut_ch5_self_trigger(m5)
        m4_c = basic_act.cut_pedestal_3sigma(ped0, ped1, m5)
        m5_c = basic_act.cut_act(m4, t4)
        m6_c = basic_act.cut_mincut(cmin, c1min, m4, t4)
        mask = m1_c & m2_c & m3_c & m4_c & m5_c & m6_c
        passing = np.where(mask)[0]
        if passing.size == 0:
            continue
        result.append((ch0_3_path, passing))
    return result

def _io_worker(
    tasks: List[Tuple[Path, np.ndarray, bool]],
    out_queue: Queue,) -> None:
    """I/O 线程：依次加载每个文件的 CH3 波形，放入 out_queue。"""
    for ch0_3_path, passing, use_full in tasks:
        try:
            with h5py.File(ch0_3_path, "r") as f:
                ch_data = f["channel_data"]
                n_samples, n_channels, n_events = ch_data.shape
                if n_channels <= CH3_INDEX:
                    out_queue.put((np.array([]), np.array([])))
                    continue
                passing = np.asarray(passing, dtype=np.int64)
                valid = (passing >= 0) & (passing < n_events)
                passing = passing[valid]
                if passing.size == 0:
                    out_queue.put((np.array([]), np.array([])))
                    continue
                if use_full:
                    ch3_full = ch_data[:, CH3_INDEX, :].astype(np.float64)
                    wf = ch3_full[:, passing].copy()
                else:
                    wf = ch_data[:, CH3_INDEX, passing].astype(np.float64)
                n_s = wf.shape[0]
                out_queue.put((wf, n_s))
        except Exception as e:
            out_queue.put((None, e))
    out_queue.put((None, None))


def _compute_chunk(args: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """单块计算：对 wf_chunk 计算 max 和 q。"""
    wf_chunk, t_ns = args
    ch_max = np.max(wf_chunk, axis=0)
    q = np.sum(wf_chunk * t_ns[:, np.newaxis], axis=0)
    return ch_max, q


def _compute_max_q_parallel(
    wf: np.ndarray,
    n_compute_workers: int,
    min_events_for_parallel: int = 500,) -> Tuple[np.ndarray, np.ndarray]:
    """对 wf (n_samples, n_events) 分块并行计算 max 和 q。"""
    n_samples, n_events = wf.shape
    t_ns = np.arange(n_samples, dtype=np.float64) * 4.0

    if n_events < min_events_for_parallel or n_compute_workers <= 1:
        ch_max = np.max(wf, axis=0)
        q = np.sum(wf * t_ns[:, np.newaxis], axis=0)
        return ch_max, q

    chunk_size = max(1, (n_events + n_compute_workers - 1) // n_compute_workers)
    chunks: List[Tuple[np.ndarray, np.ndarray]] = []
    for start in range(0, n_events, chunk_size):
        end = min(start + chunk_size, n_events)
        chunks.append((wf[:, start:end], t_ns))

    ch_max_list: List[np.ndarray] = []
    q_list: List[np.ndarray] = []
    with ThreadPoolExecutor(max_workers=n_compute_workers) as ex:
        futures = [ex.submit(_compute_chunk, c) for c in chunks]
        for fut in as_completed(futures):
            cm, qm = fut.result()
            ch_max_list.append(cm)
            q_list.append(qm)

    return np.concatenate(ch_max_list), np.concatenate(q_list)


def main() -> None:
    file_tasks = _load_passing_events_per_file()
    if not file_tasks:
        print("[ch3max-q] 没有通过 cut 的事例可处理，退出。")
        return

    n_total = sum(len(idx) for _, idx in file_tasks)
    print(f"[ch3max-q] 共 {n_total} 个通过 cut 的事例，分布在 {len(file_tasks)} 个文件中。")

    mem = psutil.virtual_memory()
    avail_bytes = max(0, mem.total - RESERVE_MEMORY_BYTES)
    n_compute = _n_compute_workers()
    print(f"[ch3max-q] 可用内存约 {avail_bytes / 1024**3:.1f} GB，计算线程数 {n_compute}。")

    # 为每个文件决定是否整通道读取
    io_tasks: List[Tuple[Path, np.ndarray, bool]] = []
    for ch0_3_path, passing in file_tasks:
        with h5py.File(ch0_3_path, "r") as f:
            n_s, _, n_ev = f["channel_data"].shape
        need = n_s * n_ev * 8
        use_full = need <= avail_bytes
        io_tasks.append((ch0_3_path, passing, use_full))

    if not any(use for _, _, use in io_tasks) and io_tasks:
        print("[ch3max-q] 单文件 CH3 超可用内存，改用按事件读取。")

    out_queue: Queue = Queue(maxsize=1)
    io_thread = threading.Thread(
        target=_io_worker,
        args=(io_tasks, out_queue),
        daemon=True,
    )
    io_thread.start()

    all_max: List[np.ndarray] = []
    all_q: List[np.ndarray] = []
    received = 0
    n_files = len(io_tasks)
    fig, ax, scatter = None, None, None

    while True:
        item = out_queue.get()
        if item[0] is None and item[1] is None:
            break
        received += 1
        if item[0] is None and isinstance(item[1], Exception):
            print(f"[ch3max-q] I/O 加载失败: {item[1]}")
            continue
        wf, n_s = item
        if wf.size == 0:
            continue
        ch_max, q = _compute_max_q_parallel(wf, n_compute)
        all_max.append(ch_max)
        all_q.append(q)
        # 主线程增量绘图（Matplotlib GUI 必须在主线程）
        if fig is None:
            plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
            plt.ion()
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.set_xlabel("CH3 max (ADC)", fontsize=12)
            ax.set_ylabel("CH3 q (Σ signal×t, ADC·ns, 4ns/sample)", fontsize=12)
            ax.grid(True, alpha=0.3)
            scatter = ax.scatter([], [], s=2, alpha=0.5, edgecolors="none")
            fig.tight_layout()
            plt.show(block=False)
        ch3_max = np.concatenate(all_max)
        ch3_q = np.concatenate(all_q)
        scatter.set_offsets(np.column_stack([ch3_max, ch3_q]))
        ax.set_title(f"CH3 max vs q (basic+act cuts passed, N={ch3_max.size}, {received}/{n_files} 文件)", fontsize=13)
        # set_offsets 后 relim/autoscale 不生效，需手动设置坐标范围
        xmin, xmax = ch3_max.min(), ch3_max.max()
        ymin, ymax = ch3_q.min(), ch3_q.max()
        pad_x = max((xmax - xmin) * 0.02, 1.0)
        pad_y = max((ymax - ymin) * 0.02, 1.0)
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        if received % 10 == 0 or received == n_files:
            print(f"[ch3max-q] 进度: {received}/{n_files} 文件")

    io_thread.join(timeout=2.0)

    if not all_max:
        print("[ch3max-q] 未获取到任何 CH3 max/q 数据，退出。")
        return

    ch3_max = np.concatenate(all_max)
    ch3_q = np.concatenate(all_q)
    print(f"[ch3max-q] 共得到 {ch3_max.size} 个 (max, q) 数据点。")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
