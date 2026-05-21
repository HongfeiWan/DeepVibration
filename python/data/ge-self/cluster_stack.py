#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
读取某一个 cluster 的事件，并把该 cluster 的所有 CH0 波形堆在一张图里绘制。

数据来源：
- 事件映射 HDF5：data/hdf5/ge_30param_umap_hdbscan_eventmap.h5
  由 30parameter&HDBSCAN.py 生成，包含每个事件对应的 (base_name, event_idx, cluster_label)。
- 原始波形 HDF5：data/hdf5/raw_pulse/CH0-3/<base_name>
  其中包含 channel_data 数据集。

与 preprocessor.py 中 bin→h5 的约定一致（见该文件 `channel_data` 分配与保存）：
  - 形状：(n_samples, n_channels_saved, n_events)，即 (时间样本, 通道, 事件序号)
  - 典型 dtype：uint16（采集原始 ADC）
  - CH0 对应 channel_list 中索引 0，故本脚本用 `channel_data[:, 0, event_idx]` 批量切片读 CH0 是最顺应该磁盘布局的读法。
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVENTMAP_PATH = PROJECT_ROOT / "data" / "hdf5" / "ge_30param_umap_hdbscan_eventmap.h5"
DEFAULT_CH0_3_DIR = PROJECT_ROOT / "data" / "hdf5" / "raw_pulse" / "CH0-3"
DEFAULT_OUT_DIR = PROJECT_ROOT / "images"

# 与 python/utils/plotstyle.md 一致（同 cut/parameterize/tradition/tradition.py）
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


def _load_event_mapping(path: Path) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"事件映射 HDF5 不存在: {path}")
    with h5py.File(path, "r") as f:
        file_paths_raw = f["file_paths"][...]
        event_file_indices = f["event_file_indices"][...]
        event_event_indices = f["event_event_indices"][...]
        labels = f["event_cluster_labels"][...]
    file_paths: List[str] = []
    for p in file_paths_raw:
        file_paths.append(p.decode("utf-8") if isinstance(p, bytes) else str(p))
    return file_paths, event_file_indices, event_event_indices, labels


def _resolve_source_path(path_entry: str, ch0_3_dir: Path) -> Path:
    p = Path(path_entry)
    return p if p.is_absolute() else (ch0_3_dir / p)


def _extract_waveform_from_channel_data(
    ch_data: h5py.Dataset,
    event_idx: int,
    channel_idx: int,
) -> np.ndarray | None:
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


def _extract_waveforms_batch(
    ch_data: h5py.Dataset,
    event_indices: List[int],
    channel_idx: int,
) -> List[np.ndarray]:
    """
    批量提取同一文件内多个 event 的同一通道波形。
    优先走常见布局 (time, channel, event) 的批量切片，失败再回退到逐条提取。
    """
    if not event_indices:
        return []
    ev = np.asarray(sorted(set(int(x) for x in event_indices if int(x) >= 0)), dtype=np.int64)
    if ev.size == 0:
        return []

    # fast path: (time, channel, event)
    if ch_data.ndim == 3:
        try:
            if channel_idx < ch_data.shape[1] and ev.max() < ch_data.shape[2]:
                block = np.asarray(ch_data[:, channel_idx, ev], dtype=np.float64)  # (time, n_ev)
                if block.ndim == 2 and block.size > 0:
                    return [block[:, i] for i in range(block.shape[1])]
        except Exception:
            pass

    # fallback: 逐 event
    out: List[np.ndarray] = []
    for e in ev.tolist():
        wf = _extract_waveform_from_channel_data(ch_data, event_idx=e, channel_idx=channel_idx)
        if wf is not None:
            out.append(wf)
    return out


def _effective_read_clip(clip_len: int | None, read_max_samples: int | None) -> int | None:
    """读取阶段截断长度：显式 clip_len 优先，否则用 read_max_samples（>0）。"""
    if clip_len is not None and int(clip_len) > 0:
        return int(clip_len)
    if read_max_samples is not None and int(read_max_samples) > 0:
        return int(read_max_samples)
    return None


def _draw_overlay_linecollections(
    ax,
    t: np.ndarray,
    X_plot: np.ndarray,
    *,
    alpha: float,
    linewidth: float,
    chunk_rows: int,
) -> None:
    """
    分块 LineCollection 叠画多条波形：比「每条线一个 Line2D」更省 artist 与路径开销；
    分块避免一次性构造 (n, T, 2) 巨型数组导致内存翻倍。
    """
    n, T = X_plot.shape
    if n == 0 or T == 0:
        return
    t_c = np.ascontiguousarray(t, dtype=np.float32)
    chunk_rows = max(8, int(chunk_rows))
    for i0 in range(0, n, chunk_rows):
        i1 = min(i0 + chunk_rows, n)
        k = i1 - i0
        seg = np.empty((k, T, 2), dtype=np.float32)
        seg[:, :, 0] = t_c
        seg[:, :, 1] = np.ascontiguousarray(X_plot[i0:i1], dtype=np.float32)
        lc = LineCollection(
            seg,
            colors="C0",
            alpha=float(alpha),
            linewidths=float(linewidth),
            rasterized=True,
        )
        ax.add_collection(lc)


def _load_waveforms_for_file_task(
    task: Tuple[str, List[int], int, int | None, int, int],
) -> np.ndarray:
    """
    子进程任务：读取单个文件内若干 event 的单通道波形，返回形状 (n_event, T)。
    优先使用 preprocessor 布局 (time, channel, event) 的一次切片 `[:, ch, ev[]]`，
    避免逐 event 重复打开/索引 HDF5。
    """
    path_str, event_indices, channel_idx, clip_len, plot_decimate, h5_rdcc_nbytes = task
    fpath = Path(path_str)
    if not fpath.exists():
        return np.empty((0, 0), dtype=np.float32)

    ev = np.asarray(sorted(set(int(x) for x in event_indices if int(x) >= 0)), dtype=np.int64)
    if ev.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    # rdcc_nbytes：放大原始块读缓存，对大块连续切片有时能略减 HDF5 层开销（0 则用 h5py 默认）
    kw: dict = {}
    if int(h5_rdcc_nbytes) > 0:
        kw["rdcc_nbytes"] = int(h5_rdcc_nbytes)

    with h5py.File(fpath, "r", **kw) as f:
        if "channel_data" not in f:
            return np.empty((0, 0), dtype=np.float32)
        ch_data = f["channel_data"]
        if ch_data.ndim != 3:
            return np.empty((0, 0), dtype=np.float32)

        block: np.ndarray | None = None
        ev_max = int(ev.max())

        # preprocessor 默认布局：(time, channel, event) — 见 preprocessor.py channel_data 形状注释
        if channel_idx < ch_data.shape[1] and ev_max < ch_data.shape[2]:
            raw = ch_data[:, channel_idx, ev]
            arr = np.asarray(raw)
            # 磁盘多为 uint16；先读原生类型再转 float32，避免无谓的中间类型
            block = arr.astype(np.float32, copy=False).T
        # (event, channel, time)
        elif ev_max < ch_data.shape[0] and channel_idx < ch_data.shape[1]:
            block = np.asarray(ch_data[ev, channel_idx, :]).astype(np.float32, copy=False)
        # (event, time, channel)
        elif ev_max < ch_data.shape[0] and channel_idx < ch_data.shape[2]:
            block = np.asarray(ch_data[ev, :, channel_idx]).astype(np.float32, copy=False)
        # (time, event, channel)
        elif ev_max < ch_data.shape[1] and channel_idx < ch_data.shape[2]:
            block = np.asarray(ch_data[:, ev, channel_idx]).astype(np.float32, copy=False).T
        else:
            return np.empty((0, 0), dtype=np.float32)

    if block.ndim != 2 or block.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    if clip_len is not None and int(clip_len) > 0:
        block = block[:, : int(clip_len)]
    plot_decimate = max(1, int(plot_decimate))
    if plot_decimate > 1:
        block = block[:, ::plot_decimate]
    return block


def plot_cluster_ch0_stack(
    mapping_path: Path,
    ch0_3_dir: Path,
    cluster_label: int,
    ch0_index: int = 0,
    max_events: int = 2000,
    seed: int = 42,
    stride: int = 1,
    clip_len: int | None = None,
    read_max_samples: int | None = 30000,
    per_event_zscore: bool = False,
    offset: float = 0.0,
    alpha: float = 0.08,
    linewidth: float = 0.6,
    plot_decimate: int = 1,
    read_workers: int = 0,
    read_backend: str = "process",
    max_files: int = 0,
    file_pick: str = "random",
    h5_rdcc_nbytes: int = 0,
    viz_mode: str = "overlay",
    line_chunk_rows: int = 256,
    save_dpi: int = 220,
    save_path: Path | None = None,
) -> None:
    t0 = time.perf_counter()
    file_paths, event_file_indices, event_event_indices, labels = _load_event_mapping(mapping_path)

    idxs_all = np.where(labels.astype(np.int64) == int(cluster_label))[0].astype(np.int64)
    if idxs_all.size == 0:
        uniq = np.unique(labels.astype(np.int64)).tolist()
        raise ValueError(f"cluster_label={cluster_label} 不存在或为空。可用 labels: {sorted(uniq)}")

    # 事件过多时抽样，避免一次画到爆
    if max_events is not None and idxs_all.size > int(max_events):
        rng = np.random.default_rng(int(seed) + int(cluster_label))
        idxs = rng.choice(idxs_all, size=int(max_events), replace=False)
    else:
        idxs = idxs_all

    # stride 再降采样（在 idxs 上等距取）
    stride = max(1, int(stride))
    if stride > 1:
        idxs = np.sort(idxs)[::stride]

    file_to_events: Dict[int, List[int]] = defaultdict(list)
    for gi in idxs:
        fi = int(event_file_indices[gi])
        ev = int(event_event_indices[gi])
        if fi < 0 or ev < 0:
            continue
        file_to_events[fi].append(ev)

    eff_clip = _effective_read_clip(clip_len, read_max_samples)

    rdcc = max(0, int(h5_rdcc_nbytes))
    tasks: List[Tuple[str, List[int], int, int | None, int, int]] = []
    for fi, ev_list in file_to_events.items():
        if fi < 0 or fi >= len(file_paths):
            continue
        fpath = _resolve_source_path(file_paths[fi], ch0_3_dir=ch0_3_dir)
        if not fpath.exists():
            continue
        tasks.append((str(fpath), ev_list, int(ch0_index), eff_clip, int(plot_decimate), rdcc))

    if not tasks:
        raise RuntimeError(f"Cluster {cluster_label}: 没有可读取的源文件任务。")

    # 控制读取文件总数，避免长时间无响应
    if int(max_files) > 0 and len(tasks) > int(max_files):
        max_files = int(max_files)
        pick_mode = str(file_pick).strip().lower()
        if pick_mode == "first":
            tasks = tasks[:max_files]
        else:
            # 默认 random：均匀随机取部分文件（可复现）
            rng = np.random.default_rng(int(seed) + 100000 + int(cluster_label))
            sel = rng.choice(len(tasks), size=max_files, replace=False)
            tasks = [tasks[i] for i in np.sort(sel).tolist()]
        print(f"[读取] 已启用文件数限制: {len(tasks)}/{len(file_to_events)} (mode={pick_mode})")

    if int(read_workers) <= 0:
        read_workers = os.cpu_count() or 1
    read_workers = max(1, int(read_workers))
    backend = str(read_backend).strip().lower()
    use_threads = backend in ("thread", "threads")
    if eff_clip is not None:
        print(f"[读取] 时间长度截断: {eff_clip} 点（减轻 HDF5 读量与内存）")
    print(
        f"[读取] 文件任务数: {len(tasks)}, 并行{'线程' if use_threads else '进程'}数: {read_workers}",
    )

    blocks: List[np.ndarray] = []
    done = 0
    Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    with Executor(max_workers=read_workers) as ex:
        futs = [ex.submit(_load_waveforms_for_file_task, t) for t in tasks]
        for fut in as_completed(futs):
            arr = fut.result()
            if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] > 1:
                blocks.append(arr)
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"[读取进度] {done}/{len(tasks)}")

    if not blocks:
        raise RuntimeError(f"Cluster {cluster_label}: 未找到任何可用 CH{ch0_index} 波形。")

    # 统一长度（否则 overlay 会错位），取最短长度
    min_len = min(int(b.shape[1]) for b in blocks)
    if min_len <= 1:
        raise RuntimeError("波形长度异常（<=1），无法绘制。")
    X = np.vstack([b[:, :min_len] for b in blocks]).astype(np.float32, copy=False)  # (n, T)
    if per_event_zscore:
        mu = X.mean(axis=1, keepdims=True)
        sig = X.std(axis=1, keepdims=True)
        sig[sig == 0] = 1.0
        X = (X - mu) / sig

    t = np.arange(min_len, dtype=np.int32)

    _apply_plotstyle_font()
    plt.rcParams.setdefault("axes.unicode_minus", False)

    fig_w = 12
    fig_h = 6 if offset == 0 else max(6, 0.02 * X.shape[0])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    if offset != 0.0:
        X_plot = X + (float(offset) * np.arange(X.shape[0], dtype=np.float64)[:, None])
    else:
        X_plot = X
    # 单次 plot 画多条线，明显减少 Python 层循环开销
    ax.plot(t, X_plot.T, color="C0", alpha=float(alpha), linewidth=float(linewidth))

    ax.set_title(
        f"Cluster {cluster_label} | CH{ch0_index} stacked (n={X.shape[0]})",
        fontsize=_PLOT_TITLE,
    )
    ax.set_xlabel("Sample index", fontsize=_PLOT_AXIS)
    ax.set_ylabel(
        "ADC (stacked)" if offset != 0.0 else "ADC (overlay)",
        fontsize=_PLOT_AXIS,
    )
    ax.tick_params(axis="both", which="major", labelsize=_PLOT_TICK)
    ax.set_xlim(0,30000)
    ax.set_ylim(0,16383)

    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220)
        print(f"[完成] 已保存: {save_path}")

    t1 = time.perf_counter()
    print(f"[完成] 共绘制 {X.shape[0]} 条波形, 每条长度 {X.shape[1]}, 总耗时 {t1 - t0:.2f} s")
    plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="按 cluster 读取事件并把所有 CH0 波形堆叠绘制在一张图里。")
    p.add_argument(
        "hdf5_path",
        nargs="?",
        default=str(DEFAULT_EVENTMAP_PATH),
        help="事件映射 HDF5 路径（默认 data/hdf5/ge_30param_umap_hdbscan_eventmap.h5）。",
    )
    p.add_argument("--cluster-label", type=int, default=-1, help="要绘制的 cluster label。")
    p.add_argument("--ch0-3-dir", default=str(DEFAULT_CH0_3_DIR), help="CH0-3 原始波形目录。")
    p.add_argument("--ch0", type=int, default=0, help="通道索引（默认 0=CH0）。")
    p.add_argument("--max-events", type=int, default=2000, help="最多绘制多少个事件（过多会很慢）。")
    p.add_argument("--seed", type=int, default=42, help="抽样随机种子。")
    p.add_argument("--stride", type=int, default=1, help="在事件列表上每隔 stride 取一个（进一步降采样）。")
    p.add_argument("--clip-len", type=int, default=0, help="截取波形长度（0 表示不截取）。")
    p.add_argument("--per-event-zscore", action="store_true", help="每条波形单独 z-score（便于形状比较）。")
    p.add_argument("--offset", type=float, default=0.0, help="每条波形的纵向偏移量（>0 更像“堆叠/瀑布图”）。")
    p.add_argument("--alpha", type=float, default=0.08, help="透明度。")
    p.add_argument("--linewidth", type=float, default=0.6, help="线宽。")
    p.add_argument("--plot-decimate", type=int, default=1, help="绘图时在时间轴上每隔 N 点取一个（加速渲染）。")
    p.add_argument("--read-workers", type=int, default=0, help="并行读取进程数（0 表示使用全部 CPU 核）。")
    p.add_argument("--max-files", type=int, default=0, help="最多读取多少个源文件（0 表示不限制）。")
    p.add_argument(
        "--file-pick",
        choices=("random", "first"),
        default="random",
        help="当 --max-files 生效时，文件选择策略：random 或 first。",
    )
    p.add_argument(
        "--h5-rdcc-mb",
        type=int,
        default=0,
        help="每个进程打开 HDF5 时 raw data chunk cache 大小（MB，0=h5py 默认）。大文件可试 256~512。",
    )
    p.add_argument(
        "--out",
        default="",
        help="保存路径（默认不保存）。例如 images/cluster1_ch0_stack.png",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    clip_len = None if int(args.clip_len) <= 0 else int(args.clip_len)
    out_path = None
    if str(args.out).strip():
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = DEFAULT_OUT_DIR / out_path

    plot_cluster_ch0_stack(
        mapping_path=Path(args.hdf5_path),
        ch0_3_dir=Path(args.ch0_3_dir).resolve(),
        cluster_label=int(args.cluster_label),
        ch0_index=int(args.ch0),
        max_events=int(args.max_events),
        seed=int(args.seed),
        stride=int(args.stride),
        clip_len=clip_len,
        per_event_zscore=bool(args.per_event_zscore),
        offset=float(args.offset),
        alpha=float(args.alpha),
        linewidth=float(args.linewidth),
        plot_decimate=int(args.plot_decimate),
        read_workers=int(args.read_workers),
        max_files=int(args.max_files),
        file_pick=str(args.file_pick),
        h5_rdcc_nbytes=int(args.h5_rdcc_mb) * 1024 * 1024,
        save_path=out_path,
    )


if __name__ == "__main__":
    main()

