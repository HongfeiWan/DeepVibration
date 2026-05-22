#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ch2ped-pedt cluster: 从 UMAP+HDBSCAN 的事件映射 HDF5 读取不同 cluster，
对每个事件直接从 CH2_parameters 读取已计算好的：
- ch2ped_mean
- ch2pedt_mean
并按 cluster 上色绘制二维散点图（一次性批量读取 + 一次性绘制）。

波形位置参考 python/data/preprocessor.py：
raw_pulse/CH0-3/*.h5 中 dataset "channel_data" 形状为 (n_samples, n_channels, n_events)，
其中 CH2 对应 channel index = 2。
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple
from collections import defaultdict

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _discover_project_root() -> Path:
    here = Path(__file__).resolve()
    python_dir = here.parents[3]
    return python_dir.parent


PROJECT_ROOT = _discover_project_root()
DATA_ROOT = PROJECT_ROOT / "data" / "hdf5" / "raw_pulse"
CH2_PARAM_DIR = DATA_ROOT / "CH2_parameters"

DEFAULT_EVENTMAP_PATH = PROJECT_ROOT / "data" / "hdf5" / "ge_30param_umap_hdbscan_eventmap.h5"

CH2_PED_MEAN_KEY = "ch2ped_mean"
CH2_PEDT_MEAN_KEY = "ch2pedt_mean"

def _parse_cluster_arg(values: Sequence[str]) -> List[int]:
    out: List[int] = []
    for v in values:
        for part in str(v).split(","):
            part = part.strip()
            if part == "":
                continue
            out.append(int(part))
    return out


def _load_eventmap(path: Path) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    读取事件映射 HDF5（ge_30param_umap_hdbscan_eventmap.h5）：
      - file_paths          : (n_files,)    base_name（通常是 CH0-3 文件名）
      - event_file_indices  : (n_events,)
      - event_event_indices : (n_events,)
      - event_cluster_labels: (n_events,)
    """
    if not path.exists():
        raise FileNotFoundError(f"事件映射 HDF5 不存在: {path}")

    with h5py.File(path, "r") as f:
        file_paths_raw = f["file_paths"][...]
        event_file_indices = np.asarray(f["event_file_indices"][...], dtype=np.int64)
        event_event_indices = np.asarray(f["event_event_indices"][...], dtype=np.int64)
        labels = np.asarray(f["event_cluster_labels"][...], dtype=np.int32)

    file_paths: List[str] = []
    for p in file_paths_raw:
        file_paths.append(p.decode("utf-8") if isinstance(p, (bytes, np.bytes_)) else str(p))

    return file_paths, event_file_indices, event_event_indices, labels


def _group_events_by_file(
    file_paths: List[str],
    event_file_indices: np.ndarray,
    event_event_indices: np.ndarray,
    labels: np.ndarray,
    target_clusters: Sequence[int] | None,
) -> Dict[Path, Tuple[np.ndarray, np.ndarray]]:
    """
    返回 {ch2_param_path: (event_indices, cluster_labels)}，两者等长且一一对应。
    """
    if target_clusters is None:
        mask = np.ones(labels.shape[0], dtype=bool)
    else:
        target_clusters = [int(x) for x in target_clusters]
        mask = np.isin(labels.astype(np.int32), np.asarray(target_clusters, dtype=np.int32))

    idx = np.where(mask)[0]
    if idx.size == 0:
        return {}

    by_file: DefaultDict[int, List[int]] = defaultdict(list)
    for gi in idx:
        fi = int(event_file_indices[gi])
        if fi < 0 or fi >= len(file_paths):
            continue
        by_file[fi].append(int(gi))

    tasks: Dict[Path, Tuple[np.ndarray, np.ndarray]] = {}
    for fi, gis in by_file.items():
        base_name = file_paths[fi]
        ch2_param_path = CH2_PARAM_DIR / base_name
        evs = event_event_indices[np.asarray(gis, dtype=np.int64)].astype(np.int64)
        labs = labels[np.asarray(gis, dtype=np.int64)].astype(np.int32)
        tasks[ch2_param_path] = (evs, labs)
    return tasks


def _phase2_worker(
    args: Tuple[Path, np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    单文件：按 event_indices 读取 ch2ped_mean/ch2pedt_mean，并返回与之对齐的 cluster_labels。
    返回 (labs, x, y)；失败/无有效点返回 None。
    """
    ch2_param_path, evs, labs = args
    if not ch2_param_path.exists():
        return None

    evs = np.asarray(evs, dtype=np.int64)
    labs = np.asarray(labs, dtype=np.int32)
    if evs.size == 0:
        return None

    try:
        with h5py.File(ch2_param_path, "r") as f:
            if CH2_PED_MEAN_KEY not in f or CH2_PEDT_MEAN_KEY not in f:
                return None
            dset_x = f[CH2_PED_MEAN_KEY]
            dset_y = f[CH2_PEDT_MEAN_KEY]
            n_ev = int(min(dset_x.shape[0], dset_y.shape[0]))
            if n_ev <= 0:
                return None

            ok = (evs >= 0) & (evs < n_ev)
            evs = evs[ok]
            labs = labs[ok]
            if evs.size == 0:
                return None

            x = np.asarray(dset_x[evs], dtype=np.float64)
            y = np.asarray(dset_y[evs], dtype=np.float64)
        return (labs, x, y)
    except Exception:
        return None

def _iter_unique_in_order(values: Iterable[int]) -> List[int]:
    seen: set[int] = set()
    out: List[int] = []
    for v in values:
        v = int(v)
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


_PALETTE6_RGB255: List[Tuple[int, int, int]] = [
    (230, 159, 0),   # orange
    (86, 180, 233),  # sky blue
    (0, 158, 115),   # bluish green
    (240, 228, 66),  # yellow
    (0, 114, 178),   # blue
    (213, 94, 0),    # vermillion
]


def _rgb255_to_mpl(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    r, g, b = rgb
    return (r / 255.0, g / 255.0, b / 255.0)


def _build_cluster_color_map(cluster_labels: Sequence[int]) -> Dict[int, Tuple[float, float, float] | str]:
    """
    将 cluster label 映射到 6 色高区分度调色板。
    - label=-1 (noise) 固定为灰色
    - 其它 label 按 cluster_labels 的顺序依次分配颜色（超过 6 会循环）
    """
    labs = [int(x) for x in cluster_labels if int(x) != -1]
    labs = sorted(set(labs))
    cmap: Dict[int, Tuple[float, float, float] | str] = {-1: "0.6"}
    for i, lab in enumerate(labs):
        cmap[lab] = _rgb255_to_mpl(_PALETTE6_RGB255[i % len(_PALETTE6_RGB255)])
    return cmap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="按 cluster 给 CH2 ped/pedt 二维散点上色，并固定左上角图例。"
    )
    p.add_argument(
        "--eventmap",
        default=str(DEFAULT_EVENTMAP_PATH),
        help="事件映射 HDF5 路径（默认 data/hdf5/ge_30param_umap_hdbscan_eventmap.h5）。",
    )
    p.add_argument(
        "--clusters",
        nargs="*",
        default=["3", "6"],
        help="要绘制的 cluster 列表，支持空格分隔或逗号分隔（例如: --clusters 3 4 或 --clusters 3,4）。默认: 3 4。",
    )
    p.add_argument(
        "--include-noise",
        action="store_true",
        help="是否包含 HDBSCAN noise（label=-1）。默认不包含。",
    )
    p.add_argument(
        "--max-points-per-cluster",
        type=int,
        default=0,
        help="每个 cluster 最多绘制多少点（0 表示不限制）。用于快速预览。",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="散点透明度。",
    )
    p.add_argument(
        "--s",
        type=float,
        default=2.0,
        help="散点大小。",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    file_paths, event_file_indices, event_event_indices, labels = _load_eventmap(Path(args.eventmap))
    target_clusters: List[int] = _parse_cluster_arg(args.clusters)
    if not args.include_noise:
        target_clusters = [c for c in target_clusters if int(c) != -1]

    # 为颜色分配建立稳定映射（按 label 排序）
    color_map = _build_cluster_color_map(target_clusters)

    tasks_by_file = _group_events_by_file(
        file_paths=file_paths,
        event_file_indices=event_file_indices,
        event_event_indices=event_event_indices,
        labels=labels,
        target_clusters=target_clusters,
    )
    if not tasks_by_file:
        print("[ch2ped-pedt cluster] 没有匹配到任何事件，退出。")
        return

    n_total = int(sum(int(evs.size) for evs, _ in tasks_by_file.values()))
    print(f"[ch2ped-pedt cluster] 共 {n_total} 个事件，分布在 {len(tasks_by_file)} 个文件中。")

    per_cluster_x: DefaultDict[int, List[np.ndarray]] = defaultdict(list)
    per_cluster_y: DefaultDict[int, List[np.ndarray]] = defaultdict(list)
    max_per = int(args.max_points_per_cluster)
    rng = np.random.default_rng(42)

    workers = max(1, os.cpu_count() or 1)
    print(f"[ch2ped-pedt cluster] 并行读取 CH2_parameters（仅目标 cluster 的 event），进程数: {workers}")

    tasks_list: List[Tuple[Path, np.ndarray, np.ndarray]] = [
        (p, evs, labs) for p, (evs, labs) in tasks_by_file.items()
    ]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_phase2_worker, t): t[0] for t in tasks_list}
        done = 0
        for fut in as_completed(futures):
            out = fut.result()
            done += 1
            if out is None:
                continue
            labs_arr, x_arr, y_arr = out
            if x_arr.size == 0:
                continue
            # 将该文件内事件按 cluster 分桶
            for lab in _iter_unique_in_order(labs_arr.tolist()):
                idx = np.where(labs_arr == int(lab))[0]
                if idx.size == 0:
                    continue
                per_cluster_x[int(lab)].append(x_arr[idx])
                per_cluster_y[int(lab)].append(y_arr[idx])
            if done % 20 == 0 or done == len(futures):
                n_acc = int(sum(int(np.concatenate(v).size) if len(v) > 1 else int(v[0].size) for v in per_cluster_x.values() if v))
                print(f"[ch2ped-pedt cluster] 进度: {done}/{len(futures)} 文件, 已累计点数 {n_acc}")

    clusters_to_plot = sorted(per_cluster_x.keys())
    if not clusters_to_plot:
        print("[ch2ped-pedt cluster] 未获取到任何 ped/pedt 数据，退出。")
        return

    total_points = int(
        sum(
            int(np.concatenate(per_cluster_x[lab]).size)
            if len(per_cluster_x[lab]) > 1
            else int(per_cluster_x[lab][0].size)
            for lab in clusters_to_plot
        )
    )
    print(f"[ch2ped-pedt cluster] 完成：共绘制 {total_points} 个点，cluster 数 {len(clusters_to_plot)}。")

    # 若限制每个 cluster 点数：在全量拼接后做一次抽样（更快更简单）
    final_x: Dict[int, np.ndarray] = {}
    final_y: Dict[int, np.ndarray] = {}
    for lab in clusters_to_plot:
        xs = np.concatenate(per_cluster_x[lab]) if len(per_cluster_x[lab]) > 1 else per_cluster_x[lab][0]
        ys = np.concatenate(per_cluster_y[lab]) if len(per_cluster_y[lab]) > 1 else per_cluster_y[lab][0]
        if max_per > 0 and xs.size > max_per:
            pick = rng.choice(xs.size, size=max_per, replace=False)
            xs = xs[pick]
            ys = ys[pick]
        final_x[lab] = xs
        final_y[lab] = ys

    title_clusters = ",".join(str(x) for x in target_clusters) if target_clusters else "(none)"
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_xlabel(f"CH2 ped mean ({CH2_PED_MEAN_KEY}) (ADC)", fontsize=12)
    ax.set_ylabel(f"CH2 pedt mean ({CH2_PEDT_MEAN_KEY}) (ADC)", fontsize=12)
    ax.grid(True, alpha=0.3)

    for lab in clusters_to_plot:
        xs = final_x[lab]
        ys = final_y[lab]
        ax.scatter(
            xs,
            ys,
            s=float(args.s),
            alpha=float(args.alpha),
            edgecolors="none",
            color=color_map.get(int(lab), _rgb255_to_mpl(_PALETTE6_RGB255[int(lab) % 6])),
            label=f"cluster {lab} (N={xs.size})",
        )

    ax.set_title(
        f"CH2 ped vs pedt mean (cluster colored, clusters={title_clusters})",
        fontsize=13,
    )
    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            legend_labels,
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            borderaxespad=0.0,
            frameon=True,
            fontsize=9,
            ncols=1,
        )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

