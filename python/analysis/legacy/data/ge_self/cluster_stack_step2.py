#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
针对 30parameter&HDBSCAN_step2.py 生成的事件映射 HDF5（如
ge_30param_umap_hdbscan_eventmap_step2_cluster0.h5），
按 **命令行指定的子 cluster**（`--cluster`，可多个）依次绘制，流程与
`cluster_stack.py` 一致：每个 cluster 单独读波形、单独弹窗，CH 默认 CH1。

不一次性预取全部 cluster，避免大缓存与长时间阻塞。
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVENTMAP_PATH = (
    PROJECT_ROOT
    / "data"
    / "hdf5"
    / "ge_30param_umap_hdbscan_eventmap_step2_cluster1.h5"
)
DEFAULT_CH0_3_DIR = PROJECT_ROOT / "data" / "hdf5" / "raw_pulse" / "CH0-3"
DEFAULT_OUT_DIR = PROJECT_ROOT / "images"
# CH1 在 preprocessor 保存的 channel_data 中为索引 1（CH0=0, CH1=1, ...）
DEFAULT_CHANNEL_INDEX = 1

# 与 cluster_stack.py、python/utils/plotstyle.md 一致
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


def _cluster_tag_for_filename(clab: int) -> str:
    if int(clab) < 0:
        return f"m{abs(int(clab))}"
    return str(int(clab))


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


def _load_waveforms_for_file_task(
    task: Tuple[str, List[int], int, int | None, int, int],
) -> np.ndarray:
    path_str, event_indices, channel_idx, clip_len, plot_decimate, h5_rdcc_nbytes = task
    fpath = Path(path_str)
    if not fpath.exists():
        return np.empty((0, 0), dtype=np.float32)

    ev = np.asarray(sorted(set(int(x) for x in event_indices if int(x) >= 0)), dtype=np.int64)
    if ev.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    kw: dict = {}
    if int(h5_rdcc_nbytes) > 0:
        kw["rdcc_nbytes"] = int(h5_rdcc_nbytes)

    with h5py.File(fpath, "r", **kw) as f:
        if "channel_data" not in f:
            return np.empty((0, 0), dtype=np.float32)
        ch_data = f["channel_data"]
        if ch_data.ndim != 3:
            return np.empty((0, 0), dtype=np.float32)

        ev_max = int(ev.max())
        if channel_idx < ch_data.shape[1] and ev_max < ch_data.shape[2]:
            raw = ch_data[:, channel_idx, ev]
            arr = np.asarray(raw)
            block = arr.astype(np.float32, copy=False).T
        elif ev_max < ch_data.shape[0] and channel_idx < ch_data.shape[1]:
            block = np.asarray(ch_data[ev, channel_idx, :]).astype(np.float32, copy=False)
        elif ev_max < ch_data.shape[0] and channel_idx < ch_data.shape[2]:
            block = np.asarray(ch_data[ev, :, channel_idx]).astype(np.float32, copy=False)
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


def collect_cluster_waveform_matrix(
    ch0_3_dir: Path,
    cluster_label: int,
    channel_idx: int,
    file_paths: List[str],
    event_file_indices: np.ndarray,
    event_event_indices: np.ndarray,
    labels: np.ndarray,
    max_events: int = 2000,
    seed: int = 42,
    stride: int = 1,
    clip_len: int | None = None,
    per_event_zscore: bool = False,
    plot_decimate: int = 1,
    read_workers: int = 0,
    max_files: int = 0,
    file_pick: str = "random",
    h5_rdcc_nbytes: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    与 cluster_stack.py 中按 cluster 读波形、组任务、并行读块逻辑对齐。
    返回 (t, X)，X 形状 (n_event, T)。
    """
    idxs_all = np.where(labels.astype(np.int64) == int(cluster_label))[0].astype(np.int64)
    if idxs_all.size == 0:
        return np.array([], dtype=np.int32), np.empty((0, 0), dtype=np.float32)

    if max_events is not None and idxs_all.size > int(max_events):
        rng = np.random.default_rng(int(seed) + int(cluster_label))
        idxs = rng.choice(idxs_all, size=int(max_events), replace=False)
    else:
        idxs = idxs_all

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

    rdcc = max(0, int(h5_rdcc_nbytes))
    tasks: List[Tuple[str, List[int], int, int | None, int, int]] = []
    for fi, ev_list in file_to_events.items():
        if fi < 0 or fi >= len(file_paths):
            continue
        fpath = _resolve_source_path(file_paths[fi], ch0_3_dir=ch0_3_dir)
        if not fpath.exists():
            continue
        tasks.append((str(fpath), ev_list, int(channel_idx), clip_len, int(plot_decimate), rdcc))

    if not tasks:
        return np.array([], dtype=np.int32), np.empty((0, 0), dtype=np.float32)

    if int(max_files) > 0 and len(tasks) > int(max_files):
        mf = int(max_files)
        pick_mode = str(file_pick).strip().lower()
        if pick_mode == "first":
            tasks = tasks[:mf]
        else:
            rng = np.random.default_rng(int(seed) + 100000 + int(cluster_label))
            sel = rng.choice(len(tasks), size=mf, replace=False)
            tasks = [tasks[i] for i in np.sort(sel).tolist()]

    if int(read_workers) <= 0:
        read_workers = os.cpu_count() or 1
    read_workers = max(1, int(read_workers))

    blocks: List[np.ndarray] = []
    done = 0
    with ProcessPoolExecutor(max_workers=read_workers) as ex:
        futs = [ex.submit(_load_waveforms_for_file_task, t) for t in tasks]
        for fut in as_completed(futs):
            arr = fut.result()
            if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] > 1:
                blocks.append(arr)
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"[读取进度] {done}/{len(tasks)}")

    if not blocks:
        return np.array([], dtype=np.int32), np.empty((0, 0), dtype=np.float32)

    min_len = min(int(b.shape[1]) for b in blocks)
    if min_len <= 1:
        return np.array([], dtype=np.int32), np.empty((0, 0), dtype=np.float32)

    X = np.vstack([b[:, :min_len] for b in blocks]).astype(np.float32, copy=False)
    if per_event_zscore:
        mu = X.mean(axis=1, keepdims=True)
        sig = X.std(axis=1, keepdims=True)
        sig[sig == 0] = 1.0
        X = (X - mu) / sig

    t = np.arange(min_len, dtype=np.int32)
    return t, X


def plot_step2_windows_ch1(
    mapping_path: Path,
    ch0_3_dir: Path,
    cluster_labels: List[int],
    channel_idx: int = DEFAULT_CHANNEL_INDEX,
    max_events: int = 2000,
    seed: int = 42,
    stride: int = 1,
    clip_len: int | None = None,
    per_event_zscore: bool = False,
    offset: float = 0.0,
    alpha: float = 0.08,
    linewidth: float = 0.6,
    plot_decimate: int = 1,
    read_workers: int = 0,
    max_files: int = 0,
    file_pick: str = "random",
    h5_rdcc_nbytes: int = 0,
    out_prefix: Optional[Path] = None,
) -> None:
    """
    与 cluster_stack.py 一致：每个 cluster 单独读盘、单独弹窗。
    `cluster_labels` 为要绘制的子 cluster 编号列表（可含 -1）。
    """
    file_paths, event_file_indices, event_event_indices, labels = _load_event_mapping(mapping_path)
    uniq = set(np.unique(labels.astype(np.int64)).tolist())
    for clab in cluster_labels:
        if int(clab) not in uniq:
            raise ValueError(
                f"cluster {clab} 不在本 eventmap 中。可用标签: {sorted(uniq)}"
            )

    n_total = len(cluster_labels)
    print(f"[信息] 将依次绘制 {n_total} 个 cluster: {cluster_labels}（与 cluster_stack 相同读法）。")

    t0 = time.perf_counter()
    stem = mapping_path.stem

    for idx, clab in enumerate(cluster_labels):
        t, X = collect_cluster_waveform_matrix(
            ch0_3_dir=ch0_3_dir,
            cluster_label=int(clab),
            channel_idx=channel_idx,
            file_paths=file_paths,
            event_file_indices=event_file_indices,
            event_event_indices=event_event_indices,
            labels=labels,
            max_events=max_events,
            seed=seed,
            stride=stride,
            clip_len=clip_len,
            per_event_zscore=per_event_zscore,
            plot_decimate=plot_decimate,
            read_workers=read_workers,
            max_files=max_files,
            file_pick=file_pick,
            h5_rdcc_nbytes=h5_rdcc_nbytes,
        )

        _apply_plotstyle_font()
        plt.rcParams.setdefault("axes.unicode_minus", False)

        fig_w, fig_h = 6.0, 6.0
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

        if X.size == 0:
            ax.text(
                0.5,
                0.5,
                f"Cluster {clab}: 无波形",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=_PLOT_AXIS,
            )
            ax.set_axis_off()
        else:
            if offset != 0.0:
                X_plot = X + (float(offset) * np.arange(X.shape[0], dtype=np.float64)[:, None])
            else:
                X_plot = X
            ax.plot(t, X_plot.T, color="C0", alpha=float(alpha), linewidth=float(linewidth))
            noise_note = " (noise)" if int(clab) < 0 else ""
            ax.set_title(
                f"{stem} | Cluster {clab}{noise_note} | CH{channel_idx} stacked (n={X.shape[0]})",
                fontsize=_PLOT_TITLE,
            )
            ax.set_xlabel("Sample index", fontsize=_PLOT_AXIS)
            ax.set_ylabel(
                "ADC (stacked)" if offset != 0.0 else "ADC (overlay)",
                fontsize=_PLOT_AXIS,
            )
            ax.tick_params(axis="both", which="major", labelsize=_PLOT_TICK)
            ax.set_xlim(0, 30000)
            ax.set_ylim(0, 16383)

        fig.tight_layout()

        if out_prefix is not None:
            tag = _cluster_tag_for_filename(int(clab))
            out_path = Path(f"{out_prefix}_cluster_{tag}.png")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=220)
            print(f"[保存] {out_path}")

        plt.show()
        print(f"[窗口 {idx + 1}/{n_total}] cluster={clab} 已关闭或继续下一窗。")

    t1 = time.perf_counter()
    print(f"[完成] 共 {n_total} 个独立窗口, 总耗时 {t1 - t0:.2f} s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="step2 eventmap：指定子 cluster，与 cluster_stack 相同方式读波形并弹窗（默认 CH1）。",
    )
    p.add_argument(
        "hdf5_path",
        nargs="?",
        default=str(DEFAULT_EVENTMAP_PATH),###记得检查这里是否存的是对的。
        help="step2 写出的事件映射（默认 ge_30param_umap_hdbscan_eventmap_step2_cluster0.h5）。",
    )
    p.add_argument(
        "--cluster",
        type=int,
        nargs="*",
        dest="clusters",
        default=[2],
        metavar="LABEL",
        help="要绘制的子 cluster，可多个，如 --cluster 0 1 2；默认 [0]。噪声为 -1。单独写 --cluster 无参数则回退为 [0]。",
    )
    p.add_argument("--ch0-3-dir", default=str(DEFAULT_CH0_3_DIR), help="CH0-3 原始波形目录。")
    p.add_argument(
        "--channel",
        type=int,
        default=DEFAULT_CHANNEL_INDEX,
        help="channel_data 通道索引（默认 1=CH1）。",
    )
    p.add_argument("--max-events", type=int, default=200, help="每个 cluster 最多绘制事件数（与 cluster_stack 一致）。")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--clip-len", type=int, default=0)
    p.add_argument("--per-event-zscore", action="store_true")
    p.add_argument("--offset", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=0.08)
    p.add_argument("--linewidth", type=float, default=0.6)
    p.add_argument("--plot-decimate", type=int, default=1)
    p.add_argument("--read-workers", type=int, default=0)
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument(
        "--file-pick",
        choices=("random", "first"),
        default="random",
        help="当 --max-files 生效时，源文件选择策略（与 cluster_stack 默认 random 一致）。",
    )
    p.add_argument("--h5-rdcc-mb", type=int, default=0)
    p.add_argument(
        "--out-prefix",
        default="",
        help="若设置则每个 cluster 保存 PNG：{prefix}_cluster_<id>.png（-1 为 _cluster_m1.png）。",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    clip_len = None if int(args.clip_len) <= 0 else int(args.clip_len)
    out_prefix: Optional[Path] = None
    if str(args.out_prefix).strip():
        out_prefix = Path(args.out_prefix)
        if not out_prefix.is_absolute():
            out_prefix = DEFAULT_OUT_DIR / out_prefix

    # nargs='*' 且用户只写 --cluster 时可能为 []，回退为 [0]
    clusters: List[int] = list(args.clusters) if args.clusters else [0]

    plot_step2_windows_ch1(
        mapping_path=Path(args.hdf5_path).resolve(),
        ch0_3_dir=Path(args.ch0_3_dir).resolve(),
        cluster_labels=clusters,
        channel_idx=int(args.channel),
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
        out_prefix=out_prefix,
    )


if __name__ == "__main__":
    main()
