#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从全量 eventmap 中筛出“剩余事件”，逐个显示 CH0/CH1/CH2/CH3 四宫格波形。

默认排除规则（对应你的描述）：
- 排除 A：step2_cluster0 里的全部事件（简称 “CH0-0 内所有事例”）
- 排除 B：step2_cluster1_step_3_cluster2 里 label ∈ {2,3} 的事件（简称 “CH0-1-CH1-2-CH3-2/3”）

关掉当前图窗后自动显示下一个事件（plt.show() 阻塞，窗口关闭即继续）。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HDF5_DIR = PROJECT_ROOT / "data" / "hdf5"
DEFAULT_RAW_CH0_3_DIR = DEFAULT_HDF5_DIR / "raw_pulse" / "CH0-3"

DEFAULT_BASE_EVENTMAP = DEFAULT_HDF5_DIR / "ge_30param_umap_hdbscan_eventmap.h5"
DEFAULT_EXCLUDE_STEP2_CLUSTER0 = DEFAULT_HDF5_DIR / "ge_30param_umap_hdbscan_eventmap_step2_cluster0.h5"
DEFAULT_EXCLUDE_STEP3 = DEFAULT_HDF5_DIR / "ge_30param_umap_hdbscan_eventmap_step2_cluster1_step_3_cluster2.h5"


def _apply_plotstyle_font() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
        }
    )


def _resolve_source_path(path_entry: str, ch0_3_dir: Path) -> Path:
    p = Path(path_entry)
    return p if p.is_absolute() else (ch0_3_dir / p)


def _load_eventmap(path: Path) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"eventmap 不存在: {path}")
    with h5py.File(path, "r") as f:
        file_paths_raw = f["file_paths"][...]
        event_file_indices = f["event_file_indices"][...].astype(np.int64, copy=False)
        event_event_indices = f["event_event_indices"][...].astype(np.int64, copy=False)
        labels = f["event_cluster_labels"][...].astype(np.int64, copy=False)

    file_paths: List[str] = []
    for p in file_paths_raw:
        file_paths.append(p.decode("utf-8") if isinstance(p, (bytes, np.bytes_)) else str(p))
    return file_paths, event_file_indices, event_event_indices, labels


EventKey = Tuple[str, int]  # (resolved_file_path_str, event_idx)


def _iter_eventkeys_from_eventmap(
    eventmap_path: Path,
    *,
    ch0_3_dir: Path,
    keep_labels: Optional[Set[int]] = None,
) -> Iterator[EventKey]:
    file_paths, ef, ee, labels = _load_eventmap(eventmap_path)
    n = int(labels.size)
    for i in range(n):
        lab = int(labels[i])
        if keep_labels is not None and lab not in keep_labels:
            continue
        fi = int(ef[i])
        ev = int(ee[i])
        if fi < 0 or ev < 0 or fi >= len(file_paths):
            continue
        fpath = _resolve_source_path(file_paths[fi], ch0_3_dir=ch0_3_dir)
        if not fpath.exists():
            continue
        yield (str(fpath), ev)


def _eventkey_set(
    eventmap_path: Path,
    *,
    ch0_3_dir: Path,
    keep_labels: Optional[Set[int]] = None,
) -> Set[EventKey]:
    return set(_iter_eventkeys_from_eventmap(eventmap_path, ch0_3_dir=ch0_3_dir, keep_labels=keep_labels))


def _read_one_event_ch0_3(
    f: h5py.File,
    event_idx: int,
    *,
    clip_len: Optional[int],
) -> Optional[np.ndarray]:
    """
    返回 shape (4, T) 的 float32；失败返回 None。
    优先使用 preprocessor 常见布局 (time, channel, event) 的切片 [:, ch, ev]。
    """
    if "channel_data" not in f:
        return None
    ch_data = f["channel_data"]
    if ch_data.ndim != 3:
        return None

    # 要绘制 CH0..CH3（0..3）
    if ch_data.shape[1] < 4:
        return None

    ev = int(event_idx)
    max_reasonable_waveform_len = 200000

    # candidate layouts
    getters = (
        lambda: np.stack([np.asarray(ch_data[:, ch, ev]) for ch in range(4)], axis=0),  # (ch, time)
        lambda: np.stack([np.asarray(ch_data[ev, ch, :]) for ch in range(4)], axis=0),
        lambda: np.stack([np.asarray(ch_data[ev, :, ch]) for ch in range(4)], axis=0),
        lambda: np.stack([np.asarray(ch_data[:, ev, ch]) for ch in range(4)], axis=0),
    )
    arr: Optional[np.ndarray] = None
    for g in getters:
        try:
            a = g()
        except Exception:
            continue
        if a.ndim != 2 or a.shape[0] != 4:
            continue
        if a.shape[1] <= 1 or a.shape[1] > max_reasonable_waveform_len:
            continue
        arr = a
        break

    if arr is None:
        return None

    if clip_len is not None and int(clip_len) > 0:
        arr = arr[:, : int(clip_len)]
    return arr.astype(np.float32, copy=False)


class _OpenFileCache:
    def __init__(self) -> None:
        self.current_path: Optional[str] = None
        self.current_file: Optional[h5py.File] = None

    def close(self) -> None:
        if self.current_file is not None:
            try:
                self.current_file.close()
            finally:
                self.current_file = None
                self.current_path = None

    def get(self, path_str: str) -> h5py.File:
        if self.current_file is not None and self.current_path == path_str:
            return self.current_file
        self.close()
        self.current_path = path_str
        self.current_file = h5py.File(Path(path_str), "r")
        return self.current_file


def _plot_event_quad(
    wf4: np.ndarray,
    *,
    title: str,
    xlim: Optional[Tuple[int, int]],
    ylim: Optional[Tuple[float, float]],
) -> None:
    _apply_plotstyle_font()
    plt.rcParams.setdefault("axes.unicode_minus", False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), squeeze=False)
    t = np.arange(wf4.shape[1], dtype=np.int32)
    for ch in range(4):
        r, c = divmod(ch, 2)
        ax = axes[r][c]
        ax.plot(t, wf4[ch], linewidth=0.9)
        ax.set_title(f"CH{ch}", fontsize=12)
        ax.grid(True, alpha=0.25)
        if xlim is not None:
            ax.set_xlim(int(xlim[0]), int(xlim[1]))
        if ylim is not None:
            ax.set_ylim(float(ylim[0]), float(ylim[1]))

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="排除指定集合后，逐个显示剩余事件的 CH0/1/2/3 四宫格波形（关窗到下一个）。"
    )
    p.add_argument("--ch0-3-dir", default=str(DEFAULT_RAW_CH0_3_DIR), help="raw_pulse/CH0-3 目录。")
    p.add_argument("--base-eventmap", default=str(DEFAULT_BASE_EVENTMAP), help="全量 eventmap（全集）。")

    p.add_argument(
        "--exclude-step2-cluster0",
        default=str(DEFAULT_EXCLUDE_STEP2_CLUSTER0),
        help="排除集合 A：step2_cluster0 的 eventmap（其内全部事件都会排除）。",
    )
    p.add_argument(
        "--exclude-step3",
        default=str(DEFAULT_EXCLUDE_STEP3),
        help="排除集合 B：step3 eventmap（默认 step2_cluster1_step_3_cluster2）。",
    )
    p.add_argument(
        "--exclude-step3-labels",
        type=int,
        nargs="*",
        default=[2, 3],
        help="在 --exclude-step3 指定的 eventmap 内，仅排除这些 label 的事件（默认 2 3）。",
    )

    p.add_argument("--clip-len", type=int, default=30000, help="每条波形截断长度（<=0 表示不截断）。")
    p.add_argument("--start", type=int, default=0, help="从剩余列表的第几个开始看（用于断点续看）。")
    p.add_argument("--limit", type=int, default=0, help="最多看多少个（0 表示不限制）。")
    p.add_argument("--xlim", type=int, nargs=2, default=[0, 30000], help="x 轴范围。")
    p.add_argument("--ylim", type=float, nargs=2, default=[0.0, 16383.0], help="y 轴范围。")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ch0_3_dir = Path(args.ch0_3_dir).resolve()
    base_eventmap = Path(args.base_eventmap).resolve()
    excl_a_path = Path(args.exclude_step2_cluster0).resolve()
    excl_b_path = Path(args.exclude_step3).resolve()

    clip_len = None if int(args.clip_len) <= 0 else int(args.clip_len)
    keep_labels_b: Optional[Set[int]] = None
    if args.exclude_step3_labels is not None:
        labs = [int(x) for x in args.exclude_step3_labels]
        keep_labels_b = set(labs)

    base_set = _eventkey_set(base_eventmap, ch0_3_dir=ch0_3_dir)
    excl_a = _eventkey_set(excl_a_path, ch0_3_dir=ch0_3_dir)
    excl_b = _eventkey_set(excl_b_path, ch0_3_dir=ch0_3_dir, keep_labels=keep_labels_b)
    remaining = sorted(base_set - excl_a - excl_b)

    print(f"[全集] {len(base_set)}")
    print(f"[排除A step2_cluster0] {len(excl_a)}")
    print(f"[排除B step3 labels={sorted(list(keep_labels_b)) if keep_labels_b else None}] {len(excl_b)}")
    print(f"[剩余] {len(remaining)}")

    start = max(0, int(args.start))
    if start >= len(remaining):
        print(f"[结束] start={start} >= 剩余事件数 {len(remaining)}")
        return
    limit = int(args.limit)
    if limit > 0:
        remaining = remaining[start : start + limit]
    else:
        remaining = remaining[start:]

    xlim = (int(args.xlim[0]), int(args.xlim[1])) if args.xlim else None
    ylim = (float(args.ylim[0]), float(args.ylim[1])) if args.ylim else None

    cache = _OpenFileCache()
    try:
        for k, (fpath_str, ev) in enumerate(remaining, start=start):
            try:
                f = cache.get(fpath_str)
                wf4 = _read_one_event_ch0_3(f, ev, clip_len=clip_len)
            except Exception as e:
                print(f"[跳过] {k}: {Path(fpath_str).name} ev={ev} 读取失败: {e}")
                continue
            if wf4 is None:
                print(f"[跳过] {k}: {Path(fpath_str).name} ev={ev} 无法提取 CH0-3")
                continue

            title = f"[{k}] {Path(fpath_str).name} | event={ev}"
            _plot_event_quad(wf4, title=title, xlim=xlim, ylim=ylim)
    finally:
        cache.close()


if __name__ == "__main__":
    main()

