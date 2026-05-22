import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVENTMAP_PATH = PROJECT_ROOT / "data" / "hdf5" / "ge_30param_umap_hdbscan_eventmap.h5"
DEFAULT_CH0_3_DIR = PROJECT_ROOT / "data" / "hdf5" / "raw_pulse" / "CH0-3"


def _load_event_mapping(path: Path) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    读取 30parameter&HDBSCAN.py 写出的事件映射 HDF5（只保留本脚本所需字段）。

    - file_paths          : (n_files,)  base_name（通常是 CH0-3 的文件名）
    - event_file_indices  : (n_events,)
    - event_event_indices : (n_events,)
    - event_cluster_labels: (n_events,)
    """
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

def _group_indices_by_cluster(labels: np.ndarray) -> Dict[int, np.ndarray]:
    cluster_to_indices: Dict[int, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        cluster_to_indices[int(lab)].append(i)
    return {lab: np.asarray(idxs, dtype=np.int64) for lab, idxs in cluster_to_indices.items()}

def _extract_waveform_from_channel_data(
    ch_data: h5py.Dataset,
    event_idx: int,
    channel_idx: int,) -> np.ndarray | None:
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

def _extract_ch0_ch3_waveforms_fast(
    ch_data: h5py.Dataset,
    event_idx: int,
    ch0_index: int,
    ch3_index: int,) -> tuple[np.ndarray | None, np.ndarray | None]:
    # fast path: (time, channel, event)
    if ch_data.ndim == 3:
        try:
            if ch_data.shape[1] > max(ch0_index, ch3_index):
                wf0 = np.asarray(ch_data[:, ch0_index, event_idx], dtype=np.float64)
                wf3 = np.asarray(ch_data[:, ch3_index, event_idx], dtype=np.float64)
                if wf0.ndim == 1 and wf3.ndim == 1 and wf0.size > 0 and wf3.size > 0:
                    return wf0, wf3
        except Exception:
            pass

    wf0 = _extract_waveform_from_channel_data(ch_data, event_idx, ch0_index)
    wf3 = _extract_waveform_from_channel_data(ch_data, event_idx, ch3_index)
    return wf0, wf3

def plot_one_cluster_9_events(
    mapping_path: Path,
    ch0_3_dir: Path,
    cluster_label: int,
    max_events: int = 9,
    ch0_index: int = 0,
    ch3_index: int = 3,) -> None:
    file_paths, event_file_indices, event_event_indices, labels = _load_event_mapping(mapping_path)
    cluster_to_indices = _group_indices_by_cluster(labels)
    if cluster_label not in cluster_to_indices:
        uniq = sorted(cluster_to_indices.keys())
        raise ValueError(f"cluster_label={cluster_label} 不存在。可用 labels: {uniq}")

    idxs = cluster_to_indices[cluster_label]
    limit = min(int(max_events), 9)
    if idxs.size > limit:
        idxs = np.random.default_rng(42 + int(cluster_label)).choice(idxs, size=limit, replace=False)

    file_to_events: Dict[int, List[int]] = defaultdict(list)
    for gi in idxs:
        file_to_events[int(event_file_indices[gi])].append(int(event_event_indices[gi]))

    waveforms_ch0: List[np.ndarray] = []
    waveforms_ch3: List[np.ndarray] = []
    titles: List[str] = []

    for fi, ev_list in file_to_events.items():
        if fi < 0 or fi >= len(file_paths):
            continue
        fpath = _resolve_source_path(file_paths[fi], ch0_3_dir=ch0_3_dir)
        if not fpath.exists():
            continue

        with h5py.File(fpath, "r") as f_ch:
            if "channel_data" not in f_ch:
                continue
            ch_data = f_ch["channel_data"]
            for ev in ev_list:
                if ev < 0:
                    continue
                wf0, wf3 = _extract_ch0_ch3_waveforms_fast(
                    ch_data=ch_data,
                    event_idx=ev,
                    ch0_index=ch0_index,
                    ch3_index=ch3_index,
                )
                if wf0 is None or wf3 is None:
                    continue

                base_name = file_paths[fi]
                stem = Path(base_name).stem
                m = re.search(r"(\d+)(?!.*\d)", stem)
                run_id = m.group(1) if m else stem

                waveforms_ch0.append(wf0)
                waveforms_ch3.append(wf3)
                titles.append(f"{run_id} | #{ev}")

                if len(waveforms_ch0) >= limit:
                    break
            if len(waveforms_ch0) >= limit:
                break

    if not waveforms_ch0:
        raise RuntimeError(f"Cluster {cluster_label}: 未找到可用的 CH{ch0_index}/CH{ch3_index} 波形。")

    fig, axes = plt.subplots(3, 3, figsize=(14, 9), squeeze=False)
    for k in range(9):
        r, c = divmod(k, 3)
        ax = axes[r][c]
        if k >= len(waveforms_ch0):
            ax.axis("off")
            continue

        wf0 = waveforms_ch0[k]
        wf3 = waveforms_ch3[k]
        x0 = np.arange(wf0.size, dtype=np.int32)
        x3 = np.arange(wf3.size, dtype=np.int32)

        ax.plot(x0, wf0, color="C0", linewidth=0.8)
        ax.set_ylabel(f"CH{ch0_index}", color="C0", fontsize=9)
        ax.tick_params(axis="y", labelcolor="C0", labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, alpha=0.25)

        ax2 = ax.twinx()
        ax2.plot(x3, wf3, color="C1", linewidth=0.8, alpha=0.9)
        ax2.set_ylabel(f"CH{ch3_index}", color="C1", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="C1", labelsize=8)

        ax.set_title(titles[k], fontsize=8)

    fig.suptitle(f"Cluster {cluster_label} | shown {len(waveforms_ch0)} events", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="指定 cluster，快速绘制 9 个 event 的 CH0/CH3 波形（双 y 轴）。")
    p.add_argument(
        "hdf5_path",
        nargs="?",
        default=str(DEFAULT_EVENTMAP_PATH),
        help="事件映射 HDF5 路径（默认 data/hdf5/ge_30param_umap_hdbscan_eventmap.h5）。",
    )
    p.add_argument("--cluster-label", type=int, default=0, help="要绘制的 cluster label（默认 0）。")
    p.add_argument(
        "--ch0-3-dir",
        default=str(DEFAULT_CH0_3_DIR),
        help="CH0-3 原始波形目录（用于从 file_paths 定位源 h5 文件）。",
    )
    p.add_argument("--max-events", type=int, default=9, help="最多展示的 event 数（<=9）。")
    p.add_argument("--ch0", type=int, default=0, help="左轴通道索引（默认 0=CH0）。")
    p.add_argument("--ch3", type=int, default=3, help="右轴通道索引（默认 3=CH3）。")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    plot_one_cluster_9_events(
        mapping_path=Path(args.hdf5_path),
        ch0_3_dir=Path(args.ch0_3_dir).resolve(),
        cluster_label=int(args.cluster_label),
        max_events=int(args.max_events),
        ch0_index=int(args.ch0),
        ch3_index=int(args.ch3),
    )

if __name__ == "__main__":
    main()

