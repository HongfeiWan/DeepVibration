#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据 30parameter&HDBSCAN.py 生成的 CLUSTER_CACHE_PATH（事件映射 HDF5），
读取每个 cluster 中所有事件的“原始 30 参数”，并画出按 cluster 分组的参数分布。

使用方式（默认使用 30parameter&HDBSCAN.py 中的 CLUSTER_CACHE_PATH）:
    python 30parameter&HDBSCAN_distribution.py
或者手动指定其它 eventmap：
    python 30parameter&HDBSCAN_distribution.py path/to/eventmap.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path


def _discover_project_root() -> Path:
    """
    推断 DeepVibration 项目根目录：
    当前文件位于:
        .../python/data/ge-self/cut/parameterize/umap-optimazation-2/30parameter&HDBSCAN_distribution.py
    向上到 python，再上一层即项目根目录。
    """
    here = Path(__file__).resolve()
    # parents 索引（parents[i] 是第 i 层父目录）：
    # 0: umap-optimazation-2
    # 1: parameterize
    # 2: cut
    # 3: ge-self
    # 4: data
    # 5: python
    # 6: DeepVibration
    python_dir = here.parents[5]           # .../python
    return python_dir.parent               # .../DeepVibration

PROJECT_ROOT = _discover_project_root()
DATA_ROOT = PROJECT_ROOT / "data" / "hdf5" / "raw_pulse"

CH_PARAM_DIRS: Dict[int, Path] = {
    0: DATA_ROOT / "CH0_parameters",
    1: DATA_ROOT / "CH1_parameters",
    2: DATA_ROOT / "CH2_parameters",
    3: DATA_ROOT / "CH3_parameters",
    4: DATA_ROOT / "CH4_parameters",
    5: DATA_ROOT / "CH5_parameters",
}

# UMAP+HDBSCAN 结果缓存默认保存路径（事件映射 HDF5）
CLUSTER_CACHE_PATH = (
    PROJECT_ROOT
    / "data"
    / "hdf5"
    / "ge_30param_umap_hdbscan_eventmap.h5"
)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "根据 30 参数 UMAP+HDBSCAN 的事件映射 HDF5，"
            "读取各 cluster 中事件的所有原始参数，并画出按 cluster 分组的参数分布。"
        )
    )
    parser.add_argument(
        "eventmap_path",
        nargs="?",
        help=(
            "事件映射 HDF5 路径；若不指定，则使用 30parameter&HDBSCAN.py 中的 CLUSTER_CACHE_PATH。"
        ),
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=30,
        help="最多绘制多少个参数的分布（按特征名排序后截断，默认 30）。",
    )
    return parser.parse_args()

def _load_eventmap(path: Path) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"事件映射 HDF5 不存在: {path}")

    with h5py.File(path, "r") as f:
        file_paths_raw = f["file_paths"][...]
        event_file_indices = f["event_file_indices"][...]
        event_event_indices = f["event_event_indices"][...]
        labels = f["event_cluster_labels"][...]

    file_paths: List[str] = []
    for p in file_paths_raw:
        if isinstance(p, bytes):
            file_paths.append(p.decode("utf-8"))
        else:
            file_paths.append(str(p))

    return file_paths, event_file_indices, event_event_indices, labels

def _plot_umap_embedding_by_cluster(path: Path) -> None:
    """
    从事件映射 HDF5 中读取 UMAP 嵌入 (umap1, umap2) 和 cluster 标签，
    绘制每个 cluster 在 UMAP 平面上的分布图。
    """
    if not path.exists():
        print(f"[UMAP] 事件映射 HDF5 不存在，无法重建 UMAP 图: {path}")
        return

    with h5py.File(path, "r") as f:
        if "umap_embedding" not in f or "event_cluster_labels" not in f:
            print(f"[UMAP] {path} 中缺少 umap_embedding 或 event_cluster_labels，跳过 UMAP 绘图。")
            return
        embedding = np.asarray(f["umap_embedding"][...], dtype=np.float64)
        labels = np.asarray(f["event_cluster_labels"][...], dtype=int)

    if embedding.ndim != 2 or embedding.shape[1] != 2:
        print(f"[UMAP] umap_embedding 形状异常: {embedding.shape}，期望为 (n_events, 2)。")
        return
    if labels.shape[0] != embedding.shape[0]:
        print(
            f"[UMAP] labels 长度 {labels.shape[0]} 与 embedding 行数 {embedding.shape[0]} 不一致，"
            "跳过 UMAP 绘图。"
        )
        return

    # 全局字体风格（与分布图保持一致）
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
    })

    unique_labels = sorted(set(int(l) for l in labels))
    cmap = plt.cm.get_cmap("tab20")

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for ci, lab in enumerate(unique_labels):
        mask = labels == lab
        if not np.any(mask):
            continue
        color = "lightgray" if lab == -1 else cmap((ci % 20 + 0.5) / 20)
        label = "Noise" if lab == -1 else f"Cluster {lab}"
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=5,
            alpha=0.7 if lab != -1 else 0.4,
            color=color,
            label=label,
            edgecolors="none",
        )

    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title("UMAP embedding by cluster", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10, frameon=False)
    fig.tight_layout()
    plt.show()

def _load_all_parameters_for_events(
    file_paths: List[str],
    event_file_indices: np.ndarray,
    event_event_indices: np.ndarray,) -> Tuple[np.ndarray, List[str]]:
    """
    对 eventmap 中的每一个 (file_idx, event_idx)，从对应的 CH0_parameters ... CH5_parameters
    读取该 event 的所有 1D 参数（与 30parameter&HDBSCAN.py 中 load_run_parameters 一致），
    最终返回:
        - X_all: (n_events, n_features) 的特征矩阵
        - feature_names: 长度为 n_features 的特征名列表
    """
    # 先按照文件名分组事件索引，避免频繁打开/关闭 HDF5
    n_events = event_file_indices.shape[0]
    file_to_events: Dict[str, List[int]] = {}
    for i in range(n_events):
        fi = int(event_file_indices[i])
        if fi < 0 or fi >= len(file_paths):
            continue
        fname = file_paths[fi]
        file_to_events.setdefault(fname, []).append(i)

    per_event_features: List[np.ndarray] = []
    feature_names_ref: List[str] | None = None

    for base_name, global_event_indices in file_to_events.items():
        # global_event_indices: 这些是在“合并后” X_all 中的行号
        ev_local_indices = event_event_indices[global_event_indices].astype(int)

        # 参照 30parameter&HDBSCAN.py 中 load_run_parameters 的逻辑:
        per_channel_features: Dict[int, Dict[str, np.ndarray]] = {}
        n_events_in_run: int | None = None

        for ch, param_dir in CH_PARAM_DIRS.items():
            param_path = Path(param_dir) / base_name
            if not param_path.exists():
                continue
            with h5py.File(param_path, "r") as f:
                feats: Dict[str, np.ndarray] = {}
                for key, dset in f.items():
                    if not isinstance(dset, h5py.Dataset):
                        continue
                    if dset.ndim != 1:
                        continue
                    data = np.asarray(dset[...], dtype=np.float64)
                    if data.size == 0:
                        continue
                    if n_events_in_run is None:
                        n_events_in_run = int(data.shape[0])
                    elif data.shape[0] != n_events_in_run:
                        # 跟主脚本保持一致：长度不符就跳过该数据集
                        continue
                    feats[f"ch{ch}_{key}"] = data
                if feats:
                    per_channel_features[ch] = feats

        if n_events_in_run is None or not per_channel_features:
            continue

        # 拼成 (n_events_in_run, n_features_run) 的矩阵
        feature_names_run: List[str] = []
        feature_arrays_run: List[np.ndarray] = []
        for ch in sorted(per_channel_features.keys()):
            feats = per_channel_features[ch]
            for name in sorted(feats.keys()):
                feature_names_run.append(name)
                feature_arrays_run.append(feats[name].reshape(n_events_in_run, 1))

        X_run = np.concatenate(feature_arrays_run, axis=1)  # (n_events_in_run, n_features_run)

        # 记录/对齐特征名顺序
        if feature_names_ref is None:
            feature_names_ref = feature_names_run
        else:
            if feature_names_run != feature_names_ref:
                raise RuntimeError(
                    f"文件 {base_name} 的特征名顺序与之前不一致，无法统一绘制参数分布。"
                )

        # 只取当前 run 中 eventmap 涉及到的那些本地 event
        if np.any(ev_local_indices >= X_run.shape[0]):
            raise IndexError(
                f"在 {base_name} 中请求的本地事件号超出范围，"
                f"最大本地事件号 = {X_run.shape[0] - 1}"
            )
        per_event_features.append(X_run[ev_local_indices])

    if not per_event_features or feature_names_ref is None:
        raise RuntimeError("未能从任何参数文件中成功提取事件参数。")

    X_all = np.vstack(per_event_features)
    return X_all, feature_names_ref

def _plot_feature_distributions_by_cluster(
    X_all: np.ndarray,
    feature_names: List[str],
    labels: np.ndarray,
    max_features: int | None = None,
) -> None:
    """
    对所有可用特征（或指定前 max_features 个特征），画出“按 cluster 分组”的 1D 分布。
    按通道分组：CH0–CH5 各自一个窗口，每个窗口内部为该通道所有参数的子图。
    """
    unique_labels = sorted(set(int(l) for l in labels))
    cmap = plt.cm.get_cmap("tab20")

    # 全局字体风格，参考 README.md 中整体文档风格（使用衬线体）
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
    })

    # 预先构建每个通道的特征索引列表
    channel_to_indices: Dict[int, List[int]] = {ch: [] for ch in range(6)}
    for idx, name in enumerate(feature_names):
        # 名称形如 "ch0_xxx"
        if len(name) >= 4 and name.startswith("ch") and name[2].isdigit() and name[3] == "_":
            ch = int(name[2])
            if ch in channel_to_indices:
                channel_to_indices[ch].append(idx)

    for ch in range(6):
        feat_indices = channel_to_indices.get(ch, [])
        if not feat_indices:
            continue
        # 可选地根据 max_features 限制每个通道的特征数
        if max_features is not None and max_features > 0:
            feat_indices = feat_indices[:max_features]
        n_features = len(feat_indices)
        if n_features == 0:
            continue

        print(f"[分布] CH{ch} 将绘制 {n_features} 个特征的按 cluster 分布。")

        n_cols = int(np.ceil(np.sqrt(n_features)))
        n_rows = int(np.ceil(n_features / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes = np.atleast_1d(axes).ravel()

        for local_idx, feat_idx in enumerate(feat_indices):
            feat_name = feature_names[feat_idx]
            ax = axes[local_idx]

            # CH2/CH3 仍使用 log-log 坐标
            use_log_xy = feat_name.startswith("ch2_") or feat_name.startswith("ch3_")

            for ci, lab in enumerate(unique_labels):
                mask = labels == lab
                if not np.any(mask):
                    continue
                data = X_all[mask, feat_idx]
                if use_log_xy:
                    data = data[data > 0]
                    if data.size == 0:
                        continue
                color = "lightgray" if lab == -1 else cmap((ci % 20 + 0.5) / 20)
                label = "Noise" if lab == -1 else f"Cluster {lab}"

                ax.hist(
                    data,
                    bins=50,
                    histtype="step",
                    linewidth=1.0,
                    alpha=0.9 if lab != -1 else 0.6,
                    color=color,
                    label=label,
                    density=True,
                )

            if use_log_xy:
                ax.set_xscale("log")
                ax.set_yscale("log")

            ax.set_xlabel(feat_name, fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.set_title(feat_name, fontsize=11)
            ax.grid(True, alpha=0.3)

        # 隐藏多余子图
        for j in range(n_features, len(axes)):
            axes[j].set_visible(False)

        # 在第一个子图内添加统一图例
        handles, labels_legend = [], []
        for ax in axes[:n_features]:
            h, l = ax.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                if ll not in labels_legend:
                    handles.append(hh)
                    labels_legend.append(ll)
        if handles:
            axes[0].legend(
                handles,
                labels_legend,
                loc="upper right",
                fontsize=10,
                frameon=False,
            )

        fig.suptitle(f"CH{ch} feature distributions by cluster", fontsize=14)
        fig.tight_layout()
        plt.show()

def main() -> None:
    args = _parse_args()

    # 1. 决定要使用的 eventmap HDF5 路径
    if args.eventmap_path:
        eventmap_path = Path(args.eventmap_path)
        if not eventmap_path.is_absolute():
            # 若为相对路径，则解释为“相对于当前脚本所在目录”
            script_dir = Path(__file__).resolve().parent
            eventmap_path = (script_dir / eventmap_path).resolve()
    else:
        eventmap_path = CLUSTER_CACHE_PATH

    print(f"[分布] 使用事件映射 HDF5: {eventmap_path}")

    # 2. 从 eventmap 加载 (file_paths, event_file_indices, event_event_indices, labels)
    file_paths, event_file_indices, event_event_indices, labels = _load_eventmap(eventmap_path)

    # 3. 从各个 CH*_parameters 文件中重建这些事件的 30 维参数
    X_all, feature_names = _load_all_parameters_for_events(
        file_paths=file_paths,
        event_file_indices=event_file_indices,
        event_event_indices=event_event_indices,
    )

    if labels.shape[0] != X_all.shape[0]:
        raise RuntimeError(
            f"labels 长度 {labels.shape[0]} 与 X_all 行数 {X_all.shape[0]} 不一致，"
            "请检查 eventmap 与参数文件是否对应一致。"
        )

    # 4. 绘制不同 cluster 的参数分布（所有可用特征）
    _plot_feature_distributions_by_cluster(
        X_all=X_all,
        feature_names=feature_names,
        labels=labels,
        max_features=args.max_features,
    )

    # 5. 重建并绘制 UMAP1 vs UMAP2 的 cluster 分布
    _plot_umap_embedding_by_cluster(eventmap_path)

if __name__ == "__main__":
    main()

