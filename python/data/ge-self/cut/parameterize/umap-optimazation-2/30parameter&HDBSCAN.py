#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为每个 event 汇总 CH0–CH5 的所有已计算参数（共 ~30 维），
后续可用于 UMAP 降维和 HDBSCAN 聚类。

本脚本当前只完成：
1. 匹配同名的参数文件：
   - data/hdf5/raw_pulse/CH0_parameters
   - data/hdf5/raw_pulse/CH1_parameters
   - data/hdf5/raw_pulse/CH2_parameters
   - data/hdf5/raw_pulse/CH3_parameters
   - data/hdf5/raw_pulse/CH4_parameters
   - data/hdf5/raw_pulse/CH5_parameters
2. 从这些 h5 中读取每个 event 的全部参数，并在内存中拼成
   (n_events, n_features) 的特征矩阵，附带特征名列表。
后续再在此基础上做 UMAP + HDBSCAN。
"""

from __future__ import annotations
import concurrent.futures
import os
import sys
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

@dataclass
class RunParameters:
    base_name: str
    n_events: int
    feature_matrix: np.ndarray          # 形状: (n_events, n_features)
    feature_names: List[str]            # 长度 = n_features

def _discover_project_root() -> Path:
    """
    推断 DeepVibration 项目根目录：
    当前文件位于:
        .../python/data/ge-self/cut/parameterize/umap-optimazation-2/30parameter&HDBSCAN.py
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
CH0_3_DIR = DATA_ROOT / "CH0-3"

CH_PARAM_DIRS: Dict[int, Path] = {
    0: DATA_ROOT / "CH0_parameters",
    1: DATA_ROOT / "CH1_parameters",
    2: DATA_ROOT / "CH2_parameters",
    3: DATA_ROOT / "CH3_parameters",
    4: DATA_ROOT / "CH4_parameters",
    5: DATA_ROOT / "CH5_parameters",}

# UMAP+HDBSCAN 结果缓存默认保存路径（事件映射 HDF5）
CLUSTER_CACHE_PATH = (
    PROJECT_ROOT
    / "data"
    / "hdf5"
    / "ge_30param_umap_hdbscan_eventmap.h5"
)

def _save_cluster_eventmap_hdf5(
    path: Path,
    all_sources: List[Tuple[str, int]],
    embedding: np.ndarray,
    labels: np.ndarray,) -> None:
    """
    将 UMAP+HDBSCAN 的结果保存为事件映射 HDF5：
        - file_paths          : (n_files,)    唯一文件路径（字符串）
        - event_file_indices  : (n_events,)   每个事件对应的 file_paths 索引
        - event_event_indices : (n_events,)   每个事件在原始文件中的 event 号
        - event_cluster_labels: (n_events,)   HDBSCAN 聚类标签
        - umap_embedding      : (n_events, 2) UMAP 2D 嵌入坐标
    方便后续脚本（例如能谱、波形可视化）直接基于该 HDF5 继续分析。
    """
    if embedding.shape[0] != len(all_sources) or labels.shape[0] != len(all_sources):
        raise ValueError("embedding/labels 长度与 all_sources 不一致，无法保存事件映射 HDF5。")

    # 先将 (base_name, local_event_idx) 映射到 file_paths 索引
    file_to_idx: Dict[str, int] = {}
    file_paths: List[bytes] = []
    event_file_indices = np.empty(len(all_sources), dtype=np.int64)
    event_event_indices = np.empty(len(all_sources), dtype=np.int64)

    for i, (base_name, ev_idx) in enumerate(all_sources):
        if base_name not in file_to_idx:
            file_to_idx[base_name] = len(file_paths)
            file_paths.append(base_name.encode("utf-8"))
        fi = file_to_idx[base_name]
        event_file_indices[i] = fi
        event_event_indices[i] = int(ev_idx)

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("file_paths", data=np.asarray(file_paths, dtype="S"))
        f.create_dataset("event_file_indices", data=event_file_indices)
        f.create_dataset("event_event_indices", data=event_event_indices)
        f.create_dataset("event_cluster_labels", data=labels.astype(np.int32))
        f.create_dataset("umap_embedding", data=embedding.astype(np.float32))

    print(f"[缓存] 已将 UMAP+HDBSCAN 结果保存到: {path}")

def _plot_waveforms_from_cluster_cache(
    path: Path,
    ch0_3_dir: Path,
    sampling_interval_ns: float = 4.0,) -> None:
    """
    从缓存的事件映射 HDF5 读取 cluster 结果，对每个 cluster 随机抽样画 CH0/CH3 波形。
    这样在已经有缓存的情况下，可以跳过 UMAP/HDBSCAN 直接可视化。
    """
    if not path.exists():
        print(f"[缓存] 事件映射 HDF5 不存在，跳过可视化: {path}")
        return

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

    n_events = labels.shape[0]
    print(f"[缓存] 从 {path} 读取到 {n_events} 个事件的聚类结果。")

    clusters_to_plot = sorted(set(labels.tolist()))
    sampling_interval_ns = float(sampling_interval_ns)

    for lab in clusters_to_plot:
        idx_in_cluster = np.where(labels == lab)[0]
        if idx_in_cluster.size == 0:
            continue

        n_sample = min(9, idx_in_cluster.size)
        rng = np.random.default_rng(42 + int(lab))
        sample_idx = rng.choice(idx_in_cluster, size=n_sample, replace=False)

        fig_wf, axes = plt.subplots(3, 3, figsize=(12, 10))
        axes = axes.ravel()

        for i, global_idx in enumerate(sample_idx):
            ax_left = axes[i]
            fi = int(event_file_indices[global_idx])
            ev_idx = int(event_event_indices[global_idx])
            if fi < 0 or fi >= len(file_paths):
                ax_left.set_visible(False)
                continue

            base_name = file_paths[fi]
            ch0_3_path = ch0_3_dir / base_name
            if not ch0_3_path.exists():
                ax_left.set_visible(False)
                continue

            with h5py.File(ch0_3_path, "r") as f_ch:
                ch_data = f_ch["channel_data"]
                time_samples, num_channels, num_events = ch_data.shape
                if num_channels <= 3 or ev_idx >= num_events:
                    ax_left.set_visible(False)
                    continue
                time_us = np.arange(time_samples) * sampling_interval_ns / 1000.0
                wf_ch0 = ch_data[:, 0, ev_idx].astype(np.float64)
                wf_ch3 = ch_data[:, 3, ev_idx].astype(np.float64)

            stem = Path(base_name).stem
            m = re.search(r"(\d+)(?!.*\d)", stem)
            if m:
                run_id = m.group(1)
            else:
                run_id = stem

            ax_right = ax_left.twinx()
            ax_left.plot(time_us, wf_ch0, color="C0", linewidth=0.8, label="CH0")
            ax_right.plot(time_us, wf_ch3, color="C3", linewidth=0.8, label="CH3")

            ax_left.set_title(f"{run_id} | #{ev_idx}", fontsize=10)
            ax_left.set_xlabel("Time (µs)", fontsize=12)
            ax_left.set_ylabel("CH0 ADC", fontsize=12)
            ax_right.set_ylabel("CH3 ADC", fontsize=12)

            ax_left.tick_params(axis="both", which="major", labelsize=10)
            ax_right.tick_params(axis="y", which="major", labelsize=10, colors="C3")
            ax_left.grid(True, alpha=0.3)

            lines_left, labels_left = ax_left.get_legend_handles_labels()
            lines_right, labels_right = ax_right.get_legend_handles_labels()
            ax_left.legend(
                lines_left + lines_right,
                labels_left + labels_right,
                fontsize=8,
            )

        for j in range(n_sample, 9):
            axes[j].set_visible(False)

        lab_name = "Noise" if lab == -1 else f"Cluster {lab}"
        fig_wf.suptitle(
            f"All Runs — {lab_name}  (n={idx_in_cluster.size}, shown {n_sample})",
            fontsize=14,
        )
        fig_wf.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

def _list_base_names_from_ch0() -> List[str]:
    """
    以 CH0_parameters 目录为“主目录”，列出所有存在的 run 文件名（不含路径）。
    后续用这些 base_name 去其他通道的参数目录中匹配同名文件。
    """
    ch0_dir = CH_PARAM_DIRS[0]
    if not ch0_dir.exists():
        raise FileNotFoundError(f"CH0_parameters 目录不存在: {ch0_dir}")

    base_names: List[str] = []
    for name in sorted(os.listdir(ch0_dir)):
        if name.lower().endswith((".h5", ".hdf5")):
            base_names.append(name)
    if not base_names:
        raise FileNotFoundError(f"CH0_parameters 目录 {ch0_dir} 下未找到任何 h5 文件")
    return base_names

def _open_param_file_if_exists(ch: int, base_name: str) -> Path | None:
    """
    在对应通道参数目录下查找与 base_name 同名的 h5 文件。
    若存在则返回完整路径，否则返回 None。
    """
    dir_path = CH_PARAM_DIRS[ch]
    path = dir_path / base_name
    if path.exists():
        return path
    return None

def _read_all_1d_datasets(
    path: Path,
    prefix: str,) -> Tuple[Dict[str, np.ndarray], int]:
    """
    从给定 h5 文件中读取所有一维数据集，返回:
    - 字典: {f"{prefix}{dataset_name}": ndarray(shape=(n_events,))}
    - 事件数 n_events（若文件为空或没有 1D 数据集，则为 0）

    只读取 dtype 可转换为 float 或整数类型的一维数据。
    """
    features: Dict[str, np.ndarray] = {}
    n_events = 0

    with h5py.File(path, "r") as f:
        for key, dset in f.items():
            if not isinstance(dset, h5py.Dataset):
                continue
            if dset.ndim != 1:
                continue
            data = np.asarray(dset[...])
            if data.size == 0:
                continue

            # 记录事件数，并确保所有 1D 数据集长度一致
            if n_events == 0:
                n_events = int(data.shape[0])
            elif data.shape[0] != n_events:
                print(
                    f"[警告] 文件 {path.name} 中数据集 {key} 长度 "
                    f"{data.shape[0]} != 预期 {n_events}，跳过该数据集。"
                )
                continue
            # 统一转为 float64，便于后续数值处理
            data_f = data.astype(np.float64)
            feat_name = f"{prefix}{key}"
            features[feat_name] = data_f
    return features, n_events

def load_run_parameters(base_name: str) -> RunParameters:
    """
    读取某个 run 的所有通道参数（CH0–CH5），并拼接成统一的特征矩阵。

    - 以 CH0_parameters 中的 base_name 为起点；
    - 对于每个通道 ch:
        若存在对应参数文件，则读取该文件中所有一维数据集作为特征；
        每个特征名加上 `chX_` 前缀（例如 ch0_max_ch0, ch2_tanh_p0）。
    - 所有通道的特征在特征维度上拼接，确保 event 维度对齐。
    """
    per_channel_features: Dict[int, Dict[str, np.ndarray]] = {}
    n_events_global: int | None = None

    for ch in range(6):
        path = _open_param_file_if_exists(ch, base_name)
        if path is None:
            print(f"[信息] 通道 CH{ch} 的参数文件缺失: {base_name}，跳过该通道。")
            continue

        prefix = f"ch{ch}_"
        feats, n_events = _read_all_1d_datasets(path, prefix=prefix)
        if not feats:
            print(f"[信息] 通道 CH{ch} 参数文件 {path.name} 中没有有效 1D 参数，跳过。")
            continue

        if n_events_global is None:
            n_events_global = n_events
        elif n_events != n_events_global:
            raise ValueError(
                f"文件 {base_name} 中通道 CH{ch} 的事件数 {n_events} "
                f"与其他通道不一致（预期 {n_events_global}）。"
            )

        per_channel_features[ch] = feats

    if n_events_global is None or not per_channel_features:
        raise RuntimeError(f"运行 {base_name} 未能收集到任何通道的参数。")

    # 按通道顺序 / 特征名顺序拼接成 (n_events, n_features)
    feature_names: List[str] = []
    feature_arrays: List[np.ndarray] = []

    for ch in sorted(per_channel_features.keys()):
        feats = per_channel_features[ch]
        for name in sorted(feats.keys()):
            feature_names.append(name)
            feature_arrays.append(feats[name].reshape(n_events_global, 1))

    feature_matrix = np.concatenate(feature_arrays, axis=1)  # (n_events, n_features)

    return RunParameters(
        base_name=base_name,
        n_events=n_events_global,
        feature_matrix=feature_matrix,
        feature_names=feature_names,
    )

def run_umap_hdbscan(
    features: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    min_cluster_size: int = 50,
    min_samples: int = None,
    prediction_data: bool = False,) -> Tuple[np.ndarray, np.ndarray]:
    """
    在给定的特征矩阵上执行 UMAP 降维和 HDBSCAN 聚类。
    返回:
        embedding: (n_events, 2) 的二维 UMAP 嵌入
        labels:    (n_events,) 的 HDBSCAN 聚类标签（-1 表示噪声）
    prediction_data: 若为 True，HDBSCAN 会预计算供 approximate_predict 使用的数据。
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="euclidean",
        random_state=42,     # 固定随机种子，确保结果可复现
        low_memory=True,     # 防止爆内存
        force_approximation_algorithm=True,
        verbose=True,
        n_jobs=-1           # 使用所有可用 CPU 核并行加速 UMAP
    )
    embedding = reducer.fit_transform(features)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        core_dist_n_jobs=-1,  # 使用所有 CPU 核并行计算 core distances
        prediction_data=prediction_data,  # 为 approximate_predict 准备数据
    )
    labels = clusterer.fit_predict(embedding)
    return embedding, labels


def _fit_umap_hdbscan_on_sample(
    X_sampled: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    min_cluster_size: int = 50,
    min_samples: int = None,
) -> Tuple[umap.UMAP, hdbscan.HDBSCAN, np.ndarray, np.ndarray]:
    """
    仅在抽样子集上拟合 UMAP 和 HDBSCAN，返回训练好的模型，供后续外推使用。
    HDBSCAN 固定使用 prediction_data=True，以支持 approximate_predict。
    返回: (reducer, clusterer, embedding_sampled, labels_sampled)
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="euclidean",
        random_state=42,
        low_memory=True,
        force_approximation_algorithm=True,
        verbose=True,
        n_jobs=-1,
    )
    embedding_s = reducer.fit_transform(X_sampled)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        core_dist_n_jobs=-1,
        prediction_data=True,  # 必需，否则 approximate_predict 无法使用
    )
    labels_s = clusterer.fit_predict(embedding_s)
    return reducer, clusterer, embedding_s, labels_s


def _resolve_extrapolate_n_jobs(n_jobs: int) -> int:
    """n_jobs=-1 表示使用全部逻辑 CPU；0 或 1 表示单进程顺序外推。"""
    if n_jobs == 0 or n_jobs == 1:
        return 1
    if n_jobs < 0:
        return max(1, os.cpu_count() or 1)
    return max(1, int(n_jobs))


def _split_index_ranges(n: int, n_parts: int) -> List[Tuple[int, int]]:
    """将 [0, n) 均分为 n_parts 段，返回 [(start, end), ...]，end 为开区间。"""
    n_parts = min(max(1, n_parts), n)
    edges = np.linspace(0, n, n_parts + 1, dtype=np.int64)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_parts)]


def _extrapolate_one_chunk_worker(
    args: Tuple[umap.UMAP, KNeighborsClassifier, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    多进程 worker：对一块 X_remaining 做 UMAP transform + KNN 预测。
    必须在模块顶层以便 pickle/spawn。
    """
    reducer, knn, X_chunk = args
    emb_batch = reducer.transform(X_chunk)
    lab_batch = knn.predict(emb_batch).astype(np.int32)
    return emb_batch.astype(np.float32), lab_batch


def extrapolate_cluster_labels(
    reducer: umap.UMAP,
    embedding_train: np.ndarray,
    labels_train: np.ndarray,
    X_remaining: np.ndarray,
    batch_size: int = 50000,
    knn_neighbors: int = 15,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对未参与训练的样本进行 UMAP transform，再用 KNN 在训练集嵌入上投票分配 cluster 标签。
    UMAP transform 会把新点投影到训练嵌入的 periphery，导致 approximate_predict 几乎全判为 noise。
    改用 KNN：根据外推点在 2D 嵌入空间中最近的 k 个训练点的标签做多数投票，更符合物理预期。

    n_jobs: 外推并行进程数。-1 为使用全部 CPU；1 为单进程按 batch_size 顺序处理（便于调试）。
    多进程时 KNN 内部 n_jobs=1，避免与多进程争抢线程。
    """
    # 进程内单线程 KNN，避免 n 进程 × 每进程多线程 过度订阅
    knn = KNeighborsClassifier(
        n_neighbors=knn_neighbors,
        weights="distance",
        n_jobs=1,
    )
    knn.fit(embedding_train.astype(np.float64), labels_train)

    n = X_remaining.shape[0]
    embedding_full = np.empty((n, 2), dtype=np.float32)
    labels_full = np.empty(n, dtype=np.int32)

    n_workers = _resolve_extrapolate_n_jobs(n_jobs)
    if n_workers <= 1 or n < 2:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            X_batch = X_remaining[start:end]
            emb_batch = reducer.transform(X_batch)
            lab_batch = knn.predict(emb_batch).astype(np.int32)
            embedding_full[start:end] = emb_batch.astype(np.float32)
            labels_full[start:end] = lab_batch
            if batch_size < n:
                pct = 100.0 * end / n
                print(f"[外推] 已处理 {end}/{n} 事例 ({pct:.1f}%)")
        return embedding_full, labels_full

    ranges = _split_index_ranges(n, n_workers)
    print(
        f"[外推] 使用 {len(ranges)} 个进程并行处理 {n} 条事例 "
        f"(每段约 {n // len(ranges)} 条)…"
    )
    tasks: List[Tuple[Tuple[int, int], Tuple[umap.UMAP, KNeighborsClassifier, np.ndarray]]] = []
    for start, end in ranges:
        X_chunk = X_remaining[start:end]
        tasks.append(((start, end), (reducer, knn, X_chunk)))

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(ranges)) as ex:
        # 保持与 ranges 顺序一致提交，便于按序写回
        future_to_slice = {
            ex.submit(_extrapolate_one_chunk_worker, t[1]): t[0] for t in tasks
        }
        done = 0
        for fut in concurrent.futures.as_completed(future_to_slice):
            start, end = future_to_slice[fut]
            emb_part, lab_part = fut.result()
            embedding_full[start:end] = emb_part
            labels_full[start:end] = lab_part
            done += end - start
            pct = 100.0 * done / n
            print(f"[外推] 已完成 {done}/{n} 事例 ({pct:.1f}%)")

    return embedding_full, labels_full

# -----------------------------------------------------------------------------
# 测试 umap
# -----------------------------------------------------------------------------
def _test_umap_with_selected_features(
    X: np.ndarray,
    feature_names: List[str],
    do_sampling: bool = True,
    max_points: int = 100000,
    extrapolate_batch_size: int = 50000,
    extrapolate_n_jobs: int = -1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    临时测试函数：仅使用 15 个物理上最重要的特征做 UMAP+HDBSCAN。
    当 do_sampling=True 且 n_events > max_points 时，采用「抽样训练 + 分批外推」流程：
    在 10 万点上训练 UMAP+HDBSCAN，再对剩余事例 transform + KNN 赋标签。
    extrapolate_n_jobs=-1 时外推阶段使用多进程（每进程一段数据）；设为 1 则单进程顺序外推。

    当前选取的 15 个维度（8 个 CH0 波形参数 + 5 个快放拟合参数 + 1 个快放高频能量占比 + 1 个快放频谱质心）：
        - CH0（来自 preprocessor.py 中的 dataset：max_ch0, ch0ped_mean, ch0pedt_mean,
           ch0ped_var, ch0pedt_var, tmax_ch0, ch0ped_rms, ch0pedt_rms）：
            ch0_max_ch0
            ch0_ch0ped_mean
            ch0_ch0pedt_mean
            ch0_ch0ped_var
            ch0_ch0pedt_var
            ch0_tmax_ch0
            ch0_ch0ped_rms
            ch0_ch0pedt_rms
        - 快放 CH3（来自 CH3_parameters 的 tanh 拟合与频谱特征）：
            ch3_tanh_p0
            ch3_tanh_p1
            ch3_tanh_p2
            ch3_tanh_p3
            ch3_tanh_rms
            ch3_highfreq_energy_ratio  (fast_highfreq_ratio)
            ch3_spectral_centroid_mhz  (频谱质心，单位 MHz)

    注意：这是一个方便试验用的函数，未来可以随时删除。
    """
    selected_names = [
        # 8 个 CH0 波形参数
        "ch0_max_ch0",
        "ch0_ch0ped_mean",
        "ch0_ch0pedt_mean",
        "ch0_ch0ped_var",
        "ch0_ch0pedt_var",
        "ch0_tmax_ch0",
        "ch0_ch0ped_rms",
        "ch0_ch0pedt_rms",
        # 4 个 tanh 拟合参数 + 1 个 tanh 残差 RMS
        "ch3_tanh_p0",
        "ch3_tanh_p1",
        "ch3_tanh_p2",
        "ch3_tanh_p3",
        "ch3_tanh_rms",
        # 1 个快放信号中高频能量占比
        "ch3_highfreq_energy_ratio",
        # 1 个快放信号频谱质心 (MHz)
        "ch3_spectral_centroid_mhz",
    ]
    idx_list: List[int] = []
    for name in selected_names:
        if name not in feature_names:
            print(f"[测试 UMAP] 特征 {name} 不在 feature_names 中，跳过该测试。")
            # 返回原始 X 及全 -1 标签，表示“仅噪声”，方便主流程继续运行
            return X, np.full(X.shape[0], -1, dtype=int)
        idx_list.append(feature_names.index(name))

    X_sel = X[:, idx_list]
    print(f"[测试 UMAP] 使用特征子集做 UMAP：{selected_names}")
    print(f"[测试 UMAP] 子矩阵形状: {X_sel.shape}")

    # 在进入 UMAP 之前对选定特征做标准化（零均值、单位方差），
    # 使不同量纲/数值尺度的特征在欧氏距离中贡献更加均衡。
    scaler = StandardScaler()
    X_sel_std = scaler.fit_transform(X_sel)

    # ------------------------------------------------------------------
    # 每个特征的初始“物理权重”（在 UMAP 距离中的相对重要性）：
    #
    #   对应顺序与 selected_names 完全一致：
    #   0  ch0_max_ch0                → weights[0]
    #   1  ch0_ch0ped_mean            → weights[1]
    #   2  ch0_ch0pedt_mean           → weights[2]
    #   3  ch0_ch0ped_var             → weights[3]
    #   4  ch0_ch0pedt_var            → weights[4]
    #   5  ch0_tmax_ch0               → weights[5]
    #   6  ch0_ch0ped_rms             → weights[6]
    #   7  ch0_ch0pedt_rms            → weights[7]
    #   8  ch3_tanh_p0                → weights[8]
    #   9  ch3_tanh_p1                → weights[9]
    #   10 ch3_tanh_p2                → weights[10]
    #   11 ch3_tanh_p3                → weights[11]
    #   12 ch3_tanh_rms               → weights[12]
    #   13 ch3_highfreq_energy_ratio  → weights[13]
    #   14 ch3_spectral_centroid_mhz  → weights[14]
    #
    #   你可以直接在下面这个数组里改数字来调每一维的权重：
    #   - >1.0  表示“更重要”，在 UMAP 距离中拉大该维度
    #   - 1.0   表示“基准权重”
    #   - <1.0  表示“弱化”该维度（接近 0 相当于忽略）
    # ------------------------------------------------------------------
    weights = np.array([
        1.0,  # 0  ch0_max_ch0             ：总幅度，决定能量               没那么重要    
        0.0,  # 1  ch0_ch0ped_mean         ：基线平均值，区分基线偏移        没那么重要
        1.0,  # 2  ch0_ch0pedt_mean        ：时间窗内基线平均，抑制慢漂移     没那么重要
        1.0,  # 3  ch0_ch0ped_var          ：基线方差，反映噪声水平          没那么重要
        1.0,  # 4  ch0_ch0pedt_var         ：时间相关的基线方差
        2.0,  # 5  ch0_tmax_ch0            ：峰位置，区分波形时间结构
        1.0,  # 6  ch0_ch0ped_rms          ：基线 RMS
        1.0,  # 7  ch0_ch0pedt_rms         ：时间窗内 RMS
        1.0,  # 8  ch3_tanh_p0             ：快放脉冲幅度
        1.0,  # 9  ch3_tanh_p1             ：时间尺度/上升沿相关
        1.0,  # 10 ch3_tanh_p2             ：形状参数
        1.0,  # 11 ch3_tanh_p3             ：形状/平顶相关
        1.0,  # 12 ch3_tanh_rms            ：tanh 残差 RMS
        1.0,  # 13 ch3_highfreq_energy_ratio ：高频能量占比
        2.0,  # 14 ch3_spectral_centroid_mhz ：频谱质心，强调峰移向高频
    ], dtype=float)

    # 将权重作用在“标准化后的特征”上，相当于在欧氏距离中调节各维度贡献
    X_weighted = X_sel_std * weights

    n_total = X_weighted.shape[0]
    umap_kw = dict(min_cluster_size=125, n_neighbors=10, min_samples=None, min_dist=0.1)

    if do_sampling and n_total > max_points:
        # 抽样训练 + 分批外推：先在 10 万点上 fit，再对剩余事例 extrapolate
        rng = np.random.default_rng(random_state)
        sampled_idx = rng.choice(n_total, size=max_points, replace=False).astype(np.int64)
        X_sampled = X_weighted[sampled_idx]

        print(f"[测试 UMAP] 抽样 {max_points} 点进行训练，剩余 {n_total - max_points} 点分批外推。")
        reducer, clusterer, emb_s, lab_s = _fit_umap_hdbscan_on_sample(
            X_sampled, **umap_kw
        )

        remaining_idx = np.setdiff1d(np.arange(n_total, dtype=np.int64), sampled_idx)
        X_remaining = X_weighted[remaining_idx]
        emb_rem, lab_rem = extrapolate_cluster_labels(
            reducer,
            emb_s,
            lab_s,
            X_remaining,
            batch_size=extrapolate_batch_size,
            n_jobs=extrapolate_n_jobs,
        )

        embedding = np.empty((n_total, 2), dtype=np.float32)
        labels = np.empty(n_total, dtype=np.int32)
        embedding[sampled_idx] = emb_s.astype(np.float32)
        labels[sampled_idx] = lab_s.astype(np.int32)
        embedding[remaining_idx] = emb_rem
        labels[remaining_idx] = lab_rem
        print(f"[测试 UMAP] 全量 {n_total} 事例的 cluster 标签已外推完成。")
    else:
        # 数据量可接受，直接全量 fit
        embedding, labels = run_umap_hdbscan(X_weighted, **umap_kw)

    uniq, counts = np.unique(labels, return_counts=True)
    stats = {int(l): int(c) for l, c in zip(uniq, counts)}
    print(f"[测试 UMAP] 子集特征 HDBSCAN 聚类标签统计 (label: count): {stats}")

    # 简单画一个二维 UMAP 图（不再重复波形可视化，以避免干扰主流程）
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"]})
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20")
    color_idx = 0
    for lab in unique_labels:
        mask = labels == lab
        if lab == -1:
            color = "lightgray"
            alpha = 0.4
            leg_label = "Noise"
        else:
            color = cmap((color_idx % 20 + 0.5) / 20)
            color_idx += 1
            alpha = 0.7
            leg_label = f"Cluster {lab}"
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            color=color,
            s=5,
            alpha=alpha,
            label=leg_label,
        )

    ax.set_xlabel("UMAP 1", fontsize=16)
    ax.set_ylabel("UMAP 2", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_title("TEST: UMAP on selected CH0/CH3 features", fontsize=18)
    ax.legend(loc="upper right", fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()

    # 将子集 UMAP 的嵌入和标签返回给主流程，用于后续波形可视化
    return embedding, labels

# -----------------------------------------------------------------------------
# 独立的 cut 函数：输入为对应参数数组，输出为 bool 掩码
# -----------------------------------------------------------------------------

def cut_ch0_min_positive(ch0_min: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    条件：ch0_min > threshold（排除抑制信号）。
    输入: ch0_min 数组
    输出: bool 掩码
    """
    return ch0_min > threshold

def cut_ch0_max_saturation(max_ch0: np.ndarray, max_val: float = 16382.0) -> np.ndarray:
    """
    条件：max_ch0 <= max_val（排除饱和事例）。
    输入: max_ch0 数组
    输出: bool 掩码
    """
    return max_ch0 <= max_val

def cut_ch5_self_trigger(max_ch5: np.ndarray, rt_threshold: float = 6000.0) -> np.ndarray:
    """
    条件：max_ch5 <= rt_threshold（排除随机触发）。
    输入: max_ch5 数组
    输出: bool 掩码
    """
    return max_ch5 <= rt_threshold

def cut_act(
    max_ch4: np.ndarray,
    tmax_ch4: np.ndarray,
    trigger_threshold: float = 7060.0,
    t_ge_us: float = 40.0,
    sampling_interval_ns: float = 4.0,
    dt_min_us: float = 1.0,
    dt_max_us: float = 16.0,) -> np.ndarray:
    """
    ACT cut：
    - 对 NaI 过阈事件 (max_ch4 >= trigger_threshold)，选取 Δt 非 [dt_min_us, dt_max_us] μs 的事例（反符合）；
    - 对 NaI 未过阈事件，视为“非 ACV 约束”，一律保留。
    参考 acv.py：Δt = t_Ge - t_CH4，t_CH4(μs) = tmax_ch4 * sampling_interval_ns * 1e-3。
    输入: max_ch4, tmax_ch4
    输出: bool 掩码（True = 通过 ACT cut 的事例）
    """
    n = max_ch4.shape[0]
    tmax_ch4 = np.asarray(tmax_ch4, dtype=np.float64)[:n]
    max_ch4 = np.asarray(max_ch4, dtype=np.float64)[:n]
    # NaI 是否过阈
    nai_ok = max_ch4 >= trigger_threshold
    t_ch4_us = tmax_ch4 * sampling_interval_ns * 1e-3
    delta_t_us = t_ge_us - t_ch4_us
    # 对 NaI 过阈事件：ACT = Δt 非 [dt_min_us, dt_max_us] 范围
    act_mask = (delta_t_us < dt_min_us) | (delta_t_us > dt_max_us)
    # 最终通过条件：
    # - NaI 未过阈（nai_ok == False）：全部保留；
    # - NaI 过阈（nai_ok == True）：要求满足 act_mask。
    return (~nai_ok) | (nai_ok & act_mask)



def main() -> None:
    """
    简单入口：
    - 遍历 CH0_parameters 目录下的所有 run（以文件名为 key）；
    - 对每个 run 读取并拼接 CH0–CH5 的所有参数；
    - 目前仅打印每个 run 的事件数与特征维度信息，供后续 UMAP/HDBSCAN 使用。
    """
    # 确保工程根目录在 sys.path 中，方便后续扩展时导入自定义模块
    if str(PROJECT_ROOT / "python") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "python"))

    base_names = _list_base_names_from_ch0()
    total_runs = len(base_names)
    print(f"在 CH0_parameters 中找到 {total_runs} 个 run，将汇总 CH0–CH5 参数后统一做 UMAP+HDBSCAN。")

    # 汇总所有 run 的特征矩阵和事件来源 (base_name, local_event_idx)
    all_feature_blocks: List[np.ndarray] = []
    all_sources: List[Tuple[str, int]] = []
    feature_names_ref: List[str] | None = None

    for idx, base_name in enumerate(base_names, 1):
        try:
            run_params = load_run_parameters(base_name)
        except Exception as e:
            print()  # 换行，避免与进度行混在一行
            print(f"[错误] 处理 {base_name} 时失败: {e}")
            continue

        if feature_names_ref is None:
            feature_names_ref = run_params.feature_names
        else:
            if run_params.feature_names != feature_names_ref:
                print(f"[警告] {base_name} 的特征名顺序与之前 run 不一致，跳过该 run。")
                continue

        all_feature_blocks.append(run_params.feature_matrix)
        for ev_idx in range(run_params.n_events):
            all_sources.append((base_name, ev_idx))

    if not all_feature_blocks:
        print("[错误] 未能从任何 run 中收集到参数，终止。")
        return

    X = np.vstack(all_feature_blocks)
    n_total_events, n_features = X.shape
    print(f"\n合并后总事件数: {n_total_events}，特征维度: {n_features}")

    if feature_names_ref is None:
        raise RuntimeError("feature_names_ref 为空，无法根据 ch0/ch5 进行筛选。")

    try:
        idx_ch0_max = feature_names_ref.index("ch0_max_ch0")
        idx_ch5_max = feature_names_ref.index("ch5_max_ch5")
        idx_ch0_min = feature_names_ref.index("ch0_ch0_min")
        idx_ch4_max = feature_names_ref.index("ch4_max_ch4")
        idx_ch4_tmax = feature_names_ref.index("ch4_tmax_ch4")
    except ValueError as e:
        raise RuntimeError(
            "在特征列表中未找到 ch0_max_ch0 / ch5_max_ch5 / ch0_ch0_min 之一，"
            "请检查参数 h5 是否由当前版本的 preprocessor.py 生成。"
        ) from e

    ch0_max_all = X[:, idx_ch0_max]
    ch5_max_all = X[:, idx_ch5_max]
    ch0_min_all = X[:, idx_ch0_min]
    max_ch4_all = X[:, idx_ch4_max]
    tmax_ch4_all = X[:, idx_ch4_tmax]

    # 分别应用三个基础 cut
    m_ch0_min = cut_ch0_min_positive(ch0_min_all)
    m_ch0_max = cut_ch0_max_saturation(ch0_max_all)
    m_ch5_rt = cut_ch5_self_trigger(ch5_max_all)
    m_act = cut_act(max_ch4_all, tmax_ch4_all)
    mask_physical = m_ch0_min & m_ch0_max & m_ch5_rt & m_act

    n_kept = int(mask_physical.sum())
    print(f"根据 CH0/CH5 条件筛选后剩余事件数: {n_kept} / {n_total_events}")
    if n_kept == 0:
        print("[错误] 所有事件均被 CH0/CH5 条件滤除，终止 UMAP+HDBSCAN。")
        return

    X = X[mask_physical]
    all_sources = [src for src, keep in zip(all_sources, mask_physical) if keep]
    n_total_events = n_kept

    #-------- 可选：UMAP 前同步抽样以加速 --------
    # do_umap_sampling = True
    # X, all_sources, sampled_idx = maybe_uniform_sample_for_umap(
    #     X,
    #     all_sources,
    #     do_sampling=do_umap_sampling,
    #     max_points=100000,
    #     random_state=42,)
    # print(f"[UMAP] 物理筛选后事件数: {n_kept}，UMAP 抽样后事件数: {X.shape[0]}。")

    # -------- 测试：仅用少数选定特征做一次 UMAP+HDBSCAN --------
    # 方便快速观察 CH0/CH3 关键参数的聚类效果，并将结果缓存为事件映射 HDF5。
    embedding, labels = _test_umap_with_selected_features(X, feature_names_ref)

    # 将 UMAP 嵌入与聚类标签 + 事件来源一起写入缓存 HDF5，供后续脚本直接使用
    _save_cluster_eventmap_hdf5(
        path=CLUSTER_CACHE_PATH,
        all_sources=all_sources,
        embedding=embedding,
        labels=labels,
    )

    # 从缓存文件中读取结果并进行波形可视化（如已有缓存则只做这一部分）
    _plot_waveforms_from_cluster_cache(
        path=CLUSTER_CACHE_PATH,
        ch0_3_dir=CH0_3_DIR,
        sampling_interval_ns=4.0,
    )

    print("=" * 70)
    print("参数汇总、统一 UMAP 降维、HDBSCAN 聚类、结果缓存及波形可视化已完成。")

if __name__ == "__main__":
    main()

