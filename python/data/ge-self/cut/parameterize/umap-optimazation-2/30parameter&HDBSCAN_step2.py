#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
二次聚类（step2）：在 30parameter&HDBSCAN.py 生成的第一层 eventmap 基础上，
对**指定第一层 cluster** 内的事例做 UMAP+HDBSCAN（**不再重复物理 cut**，事例已在第一层筛过）。

- 第一层映射默认：data/hdf5/ge_30param_umap_hdbscan_eventmap.h5
- 全量读取参数矩阵后，按 eventmap 中的 (file, event) 对齐到对应行，再按父 cluster 取子集。
- 输出文件名：{output_basename}_step2_cluster{x}.h5（x 为 --parent-cluster，默认 0）
- 特征权重：编辑 STEP2_KEY_FEATURE_WEIGHTS。
"""

from __future__ import annotations
import argparse
import concurrent.futures
import os
import sys
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.covariance import LedoitWolf
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
        .../python/data/ge-self/cut/parameterize/umap-optimazation-2/30parameter&HDBSCAN_step2.py
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

# 第一层（30parameter&HDBSCAN.py）写出的事件映射，作为 step2 输入
STEP1_EVENTMAP_DEFAULT_PATH = (
    PROJECT_ROOT
    / "data"
    / "hdf5"
    / "ge_30param_umap_hdbscan_eventmap.h5")

# 输出 HDF5 文件名主体（不含扩展名）：将保存为 {OUTPUT_BASENAME_DEFAULT}_step2_cluster{x}.h5
OUTPUT_BASENAME_DEFAULT = "ge_30param_umap_hdbscan_eventmap"


# ---------------------------------------------------------------------------
# Step2 专用：与第一层可不同的 key_feature_weights（仅权重为 1.0 的维进入 UMAP）
# 后续请直接改此字典。
# ---------------------------------------------------------------------------
STEP2_KEY_FEATURE_WEIGHTS: Dict[str, float] = {
    # CH0 (preprocessor.py)
    "ch0_max_ch0": 0.0,
    "ch0_tmax_ch0": 0.0,
    "ch0_ch0_min": 0.0,
    "ch0_ch0ped_mean": 0.0,
    "ch0_ch0ped_var": 0.0,
    "ch0_ch0pedt_mean": 0.0,
    "ch0_ch0pedt_var": 0.0,
    "ch0_ch0ped_rms": 0.0,
    "ch0_ch0pedt_rms": 0.0,
    # CH1
    "ch1_max_ch1": 1.0,
    "ch1_tmax_ch1": 1.0,
    "ch1_ch1_min": 1.0,
    "ch1_ch1ped_mean": 1.0,
    "ch1_ch1ped_var": 0.0,
    "ch1_ch1pedt_mean": 1.0,
    "ch1_ch1pedt_var": 0.0,
    "ch1_ch1ped_rms": 0.0,
    "ch1_ch1pedt_rms": 0.0,
    # CH2
    "ch2_tanh_p0": 0.0,
    "ch2_tanh_p1": 0.0,
    "ch2_tanh_p2": 0.0,
    "ch2_tanh_p3": 0.0,
    "ch2_tanh_rms": 0.0,
    "ch2_highfreq_energy_ratio": 0.0,
    "ch2_spectral_centroid_mhz": 0.0,
    "ch2_n_fit_points": 0.0,
    "ch2_max_ch2": 0.0,
    "ch2_min_ch2": 0.0,
    "ch2_tmax_ch2": 0.0,
    "ch2_tmin_ch2": 0.0,
    "ch2_ch2ped_mean": 0.0,
    "ch2_ch2pedt_mean": 0.0,
    # CH3
    "ch3_tanh_p0": 0.0,
    "ch3_tanh_p1": 0.0,
    "ch3_tanh_p2": 0.0,
    "ch3_tanh_p3": 0.0,
    "ch3_tanh_rms": 0.0,
    "ch3_highfreq_energy_ratio": 0.0,
    "ch3_spectral_centroid_mhz": 0.0,
    "ch3_n_fit_points": 0.0,
    "ch3_max_ch3": 0.0,
    "ch3_min_ch3": 0.0,
    "ch3_tmax_ch3": 0.0,
    "ch3_tmin_ch3": 0.0,
    "ch3_ch3ped_mean": 0.0,
    "ch3_ch3pedt_mean": 0.0,
    # CH4/CH5
    "ch4_max_ch4": 0.0,
    "ch4_tmax_ch4": 0.0,
    "ch5_max_ch5": 0.0,}


def _load_step1_eventmap_sources_and_labels(
    path: Path,) -> Tuple[np.ndarray, List[Tuple[str, int]]]:

    """
    从第一层 eventmap 读出与 30parameter&HDBSCAN.py 保存顺序一致的事例列表
    及第一层 cluster 标签。
    """
    with h5py.File(path, "r") as f:
        file_paths_raw = f["file_paths"][...]
        event_file_indices = f["event_file_indices"][...]
        event_event_indices = f["event_event_indices"][...]
        parent_labels = f["event_cluster_labels"][...].astype(np.int32)

    file_paths: List[str] = []
    for p in file_paths_raw:
        file_paths.append(p.decode("utf-8") if isinstance(p, bytes) else str(p))

    n = int(parent_labels.shape[0])
    sources: List[Tuple[str, int]] = []
    for k in range(n):
        fi = int(event_file_indices[k])
        ev = int(event_event_indices[k])
        if fi < 0 or fi >= len(file_paths):
            raise ValueError(f"eventmap 行 {k}: 非法 file 索引 {fi}")
        sources.append((file_paths[fi], ev))
    return parent_labels, sources

def _step2_subcluster_label_str(output_basename: str, parent_cluster: int, hdbscan_label: int) -> str:
    """每条事例一个字符串标签：{basename}_step2_cluster{parent}_sub{子类或 noise}。"""
    if int(hdbscan_label) == -1:
        sub = "noise"
    else:
        sub = str(int(hdbscan_label))
    return f"{output_basename}_step2_cluster{int(parent_cluster)}_sub{sub}"

def _save_cluster_eventmap_hdf5(
    path: Path,
    all_sources: List[Tuple[str, int]],
    embedding: np.ndarray,
    labels: np.ndarray,
    cluster_label_names: Optional[List[str]] = None,
    file_attrs: Optional[Dict[str, str]] = None,) -> None:
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
        if cluster_label_names is not None:
            if len(cluster_label_names) != len(all_sources):
                raise ValueError("cluster_label_names 长度与 all_sources 不一致。")
            dt = h5py.special_dtype(vlen=str)
            ds = f.create_dataset("cluster_label_names", (len(cluster_label_names),), dtype=dt)
            for i, s in enumerate(cluster_label_names):
                ds[i] = s
        if file_attrs:
            for k, v in file_attrs.items():
                f.attrs[k] = v

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
                loc="upper left",
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

# ------------------------------------------------------------------
# 读取文件维度信息
# ------------------------------------------------------------------

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

def _align_run_to_feature_union(
    run_params: RunParameters,
    feature_names_union: List[str],
    fill_value: float = np.nan,) -> np.ndarray:
    """
    将单个 run 的特征矩阵按“全局特征并集”顺序对齐：
    - run 缺失的维度填充为 fill_value；
    - run 额外维度（理论上不会额外）忽略。
    """
    n_events = run_params.n_events
    aligned = np.full((n_events, len(feature_names_union)), fill_value, dtype=np.float64)
    name_to_idx = {n: i for i, n in enumerate(run_params.feature_names)}
    for j, feat_name in enumerate(feature_names_union):
        src_idx = name_to_idx.get(feat_name)
        if src_idx is None:
            continue
        aligned[:, j] = run_params.feature_matrix[:, src_idx]
    return aligned

# ------------------------------------------------------------------
# 核心
# ------------------------------------------------------------------

def run_umap_hdbscan(
    features: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    min_cluster_size: int = 50,
    min_samples: int | None = None,
    cluster_selection_epsilon: float = 0.0,
    metric: str = "euclidean",
    metric_kwds: dict | None = None,
    prediction_data: bool = False,
    hdbscan_adaptive_merge: bool = False,
    hdbscan_adaptive_growth: float = 1.12,
    hdbscan_adaptive_max_steps: int = 48,) -> Tuple[np.ndarray, np.ndarray]:
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
        metric=metric,
        metric_kwds=metric_kwds,
        random_state=42,     # 固定随机种子，确保结果可复现
        low_memory=True,     # 防止爆内存
        force_approximation_algorithm=True,
        verbose=True,
        n_jobs=-1           # 使用所有可用 CPU 核并行加速 UMAP
    )
    embedding = reducer.fit_transform(features)

    clusterer, labels, eps_used = _fit_hdbscan_2d(
        embedding,
        min_cluster_size,
        min_samples,
        float(cluster_selection_epsilon),
        prediction_data=prediction_data,
        adaptive_merge=hdbscan_adaptive_merge,
        adaptive_growth=hdbscan_adaptive_growth,
        adaptive_max_steps=hdbscan_adaptive_max_steps,
        log_prefix="[UMAP+HDBSCAN]",
    )
    return embedding, labels

# ------------------------------------------------------------------
# 训练部分
# ------------------------------------------------------------------

def _fit_umap_hdbscan_on_sample(
    X_sampled: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    min_cluster_size: int = 50,
    min_samples: int | None = None,
    cluster_selection_epsilon: float = 0.0,
    metric: str = "euclidean",
    metric_kwds: dict | None = None,
    hdbscan_adaptive_merge: bool = False,
    hdbscan_adaptive_growth: float = 1.12,
    hdbscan_adaptive_max_steps: int = 48,
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
        metric=metric,
        metric_kwds=metric_kwds,
        random_state=42,
        low_memory=True,
        force_approximation_algorithm=True,
        verbose=True,
        n_jobs=-1,
    )
    embedding_s = reducer.fit_transform(X_sampled)

    clusterer, labels_s, eps_used = _fit_hdbscan_2d(
        embedding_s,
        min_cluster_size,
        min_samples,
        float(cluster_selection_epsilon),
        prediction_data=True,
        adaptive_merge=hdbscan_adaptive_merge,
        adaptive_growth=hdbscan_adaptive_growth,
        adaptive_max_steps=hdbscan_adaptive_max_steps,
        log_prefix="[抽样训练]",
    )
    return reducer, clusterer, embedding_s, labels_s

def _estimate_cluster_selection_epsilon(
    embedding: np.ndarray,
    n_neighbors: int = 12,
    quantile: float = 0.9,
    scale: float = 1.0,) -> float:
    """
    按 UMAP 2D 点云的局部尺度估计 HDBSCAN 的 cluster_selection_epsilon。
    用较高分位的 kNN 距离作为“可接受簇间缝隙”，有助于合并视觉上连在一起但被密度树切开的子簇。
    """
    n = int(embedding.shape[0])
    if n < 5:
        return 0.0
    k = max(2, min(int(n_neighbors), n - 1))
    # 第 0 列是自身距离 0，取第 k 列代表第 k 邻居距离
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    nbrs.fit(embedding.astype(np.float64))
    distances, _ = nbrs.kneighbors(embedding.astype(np.float64))
    kth = distances[:, k]
    eps = float(np.quantile(kth, quantile) * float(scale))
    if not np.isfinite(eps) or eps < 0:
        return 0.0
    return eps


def _fit_hdbscan_2d(
    embedding: np.ndarray,
    min_cluster_size: int,
    min_samples: int | None,
    cluster_selection_epsilon: float,
    *,
    prediction_data: bool,
    adaptive_merge: bool = False,
    adaptive_growth: float = 1.12,
    adaptive_max_steps: int = 48,
    log_prefix: str = "[HDBSCAN]",
) -> Tuple[hdbscan.HDBSCAN, np.ndarray, float]:
    """
    在固定 2D 嵌入上做 HDBSCAN。adaptive_merge 为 True 时：
    - 从 cluster_selection_epsilon 起按几何步长增大 epsilon，直到步数上限或超过由点云跨度推得的 eps 上限；
    - 在**所有**出现过的合法划分里取字典序最优 (非噪声簇数, 噪声点数)，避免「簇数未降但噪声变多」时错误地一直停在第一步；
    - 若某步非噪声簇数 <= 1 则提前结束；
    - 若 EOM 扫描后仍 >=2 簇，再尝试 cluster_selection_method='leaf'（仍按同样准则择优）。
    """
    emb = np.asarray(embedding, dtype=np.float64)
    eps0 = float(cluster_selection_epsilon)

    def _single_fit(eps: float, method: str) -> Tuple[hdbscan.HDBSCAN, np.ndarray]:
        cl = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method=method,
            cluster_selection_epsilon=float(eps),
            core_dist_n_jobs=-1,
            prediction_data=prediction_data,
        )
        lab = cl.fit_predict(emb)
        return cl, lab

    if not adaptive_merge:
        cl, lab = _single_fit(eps0, "eom")
        return cl, lab, float(eps0)

    def _stats(lab: np.ndarray) -> Tuple[int, int] | None:
        labeled = lab >= 0
        if not np.any(labeled):
            return None
        n_cl = int(len(np.unique(lab[labeled])))
        n_noise = int(np.sum(~labeled))
        return n_cl, n_noise

    growth = float(adaptive_growth)
    max_steps = max(1, int(adaptive_max_steps))
    span = float(np.nanmax(emb.max(axis=0) - emb.min(axis=0)))
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    # 上限：兼顾「比 kNN 估计值大很多仍能并簇」与点云直径比例，避免无限放大
    eps_floor = max(eps0, 1e-9) if eps0 > 0.0 else 1e-9
    eps_cap = max(eps_floor * 120.0, span * 0.55, eps_floor * 4.0)

    records: List[Tuple[int, int, float, hdbscan.HDBSCAN, np.ndarray, str]] = []
    eps = eps_floor
    first_stats: Tuple[int, int] | None = None
    eom_eps_tried: List[float] = []
    eom_eps_seen: set[float] = set()

    def _eom_try_eps(eps_try: float) -> bool:
        """若该 eps（按四舍五入去重）未试过则拟合 EOM；<=1 簇时打印并返回 True 表示应立刻 return。"""
        nonlocal first_stats
        key = round(float(eps_try), 9)
        if key in eom_eps_seen:
            return False
        eom_eps_seen.add(key)
        eom_eps_tried.append(float(eps_try))
        cl, lab = _single_fit(float(eps_try), "eom")
        st = _stats(lab)
        if st is None:
            return False
        n_cl, n_noise = st
        if first_stats is None:
            first_stats = (n_cl, n_noise)
        records.append((n_cl, n_noise, float(eps_try), cl, lab.copy(), "eom"))
        if n_cl <= 1:
            fs = first_stats or st
            print(
                f"{log_prefix} 自适应(EOM): 起始 eps={eps0:.6g} → 选用 eps={float(eps_try):.6g} | "
                f"首轮 簇={fs[0]} 噪声={fs[1]} → 选用 簇={n_cl} 噪声={n_noise}"
            )
            return True
        return False

    for _ in range(max_steps):
        if eps > eps_cap:
            break
        if _eom_try_eps(eps):
            return records[-1][3], records[-1][4], records[-1][2]
        eps *= growth

    # 多簇时在中小 eps 上常出现「平台」：簇数/噪声不变，看起来像没扩大 eps。
    # 在更大 epsilon 上补扫，并尝试压到单簇（字典序仍优先 簇数，再 噪声数）。
    if records:
        n_min = min(t[0] for t in records)
        if n_min > 1:
            diam = float(np.linalg.norm(np.nanmax(emb, axis=0) - np.nanmin(emb, axis=0)))
            if not np.isfinite(diam) or diam <= 0:
                diam = float(span)
            e_last = max(t[2] for t in records if t[5] == "eom")
            e_hi = max(
                e_last * 200.0,
                diam * 4.0,
                eps_floor * 400.0,
                float(span) * 6.0,
                eps_cap,
            )
            if e_hi > e_last * 1.02:
                ext = np.geomspace(max(e_last * 1.06, eps_floor * 1.01), e_hi, num=40)
                for eps_x in ext:
                    if _eom_try_eps(float(eps_x)):
                        return records[-1][3], records[-1][4], records[-1][2]
            # 极大 epsilon：促使密度树尽量扁平（若仍多簇则说明 min_cluster_size 等约束主导）
            for mult in (500.0, 2000.0, 10000.0):
                ex = max(eps_floor * mult, diam * mult * 0.05, 1e-6)
                if _eom_try_eps(ex):
                    return records[-1][3], records[-1][4], records[-1][2]

    # leaf：有时比 EOM 更少「切碎」同一团
    for eps_l in (0.0, eps_floor, min(eps_cap, eps_floor * 3.0)):
        cl, lab = _single_fit(float(eps_l), "leaf")
        st = _stats(lab)
        if st is not None:
            records.append((st[0], st[1], float(eps_l), cl, lab.copy(), "leaf"))

    if records:
        # 簇数、噪声数相同时取更大 eps（更偏合并），避免日志上「选用=起点」而实际已扫过大档 eps
        records.sort(key=lambda t: (t[0], t[1], -t[2]))
        n_cl_b, n_noise_b, eps_b, cl_b, lab_b, method_b = records[0]
        fs = first_stats or (n_cl_b, n_noise_b)
        e_lo = min(eom_eps_tried) if eom_eps_tried else eps_floor
        e_hi_log = max(eom_eps_tried) if eom_eps_tried else eps_floor
        print(
            f"{log_prefix} 自适应({method_b}): 起始 eps={eps0:.6g} → 选用 eps={eps_b:.6g} | "
            f"首轮(EOM@起始) 簇={fs[0]} 噪声={fs[1]} → 选用 簇={n_cl_b} 噪声={n_noise_b} | "
            f"EOM 共试 {len(eom_eps_tried)} 档 eps∈[{e_lo:.4g},{e_hi_log:.4g}]"
        )
        return cl_b, lab_b, eps_b

    cl, lab = _single_fit(eps_floor, "eom")
    print(f"{log_prefix} 自适应: 无合法非空簇，回退 EOM eps={eps_floor:.6g}")
    return cl, lab, float(eps_floor)


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
    args: Tuple[umap.UMAP, KNeighborsClassifier, np.ndarray],) -> Tuple[np.ndarray, np.ndarray]:
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
    n_jobs: int = -1,) -> Tuple[np.ndarray, np.ndarray]:
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
    key_feature_weights: Dict[str, float],
    do_sampling: bool = True,
    max_points: int = 100000,
    extrapolate_batch_size: int = 50000,
    extrapolate_n_jobs: int = -1,
    random_state: int = 42,
    hdbscan_adaptive_merge: bool = False,
    hdbscan_adaptive_growth: float = 1.12,
    hdbscan_adaptive_max_steps: int = 48,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    临时测试函数：使用当前输入中的全部特征做 UMAP+HDBSCAN。
    当 do_sampling=True 且 n_events > max_points 时，采用「抽样训练 + 分批外推」流程：
    在 10 万点上训练 UMAP+HDBSCAN，再对剩余事例 transform + KNN 赋标签。
    extrapolate_n_jobs=-1 时外推阶段使用多进程（每进程一段数据）；设为 1 则单进程顺序外推。

    权重策略：
    - 默认所有维度权重为 0；
    - 对当前已验证有效的 15 个物理特征，沿用既有权重；
    - 其余维度先保留在输入矩阵中，但权重为 0（相当于暂不参与距离计算）。

    注意：这是一个方便试验用的函数，未来可以随时删除。
    """
    # 输入维度策略：
    # 仅选取 key_feature_weights 中权重为 1.0 的维度进入 UMAP（例如 4 维就只用 4 维做降维）。
    selected_names = list(feature_names)

    enabled = {k for k, v in key_feature_weights.items() if float(v) == 1.0}
    keep_idx = [i for i, name in enumerate(selected_names) if name in enabled]
    if not keep_idx:
        raise RuntimeError(
            "[测试 UMAP] key_feature_weights 中没有任何权重为 1.0 且存在于 feature_names 的特征，无法继续。"
        )

    selected_names = [selected_names[i] for i in keep_idx]
    X_sel = X[:, keep_idx]
    print(f"[测试 UMAP] 仅使用 key_feature_weights==1.0 的维度做 UMAP，维度数: {len(selected_names)}")
    print(f"[测试 UMAP] 特征矩阵形状: {X_sel.shape}")

    # 标准化后直接作为 UMAP 输入，不再乘以 weights
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)

    # 2️⃣ 协方差（稳健）
    cov = LedoitWolf().fit(X_scaled)
    V = cov.covariance_
    X_umap_in = X_scaled

    # 事例数（行数）；勿用 shape[-1]，否则会把特征维当成样本数导致抽样分支失效。
    n_total = int(X_umap_in.shape[0])
    umap_kw = dict(
        min_cluster_size=80,
        n_neighbors=40,   #越小越碎
        #min_samples=18,
        min_dist=0.1,
        metric="mahalanobis",
        metric_kwds={"V": V},
    )

    if do_sampling and n_total > max_points:
        # 抽样训练 + 分批外推：先在 10 万点上 fit，再对剩余事例 extrapolate
        rng = np.random.default_rng(random_state)
        sampled_idx = rng.choice(n_total, size=max_points, replace=False).astype(np.int64)
        X_sampled = X_umap_in[sampled_idx]

        print(f"[测试 UMAP] 抽样 {max_points} 点进行训练，剩余 {n_total - max_points} 点分批外推。")
        # 先做一遍 UMAP 以估计 epsilon，再用于 HDBSCAN，减少同团块被过度拆分。
        reducer_probe = umap.UMAP(
            n_neighbors=umap_kw["n_neighbors"],
            min_dist=umap_kw["min_dist"],
            n_components=2,
            metric=umap_kw["metric"],
            metric_kwds=umap_kw["metric_kwds"],
            random_state=42,
            low_memory=True,
            force_approximation_algorithm=True,
            verbose=False,
            n_jobs=-1,
        )
        emb_probe = reducer_probe.fit_transform(X_sampled)
        eps = _estimate_cluster_selection_epsilon(emb_probe, n_neighbors=12, quantile=0.90, scale=1.22)
        umap_kw["cluster_selection_epsilon"] = eps
        print(f"[测试 UMAP] kNN 估计 cluster_selection_epsilon(起点)={eps:.4f} (scale=1.22)")
        reducer, clusterer, emb_s, lab_s = _fit_umap_hdbscan_on_sample(
            X_sampled,
            **umap_kw,
            hdbscan_adaptive_merge=hdbscan_adaptive_merge,
            hdbscan_adaptive_growth=hdbscan_adaptive_growth,
            hdbscan_adaptive_max_steps=hdbscan_adaptive_max_steps,
        )

        remaining_idx = np.setdiff1d(np.arange(n_total, dtype=np.int64), sampled_idx)
        X_remaining = X_umap_in[remaining_idx]
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
        # 先估计一次 epsilon，避免在 2D 上把视觉连通团块过度拆分。
        reducer_probe = umap.UMAP(
            n_neighbors=umap_kw["n_neighbors"],
            min_dist=umap_kw["min_dist"],
            n_components=2,
            metric=umap_kw["metric"],
            metric_kwds=umap_kw["metric_kwds"],
            random_state=42,
            low_memory=True,
            force_approximation_algorithm=True,
            verbose=False,
            n_jobs=-1,
        )
        emb_probe = reducer_probe.fit_transform(X_umap_in)
        eps = _estimate_cluster_selection_epsilon(emb_probe, n_neighbors=12, quantile=0.90, scale=1.22)
        umap_kw["cluster_selection_epsilon"] = eps
        print(f"[测试 UMAP] kNN 估计 cluster_selection_epsilon(起点)={eps:.4f} (scale=1.22)")
        embedding, labels = run_umap_hdbscan(
            X_umap_in,
            **umap_kw,
            hdbscan_adaptive_merge=hdbscan_adaptive_merge,
            hdbscan_adaptive_growth=hdbscan_adaptive_growth,
            hdbscan_adaptive_max_steps=hdbscan_adaptive_max_steps,
        )

    uniq, counts = np.unique(labels, return_counts=True)
    stats = {int(l): int(c) for l, c in zip(uniq, counts)}
    print(f"[测试 UMAP] 子集特征 HDBSCAN 聚类标签统计 (label: count): {stats}")

    # 简单画一个二维 UMAP 图（不再重复波形可视化，以避免干扰主流程）
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
    #ax.set_title("TEST: UMAP on selected CH0/CH3 features", fontsize=18)
    ax.legend(loc="upper left", fontsize=10, ncol=2)
    #ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()

    # 将子集 UMAP 的嵌入和标签返回给主流程，用于后续波形可视化
    return embedding, labels

# -----------------------------------------------------------------------------
# 独立的 cut 函数：输入为对应参数数组，输出为 bool 掩码
# （与 basic+act.py / ch2ped-pedt.py 中 ±3σ_ped、y=x±1σ、ADC 裁剪一致）
# -----------------------------------------------------------------------------

CH2_PED_PEDT_INNER_X_MAX = 2000.0
CH2_PED_PEDT_ADC_MAX = 16382.0

def cut_fit_success(
    ch2_n_fit_points: np.ndarray,
    ch3_n_fit_points: np.ndarray,
    ch2_tanh_p0: np.ndarray,
    ch3_tanh_p0: np.ndarray,
    bad_val: float = 1e6,) -> np.ndarray:
    """
    过滤 fit_ch2_ch3_parallel.py 中拟合失败/未参与拟合的事件。

    判定规则（同时满足）：
    - CH2 与 CH3 的 n_fit_points 都 > 0；
    - CH2 与 CH3 的 tanh_p0 为有限值，且不等于异常值 bad_val（默认 1e6）。
    """
    ch2_n = np.asarray(ch2_n_fit_points, dtype=np.int32)
    ch3_n = np.asarray(ch3_n_fit_points, dtype=np.int32)
    ch2_p0 = np.asarray(ch2_tanh_p0, dtype=np.float64)
    ch3_p0 = np.asarray(ch3_tanh_p0, dtype=np.float64)
    n = min(ch2_n.shape[0], ch3_n.shape[0], ch2_p0.shape[0], ch3_p0.shape[0])
    ch2_n = ch2_n[:n]
    ch3_n = ch3_n[:n]
    ch2_p0 = ch2_p0[:n]
    ch3_p0 = ch3_p0[:n]

    ok_npts = (ch2_n > 0) & (ch3_n > 0)
    ok_ch2 = np.isfinite(ch2_p0) & (~np.isclose(ch2_p0, bad_val))
    ok_ch3 = np.isfinite(ch3_p0) & (~np.isclose(ch3_p0, bad_val))
    return ok_npts & ok_ch2 & ok_ch3

def cut_ch0_min_positive(ch0_min: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    条件：ch0_min > threshold（排除抑制信号）。
    输入: ch0_min 数组
    输出: bool 掩码
    """
    return ch0_min > threshold

def cut_ch0_max_saturation(max_ch0: np.ndarray, max_ch1: np.ndarray, max_val: float = 16382.0) -> np.ndarray:
    """
    条件：max_ch0 <= max_val 且 max_ch1 <= max_val（排除 CH0/CH1 饱和事例）。
    输入: max_ch0、max_ch1 数组
    输出: bool 掩码
    """
    return (max_ch0 <= max_val) & (max_ch1 <= max_val)

def cut_ch5_self_trigger(max_ch5: np.ndarray, rt_threshold: float = 6000.0) -> np.ndarray:
    """
    条件：max_ch5 <= rt_threshold（排除随机触发）。
    输入: max_ch5 数组
    输出: bool 掩码
    """
    return max_ch5 <= rt_threshold

def cut_acv(
    max_ch4: np.ndarray,
    tmax_ch4: np.ndarray,
    trigger_threshold: float = 7060.0,
    t_ge_us: float = 40.0,
    sampling_interval_ns: float = 4.0,
    dt_min_us: float = 1.0,
    dt_max_us: float = 16.0,) -> np.ndarray:
    """
    acv cut：
    - 对 NaI 过阈事件 (max_ch4 >= trigger_threshold)，选取 Δt 非 [dt_min_us, dt_max_us] μs 的事例（反符合）；
    - 对 NaI 未过阈事件，视为“非 ACV 约束”，一律保留。
    参考 acv.py：Δt = t_Ge - t_CH4，t_CH4(μs) = tmax_ch4 * sampling_interval_ns * 1e-3。
    输入: max_ch4, tmax_ch4
    输出: bool 掩码（True = 通过 acv cut 的事例）
    """
    n = max_ch4.shape[0]
    tmax_ch4 = np.asarray(tmax_ch4, dtype=np.float64)[:n]
    max_ch4 = np.asarray(max_ch4, dtype=np.float64)[:n]
    # NaI 是否过阈
    nai_ok = max_ch4 >= trigger_threshold
    t_ch4_us = tmax_ch4 * sampling_interval_ns * 1e-3
    delta_t_us = t_ge_us - t_ch4_us
    # 对 NaI 过阈事件：acv = Δt 非 [dt_min_us, dt_max_us] 范围
    acv_mask = (delta_t_us < dt_min_us) | (delta_t_us > dt_max_us)
    # 最终通过条件：
    # - NaI 未过阈（nai_ok == False）：全部保留；
    # - NaI 过阈（nai_ok == True）：要求满足 acv_mask。
    return (~nai_ok) | (nai_ok & acv_mask)

def cut_ch3ped_min(
    ch3ped_mean: np.ndarray,
    min_ch3: np.ndarray,
    *,
    sigma_yx: float = 20.0,) -> np.ndarray:
    """
    CH3 ped-min 带状 cut：
    - 要求点位于 y=x 周边 sigma_yx 内（默认 |min_ch3 - ch3ped_mean| <= 20）
    - 并显式要求 min_ch3 > 0。
    """
    x = np.asarray(ch3ped_mean, dtype=np.float64)
    y = np.asarray(min_ch3, dtype=np.float64)
    n = min(x.shape[0], y.shape[0])
    x = x[:n]
    y = y[:n]
    fin = np.isfinite(x) & np.isfinite(y)
    sig = float(sigma_yx)
    return fin & (np.abs(y - x) <= sig) & (y > 0.0)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="在第一层 ge_30param_umap_hdbscan_eventmap.h5 基础上，对指定 cluster 做 step2 UMAP+HDBSCAN。",
    )
    p.add_argument(
        "--parent-cluster",
        type=int,
        default=1,
        help="第一层 HDBSCAN 聚类标签（与 eventmap 中 event_cluster_labels 一致，默认 0）。",
    )
    p.add_argument(
        "--input-eventmap",
        type=str,
        default=str(STEP1_EVENTMAP_DEFAULT_PATH),
        help="第一层写出的事件映射 HDF5（默认 data/hdf5/ge_30param_umap_hdbscan_eventmap.h5）。",
    )
    p.add_argument(
        "--output-basename",
        type=str,
        default=OUTPUT_BASENAME_DEFAULT,
        help=(
            "输出文件名主体（不含路径与扩展名）；将写入 "
            "data/hdf5/{basename}_step2_cluster{parent}.h5"
        ),
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="跳过波形抽样可视化。",
    )
    p.add_argument(
        "--hdbscan-adaptive-merge",
        action="store_true",
        help=(
            "启用 step2 HDBSCAN 在 2D UMAP 嵌入上的自适应合并（扫描增大 cluster_selection_epsilon，"
            "偏向更少簇/更少噪声）。默认关闭。"
        ),
    )
    p.add_argument(
        "--no-hdbscan-adaptive-merge",
        action="store_true",
        help=(
            "与默认行为相同（关闭自适应合并）。保留该参数仅为兼容旧命令行；"
            "若需旧版「默认开启自适应」请改用 --hdbscan-adaptive-merge。"
        ),
    )
    return p.parse_args()

def main() -> None:
    """
    Step2 入口：全量汇总特征（与主流程相同列），按第一层 eventmap 对齐事例（不重复 cut），
    再按 --parent-cluster 取子集，用 STEP2_KEY_FEATURE_WEIGHTS 做 UMAP+HDBSCAN。
    """
    args = parse_args()

    if str(PROJECT_ROOT / "python") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "python"))

    input_eventmap = Path(args.input_eventmap).resolve()
    if not input_eventmap.exists():
        raise FileNotFoundError(f"第一层 eventmap 不存在: {input_eventmap}")

    parent_cluster = int(args.parent_cluster)
    out_base = str(args.output_basename).strip().rstrip("/")
    if not out_base:
        raise ValueError("--output-basename 不能为空。")

    base_names = _list_base_names_from_ch0()
    total_runs = len(base_names)
    print(f"[step2] 在 CH0_parameters 中找到 {total_runs} 个 run；全量汇总参数（不重复物理 cut）。")

    run_params_list: List[RunParameters] = []
    all_sources: List[Tuple[str, int]] = []
    feature_name_union_set: set[str] = set()

    for base_name in base_names:
        try:
            run_params = load_run_parameters(base_name)
        except Exception as e:
            print()
            print(f"[错误] 处理 {base_name} 时失败: {e}")
            continue

        run_params_list.append(run_params)
        feature_name_union_set.update(run_params.feature_names)
        for ev_idx in range(run_params.n_events):
            all_sources.append((base_name, ev_idx))

    if not run_params_list:
        print("[错误] 未能从任何 run 中收集到参数，终止。")
        return

    feature_names_ref = sorted(feature_name_union_set)
    print(f"全局特征并集维度数: {len(feature_names_ref)}")

    all_feature_blocks: List[np.ndarray] = []
    for rp in run_params_list:
        all_feature_blocks.append(_align_run_to_feature_union(rp, feature_names_ref))

    X = np.vstack(all_feature_blocks)
    if np.isnan(X).any():
        col_med = np.nanmedian(X, axis=0)
        col_med = np.where(np.isfinite(col_med), col_med, 0.0)
        nan_rows, nan_cols = np.where(np.isnan(X))
        X[nan_rows, nan_cols] = col_med[nan_cols]
    n_total_events, n_features = X.shape
    print(f"\n合并后总事件数: {n_total_events}，特征维度: {n_features}")

    parent_labels, sources_step1 = _load_step1_eventmap_sources_and_labels(input_eventmap)

    source_to_row: Dict[Tuple[str, int], int] = {all_sources[i]: i for i in range(len(all_sources))}
    rows: List[int] = []
    missing: List[Tuple[str, int]] = []
    for s in sources_step1:
        j = source_to_row.get(s)
        if j is None:
            missing.append(s)
        else:
            rows.append(j)
    if missing:
        raise ValueError(
            f"第一层 eventmap 中有 {len(missing)} 个事例在当前全量参数表中找不到（run/事件号不一致？）。"
            f"示例: {missing[:3]}"
        )

    X_step1 = X[np.asarray(rows, dtype=np.int64)]
    if X_step1.shape[0] != parent_labels.shape[0]:
        raise RuntimeError("内部错误：对齐第一层事例后行数与标签数不一致。")

    print(
        f"[step2] 已与第一层 eventmap 对齐: {X_step1.shape[0]} 个事例（等于第一层保留事例数）。"
    )

    sub_mask = parent_labels == parent_cluster
    n_sub = int(sub_mask.sum())
    print(f"[step2] 第一层 cluster == {parent_cluster} 的事例数: {n_sub} / {X_step1.shape[0]}")
    if n_sub == 0:
        print("[错误] 该父 cluster 内没有事例，终止。")
        return

    X_sub = X_step1[sub_mask]
    sources_sub = [src for src, m in zip(sources_step1, sub_mask) if m]

    embedding, labels = _test_umap_with_selected_features(
        X_sub,
        feature_names_ref,
        STEP2_KEY_FEATURE_WEIGHTS,
        hdbscan_adaptive_merge=bool(args.hdbscan_adaptive_merge)
        and not bool(args.no_hdbscan_adaptive_merge),
    )

    out_path = PROJECT_ROOT / "data" / "hdf5" / f"{out_base}_step2_cluster{parent_cluster}.h5"
    label_names = [
        _step2_subcluster_label_str(out_base, parent_cluster, int(lab)) for lab in labels
    ]

    _save_cluster_eventmap_hdf5(
        path=out_path,
        all_sources=sources_sub,
        embedding=embedding,
        labels=labels,
        cluster_label_names=label_names,
        file_attrs={
            "pipeline": "step2",
            "step1_eventmap": str(input_eventmap),
            "parent_cluster": str(parent_cluster),
            "output_basename": out_base,
        },
    )

    if not args.no_plot:
        _plot_waveforms_from_cluster_cache(
            path=out_path,
            ch0_3_dir=CH0_3_DIR,
            sampling_interval_ns=4.0,
        )

    print("=" * 70)
    print(f"[step2] 已完成：{out_path}")
    print(f"  字符串标签示例（dataset cluster_label_names）: {label_names[0] if label_names else '(空)'}")


if __name__ == "__main__":
    main()

