#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
传统参数统计脚本。

模仿 30parameter&HDBSCAN.py 的目录发现与参数读取方式：
- 从 data/hdf5/raw_pulse/CH0_parameters 列出 run 文件名
- 对每个 run，在 CH0~CH5_parameters 中读取所有一维数据集作为特征
- 统计 raw_pulse 的总事件数与总参数维度
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

def _discover_project_root() -> Path:
    """
    推断 DeepVibration 项目根目录。
    当前文件位于:
        .../python/data/ge-self/cut/parameterize/tradition/tradition.py
    向上到 python，再上一层即项目根目录。
    """
    here = Path(__file__).resolve()
    # 0: tradition
    # 1: parameterize
    # 2: cut
    # 3: ge-self
    # 4: data
    # 5: python
    # 6: DeepVibration
    python_dir = here.parents[5]
    return python_dir.parent

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

@dataclass
class RunParameters:
    base_name: str
    n_events: int
    feature_names: List[str]
    feature_matrix: np.ndarray  # (n_events, n_features)

def _list_base_names_from_ch0() -> List[str]:
    """以 CH0_parameters 为主目录列出所有 h5 run 文件名。"""
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
    """在 CH{ch}_parameters 下查找同名 run 文件。"""
    path = CH_PARAM_DIRS[ch] / base_name
    return path if path.exists() else None

def _read_all_1d_datasets(path: Path, prefix: str) -> Tuple[Dict[str, np.ndarray], int]:
    """
    读取参数文件中的所有一维数据集，返回:
    - features: {f"{prefix}{dataset_name}": ndarray(shape=(n_events,))}
    - n_events: 事件数
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

            if n_events == 0:
                n_events = int(data.shape[0])
            elif data.shape[0] != n_events:
                print(
                    f"[警告] 文件 {path.name} 中数据集 {key} 长度 "
                    f"{data.shape[0]} != 预期 {n_events}，跳过该数据集。"
                )
                continue

            features[f"{prefix}{key}"] = data.astype(np.float64)

    return features, n_events

def load_run_parameters(base_name: str) -> RunParameters:
    """读取一个 run 的 CH0~CH5 参数并拼成特征矩阵。"""
    per_channel_features: Dict[int, Dict[str, np.ndarray]] = {}
    n_events_global: int | None = None

    for ch in range(6):
        path = _open_param_file_if_exists(ch, base_name)
        if path is None:
            continue

        feats, n_events = _read_all_1d_datasets(path, prefix=f"ch{ch}_")
        if not feats:
            continue

        if n_events_global is None:
            n_events_global = n_events
        elif n_events != n_events_global:
            raise ValueError(
                f"文件 {base_name} 中通道 CH{ch} 事件数 {n_events} "
                f"与其他通道不一致（预期 {n_events_global}）。"
            )

        per_channel_features[ch] = feats

    if n_events_global is None or not per_channel_features:
        raise RuntimeError(f"运行 {base_name} 未收集到任何有效参数。")

    feature_names: List[str] = []
    feature_arrays: List[np.ndarray] = []
    for ch in sorted(per_channel_features.keys()):
        feats = per_channel_features[ch]
        for name in sorted(feats.keys()):
            feature_names.append(name)
            feature_arrays.append(feats[name].reshape(n_events_global, 1))

    feature_matrix = np.concatenate(feature_arrays, axis=1)
    return RunParameters(
        base_name=base_name,
        n_events=n_events_global,
        feature_names=feature_names,
        feature_matrix=feature_matrix,
    )

def summarize_total_events_and_dims() -> Tuple[int, int]:
    """
    汇总 raw_pulse 参数目录中的:
    - 总事件数（所有 run 事件数求和）
    - 总参数维度（所有 run 特征名并集大小）
    """
    base_names = _list_base_names_from_ch0()

    total_events = 0
    all_feature_names: set[str] = set()

    for base_name in base_names:
        run = load_run_parameters(base_name)
        total_events += run.n_events
        all_feature_names.update(run.feature_names)

    total_param_dim = len(all_feature_names)
    return total_events, total_param_dim

if __name__ == "__main__":
    total_events, total_param_dim = summarize_total_events_and_dims()
    print(f"总事件数: {total_events}")
    print(f"总参数维度: {total_param_dim}")
