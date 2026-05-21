#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
汇总指定参数（来自 30parameter&HDBSCAN.py 的 key_feature_weights），
对全部 event 执行：
1) 标准化（StandardScaler）
2) LedoitWolf 协方差估计
3) 协方差矩阵可视化（热力图）
"""

from __future__ import annotations

import ast
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler


def _discover_project_root() -> Path:
    here = Path(__file__).resolve()
    python_dir = here.parents[5]  # .../python
    return python_dir.parent      # .../DeepVibration


PROJECT_ROOT = _discover_project_root()
DATA_ROOT = PROJECT_ROOT / "data" / "hdf5" / "raw_pulse"
OUT_DIR = PROJECT_ROOT / "images"

CH_PARAM_DIRS: Dict[int, Path] = {
    0: DATA_ROOT / "CH0_parameters",
    1: DATA_ROOT / "CH1_parameters",
    2: DATA_ROOT / "CH2_parameters",
    3: DATA_ROOT / "CH3_parameters",
    4: DATA_ROOT / "CH4_parameters",
    5: DATA_ROOT / "CH5_parameters",
}

KEY_FEATURE_SCRIPT = Path(__file__).resolve().parent / "30parameter&HDBSCAN.py"


def _load_target_features_from_script(script_path: Path) -> List[str]:
    """
    从 30parameter&HDBSCAN.py 中解析 key_feature_weights 的键顺序，确保与主脚本一致。
    """
    if not script_path.exists():
        raise FileNotFoundError(f"不存在: {script_path}")
    src = script_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(script_path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "key_feature_weights":
                    if not isinstance(node.value, ast.Dict):
                        continue
                    keys: List[str] = []
                    for k in node.value.keys:
                        if isinstance(k, ast.Constant) and isinstance(k.value, str):
                            keys.append(k.value)
                    if keys:
                        return keys
    raise RuntimeError(f"未在 {script_path} 中解析到 key_feature_weights。")

TARGET_FEATURES: List[str] = _load_target_features_from_script(KEY_FEATURE_SCRIPT)

def _list_base_names_from_ch0() -> List[str]:
    ch0_dir = CH_PARAM_DIRS[0]
    if not ch0_dir.exists():
        raise FileNotFoundError(f"CH0_parameters 目录不存在: {ch0_dir}")
    names = [
        n for n in sorted(os.listdir(ch0_dir))
        if n.lower().endswith((".h5", ".hdf5"))
    ]
    if not names:
        raise FileNotFoundError(f"CH0_parameters 目录 {ch0_dir} 下未找到任何 h5 文件")
    return names

def _read_1d_dataset_if_exists(path: Path, key: str) -> np.ndarray | None:
    if not path.exists():
        return None
    with h5py.File(path, "r") as f:
        dset = f.get(key)
        if dset is None or not isinstance(dset, h5py.Dataset) or dset.ndim != 1:
            return None
        arr = np.asarray(dset[...], dtype=np.float64)
        if arr.size == 0:
            return None
        return arr

def _load_one_run(base_name: str) -> Tuple[str, np.ndarray]:
    # 以 CH0 的 max_ch0 作为该 run 的事件基准长度，其他特征不足部分保留 NaN。
    base_arr = _read_1d_dataset_if_exists(CH_PARAM_DIRS[0] / base_name, "max_ch0")
    if base_arr is None:
        raise RuntimeError(f"{base_name}: 缺少基准数据集 ch0_max_ch0")
    n_events = int(base_arr.shape[0])
    run_data: Dict[str, np.ndarray] = {"ch0_max_ch0": base_arr}

    for feat_name in TARGET_FEATURES:
        ch_str, dset_name = feat_name.split("_", 1)   # ch0, max_ch0
        ch_idx = int(ch_str[2:])
        h5_path = CH_PARAM_DIRS[ch_idx] / base_name
        arr = _read_1d_dataset_if_exists(h5_path, dset_name)

        if arr is not None:
            run_data[feat_name] = arr

    mat = np.full((n_events, len(TARGET_FEATURES)), np.nan, dtype=np.float64)
    for j, feat_name in enumerate(TARGET_FEATURES):
        arr = run_data.get(feat_name)
        if arr is not None:
            n = min(n_events, int(arr.shape[0]))
            if n > 0:
                mat[:n, j] = arr[:n]
    return base_name, mat


def _impute_nan_with_column_median(X: np.ndarray) -> np.ndarray:
    X_imp = X.copy()
    for j in range(X_imp.shape[1]):
        col = X_imp[:, j]
        finite_mask = np.isfinite(col)
        if not np.any(finite_mask):
            X_imp[:, j] = 0.0
            continue
        med = float(np.median(col[finite_mask]))
        col[~finite_mask] = med
        X_imp[:, j] = col
    return X_imp


def main() -> None:
    base_names = _list_base_names_from_ch0()
    n_workers = os.cpu_count() or 1
    print(f"[信息] 检测到 run 文件数: {len(base_names)}")
    print(f"[信息] 使用特征数: {len(TARGET_FEATURES)}")
    print(f"[信息] 并行进程数: {n_workers}")

    mats: List[np.ndarray] = []
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        fut_map = {ex.submit(_load_one_run, bn): bn for bn in base_names}
        for fut in as_completed(fut_map):
            bn = fut_map[fut]
            try:
                _, mat = fut.result()
                mats.append(mat)
            except Exception as exc:  # noqa: BLE001
                print(f"[警告] 跳过 {bn}: {exc}")
            done += 1
            if done % 20 == 0 or done == len(base_names):
                print(f"[进度] {done}/{len(base_names)}")

    if not mats:
        raise RuntimeError("没有成功加载任何 run，无法计算协方差。")

    X = np.vstack(mats)
    print(f"[信息] 合并后事件总数: {X.shape[0]}, 特征维度: {X.shape[1]}")

    # 先补 NaN，再标准化
    X_imputed = _impute_nan_with_column_median(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    lw = LedoitWolf()
    lw.fit(X_scaled)
    cov = lw.covariance_
    print(f"[信息] LedoitWolf 协方差矩阵形状: {cov.shape}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "LedoitWolf_covariance_matrix.png"

    # 图整体更紧凑：色块更小、colorbar 更细
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cov, cmap="coolwarm", aspect="equal")
    ax.set_title("LedoitWolf Covariance Matrix (Standardized Features)", fontsize=14)
    ax.set_xticks(np.arange(len(TARGET_FEATURES)))
    ax.set_yticks(np.arange(len(TARGET_FEATURES)))
    ax.set_xticklabels(TARGET_FEATURES, rotation=90, fontsize=10)
    ax.set_yticklabels(TARGET_FEATURES, fontsize=10)
    ax.tick_params(axis="both", which="major", length=3, pad=2)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, shrink=0.78, aspect=26)
    cbar.set_label("Covariance", rotation=270, labelpad=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    print(f"[完成] 矩阵图已保存: {out_png}")

    plt.show()


if __name__ == "__main__":
    main()
