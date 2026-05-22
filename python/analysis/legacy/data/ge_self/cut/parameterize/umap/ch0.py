#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PN-cut 选中事件的 CH0 波形 10 参数 UMAP 降维可视化脚本。

流程：
1. 使用 PN-cut 选出 ±1σ 带内的事件；
2. 使用所有 CPU 并行处理所有波形，得到 n×10 参数矩阵；
3. 对 10 个参数使用 StandardScaler 标准化；
4. 使用 UMAP 降维到二维；
5. 可视化二维散点图。
"""

import os
import sys
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons

# umap 目录的父目录是 parameterize，需加入路径以便子进程能 import paramdistribution
current_dir = os.path.dirname(os.path.abspath(__file__))  # .../cut/parameterize/umap
parameterize_dir = os.path.dirname(current_dir)            # .../cut/parameterize
cut_dir = os.path.dirname(parameterize_dir)                # .../cut

if parameterize_dir not in sys.path:
    sys.path.insert(0, parameterize_dir)

# 使用标准 import，避免 ProcessPoolExecutor 子进程无法 pickle 动态加载的模块
import paramdistribution

_select_events_in_1sigma_band = paramdistribution._select_events_in_1sigma_band
compute_all_parameters_for_events = paramdistribution.compute_all_parameters_for_events

# UMAP 和 StandardScaler
from sklearn.preprocessing import StandardScaler
import umap


# 参数缓存文件默认路径（脚本同目录）
DEFAULT_PARAMS_CACHE_PATH = os.path.join(current_dir, "ch0_params_cache.h5")

# 10 参数名称（与 paramdistribution 一致）
PARAM_NAMES = [
    "Amax", "Amin", "Tmax", "Tmin", "Q",
    "Qpre", "Qprev", "ped", "pedt", "abs(pedt-ped)"
]


def _save_params_cache(cache_path: str, params_dict: Dict[str, np.ndarray], param_names: List[str]) -> None:
    """将参数矩阵及各列保存到 HDF5 文件。"""
    with h5py.File(cache_path, "w") as f:
        X = np.column_stack([params_dict[name] for name in param_names])
        f.create_dataset("X", data=X)
        for name in param_names:
            f.create_dataset(name, data=params_dict[name])
        f.attrs["param_names"] = "|".join(param_names)


def _load_params_cache(cache_path: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """从 HDF5 文件加载参数，返回 (params_dict, X)。"""
    with h5py.File(cache_path, "r") as f:
        X = np.asarray(f["X"])
        names_str = f.attrs["param_names"]
        if isinstance(names_str, bytes):
            names_str = names_str.decode("utf-8")
        param_names = names_str.split("|")
        params_dict = {name: np.asarray(f[name]) for name in param_names}
    return params_dict, X


def run_umap_visualization(
    ch0_3_file: Optional[str] = None,
    ch5_file: Optional[str] = None,
    rt_cut: float = 6000.0,
    ch0_threshold: float = 16382.0,
    ch0_idx: int = 0,
    ch1_idx: int = 1,
    x_min: float = 2000.0,
    x_max: float = 14000.0,
    sigma_factor: float = 1.0,
    baseline_window_us: float = 2.0,
    max_workers: Optional[int] = None,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    use_cached_params: bool = False,
    params_cache_path: Optional[str] = None,
) -> str:
    """
    对 PN-cut ±1σ 带内事件的 CH0 波形，并行提取 10 参数 → StandardScaler → UMAP 2D → 可视化。

    10 参数：Amax, Amin, Tmax, Tmin, Q, Qpre, Qprev, ped, pedt, abs(pedt-ped)

    参数：
        max_workers: 并行进程数，None 表示使用所有 CPU
        umap_n_neighbors: UMAP 的 n_neighbors
        umap_min_dist: UMAP 的 min_dist
        use_cached_params: 若 True 且缓存文件存在，则直接加载参数矩阵，跳过波形计算（仅改着色时可用）
        params_cache_path: 参数缓存 HDF5 路径，None 时使用脚本同目录下的 ch0_params_cache.h5

    返回：
        保存的图片路径
    """
    cache_path = params_cache_path if params_cache_path is not None else DEFAULT_PARAMS_CACHE_PATH

    if use_cached_params and os.path.isfile(cache_path):
        # 从缓存加载，跳过波形计算
        print(f"\n从缓存加载参数: {cache_path}")
        params_dict, X = _load_params_cache(cache_path)
        param_names = list(params_dict.keys())
        n_events = X.shape[0]
        print(f"参数矩阵形状: {X.shape} (n_events × 10)")
    else:
        # 1. 选出 ±1σ 带内的事件
        event_ranks, ch0_3_file_sel, _, selected_indices = _select_events_in_1sigma_band(
            ch0_3_file=ch0_3_file,
            ch5_file=ch5_file,
            rt_cut=rt_cut,
            ch0_threshold=ch0_threshold,
            ch0_idx=ch0_idx,
            ch1_idx=ch1_idx,
            x_min=x_min,
            x_max=x_max,
            sigma_factor=sigma_factor,
        )

        n_events = event_ranks.size
        print(f"\n将对 {n_events} 个事件进行并行参数提取...")

        # 2. 使用所有 CPU 并行处理，得到 n×10 参数矩阵
        params_dict = compute_all_parameters_for_events(
            ch0_3_file=ch0_3_file_sel,
            event_ranks=event_ranks,
            selected_indices=selected_indices,
            ch0_idx=ch0_idx,
            baseline_window_us=baseline_window_us,
            max_workers=max_workers,
        )

        # 构建 n×10 矩阵
        param_names = PARAM_NAMES
        X = np.column_stack([params_dict[name] for name in param_names])
        print(f"参数矩阵形状: {X.shape} (n_events × 10)")

        # 保存到 HDF5 缓存（脚本同目录）
        _save_params_cache(cache_path, params_dict, param_names)
        print(f"参数矩阵已保存至: {cache_path}")

    # 3. StandardScaler 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("StandardScaler 标准化完成")

    # 4. UMAP 降维到 2D
    reducer = umap.UMAP(n_components=2, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    print(f"UMAP 降维完成，嵌入形状: {embedding.shape}")

    # 5. 可视化：10 参数着色 + 按钮切换
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
    })
    fig = plt.figure(figsize=(12, 8))
    # 左侧：UMAP 散点图
    ax = fig.add_axes([0.08, 0.12, 0.62, 0.82])
    # 右侧：10 参数单选按钮
    rax = fig.add_axes([0.74, 0.25, 0.22, 0.6])
    rax.set_title("着色参数", fontsize=11, fontweight="bold")

    # 初始着色：Amax
    current_param = param_names[0]
    c_values = params_dict[current_param]
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=c_values,
        cmap="viridis",
        s=5,
        alpha=0.7,
    )
    ax.set_xlabel("UMAP 1", fontsize=14, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=14, fontweight="bold")
    ax.set_title(
        f"CH0 Waveform 10-Parameter UMAP\n"
        f"n={n_events} events  |  StandardScaler + UMAP(n_neighbors={umap_n_neighbors}, min_dist={umap_min_dist})",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    cbar_ax = fig.add_axes([0.62, 0.12, 0.02, 0.82])
    cbar = fig.colorbar(scatter, cax=cbar_ax, label=current_param)

    def on_param_select(label: str) -> None:
        """切换着色参数时更新散点颜色与 colorbar。"""
        c_new = params_dict[label]
        scatter.set_array(c_new)
        scatter.set_clim(c_new.min(), c_new.max())
        cbar.set_label(label, fontsize=10)
        fig.canvas.draw_idle()

    radio = RadioButtons(rax, param_names, active=0)
    radio.on_clicked(on_param_select)

    # 保存
    if save_path is None:
        ge_self_dir = os.path.dirname(cut_dir)
        data_dir = os.path.dirname(ge_self_dir)
        python_dir = os.path.dirname(data_dir)
        project_root = os.path.dirname(python_dir)
        output_dir = os.path.join(project_root, "images", "presentation")
        os.makedirs(output_dir, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(output_dir, f"ch0_umap_{n_events}events_{timestamp}.png")

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"UMAP 可视化已保存至: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return save_path


if __name__ == "__main__":
    try:
        run_umap_visualization(
            ch0_3_file=None,
            ch5_file=None,
            rt_cut=6000.0,
            ch0_threshold=16382.0,
            ch0_idx=0,
            ch1_idx=1,
            x_min=2000.0,
            x_max=14000.0,
            sigma_factor=1.0,
            baseline_window_us=2.0,
            max_workers=None,  # 使用所有 CPU
            umap_n_neighbors=25,
            umap_min_dist=0.1,
            save_path=None,
            show_plot=True,
            use_cached_params=False,  # 改为 True 可跳过波形计算，直接加载已保存的参数矩阵（仅改着色时用）
            params_cache_path=None,   # None 时使用脚本同目录下的 ch0_params_cache.h5
        )
    except Exception as e:
        print(f"\nUMAP 可视化失败: {e}")
        import traceback
        traceback.print_exc()
