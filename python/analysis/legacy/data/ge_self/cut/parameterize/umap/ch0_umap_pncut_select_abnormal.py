#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UMAP + PN-cut 关联可视化脚本。

流程：
1. 运行 UMAP（模仿 (ch0).py），得到二维 UMAP 分布；
2. 选取 UMAP1 < 5 的事件；
3. 在 PN-cut 图像（max_ch0 vs max_ch1）中，用不同颜色标注这些事件的位置。
"""

import os
import sys
from typing import Optional, Tuple, List, Dict

import h5py
import numpy as np
import matplotlib.pyplot as plt

# umap 目录的父目录是 parameterize
current_dir = os.path.dirname(os.path.abspath(__file__))
parameterize_dir = os.path.dirname(current_dir)
cut_dir = os.path.dirname(parameterize_dir)

if parameterize_dir not in sys.path:
    sys.path.insert(0, parameterize_dir)

import paramdistribution
from sklearn.preprocessing import StandardScaler
import umap

_select_events_in_1sigma_band = paramdistribution._select_events_in_1sigma_band
compute_all_parameters_for_events = paramdistribution.compute_all_parameters_for_events

PARAM_NAMES = [
    "Amax", "Amin", "Tmax", "Tmin", "Q",
    "Qpre", "Qprev", "ped", "pedt", "abs(pedt-ped)"
]

DEFAULT_PARAMS_CACHE_PATH = os.path.join(current_dir, "ch0_params_cache.h5")
UMAP1_THRESHOLD = 5.0
MAX_CH0_LOW = 1200.0   # UMAP1<5 & max_ch0<1200
MAX_CH0_HIGH = 1700.0  # max_ch0>7000 & UMAP1>=5

SAMPLING_INTERVAL_NS = 4.0


def _load_params_cache(cache_path: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """从 HDF5 加载参数，返回 (params_dict, X)。"""
    with h5py.File(cache_path, "r") as f:
        X = np.asarray(f["X"])
        names_str = f.attrs["param_names"]
        if isinstance(names_str, bytes):
            names_str = names_str.decode("utf-8")
        param_names = names_str.split("|")
        params_dict = {name: np.asarray(f[name]) for name in param_names}
    return params_dict, X


def _compute_max_ch0_ch1(
    ch0_3_file: str,
    selected_indices: np.ndarray,
    event_ranks: np.ndarray,
    ch0_idx: int = 0,
    ch1_idx: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 ±1σ 事件的 max_ch0, max_ch1。
    event_ranks 为在 selected_indices 中的下标。
    """
    with h5py.File(ch0_3_file, "r") as f:
        channel_data = f["channel_data"]
        phys_ch0 = channel_data[:, ch0_idx, selected_indices].astype(np.float64)
        phys_ch1 = channel_data[:, ch1_idx, selected_indices].astype(np.float64)
    max_ch0_all = phys_ch0.max(axis=0)
    max_ch1_all = phys_ch1.max(axis=0)
    max_ch0_1sigma = max_ch0_all[event_ranks]
    max_ch1_1sigma = max_ch1_all[event_ranks]
    return max_ch0_1sigma, max_ch1_1sigma


def _load_ch0_waveforms(
    ch0_3_file: str,
    global_indices: np.ndarray,
    ch0_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载指定事件的 CH0 原始波形。
    返回 (time_axis_us, waveforms) 其中 waveforms 形状为 (n_events, n_samples)。
    """
    with h5py.File(ch0_3_file, "r") as f:
        channel_data = f["channel_data"]
        n_samples = channel_data.shape[0]
        waveforms = np.stack([
            channel_data[:, ch0_idx, idx].astype(np.float64)
            for idx in global_indices
        ], axis=0)
    time_axis_us = np.arange(n_samples) * SAMPLING_INTERVAL_NS / 1000.0
    return time_axis_us, waveforms


def run_umap_pncut_visualization(
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
    umap1_threshold: float = UMAP1_THRESHOLD,
    max_ch0_low: float = MAX_CH0_LOW,
    max_ch0_high: float = MAX_CH0_HIGH,
    use_cached_params: bool = False,
    params_cache_path: Optional[str] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> str:
    """
    运行 UMAP，选取 UMAP1 < umap1_threshold 的事件，在 PN-cut 图中用不同颜色标注。

    返回：保存的图片路径（PN-cut 图）
    """
    cache_path = params_cache_path or DEFAULT_PARAMS_CACHE_PATH

    # 1. 选出 ±1σ 带内事件
    event_ranks, ch0_3_file_sel, ch5_file_sel, selected_indices = _select_events_in_1sigma_band(
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

    # 2. 计算 max_ch0, max_ch1（用于 PN-cut 图）
    max_ch0, max_ch1 = _compute_max_ch0_ch1(
        ch0_3_file_sel, selected_indices, event_ranks, ch0_idx, ch1_idx
    )
    print(f"max_ch0/max_ch1 形状: {max_ch0.shape}, {max_ch1.shape}")

    # 3. 参数矩阵与 UMAP
    if use_cached_params and os.path.isfile(cache_path):
        print(f"\n从缓存加载参数: {cache_path}")
        params_dict, X = _load_params_cache(cache_path)
        param_names = list(params_dict.keys())
        if X.shape[0] != n_events:
            raise ValueError(
                f"缓存事件数 {X.shape[0]} 与当前 ±1σ 事件数 {n_events} 不一致，"
                "请删除缓存或设置 use_cached_params=False"
            )
    else:
        params_dict = compute_all_parameters_for_events(
            ch0_3_file=ch0_3_file_sel,
            event_ranks=event_ranks,
            selected_indices=selected_indices,
            ch0_idx=ch0_idx,
            baseline_window_us=baseline_window_us,
            max_workers=max_workers,
        )
        param_names = PARAM_NAMES
        X = np.column_stack([params_dict[name] for name in param_names])
        print(f"参数矩阵形状: {X.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        random_state=42,
    )
    embedding = reducer.fit_transform(X_scaled)
    print(f"UMAP 嵌入形状: {embedding.shape}")

    # 4. 选取 UMAP1 < umap1_threshold 及异常分组
    umap1_lt5 = embedding[:, 0] < umap1_threshold
    n_umap1_lt5 = int(umap1_lt5.sum())
    print(f"UMAP1 < {umap1_threshold} 的事件数: {n_umap1_lt5} / {n_events}")

    # 异常组：A) UMAP1<5 & max_ch0<1200；B) max_ch0>7000 & UMAP1>=5
    mask_A = umap1_lt5 & (max_ch0 < max_ch0_low)
    mask_B = (~umap1_lt5) & (max_ch0 > max_ch0_high)
    n_A, n_B = int(mask_A.sum()), int(mask_B.sum())
    print(f"组 A (UMAP1<5 & max_ch0<{max_ch0_low}): {n_A} 事件")
    print(f"组 B (max_ch0>{max_ch0_high} & UMAP1≥5): {n_B} 事件")

    # 5. 绘图：UMAP + PN-cut 双图
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
    })
    fig, (ax_umap, ax_pncut) = plt.subplots(1, 2, figsize=(14, 6))

    # 6a. UMAP 图
    ax_umap.scatter(
        embedding[~umap1_lt5, 0], embedding[~umap1_lt5, 1],
        c="lightgray", s=5, alpha=0.6, label=f"UMAP1 ≥ {umap1_threshold}",
    )
    ax_umap.scatter(
        embedding[umap1_lt5, 0], embedding[umap1_lt5, 1],
        c="tab:red", s=8, alpha=0.8, label=f"UMAP1 < {umap1_threshold}",
    )
    ax_umap.scatter(
        embedding[mask_A, 0], embedding[mask_A, 1],
        c="magenta", s=80, alpha=0.9, marker="s", edgecolors="black", linewidths=1,
        label=f"A: UMAP1<5 & max_ch0<{max_ch0_low} (n={n_A})", zorder=5,
    )
    ax_umap.scatter(
        embedding[mask_B, 0], embedding[mask_B, 1],
        c="lime", s=80, alpha=0.9, marker="^", edgecolors="black", linewidths=1,
        label=f"B: max_ch0>{max_ch0_high} & UMAP1≥5 (n={n_B})", zorder=5,
    )
    ax_umap.axvline(x=umap1_threshold, color="gray", linestyle="--", alpha=0.7)
    ax_umap.set_xlabel("UMAP 1", fontsize=12, fontweight="bold")
    ax_umap.set_ylabel("UMAP 2", fontsize=12, fontweight="bold")
    ax_umap.set_title(f"UMAP 2D | n={n_events} events", fontsize=11, fontweight="bold")
    ax_umap.legend(loc="upper right", fontsize=8)
    ax_umap.grid(True, alpha=0.3)

    # 6b. PN-cut 图：所有 ±1σ 事件 + UMAP1<5 高亮 + 组 A/B 标注
    ax_pncut.scatter(
        max_ch0[~umap1_lt5], max_ch1[~umap1_lt5],
        c="lightblue", s=3, alpha=0.5, label=f"UMAP1 ≥ {umap1_threshold}",
    )
    ax_pncut.scatter(
        max_ch0[umap1_lt5], max_ch1[umap1_lt5],
        c="tab:red", s=8, alpha=0.8, label=f"UMAP1 < {umap1_threshold}",
    )
    ax_pncut.scatter(
        max_ch0[mask_A], max_ch1[mask_A],
        c="magenta", s=80, alpha=0.9, marker="s", edgecolors="black", linewidths=1,
        label=f"A: UMAP1<5 & max_ch0<{max_ch0_low} (n={n_A})", zorder=5,
    )
    ax_pncut.scatter(
        max_ch0[mask_B], max_ch1[mask_B],
        c="lime", s=80, alpha=0.9, marker="^", edgecolors="black", linewidths=1,
        label=f"B: max_ch0>{max_ch0_high} & UMAP1≥5 (n={n_B})", zorder=5,
    )
    ax_pncut.set_xlabel("max CH0 (ADC)", fontsize=12, fontweight="bold")
    ax_pncut.set_ylabel("max CH1 (ADC)", fontsize=12, fontweight="bold")
    ax_pncut.set_title(
        f"PN-cut: max CH0 vs max CH1 | UMAP1<{umap1_threshold} highlighted (n={n_umap1_lt5})",
        fontsize=11, fontweight="bold",
    )
    ax_pncut.legend(loc="upper left", fontsize=8)
    ax_pncut.grid(True, alpha=0.3)
    ax_pncut.set_xlim(max_ch0.min() - 100, max_ch0.max() + 100)
    ax_pncut.set_ylim(max_ch1.min() - 100, max_ch1.max() + 100)

    plt.tight_layout()

    # 7. 多窗口：分别显示组 A、组 B 的全部 CH0 原始波形
    indices_1sigma = event_ranks
    global_idx_A = selected_indices[indices_1sigma[mask_A]]
    global_idx_B = selected_indices[indices_1sigma[mask_B]]

    alpha_A = min(0.8, 0.4 + 0.4 / max(1, n_A)) if n_A > 0 else 0.6
    alpha_B = min(0.8, 0.4 + 0.4 / max(1, n_B)) if n_B > 0 else 0.6

    # Window: Group A (all waveforms)
    fig2, ax_wf_A = plt.subplots(1, 1, figsize=(10, 5))
    if n_A > 0:
        t_us, wfs_A = _load_ch0_waveforms(ch0_3_file_sel, global_idx_A, ch0_idx)
        for i in range(n_A):
            ax_wf_A.plot(t_us, wfs_A[i], color="magenta", alpha=alpha_A, lw=1)
        ax_wf_A.set_title(f"Group A: UMAP1<5 & max_ch0<{max_ch0_low} (n={n_A}, all waveforms)", fontsize=11, fontweight="bold")
    else:
        ax_wf_A.text(0.5, 0.5, "No events", ha="center", va="center", transform=ax_wf_A.transAxes)
    ax_wf_A.set_xlabel("Time (µs)", fontsize=10)
    ax_wf_A.set_ylabel("ADC", fontsize=10)
    ax_wf_A.grid(True, alpha=0.3)
    plt.tight_layout()

    # Window: Group B (all waveforms)
    fig3, ax_wf_B = plt.subplots(1, 1, figsize=(10, 5))
    if n_B > 0:
        t_us, wfs_B = _load_ch0_waveforms(ch0_3_file_sel, global_idx_B, ch0_idx)
        for i in range(n_B):
            ax_wf_B.plot(t_us, wfs_B[i], color="lime", alpha=alpha_B, lw=1)
        ax_wf_B.set_title(f"Group B: max_ch0>{max_ch0_high} & UMAP1≥5 (n={n_B}, all waveforms)", fontsize=11, fontweight="bold")
    else:
        ax_wf_B.text(0.5, 0.5, "No events", ha="center", va="center", transform=ax_wf_B.transAxes)
    ax_wf_B.set_xlabel("Time (µs)", fontsize=10)
    ax_wf_B.set_ylabel("ADC", fontsize=10)
    ax_wf_B.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is None:
        ge_self_dir = os.path.dirname(cut_dir)
        data_dir = os.path.dirname(ge_self_dir)
        python_dir = os.path.dirname(data_dir)
        project_root = os.path.dirname(python_dir)
        output_dir = os.path.join(project_root, "images", "presentation")
        os.makedirs(output_dir, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(
            output_dir, f"ch0_umap_pncut_{n_events}events_umap1lt{umap1_threshold}_{timestamp}.png"
        )

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"UMAP+PN-cut 可视化已保存至: {save_path}")

    base, ext = os.path.splitext(save_path)
    save_path_wf_A = base + "_waveforms_groupA" + (ext or ".png")
    save_path_wf_B = base + "_waveforms_groupB" + (ext or ".png")
    fig2.savefig(save_path_wf_A, dpi=300, bbox_inches="tight", facecolor="white")
    fig3.savefig(save_path_wf_B, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"CH0 waveforms Group A saved to: {save_path_wf_A}")
    print(f"CH0 waveforms Group B saved to: {save_path_wf_B}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig2)
        plt.close(fig3)

    return save_path


if __name__ == "__main__":
    try:
        run_umap_pncut_visualization(
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
            max_workers=None,
            umap_n_neighbors=25,
            umap_min_dist=0.1,
            umap1_threshold=5.0,
            use_cached_params=True,
            params_cache_path=os.path.join(current_dir, "ch0_params_cache.h5"),
            save_path=None,
            show_plot=True,
        )
    except Exception as e:
        print(f"\nUMAP+PN-cut 可视化失败: {e}")
        import traceback
        traceback.print_exc()
