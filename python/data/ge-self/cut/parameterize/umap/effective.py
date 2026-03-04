#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
显示 ch0_umap&pncut 筛选出来但在 ch0_ch3_umap&pncut 中未筛选出来的事件（差集）的 CH0 波形。

流程：
1. 获取 ±1σ 带内事件（与两个脚本相同）；
2. ch0_umap：10 参数 UMAP，选取 UMAP1 < 5 的事件；
3. ch0_ch3：15 参数 UMAP，选取 UMAP1 > 4 的事件；
4. 差集 = ch0 选中 且 ch0_ch3 未选中；
5. 显示差集事件的 CH0 波形。
"""

import os
import sys
import importlib.util
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import matplotlib.pyplot as plt

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

# 通过 importlib 加载 ch0_umap 和 ch0_ch3 模块（文件名含 &）
_ch0_umap_path = os.path.join(current_dir, "ch0_umap&pncut.py")
_ch0_ch3_path = os.path.join(current_dir, "ch0_ch3_umap&pncut.py")

CH0_UMAP_THRESHOLD = 5.0   # ch0_umap: UMAP1 < 5
CH0_CH3_UMAP_THRESHOLD = 4.0  # ch0_ch3: UMAP1 > 4


def _load_ch0_umap_module():
    spec = importlib.util.spec_from_file_location("ch0_umap_mod", _ch0_umap_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_ch0_ch3_umap_module():
    spec = importlib.util.spec_from_file_location("ch0_ch3_umap_mod", _ch0_ch3_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_params_cache(cache_path: str):
    with h5py.File(cache_path, "r") as f:
        X = np.asarray(f["X"])
        names_str = f.attrs["param_names"]
        if isinstance(names_str, bytes):
            names_str = names_str.decode("utf-8")
        param_names = names_str.split("|")
        params_dict = {name: np.asarray(f[name]) for name in param_names}
    return params_dict, X


def run_effective_waveform_visualization(
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
    umap_n_neighbors: int = 25,
    umap_min_dist: float = 0.1,
    use_cached_params: bool = True,
    ch0_cache_path: Optional[str] = None,
    ch0_ch3_cache_path: Optional[str] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    max_waveforms_display: int = 24,
) -> str:
    """
    计算差集（ch0 选中 - ch0_ch3 选中），并显示差集事件的 CH0 波形。

    返回：保存的图片路径
    """
    ch0_umap_mod = _load_ch0_umap_module()
    ch0_ch3_mod = _load_ch0_ch3_umap_module()

    ch0_cache = ch0_cache_path or ch0_umap_mod.DEFAULT_PARAMS_CACHE_PATH
    ch0_ch3_cache = ch0_ch3_cache_path or ch0_ch3_mod.DEFAULT_PARAMS_CACHE_PATH

    # 1. 选出 ±1σ 带内事件
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
    print(f"±1σ 带内事件数: {n_events}")

    # 2. ch0 10 参数 UMAP → 选取 UMAP1 < 5
    if use_cached_params and os.path.isfile(ch0_cache):
        params_dict_ch0, X_ch0 = _load_params_cache(ch0_cache)
        param_names_ch0 = list(params_dict_ch0.keys())
    else:
        params_dict_ch0 = compute_all_parameters_for_events(
            ch0_3_file=ch0_3_file_sel,
            event_ranks=event_ranks,
            selected_indices=selected_indices,
            ch0_idx=ch0_idx,
            baseline_window_us=baseline_window_us,
            max_workers=max_workers,
        )
        param_names_ch0 = ch0_umap_mod.PARAM_NAMES
        X_ch0 = np.column_stack([params_dict_ch0[n] for n in param_names_ch0])
    if X_ch0.shape[0] != n_events:
        raise ValueError(f"ch0 缓存事件数 {X_ch0.shape[0]} != ±1σ 事件数 {n_events}")

    scaler_ch0 = StandardScaler()
    X_ch0_scaled = scaler_ch0.fit_transform(X_ch0)
    reducer_ch0 = umap.UMAP(n_components=2, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, random_state=42)
    emb_ch0 = reducer_ch0.fit_transform(X_ch0_scaled)
    ch0_selected = emb_ch0[:, 0] < CH0_UMAP_THRESHOLD
    print(f"ch0_umap (UMAP1<{CH0_UMAP_THRESHOLD}) 选中: {ch0_selected.sum()} 个事件")

    # 3. ch0_ch3 15 参数 UMAP → 选取 UMAP1 > 4
    if use_cached_params and os.path.isfile(ch0_ch3_cache):
        params_dict_ch3, X_ch3 = _load_params_cache(ch0_ch3_cache)
    else:
        with ThreadPoolExecutor(max_workers=2) as exe:
            f_ch0 = exe.submit(
                compute_all_parameters_for_events,
                ch0_3_file=ch0_3_file_sel,
                event_ranks=event_ranks,
                selected_indices=selected_indices,
                ch0_idx=ch0_idx,
                baseline_window_us=baseline_window_us,
                max_workers=max_workers,
            )
            f_ch3 = exe.submit(
                ch0_ch3_mod._compute_fast_params_all,
                ch0_3_file_sel, selected_indices, event_ranks,
                baseline_window_us=baseline_window_us,
                max_workers=max_workers,
            )
            ch0_dict = f_ch0.result()
            fast_dict = f_ch3.result()
        params_dict_ch3 = {**ch0_dict, **fast_dict}
        X_ch3 = np.column_stack([params_dict_ch3[n] for n in ch0_ch3_mod.PARAM_NAMES])
    if X_ch3.shape[0] != n_events:
        raise ValueError(f"ch0_ch3 缓存事件数 {X_ch3.shape[0]} != ±1σ 事件数 {n_events}")

    scaler_ch3 = StandardScaler()
    X_ch3_scaled = scaler_ch3.fit_transform(X_ch3)
    reducer_ch3 = umap.UMAP(n_components=2, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, random_state=42)
    emb_ch3 = reducer_ch3.fit_transform(X_ch3_scaled)
    ch3_selected = emb_ch3[:, 0] > CH0_CH3_UMAP_THRESHOLD
    print(f"ch0_ch3 (UMAP1>{CH0_CH3_UMAP_THRESHOLD}) 选中: {ch3_selected.sum()} 个事件")

    # 4. 差集：ch0 选中 且 ch0_ch3 未选中
    diff_mask = ch0_selected & ~ch3_selected
    diff_indices = np.where(diff_mask)[0]
    n_diff = diff_indices.size
    print(f"差集 (ch0选中 且 ch0_ch3未选中): {n_diff} 个事件")

    if n_diff == 0:
        print("差集为空，无可显示波形。")
        return ""

    # 5. 获取差集事件的全局索引并加载 CH0 波形
    global_indices = selected_indices[event_ranks]
    diff_global = global_indices[diff_indices]

    with h5py.File(ch0_3_file_sel, "r") as f:
        channel_data = f["channel_data"]
        n_samples = channel_data.shape[0]
        time_us = np.arange(n_samples) * 4.0 / 1000.0  # 4 ns 采样
        waveforms = np.stack([channel_data[:, ch0_idx, int(g)].astype(np.float64) for g in diff_global], axis=0)

    # 6. 显示波形（子图网格，最多 max_waveforms_display 个）
    n_show = min(n_diff, max_waveforms_display)
    n_cols = min(4, n_show)
    n_rows = (n_show + n_cols - 1) // n_cols

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True, squeeze=False)

    for k in range(n_show):
        i, j = k // n_cols, k % n_cols
        ax = axes[i, j]
        ax.plot(time_us, waveforms[k], "b-", linewidth=1)
        ax.set_title(f"Event #{diff_global[k]} (rank {diff_indices[k]})", fontsize=9)
        ax.set_ylabel("ADC")
        ax.grid(True, alpha=0.3)

    for ax in axes.flat[n_show:]:
        ax.set_visible(False)

    axes[-1, 0].set_xlabel("Time (µs)", fontsize=11)
    fig.suptitle(
        f"CH0 Waveforms: ch0_umap 选中 且 ch0_ch3 未选中 (差集, n={n_diff})\n"
        f"ch0: UMAP1<{CH0_UMAP_THRESHOLD} | ch0_ch3: UMAP1>{CH0_CH3_UMAP_THRESHOLD}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    if save_path is None:
        ge_self_dir = os.path.dirname(cut_dir)
        data_dir = os.path.dirname(ge_self_dir)
        python_dir = os.path.dirname(data_dir)
        project_root = os.path.dirname(python_dir)
        output_dir = os.path.join(project_root, "images", "presentation")
        os.makedirs(output_dir, exist_ok=True)
        from datetime import datetime
        save_path = os.path.join(output_dir, f"effective_ch0_waveforms_{n_diff}events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"波形图已保存至: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return save_path


if __name__ == "__main__":
    try:
        run_effective_waveform_visualization(
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
            use_cached_params=True,
            max_waveforms_display=24,
            show_plot=True,
        )
    except Exception as e:
        print(f"\neffective 波形可视化失败: {e}")
        import traceback
        traceback.print_exc()
