#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PN-cut 选中事件的 CH0 波形 8 参数 AP 聚类可视化脚本。

流程：
1. 使用 PN-cut 选出 ±1σ 带内的事件；
2. 使用所有 CPU 并行处理所有波形，得到 n×8 参数矩阵；
3. 对 8 个参数使用 StandardScaler 标准化；
4. 在标准化 8 维参数空间上进行 Affinity Propagation 聚类（簇数由算法自适应决定，不做人为限制）；
5. 使用多种常见的 AP 聚类可视化形式展示各个簇的分布与特征。

8 参数（来自 parameter(ch0).py）：
Amax, μ(ped), μ(pedt), σ(ped), σ(pedt), Tmax, RMS_ped, RMS_pedt
"""

import os
import sys
import importlib.util
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import h5py
import numpy as np
import matplotlib.pyplot as plt

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))   # .../parameterize/umap-optimazation-1
parameterize_dir = os.path.dirname(current_dir)             # .../parameterize
cut_dir = os.path.dirname(parameterize_dir)                 # .../cut

if parameterize_dir not in sys.path:
    sys.path.insert(0, parameterize_dir)

# 导入 paramdistribution（事件选择）
import paramdistribution
_select_events_in_1sigma_band = paramdistribution._select_events_in_1sigma_band

# 导入 parameter(ch0).py 的 compute_eight_params（文件名含括号，需用 importlib）
param_ch0_path = os.path.join(parameterize_dir, "parameter(ch0).py")
spec_param = importlib.util.spec_from_file_location("parameter_ch0", param_ch0_path)
parameter_ch0 = importlib.util.module_from_spec(spec_param)
assert spec_param.loader is not None
spec_param.loader.exec_module(parameter_ch0)
compute_eight_params = parameter_ch0.compute_eight_params

# 中值滤波
python_dir = os.path.dirname(os.path.dirname(os.path.dirname(parameterize_dir)))
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)
from utils.filter import median_filter

# StandardScaler 与 AP 聚类
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import pairwise_distances

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# 参数缓存文件默认路径（脚本同目录）
DEFAULT_PARAMS_CACHE_PATH = os.path.join(current_dir, "ch0_8params_cache.h5")

# 8 参数名称（与 parameter(ch0).py 一致）
PARAM_NAMES = [
    "Amax", "μ(ped)", "μ(pedt)", "σ(ped)", "σ(pedt)", "Tmax", "RMS_ped", "RMS_pedt"
]
PARAM_KEYS = [
    "amax", "mu_ped", "mu_pedt", "sigma_ped", "sigma_pedt", "tmax_us", "rms_ped", "rms_pedt"
]


def _process_single_event_eight(args: Tuple) -> Tuple[int, Dict[str, float]]:
    """工作函数：对单条波形计算 8 参数。args = (rank, waveform, sampling_interval_ns, baseline_window_us)"""
    rank, waveform, sampling_interval_ns, baseline_window_us = args
    wf = median_filter(np.asarray(waveform, dtype=np.float64), kernel_size=3)
    params = compute_eight_params(wf, sampling_interval_ns=sampling_interval_ns, baseline_window_us=baseline_window_us)
    return rank, params


def compute_eight_params_for_events(
    ch0_3_file: str,
    event_ranks: np.ndarray,
    selected_indices: np.ndarray,
    ch0_idx: int = 0,
    baseline_window_us: float = 2.0,
    max_workers: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """对选中的事件计算 8 参数（多进程并行）。"""
    print("\n正在计算所有事件的 8 参数...")
    sampling_interval_ns = 4.0
    n_events = event_ranks.size

    with h5py.File(ch0_3_file, "r") as f_ch0:
        ch0_channel_data = f_ch0["channel_data"]
        global_indices = selected_indices[event_ranks]
        all_waveforms = ch0_channel_data[:, ch0_idx, global_indices].astype(np.float64)
        waveforms_list = [all_waveforms[:, i] for i in range(n_events)]

    print(f"已读取 {n_events} 个波形，准备并行处理...")
    task_args = [
        (int(rank), waveforms_list[i], sampling_interval_ns, baseline_window_us)
        for i, rank in enumerate(event_ranks)
    ]

    if max_workers is None:
        max_workers = os.cpu_count() or 1
    print(f"使用 {max_workers} 个 CPU 核心进行并行计算")

    result_arrays = {key: np.zeros(n_events, dtype=np.float64) for key in PARAM_KEYS}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_rank = {executor.submit(_process_single_event_eight, args): args[0] for args in task_args}
        with tqdm(total=n_events, desc="计算8参数", unit="事件", ncols=100) as pbar:
            for future in as_completed(future_to_rank):
                try:
                    rank, result = future.result()
                    idx = np.where(event_ranks == rank)[0][0]
                    for k in PARAM_KEYS:
                        result_arrays[k][idx] = result[k]
                except Exception as e:
                    print(f"\n警告: 处理事件 rank={future_to_rank[future]} 时出错: {e}")
                pbar.update(1)

    params_dict = {PARAM_NAMES[i]: result_arrays[PARAM_KEYS[i]] for i in range(len(PARAM_NAMES))}
    return params_dict


def _save_params_cache(cache_path: str, params_dict: Dict[str, np.ndarray], param_names: List[str]) -> None:
    """将参数矩阵保存到 HDF5。"""
    with h5py.File(cache_path, "w") as f:
        X = np.column_stack([params_dict[name] for name in param_names])
        f.create_dataset("X", data=X)
        for name in param_names:
            f.create_dataset(name, data=params_dict[name])
        f.attrs["param_names"] = "|".join(param_names)


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


def run_ap_clustering_visualization(
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
    对 PN-cut ±1σ 带内事件的 CH0 波形，并行提取 8 参数 → StandardScaler → 在 8 维参数空间上进行
    Affinity Propagation 聚类（簇数由算法自适应决定），然后使用多种 AP 聚类可视化形式展示各簇事件的分布与特征。
    """
    cache_path = params_cache_path if params_cache_path is not None else DEFAULT_PARAMS_CACHE_PATH

    if use_cached_params and os.path.isfile(cache_path):
        print(f"\n从缓存加载参数: {cache_path}")
        params_dict, X = _load_params_cache(cache_path)
        param_names = list(params_dict.keys())
        n_events = X.shape[0]
        print(f"参数矩阵形状: {X.shape} (n_events × 8)")
    else:
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
        print(f"\n将对 {n_events} 个事件进行 8 参数并行提取...")

        params_dict = compute_eight_params_for_events(
            ch0_3_file=ch0_3_file_sel,
            event_ranks=event_ranks,
            selected_indices=selected_indices,
            ch0_idx=ch0_idx,
            baseline_window_us=baseline_window_us,
            max_workers=max_workers,
        )

        param_names = PARAM_NAMES
        X = np.column_stack([params_dict[name] for name in param_names])
        print(f"参数矩阵形状: {X.shape} (n_events × 8)")

        _save_params_cache(cache_path, params_dict, param_names)
        print(f"参数矩阵已保存至: {cache_path}")

    # StandardScaler 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("StandardScaler 标准化完成")

    # 在标准化 8 参数空间上进行 AP 聚类（簇数自适应）
    print("开始在标准化 8 参数空间上进行 Affinity Propagation 聚类...")
    # 计算相似度矩阵 S = -||x_i - x_j||^2，并基于相似度分布自动选择 preference
    S = -pairwise_distances(X_scaled, metric="sqeuclidean")
    sim_values = S[np.triu_indices_from(S, k=1)]
    sim_median = float(np.median(sim_values))
    sim_p10 = float(np.percentile(sim_values, 1))
    preference = sim_p10
    print(f"AP 相似度: median={sim_median:.3f}, 10% 分位={sim_p10:.3f}，使用 preference=10% 分位以获得较少但有代表性的簇数。")

    ap = AffinityPropagation(affinity="precomputed", preference=preference, damping=0.9, random_state=42)
    ap.fit(S)
    labels = ap.labels_
    unique_labels = np.unique(labels)
    n_ap_clusters = unique_labels.size
    print(f"AP 聚类得到 {n_ap_clusters} 个簇")

    # 计算每一簇在原始 8 参数空间中的均值（簇中心），以及每簇样本数
    cluster_sizes = [int((labels == k).sum()) for k in unique_labels]
    centers = np.zeros((n_ap_clusters, len(param_names)), dtype=np.float64)
    for row_idx, k in enumerate(unique_labels):
        mask_k = (labels == k)
        if not np.any(mask_k):
            continue
        for j, name in enumerate(param_names):
            centers[row_idx, j] = params_dict[name][mask_k].mean()

    # 一些便于可视化的典型二维投影（选择几个参数对）
    # 1) Amax vs Tmax
    try:
        idx_amax = param_names.index("Amax")
    except ValueError:
        idx_amax = 0
    try:
        idx_tmax = param_names.index("Tmax")
    except ValueError:
        idx_tmax = min(1, len(param_names) - 1)

    x_amax = params_dict[param_names[idx_amax]]
    y_tmax = params_dict[param_names[idx_tmax]]

    # 2) μ(ped) vs σ(ped)
    try:
        idx_mu_ped = param_names.index("μ(ped)")
    except ValueError:
        idx_mu_ped = 0
    try:
        idx_sigma_ped = param_names.index("σ(ped)")
    except ValueError:
        idx_sigma_ped = min(1, len(param_names) - 1)

    x_mu_ped = params_dict[param_names[idx_mu_ped]]
    y_sigma_ped = params_dict[param_names[idx_sigma_ped]]

    # 准备绘图
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax_scatter1 = axes[0, 0]
    ax_scatter2 = axes[0, 1]
    ax_bar = axes[1, 0]
    ax_heatmap = axes[1, 1]

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in range(n_ap_clusters)]
    label_names = [f"Cluster {int(k)}" for k in unique_labels]

    # 图 1：Amax vs Tmax 散点，按簇着色
    for row_idx, k in enumerate(unique_labels):
        mask_k = (labels == k)
        if not np.any(mask_k):
            continue
        ax_scatter1.scatter(
            x_amax[mask_k],
            y_tmax[mask_k],
            s=8,
            alpha=0.7,
            color=colors[row_idx],
            label=f"{label_names[row_idx]} (n={cluster_sizes[row_idx]})",
        )
    ax_scatter1.set_xlabel(param_names[idx_amax], fontsize=12, fontweight="bold")
    ax_scatter1.set_ylabel(param_names[idx_tmax], fontsize=12, fontweight="bold")
    ax_scatter1.set_title("AP 各簇事件在 (Amax, Tmax) 空间的分布", fontsize=11, fontweight="bold")
    ax_scatter1.grid(True, alpha=0.3)
    ax_scatter1.legend(loc="best", fontsize=9)

    # 图 2：μ(ped) vs σ(ped) 散点，按簇着色
    for row_idx, k in enumerate(unique_labels):
        mask_k = (labels == k)
        if not np.any(mask_k):
            continue
        ax_scatter2.scatter(
            x_mu_ped[mask_k],
            y_sigma_ped[mask_k],
            s=8,
            alpha=0.7,
            color=colors[row_idx],
            label=f"{label_names[row_idx]} (n={cluster_sizes[row_idx]})",
        )
    ax_scatter2.set_xlabel(param_names[idx_mu_ped], fontsize=12, fontweight="bold")
    ax_scatter2.set_ylabel(param_names[idx_sigma_ped], fontsize=12, fontweight="bold")
    ax_scatter2.set_title("AP 各簇事件在 (μ(ped), σ(ped)) 空间的分布", fontsize=11, fontweight="bold")
    ax_scatter2.grid(True, alpha=0.3)

    # 图 3：每个簇的样本数柱状图
    x_pos = np.arange(n_ap_clusters)
    ax_bar.bar(x_pos, cluster_sizes, color=colors[:n_ap_clusters])
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(label_names, rotation=30, ha="right", fontsize=9)
    ax_bar.set_ylabel("Counts", fontsize=12, fontweight="bold")
    ax_bar.set_title("AP 各簇事件的样本数", fontsize=11, fontweight="bold")

    # 图 4：各簇在 8 个参数上的簇中心（均值）热力图
    im = ax_heatmap.imshow(centers, aspect="auto", cmap="viridis")
    ax_heatmap.set_xticks(range(len(param_names)))
    ax_heatmap.set_xticklabels(param_names, rotation=45, ha="right", fontsize=9)
    ax_heatmap.set_yticks(np.arange(n_ap_clusters))
    ax_heatmap.set_yticklabels(label_names, fontsize=9)
    ax_heatmap.set_title("AP 各簇在 8 参数空间中的簇中心（均值）", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"CH0 8-Parameter AP Clustering\n"
        f"n={n_events} events  |  StandardScaler + AP (8D)  |  n_clusters={n_ap_clusters}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path is None:
        ge_self_dir = os.path.dirname(cut_dir)
        data_dir = os.path.dirname(ge_self_dir)
        python_dir = os.path.dirname(data_dir)
        project_root = os.path.dirname(python_dir)
        output_dir = os.path.join(project_root, "images", "presentation")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(
            output_dir,
            f"ch0_8param_ap_{n_ap_clusters}clusters_{n_events}events_{timestamp}.png",
        )

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"AP 聚类可视化已保存至: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return save_path


if __name__ == "__main__":
    try:
        run_ap_clustering_visualization(
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
            save_path=None,
            show_plot=True,
            use_cached_params=True,
            params_cache_path=None,
        )
    except Exception as e:
        print(f"\n8 参数 AP 聚类可视化失败: {e}")
        import traceback
        traceback.print_exc()
