#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PN-cut 选中事件的 CH0 波形 8 参数 Diffusion Maps 降维可视化脚本。

流程：
1. 使用 PN-cut 选出 ±1σ 带内的事件；
2. 使用所有 CPU 并行处理所有波形，得到 n×8 参数矩阵；
3. 对 8 个参数使用 StandardScaler 标准化；
4. 使用 Diffusion Maps 将 8 维参数映射到二维；
5. 可视化二维散点图，并可通过 RadioButtons 切换着色参数。

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
from matplotlib.widgets import RadioButtons

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

# StandardScaler 和 Diffusion Maps 所需工具
from sklearn.preprocessing import StandardScaler
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


def _diffusion_maps_embedding(
    X: np.ndarray,
    n_components: int = 2,
    epsilon: Optional[float] = None,
) -> np.ndarray:
    """
    使用简单的 Diffusion Maps 对数据 X 进行降维。

    参数:
        X: 标准化后的数据矩阵，形状 (n_samples, n_features)
        n_components: 需要的扩散坐标维数
        epsilon: 高斯核尺度，如果为 None 则使用距离中位数自适应设定

    返回:
        embedding: 形状 (n_samples, n_components) 的二维嵌入坐标
    """
    n_samples = X.shape[0]
    if n_samples == 0:
        return np.zeros((0, n_components), dtype=np.float64)

    # 计算两两欧氏距离
    dists = pairwise_distances(X, metric="euclidean")

    if epsilon is None:
        # 使用非零距离的中位数作为尺度，并略微放大，避免过于局部导致点挤在一起
        non_zero = dists[dists > 0]
        if non_zero.size == 0:
            epsilon = 1.0
        else:
            epsilon = (np.median(non_zero) ** 2) * 2.0

    # 高斯核矩阵（全连接）
    K = np.exp(-(dists ** 2) / epsilon)

    # 仅保留每行最近的若干个邻居，增强局部结构，避免“全局一坨”
    k_neighbors = min(30, n_samples - 1) if n_samples > 1 else 0
    if k_neighbors > 0:
        # 对每一行，找出距离最近的 k_neighbors 个索引
        idx_sorted = np.argsort(dists, axis=1)
        neighbor_mask = np.zeros_like(K, dtype=bool)
        for i in range(n_samples):
            neighbor_mask[i, idx_sorted[i, 1 : k_neighbors + 1]] = True  # 跳过自己（第一个是自身）
        K[~neighbor_mask] = 0.0

    # 行归一化得到 Markov 矩阵 P
    row_sums = K.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = K / row_sums

    # 特征分解（右特征向量），取实部
    eigvals, eigvecs = np.linalg.eig(P)
    eigvals = eigvals.real
    eigvecs = eigvecs.real

    # 按特征值从大到小排序
    idx = np.argsort(-eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # 跳过第一个平凡特征向量（对应特征值约为 1），
    # 在后续非平凡特征向量中，根据方差选出“变化最大”的 n_components 个方向，
    # 避免某个扩散坐标几乎为常数、在图上挤成一条线。
    if eigvecs.shape[1] <= 1:
        return np.zeros((n_samples, n_components), dtype=np.float64)

    non_trivial = eigvecs[:, 1:]  # 形状 (n_samples, n_non_trivial)
    # 每个非平凡特征向量在样本上的方差
    variances = non_trivial.var(axis=0)
    # 方差从大到小排序
    var_idx = np.argsort(-variances)
    take = min(n_components, non_trivial.shape[1])
    selected = non_trivial[:, var_idx[:take]]

    # 如果实际选到的维数少于 n_components，就在后面补零列，保持返回形状一致
    if selected.shape[1] < n_components:
        pad = np.zeros((n_samples, n_components - selected.shape[1]), dtype=np.float64)
        selected = np.hstack([selected, pad])

    return selected


def run_diffusion_visualization(
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
    对 PN-cut ±1σ 带内事件的 CH0 波形，并行提取 8 参数 → StandardScaler → Diffusion Maps 2D → 可视化。
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

    # Diffusion Maps 降维到 2D
    embedding = _diffusion_maps_embedding(X_scaled, n_components=2)
    # 为了在图上更均匀铺开，对嵌入结果做简单标准化（零均值、单位方差）
    if embedding.shape[0] > 0:
        embedding = (embedding - embedding.mean(axis=0)) / (embedding.std(axis=0) + 1e-12)
    print(f"Diffusion Maps 降维完成，嵌入形状: {embedding.shape}")

    # 可视化
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes([0.08, 0.12, 0.62, 0.82])
    rax = fig.add_axes([0.74, 0.25, 0.22, 0.6])
    rax.set_title("着色参数", fontsize=11, fontweight="bold")

    current_param = param_names[0]
    c_values = params_dict[current_param]
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=c_values, cmap="viridis", s=5, alpha=0.7,
    )
    ax.set_xlabel("Diffusion component 1", fontsize=14, fontweight="bold")
    ax.set_ylabel("Diffusion component 2", fontsize=14, fontweight="bold")
    ax.set_title(
        f"CH0 8-Parameter Diffusion Maps\n"
        f"n={n_events} events  |  StandardScaler + Diffusion Maps (2D)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    cbar_ax = fig.add_axes([0.62, 0.12, 0.02, 0.82])
    cbar = fig.colorbar(scatter, cax=cbar_ax, label=current_param)

    def on_param_select(label: str) -> None:
        c_new = params_dict[label]
        scatter.set_array(c_new)
        scatter.set_clim(c_new.min(), c_new.max())
        cbar.set_label(label, fontsize=10)
        fig.canvas.draw_idle()

    radio = RadioButtons(rax, param_names, active=0)
    radio.on_clicked(on_param_select)

    if save_path is None:
        ge_self_dir = os.path.dirname(cut_dir)
        data_dir = os.path.dirname(ge_self_dir)
        python_dir = os.path.dirname(data_dir)
        project_root = os.path.dirname(python_dir)
        output_dir = os.path.join(project_root, "images", "presentation")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(output_dir, f"ch0_8param_diffusion_{n_events}events_{timestamp}.png")

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Diffusion Maps 可视化已保存至: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return save_path


if __name__ == "__main__":
    try:
        run_diffusion_visualization(
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
        print(f"\n8 参数 Diffusion Maps 可视化失败: {e}")
        import traceback
        traceback.print_exc()
