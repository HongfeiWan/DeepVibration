#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
±1σ 带内事件的 CH0 波形 15 参数 UMAP + HDBSCAN 聚类可视化脚本。

不显示 PNCUT 图，不按 umap1_threshold 划分，仅用 HDBSCAN 对 UMAP 空间进行聚类。

流程：
1. 选出 ±1σ 带内的事件；
2. 使用所有 CPU 并行处理所有波形，得到 n×15 参数矩阵；
3. 对 15 个参数使用 StandardScaler 标准化；
4. 使用 UMAP 降维到二维；
5. 使用 HDBSCAN 对 UMAP 空间聚类，分为若干簇；
6. 绘制 UMAP 图，按 HDBSCAN 簇着色；
7. 绘制各簇的 15 参数分布图；
8. 从 n 个簇+噪声各抽 9 个 event 显示 CH3 原始波形。

15 参数：
- 8 个 CH0 波形参数（来自 parameter(ch0).py）：
  Amax, μ(ped), μ(pedt), σ(ped), σ(pedt), Tmax, RMS_ped, RMS_pedt
- 4 个快放 tanh 拟合参数 + 1 个拟合残差 RMS：
  tanh_p0, tanh_p1, tanh_p2, tanh_p3, tanh_RMS
- 1 个快放信号中 freq > 0.2 MHz 的频率成分能量占比：
  fast_highfreq_ratio
- 1 个快放二阶差分峰均比：
  fast_second_diff_peak_mean
"""

import os
import re
import sys
import importlib.util
from typing import Optional, Tuple, List, Dict
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

python_dir = os.path.dirname(os.path.dirname(os.path.dirname(parameterize_dir)))
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)
from utils.visualize import get_h5_files

# StandardScaler、UMAP 和 HDBSCAN
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# 参数缓存文件默认路径（脚本同目录）
DEFAULT_PARAMS_CACHE_PATH = os.path.join(current_dir, "ch0_15params_cache.h5")

# 14 参数名称：8 个 CH0 基础参数 + 4 个快放 tanh 拟合参数 + 1 个拟合残差 RMS + 1 个高频能量占比
PARAM_NAMES = [
    "Amax", "μ(ped)", "μ(pedt)", "σ(ped)", "σ(pedt)", "Tmax", "RMS_ped", "RMS_pedt",
    "tanh_p0", "tanh_p1", "tanh_p2", "tanh_p3", "tanh_RMS",
    "fast_highfreq_ratio",
]
PARAM_KEYS = [
    "amax", "mu_ped", "mu_pedt", "sigma_ped", "sigma_pedt", "tmax_us", "rms_ped", "rms_pedt",
    "tanh_p0", "tanh_p1", "tanh_p2", "tanh_p3", "tanh_rms",
    "fast_highfreq_ratio",
]
TANH_PARAM_NAMES = {"tanh_p0", "tanh_p1", "tanh_p2", "tanh_p3", "tanh_rms"}
# 分布图 y 轴取 log 可视化的参数
PARAM_NAMES_LOG_Y = {"Amax", "tanh_p0", "tanh_p1", "tanh_p2", "tanh_p3", "fast_highfreq_ratio"}

def _extract_file_id_from_name(name_no_ext: str) -> str:
    """
    从不带扩展名的文件名中提取“文件号”：
    - 若存在数字串（例如 ...Data_465_processed），取最后一段连续数字（这里为 465）；
    - 若不存在数字，则返回完整 name_no_ext。
    """
    matches = list(re.finditer(r"\d+", name_no_ext))
    if not matches:
        return name_no_ext
    return matches[-1].group(0)

def compute_params_for_events(
    ch0_3_file: str,
    event_ranks: np.ndarray,
    selected_indices: np.ndarray,
    ch0_idx: int = 0,
    ch3_idx: int = 3,
    baseline_window_us: float = 2.0,
    max_workers: Optional[int] = None,) -> Dict[str, np.ndarray]:
    """
    对选中的事件读取预计算好的 14 维参数（不再重新从波形计算）。

    - CH0 的 8 个参数来自 data/hdf5/raw_pulse/CH0_parameters 下对应文件：
      max_ch0, tmax_ch0, ch0ped_mean/var, ch0pedt_mean/var, ch0ped_rms, ch0pedt_rms；
    - CH3 的 5 个参数（4 个 tanh + 1 个 RMS）和 fast_highfreq_ratio
      来自 data/hdf5/raw_pulse/CH3_parameters 下对应文件。
    """
    del baseline_window_us, max_workers  # 仅为兼容旧接口，这里不再使用

    n_events = int(event_ranks.size)
    if n_events == 0:
        raise ValueError("event_ranks 为空，无法计算参数。")

    # 对应原始 h5 的全局事件索引
    global_indices = selected_indices[event_ranks]

    # 由 CH0-3 文件路径推断参数文件路径
    ch0_3_dir = os.path.dirname(ch0_3_file)              # .../CH0-3
    raw_pulse_dir = os.path.dirname(ch0_3_dir)           # .../raw_pulse
    base_name = os.path.basename(ch0_3_file)
    ch0_param_file = os.path.join(raw_pulse_dir, "CH0_parameters", base_name)
    ch3_param_file = os.path.join(raw_pulse_dir, "CH3_parameters", base_name)

    if not os.path.isfile(ch0_param_file):
        raise FileNotFoundError(f"未找到 CH0 参数文件: {ch0_param_file}")
    if not os.path.isfile(ch3_param_file):
        raise FileNotFoundError(f"未找到 CH3 参数文件: {ch3_param_file}")

    print("\n正在从预计算文件中读取 CH0 / CH3 参数...")

    # 读取 CH0 参数并转换为 8 维特征
    with h5py.File(ch0_param_file, "r") as f_ch0:
        max_ch0 = np.asarray(f_ch0["max_ch0"][global_indices], dtype=np.float64)
        tmax_samples = np.asarray(f_ch0["tmax_ch0"][global_indices], dtype=np.float64)
        ch0ped_mean = np.asarray(f_ch0["ch0ped_mean"][global_indices], dtype=np.float64)
        ch0pedt_mean = np.asarray(f_ch0["ch0pedt_mean"][global_indices], dtype=np.float64)
        ch0ped_var = np.asarray(f_ch0["ch0ped_var"][global_indices], dtype=np.float64)
        ch0pedt_var = np.asarray(f_ch0["ch0pedt_var"][global_indices], dtype=np.float64)
        ch0ped_rms = np.asarray(f_ch0["ch0ped_rms"][global_indices], dtype=np.float64)
        ch0pedt_rms = np.asarray(f_ch0["ch0pedt_rms"][global_indices], dtype=np.float64)

    # σ = sqrt(var)，Tmax 从 sample index 转换为 µs（采样间隔 4 ns）
    sigma_ped = np.sqrt(ch0ped_var)
    sigma_pedt = np.sqrt(ch0pedt_var)
    sampling_interval_ns = 4.0
    tmax_us = tmax_samples * sampling_interval_ns / 1000.0

    # 读取 CH3 拟合参数与高频能量占比
    with h5py.File(ch3_param_file, "r") as f_ch3:
        tanh_p0 = np.asarray(f_ch3["tanh_p0"][global_indices], dtype=np.float64)
        tanh_p1 = np.asarray(f_ch3["tanh_p1"][global_indices], dtype=np.float64)
        tanh_p2 = np.asarray(f_ch3["tanh_p2"][global_indices], dtype=np.float64)
        tanh_p3 = np.asarray(f_ch3["tanh_p3"][global_indices], dtype=np.float64)
        tanh_rms = np.asarray(f_ch3["tanh_rms"][global_indices], dtype=np.float64)
        fast_highfreq_ratio = np.asarray(
            f_ch3["highfreq_energy_ratio"][global_indices], dtype=np.float64
        )

    params_dict: Dict[str, np.ndarray] = {
        "Amax": max_ch0,
        "μ(ped)": ch0ped_mean,
        "μ(pedt)": ch0pedt_mean,
        "σ(ped)": sigma_ped,
        "σ(pedt)": sigma_pedt,
        "Tmax": tmax_us,
        "RMS_ped": ch0ped_rms,
        "RMS_pedt": ch0pedt_rms,
        "tanh_p0": tanh_p0,
        "tanh_p1": tanh_p1,
        "tanh_p2": tanh_p2,
        "tanh_p3": tanh_p3,
        "tanh_RMS": tanh_rms,
        "fast_highfreq_ratio": fast_highfreq_ratio,
    }

    # 构造特征矩阵 X（n_events × 14）
    X = np.column_stack([params_dict[name] for name in PARAM_NAMES])
    print(f"参数矩阵形状: {X.shape} (n_events × {len(PARAM_NAMES)})")

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

def run_umap_hdbscan_visualization(
    ch0_3_file: Optional[str] = None,
    ch5_file: Optional[str] = None,
    rt_cut: float = 6000.0,
    ch0_threshold: float = 16382.0,
    ch0_idx: int = 0,
    ch1_idx: int = 1,
    ch3_idx: int = 3,
    x_min: float = 2000.0,
    x_max: float = 14000.0,
    sigma_factor: float = 1.0,
    baseline_window_us: float = 2.0,
    max_workers: Optional[int] = None,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    feature_weights: Optional[List[float]] = None,
    hdbscan_min_cluster_size: int = 50,
    hdbscan_min_samples: Optional[int] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    use_cached_params: bool = False,
    params_cache_path: Optional[str] = None,) -> str:
    """
    对 ±1σ 带内事件：读取 14 维预计算参数，UMAP 降维，HDBSCAN 聚类，绘制 UMAP 图按簇着色。
    """
    cache_path = params_cache_path if params_cache_path is not None else DEFAULT_PARAMS_CACHE_PATH

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

    # 2. 参数矩阵与 UMAP
    if use_cached_params and os.path.isfile(cache_path):
        print(f"\n从缓存加载参数: {cache_path}")
        params_dict, X = _load_params_cache(cache_path)
        if X.shape[0] != n_events:
            raise ValueError(
                f"缓存事件数 {X.shape[0]} 与当前 ±1σ 事件数 {n_events} 不一致，"
                "请删除缓存或设置 use_cached_params=False"
            )
        if list(params_dict.keys()) != PARAM_NAMES or X.shape[1] != len(PARAM_NAMES):
            raise ValueError(
                "缓存特征维度与当前脚本不一致，请删除缓存或设置 use_cached_params=False"
            )
        print(f"参数矩阵形状: {X.shape} (n_events × {len(PARAM_NAMES)})")
    else:
        print(f"\n将对 {n_events} 个事件读取 14 维预计算参数...")
        params_dict = compute_params_for_events(
            ch0_3_file=ch0_3_file_sel,
            event_ranks=event_ranks,
            selected_indices=selected_indices,
            ch0_idx=ch0_idx,
            ch3_idx=ch3_idx,
            baseline_window_us=baseline_window_us,
            max_workers=max_workers,
        )
        X = np.column_stack([params_dict[name] for name in PARAM_NAMES])
        print(f"参数矩阵形状: {X.shape} (n_events × {len(PARAM_NAMES)})")
        _save_params_cache(cache_path, params_dict, PARAM_NAMES)
        print(f"参数矩阵已保存至: {cache_path}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("StandardScaler 标准化完成")

    if feature_weights is not None:
        weights = np.asarray(feature_weights, dtype=np.float64)
        if weights.shape[0] != len(PARAM_NAMES):
            raise ValueError(f"feature_weights length should be {len(PARAM_NAMES)}, got {weights.shape[0]}")
        X_for_umap = X_scaled * weights.reshape(1, -1)
        print(f"Feature weights: {dict(zip(PARAM_NAMES, weights.tolist()))}")
    else:
        X_for_umap = X_scaled
        print("Feature weights: using uniform weights")

    reducer = umap.UMAP(
        n_components=2, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist,
        random_state=42, n_jobs=-1,
    )
    embedding = reducer.fit_transform(X_for_umap)
    print(f"UMAP finished, embedding shape: {embedding.shape}")

    # 3. HDBSCAN 聚类
    clusterer_kwargs = {
        "min_cluster_size": hdbscan_min_cluster_size,
        "core_dist_n_jobs": -1,
    }
    if hdbscan_min_samples is not None:
        clusterer_kwargs["min_samples"] = hdbscan_min_samples
    clusterer = hdbscan.HDBSCAN(**clusterer_kwargs)
    labels = clusterer.fit_predict(embedding)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"HDBSCAN finished: {n_clusters} clusters, {n_noise} noise points")

    # 4. Plot UMAP, colored by HDBSCAN clusters
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

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
            embedding[mask, 0], embedding[mask, 1],
            color=color, s=5, alpha=alpha, label=leg_label,
        )

    ax.set_xlabel("UMAP 1", fontsize=12, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=12, fontweight="bold")
    ax.set_title(
        f"CH0 14-Parameter UMAP + HDBSCAN | n={n_events} | {n_clusters} clusters",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # 计算保存路径（波形图需要用到）
    if save_path is None:
        ge_self_dir = os.path.dirname(cut_dir)
        data_dir = os.path.dirname(ge_self_dir)
        python_dir = os.path.dirname(data_dir)
        project_root = os.path.dirname(python_dir)
        output_dir = os.path.join(project_root, "images", "presentation")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(output_dir, f"ch0_13param_umap_hdbscan_{n_events}events_{timestamp}.png")

    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"15-parameter UMAP+HDBSCAN figure saved to: {save_path}")

    # 5. Parameter distributions per cluster (excluding noise)
    unique_labels_ordered = sorted([l for l in set(labels) if l != -1])
    if len(unique_labels_ordered) == 0:
        print("No valid clusters, skip parameter distribution plots")
    else:
        fig_params, axes = plt.subplots(4, 4, figsize=(16, 14))
        axes = axes.ravel()
        cmap_params = plt.cm.get_cmap("tab20")

        for i, name in enumerate(PARAM_NAMES):
            ax = axes[i]
            param_key = PARAM_KEYS[i]
            values_all = params_dict[name]
            bins_use = 80 if param_key in TANH_PARAM_NAMES else 40

            for c_idx, lab in enumerate(unique_labels_ordered):
                mask = labels == lab
                values = values_all[mask]
                color = cmap_params((c_idx % 20 + 0.5) / 20)
                ax.hist(
                    values,
                    bins=bins_use,
                    alpha=0.6,
                    color=color,
                    density=True,
                    label=f"Cluster {lab}",
                )
            ax.set_title(name, fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3)
            if name in PARAM_NAMES_LOG_Y:
                ax.set_yscale("log")
            if i == 0:
                ax.legend(fontsize=8)

        if len(PARAM_NAMES) < len(axes):
            for j in range(len(PARAM_NAMES), len(axes)):
                fig_params.delaxes(axes[j])

        fig_params.suptitle(
            f"CH0 15 Parameters Distribution | HDBSCAN {n_clusters} clusters",
            fontsize=13, fontweight="bold",
        )
        fig_params.tight_layout(rect=[0, 0, 1, 0.95])

        save_path_params = save_path.replace(
            "ch0_13param_umap_hdbscan_", "ch0_15param_distributions_hdbscan_"
        )
        fig_params.savefig(save_path_params, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"15-parameter distribution figure saved to: {save_path_params}")

        # 6. 从 n 个簇 + 噪声类各抽 9 个 event，分别绘制 n+1 幅图，每幅图 3×3 展示 9 条 CH3 波形
        WAVEFORMS_PER_FIG = 9
        n_rows, n_cols = 3, 3
        sampling_interval_ns = 4.0

        with h5py.File(ch0_3_file_sel, "r") as f:
            ch_data = f["channel_data"]
            # global_indices 对应在原始 hdf5 文件中的真实 event 号
            global_indices = selected_indices[event_ranks]
            all_ch3_waveforms = ch_data[:, ch3_idx, global_indices].astype(np.float64)

        time_us = np.arange(all_ch3_waveforms.shape[0]) * sampling_interval_ns / 1000.0
        classes_to_plot = list(unique_labels_ordered) + [-1]

        for lab in classes_to_plot:
            idx_arr_all = np.where(labels == lab)[0]
            if len(idx_arr_all) == 0:
                continue

            lab_name = "Noise" if lab == -1 else f"Cluster {lab}"
            total_in_cluster = len(idx_arr_all)

            fig_ch3, axes_ch3 = plt.subplots(n_rows, n_cols, figsize=(12, 10))
            axes_flat = np.array(axes_ch3).flatten()

            state = {"page": 0}

            def _plot_page():
                # 清空当前 axes
                for ax_ in axes_flat:
                    ax_.cla()

                start = state["page"] * WAVEFORMS_PER_FIG
                end = min((state["page"] + 1) * WAVEFORMS_PER_FIG, total_in_cluster)
                samp_idx_local = idx_arr_all[start:end]
                n_sample_local = samp_idx_local.size

                cmap_ch3_local = plt.cm.get_cmap("tab20")
                color_local = (
                    (0.7, 0.7, 0.7, 0.8)
                    if lab == -1
                    else cmap_ch3_local((list(unique_labels_ordered).index(lab) % 20 + 0.5) / 20)
                )

                for i, loc_idx in enumerate(samp_idx_local):
                    ax_ch3 = axes_flat[i]
                    wf = all_ch3_waveforms[:, loc_idx]
                    ax_ch3.plot(time_us, wf, color=color_local, linewidth=0.8)
                    true_event_idx = int(global_indices[loc_idx])
                    base = os.path.basename(ch0_3_file_sel)
                    name_no_ext, _ = os.path.splitext(base)
                    file_id = _extract_file_id_from_name(name_no_ext)
                    ax_ch3.set_title(f"{file_id}#{true_event_idx}", fontsize=9)
                    ax_ch3.set_xlabel("Time (μs)", fontsize=9)
                    ax_ch3.set_ylabel("ADC", fontsize=9)
                    ax_ch3.grid(True, alpha=0.3)

                for j in range(n_sample_local, WAVEFORMS_PER_FIG):
                    axes_flat[j].set_visible(False)

                fig_ch3.suptitle(
                    f"CH3 raw waveforms — {lab_name} "
                    f"(cluster total {total_in_cluster} events, this page {n_sample_local}, "
                    f"page {state['page'] + 1})",
                    fontsize=12,
                    fontweight="bold",
                )
                fig_ch3.tight_layout(rect=[0, 0, 1, 0.95])
                fig_ch3.canvas.draw_idle()

            def _on_key(event):
                if event.key == " ":
                    # 切到下一页，如果超出则关闭窗口
                    if (state["page"] + 1) * WAVEFORMS_PER_FIG >= total_in_cluster:
                        plt.close(fig_ch3)
                    else:
                        state["page"] += 1
                        _plot_page()

            _plot_page()
            cid = fig_ch3.canvas.mpl_connect("key_press_event", _on_key)

            if show_plot:
                plt.show()
            else:
                plt.close(fig_ch3)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        if len(unique_labels_ordered) > 0:
            plt.close(fig_params)

    return save_path

def _get_h5_files_in_dir_or_default(folder: Optional[str]) -> List[str]:
    """
    获取待处理的 HDF5 文件列表：
    - 若 folder 为 None，则使用 utils.visualize.get_h5_files() 中的 CH0-3 列表；
    - 若 folder 非 None，则从该目录中枚举所有以 .h5 结尾的文件。
    """
    if folder is None:
        h5_files = get_h5_files()
        if "CH0-3" not in h5_files or not h5_files["CH0-3"]:
            raise FileNotFoundError("在 data/hdf5/raw_pulse/CH0-3 目录中未找到 h5 文件")
        return h5_files["CH0-3"]

    if not os.path.isdir(folder):
        raise NotADirectoryError(f"指定的路径不是目录: {folder}")

    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".h5")
    ]
    if not files:
        raise FileNotFoundError(f"在目录中未找到 h5 文件: {folder}")
    files.sort()
    return files

def _build_ch5_basename_to_path(
    folder: Optional[str],
    ch0_3_files: List[str],
    ch5_folder: Optional[str] = None,) -> Dict[str, str]:
    """
    构建 {basename -> CH5 文件路径} 的映射，使得每个 CH0-3 文件能匹配到同名 CH5 文件。

    参数：
        folder      : CH0-3 文件所在目录（None 时使用默认 raw_pulse 结构）
        ch0_3_files : CH0-3 文件路径列表
        ch5_folder  : 显式指定的 CH5 目录；为 None 时自动推断（与 CH0-3 同级的 CH5 目录）

    返回：
        basename -> 完整 CH5 路径 的字典
    """
    if ch5_folder is not None and os.path.isdir(ch5_folder):
        ch5_files = [
            os.path.join(ch5_folder, f)
            for f in os.listdir(ch5_folder)
            if f.lower().endswith(".h5")
        ]
        return {os.path.basename(p): p for p in ch5_files}

    if folder is None:
        h5_files = get_h5_files()
        if "CH5" not in h5_files or not h5_files["CH5"]:
            return {}
        return {os.path.basename(p): p for p in h5_files["CH5"]}

    ch5_dir = os.path.join(os.path.dirname(folder), "CH5")
    if not os.path.isdir(ch5_dir):
        h5_files = get_h5_files()
        if "CH5" in h5_files and h5_files["CH5"]:
            return {os.path.basename(p): p for p in h5_files["CH5"]}
        return {}

    ch5_files = [
        os.path.join(ch5_dir, f)
        for f in os.listdir(ch5_dir)
        if f.lower().endswith(".h5")
    ]
    return {os.path.basename(p): p for p in ch5_files}

def run_batch_umap_hdbscan_for_ch0_3_files(
    folder: Optional[str] = None,
    ch0_3_files: Optional[List[str]] = None,
    max_files: int = 20,
    **kwargs,) -> None:
    """
    对指定文件夹（或指定文件列表）中的 CH0-3 HDF5 文件进行批量 15 参数 UMAP+HDBSCAN 处理。
    所有文件的事件参数合并成一个大的 n_total×15 矩阵，再统一进行 UMAP 和聚类分析。

    参数：
        folder      : 指定 HDF5 所在目录；为 None 时使用 get_h5_files() 中的默认 CH0-3 目录。
        ch0_3_files : 直接给定的 CH0-3 文件路径列表（优先于 folder）；为 None 时根据 folder 自动获取。
        max_files   : 最多处理的文件数（如果文件数超过该值，只取前 max_files 个）。
        **kwargs    : 传递给内部处理的其他参数。支持 ch5_folder（CH5 目录路径）、
                      阈值、UMAP 参数、HDBSCAN 参数等。每个 CH0-3 必须与同名 CH5 配对，否则跳过。
    """
    if ch0_3_files is None:
        ch0_3_files = _get_h5_files_in_dir_or_default(folder)

    if not ch0_3_files:
        raise FileNotFoundError("未提供任何 CH0-3 文件路径，且默认目录中也未找到文件。")

    total_files = len(ch0_3_files)
    if total_files > max_files:
        print(f"共找到 {total_files} 个 CH0-3 HDF5 文件，仅处理前 {max_files} 个。")
        ch0_3_files = ch0_3_files[:max_files]

    print(f"\n即将批量处理 {len(ch0_3_files)} 个 HDF5 文件，并汇总为一个整体参数矩阵。")

    # 构建 CH5 文件名匹配映射：每个 CH0-3 文件必须匹配同名的 CH5 文件，否则跳过该对
    ch5_folder = kwargs.get("ch5_folder", None)
    ch5_basename_to_path = _build_ch5_basename_to_path(folder, ch0_3_files, ch5_folder)
    if not ch5_basename_to_path:
        raise FileNotFoundError(
            "未找到任何 CH5 文件，无法进行 basename 匹配。请确认 CH5 目录存在（如 raw_pulse/CH5 或通过 ch5_folder 指定）。"
        )
    print(f"CH5 文件名匹配: 已加载 {len(ch5_basename_to_path)} 个 CH5 文件（按 basename 与 CH0-3 一一对应）")

    # 汇总所有文件的 15 参数，并记录每个事件来自哪个文件及其真实 event 编号
    combined_params: Dict[str, List[np.ndarray]] = {name: [] for name in PARAM_NAMES}
    event_sources: List[Tuple[str, int]] = []  # (file_path, true_event_index)

    for idx, fpath in enumerate(ch0_3_files, start=1):
        print("\n" + "=" * 80)
        print(f"[{idx}/{len(ch0_3_files)}] 正在处理文件: {fpath}")
        print("=" * 80)

        # 每个 CH0-3 文件必须匹配同名的 CH5 文件，否则跳过该对
        ch5_for_this = ch5_basename_to_path.get(os.path.basename(fpath)) if ch5_basename_to_path else None
        if ch5_for_this is None:
            print(f"  警告: 未找到与 {os.path.basename(fpath)} 同名的 CH5 文件，跳过该对")
            continue
        print(f"  匹配 CH5: {os.path.basename(ch5_for_this)}")

        # 1. 对当前文件做 ±1σ 事件选择
        event_ranks, ch0_3_file_sel, _, selected_indices = _select_events_in_1sigma_band(
            ch0_3_file=fpath,
            ch5_file=ch5_for_this,
            rt_cut=kwargs.get("rt_cut", 6000.0),
            ch0_threshold=kwargs.get("ch0_threshold", 16382.0),
            ch0_idx=kwargs.get("ch0_idx", 0),
            ch1_idx=kwargs.get("ch1_idx", 1),
            x_min=kwargs.get("x_min", 2000.0),
            x_max=kwargs.get("x_max", 14000.0),
            sigma_factor=kwargs.get("sigma_factor", 1.0),
        )
        n_events = event_ranks.size
        print(f"文件 {ch0_3_file_sel} 中选出 ±1σ 事件数: {n_events}")
        if n_events == 0:
            continue

        # 记录该文件中每个选中事件在原始 HDF5 中的真实 event 号
        global_indices = selected_indices[event_ranks]
        for g in global_indices:
            event_sources.append((ch0_3_file_sel, int(g)))

        # 2. 计算当前文件的 15 参数（内部已经使用所有 CPU 核心并行）
        params_dict_file = compute_params_for_events(
            ch0_3_file=ch0_3_file_sel,
            event_ranks=event_ranks,
            selected_indices=selected_indices,
            ch0_idx=kwargs.get("ch0_idx", 0),
            ch3_idx=kwargs.get("ch3_idx", 3),
            baseline_window_us=kwargs.get("baseline_window_us", 2.0),
            max_workers=kwargs.get("max_workers", None),
        )

        # 3. 累加到汇总参数中
        for name in PARAM_NAMES:
            combined_params[name].append(params_dict_file[name])

    # 将所有文件的参数拼接为一个大矩阵
    for name in PARAM_NAMES:
        if combined_params[name]:
            combined_params[name] = np.concatenate(combined_params[name], axis=0)
        else:
            combined_params[name] = np.empty((0,), dtype=np.float64)

    n_total_events = combined_params[PARAM_NAMES[0]].shape[0]
    if n_total_events == 0:
        raise RuntimeError("所有文件均未选出 ±1σ 事件，无法进行 UMAP+HDBSCAN。")

    print(f"\nAfter merging all files: total events = {n_total_events}, feature dim = {len(PARAM_NAMES)}")

    X = np.column_stack([combined_params[name] for name in PARAM_NAMES])
    print(f"Merged parameter matrix shape: {X.shape} (n_events_total × {len(PARAM_NAMES)})")

    # 使用所有 CPU 核心进行 UMAP（n_jobs=-1）和 HDBSCAN（core_dist_n_jobs=-1）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("StandardScaler finished (merged matrix)")

    feature_weights = kwargs.get("feature_weights", None)
    if feature_weights is not None:
        weights = np.asarray(feature_weights, dtype=np.float64)
        if weights.shape[0] != len(PARAM_NAMES):
            raise ValueError(f"feature_weights length should be {len(PARAM_NAMES)}, got {weights.shape[0]}")
        X_for_umap = X_scaled * weights.reshape(1, -1)
        print(f"Feature weights: {dict(zip(PARAM_NAMES, weights.tolist()))}")
    else:
        X_for_umap = X_scaled
        print("Feature weights: using uniform weights")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=kwargs.get("umap_n_neighbors", 15),
        min_dist=kwargs.get("umap_min_dist", 0.1),
        random_state=42,
        n_jobs=-1,
    )
    embedding = reducer.fit_transform(X_for_umap)
    print(f"UMAP finished, embedding shape: {embedding.shape}")

    hdbscan_min_cluster_size = kwargs.get("hdbscan_min_cluster_size", 50)
    hdbscan_min_samples = kwargs.get("hdbscan_min_samples", None)
    clusterer_kwargs = {
        "min_cluster_size": hdbscan_min_cluster_size,
        "core_dist_n_jobs": -1,
    }
    if hdbscan_min_samples is not None:
        clusterer_kwargs["min_samples"] = hdbscan_min_samples
    clusterer = hdbscan.HDBSCAN(**clusterer_kwargs)
    labels = clusterer.fit_predict(embedding)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"HDBSCAN finished: {n_clusters} clusters, {n_noise} noise points")

    # 绘制合并后的 UMAP 图
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

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
            embedding[mask, 0], embedding[mask, 1],
            color=color, s=5, alpha=alpha, label=leg_label,
        )

    ax.set_xlabel("UMAP 1", fontsize=12, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=12, fontweight="bold")
    ax.set_title(
        f"CH0 14-Parameter UMAP + HDBSCAN (All Files) | n={n_total_events} | {n_clusters} clusters",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # 15 参数分布图（合并后，按簇分布）
    unique_labels_ordered = sorted([l for l in set(labels) if l != -1])
    if unique_labels_ordered:
        fig_params, axes = plt.subplots(4, 4, figsize=(16, 14))
        axes = axes.ravel()
        cmap_params = plt.cm.get_cmap("tab20")

        for i, name in enumerate(PARAM_NAMES):
            ax_p = axes[i]
            param_key = PARAM_KEYS[i]
            values_all = combined_params[name]
            bins_use = 80 if param_key in TANH_PARAM_NAMES else 40

            for c_idx, lab in enumerate(unique_labels_ordered):
                mask = labels == lab
                values = values_all[mask]
                color = cmap_params((c_idx % 20 + 0.5) / 20)
                ax_p.hist(
                    values,
                    bins=bins_use,
                    alpha=0.6,
                    color=color,
                    density=True,
                    label=f"Cluster {lab}",
                )
            ax_p.set_title(name, fontsize=11, fontweight="bold")
            ax_p.grid(True, alpha=0.3)
            if name in PARAM_NAMES_LOG_Y:
                ax_p.set_yscale("log")
            if i == 0:
                ax_p.legend(fontsize=8)

        if len(PARAM_NAMES) < len(axes):
            for j in range(len(PARAM_NAMES), len(axes)):
                fig_params.delaxes(axes[j])

        fig_params.suptitle(
            f"CH0 14 Parameters Distribution (All Files) | HDBSCAN {n_clusters} clusters",
            fontsize=13, fontweight="bold",
        )
        fig_params.tight_layout(rect=[0, 0, 1, 0.95])

    # 从 n 个簇 + 噪声各抽 9 个 event，一页展示 9 条 CH3 波形；
    # 在波形窗口中按空格键可以切换到该簇的下一批 9 条，直到该簇事件用完或关闭窗口。
    if n_total_events > 0 and len(event_sources) == n_total_events:
        WAVEFORMS_PER_FIG = 9
        n_rows, n_cols = 3, 3
        sampling_interval_ns = 4.0
        ch3_idx = kwargs.get("ch3_idx", 3)
        classes_to_plot = list(unique_labels_ordered) + [-1]

        for lab in classes_to_plot:
            idx_arr_all = np.where(labels == lab)[0]
            if len(idx_arr_all) == 0:
                continue

            lab_name = "Noise" if lab == -1 else f"Cluster {lab}"
            total_in_cluster = len(idx_arr_all)

            fig_ch3, axes_ch3 = plt.subplots(n_rows, n_cols, figsize=(12, 10))
            axes_flat = np.array(axes_ch3).flatten()

            state = {"page": 0}

            def _plot_page():
                # 清空当前 axes
                for ax_ in axes_flat:
                    ax_.cla()

                start = state["page"] * WAVEFORMS_PER_FIG
                end = min((state["page"] + 1) * WAVEFORMS_PER_FIG, total_in_cluster)
                samp_idx = idx_arr_all[start:end]
                n_sample_local = samp_idx.size

                cmap_ch3_local = plt.cm.get_cmap("tab20")
                color_local = (
                    (0.7, 0.7, 0.7, 0.8)
                    if lab == -1
                    else cmap_ch3_local((list(unique_labels_ordered).index(lab) % 20 + 0.5) / 20)
                )

                for i, glob_idx in enumerate(samp_idx):
                    ax_ch = axes_flat[i]
                    file_path, true_event_idx = event_sources[int(glob_idx)]
                    with h5py.File(file_path, "r") as f_ch:
                        ch_data = f_ch["channel_data"]
                        wf = ch_data[:, ch3_idx, true_event_idx].astype(np.float64)
                    time_us = np.arange(wf.size) * sampling_interval_ns / 1000.0
                    ax_ch.plot(time_us, wf, color=color_local, linewidth=0.8)
                    base = os.path.basename(file_path)
                    name_no_ext, _ = os.path.splitext(base)
                    file_id = _extract_file_id_from_name(name_no_ext)
                    ax_ch.set_title(f"{file_id}#{true_event_idx}", fontsize=9)
                    ax_ch.set_xlabel("Time (μs)", fontsize=9)
                    ax_ch.set_ylabel("ADC", fontsize=9)
                    ax_ch.grid(True, alpha=0.3)

                for j in range(n_sample_local, WAVEFORMS_PER_FIG):
                    axes_flat[j].set_visible(False)

                fig_ch3.suptitle(
                    f"CH3 raw waveforms — {lab_name} "
                    f"(cluster total {total_in_cluster} events, this page {n_sample_local}, "
                    f"page {state['page'] + 1})",
                    fontsize=12,
                    fontweight="bold",
                )
                fig_ch3.tight_layout(rect=[0, 0, 1, 0.95])
                fig_ch3.canvas.draw_idle()

            def _on_key(event):
                if event.key == " ":
                    # 切到下一页，如果超出则关闭窗口
                    if (state["page"] + 1) * WAVEFORMS_PER_FIG >= total_in_cluster:
                        plt.close(fig_ch3)
                    else:
                        state["page"] += 1
                        _plot_page()

            _plot_page()
            cid = fig_ch3.canvas.mpl_connect("key_press_event", _on_key)

            if kwargs.get("show_plot", True):
                plt.show()
            else:
                plt.close(fig_ch3)

    # 显示或关闭图像窗口
    if kwargs.get("show_plot", True):
        plt.show()
    else:
        plt.close(fig)
        fig_params.tight_layout(rect=[0, 0, 1, 0.95])

if __name__ == "__main__":
    try:
        # 对指定目录或默认 CH0-3 目录下的 HDF5 文件进行批量处理（最多 20 个），
        # 并将所有文件的 14 参数合并后再做一次 UMAP + HDBSCAN。
        run_batch_umap_hdbscan_for_ch0_3_files(
            folder=None,               # 若想指定目录，可改为具体路径字符串
            ch0_3_files=None,          # 若直接给定文件列表，可在此处传入列表，优先于 folder
            max_files=20,
            ch5_file=None,
            rt_cut=6000.0,
            ch0_threshold=16382.0,
            ch0_idx=0,
            ch1_idx=1,
            ch3_idx=3,
            x_min=2000.0,
            x_max=14000.0,
            sigma_factor=1.0,
            baseline_window_us=2.0,
            max_workers=None,
            umap_n_neighbors=10,
            umap_min_dist=0.1,
            # 14 维特征权重（依次对应 PARAM_NAMES）：
            # Amax, μ(ped), μ(pedt), σ(ped), σ(pedt), Tmax, RMS_ped, RMS_pedt,
            # tanh_p0, tanh_p1, tanh_p2, tanh_p3, tanh_RMS, fast_highfreq_ratio
            feature_weights=[0.5, 0.5, 0.5, 0.5, 0.5, 2.0, 1.0, 1.0,
                             1.0, 1.0, 2.0, 1.0, 1.0, 1],
            hdbscan_min_cluster_size=7,
            hdbscan_min_samples=None,
            save_path=None,
            show_plot=True,
            use_cached_params=True,
            params_cache_path=None,
        )
    except Exception as e:
        print(f"\n15 参数 UMAP+HDBSCAN 批量可视化失败: {e}")
        import traceback
        traceback.print_exc()
