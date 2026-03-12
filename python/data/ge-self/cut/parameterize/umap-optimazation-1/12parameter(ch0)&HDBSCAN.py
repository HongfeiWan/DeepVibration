#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
±1σ 带内事件的 CH0 波形 12 参数 UMAP + HDBSCAN 聚类可视化脚本。

不显示 PNCUT 图，不按 umap1_threshold 划分，仅用 HDBSCAN 对 UMAP 空间进行聚类。

流程：
1. 选出 ±1σ 带内的事件；
2. 使用所有 CPU 并行处理所有波形，得到 n×12 参数矩阵；
3. 对 12 个参数使用 StandardScaler 标准化；
4. 使用 UMAP 降维到二维；
5. 使用 HDBSCAN 对 UMAP 空间聚类，分为若干簇；
6. 绘制 UMAP 图，按 HDBSCAN 簇着色；
7. 绘制各簇的 12 参数分布图（叠加直方图，风格同 pncut(distribution)）。

12 参数：
- 8 个 CH0 波形参数（来自 parameter(ch0).py）：
  Amax, μ(ped), μ(pedt), σ(ped), σ(pedt), Tmax, RMS_ped, RMS_pedt
- 4 个快放 tanh 拟合参数（参考 pncutvisualization(ch3) .py）：
  tanh_p0, tanh_p1, tanh_p2, tanh_p3
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
from scipy.optimize import curve_fit

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
DEFAULT_PARAMS_CACHE_PATH = os.path.join(current_dir, "ch0_12params_cache.h5")

# 12 参数名称：8 个基础参数 + 4 个快放 tanh 拟合参数
PARAM_NAMES = [
    "Amax", "μ(ped)", "μ(pedt)", "σ(ped)", "σ(pedt)", "Tmax", "RMS_ped", "RMS_pedt",
    "tanh_p0", "tanh_p1", "tanh_p2", "tanh_p3",
]
PARAM_KEYS = [
    "amax", "mu_ped", "mu_pedt", "sigma_ped", "sigma_pedt", "tmax_us", "rms_ped", "rms_pedt",
    "tanh_p0", "tanh_p1", "tanh_p2", "tanh_p3",
]
TANH_PARAM_NAMES = {"tanh_p0", "tanh_p1", "tanh_p2", "tanh_p3"}


def _smooth_waveform_for_fast_fit(
    waveform: np.ndarray,
    smooth_window: int = 5,
    smooth_times: int = 20,
) -> np.ndarray:
    """按快放拟合脚本中的方式对波形做重复滑动平均平滑。"""
    waveform_smooth = np.asarray(waveform, dtype=np.float64).copy()
    if smooth_window is not None and smooth_window > 1:
        if smooth_window % 2 == 0:
            raise ValueError(f"smooth_window 必须为奇数，当前为 {smooth_window}")
        if smooth_times < 1:
            smooth_times = 1
        if waveform_smooth.size >= smooth_window:
            half = smooth_window // 2
            for _ in range(smooth_times):
                tmp = waveform_smooth.copy()
                for i in range(half, waveform_smooth.size - half):
                    waveform_smooth[i] = float(np.mean(tmp[i - half : i + half + 1]))
    return waveform_smooth


def _tanh_rise(x: np.ndarray, p0: float, p1: float, p2: float, p3: float) -> np.ndarray:
    """快放前沿 tanh 拟合模型。"""
    return 0.5 * p0 * np.tanh(p1 * (x - p2)) + p3


def _compute_fast_fit_params(
    waveform: np.ndarray,
    sampling_interval_ns: float = 4.0,
    baseline_window_us: float = 2.0,
) -> Dict[str, float]:
    """参考 pncutvisualization(ch3) .py，提取快放 tanh 拟合的 4 个参数。拟合失败时统一返回 0。"""
    waveform_smooth = _smooth_waveform_for_fast_fit(waveform)
    time_axis_us = np.arange(waveform_smooth.size) * sampling_interval_ns / 1000.0

    baseline_window_ns = baseline_window_us * 1000.0
    samples_baseline = int(round(baseline_window_ns / sampling_interval_ns))
    samples_baseline = max(1, min(samples_baseline, waveform_smooth.size // 2))

    baseline_front = float(np.mean(waveform_smooth[:samples_baseline]))
    amp = waveform_smooth - baseline_front
    max_amp = float(np.max(amp))

    if max_amp <= 0:
        return {
            "tanh_p0": 0.0,
            "tanh_p1": 0.0,
            "tanh_p2": 0.0,
            "tanh_p3": 0.0,
        }

    p0_init = max(max_amp, 0.0)
    p3_init = baseline_front
    p2_init = float(np.mean(time_axis_us)) if time_axis_us.size > 0 else 0.0
    p1_init = 1.0

    idx_max = int(np.argmax(amp))
    t_max = float(time_axis_us[idx_max])

    mid_level = 0.5 * max_amp
    idx_mid = int(np.argmax(amp >= mid_level))
    if amp[idx_mid] >= mid_level:
        p2_init = float(time_axis_us[idx_mid])
    else:
        p2_init = float(np.mean(time_axis_us[: idx_max + 1])) if idx_max >= 0 else p2_init

    level_5 = 0.05 * max_amp
    level_95 = 0.95 * max_amp
    idx_5 = int(np.argmax(amp >= level_5))
    idx_95 = int(np.argmax(amp >= level_95))
    if amp[idx_5] >= level_5 and amp[idx_95] >= level_95 and idx_95 > idx_5:
        rise_time = float(time_axis_us[idx_95] - time_axis_us[idx_5])
        rise_time = max(rise_time, 1e-6)
        p1_init = float(np.log(19.0) / rise_time)
    else:
        p1_init = 1.0 / max(t_max, 1e-6)

    t_back_start = float(time_axis_us[-samples_baseline])
    mask_time = time_axis_us <= t_back_start
    mask_before_max = time_axis_us <= t_max
    mask = mask_time & mask_before_max

    if np.count_nonzero(mask) < 5:
        return {
            "tanh_p0": 0.0,
            "tanh_p1": 0.0,
            "tanh_p2": 0.0,
            "tanh_p3": 0.0,
        }

    x_data = time_axis_us[mask]
    y_data = waveform_smooth[mask]
    try:
        popt, _ = curve_fit(
            _tanh_rise,
            x_data,
            y_data,
            p0=[p0_init, p1_init, p2_init, p3_init],
            maxfev=10000,
        )
        p0_fit, p1_fit, p2_fit, p3_fit = [float(v) for v in popt]
        return {
            "tanh_p0": p0_fit,
            "tanh_p1": p1_fit,
            "tanh_p2": p2_fit,
            "tanh_p3": p3_fit,
        }
    except Exception:
        return {
            "tanh_p0": 0.0,
            "tanh_p1": 0.0,
            "tanh_p2": 0.0,
            "tanh_p3": 0.0,
        }


def _process_single_event_params(args: Tuple) -> Tuple[int, Dict[str, float]]:
    """工作函数：对单条波形计算 12 参数。args = (rank, waveform, sampling_interval_ns, baseline_window_us)"""
    rank, waveform, sampling_interval_ns, baseline_window_us = args
    waveform_raw = np.asarray(waveform, dtype=np.float64)
    wf = median_filter(waveform_raw, kernel_size=3)
    params = compute_eight_params(wf, sampling_interval_ns=sampling_interval_ns, baseline_window_us=baseline_window_us)
    params.update(
        _compute_fast_fit_params(
            waveform_raw,
            sampling_interval_ns=sampling_interval_ns,
            baseline_window_us=baseline_window_us,
        )
    )
    return rank, params


def compute_params_for_events(
    ch0_3_file: str,
    event_ranks: np.ndarray,
    selected_indices: np.ndarray,
    ch0_idx: int = 0,
    baseline_window_us: float = 2.0,
    max_workers: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """对选中的事件计算 12 参数（多进程并行）。"""
    print("\n正在计算所有事件的 12 参数...")
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
        future_to_rank = {executor.submit(_process_single_event_params, args): args[0] for args in task_args}
        with tqdm(total=n_events, desc="计算12参数", unit="事件", ncols=100) as pbar:
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


def run_umap_hdbscan_visualization(
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
    feature_weights: Optional[List[float]] = None,
    hdbscan_min_cluster_size: int = 50,
    hdbscan_min_samples: Optional[int] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    use_cached_params: bool = False,
    params_cache_path: Optional[str] = None,
) -> str:
    """
    对 ±1σ 带内事件：计算 12 参数，UMAP 降维，HDBSCAN 聚类，绘制 UMAP 图按簇着色。
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
        print(f"\n将对 {n_events} 个事件进行 12 参数并行提取...")
        params_dict = compute_params_for_events(
            ch0_3_file=ch0_3_file_sel,
            event_ranks=event_ranks,
            selected_indices=selected_indices,
            ch0_idx=ch0_idx,
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
            raise ValueError(f"feature_weights 长度应为 {len(PARAM_NAMES)}，当前为 {weights.shape[0]}")
        X_for_umap = X_scaled * weights.reshape(1, -1)
        print(f"特征权重: {dict(zip(PARAM_NAMES, weights.tolist()))}")
    else:
        X_for_umap = X_scaled
        print("特征权重: 使用默认等权重")

    reducer = umap.UMAP(n_components=2, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, random_state=42)
    embedding = reducer.fit_transform(X_for_umap)
    print(f"UMAP 降维完成，嵌入形状: {embedding.shape}")

    # 3. HDBSCAN 聚类
    clusterer_kwargs = {"min_cluster_size": hdbscan_min_cluster_size}
    if hdbscan_min_samples is not None:
        clusterer_kwargs["min_samples"] = hdbscan_min_samples
    clusterer = hdbscan.HDBSCAN(**clusterer_kwargs)
    labels = clusterer.fit_predict(embedding)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"HDBSCAN 聚类完成: {n_clusters} 个簇, 噪声点 {n_noise} 个")

    # 4. 绘制 UMAP 图，按 HDBSCAN 簇着色
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
            leg_label = "噪声"
        else:
            color = cmap((color_idx % 20 + 0.5) / 20)
            color_idx += 1
            alpha = 0.7
            leg_label = f"簇 {lab}"
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=color, s=5, alpha=alpha, label=leg_label,
        )

    ax.set_xlabel("UMAP 1", fontsize=12, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=12, fontweight="bold")
    ax.set_title(
        f"CH0 12-Parameter UMAP + HDBSCAN | n={n_events} | {n_clusters} 簇",
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
        save_path = os.path.join(output_dir, f"ch0_12param_umap_hdbscan_{n_events}events_{timestamp}.png")

    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"12 参数 UMAP+HDBSCAN 可视化已保存至: {save_path}")

    # 5. 各簇的 12 参数分布图（排除噪声，仅 n 个有效簇）
    unique_labels_ordered = sorted([l for l in set(labels) if l != -1])
    if len(unique_labels_ordered) == 0:
        print("无有效簇，跳过参数分布图")
    else:
        fig_params, axes = plt.subplots(3, 4, figsize=(16, 11))
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
                    label=f"簇 {lab}",
                )
            ax.set_title(name, fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(fontsize=8)

        if len(PARAM_NAMES) < len(axes):
            for j in range(len(PARAM_NAMES), len(axes)):
                fig_params.delaxes(axes[j])

        fig_params.suptitle(
            f"CH0 12 Parameters Distribution | HDBSCAN {n_clusters} 簇",
            fontsize=13, fontweight="bold",
        )
        fig_params.tight_layout(rect=[0, 0, 1, 0.95])

        save_path_params = save_path.replace(
            "ch0_12param_umap_hdbscan_", "ch0_12param_distributions_hdbscan_"
        )
        fig_params.savefig(save_path_params, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"各簇 12 参数分布图已保存至: {save_path_params}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        if len(unique_labels_ordered) > 0:
            plt.close(fig_params)

    return save_path


if __name__ == "__main__":
    try:
        run_umap_hdbscan_visualization(
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
            umap_n_neighbors=10,
            umap_min_dist=0.1,
            feature_weights=[2.0, 0.5, 0.5, 0.5, 0.5, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            hdbscan_min_cluster_size=50,
            hdbscan_min_samples=None,
            save_path=None,
            show_plot=True,
            use_cached_params=True,
            params_cache_path=None,
        )
    except Exception as e:
        print(f"\n12 参数 UMAP+HDBSCAN 可视化失败: {e}")
        import traceback
        traceback.print_exc()
