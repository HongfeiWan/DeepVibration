#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UMAP + PN-cut 关联可视化脚本。

流程：
1. 运行 15 参数 UMAP（CH0 10 + CH3 快放 5 参数），得到二维 UMAP 分布；
2. 选取 UMAP1 > 4 的事件；
3. 在 PN-cut 图像（max_ch0 vs max_ch1）中，用不同颜色标注这些事件的位置。
"""

import os
import sys
import warnings
from typing import Optional, Tuple, List, Dict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

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
from scipy.optimize import curve_fit
import umap

_select_events_in_1sigma_band = paramdistribution._select_events_in_1sigma_band
compute_all_parameters_for_events = paramdistribution.compute_all_parameters_for_events

# CH0 10 参数 + CH3 快放 5 参数（15 参数 UMAP）
PARAM_NAMES_CH0 = [
    "Amax", "Amin", "Tmax", "Tmin", "Q",
    "Qpre", "Qprev", "ped", "pedt", "abs(pedt-ped)"
]
FAST_PARAM_NAMES = ["rise_t", "famp", "fslope", "fcross", "fped"]
PARAM_NAMES = PARAM_NAMES_CH0 + FAST_PARAM_NAMES

CH3_IDX = 3
SAMPLING_INTERVAL_NS = 4.0
LN19 = np.log(19.0)

DEFAULT_PARAMS_CACHE_PATH = os.path.join(current_dir, "ch0_ch3_params_cache.h5")
UMAP1_THRESHOLD = 4.0


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


def _save_params_cache(cache_path: str, params_dict: Dict[str, np.ndarray], param_names: List[str]) -> None:
    """将参数矩阵及各列保存到 HDF5 文件。"""
    with h5py.File(cache_path, "w") as f:
        X = np.column_stack([params_dict[name] for name in param_names])
        f.create_dataset("X", data=X)
        for name in param_names:
            f.create_dataset(name, data=params_dict[name])
        f.attrs["param_names"] = "|".join(param_names)


def _tanh_rise(x: np.ndarray, p0: float, p1: float, p2: float, p3: float) -> np.ndarray:
    """f(x) = 0.5 * p0 * tanh(p1 * (x - p2)) + p3"""
    return 0.5 * p0 * np.tanh(p1 * (x - p2)) + p3


def _smooth_waveform(waveform: np.ndarray, window: int = 5, times: int = 20) -> np.ndarray:
    """滑动平均平滑，window 需为正奇数。"""
    wf = waveform.copy().astype(np.float64)
    if window < 2 or window % 2 == 0 or wf.size < window:
        return wf
    half = window // 2
    for _ in range(times):
        tmp = wf.copy()
        for i in range(half, wf.size - half):
            wf[i] = float(np.mean(tmp[i - half : i + half + 1]))
    return wf


def _compute_fast_params_one(
    waveform: np.ndarray,
    sampling_interval_ns: float = SAMPLING_INTERVAL_NS,
    baseline_window_us: float = 2.0,
    smooth_window: int = 5,
    smooth_times: int = 20,
) -> Optional[Tuple[float, float, float, float, float]]:
    """对单条波形做 tanh 拟合，返回 (rise_t, famp, fslope, fcross, fped) 或 None。rise_t = ln(19) / p1"""
    wf_smooth = _smooth_waveform(waveform, smooth_window, smooth_times)
    time_axis_us = np.arange(wf_smooth.size) * sampling_interval_ns / 1000.0
    baseline_window_ns = baseline_window_us * 1000.0
    samples_baseline = max(1, min(int(round(baseline_window_ns / sampling_interval_ns)), wf_smooth.size // 2))
    baseline_front = float(np.mean(wf_smooth[:samples_baseline]))
    amp = wf_smooth - baseline_front
    max_amp = float(np.max(amp))
    if max_amp <= 0:
        return None
    idx_max = int(np.argmax(amp))
    t_max = float(time_axis_us[idx_max])
    low, high = 0.05 * max_amp, 0.95 * max_amp
    mask_amp = (amp >= low) & (amp <= high)
    t_front_end = time_axis_us[samples_baseline - 1]
    t_back_start = time_axis_us[-samples_baseline]
    mask_time = (time_axis_us >= t_front_end) & (time_axis_us <= t_back_start)
    mask_before_max = time_axis_us <= t_max
    mask = mask_amp & mask_time & mask_before_max
    if np.count_nonzero(mask) < 5:
        return None
    x_data = time_axis_us[mask]
    y_data = wf_smooth[mask]
    p0_init = 2.0 * max_amp
    p3_init = baseline_front
    mid_level = 0.5 * max_amp
    idx_mid = int(np.argmax(amp >= mid_level))
    p2_init = float(time_axis_us[idx_mid]) if amp[idx_mid] >= mid_level else float(np.mean(x_data))
    idx_5 = np.argmax(amp >= low)
    idx_95 = np.argmax(amp >= high)
    if amp[idx_5] >= low and amp[idx_95] >= high and idx_95 > idx_5:
        rise_time = max(float(time_axis_us[idx_95] - time_axis_us[idx_5]), 1e-6)
        p1_init = LN19 / rise_time
    else:
        p1_init = 1.0 / (x_data[-1] - x_data[0] + 1e-6)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*[Cc]ovariance of the parameters could not be estimated.*")
            popt, _ = curve_fit(_tanh_rise, x_data, y_data, p0=[p0_init, p1_init, p2_init, p3_init], maxfev=10000)
        p0, p1, p2, p3 = float(popt[0]), float(popt[1]), float(popt[2]), float(popt[3])
        if p1 <= 0:
            return None
        rise_t = LN19 / p1
        return (rise_t, p0, p1, p2, p3)
    except Exception:
        return None


def _fast_params_worker(args: Tuple) -> Optional[Tuple[float, float, float, float, float]]:
    """供 ProcessPoolExecutor 调用的顶层 worker，便于 pickle。"""
    waveform, baseline_window_us, smooth_window, smooth_times = args
    return _compute_fast_params_one(
        waveform,
        baseline_window_us=baseline_window_us,
        smooth_window=smooth_window,
        smooth_times=smooth_times,
    )


def _fast_params_batch_worker(args: Tuple) -> List[Optional[Tuple[float, float, float, float, float]]]:
    """
    批量 worker：从 HDF5 文件读取波形并计算，避免通过 IPC 传递大量波形数据。
    仅传递文件路径和索引，显著减少 pickle/IPC 开销，提升 CPU 利用率。
    """
    (ch0_3_file, indices_batch, ch3_idx, baseline_window_us, smooth_window, smooth_times) = args
    results = []
    with h5py.File(ch0_3_file, "r") as f:
        channel_data = f["channel_data"]
        for idx in indices_batch:
            waveform = channel_data[:, ch3_idx, int(idx)].astype(np.float64)
            r = _compute_fast_params_one(
                waveform,
                baseline_window_us=baseline_window_us,
                smooth_window=smooth_window,
                smooth_times=smooth_times,
            )
            results.append(r)
    return results


def _compute_fast_params_all(
    ch0_3_file: str,
    selected_indices: np.ndarray,
    event_ranks: np.ndarray,
    ch3_idx: int = CH3_IDX,
    baseline_window_us: float = 2.0,
    smooth_window: int = 5,
    smooth_times: int = 20,
    max_workers: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    对所有 ±1σ 事件并行计算 CH3 快放 5 参数。拟合失败用中位数填充。
    采用「worker 自读文件」策略：仅传递文件路径和索引，避免 IPC 传输大量波形，
    减少 pickle 开销，使各进程 CPU 真正跑满。
    """
    n_workers = max_workers if max_workers is not None else (os.cpu_count() or 1)
    global_indices = selected_indices[event_ranks]
    n_wf = global_indices.size

    # 按批划分索引，每批由 worker 从文件读取，避免传递波形数据
    n_batches = min(n_wf, n_workers * 4)  # 批数略多于 worker 数以利于负载均衡
    batch_size = max(1, (n_wf + n_batches - 1) // n_batches)
    batches = []
    for i in range(0, n_wf, batch_size):
        batch = global_indices[i : i + batch_size].tolist()
        batches.append(batch)

    tasks = [
        (ch0_3_file, batch, ch3_idx, baseline_window_us, smooth_window, smooth_times)
        for batch in batches
    ]
    print(f"CH3 快放参数: 使用 {n_workers} 个 CPU 核心，{len(batches)} 批，每批约 {batch_size} 个事件...")

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        batch_results = list(exe.map(_fast_params_batch_worker, tasks))
    for br in batch_results:
        results.extend(br)
    print("CH3 快放参数计算完成")
    arr = np.array([r if r is not None else (np.nan, np.nan, np.nan, np.nan, np.nan) for r in results])
    out = {}
    for j, name in enumerate(FAST_PARAM_NAMES):
        col = arr[:, j].astype(np.float64)
        valid = ~np.isnan(col)
        if valid.any():
            col[~valid] = np.nanmedian(col[valid])
        out[name] = col
    return out


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
    use_cached_params: bool = False,
    params_cache_path: Optional[str] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> str:
    """
    运行 15 参数 UMAP，选取 UMAP1 > umap1_threshold 的事件，在 PN-cut 图中用不同颜色标注。

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

    # 3. 15 参数矩阵（CH0 10 + CH3 快放 5）与 UMAP
    if use_cached_params and os.path.isfile(cache_path):
        print(f"\n从缓存加载参数: {cache_path}")
        params_dict, X = _load_params_cache(cache_path)
        param_names = list(params_dict.keys())
        if X.shape[0] != n_events:
            raise ValueError(
                f"缓存事件数 {X.shape[0]} 与当前 ±1σ 事件数 {n_events} 不一致，"
                "请删除缓存或设置 use_cached_params=False"
            )
        if not all(p in params_dict for p in FAST_PARAM_NAMES):
            print("缓存缺少快放参数，并行补算 CH3 快放 5 参数...")
            fast_dict = _compute_fast_params_all(
                ch0_3_file_sel, selected_indices, event_ranks,
                ch3_idx=CH3_IDX, baseline_window_us=baseline_window_us,
                max_workers=max_workers,
            )
            params_dict = {**params_dict, **fast_dict}
            param_names = list(params_dict.keys())
            X = np.column_stack([params_dict[name] for name in param_names])
            _save_params_cache(cache_path, params_dict, param_names)
    else:
        def _run_ch0():
            return compute_all_parameters_for_events(
                ch0_3_file=ch0_3_file_sel,
                event_ranks=event_ranks,
                selected_indices=selected_indices,
                ch0_idx=ch0_idx,
                baseline_window_us=baseline_window_us,
                max_workers=max_workers,
            )

        def _run_ch3():
            return _compute_fast_params_all(
                ch0_3_file_sel, selected_indices, event_ranks,
                ch3_idx=CH3_IDX, baseline_window_us=baseline_window_us,
                max_workers=max_workers,
            )

        with ThreadPoolExecutor(max_workers=2) as exe:
            fut_ch0 = exe.submit(_run_ch0)
            fut_ch3 = exe.submit(_run_ch3)
            ch0_dict = fut_ch0.result()
            fast_dict = fut_ch3.result()

        params_dict = {**ch0_dict, **fast_dict}
        param_names = PARAM_NAMES
        X = np.column_stack([params_dict[name] for name in param_names])
        _save_params_cache(cache_path, params_dict, param_names)
    print(f"参数矩阵形状: {X.shape} (15 参数 UMAP)")

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

    # 4. 选取 UMAP1 > umap1_threshold 的事件
    umap1_gt4 = embedding[:, 0] > umap1_threshold
    n_umap1_gt4 = int(umap1_gt4.sum())
    print(f"UMAP1 > {umap1_threshold} 的事件数: {n_umap1_gt4} / {n_events}")

    # 5. 绘图：UMAP + PN-cut 双图
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
    })
    fig, (ax_umap, ax_pncut) = plt.subplots(1, 2, figsize=(14, 6))

    # 6a. UMAP 图（15 参数）
    ax_umap.scatter(
        embedding[~umap1_gt4, 0], embedding[~umap1_gt4, 1],
        c="lightgray", s=5, alpha=0.6, label=f"UMAP1 ≤ {umap1_threshold}",
    )
    ax_umap.scatter(
        embedding[umap1_gt4, 0], embedding[umap1_gt4, 1],
        c="tab:red", s=8, alpha=0.8, label=f"UMAP1 > {umap1_threshold}",
    )
    ax_umap.axvline(x=umap1_threshold, color="gray", linestyle="--", alpha=0.7)
    ax_umap.set_xlabel("UMAP 1", fontsize=12, fontweight="bold")
    ax_umap.set_ylabel("UMAP 2", fontsize=12, fontweight="bold")
    ax_umap.set_title(f"UMAP 2D (15 参数) | n={n_events} events", fontsize=11, fontweight="bold")
    ax_umap.legend(loc="upper right", fontsize=9)
    ax_umap.grid(True, alpha=0.3)

    # 6b. PN-cut 图：所有 ±1σ 事件 + UMAP1>4 高亮
    ax_pncut.scatter(
        max_ch0[~umap1_gt4], max_ch1[~umap1_gt4],
        c="lightblue", s=3, alpha=0.5, label=f"UMAP1 ≤ {umap1_threshold}",
    )
    ax_pncut.scatter(
        max_ch0[umap1_gt4], max_ch1[umap1_gt4],
        c="tab:red", s=8, alpha=0.8, label=f"UMAP1 > {umap1_threshold}",
    )
    ax_pncut.set_xlabel("max CH0 (ADC)", fontsize=12, fontweight="bold")
    ax_pncut.set_ylabel("max CH1 (ADC)", fontsize=12, fontweight="bold")
    ax_pncut.set_title(
        f"PN-cut: max CH0 vs max CH1 | UMAP1>{umap1_threshold} 高亮 (n={n_umap1_gt4})",
        fontsize=11, fontweight="bold",
    )
    ax_pncut.legend(loc="upper left", fontsize=9)
    ax_pncut.grid(True, alpha=0.3)
    ax_pncut.set_xlim(max_ch0.min() - 100, max_ch0.max() + 100)
    ax_pncut.set_ylim(max_ch1.min() - 100, max_ch1.max() + 100)

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
            output_dir, f"ch0_ch3_umap_pncut_{n_events}events_umap1gt{umap1_threshold}_{timestamp}.png"
        )

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"UMAP+PN-cut 可视化已保存至: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

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
            umap1_threshold=5.4,
            use_cached_params=False,
            params_cache_path=None,
            save_path=None,
            show_plot=True,
        )
    except Exception as e:
        print(f"\nUMAP+PN-cut 可视化失败: {e}")
        import traceback
        traceback.print_exc()
