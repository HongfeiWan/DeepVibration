#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对指定目录下的所有 CH0-3 bin 文件，使用
`15parameter(ch0)&HDBSCAN（ch3）-batch.py` 中 **相同的 15 参数提取 + UMAP + HDBSCAN 聚类方法**，
并做 UMAP 可视化。

额外功能：
- 把 **每一个聚类后的事件** 对应的：
  - 文件路径（通过文件索引间接给出）；
  - 在该文件内的绝对 event 号；
  - 所属 HDBSCAN cluster label；
 统一写入一个 HDF5 文件，方便之后“按簇回溯到原始波形”。

本脚本不再区分“训练 bin / 搜索 bin”，只是一遍跑完、画图、存索引，
逻辑与 batch 版本保持一致，只是多了 HDF5 输出。
"""

import os
import sys
import importlib.util
from typing import List, Tuple, Dict, Optional

from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


# 路径和依赖脚本导入
current_dir = os.path.dirname(os.path.abspath(__file__))   # .../parameterize/umap-optimazation-1
parameterize_dir = os.path.dirname(current_dir)             # .../parameterize
cut_dir = os.path.dirname(parameterize_dir)                 # .../cut

if parameterize_dir not in sys.path:
    sys.path.insert(0, parameterize_dir)

# 导入 15 参数批量脚本作为模块，以复用其中的参数计算和文件枚举逻辑
batch_script_path = os.path.join(current_dir, "15parameter(ch0)&HDBSCAN（ch3）-batch.py")
spec_batch = importlib.util.spec_from_file_location("param15_batch", batch_script_path)
param15_batch = importlib.util.module_from_spec(spec_batch)
assert spec_batch.loader is not None
spec_batch.loader.exec_module(param15_batch)  # type: ignore

# 复用 batch 脚本里的对象
PARAM_NAMES: List[str] = param15_batch.PARAM_NAMES
_get_h5_files_in_dir_or_default = param15_batch._get_h5_files_in_dir_or_default
_build_ch5_basename_to_path = param15_batch._build_ch5_basename_to_path
_select_events_in_1sigma_band = param15_batch._select_events_in_1sigma_band


def _process_single_event_params_wrapper(args):
    """顶层包装函数，内部直接调用 batch 脚本里的 _process_single_event_params。

    注意：必须定义在本模块顶层，才能被 Windows 多进程安全 pickle。
    """
    return param15_batch._process_single_event_params(args)


def _compute_params_for_events_multiprocess(
    ch0_3_file: str,
    event_ranks: np.ndarray,
    selected_indices: np.ndarray,
    ch0_idx: int = 0,
    ch3_idx: int = 3,
    baseline_window_us: float = 2.0,
    max_workers: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    多进程版本的 15 参数计算。

    逻辑基本复制自 `15parameter(ch0)&HDBSCAN（ch3）-batch.py` 中的 compute_params_for_events，
    区别在于：
    - 这里的 worker 函数是本模块顶层的 `_process_single_event_params_wrapper`，
      避免了 Windows 下动态导入模块无法被子进程 pickle 的问题；
    - 仍然调用 batch 脚本内部的 `_process_single_event_params` 保证完全一致的物理含义。
    """
    print("\n正在计算所有事件的 15 参数（多进程版本）...")
    sampling_interval_ns = 4.0
    n_events = event_ranks.size

    with h5py.File(ch0_3_file, "r") as f_ch0:
        ch0_channel_data = f_ch0["channel_data"]
        global_indices = selected_indices[event_ranks]
        all_waveforms_ch0 = ch0_channel_data[:, ch0_idx, global_indices].astype(np.float64)
        all_waveforms_ch3 = ch0_channel_data[:, ch3_idx, global_indices].astype(np.float64)
        waveforms_list_ch0 = [all_waveforms_ch0[:, i] for i in range(n_events)]
        waveforms_list_ch3 = [all_waveforms_ch3[:, i] for i in range(n_events)]

    result_arrays = {key: np.zeros(n_events, dtype=np.float64) for key in param15_batch.PARAM_KEYS}
    rank_to_idx = {int(r): i for i, r in enumerate(event_ranks)}

    task_args = [
        (int(rank), waveforms_list_ch0[i], waveforms_list_ch3[i], sampling_interval_ns, baseline_window_us)
        for i, rank in enumerate(event_ranks)
    ]

    if max_workers is None:
        max_workers = os.cpu_count() or 1
    chunksize = max(1, n_events // (max_workers * 8) if max_workers > 0 else 1)
    print(f"使用 {max_workers} 个 CPU 核心并行计算 (chunksize={chunksize})")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results_iter = executor.map(
            _process_single_event_params_wrapper,
            task_args,
            chunksize=chunksize,
        )
        for rank, result in tqdm(
            results_iter,
            total=n_events,
            desc="计算15参数(多进程)",
            unit="事件",
            ncols=100,
        ):
            try:
                idx = rank_to_idx[int(rank)]
                for k in param15_batch.PARAM_KEYS:
                    result_arrays[k][idx] = result[k]
            except Exception as e:
                print(f"\n警告: 处理事件 rank={rank} 时出错: {e}")

    params_dict = {PARAM_NAMES[i]: result_arrays[param15_batch.PARAM_KEYS[i]] for i in range(len(PARAM_NAMES))}
    return params_dict


def _collect_all_params_and_sources(
    ch0_3_files: List[str],
    ch5_folder: Optional[str] = None,
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
) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
    """
    按 batch 脚本的逻辑：
    - 每个 CH0-3 文件与同名 CH5 文件配对；
    - 选出每个文件中 ±1σ 带内事件；
    - 计算 15 参数；
    - 把所有文件的参数拼成一个大矩阵；
    - 同时记录每个事件来自哪个文件、绝对 event 号。

    返回：
        X               : (n_total_events, 15) 参数矩阵（float32，节省内存）
        event_sources   : [(file_path, true_event_index)] * n_total_events
    """
    if not ch0_3_files:
        raise FileNotFoundError("未提供任何 CH0-3 文件路径。")

    ch5_basename_to_path = _build_ch5_basename_to_path(
        folder=os.path.dirname(ch0_3_files[0]),
        ch0_3_files=ch0_3_files,
        ch5_folder=ch5_folder,
    )
    if not ch5_basename_to_path:
        raise FileNotFoundError(
            "未找到任何 CH5 文件，无法进行 basename 匹配。"
            "请确认 CH5 目录存在（如 raw_pulse/CH5 或通过 ch5_folder 指定）。"
        )

    combined_params: Dict[str, List[np.ndarray]] = {name: [] for name in PARAM_NAMES}
    event_sources: List[Tuple[str, int]] = []

    for idx, fpath in enumerate(ch0_3_files, start=1):
        print("\n" + "=" * 80)
        print(f"[{idx}/{len(ch0_3_files)}] 处理文件: {fpath}")
        print("=" * 80)

        ch5_for_this = ch5_basename_to_path.get(os.path.basename(fpath))
        if ch5_for_this is None:
            print(f"  警告: 未找到与 {os.path.basename(fpath)} 同名的 CH5 文件，跳过该对")
            continue
        print(f"  匹配 CH5: {os.path.basename(ch5_for_this)}")

        # 事件选择（±1σ）
        event_ranks, ch0_3_file_sel, _, selected_indices = _select_events_in_1sigma_band(
            ch0_3_file=fpath,
            ch5_file=ch5_for_this,
            rt_cut=rt_cut,
            ch0_threshold=ch0_threshold,
            ch0_idx=ch0_idx,
            ch1_idx=ch1_idx,
            x_min=x_min,
            x_max=x_max,
            sigma_factor=sigma_factor,
        )
        n_events = event_ranks.size
        print(f"  文件 {ch0_3_file_sel} 中 ±1σ 事件数: {n_events}")
        if n_events == 0:
            continue

        # 真实 event 号
        global_indices = selected_indices[event_ranks]
        for g in global_indices:
            event_sources.append((ch0_3_file_sel, int(g)))

        # 15 参数（与 batch 一致，这里使用本脚本的多进程版本，充分利用 CPU）
        params_dict_file = _compute_params_for_events_multiprocess(
            ch0_3_file=ch0_3_file_sel,
            event_ranks=event_ranks,
            selected_indices=selected_indices,
            ch0_idx=ch0_idx,
            ch3_idx=ch3_idx,
            baseline_window_us=baseline_window_us,
            max_workers=max_workers,
        )

        for name in PARAM_NAMES:
            combined_params[name].append(params_dict_file[name])

    # 合并所有文件参数
    for name in PARAM_NAMES:
        if combined_params[name]:
            combined_params[name] = np.concatenate(combined_params[name], axis=0)
        else:
            combined_params[name] = np.empty((0,), dtype=np.float64)

    if combined_params[PARAM_NAMES[0]].size == 0:
        raise RuntimeError("所有文件均未选出 ±1σ 事件，无法进行 UMAP+HDBSCAN。")

    X = np.column_stack([combined_params[name] for name in PARAM_NAMES]).astype(np.float32)
    print(f"\n合并后: total events = {X.shape[0]}, feature dim = {X.shape[1]} (dtype={X.dtype})")

    # 及时释放中间 dict，减小内存占用
    del combined_params

    return X, event_sources


def _run_umap_hdbscan(
    X: np.ndarray,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    hdbscan_min_cluster_size: int = 50,
    hdbscan_min_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    与 batch 脚本相同的 StandardScaler + UMAP + HDBSCAN 聚类流程。
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("StandardScaler 完成")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        random_state=42,
        n_jobs=-1,
    )
    embedding = reducer.fit_transform(X_scaled)
    print(f"UMAP finished, embedding shape: {embedding.shape}")

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

    return embedding, labels, scaler


def _plot_umap(embedding: np.ndarray, labels: np.ndarray) -> None:
    """绘制 UMAP + HDBSCAN 结果，可视化方法与 batch 版本一致（只画 UMAP 散点）。"""
    import matplotlib.pyplot as plt

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
            embedding[mask, 0],
            embedding[mask, 1],
            color=color,
            s=5,
            alpha=alpha,
            label=leg_label,
        )

    ax.set_xlabel("UMAP 1", fontsize=12, fontweight="bold")
    ax.set_ylabel("UMAP 2", fontsize=12, fontweight="bold")
    ax.set_title(
        f"CH0 15-Parameter UMAP + HDBSCAN | n={embedding.shape[0]}",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def _save_event_mapping_hdf5(
    output_path: str,
    event_sources: List[Tuple[str, int]],
    labels: np.ndarray,
) -> None:
    """
    把每个事件对应的 (文件路径, 文件索引, 绝对 event 号, cluster label) 写入 HDF5。

    结构：
    - file_paths               : shape = (n_files,)，utf-8 字符串，索引即 file_index
    - event_file_indices       : shape = (n_events,)
    - event_event_indices      : shape = (n_events,)
    - event_cluster_labels     : shape = (n_events,)
    """
    assert len(event_sources) == labels.shape[0]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 构建文件路径 -> 索引 映射
    unique_paths: List[str] = []
    path_to_idx: Dict[str, int] = {}

    event_file_indices = np.empty(len(event_sources), dtype=np.int32)
    event_event_indices = np.empty(len(event_sources), dtype=np.int64)

    for i, (path, ev) in enumerate(event_sources):
        if path not in path_to_idx:
            path_to_idx[path] = len(unique_paths)
            unique_paths.append(path)
        event_file_indices[i] = path_to_idx[path]
        event_event_indices[i] = ev

    labels = labels.astype(np.int32, copy=False)

    max_len = max((len(p) for p in unique_paths), default=1)
    dt_str = f"S{max_len}"

    with h5py.File(output_path, "w") as f:
        f.create_dataset(
            "file_paths",
            data=np.array([p.encode("utf-8") for p in unique_paths], dtype=dt_str),
        )
        f.create_dataset("event_file_indices", data=event_file_indices)
        f.create_dataset("event_event_indices", data=event_event_indices)
        f.create_dataset("event_cluster_labels", data=labels)

        f.attrs["description"] = (
            "Per-event mapping for 15-parameter CH0 UMAP+HDBSCAN.\n"
            "Each event i has (file_paths[event_file_indices[i]], event_event_indices[i], "
            "event_cluster_labels[i])."
        )
        f.attrs["n_events"] = int(len(event_sources))
        f.attrs["n_files"] = int(len(unique_paths))

    print(f"\n事件映射已保存到: {output_path}")
    print(f"  覆盖文件数: {len(unique_paths)}, 事件总数: {len(event_sources)}")


def main() -> None:
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(
        description=(
            "对指定目录下的 CH0-3 bin 做 15 参数 UMAP+HDBSCAN 聚类，并把每个事件的 "
            "(文件名, 绝对 event 号, cluster label) 写入一个 HDF5。"
        )
    )

    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="CH0-3 HDF5 所在目录；若为 None，则使用 utils.visualize.get_h5_files() 默认 CH0-3 目录。",
    )
    parser.add_argument(
        "--ch5-folder",
        type=str,
        default=None,
        help="CH5 HDF5 所在目录；若为 None，则按 batch 脚本逻辑自动推断。",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=-1,
        help="最多处理的 CH0-3 文件数；<=0 表示处理目录下的所有文件。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 HDF5 路径；若未指定，则保存在项目根目录 images/presentation/ 下自动命名。",
    )
    parser.add_argument("--rt-cut", type=float, default=6000.0)
    parser.add_argument("--ch0-threshold", type=float, default=16382.0)
    parser.add_argument("--x-min", type=float, default=2000.0)
    parser.add_argument("--x-max", type=float, default=14000.0)
    parser.add_argument("--sigma-factor", type=float, default=1.0)
    parser.add_argument("--baseline-window-us", type=float, default=2.0)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=50)
    parser.add_argument("--hdbscan-min-samples", type=int, default=None)

    args = parser.parse_args()

    # 1. 获取 CH0-3 文件列表
    ch0_3_files = _get_h5_files_in_dir_or_default(args.folder)
    if not ch0_3_files:
        raise FileNotFoundError("未找到任何 CH0-3 h5 文件。")
    if args.max_files is not None and args.max_files > 0 and len(ch0_3_files) > args.max_files:
        print(
            f"共找到 {len(ch0_3_files)} 个 CH0-3 文件，仅处理前 {args.max_files} 个。"
        )
        ch0_3_files = ch0_3_files[: args.max_files]

    # 2. 收集所有事件的 15 参数 + 来源信息
    X, event_sources = _collect_all_params_and_sources(
        ch0_3_files=ch0_3_files,
        ch5_folder=args.ch5_folder,
        rt_cut=args.rt_cut,
        ch0_threshold=args.ch0_threshold,
        ch0_idx=0,
        ch1_idx=1,
        ch3_idx=3,
        x_min=args.x_min,
        x_max=args.x_max,
        sigma_factor=args.sigma_factor,
        baseline_window_us=args.baseline_window_us,
        max_workers=args.max_workers,
    )

    # 3. UMAP + HDBSCAN 聚类
    embedding, labels, _ = _run_umap_hdbscan(
        X,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
    )

    # 4. UMAP 可视化
    _plot_umap(embedding, labels)

    # 5. 写 HDF5 事件映射
    if args.output is None:
        ge_self_dir = os.path.dirname(cut_dir)
        data_dir = os.path.dirname(ge_self_dir)
        python_dir = os.path.dirname(data_dir)
        project_root = os.path.dirname(python_dir)
        output_dir = os.path.join(project_root, "images", "presentation")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            output_dir,
            f"ch0_15param_umap_hdbscan_eventmap_{X.shape[0]}events_{timestamp}.h5",
        )
    else:
        output_path = args.output

    _save_event_mapping_hdf5(
        output_path=output_path,
        event_sources=event_sources,
        labels=labels,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n15 参数 UMAP+HDBSCAN 事件索引脚本运行失败: {e}")
        import traceback
        traceback.print_exc()

