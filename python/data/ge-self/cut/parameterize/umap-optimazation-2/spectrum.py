#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 30 参数 UMAP+HDBSCAN 事件映射 HDF5 中，选出指定 cluster 的所有事件，
根据 (文件路径, event 号) 到 `raw_pulse/CH0max` 中读取预先计算好的 max(ch0)，
并将其与对应文件在 `raw_pulse/RTCH0max` 中的 RT 事件（rt_event_indices 的并集）
一起计入能谱（即 RT 事件也算在 count 中，且其 CH0max 直接从 CH0max 读取）。

HDF5 结构参考 `30parameter&HDBSCAN.py` 中的 `_save_cluster_eventmap_hdf5`：
    - file_paths               : (n_files,)  字节串/字符串
    - event_file_indices       : (n_events,)
    - event_event_indices      : (n_events,)
    - event_cluster_labels     : (n_events,)

ch0max HDF5 结构（由 python/data/ch0max.py 写出）：
    - max_ch0                  : (n_events,) 每个 event 的 max(ch0)
    - attrs['source_file']     : 对应的 CH0-3 HDF5 路径

RTCH0max HDF5 结构（由 python/data/randomtrigger/ch0max.py 写出）：
    - rt_event_indices         : (n_rt_events,) RT 事件在原始文件中的索引
    - rt_ch0max                : (n_rt_events,) 对应事件的 max_ch0（与 CH0max 中一致）

脚本默认会使用一个相对于本脚本路径推导出的事件映射 HDF5 文件；
也可以通过命令行参数指定一个“相对于本脚本”的 HDF5 路径。
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def _load_event_mapping(path: Path) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    读取事件映射 HDF5，返回：
        file_paths, event_file_indices, event_event_indices, labels
    """
    if not path.exists():
        raise FileNotFoundError(f"事件映射 HDF5 不存在: {path}")

    with h5py.File(path, "r") as f:
        file_paths_raw = f["file_paths"][...]
        event_file_indices = f["event_file_indices"][...]
        event_event_indices = f["event_event_indices"][...]
        labels = f["event_cluster_labels"][...]

    file_paths: List[str] = []
    for p in file_paths_raw:
        if isinstance(p, bytes):
            file_paths.append(p.decode("utf-8"))
        else:
            file_paths.append(str(p))

    return file_paths, event_file_indices, event_event_indices, labels


def _compute_max_ch0_for_cluster(
    file_paths: Sequence[str],
    event_file_indices: np.ndarray,
    event_event_indices: np.ndarray,
    labels: np.ndarray,
    target_cluster: int = 0,
    ch0_index: int = 0,
) -> np.ndarray:
    """
    对给定 cluster 中的所有事件，从 max(ch0) 源文件中读取 max(ch0)；
    同时把对应文件在 RTCH0max 中的 RT 事件一起纳入（按 event 索引取并集），
    返回所有这些事件的 max(ch0) 一维数组。
    """
    mask = labels.astype(int) == int(target_cluster)
    if not np.any(mask):
        print(f"cluster={target_cluster}: 在映射文件中没有事件。")
        return np.array([], dtype=np.float64)

    indices = np.nonzero(mask)[0]
    print(f"cluster={target_cluster}: 共有事件数 = {indices.size}")

    # 先按文件索引分组，避免频繁打开/关闭文件
    file_to_events: Dict[int, List[int]] = defaultdict(list)
    for i in indices:
        fi = int(event_file_indices[i])
        ev = int(event_event_indices[i])
        file_to_events[fi].append(ev)

    # 通过脚本位置推导 ch0max 目录（与相关脚本保持一致的层级约定）
    script_dir = Path(__file__).resolve().parent  # .../parameterize/umap-optimazation-2
    parameterize_dir = script_dir.parent          # .../parameterize
    cut_dir = parameterize_dir.parent             # .../cut
    ge_self_dir = cut_dir.parent                  # .../ge-self
    data_dir = ge_self_dir.parent                 # .../data
    python_dir = data_dir.parent                  # .../python
    project_root = python_dir.parent              # 项目根
    ch0max_dir = project_root / "data" / "hdf5" / "raw_pulse" / "CH0max"
    ch0_param_dir = project_root / "data" / "hdf5" / "raw_pulse" / "CH0_parameters"
    rtch0max_dir = project_root / "data" / "hdf5" / "raw_pulse" / "RTCH0max"

    max_values: List[float] = []

    for fi, ev_list in file_to_events.items():
        if fi < 0 or fi >= len(file_paths):
            continue
        src_path = Path(file_paths[fi])

        # max(ch0) 源文件优先级：
        # 1) raw_pulse/CH0max/<同名文件>（若存在）
        # 2) raw_pulse/CH0_parameters/<同名文件>（由 preprocessor.py 输出）
        ch0max_path = ch0max_dir / src_path.name
        ch0_param_path = ch0_param_dir / src_path.name
        max_source_path: Path | None = None
        if ch0max_path.exists():
            max_source_path = ch0max_path
        elif ch0_param_path.exists():
            max_source_path = ch0_param_path

        if max_source_path is None:
            print(
                f"警告: 未找到 max(ch0) 源文件，跳过: "
                f"{ch0max_path} 或 {ch0_param_path}"
            )
            continue
        rtch0max_path = rtch0max_dir / src_path.name

        ev_arr = np.asarray(ev_list, dtype=np.int64)

        with h5py.File(max_source_path, "r") as f_max:
            if "max_ch0" not in f_max:
                print(
                    f"警告: max(ch0) 源文件中缺少 'max_ch0' 数据集，跳过: {max_source_path}"
                )
                continue
            dset = f_max["max_ch0"]
            if dset.ndim != 1:
                print(f"警告: 'max_ch0' 维度不是 1, shape={dset.shape}，跳过: {ch0max_path}")
                continue
            n_events_in_file = dset.shape[0]

            valid_mask = (ev_arr >= 0) & (ev_arr < n_events_in_file)
            ev_arr = ev_arr[valid_mask]
            if ev_arr.size == 0:
                continue

            # 将 cluster 事件与 RT 事件按 event index 做并集
            union_indices = ev_arr
            if rtch0max_path.exists():
                with h5py.File(rtch0max_path, "r") as f_rt:
                    if "rt_event_indices" in f_rt:
                        rt_indices = np.asarray(f_rt["rt_event_indices"][...], dtype=np.int64)
                        rt_valid = (rt_indices >= 0) & (rt_indices < n_events_in_file)
                        rt_indices_valid = rt_indices[rt_valid]
                        if rt_indices_valid.size > 0:
                            union_indices = np.unique(
                                np.concatenate([ev_arr, rt_indices_valid])
                            )

            max_vals_file = np.asarray(dset[union_indices], dtype=np.float64)
            max_values.extend(max_vals_file.tolist())

    max_values_arr = np.asarray(max_values, dtype=np.float64)
    print(f"cluster={target_cluster}: 成功计算 max(ch0) 的事件数 = {max_values_arr.size}")
    return max_values_arr


def _plot_histogram(max_values: np.ndarray, bins: int = 100, cluster: int = 0) -> None:
    """画出 max(ch0) 的直方图。"""
    if max_values.size == 0:
        print("没有可用于画图的 max(ch0) 数据。")
        return

    # ADC -> 能量(keV) 线性变换: E = a * x + b
    a = 0.0008432447500464594
    b = -0.826976770117076
    energy_values = a * max_values + b

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})

    # 先用 numpy.histogram 得到计数和能量 bin 边界
    counts, bin_edges = np.histogram(energy_values, bins=bins)
    bin_widths = np.diff(bin_edges)  # keV
    bin_centers = bin_edges[:-1] + bin_widths / 2.0

    # 归一化：rate = count / (0.5 kg * bin_width (keV) * 20 day)
    exposure_kg = 0.5
    exposure_days = 20.0
    denom = exposure_kg * bin_widths * exposure_days
    # 避免除以 0（理论上 bin_width 不会为 0，这里只是保险）
    denom[denom == 0] = np.inf
    rates = counts / denom

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.bar(bin_centers, rates, width=bin_widths, color="C0", alpha=0.8, align="center")
    ax.set_yscale("log")
    ax.set_xlabel("Energy (keV)", fontsize=12)
    ax.set_ylabel(r"Rate [counts / (keV·kg·day)]", fontsize=12)
    ax.set_title(f"Energy spectrum for cluster={cluster}", fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # ------------------------------------------------------------------
    # 在 10–11 keV 能区内寻找峰，并进行高斯 + 线性本底拟合
    # ------------------------------------------------------------------
    e_min, e_max = 10.0, 11.0
    mask_roi = (bin_centers >= e_min) & (bin_centers <= e_max) & (rates > 0)

    if np.count_nonzero(mask_roi) >= 5:
        x_roi = bin_centers[mask_roi]
        y_roi = rates[mask_roi]

        # 初始参数估计
        peak_idx = np.argmax(y_roi)
        mu0 = x_roi[peak_idx]
        amp0 = y_roi[peak_idx] - np.min(y_roi)
        sigma0 = 0.05  # keV，经验初始值
        c0 = np.min(y_roi)
        d0 = 0.0

        def gauss_linear(x, A, mu, sigma, c, d):
            return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + c + d * x

        try:
            popt, pcov = curve_fit(
                gauss_linear,
                x_roi,
                y_roi,
                p0=[amp0, mu0, sigma0, c0, d0],
                maxfev=10000,
            )
            A_fit, mu_fit, sigma_fit, c_fit, d_fit = popt

            # 计算 FWHM
            fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * abs(sigma_fit)
            print(f"10–11 keV 峰拟合结果: mu = {mu_fit:.4f} keV, FWHM = {fwhm:.4f} keV")

            # 在新窗口内画出拟合函数与数据点
            fig_fit, ax_fit = plt.subplots(1, 1, figsize=(8, 6))
            ax_fit.scatter(x_roi, y_roi, color="C0", label="Data (10–11 keV)", zorder=3)

            x_fit = np.linspace(x_roi.min(), x_roi.max(), 400)
            y_fit = gauss_linear(x_fit, *popt)
            ax_fit.plot(x_fit, y_fit, color="C1", label="Gaussian + linear fit")

            ax_fit.set_xlabel("Energy (keV)", fontsize=12)
            ax_fit.set_ylabel(r"Rate [counts / (keV·kg·day)]", fontsize=12)
            ax_fit.set_title(
                f"Peak fit in [{e_min}, {e_max}] keV\n"
                f"mu = {mu_fit:.4f} keV, FWHM = {fwhm:.4f} keV",
                fontsize=13,
            )
            ax_fit.grid(True, alpha=0.3)
            ax_fit.legend()
            fig_fit.tight_layout()
        except Exception as exc:
            print(f"10–11 keV 拟合失败: {exc}")
    else:
        print("10–11 keV 区间内有效点太少，跳过拟合。")

    plt.show()


def _default_hdf5_path_relative_to_script() -> Path:
    """
    构造一个默认的事件映射 HDF5 路径：
    - 先从本脚本所在目录推导出项目根目录；
    - 再在项目根目录下拼出 data/hdf5/xxx.h5。

    这里默认使用 `30parameter&HDBSCAN.py` 的缓存输出文件名。
    """
    script_dir = Path(__file__).resolve().parent  # .../cut/parameterize/umap-optimazation-2

    # 对应 15parameter_spectrum.py 里推导 project_root 的逻辑：
    parameterize_dir = script_dir.parent          # .../cut/parameterize
    cut_dir = parameterize_dir.parent             # .../cut
    ge_self_dir = cut_dir.parent                  # .../ge-self
    data_dir = ge_self_dir.parent                 # .../data
    python_dir = data_dir.parent                  # .../python
    project_root = python_dir.parent              # 项目根目录

    return project_root / "data" / "hdf5" / "ge_30param_umap_hdbscan_eventmap.h5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "读取 30 参数 UMAP+HDBSCAN 事件映射 HDF5，"
            "对指定 cluster 的所有事件从 CH0max 读取 max(ch0)，并画出直方图。"
        )
    )
    parser.add_argument(
        "hdf5_path",
        nargs="?",
        help=(
            "事件映射 HDF5 路径；若不指定，则使用一个相对于本脚本路径推导出的默认文件。"
            "若提供的是相对路径，将被解释为“相对于本脚本所在目录”。"
        ),
    )
    parser.add_argument(
        "--cluster",
        type=int,
        default=0,
        help="要分析的 cluster label（默认 0）。",
    )
    parser.add_argument(
        "--ch0-index",
        type=int,
        default=0,
        help="ch0 在 channel_data 中的通道索引（默认 0）。",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="直方图的 bin 数（默认 100）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    if args.hdf5_path:
        raw_path = Path(args.hdf5_path)
        # 若为相对路径，则解释为“相对于脚本”的路径
        if not raw_path.is_absolute():
            hdf5_path = (script_dir / raw_path).resolve()
        else:
            hdf5_path = raw_path
    else:
        hdf5_path = _default_hdf5_path_relative_to_script()

    print(f"使用事件映射 HDF5: {hdf5_path}")

    file_paths, event_file_indices, event_event_indices, labels = _load_event_mapping(
        hdf5_path
    )

    max_values = _compute_max_ch0_for_cluster(
        file_paths=file_paths,
        event_file_indices=event_file_indices,
        event_event_indices=event_event_indices,
        labels=labels,
        target_cluster=args.cluster,
        ch0_index=args.ch0_index,
    )

    _plot_histogram(max_values, bins=args.bins, cluster=args.cluster)


if __name__ == "__main__":
    main()
