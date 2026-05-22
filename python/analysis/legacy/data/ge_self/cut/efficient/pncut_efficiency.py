#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ACV 效率：按 max(CH0) 分 bin，每 bin 效率 = 该 bin 内 ACV 事例数 / 该 bin 总事例数，并绘图。

ACV 判据与 `maxch0maxch1/distinguish/background&ACV.py` 一致：
- RT: CH5 最大值 > rt_cut（默认 6000）
- Inhibit: CH0 最小值 == 0
- ge-self: 既非 RT 也非 Inhibit
- ACV: ge-self 且 NaI 触发（CH4 max >= nai_trigger_threshold）且 1 <= Δt <= 16 µs（Δt = t_ge - t_nai）

输出：总事例数、ACV 事例数、总效率一行；横轴 max(CH0)、纵轴 efficiency 的曲线图。
"""

from __future__ import annotations

import os
import argparse
from typing import List, Tuple, Optional

import h5py
import numpy as np
import matplotlib.pyplot as plt


# 与 background&ACV.py 一致的常数
RT_CUT = 6000.0
NAI_TRIGGER_THRESHOLD = 7060.0
T_GE_US = 40.0
SAMPLING_INTERVAL_NS = 4.0


def _get_project_root(script_dir: str) -> str:
    """从 efficient 目录向上到项目根。"""
    # efficient -> cut -> ge-self -> data -> python -> 项目根
    return os.path.abspath(os.path.join(script_dir, "..", "..", "..", "..", ".."))


def _default_ch0_3_dir(script_dir: str) -> str:
    return os.path.join(
        _get_project_root(script_dir),
        "data", "hdf5", "raw_pulse", "CH0-3",
    )


def _default_ch4_ch5_dirs(script_dir: str) -> Tuple[str, str]:
    root = _get_project_root(script_dir)
    ch4_dir = os.path.join(root, "data", "hdf5", "raw_pulse", "CH4")
    ch5_dir = os.path.join(root, "data", "hdf5", "raw_pulse", "CH5")
    return ch4_dir, ch5_dir


def compute_acv_mask_one_file(
    ch0_3_path: str,
    ch4_path: Optional[str],
    ch5_path: Optional[str],
    rt_cut: float = RT_CUT,
    nai_trigger_threshold: float = NAI_TRIGGER_THRESHOLD,
    t_ge_us: float = T_GE_US,
    sampling_interval_ns: float = SAMPLING_INTERVAL_NS,
) -> Tuple[int, int, np.ndarray, int, np.ndarray]:
    """
    对单个 CH0-3 文件计算 ACV 判据，得到总事例数、ACV 事例数、ACV 掩码、通道数和 max(CH0)。

    返回:
        total_events: 总事件数
        acv_count: 通过 ACV 的事件数
        acv_mask: 形状 (num_events,) 的 bool 数组，True 表示该事件为 ACV
        num_channels: CH0-3 通道数
        max_ch0: 形状 (num_events,) 的 float 数组，每个事件的 CH0 最大值
    """
    with h5py.File(ch0_3_path, "r") as f:
        if "channel_data" not in f:
            raise KeyError(f"文件中未找到 'channel_data': {ch0_3_path}")
        channel_data = f["channel_data"]
        if channel_data.ndim != 3:
            raise ValueError(
                f"'channel_data' 维度应为 3，当前为 {channel_data.ndim}，"
                f"文件: {ch0_3_path}"
            )
        _, num_channels, num_events = channel_data.shape

        ch0_waveforms = channel_data[:, 0, :].astype(np.float64)  # (time, evt)
        max_ch0 = ch0_waveforms.max(axis=0)  # (evt,)
        ch0_all = channel_data[:, :, :].astype(np.float64)  # (time, ch, evt)
        ch0_mins = ch0_all.min(axis=0)  # (ch, evt)
        ch0_min_ch0 = ch0_mins[0, :]
        inhibit_mask = ch0_min_ch0 == 0

    ch5_max_values = np.zeros(num_events, dtype=np.float64)
    if ch5_path and os.path.isfile(ch5_path):
        with h5py.File(ch5_path, "r") as f5:
            if "channel_data" in f5:
                cd5 = f5["channel_data"]
                if cd5.ndim == 3:
                    _, _, n5 = cd5.shape
                    if n5 == num_events:
                        ch5_max_values = cd5[:, 0, :].astype(np.float64).max(axis=0)

    nai_max_values = np.zeros(num_events, dtype=np.float64)
    delta_t_us = np.full(num_events, np.nan, dtype=np.float64)
    if ch4_path and os.path.isfile(ch4_path):
        with h5py.File(ch4_path, "r") as f4:
            if "channel_data" in f4:
                cd4 = f4["channel_data"]
                if cd4.ndim == 3:
                    _, _, n4 = cd4.shape
                    if n4 == num_events:
                        nai_wf = cd4[:, 0, :].astype(np.float64)
                        nai_max_values = nai_wf.max(axis=0)
                        nai_max_indices = nai_wf.argmax(axis=0)
                        t_nai_us = nai_max_indices.astype(np.float64) * sampling_interval_ns * 1e-3
                        delta_t_us = t_ge_us - t_nai_us

    rt_mask = ch5_max_values > rt_cut
    geself_mask = (~rt_mask) & (~inhibit_mask)
    acv_mask = (
        geself_mask
        & (nai_max_values >= nai_trigger_threshold)
        & (delta_t_us >= 1.0)
        & (delta_t_us <= 16.0)
    )
    acv_count = int(np.sum(acv_mask))
    return num_events, acv_count, acv_mask, num_channels, max_ch0


def compute_efficiency_vs_max_ch0(
    ch0_3_paths: List[str],
    ch4_dir: Optional[str] = None,
    ch5_dir: Optional[str] = None,
    rt_cut: float = RT_CUT,
    nai_trigger_threshold: float = NAI_TRIGGER_THRESHOLD,
    bins: Optional[np.ndarray] = None,
    min_count_per_bin: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    对每个 max(CH0) 的 bin，计算该 bin 内 ACV 事例数 / 该 bin 总事例数作为效率。

    返回:
        bins: bin 边界（用于阶梯图）
        bin_centers: 各 bin 中心（max CH0）
        efficiencies: 各 bin 效率，分母为 0 的 bin 为 np.nan
        counts_total: 各 bin 总事例数
        total_events: 汇总总事例数
        acv_events: 汇总 ACV 事例数
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if ch4_dir is None or ch5_dir is None:
        _ch4, _ch5 = _default_ch4_ch5_dirs(script_dir)
        ch4_dir = ch4_dir or _ch4
        ch5_dir = ch5_dir or _ch5

    max_ch0_list: List[np.ndarray] = []
    acv_mask_list: List[np.ndarray] = []

    for ch0_3_path in ch0_3_paths:
        if not os.path.isfile(ch0_3_path):
            continue
        ch4_path = os.path.join(ch4_dir, os.path.basename(ch0_3_path)) if ch4_dir else None
        ch5_path = os.path.join(ch5_dir, os.path.basename(ch0_3_path)) if ch5_dir else None
        try:
            _, _, acv_mask, _, max_ch0 = compute_acv_mask_one_file(
                ch0_3_path,
                ch4_path=ch4_path,
                ch5_path=ch5_path,
                rt_cut=rt_cut,
                nai_trigger_threshold=nai_trigger_threshold,
            )
        except Exception as e:
            print(f"[警告] 跳过文件 {ch0_3_path}: {e}")
            continue
        max_ch0_list.append(max_ch0)
        acv_mask_list.append(acv_mask)

    if not max_ch0_list:
        return np.array([]), np.array([]), np.array([]), np.array([]), 0, 0

    max_ch0_all = np.concatenate(max_ch0_list)
    acv_mask_all = np.concatenate(acv_mask_list)

    if bins is None:
        # 每个 bin 宽度 100 FADC：边界的 0, 100, 200, ..., 17000
        bins = np.arange(0, 17000 + 1, 100)

    digitized = np.digitize(max_ch0_all, bins) - 1  # 0-based bin index
    n_bins = len(bins) - 1
    # 落在最后一个边界上的点会被 digitize 到 n_bins，归入最后一 bin
    digitized = np.minimum(digitized, n_bins - 1)
    counts_total = np.zeros(n_bins, dtype=np.int64)
    counts_acv = np.zeros(n_bins, dtype=np.int64)
    for i in range(n_bins):
        mask_bin = digitized == i
        counts_total[i] = mask_bin.sum()
        counts_acv[i] = (acv_mask_all & mask_bin).sum()

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    efficiencies = np.full(n_bins, np.nan, dtype=np.float64)
    valid = counts_total >= min_count_per_bin
    efficiencies[valid] = counts_acv[valid].astype(np.float64) / counts_total[valid]

    total_events = int(max_ch0_all.size)
    acv_events = int(acv_mask_all.sum())
    return bins, bin_centers, efficiencies, counts_total, total_events, acv_events


def plot_efficiency_vs_max_ch0(
    bins: np.ndarray,
    bin_centers: np.ndarray,
    efficiencies: np.ndarray,
    counts_total: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
) -> str:
    """绘制横轴 max(CH0)、纵轴 efficiency，阶梯图（每 bin 一水平段），符合高纯锗/核物理分 bin 效率惯例；X 轴每小格 100 FADC。"""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
    })
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = ~np.isnan(efficiencies)
    # 只画有效 bin 的连续区间，用阶梯图（stairs）表示分 bin 效率，避免散点杂乱
    valid_idx = np.where(valid)[0]
    if len(valid_idx) > 0:
        i0, i1 = int(valid_idx[0]), int(valid_idx[-1])
        edges = bins[i0 : i1 + 2]
        vals = efficiencies[i0 : i1 + 1]
        ax.stairs(vals, edges, color="tab:blue", label="ACV efficiency", linewidth=1.2)
    ax.set_xlabel("max(CH0) (FADC)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Efficiency (ACV / total)", fontsize=12, fontweight="bold")
    ax.set_title(
        "ACV efficiency vs max(CH0)\n(bin width 100 FADC, step plot)",
        fontsize=11,
        fontweight="bold",
    )
    # X 轴每小格 100 FADC
    ax.xaxis.set_minor_locator(plt.MultipleLocator(100))
    ax.xaxis.set_major_locator(plt.MultipleLocator(1000))
    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.2, linestyle=":")
    ax.set_ylim(-0.05, 1.05)
    if save_path:
        d = os.path.dirname(save_path)
        if d:
            os.makedirs(d, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"效率曲线已保存: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return save_path or ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按 max(CH0) 分 bin 计算 ACV 效率并绘制效率曲线。"
    )
    parser.add_argument(
        "--folder",
        "-f",
        type=str,
        default=None,
        help="CH0-3 HDF5 所在目录；默认使用 data/hdf5/raw_pulse/CH0-3",
    )
    parser.add_argument(
        "--ch4-dir",
        type=str,
        default=None,
        help="CH4 所在目录；默认使用 data/hdf5/raw_pulse/CH4",
    )
    parser.add_argument(
        "--ch5-dir",
        type=str,
        default=None,
        help="CH5 所在目录；默认使用 data/hdf5/raw_pulse/CH5",
    )
    parser.add_argument(
        "--rt-cut",
        type=float,
        default=RT_CUT,
        help=f"RT 判据：CH5 最大值 > 该值（默认 {RT_CUT}）",
    )
    parser.add_argument(
        "--nai-threshold",
        type=float,
        default=NAI_TRIGGER_THRESHOLD,
        help=f"NaI 触发阈值（CH4 max >= 该值，默认 {NAI_TRIGGER_THRESHOLD}）",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=-1,
        help="最多处理文件数；<=0 表示全部",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="ACV 效率 vs max(CH0) 曲线图保存路径；不指定则自动生成到 images/presentation",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="不弹出效率曲线图窗口，仅保存（若指定了 --output）",
    )
    parser.add_argument(
        "--bin-width",
        type=int,
        default=100,
        help="每个 bin 的 FADC 宽度（默认 100，即每 bin 100 FADC）",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder = args.folder or _default_ch0_3_dir(script_dir)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"CH0-3 目录不存在: {folder}")

    ch0_3_paths = [
        os.path.join(folder, name)
        for name in sorted(os.listdir(folder))
        if name.lower().endswith(".h5")
    ]
    if not ch0_3_paths:
        raise FileNotFoundError(f"目录下未找到 .h5 文件: {folder}")

    if args.max_files > 0 and len(ch0_3_paths) > args.max_files:
        ch0_3_paths = ch0_3_paths[: args.max_files]
        print(f"仅使用前 {args.max_files} 个文件。")

    ch4_dir = args.ch4_dir
    ch5_dir = args.ch5_dir
    if ch4_dir is None or ch5_dir is None:
        _ch4, _ch5 = _default_ch4_ch5_dirs(script_dir)
        ch4_dir = ch4_dir or _ch4
        ch5_dir = ch5_dir or _ch5

    bins = np.arange(0, 17000 + 1, args.bin_width)
    bins, bin_centers, efficiencies, counts_total, total_events, acv_events = compute_efficiency_vs_max_ch0(
        ch0_3_paths,
        ch4_dir=ch4_dir,
        ch5_dir=ch5_dir,
        rt_cut=args.rt_cut,
        nai_trigger_threshold=args.nai_threshold,
        bins=bins,
    )
    if total_events == 0:
        print("未得到任何统计结果（无有效文件）。")
        return

    eff_overall = acv_events / total_events if total_events > 0 else 0.0
    print(f"CH0-3 目录: {folder}")
    print(f"文件数: {len(ch0_3_paths)}")
    print(f"总事例数: {total_events},  ACV 事例数: {acv_events},  效率: {eff_overall:.4f}")

    if bin_centers.size > 0:
        if args.output is None:
            root = _get_project_root(script_dir)
            args.output = os.path.join(root, "images", "presentation", "acv_efficiency_vs_max_ch0.png")
        plot_efficiency_vs_max_ch0(
            bins,
            bin_centers,
            efficiencies,
            counts_total,
            save_path=args.output,
            show=not args.no_plot,
        )


if __name__ == "__main__":
    main()
