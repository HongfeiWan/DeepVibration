#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对 HDF5 文件中所有事件绘制 PN-cut 散点图：
- 读取数据集 `channel_data`，形状应为 (time_samples, num_channels, num_events)；
- 选择 CH0 与 CH1 通道，计算每个事件的 max(CH0)、max(CH1)；
- 绘制 max(CH0) vs max(CH1) 的散点图并保存。
"""

import os
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

def plot_pncut_scatter(
    h5_path: str,
    ch0_idx: int = 0,
    ch1_idx: int = 1,
    save_path: str | None = None,
    show: bool = True,) -> str:
    """
    对指定 HDF5 文件中所有事件，绘制 max(CH0) vs max(CH1) 散点图。

    参数：
        h5_path: HDF5 文件路径，需包含数据集 `channel_data`
        ch0_idx: CH0 通道索引（默认 0）
        ch1_idx: CH1 通道索引（默认 1）
        save_path: 图片保存路径；为 None 时自动生成
        show: 是否调用 plt.show()

    返回：
        实际保存的图片路径
    """
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"HDF5 文件不存在: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        if "channel_data" not in f:
            raise KeyError("HDF5 文件中未找到数据集 'channel_data'")

        channel_data = f["channel_data"]
        if channel_data.ndim != 3:
            raise ValueError(
                f"'channel_data' 维度应为 3，当前为 {channel_data.ndim}，"
                "预期形状 (time_samples, num_channels, num_events)"
            )

        time_samples, num_channels, num_events = channel_data.shape

        if ch0_idx < 0 or ch0_idx >= num_channels:
            raise ValueError(f"ch0_idx={ch0_idx} 超出通道数范围 [0, {num_channels-1}]")
        if ch1_idx < 0 or ch1_idx >= num_channels:
            raise ValueError(f"ch1_idx={ch1_idx} 超出通道数范围 [0, {num_channels-1}]")

        # 读取全部事件的 CH0 / CH1 波形并计算最大值
        ch0_waveforms = channel_data[:, ch0_idx, :].astype(np.float64)
        ch1_waveforms = channel_data[:, ch1_idx, :].astype(np.float64)
        max_ch0 = ch0_waveforms.max(axis=0)
        max_ch1 = ch1_waveforms.max(axis=0)

    print(f"读取文件: {h5_path}")
    print(f"事件数: {num_events}，通道数: {num_channels}")

    # 画散点
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
        }
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        max_ch0,
        max_ch1,
        s=4,
        alpha=0.6,
        c="tab:blue",
        edgecolors="none",
    )

    ax.set_xlabel("max CH0 (ADC)", fontsize=14, fontweight="bold")
    ax.set_ylabel("max CH1 (ADC)", fontsize=14, fontweight="bold")
    # 去掉文件名，只保留整体标题
    ax.set_title(
        "PN-cut Scatter: max CH0 vs max CH1",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontweight("bold")
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontweight("bold")

    plt.tight_layout()

    if save_path is None:
        # 推断项目根目录：.../python/data/pncut -> data -> python -> 项目根
        pncut_dir = os.path.dirname(os.path.abspath(__file__))    # .../python/data/pncut
        data_dir = os.path.dirname(pncut_dir)                     # .../python/data
        python_dir = os.path.dirname(data_dir)                    # .../python
        project_root = os.path.dirname(python_dir)                # 项目根

        # 默认输出到 images/presentation 目录
        output_dir = os.path.join(project_root, "images", "presentation")
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(h5_path))[0]
        save_path = os.path.join(output_dir, f"pncut_scatter_{base_name}.png")

    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"PN-cut 散点图已保存至: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path

def plot_pncut_scatter_multi(
    h5_paths: list[str],
    ch0_idx: int = 0,
    ch1_idx: int = 1,
    save_path: str | None = None,
    show: bool = True,) -> str:
    """
    将多个 HDF5 文件中的事件汇总到一张 PN-cut 散点图上。
    """
    if not h5_paths:
        raise ValueError("h5_paths 为空，未提供任何 HDF5 文件路径。")

    all_max_ch0: list[np.ndarray] = []
    all_max_ch1: list[np.ndarray] = []
    total_events = 0
    num_channels_ref: int | None = None

    for h5_path in h5_paths:
        if not os.path.isfile(h5_path):
            print(f"[警告] 跳过不存在的文件: {h5_path}")
            continue

        with h5py.File(h5_path, "r") as f:
            if "channel_data" not in f:
                print(f"[警告] 文件中未找到 'channel_data'，跳过: {h5_path}")
                continue

            channel_data = f["channel_data"]
            if channel_data.ndim != 3:
                print(
                    f"[警告] 'channel_data' 维度应为 3，当前为 {channel_data.ndim}，跳过: {h5_path}"
                )
                continue

            _, num_channels, num_events = channel_data.shape

            if num_channels_ref is None:
                num_channels_ref = num_channels
            elif num_channels_ref != num_channels:
                print(
                    f"[警告] 文件通道数不一致 (期望 {num_channels_ref}, 实际 {num_channels})，跳过: {h5_path}"
                )
                continue

            if ch0_idx < 0 or ch0_idx >= num_channels:
                print(
                    f"[警告] ch0_idx={ch0_idx} 超出通道数范围 [0, {num_channels-1}]，跳过: {h5_path}"
                )
                continue
            if ch1_idx < 0 or ch1_idx >= num_channels:
                print(
                    f"[警告] ch1_idx={ch1_idx} 超出通道数范围 [0, {num_channels-1}]，跳过: {h5_path}"
                )
                continue

            ch0_waveforms = channel_data[:, ch0_idx, :].astype(np.float64)
            ch1_waveforms = channel_data[:, ch1_idx, :].astype(np.float64)
            max_ch0 = ch0_waveforms.max(axis=0)
            max_ch1 = ch1_waveforms.max(axis=0)

            all_max_ch0.append(max_ch0)
            all_max_ch1.append(max_ch1)
            total_events += num_events

            print(f"读取文件: {h5_path} | 事件数: {num_events}，通道数: {num_channels}")

    if not all_max_ch0:
        raise RuntimeError("没有任何有效的 HDF5 文件被成功读取，无法绘图。")

    max_ch0_all = np.concatenate(all_max_ch0)
    max_ch1_all = np.concatenate(all_max_ch1)

    print(f"总共使用 {len(all_max_ch0)} 个文件，汇总事件数: {total_events}")

    # 画散点
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
        }
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        max_ch0_all,
        max_ch1_all,
        s=4,
        alpha=0.6,
        c="tab:blue",
        edgecolors="none",
    )

    ax.set_xlabel("max CH0 (ADC)", fontsize=14, fontweight="bold")
    ax.set_ylabel("max CH1 (ADC)", fontsize=14, fontweight="bold")
    ax.set_title(
        "PN-cut Scatter: max CH0 vs max CH1 (CH0-3 all files)",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontweight("bold")
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontweight("bold")

    plt.tight_layout()

    if save_path is None:
        pncut_dir = os.path.dirname(os.path.abspath(__file__))    # .../python/data/pncut
        data_dir = os.path.dirname(pncut_dir)                     # .../python/data
        python_dir = os.path.dirname(data_dir)                    # .../python
        project_root = os.path.dirname(python_dir)                # 项目根

        output_dir = os.path.join(project_root, "images", "presentation")
        os.makedirs(output_dir, exist_ok=True)

        save_path = os.path.join(output_dir, "pncut_scatter_CH0-3_all.png")

    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"PN-cut 散点图（多文件）已保存至: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path

def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="对 HDF5 文件所有事件绘制 max(CH0) vs max(CH1) PN-cut 散点图。"
    )
    parser.add_argument(
        "--file",
        "-f",
        dest="h5_path",
        required=False,
        default=None,
        help="输入 HDF5 文件路径（需包含数据集 'channel_data'）；"
        "若不提供，则默认读取 data/hdf5/raw_pulse/CH0-3 目录下的所有 .h5 文件并汇总绘制。",
    )
    parser.add_argument(
        "--ch0-idx",
        type=int,
        default=0,
        help="CH0 通道索引（默认 0）",
    )
    parser.add_argument(
        "--ch1-idx",
        type=int,
        default=1,
        help="CH1 通道索引（默认 1）",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="save_path",
        default=None,
        help="输出图片路径（默认自动生成）",
    )
    return parser

if __name__ == "__main__":
    args = _build_argparser().parse_args()

    # 若未指定单个文件，则默认读取 CH0-3 目录下的所有 .h5 文件并汇总到一张图中
    if args.h5_path is None:
        pncut_dir = os.path.dirname(os.path.abspath(__file__))    # .../python/data/pncut
        data_dir = os.path.dirname(pncut_dir)                     # .../python/data
        python_dir = os.path.dirname(data_dir)                    # .../python
        project_root = os.path.dirname(python_dir)                # 项目根

        ch_dir = os.path.join(project_root, "data", "hdf5", "raw_pulse", "CH0-3")
        if not os.path.isdir(ch_dir):
            raise FileNotFoundError(f"默认目录不存在: {ch_dir}")

        all_h5_paths = [
            os.path.join(ch_dir, name)
            for name in sorted(os.listdir(ch_dir))
            if name.lower().endswith(".h5")
        ]
        if not all_h5_paths:
            raise FileNotFoundError(f"目录中未找到任何 .h5 文件: {ch_dir}")

        # 若默认目录中的 h5 文件数大于 20，只取前 20 个进行绘制
        if len(all_h5_paths) > 20:
            print(f"在目录 {ch_dir} 中共找到 {len(all_h5_paths)} 个 .h5 文件，仅使用前 20 个进行绘图。")
            h5_paths = all_h5_paths[:20]
        else:
            h5_paths = all_h5_paths

        plot_pncut_scatter_multi(
            h5_paths=h5_paths,
            ch0_idx=args.ch0_idx,
            ch1_idx=args.ch1_idx,
            save_path=args.save_path,
            show=True,
        )
    else:
        plot_pncut_scatter(
            h5_path=args.h5_path,
            ch0_idx=args.ch0_idx,
            ch1_idx=args.ch1_idx,
            save_path=args.save_path,
            show=True,
        )

