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
        s=4,  # 点更小
        alpha=0.6,
        c="tab:blue",
        edgecolors="none",
    )

    ax.set_xlabel("max CH0 (ADC)", fontsize=14, fontweight="bold")
    ax.set_ylabel("max CH1 (ADC)", fontsize=14, fontweight="bold")
    # 去掉文件名，只保留整体标题
    ax.set_title(
        "PN-cut Scatter: max CH0 vs max CH1",
        fontsize=10,
        fontweight="bold",
    )
    #ax.grid(True, alpha=0.3)

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
    show: bool = True,
) -> str:
    """
    将多个 HDF5 文件中的事件汇总到一张 PN-cut 散点图上，
    并根据 RT / Inhibit / ge-self 条件用不同颜色区分：
    - RT:   CH5 最大值 > 6000 (蓝色点)
    - Inhibit: CH0 最小值 == 0 (黑色点)
    - ge-self (Physical): 既非 RT 也非 Inhibit (红色点)
    """
    if not h5_paths:
        raise ValueError("h5_paths 为空，未提供任何 HDF5 文件路径。")

    # 分类结果
    rt_x, rt_y = [], []
    inhibit_x, inhibit_y = [], []
    geself_x, geself_y = [], []
    acv_x, acv_y = [], []
    total_events = 0

    # RT 阈值 (CH5) 与 NaI 触发阈值 (CH4)
    rt_cut = 6000.0
    nai_trigger_threshold = 7060.0
    t_ge_us = 40.0
    sampling_interval_ns = 4.0

    # 写死 CH5 / CH4 目录：相对当前脚本向上四级即项目根，再拼接 data/hdf5/raw_pulse/CH5 / CH4
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
    ch5_dir = os.path.join(project_root, "data", "hdf5", "raw_pulse", "CH5")
    ch4_dir = os.path.join(project_root, "data", "hdf5", "raw_pulse", "CH4")

    for h5_path in h5_paths:
        if not os.path.isfile(h5_path):
            print(f"[警告] 跳过不存在的文件: {h5_path}")
            continue

        with h5py.File(h5_path, "r") as f_ch0:
            if "channel_data" not in f_ch0:
                print(f"[警告] 文件中未找到 'channel_data'，跳过: {h5_path}")
                continue

            channel_data = f_ch0["channel_data"]
            if channel_data.ndim != 3:
                print(
                    f"[警告] 'channel_data' 维度应为 3，当前为 {channel_data.ndim}，跳过: {h5_path}"
                )
                continue

            _, num_channels, num_events = channel_data.shape

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

            # 计算 max CH0 / max CH1
            ch0_waveforms = channel_data[:, ch0_idx, :].astype(np.float64)
            ch1_waveforms = channel_data[:, ch1_idx, :].astype(np.float64)
            max_ch0 = ch0_waveforms.max(axis=0)
            max_ch1 = ch1_waveforms.max(axis=0)

            # 计算 CH0 最小值（用于 Inhibit 判据）
            ch0_min_values = ch0_waveforms.min(axis=0)

            # 尝试找到对应的 CH5 文件并计算 max CH5（用于 RT 判据）
            ch5_path = os.path.join(ch5_dir, os.path.basename(h5_path))
            if os.path.isfile(ch5_path):
                with h5py.File(ch5_path, "r") as f_ch5:
                    if "channel_data" not in f_ch5:
                        print(f"[警告] CH5 文件中未找到 'channel_data'，跳过 RT 判据: {ch5_path}")
                        ch5_max_values = np.zeros(num_events, dtype=np.float64)
                    else:
                        ch5_channel_data = f_ch5["channel_data"]
                        if ch5_channel_data.ndim != 3:
                            print(
                                f"[警告] CH5 'channel_data' 维度应为 3，当前为 {ch5_channel_data.ndim}，跳过 RT 判据: {ch5_path}"
                            )
                            ch5_max_values = np.zeros(num_events, dtype=np.float64)
                        else:
                            ts5, n_ch5, n_evt5 = ch5_channel_data.shape
                            if n_evt5 != num_events:
                                print(
                                    f"[警告] CH5 事件数 ({n_evt5}) 与 CH0-3 ({num_events}) 不一致，跳过 RT 判据: {ch5_path}"
                                )
                                ch5_max_values = np.zeros(num_events, dtype=np.float64)
                            else:
                                # 假设 CH5 文件只有一个物理通道（索引 0）
                                ch5_waveforms = ch5_channel_data[:, 0, :].astype(np.float64)
                                ch5_max_values = ch5_waveforms.max(axis=0)
            else:
                print(f"[警告] 未找到对应的 CH5 文件，跳过 RT 判据: {ch5_path}")
                ch5_max_values = np.zeros(num_events, dtype=np.float64)

            # 尝试找到对应的 CH4 文件并计算 NaI 触发信息（用于 ACV 判据）
            ch4_path = os.path.join(ch4_dir, os.path.basename(h5_path))
            if os.path.isfile(ch4_path):
                with h5py.File(ch4_path, "r") as f_ch4:
                    if "channel_data" not in f_ch4:
                        print(f"[警告] CH4 文件中未找到 'channel_data'，跳过 ACV 判据: {ch4_path}")
                        nai_max_values = np.zeros(num_events, dtype=np.float64)
                        delta_t_us = np.full(num_events, np.nan, dtype=np.float64)
                    else:
                        ch4_channel_data = f_ch4["channel_data"]
                        if ch4_channel_data.ndim != 3:
                            print(
                                f"[警告] CH4 'channel_data' 维度应为 3，当前为 {ch4_channel_data.ndim}，跳过 ACV 判据: {ch4_path}"
                            )
                            nai_max_values = np.zeros(num_events, dtype=np.float64)
                            delta_t_us = np.full(num_events, np.nan, dtype=np.float64)
                        else:
                            _, _, n_evt4 = ch4_channel_data.shape
                            if n_evt4 != num_events:
                                print(
                                    f"[警告] CH4 事件数 ({n_evt4}) 与 CH0-3 ({num_events}) 不一致，跳过 ACV 判据: {ch4_path}"
                                )
                                nai_max_values = np.zeros(num_events, dtype=np.float64)
                                delta_t_us = np.full(num_events, np.nan, dtype=np.float64)
                            else:
                                nai_waveforms = ch4_channel_data[:, 0, :].astype(np.float64)
                                nai_max_values = nai_waveforms.max(axis=0)
                                nai_max_indices = nai_waveforms.argmax(axis=0)
                                t_nai_us = nai_max_indices.astype(np.float64) * sampling_interval_ns * 1e-3
                                delta_t_us = t_ge_us - t_nai_us
            else:
                print(f"[警告] 未找到对应的 CH4 文件，跳过 ACV 判据: {ch4_path}")
                nai_max_values = np.zeros(num_events, dtype=np.float64)
                delta_t_us = np.full(num_events, np.nan, dtype=np.float64)

            # 根据条件分类
            rt_mask = ch5_max_values > rt_cut
            inhibit_mask = ch0_min_values == 0
            geself_mask = (~rt_mask) & (~inhibit_mask)
            acv_mask = (
                geself_mask
                & (nai_max_values >= nai_trigger_threshold)
                & (delta_t_us >= 1.0)
                & (delta_t_us <= 16.0)
            )
            geself_mask = geself_mask & (~acv_mask)

            # 各类事件的坐标
            rt_x.append(max_ch0[rt_mask])
            rt_y.append(max_ch1[rt_mask])
            inhibit_x.append(max_ch0[inhibit_mask])
            inhibit_y.append(max_ch1[inhibit_mask])
            geself_x.append(max_ch0[geself_mask])
            geself_y.append(max_ch1[geself_mask])
            acv_x.append(max_ch0[acv_mask])
            acv_y.append(max_ch1[acv_mask])

            total_events += num_events
            print(
                f"读取文件: {h5_path} | 事件数: {num_events}，"
                f"RT: {rt_mask.sum()}，Inhibit: {inhibit_mask.sum()}，"
                f"ge-self: {geself_mask.sum()}，ACV: {acv_mask.sum()}"
            )

    if total_events == 0:
        raise RuntimeError("没有任何有效的 HDF5 文件被成功读取，无法绘图。")

    # 拼接所有文件的数据
    def _concat(arrs: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(arrs) if arrs else np.array([], dtype=np.float64)

    rt_x_all = _concat(rt_x)
    rt_y_all = _concat(rt_y)
    inh_x_all = _concat(inhibit_x)
    inh_y_all = _concat(inhibit_y)
    ges_x_all = _concat(geself_x)
    ges_y_all = _concat(geself_y)
    acv_x_all = _concat(acv_x)
    acv_y_all = _concat(acv_y)

    # 画散点（按类别着色）
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
        }
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    # RT：蓝色
    if rt_x_all.size > 0:
        ax.scatter(rt_x_all, rt_y_all, s=2, alpha=0.6, c="blue", edgecolors="none", label="RT")
    # Inhibit：黑色
    if inh_x_all.size > 0:
        ax.scatter(inh_x_all, inh_y_all, s=2, alpha=0.6, c="black", edgecolors="none", label="Inhibit")
    # ge-self：红色
    if ges_x_all.size > 0:
        ax.scatter(ges_x_all, ges_y_all, s=2, alpha=0.6, c="red", edgecolors="none", label="ge-self")
    # ACV：绿色
    if acv_x_all.size > 0:
        ax.scatter(acv_x_all, acv_y_all, s=3, alpha=0.7, c="green", edgecolors="none", label="ACV")

    ax.set_xlabel("max CH0 (ADC)", fontsize=14, fontweight="bold")
    ax.set_ylabel("max CH1 (ADC)", fontsize=14, fontweight="bold")
    ax.set_title(
        "PN-cut Scatter: max CH0 vs max CH1\n(RT / Inhibit / ge-self / ACV)",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontweight("bold")
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontweight("bold")

    if any([
        rt_x_all.size > 0,
        inh_x_all.size > 0,
        ges_x_all.size > 0,
        acv_x_all.size > 0,
    ]):
        # 图例中点的大小调大一些（markerscale > 1）
        ax.legend(loc="best", fontsize=9, markerscale=3)

    plt.tight_layout()

    if save_path is None:
        # 写死输出目录：项目根下的 images/presentation
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
        output_dir = os.path.join(project_root, "images", "presentation")
        os.makedirs(output_dir, exist_ok=True)

        save_path = os.path.join(output_dir, "pncut_scatter_CH0-3_all.png")

    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"已保存至: {save_path}")

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

    # 若未指定单个文件，则默认读取 data/hdf5/raw_pulse/CH0-3 下的所有 .h5 文件并汇总到一张图中
    if args.h5_path is None:
        # 写死 CH0-3 目录：相对当前脚本向上四级即项目根，再拼接 data/hdf5/raw_pulse/CH0-3
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
        ch_dir = os.path.join(project_root, "data", "hdf5", "raw_pulse", "CH0-3")

        all_h5_paths = [
            os.path.join(ch_dir, name)
            for name in sorted(os.listdir(ch_dir))
            if name.lower().endswith(".h5")
        ]

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

