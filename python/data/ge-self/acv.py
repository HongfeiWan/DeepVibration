#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CH0 主放最大值与触发时间差 Δt 关系

数据来源:
- 使用 data/hdf5/raw_pulse/CH0-3 与 CH4 目录中的 HDF5 文件；
- 对同一 run 的 CH0-3 与 CH4 文件按文件名一一对应；
- 对每个事件：
  * 从 CH0-3 文件中读取 CH0 通道波形，计算主放最大值 Amax(CH0)，作为 x 轴；
  * 从对应的 CH4 文件中读取通道波形（默认通道 0），找到最大值所在采样点，对应时间记为 t_CH4；
  * 定义 Δt = t_Ge - t_CH4，其中 t_Ge 由系统设置决定 (实验中 t_Ge = 40 μs)。

图像:
- x 轴: CH0 主放最大值 Amax(CH0) (FADC)
- y 轴: Δt = t_Ge - t_CH4 (μs)
"""

import os
import sys
from typing import Optional, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt

# 添加路径以便导入 utils 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.dirname(os.path.dirname(current_dir))
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

from utils.visualize import get_h5_files

def analyze_ch0_max_vs_delta_t(
    h5_file: Optional[str] = None,
    ch0_channel_idx: int = 0,
    ch4_channel_idx: int = 0,
    trigger_threshold: float = 7060.0,
    t_ge_us: float = 40.0,
    max_events_per_file: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show_plot: bool = True,):
    """
    计算 NaI 过阈事件的主放最大值与 Δt，并绘制散点图。

    参数:
        h5_file: 指定单个 CH0-3 HDF5 文件路径；为 None 时自动从 CH0-3 / CH4 目录中按文件名配对并处理所有文件对
        ch0_channel_idx: CH0 主放所在通道索引 (默认 0)
        ch4_channel_idx: CH4 (NaI) 文件中用于取最大值时间的通道索引 (默认 0)
        trigger_threshold: NaI 触发阈值 (FADC)，仅用于判断是否过阈
        t_ge_us: 高纯锗触发时间 t_Ge (单位: μs)，实验中为 40 μs
        max_events_per_file: 每个文件最多处理的事件数；为 None 时处理该文件中的所有事件
        figsize: 图像大小
        save_path: 图片保存路径；为 None 时不保存
        show_plot: 是否显示图像

    返回:
        (ch0_max_values, delta_t_values)，均为 numpy 数组
    """
    # 自动选择 CH0-3 与 CH4 目录中的配对文件
    if h5_file is None:
        h5_files = get_h5_files()
        if "CH0-3" not in h5_files or not h5_files["CH0-3"]:
            raise FileNotFoundError("在 data/hdf5/raw_pulse/CH0-3 目录中未找到 h5 文件")
        if "CH4" not in h5_files or not h5_files["CH4"]:
            raise FileNotFoundError("在 data/hdf5/raw_pulse/CH4 目录中未找到 h5 文件")

        ch03_files = sorted(h5_files["CH0-3"])
        ch4_files = sorted(h5_files["CH4"])
        ch4_map = {os.path.basename(p): p for p in ch4_files}

        paired_files = []
        for fp in ch03_files:
            base = os.path.basename(fp)
            if base in ch4_map:
                paired_files.append((fp, ch4_map[base]))

        if not paired_files:
            raise FileNotFoundError("在 CH0-3 与 CH4 目录中未找到可配对的 h5 文件")

        # 最多处理 20 个文件对
        if len(paired_files) > 20:
            paired_files = paired_files[:20]
    else:
        # 用户指定的是 CH0-3 文件，自动推断对应的 CH4 文件
        if not os.path.exists(h5_file):
            raise FileNotFoundError(f"文件不存在: {h5_file}")

        ch03_file = h5_file
        # 尝试用路径中的 CH0-3 替换为 CH4
        if os.sep + "CH0-3" + os.sep in ch03_file:
            ch4_file = ch03_file.replace(os.sep + "CH0-3" + os.sep, os.sep + "CH4" + os.sep)
        else:
            # 回退策略：在 CH4 目录下按同名文件寻找
            h5_files = get_h5_files()
            if "CH4" not in h5_files or not h5_files["CH4"]:
                raise FileNotFoundError("在 data/hdf5/raw_pulse/CH4 目录中未找到 h5 文件")
            base = os.path.basename(ch03_file)
            ch4_map = {os.path.basename(p): p for p in h5_files["CH4"]}
            if base not in ch4_map:
                raise FileNotFoundError(f"未在 CH4 目录中找到与 {base} 同名的 h5 文件")
            ch4_file = ch4_map[base]

        if not os.path.exists(ch4_file):
            raise FileNotFoundError(f"推断得到的 CH4 文件不存在: {ch4_file}")

        paired_files = [(ch03_file, ch4_file)]

        print("使用指定的 CH0-3 文件及其对应的 CH4 文件进行分析:")
        print(f"  - CH0-3: {os.path.basename(ch03_file)}")
        print(f"  - CH4  : {os.path.basename(ch4_file)}")

    # 采样间隔 (ns)，根据 V1725 采样率假定为 4 ns，与 utils.visualize 中保持一致
    sampling_interval_ns = 4.0

    ch0_max_list = []
    delta_t_list = []
    total_events = 0
    selected_events = 0

    for idx, (ch03_path, ch4_path) in enumerate(paired_files, start=1):
        print("=" * 70)
        print(f"[{idx}/{len(paired_files)}] 分析文件对:")
        print(f"  CH0-3: {os.path.basename(ch03_path)}")
        print(f"  CH4  : {os.path.basename(ch4_path)}")
        print("=" * 70)

        with h5py.File(ch03_path, "r") as f_ch03, h5py.File(ch4_path, "r") as f_ch4:
            if "channel_data" not in f_ch03:
                raise KeyError(f"CH0-3 文件中没有找到 channel_data 数据集: {ch03_path}")
            if "channel_data" not in f_ch4:
                raise KeyError(f"CH4 文件中没有找到 channel_data 数据集: {ch4_path}")

            ch03_data = f_ch03["channel_data"]
            ch4_data = f_ch4["channel_data"]

            t_samples_ch03, n_ch_ch03, n_evt_ch03 = ch03_data.shape
            t_samples_ch4, n_ch_ch4, n_evt_ch4 = ch4_data.shape

            print(f"\nCH0-3 数据维度: (时间采样点数={t_samples_ch03}, 通道数={n_ch_ch03}, 事件数={n_evt_ch03})")
            print(f"CH4   数据维度: (时间采样点数={t_samples_ch4}, 通道数={n_ch_ch4}, 事件数={n_evt_ch4})")

            if ch0_channel_idx < 0 or ch0_channel_idx >= n_ch_ch03:
                raise IndexError(f"CH0-3 通道索引 {ch0_channel_idx} 超出范围 [0, {n_ch_ch03-1}]")
            if ch4_channel_idx < 0 or ch4_channel_idx >= n_ch_ch4:
                raise IndexError(f"CH4 通道索引 {ch4_channel_idx} 超出范围 [0, {n_ch_ch4-1}]")

            if n_evt_ch03 != n_evt_ch4:
                raise ValueError(
                    f"CH0-3 与 CH4 事件数不一致: {n_evt_ch03} vs {n_evt_ch4}，无法逐事件配对"
                )

            if max_events_per_file is not None:
                num_events_to_process = min(n_evt_ch03, max_events_per_file)
            else:
                num_events_to_process = n_evt_ch03

            total_events += num_events_to_process

            print(f"将处理每对文件的前 {num_events_to_process} 个事件。")

            batch_size = 1000
            for start_evt in range(0, num_events_to_process, batch_size):
                end_evt = min(start_evt + batch_size, num_events_to_process)

                # 读取当前批次 CH0 波形 (time_samples, batch_events)
                batch_ch0 = ch03_data[:, ch0_channel_idx, start_evt:end_evt]
                # 读取当前批次 CH4 波形 (time_samples, batch_events)
                batch_ch4 = ch4_data[:, ch4_channel_idx, start_evt:end_evt]

                # CH0: 计算每个事件的最大值
                max_vals_ch0 = np.max(batch_ch0, axis=0)
                # CH4 (NaI): 计算每个事件的最大值及其对应的采样点索引
                max_vals_ch4 = np.max(batch_ch4, axis=0)
                max_indices_ch4 = np.argmax(batch_ch4, axis=0)

                # 只保留 NaI (CH4) 过阈事件
                mask = max_vals_ch4 >= trigger_threshold
                if not np.any(mask):
                    continue

                selected_max_ch0 = max_vals_ch0[mask]
                selected_indices_ch4 = max_indices_ch4[mask]

                # t_NaI (μs) = t_max_sample * 采样间隔 (ns) * 1e-3
                t_nai_us = selected_indices_ch4.astype(np.float64) * sampling_interval_ns * 1e-3

                # Δt = t_Ge - t_NaI (μs)
                delta_t_us = t_ge_us - t_nai_us

                ch0_max_list.append(selected_max_ch0)
                delta_t_list.append(delta_t_us)

                selected_events += selected_max_ch0.size

                if (start_evt // batch_size + 1) % 10 == 0 or end_evt == num_events_to_process:
                    print(
                        f"  已处理 {end_evt}/{num_events_to_process} 个事件，"
                        f"当前累计选中 {selected_events} 个 NaI 过阈事件 (阈值 = {trigger_threshold:.0f} FADC)"
                    )

    if not ch0_max_list:
        raise RuntimeError("在所有文件中均未找到 NaI 过阈事件，请检查阈值设置。")

    ch0_max_values = np.concatenate(ch0_max_list)
    delta_t_values = np.concatenate(delta_t_list)

    print("\n汇总统计信息:")
    print(f"  总处理事件数: {total_events}")
    print(f"  NaI 过阈事件数: {selected_events}")
    print(f"  CH0 最大值范围: [{np.min(ch0_max_values):.2f}, {np.max(ch0_max_values):.2f}] FADC")
    print(f"  Δt 范围: [{np.min(delta_t_values):.3f}, {np.max(delta_t_values):.3f}] μs")

    # 绘制散点图（仅散点，无参考线）
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(ch0_max_values, delta_t_values, s=5, alpha=0.4, edgecolors="none")

    ax.set_xlabel("CH0 maximum amplitude (FADC)", fontsize=12)
    ax.set_ylabel("Δt = t_Ge - t_NaI (μs)", fontsize=12)
    ax.set_title("CH0 maximum amplitude vs NaI Δt", fontsize=13)
    #ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n图片已保存至: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return ch0_max_values, delta_t_values


if __name__ == "__main__":
    print("=" * 70)
    print("CH0 过阈事件主放最大值与 Δt 分布分析")
    print("=" * 70)

    try:
        ch0_max, delta_t = analyze_ch0_max_vs_delta_t(
            h5_file=None,          # 自动从 CH0-3 / CH4 目录按文件名配对并处理所有文件对
            ch0_channel_idx=0,     # CH0 通道索引（在 CH0-3 文件中）
            ch4_channel_idx=0,     # CH4 通道索引（在 CH4 文件中，用于取最大值时间）
            trigger_threshold=7060,
            t_ge_us=40.0,
            max_events_per_file=None,
            show_plot=True,
        )
        print(f"\n分析完成！共选中 {len(ch0_max)} 个 CH0 过阈事件用于绘制散点图。")
    except Exception as e:
        print(f"分析失败: {e}")