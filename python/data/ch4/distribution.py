#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CH4 通道信号最大值分布分析
分析 data/hdf5/raw_pulse/CH4 目录中事件波形最大值的分布
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


def analyze_ch4_max_distribution(
    h5_file: Optional[str] = None,
    channel_idx: int = 0,
    bins: int = 100,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    trigger_threshold: Optional[float] = None,
    max_files: int = 20,
) -> Tuple[np.ndarray, dict]:
    """
    分析 CH4 文件中所有事件波形最大值的分布。

    参数:
        h5_file: HDF5 文件路径，如果为 None 则自动获取 CH4 目录中的多个文件（最多 max_files 个）并合并分析
        channel_idx: 通道索引（CH4 目录中通常只有通道 0）
        bins: 直方图的 bins 数量
        save_path: 保存图片的路径，如果为 None 则不保存
        show_plot: 是否显示图片
        figsize: 图片大小 (宽度, 高度)
        trigger_threshold: 触发阈值（FADC），用于在图中画出竖线
        max_files: 当 h5_file 为 None 时，最多自动选择的文件数

    返回:
        (max_values 数组, histogram 统计结果) 的元组
    """
    # 如果没有指定文件，自动获取 CH4 目录中的多个文件（最多 max_files 个）
    if h5_file is None:
        h5_files = get_h5_files()
        if "CH4" not in h5_files or not h5_files["CH4"]:
            raise FileNotFoundError("在 data/hdf5/raw_pulse/CH4 目录中未找到 h5 文件")
        all_files = h5_files["CH4"][:max_files]
        if not all_files:
            raise FileNotFoundError("在 data/hdf5/raw_pulse/CH4 目录中未找到可用的 h5 文件")
        print(f"自动选择 {len(all_files)} 个文件 (最多 {max_files} 个):")
        for fp in all_files:
            print(f"  - {os.path.basename(fp)}")
    else:
        if not os.path.exists(h5_file):
            raise FileNotFoundError(f"文件不存在: {h5_file}")
        all_files = [h5_file]
        print("使用指定文件进行分析:")
        print(f"  - {os.path.basename(h5_file)}")

    try:
        all_max_values = []
        total_events = 0
        total_files = len(all_files)

        for idx, file_path in enumerate(all_files, start=1):
            print("=" * 70)
            print(f"[{idx}/{total_files}] 分析文件: {os.path.basename(file_path)}")
            print(f"文件路径: {file_path}")
            print("=" * 70)

            with h5py.File(file_path, "r") as f:
                if "channel_data" not in f:
                    raise KeyError("文件中没有找到 channel_data 数据集")

                channel_data = f["channel_data"]
                time_samples, num_channels, num_events = channel_data.shape

                print(f"\n数据维度: (时间采样点数={time_samples}, 通道数={num_channels}, 事件数={num_events})")

                if channel_idx < 0 or channel_idx >= num_channels:
                    raise IndexError(f"通道索引 {channel_idx} 超出范围 [0, {num_channels-1}]")

                # 提取所有事件的波形最大值
                print("\n正在计算所有事件的最大值...")
                file_max_values = np.zeros(num_events, dtype=np.float64)

                batch_size = 1000
                for i in range(0, num_events, batch_size):
                    end_idx = min(i + batch_size, num_events)
                    batch_data = channel_data[:, channel_idx, i:end_idx]
                    batch_max = np.max(batch_data, axis=0)
                    file_max_values[i:end_idx] = batch_max
                    if (i // batch_size + 1) % 10 == 0 or end_idx == num_events:
                        print(f"  已处理 {end_idx}/{num_events} 个事件 ({end_idx/num_events*100:.1f}%)")

                all_max_values.append(file_max_values)
                total_events += num_events

        if not all_max_values:
            raise RuntimeError("没有从任何文件中读取到最大值数据。")

        max_values = np.concatenate(all_max_values)

        print("\n汇总统计信息:")
        print(f"  文件数: {total_files}")
        print(f"  总事件数: {total_events}")
        print(f"  最大值范围: [{np.min(max_values):.2f}, {np.max(max_values):.2f}]")
        print(f"  平均值: {np.mean(max_values):.2f}")
        print(f"  中位数: {np.median(max_values):.2f}")
        print(f"  标准差: {np.std(max_values):.2f}")

        fig, ax = plt.subplots(figsize=figsize)
        n, bins_edges, patches = ax.hist(max_values, bins=bins, edgecolor="black", alpha=0.7)

        if trigger_threshold is not None:
            ax.axvline(
                trigger_threshold,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Trigger = {trigger_threshold:.0f} FADC",
            )

        mean_val = np.mean(max_values)
        median_val = np.median(max_values)
        std_val = np.std(max_values)

        stats_text = (
            f"Total Events: {total_events}\n"
            f"Mean: {mean_val:.2f}\n"
            f"Median: {median_val:.2f}\n"
            f"Std: {std_val:.2f}\n"
            f"Min: {np.min(max_values):.2f}\n"
            f"Max: {np.max(max_values):.2f}"
        )

        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=10,
        )

        ax.set_xlabel("Maximum Amplitude (ADC counts)", fontsize=12)
        ax.set_ylabel("Number of Events", fontsize=12)
        if trigger_threshold is not None:
            ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"\n图片已保存至: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        hist_stats = {
            "counts": n,
            "bins": bins_edges,
            "mean": mean_val,
            "median": median_val,
            "std": std_val,
            "min": np.min(max_values),
            "max": np.max(max_values),
        }

        return max_values, hist_stats

    except Exception as e:
        print(f"分析过程中出错: {e}")
        raise


if __name__ == "__main__":
    print("=" * 70)
    print("分析 CH4 单个文件")
    print("=" * 70)
    try:
        max_values, stats = analyze_ch4_max_distribution(
            h5_file=None,  # 自动选择 CH4 目录中的第一个文件
            bins=1000,
            show_plot=True,
            trigger_threshold=7060,
        )
        print(f"\n分析完成！总事件数: {stats['counts'].sum()}")
    except Exception as e:
        print(f"分析失败: {e}")
