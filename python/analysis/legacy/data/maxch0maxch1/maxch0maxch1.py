#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
针对所有 run，直接从参数目录中读取 CH0/CH1 的最大值（max_ch0 / max_ch1），
并绘制统一的 PN-cut 散点图：max(CH0) vs max(CH1)。

相比从 CH0-3 原始波形 h5 中逐事件计算最大值，这里直接使用
`preprocessor.py` 已经写好的参数文件：
    - data/hdf5/raw_pulse/CH0_parameters  中的 dataset 'max_ch0'
    - data/hdf5/raw_pulse/CH1_parameters  中的 dataset 'max_ch1'

这样既更快，也与其他分析脚本的参数来源保持一致。
"""

import os
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt


def _discover_project_root() -> str:
    """
    推断 DeepVibration 项目根目录：
    当前文件位于:
        .../python/data/maxch0maxch1/maxch0maxch1.py
    向上到 python，再上一层即项目根目录。
    """
    here = os.path.abspath(__file__)
    pncut_dir = os.path.dirname(here)              # .../python/data/maxch0maxch1
    data_dir = os.path.dirname(pncut_dir)          # .../python/data
    python_dir = os.path.dirname(data_dir)         # .../python
    project_root = os.path.dirname(python_dir)     # .../DeepVibration
    return project_root


def _collect_max_from_param_dirs(
    ch0_param_dir: str,
    ch1_param_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    从 CH0_parameters / CH1_parameters 目录中，读取所有同时存在的 run，
    直接提取每个事件的 max_ch0 / max_ch1，并汇总返回。
    """
    if not os.path.isdir(ch0_param_dir):
        raise FileNotFoundError(f"CH0_parameters 目录不存在: {ch0_param_dir}")
    if not os.path.isdir(ch1_param_dir):
        raise FileNotFoundError(f"CH1_parameters 目录不存在: {ch1_param_dir}")

    ch0_files = sorted(
        name
        for name in os.listdir(ch0_param_dir)
        if name.lower().endswith((".h5", ".hdf5"))
    )
    if not ch0_files:
        raise FileNotFoundError(f"CH0_parameters 目录下未找到任何 h5 文件: {ch0_param_dir}")

    all_max_ch0: list[np.ndarray] = []
    all_max_ch1: list[np.ndarray] = []
    n_runs_used = 0
    total_events = 0

    for name in ch0_files:
        ch0_path = os.path.join(ch0_param_dir, name)
        ch1_path = os.path.join(ch1_param_dir, name)
        if not os.path.isfile(ch1_path):
            print(f"[信息] CH1 参数文件缺失，跳过该 run: {name}")
            continue

        with h5py.File(ch0_path, "r") as f0, h5py.File(ch1_path, "r") as f1:
            if "max_ch0" not in f0 or "max_ch1" not in f1:
                print(f"[警告] {name} 中缺少 max_ch0 或 max_ch1，跳过该 run。")
                continue

            max_ch0 = np.asarray(f0["max_ch0"][...], dtype=np.float64)
            max_ch1 = np.asarray(f1["max_ch1"][...], dtype=np.float64)

            if max_ch0.ndim != 1 or max_ch1.ndim != 1:
                print(f"[警告] {name} 中 max_ch0/max_ch1 不是一维数组，跳过该 run。")
                continue

            n0, n1 = max_ch0.shape[0], max_ch1.shape[0]
            if n0 != n1:
                n = min(n0, n1)
                print(
                    f"[警告] {name} 中 CH0/CH1 事件数不一致 ({n0} vs {n1})，"
                    f"仅使用前 {n} 个事件。"
                )
                max_ch0 = max_ch0[:n]
                max_ch1 = max_ch1[:n]
            else:
                n = n0

        all_max_ch0.append(max_ch0)
        all_max_ch1.append(max_ch1)
        n_runs_used += 1
        total_events += n
        print(f"[读取] {name} | 事件数: {n}")

    if not all_max_ch0:
        raise RuntimeError("未能从任何 run 中收集到 max_ch0/max_ch1 参数，无法绘图。")

    max_ch0_all = np.concatenate(all_max_ch0)
    max_ch1_all = np.concatenate(all_max_ch1)

    print(f"总共使用 {n_runs_used} 个 run，汇总事件数: {total_events}")

    return max_ch0_all, max_ch1_all


def plot_pncut_from_parameters(
    ch0_param_dir: str,
    ch1_param_dir: str,
    save_path: str | None = None,
    show: bool = True,
) -> str:
    """
    直接从 CH0_parameters / CH1_parameters 中的 max_ch0 / max_ch1 绘制 PN-cut 散点图。
    """
    max_ch0_all, max_ch1_all = _collect_max_from_param_dirs(ch0_param_dir, ch1_param_dir)

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
        "PN-cut Scatter: max CH0 vs max CH1 (from parameters)",
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
        project_root = _discover_project_root()
        output_dir = os.path.join(project_root, "images", "presentation")
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "pncut_scatter_CH0_CH1_parameters.png")

    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"PN-cut 散点图（参数文件）已保存至: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="基于 CH0_parameters / CH1_parameters 中的 max_ch0 / max_ch1 绘制 PN-cut 散点图。"
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

    project_root = _discover_project_root()
    ch0_param_dir = os.path.join(project_root, "data", "hdf5", "raw_pulse", "CH0_parameters")
    ch1_param_dir = os.path.join(project_root, "data", "hdf5", "raw_pulse", "CH1_parameters")

    if not os.path.isdir(ch0_param_dir):
        raise FileNotFoundError(f"CH0_parameters 目录不存在: {ch0_param_dir}")
    if not os.path.isdir(ch1_param_dir):
        raise FileNotFoundError(f"CH1_parameters 目录不存在: {ch1_param_dir}")

    plot_pncut_from_parameters(
        ch0_param_dir=ch0_param_dir,
        ch1_param_dir=ch1_param_dir,
        save_path=args.save_path,
        show=True,
    )
      

