#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对 HDF5 参数文件中的事件绘制 PN-cut 散点图（不读原始波形）：
- 从 data/hdf5/raw_pulse/CH0_parameters 读 max_ch0、ch0_min；
- 从 CH1_parameters 读 max_ch1；
- 从 CH5_parameters 读 max_ch5（用于 RT 判据）。
数据集命名与 preprocessor.py 一致。

批量模式：遍历 CH0_parameters 下所有 .h5，与同名的 CH1/CH5 参数文件对齐后汇总绘图（显示全部点）。
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _project_root() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))


def _param_paths(project_root: str) -> Tuple[str, str, str]:
    root = os.path.join(project_root, "data", "hdf5", "raw_pulse")
    return (
        os.path.join(root, "CH0_parameters"),
        os.path.join(root, "CH1_parameters"),
        os.path.join(root, "CH5_parameters"),
    )


def load_pncut_arrays_from_params(
    project_root: str,
    base_name: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    从 raw_pulse 下各 *parameters 目录读取与 base_name 同名的 h5。
    返回 (max_ch0, max_ch1, ch0_min, max_ch5)，长度对齐为各数组最小长度；
    若缺少 CH0/CH1 参数文件或必要数据集则返回 None；
    若缺少 CH5 参数文件，则 max_ch5 全 0（与原先不读 CH5 波形时的 RT 行为一致）。
    """
    ch0_dir, ch1_dir, ch5_dir = _param_paths(project_root)
    ch0_path = os.path.join(ch0_dir, base_name)
    ch1_path = os.path.join(ch1_dir, base_name)
    ch5_path = os.path.join(ch5_dir, base_name)

    if not os.path.isfile(ch0_path) or not os.path.isfile(ch1_path):
        return None

    with h5py.File(ch0_path, "r") as f0:
        if "max_ch0" not in f0 or "ch0_min" not in f0:
            return None
        max_ch0 = np.asarray(f0["max_ch0"][...], dtype=np.float64)
        ch0_min = np.asarray(f0["ch0_min"][...], dtype=np.float64)

    with h5py.File(ch1_path, "r") as f1:
        if "max_ch1" not in f1:
            return None
        max_ch1 = np.asarray(f1["max_ch1"][...], dtype=np.float64)

    n = min(max_ch0.size, max_ch1.size, ch0_min.size)
    max_ch0 = max_ch0[:n]
    max_ch1 = max_ch1[:n]
    ch0_min = ch0_min[:n]

    if os.path.isfile(ch5_path):
        with h5py.File(ch5_path, "r") as f5:
            if "max_ch5" not in f5:
                max_ch5 = np.zeros(n, dtype=np.float64)
            else:
                max_ch5 = np.asarray(f5["max_ch5"][...], dtype=np.float64)
                n5 = min(n, max_ch5.size)
                max_ch0 = max_ch0[:n5]
                max_ch1 = max_ch1[:n5]
                ch0_min = ch0_min[:n5]
                max_ch5 = max_ch5[:n5]
    else:
        max_ch5 = np.zeros(n, dtype=np.float64)

    return max_ch0, max_ch1, ch0_min, max_ch5


def load_pncut_arrays_simple_from_params(
    project_root: str,
    base_name: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """仅 max_ch0 / max_ch1，用于单文件无分类散点图。"""
    ch0_dir, ch1_dir, _ = _param_paths(project_root)
    ch0_path = os.path.join(ch0_dir, base_name)
    ch1_path = os.path.join(ch1_dir, base_name)
    if not os.path.isfile(ch0_path) or not os.path.isfile(ch1_path):
        return None
    with h5py.File(ch0_path, "r") as f0:
        if "max_ch0" not in f0:
            return None
        max_ch0 = np.asarray(f0["max_ch0"][...], dtype=np.float64)
    with h5py.File(ch1_path, "r") as f1:
        if "max_ch1" not in f1:
            return None
        max_ch1 = np.asarray(f1["max_ch1"][...], dtype=np.float64)
    n = min(max_ch0.size, max_ch1.size)
    return max_ch0[:n], max_ch1[:n]


def plot_pncut_scatter(
    h5_path: str,
    ch0_idx: int = 0,
    ch1_idx: int = 1,
    save_path: str | None = None,
    show: bool = True,
) -> str:
    """
    对指定「运行」绘制 max(CH0) vs max(CH1) 散点图。
    h5_path 可为 CH0-3 原始脉冲文件路径或任意同名 base 的路径；实际从 CH0_parameters / CH1_parameters 读取。
    ch0_idx / ch1_idx 保留与旧接口兼容，此处忽略（通道由参数文件固定）。
    """
    del ch0_idx, ch1_idx
    base_name = os.path.basename(h5_path)
    project_root = _project_root()
    loaded = load_pncut_arrays_simple_from_params(project_root, base_name)
    if loaded is None:
        raise FileNotFoundError(
            f"未找到或未完整读取参数文件（需 CH0_parameters 与 CH1_parameters 下同名 {base_name}，"
            "且含数据集 max_ch0、max_ch1）"
        )
    max_ch0, max_ch1 = loaded
    num_events = max_ch0.size

    print(f"参数文件 basename: {base_name}")
    print(f"事件数: {num_events}")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
        }
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        max_ch0,
        max_ch1,
        s=4,
        alpha=0.6,
        c="tab:blue",
        edgecolors="none",
    )

    ax.set_xlabel("max CH0 (ADC)", fontsize=14, fontweight="bold")
    ax.set_ylabel("max CH1 (ADC)", fontsize=14, fontweight="bold")
    ax.set_title(
        "PN-cut Scatter: max CH0 vs max CH1 (from parameters)",
        fontsize=10,
        fontweight="bold",
    )

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontweight("bold")
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontweight("bold")

    plt.tight_layout()

    if save_path is None:
        pncut_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.dirname(pncut_dir)
        python_dir = os.path.dirname(data_dir)
        project_root = os.path.dirname(python_dir)
        output_dir = os.path.join(project_root, "images", "presentation")
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"pncut_scatter_{base_name}.png")

    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"PN-cut 散点图已保存至: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


def plot_pncut_scatter_multi(
    base_names: Optional[list[str]] = None,
    project_root: Optional[str] = None,
    save_path: str | None = None,
    show: bool = True,
) -> str:
    """
    将多个运行中的事件汇总到一张 PN-cut 散点图上，
    根据 RT / Inhibit / ge-self 条件着色（参数来自 HDF5，不读波形）：
    - RT:   max_ch5 > 6000（蓝色）
    - Inhibit: ch0_min == 0（黑色）
    - ge-self: 既非 RT 也非 Inhibit（红色）

    base_names: 若为 None，则处理 CH0_parameters 目录下全部 .h5 文件名；
    否则仅处理给定列表（不含路径，仅 basename）。
    """
    if project_root is None:
        project_root = _project_root()
    ch0_dir, _, _ = _param_paths(project_root)

    if base_names is None:
        if not os.path.isdir(ch0_dir):
            raise FileNotFoundError(f"目录不存在: {ch0_dir}")
        base_names = sorted(
            n for n in os.listdir(ch0_dir) if n.lower().endswith(".h5")
        )
    if not base_names:
        raise ValueError("未指定任何参数文件（base_names 为空）。")

    rt_cut = 6000.0
    rt_x, rt_y = [], []
    inhibit_x, inhibit_y = [], []
    geself_x, geself_y = [], []
    total_events = 0

    for base_name in base_names:
        loaded = load_pncut_arrays_from_params(project_root, base_name)
        if loaded is None:
            print(f"[警告] 跳过（缺 CH0/CH1 参数或数据集不全）: {base_name}")
            continue
        max_ch0, max_ch1, ch0_min, max_ch5 = loaded
        num_events = max_ch0.size
        if num_events == 0:
            continue

        rt_mask = max_ch5 > rt_cut
        inhibit_mask = ch0_min == 0
        geself_mask = (~rt_mask) & (~inhibit_mask)

        rt_x.append(max_ch0[rt_mask])
        rt_y.append(max_ch1[rt_mask])
        inhibit_x.append(max_ch0[inhibit_mask])
        inhibit_y.append(max_ch1[inhibit_mask])
        geself_x.append(max_ch0[geself_mask])
        geself_y.append(max_ch1[geself_mask])

        total_events += num_events
        print(
            f"参数文件: {base_name} | 事件数: {num_events}，"
            f"RT: {rt_mask.sum()}，Inhibit: {inhibit_mask.sum()}，ge-self: {geself_mask.sum()}"
        )

    if total_events == 0:
        raise RuntimeError("没有任何有效的参数文件被成功读取，无法绘图。")

    def _concat(arrs: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(arrs) if arrs else np.array([], dtype=np.float64)

    rt_x_all = _concat(rt_x)
    rt_y_all = _concat(rt_y)
    inh_x_all = _concat(inhibit_x)
    inh_y_all = _concat(inhibit_y)
    ges_x_all = _concat(geself_x)
    ges_y_all = _concat(geself_y)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
        }
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    if rt_x_all.size > 0:
        ax.scatter(rt_x_all, rt_y_all, s=2, alpha=0.6, c="tab:blue", edgecolors="none", label="RT")
    if inh_x_all.size > 0:
        ax.scatter(inh_x_all, inh_y_all, s=2, alpha=0.6, c="black", edgecolors="none", label="Inhibit")
    if ges_x_all.size > 0:
        ax.scatter(ges_x_all, ges_y_all, s=2, alpha=0.6, c="tab:red", edgecolors="none", label="ge-self")

    ax.set_xlabel("max CH0 (ADC)", fontsize=14, fontweight="bold")
    ax.set_ylabel("max CH1 (ADC)", fontsize=14, fontweight="bold")
    ax.set_title(
        "PN-cut Scatter: max CH0 vs max CH1\n(RT / Inhibit / ge-self, from parameters)",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontweight("bold")
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontweight("bold")

    if any([rt_x_all.size > 0, inh_x_all.size > 0, ges_x_all.size > 0]):
        ax.legend(loc="best", fontsize=9, markerscale=3)

    plt.tight_layout()

    if save_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pr = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
        output_dir = os.path.join(pr, "images", "presentation")
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "pncut_scatter_CH0-3_all.png")

    # fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    # print(f"已保存至: {save_path}（总事件数 {total_events}）")

    if show:
        ax.set_xlim(0, 16382)
        ax.set_ylim(0, 16382)
        plt.show()
    else:
        plt.close(fig)

    return save_path


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "从 data/hdf5/raw_pulse/*_parameters 读取 per-event 参数，"
            "绘制 max(CH0) vs max(CH1) PN-cut 散点图（不读取 channel_data 波形）。"
        )
    )
    parser.add_argument(
        "--file",
        "-f",
        dest="h5_path",
        required=False,
        default=None,
        help=(
            "单个运行的 h5 文件名或路径（取 basename）；"
            "从 CH0_parameters / CH1_parameters 读 max_ch0、max_ch1。"
            "若不提供，则批量读取 CH0_parameters 下全部 .h5 并汇总为 RT/Inhibit/ge-self 分类图。"
        ),
    )
    parser.add_argument(
        "--ch0-idx",
        type=int,
        default=0,
        help="保留兼容，忽略（通道由参数文件固定）。",
    )
    parser.add_argument(
        "--ch1-idx",
        type=int,
        default=1,
        help="保留兼容，忽略。",
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

    if args.h5_path is None:
        plot_pncut_scatter_multi(
            base_names=None,
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
