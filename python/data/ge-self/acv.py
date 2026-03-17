#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CH0 主放最大值与触发时间差 Δt 关系

更新后的数据来源（更快）：
- 直接使用 data/hdf5/raw_pulse/CH0_parameters 与 CH4_parameters 中的参数文件；
- 对同一 run 的 CH0_parameters 与 CH4_parameters 按文件名一一对应；
- 对每个事件：
  * 从 CH0_parameters 文件中读取 dataset `max_ch0` 作为主放最大值 Amax(CH0)（x 轴）；
  * 从对应的 CH4_parameters 文件中读取 `max_ch4` 与 `tmax_ch4`，
    其中 t_CH4 = tmax_ch4 * 采样间隔 (4 ns)；
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

def _discover_project_root() -> str:
    """
    推断 DeepVibration 项目根目录：
    当前文件位于:
        .../python/data/ge-self/acv.py
    向上到 python，再上一层即项目根目录。
    """
    here = os.path.abspath(__file__)
    ge_self_dir = os.path.dirname(here)           # .../python/data/ge-self
    data_dir = os.path.dirname(ge_self_dir)       # .../python/data
    python_dir = os.path.dirname(data_dir)        # .../python
    project_root = os.path.dirname(python_dir)    # .../DeepVibration
    return project_root


def analyze_ch0_max_vs_delta_t(
    h5_file: Optional[str] = None,
    trigger_threshold: float = 7060.0,
    t_ge_us: float = 40.0,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show_plot: bool = True,):
    """
    基于 CH0_parameters / CH4_parameters 中的预计算参数，
    计算 NaI 过阈事件的主放最大值与 Δt，并绘制散点图。

    参数:
        h5_file: 指定单个 run 的基名对应的参数文件（例如某个 CH0_parameters 文件路径）；
                 为 None 时自动从 CH0_parameters / CH4_parameters 目录中按文件名配对并处理所有文件对
        trigger_threshold: NaI 触发阈值 (FADC)，仅用于判断是否过阈
        t_ge_us: 高纯锗触发时间 t_Ge (单位: μs)，实验中为 40 μs
        max_events_per_file: 每个文件最多处理的事件数；为 None 时处理该文件中的所有事件
        figsize: 图像大小
        save_path: 图片保存路径；为 None 时不保存
        show_plot: 是否显示图像

    返回:
        (ch0_max_values, delta_t_values)，均为 numpy 数组
    """
    project_root = _discover_project_root()
    ch0_param_dir = os.path.join(project_root, "data", "hdf5", "raw_pulse", "CH0_parameters")
    ch4_param_dir = os.path.join(project_root, "data", "hdf5", "raw_pulse", "CH4_parameters")

    if not os.path.isdir(ch0_param_dir):
        raise FileNotFoundError(f"CH0_parameters 目录不存在: {ch0_param_dir}")
    if not os.path.isdir(ch4_param_dir):
        raise FileNotFoundError(f"CH4_parameters 目录不存在: {ch4_param_dir}")

    # 自动选择 CH0_parameters 与 CH4_parameters 目录中的配对文件
    if h5_file is None:
        ch0_files = sorted(
            name for name in os.listdir(ch0_param_dir)
            if name.lower().endswith((".h5", ".hdf5"))
        )
        if not ch0_files:
            raise FileNotFoundError(f"在 {ch0_param_dir} 中未找到任何 h5 参数文件")

        ch4_existing = {name for name in os.listdir(ch4_param_dir)
                        if name.lower().endswith((".h5", ".hdf5"))}

        paired_files = []
        for name in ch0_files:
            if name in ch4_existing:
                paired_files.append(
                    (os.path.join(ch0_param_dir, name), os.path.join(ch4_param_dir, name))
                )

        if not paired_files:
            raise FileNotFoundError("在 CH0_parameters 与 CH4_parameters 目录中未找到可配对的参数文件")

        # 最多处理 50 个文件对，避免一次性读入过多数据
        if len(paired_files) > 50:
            print(f"找到 {len(paired_files)} 个参数文件对，仅使用前 50 个。")
            paired_files = paired_files[:50]
    else:
        # 用户可以传入一个 CH0_parameters 文件路径，或者任意包含该文件名的路径
        if not os.path.exists(h5_file):
            raise FileNotFoundError(f"文件不存在: {h5_file}")

        base = os.path.basename(h5_file)
        ch0_file = os.path.join(ch0_param_dir, base)
        ch4_file = os.path.join(ch4_param_dir, base)

        if not os.path.exists(ch0_file):
            raise FileNotFoundError(f"未在 CH0_parameters 中找到文件: {ch0_file}")
        if not os.path.exists(ch4_file):
            raise FileNotFoundError(f"未在 CH4_parameters 中找到文件: {ch4_file}")

        paired_files = [(ch0_file, ch4_file)]

        print("使用指定 run 对应的参数文件进行分析:")
        print(f"  - CH0_parameters: {os.path.basename(ch0_file)}")
        print(f"  - CH4_parameters: {os.path.basename(ch4_file)}")

    # 采样间隔 (ns)，根据 V1725 采样率假定为 4 ns，与 preprocessor / utils.visualize 中保持一致
    sampling_interval_ns = 4.0

    ch0_max_list = []
    delta_t_list = []
    total_events = 0          # 所有 run 的总事件数
    selected_events = 0       # 通过 NaI 触发阈值筛选的事件数

    for idx, (ch0_param_path, ch4_param_path) in enumerate(paired_files, start=1):
        print("=" * 70)
        print(f"[{idx}/{len(paired_files)}] 分析参数文件对:")
        print(f"  CH0_parameters: {os.path.basename(ch0_param_path)}")
        print(f"  CH4_parameters: {os.path.basename(ch4_param_path)}")
        print("=" * 70)

        with h5py.File(ch0_param_path, "r") as f_ch0, h5py.File(ch4_param_path, "r") as f_ch4:
            if "max_ch0" not in f_ch0:
                raise KeyError(f"CH0_parameters 中没有找到 max_ch0 数据集: {ch0_param_path}")
            if "max_ch4" not in f_ch4 or "tmax_ch4" not in f_ch4:
                raise KeyError(
                    f"CH4_parameters 中缺少 max_ch4 或 tmax_ch4 数据集: {ch4_param_path}"
                )

            max_ch0 = np.asarray(f_ch0["max_ch0"][...], dtype=np.float64)
            max_ch4 = np.asarray(f_ch4["max_ch4"][...], dtype=np.float64)
            tmax_ch4 = np.asarray(f_ch4["tmax_ch4"][...], dtype=np.int64)

            if max_ch0.ndim != 1 or max_ch4.ndim != 1 or tmax_ch4.ndim != 1:
                raise ValueError(
                    f"{os.path.basename(ch0_param_path)} 或 {os.path.basename(ch4_param_path)} "
                    "中的 max_ch0/max_ch4/tmax_ch4 不是一维数组。"
                )

            n0, n4, nt = max_ch0.shape[0], max_ch4.shape[0], tmax_ch4.shape[0]
            n = min(n0, n4, nt)
            if n0 != n or n4 != n or nt != n:
                print(
                    f"[警告] 事件数不一致 (CH0={n0}, CH4_max={n4}, CH4_tmax={nt})，"
                    f"仅使用前 {n} 个事件。"
                )
                max_ch0 = max_ch0[:n]
                max_ch4 = max_ch4[:n]
                tmax_ch4 = tmax_ch4[:n]

            total_events += n

            # 只保留 NaI (CH4) 过阈事件
            mask = max_ch4 >= trigger_threshold
            if not np.any(mask):
                print(f"[信息] 本 run 中无 NaI 过阈事件，跳过。")
                continue

            selected_max_ch0 = max_ch0[mask]
            selected_tmax_ch4 = tmax_ch4[mask]

            # t_NaI (μs) = t_max_sample * 采样间隔 (ns) * 1e-3
            t_nai_us = selected_tmax_ch4.astype(np.float64) * sampling_interval_ns * 1e-3

            # Δt = t_Ge - t_NaI (μs)
            delta_t_us = t_ge_us - t_nai_us

            ch0_max_list.append(selected_max_ch0)
            delta_t_list.append(delta_t_us)

            selected_events += selected_max_ch0.size

            print(
                f"  本 run 事件数: {n}，NaI 过阈事件数: {selected_max_ch0.size} "
                f"(阈值 = {trigger_threshold:.0f} FADC)"
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

    # -------- 按项目统一绘图风格绘制散点图 --------
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
    })

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(ch0_max_values, delta_t_values, s=5, alpha=0.4, edgecolors="none")

    ax.set_xlabel("CH0 maximum amplitude (FADC)", fontsize=16)
    ax.set_ylabel("Δt = t_Ge - t_NaI (μs)", fontsize=16)
    ax.set_title("CH0 maximum amplitude vs NaI Δt", fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_xlim(0, 16382)

    # 在 Δt = 1 μs 和 16 μs 处画两条参考横线
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5)
    ax.axhline(16.0, color="green", linestyle="--", linewidth=1.5)

    # 可根据需要打开网格
    # ax.grid(True, alpha=0.3)

    fig.tight_layout()

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
            h5_file=None,          # 自动从 CH0_parameters / CH4_parameters 目录按文件名配对并处理所有文件对
            trigger_threshold=7060,
            t_ge_us=40.0,
            show_plot=True,
        )
        print(f"\n分析完成！共选中 {len(ch0_max)} 个 CH0 过阈事件用于绘制散点图。")
    except Exception as e:
        print(f"分析失败: {e}")