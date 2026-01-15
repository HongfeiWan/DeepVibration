#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
振动传感器频率分布分析脚本

功能：
- 复用 `select.py` 中的数据读取与筛选函数
- 对每个传感器的三个轴（Frequency_x / y / z）做频率分布统计
- 将各轴分布绘制为直方图（可选叠加核密度曲线）
"""

import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 配置路径：将项目中的 python 目录加入 sys.path，方便作为脚本直接运行
current_dir = os.path.dirname(os.path.abspath(__file__))
# 当前在 .../python/data/sensor/vibration/frequency
# 向上 4 层到达 .../python 目录
python_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
)
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

# 复用数据读取与筛选函数
from data.sensor.vibration.frequency.select import select_by_date_range_vibration


def plot_frequency_distribution_per_detector(
    data_dict: Dict[str, np.ndarray],
    bins: int = 80,
    save_dir: Optional[str] = None,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (14, 8),) -> None:
    """
    根据 select.py 读取的结果，按探测器绘制三个轴的频率分布直方图。

    参数:
        data_dict: select_by_date_range_vibration 的返回值
        bins: 直方图的 bin 数
        save_dir: 图像保存目录；为 None 时不保存
        show_plot: 是否显示图像
        figsize: 单个探测器图像的尺寸
    """
    if not data_dict:
        print("警告：数据为空，无法绘制分布图")
        return

    if "Frequency_x" not in data_dict and "Frequency_y" not in data_dict and "Frequency_z" not in data_dict:
        print("警告：数据中没有 Frequency_x / y / z 列，无法绘制分布图")
        return

    freq_cols = [c for c in ["Frequency_x", "Frequency_y", "Frequency_z"] if c in data_dict]
    if not freq_cols:
        print("警告：没有可用的频率列")
        return

    has_detector = "detector_num" in data_dict

    # 统一的绘图风格
    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "axes.linewidth": 1.2,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.minor.width": 0.8,
            "ytick.minor.width": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.fontsize": 10,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "figure.dpi": 100,
        }
    )

    freq_colors = {"Frequency_x": "#2E86AB", "Frequency_y": "#A23B72", "Frequency_z": "#F18F01"}

    if has_detector:
        detector_nums = np.unique(data_dict["detector_num"])
    else:
        detector_nums = [None]  # 视为单个“虚拟探测器”

    for det in detector_nums:
        if has_detector:
            det_mask = data_dict["detector_num"] == det
            title_prefix = f"Detector {det}"
        else:
            det_mask = slice(None)
            title_prefix = "All data"

        # 为该探测器创建一张图，3 行 1 列子图
        fig, axes = plt.subplots(len(freq_cols), 1, figsize=figsize, sharex=False)
        if len(freq_cols) == 1:
            axes = [axes]

        for ax, col in zip(axes, freq_cols):
            values = data_dict[col][det_mask]
            values = values[~np.isnan(values)]

            if values.size == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No valid data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{title_prefix} - {col} (no data)")
                continue

            # 计算统计量
            v_min, v_max = np.min(values), np.max(values)
            v_mean, v_std = np.mean(values), np.std(values)

            # 画直方图
            color = freq_colors.get(col, "#2E86AB")
            ax.hist(
                values,
                bins=bins,
                density=True,
                alpha=0.75,
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )

            # 也可以叠加一个高斯近似（可选）
            try:
                from scipy.stats import norm  # 若环境中无 scipy，会抛异常

                x_grid = np.linspace(v_min, v_max, 400)
                pdf = norm.pdf(x_grid, v_mean, v_std if v_std > 0 else 1.0)
                ax.plot(x_grid, pdf, "r--", linewidth=1.2, label="Gaussian fit")
                ax.legend(loc="upper right", framealpha=0.9)
            except Exception:
                # 没有 scipy 就只画直方图
                pass

            ax.set_xlabel(f"{col} (Hz)")
            ax.set_ylabel("Density")
            ax.set_title(f"{title_prefix} - {col} distribution")

            # y 轴格式与网格
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
            ax.yaxis.get_major_formatter().set_scientific(False)
            ax.grid(True, which="major", linestyle="-", linewidth=0.7, alpha=0.3, color="gray")
            ax.grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.2, color="gray")

            # 统计信息框
            stats_text = (
                f"N = {values.size}\n"
                f"Min = {v_min:.3f} Hz\n"
                f"Max = {v_max:.3f} Hz\n"
                f"Mean = {v_mean:.3f} Hz\n"
                f"Std = {v_std:.3f} Hz"
            )
            ax.text(
                0.98,
                0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="white",
                    edgecolor="gray",
                    alpha=0.8,
                    linewidth=0.8,
                ),
                fontsize=9,
                family="monospace",
            )

        plt.tight_layout()

        # 保存
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            if has_detector:
                fname = f"frequency_distribution_detector_{det}.png"
            else:
                fname = "frequency_distribution_all.png"
            save_path = os.path.join(save_dir, fname)
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"频率分布图已保存至: {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)


if __name__ == "__main__":
    """
    示例：
    1. 先用 select_by_date_range_vibration 读取/筛选数据
    2. 再按探测器绘制三个轴的频率分布
    """
    # 当前脚本所在目录：python/data/sensor/vibration/frequency
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 向上 5 层到项目根目录
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
    )
    data_dir = os.path.join(project_root, "data", "vibration", "hdf5")

    print("=" * 70)
    print("振动传感器频率分布分析示例")
    print("=" * 70)

    try:
        # 这里的日期范围、探测器编号可以根据需要修改
        data_multi = select_by_date_range_vibration(
            data_dir,
            detector_num=[1, 2, 3, 4, 5],
            start_date="2025-05-28",
            end_date="2025-06-10",
            downsample_factor=100,
        )

        if not data_multi:
            print("未能读取到任何数据，无法绘制分布")
        else:
            print("开始绘制每个传感器三个轴的频率分布直方图...")
            # 若不想保存，只需将 save_dir 设为 None
            out_dir = os.path.join(project_root, "imgaes", "frequency_distribution")
            plot_frequency_distribution_per_detector(
                data_multi,
                bins=80,
                save_dir=out_dir,
                show_plot=True,
                figsize=(12, 9),
            )

    except Exception as e:
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()

