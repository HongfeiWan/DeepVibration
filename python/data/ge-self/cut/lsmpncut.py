#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lsmpncut 简化版工具函数

只对给定二维散点 (x, y) 中 **2000 < x < 14000** 的点，
使用普通最小二乘法拟合一条直线 y = a * x + b。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def fit_single_line_in_range(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float = 2000.0,
    x_max: float = 14000.0,
) -> Tuple[float, float]:
    """
    在区间 (x_min, x_max) 内，使用最小二乘法拟合一条直线 y = a * x + b。
    第一次拟合后，剔除残差过大的“远点”，再做一次最小二乘拟合。

    参数
    ----
    x, y:
        一维数组，长度为 N，对应 N 个散点。
    x_min, x_max:
        只使用满足 x_min < x < x_max 的点参与拟合。

    返回
    ----
    a, b:
        拟合得到的直线参数，使得 y ≈ a * x + b。
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    if x.shape != y.shape:
        raise ValueError(f"x 与 y 形状不一致: {x.shape} vs {y.shape}")

    # 只保留 x 在指定区间内的点
    mask_range = (x > x_min) & (x < x_max)
    n_range = int(mask_range.sum())
    if n_range < 2:
        raise ValueError(
            f"在区间 ({x_min}, {x_max}) 内有效点数不足 2 个，无法拟合直线 "
            f"(有效点数={n_range})"
        )

    x_sel = x[mask_range]
    y_sel = y[mask_range]

    # 第一次普通最小二乘线性拟合
    a1, b1 = np.polyfit(x_sel, y_sel, deg=1)

    # 计算残差，并按 3σ 规则剔除“太远”的点，然后再拟合一次
    y_pred = a1 * x_sel + b1
    residuals = y_sel - y_pred
    sigma = residuals.std(ddof=1) if residuals.size > 1 else 0.0

    if sigma > 0.0:
        inlier_mask = np.abs(residuals) <= 3.0 * sigma
        n_inliers = int(inlier_mask.sum())

        # 只有在有足够多的内点、且确实剔除了部分点时才进行第二次拟合
        if 2 <= n_inliers < x_sel.size:
            x_in = x_sel[inlier_mask]
            y_in = y_sel[inlier_mask]
            a2, b2 = np.polyfit(x_in, y_in, deg=1)
            return float(a2), float(b2)

    # 若 sigma 为 0 或内点过少，则退回首次拟合结果
    return float(a1), float(b1)


__all__ = [
    "fit_single_line_in_range",
]


if __name__ == "__main__":
    """
    测试函数：使用真实 HDF5 物理事例数据，在 2000 < x < 14000 范围内
    使用最小二乘法拟合一条直线。
    """
    import os
    import sys

    import h5py
    import matplotlib.pyplot as plt

    # 添加路径以便导入 utils 模块
    current_dir = os.path.dirname(os.path.abspath(__file__))
    python_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)

    from utils.visualize import get_h5_files

    print("=" * 70)
    print("lsmpncut 测试：在 2000 < x < 14000 范围内拟合一条直线")
    print("=" * 70)

    try:
        # 1. 自动获取文件并筛选物理事例（既非RT也非Inhibit）
        print("\n正在筛选物理事例（既非RT也非Inhibit）...")

        # 获取文件
        h5_files = get_h5_files()
        if "CH0-3" not in h5_files or not h5_files["CH0-3"]:
            raise FileNotFoundError("在 data/hdf5/raw_pulse/CH0-3 目录中未找到 h5 文件")
        if "CH5" not in h5_files or not h5_files["CH5"]:
            raise FileNotFoundError("在 data/hdf5/raw_pulse/CH5 目录中未找到 h5 文件")

        # 查找匹配的文件对
        ch0_3_files = h5_files["CH0-3"]
        ch5_files = h5_files["CH5"]
        ch0_3_dict = {os.path.basename(f): f for f in ch0_3_files}
        ch5_dict = {os.path.basename(f): f for f in ch5_files}

        matched = False
        ch0_3_file = None
        ch5_file = None
        for filename in ch0_3_dict.keys():
            if filename in ch5_dict:
                ch0_3_file = ch0_3_dict[filename]
                ch5_file = ch5_dict[filename]
                matched = True
                break

        if not matched:
            raise ValueError("未找到匹配的CH0-3和CH5文件对")

        print(f"使用文件: {os.path.basename(ch0_3_file)}")

        # 筛选物理事例
        rt_cut = 6000.0
        batch_size = 1000

        with h5py.File(ch0_3_file, "r") as f_ch0:
            ch0_channel_data = f_ch0["channel_data"]
            _, _, ch0_num_events = ch0_channel_data.shape
            ch0_min_values = np.zeros(ch0_num_events, dtype=np.float64)
            for i in range(0, ch0_num_events, batch_size):
                end_idx = min(i + batch_size, ch0_num_events)
                batch_data = ch0_channel_data[:, 0, i:end_idx]
                ch0_min_values[i:end_idx] = np.min(batch_data, axis=0)

        with h5py.File(ch5_file, "r") as f_ch5:
            ch5_channel_data = f_ch5["channel_data"]
            _, _, ch5_num_events = ch5_channel_data.shape
            ch5_max_values = np.zeros(ch5_num_events, dtype=np.float64)
            for i in range(0, ch5_num_events, batch_size):
                end_idx = min(i + batch_size, ch5_num_events)
                batch_data = ch5_channel_data[:, 0, i:end_idx]
                ch5_max_values[i:end_idx] = np.max(batch_data, axis=0)

            # 判断信号类型
            rt_mask = ch5_max_values > rt_cut
            inhibit_mask = ch0_min_values == 0
            neither_mask = ~rt_mask & ~inhibit_mask
            selected_indices = np.where(neither_mask)[0]

        physical_count = len(selected_indices)

        if physical_count == 0:
            print("未发现物理事例，无法进行测试")
            sys.exit(1)

        print(f"找到 {physical_count} 个物理事例")

        # 2. 读取物理事例的波形数据
        print("\n正在读取波形数据...")
        batch_size = 1000
        waveforms_list = []

        with h5py.File(ch0_3_file, "r") as f:
            channel_data = f["channel_data"]
            time_samples, num_channels, _ = channel_data.shape

            # 确保有 CH1 和 CH2（索引 1 和 2）
            if num_channels < 3:
                raise ValueError(
                    f"文件只有 {num_channels} 个通道，需要至少 3 个通道（CH0, CH1, CH2）"
                )

            # 批量读取选中的物理事例
            for i in range(0, len(selected_indices), batch_size):
                end_idx = min(i + batch_size, len(selected_indices))
                batch_indices = selected_indices[i:end_idx]
                batch_waveforms = channel_data[:, :, batch_indices]
                waveforms_list.append(batch_waveforms)
                if (i // batch_size + 1) % 10 == 0 or end_idx == len(selected_indices):
                    print(f"  已读取 {end_idx}/{len(selected_indices)} 个事件")

        # 合并所有批次
        phys_waveforms = np.concatenate(waveforms_list, axis=2)
        print(f"波形数据形状: {phys_waveforms.shape}")

        # 3. 计算 CH1 和 CH2 的最大值
        print("\n正在计算 CH1 和 CH2 的最大值...")
        # phys_waveforms: (time_samples, n_channels, n_events)
        max_ch1 = phys_waveforms[:, 1, :].max(axis=0).astype(np.float64)
        max_ch2 = phys_waveforms[:, 2, :].max(axis=0).astype(np.float64)

        # 4. 在 2000 < x < 14000 范围内使用“两步”最小二乘法拟合一条直线
        print("\n正在对 2000 < x < 14000 范围内的数据进行两步最小二乘拟合...")

        # 先记录范围内的原始点数
        mask_range = (max_ch1 > 2000.0) & (max_ch1 < 14000.0)
        n_in_range = int(mask_range.sum())

        # 调用函数完成：第一次拟合 -> 剔除远点 -> 第二次拟合
        a, b = fit_single_line_in_range(max_ch1, max_ch2, x_min=2000.0, x_max=14000.0)

        # 基于最终直线，在 2000 < x < 14000 范围内计算残差的标准差 σ
        x_fit = max_ch1[mask_range]
        y_fit = max_ch2[mask_range]
        y_fit_pred = a * x_fit + b
        residuals_final = y_fit - y_fit_pred
        sigma_final = residuals_final.std(ddof=1) if residuals_final.size > 1 else 0.0

        print("=" * 70)
        print("两步最小二乘拟合结果：")
        print(f"  最终拟合直线: y = {a:.6f} * x + {b:.3f}")
        print(f"  参与拟合的初始点数 (2000 < x < 14000): {n_in_range}")
        print(f"  最终残差标准差 σ (y 方向): {sigma_final:.3f}")
        print("=" * 70)

        # 5. 绘制图像：所有散点 + 拟合直线 + σ 区域
        print("\n正在绘制图形...")
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # 原始所有物理事例散点（灰色）
        ax.scatter(
            max_ch1,
            max_ch2,
            s=8,
            alpha=0.4,
            c="lightgray",
            label="All physical events",
        )

        # 2000 < x < 14000 范围内的点（红色）
        x_range = max_ch1[mask_range]
        y_range = max_ch2[mask_range]
        ax.scatter(
            x_range,
            y_range,
            s=20,
            alpha=0.9,
            edgecolors="k",
            linewidths=0.3,
            c="tab:red",
            label=f"2000 < x < 14000 ({n_in_range} events)",
        )

        # 画出最终拟合直线及其 σ 区域（在全局 x 范围内）
        x_min, x_max = max_ch1.min(), max_ch1.max()
        x_line = np.linspace(x_min, x_max, 200)
        y_line = a * x_line + b
        ax.plot(x_line, y_line, "r-", lw=2, label=f"Refined fit: y = {a:.6f} * x + {b:.3f}")

        # 若 σ>0，则在直线周围绘制 ±1σ 的绿色透明带
        if sigma_final > 0.0:
            y_upper = y_line + 1.0 * sigma_final
            y_lower = y_line - 1.0 * sigma_final
            ax.fill_between(
                x_line,
                y_lower,
                y_upper,
                color="green",
                alpha=0.2,
                label="±1σ band",
            )

        ax.set_xlabel("CH0 Max (ADC counts)", fontsize=18)
        ax.set_ylabel("CH1 Max (ADC counts)", fontsize=18)
        ax.set_title(
            "Two-step least-squares fit on CH0 Max vs CH1 Max\n"
            "View: x < 2000, y < 3000 (fit on 2000 < x < 14000)",
            fontsize=11,
        )

        # 只显示 x < 2000, y < 3000 的局部区域
        #ax.set_xlim(1100, 1400)
        #ax.set_ylim(1000, 2500)
        ax.legend(fontsize=18)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print("\n测试完成！图形已显示。")

    except Exception as e:
        print(f"\n测试过程中出错: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
