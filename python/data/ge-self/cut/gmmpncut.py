#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gmmpncut 工具脚本

在 lsmpncut 的基础上：
- 使用真实 HDF5 物理事例数据得到散点 (x, y) = (max_ch1, max_ch2)；
- 在 2000 < x < 14000 范围内做“两步”最小二乘拟合，得到直线 y = a * x + b 及残差 σ；
- 只选取同时满足
    1100 < x < 1400,
    1000 < y < 2200,
    且 |y - (a*x + b)| <= 1σ
  的事件；
- 对这些事件的二维分布 (x, y) 进行 2 成分 GMM 拟合；
- 在散点图上绘制两个高斯分量对应的 1σ 椭圆区域。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _fit_single_line_two_step(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float = 2000.0,
    x_max: float = 14000.0,
) -> Tuple[float, float, float]:
    """
    在区间 (x_min, x_max) 内，使用“两步”最小二乘法拟合一条直线 y = a * x + b，
    并返回最终残差的标准差 σ。

    返回 (a, b, sigma)，其中 sigma 为最终残差 std。
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    if x.shape != y.shape:
        raise ValueError(f"x 与 y 形状不一致: {x.shape} vs {y.shape}")

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
    sigma1 = residuals.std(ddof=1) if residuals.size > 1 else 0.0

    if sigma1 > 0.0:
        inlier_mask = np.abs(residuals) <= 3.0 * sigma1
        n_inliers = int(inlier_mask.sum())

        # 只有在有足够多的内点、且确实剔除了部分点时才进行第二次拟合
        if 2 <= n_inliers < x_sel.size:
            x_in = x_sel[inlier_mask]
            y_in = y_sel[inlier_mask]
            a2, b2 = np.polyfit(x_in, y_in, deg=1)

            # 最终残差
            y_pred2 = a2 * x_in + b2
            residuals2 = y_in - y_pred2
            sigma2 = residuals2.std(ddof=1) if residuals2.size > 1 else 0.0
            return float(a2), float(b2), float(sigma2)

    # 若 sigma1 为 0 或内点过少，则退回首次拟合结果
    return float(a1), float(b1), float(sigma1)


if __name__ == "__main__":
    """
    使用 GMM 对位于指定窗口且在直线 ±1σ 带内的事件 (x, y) 做二维高斯混合拟合，
    并绘制两个高斯分量的 1σ 椭圆区域。
    """

    import os
    import sys

    import h5py
    import matplotlib.pyplot as plt

    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as e:  # pragma: no cover - 仅在缺少依赖时触发
        print("错误：需要 scikit-learn 来进行 GMM 拟合，请先安装：")
        print("    pip install scikit-learn")
        sys.exit(1)

    # 添加路径以便导入 utils 模块
    current_dir = os.path.dirname(os.path.abspath(__file__))
    python_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)

    from utils.visualize import get_h5_files

    print("=" * 70)
    print("gmmpncut 测试：GMM 拟合 1100<x<1400, 1000<y<2200 且在直线 ±1σ 内的事件")
    print("=" * 70)

    try:
        # 1. 自动获取文件并筛选物理事例（既非RT也非Inhibit）
        print("\n正在筛选物理事例（既非RT也非Inhibit）...")

        h5_files = get_h5_files()
        if "CH0-3" not in h5_files or not h5_files["CH0-3"]:
            raise FileNotFoundError("在 data/hdf5/raw_pulse/CH0-3 目录中未找到 h5 文件")
        if "CH5" not in h5_files or not h5_files["CH5"]:
            raise FileNotFoundError("在 data/hdf5/raw_pulse/CH5 目录中未找到 h5 文件")

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

            if num_channels < 3:
                raise ValueError(
                    f"文件只有 {num_channels} 个通道，需要至少 3 个通道（CH0, CH1, CH2）"
                )

            for i in range(0, len(selected_indices), batch_size):
                end_idx = min(i + batch_size, len(selected_indices))
                batch_indices = selected_indices[i:end_idx]
                batch_waveforms = channel_data[:, :, batch_indices]
                waveforms_list.append(batch_waveforms)
                if (i // batch_size + 1) % 10 == 0 or end_idx == len(selected_indices):
                    print(f"  已读取 {end_idx}/{len(selected_indices)} 个事件")

        phys_waveforms = np.concatenate(waveforms_list, axis=2)
        print(f"波形数据形状: {phys_waveforms.shape}")

        # 3. 计算 CH1 和 CH2 的最大值
        print("\n正在计算 CH1 和 CH2 的最大值...")
        max_ch1 = phys_waveforms[:, 1, :].max(axis=0).astype(np.float64)
        max_ch2 = phys_waveforms[:, 2, :].max(axis=0).astype(np.float64)

        # 4. 两步最小二乘拟合 2000 < x < 14000，得到直线和 σ
        print("\n正在对 2000 < x < 14000 范围内的数据进行两步最小二乘拟合...")
        a, b, sigma = _fit_single_line_two_step(
            max_ch1,
            max_ch2,
            x_min=2000.0,
            x_max=14000.0,
        )

        print(f"  拟合得到直线: y = {a:.6f} * x + {b:.3f}")
        print(f"  残差标准差 σ (y 方向): {sigma:.3f}")

        # 5. 选取 1100<x<1400, 1000<y<2200 且在直线 ±1σ 带内的事件
        print("\n正在筛选用于 GMM 拟合的事件...")
        mask_window = (
            (max_ch1 > 1100.0)
            & (max_ch1 < 1400.0)
            & (max_ch2 > 1000.0)
            & (max_ch2 < 2200.0)
        )
        if sigma > 0.0:
            residuals_all = max_ch2 - (a * max_ch1 + b)
            mask_band = np.abs(residuals_all) <= sigma
        else:
            mask_band = np.ones_like(max_ch1, dtype=bool)

        mask_sel = mask_window & mask_band
        x_sel = max_ch1[mask_sel]
        y_sel = max_ch2[mask_sel]

        n_sel = int(mask_sel.sum())
        print(f"  满足窗口和 ±1σ 条件的事件数: {n_sel}")
        if n_sel < 10:
            print("选中事件数过少，无法稳定进行 GMM 拟合")
            sys.exit(1)

        # 记录这些事件在物理事例数组中的索引，方便后续画 CH0 波形
        sel_indices = np.where(mask_sel)[0]

        data = np.column_stack([x_sel, y_sel])

        # 6. 进行 2 成分 GMM 拟合
        print("\n正在进行 2 成分 GMM 拟合...")
        gmm = GaussianMixture(
            n_components=2,
            covariance_type="full",
            random_state=0,
        )
        gmm.fit(data)

        labels = gmm.predict(data)
        means = gmm.means_          # (2, 2)
        covariances = gmm.covariances_  # (2, 2, 2)

        # 按 y 坐标从小到大对两个 GMM 分量排序，便于定义“区域1/区域2”
        # （区域1：y 更小的成分；区域2：y 更大的成分）
        order = np.argsort(means[:, 1])
        region1_label = int(order[0])
        region2_label = int(order[1])

        print("GMM 拟合完成。均值为（已按 y 从小到大排序）：")
        for rank, k in enumerate(order, start=1):
            print(f"  区域 {rank} (组件 {k}): mean = ({means[k,0]:.2f}, {means[k,1]:.2f})")

        # 7. 绘制散点 + 叠加 GMM 热力图（第一个窗口）
        print("\n正在绘制 GMM 结果图像...")
        fig_gmm, ax = plt.subplots(1, 1, figsize=(8, 6))

        # 背景：所有物理事例（浅灰）
        ax.scatter(
            max_ch1,
            max_ch2,
            s=8,
            alpha=0.15,
            c="lightgray",
            label="All physical events",
        )

        # 被选中用于 GMM 的事件，按 GMM 标签着色
        colors = np.array(["tab:red", "tab:blue"])
        ax.scatter(
            x_sel,
            y_sel,
            c=colors[labels],
            s=25,
            alpha=0.9,
            edgecolors="k",
            linewidths=0.4,
            label="Selected events for GMM",
        )

        # 在同一张图上绘制原来的拟合直线及其 ±1σ 带（方便参考）
        x_min_plot, x_max_plot = 1100.0, 1400.0
        x_line = np.linspace(x_min_plot, x_max_plot, 200)
        y_line = a * x_line + b
        ax.plot(x_line, y_line, "r--", lw=1.5, label="LS fit line")
        if sigma > 0.0:
            y_upper = y_line + sigma
            y_lower = y_line - sigma
            ax.fill_between(
                x_line,
                y_lower,
                y_upper,
                color="green",
                alpha=0.15,
                label="±1σ band (LS fit)",
            )

        # 使用 GMM 在同一窗口内生成概率密度热力图，并叠加到最上层
        grid_size = 200
        xx, yy = np.meshgrid(
            np.linspace(1100.0, 1400.0, grid_size),
            np.linspace(1000.0, 2200.0, grid_size),
        )
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        # score_samples 返回对数概率密度，取 exp 变成实际密度
        log_probs = gmm.score_samples(grid_points)
        zz = np.exp(log_probs).reshape(xx.shape)

        # 叠加热力图（半透明，放在散点之上）
        cf = ax.contourf(
            xx,
            yy,
            zz,
            levels=30,
            cmap="viridis",
            alpha=0.6,
            zorder=5,
        )
        plt.colorbar(cf, ax=ax, label="GMM density")

        ax.set_xlabel("CH1 Max (ADC counts)", fontsize=12)
        ax.set_ylabel("CH2 Max (ADC counts)", fontsize=12)
        ax.set_title(
            "GMM on events within 1100<x<1400, 1000<y<2200\n"
            "and inside ±1σ band of LS fit line",
            fontsize=11,
        )

        # 只显示指定窗口
        ax.set_xlim(1100, 1400)
        ax.set_ylim(1000, 2200)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        # 第一个窗口：GMM 结果。用户关闭该窗口后，再依次弹出后两个窗口。
        plt.show()

        print("\nGMM 拟合与绘图完成，接下来将依次展示两个 GMM 区域中各选一个事件的 CH0 波形。")

        # 8. 从两个 GMM 区域中各选取一个事件，绘制其 CH0 波形
        time_samples = phys_waveforms.shape[0]
        t_axis = np.arange(time_samples)

        # 为每个区域找一个样本索引（在 x_sel/y_sel 中的位置）
        region1_indices = np.where(labels == region1_label)[0]
        region2_indices = np.where(labels == region2_label)[0]

        # 只有在该区域中至少有一个事件时才画波形
        if region1_indices.size > 0:
            idx_r1 = region1_indices[0]
            global_idx_r1 = sel_indices[idx_r1]
            ch0_wave_r1 = phys_waveforms[:, 0, global_idx_r1]

            fig_r1, ax_r1 = plt.subplots(1, 1, figsize=(8, 4))
            ax_r1.plot(t_axis, ch0_wave_r1, color="tab:blue")
            ax_r1.set_xlabel("Sample index")
            ax_r1.set_ylabel("CH0 ADC counts")
            ax_r1.set_title(
                "GMM 区域 1 中某事件的 CH0 波形\n"
                f"(x={x_sel[idx_r1]:.1f}, y={y_sel[idx_r1]:.1f})"
            )
            ax_r1.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        if region2_indices.size > 0:
            idx_r2 = region2_indices[0]
            global_idx_r2 = sel_indices[idx_r2]
            ch0_wave_r2 = phys_waveforms[:, 0, global_idx_r2]

            fig_r2, ax_r2 = plt.subplots(1, 1, figsize=(8, 4))
            ax_r2.plot(t_axis, ch0_wave_r2, color="tab:orange")
            ax_r2.set_xlabel("Sample index")
            ax_r2.set_ylabel("CH0 ADC counts")
            ax_r2.set_title(
                "GMM 区域 2 中某事件的 CH0 波形\n"
                f"(x={x_sel[idx_r2]:.1f}, y={y_sel[idx_r2]:.1f})"
            )
            ax_r2.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"\n测试过程中出错: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
