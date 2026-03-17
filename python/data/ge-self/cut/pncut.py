#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pncut 工具函数

用于根据两路主放最大值的线性关系，剔除在能量阈值附近偏离线性关系的噪声信号。

物理图像
--------
- 对每个事件，计算两路主放（例如 CH0 和 CH1）的最大值：max_ch0, max_ch1。
- 正常情况下，max_ch0 与 max_ch1 应服从线性关系：max_ch1 ≈ a * max_ch0 + b。
- 在能量阈值附近的噪声信号会偏离这种线性关系。
- 通过选取紧贴噪声分布的一条直线（cut line）进行筛除，保留符合线性关系的事例。
- 使用 ACT 事例计算 PNcut 在能量阈值附近的效率曲线。

说明
----
- 本文件只提供 **数组级工具函数**，不直接读写 HDF5 文件。
- 上层脚本可以自行根据返回的 max_ch0 / max_ch1 画出二维散点图，并叠加 cut line。
"""

from __future__ import annotations

from typing import Dict, Tuple, Union, Optional

import numpy as np

ArrayLike = Union[np.ndarray]


def _compute_max_in_window(
    waveforms: np.ndarray,
    ch_idx: int,
    max_window: Optional[Union[slice, Tuple[int, int]]] = None,
) -> np.ndarray:
    """
    在给定时间窗上计算某一通道每个事件的 **最大值**。

    参数:
        waveforms: 波形数组，形状为 (n_samples, n_channels, n_events)
        ch_idx:    需要提取的通道索引（如 CH0 -> 0, CH1 -> 1）
        max_window:
            - 若为 None，则在整个波形上求最大值；
            - 若为 slice，对应 waveforms[max_window, ch_idx, :] 的时间区间；
            - 若为 (start, stop) 元组，则自动转换为 slice(start, stop)。

    返回:
        amaxs: 形状为 (n_events,) 的最大值数组，每个事件一个最大值。
    """
    if not isinstance(waveforms, np.ndarray):
        waveforms = np.asarray(waveforms)

    if waveforms.ndim != 3:
        raise ValueError(
            f"waveforms 维度应为 3，形如 (n_samples, n_channels, n_events)，"
            f"当前形状为 {waveforms.shape}"
        )

    n_samples, n_channels, _ = waveforms.shape

    if not (0 <= ch_idx < n_channels):
        raise IndexError(f"通道索引 ch_idx={ch_idx} 超出范围 [0, {n_channels - 1}]")

    if max_window is None:
        # 整个波形上求最大值
        amaxs = waveforms[:, ch_idx, :].max(axis=0).astype(np.float64)
    else:
        if isinstance(max_window, tuple):
            if len(max_window) != 2:
                raise ValueError("max_window 元组必须为 (start, stop)")
            start, stop = max_window
            window_slice = slice(start, stop)
        elif isinstance(max_window, slice):
            window_slice = max_window
        else:
            raise TypeError("max_window 必须是 None、slice 或 (start, stop) 元组")

        # 规范化 slice 边界
        start, stop, step = window_slice.indices(n_samples)
        if step != 1:
            raise ValueError("当前实现仅支持连续的时间窗（step 必须为 1）")
        if start >= stop:
            raise ValueError(f"max_window 无效: start={start}, stop={stop}")

        # 取出该通道在时间窗内的波形: 形状 (n_win_samples, n_events)
        seg = waveforms[window_slice, ch_idx, :]  # type: ignore[arg-type]

        # 沿时间维度取最大值
        amaxs = seg.max(axis=0).astype(np.float64)

    return amaxs


def _fit_cut_line(
    max_ch0: np.ndarray,
    max_ch1: np.ndarray,
    method: str = "quantile",
    quantile: float = 0.05,
) -> Tuple[float, float]:
    """
    拟合紧贴噪声分布的 cut line（直线）。

    参数:
        max_ch0: CH0 的最大值数组
        max_ch1: CH1 的最大值数组
        method: 拟合方法
            - "quantile": 使用分位数方法，找到偏离主线性关系的噪声点，拟合紧贴噪声的直线
            - "linear": 直接线性拟合所有点，然后向下偏移
        quantile: 当 method="quantile" 时，用于识别噪声点的分位数阈值（默认 0.05）

    返回:
        (slope, intercept): 直线参数 y = slope * x + intercept
    """
    if method == "quantile":
        # 方法1：先对主线性关系做拟合，然后找偏离的点（噪声），再拟合噪声边界
        # 1. 对全部点做线性拟合（主线性关系）
        coeffs = np.polyfit(max_ch0, max_ch1, deg=1)
        main_slope, main_intercept = coeffs[0], coeffs[1]

        # 2. 计算每个点到主直线的距离（垂直距离）
        # 点到直线 ax + by + c = 0 的距离 = |ax0 + by0 + c| / sqrt(a^2 + b^2)
        # 对于 y = slope*x + intercept，转换为标准形式：slope*x - y + intercept = 0
        # 距离 = |slope*x0 - y0 + intercept| / sqrt(slope^2 + 1)
        distances = np.abs(main_slope * max_ch0 - max_ch1 + main_intercept) / np.sqrt(
            main_slope**2 + 1
        )

        # 3. 找到距离较大的点（可能是噪声），使用下分位数
        noise_threshold = np.quantile(distances, 1.0 - quantile)
        noise_mask = distances >= noise_threshold

        if noise_mask.sum() < 2:
            # 噪声点太少，回退到直接线性拟合
            print(
                f"警告：噪声点数量过少 ({noise_mask.sum()})，使用全部点进行线性拟合"
            )
            coeffs = np.polyfit(max_ch0, max_ch1, deg=1)
            return coeffs[0], coeffs[1]

        # 4. 对噪声点做线性拟合，得到紧贴噪声的直线
        noise_ch0 = max_ch0[noise_mask]
        noise_ch1 = max_ch1[noise_mask]
        coeffs = np.polyfit(noise_ch0, noise_ch1, deg=1)
        return coeffs[0], coeffs[1]

    elif method == "linear":
        # 方法2：直接线性拟合，然后向下偏移（简单但可能不够精确）
        coeffs = np.polyfit(max_ch0, max_ch1, deg=1)
        slope, intercept = coeffs[0], coeffs[1]

        # 计算残差的标准差，向下偏移 2-3σ
        predicted = slope * max_ch0 + intercept
        residuals = max_ch1 - predicted
        std_residual = residuals.std()
        intercept_cut = intercept - 2.5 * std_residual  # 向下偏移

        return slope, intercept_cut

    else:
        raise ValueError(f"不支持的拟合方法: {method}，支持: 'quantile', 'linear'")


def pncut_ch0_ch1_from_act(
    phys_waveforms: np.ndarray,
    energy: np.ndarray,
    max_window: Optional[Union[slice, Tuple[int, int]]] = None,
    ch0_idx: int = 0,
    ch1_idx: int = 1,
    cut_method: str = "quantile",
    quantile: float = 0.05,
    cut_side: str = "below",
    energy_bins: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    使用 ACT 物理事例，对两路主放（CH0/CH1）的最大值做 PNcut。

    典型用法：
      - phys_waveforms: 已经筛过 RT / Inhibit 等，主要包含 ACT 物理事例。
      - energy:         与 phys_waveforms 对应的能量数组（长度 = n_phys_events），
                        用于计算效率曲线。

    参数:
        phys_waveforms: 物理事例波形，形状 (n_samples, n_channels, n_events)
        energy:         每个事件的能量数组，形状 (n_events,)
        max_window:     计算最大值的时间窗（None 表示整个波形，或 slice / (start, stop)）
        ch0_idx:        CH0 的通道索引（默认 0）
        ch1_idx:        CH1 的通道索引（默认 1）
        cut_method:     拟合 cut line 的方法：'quantile'（默认）或 'linear'
        quantile:       当 cut_method='quantile' 时，用于识别噪声点的分位数（默认 0.05）
        cut_side:       保留哪一侧的点：
                        - 'below': 保留在 cut line 下方的点（默认，用于剔除上方噪声）
                        - 'above': 保留在 cut line 上方的点
        energy_bins:    用于计算效率曲线的能量区间（若为 None，自动生成）

    返回:
        keep_mask, stats 字典:
        - keep_mask: 形状为 (n_events,) 的 bool 数组，True 表示该事件通过 PNcut
        - stats: 包含以下键值的字典:
            * 'max_ch0', 'max_ch1'      : CH0/CH1 的最大值数组
            * 'energy'                  : 对应的能量数组（原样返回）
            * 'cut_slope', 'cut_intercept': Cut line 的参数（y = slope * x + intercept）
            * 'efficiency', 'energy_centers', 'energy_edges': 效率曲线数据
    """
    if not isinstance(phys_waveforms, np.ndarray):
        phys_waveforms = np.asarray(phys_waveforms)

    if phys_waveforms.ndim != 3:
        raise ValueError(
            f"phys_waveforms 维度应为 3，形如 (n_samples, n_channels, n_events)，"
            f"当前形状为 {phys_waveforms.shape}"
        )

    n_samples, n_channels, n_events = phys_waveforms.shape

    energy = np.asarray(energy, dtype=np.float64)
    if energy.ndim != 1 or energy.shape[0] != n_events:
        raise ValueError(
            f"energy 长度必须等于事件数 n_events={n_events}，"
            f"当前 energy 形状为 {energy.shape}"
        )

    # 1. 计算两路主放的最大值
    max_ch0 = _compute_max_in_window(phys_waveforms, ch0_idx, max_window)
    max_ch1 = _compute_max_in_window(phys_waveforms, ch1_idx, max_window)

    # 2. 拟合紧贴噪声分布的 cut line
    cut_slope, cut_intercept = _fit_cut_line(
        max_ch0, max_ch1, method=cut_method, quantile=quantile
    )

    # 3. 应用 PNcut：判断点是否在 cut line 的指定一侧
    predicted_ch1 = cut_slope * max_ch0 + cut_intercept

    if cut_side == "below":
        # 保留在 cut line 下方的点（max_ch1 < predicted_ch1）
        keep_mask = max_ch1 < predicted_ch1
    elif cut_side == "above":
        # 保留在 cut line 上方的点（max_ch1 > predicted_ch1）
        keep_mask = max_ch1 > predicted_ch1
    else:
        raise ValueError(f"cut_side 必须是 'below' 或 'above'，当前为 {cut_side}")

    # 4. 计算效率曲线（在不同能量区间下的通过率）
    if energy_bins is None:
        # 自动生成能量区间
        energy_min, energy_max = energy.min(), energy.max()
        n_bins = 20
        energy_bins = np.linspace(energy_min, energy_max, n_bins + 1)
    else:
        energy_bins = np.asarray(energy_bins)

    energy_centers = (energy_bins[:-1] + energy_bins[1:]) / 2.0
    efficiency = np.zeros(len(energy_centers))

    for i in range(len(energy_centers)):
        bin_mask = (energy >= energy_bins[i]) & (energy < energy_bins[i + 1])
        if bin_mask.sum() > 0:
            efficiency[i] = keep_mask[bin_mask].mean()
        else:
            efficiency[i] = np.nan

    stats: Dict[str, np.ndarray] = {
        "max_ch0": max_ch0,
        "max_ch1": max_ch1,
        "energy": energy,
        "cut_slope": np.array(cut_slope),
        "cut_intercept": np.array(cut_intercept),
        "efficiency": efficiency,
        "energy_centers": energy_centers,
        "energy_edges": energy_bins,
    }

    # 5. 打印简要统计信息
    print("=" * 70)
    print("PN Cut 统计信息（基于 ACT 物理事例两路主放最大值线性关系）")
    print("=" * 70)
    print(f"Cut line: max_ch1 = {cut_slope:.6f} * max_ch0 + {cut_intercept:.3f}")
    print(f"保留策略: {cut_side} cut line")
    print(f"\n事件总数: {n_events}")
    print(f"通过 PN Cut 的事例数: {int(keep_mask.sum())} " f"({keep_mask.mean() * 100:.2f}%)")
    print("=" * 70)

    return keep_mask, stats


__all__ = [
    "pncut_ch0_ch1_from_act",
    "_compute_max_in_window",
    "_fit_cut_line",
]


if __name__ == "__main__":
    """
    测试函数：直接从 CH0_parameters / CH1_parameters 中读取 max_ch0 / max_ch1，
    绘制 PNcut 前的 max CH0 vs max CH1 散点图。
    """
    import os
    import sys
    import h5py
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("PNcut 测试：从参数文件绘制 max CH0 vs max CH1 图形")
    print("=" * 70)

    try:
        # 推断项目根目录：.../python/data/ge-self/cut/pncut.py -> .../DeepVibration
        current_dir = os.path.dirname(os.path.abspath(__file__))      # .../cut
        ge_self_dir = os.path.dirname(current_dir)                    # .../ge-self
        data_dir = os.path.dirname(ge_self_dir)                       # .../data
        python_dir = os.path.dirname(data_dir)                        # .../python
        project_root = os.path.dirname(python_dir)                    # .../DeepVibration

        ch0_param_dir = os.path.join(project_root, "data", "hdf5", "raw_pulse", "CH0_parameters")
        ch1_param_dir = os.path.join(project_root, "data", "hdf5", "raw_pulse", "CH1_parameters")

        if not os.path.isdir(ch0_param_dir):
            raise FileNotFoundError(f"CH0_parameters 目录不存在: {ch0_param_dir}")
        if not os.path.isdir(ch1_param_dir):
            raise FileNotFoundError(f"CH1_parameters 目录不存在: {ch1_param_dir}")

        # 基于文件名在 CH0/CH1 参数目录中匹配 run
        ch0_files = sorted(
            name for name in os.listdir(ch0_param_dir)
            if name.lower().endswith((".h5", ".hdf5"))
        )
        if not ch0_files:
            raise FileNotFoundError(f"在 {ch0_param_dir} 中未找到任何 h5 参数文件")

        ch1_existing = {
            name for name in os.listdir(ch1_param_dir)
            if name.lower().endswith((".h5", ".hdf5"))
        }

        all_max_ch0 = []
        all_max_ch1 = []
        n_runs_used = 0

        for name in ch0_files:
            if name not in ch1_existing:
                continue

            ch0_path = os.path.join(ch0_param_dir, name)
            ch1_path = os.path.join(ch1_param_dir, name)

            with h5py.File(ch0_path, "r") as f_ch0, h5py.File(ch1_path, "r") as f_ch1:
                if "max_ch0" not in f_ch0 or "max_ch1" not in f_ch1:
                    print(f"[警告] {name} 中缺少 max_ch0 或 max_ch1，跳过该 run。")
                    continue

                max_ch0 = np.asarray(f_ch0["max_ch0"][...], dtype=np.float64)
                max_ch1 = np.asarray(f_ch1["max_ch1"][...], dtype=np.float64)

                if max_ch0.ndim != 1 or max_ch1.ndim != 1:
                    print(f"[警告] {name} 中 max_ch0/max_ch1 不是一维数组，跳过该 run。")
                    continue

                n0, n1 = max_ch0.shape[0], max_ch1.shape[0]
                n = min(n0, n1)
                if n0 != n1:
                    print(
                        f"[警告] {name} 中 CH0/CH1 事件数不一致 ({n0} vs {n1})，"
                        f"仅使用前 {n} 个事件。"
                    )
                    max_ch0 = max_ch0[:n]
                    max_ch1 = max_ch1[:n]

            all_max_ch0.append(max_ch0)
            all_max_ch1.append(max_ch1)
            n_runs_used += 1
            print(f"[读取] {name} | 事件数: {n}")

        if not all_max_ch0:
            raise RuntimeError("未能从任何 run 中收集到 max_ch0/max_ch1，无法绘图。")

        max_ch0_all = np.concatenate(all_max_ch0)
        max_ch1_all = np.concatenate(all_max_ch1)

        print("=" * 70)
        print(f"总共使用 {n_runs_used} 个 run，汇总事件数: {max_ch0_all.size}")

        # 在指定范围内做快速线性拟合：
        #   6000 <= CH0max <= 12000 且 CH1max > 3000
        print("\n在 6000 <= CH0max <= 12000 且 CH1max > 3000 条件下进行快速线性拟合...")
        mask_fit = (
            (max_ch0_all >= 6000.0) & (max_ch0_all <= 12000.0) &
            (max_ch1_all > 3000.0)
        )
        x_fit = max_ch0_all[mask_fit]
        y_fit = max_ch1_all[mask_fit]
        n_fit = x_fit.size
        if n_fit < 2:
            raise RuntimeError(
                f"满足拟合条件的点数不足 2 个 (n_fit={n_fit})，无法进行线性拟合。"
            )

        # 使用 numpy.polyfit 做普通最小二乘直线拟合。
        # 该操作内部使用向量化/BLAS，已能较好利用 CPU 资源。
        a, b = np.polyfit(x_fit, y_fit, deg=1)

        # 在拟合点上计算残差的标准差 σ，用于构建平行于直线的 ±σ 带
        y_fit_pred = a * x_fit + b
        residuals = y_fit - y_fit_pred
        sigma = residuals.std(ddof=1) if residuals.size > 1 else 0.0

        print(f"线性拟合结果（在筛选条件下）: CH1max ≈ {a:.6f} * CH0max + {b:.3f}")
        print(f"参与拟合点数: {n_fit}")
        print(f"残差标准差 σ（沿 y 方向）: {sigma:.3f}")

        # 如散点数过多，随机下采样到可快速显示的数量（例如 200k）用于绘图
        max_points = 200_000
        n_total = max_ch0_all.size
        if n_total > max_points:
            print(f"散点数量 {n_total} 过多，随机下采样到 {max_points} 个点用于绘图。")
            rng = np.random.default_rng(42)
            idx_sub = rng.choice(n_total, size=max_points, replace=False)
            x_plot = max_ch0_all[idx_sub]
            y_plot = max_ch1_all[idx_sub]
        else:
            x_plot = max_ch0_all
            y_plot = max_ch1_all

        # 1. 绘制全局 max CH0 vs max CH1 散点图，并叠加拟合直线
        print("\n正在绘制全局图形...")
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        ax.scatter(
            x_plot, y_plot,
            s=3, alpha=0.4, c="blue", edgecolors="none",
            label="Events (from parameters)",
        )

        # 在全局 x 范围上画出拟合直线及其平行的 ±σ 带
        x_line = np.linspace(x_plot.min(), x_plot.max(), 200)
        y_line = a * x_line + b
        ax.plot(x_line, y_line, "r-", lw=2, alpha=0.8, label="Linear fit (range-filtered)")

        if sigma > 0.0:
            y_upper = y_line + 2*sigma
            y_lower = y_line - 2*sigma
            ax.fill_between(
                x_line,
                y_lower,
                y_upper,
                color="red",
                alpha=0.15,
                label="±2σ band (parallel to line)",
            )

        ax.set_xlabel("CH0 Max (ADC counts)", fontsize=12)
        ax.set_ylabel("CH1 Max (ADC counts)", fontsize=12)
        ax.set_title(f"CH0 Max vs CH1 Max\nTotal: {len(max_ch0_all)} events", fontsize=11)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


    except Exception as e:
        print(f"\n测试过程中出错: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

