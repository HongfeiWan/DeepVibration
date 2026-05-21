#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于 basic+acv.py 的 cut 逻辑，绘制 mask 后事件的 max(CH0) 分布图。

mask 定义与 basic+acv.py 保持一致：
    mask = m1 & m2 & m4 & m5 & m6 & m8
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def _load_basic_acv_module():
    """从同目录的 basic+acv.py 动态加载模块。"""
    current_dir = Path(__file__).resolve().parent
    basic_acv_path = current_dir / "basic+acv.py"
    if not basic_acv_path.is_file():
        raise FileNotFoundError(f"未找到文件: {basic_acv_path}")

    spec = importlib.util.spec_from_file_location("basic_acv_module", basic_acv_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {basic_acv_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_top_peaks_from_hist(
    values: np.ndarray,
    centers: np.ndarray,
    top_k: int = 3,
    prominence_window: int = 80,
    exclude_ranges: List[tuple[float, float]] | None = None,) -> List[tuple[float, float, float]]:
    """
    从直方图中寻找最显著峰。
    返回: [(x_peak, y_peak, prominence), ...]，按 prominence 从高到低排序。
    """
    if values.size < 3:
        return []

    raw = values.astype(np.float64)

    # 局部极大值候选
    candidate_idx = np.where((raw[1:-1] > raw[:-2]) & (raw[1:-1] >= raw[2:]))[0] + 1
    if candidate_idx.size == 0:
        return []

    # 排除指定 x 区间内的峰候选
    if exclude_ranges:
        keep = np.ones(candidate_idx.size, dtype=bool)
        cand_x = centers[candidate_idx]
        for x_low, x_high in exclude_ranges:
            keep &= ~((cand_x >= x_low) & (cand_x <= x_high))
        candidate_idx = candidate_idx[keep]
        if candidate_idx.size == 0:
            return []

    # 粗略 prominence: 峰高减去左右窗口基线较高者
    pwin = max(10, int(prominence_window))
    peak_infos: List[tuple[float, float, float, int]] = []
    for i in candidate_idx:
        left = max(0, i - pwin)
        right = min(raw.size, i + pwin + 1)
        left_min = np.min(raw[left:i]) if i > left else raw[i]
        right_min = np.min(raw[i + 1:right]) if (i + 1) < right else raw[i]
        base = max(left_min, right_min)
        prominence = raw[i] - base
        if prominence > 0:
            peak_infos.append((centers[i], values[i], float(prominence), i))

    if not peak_infos:
        return []

    # 先按 prominence 选 top_k，再按 x 位置排序用于标注
    top = sorted(peak_infos, key=lambda t: t[2], reverse=True)[:top_k]
    top = sorted(top, key=lambda t: t[0])
    return [(x_peak, y_peak, prom) for x_peak, y_peak, prom, _ in top]


def _find_peak_in_range(
    values: np.ndarray,
    centers: np.ndarray,
    x_low: float,
    x_high: float,
) -> tuple[float, float] | None:
    """
    在给定 x 区间内指定一个峰（选该区间内 values 最大点）。
    返回 (x_peak, y_value)；若区间内无有效点则返回 None。
    """
    in_range = (centers >= x_low) & (centers <= x_high)
    if not np.any(in_range):
        return None
    idx_local = np.where(in_range)[0]
    local_vals = values[idx_local]
    if local_vals.size == 0:
        return None
    best_i = idx_local[int(np.argmax(local_vals))]
    return float(centers[best_i]), float(values[best_i])


if __name__ == "__main__":
    basic_acv = _load_basic_acv_module()

    pairs = basic_acv._list_paired_param_files()
    print(f"找到 {len(pairs)} 个可配对参数文件。")
    if not pairs:
        raise RuntimeError("未找到任何可配对参数文件，无法继续。")

    all_max_ch0: List[np.ndarray] = []
    all_ch0_min: List[np.ndarray] = []
    all_max_ch5: List[np.ndarray] = []
    all_ch0_ped_mean: List[np.ndarray] = []
    all_ch1_ped_mean: List[np.ndarray] = []
    all_max_ch1: List[np.ndarray] = []
    all_ch1_min: List[np.ndarray] = []
    all_max_ch4: List[np.ndarray] = []
    all_tmax_ch4: List[np.ndarray] = []

    for ch0_path, ch5_path in pairs:
        m0, cmin, m5, ped0, ped1, m1, c1min, m4, t4, _, _ = basic_acv._load_basic_features_for_run(
            ch0_path, ch5_path
        )
        all_max_ch0.append(m0)
        all_ch0_min.append(cmin)
        all_max_ch5.append(m5)
        all_ch0_ped_mean.append(ped0)
        all_ch1_ped_mean.append(ped1)
        all_max_ch1.append(m1)
        all_ch1_min.append(c1min)
        all_max_ch4.append(m4)
        all_tmax_ch4.append(t4)

    max_ch0 = np.concatenate(all_max_ch0)
    ch0_min = np.concatenate(all_ch0_min)
    max_ch5 = np.concatenate(all_max_ch5)
    ch0_ped_mean = np.concatenate(all_ch0_ped_mean)
    ch1_ped_mean = np.concatenate(all_ch1_ped_mean)
    max_ch1 = np.concatenate(all_max_ch1)
    ch1_min = np.concatenate(all_ch1_min)
    max_ch4 = np.concatenate(all_max_ch4)
    tmax_ch4 = np.concatenate(all_tmax_ch4)

    n_raw = max_ch0.shape[0]
    print(f"原始事件数: {n_raw}")

    # 与 basic+acv.py 中一致的 cut 组合
    m1 = basic_acv.cut_ch0_min_positive(ch0_min)
    m2 = basic_acv.cut_ch0_max_saturation(max_ch0, max_ch1)
    m4 = basic_acv.cut_pedestal_3sigma(ch0_ped_mean, ch1_ped_mean, max_ch5)
    m5 = basic_acv.cut_acv(max_ch4, tmax_ch4)
    m6 = basic_acv.cut_mincut(ch0_min, ch1_min, max_ch4, tmax_ch4)
    m8 = basic_acv.cut_pncut(m1 & m2 & m4 & m5 & m6, max_ch0, max_ch1)
    mask = m1 & m2 & m4 & m5 & m6 & m8

    x = max_ch0[mask]
    print(f"通过 mask 事件数: {x.size} / {n_raw} ({(x.size / max(n_raw, 1)) * 100:.2f}%)")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    counts, edges, _ = ax.hist(
        x,
        bins=700,
        color="C0",
        alpha=0.8
    )
    centers = edges[:-1] + 0.5 * np.diff(edges)

    # 在 log(counts) 分布上寻找最显著的 3 个峰，并排除 1180~1200 区间
    log_counts = np.log10(np.maximum(counts, 1.0))
    peaks = _find_top_peaks_from_hist(
        values=log_counts,
        centers=centers,
        top_k=3,
        exclude_ranges=[(1180.0, 1200.0)],
    )

    # 指定峰区间：强制包含 2400~2600 和 1000~1150 内各一个代表峰
    forced_ranges = [(2400.0, 2600.0), (1000.0, 1150.0)]
    for x_low, x_high in forced_ranges:
        forced_peak = _find_peak_in_range(
            values=log_counts,
            centers=centers,
            x_low=x_low,
            x_high=x_high,
        )
        if forced_peak is None:
            continue
        fp_x, fp_y = forced_peak
        # 若与已选峰位置过近，则替换为指定峰；否则追加后再按强度截断为 3 个
        updated = []
        replaced = False
        for p in peaks:
            if abs(p[0] - fp_x) <= 1.0:
                updated.append((fp_x, fp_y, p[2]))
                replaced = True
            else:
                updated.append(p)
        if not replaced:
            updated.append((fp_x, fp_y, 1e9))
        peaks = sorted(updated, key=lambda t: t[2], reverse=True)[:3]
        peaks = sorted(peaks, key=lambda t: t[0])

    if peaks:
        x_peaks = [p[0] for p in peaks]
        # 图上圈在原始计数高度位置（y 轴显示为 log）
        y_peaks = [counts[int(np.argmin(np.abs(centers - xp)))] for xp in x_peaks]
        print("最终三个峰信息：")
        for i, (xp, yp) in enumerate(zip(x_peaks, y_peaks), start=1):
            print(
                f"  峰{i}: x={xp:.2f}, count={yp:.0f}, log10(count)={np.log10(max(yp, 1.0)):.4f}"
            )
        ax.scatter(
            x_peaks,
            y_peaks,
            s=2,
            facecolors="none",
            edgecolors="crimson",
            linewidths=2.2,
            zorder=5,
        )

        # Figure 2: 三个峰的能量标定最小二乘直线拟合
        # 从左到右对应: RT(0 keV), Ge-LX(1.298 keV), Ge-KX(10.373 keV)
        energies = np.array([0.0, 1.298, 10.373], dtype=np.float64)
        ch0_peaks = np.array(x_peaks[:3], dtype=np.float64)
        labels = ["RT", "Ge-LX", "Ge-KX"]

        # 最小二乘拟合: max(CH0) = a * Energy + b
        a_fit, b_fit = np.polyfit(energies, ch0_peaks, deg=1)
        print(f"能量标定拟合: max(CH0) = {a_fit:.6f} * Energy + {b_fit:.6f}")

        fig_fit, ax_fit = plt.subplots(1, 1, figsize=(8, 6))
        e_line = np.linspace(energies.min(), energies.max(), 300)
        ch0_line = a_fit * e_line + b_fit
        ax_fit.plot(e_line, ch0_line, color="C1", linewidth=2.0, label="Least-squares fit")

        # ADC 峰位不确定度：半箱宽；标定 CH0 = a·E + b 时 σ_E = σ_ADC / |a|，在图上画水平误差棒
        bin_width_adc = float(edges[1] - edges[0])
        sigma_adc = np.full(ch0_peaks.shape[0], 0.5 * bin_width_adc, dtype=np.float64)
        a_abs = max(abs(float(a_fit)), 1e-12)
        xerr_peaks = sigma_adc / a_abs
        ax_fit.errorbar(
            energies,
            ch0_peaks,
            xerr=xerr_peaks,
            fmt=".",
            color="crimson",
            markersize=3,
            elinewidth=1.5,
            ecolor="crimson",
            capsize=0.1,
            capthick=0.1,
            zorder=5,
        )
        for e, ch0, name in zip(energies, ch0_peaks, labels):
            ax_fit.text(
                e + 0.12,
                ch0,
                name,
                fontsize=12,
                va="center",
                ha="left",
                color="black",
            )

        ax_fit.set_xlabel("Energy (keV)", fontsize=16)
        ax_fit.set_ylabel("max(CH0) [ADC counts]", fontsize=16)
        # ax_fit.set_title("Energy calibration (RT / Ge-LX / Ge-KX)", fontsize=18)
        ax_fit.tick_params(axis="both", which="major", labelsize=12)
        ax_fit.grid(alpha=0.25)
        ax_fit.legend(fontsize=12)
        fig_fit.tight_layout()
    else:
        print("未找到显著峰。")

    ax.set_yscale("log")
    ax.set_xlabel("max(CH0) [ADC counts]", fontsize=16)
    ax.set_ylabel("Counts (log scale)", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(alpha=0.25)
    ax.set_xlim(1000,16340)
    fig.tight_layout()
    plt.show()
