#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ACT 能谱与 cut 效率检查脚本。

思路：
1) 复用 tradition.py 的最小特征读取与 cut 函数；
2) 定义 ACT 事件为 acv 的取反：act_mask = ~m_acv；
3) 以 ACT 为起点，按顺序叠加 cut 绘制能谱；
4) 打印各步 cut 的累计效率；效率为每能量箱 N_pass/N_basic，误差为 Wilson 区间（非 rate 比 Poisson 传播）。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import traditionACT as tr

# 与 traditionACT.py 中曲线颜色顺序一致（用于效率曲线 errorbar/拟合曲线）
_FIXED_COLORS = [
    "C0",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
]

# Wilson 区间 z：1.0 ≈ 68% 中心覆盖（与常见「1σ」展示相当）；1.96 ≈ 95%
WILSON_Z = 1.0


def _histogram_counts_per_energy_bin(
    max_ch0: np.ndarray,
    mask: np.ndarray,
    *,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """在给定能量箱边界下，对通过 mask 的事件按 keV 能量计数。"""
    x = np.asarray(max_ch0, dtype=np.float64).reshape(-1)
    m = np.asarray(mask, dtype=bool).reshape(-1)
    n = min(x.size, m.size)
    e = tr._E_CAL_A * x[:n][m[:n]] + tr._E_CAL_B
    counts, _ = np.histogram(e, bins=bin_edges)
    return counts.astype(np.float64, copy=False)


def _wilson_interval(
    k: np.ndarray,
    n: np.ndarray,
    *,
    z: float = WILSON_Z,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    二项比例 k/n 的 Wilson 置信区间 [lower, upper]；返回 (p_hat, lower, upper)。
    n=0 的箱为 nan；k 截断到不超过 n。
    """
    k = np.asarray(k, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    k = np.minimum(k, n)
    p_hat = np.divide(k, n, out=np.full_like(k, np.nan, dtype=np.float64), where=n > 0)
    p_out = np.where(n > 0, np.clip(p_hat, 0.0, 1.0), np.nan)

    lower = np.full_like(p_hat, np.nan, dtype=np.float64)
    upper = np.full_like(p_hat, np.nan, dtype=np.float64)
    nz = n > 0
    z2 = z * z
    denom = np.empty_like(n, dtype=np.float64)
    denom[nz] = 1.0 + z2 / n[nz]
    center = np.empty_like(p_hat, dtype=np.float64)
    center[nz] = (p_hat[nz] + z2 / (2.0 * n[nz])) / denom[nz]
    radic = np.zeros_like(p_hat, dtype=np.float64)
    radic[nz] = p_hat[nz] * (1.0 - p_hat[nz]) / n[nz] + z2 / (4.0 * n[nz] * n[nz])
    radic[nz] = np.maximum(radic[nz], 0.0)
    margin = np.empty_like(p_hat, dtype=np.float64)
    margin[nz] = z * np.sqrt(radic[nz]) / denom[nz]
    lower[nz] = np.clip(center[nz] - margin[nz], 0.0, 1.0)
    upper[nz] = np.clip(center[nz] + margin[nz], 0.0, 1.0)
    return p_out, lower, upper


def _efficiency_binomial_wilson(
    max_ch0: np.ndarray,
    mask_basic: np.ndarray,
    mask_last: np.ndarray,
    *,
    bin_edges: np.ndarray,
    z: float = WILSON_Z,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    每能量箱：efficiency = N_last / N_basic（N_basic = 过 basic 且落入该箱；N_last = 过 last 且落入该箱）。
    要求 last 为 basic 的子集（本脚本 cumulative 构造满足）。返回 p_hat 与 errorbar 用的非对称 y 误差 (y_down, y_up)。
    """
    n_den = _histogram_counts_per_energy_bin(max_ch0, mask_basic, bin_edges=bin_edges)
    n_num = _histogram_counts_per_energy_bin(max_ch0, mask_last, bin_edges=bin_edges)
    n_num = np.minimum(n_num, n_den)
    p_hat, lo, up = _wilson_interval(n_num, n_den, z=z)
    y_down = p_hat - lo
    y_up = up - p_hat
    y_down = np.where(np.isfinite(p_hat), y_down, np.nan)
    y_up = np.where(np.isfinite(p_hat), y_up, np.nan)
    # matplotlib.errorbar 要求 yerr >= 0；clip 与浮点舍入可能使差值略为负
    y_down = np.maximum(y_down, 0.0)
    y_up = np.maximum(y_up, 0.0)
    return p_hat, y_down, y_up


def _compute_spectrum_rate_and_err(
    max_ch0: np.ndarray,
    mask: np.ndarray,
    *,
    n_bins: int,
    e_min: float,
    e_max: float,
    exposure_days: float | None = None,) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    与 traditionACT.py 的谱计算保持一致的 rate 定义，并额外给出 Poisson 误差：
        rate = counts / (EXPOSURE_KG * ΔE * exposure_days)
        sigma(rate) ≈ sqrt(counts) / (EXPOSURE_KG * ΔE * exposure_days)
    """
    x = np.asarray(max_ch0, dtype=np.float64).reshape(-1)
    m = np.asarray(mask, dtype=bool).reshape(-1)
    n = min(x.size, m.size)
    x = x[:n]
    m = m[:n]

    energy_values = tr._E_CAL_A * x[m] + tr._E_CAL_B
    counts, bin_edges = np.histogram(energy_values, bins=int(n_bins), range=(float(e_min), float(e_max)))
    counts = counts.astype(np.float64, copy=False)

    bin_widths = np.diff(bin_edges).astype(np.float64, copy=False)
    bin_widths[bin_widths == 0] = np.inf

    exp_days = float(tr.EXPOSURE_DAYS if exposure_days is None else exposure_days)
    denom = float(tr.EXPOSURE_KG) * bin_widths * exp_days
    rates = counts / denom
    err = np.sqrt(counts) / denom
    return bin_edges, rates, err

def _load_required_arrays(base_names: List[str]) -> dict[str, np.ndarray]:
    """按需读取当前分析需要的最小参数集合。"""
    max_values: List[np.ndarray] = []
    ch0_min_values: List[np.ndarray] = []
    max_ch5_values: List[np.ndarray] = []
    ch0_ped_mean_values: List[np.ndarray] = []
    ch1_ped_mean_values: List[np.ndarray] = []
    ch1_min_values: List[np.ndarray] = []
    max_ch4_values: List[np.ndarray] = []
    tmax_ch4_values: List[np.ndarray] = []
    max_ch1_values: List[np.ndarray] = []
    tmax_ch0_values: List[np.ndarray] = []
    tmax_ch1_values: List[np.ndarray] = []
    ch2_n_fit_points_values: List[np.ndarray] = []
    ch3_n_fit_points_values: List[np.ndarray] = []
    ch2_tanh_p0_values: List[np.ndarray] = []
    ch3_tanh_p0_values: List[np.ndarray] = []
    ch3_tanh_p1_values: List[np.ndarray] = []
    ch3_ped_mean_values: List[np.ndarray] = []
    ch3_min_values: List[np.ndarray] = []
    time_mpl_values: List[np.ndarray] = []
    run_event_counts: List[int] = []

    for base_name in base_names:
        feats = tr._read_required_features_for_run(base_name)
        max_values.append(feats["ch0_max_ch0"])
        ch0_min_values.append(feats["ch0_ch0_min"])
        max_ch5_values.append(feats["ch5_max_ch5"])
        ch0_ped_mean_values.append(feats["ch0_ch0ped_mean"])
        ch1_ped_mean_values.append(feats["ch1_ch1ped_mean"])
        ch1_min_values.append(feats["ch1_ch1_min"])
        max_ch4_values.append(feats["ch4_max_ch4"])
        tmax_ch4_values.append(feats["ch4_tmax_ch4"])
        max_ch1_values.append(feats["ch1_max_ch1"])
        tmax_ch0_values.append(feats["ch0_tmax_ch0"])
        tmax_ch1_values.append(feats["ch1_tmax_ch1"])
        ch2_n_fit_points_values.append(feats["ch2_n_fit_points"])
        ch3_n_fit_points_values.append(feats["ch3_n_fit_points"])
        ch2_tanh_p0_values.append(feats["ch2_tanh_p0"])
        ch3_tanh_p0_values.append(feats["ch3_tanh_p0"])
        ch3_tanh_p1_values.append(feats["ch3_tanh_p1"])
        ch3_ped_mean_values.append(feats["ch3_ch3ped_mean"])
        ch3_min_values.append(feats["ch3_min_ch3"])
        time_mpl_values.append(
            tr._datetime64_ns_to_mpl_date(tr._read_event_time_datetime64_ns_from_ch03(base_name))
        )
        run_event_counts.append(int(feats["ch0_max_ch0"].shape[0]))

    return {
        "max_ch0": np.concatenate(max_values, axis=0),
        "ch0_min": np.concatenate(ch0_min_values, axis=0),
        "max_ch5": np.concatenate(max_ch5_values, axis=0),
        "ch0_ped_mean": np.concatenate(ch0_ped_mean_values, axis=0),
        "ch1_ped_mean": np.concatenate(ch1_ped_mean_values, axis=0),
        "ch1_min": np.concatenate(ch1_min_values, axis=0),
        "max_ch4": np.concatenate(max_ch4_values, axis=0),
        "tmax_ch4": np.concatenate(tmax_ch4_values, axis=0),
        "max_ch1": np.concatenate(max_ch1_values, axis=0),
        "tmax_ch0": np.concatenate(tmax_ch0_values, axis=0),
        "tmax_ch1": np.concatenate(tmax_ch1_values, axis=0),
        "ch2_n_fit_points": np.concatenate(ch2_n_fit_points_values, axis=0),
        "ch3_n_fit_points": np.concatenate(ch3_n_fit_points_values, axis=0),
        "ch2_tanh_p0": np.concatenate(ch2_tanh_p0_values, axis=0),
        "ch3_tanh_p0": np.concatenate(ch3_tanh_p0_values, axis=0),
        "ch3_tanh_p1": np.concatenate(ch3_tanh_p1_values, axis=0),
        "ch3_ped_mean": np.concatenate(ch3_ped_mean_values, axis=0),
        "ch3_min": np.concatenate(ch3_min_values, axis=0),
        "time_mpl": np.concatenate(time_mpl_values, axis=0),
        "run_event_counts": np.asarray(run_event_counts, dtype=np.int64),
    }

def _report_efficiency(
    max_ch0_all: np.ndarray,
    cut_steps: List[Tuple[str, np.ndarray]],
    n_bins: int = 200,
    e_min: float = 0.01,
    e_max: float = 2.0,) -> None:
    """绘制每一步累计 cut 相对 basic 的逐 bin 效率 N_last/N_basic 及 Wilson 误差棒。"""
    if not cut_steps:
        return
    n_total = max_ch0_all.shape[0]
    m0 = np.asarray(cut_steps[0][1], dtype=bool).ravel()
    if m0.size != n_total:
        raise ValueError("basic mask 长度与 max_ch0_all 不一致。")
    if int(m0.sum()) <= 0:
        print("[效率] basic 样本数为 0，无法绘图。")
        return

    bin_edges, _, _ = _compute_spectrum_rate_and_err(
        max_ch0_all, m0, n_bins=n_bins, e_min=e_min, e_max=e_max
    )
    centers = bin_edges[:-1] + 0.5 * np.diff(bin_edges)

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"]})
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    cumulative = np.ones(n_total, dtype=bool)
    for i, (name, m) in enumerate(cut_steps):
        mask = np.asarray(m, dtype=bool).ravel()
        if mask.size != n_total:
            raise ValueError(f"cut_steps[{i}] mask 长度与 max_ch0_all 不一致。")
        cumulative &= mask
        if int(cumulative.sum()) <= 0:
            continue
        p_hat, y_down, y_up = _efficiency_binomial_wilson(
            max_ch0_all,
            m0,
            cumulative,
            bin_edges=bin_edges,
            z=WILSON_Z,
        )

        color = _FIXED_COLORS[i % len(_FIXED_COLORS)]
        label = "basic" if i == 0 else f"above+{name}"
        ax.errorbar(
            centers,
            p_hat,
            yerr=np.vstack([y_down, y_up]),
            fmt="o",
            linestyle="none",
            color=color,
            markersize=2,
            capsize=1,
            label=label,
        )

    ax.set_xlabel("Energy (keV)", fontsize=16)
    ax.set_ylabel("Efficiency (N_pass / N_basic)", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc="best")
    ax.set_xlim(e_min, e_max)
    fig.tight_layout()
    plt.show()

def _report_last_efficiency_atanh_fit(
    max_ch0_all: np.ndarray,
    cut_steps: List[Tuple[str, np.ndarray]],
    n_bins: int = 40,
    e_min: float = 0.01,
    e_max: float = 0.5,) -> None:
    """
    绘制 cut_steps 中最后一步（即累计到最后一步）的效率曲线，并用：
        y(x) = a * tanh(b * (x - c)) + d
    做非线性拟合。

    这里的效率定义与 `_report_efficiency` 一致：
        efficiency = N_last / N_basic（每能量箱二项比例 + Wilson 区间）。
    """
    if not cut_steps:
        return

    n_total = max_ch0_all.shape[0]

    m0 = np.asarray(cut_steps[0][1], dtype=bool).ravel()
    if m0.size != n_total:
        raise ValueError("basic mask 长度与 max_ch0_all 不一致。")
    if int(m0.sum()) <= 0:
        print("[效率-拟合] basic 样本数为 0，无法拟合。")
        return

    bin_edges, _, _ = _compute_spectrum_rate_and_err(
        max_ch0_all, m0, n_bins=n_bins, e_min=e_min, e_max=e_max
    )
    centers = bin_edges[:-1] + 0.5 * np.diff(bin_edges)

    cumulative = np.ones(n_total, dtype=bool)
    for _, m in cut_steps:
        mask = np.asarray(m, dtype=bool).ravel()
        if mask.size != n_total:
            raise ValueError("cut_steps mask 长度与 max_ch0_all 不一致。")
        cumulative &= mask

    if int(cumulative.sum()) <= 0:
        print("[效率-拟合] 最后一段累计样本数为 0，无法拟合。")
        return

    y, y_down, y_up = _efficiency_binomial_wilson(
        max_ch0_all,
        m0,
        cumulative,
        bin_edges=bin_edges,
        z=WILSON_Z,
    )
    x = centers
    # curve_fit 需要对称 sigma：取 Wilson 上下半宽平均作为权重
    sigma_sym = 0.5 * (y_down + y_up)

    fit_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma_sym)
    fit_mask &= (y >= 0.0) & (y <= 1.0)

    n_fit = int(fit_mask.sum())
    if n_fit < 4:
        print(
            f"[效率-拟合] 有效拟合点不足（{n_fit}）。"
            f"y(min..max)={np.nanmin(y):.3g}..{np.nanmax(y):.3g}，"
            f"请检查 cut_steps 最后一步是否足够统计。"
        )
    else:
        print(
            f"[效率-拟合] 有效拟合点 {n_fit}/{len(x)}，"
            f"y范围={float(np.nanmin(y[fit_mask])):.3g}..{float(np.nanmax(y[fit_mask])):.3g}"
        )

    def _model_nonneg_d_s_b_c(
        xv: np.ndarray, d: float, s: float, b: float, c: float
    ) -> np.ndarray:
        """
        非负约束版本：
            y(x) = d * (1 + s * tanh(b * (x - c)))
        若 d >= 0 且 s in [-1, 1]，则对所有 x 都有 y(x) >= 0（理论上）。
        """
        return d * (1.0 + s * np.tanh(b * (xv - c)))

    popt = None

    # 风险较小的初值策略（基于 tanh 的形状）：
    # - d0 取效率的中位数（模型中 y(c)=d）
    # - c0 取效率最接近 d0 的能量点
    # - a0 取数据上下饱和幅度的半差（带符号）
    # - b0 由 c0 附近的局部斜率（线性近似：slope|c ≈ a*b）
    if n_fit >= 4:
        xf = x[fit_mask]
        yf = y[fit_mask]

        y_min = float(np.nanmin(yf))
        y_max = float(np.nanmax(yf))
        d0 = float(np.nanmedian(yf))
        c0 = float(xf[np.argmin(np.abs(yf - d0))])

        # a 的符号：看两端相对 d0 的偏移谁更大
        a_pos = y_max - d0
        a_neg = d0 - y_min
        a0 = float(a_pos if abs(a_pos) >= abs(a_neg) else -a_neg)
        if abs(a0) < 1e-6:
            a0 = 1e-3

        # 局部斜率：在 c0 附近挑几个点做一次一元线性拟合
        span = 0.08 * (e_max - e_min)  # 经验值；能量范围越宽，窗口越大
        close = np.abs(xf - c0) <= max(span, 1e-6)
        if int(np.sum(close)) >= 2:
            xs = xf[close]
            ys = yf[close]
            # dy/dx 的线性回归
            slope = float(np.polyfit(xs, ys, deg=1)[0])  # ≈ a*b
            b0 = slope / a0
        else:
            b0 = 0.0

        # b 的数量级：让过渡区间落在能量跨度内，避免参数高度相关
        b_scale = 1.0 / max(e_max - e_min, 1e-9)
        b_bound = 50.0 * b_scale

        # Re-parameterize: a = d * s
        #   y = d + a*tanh(...) = d * (1 + s*tanh(...))
        d0 = max(d0, 1e-12)  # 保证 s0 不会除以 0
        s0 = float(np.clip(a0 / d0, -1.0, 1.0))

        d_pad = 0.2 * (y_max - y_min + 1e-9)
        d_low = 0.0
        d_high = max(y_max + d_pad, 1e-6)

        p0 = [d0, s0, b0, c0]
        bounds_lower = [d_low, -1.0, -b_bound, e_min]
        bounds_upper = [d_high, 1.0, b_bound, e_max]

        print(
            "[效率-拟合] p0: "
            f"d0={p0[0]:.4g}, s0={p0[1]:.4g}, b0={p0[2]:.4g}, c0={p0[3]:.4g}, "
            f"d_high={d_high:.4g}, b_bound={b_bound:.4g}"
        )

        sigma = sigma_sym[fit_mask].astype(np.float64, copy=False)
        sigma_pos = sigma[np.isfinite(sigma) & (sigma > 0)]
        sigma_med = float(np.nanmedian(sigma_pos)) if sigma_pos.size > 0 else None
        if sigma_med is None:
            print("[效率-拟合] Wilson 半宽没有有效的正值，改用无权重拟合。")
            try:
                popt, _ = curve_fit(
                    _model_nonneg_d_s_b_c,
                    xf,
                    yf,
                    p0=p0,
                    bounds=(bounds_lower, bounds_upper),
                    maxfev=200000,
                )
            except Exception as e:
                popt = None
                print(f"[效率-拟合] curve_fit(无权重) 失败：{e}")
        else:
            # sigma <=0 统一用中位数替代，避免异常权重
            sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, sigma_med)
            try:
                popt, _ = curve_fit(
                    _model_nonneg_d_s_b_c,
                    xf,
                    yf,
                    p0=p0,
                    sigma=sigma,
                    absolute_sigma=True,
                    bounds=(bounds_lower, bounds_upper),
                    maxfev=200000,
                )
            except Exception as e:
                print(f"[效率-拟合] curve_fit(加权) 失败：{e}，尝试无权重再拟合一次。")
                popt = None
                try:
                    popt, _ = curve_fit(
                        _model_nonneg_d_s_b_c,
                        xf,
                        yf,
                        p0=p0,
                        bounds=(bounds_lower, bounds_upper),
                        maxfev=200000,
                    )
                except Exception as e2:
                    popt = None
                    print(f"[效率-拟合] curve_fit(无权重) 也失败：{e2}")

        if popt is not None:
            d_fit = float(popt[0])
            s_fit = float(popt[1])
            b_fit = float(popt[2])
            c_fit = float(popt[3])
            a_fit = d_fit * s_fit
            print(
                f"[效率-拟合] a*tanh(b*(x-c))+d: a={a_fit:.6g}, b={b_fit:.6g}, c={c_fit:.6g}, d={d_fit:.6g}"
            )

    # 画图（包含数据点与拟合曲线）
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"]})
    fixed_color = _FIXED_COLORS[0]
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    if n_fit > 0:
        y_fit = y[fit_mask]
        ax.errorbar(
            x[fit_mask],
            y_fit,
            yerr=np.vstack([y_down[fit_mask], y_up[fit_mask]]),
            fmt="o",
            linestyle="none",
            markersize=3,
            capsize=2,
            color=fixed_color,
            label="efficiency",
        )

    if popt is not None:
        x_line = np.linspace(e_min, e_max, 400)
        d_fit = float(popt[0])
        s_fit = float(popt[1])
        b_fit = float(popt[2])
        c_fit = float(popt[3])
        y_line = _model_nonneg_d_s_b_c(x_line, d_fit, s_fit, b_fit, c_fit)
        ax.plot(x_line, y_line, color=fixed_color, lw=2.0, label="curve_fit")

    ax.set_xlabel("Energy (keV)", fontsize=16)
    ax.set_ylabel("Efficiency", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc="best")
    #ax.set_title("Last cut efficiency fit", fontsize=18)
    ax.set_xlim(e_min, e_max)
    fig.tight_layout()
    plt.show()

def main() -> None:
    base_names = tr._list_base_names_from_ch0()
    data = _load_required_arrays(base_names)

    max_ch0_all = data["max_ch0"]
    ch0_min_all = data["ch0_min"]
    max_ch5_all = data["max_ch5"]
    ch0_ped_mean_all = data["ch0_ped_mean"]
    ch1_ped_mean_all = data["ch1_ped_mean"]
    ch1_min_all = data["ch1_min"]
    max_ch4_all = data["max_ch4"]
    tmax_ch4_all = data["tmax_ch4"]
    max_ch1_all = data["max_ch1"]
    tmax_ch0_all = data["tmax_ch0"]
    tmax_ch1_all = data["tmax_ch1"]
    ch2_n_fit_points_all = data["ch2_n_fit_points"]
    ch3_n_fit_points_all = data["ch3_n_fit_points"]
    ch2_tanh_p0_all = data["ch2_tanh_p0"]
    ch3_tanh_p0_all = data["ch3_tanh_p0"]
    ch3_tanh_p1_all = data["ch3_tanh_p1"]
    ch3_ped_mean_all = data["ch3_ped_mean"]
    ch3_min_all = data["ch3_min"]
    time_mpl_all = data["time_mpl"]

    # 与 tradition.py 一致的各项 cut
    m_fit_ok = tr.cut_fit_success(
        ch2_n_fit_points_all,
        ch3_n_fit_points_all,
        ch2_tanh_p0_all,
        ch3_tanh_p0_all,
    )
    m_ch0_min = tr.cut_ch0_min_positive(ch0_min_all)
    m_ch0_sat = tr.cut_ch0_max_saturation(max_ch0_all, max_ch1_all)
    m_ch5_rt = tr.cut_ch5_self_trigger(max_ch5_all)
    m_ped = tr.cut_pedestal_3sigma(ch0_ped_mean_all, ch1_ped_mean_all, max_ch5_all)
    m_acv = tr.cut_acv(max_ch4_all, tmax_ch4_all)
    m_mincut = tr.cut_mincut(ch0_min_all, ch1_min_all, max_ch4_all, tmax_ch4_all)
    m_ch3ped_min = tr.cut_ch3ped_min(ch3_ped_mean_all, ch3_min_all)
    
    m_bscut = tr.cut_bscut(ch3_tanh_p1_all)

    base_after_pre = ~m_acv & m_fit_ok & m_ch0_min & m_ch0_sat & m_ch5_rt & m_ped & m_mincut & m_ch3ped_min
    m_time = tr.cut_time(
        time_mpl_all,
        max_ch0=max_ch0_all,
        pre_mask=base_after_pre,
    )

    m_pn = tr.cut_pncut(base_after_pre, max_ch0_all, max_ch1_all)
    # 跟 traditionACT.py 的 cut_steps 对齐：仅保留 4 个阶段
    # - 第 1 步：ACT 约束 -> ~m_acv
    # - 第 2 步：basic_cut
    # - 第 3 步：event_cut
    # - 第 4 步：pncut（用于估计阈值的拟合基准按 ACV 侧构造）
    act_mask = ~m_acv & m_time & m_ch0_min & m_ch0_sat & m_ch5_rt & m_fit_ok & m_ped & m_mincut & m_pn & m_bscut
    
    cut_steps = [
        ("ACT", act_mask),
        #("pncut", m_pn),
        ("ch3pedmin", m_ch3ped_min),
        #("bscut", m_bscut),
    ]

    # tr.plot_cumulative_cut_spectra(
    #     max_ch0_all=max_ch0_all,
    #     cut_steps=cut_steps,
    #     n_bins=200,
    #     e_min=0.01,
    #     e_max=2.0,
    # )
    # _report_efficiency(max_ch0_all, cut_steps, n_bins=40, e_min=0.01, e_max=0.5)
    # 拟合曲线分箱加密：n_bins 越大，效率点越密
    _report_last_efficiency_atanh_fit(max_ch0_all, cut_steps, n_bins=120, e_min=0.01, e_max=0.5)

    # # 查看 m_ch0_t 与 m_ch1_t 差异部分的 CH0/CH3 原始波形
    # tr.plot_waveforms_from_mask_diff(
    #     mask_a=base_after_pre&m_pn&m_ch0_t,
    #     mask_b=base_after_pre&m_pn&m_ch1_t,
    #     base_names=base_names,
    #     run_event_counts=data["run_event_counts"].tolist(),
    #     ch0_3_dir=tr.DATA_ROOT / "CH0-3",
    #     n_show=9,
    #     random_state=42,
    #     sampling_interval_ns=4.0,
    #     alpha=0.2,
    # )


if __name__ == "__main__":
    main()
