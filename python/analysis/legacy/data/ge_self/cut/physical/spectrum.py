#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对 basic+acv 中与 cuts_distribution / tradition 一致的 cut 组合绘制能谱。

交互：左侧复选框勾选若干 cut，图中显示「同时通过所有已勾选 cut」事例的能谱；选项名与 _CUT_KEYS_SPECTRUM 一致。
掩码与 maxch0maxch1/distinguish/cuts_distribution._build_cut_pass_fns 一致：
fit_success → cut_time（坏时段由 m_pre_m6 & pncut & ch3ped_min 自算）→ … → pncut（base 为 m0&…&m6）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons


def _load_basic_act_module() -> Tuple[object, Path]:
    """从 ge-self/cut 的 basic+act.py 动态加载模块（若无则 basic+acv.py）。返回 (模块, cut 目录)。"""
    current_dir = Path(__file__).resolve().parent.parent  # .../cut
    basic_act_path = current_dir / "basic+act.py"
    if not basic_act_path.is_file():
        alt = current_dir / "basic+acv.py"
        if alt.is_file():
            basic_act_path = alt
        else:
            raise FileNotFoundError(f"未找到文件: {current_dir / 'basic+act.py'}")

    spec = importlib.util.spec_from_file_location("basic_act_module", str(basic_act_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {basic_act_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["basic_act_module"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "cut_act") and hasattr(module, "cut_acv"):
        module.cut_act = module.cut_acv
    return module, current_dir


# 与 cuts_distribution._build_cut_pass_fns 返回字典键顺序一致；复选框显示名称与此列表一致
_CUT_KEYS_SPECTRUM: List[str] = [
    "fit_success",
    "cut_time",
    "ch0_min_pos",
    "ch0_max_sat",
    "ch5_rt",
    "ch3ped_min",
    "ped_3sigma",
    "act",
    "mincut",
    "pncut",
]

# 能量刻度系数（与 basic+acv.py 中 _plot_ch0max_hist_passing_cuts 一致）
_E_CAL_A = 0.0008432447500464594
_E_CAL_B = -0.826976770117076
EXPOSURE_KG = 0.5
EXPOSURE_DAYS = 20.0
N_BINS = 500


def _load_bundle_one_run(
    ch0_path: Path,
    ch5_path: Path,
    ba: Any,
) -> Optional[Dict[str, np.ndarray]]:
    """与 cuts_distribution.load_cut_bundle_one_run 对齐。"""
    try:
        (
            max_ch0,
            ch0_min,
            max_ch5,
            ch0_ped_mean,
            ch1_ped_mean,
            max_ch1,
            ch1_min,
            max_ch4,
            tmax_ch4,
            _tmax_ch0,
            _tmax_ch1,
        ) = ba._load_basic_features_for_run(ch0_path, ch5_path)
    except (KeyError, OSError) as e:
        print(f"[跳过 {ch0_path.name}] 基准特征读取失败: {e}")
        return None

    n = int(max_ch0.shape[0])
    ch2_n, ch3_n, ch2_p0, ch3_p0 = ba._load_ch2_ch3_fit_quality_aligned(ch0_path, n)
    ch3ped_mean, min_ch3 = ba._load_ch3_ped_min_aligned(ch0_path, n)

    try:
        time_ns = ba.read_event_time_datetime64_ns_from_ch03(ch0_path)
        time_ns = np.asarray(time_ns).reshape(-1)
        nt = min(n, time_ns.size)
        time_mpl = ba.datetime64_ns_to_mpl_date(time_ns[:nt])
        if nt < n:
            time_mpl = np.pad(time_mpl, (0, n - nt), constant_values=np.nan)
    except Exception:
        time_mpl = np.full(n, np.nan, dtype=np.float64)

    return {
        "max_ch0": max_ch0,
        "max_ch1": max_ch1,
        "ch0_min": ch0_min,
        "ch1_min": ch1_min,
        "max_ch5": max_ch5,
        "ch0_ped_mean": ch0_ped_mean,
        "ch1_ped_mean": ch1_ped_mean,
        "max_ch4": max_ch4,
        "tmax_ch4": tmax_ch4,
        "ch2_n_fit_points": ch2_n,
        "ch3_n_fit_points": ch3_n,
        "ch2_tanh_p0": ch2_p0,
        "ch3_tanh_p0": ch3_p0,
        "ch3ped_mean": ch3ped_mean,
        "min_ch3": min_ch3,
        "time_mpl": time_mpl,
    }


def _concatenate_bundles(runs: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if not runs:
        raise ValueError("无有效 run")
    keys = runs[0].keys()
    out: Dict[str, np.ndarray] = {}
    for k in keys:
        out[k] = np.concatenate([r[k] for r in runs], axis=0)
    return out


def _build_cut_masks_aligned_with_cuts_distribution(
    ba: Any,
    bundle: Dict[str, np.ndarray],
) -> List[np.ndarray]:
    """
    与 cuts_distribution._build_cut_pass_fns 中预计算量一致：
    m_pre_m6 = m0&…&m6，pncut(base=m_pre_m6)，ch3ped=m11，坏时段由 m_pre_m6&m8&m11 与 ch0-time 规则生成。
    """
    d0 = bundle
    m0 = ba.cut_fit_success(
        d0["ch2_n_fit_points"],
        d0["ch3_n_fit_points"],
        d0["ch2_tanh_p0"],
        d0["ch3_tanh_p0"],
    )
    m1 = ba.cut_ch0_min_positive(d0["ch0_min"])
    m2 = ba.cut_ch0_max_saturation(d0["max_ch0"], d0["max_ch1"])
    m3 = ba.cut_ch5_self_trigger(d0["max_ch5"])
    m4 = ba.cut_pedestal_3sigma(
        d0["ch0_ped_mean"], d0["ch1_ped_mean"], d0["max_ch5"]
    )
    m5 = ba.cut_act(d0["max_ch4"], d0["tmax_ch4"])
    m6 = ba.cut_mincut(
        d0["ch0_min"],
        d0["ch1_min"],
        d0["max_ch4"],
        d0["tmax_ch4"],
    )
    m_pre_m6 = m0 & m1 & m2 & m3 & m4 & m5 & m6
    m8 = ba.cut_pncut(m_pre_m6, d0["max_ch0"], d0["max_ch1"])
    m11 = ba.cut_ch3ped_min(d0["ch3ped_mean"], d0["min_ch3"])
    mask_pre_time = m_pre_m6 & m8 & m11
    bad_intervals_mpl = ba.build_ch0_time_exclude_intervals_global(
        d0["time_mpl"],
        d0["max_ch0"],
        pre_mask=mask_pre_time,
        rate_threshold=ba.CH0_TIME_BAND_BURST_RATE_THRESHOLD,
    )
    mask_time = ba.cut_time(d0["time_mpl"], bad_intervals=bad_intervals_mpl)

    # 键顺序与 _CUT_KEYS_SPECTRUM / pass_fns 一致
    return [
        m0,
        mask_time,
        m1,
        m2,
        m3,
        m11,
        m4,
        m5,
        m6,
        m8,
    ]


def _compute_spectrum(max_ch0: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """对给定 mask 下的 max_ch0 计算能谱 (bin_edges, rates)。"""
    x = max_ch0[mask]
    energy_values = _E_CAL_A * x + _E_CAL_B

    counts, bin_edges = np.histogram(energy_values, bins=N_BINS)
    bin_widths = np.diff(bin_edges)
    bin_widths[bin_widths == 0] = np.inf
    denom = EXPOSURE_KG * bin_widths * EXPOSURE_DAYS
    rates = counts / denom

    return bin_edges, rates


def _plot_interactive_spectrum(
    max_ch0: np.ndarray,
    masks_ordered: List[np.ndarray],
    check_labels: List[str],
) -> None:
    """布局与 maxch0maxch1/distinguish/cuts_distribution.py 一致。"""
    plt.rcParams.setdefault("axes.unicode_minus", False)

    n_raw = int(max_ch0.shape[0])
    if len(check_labels) != len(masks_ordered):
        raise ValueError("check_labels 与 masks_ordered 长度须一致")

    fig = plt.figure(figsize=(10, 6))
    fig.subplots_adjust(left=0.08, bottom=0.10, right=0.98, top=0.92)
    ax = fig.add_axes([0.22, 0.12, 0.76, 0.80])

    rax = fig.add_axes([0.015, 0.14, 0.185, 0.76])
    rax.set_facecolor("none")
    rax.patch.set_visible(False)
    for s in rax.spines.values():
        s.set_visible(False)

    check = CheckButtons(
        rax,
        check_labels,
        actives=[False] * len(check_labels),
        label_props={"fontsize": [6]},
        frame_props={"s": 12},
        check_props={"s": 12, "linewidths": 0.65},
    )

    ax.set_yscale("log")
    ax.set_xlabel("Energy (keV)", fontsize=16)
    ax.set_ylabel(r"Rate [counts / (keV·kg·day)]", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=12)
    #ax.grid(True, alpha=0.3)
    ax.set_xlim(0.1, 12.0)

    # 对数纵轴下 stairs 填充不能以 0 为底，需正的下限
    _log_floor = 1

    spectrum_artist = None

    def _combined_mask() -> np.ndarray:
        vis = np.ones(n_raw, dtype=bool)
        status = check.get_status()
        for i, on in enumerate(status):
            if on:
                vis &= masks_ordered[i]
        return vis

    def _apply() -> None:
        nonlocal spectrum_artist
        mask = _combined_mask()
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        if spectrum_artist is not None:
            spectrum_artist.remove()
            spectrum_artist = None

        bin_edges, rates = _compute_spectrum(max_ch0, mask)
        n_pass = int(mask.sum())
        print(f"当前剩余事例总数: {n_pass}")
        rates_plot = np.maximum(rates, _log_floor)

        spectrum_artist = ax.stairs(
            rates_plot,
            bin_edges,
            orientation="vertical",
            fill=True,
            color="C0",
            alpha=0.45,
            edgecolor="C0",
            linewidth=0.9,
            baseline=_log_floor * 0.1,
            label=f"spectrum (N={n_pass})",
            zorder=3,
        )
        #ax.legend(loc="best", fontsize=12)
        fig.canvas.draw_idle()

    def on_check(_label: str) -> None:
        _apply()

    check.on_clicked(on_check)
    _apply()
    plt.show()


if __name__ == "__main__":
    basic_act, _ = _load_basic_act_module()
    check_labels = list(_CUT_KEYS_SPECTRUM)

    pairs = basic_act._list_paired_param_files()
    print(f"找到 {len(pairs)} 个可配对的参数文件。")
    if not pairs:
        raise RuntimeError("未找到任何可配对参数文件。")

    bundles: List[Dict[str, np.ndarray]] = []
    for ch0_path, ch5_path in pairs:
        b = _load_bundle_one_run(ch0_path, ch5_path, basic_act)
        if b is not None:
            bundles.append(b)
            print(f"已读 {ch0_path.name} | N={b['max_ch0'].shape[0]}")
    if not bundles:
        raise RuntimeError("没有成功加载任何 run")

    bundle = _concatenate_bundles(bundles)
    max_ch0 = bundle["max_ch0"]

    n_raw = max_ch0.shape[0]
    print(f"原始事件数: {n_raw}")

    masks_ordered = _build_cut_masks_aligned_with_cuts_distribution(basic_act, bundle)

    _plot_interactive_spectrum(max_ch0, masks_ordered, check_labels)
