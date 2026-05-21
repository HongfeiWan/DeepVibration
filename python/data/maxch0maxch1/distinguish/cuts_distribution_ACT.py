#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CH0max–CH1max 散点：交互式勾选 cut 过滤显示，并叠加 PNcut 拟合线与带宽。

要求（按用户约定）：
- 删除所有与 basic+act.py / basic+acv.py 相关的代码
- 所有 cut 逻辑与参数读取均来自
  python/data/ge-self/cut/parameterize/tradition/traditionACT.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons


#
# PNcut 拟合/可视化共享参数（只改这里即可同步全脚本）
#
PNCUT_FIT_CH0_MIN: float = 1000.0
PNCUT_FIT_CH0_MAX: float = 16000.0
PNCUT_BAND_N_SIGMA: float = 1.0


def _pncut_fit_kwargs() -> Dict[str, float]:
    """与 traditionACT.cut_pncut(..., fit_ch0_min, fit_ch0_max, n_sigma) 对齐，供真实 mask 与可视化共用。"""
    return {
        "fit_ch0_min": float(PNCUT_FIT_CH0_MIN),
        "fit_ch0_max": float(PNCUT_FIT_CH0_MAX),
        "n_sigma": float(PNCUT_BAND_N_SIGMA),
    }


def _project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    return script_dir.parents[3]


def _load_tradition_act_module(project_root: Path):
    """
    动态加载 traditionACT.py（路径包含 ge-self，无法常规包导入）。
    """
    ta_path = (
        project_root
        / "python"
        / "data"
        / "ge-self"
        / "cut"
        / "parameterize"
        / "tradition"
        / "traditionACT.py"
    )
    if not ta_path.is_file():
        raise FileNotFoundError(f"未找到 traditionACT.py: {ta_path}")
    spec = importlib.util.spec_from_file_location("traditionACT_mod", str(ta_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载: {ta_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["traditionACT_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_cut_bundle_one_run(base_name: str, ta: Any) -> Optional[Dict[str, np.ndarray]]:
    """
    读取一个 run 的最小数组集合，并统一键名供后续 cut 使用。
    """
    try:
        feats = ta._read_required_features_for_run(base_name)
    except Exception as e:
        print(f"[跳过 {base_name}] 参数读取失败: {e}")
        return None

    # traditionACT.py 内的 key 命名为 ch{ch}_<dataset>
    max_ch0 = np.asarray(feats["ch0_max_ch0"], dtype=np.float64)
    max_ch1 = np.asarray(feats["ch1_max_ch1"], dtype=np.float64)
    ch0_min = np.asarray(feats["ch0_ch0_min"], dtype=np.float64)
    ch1_min = np.asarray(feats["ch1_ch1_min"], dtype=np.float64)
    max_ch5 = np.asarray(feats["ch5_max_ch5"], dtype=np.float64)
    ch0_ped_mean = np.asarray(feats["ch0_ch0ped_mean"], dtype=np.float64)
    ch1_ped_mean = np.asarray(feats["ch1_ch1ped_mean"], dtype=np.float64)
    max_ch4 = np.asarray(feats["ch4_max_ch4"], dtype=np.float64)
    tmax_ch4 = np.asarray(feats["ch4_tmax_ch4"], dtype=np.float64)
    ch2_n_fit_points = np.asarray(feats["ch2_n_fit_points"], dtype=np.float64)
    ch3_n_fit_points = np.asarray(feats["ch3_n_fit_points"], dtype=np.float64)
    ch2_tanh_p0 = np.asarray(feats["ch2_tanh_p0"], dtype=np.float64)
    ch3_tanh_p0 = np.asarray(feats["ch3_tanh_p0"], dtype=np.float64)
    ch3_tanh_p1 = np.asarray(feats["ch3_tanh_p1"], dtype=np.float64)
    ch3ped_mean = np.asarray(feats["ch3_ch3ped_mean"], dtype=np.float64)
    min_ch3 = np.asarray(feats["ch3_min_ch3"], dtype=np.float64)

    n = int(
        min(
            max_ch0.size,
            max_ch1.size,
            ch0_min.size,
            ch1_min.size,
            max_ch5.size,
            ch0_ped_mean.size,
            ch1_ped_mean.size,
            max_ch4.size,
            tmax_ch4.size,
            ch2_n_fit_points.size,
            ch3_n_fit_points.size,
            ch2_tanh_p0.size,
            ch3_tanh_p0.size,
            ch3_tanh_p1.size,
            ch3ped_mean.size,
            min_ch3.size,
        )
    )
    if n <= 0:
        return None

    max_ch0 = max_ch0[:n]
    max_ch1 = max_ch1[:n]
    ch0_min = ch0_min[:n]
    ch1_min = ch1_min[:n]
    max_ch5 = max_ch5[:n]
    ch0_ped_mean = ch0_ped_mean[:n]
    ch1_ped_mean = ch1_ped_mean[:n]
    max_ch4 = max_ch4[:n]
    tmax_ch4 = tmax_ch4[:n]
    ch2_n_fit_points = ch2_n_fit_points[:n]
    ch3_n_fit_points = ch3_n_fit_points[:n]
    ch2_tanh_p0 = ch2_tanh_p0[:n]
    ch3_tanh_p0 = ch3_tanh_p0[:n]
    ch3_tanh_p1 = ch3_tanh_p1[:n]
    ch3ped_mean = ch3ped_mean[:n]
    min_ch3 = min_ch3[:n]

    # 时间
    try:
        time_ns = ta._read_event_time_datetime64_ns_from_ch03(base_name, ta.EPOCH_OFFSET_DEFAULT)
        time_mpl = ta._datetime64_ns_to_mpl_date(time_ns)
        time_mpl = np.asarray(time_mpl, dtype=np.float64).reshape(-1)
        nt = min(n, time_mpl.size)
        time_mpl = time_mpl[:nt]
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
        "ch2_n_fit_points": ch2_n_fit_points,
        "ch3_n_fit_points": ch3_n_fit_points,
        "ch2_tanh_p0": ch2_tanh_p0,
        "ch3_tanh_p0": ch3_tanh_p0,
        "ch3_tanh_p1": ch3_tanh_p1,
        "ch3ped_mean": ch3ped_mean,
        "min_ch3": min_ch3,
        "time_mpl": time_mpl,
    }


def _concat_runs(runs: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if not runs:
        raise ValueError("无有效 run")
    keys = list(runs[0].keys())
    return {k: np.concatenate([r[k] for r in runs], axis=0) for k in keys}


def load_cut_bundle(
    project_root: Path,
    base_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, np.ndarray], Any]:
    ta = _load_tradition_act_module(project_root)
    if base_names is None:
        base_names = ta._list_base_names_from_ch0()
    bundles: List[Dict[str, np.ndarray]] = []
    for bn in base_names:
        b = _load_cut_bundle_one_run(bn, ta)
        if b is not None:
            bundles.append(b)
            print(f"已读 {bn} | N={b['max_ch0'].shape[0]}")
    if not bundles:
        raise RuntimeError("没有成功加载任何 run")
    return _concat_runs(bundles), ta


def _build_cut_pass_fns(
    ta: Any,
    bundle: Dict[str, np.ndarray],
) -> Dict[str, Callable[[Dict[str, np.ndarray]], np.ndarray]]:
    """
    返回每个 cut 的 pass 函数（True=通过/保留）。
    其中 cut_time 使用 traditionACT.cut_time 自算坏区间，再用 bad_intervals 过滤。
    """
    d0 = bundle

    m_fit_ok = ta.cut_fit_success(
        d0["ch2_n_fit_points"],
        d0["ch3_n_fit_points"],
        d0["ch2_tanh_p0"],
        d0["ch3_tanh_p0"],
    )
    m_ch0_min = ta.cut_ch0_min_positive(d0["ch0_min"])
    m_ch0_sat = ta.cut_ch0_max_saturation(d0["max_ch0"], d0["max_ch1"])
    m_ch5_rt = ta.cut_ch5_self_trigger(d0["max_ch5"])
    m_ped = ta.cut_pedestal_3sigma(d0["ch0_ped_mean"], d0["ch1_ped_mean"], d0["max_ch5"])
    m_acv = ta.cut_acv(d0["max_ch4"], d0["tmax_ch4"])
    m_mincut = ta.cut_mincut(d0["ch0_min"], d0["ch1_min"], d0["max_ch4"], d0["tmax_ch4"])
    m_ch3ped_min = ta.cut_ch3ped_min(d0["ch3ped_mean"], d0["min_ch3"])
    m_bscut = ta.cut_bscut(d0["ch3_tanh_p1"])

    # traditionACT.py：坏时间段统计使用 ~m_acv 侧
    base_after_pre = (
        (~m_acv)
        & m_fit_ok
        & m_ch0_min
        & m_ch0_sat
        & m_ch5_rt
        & m_ped
        & m_mincut
        & m_ch3ped_min
    )

    _, cut_time_intervals = ta.cut_time(
        d0["time_mpl"],
        bad_intervals=None,
        max_ch0=d0["max_ch0"],
        pre_mask=base_after_pre,
        rate_threshold=float(ta.CH0_TIME_BAND_BURST_RATE_THRESHOLD),
        return_intervals=True,
    )
    m_time = ta.cut_time(d0["time_mpl"], bad_intervals=cut_time_intervals)

    # traditionACT.py：用于 pncut 的 ACT 基准集合
    act_mask_for_pncut = (
        (~m_acv)
        & m_fit_ok
        & m_ch0_min
        & m_ch0_sat
        & m_ch5_rt
        & m_ped
        & m_mincut
        & m_ch3ped_min
        & m_bscut
    )
    m_pn = ta.cut_pncut(
        act_mask_for_pncut,
        d0["max_ch0"],
        d0["max_ch1"],
        **_pncut_fit_kwargs(),
    )

    def pass_fit_success(d: Dict[str, np.ndarray]) -> np.ndarray:
        return m_fit_ok

    def pass_cut_time(d: Dict[str, np.ndarray]) -> np.ndarray:
        return m_time

    def pass_ch0_min(d: Dict[str, np.ndarray]) -> np.ndarray:
        return m_ch0_min

    def pass_ch0_sat(d: Dict[str, np.ndarray]) -> np.ndarray:
        return m_ch0_sat

    def pass_ch5_rt(d: Dict[str, np.ndarray]) -> np.ndarray:
        return m_ch5_rt

    def pass_ch3ped_min(d: Dict[str, np.ndarray]) -> np.ndarray:
        return m_ch3ped_min

    def pass_ped_3sigma(d: Dict[str, np.ndarray]) -> np.ndarray:
        return m_ped

    def pass_bscut(d: Dict[str, np.ndarray]) -> np.ndarray:
        return m_bscut

    def pass_mincut(d: Dict[str, np.ndarray]) -> np.ndarray:
        return m_mincut

    def pass_pncut(d: Dict[str, np.ndarray]) -> np.ndarray:
        return m_pn

    return {
        "fit_success": pass_fit_success,
        "cut_time": pass_cut_time,
        "ch0_min_pos": pass_ch0_min,
        "ch0_max_sat": pass_ch0_sat,
        "ch5_rt": pass_ch5_rt,
        "ch3ped_min": pass_ch3ped_min,
        "ped_3sigma": pass_ped_3sigma,
        "mincut": pass_mincut,
        "bscut": pass_bscut,
        "pncut": pass_pncut,
    }


def _pncut_fit_line_params(
    bundle: Dict[str, np.ndarray],
    base_mask: np.ndarray,
) -> Optional[Tuple[float, float, float, float]]:
    """
    与 traditionACT.cut_pncut 内部拟合一致：在传入的 base_mask（与 cut_pncut 第一个参数相同）
    且 CH0max∈(PNCUT_FIT_CH0_MIN, PNCUT_FIT_CH0_MAX) 上 polyfit，
    返回 (斜率 a, 截距 b, 残差 σ, n_sigma)，失败时返回 None。
    """
    d0 = bundle
    max_ch0 = d0["max_ch0"]
    max_ch1 = d0["max_ch1"]
    bm = np.asarray(base_mask, dtype=bool).reshape(-1)
    fit_ch0_min = float(PNCUT_FIT_CH0_MIN)
    fit_ch0_max = float(PNCUT_FIT_CH0_MAX)
    min_fit_events = 10
    fit_mask = bm & (max_ch0 > fit_ch0_min) & (max_ch0 < fit_ch0_max)
    x_fit = max_ch0[fit_mask]
    y_fit = max_ch1[fit_mask]
    if x_fit.size < min_fit_events:
        return None
    a, b = np.polyfit(x_fit, y_fit, deg=1)
    y_pred_fit = a * x_fit + b
    resid_fit = y_fit - y_pred_fit
    sigma = float(resid_fit.std(ddof=1))
    if sigma <= 0.0:
        return None
    n_sigma = float(PNCUT_BAND_N_SIGMA)
    return float(a), float(b), float(sigma), float(n_sigma)


CUT_LABELS_ZH: Dict[str, str] = {
    "fit_success": "CH2/CH3 拟合成功",
    "cut_time": "时间 cut（红区暴发排除，与 traditionACT 一致）",
    "ch0_min_pos": "ch0_min>0（非抑制）",
    "ch0_max_sat": "CH0 未饱和",
    "ch5_rt": "CH5 非随机触发",
    "ch3ped_min": "CH3 ped-min 带",
    "ped_3sigma": "前沿基线 3σ",
    "mincut": "mincut（ACT 后 CH0/CH1 min）",
    "bscut": "bscut（CH3 rise_time 范围）",
    "pncut": "PN 线性带",
}


def plot_interactive_cuts_ch0_ch1(
    project_root: Optional[Path] = None,
    base_names: Optional[List[str]] = None,
    *,
    max_points: int = 1000_000,
    seed: int = 0,
    show: bool = True,
) -> None:
    if project_root is None:
        project_root = _project_root()
    bundle, ta = load_cut_bundle(project_root, base_names)

    max_ch0 = bundle["max_ch0"]
    max_ch1 = bundle["max_ch1"]
    n = int(max_ch0.shape[0])
    rng = np.random.default_rng(seed)

    pass_fns = _build_cut_pass_fns(ta, bundle)

    max_ch0 = bundle["max_ch0"]
    m_acv = ta.cut_acv(bundle["max_ch4"], bundle["tmax_ch4"])
    base_mask = ~np.asarray(m_acv, dtype=bool).reshape(-1)[:n]
    base_idx_all = np.where(base_mask)[0]
    if base_idx_all.size == 0:
        idx = np.array([], dtype=np.int64)
        print("[警告] ~m_acv 为空：背景散点将为空。")
    elif base_idx_all.size > max_points:
        idx = rng.choice(base_idx_all, size=max_points, replace=False)
        print(f"背景散点采样(~m_acv): {max_points}/{base_idx_all.size}")
    else:
        idx = base_idx_all

    # PNcut 拟合线：使用 traditionACT.py 中用于 pncut 的 ACT 基准集合（与 pass_pncut 一致）
    m_fit_ok = ta.cut_fit_success(
        bundle["ch2_n_fit_points"],
        bundle["ch3_n_fit_points"],
        bundle["ch2_tanh_p0"],
        bundle["ch3_tanh_p0"],
    )
    m_ch0_min = ta.cut_ch0_min_positive(bundle["ch0_min"])
    m_ch0_sat = ta.cut_ch0_max_saturation(bundle["max_ch0"], bundle["max_ch1"])
    m_ch5_rt = ta.cut_ch5_self_trigger(bundle["max_ch5"])
    m_ped = ta.cut_pedestal_3sigma(
        bundle["ch0_ped_mean"], bundle["ch1_ped_mean"], bundle["max_ch5"]
    )
    m_mincut = ta.cut_mincut(bundle["ch0_min"], bundle["ch1_min"], bundle["max_ch4"], bundle["tmax_ch4"])
    m_ch3ped_min = ta.cut_ch3ped_min(bundle["ch3ped_mean"], bundle["min_ch3"])
    m_bscut = ta.cut_bscut(bundle["ch3_tanh_p1"])
    act_mask_for_pncut = (
        (~m_acv)
        & m_fit_ok
        & m_ch0_min
        & m_ch0_sat
        & m_ch5_rt
        & m_ped
        & m_mincut
        & m_ch3ped_min
        & m_bscut
    )
    pncut_fit = _pncut_fit_line_params(bundle, act_mask_for_pncut)

    plt.rcParams.setdefault("axes.unicode_minus", False)
    fig = plt.figure(figsize=(10, 6))
    fig.subplots_adjust(left=0.08, bottom=0.10, right=0.98, top=0.92)
    ax = fig.add_axes([0.22, 0.12, 0.76, 0.80])

    sc_bg = ax.scatter(
        max_ch0[idx],
        max_ch1[idx],
        s=3,
        alpha=0.4,
        c="blue",
        edgecolors="none",
        rasterized=True,
        label="Events",
    )

    x_line = np.linspace(1000.0, 16382.0, 200)
    if pncut_fit is not None:
        a, b, sigma, n_sigma = pncut_fit
        y_mid = a * x_line + b
        y_lo = y_mid - n_sigma * sigma
        y_hi = y_mid + n_sigma * sigma
    else:
        x_line = np.array([])
        y_mid = y_lo = y_hi = np.array([])
        n_sigma = float(PNCUT_BAND_N_SIGMA)

    (line_pn,) = ax.plot(x_line, y_mid, "r-", lw=9, alpha=0.4, label="Linear fit", zorder=3)
    (line_lo,) = ax.plot(
        x_line,
        y_lo,
        "r--",
        lw=1.5,
        alpha=0.35,
        zorder=2,
        label=f"±{PNCUT_BAND_N_SIGMA:g}σ band",
    )
    (line_hi,) = ax.plot(x_line, y_hi, "r--", lw=1.5, alpha=0.35, zorder=2)

    # 多选项：从 ~m_acv 的背景逐步加入各种 cut
    cut_keys = [
        "fit_success",
        "cut_time",
        "ch0_min_pos",
        "ch0_max_sat",
        "ch5_rt",
        "ped_3sigma",
        "mincut",
        "ch3ped_min",
        "bscut",
        "pncut",
    ]
    rax = fig.add_axes([0.015, 0.14, 0.185, 0.76])
    rax.set_facecolor("none")
    rax.patch.set_visible(False)
    for s in rax.spines.values():
        s.set_visible(False)
    check = CheckButtons(
        rax,
        cut_keys,
        actives=[False] * len(cut_keys),
        label_props={"fontsize": [6]},
        frame_props={"s": 12},
        check_props={"s": 12, "linewidths": 0.65},
    )

    def _apply_checks() -> None:
        status = check.get_status()
        vis = np.ones(idx.shape[0], dtype=bool)
        for i, k in enumerate(cut_keys):
            if status[i]:
                vis &= pass_fns[k](bundle)[idx]
        x_vis = max_ch0[idx][vis]
        y_vis = max_ch1[idx][vis]
        sc_bg.set_offsets(
            np.column_stack([x_vis, y_vis]) if x_vis.size else np.empty((0, 2))
        )
        fig.canvas.draw_idle()

    def on_check(_label: str) -> None:
        _apply_checks()

    check.on_clicked(on_check)
    _apply_checks()

    ax.set_xlabel("CH0 max (ADC)", fontsize=16)
    ax.set_ylabel("CH1 max (ADC)", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=10, loc="best")

    if show:
        plt.show()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CH0max–CH1max 散点：复选框勾选 cut 后过滤点；叠加 PNcut 拟合线",
    )
    p.add_argument("--file", "-f", dest="h5_name", default=None, help="单个运行的 h5 文件名（basename）")
    p.add_argument("--max-points", type=int, default=1000_000, help="背景散点最大点数（过大则随机采样）")
    p.add_argument("--no-show", action="store_true", help="仅加载数据不弹窗（调试用）")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    root = _project_root()
    base_names = [args.h5_name] if args.h5_name else None
    plot_interactive_cuts_ch0_ch1(
        project_root=root,
        base_names=base_names,
        max_points=int(args.max_points),
        seed=0,
        show=(not args.no_show),
    )