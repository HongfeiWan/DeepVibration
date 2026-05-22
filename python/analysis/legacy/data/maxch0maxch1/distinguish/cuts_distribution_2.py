#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CH0max–CH1max 散点 + basic+act.py 中若干 cut 的交互式过滤显示。

数据读取：与 rt&inhibit&ge-self.py 相同思路（从 raw_pulse/*_parameters 读 HDF5，不读波形），
并扩展加载 CH2/CH3/CH4 及 CH0-3 时间，以对齐 basic+act 中的 cut 所需数组。

交互：左侧小型复选框勾选若干 cut，图中仅保留「同时通过所有已勾选 cut」的事例（未通过者从散点中移除）。
叠加 PNcut 拟合直线（及 ±nσ 带），线型与 ge-self/cut/pncut.py 测试图一致。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.widgets import CheckButtons


def _project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    return script_dir.parents[3]

def _param_dirs(project_root: Path) -> Dict[str, Path]:
    root = project_root / "data" / "hdf5" / "raw_pulse"
    return {
        "ch0": root / "CH0_parameters",
        "ch1": root / "CH1_parameters",
        "ch2": root / "CH2_parameters",
        "ch3": root / "CH3_parameters",
        "ch4": root / "CH4_parameters",
        "ch5": root / "CH5_parameters",
    }

def _load_basic_act_module(project_root: Path):
    """动态加载 ge-self/cut/basic+act.py（文件名含 +，不能常规 import）。"""
    cut_dir = project_root / "python" / "data" / "ge-self" / "cut"
    mod_path = cut_dir / "basic+act.py"
    if not mod_path.is_file():
        alt = cut_dir / "basic+acv.py"
        if alt.is_file():
            mod_path = alt
        else:
            raise FileNotFoundError(f"未找到 basic+act 模块: {cut_dir / 'basic+act.py'}")
    spec = importlib.util.spec_from_file_location("basic_act_cut", str(mod_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载: {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["basic_act_cut"] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "cut_act") and hasattr(mod, "cut_acv"):
        mod.cut_act = mod.cut_acv  # basic+acv 命名
    return mod

def load_pncut_arrays_from_params(
    project_root: str | Path,
    base_name: str,) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    与 rt&inhibit&ge-self.py 一致：返回 (max_ch0, max_ch1, ch0_min, max_ch5)。
    """
    root = Path(project_root)
    dirs = _param_dirs(root)
    ch0_path = dirs["ch0"] / base_name
    ch1_path = dirs["ch1"] / base_name
    ch5_path = dirs["ch5"] / base_name
    if not ch0_path.is_file() or not ch1_path.is_file():
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
    if ch5_path.is_file():
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

def load_cut_bundle_one_run(
    base_name: str,
    ba: Any,) -> Optional[Dict[str, np.ndarray]]:
    """
    单 run：对齐长度后返回 basic+act 各 cut 所需数组。
    """
    ch0_path = ba.CH0_PARAM_DIR / base_name
    ch5_path = ba.CH5_PARAM_DIR / base_name
    if not ch0_path.is_file() or not ch5_path.is_file():
        return None
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
            tmax_ch0,
            tmax_ch1,
        ) = ba._load_basic_features_for_run(ch0_path, ch5_path)
    except (KeyError, OSError) as e:
        print(f"[跳过 {base_name}] 基准特征读取失败: {e}")
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
        "tmax_ch0": tmax_ch0,
        "tmax_ch1": tmax_ch1,
        "ch2_n_fit_points": ch2_n,
        "ch3_n_fit_points": ch3_n,
        "ch2_tanh_p0": ch2_p0,
        "ch3_tanh_p0": ch3_p0,
        "ch3ped_mean": ch3ped_mean,
        "min_ch3": min_ch3,
        "time_mpl": time_mpl,
    }


def concatenate_bundles(runs: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if not runs:
        raise ValueError("无有效 run")
    keys = runs[0].keys()
    out: Dict[str, np.ndarray] = {}
    for k in keys:
        out[k] = np.concatenate([r[k] for r in runs], axis=0)
    return out


def load_cut_bundle(
    project_root: Path,
    base_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, np.ndarray], Any]:
    ba = _load_basic_act_module(project_root)
    ch0_dir = project_root / "data" / "hdf5" / "raw_pulse" / "CH0_parameters"
    if base_names is None:
        if not ch0_dir.is_dir():
            raise FileNotFoundError(f"目录不存在: {ch0_dir}")
        base_names = sorted(x for x in os.listdir(ch0_dir) if x.lower().endswith(".h5"))
    bundles: List[Dict[str, np.ndarray]] = []
    for bn in base_names:
        b = load_cut_bundle_one_run(bn, ba)
        if b is not None:
            bundles.append(b)
            print(f"已读 {bn} | N={b['max_ch0'].shape[0]}")
    if not bundles:
        raise RuntimeError("没有成功加载任何 run")
    return concatenate_bundles(bundles), ba


def _build_cut_pass_fns(
    ba: Any,
    bundle: Dict[str, np.ndarray],
) -> Dict[str, Callable[[Dict[str, np.ndarray]], np.ndarray]]:
    """各键：返回「通过」掩码 True=保留。cut_time 与 basic+act / tradition 一致：先算坏时间段再筛除。"""

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

    def pass_fit_success(d: Dict[str, np.ndarray]) -> np.ndarray:
        return ba.cut_fit_success(
            d["ch2_n_fit_points"],
            d["ch3_n_fit_points"],
            d["ch2_tanh_p0"],
            d["ch3_tanh_p0"],
        )

    def pass_cut_time(d: Dict[str, np.ndarray]) -> np.ndarray:
        return ba.cut_time(d["time_mpl"], bad_intervals=bad_intervals_mpl)

    def pass_ch0_min(d: Dict[str, np.ndarray]) -> np.ndarray:
        return ba.cut_ch0_min_positive(d["ch0_min"])

    def pass_ch0_sat(d: Dict[str, np.ndarray]) -> np.ndarray:
        return ba.cut_ch0_max_saturation(d["max_ch0"], d["max_ch1"])

    def pass_ch5_rt(d: Dict[str, np.ndarray]) -> np.ndarray:
        return ba.cut_ch5_self_trigger(d["max_ch5"])

    def pass_ch3ped_min(d: Dict[str, np.ndarray]) -> np.ndarray:
        return ba.cut_ch3ped_min(d["ch3ped_mean"], d["min_ch3"])

    def pass_ped_3sigma(d: Dict[str, np.ndarray]) -> np.ndarray:
        return ba.cut_pedestal_3sigma(
            d["ch0_ped_mean"], d["ch1_ped_mean"], d["max_ch5"]
        )

    def pass_act(d: Dict[str, np.ndarray]) -> np.ndarray:
        return ba.cut_act(d["max_ch4"], d["tmax_ch4"])

    def pass_mincut(d: Dict[str, np.ndarray]) -> np.ndarray:
        return ba.cut_mincut(
            d["ch0_min"],
            d["ch1_min"],
            d["max_ch4"],
            d["tmax_ch4"],
        )

    def pass_pncut(d: Dict[str, np.ndarray]) -> np.ndarray:
        return ba.cut_pncut(m_pre_m6, d["max_ch0"], d["max_ch1"])


    return {
        "fit_success": pass_fit_success,
        "cut_time": pass_cut_time,
        "ch0_min_pos": pass_ch0_min,
        "ch0_max_sat": pass_ch0_sat,
        "ch5_rt": pass_ch5_rt,
        "ch3ped_min": pass_ch3ped_min,
        "ped_3sigma": pass_ped_3sigma,
        "act": pass_act,
        "mincut": pass_mincut,
        "pncut": pass_pncut,
    }


def _pncut_fit_line_params(
    ba: Any,
    bundle: Dict[str, np.ndarray],
) -> Optional[Tuple[float, float, float, float]]:
    """
    与 basic+acv.cut_pncut 相同拟合：在 m_pre_m6 且 CH0max∈(3000,12000) 上 polyfit，
    返回 (斜率 a, 截距 b, 残差 σ, n_sigma)，失败时返回 None。
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
    max_ch0 = d0["max_ch0"]
    max_ch1 = d0["max_ch1"]
    fit_ch0_min = 3000.0
    fit_ch0_max = 12000.0
    min_fit_events = 10
    fit_mask = m_pre_m6 & (max_ch0 > fit_ch0_min) & (max_ch0 < fit_ch0_max)
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
    n_sigma = 0.3
    return a, b, sigma, n_sigma


CUT_LABELS_ZH: Dict[str, str] = {
    "fit_success": "CH2/CH3 拟合成功",
    "cut_time": "时间 cut（红区暴发排除，与 basic+act 一致）",
    "ch0_min_pos": "ch0_min>0（非抑制）",
    "ch0_max_sat": "CH0 未饱和",
    "ch5_rt": "CH5 非随机触发",
    "ch3ped_min": "CH3 ped-min 带",
    "ped_3sigma": "前沿基线 3σ",
    "act": "ACT 反符合",
    "mincut": "mincut（ACT 后 CH0/CH1 min）",
    "pncut": "PN 线性带",
    "ch0max_tmax": "CH0max–tmax 带",
    "ch1max_tmax": "CH1max–tmax 带",
}


def plot_interactive_cuts_ch0_ch1(
    project_root: Optional[Path] = None,
    base_names: Optional[List[str]] = None,
    *,
    max_points: int = 40000_000,
    seed: int = 0,
    show: bool = True,
) -> None:
    """
    交互图：灰色散点为（子）样本；勾选若干 cut 后，仅保留同时通过所有已勾选项的点。
    叠加 PNcut 在 m0–m6 上的拟合直线及 ±nσ 带，线宽/透明度与 pncut.py 测试图一致。
    """
    if project_root is None:
        project_root = _project_root()
    bundle, ba = load_cut_bundle(project_root, base_names)
    pass_fns = _build_cut_pass_fns(ba, bundle)
    pncut_fit = _pncut_fit_line_params(ba, bundle)

    max_ch0 = bundle["max_ch0"]
    max_ch1 = bundle["max_ch1"]
    n = max_ch0.shape[0]
    rng = np.random.default_rng(seed)

    idx = np.arange(n)
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        print(f"背景散点采样: {max_points}/{n}")

    # 坐标轴英数字体与 ge-self/cut/pncut.py 测试块一致（默认 sans-serif，16/12 pt）
    plt.rcParams.setdefault("axes.unicode_minus", False)

    fig = plt.figure(figsize=(10, 6))
    fig.subplots_adjust(left=0.08, bottom=0.10, right=0.98, top=0.92)
    # 主图靠右，左侧窄条仅放复选框，避免与 y 轴刻度区重叠
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

    (line_pn,) = ax.plot(
        x_line,
        y_mid,
        "r-",
        lw=9,
        alpha=0.4,
        label="Linear fit",
        zorder=3,
    )
    (line_lo,) = ax.plot(
        x_line,
        y_lo,
        "r--",
        lw=1.5,
        alpha=0.35,
        zorder=2,
        label="±3σ band",
    )
    (line_hi,) = ax.plot(x_line, y_hi, "r--", lw=1.5, alpha=0.35, zorder=2)

    # 勾选 PNcut 后显示：y=6.7x-7066.7 左侧 与 y<1250 下边 的并集（半透明蓝，仅可视化）
    _xlim = ax.get_xlim()
    _ylim = ax.get_ylim()
    _nx, _ny = 1200, 1200
    _gx = np.linspace(_xlim[0], _xlim[1], _nx)
    _gy = np.linspace(_ylim[0], _ylim[1], _ny)
    _GX, _GY = np.meshgrid(_gx, _gy)
    _F_line = 6.7 * _GX - _GY - 7066.7
    _union = (_GY < 1250.0) | (_F_line < 0.0)
    _Z = np.where(_union, 1.0, 0.0).astype(np.float64)
    _cmap = ListedColormap([(0.0, 0.45, 0.9, 0.0), (0.0, 0.45, 0.9, 0.28)])
    pncut_region = ax.pcolormesh(
        _GX,
        _GY,
        _Z,
        cmap=_cmap,
        vmin=0.0,
        vmax=1.0,
        shading="auto",
        zorder=0,
        visible=False,
        rasterized=True,
    )

    ax.set_xlabel("CH0 Max (ADC counts)", fontsize=16)
    ax.set_ylabel("CH1 Max (ADC counts)", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=12)

    ax.set_xlim(1000, 16382)
    ax.set_ylim(1000, 16382)
    ax.legend(loc="upper left", fontsize=12)


    cut_keys = list(pass_fns.keys())
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
        vis = np.ones(idx.shape[0], dtype=bool)
        status = check.get_status()
        for i, k in enumerate(cut_keys):
            if status[i]:
                vis &= pass_fns[k](bundle)[idx]
        x_vis = max_ch0[idx][vis]
        y_vis = max_ch1[idx][vis]
        sc_bg.set_offsets(np.column_stack([x_vis, y_vis]))
        n_vis = int(vis.sum())
        checked = [cut_keys[i] for i, on in enumerate(status) if on]
        extra = ",".join(checked) if checked else "none"
        pncut_region.set_visible(status[cut_keys.index("pncut")])
        fig.canvas.draw_idle()

    def on_check(_label: str) -> None:
        _apply_checks()

    check.on_clicked(on_check)

    _apply_checks()

    if show:
        plt.show()


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CH0max–CH1max 散点：复选框勾选 cut 后从图中去掉未通过该 cut 的点；叠加 PNcut 拟合线"
    )
    p.add_argument(
        "--file",
        "-f",
        dest="h5_name",
        default=None,
        help="单个运行的 h5 文件名（basename）；不指定则读取 CH0_parameters 下全部 .h5",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=4000_0000,
        help="背景散点最大点数（过大则随机采样）",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="仅加载数据不弹窗（调试用）",
    )
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    root = _project_root()
    names = [args.h5_name] if args.h5_name else None
    if not args.no_show:
        plot_interactive_cuts_ch0_ch1(
            project_root=root,
            base_names=names,
            max_points=args.max_points,
            show=True,
        )
    else:
        b, _ = load_cut_bundle(root, names)
        print("keys:", list(b.keys()), "N=", b["max_ch0"].shape[0])
