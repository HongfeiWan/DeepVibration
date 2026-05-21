#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
传统参数统计脚本。

模仿 30parameter&HDBSCAN.py 的目录发现与参数读取方式：
- 从 data/hdf5/raw_pulse/CH0_parameters 列出 run 文件名
- 对每个 run，在 CH0~CH5_parameters 中读取所有一维数据集作为特征
- 统计 raw_pulse 的总事件数与总参数维度
"""

from __future__ import annotations
import argparse
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MultipleLocator
import pandas as pd
from scipy.optimize import curve_fit

def _discover_project_root() -> Path:
    """
    推断 DeepVibration 项目根目录。
    当前文件位于:
        .../python/data/ge-self/cut/parameterize/tradition/tradition.py
    向上到 python，再上一层即项目根目录。
    """
    here = Path(__file__).resolve()
    # 0: tradition
    # 1: parameterize
    # 2: cut
    # 3: ge-self
    # 4: data
    # 5: python
    # 6: DeepVibration
    python_dir = here.parents[5]
    return python_dir.parent

PROJECT_ROOT = _discover_project_root()
DATA_ROOT = PROJECT_ROOT / "data" / "hdf5" / "raw_pulse"

# 本脚本目录（用于默认 MATLAB .fig 路径）
_SCRIPT_DIR = Path(__file__).resolve().parent
# read_matlab_fig 与 umap-optimazation-2 同层
_READ_MATLAB_FIG_PKG_DIR = _SCRIPT_DIR.parent / "umap-optimazation-2"

# 与 combine_wyf_plot / read_matlab_fig 默认图例名一致
_MATLAB_ACV_DISPLAY_NAME = "basic cut + ACV"
_DEFAULT_MATLAB_FIG_PATH = _SCRIPT_DIR / "DZL_vetospec_12kev_0615.fig"

CH_PARAM_DIRS: Dict[int, Path] = {
    0: DATA_ROOT / "CH0_parameters",
    1: DATA_ROOT / "CH1_parameters",
    2: DATA_ROOT / "CH2_parameters",
    3: DATA_ROOT / "CH3_parameters",
    4: DATA_ROOT / "CH4_parameters",
    5: DATA_ROOT / "CH5_parameters",}

# 能量刻度系数（与 physical/spectrum.py 一致）
_E_CAL_A = 0.0008432447500464594
_E_CAL_B = -0.826976770117076
EXPOSURE_KG = 0.5
EXPOSURE_DAYS = 20.0
# 与 physical/spectrum.py 直方图分箱数一致
SPECTRUM_N_BINS = 1000
# CH3 上升时间 rise_time = ln(19)/p1（μs），与 ge-self/cut/bscut 一致
_LN_19 = np.log(19.0)
EPOCH_OFFSET_DEFAULT = 2.082816000000000e09
CH3PED_MIN_X_RANGE = (960.0, 980.0)
# 与 ch3ped-min.py 红区 axvspan 一致：半宽 h·σ_x（h 默认 0.5，σ_x 为全体有限点 ch3ped_mean 的样本标准差）
CH3PED_X_MEAN_BAND_HALF_SIGMA = 0.5
# 与 ch3ped-min.py 绿区 fill_between 一致：|残差| ≤ n·σ_res（默认 n=6）
CH3PED_RESIDUAL_N_SIGMA = 6.0
# cut_time：在 MAX(CH0)∈[1250,1500] 子样本上按时间分箱后的计数率阈值 (count/min)。
# 某 bin 的 rate > 该值则视为「红区」并整段排除；数值越大越宽松、红区越少。
# 若为 None，则使用 CH0_TIME_BAND_BURST_RATE_THRESHOLD（与 ch0-time 下图一致）。
CH0_BAND_BURST_LO = 1250.0
CH0_BAND_BURST_HI = 1500.0
CH0_TIME_BAND_BURST_RATE_THRESHOLD = 0.5  # count/min
CH0_TIME_EXCLUDE_BAD_INTERVALS_MPL: Tuple[Tuple[float, float], ...] = ()
CUT_TIME_RATE_THRESHOLD: Optional[float] = None

# 与 python/utils/plotstyle.md 一致
_PLOT_TICK = 12
_PLOT_AXIS = 16
_PLOT_TITLE = 18
_PLOT_LEGEND = 12
_PLOT_SUBPLOT_TITLE = 14

def _apply_plotstyle_font() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
        }
    )

@dataclass
class RunParameters:
    base_name: str
    n_events: int
    feature_names: List[str]
    feature_matrix: np.ndarray  # (n_events, n_features)

def _list_base_names_from_ch0() -> List[str]:
    """以 CH0_parameters 为主目录列出所有 h5 run 文件名。"""
    ch0_dir = CH_PARAM_DIRS[0]
    if not ch0_dir.exists():
        raise FileNotFoundError(f"CH0_parameters 目录不存在: {ch0_dir}")

    base_names: List[str] = []
    for name in sorted(os.listdir(ch0_dir)):
        if name.lower().endswith((".h5", ".hdf5")):
            base_names.append(name)
    if not base_names:
        raise FileNotFoundError(f"CH0_parameters 目录 {ch0_dir} 下未找到任何 h5 文件")
    return base_names

def _open_param_file_if_exists(ch: int, base_name: str) -> Path | None:
    """在 CH{ch}_parameters 下查找同名 run 文件。"""
    path = CH_PARAM_DIRS[ch] / base_name
    return path if path.exists() else None

def _map_base_name_to_ch03_files(base_name: str) -> List[Path]:
    ch03_dir = DATA_ROOT / "CH0-3"
    stem = Path(base_name).stem
    candidates = [ch03_dir / f"{stem}.h5"]
    if stem.endswith("_processed"):
        candidates.append(ch03_dir / f"{stem[:-10]}_processed.h5")
    else:
        candidates.append(ch03_dir / f"{stem}_processed.h5")
    return [c for c in candidates if c.is_file()]

def _read_event_time_datetime64_ns_from_ch03(
    base_name: str,
    epoch_offset: float = EPOCH_OFFSET_DEFAULT,) -> np.ndarray:
    """从 CH0-3 raw_pulse 读取 time_data 并转换为 datetime64[ns]。"""
    ch03_candidates = _map_base_name_to_ch03_files(base_name)
    time_data: np.ndarray | None = None
    for c in ch03_candidates:
        try:
            with h5py.File(c, "r") as f_ch03:
                if "time_data" not in f_ch03:
                    continue
                t = np.asarray(f_ch03["time_data"][...], dtype=np.float64)
                if t.size == 0:
                    continue
                time_data = t
                break
        except Exception:
            continue
    if time_data is None:
        raise RuntimeError(f"未能在 CH0-3 为 {base_name} 找到包含 time_data 的文件")
    if time_data.size == 0:
        raise RuntimeError(f"{base_name} 对应的 time_data 为空")
    epoch_start = datetime(1970, 1, 1)
    eventtime = time_data - float(epoch_offset)
    dt = epoch_start + pd.to_timedelta(eventtime, unit="s")
    return dt.to_numpy(dtype="datetime64[ns]")

def _datetime64_ns_to_mpl_date(time_ns: np.ndarray) -> np.ndarray:
    """datetime64[ns] -> matplotlib date float。"""
    time_ns = np.asarray(time_ns)
    time_py = time_ns.astype("M8[ms]").astype(datetime)
    return np.asarray(mdates.date2num(time_py), dtype=np.float64)

def _read_required_features_for_run(base_name: str) -> Dict[str, np.ndarray]:
    """
    只读取当前流程需要的最小特征集合，避免读取全部参数导致的 I/O 开销。
    """
    need_by_ch: Dict[int, List[str]] = {
        0: ["max_ch0", "ch0_min", "ch0ped_mean", "tmax_ch0"],
        1: ["ch1ped_mean", "ch1_min", "max_ch1", "tmax_ch1"],
        2: ["n_fit_points", "tanh_p0"],
        3: ["ch3ped_mean", "min_ch3", "n_fit_points", "tanh_p0", "tanh_p1"],
        4: ["max_ch4", "tmax_ch4"],
        5: ["max_ch5"],
    }
    out: Dict[str, np.ndarray] = {}
    n_events_ref: int | None = None

    for ch, keys in need_by_ch.items():
        path = _open_param_file_if_exists(ch, base_name)
        if path is None:
            raise FileNotFoundError(f"缺少参数文件: CH{ch} / {base_name}")
        with h5py.File(path, "r") as f:
            for key in keys:
                if key not in f:
                    raise KeyError(f"{path} 中缺少数据集: {key}")
                arr = np.asarray(f[key][...], dtype=np.float64)
                if arr.ndim != 1:
                    raise ValueError(f"{path}::{key} 不是 1D 数据集。")
                if n_events_ref is None:
                    n_events_ref = int(arr.shape[0])
                elif int(arr.shape[0]) != n_events_ref:
                    raise ValueError(
                        f"{base_name} 事件数不一致: 预期 {n_events_ref}, 实际 {arr.shape[0]} ({path.name}:{key})"
                    )
                out[f"ch{ch}_{key}"] = arr

    if n_events_ref is None:
        raise RuntimeError(f"未能读取到有效特征: {base_name}")
    return out

def _read_all_1d_datasets(path: Path, prefix: str) -> Tuple[Dict[str, np.ndarray], int]:
    """
    读取参数文件中的所有一维数据集，返回:
    - features: {f"{prefix}{dataset_name}": ndarray(shape=(n_events,))}
    - n_events: 事件数
    """
    features: Dict[str, np.ndarray] = {}
    n_events = 0

    with h5py.File(path, "r") as f:
        for key, dset in f.items():
            if not isinstance(dset, h5py.Dataset):
                continue
            if dset.ndim != 1:
                continue

            data = np.asarray(dset[...])
            if data.size == 0:
                continue

            if n_events == 0:
                n_events = int(data.shape[0])
            elif data.shape[0] != n_events:
                print(
                    f"[警告] 文件 {path.name} 中数据集 {key} 长度 "
                    f"{data.shape[0]} != 预期 {n_events}，跳过该数据集。"
                )
                continue

            features[f"{prefix}{key}"] = data.astype(np.float64)

    return features, n_events

def load_run_parameters(base_name: str) -> RunParameters:
    """读取一个 run 的 CH0~CH5 参数并拼成特征矩阵。"""
    per_channel_features: Dict[int, Dict[str, np.ndarray]] = {}
    n_events_global: int | None = None

    for ch in range(6):
        path = _open_param_file_if_exists(ch, base_name)
        if path is None:
            continue

        feats, n_events = _read_all_1d_datasets(path, prefix=f"ch{ch}_")
        if not feats:
            continue

        if n_events_global is None:
            n_events_global = n_events
        elif n_events != n_events_global:
            raise ValueError(
                f"文件 {base_name} 中通道 CH{ch} 事件数 {n_events} "
                f"与其他通道不一致（预期 {n_events_global}）。"
            )

        per_channel_features[ch] = feats

    if n_events_global is None or not per_channel_features:
        raise RuntimeError(f"运行 {base_name} 未收集到任何有效参数。")

    feature_names: List[str] = []
    feature_arrays: List[np.ndarray] = []
    for ch in sorted(per_channel_features.keys()):
        feats = per_channel_features[ch]
        for name in sorted(feats.keys()):
            feature_names.append(name)
            feature_arrays.append(feats[name].reshape(n_events_global, 1))

    feature_matrix = np.concatenate(feature_arrays, axis=1)
    return RunParameters(
        base_name=base_name,
        n_events=n_events_global,
        feature_names=feature_names,
        feature_matrix=feature_matrix,
    )

def summarize_total_events_and_dims() -> Tuple[int, int]:
    """
    汇总 raw_pulse 参数目录中的:
    - 总事件数（所有 run 事件数求和）
    - 总参数维度（所有 run 特征名并集大小）
    """
    base_names = _list_base_names_from_ch0()

    total_events = 0
    all_feature_names: set[str] = set()

    for base_name in base_names:
        run = load_run_parameters(base_name)
        total_events += run.n_events
        all_feature_names.update(run.feature_names)

    total_param_dim = len(all_feature_names)
    return total_events, total_param_dim

def _compute_spectrum_for_mask(
    max_ch0: np.ndarray,
    mask: np.ndarray,
    n_bins: int,
    e_min: float = 0.1,
    e_max: float = 12.0,
    *,
    exposure_days: float = EXPOSURE_DAYS,) -> Tuple[np.ndarray, np.ndarray]:
    """
    对给定 mask 下的 max_ch0 计算能谱 rate，与 physical/spectrum._compute_spectrum 一致：
    rate = counts / (EXPOSURE_KG * ΔE_keV * exposure_days)。
    """
    x = max_ch0[mask]
    energy_values = _E_CAL_A * x + _E_CAL_B
    counts, bin_edges = np.histogram(energy_values, bins=n_bins, range=(e_min, e_max))
    bin_widths = np.diff(bin_edges)
    bin_widths[bin_widths == 0] = np.inf
    exp_days = float(exposure_days)
    if exp_days <= 0.0:
        raise ValueError(f"exposure_days 必须 > 0，当前为 {exp_days}")
    denom = EXPOSURE_KG * bin_widths * exp_days
    rates = counts / denom
    return bin_edges, rates

def _overlay_matlab_basic_cut_acv(ax: plt.Axes, fig_path: Path) -> bool:
    """
    从 MATLAB .fig 中仅绘制 DisplayName 为「basic cut + ACV」的曲线（散点），
    逻辑与 umap-optimazation-2/combine_wyf_plot.py 一致。
    """
    if not fig_path.is_file():
        print(f"[MATLAB .fig] 文件不存在: {fig_path}")
        return False
    if str(_READ_MATLAB_FIG_PKG_DIR) not in sys.path:
        sys.path.insert(0, str(_READ_MATLAB_FIG_PKG_DIR))
    import read_matlab_fig as rmf

    print(f"[MATLAB .fig] 叠加 basic cut + ACV: {fig_path}")
    raw = rmf.load_matlab_fig(fig_path)
    series = rmf.extract_xy_series(raw, include_child_lines=True)
    to_plot = rmf._series_for_plot(series, plot_all=False)
    axinfo = rmf.extract_axes_info(raw)
    y_is_log = axinfo.get("yscale", "linear") == "log"

    for s in to_plot:
        label = str(s.get("display_name", "")).strip()
        if label != _MATLAB_ACV_DISPLAY_NAME:
            continue
        x = np.asarray(s["x"], dtype=np.float64).ravel()
        y = np.asarray(s["y"], dtype=np.float64).ravel()
        n = min(x.size, y.size)
        x, y = x[:n], y[:n]
        if y_is_log:
            ok = y > 0
            x, y = x[ok], y[ok]
        ax.scatter(
            x,
            y,
            s=12,
            c="C2",
            alpha=0.85,
            edgecolors="none",
            label=label,
            zorder=5,
        )
        return True

    print(f"[MATLAB .fig] 未找到「{_MATLAB_ACV_DISPLAY_NAME}」序列（read_matlab_fig 默认筛选）。")
    return False

# -----------------------------------------------------------------------------
# 独立的 绘图 函数：绘制能谱图
# -----------------------------------------------------------------------------

def plot_cumulative_cut_spectra(
    max_ch0_all: np.ndarray,
    cut_steps: List[Tuple[str, np.ndarray]],
    n_bins: int = SPECTRUM_N_BINS,
    e_min: float = 0.1,
    e_max: float = 12.0,
    *,
    overlay_matlab_acv: bool = False,
    matlab_fig_path: Optional[Path] = None,
    exposure_days_after_cut_time: Optional[float] = None,) -> None:
    """
    按给定顺序累计应用 cut，并在同一张图上绘制每一步的能谱。
    绘图风格与 physical/spectrum.py 一致：figsize、subplots_adjust、坐标轴字号 16/12、
    对数纵轴、stairs 填充阶梯（无误差棒）；无复选框，主图占满宽度。

    cut_steps 的第一个元素应为基础掩码，label 会显示为 basic。
    后续每一步显示为 above+<name>，并附带当前 N。

    overlay_matlab_acv:
        若 True，在同轴上叠加同目录（或 matlab_fig_path）下 MATLAB .fig 中的
        「basic cut + ACV」散点（与 combine_wyf_plot 一致）。
    matlab_fig_path:
        自定义 .fig；默认使用本脚本目录下 DZL_vetospec_12kev_0615.fig。
    """
    if max_ch0_all.ndim != 1:
        raise ValueError("max_ch0_all 必须是一维数组。")
    n_total = max_ch0_all.shape[0]
    if not cut_steps:
        raise ValueError("cut_steps 不能为空。")

    _apply_plotstyle_font()
    plt.rcParams.setdefault("axes.unicode_minus", False)
    fig = plt.figure(figsize=(10, 6))
    fig.subplots_adjust(left=0.08, bottom=0.10, right=0.98, top=0.92)
    ax = fig.add_axes([0.10, 0.12, 0.86, 0.80])
    # 与 physical/spectrum.py _plot_interactive_spectrum 中 stairs 一致
    _log_floor = 1.0
    # 固定颜色顺序：第 i 条曲线始终映射到同一颜色
    fixed_colors = [
        "C0",  # 1st: blue
        "C1",  # 2nd: orange
        "C2",  # 3rd: green
        "C3",  # 4th: red
        "C4",  # 5th: purple
        "C5",  # 6th: brown
        "C6",  # 7th: pink
        "C7",  # 8th: gray
        "C8",  # 9th: olive
        "C9",  # 10th: cyan
        "#1f77b4",  # 11th
        "#ff7f0e",  # 12th
        "#2ca02c",  # 13th
        "#d62728",  # 14th
        "#9467bd",  # 15th
        "#8c564b",  # 16th
        "#e377c2",  # 17th
        "#7f7f7f",  # 18th
        "#bcbd22",  # 19th
        "#17becf",  # 20th
        "#393b79",  # 21st
        "#637939",  # 22nd
        "#8c6d31",  # 23rd
        "#843c39",  # 24th
        "#7b4173",  # 25th
        "#3182bd",  # 26th
        "#31a354",  # 27th
        "#756bb1",  # 28th
        "#636363",  # 29th
        "#e6550d",  # 30th
    ]
    cumulative = np.ones(n_total, dtype=bool)
    # 最后一条实际会画出的阶梯（累计后 N>0）用不透明填充，其余保持半透明
    _cum_probe = np.ones(n_total, dtype=bool)
    last_plotted_i = -1
    for _j, (_nm, _sm) in enumerate(cut_steps):
        _cum_probe &= np.asarray(_sm, dtype=bool).ravel()
        if int(_cum_probe.sum()) > 0:
            last_plotted_i = _j

    # 需求更新：
    # - 从 ("basic", m_basic_cut) 这条曲线开始，以及其后的所有 step，
    #   计算 rate 时都使用 exposure_days_after_cut_time 作为 live time；
    # - 在此之前（如 "total"）仍使用原始 EXPOSURE_DAYS。
    use_reduced_exposure_now = False
    for i, (name, step_mask) in enumerate(cut_steps):
        m = np.asarray(step_mask, dtype=bool).ravel()
        if m.size != n_total:
            raise ValueError(f"cut_steps[{i}] 掩码长度 {m.size} 与事件数 {n_total} 不一致。")
        cumulative = cumulative & m
        n_pass = int(cumulative.sum())
        if n_pass == 0:
            continue
        # basic / cut_time 以及之后的所有 step，统一使用 reduced exposure
        if name in ("basic", "cut_time"):
            use_reduced_exposure_now = True

        if use_reduced_exposure_now and exposure_days_after_cut_time is not None:
            exp_days = float(exposure_days_after_cut_time)
        else:
            exp_days = float(EXPOSURE_DAYS)
        bin_edges, rates = _compute_spectrum_for_mask(
            max_ch0_all,
            cumulative,
            n_bins=n_bins,
            e_min=e_min,
            e_max=e_max,
            exposure_days=exp_days,
        )
        label = f"basic (N={n_pass})" if i == 0 else f"above+{name} (N={n_pass})"
        color = fixed_colors[i % len(fixed_colors)]
        rates_plot = np.maximum(rates, _log_floor)
        _fill_alpha = 1.0 if i == last_plotted_i else 0.45
        ax.stairs(
            rates_plot,
            bin_edges,
            orientation="vertical",
            fill=True,
            color=color,
            alpha=_fill_alpha,
            edgecolor=color,
            linewidth=0.9,
            baseline=_log_floor * 0.1,
            label=label,
            zorder=3 + i,
        )

    if overlay_matlab_acv:
        fig_p = matlab_fig_path if matlab_fig_path is not None else _DEFAULT_MATLAB_FIG_PATH
        if not fig_p.is_absolute():
            fig_p = (_SCRIPT_DIR / fig_p).resolve()
        _overlay_matlab_basic_cut_acv(ax, fig_p)

    ax.set_yscale("log")
    ax.set_xlabel("Energy (keV)", fontsize=16)
    ax.set_ylabel(r"Rate [counts / (keV·kg·day)]", fontsize=16)

    # x 轴主刻度：全范围时固定 0.5 keV；交互缩放后按视窗跨度自动变细，避免“刻度总是密一倍”
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:g}"))

    def _choose_major_step(span: float) -> float:
        # span 单位 keV。经验阈值：保证全范围 (~12 keV) 时是 0.5 keV，
        # 放大后可逐级变细，但不会一开始就变成 0.25 这种“密一倍”的效果。
        if not np.isfinite(span) or span <= 0:
            return 0.5
        if span >= 10.0:
            return 0.5
        if span >= 5.0:
            return 0.5
        if span >= 2.0:
            return 0.1
        if span >= 1.0:
            return 0.1
        if span >= 0.5:
            return 0.1
        return 0.1

    def _update_x_locator(_ax: plt.Axes) -> None:
        x0, x1 = _ax.get_xlim()
        step = _choose_major_step(abs(float(x1) - float(x0)))
        _ax.xaxis.set_major_locator(MultipleLocator(step))
        # minor ticks：每个 major 再细分 5 份
        _ax.xaxis.set_minor_locator(AutoMinorLocator(5))

    # 初始设置
    _update_x_locator(ax)
    # 缩放/平移后自动更新
    ax.callbacks.connect("xlim_changed", _update_x_locator)

    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="x", which="minor", length=3, width=0.8, direction="in")
    ax.legend(fontsize=12, loc="best")
    ax.set_xlim(e_min, e_max)
    ax.set_ylim(1e0, 1e7)
    plt.show()

def plot_waveforms_from_mask_diff(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    base_names: List[str],
    run_event_counts: List[int],
    ch0_3_dir: Path | None = None,
    n_show: int = 9,
    random_state: int = 42,
    sampling_interval_ns: float = 4.0,
    alpha: float = 0.2,) -> None:
    """
    对两个 mask 中“在 mask_a 但不在 mask_b”的事件随机抽样并绘制 CH0/CH3 原始波形。

    参数:
    - mask_a, mask_b: 两个等长 bool 掩码
    - base_names: run 文件名列表（与 run_event_counts 一一对应）
    - run_event_counts: 每个 run 的事件数
    - ch0_3_dir: CH0-3 原始波形目录；默认 data/hdf5/raw_pulse/CH0-3
    - n_show: 目标绘制事件数（默认 9）
    - random_state: 随机数种子（默认 42）
    - sampling_interval_ns: 采样间隔（ns）
    - alpha: 波形透明度（CH0/CH3 均使用该值）
    """
    ma = np.asarray(mask_a, dtype=bool).ravel()
    mb = np.asarray(mask_b, dtype=bool).ravel()
    if ma.shape != mb.shape:
        raise ValueError("mask_a 与 mask_b 长度不一致。")
    if len(base_names) != len(run_event_counts):
        raise ValueError("base_names 与 run_event_counts 长度不一致。")
    run_counts = np.asarray(run_event_counts, dtype=np.int64)
    if np.any(run_counts < 0):
        raise ValueError("run_event_counts 中不能有负数。")
    cum_counts = np.cumsum(run_counts)
    if cum_counts.size == 0 or int(cum_counts[-1]) != ma.size:
        raise ValueError(
            f"run_event_counts 总和 {int(cum_counts[-1]) if cum_counts.size else 0} 与 mask 长度 {ma.size} 不一致。"
        )

    if ch0_3_dir is None:
        ch0_3_dir = DATA_ROOT / "CH0-3"

    diff_idx = np.where(ma & (~mb))[0]
    if diff_idx.size == 0:
        print("[信息] 两个 mask 没有差异事件可绘制。")
        return

    n_pick = min(int(n_show), int(diff_idx.size))
    rng = np.random.default_rng(random_state)
    picked = rng.choice(diff_idx, size=n_pick, replace=False)

    _apply_plotstyle_font()
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.ravel()

    for i, gidx in enumerate(picked):
        ax_left = axes[i]
        run_idx = int(np.searchsorted(cum_counts, int(gidx), side="right"))
        prev_end = 0 if run_idx == 0 else int(cum_counts[run_idx - 1])
        ev_idx = int(gidx) - prev_end
        base_name = base_names[run_idx]
        ch0_3_path = ch0_3_dir / base_name
        if not ch0_3_path.exists():
            ax_left.set_visible(False)
            continue

        with h5py.File(ch0_3_path, "r") as f_ch:
            ch_data = f_ch["channel_data"]
            time_samples, num_channels, num_events = ch_data.shape
            if num_channels <= 3 or ev_idx >= num_events:
                ax_left.set_visible(False)
                continue
            time_us = np.arange(time_samples) * float(sampling_interval_ns) / 1000.0
            wf_ch0 = ch_data[:, 0, ev_idx].astype(np.float64)
            wf_ch3 = ch_data[:, 3, ev_idx].astype(np.float64)

        ax_right = ax_left.twinx()
        ax_left.plot(time_us, wf_ch0, color="C0", linewidth=0.8, alpha=alpha, label="CH0")
        ax_right.plot(time_us, wf_ch3, color="C3", linewidth=0.8, alpha=alpha, label="CH3")
        stem = Path(base_name).stem
        m = re.search(r"(\d+)(?!.*\d)", stem)
        run_id = m.group(1) if m else stem
        ax_left.set_title(f"{run_id} | #{ev_idx}", fontsize=_PLOT_SUBPLOT_TITLE)
        ax_left.set_xlabel("Time (us)", fontsize=_PLOT_AXIS)
        ax_left.set_ylabel("CH0 ADC", fontsize=_PLOT_AXIS)
        ax_right.set_ylabel("CH3 ADC", fontsize=_PLOT_AXIS)
        ax_left.tick_params(axis="both", which="major", labelsize=_PLOT_TICK)
        ax_right.tick_params(axis="both", which="major", labelsize=_PLOT_TICK)
        ax_left.grid(True, alpha=0.3)

    for j in range(n_pick, 9):
        axes[j].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_waveforms_from_mask_diff_top_spectral_centroid(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    base_names: List[str],
    run_event_counts: List[int],
    ch0_3_dir: Path | None = None,
    n_show: int = 9,
    sampling_interval_ns: float = 4.0,
    alpha: float = 0.2,) -> None:
    """
    在 mask_a & (~mask_b) 的差异事件中，按 CH3 的 spectral_centroid_mhz 降序选前 n_show 个，
    并绘制对应事件的 CH0/CH3 原始波形。
    """
    ma = np.asarray(mask_a, dtype=bool).ravel()
    mb = np.asarray(mask_b, dtype=bool).ravel()
    if ma.shape != mb.shape:
        raise ValueError("mask_a 与 mask_b 长度不一致。")
    if len(base_names) != len(run_event_counts):
        raise ValueError("base_names 与 run_event_counts 长度不一致。")

    run_counts = np.asarray(run_event_counts, dtype=np.int64)
    if np.any(run_counts < 0):
        raise ValueError("run_event_counts 中不能有负数。")
    cum_counts = np.cumsum(run_counts)
    if cum_counts.size == 0 or int(cum_counts[-1]) != ma.size:
        raise ValueError(
            f"run_event_counts 总和 {int(cum_counts[-1]) if cum_counts.size else 0} 与 mask 长度 {ma.size} 不一致。"
        )
    if ch0_3_dir is None:
        ch0_3_dir = DATA_ROOT / "CH0-3"

    diff_idx = np.where(ma & (~mb))[0]
    if diff_idx.size == 0:
        print("[信息] 两个 mask 没有差异事件可绘制。")
        return

    # 每个候选事件查找其 CH3 spectral_centroid_mhz，按值降序取前 n_show
    scores = np.full(diff_idx.shape[0], -np.inf, dtype=np.float64)
    run_to_positions: Dict[int, List[Tuple[int, int]]] = {}
    for pos, gidx in enumerate(diff_idx):
        run_idx = int(np.searchsorted(cum_counts, int(gidx), side="right"))
        prev_end = 0 if run_idx == 0 else int(cum_counts[run_idx - 1])
        ev_idx = int(gidx) - prev_end
        run_to_positions.setdefault(run_idx, []).append((pos, ev_idx))

    for run_idx, pos_list in run_to_positions.items():
        base_name = base_names[run_idx]
        ch3_param_path = CH_PARAM_DIRS[3] / base_name
        if not ch3_param_path.exists():
            continue
        with h5py.File(ch3_param_path, "r") as f_ch3:
            if "spectral_centroid_mhz" not in f_ch3:
                continue
            sc = np.asarray(f_ch3["spectral_centroid_mhz"][...], dtype=np.float64)
            for pos, ev_idx in pos_list:
                if 0 <= ev_idx < sc.shape[0] and np.isfinite(sc[ev_idx]):
                    scores[pos] = sc[ev_idx]

    valid = np.isfinite(scores)
    if not np.any(valid):
        print("[信息] 差异事件中未找到可用的 spectral_centroid_mhz。")
        return
    order = np.argsort(scores[valid])[::-1]
    valid_idx = np.where(valid)[0]
    pick_pos = valid_idx[order[: min(int(n_show), int(valid_idx.size))]]
    picked = diff_idx[pick_pos]

    _apply_plotstyle_font()
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.ravel()

    for i, gidx in enumerate(picked):
        ax_left = axes[i]
        run_idx = int(np.searchsorted(cum_counts, int(gidx), side="right"))
        prev_end = 0 if run_idx == 0 else int(cum_counts[run_idx - 1])
        ev_idx = int(gidx) - prev_end
        base_name = base_names[run_idx]
        ch0_3_path = ch0_3_dir / base_name
        if not ch0_3_path.exists():
            ax_left.set_visible(False)
            continue

        with h5py.File(ch0_3_path, "r") as f_ch:
            ch_data = f_ch["channel_data"]
            time_samples, num_channels, num_events = ch_data.shape
            if num_channels <= 3 or ev_idx >= num_events:
                ax_left.set_visible(False)
                continue
            time_us = np.arange(time_samples) * float(sampling_interval_ns) / 1000.0
            wf_ch0 = ch_data[:, 0, ev_idx].astype(np.float64)
            wf_ch3 = ch_data[:, 3, ev_idx].astype(np.float64)

        ax_right = ax_left.twinx()
        ax_left.plot(time_us, wf_ch0, color="C0", linewidth=0.8, alpha=alpha, label="CH0")
        ax_right.plot(time_us, wf_ch3, color="C3", linewidth=0.8, alpha=alpha, label="CH3")
        stem = Path(base_name).stem
        m = re.search(r"(\d+)(?!.*\d)", stem)
        run_id = m.group(1) if m else stem
        sc_val = scores[pick_pos[i]]
        ax_left.set_title(
            f"{run_id} | #{ev_idx} | sc={sc_val:.3f} MHz",
            fontsize=_PLOT_SUBPLOT_TITLE,
        )
        ax_left.set_xlabel("Time (us)", fontsize=_PLOT_AXIS)
        ax_left.set_ylabel("CH0 ADC", fontsize=_PLOT_AXIS)
        ax_right.set_ylabel("CH3 ADC", fontsize=_PLOT_AXIS)
        ax_left.tick_params(axis="both", which="major", labelsize=_PLOT_TICK)
        ax_right.tick_params(axis="both", which="major", labelsize=_PLOT_TICK)
        ax_left.grid(True, alpha=0.3)

    for j in range(len(picked), 9):
        axes[j].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# -----------------------------------------------------------------------------
# 独立的 cut 函数：输入为对应参数数组，输出为 bool 掩码
# -----------------------------------------------------------------------------

def cut_fit_success(
    ch2_n_fit_points: np.ndarray,
    ch3_n_fit_points: np.ndarray,
    ch2_tanh_p0: np.ndarray,
    ch3_tanh_p0: np.ndarray,
    bad_val: float = 1e6,) -> np.ndarray:
    """
    过滤 fit_ch2_ch3_parallel.py 中拟合失败/未参与拟合的事件。

    判定规则（同时满足）：
    - CH2 与 CH3 的 n_fit_points 都 > 0；
    - CH2 与 CH3 的 tanh_p0 为有限值，且不等于异常值 bad_val（默认 1e6）。
    """
    ch2_n = np.asarray(ch2_n_fit_points, dtype=np.int32)
    ch3_n = np.asarray(ch3_n_fit_points, dtype=np.int32)
    ch2_p0 = np.asarray(ch2_tanh_p0, dtype=np.float64)
    ch3_p0 = np.asarray(ch3_tanh_p0, dtype=np.float64)
    n = min(ch2_n.shape[0], ch3_n.shape[0], ch2_p0.shape[0], ch3_p0.shape[0])
    ch2_n = ch2_n[:n]
    ch3_n = ch3_n[:n]
    ch2_p0 = ch2_p0[:n]
    ch3_p0 = ch3_p0[:n]

    ok_npts = (ch2_n > 0) & (ch3_n > 0)
    ok_ch2 = np.isfinite(ch2_p0) & (~np.isclose(ch2_p0, bad_val))
    ok_ch3 = np.isfinite(ch3_p0) & (~np.isclose(ch3_p0, bad_val))
    return ok_npts & ok_ch2 & ok_ch3

def cut_ch0_min_positive(ch0_min: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    条件：ch0_min > threshold（排除抑制信号）。
    输入: ch0_min 数组
    输出: bool 掩码
    """
    return ch0_min > threshold

def cut_ch0_max_saturation(max_ch0: np.ndarray, max_ch1: np.ndarray, max_val: float = 16382.0) -> np.ndarray:
    """
    条件：max_ch0 <= max_val 且 max_ch1 <= max_val（排除 CH0/CH1 饱和事例）。
    输入: max_ch0、max_ch1 数组
    输出: bool 掩码
    """
    return (max_ch0 <= max_val) & (max_ch1 <= max_val)

def cut_ch3_min(ch3_min: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    条件：ch3_min > threshold（默认要求 min(ch3) > 0）。
    输入: ch3_min 数组
    输出: bool 掩码
    """
    return ch3_min > threshold

def cut_ch5_self_trigger(max_ch5: np.ndarray, rt_threshold: float = 6000.0) -> np.ndarray:
    """
    条件：max_ch5 <= rt_threshold（排除随机触发）。
    输入: max_ch5 数组
    输出: bool 掩码
    """
    return max_ch5 <= rt_threshold

def cut_pedestal_3sigma(
    ch0_ped_mean: np.ndarray,
    ch1_ped_mean: np.ndarray,
    max_ch5: np.ndarray,
    rt_threshold: float = 6000.0,
    n_sigma: float = 3.0,
    min_rt_events: int = 10,) -> np.ndarray:
    """
    前沿基线 cut：使用随机触发事例 (max_ch5 > rt_threshold) 的 CH0/CH1 pedestal 分别拟合高斯，
    保留 |ch0_ped - μ0| <= n_sigma*σ0 且 |ch1_ped - μ1| <= n_sigma*σ1 的事件。
    若 RT 事例不足或 σ=0，对应通道返回全 True 掩码（不剔除）。
    输入: ch0_ped_mean, ch1_ped_mean, max_ch5
    输出: bool 掩码
    """
    n = ch0_ped_mean.shape[0]
    mask = np.ones(n, dtype=bool)
    rt_mask = max_ch5 > rt_threshold

    ch0_ped_rt = ch0_ped_mean[rt_mask]
    if ch0_ped_rt.size >= min_rt_events:
        ped_mu0 = float(ch0_ped_rt.mean())
        ped_sigma0 = float(ch0_ped_rt.std(ddof=1))
        if ped_sigma0 > 0.0:
            mask = mask & (np.abs(ch0_ped_mean - ped_mu0) <= n_sigma * ped_sigma0)

    ch1_ped_rt = ch1_ped_mean[rt_mask]
    if ch1_ped_rt.size >= min_rt_events:
        ped_mu1 = float(ch1_ped_rt.mean())
        ped_sigma1 = float(ch1_ped_rt.std(ddof=1))
        if ped_sigma1 > 0.0:
            mask = mask & (np.abs(ch1_ped_mean - ped_mu1) <= n_sigma * ped_sigma1)

    return mask

def cut_acv(
    max_ch4: np.ndarray,
    tmax_ch4: np.ndarray,
    trigger_threshold: float = 7060.0,
    t_ge_us: float = 40.0,
    sampling_interval_ns: float = 4.0,
    dt_min_us: float = 1.0,
    dt_max_us: float = 16.0,) -> np.ndarray:
    """
    acv cut：
    - 对 NaI 过阈事件 (max_ch4 >= trigger_threshold)，选取 Δt 非 [dt_min_us, dt_max_us] μs 的事例（反符合）；
    - 对 NaI 未过阈事件，视为“非 ACV 约束”，一律保留。
    参考 acv.py：Δt = t_Ge - t_CH4，t_CH4(μs) = tmax_ch4 * sampling_interval_ns * 1e-3。
    输入: max_ch4, tmax_ch4
    输出: bool 掩码（True = 通过 acv cut 的事例）
    """
    n = max_ch4.shape[0]
    tmax_ch4 = np.asarray(tmax_ch4, dtype=np.float64)[:n]
    max_ch4 = np.asarray(max_ch4, dtype=np.float64)[:n]
    # NaI 是否过阈
    nai_ok = max_ch4 >= trigger_threshold
    t_ch4_us = tmax_ch4 * sampling_interval_ns * 1e-3
    delta_t_us = t_ge_us - t_ch4_us
    # 对 NaI 过阈事件：acv = Δt 非 [dt_min_us, dt_max_us] 范围
    acv_mask = (delta_t_us < dt_min_us) | (delta_t_us > dt_max_us)
    # 最终通过条件：
    # - NaI 未过阈（nai_ok == False）：全部保留；
    # - NaI 过阈（nai_ok == True）：要求满足 acv_mask。
    return (~nai_ok) | (nai_ok & acv_mask)

def cut_mincut(
    ch0_min: np.ndarray,
    ch1_min: np.ndarray,
    max_ch4: np.ndarray,
    tmax_ch4: np.ndarray,
    n_sigma: float = 3.0,
    min_fit_events: int = 10,
    trigger_threshold: float = 7060.0,
    t_ge_us: float = 40.0,
    sampling_interval_ns: float = 4.0,
    dt_min_us: float = 1.0,
    dt_max_us: float = 16.0,) -> np.ndarray:
    """
    mincut：在 act 基础上，用 act 事例拟合 CH0min/CH1min 分布，
    保留 CH0min、CH1min 均在中心值 ± n_sigma*σ 内的事件。
    若拟合样本不足或 σ=0，对应通道不剔除。
    输入: ch0_min, ch1_min, max_ch5, max_ch4, tmax_ch4
    输出: bool 掩码
    """
    n = ch0_min.shape[0]
    mask = np.ones(n, dtype=bool)
    acv_mask = cut_acv(max_ch4, tmax_ch4, trigger_threshold, t_ge_us, sampling_interval_ns, dt_min_us, dt_max_us)
    fit_mask = ~acv_mask # 使用的是ACT事例

    ch0_min_fit = ch0_min[fit_mask]
    if ch0_min_fit.size >= min_fit_events:
        mu0 = float(ch0_min_fit.mean())
        sigma0 = float(ch0_min_fit.std(ddof=1))
        if sigma0 > 0.0:
            mask = mask & (np.abs(ch0_min - mu0) <= n_sigma * sigma0)

    ch1_min_fit = ch1_min[fit_mask]
    if ch1_min_fit.size >= min_fit_events:
        mu1 = float(ch1_min_fit.mean())
        sigma1 = float(ch1_min_fit.std(ddof=1))
        if sigma1 > 0.0:
            mask = mask & (np.abs(ch1_min - mu1) <= n_sigma * sigma1)

    return mask

def cut_time(
    time_mpl: np.ndarray,
    bad_intervals: Optional[Sequence[Tuple[float, float]]] = None,
    *,
    max_ch0: Optional[np.ndarray] = None,
    pre_mask: Optional[np.ndarray] = None,
    rate_threshold: float = CH0_TIME_BAND_BURST_RATE_THRESHOLD,
    year: Optional[int] = None,
    return_intervals: bool = False,) -> np.ndarray | Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    时间 cut：排除「红区」时间段内的事例，True 表示保留，False 表示剔除。
    - 若 bad_intervals 已显式传入（非 None），直接使用；
    - 若 bad_intervals 为 None 且同时提供 max_ch0 与 pre_mask，则按 ch0-time 规则在带内
      统计计数率并自算坏时间段（辅助逻辑均在本函数内部，不对外暴露）；
    - 否则回退到 CH0_TIME_EXCLUDE_BAD_INTERVALS_MPL；若仍为空则全部保留。

    嵌套的 ``_bad_intervals_total_days`` 亦绑定到 ``cut_time.bad_intervals_total_days``，供外部对
    ``return_intervals`` 得到的区间求总天数（须先至少调用过一次 ``cut_time``）。
    """
    def _bad_intervals_total_days(intervals: Sequence[Tuple[float, float]]) -> float:
        """统计坏时间区间总时长（单位：天，matplotlib date num 差值即天数）。"""
        total_days = 0.0
        for a, b in intervals:
            a_f = float(a)
            b_f = float(b)
            if np.isfinite(a_f) and np.isfinite(b_f) and b_f > a_f:
                total_days += (b_f - a_f)
        return float(total_days)

    cut_time.bad_intervals_total_days = _bad_intervals_total_days  # type: ignore[attr-defined]

    def _plot_window_mpl(yr: int) -> Tuple[float, float]:
        """与 ch0-time.py 相同：当年 5/20 0:00 — 6/10 24:00（matplotlib date num）。"""
        x_lo = mdates.date2num(datetime(yr, 5, 20))
        x_hi = mdates.date2num(datetime(yr, 6, 10))
        return x_lo, x_hi

    def _merge_bad_bins(
        edges: np.ndarray,
        bad_bin: np.ndarray,) -> List[Tuple[float, float]]:
        """相邻坏 bin 合并为 [left, right) 区间。"""
        out: List[Tuple[float, float]] = []
        nbin = int(bad_bin.size)
        i = 0
        while i < nbin:
            if not bad_bin[i]:
                i += 1
                continue
            j = i
            while j + 1 < nbin and bad_bin[j + 1]:
                j += 1
            out.append((float(edges[i]), float(edges[j + 1])))
            i = j + 1
        return out

    def _compute_exclude_intervals(
        tm: np.ndarray,
        m0: np.ndarray,
        yr: int,
        *,
        x_lo: Optional[float] = None,
        x_hi: Optional[float] = None,
        band_lo: float = CH0_BAND_BURST_LO,
        band_hi: float = CH0_BAND_BURST_HI,
        rt: float = rate_threshold,) -> List[Tuple[float, float]]:
        tm = np.asarray(tm, dtype=np.float64).reshape(-1)
        m0 = np.asarray(m0, dtype=np.float64).reshape(-1)
        n = min(tm.size, m0.size)
        tm = tm[:n]
        m0 = m0[:n]
        if x_lo is None or x_hi is None:
            x_lo, x_hi = _plot_window_mpl(yr)
        n_bins = max(30, int((x_hi - x_lo) * 24))
        band = (m0 >= band_lo) & (m0 <= band_hi)
        time_band = tm[band]
        counts, edges = np.histogram(time_band, bins=n_bins, range=(x_lo, x_hi))
        bin_width_days = float(edges[1] - edges[0])
        bin_width_min = bin_width_days * 24.0 * 60.0
        rate_per_min = counts.astype(np.float64) / bin_width_min
        bad = rate_per_min > float(rt)
        return _merge_bad_bins(edges, bad)

    def _build_bad_intervals() -> List[Tuple[float, float]]:
        assert max_ch0 is not None and pre_mask is not None
        t = np.asarray(time_mpl, dtype=np.float64).reshape(-1)
        x = np.asarray(max_ch0, dtype=np.float64).reshape(-1)
        n = min(t.size, x.size)
        if n == 0:
            return []
        t = t[:n]
        x = x[:n]
        m = np.asarray(pre_mask, dtype=bool).reshape(-1)[:n]
        t = t[m]
        x = x[m]
        if t.size == 0:
            return []
        y = int(mdates.num2date(float(t[0])).year if year is None else int(year))
        return _compute_exclude_intervals(t, x, y, rt=rate_threshold)

    if bad_intervals is not None:
        intervals: List[Tuple[float, float]] = [(float(a), float(b)) for a, b in bad_intervals]
    elif max_ch0 is not None and pre_mask is not None:
        intervals = _build_bad_intervals()
    else:
        intervals = [(float(a), float(b)) for a, b in CH0_TIME_EXCLUDE_BAD_INTERVALS_MPL]

    t = np.asarray(time_mpl, dtype=np.float64).reshape(-1)
    n = t.size
    ok = np.ones(n, dtype=bool)
    if not intervals:
        return (ok, intervals) if return_intervals else ok
    for a, b in intervals:
        ok &= ~((t >= a) & (t < b))
    return (ok, intervals) if return_intervals else ok

def cut_ch3ped_min(
    ch3ped_mean: np.ndarray,
    min_ch3: np.ndarray,
    *,
    sigma_yx: float = 20.0,
    n_sigma_residual: float = CH3PED_RESIDUAL_N_SIGMA,
    x_mean_band_half_sigma: float = CH3PED_X_MEAN_BAND_HALF_SIGMA,) -> np.ndarray:
    """
    CH3 ped-min：与 ch3ped-min.py 图中红区 ∩ 绿区的数学交集一致。
    - 红区：在全体有限点上 mean(x)、σ_x，保留 x ∈ [mean - h·σ_x, mean + h·σ_x]（h 默认 0.5）。
    - 绿区：在 |y-x|≤sigma_yx 上拟合 y=x+b，σ_res 为拟合残差标准差，保留 |y-(x+b)|≤n·σ_res（n 默认 6）。
    - 另要求 ch3ped_mean∈CH3PED_MIN_X_RANGE、min_ch3>0。
    红/绿任一侧无法统计时，该侧不额外收紧（全 True）。
    """
    x = np.asarray(ch3ped_mean, dtype=np.float64)
    y = np.asarray(min_ch3, dtype=np.float64)
    n = min(x.shape[0], y.shape[0])
    x = x[:n]
    y = y[:n]
    fin = np.isfinite(x) & np.isfinite(y)
    sig = float(sigma_yx)

    x_all = x[fin]
    if x_all.size >= 2:
        x_mean_all = float(np.mean(x_all))
        sigma_x = float(np.std(x_all, ddof=1))
        if np.isfinite(sigma_x) and sigma_x > 0.0:
            h = float(x_mean_band_half_sigma) * sigma_x
            red_ok = (x >= x_mean_all - h) & (x <= x_mean_all + h)
        else:
            red_ok = np.ones(n, dtype=bool)
    else:
        red_ok = np.ones(n, dtype=bool)

    in_yx_band = fin & (np.abs(y - x) <= sig)
    xf = x[in_yx_band]
    yf = y[in_yx_band]
    if xf.size >= 2:
        b_fit = float(np.mean(yf - xf))
        resid_fit = yf - xf - b_fit
        sigma_res = float(np.std(resid_fit, ddof=1))
        if np.isfinite(sigma_res) and sigma_res > 0.0:
            resid_all = y - x - b_fit
            band_ok = np.abs(resid_all) <= float(n_sigma_residual) * sigma_res
        else:
            band_ok = np.ones(n, dtype=bool)
    else:
        band_ok = np.ones(n, dtype=bool)

    return (
        (y > 0.0)
        & red_ok
        & band_ok)

def cut_pncut(
    base_mask: np.ndarray,
    max_ch0: np.ndarray,
    max_ch1: np.ndarray,
    fit_ch0_min: float = 3000.0,
    fit_ch0_max: float = 12000.0,
    n_sigma: float = 0.3,
    min_fit_events: int = 10,) -> np.ndarray:
    """
    pncut：
    - 先在传入 base_mask 选中的事件中，限定 CH0max 落在 (fit_ch0_min, fit_ch0_max) 区间，
      对对应的 (CH0max, CH1max) 做一次线性拟合 y = a * x + b；
    - 计算所有事件相对这条直线的残差 r = CH1max - (a * CH0max + b)；
    - 将 |r| <= n_sigma * σ 的事件（σ 为残差标准差）视为“主相关带”上的事件，输出 True。
    若拟合样本不足或 σ=0，则返回与 base_mask 相同的掩码。
    """
    def _pncut_fit_ab_sigma(
        base_mask: np.ndarray,
        max_ch0: np.ndarray,
        max_ch1: np.ndarray,
        fit_ch0_min: float = 3000.0,
        fit_ch0_max: float = 12000.0,
        min_fit_events: int = 10,) -> Optional[Tuple[float, float, float]]:
        """
        与 cut_pncut 相同的拟合：在 base_mask 且 CH0max∈(fit_ch0_min, fit_ch0_max) 上拟合直线，
        返回 (a, b, σ)；失败时返回 None。
        """
        n = max_ch0.shape[0]
        assert max_ch1.shape[0] == n and base_mask.shape[0] == n
        fit_mask = (
            base_mask
            & (max_ch0 > fit_ch0_min)
            & (max_ch0 < fit_ch0_max)
        )
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
        return (float(a), float(b), sigma)

    n = max_ch0.shape[0]
    assert max_ch1.shape[0] == n and base_mask.shape[0] == n

    fit = _pncut_fit_ab_sigma(
        base_mask,
        max_ch0,
        max_ch1,
        fit_ch0_min=fit_ch0_min,
        fit_ch0_max=fit_ch0_max,
        min_fit_events=min_fit_events,
    )
    if fit is None:
        return base_mask.copy()
    a, b, sigma = fit

    y_pred_all = a * max_ch0 + b
    resid_all = max_ch1 - y_pred_all
    band_mask = np.abs(resid_all) <= n_sigma * sigma

    return band_mask

def cut_bscut(
    tanh_p1: np.ndarray,
    rise_time_max_us: float = 0.8,) -> np.ndarray:
    """
    bscut：CH3 上升时间 rise_time = ln(19)/p1（单位 μs，与 ge-self/cut/bscut 中定义一致），
    选取 rise_time 落在 [rise_time_min_us, rise_time_max_us] 内的事例。
    p1<=0 或无效时对应位置为 False。
    """
    p1 = np.asarray(tanh_p1, dtype=np.float64)
    p1_safe = np.where(p1 > 1e-10, p1, np.nan)
    rise_us = np.where(np.isfinite(p1_safe), _LN_19 / p1_safe, np.nan)
    return (
        np.isfinite(rise_us)
        & (rise_us <= float(rise_time_max_us)))

if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="传统参数统计 / 累计 cut 能谱")
    _parser.add_argument(
        "--overlay-matlab-acv",
        action="store_true",
        help=(
            "在叠加能谱图中绘制本脚本目录下 DZL_vetospec_12kev_0615.fig 中的 "
            "「basic cut + ACV」曲线（散点，与 combine_wyf_plot 一致）"
        ),
    )
    _parser.add_argument(
        "--matlab-fig",
        type=Path,
        default=None,
        help="自定义 MATLAB .fig 路径（默认：本脚本目录 DZL_vetospec_12kev_0615.fig）",
    )
    _cli = _parser.parse_args()

    base_names = _list_base_names_from_ch0()
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
    ch3_ped_mean_values: List[np.ndarray] = []
    ch3_min_values: List[np.ndarray] = []
    ch3_tanh_p1_values: List[np.ndarray] = []
    time_mpl_values: List[np.ndarray] = []
    run_event_counts: List[int] = []

    for base_name in base_names:
        feats = _read_required_features_for_run(base_name)
        time_ns = _read_event_time_datetime64_ns_from_ch03(base_name, EPOCH_OFFSET_DEFAULT)
        time_mpl = _datetime64_ns_to_mpl_date(time_ns)
        n_run = min(int(feats["ch0_max_ch0"].shape[0]), int(time_mpl.shape[0]))
        if n_run <= 0:
            continue
        max_values.append(feats["ch0_max_ch0"][:n_run])
        ch0_min_values.append(feats["ch0_ch0_min"][:n_run])
        max_ch5_values.append(feats["ch5_max_ch5"][:n_run])
        ch0_ped_mean_values.append(feats["ch0_ch0ped_mean"][:n_run])
        ch1_ped_mean_values.append(feats["ch1_ch1ped_mean"][:n_run])
        ch1_min_values.append(feats["ch1_ch1_min"][:n_run])
        max_ch4_values.append(feats["ch4_max_ch4"][:n_run])
        tmax_ch4_values.append(feats["ch4_tmax_ch4"][:n_run])
        max_ch1_values.append(feats["ch1_max_ch1"][:n_run])
        tmax_ch0_values.append(feats["ch0_tmax_ch0"][:n_run])
        tmax_ch1_values.append(feats["ch1_tmax_ch1"][:n_run])
        ch2_n_fit_points_values.append(feats["ch2_n_fit_points"][:n_run])
        ch3_n_fit_points_values.append(feats["ch3_n_fit_points"][:n_run])
        ch2_tanh_p0_values.append(feats["ch2_tanh_p0"][:n_run])
        ch3_tanh_p0_values.append(feats["ch3_tanh_p0"][:n_run])
        ch3_ped_mean_values.append(feats["ch3_ch3ped_mean"][:n_run])
        ch3_min_values.append(feats["ch3_min_ch3"][:n_run])
        ch3_tanh_p1_values.append(feats["ch3_tanh_p1"][:n_run])
        time_mpl_values.append(time_mpl[:n_run])
        run_event_counts.append(n_run)

    max_ch0_all = np.concatenate(max_values, axis=0)
    ch0_min_all = np.concatenate(ch0_min_values, axis=0)
    max_ch5_all = np.concatenate(max_ch5_values, axis=0)
    ch0_ped_mean_all = np.concatenate(ch0_ped_mean_values, axis=0)
    ch1_ped_mean_all = np.concatenate(ch1_ped_mean_values, axis=0)
    ch1_min_all = np.concatenate(ch1_min_values, axis=0)
    max_ch4_all = np.concatenate(max_ch4_values, axis=0)
    tmax_ch4_all = np.concatenate(tmax_ch4_values, axis=0)
    max_ch1_all = np.concatenate(max_ch1_values, axis=0)
    tmax_ch0_all = np.concatenate(tmax_ch0_values, axis=0)
    tmax_ch1_all = np.concatenate(tmax_ch1_values, axis=0)
    ch2_n_fit_points_all = np.concatenate(ch2_n_fit_points_values, axis=0)
    ch3_n_fit_points_all = np.concatenate(ch3_n_fit_points_values, axis=0)
    ch2_tanh_p0_all = np.concatenate(ch2_tanh_p0_values, axis=0)
    ch3_tanh_p0_all = np.concatenate(ch3_tanh_p0_values, axis=0)
    ch3_ped_mean_all = np.concatenate(ch3_ped_mean_values, axis=0)
    ch3_min_all = np.concatenate(ch3_min_values, axis=0)
    ch3_tanh_p1_all = np.concatenate(ch3_tanh_p1_values, axis=0)
    time_mpl_all = np.concatenate(time_mpl_values, axis=0)

    # 先计算各个基础 cut
    m_fit_ok = cut_fit_success(
        ch2_n_fit_points_all,
        ch3_n_fit_points_all,
        ch2_tanh_p0_all,
        ch3_tanh_p0_all,)
    m_ch0_min = cut_ch0_min_positive(ch0_min_all)
    m_ch0_sat = cut_ch0_max_saturation(max_ch0_all, max_ch1_all)
    m_ch5_rt = cut_ch5_self_trigger(max_ch5_all)
    m_ped = cut_pedestal_3sigma(ch0_ped_mean_all, ch1_ped_mean_all, max_ch5_all)
    m_acv = cut_acv(max_ch4_all, tmax_ch4_all)
    m_mincut = cut_mincut(ch0_min_all, ch1_min_all, max_ch4_all, tmax_ch4_all)
    m_ch3ped_min = cut_ch3ped_min(ch3_ped_mean_all, ch3_min_all)
    m_bscut = cut_bscut(ch3_tanh_p1_all)

    # 与 ch0-time.py 一致：在 m1–m6 & m7(pncut) & m8(ch3ped_min) 上统计带内计数率，生成红区；
    # 此处另加 m_fit_ok（本脚本管线）；bscut 不参与红区统计（与 ch0-time 作图无关）。
    m_pre_m6 = (
        m_fit_ok
        & m_ch0_min
        & m_ch0_sat
        & m_ch5_rt
        & m_ped
        & m_acv
        & m_mincut
    )
    m_pn_for_ch0_time = cut_pncut(m_pre_m6, max_ch0_all, max_ch1_all)
    mask_pre_ch0_time = m_pre_m6 & m_pn_for_ch0_time & m_ch3ped_min
    _cut_time_rate = (
        float(CUT_TIME_RATE_THRESHOLD)
        if CUT_TIME_RATE_THRESHOLD is not None
        else float(CH0_TIME_BAND_BURST_RATE_THRESHOLD)
    )
    m_ch0_time, cut_time_intervals = cut_time(
        time_mpl_all,
        bad_intervals=None,
        max_ch0=max_ch0_all,
        pre_mask=mask_pre_ch0_time,
        rate_threshold=_cut_time_rate,
        return_intervals=True,
    )
    cut_time_removed_days = cut_time.bad_intervals_total_days(cut_time_intervals)
    exposure_days_after_cut_time = max(1e-12, float(EXPOSURE_DAYS) - cut_time_removed_days)
    print(
        f"[cut_time] excluded={cut_time_removed_days:.6f} day, "
        f"exposure: {EXPOSURE_DAYS:.6f} -> {exposure_days_after_cut_time:.6f} day"
    )

    # 能谱累计曲线：将 fit_success、pedestal 3σ、mincut、cut_time 合并为一步 basic
    m_basic_cut = m_fit_ok & m_ped & m_mincut & m_ch0_time
    # inhibitcut / maxcut / RTcut / acvcut 合并为一步 event cut
    m_event_cut = m_ch0_min & m_ch0_sat & m_ch5_rt & m_acv

    base_mask = (m_basic_cut & m_event_cut & m_ch3ped_min & m_bscut)
    m_pn = cut_pncut(base_mask , max_ch0_all, max_ch1_all)

    cut_steps = [
        ("total", np.ones(max_ch0_all.shape[0], dtype=bool)),
        ("basic", m_basic_cut),
        ("event_cut", m_event_cut),
        ("pncut", m_pn),
        #("ch3pedmin", m_ch3ped_min),
        #("bscut", m_bscut),
    ]

    _mf = _cli.matlab_fig
    if _mf is not None and not _mf.is_absolute():
        _mf = (_SCRIPT_DIR / _mf).resolve()

    plot_cumulative_cut_spectra(
        max_ch0_all=max_ch0_all,
        cut_steps=cut_steps,
        n_bins=SPECTRUM_N_BINS,
        e_min=0.1,
        e_max=12.0,
        overlay_matlab_acv=_cli.overlay_matlab_acv,
        matlab_fig_path=_mf,
        exposure_days_after_cut_time=exposure_days_after_cut_time,
    )

    # 测试示例z：比较 base_mask 与 m_ch0_t 的差异事件，并随机画 9 个 CH0/CH3 原始波形
    # plot_waveforms_from_mask_diff(
    #     mask_a=base_mask&m_pn,
    #     mask_b=base_mask&m_pn&m_ch0_t,
    #     base_names=base_names,
    #     run_event_counts=run_event_counts,
    #     ch0_3_dir=DATA_ROOT / "CH0-3",
    #     n_show=9,
    #     random_state=24,
    #     sampling_interval_ns=4.0,
    #     alpha=0.5,
    # )

    # # 测试示例：在同一差异集合中，按 CH3 spectral_centroid_mhz 最高优先选取 9 个事件绘制
    # plot_waveforms_from_mask_diff_top_spectral_centroid(
    #     mask_a=base_mask & m_pn & m_ch0_t & m_ch1_t,
    #     mask_b=base_mask & m_pn & m_ch0_t & ~m_ch1_t,
    #     base_names=base_names,
    #     run_event_counts=run_event_counts,
    #     ch0_3_dir=DATA_ROOT / "CH0-3",
    #     n_show=9,
    #     sampling_interval_ns=4.0,
    #     alpha=0.5,
    # )
