#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基础物理事例筛选 + 统计：

使用 CH0/CH5 的三个基准特征直接从参数文件中排除掉 inhibit 信号和 RT 信号，
对应于 30parameter&HDBSCAN.py 中的筛选条件：

    ch0_ch0_min > 0
    ch0_max_ch0 <= 16382
    ch5_max_ch5 <= 6000

其中三个特征分别来源于：
    - data/hdf5/raw_pulse/CH0_parameters  中的 dataset:
        max_ch0, ch0_min
    - data/hdf5/raw_pulse/CH5_parameters  中的 dataset:
        max_ch5
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def _discover_project_root() -> Path:
    """
    推断 DeepVibration 项目根目录：
    当前文件位于:
        .../python/data/ge-self/cut/basic+acv.py
    向上到 python，再上一层即项目根目录。
    """
    here = Path(__file__).resolve()
    # parents 索引：
    # 0: cut
    # 1: ge-self
    # 2: data
    # 3: python
    # 4: DeepVibration
    python_dir = here.parents[3]          # .../python
    return python_dir.parent              # .../DeepVibration

PROJECT_ROOT = _discover_project_root()
DATA_ROOT = PROJECT_ROOT / "data" / "hdf5" / "raw_pulse"
CH0_PARAM_DIR = DATA_ROOT / "CH0_parameters"
CH1_PARAM_DIR = DATA_ROOT / "CH1_parameters"
CH4_PARAM_DIR = DATA_ROOT / "CH4_parameters"
CH5_PARAM_DIR = DATA_ROOT / "CH5_parameters"
CH2_PARAM_DIR = DATA_ROOT / "CH2_parameters"
CH3_PARAM_DIR = DATA_ROOT / "CH3_parameters"

# CH3 ped-min 带状 cut 中 x 方向默认允许范围
CH3PED_MIN_X_RANGE = (960.0, 980.0)

# -----------------------------------------------------------------------------
# 时间 cut（cut_time）：与 ch0-time.py、parameterize/tradition/tradition.py 一致
# - 在 m0(fit) & m1–m6 & pncut & ch3ped_min 上统计 MAX(CH0)∈[band_lo,band_hi] 带内按时间分箱的计数率；
# - 时间窗为当年 5/20—6/10（matplotlib date num），分箱方式与 ch0-time.py 下图相同；
# - 若某箱计数率 > CH0_TIME_BAND_BURST_RATE_THRESHOLD (count/min)，该箱对应时间段内事例全部剔除；
# - 坏区间由 build_ch0_time_exclude_intervals_global 生成后传入 cut_time；亦可预存于
#   CH0_TIME_EXCLUDE_BAD_INTERVALS_MPL（bad_intervals=None 且未自算时回退）。
# -----------------------------------------------------------------------------
EPOCH_OFFSET_DEFAULT = 2.082816000000000e09
CH0_BAND_BURST_LO = 1250.0
CH0_BAND_BURST_HI = 1500.0
CH0_TIME_BAND_BURST_RATE_THRESHOLD = 0.5  # count/min，与 ch0-time.py 下图 rate_threshold 一致
CH0_TIME_EXCLUDE_BAD_INTERVALS_MPL: Tuple[Tuple[float, float], ...] = ()


def _ch0_time_plot_window_mpl(year: int) -> Tuple[float, float]:
    """与 ch0-time.py 相同：当年 5/20 0:00 — 6/10 24:00（matplotlib date num）。"""
    x_lo = mdates.date2num(datetime(year, 5, 20))
    x_hi = mdates.date2num(datetime(year, 6, 10))
    return x_lo, x_hi

def _merge_histogram_bad_bins_to_intervals(
    edges: np.ndarray,
    bad_bin: np.ndarray,) -> List[Tuple[float, float]]:
    """相邻坏 bin 合并为 [left, right) 区间（右端与 np.histogram 分箱一致）。"""
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

def compute_ch0_time_exclude_intervals_mpl(
    time_mpl: np.ndarray,
    max_ch0: np.ndarray,
    year: int,
    *,
    x_lo: Optional[float] = None,
    x_hi: Optional[float] = None,
    band_lo: float = CH0_BAND_BURST_LO,
    band_hi: float = CH0_BAND_BURST_HI,
    rate_threshold: float = CH0_TIME_BAND_BURST_RATE_THRESHOLD,) -> List[Tuple[float, float]]:
    """
    与 ch0-time.py 下图完全同一套分箱与计数率定义，返回「应排除」的时间段
    （matplotlib date num，左闭右开 [edges[i], edges[i+1]) 合并后的列表）。
    用于统计的样本应与 tradition 一致：m0 & m1–m6 & pncut & ch3ped_min（经 pre_mask 传入）。
    """
    time_mpl = np.asarray(time_mpl, dtype=np.float64).reshape(-1)
    max_ch0 = np.asarray(max_ch0, dtype=np.float64).reshape(-1)
    n = min(time_mpl.size, max_ch0.size)
    time_mpl = time_mpl[:n]
    max_ch0 = max_ch0[:n]
    if x_lo is None or x_hi is None:
        x_lo, x_hi = _ch0_time_plot_window_mpl(year)
    n_bins = max(30, int((x_hi - x_lo) * 24))
    band = (max_ch0 >= band_lo) & (max_ch0 <= band_hi)
    time_band = time_mpl[band]
    counts, edges = np.histogram(time_band, bins=n_bins, range=(x_lo, x_hi))
    bin_width_days = float(edges[1] - edges[0])
    bin_width_min = bin_width_days * 24.0 * 60.0
    rate_per_min = counts.astype(np.float64) / bin_width_min
    bad = rate_per_min > float(rate_threshold)
    return _merge_histogram_bad_bins_to_intervals(edges, bad)

def _map_param_to_ch03_files(ch0_param_path: Path) -> List[Path]:
    ch03_dir = PROJECT_ROOT / "data" / "hdf5" / "raw_pulse" / "CH0-3"
    stem = ch0_param_path.stem
    candidates = [ch03_dir / f"{stem}.h5"]
    if stem.endswith("_processed"):
        candidates.append(ch03_dir / f"{stem[:-10]}_processed.h5")
    else:
        candidates.append(ch03_dir / f"{stem}_processed.h5")
    return [c for c in candidates if c.is_file()]

def read_event_time_datetime64_ns_from_ch03(
    ch0_param_path: Path,
    epoch_offset: float = EPOCH_OFFSET_DEFAULT,) -> np.ndarray:
    """
    从 CH0-3 raw_pulse 读取 time_data，转为与 ch0-time.py 一致的 datetime64[ns]。
    """
    ch03_candidates = _map_param_to_ch03_files(ch0_param_path)
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
        raise RuntimeError(
            f"未能在 CH0-3 目录为 {ch0_param_path.name} 找到包含 time_data 的文件"
        )
    if time_data.size == 0:
        raise RuntimeError(f"{ch0_param_path.name} 对应的 time_data 为空")
    epoch_start = datetime(1970, 1, 1)
    eventtime = time_data - float(epoch_offset)
    dt = epoch_start + pd.to_timedelta(eventtime, unit="s")
    return dt.to_numpy(dtype="datetime64[ns]")

def datetime64_ns_to_mpl_date(time_ns: np.ndarray) -> np.ndarray:
    """datetime64[ns] -> matplotlib date float（与 ch0-time 一致）。"""
    time_ns = np.asarray(time_ns)
    time_py = time_ns.astype("M8[ms]").astype(datetime)
    return np.asarray(mdates.date2num(time_py), dtype=np.float64)

def build_ch0_time_exclude_intervals_global(
    time_mpl: np.ndarray,
    max_ch0: np.ndarray,
    pre_mask: Optional[np.ndarray] = None,
    *,
    rate_threshold: float = CH0_TIME_BAND_BURST_RATE_THRESHOLD,
    year: Optional[int] = None,) -> List[Tuple[float, float]]:
    """
    由外部已准备好的数组（可选 pre_mask）直接构建时间排除区间。

    - 不再在函数内部重复读取文件和重建 m1~mN；
    - pre_mask=True 的事件用于统计 1250–1500 带内计数率并生成坏时间段；
    - 返回值可直接传给 cut_time。
    """
    t = np.asarray(time_mpl, dtype=np.float64).reshape(-1)
    x = np.asarray(max_ch0, dtype=np.float64).reshape(-1)
    n = min(t.size, x.size)
    if n == 0:
        return []
    t = t[:n]
    x = x[:n]

    if pre_mask is not None:
        m = np.asarray(pre_mask, dtype=bool).reshape(-1)[:n]
        t = t[m]
        x = x[m]
    if t.size == 0:
        return []

    y = int(mdates.num2date(float(t[0])).year) if year is None else int(year)
    return compute_ch0_time_exclude_intervals_mpl(
        t,
        x,
        y,
        rate_threshold=rate_threshold,
    )

# CH2 ped–pedt cut：与 ch2ped-pedt.py 中 fit 矩形、竖向 ±3σ、y=x±1σ 同源
CH2_PED_PEDT_INNER_X_MAX = 2000.0  # 用于估计 a、σ_ped 的 ped 上界（fit_mask 与作图一致）
CH2_PED_PEDT_ADC_MAX = 16382.0  # ADC 开区间上界；ped/pedt 最终均约束在 (0, hi)

def _list_paired_param_files() -> List[Tuple[Path, Path]]:
    """
    基于 CH0_parameters 目录的文件名，寻找同时存在于 CH1_parameters 与 CH5_parameters 中的参数文件对。
    """
    if not CH0_PARAM_DIR.exists():
        raise FileNotFoundError(f"CH0_parameters 目录不存在: {CH0_PARAM_DIR}")
    if not CH1_PARAM_DIR.exists():
        raise FileNotFoundError(f"CH1_parameters 目录不存在: {CH1_PARAM_DIR}")
    if not CH4_PARAM_DIR.exists():
        raise FileNotFoundError(f"CH4_parameters 目录不存在: {CH4_PARAM_DIR}")
    if not CH5_PARAM_DIR.exists():
        raise FileNotFoundError(f"CH5_parameters 目录不存在: {CH5_PARAM_DIR}")

    ch0_files = sorted(
        name for name in os.listdir(CH0_PARAM_DIR)
        if name.lower().endswith((".h5", ".hdf5"))
    )
    if not ch0_files:
        raise FileNotFoundError(f"CH0_parameters 目录下未找到任何 h5 文件: {CH0_PARAM_DIR}")

    ch1_existing = {
        name for name in os.listdir(CH1_PARAM_DIR)
        if name.lower().endswith((".h5", ".hdf5"))
    }
    ch4_existing = {
        name for name in os.listdir(CH4_PARAM_DIR)
        if name.lower().endswith((".h5", ".hdf5"))
    }
    ch5_existing = {
        name for name in os.listdir(CH5_PARAM_DIR)
        if name.lower().endswith((".h5", ".hdf5"))
    }

    pairs: List[Tuple[Path, Path]] = []
    for name in ch0_files:
        if name in ch1_existing and name in ch4_existing and name in ch5_existing:
            pairs.append((CH0_PARAM_DIR / name, CH5_PARAM_DIR / name))

    if not pairs:
        raise RuntimeError(
            f"在 {CH0_PARAM_DIR}、{CH1_PARAM_DIR}、{CH4_PARAM_DIR} 与 {CH5_PARAM_DIR} 中未找到任何可配对的参数文件。"
        )

    return pairs

def _load_basic_features_for_run(
    ch0_param_path: Path,
    ch5_param_path: Path,) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,]:
    """
    从单个 run 的 CH0/CH5/CH1/CH4 参数文件中读取基准特征：
        - ch0_max_ch0    ← max_ch0
        - ch0_ch0_min    ← ch0_min
        - ch5_max_ch5    ← max_ch5
        - ch0_ped_mean   ← ch0ped_mean
        - ch1_ped_mean   ← ch1ped_mean
        - max_ch1        ← max_ch1
        - ch1_min        ← ch1_min
        - max_ch4        ← max_ch4
        - tmax_ch4       ← tmax_ch4
        - tmax_ch0       ← tmax_ch0
        - tmax_ch1       ← tmax_ch1

    返回:
        max_ch0, ch0_min, max_ch5, ch0_ped_mean, ch1_ped_mean, max_ch1, ch1_min, max_ch4, tmax_ch4, tmax_ch0, tmax_ch1
    """
    with h5py.File(ch0_param_path, "r") as f_ch0:
        if (
            "max_ch0" not in f_ch0
            or "ch0_min" not in f_ch0
            or "ch0ped_mean" not in f_ch0
            or "tmax_ch0" not in f_ch0
        ):
            raise KeyError(
                f"{ch0_param_path.name} 中缺少 max_ch0 / ch0_min / ch0ped_mean / tmax_ch0 数据集，"
                "请确认该文件由当前版本的 preprocessor.py 生成。"
            )
        max_ch0 = np.asarray(f_ch0["max_ch0"][...], dtype=np.float64)
        ch0_min = np.asarray(f_ch0["ch0_min"][...], dtype=np.float64)
        ch0_ped_mean = np.asarray(f_ch0["ch0ped_mean"][...], dtype=np.float64)
        tmax_ch0 = np.asarray(f_ch0["tmax_ch0"][...], dtype=np.float64)

    with h5py.File(ch5_param_path, "r") as f_ch5:
        if "max_ch5" not in f_ch5:
            raise KeyError(
                f"{ch5_param_path.name} 中缺少 max_ch5 数据集，"
                "请确认该文件由当前版本的 preprocessor.py 生成。"
            )
        max_ch5 = np.asarray(f_ch5["max_ch5"][...], dtype=np.float64)

    ch1_param_path = CH1_PARAM_DIR / ch0_param_path.name
    with h5py.File(ch1_param_path, "r") as f_ch1:
        if (
            "ch1ped_mean" not in f_ch1
            or "max_ch1" not in f_ch1
            or "ch1_min" not in f_ch1
            or "tmax_ch1" not in f_ch1
        ):
            raise KeyError(
                f"{ch1_param_path.name} 中缺少 ch1ped_mean / max_ch1 / ch1_min / tmax_ch1 之一，"
                "请确认该文件由当前版本的 preprocessor.py 生成。"
            )
        ch1_ped_mean = np.asarray(f_ch1["ch1ped_mean"][...], dtype=np.float64)
        max_ch1 = np.asarray(f_ch1["max_ch1"][...], dtype=np.float64)
        ch1_min = np.asarray(f_ch1["ch1_min"][...], dtype=np.float64)
        tmax_ch1 = np.asarray(f_ch1["tmax_ch1"][...], dtype=np.float64)

    ch4_param_path = CH4_PARAM_DIR / ch0_param_path.name
    with h5py.File(ch4_param_path, "r") as f_ch4:
        if "max_ch4" not in f_ch4 or "tmax_ch4" not in f_ch4:
            raise KeyError(
                f"{ch4_param_path.name} 中缺少 max_ch4 或 tmax_ch4 数据集，"
                "请确认该文件由当前版本的 preprocessor.py 生成。"
            )
        max_ch4 = np.asarray(f_ch4["max_ch4"][...], dtype=np.float64)
        tmax_ch4 = np.asarray(f_ch4["tmax_ch4"][...], dtype=np.float64)

    n0, nmin, n5, nped0, nped1, n1, nc1min, n4, nt4, nt0, nt1 = (
        max_ch0.shape[0],
        ch0_min.shape[0],
        max_ch5.shape[0],
        ch0_ped_mean.shape[0],
        ch1_ped_mean.shape[0],
        max_ch1.shape[0],
        ch1_min.shape[0],
        max_ch4.shape[0],
        tmax_ch4.shape[0],
        tmax_ch0.shape[0],
        tmax_ch1.shape[0],
    )
    n = min(n0, nmin, n5, nped0, nped1, n1, nc1min, n4, nt4, nt0, nt1)
    if not (n0 == nmin == n5 == nped0 == nped1 == n1 == nc1min == n4 == nt4 == nt0 == nt1):
        print(
            f"[警告] 事件数不一致，仅使用前 {n} 个事件。"
        )
    max_ch0 = max_ch0[:n]
    ch0_min = ch0_min[:n]
    max_ch5 = max_ch5[:n]
    ch0_ped_mean = ch0_ped_mean[:n]
    ch1_ped_mean = ch1_ped_mean[:n]
    max_ch1 = max_ch1[:n]
    ch1_min = ch1_min[:n]
    max_ch4 = max_ch4[:n]
    tmax_ch4 = tmax_ch4[:n]
    tmax_ch0 = tmax_ch0[:n]
    tmax_ch1 = tmax_ch1[:n]

    return (
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
    )

def _load_ch2_ped_pedt_aligned(ch0_param_path: Path, n_events: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 CH2_parameters 读取 ch2ped_mean / ch2pedt_mean，长度与 n_events 对齐（缺文件或 dataset 时以 NaN 填充）。
    """
    ch2_ped = np.full(n_events, np.nan, dtype=np.float64)
    ch2_pedt = np.full(n_events, np.nan, dtype=np.float64)
    ch2_path = CH2_PARAM_DIR / ch0_param_path.name
    if not ch2_path.is_file():
        return ch2_ped, ch2_pedt
    try:
        with h5py.File(ch2_path, "r") as f:
            if "ch2ped_mean" not in f or "ch2pedt_mean" not in f:
                return ch2_ped, ch2_pedt
            p = np.asarray(f["ch2ped_mean"][...], dtype=np.float64)
            t = np.asarray(f["ch2pedt_mean"][...], dtype=np.float64)
    except Exception:
        return ch2_ped, ch2_pedt
    n2 = min(int(p.shape[0]), int(t.shape[0]), n_events)
    if n2 > 0:
        ch2_ped[:n2] = p[:n2]
        ch2_pedt[:n2] = t[:n2]
    return ch2_ped, ch2_pedt

def _load_ch3_ped_min_aligned(ch0_param_path: Path, n_events: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 CH3_parameters 读取 ch3ped_mean / min_ch3，长度与 n_events 对齐（缺文件或 dataset 时以 NaN 填充）。
    """
    ch3_ped = np.full(n_events, np.nan, dtype=np.float64)
    ch3_min = np.full(n_events, np.nan, dtype=np.float64)
    ch3_path = CH3_PARAM_DIR / ch0_param_path.name
    if not ch3_path.is_file():
        return ch3_ped, ch3_min
    try:
        with h5py.File(ch3_path, "r") as f:
            if "ch3ped_mean" not in f or "min_ch3" not in f:
                return ch3_ped, ch3_min
            p = np.asarray(f["ch3ped_mean"][...], dtype=np.float64)
            m = np.asarray(f["min_ch3"][...], dtype=np.float64)
    except Exception:
        return ch3_ped, ch3_min
    n3 = min(int(p.shape[0]), int(m.shape[0]), n_events)
    if n3 > 0:
        ch3_ped[:n3] = p[:n3]
        ch3_min[:n3] = m[:n3]
    return ch3_ped, ch3_min

def _load_ch2_ch3_fit_quality_aligned(
    ch0_param_path: Path, n_events: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从 CH2/CH3 参数文件读取拟合质量相关量，并与 n_events 对齐：
        - ch2_n_fit_points
        - ch3_n_fit_points
        - ch2_tanh_p0
        - ch3_tanh_p0
    缺文件/缺数据集时返回默认值（n_fit_points=0, tanh_p0=NaN）。
    """
    ch2_n_fit_points = np.zeros(n_events, dtype=np.int32)
    ch3_n_fit_points = np.zeros(n_events, dtype=np.int32)
    ch2_tanh_p0 = np.full(n_events, np.nan, dtype=np.float64)
    ch3_tanh_p0 = np.full(n_events, np.nan, dtype=np.float64)

    ch2_path = CH2_PARAM_DIR / ch0_param_path.name
    if ch2_path.is_file():
        try:
            with h5py.File(ch2_path, "r") as f:
                if "n_fit_points" in f:
                    n2 = np.asarray(f["n_fit_points"][...], dtype=np.int32)
                    n = min(int(n2.shape[0]), n_events)
                    if n > 0:
                        ch2_n_fit_points[:n] = n2[:n]
                if "tanh_p0" in f:
                    p2 = np.asarray(f["tanh_p0"][...], dtype=np.float64)
                    n = min(int(p2.shape[0]), n_events)
                    if n > 0:
                        ch2_tanh_p0[:n] = p2[:n]
        except Exception:
            pass

    ch3_path = CH3_PARAM_DIR / ch0_param_path.name
    if ch3_path.is_file():
        try:
            with h5py.File(ch3_path, "r") as f:
                if "n_fit_points" in f:
                    n3 = np.asarray(f["n_fit_points"][...], dtype=np.int32)
                    n = min(int(n3.shape[0]), n_events)
                    if n > 0:
                        ch3_n_fit_points[:n] = n3[:n]
                if "tanh_p0" in f:
                    p3 = np.asarray(f["tanh_p0"][...], dtype=np.float64)
                    n = min(int(p3.shape[0]), n_events)
                    if n > 0:
                        ch3_tanh_p0[:n] = p3[:n]
        except Exception:
            pass

    return ch2_n_fit_points, ch3_n_fit_points, ch2_tanh_p0, ch3_tanh_p0

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

def cut_ch5_self_trigger(max_ch5: np.ndarray, rt_threshold: float = 6000.0) -> np.ndarray:
    """
    条件：max_ch5 <= rt_threshold（排除随机触发）。
    输入: max_ch5 数组
    输出: bool 掩码
    """
    return max_ch5 <= rt_threshold

def cut_ch3ped_min(
    ch3ped_mean: np.ndarray,
    min_ch3: np.ndarray,
    *,
    sigma_yx: float = 20.0,) -> np.ndarray:
    """
    CH3 ped-min 带状 cut：
    - 要求点位于 y=x 周边 sigma_yx 内（默认 |min_ch3 - ch3ped_mean| <= 20）
    - 并显式要求 min_ch3 > 0。
    """
    x = np.asarray(ch3ped_mean, dtype=np.float64)
    y = np.asarray(min_ch3, dtype=np.float64)
    n = min(x.shape[0], y.shape[0])
    x = x[:n]
    y = y[:n]
    fin = np.isfinite(x) & np.isfinite(y)
    sig = float(sigma_yx)
    x_lo, x_hi = CH3PED_MIN_X_RANGE
    return (
        fin
        & (x >= float(x_lo))
        & (x <= float(x_hi))
        & (np.abs(y - x) <= sig)
        & (y > 0.0)
    )

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
    fit_mask = ~acv_mask #act事例

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
    year: Optional[int] = None,) -> np.ndarray:
    """
    时间 cut：排除「红区」时间段内的事例，True 表示保留，False 表示剔除。
    - 若 bad_intervals 已显式传入（非 None），直接使用；
    - 若 bad_intervals 为 None 且同时提供 max_ch0 与 pre_mask，则按 ch0-time 规则在带内
      统计计数率并自算坏时间段（辅助逻辑均在本函数内部，不对外暴露）；
    - 否则回退到 CH0_TIME_EXCLUDE_BAD_INTERVALS_MPL；若仍为空则全部保留。
    """
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
        intervals: Sequence[Tuple[float, float]] = bad_intervals
    elif max_ch0 is not None and pre_mask is not None:
        intervals = _build_bad_intervals()
    else:
        intervals = CH0_TIME_EXCLUDE_BAD_INTERVALS_MPL

    t = np.asarray(time_mpl, dtype=np.float64).reshape(-1)
    n = t.size
    ok = np.ones(n, dtype=bool)
    if not intervals:
        return ok
    for a, b in intervals:
        ok &= ~((t >= a) & (t < b))
    return ok

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
    n = max_ch0.shape[0]
    assert max_ch1.shape[0] == n and base_mask.shape[0] == n

    # 拟合样本：在 base_mask 内，且 CH0max 在指定范围内
    fit_mask = (
        base_mask
        & (max_ch0 > fit_ch0_min)
        & (max_ch0 < fit_ch0_max)
    )
    x_fit = max_ch0[fit_mask]
    y_fit = max_ch1[fit_mask]

    if x_fit.size < min_fit_events:
        # 样本太少，不做任何额外切除
        return base_mask.copy()

    # 一阶多项式拟合：y = a * x + b
    a, b = np.polyfit(x_fit, y_fit, deg=1)
    y_pred_fit = a * x_fit + b
    resid_fit = y_fit - y_pred_fit
    sigma = float(resid_fit.std(ddof=1))
    if sigma <= 0.0:
        return base_mask.copy()

    # 对所有事件计算残差，并要求在 ± n_sigma * σ 内
    y_pred_all = a * max_ch0 + b
    resid_all = max_ch1 - y_pred_all
    band_mask = np.abs(resid_all) <= n_sigma * sigma

    return band_mask

# -----------------------------------------------------------------------------
# 临时测试函数（之后可删）
# -----------------------------------------------------------------------------
def _plot_ch0max_hist_passing_cuts(
    mask: np.ndarray,
    max_ch0: np.ndarray,
    max_ch1: np.ndarray,) -> None:
    """临时：绘制通过所有 cut 的事例的 CH0max 直方图和 CH0max vs CH1max 散点图。"""
    x = max_ch0[mask]
    y = max_ch1[mask]
    # ------------------------------------------------------------------
    # 图 1：按 spectrum.py 中的能量刻度与归一化画能谱
    # ADC -> 能量(keV) 线性变换: E = a * x + b
    # （系数与 spectrum.py 中保持一致）
    # ------------------------------------------------------------------
    a = 0.0008432447500464594
    b = -0.826976770117076
    energy_values = a * x + b

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})

    # 先用 numpy.histogram 得到计数和能量 bin 边界
    counts, bin_edges = np.histogram(energy_values, bins=500)
    bin_widths = np.diff(bin_edges)  # keV
    bin_centers = bin_edges[:-1] + bin_widths / 2.0

    # 归一化：rate = count / (0.5 kg * bin_width (keV) * 20 day)
    exposure_kg = 0.5
    exposure_days = 20.0
    denom = exposure_kg * bin_widths * exposure_days
    # 避免除以 0（理论上 bin_width 不会为 0，这里只是保险）
    denom[denom == 0] = np.inf
    rates = counts / denom

    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    ax1.bar(bin_centers, rates, width=bin_widths, color="C0", alpha=0.8, align="center")
    ax1.set_yscale("log")
    ax1.set_xlabel("Energy (keV)", fontsize=12)
    ax1.set_ylabel(r"Rate [counts / (keV·kg·day)]", fontsize=12)
    ax1.set_title(f"Energy spectrum of events passing all cuts (N={x.size})", fontsize=13)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()

    # 图 2：CH0max vs CH1max 散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=2, alpha=0.5, edgecolors="none")
    plt.xlabel("CH0 maximum amplitude (FADC)")
    plt.ylabel("CH1 maximum amplitude (FADC)")
    plt.title(f"CH0max vs CH1max (events passing all cuts, N={x.size})")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    pairs = _list_paired_param_files()
    print(f"找到 {len(pairs)} 个可配对的参数文件。")
    all_max_ch0: List[np.ndarray] = []
    all_ch0_min: List[np.ndarray] = []
    all_max_ch5: List[np.ndarray] = []
    all_ch0_ped_mean: List[np.ndarray] = []
    all_ch1_ped_mean: List[np.ndarray] = []
    all_max_ch1: List[np.ndarray] = []
    all_ch1_min: List[np.ndarray] = []
    all_max_ch4: List[np.ndarray] = []
    all_tmax_ch4: List[np.ndarray] = []
    all_tmax_ch0: List[np.ndarray] = []
    all_tmax_ch1: List[np.ndarray] = []
    all_ch2_ped_mean: List[np.ndarray] = []
    all_ch2_pedt_mean: List[np.ndarray] = []
    all_ch3_ped_mean: List[np.ndarray] = []
    all_ch3_min: List[np.ndarray] = []
    all_ch2_n_fit_points: List[np.ndarray] = []
    all_ch3_n_fit_points: List[np.ndarray] = []
    all_ch2_tanh_p0: List[np.ndarray] = []
    all_ch3_tanh_p0: List[np.ndarray] = []
    all_time_mpl: List[np.ndarray] = []
    for ch0_path, ch5_path in pairs:
        m0, cmin, m5, ped0, ped1, m1, c1min, m4, t4, tc0, tc1 = _load_basic_features_for_run(
            ch0_path, ch5_path
        )
        time_ns = read_event_time_datetime64_ns_from_ch03(ch0_path, EPOCH_OFFSET_DEFAULT)
        time_mpl = datetime64_ns_to_mpl_date(time_ns)
        ch2p, ch2t = _load_ch2_ped_pedt_aligned(ch0_path, m0.shape[0])
        ch3p, ch3m = _load_ch3_ped_min_aligned(ch0_path, m0.shape[0])
        ch2_nfit, ch3_nfit, ch2_p0, ch3_p0 = _load_ch2_ch3_fit_quality_aligned(ch0_path, m0.shape[0])

        n = min(
            m0.shape[0],
            cmin.shape[0],
            m5.shape[0],
            ped0.shape[0],
            ped1.shape[0],
            m1.shape[0],
            c1min.shape[0],
            m4.shape[0],
            t4.shape[0],
            tc0.shape[0],
            tc1.shape[0],
            ch2p.shape[0],
            ch2t.shape[0],
            ch3p.shape[0],
            ch3m.shape[0],
            ch2_nfit.shape[0],
            ch3_nfit.shape[0],
            ch2_p0.shape[0],
            ch3_p0.shape[0],
            time_mpl.shape[0],
        )
        if n <= 0:
            continue

        m0 = m0[:n]
        cmin = cmin[:n]
        m5 = m5[:n]
        ped0 = ped0[:n]
        ped1 = ped1[:n]
        m1 = m1[:n]
        c1min = c1min[:n]
        m4 = m4[:n]
        t4 = t4[:n]
        tc0 = tc0[:n]
        tc1 = tc1[:n]
        ch2p = ch2p[:n]
        ch2t = ch2t[:n]
        ch3p = ch3p[:n]
        ch3m = ch3m[:n]
        ch2_nfit = ch2_nfit[:n]
        ch3_nfit = ch3_nfit[:n]
        ch2_p0 = ch2_p0[:n]
        ch3_p0 = ch3_p0[:n]
        time_mpl = time_mpl[:n]

        all_ch2_ped_mean.append(ch2p)
        all_ch2_pedt_mean.append(ch2t)
        all_ch3_ped_mean.append(ch3p)
        all_ch3_min.append(ch3m)
        all_ch2_n_fit_points.append(ch2_nfit)
        all_ch3_n_fit_points.append(ch3_nfit)
        all_ch2_tanh_p0.append(ch2_p0)
        all_ch3_tanh_p0.append(ch3_p0)
        all_time_mpl.append(time_mpl)
        all_max_ch0.append(m0)
        all_ch0_min.append(cmin)
        all_max_ch5.append(m5)
        all_ch0_ped_mean.append(ped0)
        all_ch1_ped_mean.append(ped1)
        all_max_ch1.append(m1)
        all_ch1_min.append(c1min)
        all_max_ch4.append(m4)
        all_tmax_ch4.append(t4)
        all_tmax_ch0.append(tc0)
        all_tmax_ch1.append(tc1)
    max_ch0 = np.concatenate(all_max_ch0)
    ch0_min = np.concatenate(all_ch0_min)
    max_ch5 = np.concatenate(all_max_ch5)
    ch0_ped_mean = np.concatenate(all_ch0_ped_mean)
    ch1_ped_mean = np.concatenate(all_ch1_ped_mean)
    max_ch1 = np.concatenate(all_max_ch1)
    ch1_min = np.concatenate(all_ch1_min)
    max_ch4 = np.concatenate(all_max_ch4)
    tmax_ch4 = np.concatenate(all_tmax_ch4)
    tmax_ch0 = np.concatenate(all_tmax_ch0)
    tmax_ch1 = np.concatenate(all_tmax_ch1)
    ch2ped_mean = np.concatenate(all_ch2_ped_mean)
    ch2pedt_mean = np.concatenate(all_ch2_pedt_mean)
    ch3ped_mean = np.concatenate(all_ch3_ped_mean)
    min_ch3 = np.concatenate(all_ch3_min)
    ch2_n_fit_points = np.concatenate(all_ch2_n_fit_points)
    ch3_n_fit_points = np.concatenate(all_ch3_n_fit_points)
    ch2_tanh_p0 = np.concatenate(all_ch2_tanh_p0)
    ch3_tanh_p0 = np.concatenate(all_ch3_tanh_p0)
    time_mpl_all = np.concatenate(all_time_mpl)

    n_raw = max_ch0.shape[0]
    print(f"\n原始事件数: {n_raw}")

    m0 = cut_fit_success(ch2_n_fit_points, ch3_n_fit_points, ch2_tanh_p0, ch3_tanh_p0)
    n0 = int(m0.sum())
    print(f"cut_fit_success 单独使用后: {n0} / {n_raw}")

    # 1. 单独使用各 cut
    m1 = cut_ch0_min_positive(ch0_min)
    n1 = int(m1.sum())
    print(f"cut_ch0_min_positive 单独使用后: {n1} / {n_raw}")

    m2 = cut_ch0_max_saturation(max_ch0, max_ch1)
    n2 = int(m2.sum())
    print(f"cut_ch0_max_saturation 单独使用后: {n2} / {n_raw}")

    m3 = cut_ch5_self_trigger(max_ch5)
    n3 = int(m3.sum())
    print(f"cut_ch5_self_trigger 单独使用后: {n3} / {n_raw}")

    m4 = cut_pedestal_3sigma(ch0_ped_mean, ch1_ped_mean, max_ch5)
    n4 = int(m4.sum())
    print(f"cut_pedestal_3sigma 单独使用后: {n4} / {n_raw}")

    m5 = cut_acv(max_ch4, tmax_ch4)
    n5 = int(m5.sum())
    print(f"cut_acv 单独使用后: {n5} / {n_raw}")

    m6 = cut_mincut(ch0_min, ch1_min, max_ch4, tmax_ch4)
    n6 = int(m6.sum())
    print(f"cut_mincut 单独使用后: {n6} / {n_raw}")

    # m7 = cut_ch0max_tmax(m1 & m2 & m3 & m4 & m5 & m6, max_ch0, tmax_ch0)
    # n7 = int((m1 & m2 & m3 & m4 & m5 & m6 & m7).sum())
    # print(f"cut_ch0max_tmax 单独使用后: {n7} / {n_raw}")

    m8 = cut_pncut(m0 & m1 & m2 & m3 & m4 & m5 & m6 , max_ch0, max_ch1)
    n8 = int((m0 & m1 & m2 & m3 & m4 & m5 & m6 & m8).sum())
    print(f"cut_pncut 单独使用后: {n8} / {n_raw}")

    # m9 = cut_ch1max_tmax(m1 & m2 & m3 & m4 & m5 & m6  & m8, max_ch1, tmax_ch1)
    # n9 = int((m1 & m2 & m3 & m4 & m5 & m6 & m8 & m9).sum())
    # print(f"cut_ch1max_tmax 单独使用后: {n9} / {n_raw}")

    # m10 = cut_ch2ped_pedt_fit_range(ch2ped_mean, ch2pedt_mean)
    # n10 = int(m10.sum())
    # print(f"cut_ch2ped_pedt_fit_range(±3σ_ped∪y=x±1σ, ADC) 单独使用后: {n10} / {n_raw}")

    m11 = cut_ch3ped_min(ch3ped_mean, min_ch3)
    n11 = int(m11.sum())
    print(f"cut_ch3ped_min 单独使用后: {n11} / {n_raw}")


    m_pre_m6 = m0 & m1 & m2 & m3 & m4 & m5 & m6
    # 与 tradition / ch0-time 一致：m0–m6 & pncut(m8) & ch3ped_min(m11)
    mask_pre_time = m_pre_m6 & m8 & m11

    bad_intervals_mpl = build_ch0_time_exclude_intervals_global(
        time_mpl_all,
        max_ch0,
        pre_mask=mask_pre_time,
        rate_threshold=CH0_TIME_BAND_BURST_RATE_THRESHOLD,
    )
    m12 = cut_time(time_mpl_all, bad_intervals=bad_intervals_mpl)
    n12 = int(m12.sum())
    print(
        f"cut_time 单独使用后: {n12} / {n_raw} "
        f"(剔除 {n_raw - n12}, 坏时间段 {len(bad_intervals_mpl)} 段)"
    )

    # 2. 依次使用九种 cut 后剩余（与上面 m1~m9 定义一致）
    # mask = m1 & m2 & m3 & m4 & m5 & m6 & m7 & m9 & m10
    mask = m0 & m1 & m2 & m3 & m4 & m5 & m6 
    n_final = int(mask.sum())
    print(f"\n依次使用 cut 后最终剩余: {n_final} / {n_raw}")

    # 3. 绘制最终剩余事件的 CH0max 直方图 和 CH0max vs CH1max 散点图
    _plot_ch0max_hist_passing_cuts(mask, max_ch0, max_ch1)

