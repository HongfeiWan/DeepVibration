#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通过基础 cut 且 **MAX(CH0) ∈ [0, 1500] ADC** 的事件，按**时间**分箱绘制计数率（纵轴 count/min，横轴时间）。

时间：CH0-3 HDF5 的 time_data，eventtime = time_data - epoch_offset（与 preprocessor / tradition 一致）。

时间分箱：改脚本内 TIME_HIST_BINS_PER_DAY / MIN / MAX，或命令行 ``--time-bins``、``--bins-per-day``。
"""

from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import List

import h5py
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

EPOCH_OFFSET_DEFAULT = 2.082816000000000e09

# 计数率子样本：仅统计该 MAX(CH0) 区间内的通过事件
CH0_RATE_BAND_LO = 1250.0
CH0_RATE_BAND_HI = 1500.0

# 参考阈值：高于该计数率（counts/min）的时间箱用红色底纹标出，并绘制红色虚线
RATE_REF_COUNTS_PER_MIN = 1.0

# 时间轴直方图分箱（自动模式）：
#   n_bins = clamp( span_days * TIME_HIST_BINS_PER_DAY, MIN, MAX )
# 默认 96 ≈ 每天 96 格、约每 15 分钟一格（与旧版 span_days * 24 * 4 一致）。
TIME_HIST_BINS_PER_DAY = 12
TIME_HIST_MIN_BINS = 40
TIME_HIST_MAX_BINS = 2000

# 与 tradition.py / python/utils/plotstyle.md 一致
_PLOT_TICK = 12
_PLOT_AXIS = 16


def _discover_project_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent.parent.parent.parent


def _load_basic_module(project_root: Path):
    basic_py = project_root / "python" / "data" / "ge-self" / "cut" / "basic+acv.py"
    if not basic_py.is_file():
        raise FileNotFoundError(f"未找到 basic+acv.py: {basic_py}")
    spec = importlib.util.spec_from_file_location("basic_acv_module", str(basic_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {basic_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _map_param_to_ch03_files(project_root: Path, param_file: Path) -> List[Path]:
    ch03_dir = project_root / "data" / "hdf5" / "raw_pulse" / "CH0-3"
    stem = param_file.stem
    candidates = [ch03_dir / f"{stem}.h5"]
    if stem.endswith("_processed"):
        candidates.append(ch03_dir / f"{stem[:-10]}_processed.h5")
    else:
        candidates.append(ch03_dir / f"{stem}_processed.h5")
    return [c for c in candidates if c.is_file()]


def _read_time_for_param(
    param_file: Path,
    project_root: Path,
    epoch_offset: float,
) -> np.ndarray:
    ch03_candidates = _map_param_to_ch03_files(project_root, param_file)
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
            f"未能在 CH0-3 目录为 {param_file.name} 找到包含 time_data 的文件"
        )
    epoch_start = datetime(1970, 1, 1)
    eventtime = time_data - float(epoch_offset)
    dt = epoch_start + pd.to_timedelta(eventtime, unit="s")
    return dt.to_numpy(dtype="datetime64[ns]")


def _apply_plotstyle_font() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
        }
    )


def _resolve_time_bin_count(
    span_days: float,
    *,
    time_bins: int,
    bins_per_day: float | None,
) -> int:
    """由时间跨度得到直方图箱数：固定箱数优先，否则按「每天多少箱」缩放。"""
    if time_bins > 0:
        return max(2, int(time_bins))
    bpd = float(TIME_HIST_BINS_PER_DAY if bins_per_day is None else bins_per_day)
    n = int(max(span_days, 1e-9) * bpd)
    return max(TIME_HIST_MIN_BINS, min(TIME_HIST_MAX_BINS, n))


def main() -> None:
    ap = argparse.ArgumentParser(description="CH0 子区间时间计数率（见文件头说明）。")
    ap.add_argument(
        "--time-bins",
        type=int,
        default=0,
        help="时间轴直方图总箱数（覆盖整段数据时间跨度）。0 表示按 --bins-per-day 自动算。",
    )
    ap.add_argument(
        "--bins-per-day",
        type=float,
        default=None,
        metavar="N",
        help=(
            "自动分箱时：每天划成约 N 个等宽时间箱（默认用脚本内 TIME_HIST_BINS_PER_DAY）。"
            "调大 → 时间分辨率更细、单箱计数更少。"
        ),
    )
    args = ap.parse_args()

    project_root = _discover_project_root()
    basic_mod = _load_basic_module(project_root)
    epoch_offset = float(EPOCH_OFFSET_DEFAULT)

    all_time: List[np.ndarray] = []
    n_skip = 0
    pairs = basic_mod._list_paired_param_files()

    for ch0_path, ch5_path in pairs:
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
            ) = basic_mod._load_basic_features_for_run(ch0_path, ch5_path)
            ch3ped_mean, min_ch3 = basic_mod._load_ch3_ped_min_aligned(
                ch0_path, max_ch0.shape[0]
            )
            m1 = basic_mod.cut_ch0_min_positive(ch0_min)
            m2 = basic_mod.cut_ch0_max_saturation(max_ch0, max_ch1)
            m3 = basic_mod.cut_ch5_self_trigger(max_ch5)
            m4 = basic_mod.cut_pedestal_3sigma(ch0_ped_mean, ch1_ped_mean, max_ch5)
            m5 = basic_mod.cut_acv(max_ch4, tmax_ch4)
            m6 = basic_mod.cut_mincut(ch0_min, ch1_min, max_ch4, tmax_ch4)
            m7 = basic_mod.cut_pncut(m1 & m2 & m3 & m4 & m5 & m6, max_ch0, max_ch1)
            m8 = basic_mod.cut_ch3ped_min(ch3ped_mean, min_ch3)
            mask = m1 & m2 & m3 & m4 & m5 & m6 & m7 & m8
            if not np.any(mask):
                continue

            time_arr = _read_time_for_param(ch0_path, project_root, epoch_offset)
            n = min(mask.shape[0], time_arr.shape[0], max_ch0.shape[0])
            if n == 0:
                continue
            mask = mask[:n]
            time_arr = time_arr[:n]
            max_ch0_n = np.asarray(max_ch0[:n], dtype=np.float64)
            ch0_band = (max_ch0_n >= CH0_RATE_BAND_LO) & (max_ch0_n <= CH0_RATE_BAND_HI)
            sel = mask & ch0_band
            if not np.any(sel):
                continue
            all_time.append(time_arr[sel])
        except Exception as e:
            n_skip += 1
            print(f"[跳过] {ch0_path.name}: {e}")

    if not all_time:
        raise RuntimeError("没有成功读取到任何可用事件的 time_data")

    time_all = np.concatenate(all_time)
    print("=" * 70)
    print(
        f"MAX(CH0)∈[{CH0_RATE_BAND_LO:g},{CH0_RATE_BAND_HI:g}] 且通过 cut，用于计数率的事件总数: {time_all.size}"
    )
    print(f"跳过文件数: {n_skip}")
    print("=" * 70)

    time_py = time_all.astype("M8[ms]").astype(datetime)
    time_float = mdates.date2num(time_py)
    t_lo, t_hi = float(np.min(time_float)), float(np.max(time_float))
    span_days = max(t_hi - t_lo, 1e-9)
    n_bins = _resolve_time_bin_count(
        span_days,
        time_bins=int(args.time_bins),
        bins_per_day=args.bins_per_day,
    )
    print(f"[时间分箱] span≈{span_days:.4f} d, n_bins={n_bins}")

    counts, edges = np.histogram(time_float, bins=n_bins, range=(t_lo, t_hi))
    bin_width_days = float(edges[1] - edges[0])
    bin_width_min = bin_width_days * 24.0 * 60.0
    rate_per_min = counts.astype(np.float64) / bin_width_min

    _apply_plotstyle_font()
    plt.rcParams.setdefault("axes.unicode_minus", False)

    fig, ax = plt.subplots(figsize=(11, 5))
    pos = rate_per_min[rate_per_min > 0]
    if pos.size:
        floor = float(np.min(pos)) * 0.25
    else:
        floor = 1e-12
    rate_draw = np.maximum(rate_per_min, floor)

    for i in range(len(rate_per_min)):
        if rate_per_min[i] > RATE_REF_COUNTS_PER_MIN:
            ax.axvspan(
                edges[i],
                edges[i + 1],
                facecolor="red",
                alpha=0.22,
                zorder=0,
                linewidth=0,
            )

    ax.bar(
        edges[:-1],
        rate_draw,
        width=np.diff(edges),
        align="edge",
        color="C0",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.25,
        zorder=2,
    )
    ax.axhline(
        RATE_REF_COUNTS_PER_MIN,
        color="red",
        linestyle="--",
        linewidth=1.2,
        zorder=4,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Time", fontsize=_PLOT_AXIS)
    ax.set_ylabel(r"Rate [counts / min] (log)", fontsize=_PLOT_AXIS)
    ax.tick_params(axis="both", which="major", labelsize=_PLOT_TICK)
    ax.set_xlim(t_lo, t_hi)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.legend(
        handles=[
            Patch(
                facecolor="red",
                edgecolor="none",
                alpha=0.22,
                label="cut region",
            ),
            Line2D(
                [0],
                [0],
                color="red",
                linestyle="--",
                linewidth=1.2,
                label=f"Rate = {RATE_REF_COUNTS_PER_MIN:g} counts/min",
            ),
        ],
        loc="best",
        fontsize=_PLOT_TICK,
        framealpha=0.95,
    )

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
