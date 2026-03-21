#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
combined_plot.py

将 MATLAB .fig 能谱曲线与 combine_spectrum 能谱（cluster + basic+act）绘制在同一坐标轴内，
便于直接对比（柱状图在下、曲线在上）。

可选：--scatter 额外输出 CH0max vs CH1max 散点图。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

import read_matlab_fig as rmf
import combine_spectrum as cs

ENERGY_A = cs.ENERGY_A
ENERGY_B = cs.ENERGY_B


def _draw_matlab_series_only(
    ax: plt.Axes,
    series: List[Dict[str, Any]],
    axes_info: Dict[str, Any],
    *,
    zorder: float = 4,
) -> None:
    """仅绘制 MATLAB 曲线，不设置坐标轴范围与图例。"""
    y_is_log = axes_info.get("yscale", "linear") == "log"

    for i, s in enumerate(series):
        x = np.asarray(s["x"], dtype=np.float64).ravel()
        y = np.asarray(s["y"], dtype=np.float64).ravel()
        n = min(x.size, y.size)
        x, y = x[:n], y[:n]
        label = str(s.get("display_name", "")).strip() or f"{s.get('type', 'series')}[{i}]"
        fmt, markersize = "o", 1.8

        if "l_data" in s and "u_data" in s:
            lo = np.asarray(s["l_data"], dtype=np.float64).ravel()
            hi = np.asarray(s["u_data"], dtype=np.float64).ravel()
            m = min(lo.size, hi.size, n)
            lo, hi, x, y = lo[:m], hi[:m], x[:m], y[:m]
            if y_is_log:
                ok = y > 0
                x, y, lo, hi = x[ok], y[ok], lo[ok], hi[ok]
            yerr = np.vstack([lo, hi])
            ax.errorbar(
                x, y, yerr=yerr, fmt=fmt, label=label,
                capsize=2, capthick=0.8, elinewidth=1.2, markersize=markersize,
                zorder=zorder,
            )
        else:
            if y_is_log:
                ok = y > 0
                x, y = x[ok], y[ok]
            ax.plot(x, y, fmt, label=label, markersize=markersize, zorder=zorder)


def _draw_combine_spectrum_only(
    ax: plt.Axes,
    bin_centers: np.ndarray,
    bin_widths: np.ndarray,
    rates_cluster: np.ndarray,
    rates_basic: np.ndarray,
    cluster: int,
    *,
    zorder: float = 1,
) -> None:
    """仅绘制 combine 柱状图，不设置坐标轴范围与图例。"""
    ax.bar(
        bin_centers, rates_cluster, width=bin_widths,
        color="C0", alpha=0.45, align="center",
        label=f"UMAP+HDBSCAN cluster={cluster}",
        zorder=zorder,
    )
    ax.bar(
        bin_centers, rates_basic, width=bin_widths,
        color="C1", alpha=0.45, align="center",
        label="basic+act cuts",
        zorder=zorder + 0.1,
    )


def _load_matlab_data(
    fig_path: Path,
    plot_all: bool = False,
    no_child_lines: bool = True,
) -> tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    if not fig_path.exists():
        print(f"[MATLAB .fig] 文件不存在: {fig_path}")
        return None, None
    print(f"[MATLAB .fig] 读取: {fig_path}")
    raw = rmf.load_matlab_fig(fig_path)
    series = rmf.extract_xy_series(raw, include_child_lines=not no_child_lines)
    to_plot = rmf._series_for_plot(series, plot_all=plot_all)
    if not to_plot:
        print("[MATLAB .fig] 无匹配曲线。")
        return None, None
    axinfo = rmf.extract_axes_info(raw)
    return to_plot, axinfo


def _load_combine_spectrum_data(
    hdf5_path: Path,
    cluster: int,
    bins: int,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if not hdf5_path.exists():
        print(f"[combine_spectrum] 事件映射 HDF5 不存在: {hdf5_path}")
        return None
    print(f"[combine_spectrum] 使用: {hdf5_path}")
    file_paths, event_file_indices, event_event_indices, labels = (
        cs._load_event_mapping(hdf5_path)
    )
    max_values_cluster = cs._compute_max_ch0_for_cluster(
        file_paths=file_paths,
        event_file_indices=event_file_indices,
        event_event_indices=event_event_indices,
        labels=labels,
        target_cluster=cluster,
    )
    energy_cluster = ENERGY_A * max_values_cluster + ENERGY_B
    passed_max_ch0, passed_max_ch1, energy_basic = cs._load_basic_act_pass_events()

    energy_all = (
        np.concatenate([energy_cluster, energy_basic])
        if energy_basic.size > 0
        else energy_cluster
    )
    if energy_all.size == 0:
        print("[combine_spectrum] 无可用事件。")
        return None

    bin_edges = np.histogram_bin_edges(energy_all, bins=bins)
    bin_centers, bin_widths, rates_cluster = cs._compute_rates_from_energy(
        energy_cluster, bin_edges
    )
    _, _, rates_basic = cs._compute_rates_from_energy(energy_basic, bin_edges)
    return bin_edges, bin_centers, bin_widths, rates_cluster, rates_basic


def _collect_positive_y_from_matlab(
    series: List[Dict[str, Any]], axes_info: Dict[str, Any]
) -> np.ndarray:
    """用于对数轴下估计 y 范围。"""
    y_is_log = axes_info.get("yscale", "linear") == "log"
    ys: List[float] = []
    for s in series:
        y = np.asarray(s["y"], dtype=np.float64).ravel()
        if y_is_log:
            y = y[y > 0]
        ys.extend(y.tolist())
        if "l_data" in s and "u_data" in s:
            y = np.asarray(s["y"], dtype=np.float64).ravel()
            lo = np.asarray(s["l_data"], dtype=np.float64).ravel()
            hi = np.asarray(s["u_data"], dtype=np.float64).ravel()
            m = min(y.size, lo.size, hi.size)
            y = y[:m] - lo[:m]
            y = y[y > 0]
            ys.extend(y.tolist())
    return np.asarray(ys, dtype=np.float64)


def plot_spectrum_comparison(
    fig_path: Path,
    hdf5_path: Path,
    cluster: int = 1,
    bins: int = 500,
    plot_all: bool = False,
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> None:
    """
    将 MATLAB .fig 与 combine_spectrum 能谱绘制在同一坐标轴内。
    """
    matlab_series, matlab_axinfo = _load_matlab_data(
        fig_path, plot_all=plot_all, no_child_lines=True
    )
    cs_data = _load_combine_spectrum_data(hdf5_path, cluster=cluster, bins=bins)

    has_matlab = matlab_series is not None and matlab_axinfo is not None
    has_cs = cs_data is not None

    if not has_matlab and not has_cs:
        print("无可用数据，退出。")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5), dpi=dpi)

    bin_edges: Optional[np.ndarray] = None
    if has_cs:
        bin_edges, bin_centers, bin_widths, rates_cluster, rates_basic = cs_data
        _draw_combine_spectrum_only(
            ax, bin_centers, bin_widths, rates_cluster, rates_basic, cluster
        )

    if has_matlab:
        _draw_matlab_series_only(ax, matlab_series, matlab_axinfo, zorder=4)

    # 仅一侧数据时沿用该侧轴样式
    if has_matlab and not has_cs:
        if matlab_axinfo.get("yscale", "linear") == "log":
            ax.set_yscale("log")
        xl = matlab_axinfo.get("xlim")
        if xl is not None and len(xl) >= 2:
            ax.set_xlim(float(xl[0]), float(xl[1]))
        yl = matlab_axinfo.get("ylim")
        if yl is not None and len(yl) >= 2:
            ax.set_ylim(float(yl[0]), float(yl[1]))
        if matlab_axinfo.get("xlabel"):
            ax.set_xlabel(matlab_axinfo["xlabel"])
        if matlab_axinfo.get("ylabel"):
            ax.set_ylabel(matlab_axinfo["ylabel"])
        ax.set_title("MATLAB .fig 能谱", fontsize=12)
    elif has_cs and not has_matlab:
        ax.set_yscale("log")
        if bin_edges is not None:
            ax.set_xlim(float(bin_edges[0]), float(bin_edges[-1]))
        ax.set_xlabel("Energy (keV)", fontsize=12)
        ax.set_ylabel(r"Rate [counts / (keV·kg·day)]", fontsize=12)
        ax.set_title("UMAP+HDBSCAN cluster vs basic+act", fontsize=12)
    else:
        # 合并：对数 y，x/y 范围取并集
        ax.set_yscale("log")
        assert bin_edges is not None
        x_lo = float(bin_edges[0])
        x_hi = float(bin_edges[-1])
        xl = matlab_axinfo.get("xlim")
        if xl is not None and len(xl) >= 2:
            x_lo = min(x_lo, float(xl[0]))
            x_hi = max(x_hi, float(xl[1]))
        ax.set_xlim(x_lo, x_hi)

        y_candidates: List[float] = []
        r = np.concatenate([rates_cluster, rates_basic])
        r = r[np.isfinite(r) & (r > 0)]
        if r.size:
            y_candidates.extend([float(r.min()), float(r.max())])
        ym = _collect_positive_y_from_matlab(matlab_series, matlab_axinfo)
        if ym.size:
            y_candidates.extend([float(ym.min()), float(ym.max())])
        yl = matlab_axinfo.get("ylim")
        if yl is not None and len(yl) >= 2 and float(yl[0]) > 0:
            y_candidates.extend([float(yl[0]), float(yl[1])])
        if y_candidates:
            y_min = min(y_candidates)
            y_max = max(y_candidates)
            if y_min > 0 and y_max > y_min:
                ax.set_ylim(y_min * 0.8, y_max * 1.15)

        ax.set_xlabel("Energy (keV)", fontsize=12)
        ax.set_ylabel(r"Rate [counts / (keV·kg·day)]", fontsize=12)
        ax.set_title("能谱对比：MATLAB .fig + UMAP cluster / basic+act", fontsize=12)
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"已保存: {save_path}")
    plt.show()


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_fig = here / "DZL_vetospec_12kev_0615.fig"

    parser = argparse.ArgumentParser(
        description="将 MATLAB .fig 与 combine_spectrum 能谱绘制在同一坐标轴内"
    )
    parser.add_argument("--fig", type=Path, default=default_fig, help="MATLAB .fig 路径")
    parser.add_argument("--hdf5", type=Path, default=None, help="事件映射 HDF5 路径")
    parser.add_argument("--cluster", type=int, default=3, help="cluster label")
    parser.add_argument("--bins", type=int, default=500, help="能谱 bin 数")
    parser.add_argument("--plot-all", action="store_true", help="MATLAB .fig 绘制全部序列")
    parser.add_argument("--save", type=Path, default=None, help="保存路径")
    parser.add_argument(
        "--scatter",
        action="store_true",
        help="额外输出 CH0max vs CH1max 散点图",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    hdf5_path = args.hdf5 or cs._default_hdf5_path_relative_to_project_root()
    if args.hdf5 is not None and not args.hdf5.is_absolute():
        hdf5_path = (Path(__file__).resolve().parent / args.hdf5).resolve()

    plot_spectrum_comparison(
        fig_path=args.fig,
        hdf5_path=hdf5_path,
        cluster=args.cluster,
        bins=args.bins,
        plot_all=args.plot_all,
        save_path=args.save,
    )

    if args.scatter:
        passed_max_ch0, passed_max_ch1, _ = cs._load_basic_act_pass_events()
        if passed_max_ch0.size > 0:
            plt.figure(figsize=(8, 6))
            plt.scatter(passed_max_ch0, passed_max_ch1, s=2, alpha=0.5, edgecolors="none")
            plt.xlabel("CH0 maximum amplitude (FADC)")
            plt.ylabel("CH1 maximum amplitude (FADC)")
            plt.title(f"CH0max vs CH1max (basic+act cuts, N={passed_max_ch0.size})")
            plt.tight_layout()
            plt.show()

    print("combined_plot 完成。")


if __name__ == "__main__":
    main()
