#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
combine_wyf_plot.py

将 MATLAB .fig 能谱与 combine_spectrum 的 UMAP cluster 能谱画在同一坐标轴内。

用法与 combine_spectrum.py 对齐：
- 可指定事件映射 HDF5（位置参数或 --hdf5）
- 可指定多个 cluster（--cluster C1 C2 ...）
- 未指定 --fig 时，自动使用「与 HDF5 同目录」下第一个 *.fig 文件

- 不绘制 Python 侧的 basic+acv cuts 能谱柱。
- MATLAB：raw data、basic cut 等仍为曲线/误差棒；“basic cut + ACV” 仅绘制成散点。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

import read_matlab_fig as rmf
import combine_spectrum as cs

ENERGY_A = cs.ENERGY_A
ENERGY_B = cs.ENERGY_B

# MATLAB 图例中 “basic cut + ACV” 的 DisplayName（与 read_matlab_fig 默认一致）
_MATLAB_ACV_NAME = "basic cut + ACV"


def _draw_matlab_series_only(
    ax: plt.Axes,
    series: List[Dict[str, Any]],
    axes_info: Dict[str, Any],
    *,
    zorder: float = 4,
) -> None:
    """绘制 MATLAB 曲线；其中 “basic cut + ACV” 仅画散点，其余为 errorbar/plot。"""
    y_is_log = axes_info.get("yscale", "linear") == "log"

    for i, s in enumerate(series):
        x = np.asarray(s["x"], dtype=np.float64).ravel()
        y = np.asarray(s["y"], dtype=np.float64).ravel()
        n = min(x.size, y.size)
        x, y = x[:n], y[:n]
        label = str(s.get("display_name", "")).strip() or f"{s.get('type', 'series')}[{i}]"
        fmt, markersize = "o", 1.8
        is_acv = label == _MATLAB_ACV_NAME

        if is_acv:
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
                zorder=zorder + 0.5,
            )
            continue

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
    cluster_label: str,
    *,
    zorder: float = 1,
    color: str = "C0",
) -> None:
    """仅绘制 UMAP cluster 能谱柱（不绘制 basic+acv cuts）。"""
    ax.bar(
        bin_centers, rates_cluster, width=bin_widths,
        color=color, alpha=0.45, align="center",
        label=f"UMAP+HDBSCAN cluster={cluster_label}",
        zorder=zorder,
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
    clusters: Sequence[int],
    bins: int,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if not hdf5_path.exists():
        print(f"[combine_spectrum] 事件映射 HDF5 不存在: {hdf5_path}")
        return None
    print(f"[combine_spectrum] 使用: {hdf5_path}")
    file_paths, event_file_indices, event_event_indices, labels = (
        cs._load_event_mapping(hdf5_path)
    )
    max_values_cluster = cs._compute_max_ch0_for_clusters(
        file_paths=file_paths,
        event_file_indices=event_file_indices,
        event_event_indices=event_event_indices,
        labels=labels,
        target_clusters=list(clusters),
    )
    energy_cluster = ENERGY_A * max_values_cluster + ENERGY_B
    if energy_cluster.size == 0:
        print("[combine_spectrum] 无可用事件。")
        return None

    bin_edges = np.linspace(cs.E_MIN_BIN, cs.E_MAX_BIN, bins + 1)
    ec_in = energy_cluster[
        (energy_cluster >= cs.E_MIN_BIN) & (energy_cluster <= cs.E_MAX_BIN)
    ]
    bin_centers, bin_widths, rates_cluster = cs._compute_rates_from_energy(
        ec_in, bin_edges
    )
    return bin_edges, bin_centers, bin_widths, rates_cluster


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


def _default_fig_in_hdf5_dir(hdf5_path: Path) -> Optional[Path]:
    """与 HDF5 同目录下第一个 *.fig（按文件名排序）。"""
    parent = hdf5_path.resolve().parent
    figs = sorted(parent.glob("*.fig"))
    if not figs:
        return None
    if len(figs) > 1:
        print(
            f"[MATLAB .fig] 目录 {parent} 下存在多个 .fig，将使用第一个（按名排序）: {figs[0].name}"
        )
    return figs[0]


def plot_spectrum_comparison(
    fig_path: Path,
    hdf5_path: Path,
    clusters: Sequence[int],
    bins: int = 500,
    plot_all: bool = False,
    save_path: Optional[Path] = None,
    dpi: int = 150,
) -> None:
    """
    将 MATLAB .fig 与 combine_spectrum（多 cluster 合并）能谱绘制在同一坐标轴内。
    """
    cluster_label = cs._format_cluster_label(list(clusters))

    matlab_series, matlab_axinfo = _load_matlab_data(
        fig_path, plot_all=plot_all, no_child_lines=True
    )
    cs_data = _load_combine_spectrum_data(hdf5_path, clusters=clusters, bins=bins)

    has_matlab = matlab_series is not None and matlab_axinfo is not None
    has_cs = cs_data is not None

    if not has_matlab and not has_cs:
        print("无可用数据，退出。")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5), dpi=dpi)

    bin_edges: Optional[np.ndarray] = None
    rates_cluster: Optional[np.ndarray] = None
    if has_cs:
        bin_edges, bin_centers, bin_widths, rates_cluster = cs_data
        assert rates_cluster is not None
        _draw_combine_spectrum_only(
            ax, bin_centers, bin_widths, rates_cluster, cluster_label
        )

    if has_matlab:
        _draw_matlab_series_only(ax, matlab_series, matlab_axinfo, zorder=4)

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
        ax.set_title(f"UMAP+HDBSCAN cluster={cluster_label}", fontsize=12)
    else:
        ax.set_yscale("log")
        assert bin_edges is not None and rates_cluster is not None
        x_lo = float(bin_edges[0])
        x_hi = float(bin_edges[-1])
        xl = matlab_axinfo.get("xlim")
        if xl is not None and len(xl) >= 2:
            x_lo = min(x_lo, float(xl[0]))
            x_hi = max(x_hi, float(xl[1]))
        ax.set_xlim(x_lo, x_hi)

        y_candidates: List[float] = []
        r = rates_cluster[np.isfinite(rates_cluster) & (rates_cluster > 0)]
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
        ax.set_title(
            f"能谱对比：MATLAB .fig + UMAP cluster={cluster_label}",
            fontsize=12,
        )
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"已保存: {save_path}")
    plt.show()


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="MATLAB .fig + UMAP cluster 能谱（与 combine_spectrum 一致：指定 h5、多 cluster）"
    )
    parser.add_argument(
        "hdf5_path",
        nargs="?",
        default=None,
        help="事件映射 HDF5；若不指定则用 combine_spectrum 默认路径。相对路径相对于本脚本目录。",
    )
    parser.add_argument(
        "--fig",
        type=Path,
        default=None,
        help="MATLAB .fig 路径；默认取「与 h5 同目录」下第一个 *.fig",
    )
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=None,
        help="与位置参数等效，便于只写 --cluster 时用",
    )
    parser.add_argument(
        "--cluster",
        type=int,
        nargs="+",
        default=[0],
        metavar="C",
        help="cluster label，可多个（如 --cluster 4 5 6），与 combine_spectrum 一致",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=500,
        help=f"能谱 bin 数，能量范围 [{cs.E_MIN_BIN}, {cs.E_MAX_BIN}] keV",
    )
    parser.add_argument(
        "--plot-all",
        action="store_true",
        help="MATLAB .fig 绘制全部序列",
    )
    parser.add_argument("--save", type=Path, default=None, help="保存路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent

    raw = args.hdf5_path or args.hdf5
    if raw:
        raw_path = Path(raw)
        if not raw_path.is_absolute():
            hdf5_path = (Path(__file__).resolve().parent / raw_path).resolve()
        else:
            hdf5_path = raw_path
    else:
        hdf5_path = cs._default_hdf5_path_relative_to_project_root()

    if args.fig is None:
        fig_path = _default_fig_in_hdf5_dir(hdf5_path)
        if fig_path is None:
            fallback = here / "DZL_vetospec_12kev_0615.fig"
            print(
                f"[MATLAB .fig] hdf5 同目录无 *.fig，回退: {fallback}"
            )
            fig_path = fallback
    else:
        fig_path = args.fig
        if not fig_path.is_absolute():
            fig_path = (Path(__file__).resolve().parent / fig_path).resolve()

    plot_spectrum_comparison(
        fig_path=fig_path,
        hdf5_path=hdf5_path,
        clusters=args.cluster,
        bins=args.bins,
        plot_all=args.plot_all,
        save_path=args.save,
    )

    print("combine_wyf_plot 完成。")


if __name__ == "__main__":
    main()
