#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
combine_wyf_plot.py

将 MATLAB .fig 能谱与 combine_spectrum 的 UMAP cluster 能谱画在同一坐标轴内。

用法与 combine_spectrum.py 对齐：
- 可指定事件映射 HDF5（位置参数或 --hdf5）
- 可指定多个 cluster（--cluster C1 C2 ...）
- 或使用 --spec 简记多文件合并（见 --spec 帮助），例如 2.1、3.1.2.1（文件 …_step2_cluster1_step_3_cluster2.h5 内仅簇 1）
- 未指定 --fig 时，自动使用「与 HDF5 同目录」下第一个 *.fig 文件（多 --spec 时取第一个 spec 对应 h5 所在目录）

- 不绘制 Python 侧的 basic+acv cuts 能谱柱。
- MATLAB：raw data、basic cut 等仍为曲线/误差棒；“basic cut + ACV” 仅绘制成散点。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MultipleLocator

import read_matlab_fig as rmf
import combine_spectrum as cs

ENERGY_A = cs.ENERGY_A
ENERGY_B = cs.ENERGY_B

# 与 30parameter&HDBSCAN_step2 / step3 中 OUTPUT_BASENAME_DEFAULT 一致
DEFAULT_EVENTMAP_BASENAME = "ge_30param_umap_hdbscan_eventmap"

# MATLAB 图例中 “basic cut + ACV” 的 DisplayName（与 read_matlab_fig 默认一致）
_MATLAB_ACV_NAME = "basic cut + ACV"


def _choose_major_step(span: float) -> float:
    """
    与 tradition.py 的 x 轴主刻度密度保持一致。

    span 单位 keV。经验阈值：保证全范围 (~12 keV) 时是 0.5 keV，
    放大后可逐级变细，但不会一开始就变成 0.25 这种“密一倍”的效果。
    """
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


def _attach_tradition_like_x_ticks(ax: plt.Axes) -> None:
    """让 x 轴刻度密度随缩放/平移动态更新（与 tradition.py 一致）。"""
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:g}"))

    def _update_x_locator(_ax: plt.Axes) -> None:
        x0, x1 = _ax.get_xlim()
        step = _choose_major_step(abs(float(x1) - float(x0)))
        _ax.xaxis.set_major_locator(MultipleLocator(step))
        # minor ticks：每个 major 再细分 5 份
        _ax.xaxis.set_minor_locator(AutoMinorLocator(5))

    _update_x_locator(ax)
    ax.callbacks.connect("xlim_changed", _update_x_locator)


def _parse_spec_token(
    token: str,
    *,
    basename: str,
    hdf5_dir: Path,) -> Tuple[Path, Optional[List[int]]]:
    """
    将「step 简记」解析为 HDF5 路径（与 step2/step3 产物命名一致）。

    - 1.c   → {basename}.h5，且仅 cluster 标签 c（第一层 eventmap）
    - 2.p   → {basename}_step2_cluster{p}.h5
    - 2.p.k[.k2…] → 同上文件，且仅文件内 event_cluster_labels 为 k（可多段）
    - 3.p.s → {basename}_step2_cluster{p}_step_3_cluster{s}.h5（p=step2 父簇，s=文件名里 _step_3_cluster{s}）
    - 3.p.s.k[.k2…] → 同上文件，且仅标签 k…（多段表示多簇合并到本 token）

    返回 (path, fixed_clusters_or_none)。第二项非 None 时只使用这些标签；为 None 时由
    --cluster 或「文件内全部标签」决定。
    """
    t = token.strip()
    if not t:
        raise ValueError("空的 spec token")
    parts = t.split(".")
    if len(parts) < 2:
        raise ValueError(
            f"无效 spec「{token}」：至少需要两段，例如 1.0、2.1、3.1.0"
        )
    try:
        step = int(parts[0])
        nums = [int(x) for x in parts[1:]]
    except ValueError as e:
        raise ValueError(f"spec「{token}」含非整数段: {e}") from e

    if step == 1:
        if len(nums) != 1:
            raise ValueError(f"1 层需要恰好两段，如 1.3，收到: {token}")
        return hdf5_dir / f"{basename}.h5", [nums[0]]
    if step == 2:
        if len(nums) < 1:
            raise ValueError(f"2 层至少需要两段，如 2.1，收到: {token}")
        p = nums[0]
        path = hdf5_dir / f"{basename}_step2_cluster{p}.h5"
        if len(nums) == 1:
            return path, None
        return path, nums[1:]
    if step == 3:
        if len(nums) < 2:
            raise ValueError(
                f"3 层至少需要三段（文件 p.s），如 3.1.2；若指定文件内簇再加 .k，如 3.1.2.1，收到: {token}"
            )
        p2, p3 = nums[0], nums[1]
        path = hdf5_dir / f"{basename}_step2_cluster{p2}_step_3_cluster{p3}.h5"
        if len(nums) == 2:
            return path, None
        return path, nums[2:]
    raise ValueError(f"不支持的 step={step}（仅支持 1/2/3），token={token}")

def _resolve_target_clusters(
    labels: np.ndarray,
    fixed_clusters: Optional[List[int]],
    global_clusters: Optional[Sequence[int]],) -> List[int]:
    """在单个 eventmap 内确定要纳入能谱的 cluster 标签列表。"""
    if fixed_clusters is not None:
        return [int(c) for c in fixed_clusters]
    if global_clusters is not None:
        return [int(c) for c in global_clusters]
    u = np.unique(labels.astype(int))
    return [int(x) for x in u.tolist()]


def _load_combine_spectrum_data_multi(
    specs: Sequence[Tuple[Path, Optional[List[int]]]],
    *,
    global_clusters: Optional[Sequence[int]],
    bins: int,) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    多个 eventmap 文件：对每个文件按 cluster 取 max_ch0，再合并能量后做一次直方图。
    """
    energy_chunks: List[np.ndarray] = []
    for hdf5_path, fixed_clusters in specs:
        if not hdf5_path.exists():
            print(f"[combine_spectrum] 事件映射 HDF5 不存在: {hdf5_path}")
            return None
        with h5py.File(hdf5_path, "r") as f:
            labels = f["event_cluster_labels"][...]
        target = _resolve_target_clusters(labels, fixed_clusters, global_clusters)
        print(f"[combine_spectrum] 使用: {hdf5_path.name}  cluster={target}")
        file_paths, event_file_indices, event_event_indices, labels = (
            cs._load_event_mapping(hdf5_path)
        )
        max_values_cluster = cs._compute_max_ch0_for_clusters(
            file_paths=file_paths,
            event_file_indices=event_file_indices,
            event_event_indices=event_event_indices,
            labels=labels,
            target_clusters=target,
        )
        if max_values_cluster.size:
            energy_chunks.append(ENERGY_A * max_values_cluster + ENERGY_B)

    if not energy_chunks:
        print("[combine_spectrum] 无可用事件。")
        return None
    energy_cluster = np.concatenate(energy_chunks)
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


def _format_merge_label(spec_strings: Sequence[str]) -> str:
    return "+".join(spec_strings)


def _draw_matlab_series_only(
    ax: plt.Axes,
    series: List[Dict[str, Any]],
    axes_info: Dict[str, Any],
    *,
    zorder: float = 4,) -> None:
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
    color: str = "C0",) -> None:
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
    no_child_lines: bool = True,) -> tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
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
    bins: int,) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
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
    *,
    merge_specs: Optional[Sequence[str]] = None,
    eventmap_basename: str = DEFAULT_EVENTMAP_BASENAME,
    merge_global_clusters: Optional[Sequence[int]] = None,
) -> None:
    """
    将 MATLAB .fig 与 combine_spectrum（多 cluster 合并）能谱绘制在同一坐标轴内。

    merge_specs 非空时：按「简记」解析多个 eventmap，合并所有事例能量后绘制一条柱谱。
    spec 中若已写文件内簇（如 3.1.2.1），优先生效；否则 merge_global_clusters 非 None 时
    对各文件用同一组标签；否则对 step2/step3 使用文件内全部标签。
    """
    if merge_specs:
        cluster_label = _format_merge_label(merge_specs)
        hdf5_dir = cs.PROJECT_ROOT / "data" / "hdf5"
        parsed: List[Tuple[Path, Optional[List[int]]]] = []
        for s in merge_specs:
            parsed.append(
                _parse_spec_token(
                    s, basename=eventmap_basename, hdf5_dir=hdf5_dir
                )
            )
        cs_data = _load_combine_spectrum_data_multi(
            parsed,
            global_clusters=merge_global_clusters,
            bins=bins,
        )
    else:
        cluster_label = cs._format_cluster_label(list(clusters))
        cs_data = _load_combine_spectrum_data(hdf5_path, clusters=clusters, bins=bins)

    matlab_series, matlab_axinfo = _load_matlab_data(
        fig_path, plot_all=plot_all, no_child_lines=True
    )

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

    # x 轴刻度密度：与 tradition.py 保持一致（含交互缩放后自动更新）
    _attach_tradition_like_x_ticks(ax)
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
        description=(
            "MATLAB .fig + UMAP cluster 能谱。"
            "单文件模式与 combine_spectrum 一致；--spec 模式按简记合并多个 step 的 eventmap 事例。"
        )
    )
    parser.add_argument(
        "hdf5_path",
        nargs="?",
        default=None,
        help="事件映射 HDF5；若不指定则用 combine_spectrum 默认路径。相对路径相对于本脚本目录。使用 --spec 时可省略。",
    )
    parser.add_argument(
        "--spec",
        type=str,
        nargs="+",
        default=None,
        metavar="S",
        help=(
            "多文件合并简记（可多个，空格分隔）。"
            "1.c → {basename}.h5 仅簇 c；"
            "2.p → step2 文件；2.p.k[.k2…] → 该文件内仅标签 k…；"
            "3.p.s → step3 文件（p=step2 父簇，s=文件名 _step_3_cluster{s}）；"
            "3.p.s.k[.k2…] → 该文件内仅标签 k…（例：3.1.2.1 对应 …_step2_cluster1_step_3_cluster2.h5 且仅簇 1）。"
            "与 --cluster 同用时仅对未在 spec 里写 .k 的项生效；"
            "未写 .k 且未传 --cluster 时用文件内全部 event_cluster_labels。"
        ),
    )
    parser.add_argument(
        "--eventmap-basename",
        type=str,
        default=DEFAULT_EVENTMAP_BASENAME,
        metavar="NAME",
        help=f"--spec 解析时文件名主体（默认 {DEFAULT_EVENTMAP_BASENAME}）",
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
        nargs="*",
        default=None,
        metavar="C",
        help=(
            "cluster 标签（可多个）。单文件模式：默认 0。"
            "--spec 模式：省略则对 step2/step3 文件使用各文件内全部簇；"
            "指定则对每个文件用同一组标签筛选。"
        ),
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
    hdf5_dir = cs.PROJECT_ROOT / "data" / "hdf5"

    if args.spec:
        first, _ = _parse_spec_token(
            args.spec[0],
            basename=args.eventmap_basename,
            hdf5_dir=hdf5_dir,
        )
        hdf5_path = first
    else:
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

    clusters_single: List[int] = (
        args.cluster if args.cluster else [0]
    )
    merge_global: Optional[List[int]] = (
        args.cluster if args.cluster else None
    )

    plot_spectrum_comparison(
        fig_path=fig_path,
        hdf5_path=hdf5_path,
        clusters=clusters_single,
        bins=args.bins,
        plot_all=args.plot_all,
        save_path=args.save,
        merge_specs=args.spec,
        eventmap_basename=args.eventmap_basename,
        merge_global_clusters=merge_global,
    )

    print("combine_wyf_plot 完成。")


if __name__ == "__main__":
    main()
