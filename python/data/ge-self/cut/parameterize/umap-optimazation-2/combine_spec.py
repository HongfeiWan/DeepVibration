#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 combine_wyf_plot_multistep.py 的 --spec / --cluster 选择逻辑，
接到 tradition.py 的累计 cut 能谱流程中。

与 combine_wyf_plot_multistep.py 的区别：
- 不读取 MATLAB .fig；
- 不做两条谱的叠加对比；
- 而是把 spec/cluster 选出的事件映射为全局布尔掩码，
  作为 tradition 累计 cut 图中的最后一个 cut。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_TRADITION_DIR = _THIS_DIR.parent / "tradition"
if str(_TRADITION_DIR) not in sys.path:
    sys.path.insert(0, str(_TRADITION_DIR))

import combine_spectrum as cs
import tradition as tr


DEFAULT_EVENTMAP_BASENAME = "ge_30param_umap_hdbscan_eventmap"


def _parse_spec_token(
    token: str,
    *,
    basename: str,
    hdf5_dir: Path,
) -> Tuple[Path, Optional[List[int]]]:
    """
    将 step 简记解析为 eventmap 路径与文件内固定 cluster。

    - 1.c   -> {basename}.h5，且仅 cluster=c
    - 2.p   -> {basename}_step2_cluster{p}.h5
    - 2.p.k[.k2...] -> 同文件内仅簇 k...
    - 3.p.s -> {basename}_step2_cluster{p}_step_3_cluster{s}.h5
    - 3.p.s.k[.k2...] -> 同文件内仅簇 k...
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
        parent_cluster = nums[0]
        path = hdf5_dir / f"{basename}_step2_cluster{parent_cluster}.h5"
        return (path, None) if len(nums) == 1 else (path, nums[1:])

    if step == 3:
        if len(nums) < 2:
            raise ValueError(
                f"3 层至少需要三段（文件 p.s），如 3.1.2；"
                f"若指定文件内簇再加 .k，如 3.1.2.1，收到: {token}"
            )
        parent_cluster, child_cluster = nums[0], nums[1]
        path = (
            hdf5_dir
            / f"{basename}_step2_cluster{parent_cluster}_step_3_cluster{child_cluster}.h5"
        )
        return (path, None) if len(nums) == 2 else (path, nums[2:])

    raise ValueError(f"不支持的 step={step}（仅支持 1/2/3），token={token}")


def _resolve_target_clusters(
    labels: np.ndarray,
    fixed_clusters: Optional[List[int]],
    global_clusters: Optional[Sequence[int]],
) -> List[int]:
    if fixed_clusters is not None:
        return [int(c) for c in fixed_clusters]
    if global_clusters is not None:
        return [int(c) for c in global_clusters]
    unique_labels = np.unique(labels.astype(int))
    return [int(x) for x in unique_labels.tolist()]


def _resolve_single_hdf5_path(raw: Optional[str]) -> Path:
    if raw is None:
        return cs._default_hdf5_path_relative_to_project_root()
    raw_path = Path(raw)
    if raw_path.is_absolute():
        return raw_path
    return (_THIS_DIR / raw_path).resolve()


def _build_run_lookup(
    base_names: Sequence[str],
    run_event_counts: Sequence[int],
) -> Tuple[Dict[str, Tuple[int, int]], int]:
    lookup: Dict[str, Tuple[int, int]] = {}
    start = 0
    for base_name, n_events in zip(base_names, run_event_counts):
        lookup[Path(base_name).name] = (start, int(n_events))
        start += int(n_events)
    return lookup, start


def _collect_selected_global_indices(
    hdf5_path: Path,
    *,
    fixed_clusters: Optional[List[int]],
    global_clusters: Optional[Sequence[int]],
    run_lookup: Dict[str, Tuple[int, int]],
) -> np.ndarray:
    file_paths, event_file_indices, event_event_indices, labels = cs._load_event_mapping(
        hdf5_path
    )
    target_clusters = _resolve_target_clusters(labels, fixed_clusters, global_clusters)
    print(f"[spec] 使用: {hdf5_path.name}  cluster={target_clusters}")

    selected = np.isin(labels.astype(int), np.asarray(target_clusters, dtype=int))
    selected_rows = np.nonzero(selected)[0]
    if selected_rows.size == 0:
        return np.array([], dtype=np.int64)

    global_indices: List[int] = []
    missing_files: set[str] = set()
    invalid_events = 0

    for row_idx in selected_rows:
        file_idx = int(event_file_indices[row_idx])
        event_idx = int(event_event_indices[row_idx])
        if file_idx < 0 or file_idx >= len(file_paths):
            invalid_events += 1
            continue

        src_name = Path(file_paths[file_idx]).name
        run_meta = run_lookup.get(src_name)
        if run_meta is None:
            missing_files.add(src_name)
            continue

        run_start, run_count = run_meta
        if 0 <= event_idx < run_count:
            global_indices.append(run_start + event_idx)
        else:
            invalid_events += 1

    if missing_files:
        preview = ", ".join(sorted(missing_files)[:5])
        suffix = " ..." if len(missing_files) > 5 else ""
        print(
            f"[spec] 警告: {len(missing_files)} 个源文件未在 tradition 数据集中找到: "
            f"{preview}{suffix}"
        )
    if invalid_events:
        print(f"[spec] 警告: {invalid_events} 个 event 索引越界或文件索引无效，已跳过")

    if not global_indices:
        return np.array([], dtype=np.int64)
    return np.unique(np.asarray(global_indices, dtype=np.int64))


def _build_cluster_mask(
    *,
    spec_tokens: Optional[Sequence[str]],
    global_clusters: Optional[Sequence[int]],
    eventmap_basename: str,
    single_hdf5_path: Path,
    run_lookup: Dict[str, Tuple[int, int]],
    n_total: int,
) -> np.ndarray:
    hdf5_dir = cs.PROJECT_ROOT / "data" / "hdf5"

    if spec_tokens:
        parsed_specs = [
            _parse_spec_token(
                token,
                basename=eventmap_basename,
                hdf5_dir=hdf5_dir,
            )
            for token in spec_tokens
        ]
    else:
        default_clusters = list(global_clusters) if global_clusters is not None else [0]
        parsed_specs = [(single_hdf5_path, default_clusters)]

    picked_chunks: List[np.ndarray] = []
    for hdf5_path, fixed_clusters in parsed_specs:
        picked = _collect_selected_global_indices(
            hdf5_path,
            fixed_clusters=fixed_clusters,
            global_clusters=global_clusters,
            run_lookup=run_lookup,
        )
        if picked.size:
            picked_chunks.append(picked)

    cluster_mask = np.zeros(int(n_total), dtype=bool)
    if not picked_chunks:
        print("[spec] 未选中任何事件。")
        return cluster_mask

    picked_all = np.unique(np.concatenate(picked_chunks))
    cluster_mask[picked_all] = True
    print(f"[spec] 最终 union 选中事件数: {int(cluster_mask.sum())}")
    return cluster_mask


def _load_tradition_inputs() -> Tuple[Dict[str, np.ndarray], List[str], List[int]]:
    base_names = tr._list_base_names_from_ch0()

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
    valid_base_names: List[str] = []

    for base_name in base_names:
        feats = tr._read_required_features_for_run(base_name)
        time_ns = tr._read_event_time_datetime64_ns_from_ch03(
            base_name, tr.EPOCH_OFFSET_DEFAULT
        )
        time_mpl = tr._datetime64_ns_to_mpl_date(time_ns)
        n_run = min(int(feats["ch0_max_ch0"].shape[0]), int(time_mpl.shape[0]))
        if n_run <= 0:
            continue

        valid_base_names.append(base_name)
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

    arrays = {
        "max_ch0_all": np.concatenate(max_values, axis=0),
        "ch0_min_all": np.concatenate(ch0_min_values, axis=0),
        "max_ch5_all": np.concatenate(max_ch5_values, axis=0),
        "ch0_ped_mean_all": np.concatenate(ch0_ped_mean_values, axis=0),
        "ch1_ped_mean_all": np.concatenate(ch1_ped_mean_values, axis=0),
        "ch1_min_all": np.concatenate(ch1_min_values, axis=0),
        "max_ch4_all": np.concatenate(max_ch4_values, axis=0),
        "tmax_ch4_all": np.concatenate(tmax_ch4_values, axis=0),
        "max_ch1_all": np.concatenate(max_ch1_values, axis=0),
        "tmax_ch0_all": np.concatenate(tmax_ch0_values, axis=0),
        "tmax_ch1_all": np.concatenate(tmax_ch1_values, axis=0),
        "ch2_n_fit_points_all": np.concatenate(ch2_n_fit_points_values, axis=0),
        "ch3_n_fit_points_all": np.concatenate(ch3_n_fit_points_values, axis=0),
        "ch2_tanh_p0_all": np.concatenate(ch2_tanh_p0_values, axis=0),
        "ch3_tanh_p0_all": np.concatenate(ch3_tanh_p0_values, axis=0),
        "ch3_ped_mean_all": np.concatenate(ch3_ped_mean_values, axis=0),
        "ch3_min_all": np.concatenate(ch3_min_values, axis=0),
        "ch3_tanh_p1_all": np.concatenate(ch3_tanh_p1_values, axis=0),
        "time_mpl_all": np.concatenate(time_mpl_values, axis=0),
    }
    return arrays, valid_base_names, run_event_counts


def _build_tradition_cut_steps(
    arrays: Dict[str, np.ndarray],
    cluster_mask: np.ndarray,
) -> Tuple[List[Tuple[str, np.ndarray]], float]:
    max_ch0_all = arrays["max_ch0_all"]
    ch0_min_all = arrays["ch0_min_all"]
    max_ch5_all = arrays["max_ch5_all"]
    ch0_ped_mean_all = arrays["ch0_ped_mean_all"]
    ch1_ped_mean_all = arrays["ch1_ped_mean_all"]
    ch1_min_all = arrays["ch1_min_all"]
    max_ch4_all = arrays["max_ch4_all"]
    tmax_ch4_all = arrays["tmax_ch4_all"]
    max_ch1_all = arrays["max_ch1_all"]
    time_mpl_all = arrays["time_mpl_all"]
    ch2_n_fit_points_all = arrays["ch2_n_fit_points_all"]
    ch3_n_fit_points_all = arrays["ch3_n_fit_points_all"]
    ch2_tanh_p0_all = arrays["ch2_tanh_p0_all"]
    ch3_tanh_p0_all = arrays["ch3_tanh_p0_all"]
    ch3_ped_mean_all = arrays["ch3_ped_mean_all"]
    ch3_min_all = arrays["ch3_min_all"]
    ch3_tanh_p1_all = arrays["ch3_tanh_p1_all"]

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

    m_pre_m6 = (
        m_fit_ok
        & m_ch0_min
        & m_ch0_sat
        & m_ch5_rt
        & m_ped
        & m_acv
        & m_mincut
    )
    m_pn_for_ch0_time = tr.cut_pncut(m_pre_m6, max_ch0_all, max_ch1_all)
    mask_pre_ch0_time = m_pre_m6 & m_pn_for_ch0_time & m_ch3ped_min
    cut_time_rate = (
        float(tr.CUT_TIME_RATE_THRESHOLD)
        if tr.CUT_TIME_RATE_THRESHOLD is not None
        else float(tr.CH0_TIME_BAND_BURST_RATE_THRESHOLD)
    )
    m_ch0_time, cut_time_intervals = tr.cut_time(
        time_mpl_all,
        bad_intervals=None,
        max_ch0=max_ch0_all,
        pre_mask=mask_pre_ch0_time,
        rate_threshold=cut_time_rate,
        return_intervals=True,
    )
    cut_time_removed_days = tr.cut_time.bad_intervals_total_days(cut_time_intervals)
    exposure_days_after_cut_time = max(
        1e-12, float(tr.EXPOSURE_DAYS) - cut_time_removed_days
    )
    print(
        f"[cut_time] excluded={cut_time_removed_days:.6f} day, "
        f"exposure: {tr.EXPOSURE_DAYS:.6f} -> {exposure_days_after_cut_time:.6f} day"
    )

    m_basic_cut = m_fit_ok & m_ped & m_mincut & m_ch0_time
    m_event_cut = m_ch0_min & m_ch0_sat & m_ch5_rt & m_acv
    base_mask = m_basic_cut & m_event_cut & m_ch3ped_min & m_bscut
    m_pn = tr.cut_pncut(base_mask, max_ch0_all, max_ch1_all)

    cut_steps = [
        #("total", np.ones(max_ch0_all.shape[0], dtype=bool)),
        ("traditional cut", base_mask&m_pn),
        ("cluster", cluster_mask),
    ]
    return cut_steps, exposure_days_after_cut_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "把 combine_wyf_plot_multistep.py 的 spec/cluster 选择结果，"
            "作为 tradition.py 累计能谱图中的最后一个 cut。"
        )
    )
    parser.add_argument(
        "hdf5_path",
        nargs="?",
        default=None,
        help="事件映射 HDF5；若不指定则使用 combine_spectrum 默认路径。使用 --spec 时可省略。",
    )
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=None,
        help="与位置参数等效，便于只写 --cluster 时用。",
    )
    parser.add_argument(
        "--spec",
        type=str,
        nargs="+",
        default=None,
        metavar="S",
        help=(
            "多文件合并简记（可多个，空格分隔）。"
            "1.c -> {basename}.h5 仅簇 c；"
            "2.p -> step2 文件；2.p.k[.k2…] -> 该文件内仅标签 k…；"
            "3.p.s -> step3 文件；"
            "3.p.s.k[.k2…] -> 该文件内仅标签 k…。"
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
        "--cluster",
        type=int,
        nargs="*",
        default=None,
        metavar="C",
        help=(
            "cluster 标签（可多个）。单文件模式默认 0。"
            "--spec 模式省略时，对 step2/step3 文件使用文件内全部簇；"
            "指定时则对未写 .k 的每个文件使用同一组标签。"
        ),
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=tr.SPECTRUM_N_BINS,
        help=f"能谱 bin 数，默认 {tr.SPECTRUM_N_BINS}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    single_hdf5_arg = args.hdf5_path
    if args.hdf5 is not None:
        single_hdf5_arg = str(args.hdf5)
    single_hdf5_path = _resolve_single_hdf5_path(single_hdf5_arg)

    arrays, base_names, run_event_counts = _load_tradition_inputs()
    run_lookup, n_total = _build_run_lookup(base_names, run_event_counts)

    cluster_mask = _build_cluster_mask(
        spec_tokens=args.spec,
        global_clusters=args.cluster,
        eventmap_basename=args.eventmap_basename,
        single_hdf5_path=single_hdf5_path,
        run_lookup=run_lookup,
        n_total=n_total,
    )

    if cluster_mask.shape[0] != n_total:
        raise RuntimeError(
            f"cluster_mask 长度 {cluster_mask.shape[0]} 与总事件数 {n_total} 不一致"
        )

    cut_steps, exposure_days_after_cut_time = _build_tradition_cut_steps(
        arrays, cluster_mask
    )
    tr.plot_cumulative_cut_spectra(
        max_ch0_all=arrays["max_ch0_all"],
        cut_steps=cut_steps,
        n_bins=args.bins,
        e_min=0.1,
        e_max=12.0,
        overlay_matlab_acv=False,
        matlab_fig_path=None,
        exposure_days_after_cut_time=exposure_days_after_cut_time,
    )


if __name__ == "__main__":
    main()
