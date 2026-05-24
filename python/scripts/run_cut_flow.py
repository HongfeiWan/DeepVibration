#!/usr/bin/env python
"""Report legacy cut-flow counts and cumulative survivors per run and in total."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from analysis.cuts import (  # noqa: E402
    acv_mask,
    bscut_mask,
    ch3ped_min_mask,
    cut_time,
    fit_success_mask,
    inhibit_mask,
    mincut_mask,
    pedestal_3sigma_mask,
    pncut_mask,
    rt_mask,
    saturation_mask,
)
from analysis.io import find_project_root, pair_parameter_files, read_raw_pulse_event_time_mpl  # noqa: E402
from analysis.io.parameters import load_parameter_file  # noqa: E402
from analysis.parallel import add_parallel_arguments, iter_completed  # noqa: E402
from analysis.pipelines import evaluate_cut_flow  # noqa: E402


STAGE_GROUPS = {
    "basic_mincut": "basic_cut",
    "basic_pedestal_3sigma": "basic_cut",
    "basic_time_keep": "basic_cut",
    "basic_fit_success": "basic_cut",
    "event_ch3ped_min": "event_cut",
    "event_acv_keep": "event_cut",
    "event_rt_keep": "event_cut",
    "event_inhibit_keep": "event_cut",
    "event_saturation_keep": "event_cut",
    "pncut": "pncut",
    "event_bscut": "bscut",
}

GROUP_END_STAGE = {
    "basic_cut": "basic_fit_success",
    "event_cut": "event_saturation_keep",
    "pncut": "pncut",
    "bscut": "event_bscut",
}


def _count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(mask, dtype=bool)))


def _truncate_to_common_length(arrays: Iterable[np.ndarray]) -> int:
    sizes = [np.asarray(arr).size for arr in arrays]
    return min(sizes) if sizes else 0


def _load_run_arrays(run_name: str, files: Dict[str, str], *, project_root: str) -> Dict[str, np.ndarray]:
    ch0 = load_parameter_file(files["CH0"], ["max_ch0", "ch0_min", "ch0ped_mean"])
    ch1 = load_parameter_file(files["CH1"], ["max_ch1", "ch1_min", "ch1ped_mean"])
    ch2 = load_parameter_file(files["CH2"], ["n_fit_points", "tanh_p0"])
    ch3 = load_parameter_file(files["CH3"], ["n_fit_points", "tanh_p0", "tanh_p1", "ch3ped_mean", "min_ch3"])
    ch4 = load_parameter_file(files["CH4"], ["max_ch4", "tmax_ch4"])
    ch5 = load_parameter_file(files["CH5"], ["max_ch5"])
    time_mpl = read_raw_pulse_event_time_mpl(run_name, project_root=project_root)

    n = _truncate_to_common_length(
        [
            ch0["max_ch0"],
            ch0["ch0_min"],
            ch0["ch0ped_mean"],
            ch1["max_ch1"],
            ch1["ch1_min"],
            ch1["ch1ped_mean"],
            ch2["n_fit_points"],
            ch2["tanh_p0"],
            ch3["n_fit_points"],
            ch3["tanh_p0"],
            ch3["tanh_p1"],
            ch3["ch3ped_mean"],
            ch3["min_ch3"],
            ch4["max_ch4"],
            ch4["tmax_ch4"],
            ch5["max_ch5"],
            time_mpl,
        ]
    )
    if n <= 0:
        raise ValueError(f"{run_name}: no aligned events found")

    arrays = {
        "max_ch0": np.asarray(ch0["max_ch0"], dtype=np.float64)[:n],
        "ch0_min": np.asarray(ch0["ch0_min"], dtype=np.float64)[:n],
        "ch0_ped_mean": np.asarray(ch0["ch0ped_mean"], dtype=np.float64)[:n],
        "max_ch1": np.asarray(ch1["max_ch1"], dtype=np.float64)[:n],
        "ch1_min": np.asarray(ch1["ch1_min"], dtype=np.float64)[:n],
        "ch1_ped_mean": np.asarray(ch1["ch1ped_mean"], dtype=np.float64)[:n],
        "ch2_n_fit_points": np.asarray(ch2["n_fit_points"], dtype=np.int32)[:n],
        "ch2_tanh_p0": np.asarray(ch2["tanh_p0"], dtype=np.float64)[:n],
        "ch3_n_fit_points": np.asarray(ch3["n_fit_points"], dtype=np.int32)[:n],
        "ch3_tanh_p0": np.asarray(ch3["tanh_p0"], dtype=np.float64)[:n],
        "ch3_tanh_p1": np.asarray(ch3["tanh_p1"], dtype=np.float64)[:n],
        "ch3ped_mean": np.asarray(ch3["ch3ped_mean"], dtype=np.float64)[:n],
        "min_ch3": np.asarray(ch3["min_ch3"], dtype=np.float64)[:n],
        "max_ch4": np.asarray(ch4["max_ch4"], dtype=np.float64)[:n],
        "tmax_ch4": np.asarray(ch4["tmax_ch4"], dtype=np.float64)[:n],
        "max_ch5": np.asarray(ch5["max_ch5"], dtype=np.float64)[:n],
        "time_mpl": np.asarray(time_mpl, dtype=np.float64)[:n],
    }
    return arrays


def _run_one(task: Dict[str, object]) -> Dict[str, object]:
    run_name = str(task["run_name"])
    files = {key: str(value) for key, value in dict(task["files"]).items()}
    project_root = str(task["project_root"])
    pn_fit_ch0_min = float(task["pn_fit_ch0_min"])
    pn_fit_ch0_max = float(task["pn_fit_ch0_max"])
    pn_sigma = float(task["pn_sigma"])
    pn_min_fit_events = int(task["pn_min_fit_events"])
    ped_n_sigma = float(task["ped_n_sigma"])
    ped_min_fit_events = int(task["ped_min_fit_events"])
    mincut_n_sigma = float(task["mincut_n_sigma"])
    mincut_min_fit_events = int(task["mincut_min_fit_events"])
    time_rate_threshold = float(task["time_rate_threshold"])
    ch3ped_sigma_yx = float(task["ch3ped_sigma_yx"])
    ch3ped_x_band_half_sigma = float(task["ch3ped_x_band_half_sigma"])
    ch3ped_residual_n_sigma = float(task["ch3ped_residual_n_sigma"])
    bscut_rise_time_max_us = float(task["bscut_rise_time_max_us"])

    d = _load_run_arrays(run_name, files, project_root=project_root)

    max_ch0 = d["max_ch0"]
    ch0_min = d["ch0_min"]
    ch0_ped_mean = d["ch0_ped_mean"]
    max_ch1 = d["max_ch1"]
    ch1_min = d["ch1_min"]
    ch1_ped_mean = d["ch1_ped_mean"]
    ch2_n_fit_points = d["ch2_n_fit_points"]
    ch2_tanh_p0 = d["ch2_tanh_p0"]
    ch3_n_fit_points = d["ch3_n_fit_points"]
    ch3_tanh_p0 = d["ch3_tanh_p0"]
    ch3_tanh_p1 = d["ch3_tanh_p1"]
    ch3ped_mean = d["ch3ped_mean"]
    min_ch3 = d["min_ch3"]
    max_ch4 = d["max_ch4"]
    tmax_ch4 = d["tmax_ch4"]
    max_ch5 = d["max_ch5"]
    time_mpl = d["time_mpl"]

    m_fit = np.asarray(
        fit_success_mask(ch2_n_fit_points, ch3_n_fit_points, ch2_tanh_p0, ch3_tanh_p0),
        dtype=bool,
    )
    m_rt = np.asarray(rt_mask(max_ch5), dtype=bool)
    m_rt_keep = ~m_rt
    m_inhibit_keep = ~np.asarray(inhibit_mask(ch0_min), dtype=bool)
    m_sat_keep = np.asarray(saturation_mask(max_ch0, max_ch1), dtype=bool)
    m_acv = np.asarray(acv_mask(max_ch4, tmax_ch4), dtype=bool)
    m_ped = np.asarray(
        pedestal_3sigma_mask(
            ch0_ped_mean,
            ch1_ped_mean,
            m_rt,
            n_sigma=ped_n_sigma,
            min_fit_events=ped_min_fit_events,
        ),
        dtype=bool,
    )
    m_mincut = np.asarray(
        mincut_mask(
            ch0_min,
            ch1_min,
            fit_mask=~m_acv,
            n_sigma=mincut_n_sigma,
            min_fit_events=mincut_min_fit_events,
        ),
        dtype=bool,
    )
    m_ch3ped_min = np.asarray(
        ch3ped_min_mask(
            ch3ped_mean,
            min_ch3,
            sigma_yx=ch3ped_sigma_yx,
            x_mean_band_half_sigma=ch3ped_x_band_half_sigma,
            n_sigma_residual=ch3ped_residual_n_sigma,
        ),
        dtype=bool,
    )
    m_bscut = np.asarray(bscut_mask(ch3_tanh_p1, rise_time_max_us=bscut_rise_time_max_us), dtype=bool)

    pre_m6 = m_fit & m_inhibit_keep & m_sat_keep & m_rt_keep & m_ped & m_acv & m_mincut
    pn_pre = np.asarray(
        pncut_mask(
            max_ch0,
            max_ch1,
            base_mask=pre_m6,
            fit_ch0_min=pn_fit_ch0_min,
            fit_ch0_max=pn_fit_ch0_max,
            n_sigma=pn_sigma,
            min_fit_events=pn_min_fit_events,
        ),
        dtype=bool,
    )
    time_pre_mask = pre_m6 & pn_pre & m_ch3ped_min
    m_time, intervals = cut_time(
        time_mpl,
        bad_intervals=None,
        max_ch0=max_ch0,
        pre_mask=time_pre_mask,
        rate_threshold=time_rate_threshold,
        return_intervals=True,
    )

    stages_before_pn = [
        ("basic_mincut", m_mincut),
        ("basic_pedestal_3sigma", m_ped),
        ("basic_time_keep", m_time),
        ("basic_fit_success", m_fit),
        ("event_ch3ped_min", m_ch3ped_min),
        ("event_acv_keep", m_acv),
        ("event_rt_keep", m_rt_keep),
        ("event_inhibit_keep", m_inhibit_keep),
        ("event_saturation_keep", m_sat_keep),
    ]
    pre_pn_flow = evaluate_cut_flow(stages_before_pn)
    m_pn = np.asarray(
        pncut_mask(
            max_ch0,
            max_ch1,
            base_mask=pre_pn_flow.final_mask,
            fit_ch0_min=pn_fit_ch0_min,
            fit_ch0_max=pn_fit_ch0_max,
            n_sigma=pn_sigma,
            min_fit_events=pn_min_fit_events,
        ),
        dtype=bool,
    )
    all_stages = stages_before_pn + [("pncut", m_pn), ("event_bscut", m_bscut)]
    flow = evaluate_cut_flow(all_stages)
    masks_by_stage = {name: np.asarray(mask, dtype=bool)[: flow.total] for name, mask in all_stages}

    stage_rows = []
    for stage in flow.stages:
        single_passed = _count(masks_by_stage[stage.name])
        stage_rows.append(
            {
                "run": run_name,
                "group": STAGE_GROUPS.get(stage.name, ""),
                "stage": stage.name,
                "total": flow.total,
                "single_passed": single_passed,
                "single_failed": int(flow.total - single_passed),
                "single_fraction": float(single_passed / flow.total) if flow.total else 0.0,
                "removed_by_step": stage.removed,
                "cumulative_passed": stage.cumulative_passed,
                "cumulative_fraction": stage.cumulative_fraction,
            }
        )
    cumulative_by_stage = {row["stage"]: int(row["cumulative_passed"]) for row in stage_rows}

    detail_counts = {
        "basic_fit_success": _count(m_fit),
        "basic_mincut": _count(m_mincut),
        "basic_pedestal_3sigma": _count(m_ped),
        "basic_time_keep": _count(m_time),
        "event_act_events": int(m_acv.size - _count(m_acv)),
        "event_acv_keep": _count(m_acv),
        "event_ch3ped_min": _count(m_ch3ped_min),
        "event_inhibit_events": int(m_inhibit_keep.size - _count(m_inhibit_keep)),
        "event_inhibit_keep": _count(m_inhibit_keep),
        "event_rt_events": _count(m_rt),
        "event_rt_keep": _count(m_rt_keep),
        "event_saturated_events": int(m_sat_keep.size - _count(m_sat_keep)),
        "event_saturation_keep": _count(m_sat_keep),
        "pncut_pre_time": _count(pn_pre),
        "event_bscut": _count(m_bscut),
    }
    group_counts = {
        group: int(cumulative_by_stage[stage])
        for group, stage in GROUP_END_STAGE.items()
        if stage in cumulative_by_stage
    }

    return {
        "run": run_name,
        "total": int(flow.total),
        "detail_counts": detail_counts,
        "group_counts": group_counts,
        "stage_rows": stage_rows,
        "time_removed_days": float(cut_time.bad_intervals_total_days(intervals)),
        "time_intervals": int(len(intervals)),
    }


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Legacy cut-flow report: mincut -> pedestal_3sigma -> time -> fit_success -> "
            "event-shape -> ACV -> RT -> inhibit -> saturation -> pncut -> bscut."
        )
    )
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--output-dir", default=None, help="Default: data/cache/cut_flow")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--per-run", action="store_true", help="Print a per-run summary line.")
    parser.add_argument("--pn-fit-ch0-min", type=float, default=3000.0)
    parser.add_argument("--pn-fit-ch0-max", type=float, default=12000.0)
    parser.add_argument("--pn-sigma", type=float, default=0.3)
    parser.add_argument("--pn-min-fit-events", type=int, default=10)
    parser.add_argument("--ped-n-sigma", type=float, default=3.0)
    parser.add_argument("--ped-min-fit-events", type=int, default=10)
    parser.add_argument("--mincut-n-sigma", type=float, default=3.0)
    parser.add_argument("--mincut-min-fit-events", type=int, default=10)
    parser.add_argument("--time-rate-threshold", type=float, default=0.5)
    parser.add_argument("--ch3ped-sigma-yx", type=float, default=20.0)
    parser.add_argument("--ch3ped-x-band-half-sigma", type=float, default=0.5)
    parser.add_argument("--ch3ped-residual-n-sigma", type=float, default=6.0)
    parser.add_argument("--bscut-rise-time-max-us", type=float, default=0.8)
    add_parallel_arguments(parser)
    return parser.parse_args(argv)


def _aggregate_stage_rows(rows: list[dict[str, object]], total: int) -> list[dict[str, object]]:
    by_stage: Dict[str, Dict[str, object]] = {}
    for row in rows:
        stage = str(row["stage"])
        item = by_stage.setdefault(
            stage,
            {
                "group": row.get("group", ""),
                "stage": stage,
                "total": 0,
                "single_passed": 0,
                "single_failed": 0,
                "removed_by_step": 0,
                "cumulative_passed": 0,
            },
        )
        item["total"] = int(item["total"]) + int(row["total"])
        item["single_passed"] = int(item["single_passed"]) + int(row["single_passed"])
        item["single_failed"] = int(item["single_failed"]) + int(row["single_failed"])
        item["removed_by_step"] = int(item["removed_by_step"]) + int(row["removed_by_step"])
        item["cumulative_passed"] = int(item["cumulative_passed"]) + int(row["cumulative_passed"])

    ordered: list[dict[str, object]] = []
    for stage in STAGE_GROUPS:
        if stage not in by_stage:
            continue
        row = by_stage[stage]
        single_passed = int(row["single_passed"])
        cumulative_passed = int(row["cumulative_passed"])
        row["single_fraction"] = float(single_passed / total) if total else 0.0
        row["cumulative_fraction"] = float(cumulative_passed / total) if total else 0.0
        ordered.append(row)
    return ordered


def _aggregate_group_rows(group_summary: Dict[str, int], total: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    previous = total
    for group in ("basic_cut", "event_cut", "pncut", "bscut"):
        passed = int(group_summary.get(group, 0))
        rows.append(
            {
                "group": group,
                "end_stage": GROUP_END_STAGE[group],
                "total": total,
                "cumulative_passed": passed,
                "removed_since_previous_group": max(0, previous - passed),
                "cumulative_fraction": float(passed / total) if total else 0.0,
            }
        )
        previous = passed
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary(
    output_dir: Path,
    *,
    total: int,
    stage_rows: list[dict[str, object]],
    detail_counts: Dict[str, int],
    title: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_names = [row["stage"] for row in stage_rows]
    stage_passed = [int(row["cumulative_passed"]) for row in stage_rows]

    detail_order = [
        "basic_mincut",
        "basic_pedestal_3sigma",
        "basic_time_keep",
        "basic_fit_success",
        "event_ch3ped_min",
        "event_acv_keep",
        "event_rt_keep",
        "event_inhibit_keep",
        "event_saturation_keep",
        "pncut_pre_time",
        "event_bscut",
    ]
    detail_names = [name for name in detail_order if name in detail_counts]
    detail_values = [int(detail_counts[name]) for name in detail_names]

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True)
    ax0, ax1 = axes
    ax0.bar(stage_names, stage_passed, color="C0")
    ax0.set_title(title)
    ax0.set_ylabel("Cumulative survivors")
    ax0.set_ylim(0, max(total, max(stage_passed) if stage_passed else 0) * 1.05 + 1)
    ax0.tick_params(axis="x", rotation=25)
    for idx, value in enumerate(stage_passed):
        ax0.text(idx, value, f"{value}", ha="center", va="bottom", fontsize=9)

    ax1.bar(detail_names, detail_values, color="C4")
    ax1.set_ylabel("Single-cut survivors")
    ax1.set_ylim(0, max(total, max(detail_values) if detail_values else 0) * 1.05 + 1)
    ax1.tick_params(axis="x", rotation=25)
    for idx, value in enumerate(detail_values):
        ax1.text(idx, value, f"{value}", ha="center", va="bottom", fontsize=8)

    path = output_dir / "cut_flow_summary.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main(argv=None) -> None:
    args = _parse_args(argv)
    root = find_project_root(args.project_root)
    out_dir = Path(args.output_dir) if args.output_dir else root / "data" / "cache" / "cut_flow"
    out_dir.mkdir(parents=True, exist_ok=True)

    channels = ("CH0", "CH1", "CH2", "CH3", "CH4", "CH5")
    runs = pair_parameter_files(channels, project_root=root)
    if args.max_runs is not None:
        runs = runs[: int(args.max_runs)]
    if not runs:
        raise FileNotFoundError("No paired parameter files found.")

    tasks = [
        {
            "run_name": run.name,
            "files": {key: str(path) for key, path in run.files.items()},
            "project_root": str(root),
            "pn_fit_ch0_min": float(args.pn_fit_ch0_min),
            "pn_fit_ch0_max": float(args.pn_fit_ch0_max),
            "pn_sigma": float(args.pn_sigma),
            "pn_min_fit_events": int(args.pn_min_fit_events),
            "ped_n_sigma": float(args.ped_n_sigma),
            "ped_min_fit_events": int(args.ped_min_fit_events),
            "mincut_n_sigma": float(args.mincut_n_sigma),
            "mincut_min_fit_events": int(args.mincut_min_fit_events),
            "time_rate_threshold": float(args.time_rate_threshold),
            "ch3ped_sigma_yx": float(args.ch3ped_sigma_yx),
            "ch3ped_x_band_half_sigma": float(args.ch3ped_x_band_half_sigma),
            "ch3ped_residual_n_sigma": float(args.ch3ped_residual_n_sigma),
            "bscut_rise_time_max_us": float(args.bscut_rise_time_max_us),
        }
        for run in runs
    ]

    aggregated_detail: Dict[str, int] = defaultdict(int)
    aggregated_group: Dict[str, int] = defaultdict(int)
    all_stage_rows: list[dict[str, object]] = []
    total_events = 0
    total_time_removed_days = 0.0
    run_count = 0
    for _, result in iter_completed(_run_one, tasks, workers=args.workers):
        run_count += 1
        total_events += int(result["total"])
        total_time_removed_days += float(result["time_removed_days"])
        for key, value in dict(result["detail_counts"]).items():
            aggregated_detail[key] += int(value)
        for key, value in dict(result["group_counts"]).items():
            aggregated_group[key] += int(value)
        all_stage_rows.extend(list(result["stage_rows"]))

        if args.per_run:
            print(
                f"{result['run']}: basic={int(result['group_counts']['basic_cut'])}/"
                f"{int(result['total'])}, event={int(result['group_counts']['event_cut'])}/"
                f"{int(result['total'])}, pn={int(result['group_counts']['pncut'])}/"
                f"{int(result['total'])}, bscut={int(result['group_counts']['bscut'])}/"
                f"{int(result['total'])}"
            )

    stage_rows = _aggregate_stage_rows(all_stage_rows, total_events)
    group_rows = _aggregate_group_rows(dict(aggregated_group), total_events)
    detail_rows = [
        {"stage": key, "passed": value, "fraction": float(value / total_events) if total_events else 0.0}
        for key, value in sorted(aggregated_detail.items())
    ]

    print(f"runs: {run_count}")
    print(f"events: {total_events}")
    print(f"time_removed_days: {total_time_removed_days:.6f}")
    print("\nGrouped flow:")
    for row in group_rows:
        print(
            f"{row['group']}: {int(row['cumulative_passed'])}/{total_events} "
            f"({float(row['cumulative_fraction']) * 100:.2f}%), "
            f"removed_since_previous_group={int(row['removed_since_previous_group'])}"
        )

    print("\nCumulative cut flow:")
    for row in stage_rows:
        print(
            f"{row['stage']}: single={int(row['single_passed'])}/{total_events} "
            f"({float(row['single_fraction']) * 100:.2f}%), "
            f"cumulative={int(row['cumulative_passed'])}/{total_events} "
            f"({float(row['cumulative_fraction']) * 100:.2f}%), "
            f"removed_by_step={int(row['removed_by_step'])}"
        )

    print("\nAtomic masks:")
    for row in detail_rows:
        print(f"{row['stage']}: {int(row['passed'])}/{total_events} ({float(row['fraction']) * 100:.2f}%)")

    _write_csv(out_dir / "cut_flow_grouped.csv", group_rows)
    _write_csv(out_dir / "cut_flow_cumulative.csv", stage_rows)
    _write_csv(out_dir / "cut_flow_by_run.csv", all_stage_rows)
    _write_csv(out_dir / "cut_flow_atomic.csv", detail_rows)
    plot_path = _plot_summary(
        out_dir,
        total=total_events,
        stage_rows=stage_rows,
        detail_counts=dict(aggregated_detail),
        title="Legacy cut flow summary",
    )
    print(f"\nWrote {plot_path}")


if __name__ == "__main__":
    main()
