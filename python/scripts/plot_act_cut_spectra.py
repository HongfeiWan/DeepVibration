#!/usr/bin/env python
"""Plot ACT cumulative-cut spectra with the legacy keV/rate definition."""

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
    ch3ped_min_mask,
    cut_time,
    fit_success_mask,
    inhibit_mask,
    mincut_mask,
    pedestal_3sigma_mask,
    rt_mask,
    saturation_mask,
)
from analysis.features import (  # noqa: E402
    EXPOSURE_DAYS,
    EXPOSURE_KG,
    SPECTRUM_E_MAX_KEV,
    SPECTRUM_E_MIN_KEV,
    SPECTRUM_N_BINS_ACT,
    ch0_energy_kev,
    spectrum_rate_from_counts,
)
from analysis.io import find_project_root, pair_parameter_files, read_raw_pulse_event_time_mpl  # noqa: E402
from analysis.io.parameters import load_parameter_file  # noqa: E402
from analysis.parallel import add_parallel_arguments, iter_completed  # noqa: E402
from analysis.pipelines import evaluate_cut_flow  # noqa: E402

STAGE_ORDER = (
    "act",
    "basic_mincut",
    "basic_pedestal_3sigma",
    "basic_time_keep",
    "event_ch3ped_min",
    "event_rt_keep",
    "event_inhibit_keep",
    "event_saturation_keep",
)

PLOT_STAGE_ORDER = STAGE_ORDER

STAGE_LABELS = {
    "act": "ACT",
    "basic_mincut": "ACT+mincut",
    "basic_pedestal_3sigma": "above+pedestal_3sigma",
    "basic_time_keep": "above+time_keep",
    "event_ch3ped_min": "above+ch3ped_min",
    "event_rt_keep": "above+RT_keep",
    "event_inhibit_keep": "above+inhibit_keep",
    "event_saturation_keep": "above+saturation_keep",
}


def _count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(mask, dtype=bool)))


def _common_length(arrays: Iterable[np.ndarray]) -> int:
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

    n = _common_length(
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

    return {
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


def _run_one(task: Dict[str, object]) -> Dict[str, object]:
    run_name = str(task["run_name"])
    files = {key: str(value) for key, value in dict(task["files"]).items()}
    project_root = str(task["project_root"])
    bin_edges = np.asarray(task["bin_edges_kev"], dtype=np.float64)

    d = _load_run_arrays(run_name, files, project_root=project_root)
    max_ch0 = d["max_ch0"]
    max_ch1 = d["max_ch1"]
    ch0_min = d["ch0_min"]
    ch1_min = d["ch1_min"]
    max_ch5 = d["max_ch5"]
    max_ch4 = d["max_ch4"]
    tmax_ch4 = d["tmax_ch4"]

    m_acv = np.asarray(acv_mask(max_ch4, tmax_ch4), dtype=bool)
    m_act = ~m_acv
    m_fit = np.asarray(
        fit_success_mask(
            d["ch2_n_fit_points"],
            d["ch3_n_fit_points"],
            d["ch2_tanh_p0"],
            d["ch3_tanh_p0"],
        ),
        dtype=bool,
    )
    m_rt = np.asarray(rt_mask(max_ch5), dtype=bool)
    m_rt_keep = ~m_rt
    m_inhibit_keep = ~np.asarray(inhibit_mask(ch0_min), dtype=bool)
    m_sat_keep = np.asarray(saturation_mask(max_ch0, max_ch1), dtype=bool)
    m_ped = np.asarray(
        pedestal_3sigma_mask(
            d["ch0_ped_mean"],
            d["ch1_ped_mean"],
            m_rt,
            n_sigma=float(task["ped_n_sigma"]),
            min_fit_events=int(task["ped_min_fit_events"]),
        ),
        dtype=bool,
    )
    m_mincut = np.asarray(
        mincut_mask(
            ch0_min,
            ch1_min,
            fit_mask=m_act,
            n_sigma=float(task["mincut_n_sigma"]),
            min_fit_events=int(task["mincut_min_fit_events"]),
        ),
        dtype=bool,
    )
    m_ch3ped_min = np.asarray(
        ch3ped_min_mask(
            d["ch3ped_mean"],
            d["min_ch3"],
            sigma_yx=float(task["ch3ped_sigma_yx"]),
            x_mean_band_half_sigma=float(task["ch3ped_x_band_half_sigma"]),
            n_sigma_residual=float(task["ch3ped_residual_n_sigma"]),
        ),
        dtype=bool,
    )
    time_pre_mask = (
        m_act
        & m_fit
        & m_inhibit_keep
        & m_sat_keep
        & m_rt_keep
        & m_ped
        & m_mincut
        & m_ch3ped_min
    )
    m_time, intervals = cut_time(
        d["time_mpl"],
        bad_intervals=None,
        max_ch0=max_ch0,
        pre_mask=time_pre_mask,
        rate_threshold=float(task["time_rate_threshold"]),
        return_intervals=True,
    )

    stages_before_pn = [
        ("act", m_act),
        ("basic_mincut", m_mincut),
        ("basic_pedestal_3sigma", m_ped),
        ("basic_time_keep", m_time),
        ("event_ch3ped_min", m_ch3ped_min),
        ("event_rt_keep", m_rt_keep),
        ("event_inhibit_keep", m_inhibit_keep),
        ("event_saturation_keep", m_sat_keep),
    ]
    pre_pn_flow = evaluate_cut_flow(stages_before_pn)

    stage_masks = stages_before_pn

    energy = ch0_energy_kev(max_ch0[: pre_pn_flow.total])
    cumulative = np.ones(pre_pn_flow.total, dtype=bool)
    counts_by_stage: Dict[str, np.ndarray] = {}
    events_by_stage: Dict[str, int] = {}
    for name, mask in stage_masks:
        m = np.asarray(mask, dtype=bool).reshape(-1)[: pre_pn_flow.total]
        cumulative &= m
        counts_by_stage[name] = np.histogram(energy[cumulative], bins=bin_edges)[0].astype(np.int64)
        events_by_stage[name] = _count(cumulative)

    return {
        "run": run_name,
        "total": int(pre_pn_flow.total),
        "counts_by_stage": counts_by_stage,
        "events_by_stage": events_by_stage,
        "time_removed_days": float(cut_time.bad_intervals_total_days(intervals)),
        "time_intervals": int(len(intervals)),
    }


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ACT and cumulative-cut spectra using the legacy keV/cpkkd definition."
    )
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--output-dir", default=None, help="Default: data/cache/act_cut_spectrum")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--bins", type=int, default=SPECTRUM_N_BINS_ACT)
    parser.add_argument("--e-min", type=float, default=SPECTRUM_E_MIN_KEV)
    parser.add_argument("--e-max", type=float, default=SPECTRUM_E_MAX_KEV)
    parser.add_argument("--exposure-kg", type=float, default=EXPOSURE_KG)
    parser.add_argument("--exposure-days", type=float, default=EXPOSURE_DAYS)
    parser.add_argument("--ped-n-sigma", type=float, default=3.0)
    parser.add_argument("--ped-min-fit-events", type=int, default=10)
    parser.add_argument("--mincut-n-sigma", type=float, default=3.0)
    parser.add_argument("--mincut-min-fit-events", type=int, default=10)
    parser.add_argument("--time-rate-threshold", type=float, default=0.5)
    parser.add_argument("--ch3ped-sigma-yx", type=float, default=20.0)
    parser.add_argument("--ch3ped-x-band-half-sigma", type=float, default=0.5)
    parser.add_argument("--ch3ped-residual-n-sigma", type=float, default=6.0)
    parser.add_argument("--linear", action="store_true", help="Use a linear y-axis instead of log.")
    add_parallel_arguments(parser)
    return parser.parse_args(argv)


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_spectrum_csv(
    path: Path,
    *,
    bin_edges: np.ndarray,
    counts_by_stage: Dict[str, np.ndarray],
    rates_by_stage: Dict[str, np.ndarray],
    exposure_by_stage: Dict[str, float],
) -> None:
    rows: list[dict[str, object]] = []
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    for stage in STAGE_ORDER:
        counts = counts_by_stage[stage]
        rates = rates_by_stage[stage]
        for idx in range(counts.size):
            rows.append(
                {
                    "stage": stage,
                    "label": STAGE_LABELS[stage],
                    "bin_left_kev": float(bin_edges[idx]),
                    "bin_right_kev": float(bin_edges[idx + 1]),
                    "bin_center_kev": float(centers[idx]),
                    "counts": int(counts[idx]),
                    "rate_cpkkd": float(rates[idx]),
                    "exposure_days": float(exposure_by_stage[stage]),
                }
            )
    _write_summary(path, rows)


def _plot_spectra(
    path: Path,
    *,
    bin_edges: np.ndarray,
    rates_by_stage: Dict[str, np.ndarray],
    events_by_stage: Dict[str, int],
    log_y: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.0, 6.8))
    colors = plt.get_cmap("tab20").colors
    floor = 1.0
    for idx, stage in enumerate(PLOT_STAGE_ORDER):
        rates = rates_by_stage[stage]
        rates_plot = np.maximum(rates, floor) if log_y else rates
        alpha = 0.95 if stage == PLOT_STAGE_ORDER[-1] else 0.72
        ax.stairs(
            rates_plot,
            bin_edges,
            linewidth=1.2,
            color=colors[idx % len(colors)],
            alpha=alpha,
            label=f"{STAGE_LABELS[stage]} (N={events_by_stage[stage]})",
        )
    if log_y:
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor * 0.8)
    ax.set_xlim(float(bin_edges[0]), float(bin_edges[-1]))
    ax.set_xlabel("Energy (keV)", fontsize=14)
    ax.set_ylabel(r"Rate [counts / (keV kg day)]", fontsize=14)
    ax.set_title("ACT cumulative-cut spectra", fontsize=15)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main(argv=None) -> None:
    args = _parse_args(argv)
    root = find_project_root(args.project_root)
    out_dir = Path(args.output_dir) if args.output_dir else root / "data" / "cache" / "act_cut_spectrum"
    out_dir.mkdir(parents=True, exist_ok=True)

    bin_edges = np.linspace(float(args.e_min), float(args.e_max), int(args.bins) + 1)
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
            "bin_edges_kev": bin_edges,
            "ped_n_sigma": float(args.ped_n_sigma),
            "ped_min_fit_events": int(args.ped_min_fit_events),
            "mincut_n_sigma": float(args.mincut_n_sigma),
            "mincut_min_fit_events": int(args.mincut_min_fit_events),
            "time_rate_threshold": float(args.time_rate_threshold),
            "ch3ped_sigma_yx": float(args.ch3ped_sigma_yx),
            "ch3ped_x_band_half_sigma": float(args.ch3ped_x_band_half_sigma),
            "ch3ped_residual_n_sigma": float(args.ch3ped_residual_n_sigma),
        }
        for run in runs
    ]

    counts_by_stage = {stage: np.zeros(int(args.bins), dtype=np.int64) for stage in STAGE_ORDER}
    events_by_stage: Dict[str, int] = defaultdict(int)
    total_events = 0
    total_time_removed_days = 0.0
    run_count = 0
    for _, result in iter_completed(_run_one, tasks, workers=args.workers):
        run_count += 1
        total_events += int(result["total"])
        total_time_removed_days += float(result["time_removed_days"])
        for stage, counts in dict(result["counts_by_stage"]).items():
            counts_by_stage[stage] += np.asarray(counts, dtype=np.int64)
        for stage, count in dict(result["events_by_stage"]).items():
            events_by_stage[stage] += int(count)

    reduced_exposure_days = max(1e-12, float(args.exposure_days) - total_time_removed_days)
    exposure_by_stage: Dict[str, float] = {}
    use_reduced = False
    for stage in STAGE_ORDER:
        if stage == "basic_time_keep":
            use_reduced = True
        exposure_by_stage[stage] = reduced_exposure_days if use_reduced else float(args.exposure_days)
    rates_by_stage = {
        stage: spectrum_rate_from_counts(
            counts,
            bin_edges,
            exposure_kg=float(args.exposure_kg),
            exposure_days=exposure_by_stage[stage],
        )
        for stage, counts in counts_by_stage.items()
    }

    summary_rows = [
        {
            "stage": stage,
            "label": STAGE_LABELS[stage],
            "events": int(events_by_stage[stage]),
            "fraction_of_total": float(events_by_stage[stage] / total_events) if total_events else 0.0,
            "exposure_days": float(exposure_by_stage[stage]),
        }
        for stage in STAGE_ORDER
    ]
    _write_summary(out_dir / "act_cut_spectrum_summary.csv", summary_rows)
    _write_spectrum_csv(
        out_dir / "act_cut_spectrum.csv",
        bin_edges=bin_edges,
        counts_by_stage=counts_by_stage,
        rates_by_stage=rates_by_stage,
        exposure_by_stage=exposure_by_stage,
    )
    _plot_spectra(
        out_dir / "act_cut_spectrum.png",
        bin_edges=bin_edges,
        rates_by_stage=rates_by_stage,
        events_by_stage=dict(events_by_stage),
        log_y=not bool(args.linear),
    )

    print(f"runs: {run_count}")
    print(f"events: {total_events}")
    print(f"time_removed_days: {total_time_removed_days:.6f}")
    print(f"exposure_days_after_time: {reduced_exposure_days:.6f}")
    for row in summary_rows:
        print(
            f"{row['stage']}: {int(row['events'])}/{total_events} "
            f"({float(row['fraction_of_total']) * 100:.2f}%), exposure_days={float(row['exposure_days']):.6f}"
        )
    print(f"Wrote {out_dir / 'act_cut_spectrum.png'}")
    print(f"Wrote {out_dir / 'act_cut_spectrum.csv'}")


if __name__ == "__main__":
    main()
