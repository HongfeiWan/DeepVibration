#!/usr/bin/env python
"""Run reusable DeepVibration event cuts across paired parameter files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from analysis.parallel import add_parallel_arguments, iter_completed
import h5py
import numpy as np

from analysis.io import pair_parameter_files
from analysis.io.parameters import load_parameter_file
from analysis.pipelines import run_act_acv_selection, run_basic_selection


def _run_one(args: Tuple[str, Dict[str, str], str, str | None]) -> Dict[str, float | str]:
    run_name, file_map, mode, output_dir = args
    ch0 = load_parameter_file(file_map["CH0"], ["max_ch0", "ch0_min"])
    ch1 = load_parameter_file(file_map["CH1"], ["max_ch1"])
    ch5 = load_parameter_file(file_map["CH5"], ["max_ch5"])
    if mode in {"act", "acv"}:
        ch4 = load_parameter_file(file_map["CH4"], ["max_ch4", "tmax_ch4"])
        result = run_act_acv_selection(
            max_ch5=ch5["max_ch5"],
            ch0_min=ch0["ch0_min"],
            max_ch0=ch0["max_ch0"],
            max_ch1=ch1["max_ch1"],
            max_ch4=ch4["max_ch4"],
            tmax_ch4=ch4["tmax_ch4"],
            mode=mode,
        )
    else:
        result = run_basic_selection(
            max_ch5=ch5["max_ch5"],
            ch0_min=ch0["ch0_min"],
            max_ch0=ch0["max_ch0"],
            max_ch1=ch1["max_ch1"],
        )

    if output_dir:
        out_path = Path(output_dir) / run_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out_path, "w") as handle:
            handle.create_dataset("mask", data=result.mask.astype(np.uint8))
            for step_name, step_mask in result.steps.items():
                handle.create_dataset(f"steps/{step_name}", data=step_mask.astype(np.uint8))
            for key, value in result.stats.items():
                handle.attrs[key] = value
            handle.attrs["mode"] = mode

    return {
        "run": run_name,
        "total": float(result.mask.size),
        "passed": float(np.count_nonzero(result.mask)),
        "pass_fraction": float(np.count_nonzero(result.mask) / result.mask.size) if result.mask.size else 0.0,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Physical/ACT/ACV cuts from parameter HDF5 files.")
    parser.add_argument("--mode", choices=["physical", "act", "acv"], default="physical")
    parser.add_argument("--project-root", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for per-run mask HDF5 files.",
    )
    add_parallel_arguments(parser)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    channels = ("CH0", "CH1", "CH5") if args.mode == "physical" else ("CH0", "CH1", "CH4", "CH5")
    runs = pair_parameter_files(channels, project_root=args.project_root)
    if not runs:
        raise FileNotFoundError("No paired parameter files found.")

    tasks = [
        (run.name, {key: str(path) for key, path in run.files.items()}, args.mode, args.output_dir)
        for run in runs
    ]
    total_passed = 0
    total_events = 0
    for _, result in iter_completed(_run_one, tasks, workers=args.workers):
        total_passed += int(result["passed"])
        total_events += int(result["total"])
        print(
            f"{result['run']}: passed={int(result['passed'])}/{int(result['total'])} "
            f"({result['pass_fraction'] * 100:.2f}%)"
        )
    frac = total_passed / total_events if total_events else 0.0
    print(f"Combined {args.mode}: passed={total_passed}/{total_events} ({frac * 100:.2f}%)")


if __name__ == "__main__":
    main()
