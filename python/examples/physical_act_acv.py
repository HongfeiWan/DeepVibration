#!/usr/bin/env python
"""Example: run Physical plus ACT/ACV selection from parameter files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from analysis.io.parameters import load_parameter_file
from analysis.pipelines import run_act_acv_selection, run_basic_selection
import numpy as np


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Physical/ACT/ACV selection example.")
    parser.add_argument("--ch0", required=True, help="CH0_parameters HDF5 file.")
    parser.add_argument("--ch1", required=True, help="CH1_parameters HDF5 file.")
    parser.add_argument("--ch4", help="CH4_parameters HDF5 file for ACT/ACV.")
    parser.add_argument("--ch5", required=True, help="CH5_parameters HDF5 file.")
    parser.add_argument("--mode", choices=["physical", "act", "acv"], default="physical")
    args = parser.parse_args(argv)

    ch0 = load_parameter_file(args.ch0, ["max_ch0", "ch0_min"])
    ch1 = load_parameter_file(args.ch1, ["max_ch1"])
    ch5 = load_parameter_file(args.ch5, ["max_ch5"])
    if args.mode == "physical":
        result = run_basic_selection(
            max_ch5=ch5["max_ch5"],
            ch0_min=ch0["ch0_min"],
            max_ch0=ch0["max_ch0"],
            max_ch1=ch1["max_ch1"],
        )
    else:
        if not args.ch4:
            raise ValueError("--ch4 is required for ACT/ACV modes")
        ch4 = load_parameter_file(args.ch4, ["max_ch4", "tmax_ch4"])
        result = run_act_acv_selection(
            max_ch5=ch5["max_ch5"],
            ch0_min=ch0["ch0_min"],
            max_ch0=ch0["max_ch0"],
            max_ch1=ch1["max_ch1"],
            max_ch4=ch4["max_ch4"],
            tmax_ch4=ch4["tmax_ch4"],
            mode=args.mode,
        )
    print(f"{args.mode}: {np.count_nonzero(result.mask)}/{result.mask.size}")


if __name__ == "__main__":
    main()
