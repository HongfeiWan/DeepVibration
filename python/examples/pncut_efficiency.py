#!/usr/bin/env python
"""Example: compute a PN-cut mask and simple efficiency."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from analysis.cuts import pncut_mask
from analysis.io.parameters import load_parameter_file
import numpy as np


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="PN-cut efficiency example.")
    parser.add_argument("--ch0", required=True, help="CH0_parameters HDF5 file.")
    parser.add_argument("--ch1", required=True, help="CH1_parameters HDF5 file.")
    parser.add_argument("--n-sigma", type=float, default=0.8)
    args = parser.parse_args(argv)

    ch0 = load_parameter_file(args.ch0, ["max_ch0"])
    ch1 = load_parameter_file(args.ch1, ["max_ch1"])
    mask, stats = pncut_mask(
        ch0["max_ch0"],
        ch1["max_ch1"],
        n_sigma=args.n_sigma,
        return_stats=True,
    )
    print(f"pncut: {np.count_nonzero(mask)}/{mask.size}")
    print(stats)


if __name__ == "__main__":
    main()
