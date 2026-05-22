#!/usr/bin/env python
"""Plot a CH0-derived energy spectrum from parameter files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from analysis.parallel import configure_blas_threads  # noqa: F401
import h5py
import matplotlib.pyplot as plt
import numpy as np

from analysis.io.parameters import load_parameter_file
from analysis.plotting import plot_spectrum


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot spectrum from CH0_parameters/max_ch0.")
    parser.add_argument("ch0_parameter_file")
    parser.add_argument("--mask-file", help="Optional mask HDF5 produced by scripts/run_cuts.py.")
    parser.add_argument("--cal-a", type=float, default=1.0, help="Energy calibration slope.")
    parser.add_argument("--cal-b", type=float, default=0.0, help="Energy calibration intercept.")
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--save", help="Optional output image path.")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    data = load_parameter_file(args.ch0_parameter_file, ["max_ch0"])
    energy = args.cal_a * np.asarray(data["max_ch0"], dtype=np.float64) + args.cal_b
    mask = None
    if args.mask_file:
        with h5py.File(args.mask_file, "r") as handle:
            mask = np.asarray(handle["mask"][...], dtype=bool)
    ax = plot_spectrum(energy, mask=mask, bins=args.bins)
    ax.set_xlabel("Energy")
    ax.set_title(Path(args.ch0_parameter_file).name)
    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=150)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
