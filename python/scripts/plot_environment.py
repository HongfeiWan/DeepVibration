#!/usr/bin/env python
"""Plot vibration, temperature, and compressor environment overview."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Environment overview plot.")
    parser.add_argument("--vibration-data-dir", required=True, help="Directory with detector_*.h5 vibration files.")
    parser.add_argument("--compressor-file", required=True, help="Compressor text file path.")
    parser.add_argument("--detectors", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--start-time")
    parser.add_argument("--end-time")
    parser.add_argument("--downsample-factor", type=int, default=1000)
    parser.add_argument("--output")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    from analysis.environment import plot_environment_overview

    plot_environment_overview(
        vibration_data_dir=args.vibration_data_dir,
        compressor_file_path=args.compressor_file,
        detector_nums=args.detectors,
        start_date=args.start_date,
        end_date=args.end_date,
        start_time=args.start_time,
        end_time=args.end_time,
        downsample_factor=args.downsample_factor,
        save_path=args.output,
        show_plot=not args.no_show,
    )


if __name__ == "__main__":
    main()
