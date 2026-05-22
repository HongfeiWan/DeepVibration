#!/usr/bin/env python
"""Example: select random-trigger events from a CH5 raw-pulse file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from analysis.cuts import rt_mask
from analysis.io import compute_channel_stat
import numpy as np


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="RT example: max(CH5) > threshold.")
    parser.add_argument("ch5_file")
    parser.add_argument("--threshold", type=float, default=6000.0)
    parser.add_argument("--chunk-size", type=int, default=1000)
    args = parser.parse_args(argv)

    max_ch5 = compute_channel_stat(args.ch5_file, channel_idx=0, stat="max", chunk_size=args.chunk_size)
    mask = rt_mask(max_ch5, threshold=args.threshold)
    idx = np.flatnonzero(mask)
    print(f"RT events: {idx.size}/{mask.size}")
    if idx.size:
        print(f"first={int(idx[0])}, max_ch5={max_ch5[idx[0]]:.3f}")


if __name__ == "__main__":
    main()
