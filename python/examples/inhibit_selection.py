#!/usr/bin/env python
"""Example: select inhibit events from a CH0-3 raw-pulse file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from analysis.cuts import inhibit_mask
from analysis.io import compute_channel_stat
import numpy as np


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Inhibit example: min(CH0) == 0.")
    parser.add_argument("ch0_3_file")
    parser.add_argument("--zero-value", type=float, default=0.0)
    parser.add_argument("--chunk-size", type=int, default=1000)
    args = parser.parse_args(argv)

    ch0_min = compute_channel_stat(args.ch0_3_file, channel_idx=0, stat="min", chunk_size=args.chunk_size)
    mask = inhibit_mask(ch0_min, zero_value=args.zero_value)
    idx = np.flatnonzero(mask)
    print(f"Inhibit events: {idx.size}/{mask.size}")
    if idx.size:
        print(f"first={int(idx[0])}, ch0_min={ch0_min[idx[0]]:.3f}")


if __name__ == "__main__":
    main()
