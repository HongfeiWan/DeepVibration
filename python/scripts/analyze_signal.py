#!/usr/bin/env python
"""Run FFT/Lomb/Hilbert/Wavelet analysis for one waveform event."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from analysis.features.signal import (  # noqa: E402
    continuous_wavelet_power,
    fft_power_spectrum,
    hilbert_envelope,
    lomb_scargle_power,
)
from analysis.io.hdf5 import read_waveform  # noqa: E402
from analysis.parallel import add_parallel_arguments  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Signal transforms for raw-pulse HDF5 waveforms.")
    parser.add_argument("h5_file", help="Raw-pulse HDF5 file containing channel_data.")
    parser.add_argument("--method", choices=("fft", "lomb", "hilbert", "wavelet"), default="fft")
    parser.add_argument("--event", type=int, default=0, help="Event index.")
    parser.add_argument("--channel", type=int, default=0, help="Channel index.")
    parser.add_argument("--sampling-interval-ns", type=float, default=4.0)
    parser.add_argument("--output", help="Output HDF5 path. Defaults to <input>_<method>.h5.")
    parser.add_argument("--wavelet", default="cmor3-3", help="PyWavelets CWT wavelet name.")
    parser.add_argument("--min-scale", type=float, default=1.0)
    parser.add_argument("--max-scale", type=float, default=128.0)
    parser.add_argument("--num-scales", type=int, default=64)
    add_parallel_arguments(parser, include_chunk_size=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    src = Path(args.h5_file)
    out = Path(args.output) if args.output else src.with_name(f"{src.stem}_{args.method}.h5")
    waveform = read_waveform(src, event_idx=args.event, channel_idx=args.channel)

    with h5py.File(out, "w") as handle:
        handle.attrs["source_file"] = str(src.resolve())
        handle.attrs["event_index"] = int(args.event)
        handle.attrs["channel_index"] = int(args.channel)
        handle.attrs["method"] = args.method
        if args.method == "fft":
            freq_hz, power = fft_power_spectrum(waveform, sampling_interval_ns=args.sampling_interval_ns)
            handle.create_dataset("frequency_hz", data=freq_hz)
            handle.create_dataset("power", data=power)
        elif args.method == "lomb":
            time_s = np.arange(waveform.size, dtype=np.float64) * args.sampling_interval_ns * 1e-9
            freq_hz = np.linspace(1.0, 0.5 / (args.sampling_interval_ns * 1e-9), 4096)
            power = lomb_scargle_power(time_s, waveform, freq_hz)
            handle.create_dataset("frequency_hz", data=freq_hz)
            handle.create_dataset("power", data=power)
        elif args.method == "hilbert":
            handle.create_dataset("envelope", data=hilbert_envelope(waveform))
        else:
            scales = np.linspace(args.min_scale, args.max_scale, args.num_scales)
            freq_hz, power = continuous_wavelet_power(
                waveform,
                scales=scales,
                wavelet=args.wavelet,
                sampling_interval_ns=args.sampling_interval_ns,
            )
            handle.create_dataset("scale", data=scales)
            handle.create_dataset("frequency_hz", data=freq_hz)
            handle.create_dataset("power", data=power)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
