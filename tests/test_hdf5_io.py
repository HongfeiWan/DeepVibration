from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from analysis.io import compute_channel_stat, read_waveform  # noqa: E402
from analysis.parallel import process_map  # noqa: E402


def _sum_hdf5_dataset(path: str) -> int:
    with h5py.File(path, "r") as handle:
        return int(np.sum(handle["x"][...]))


class Hdf5IoTests(unittest.TestCase):
    def test_compute_channel_stat_and_waveform(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.h5"
            data = np.arange(4 * 2 * 3, dtype=np.uint16).reshape(4, 2, 3)
            with h5py.File(path, "w") as handle:
                handle.create_dataset("channel_data", data=data)
                handle.create_dataset("time_data", data=np.arange(3, dtype=np.float64))

            self.assertTrue(np.array_equal(compute_channel_stat(path, channel_idx=0, stat="max"), [18, 19, 20]))
            self.assertTrue(np.array_equal(compute_channel_stat(path, channel_idx=1, stat="min"), [3, 4, 5]))
            self.assertTrue(np.array_equal(read_waveform(path, event_idx=2, channel_idx=1), data[:, 1, 2]))

    def test_worker_opens_hdf5_file_itself(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i in range(2):
                path = Path(tmp) / f"sample_{i}.h5"
                with h5py.File(path, "w") as handle:
                    handle.create_dataset("x", data=np.array([i, i + 1], dtype=np.int32))
                paths.append(str(path))
            self.assertEqual(process_map(_sum_hdf5_dataset, paths, workers=2), [1, 3])


if __name__ == "__main__":
    unittest.main()
