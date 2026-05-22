from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from analysis.features.filters import median_filter  # noqa: E402
from analysis.features.frequency import compute_highfreq_energy_ratio, compute_spectral_centroid_mhz  # noqa: E402
from analysis.io.writers import save_hdf5  # noqa: E402


class FeatureIoTests(unittest.TestCase):
    def test_median_filter(self) -> None:
        self.assertTrue(np.array_equal(median_filter([1, 100, 2], kernel_size=3), [1, 2, 2]))

    def test_frequency_features_are_finite(self) -> None:
        waveform = np.sin(np.linspace(0, 10, 1024))
        self.assertGreaterEqual(compute_highfreq_energy_ratio(waveform), 0.0)
        self.assertGreater(compute_spectral_centroid_mhz(waveform), 0.0)

    def test_save_hdf5(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.h5"
            save_hdf5(path, {"x": np.array([1, 2, 3]), "flag": np.array([True, False])})
            with h5py.File(path, "r") as handle:
                self.assertTrue(np.array_equal(handle["x"][...], [1, 2, 3]))
                self.assertTrue(np.array_equal(handle["flag"][...], [1, 0]))


if __name__ == "__main__":
    unittest.main()
