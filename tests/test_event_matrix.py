from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from analysis.ml.event_matrix import build_feature_cache  # noqa: E402


class EventMatrixTests(unittest.TestCase):
    def test_build_feature_cache_from_paired_parameter_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("test", encoding="utf-8")
            (root / "python").mkdir()
            raw = root / "data" / "hdf5" / "raw_pulse"
            for folder in ["CH0_parameters", "CH1_parameters", "CH4_parameters", "CH5_parameters", "CH0-3"]:
                (raw / folder).mkdir(parents=True, exist_ok=True)
            name = "run_001_processed.h5"
            with h5py.File(raw / "CH0_parameters" / name, "w") as h:
                h.create_dataset("max_ch0", data=np.array([1000, 7000, 9000], dtype=np.float32))
                h.create_dataset("ch0_min", data=np.array([1, 0, 2], dtype=np.float32))
            with h5py.File(raw / "CH1_parameters" / name, "w") as h:
                h.create_dataset("max_ch1", data=np.array([1100, 7100, 9100], dtype=np.float32))
            with h5py.File(raw / "CH4_parameters" / name, "w") as h:
                h.create_dataset("max_ch4", data=np.array([1, 1, 8000], dtype=np.float32))
                h.create_dataset("tmax_ch4", data=np.array([0, 0, 9000], dtype=np.uint32))
            with h5py.File(raw / "CH5_parameters" / name, "w") as h:
                h.create_dataset("max_ch5", data=np.array([1, 1, 7000], dtype=np.float32))
            with h5py.File(raw / "CH0-3" / name, "w") as h:
                h.create_dataset("time_data", data=np.array([10, 11, 12], dtype=np.float64))

            cache = build_feature_cache(
                project_root=root,
                cache_dir=root / "data" / "cache" / "event_feature_umap",
                workers=1,
                rebuild=True,
            )
            with h5py.File(cache, "r") as h:
                self.assertEqual(h["features"].shape, (3, 6))
                self.assertEqual(int(np.count_nonzero(h["masks/rt"][...])), 1)
                self.assertEqual(int(np.count_nonzero(h["masks/inhibit"][...])), 1)
                self.assertTrue(np.array_equal(h["event_time"][...], [10, 11, 12]))


if __name__ == "__main__":
    unittest.main()
