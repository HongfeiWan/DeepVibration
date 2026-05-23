from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from analysis.cuts import (  # noqa: E402
    acv_mask,
    act_mask,
    inhibit_mask,
    mincut_mask,
    pedestal_3sigma_mask,
    pncut_mask,
    rt_mask,
    saturation_mask,
    time_exclusion_mask,
)


class CutTests(unittest.TestCase):
    def test_basic_masks(self) -> None:
        max_ch5 = np.array([100.0, 7000.0, 100.0])
        ch0_min = np.array([10.0, 10.0, 0.0])
        self.assertTrue(np.array_equal(rt_mask(max_ch5), [False, True, False]))
        self.assertTrue(np.array_equal(inhibit_mask(ch0_min), [False, False, True]))
        self.assertTrue(np.array_equal((~rt_mask(max_ch5)) & (~inhibit_mask(ch0_min)), [True, False, False]))

    def test_saturation_mask(self) -> None:
        self.assertTrue(
            np.array_equal(
                saturation_mask(np.array([10, 17000]), np.array([20, 30])),
                [True, False],
            )
        )

    def test_acv_mask(self) -> None:
        max_ch4 = np.array([100.0, 8000.0, 8000.0])
        # t_ch4 = [0, 39.996, 20] us, delta = [40, 0.004, 20] us.
        tmax_ch4 = np.array([0.0, 9999.0, 5000.0])
        self.assertTrue(np.array_equal(acv_mask(max_ch4, tmax_ch4), [True, True, True]))
        tmax_inside = np.array([0.0, 9000.0, 5000.0])  # delta for event 1 = 4 us.
        self.assertTrue(np.array_equal(acv_mask(max_ch4, tmax_inside), [True, False, True]))
        self.assertTrue(np.array_equal(act_mask(max_ch4, tmax_inside), [False, True, False]))

    def test_pedestal_3sigma_mask(self) -> None:
        ch0_ped = np.array([10.0, 10.1, 9.9, 10.0, 20.0])
        ch1_ped = np.array([30.0, 30.1, 29.9, 30.0, 40.0])
        reference = np.array([True, True, True, True, False])
        mask = pedestal_3sigma_mask(ch0_ped, ch1_ped, reference, n_sigma=3, min_fit_events=4)
        self.assertTrue(np.array_equal(mask, [True, True, True, True, False]))

    def test_mincut_mask(self) -> None:
        ch0_min = np.array([10.0, 10.2, 9.8, 30.0])
        ch1_min = np.array([20.0, 20.1, 19.9, 60.0])
        mask = mincut_mask(
            ch0_min,
            ch1_min,
            fit_mask=np.array([True, True, True, False]),
            n_sigma=3,
            min_fit_events=3,
        )
        self.assertTrue(np.array_equal(mask, [True, True, True, False]))

    def test_pncut_mask(self) -> None:
        x = np.linspace(6000.0, 15000.0, 20)
        y = 2.0 * x + 10.0
        y[-1] += 1000.0
        mask = pncut_mask(x, y, n_sigma=3.0, min_fit_events=5)
        self.assertLess(np.count_nonzero(mask), mask.size)

    def test_time_exclusion_mask(self) -> None:
        mask = time_exclusion_mask(np.array([1.0, 2.0, 3.0, 4.0]), [(1.5, 3.5)])
        self.assertTrue(np.array_equal(mask, [True, False, False, True]))


if __name__ == "__main__":
    unittest.main()
