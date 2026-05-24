from __future__ import annotations

import sys
import unittest
from pathlib import Path
from datetime import datetime

import matplotlib.dates as mdates
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from analysis.cuts import (  # noqa: E402
    build_ch0_time_exclude_intervals_global,
    acv_mask,
    act_mask,
    bscut_mask,
    ch3ped_min_mask,
    fit_success_mask,
    inhibit_mask,
    mincut_mask,
    pedestal_3sigma_mask,
    pncut_mask,
    rt_mask,
    saturation_mask,
    cut_time,
    time_exclusion_mask,
)
from analysis.pipelines import evaluate_cut_flow  # noqa: E402


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

    def test_fit_success_mask(self) -> None:
        mask = fit_success_mask(
            np.array([1, 1, 0, 1]),
            np.array([1, 1, 1, 1]),
            np.array([0.1, 1e6, 0.3, np.nan]),
            np.array([0.2, 0.2, 0.2, 0.2]),
        )
        self.assertTrue(np.array_equal(mask, [True, False, False, False]))

    def test_ch3ped_min_mask(self) -> None:
        mask = ch3ped_min_mask(
            np.array([970.0, 971.0, 990.0, 970.0]),
            np.array([971.0, 972.0, 991.0, -1.0]),
            x_mean_band_half_sigma=100.0,
            n_sigma_residual=100.0,
        )
        self.assertTrue(np.array_equal(mask, [True, True, False, False]))

    def test_bscut_mask(self) -> None:
        mask = bscut_mask(np.array([10.0, 1.0, 0.0]), rise_time_max_us=0.8)
        self.assertTrue(np.array_equal(mask, [True, False, False]))

    def test_evaluate_cut_flow(self) -> None:
        result = evaluate_cut_flow(
            [
                ("first", np.array([True, True, False, True])),
                ("second", np.array([True, False, True, True])),
            ]
        )
        self.assertEqual(result.total, 4)
        self.assertTrue(np.array_equal(result.final_mask, [True, False, False, True]))
        self.assertEqual(result.stages[0].cumulative_passed, 3)
        self.assertEqual(result.stages[1].removed, 1)
        self.assertEqual(result.stages[1].cumulative_passed, 2)

    def test_time_exclusion_mask(self) -> None:
        mask = time_exclusion_mask(np.array([1.0, 2.0, 3.0, 4.0]), [(1.5, 3.5)])
        self.assertTrue(np.array_equal(mask, [True, False, False, True]))

    def test_legacy_cut_time_auto_builds_burst_interval(self) -> None:
        year = 2024
        bad_time = mdates.date2num(datetime(year, 5, 25, 12, 30))
        good_time = mdates.date2num(datetime(year, 6, 1, 12, 30))
        time_mpl = np.array([bad_time] * 31 + [good_time] * 9, dtype=np.float64)
        max_ch0 = np.full(time_mpl.size, 1300.0, dtype=np.float64)
        pre_mask = np.ones(time_mpl.size, dtype=bool)

        intervals = build_ch0_time_exclude_intervals_global(
            time_mpl,
            max_ch0,
            pre_mask=pre_mask,
            year=year,
            rate_threshold=0.5,
        )
        self.assertEqual(len(intervals), 1)
        self.assertLessEqual(intervals[0][0], bad_time)
        self.assertGreater(intervals[0][1], bad_time)

        mask, returned_intervals = cut_time(
            time_mpl,
            max_ch0=max_ch0,
            pre_mask=pre_mask,
            year=year,
            return_intervals=True,
        )
        self.assertTrue(np.array_equal(mask[:31], np.zeros(31, dtype=bool)))
        self.assertTrue(np.array_equal(mask[31:], np.ones(9, dtype=bool)))
        self.assertEqual(returned_intervals, intervals)
        self.assertGreater(cut_time.bad_intervals_total_days(intervals), 0.0)

    def test_legacy_cut_time_fallback_keeps_all_when_no_bad_intervals(self) -> None:
        mask = cut_time(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        self.assertTrue(np.array_equal(mask, [True, True, True]))


if __name__ == "__main__":
    unittest.main()
