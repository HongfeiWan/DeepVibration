from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from analysis.parallel import (  # noqa: E402
    ParallelConfig,
    WORKER_ENV_VAR,
    chunk_ranges,
    process_map,
    resolve_workers,
)


def _square(x: int) -> int:
    return x * x


class ParallelPolicyTests(unittest.TestCase):
    def test_resolve_workers_auto_and_maximum(self) -> None:
        self.assertGreaterEqual(resolve_workers("auto"), 1)
        self.assertEqual(resolve_workers("auto", maximum=2), min(resolve_workers("auto"), 2))
        self.assertEqual(resolve_workers(0, maximum=3), min(resolve_workers("auto"), 3))

    def test_chunk_ranges(self) -> None:
        self.assertEqual(list(chunk_ranges(10, 4)), [(0, 4), (4, 8), (8, 10)])

    def test_nested_worker_defaults_to_serial(self) -> None:
        old = os.environ.get(WORKER_ENV_VAR)
        os.environ[WORKER_ENV_VAR] = "1"
        try:
            self.assertEqual(ParallelConfig(workers="auto").resolved_workers(), 1)
            self.assertGreaterEqual(ParallelConfig(workers=2, allow_nested=True).resolved_workers(), 1)
        finally:
            if old is None:
                os.environ.pop(WORKER_ENV_VAR, None)
            else:
                os.environ[WORKER_ENV_VAR] = old

    def test_process_map(self) -> None:
        self.assertEqual(process_map(_square, [1, 2, 3], workers=2), [1, 4, 9])


if __name__ == "__main__":
    unittest.main()
