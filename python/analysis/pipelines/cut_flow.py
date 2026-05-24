"""Helpers for cumulative cut-flow reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CutStageSummary:
    """One step in a cumulative cut-flow report."""

    name: str
    passed: int
    removed: int
    cumulative_passed: int
    cumulative_fraction: float
    step_fraction: float


@dataclass(frozen=True)
class CutFlowResult:
    """Cumulative cut-flow outcome for one event collection."""

    total: int
    final_mask: np.ndarray
    stages: tuple[CutStageSummary, ...]


def evaluate_cut_flow(
    steps: Sequence[tuple[str, np.ndarray]] | Mapping[str, np.ndarray],
    *,
    base_mask: np.ndarray | None = None,
) -> CutFlowResult:
    """Apply masks in order and record cumulative counts after each step."""

    items = list(steps.items()) if isinstance(steps, Mapping) else list(steps)
    if not items:
        empty = np.array([], dtype=bool)
        return CutFlowResult(total=0, final_mask=empty, stages=())

    arrays = [np.asarray(mask, dtype=bool).reshape(-1) for _, mask in items]
    if base_mask is not None:
        arrays.append(np.asarray(base_mask, dtype=bool).reshape(-1))
    n = min(arr.size for arr in arrays)
    if n <= 0:
        empty = np.array([], dtype=bool)
        return CutFlowResult(total=0, final_mask=empty, stages=())

    cumulative = np.ones(n, dtype=bool)
    if base_mask is not None:
        cumulative &= np.asarray(base_mask, dtype=bool).reshape(-1)[:n]

    stages: list[CutStageSummary] = []
    for name, mask in items:
        step = np.asarray(mask, dtype=bool).reshape(-1)[:n]
        before = int(np.count_nonzero(cumulative))
        cumulative &= step
        after = int(np.count_nonzero(cumulative))
        stages.append(
            CutStageSummary(
                name=str(name),
                passed=after,
                removed=max(0, before - after),
                cumulative_passed=after,
                cumulative_fraction=float(after / n) if n else 0.0,
                step_fraction=float(np.count_nonzero(step) / n) if n else 0.0,
            )
        )

    return CutFlowResult(total=n, final_mask=cumulative.copy(), stages=tuple(stages))


__all__ = ["CutFlowResult", "CutStageSummary", "evaluate_cut_flow"]
