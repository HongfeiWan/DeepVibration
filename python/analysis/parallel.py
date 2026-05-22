"""Central parallel execution policy for DeepVibration.

The default worker policy is intentionally simple: ``auto`` means all logical
CPU cores reported by the operating system. Scripts can still pass an explicit
integer when memory pressure or interactive use needs a smaller pool.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Iterable, Iterator, List, Sequence, Tuple, TypeVar

T = TypeVar("T")
R = TypeVar("R")

THREAD_ENV_VARS: Tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_blas_threads(num_threads: int = 1, *, force: bool = False) -> None:
    """Limit per-process BLAS threads to prevent process/thread oversubscription."""

    value = str(max(1, int(num_threads)))
    for name in THREAD_ENV_VARS:
        if force or not os.environ.get(name):
            os.environ[name] = value


configure_blas_threads(1)


def cpu_count() -> int:
    """Return a usable CPU count, never less than one."""

    return max(1, os.cpu_count() or 1)


def resolve_workers(
    workers: int | str | None = "auto",
    *,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    """Resolve ``auto`` or an integer-like value into a concrete worker count."""

    if workers is None or str(workers).lower() == "auto":
        count = cpu_count()
    else:
        try:
            count = int(workers)
        except (TypeError, ValueError) as exc:
            raise ValueError("workers must be 'auto' or a positive integer") from exc
        if count <= 0:
            count = cpu_count()

    count = max(int(minimum), count)
    if maximum is not None:
        count = min(count, int(maximum))
    return max(1, count)


def add_parallel_arguments(parser: argparse.ArgumentParser, *, default: str = "auto") -> None:
    """Add the shared ``--workers`` option to a CLI parser."""

    parser.add_argument(
        "--workers",
        default=default,
        help="Number of worker processes. Use 'auto' to use all CPU cores.",
    )


def chunk_ranges(n_items: int, chunk_size: int) -> Iterator[Tuple[int, int]]:
    """Yield ``(start, stop)`` chunks for event-wise processing."""

    if n_items < 0:
        raise ValueError("n_items must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, n_items, chunk_size):
        yield start, min(start + chunk_size, n_items)


def process_map(
    func: Callable[[T], R],
    items: Sequence[T] | Iterable[T],
    *,
    workers: int | str | None = "auto",
) -> List[R]:
    """Map a function across items with a process pool and all-CPU default."""

    item_list = list(items)
    if not item_list:
        return []
    worker_count = resolve_workers(workers, maximum=len(item_list))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(func, item) for item in item_list]
        return [future.result() for future in futures]


def iter_completed(
    func: Callable[[T], R],
    items: Sequence[T] | Iterable[T],
    *,
    workers: int | str | None = "auto",
) -> Iterator[Tuple[T, R]]:
    """Yield ``(item, result)`` as process-pool tasks complete."""

    item_list = list(items)
    if not item_list:
        return
    worker_count = resolve_workers(workers, maximum=len(item_list))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_item = {executor.submit(func, item): item for item in item_list}
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            yield item, future.result()


__all__ = [
    "add_parallel_arguments",
    "chunk_ranges",
    "configure_blas_threads",
    "cpu_count",
    "iter_completed",
    "process_map",
    "resolve_workers",
]
