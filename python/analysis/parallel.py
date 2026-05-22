"""Central parallel execution policy for DeepVibration.

``--workers auto`` means all logical CPU cores by default.  The helpers in this
module also prevent nested process pools and cap BLAS-style thread pools to one
thread per worker, so file-level or chunk-level multiprocessing can actually use
the machine without fighting NumPy/OpenBLAS/MKL underneath.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
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
WORKER_ENV_VAR = "DEEPVIBRATION_PARALLEL_WORKER"


@dataclass(frozen=True)
class ParallelConfig:
    """Resolved process-pool settings shared by scripts and pipelines."""

    workers: int | str | None = "auto"
    chunk_size: int = 1000
    blas_threads: int = 1
    allow_nested: bool = False

    def resolved_workers(self, *, maximum: int | None = None) -> int:
        if in_parallel_worker() and not self.allow_nested:
            return 1
        return resolve_workers(self.workers, maximum=maximum)


def configure_blas_threads(num_threads: int = 1, *, force: bool = False) -> None:
    """Limit per-process BLAS threads to prevent oversubscription."""

    value = str(max(1, int(num_threads)))
    for name in THREAD_ENV_VARS:
        if force or not os.environ.get(name):
            os.environ[name] = value


def _threadpool_context(num_threads: int = 1):
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        return nullcontext()
    return threadpool_limits(limits=max(1, int(num_threads)))


configure_blas_threads(1)


def _worker_initializer(blas_threads: int = 1) -> None:
    os.environ[WORKER_ENV_VAR] = "1"
    configure_blas_threads(blas_threads, force=True)


def in_parallel_worker() -> bool:
    return os.environ.get(WORKER_ENV_VAR) == "1"


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


def add_parallel_arguments(
    parser: argparse.ArgumentParser,
    *,
    default: str = "auto",
    include_chunk_size: bool = False,
    default_chunk_size: int = 1000,
) -> None:
    """Add shared ``--workers`` and optional ``--chunk-size`` arguments."""

    parser.add_argument(
        "--workers",
        default=default,
        help="Number of worker processes. Use 'auto' to use all CPU cores.",
    )
    if include_chunk_size:
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=default_chunk_size,
            help="Number of events per chunk for single-file processing.",
        )


def config_from_args(args: argparse.Namespace, *, chunk_size: int | None = None) -> ParallelConfig:
    return ParallelConfig(
        workers=getattr(args, "workers", "auto"),
        chunk_size=int(chunk_size if chunk_size is not None else getattr(args, "chunk_size", 1000)),
    )


def chunk_ranges(n_items: int, chunk_size: int) -> Iterator[Tuple[int, int]]:
    """Yield ``(start, stop)`` chunks for event-wise processing."""

    if n_items < 0:
        raise ValueError("n_items must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, n_items, chunk_size):
        yield start, min(start + chunk_size, n_items)


def _serial_completed(func: Callable[[T], R], item_list: Sequence[T]) -> Iterator[Tuple[T, R]]:
    for item in item_list:
        yield item, func(item)


def process_map(
    func: Callable[[T], R],
    items: Sequence[T] | Iterable[T],
    *,
    workers: int | str | None = "auto",
    config: ParallelConfig | None = None,
) -> List[R]:
    """Map a function across items, preserving input order."""

    item_list = list(items)
    if not item_list:
        return []

    cfg = config or ParallelConfig(workers=workers)
    worker_count = cfg.resolved_workers(maximum=len(item_list))
    if worker_count <= 1:
        return [func(item) for item in item_list]

    with _threadpool_context(cfg.blas_threads):
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_worker_initializer,
            initargs=(cfg.blas_threads,),
        ) as executor:
            futures = [executor.submit(func, item) for item in item_list]
            return [future.result() for future in futures]


def iter_completed(
    func: Callable[[T], R],
    items: Sequence[T] | Iterable[T],
    *,
    workers: int | str | None = "auto",
    config: ParallelConfig | None = None,
) -> Iterator[Tuple[T, R]]:
    """Yield ``(item, result)`` as process-pool tasks complete."""

    item_list = list(items)
    if not item_list:
        return

    cfg = config or ParallelConfig(workers=workers)
    worker_count = cfg.resolved_workers(maximum=len(item_list))
    if worker_count <= 1:
        yield from _serial_completed(func, item_list)
        return

    with _threadpool_context(cfg.blas_threads):
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_worker_initializer,
            initargs=(cfg.blas_threads,),
        ) as executor:
            future_to_item = {executor.submit(func, item): item for item in item_list}
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                yield item, future.result()


def map_unordered(
    func: Callable[[T], R],
    items: Sequence[T] | Iterable[T],
    *,
    workers: int | str | None = "auto",
    config: ParallelConfig | None = None,
) -> Iterator[Tuple[T, R]]:
    """Alias for unordered process-pool iteration."""

    yield from iter_completed(func, items, workers=workers, config=config)


def map_files(
    func: Callable[[T], R],
    files: Sequence[T] | Iterable[T],
    *,
    workers: int | str | None = "auto",
    config: ParallelConfig | None = None,
) -> List[R]:
    """File-level multiprocessing helper."""

    return process_map(func, files, workers=workers, config=config)


def map_chunks(
    func: Callable[[Tuple[int, int]], R],
    n_items: int,
    *,
    chunk_size: int = 1000,
    workers: int | str | None = "auto",
    config: ParallelConfig | None = None,
) -> List[R]:
    """Chunk-level multiprocessing helper for one large event collection."""

    cfg = config or ParallelConfig(workers=workers, chunk_size=chunk_size)
    return process_map(func, list(chunk_ranges(n_items, cfg.chunk_size)), config=cfg)


@contextmanager
def joblib_parallel(
    *,
    workers: int | str | None = "auto",
    backend: str = "loky",
    config: ParallelConfig | None = None,
    **kwargs,
):
    """Yield a configured ``joblib.Parallel`` instance.

    Importing joblib is optional.  When called from inside one of our process
    workers, the returned object uses ``n_jobs=1`` unless the caller explicitly
    creates a ``ParallelConfig(allow_nested=True)``.
    """

    try:
        from joblib import Parallel, parallel_config
    except Exception as exc:
        raise ImportError("Install the 'signal' or 'ml' extra to use joblib parallelism") from exc

    cfg = config or ParallelConfig(workers=workers)
    worker_count = cfg.resolved_workers()
    with _threadpool_context(cfg.blas_threads):
        if backend == "loky":
            with parallel_config(backend=backend, inner_max_num_threads=cfg.blas_threads):
                yield Parallel(n_jobs=worker_count, **kwargs)
        else:
            yield Parallel(n_jobs=worker_count, backend=backend, **kwargs)


__all__ = [
    "ParallelConfig",
    "add_parallel_arguments",
    "chunk_ranges",
    "config_from_args",
    "configure_blas_threads",
    "cpu_count",
    "in_parallel_worker",
    "iter_completed",
    "joblib_parallel",
    "map_chunks",
    "map_files",
    "map_unordered",
    "process_map",
    "resolve_workers",
]
