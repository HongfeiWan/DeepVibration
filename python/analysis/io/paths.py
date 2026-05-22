"""Repository and data path discovery helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


@dataclass(frozen=True)
class RunFiles:
    """Matched files for one run in ``data/hdf5/raw_pulse``."""

    name: str
    files: Mapping[str, Path]

    def get(self, key: str) -> Path:
        return self.files[key]


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the repository root by walking upward from ``start``."""

    here = Path.cwd() if start is None else Path(start).resolve()
    if here.is_file():
        here = here.parent
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists() or (
            (candidate / "README.md").exists() and (candidate / "python").exists()
        ):
            return candidate
    raise RuntimeError(f"Could not locate project root from {here}")


def raw_pulse_dir(project_root: str | Path | None = None) -> Path:
    root = find_project_root(project_root)
    return root / "data" / "hdf5" / "raw_pulse"


def list_h5_files(directory: str | Path) -> List[Path]:
    """Return sorted HDF5 files from a directory."""

    path = Path(directory)
    if not path.exists():
        return []
    return sorted(
        p for p in path.iterdir() if p.is_file() and p.suffix.lower() in {".h5", ".hdf5"}
    )


def list_raw_pulse_files(
    channel: str,
    *,
    project_root: str | Path | None = None,
) -> List[Path]:
    return list_h5_files(raw_pulse_dir(project_root) / channel)


def pair_raw_pulse_files(
    channels: Sequence[str] = ("CH0-3", "CH5"),
    *,
    project_root: str | Path | None = None,
) -> List[RunFiles]:
    """Pair raw-pulse files by basename across the requested channel folders."""

    base = raw_pulse_dir(project_root)
    by_channel: Dict[str, Dict[str, Path]] = {}
    for channel in channels:
        by_channel[channel] = {p.name: p for p in list_h5_files(base / channel)}
    if not by_channel:
        return []
    common_names = set.intersection(*(set(v) for v in by_channel.values()))
    return [
        RunFiles(name=name, files={channel: by_channel[channel][name] for channel in channels})
        for name in sorted(common_names)
    ]


def parameter_dir(channel: int | str, *, project_root: str | Path | None = None) -> Path:
    label = str(channel).upper()
    if not label.startswith("CH"):
        label = f"CH{label}"
    return raw_pulse_dir(project_root) / f"{label}_parameters"


def pair_parameter_files(
    channels: Iterable[int | str],
    *,
    project_root: str | Path | None = None,
) -> List[RunFiles]:
    """Pair parameter files by basename across channel parameter folders."""

    labels = [str(ch).upper() if str(ch).upper().startswith("CH") else f"CH{ch}" for ch in channels]
    by_channel: Dict[str, Dict[str, Path]] = {}
    for label in labels:
        by_channel[label] = {p.name: p for p in list_h5_files(parameter_dir(label, project_root=project_root))}
    if not by_channel:
        return []
    common_names = set.intersection(*(set(v) for v in by_channel.values()))
    return [
        RunFiles(name=name, files={label: by_channel[label][name] for label in labels})
        for name in sorted(common_names)
    ]


__all__ = [
    "RunFiles",
    "find_project_root",
    "list_h5_files",
    "list_raw_pulse_files",
    "pair_parameter_files",
    "pair_raw_pulse_files",
    "parameter_dir",
    "raw_pulse_dir",
]
