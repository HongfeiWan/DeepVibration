"""Legacy-compatible energy spectrum helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ENERGY_CAL_A = 0.0008432447500464594
ENERGY_CAL_B = -0.826976770117076
EXPOSURE_KG = 0.5
EXPOSURE_DAYS = 20.0
SPECTRUM_N_BINS_ACT = 500
SPECTRUM_E_MIN_KEV = 0.1
SPECTRUM_E_MAX_KEV = 12.0


@dataclass(frozen=True)
class Spectrum:
    """One binned energy spectrum."""

    bin_edges_kev: np.ndarray
    counts: np.ndarray
    rate_cpkkd: np.ndarray
    exposure_days: float

    @property
    def bin_centers_kev(self) -> np.ndarray:
        return 0.5 * (self.bin_edges_kev[:-1] + self.bin_edges_kev[1:])


def ch0_energy_kev(
    max_ch0: np.ndarray,
    *,
    cal_a: float = ENERGY_CAL_A,
    cal_b: float = ENERGY_CAL_B,
) -> np.ndarray:
    """Convert CH0 max ADC values to keV with the legacy calibration."""

    return float(cal_a) * np.asarray(max_ch0, dtype=np.float64).reshape(-1) + float(cal_b)


def spectrum_rate_from_counts(
    counts: np.ndarray,
    bin_edges_kev: np.ndarray,
    *,
    exposure_kg: float = EXPOSURE_KG,
    exposure_days: float = EXPOSURE_DAYS,
) -> np.ndarray:
    """Convert histogram counts to counts / (keV kg day)."""

    c = np.asarray(counts, dtype=np.float64).reshape(-1)
    edges = np.asarray(bin_edges_kev, dtype=np.float64).reshape(-1)
    if edges.size != c.size + 1:
        raise ValueError("bin_edges_kev must have len(counts) + 1 values")
    days = float(exposure_days)
    kg = float(exposure_kg)
    if days <= 0.0:
        raise ValueError("exposure_days must be positive")
    if kg <= 0.0:
        raise ValueError("exposure_kg must be positive")
    widths = np.diff(edges)
    widths = np.where(widths > 0.0, widths, np.inf)
    return c / (kg * widths * days)


def compute_spectrum(
    max_ch0: np.ndarray,
    mask: np.ndarray,
    *,
    bin_edges_kev: np.ndarray | None = None,
    n_bins: int = SPECTRUM_N_BINS_ACT,
    e_min_kev: float = SPECTRUM_E_MIN_KEV,
    e_max_kev: float = SPECTRUM_E_MAX_KEV,
    exposure_kg: float = EXPOSURE_KG,
    exposure_days: float = EXPOSURE_DAYS,
    cal_a: float = ENERGY_CAL_A,
    cal_b: float = ENERGY_CAL_B,
) -> Spectrum:
    """Compute a CH0-derived energy spectrum for the selected events."""

    x = np.asarray(max_ch0, dtype=np.float64).reshape(-1)
    m = np.asarray(mask, dtype=bool).reshape(-1)
    n = min(x.size, m.size)
    energy = ch0_energy_kev(x[:n][m[:n]], cal_a=cal_a, cal_b=cal_b)
    edges = (
        np.asarray(bin_edges_kev, dtype=np.float64).reshape(-1)
        if bin_edges_kev is not None
        else np.linspace(float(e_min_kev), float(e_max_kev), int(n_bins) + 1)
    )
    counts, edges = np.histogram(energy, bins=edges)
    rates = spectrum_rate_from_counts(
        counts,
        edges,
        exposure_kg=exposure_kg,
        exposure_days=exposure_days,
    )
    return Spectrum(
        bin_edges_kev=np.asarray(edges, dtype=np.float64),
        counts=np.asarray(counts, dtype=np.int64),
        rate_cpkkd=np.asarray(rates, dtype=np.float64),
        exposure_days=float(exposure_days),
    )


__all__ = [
    "ENERGY_CAL_A",
    "ENERGY_CAL_B",
    "EXPOSURE_DAYS",
    "EXPOSURE_KG",
    "SPECTRUM_E_MAX_KEV",
    "SPECTRUM_E_MIN_KEV",
    "SPECTRUM_N_BINS_ACT",
    "Spectrum",
    "ch0_energy_kev",
    "compute_spectrum",
    "spectrum_rate_from_counts",
]
