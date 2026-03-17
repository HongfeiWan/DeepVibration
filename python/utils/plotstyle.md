# Plotting Style Guidelines for DeepVibration

This document defines a **standard plotting style** that all analysis / diagnostic
scripts should follow, so that:

- Figures are visually consistent across the project.
- Plots are easy for humans and large models to read.
- Downstream automated analysis (e.g. captioning, report generation) can rely on
  consistent visual semantics.

## 1. Font Family and Sizes

- **Global font family**: `Arial` (fall back to generic sans-serif if missing).
- **Axis tick labels**:
  - Font size: **12 pt**
  - Apply via:
    ```python
    ax.tick_params(axis="both", which="major", labelsize=12)
    ```
- **Axis labels (X / Y)**:
  - Font size: **16 pt**
  - Apply via:
    ```python
    ax.set_xlabel("X label", fontsize=16)
    ax.set_ylabel("Y label", fontsize=16)
    ```
- **Title**:
  - Font size: **18 pt** (slightly larger than axis labels)
  - Apply via:
    ```python
    ax.set_title("Figure title", fontsize=18)
    ```
### Recommended global font setup
At the beginning of the plotting code (before creating figures), set:

```python
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
})
```

This ensures that all text elements (titles, labels, legends, ticks) use Arial
by default.

## 2. Histogram Standard (as used in CH0 max distribution)
When plotting a 1D distribution (e.g. `max_ch0`) as a histogram, use:
- **Figure size**: `figsize=(8, 6)` or `(10, 6)` depending on content density.
- **Histogram call**:
  ```python
  fig, ax = plt.subplots(1, 1, figsize=(8, 6))
  ax.hist(values, bins=bins, color="C0", alpha=0.8, edgecolor="black")
  ```
- **Axis labels**:
  - X label: physical quantity name, e.g. `"max_ch0"`
  - Y label: `"Count"`
- **Title**:
  - Should include the variable name and total sample count, e.g.:
    ```python
    ax.set_title(f"Distribution of max_ch0 (N={values.size})", fontsize=18)
    ```
- **Layout**:
  - Always call `fig.tight_layout()` before `plt.show()` or `plt.savefig()`:
    ```python
    fig.tight_layout()
    ```

## 3. Legend and Lines
- **Legend font size**:
  - Use **12 pt** by default:
    ```python
    ax.legend(fontsize=12)
    ```
- **Reference lines** (e.g. thresholds):
  - Use clear colors and line styles, and add a legend entry:
    ```python
    ax.axvline(
        threshold_value,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold_value:.0f}",
    )
    ax.legend(fontsize=12)
    ```

All future plotting code for distribution-like figures should follow this
standard unless there is a strong, documented reason to deviate.

