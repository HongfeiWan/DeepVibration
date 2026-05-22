# Plotting Style Guidelines for DeepVibration

This document defines a standard plotting style that all analysis and diagnostic
scripts should follow.

## Font Family and Sizes

- Global font family: Arial, with generic sans-serif fallback.
- Axis tick labels: 12 pt.
- Axis labels: 16 pt.
- Title: 18 pt.

Recommended global setup:

```python
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
})
```

## Histogram Standard

When plotting a 1D distribution such as `max_ch0`:

```python
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.hist(values, bins=bins, color="C0", alpha=0.8, edgecolor="black")
ax.set_xlabel("max_ch0", fontsize=16)
ax.set_ylabel("Count", fontsize=16)
ax.set_title(f"Distribution of max_ch0 (N={values.size})", fontsize=18)
fig.tight_layout()
```

## Legend and Lines

- Use 12 pt legend text by default.
- Threshold/reference lines should use clear color and line style, with a legend entry.
