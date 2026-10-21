# coding=utf-8
"""
Turning numbers into colors: matplotlib colormap lookups (vectorized, for
whole slices at once) and a perceptually-distinct color palette generator
for telling multiple ROIs apart at a glance.

Importing this module also registers nilearn's neuroimaging-specific
colormaps (``cold_hot``, ``roy_big_bl``, ...) with matplotlib, so any of
the names listed at
https://nilearn.github.io/stable/auto_examples/01_plotting/plot_colormaps.html
can be used anywhere a ``cmap`` string is expected in this package.
"""

from __future__ import annotations

import colorsys

import matplotlib as mpl
import nilearn.plotting.cm  # noqa: F401  (side effect: registers nilearn cmaps)
import numpy as np

# A hand-picked, high-contrast qualitative palette (Okabe-Ito-inspired plus
# extensions), used before falling back to procedurally generated colors.
_QUALITATIVE_PALETTE = [
    "#e63946",  # red
    "#457b9d",  # blue
    "#2a9d8f",  # teal
    "#f4a261",  # orange
    "#8338ec",  # purple
    "#ffbe0b",  # yellow
    "#06d6a0",  # green
    "#ef476f",  # pink
    "#118ab2",  # cyan-blue
    "#fb8500",  # amber
]


def get_cmap(name: str):
    """Look up a matplotlib (or nilearn-registered) colormap by name."""
    return mpl.colormaps[name]


def apply_colormap(
    values: np.ndarray, cmap: str, vmin: float, vmax: float
) -> np.ndarray:
    """Map a 2D array of scalars to RGB using a named colormap.

    Parameters
    ----------
    values : numpy.ndarray
        2D array of any numeric dtype.
    cmap : str
        Matplotlib (or nilearn) colormap name.
    vmin, vmax : float
        Values are linearly normalized to [0, 1] using this range before
        colormapping (clipped at the ends). If ``vmin == vmax``, a
        constant mid-scale color is returned to avoid division by zero.

    Returns
    -------
    numpy.ndarray
        Array of shape ``values.shape + (3,)``, dtype ``uint8``.
    """
    colormap = get_cmap(cmap)
    span = vmax - vmin
    if span <= 0:
        normalized = np.full_like(values, 0.5, dtype=np.float32)
    else:
        normalized = np.clip((values - vmin) / span, 0.0, 1.0)
    rgba = colormap(normalized)  # (..., 4) float in [0, 1], fully vectorized
    return (rgba[..., :3] * 255).astype(np.uint8)


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """``"#rrggbb"`` -> ``(r, g, b)`` in [0, 255]."""
    return tuple(int(hex_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def distinct_palette(n: int) -> list[str]:
    """``n`` hex colors, chosen to be as easy to tell apart as possible.

    Uses a hand-picked qualitative palette for the first 10 colors (these
    are the ones most likely to be seen together, e.g. a handful of ROIs),
    then falls back to hues spaced evenly around the color wheel with
    alternating lightness for any additional colors needed.
    """
    if n <= len(_QUALITATIVE_PALETTE):
        return _QUALITATIVE_PALETTE[:n]

    colors = list(_QUALITATIVE_PALETTE)
    remaining = n - len(colors)
    for idx in range(remaining):
        hue = (idx / max(remaining, 1)) % 1.0
        lightness = 0.45 if idx % 2 == 0 else 0.65
        r, g, b = colorsys.hls_to_rgb(hue, lightness, 0.65)
        colors.append(
            "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        )
    return colors
