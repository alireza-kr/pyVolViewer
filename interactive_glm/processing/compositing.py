# coding=utf-8
"""
Turn a background volume plus any number of overlay layers into a single
RGB image for one slice, entirely with vectorized numpy - this is the
performance-critical path, since it re-runs on every click / slider move /
opacity change.

This module intentionally knows nothing about Dash, Plotly, or where the
data came from - it works on plain arrays and
:class:`~interactive_glm.config.OverlayLayer` objects. That separation is
what makes it possible to swap the rendering frontend later (e.g. render
straight to PNG for a static report) without touching this code at all.

The compositing itself is a standard "painter's algorithm" alpha blend:
start from the grayscale background, then paint each visible overlay on
top in order, each pixel blended by its own alpha.
"""

from __future__ import annotations

import numpy as np

from interactive_glm.config import OverlayLayer
from interactive_glm.processing import alignment as al
from interactive_glm.processing.colors import apply_colormap, hex_to_rgb


def _roi_alpha(values: np.ndarray, opacity: float, threshold: float) -> np.ndarray:
    """Smooth (anti-aliased) alpha ramp for an ROI layer.

    ``values`` are membership scores in [0, 1] (soft, thanks to continuous
    resampling at load time - see :func:`interactive_glm.processing.alignment.align_to_reference`).
    Below ``threshold`` a voxel is fully transparent; above it, alpha ramps
    from 0 up to ``opacity`` as the membership score approaches 1, which is
    what gives ROI outlines a smooth edge instead of jagged voxel steps.
    """
    alpha = np.zeros_like(values, dtype=np.float32)
    span = max(1.0 - threshold, 1e-6)
    inside = values > threshold
    alpha[inside] = opacity * np.clip((values[inside] - threshold) / span, 0.0, 1.0)
    return alpha


def _stat_map_alpha(values: np.ndarray, opacity: float, threshold: float) -> np.ndarray:
    """Flat alpha above ``threshold`` (in either direction), zero below - a
    standard statistical-map style hard threshold."""
    return np.where(np.abs(values) >= threshold, opacity, 0.0).astype(np.float32)


def layer_rgba(layer: OverlayLayer, slice_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Color and alpha for one overlay layer's already-extracted 2D slice.

    Returns
    -------
    rgb : numpy.ndarray
        ``(H, W, 3)`` uint8.
    alpha : numpy.ndarray
        ``(H, W)`` float32 in [0, 1].
    """
    if layer.kind == "roi":
        r, g, b = hex_to_rgb(layer.color)
        rgb = np.empty(slice_2d.shape + (3,), dtype=np.uint8)
        rgb[..., 0], rgb[..., 1], rgb[..., 2] = r, g, b
        alpha = _roi_alpha(slice_2d, layer.opacity, layer.threshold)
        return rgb, alpha

    # kind == "stat_map"
    vmax = layer.vmax if layer.vmax is not None else float(np.nanmax(np.abs(slice_2d)) or 1.0)
    vmin = layer.vmin if layer.vmin is not None else -vmax
    rgb = apply_colormap(slice_2d, layer.cmap, vmin=vmin, vmax=vmax)
    alpha = _stat_map_alpha(slice_2d, layer.opacity, layer.threshold)
    return rgb, alpha


def alpha_composite(base_rgb: np.ndarray, overlay_rgb: np.ndarray, overlay_alpha: np.ndarray) -> np.ndarray:
    """Standard "over" blend of ``overlay_rgb`` onto ``base_rgb``.

    Parameters
    ----------
    base_rgb : numpy.ndarray
        ``(H, W, 3)`` uint8.
    overlay_rgb : numpy.ndarray
        ``(H, W, 3)`` uint8.
    overlay_alpha : numpy.ndarray
        ``(H, W)`` float in [0, 1].

    Returns
    -------
    numpy.ndarray
        ``(H, W, 3)`` uint8.
    """
    alpha = overlay_alpha[..., None]
    blended = base_rgb.astype(np.float32) * (1 - alpha) + overlay_rgb.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def compose_panel(
    background: np.ndarray,
    background_vmin: float,
    background_vmax: float,
    background_cmap: str,
    axis: int,
    index: int,
    overlays: list[OverlayLayer],
) -> np.ndarray:
    """Build the final, display-oriented RGB image for one orthogonal view.

    Parameters
    ----------
    background : numpy.ndarray
        3D array on the reference grid.
    background_vmin, background_vmax : float
        Display intensity range for the background.
    background_cmap : str
        Colormap for the background (typically ``"gray"``).
    axis : int
        One of :data:`interactive_glm.processing.alignment.SAGITTAL` /
        ``CORONAL`` / ``AXIAL``.
    index : int
        Voxel index along ``axis`` to slice at.
    overlays : list of OverlayLayer
        Overlays to paint on top, in order. Layers with ``visible=False``
        are skipped by the caller (or here - both are safe).

    Returns
    -------
    numpy.ndarray
        ``(H, W, 3)`` uint8 RGB image, ready to hand to a Plotly
        ``go.Image`` trace.
    """
    bg_slice = al.to_display(al.get_slice_2d(background, axis, index))
    rgb = apply_colormap(bg_slice, background_cmap, background_vmin, background_vmax)

    for layer in overlays:
        if not layer.visible:
            continue
        layer_slice = al.to_display(al.get_slice_2d(layer.data, axis, index))
        overlay_rgb, overlay_alpha = layer_rgba(layer, layer_slice)
        rgb = alpha_composite(rgb, overlay_rgb, overlay_alpha)

    return rgb
