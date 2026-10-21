# coding=utf-8
"""
Recomputing a GLM contrast - the one operation in the app expensive enough
(~400ms at typical resolution; see the benchmark this number came from in
``tests/test_performance.py``) to deliberately gate behind an explicit
action instead of firing on every keystroke or slider drag.

:func:`compute_and_apply_contrast` does the actual work and is used both by
the Dash callback below *and* by :func:`interactive_glm.app.create_app` to
eagerly compute ``settings.default_contrast`` once at startup, so the two
never drift apart.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from dash import Dash, Input, Output, State
from dash.exceptions import PreventUpdate

from interactive_glm.config import BrainData, OverlayLayer, ViewerSettings
from interactive_glm.processing.alignment import align_to_reference
from interactive_glm.processing.glm import compute_contrast_img


def compute_and_apply_contrast(
    data: BrainData,
    settings: ViewerSettings,
    contrast_text: str,
    threshold: Optional[float] = None,
    cmap: Optional[str] = None,
    opacity: Optional[float] = None,
) -> str:
    """Compute ``contrast_text``, install it as ``data.overlays["stat_map"]``, return a status string.

    Any styling parameter left as None is carried over from the layer
    being replaced (or from ``settings`` if there isn't one yet), so this
    can be called just to recompute the map without disturbing the user's
    current threshold/colormap/opacity choices.

    Parameters
    ----------
    data : BrainData
        Mutated in place: ``data.overlays["stat_map"]`` is replaced.
    settings : ViewerSettings
    contrast_text : str
        Any contrast expression ``model.compute_contrast`` accepts.
    threshold, cmap, opacity : optional
        Styling overrides for the new layer.

    Returns
    -------
    str
        A short, human-readable status message - either a success summary
        (with a voxel count, as a quick sanity check that the contrast
        found *something*) or a friendly description of what went wrong.
        Never raises: invalid contrast expressions are a normal, expected
        user action (typos, unknown condition names), not a bug.
    """
    if not data.has_glm:
        return ""

    contrast_text = (contrast_text or "").strip()
    if not contrast_text:
        return 'Enter a contrast expression, e.g. a condition name or "A - B".'

    existing = data.overlays.get("stat_map")
    threshold = threshold if threshold is not None else (existing.threshold if existing else settings.default_threshold)
    cmap = cmap or (existing.cmap if existing else settings.stat_cmap)
    opacity = opacity if opacity is not None else (existing.opacity if existing else settings.stat_opacity)

    try:
        contrast_img = compute_contrast_img(data.glm_model, contrast_text, stat_type="t", output_type="z_score")
        aligned = align_to_reference(contrast_img, data.reference_template, interpolation="continuous")
    except Exception as exc:  # noqa: BLE001 - any failure here is a user-facing message, not a crash
        return f'Could not compute "{contrast_text}": {exc}'

    data.overlays["stat_map"] = OverlayLayer(
        key="stat_map", label=contrast_text, kind="stat_map", data=aligned,
        cmap=cmap, opacity=opacity, threshold=threshold,
    )
    n_significant = int(np.sum(np.abs(aligned) >= threshold))
    return f'Showing "{contrast_text}"  ·  {n_significant:,} voxels ≥ {threshold:g}'


def register_contrast_callbacks(app: Dash, data: BrainData, settings: ViewerSettings) -> None:
    """Register the "Apply contrast" callback."""

    @app.callback(
        Output("contrast-status", "children"),
        Input("contrast-apply", "n_clicks"),
        Input("contrast-input", "n_submit"),
        State("contrast-input", "value"),
        State("stat-threshold-slider", "value"),
        State("stat-cmap-dropdown", "value"),
        State("stat-opacity-slider", "value"),
        prevent_initial_call=True,
    )
    def apply_contrast(_n_clicks, _n_submit, contrast_text, threshold, cmap, opacity):
        """Dash callback: recompute the active contrast when Apply is clicked or Enter is pressed."""
        if not data.has_glm:
            raise PreventUpdate
        return compute_and_apply_contrast(data, settings, contrast_text, threshold, cmap, opacity)
