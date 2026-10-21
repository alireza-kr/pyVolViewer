# coding=utf-8
"""
The single public entry point most users need:
:func:`create_app` turns a :class:`~interactive_glm.config.BrainData` and
:class:`~interactive_glm.config.ViewerSettings` into a ready-to-run Dash app.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from dash import Dash

from interactive_glm.callbacks import register_all_callbacks
from interactive_glm.callbacks.contrast import compute_and_apply_contrast
from interactive_glm.components.layout import build_layout
from interactive_glm.config import BrainData, ViewerSettings

# NOTE: assets_folder is deliberately *not* overridden here. Dash's own
# default already resolves "assets/" relative to wherever Dash(__name__) is
# instantiated (i.e. right next to this file, at interactive_glm/assets/),
# using Flask's cross-platform root-path resolution. An earlier version of
# this file computed a custom path with `__file__.rsplit("/", 1)`, which
# silently produced a bogus, nonexistent path on Windows (backslash paths
# have no "/" to split on) - the app would still run, just with zero CSS
# applied, which is a much more confusing failure than a crash. Let Dash do
# this itself; don't second-guess it with manual string splitting.


def create_app(data: BrainData, settings: Optional[ViewerSettings] = None) -> Dash:
    """Build a ready-to-run Dash app from already-loaded brain data.

    Parameters
    ----------
    data : BrainData
        Typically built with :func:`interactive_glm.data.loading.load_brain_data`
        or :func:`interactive_glm.data.demo.build_demo_dataset`.
    settings : ViewerSettings, optional
        Display and server settings. Defaults to ``ViewerSettings()``.

    Returns
    -------
    dash.Dash
        Call ``app.run(debug=settings.debug, host=settings.host, port=settings.port)``
        to start serving.

    Examples
    --------
    >>> from interactive_glm.data.demo import build_demo_dataset
    >>> from interactive_glm.app import create_app
    >>> app = create_app(build_demo_dataset())
    >>> app.run(debug=False)  # doctest: +SKIP
    """
    settings = settings or ViewerSettings()

    if data.backgrounds and (settings.default_background not in data.backgrounds):
        settings.default_background = next(iter(data.backgrounds))

    initial_status = ""
    if data.has_glm and settings.default_contrast:
        initial_status = compute_and_apply_contrast(data, settings, settings.default_contrast)

    cursor = settings.initial_voxel or _pick_initial_voxel(data)

    app = Dash(__name__, title=settings.title, update_title=None)
    app.layout = build_layout(data, settings, cursor, initial_status=initial_status)
    register_all_callbacks(app, data, settings)
    return app


def _pick_initial_voxel(data: BrainData) -> tuple[int, int, int]:
    """A reasonable starting crosshair position: stat-map peak, else first ROI's
    centroid, else the volume center."""
    stat_layer = data.overlays.get("stat_map")
    if stat_layer is not None and np.any(stat_layer.data):
        flat_index = int(np.argmax(np.abs(stat_layer.data)))
        return tuple(int(v) for v in np.unravel_index(flat_index, stat_layer.data.shape))

    for layer in data.roi_layers:
        mask = layer.data > 0.5
        if mask.any():
            centroid = np.argwhere(mask).mean(axis=0).round()
            return tuple(int(v) for v in centroid)

    return tuple(s // 2 for s in data.shape)