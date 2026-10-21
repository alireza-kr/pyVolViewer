# coding=utf-8
"""
Dash callbacks, split by responsibility:

* :mod:`interactive_glm.callbacks.viewer` - click-to-navigate and redrawing
  the three slice panels.
* :mod:`interactive_glm.callbacks.rois` - toggling ROI visibility from the
  sidebar's custom toggle rows.
* :mod:`interactive_glm.callbacks.contrast` - recomputing a GLM contrast,
  gated behind an explicit "Apply" action since it's meaningfully slower
  than everything else in the app.
* :mod:`interactive_glm.callbacks.info` - the beta bar chart and the
  measured-vs-fitted signal panel.
"""

from __future__ import annotations

from dash import Dash

from interactive_glm.callbacks.contrast import register_contrast_callbacks
from interactive_glm.callbacks.info import register_info_callbacks
from interactive_glm.callbacks.rois import register_roi_callbacks
from interactive_glm.callbacks.viewer import register_viewer_callbacks
from interactive_glm.config import BrainData, ViewerSettings


def register_all_callbacks(app: Dash, data: BrainData, settings: ViewerSettings) -> None:
    """Register every callback the app needs. Call once, after the layout is set."""
    register_viewer_callbacks(app, data, settings)
    register_roi_callbacks(app, data, settings)
    register_contrast_callbacks(app, data, settings)
    register_info_callbacks(app, data, settings)