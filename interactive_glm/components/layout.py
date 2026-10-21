# coding=utf-8
"""Assembles the full page layout from the individual component builders."""

from __future__ import annotations

from dash import dcc, html

from interactive_glm.components.controls import build_controls
from interactive_glm.components.info_panels import build_info_panels
from interactive_glm.components.viewer import build_viewer
from interactive_glm.config import BrainData, ViewerSettings


def build_layout(
    data: BrainData, settings: ViewerSettings, cursor: tuple[int, int, int], initial_status: str = ""
) -> html.Div:
    """Build ``app.layout``.

    Parameters
    ----------
    data : BrainData
    settings : ViewerSettings
    cursor : tuple of int
        Initial (i, j, k) voxel - see ``interactive_glm.app._pick_initial_voxel``.
    initial_status : str
        Status text for the contrast panel, if a default contrast was
        computed eagerly at startup (see ``interactive_glm.app.create_app``).

    Returns
    -------
    dash.html.Div
    """
    return html.Div(
        id="app-root",
        children=[
            html.Div(
                id="app-header",
                children=[
                    html.H1(settings.title, id="app-title"),
                    html.Div(
                        "Click any view to move the crosshair.",
                        id="app-subtitle",
                    ),
                ],
            ),
            html.Div(
                id="app-body",
                children=[
                    build_controls(data, settings, initial_status=initial_status),
                    html.Div(
                        id="main-column",
                        children=[
                            build_viewer(data, settings, cursor),
                            build_info_panels(data, cursor),
                        ],
                    ),
                ],
            ),
            # Holds the current (i, j, k) voxel; updated by clicks on any
            # slice panel, read by every callback that redraws the view.
            dcc.Store(id="cursor-position", data={"i": cursor[0], "j": cursor[1], "k": cursor[2]}),
        ],
    )
