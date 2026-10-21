# coding=utf-8
"""
Handles clicks on the custom ROI toggle rows built in
``components.controls._roi_rows`` (see that module's docstring for why
these are hand-built ``html.Div`` rows rather than a ``dcc.Checklist``).

A single callback both updates the "currently visible" store and repaints
every row's own checked/unchecked appearance, since a click only tells us
which *one* row changed - every row's className still needs to be
re-emitted so the ones that didn't change keep their current look.
"""

from __future__ import annotations

from dash import ALL, Dash, Input, Output, State, ctx
from dash.exceptions import PreventUpdate

from interactive_glm.config import BrainData, ViewerSettings


def register_roi_callbacks(app: Dash, data: BrainData, settings: ViewerSettings) -> None:
    """Register the ROI row click-to-toggle callback. A no-op if there are no ROIs."""
    if not data.roi_layers:
        return

    @app.callback(
        Output("roi-visibility-store", "data"),
        Output({"type": "roi-toggle-row", "key": ALL}, "className"),
        Input({"type": "roi-toggle-row", "key": ALL}, "n_clicks"),
        State("roi-visibility-store", "data"),
        State({"type": "roi-toggle-row", "key": ALL}, "id"),
        prevent_initial_call=True,
    )
    def toggle_roi(_n_clicks_list, visible_keys, row_ids):
        """Dash callback: flip the clicked ROI's visibility, restyle every row to match."""
        triggered_id = ctx.triggered_id
        if triggered_id is None:
            raise PreventUpdate

        visible = set(visible_keys or [])
        clicked_key = triggered_id["key"]
        if clicked_key in visible:
            visible.discard(clicked_key)
        else:
            visible.add(clicked_key)

        class_names = ["roi-row" + ("" if row_id["key"] in visible else " roi-row-off") for row_id in row_ids]
        new_visible_keys = [layer.key for layer in data.roi_layers if layer.key in visible]
        return new_visible_keys, class_names