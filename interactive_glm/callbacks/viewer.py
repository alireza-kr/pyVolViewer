from __future__ import annotations
from dash import ALL, Input, Output, State, ctx
from dash.exceptions import PreventUpdate
from interactive_glm.components.viewer import PANEL_ORDER, coordinate_readout_text, panel_title, slice_graph_id
from interactive_glm.figures.slice_views import build_slice_figure
from interactive_glm.processing import alignment as al
from interactive_glm.processing import compositing as comp


def register_viewer_callbacks(app, data, settings):
    @app.callback(
        Output("cursor-position", "data"),
        Input({"type": "slice-graph", "axis": ALL}, "clickData"),
        State("cursor-position", "data"),
        prevent_initial_call=True,
    )
    def handle_click(_all_click_data, cursor_data):
        triggered_id = ctx.triggered_id
        if triggered_id is None:
            raise PreventUpdate
        click = ctx.triggered[0]["value"]
        if not click or not click.get("points"):
            raise PreventUpdate
        point = click["points"][0]
        row, col = point["y"], point["x"]
        current_ijk = (cursor_data["i"], cursor_data["j"], cursor_data["k"])
        new_ijk = al.panel_rc_to_voxel(row, col, current_ijk, data.shape, triggered_id["axis"])
        return {"i": new_ijk[0], "j": new_ijk[1], "k": new_ijk[2]}

    @app.callback(
        Output(slice_graph_id(PANEL_ORDER[0][0]), "figure"),
        Output(slice_graph_id(PANEL_ORDER[1][0]), "figure"),
        Output(slice_graph_id(PANEL_ORDER[2][0]), "figure"),
        Output("coordinate-readout", "children"),
        Input("cursor-position", "data"),
        Input("background-selector", "value"),
        Input("roi-visibility-store", "data"),
        Input("roi-opacity-slider", "value"),
        Input("stat-threshold-slider", "value"),
        Input("stat-cmap-dropdown", "value"),
        Input("stat-opacity-slider", "value"),
        Input("contrast-status", "children"),
    )
    def redraw(cursor_data, background_key, roi_visible_keys, roi_opacity, stat_threshold, stat_cmap, stat_opacity, _contrast_status):
        cursor = (cursor_data["i"], cursor_data["j"], cursor_data["k"])
        visible = set(roi_visible_keys or [])
        for layer in data.roi_layers:
            layer.visible = layer.key in visible
            layer.opacity = roi_opacity
        stat_layer = data.overlays.get("stat_map")
        if stat_layer is not None:
            stat_layer.threshold = stat_threshold
            stat_layer.cmap = stat_cmap
            stat_layer.opacity = stat_opacity
        background = data.backgrounds[background_key]
        overlays = list(data.overlays.values())
        figures = []
        for axis, _label in PANEL_ORDER:
            rgb = comp.compose_panel(background.data, background.vmin, background.vmax, background.cmap, axis, cursor[axis], overlays)
            figures.append(build_slice_figure(rgb, axis, data.shape, cursor, title=panel_title(axis, cursor, data.affine),
                                                show_direction_labels=settings.show_direction_labels))
        return (*figures, coordinate_readout_text(cursor, data.affine))