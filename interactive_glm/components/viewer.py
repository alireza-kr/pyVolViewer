# coding=utf-8
"""The main viewer: three orthogonal-view graphs plus a coordinate readout."""

from __future__ import annotations

from dash import dcc, html

from interactive_glm.config import BrainData, ViewerSettings
from interactive_glm.processing import alignment as al
from interactive_glm.processing import compositing as comp

#: (axis constant, display label), in the order panels are laid out.
#: Reused by interactive_glm.callbacks.viewer so the redraw callback's
#: outputs line up with these panels - the one place to edit if you ever
#: want to reorder, rename, or add a view.
PANEL_ORDER = [
    (al.SAGITTAL, "Sagittal"),
    (al.CORONAL, "Coronal"),
    (al.AXIAL, "Axial"),
]


def slice_graph_id(axis: int) -> dict:
    """Pattern-matching id for the graph of one view - shared with the callbacks module."""
    return {"type": "slice-graph", "axis": axis}


def coordinate_readout_text(ijk: tuple[int, int, int], affine) -> str:
    """``"voxel (i, j, k)  ·  MNI (x, y, z) mm"``, for the readout under the viewer."""
    x, y, z = al.voxel_to_mm(ijk, affine)
    return f"voxel ({ijk[0]}, {ijk[1]}, {ijk[2]})   ·   MNI ({x:.0f}, {y:.0f}, {z:.0f}) mm"


def panel_title(axis: int, ijk: tuple[int, int, int], affine) -> str:
    """Small per-panel title, e.g. ``"x = -42 mm"`` - shown in the figure's corner."""
    x, y, z = al.voxel_to_mm(ijk, affine)
    mm_by_axis = {al.SAGITTAL: x, al.CORONAL: y, al.AXIAL: z}
    axis_letter = {al.SAGITTAL: "x", al.CORONAL: "y", al.AXIAL: "z"}
    return f"{axis_letter[axis]} = {mm_by_axis[axis]:.0f} mm"


def build_initial_figure(data: BrainData, settings: ViewerSettings, axis: int, cursor: tuple[int, int, int]):
    """Compose and build the figure for one panel - used at layout-build time.

    The redraw callback (:mod:`interactive_glm.callbacks.viewer`) rebuilds
    figures the same way in response to interaction; this standalone
    version exists so the initial page load doesn't need a round-trip
    through the callback system just to see something.
    """
    background = data.backgrounds[settings.default_background or next(iter(data.backgrounds))]
    index = cursor[axis]
    rgb = comp.compose_panel(
        background.data, background.vmin, background.vmax, background.cmap,
        axis, index, list(data.overlays.values()),
    )
    from interactive_glm.figures.slice_views import build_slice_figure  # local import avoids a cycle at module load

    return build_slice_figure(
        rgb, axis, data.shape, cursor,
        title=panel_title(axis, cursor, data.affine),
        show_direction_labels=settings.show_direction_labels,
    )


def build_viewer(data: BrainData, settings: ViewerSettings, cursor: tuple[int, int, int]) -> html.Div:
    """The full viewer region: three panels in a row, plus the coordinate readout.

    Parameters
    ----------
    data : BrainData
    settings : ViewerSettings
    cursor : tuple of int
        Initial (i, j, k) voxel to center each panel on.

    Returns
    -------
    dash.html.Div
    """
    panels = []
    for axis, label in PANEL_ORDER:
        fig = build_initial_figure(data, settings, axis, cursor)
        panels.append(
            html.Div(
                className="slice-panel",
                children=[
                    html.Div(label, className="slice-panel-label"),
                    dcc.Graph(
                        id=slice_graph_id(axis),
                        figure=fig,
                        config={"displayModeBar": False, "scrollZoom": False},
                        className="slice-graph",
                    ),
                ],
            )
        )

    return html.Div(
        id="viewer",
        children=[
            html.Div(id="viewer-row", children=panels),
            html.Div(
                id="coordinate-readout",
                className="coordinate-readout",
                children=coordinate_readout_text(cursor, data.affine),
            ),
        ],
    )
