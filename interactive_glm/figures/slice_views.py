# coding=utf-8
"""
Builds the three orthogonal-view figures (sagittal, coronal, axial) from an
already-composited RGB image (see
:func:`interactive_glm.processing.compositing.compose_panel`).

Orientation note
-----------------
This uses a raw ``go.Image(z=rgb_array, colormodel="rgb")`` trace with no
custom ``x0``/``dx``/``y0``/``dy``. Verified against Plotly's own
``px.imshow`` source (the code path used for a plain RGB array with the
default ``origin="upper"``): no y-axis reversal is applied there either,
because ``go.Image`` places row 0 at the top by default - unlike
``go.Heatmap``, which needs an explicit reversed range for that. That
default is exactly what :func:`interactive_glm.processing.alignment.to_display`
was built around, so row/col here can be read directly off Plotly's
``clickData`` (``point["y"]`` = row, ``point["x"]`` = column) with no extra
transform.

Hover/click note
------------------
The image trace uses ``hoverinfo="none"``, not ``"skip"``. This is
deliberate and easy to get backwards: per Plotly's own documentation,
``hoverinfo="skip"`` suppresses the tooltip *and* disables click/hover
events on that trace entirely, while ``"none"`` suppresses only the
tooltip and leaves click events (which ``callbacks.viewer`` depends on to
move the crosshair) working normally.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from interactive_glm.figures import theme
from interactive_glm.processing import alignment as al


def build_slice_figure(
    rgb_image: np.ndarray,
    axis: int,
    shape: tuple[int, int, int],
    cursor_ijk: tuple[int, int, int],
    title: str = "",
    show_direction_labels: bool = True,
) -> go.Figure:
    """One orthogonal-view panel: the image, a crosshair, and direction labels.

    Parameters
    ----------
    rgb_image : numpy.ndarray
        ``(H, W, 3)`` uint8, from :func:`~interactive_glm.processing.compositing.compose_panel`.
    axis : int
        :data:`~interactive_glm.processing.alignment.SAGITTAL` / ``CORONAL`` / ``AXIAL``.
    shape : tuple of int
        Reference grid shape, used to place the crosshair correctly.
    cursor_ijk : tuple of int
        Current (i, j, k) voxel position.
    title : str
        Small title drawn in the top-left corner (e.g. an mm coordinate).
    show_direction_labels : bool
        Whether to draw L/R/A/P/S/I labels at the panel edges.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    rows, cols = rgb_image.shape[0], rgb_image.shape[1]
    row, col = al.voxel_to_panel_rc(cursor_ijk, shape, axis)

    fig = go.Figure()
    fig.add_trace(go.Image(z=rgb_image, colormodel="rgb", hoverinfo="none"))

    crosshair_style = dict(color=theme.ACCENT, width=1)
    fig.add_shape(
        type="line", x0=col, x1=col, y0=-0.5, y1=rows - 0.5,
        line=crosshair_style, opacity=0.85, layer="above",
    )
    fig.add_shape(
        type="line", x0=-0.5, x1=cols - 0.5, y0=row, y1=row,
        line=crosshair_style, opacity=0.85, layer="above",
    )

    if show_direction_labels:
        labels = al.panel_direction_labels(axis)
        pad_x, pad_y = cols * 0.05, rows * 0.07
        label_kwargs = dict(
            showarrow=False,
            font=dict(size=12, color=theme.TEXT_MUTED, family=theme.FONT_FAMILY),
        )
        fig.add_annotation(x=cols / 2, y=pad_y, text=labels["top"], **label_kwargs)
        fig.add_annotation(x=cols / 2, y=rows - pad_y, text=labels["bottom"], **label_kwargs)
        fig.add_annotation(x=pad_x, y=rows / 2, text=labels["left"], **label_kwargs)
        fig.add_annotation(x=cols - pad_x, y=rows / 2, text=labels["right"], **label_kwargs)

    fig.update_xaxes(visible=False, range=[-0.5, cols - 0.5], fixedrange=True)
    fig.update_yaxes(visible=False, scaleanchor="x", constrain="domain", fixedrange=True)
    fig.update_layout(
        theme.base_layout_kwargs(
            title=dict(
                text=title,
                x=0.03,
                xanchor="left",
                y=0.96,
                yanchor="top",
                font=dict(size=12, color=theme.TEXT_MUTED, family=theme.FONT_FAMILY),
            ),
            dragmode=False,
        )
    )
    return fig