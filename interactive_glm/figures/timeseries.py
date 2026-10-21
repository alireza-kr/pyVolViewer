# coding=utf-8
"""
Measured BOLD signal, alongside each condition's own contribution to the
GLM's fitted signal, at one voxel.

Each condition gets its own line colored to match
:mod:`interactive_glm.figures.timeline` (both use
``theme.condition_colors`` over the same ``condition_names`` list in the
same order, so "FaceLoc" is the same color in both panels). Clicking a
name in the legend shows/hides that one line - this is Plotly's own
default legend behavior (a single click toggles that trace; a double
click isolates it, hiding all the others), not something this app wires
up itself.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go

from interactive_glm.figures import theme

#: Distinct from CONDITION_PALETTE so the measured signal never gets
#: visually confused with any condition's fitted contribution.
MEASURED_COLOR = "#8b91a3"


def build_timeseries_figure(
    frame_times: np.ndarray,
    measured: np.ndarray,
    condition_fits: dict,
    condition_names: list,
    total_duration: Optional[float] = None,
) -> go.Figure:
    """The measured signal plus one toggleable line per condition's fitted contribution.

    Parameters
    ----------
    frame_times : numpy.ndarray
        Seconds, one per scan.
    measured : numpy.ndarray
        The actual BOLD signal at the voxel.
    condition_fits : dict of str to numpy.ndarray
        Each condition's own fitted contribution - see
        :func:`interactive_glm.processing.timeseries.voxel_timeseries_and_fit`.
    condition_names : list of str
        Determines both line order and color assignment (via
        ``theme.condition_colors``) - pass the same list used for the beta
        bar chart and the design timeline so colors line up across panels.
    total_duration : float, optional
        Run length in seconds - sets the x-axis range to match the design
        timeline panel above this one, so the two line up visually.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame_times, y=measured, mode="lines+markers", name="Measured",
            line=dict(color=MEASURED_COLOR, width=1),
            marker=dict(size=3, color=MEASURED_COLOR),
            hovertemplate="t=%{x:.0f}s<br>measured=%{y:.1f}<extra></extra>",
        )
    )

    colors = theme.condition_colors(len(condition_names))
    for name, color in zip(condition_names, colors):
        if name not in condition_fits:
            continue
        fig.add_trace(
            go.Scatter(
                x=frame_times, y=condition_fits[name], mode="lines", name=name,
                line=dict(color=color, width=2),
                hovertemplate=f"{name}<br>t=" + "%{x:.0f}s<br>fit=%{y:.1f}<extra></extra>",
            )
        )

    fig.update_layout(
        theme.base_layout_kwargs(
            title=dict(
                text="Measured signal vs. GLM fit at selected voxel",
                x=0.01,
                font=dict(size=13, color=theme.TEXT_PRIMARY, family=theme.FONT_FAMILY),
            ),
            showlegend=True,
            legend=dict(
                orientation="h", x=0, y=1.22, font=dict(size=11, color=theme.TEXT_MUTED),
                bgcolor="rgba(0,0,0,0)",
            ),
            xaxis=dict(
                title=dict(text="time (s)", font=dict(size=11)),
                showgrid=True, gridcolor=theme.BORDER, color=theme.TEXT_MUTED,
                range=[0, total_duration] if total_duration else None,
            ),
            yaxis=dict(
                title=dict(text="signal", font=dict(size=11)),
                showgrid=True, gridcolor=theme.BORDER, zeroline=False, color=theme.TEXT_MUTED,
            ),
            margin=dict(l=48, r=8, t=50, b=36),
            height=250,
        )
    )
    return fig


def build_empty_timeseries_figure(message: str = "No functional coverage at this voxel") -> go.Figure:
    """Placeholder shown when the clicked voxel has no signal (outside the functional field of view)."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(size=12, color=theme.TEXT_MUTED, family=theme.FONT_FAMILY),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(theme.base_layout_kwargs(height=250))
    return fig