# coding=utf-8
"""Bar chart of per-condition GLM effect sizes (betas) at the selected voxel."""

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go

from interactive_glm.figures import theme


def build_beta_figure(
    condition_names: list,
    values: Optional[np.ndarray] = None,
    y_range: Optional[tuple[float, float]] = None,
) -> go.Figure:
    """Bar chart of one beta value per condition.

    Parameters
    ----------
    condition_names : list of str
        X-axis categories, in display order.
    values : numpy.ndarray, optional
        One value per condition. If None, all bars are drawn at zero (used
        before any voxel has been selected).
    y_range : tuple of float, optional
        Fixed y-axis range - see :func:`interactive_glm.processing.glm.beta_display_range`.
        Keeping this fixed across updates avoids the chart rescaling (and
        re-anchoring the reader's sense of scale) on every click.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if values is None:
        values = np.zeros(len(condition_names))

    colors = theme.condition_colors(len(condition_names))

    fig = go.Figure(
        go.Bar(
            x=condition_names,
            y=values,
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="%{x}: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        theme.base_layout_kwargs(
            title=dict(
                text="Effect size (β) at selected voxel",
                font=dict(size=13, color=theme.TEXT_PRIMARY, family=theme.FONT_FAMILY),
                x=0.01,
            ),
            xaxis=dict(showgrid=False, tickangle=-25, color=theme.TEXT_MUTED, linecolor=theme.BORDER),
            yaxis=dict(
                showgrid=True,
                gridcolor=theme.BORDER,
                zeroline=True,
                zerolinecolor=theme.BORDER,
                color=theme.TEXT_MUTED,
                title=dict(text="beta", font=dict(size=11)),
            ),
            margin=dict(l=40, r=8, t=32, b=50),
        )
    )
    if y_range is not None:
        fig.update_yaxes(range=list(y_range))
    return fig
