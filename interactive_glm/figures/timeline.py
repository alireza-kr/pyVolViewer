# coding=utf-8
"""
The experimental design timeline: when each trial type occurred over the
course of the run.

This is a Gantt-style chart (one horizontal lane per condition, one bar per
event) rather than a literal measured BOLD trace - it shows the paradigm,
not the signal, matching what the original app's "time series" panel
actually displayed. Condition colors match :mod:`interactive_glm.figures.beta_bar`
so the two panels read as one coherent story about each condition.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from interactive_glm.figures import theme


def build_timeline_figure(
    events: pd.DataFrame,
    condition_names: list,
    total_duration: Optional[float] = None,
) -> go.Figure:
    """A Gantt-style timeline of ``events``, one lane per condition.

    Parameters
    ----------
    events : pandas.DataFrame
        Columns ``onset``, ``duration``, ``trial_type`` (seconds).
    condition_names : list of str
        Preferred lane order / color assignment (matches the beta bar
        chart). Any trial type in ``events`` not present here is appended
        afterwards, so unexpected trial types still show up rather than
        being silently dropped.
    total_duration : float, optional
        Total run length in seconds, e.g. ``t_r * n_scans``. Sets the
        x-axis range so the timeline reflects the whole run even if the
        last event ends earlier.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    present = set(events["trial_type"].unique())
    ordered = [c for c in condition_names if c in present]
    ordered += sorted(present - set(ordered))
    colors = dict(zip(ordered, theme.condition_colors(len(ordered))))

    fig = go.Figure()
    for trial_type in ordered:
        sub = events[events["trial_type"] == trial_type]
        fig.add_trace(
            go.Bar(
                name=trial_type,
                x=sub["duration"],
                y=[trial_type] * len(sub),
                base=sub["onset"],
                orientation="h",
                marker=dict(color=colors[trial_type], line=dict(width=0)),
                hovertemplate=(
                    f"<b>{trial_type}</b><br>onset: %{{base:.1f}}s"
                    "<br>duration: %{x:.1f}s<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        theme.base_layout_kwargs(
            title=dict(
                text="Experimental design timeline",
                x=0.01,
                font=dict(size=13, color=theme.TEXT_PRIMARY, family=theme.FONT_FAMILY),
            ),
            barmode="overlay",
            bargap=0.35,
            xaxis=dict(
                title=dict(text="time (s)", font=dict(size=11)),
                showgrid=True,
                gridcolor=theme.BORDER,
                color=theme.TEXT_MUTED,
                range=[0, total_duration] if total_duration else None,
            ),
            yaxis=dict(
                showgrid=False,
                color=theme.TEXT_MUTED,
                categoryorder="array",
                categoryarray=ordered[::-1],
            ),
            margin=dict(l=100, r=8, t=32, b=36),
            height=max(120, 34 * len(ordered) + 50),
        )
    )
    return fig
