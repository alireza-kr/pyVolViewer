# coding=utf-8
"""Beta bar chart + design timeline + measured-vs-fitted signal panels (GLM-only)."""

from __future__ import annotations

from dash import dcc, html

from interactive_glm.config import BrainData
from interactive_glm.figures.beta_bar import build_beta_figure
from interactive_glm.figures.timeline import build_timeline_figure
from interactive_glm.figures.timeseries import build_empty_timeseries_figure, build_timeseries_figure
from interactive_glm.processing.glm import voxel_betas
from interactive_glm.processing.timeseries import voxel_timeseries_and_fit


def build_info_panels(data: BrainData, cursor: tuple[int, int, int]) -> html.Div:
    """Beta bar chart (updates on click), design timeline (static), and - if a
    functional run was loaded alongside the GLM - the measured-vs-fitted
    signal plot (updates on click), stacked below the timeline. Empty if no
    GLM."""
    if not data.has_glm:
        return html.Div()

    values = voxel_betas(data.betas, cursor)
    beta_fig = build_beta_figure(data.condition_names, values=values, y_range=data.beta_range)

    beta_panel = html.Div(
        className="info-panel",
        children=[
            dcc.Graph(
                id="beta-graph", figure=beta_fig,
                config={"displayModeBar": False}, className="info-graph",
            )
        ],
    )

    timeline_panel_children = []
    if data.events is not None:
        total_duration = (data.t_r * data.n_scans) if (data.t_r and data.n_scans) else None
        timeline_fig = build_timeline_figure(data.events, data.condition_names, total_duration=total_duration)
        timeline_panel_children.append(
            dcc.Graph(
                id="timeline-graph", figure=timeline_fig,
                config={"displayModeBar": False}, className="info-graph",
            )
        )

    if data.has_timeseries:
        total_duration = (data.t_r * data.n_scans) if (data.t_r and data.n_scans) else None
        ts_fig = build_current_timeseries_figure(data, cursor, total_duration)
        timeline_panel_children.append(
            dcc.Graph(
                id="timeseries-graph", figure=ts_fig,
                config={"displayModeBar": False}, className="info-graph",
            )
        )

    children = [beta_panel]
    if timeline_panel_children:
        children.append(html.Div(className="info-panel", children=timeline_panel_children))

    return html.Div(id="info-panels", children=children)


def build_current_timeseries_figure(data: BrainData, cursor: tuple[int, int, int], total_duration):
    """The measured-vs-per-condition-fit figure for one voxel.

    Public (not underscore-prefixed) because both the initial layout build
    (here) and the click callback (``callbacks.info``) need to build this
    exact figure the same way - same pattern as
    ``components.viewer.panel_title``.
    """
    result = voxel_timeseries_and_fit(
        data.glm_model, data.func_data, data.func_affine, cursor, data.affine, data.condition_names
    )
    if result is None:
        return build_empty_timeseries_figure()
    frame_times, measured, condition_fits = result
    return build_timeseries_figure(frame_times, measured, condition_fits, data.condition_names, total_duration=total_duration)