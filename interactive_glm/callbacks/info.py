# coding=utf-8
"""
Updates the beta bar chart and the measured-vs-fitted signal plot when the
cursor moves.

The design timeline isn't wired up here because it doesn't depend on
anything interactive - it shows the fixed experimental paradigm, not the
current voxel - so it's built once, in
:func:`interactive_glm.components.info_panels.build_info_panels`, and
never touched again.
"""

from __future__ import annotations

from dash import Dash, Input, Output

from interactive_glm.components.info_panels import build_current_timeseries_figure
from interactive_glm.config import BrainData, ViewerSettings
from interactive_glm.figures.beta_bar import build_beta_figure
from interactive_glm.processing.glm import voxel_betas


def register_info_callbacks(app: Dash, data: BrainData, settings: ViewerSettings) -> None:
    """Register the beta-bar-chart and measured-vs-fitted-signal callbacks.

    Each is a no-op (not registered at all) if the corresponding data isn't
    available, mirroring how the panels themselves are only built when
    there's something to show - see ``BrainData.has_glm`` / ``has_timeseries``.
    """
    if not data.has_glm:
        return

    @app.callback(
        Output("beta-graph", "figure"),
        Input("cursor-position", "data"),
    )
    def update_beta_chart(cursor_data: dict):
        """Dash callback: rebuild the beta bar chart for the voxel at ``cursor_data``."""
        cursor = (cursor_data["i"], cursor_data["j"], cursor_data["k"])
        values = voxel_betas(data.betas, cursor)
        return build_beta_figure(data.condition_names, values=values, y_range=data.beta_range)

    if data.has_timeseries:
        total_duration = (data.t_r * data.n_scans) if (data.t_r and data.n_scans) else None

        @app.callback(
            Output("timeseries-graph", "figure"),
            Input("cursor-position", "data"),
        )
        def update_timeseries_chart(cursor_data: dict):
            """Dash callback: rebuild the measured-vs-fitted plot for the voxel at ``cursor_data``."""
            cursor = (cursor_data["i"], cursor_data["j"], cursor_data["k"])
            return build_current_timeseries_figure(data, cursor, total_duration)