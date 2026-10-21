# coding=utf-8
"""
The sidebar: background selector, ROI toggles, and GLM contrast controls.

ROI rows are deliberately *not* built with ``dcc.Checklist``. Dash's
Checklist renders each option as ``<label><input type="checkbox"/>{label}
</label>``, and across the versions of Dash/React actually in use, that
label's internal layout (checkbox lining up with its text on one row) has
turned out to depend on internal markup Dash owns, not on CSS this project
controls - in practice it can render the checkbox and the label content
stacked on separate lines regardless of the ``display: flex`` rule applied
to the label. Rather than fight an implementation detail of a third-party
component, each ROI row here is a plain, fully hand-built ``html.Div``
(custom checkbox indicator + swatch + truncating label, click-toggled by
its own callback in ``callbacks.rois``) - every pixel of the DOM is
something this file controls directly, so the layout is exactly what's
written in ``assets/style.css``, nothing more.
"""

from __future__ import annotations

from dash import dcc, html

from interactive_glm.config import BrainData, ViewerSettings

#: Curated diverging colormaps that read well for statistical maps.
#: All are registered with matplotlib either natively or by nilearn
#: (see interactive_glm.processing.colors).
STAT_CMAP_OPTIONS = [
    {"label": "Cold / Hot", "value": "cold_hot"},
    {"label": "Blue / Red", "value": "bwr"},
    {"label": "Seismic", "value": "seismic"},
    {"label": "Cool / Warm", "value": "coolwarm"},
    {"label": "Spectral", "value": "Spectral_r"},
]


def _section(title: str, children: list) -> html.Div:
    return html.Div(
        className="control-section",
        children=[html.Div(title, className="control-section-title"), *children],
    )


def _roi_rows(data: BrainData) -> list:
    """One clickable row per ROI: a custom checkbox indicator, a color swatch,
    and a single-line, truncating label - see the module docstring for why
    this is hand-built instead of using dcc.Checklist.
    """
    rows = []
    for layer in data.roi_layers:
        rows.append(
            html.Div(
                id={"type": "roi-toggle-row", "key": layer.key},
                n_clicks=0,
                className="roi-row" + ("" if layer.visible else " roi-row-off"),
                children=[
                    html.Span(className="roi-checkbox"),
                    html.Span(className="roi-swatch", style={"backgroundColor": layer.color}),
                    html.Span(layer.label, className="roi-label-text", title=layer.label),
                ],
            )
        )
    return rows


def build_background_selector(data: BrainData, settings: ViewerSettings) -> html.Div:
    """Radio selector for which loaded image is the background."""
    default = settings.default_background or next(iter(data.backgrounds))
    options = [{"label": bg.label, "value": bg.key} for bg in data.backgrounds.values()]
    return _section(
        "Background",
        [
            dcc.RadioItems(
                id="background-selector",
                options=options,
                value=default,
                className="radio-group",
                labelClassName="radio-option",
            )
        ],
    )


def build_roi_controls(data: BrainData, settings: ViewerSettings) -> html.Div:
    """ROI visibility toggles + a shared opacity slider.

    Always renders the underlying controls (even with zero ROIs) so that
    callbacks referencing their ids stay valid regardless of whether any
    ROIs were loaded; the section is simply hidden via CSS when there is
    nothing to show. This keeps the callback graph static instead of
    needing to be assembled conditionally at runtime.
    """
    has_rois = bool(data.roi_layers)
    visible_keys = [layer.key for layer in data.roi_layers if layer.visible]
    return html.Div(
        className="control-section" + ("" if has_rois else " control-section-empty"),
        children=[
            html.Div("Regions of interest", className="control-section-title"),
            html.Div(_roi_rows(data), className="roi-checklist"),
            dcc.Store(id="roi-visibility-store", data=visible_keys),
            html.Div("Opacity", className="control-label"),
            dcc.Slider(
                id="roi-opacity-slider",
                min=0, max=1, step=0.05, value=settings.roi_opacity,
                marks=None, tooltip={"placement": "bottom", "always_visible": False},
                updatemode="mouseup",
            ),
        ],
    )


def build_contrast_controls(data: BrainData, settings: ViewerSettings, initial_status: str = "") -> html.Div:
    """Contrast expression, threshold, colormap and opacity.

    As with :func:`build_roi_controls`, the controls always exist in the
    layout (hidden via CSS when ``not data.has_glm``) so the callback graph
    doesn't need to change shape depending on what was loaded.
    """
    default_contrast = settings.default_contrast or (data.condition_names[0] if data.condition_names else "")
    conditions_hint = ", ".join(data.condition_names) if data.condition_names else ""

    return html.Div(
        className="control-section" + ("" if data.has_glm else " control-section-empty"),
        children=[
            html.Div("Statistical map", className="control-section-title"),
            html.Div(f"Available conditions: {conditions_hint}", className="control-hint"),
            dcc.Input(
                id="contrast-input",
                type="text",
                value=default_contrast,
                placeholder="e.g. FaceLoc - SceneLoc",
                className="text-input",
                debounce=False,
            ),
            html.Button("Apply contrast", id="contrast-apply", className="apply-button", n_clicks=0),
            html.Div(id="contrast-status", className="contrast-status", children=initial_status),
            html.Div("Threshold", className="control-label"),
            dcc.Slider(
                id="stat-threshold-slider",
                min=0, max=8, step=0.1, value=settings.default_threshold,
                marks={0: "0", 2: "2", 4: "4", 6: "6", 8: "8"},
                tooltip={"placement": "bottom", "always_visible": False},
                updatemode="mouseup",
            ),
            html.Div("Colormap", className="control-label"),
            dcc.Dropdown(
                id="stat-cmap-dropdown",
                options=STAT_CMAP_OPTIONS,
                value=settings.stat_cmap,
                clearable=False,
                className="dropdown",
            ),
            html.Div("Opacity", className="control-label"),
            dcc.Slider(
                id="stat-opacity-slider",
                min=0, max=1, step=0.05, value=settings.stat_opacity,
                marks=None, tooltip={"placement": "bottom", "always_visible": False},
                updatemode="mouseup",
            ),
        ],
    )


def build_controls(data: BrainData, settings: ViewerSettings, initial_status: str = "") -> html.Div:
    """The full sidebar."""
    return html.Div(
        id="controls",
        children=[
            build_background_selector(data, settings),
            build_roi_controls(data, settings),
            build_contrast_controls(data, settings, initial_status=initial_status),
        ],
    )