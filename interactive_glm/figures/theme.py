# coding=utf-8
"""
One place for the color palette and typography used by every figure
builder (and mirrored in ``assets/style.css`` for the surrounding page
chrome), so the plots and the page they sit in always look like one
coherent app instead of a chart library's default theme dropped onto a
custom page.
"""

from __future__ import annotations

# Page / panel surfaces (also used in assets/style.css - keep in sync).
BG_PAGE = "#0b0d12"
BG_PANEL = "#12151c"
BG_PANEL_ALT = "#171b24"
BORDER = "#242938"

# Text.
TEXT_PRIMARY = "#e8e9ed"
TEXT_MUTED = "#8b91a3"

# Accent / interactive elements.
ACCENT = "#5eead4"  # crosshair, active controls
ACCENT_SOFT = "rgba(94, 234, 212, 0.35)"

FONT_FAMILY = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, Roboto, "
    "'Helvetica Neue', Arial, sans-serif"
)

# Qualitative palette for GLM condition bars - distinct from the ROI
# palette in processing.colors so ROI colors and condition colors never
# get confused if both are visible at once.
CONDITION_PALETTE = [
    "#5eead4",  # teal
    "#f9a8d4",  # pink
    "#fcd34d",  # amber
    "#93c5fd",  # blue
    "#c4b5fd",  # violet
    "#86efac",  # green
    "#fca5a5",  # red
    "#fdba74",  # orange
]


def condition_colors(n: int) -> list[str]:
    """``n`` colors from :data:`CONDITION_PALETTE`, cycling if there are more conditions than colors."""
    return [CONDITION_PALETTE[i % len(CONDITION_PALETTE)] for i in range(n)]


def base_layout_kwargs(**overrides) -> dict:
    """Shared ``go.Figure.update_layout`` kwargs for a transparent, dark-themed plot."""
    layout = dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, color=TEXT_PRIMARY, size=13),
        margin=dict(l=8, r=8, t=28, b=8),
        showlegend=False,
        hoverlabel=dict(bgcolor=BG_PANEL_ALT, font=dict(color=TEXT_PRIMARY, family=FONT_FAMILY)),
    )
    layout.update(overrides)
    return layout
