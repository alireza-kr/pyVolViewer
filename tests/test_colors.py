# coding=utf-8
"""Tests for interactive_glm.processing.colors."""

from __future__ import annotations

import numpy as np

from interactive_glm.processing import colors


def test_apply_colormap_shape_and_dtype():
    values = np.linspace(-5, 5, 100).reshape(10, 10)
    rgb = colors.apply_colormap(values, "cold_hot", vmin=-5, vmax=5)
    assert rgb.shape == (10, 10, 3)
    assert rgb.dtype == np.uint8


def test_apply_colormap_constant_range_does_not_crash():
    values = np.zeros((5, 5))
    rgb = colors.apply_colormap(values, "gray", vmin=0, vmax=0)
    assert rgb.shape == (5, 5, 3)


def test_hex_to_rgb():
    assert colors.hex_to_rgb("#ff0000") == (255, 0, 0)
    assert colors.hex_to_rgb("00ff00") == (0, 255, 0)


def test_distinct_palette_length_and_no_duplicates_within_hand_picked_range():
    palette = colors.distinct_palette(6)
    assert len(palette) == 6
    assert len(set(palette)) == 6


def test_distinct_palette_beyond_hand_picked_still_returns_unique_colors():
    n = 25
    palette = colors.distinct_palette(n)
    assert len(palette) == n
    assert len(set(palette)) == n, "generated fallback colors should still be distinct"
    assert all(c.startswith("#") and len(c) == 7 for c in palette)
