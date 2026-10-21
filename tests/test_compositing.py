# coding=utf-8
"""Tests for interactive_glm.processing.compositing."""

from __future__ import annotations

import numpy as np

from interactive_glm.config import OverlayLayer
from interactive_glm.processing import alignment as al
from interactive_glm.processing import compositing as comp


def _flat_background(shape, value=400.0):
    return np.full(shape, value, dtype=np.float32)


def test_compose_panel_shape_and_dtype_with_no_overlays():
    shape = (30, 40, 20)
    bg = _flat_background(shape)
    rgb = comp.compose_panel(bg, 0, 800, "gray", al.AXIAL, 10, overlays=[])
    rows, cols = al.display_shape(shape, al.AXIAL)
    assert rgb.shape == (rows, cols, 3)
    assert rgb.dtype == np.uint8


def test_roi_layer_full_opacity_interior_is_exact_color():
    shape = (30, 40, 20)
    bg = _flat_background(shape)
    roi_data = np.zeros(shape, dtype=np.float32)
    roi_data[10:15, 10:15, 10:15] = 1.0  # interior membership == 1.0 exactly
    layer = OverlayLayer(key="roi", label="ROI", kind="roi", data=roi_data, color="#00ff00", opacity=1.0, threshold=0.5)

    rgb = comp.compose_panel(bg, 0, 800, "gray", al.AXIAL, 12, overlays=[layer])
    row, col = al.voxel_to_panel_rc((12, 12, 12), shape, al.AXIAL)
    assert tuple(rgb[row, col]) == (0, 255, 0)


def test_invisible_layer_is_skipped():
    shape = (20, 20, 20)
    bg = _flat_background(shape)
    roi_data = np.ones(shape, dtype=np.float32)
    layer = OverlayLayer(key="roi", label="ROI", kind="roi", data=roi_data, color="#ff0000", opacity=1.0, visible=False)

    rgb = comp.compose_panel(bg, 0, 800, "gray", al.AXIAL, 10, overlays=[layer])
    # every pixel should just be the (grayscale) background - no red anywhere.
    assert np.all(rgb[..., 0] == rgb[..., 1]) and np.all(rgb[..., 1] == rgb[..., 2])


def test_stat_map_below_threshold_is_fully_transparent():
    shape = (20, 20, 20)
    bg = _flat_background(shape, value=100.0)
    stat_data = np.full(shape, 2.0, dtype=np.float32)  # everywhere below threshold
    layer = OverlayLayer(
        key="stat_map", label="A", kind="stat_map", data=stat_data, cmap="bwr",
        opacity=1.0, threshold=3.1, vmin=-5, vmax=5,
    )
    rgb_with = comp.compose_panel(bg, 0, 200, "gray", al.AXIAL, 10, overlays=[layer])
    rgb_without = comp.compose_panel(bg, 0, 200, "gray", al.AXIAL, 10, overlays=[])
    assert np.array_equal(rgb_with, rgb_without)


def test_stat_map_above_threshold_is_colored():
    shape = (20, 20, 20)
    bg = _flat_background(shape, value=100.0)
    stat_data = np.zeros(shape, dtype=np.float32)
    stat_data[8:12, 8:12, 8:12] = 5.0
    layer = OverlayLayer(
        key="stat_map", label="A", kind="stat_map", data=stat_data, cmap="bwr",
        opacity=1.0, threshold=3.1, vmin=-5, vmax=5,
    )
    rgb = comp.compose_panel(bg, 0, 200, "gray", al.AXIAL, 10, overlays=[layer])
    row, col = al.voxel_to_panel_rc((10, 10, 10), shape, al.AXIAL)
    pixel = rgb[row, col]
    assert not (pixel[0] == pixel[1] == pixel[2]), "a suprathreshold positive stat voxel should not be gray"


def test_layers_painted_in_order_last_wins_where_both_fully_opaque():
    shape = (10, 10, 10)
    bg = _flat_background(shape)
    layer_a = OverlayLayer(key="a", label="A", kind="roi", data=np.ones(shape, np.float32), color="#ff0000", opacity=1.0)
    layer_b = OverlayLayer(key="b", label="B", kind="roi", data=np.ones(shape, np.float32), color="#0000ff", opacity=1.0)

    rgb_ab = comp.compose_panel(bg, 0, 800, "gray", al.AXIAL, 5, overlays=[layer_a, layer_b])
    rgb_ba = comp.compose_panel(bg, 0, 800, "gray", al.AXIAL, 5, overlays=[layer_b, layer_a])

    assert tuple(rgb_ab[0, 0]) == (0, 0, 255)  # b painted last -> wins
    assert tuple(rgb_ba[0, 0]) == (255, 0, 0)  # a painted last -> wins
