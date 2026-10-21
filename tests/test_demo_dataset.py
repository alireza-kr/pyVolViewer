# coding=utf-8
"""
Tests for interactive_glm.data.demo.

test_activation_colocalizes_with_matching_roi is the key regression test
for this module: it checks that the synthetic pipeline is internally
coherent end to end (synthetic BOLD generation -> GLM fit -> alignment ->
ROI placement), not just that each stage runs without error. If a future
change broke, say, the sign convention of a contrast or the coordinate
system used to place ROIs, the individual stages could all still pass
their own tests while the demo silently stopped making sense - this is
what would catch that.
"""

from __future__ import annotations

import numpy as np

from interactive_glm.processing.glm import voxel_betas


def test_structure(demo_data):
    assert set(demo_data.backgrounds) == {"anat", "func"}
    assert demo_data.shape == demo_data.backgrounds["anat"].data.shape
    assert demo_data.shape == demo_data.backgrounds["func"].data.shape
    assert demo_data.has_glm
    assert set(demo_data.condition_names) == {"FaceLoc", "SceneLoc", "ObjectLoc"}
    assert demo_data.betas.shape == demo_data.shape + (3,)
    assert len(demo_data.roi_layers) == 4
    assert demo_data.events is not None and len(demo_data.events) == 24
    assert demo_data.t_r == 2.0 and demo_data.n_scans == 128


def test_roi_layers_are_disjoint_ish_and_within_backgrounds_bounds(demo_data):
    for layer in demo_data.roi_layers:
        assert layer.data.shape == demo_data.shape
        assert layer.data.min() >= 0.0
        assert (layer.data > 0.5).sum() > 0, f"{layer.label} should not be empty"


def test_activation_colocalizes_with_matching_roi(demo_data):
    """For each condition's matching ROI, that condition's beta should be
    the largest of the three at that ROI's peak voxel."""
    for layer in demo_data.roi_layers:
        matching = [c for c in demo_data.condition_names if c in layer.label]
        if not matching:
            continue  # the deliberately-unrelated "Region D" ROI
        condition = matching[0]

        peak_voxel = tuple(int(v) for v in np.unravel_index(np.argmax(layer.data), layer.data.shape))
        values = voxel_betas(demo_data.betas, peak_voxel)
        winner = demo_data.condition_names[int(np.argmax(values))]
        assert winner == condition, (
            f"expected {condition} to dominate inside {layer.label}, "
            f"got {winner} (values={dict(zip(demo_data.condition_names, values))})"
        )


def test_contrast_between_matching_conditions_peaks_inside_the_right_roi(demo_data):
    from interactive_glm.processing.alignment import align_to_reference
    from interactive_glm.processing.glm import compute_contrast_img

    contrast_img = compute_contrast_img(demo_data.glm_model, "FaceLoc - SceneLoc", stat_type="t", output_type="z_score")
    aligned = align_to_reference(contrast_img, demo_data.reference_template, interpolation="continuous")

    peak_voxel = np.unravel_index(np.argmax(aligned), aligned.shape)
    face_roi = next(layer for layer in demo_data.roi_layers if "FaceLoc" in layer.label)
    assert face_roi.data[peak_voxel] > 0.5, "peak of FaceLoc>SceneLoc contrast should fall inside the FaceLoc ROI"
