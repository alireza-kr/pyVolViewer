# coding=utf-8
"""
Tests for interactive_glm.processing.alignment.

test_marker_lands_in_expected_screen_quadrant is the most important test
in this whole project: it's an *empirical* check that a point at a known
Right/Anterior/Superior location ends up in the correct on-screen position
for each of the three views, starting from a deliberately non-canonical
native affine. Getting this wrong would silently mislabel left/right or
flip a view upside down - exactly the kind of bug that's invisible until
someone notices their "left" activation is drawn on the right.
"""

from __future__ import annotations

import numpy as np
import nibabel as nib
import pytest

from interactive_glm.processing import alignment as al


@pytest.fixture
def weird_affine_volume():
    """A volume in a deliberately non-RAS native orientation, with a bright
    marker placed at a known Right/Anterior/Superior world coordinate."""
    shape = (40, 50, 60)
    vol = np.zeros(shape, dtype=np.float32)
    # axis0 <-> -X, axis1 <-> Z, axis2 <-> -Y: nothing lines up with RAS by luck.
    affine = np.array(
        [[-1, 0, 0, 20], [0, 0, -1, 30], [0, 1, 0, -25], [0, 0, 0, 1]], dtype=float
    )
    target_world = np.array([15.0, 15.0, 15.0, 1.0])  # Right, Anterior, Superior
    ijk = np.linalg.inv(affine) @ target_world
    i0, j0, k0 = (int(round(v)) for v in ijk[:3])
    vol[i0 - 2 : i0 + 2, j0 - 2 : j0 + 2, k0 - 2 : k0 + 2] = 100.0
    return nib.Nifti1Image(vol, affine)


def test_build_reference_grid_is_isotropic_and_ras_positive(weird_affine_volume):
    reference = al.build_reference_grid(weird_affine_volume)
    diag = np.diag(reference.affine)[:3]
    assert np.allclose(diag, diag[0]), "reference grid should be isotropic"
    assert np.all(diag > 0), "reference grid should be RAS+ (positive diagonal)"
    off_diag = reference.affine[:3, :3] - np.diag(diag)
    assert np.allclose(off_diag, 0), "reference grid should be axis-aligned"


def test_marker_lands_in_expected_screen_quadrant(weird_affine_volume):
    reference = al.build_reference_grid(weird_affine_volume, voxel_size=1.0)
    volume = al.align_to_reference(weird_affine_volume, reference, interpolation="nearest")

    marker = np.argwhere(volume > 50)
    assert marker.size > 0, "marker should survive canonicalization"
    i_mid, j_mid, k_mid = (int(v) for v in marker.mean(axis=0))
    shape = volume.shape

    # Axial (fix k=S): marker should be near the TOP (low row = Anterior) and RIGHT (high col = R).
    axial = al.to_display(al.get_slice_2d(volume, al.AXIAL, k_mid))
    r, c = np.argwhere(axial > 50)[:, 0], np.argwhere(axial > 50)[:, 1]
    assert r.max() < axial.shape[0] * 0.5, "axial marker should be in the top half (Anterior)"
    assert c.min() > axial.shape[1] * 0.5, "axial marker should be in the right half (Right)"

    # Coronal (fix j=A): TOP (Superior) and RIGHT (R).
    coronal = al.to_display(al.get_slice_2d(volume, al.CORONAL, j_mid))
    r, c = np.argwhere(coronal > 50)[:, 0], np.argwhere(coronal > 50)[:, 1]
    assert r.max() < coronal.shape[0] * 0.5, "coronal marker should be in the top half (Superior)"
    assert c.min() > coronal.shape[1] * 0.5, "coronal marker should be in the right half (Right)"

    # Sagittal (fix i=R): TOP (Superior) and RIGHT (Anterior, by this module's convention).
    sagittal = al.to_display(al.get_slice_2d(volume, al.SAGITTAL, i_mid))
    r, c = np.argwhere(sagittal > 50)[:, 0], np.argwhere(sagittal > 50)[:, 1]
    assert r.max() < sagittal.shape[0] * 0.5, "sagittal marker should be in the top half (Superior)"
    assert c.min() > sagittal.shape[1] * 0.5, "sagittal marker should be in the right half (Anterior)"


def test_align_to_reference_shapes_match_across_different_native_grids():
    anat = nib.Nifti1Image(
        np.random.default_rng(0).normal(500, 20, (40, 45, 40)).astype(np.float32),
        np.diag([1.5, 1.5, 1.5, 1.0]),
    )
    other = nib.Nifti1Image(
        np.random.default_rng(1).normal(1000, 50, (12, 14, 12)).astype(np.float32),
        np.diag([4.0, 4.0, 4.0, 1.0]),
    )
    reference = al.build_reference_grid(anat)
    aligned = al.align_to_reference(other, reference)
    assert aligned.shape == reference.shape[:3]


def test_voxel_mm_roundtrip():
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = [-30, -35, -30]
    for ijk in [(0, 0, 0), (10, 20, 15), (29, 34, 29)]:
        mm = al.voxel_to_mm(ijk, affine)
        back = al.mm_to_voxel(mm, affine, shape=(30, 35, 30))
        assert back == ijk


@pytest.mark.parametrize("axis", [al.SAGITTAL, al.CORONAL, al.AXIAL])
def test_panel_rc_roundtrip(axis, rng):
    shape = (37, 41, 29)
    for _ in range(100):
        ijk = tuple(int(rng.integers(0, s)) for s in shape)
        row, col = al.voxel_to_panel_rc(ijk, shape, axis)
        rows, cols = al.display_shape(shape, axis)
        assert 0 <= row < rows
        assert 0 <= col < cols
        assert al.panel_rc_to_voxel(row, col, ijk, shape, axis) == ijk


@pytest.mark.parametrize("axis", [al.SAGITTAL, al.CORONAL, al.AXIAL])
def test_to_display_shape_matches_display_shape(axis):
    shape = (33, 44, 55)
    volume = np.zeros(shape, dtype=np.float32)
    sl = al.get_slice_2d(volume, axis, shape[axis] // 2)
    assert al.to_display(sl).shape == al.display_shape(shape, axis)


@pytest.mark.parametrize("axis", [al.SAGITTAL, al.CORONAL, al.AXIAL])
def test_panel_direction_labels_has_all_four_edges(axis):
    labels = al.panel_direction_labels(axis)
    assert set(labels) == {"top", "bottom", "left", "right"}
    assert all(isinstance(v, str) and len(v) == 1 for v in labels.values())
