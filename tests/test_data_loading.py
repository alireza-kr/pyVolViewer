# coding=utf-8
"""Tests for interactive_glm.data.loading, using small hand-built images
(not the full demo dataset - these focus on the loading/wiring logic)."""

from __future__ import annotations

import numpy as np
import nibabel as nib
import pytest

from interactive_glm.config import RoiSource
from interactive_glm.data.loading import load_brain_data, split_atlas


@pytest.fixture
def small_anat():
    data = np.random.default_rng(0).normal(500, 30, (20, 24, 20)).astype(np.float32)
    return nib.Nifti1Image(data, np.diag([2.0, 2.0, 2.0, 1.0]))


@pytest.fixture
def small_func():
    data = np.random.default_rng(1).normal(1000, 50, (10, 12, 10, 15)).astype(np.float32)
    return nib.Nifti1Image(data, np.diag([4.0, 4.0, 4.0, 1.0]))


def test_load_brain_data_requires_anat_or_func():
    with pytest.raises(ValueError):
        load_brain_data()


def test_load_brain_data_anat_only(small_anat):
    data = load_brain_data(anat=small_anat)
    assert set(data.backgrounds) == {"anat"}
    assert data.overlays == {}
    assert not data.has_glm


def test_load_brain_data_anat_and_func_both_become_backgrounds(small_anat, small_func):
    data = load_brain_data(anat=small_anat, func=small_func)
    assert set(data.backgrounds) == {"anat", "func"}
    # both backgrounds must share the reference grid, regardless of native resolution
    assert data.backgrounds["anat"].data.shape == data.shape
    assert data.backgrounds["func"].data.shape == data.shape


def test_load_brain_data_func_only_uses_func_as_anchor(small_func):
    data = load_brain_data(func=small_func)
    assert set(data.backgrounds) == {"func"}


def test_load_rois_auto_assigns_distinct_colors_and_keys(small_anat):
    roi_a = nib.Nifti1Image(np.zeros((20, 24, 20), np.float32), small_anat.affine)
    roi_b = nib.Nifti1Image(np.zeros((20, 24, 20), np.float32), small_anat.affine)
    rois = [RoiSource(img=roi_a, label="Region One"), RoiSource(img=roi_b, label="Region Two")]

    data = load_brain_data(anat=small_anat, rois=rois)
    assert len(data.overlays) == 2
    layers = list(data.overlays.values())
    assert layers[0].color != layers[1].color
    assert layers[0].key != layers[1].key
    assert {layer.label for layer in data.overlays.values()} == {"Region One", "Region Two"}


def test_load_rois_respects_explicit_color(small_anat):
    roi = nib.Nifti1Image(np.zeros((20, 24, 20), np.float32), small_anat.affine)
    data = load_brain_data(anat=small_anat, rois=[RoiSource(img=roi, label="X", color="#123456")])
    assert list(data.overlays.values())[0].color == "#123456"


def test_split_atlas_skips_background_label_zero(small_anat):
    atlas_data = np.zeros((20, 24, 20), dtype=np.int16)
    atlas_data[0:5, 0:5, 0:5] = 1
    atlas_data[10:15, 10:15, 10:15] = 2
    atlas_img = nib.Nifti1Image(atlas_data, small_anat.affine)

    rois = split_atlas(atlas_img, labels={0: "background", 1: "Region A", 2: "Region B"})
    assert len(rois) == 2
    assert {r.label for r in rois} == {"Region A", "Region B"}


def test_split_atlas_regions_are_disjoint_binary_masks(small_anat):
    atlas_data = np.zeros((20, 24, 20), dtype=np.int16)
    atlas_data[0:5, 0:5, 0:5] = 1
    atlas_data[10:15, 10:15, 10:15] = 2
    atlas_img = nib.Nifti1Image(atlas_data, small_anat.affine)

    rois = split_atlas(atlas_img, labels={1: "A", 2: "B"})
    mask_a = np.asarray(rois[0].img.dataobj)
    mask_b = np.asarray(rois[1].img.dataobj)
    assert not np.any((mask_a > 0) & (mask_b > 0))
    assert mask_a.sum() == 5 * 5 * 5


def test_glm_loading_end_to_end(small_func):
    import warnings

    import pandas as pd
    from nilearn.glm.first_level import FirstLevelModel

    events = pd.DataFrame({"onset": [0, 6], "duration": [2, 2], "trial_type": ["A", "B"]})
    mask_img = nib.Nifti1Image(np.ones(small_func.shape[:3], dtype=np.int8), small_func.affine)
    model = FirstLevelModel(t_r=1.0, minimize_memory=True, mask_img=mask_img, noise_model="ols")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(small_func, events=events)

    data = load_brain_data(func=small_func, glm_model=model, events=events)
    assert data.has_glm
    assert set(data.condition_names) == {"A", "B"}
    assert data.betas.shape == data.shape + (2,)
    assert data.beta_range is not None
    assert data.n_scans == small_func.shape[3]
    assert data.t_r == 1.0
    assert list(data.events.columns) == ["onset", "duration", "trial_type"]
