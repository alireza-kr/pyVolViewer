# coding=utf-8
"""Tests for interactive_glm.processing.glm, against a small synthetic
fitted FirstLevelModel (not the full demo dataset - this is deliberately
tiny and fast, since these tests only care about the GLM plumbing, not
realistic results)."""

from __future__ import annotations

import warnings

import numpy as np
import nibabel as nib
import pandas as pd
import pytest

from interactive_glm.processing import glm as glm_mod


@pytest.fixture(scope="module")
def fitted_model():
    from nilearn.glm.first_level import FirstLevelModel

    shape = (10, 10, 10)
    n_scans = 40
    t_r = 1.5
    rng = np.random.default_rng(0)
    data = (rng.normal(size=shape + (n_scans,)) * 5 + 1000).astype(np.float32)
    affine = np.diag([3.0, 3.0, 3.0, 1.0])
    func_img = nib.Nifti1Image(data, affine)
    mask_img = nib.Nifti1Image(np.ones(shape, dtype=np.int8), affine)

    events = pd.DataFrame(
        {"onset": [0, 15, 30, 45], "duration": [2, 2, 2, 2], "trial_type": ["A", "B", "A", "B"]}
    )
    model = FirstLevelModel(t_r=t_r, minimize_memory=True, mask_img=mask_img, noise_model="ols")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(func_img, events=events)
    return model


def test_list_conditions_excludes_nuisance(fitted_model):
    names, mask = glm_mod.list_conditions(fitted_model)
    assert set(names) == {"A", "B"}
    assert mask.sum() == 2
    assert mask.dtype == bool


def test_list_conditions_custom_exclude_pattern(fitted_model):
    import re

    # Excluding "A" too (in addition to the defaults) should leave only "B".
    names, mask = glm_mod.list_conditions(fitted_model, exclude=re.compile("A|drift|constant"))
    assert names == ["B"]


def test_compute_contrast_img_shape(fitted_model):
    img = glm_mod.compute_contrast_img(fitted_model, "A - B", stat_type="t", output_type="z_score")
    assert img.shape == (10, 10, 10)


def test_compute_all_betas_shape(fitted_model):
    _, mask = glm_mod.list_conditions(fitted_model)
    betas_img = glm_mod.compute_all_betas(fitted_model, mask)
    assert betas_img.shape == (10, 10, 10, int(mask.sum()))


def test_voxel_betas_indexing():
    betas = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).astype(float)
    values = glm_mod.voxel_betas(betas, (1, 2, 3))
    assert np.array_equal(values, betas[1, 2, 3, :])


def test_beta_display_range_is_symmetric_and_positive_span():
    betas = np.random.default_rng(0).normal(0, 2, (5, 5, 5, 3))
    vmin, vmax = glm_mod.beta_display_range(betas)
    assert vmin < 0 < vmax
    assert abs(vmin) == pytest.approx(vmax)


def test_beta_display_range_handles_all_zero_input():
    betas = np.zeros((4, 4, 4, 2))
    vmin, vmax = glm_mod.beta_display_range(betas)
    assert vmin < vmax
