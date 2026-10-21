# coding=utf-8
"""
First-level GLM helpers: turning a fitted
``nilearn.glm.first_level.FirstLevelModel`` into the three things the app
needs - a statistical map for an arbitrary contrast, a 4D volume of
per-condition effect sizes (betas) for the bar chart, and the list of
"real" experimental conditions with nuisance regressors filtered out.

Nothing here talks to Dash, Plotly, or the reference grid - callers are
responsible for resampling the results with
:mod:`interactive_glm.processing.alignment` if needed. That keeps this
module usable on its own, e.g. from a notebook, and makes it trivial to
test without spinning up the whole app.
"""

from __future__ import annotations

import re

import numpy as np
from nibabel.nifti1 import Nifti1Image

#: Design-matrix columns that are never experimental conditions of interest.
#: Extend this if your design matrices use different nuisance-regressor
#: naming conventions.
NUISANCE_PATTERN = re.compile(
    r"trans|rot|drift|constant|comp_cor|csf|white_matter", re.IGNORECASE
)


def list_conditions(model, exclude: re.Pattern = NUISANCE_PATTERN) -> tuple[list, np.ndarray]:
    """Experimental condition names from a fitted model's design matrix.

    Parameters
    ----------
    model : nilearn FirstLevelModel
        Must already be fitted (``model.design_matrices_`` populated).
    exclude : re.Pattern
        Columns whose name matches this pattern are treated as nuisance
        regressors and left out.

    Returns
    -------
    condition_names : list of str
        The design-matrix columns considered real conditions, in their
        original order.
    condition_mask : numpy.ndarray of bool
        Boolean mask into the *full* set of design-matrix columns
        (``model.design_matrices_[0].columns``), True for kept conditions.
        Use this to index into anything shaped like the full design matrix,
        e.g. per-regressor beta arrays.
    """
    all_columns = model.design_matrices_[0].columns
    condition_mask = ~all_columns.str.contains(exclude, regex=True)
    condition_names = list(all_columns[condition_mask])
    return condition_names, np.asarray(condition_mask)


def compute_contrast_img(
    model, contrast_def: str, stat_type: str = "t", output_type: str = "z_score"
) -> Nifti1Image:
    """Compute one contrast and return it as a NIfTI image, in the model's native grid.

    A thin, documented wrapper around ``model.compute_contrast`` - kept as
    its own function so callers (and tests) don't need to know nilearn's
    exact call signature, and so we have one place to extend if we later
    want to support e.g. F-contrasts or contrast caching.

    Parameters
    ----------
    model : nilearn FirstLevelModel
        Must already be fitted.
    contrast_def : str
        Any contrast expression nilearn's ``compute_contrast`` accepts,
        e.g. a single condition name or an algebraic expression like
        ``"Face - House"``.
    stat_type : {"t", "F"}
        Type of statistic to compute.
    output_type : str
        Passed straight through to ``compute_contrast`` (e.g.
        ``"z_score"``, ``"stat"``, ``"effect_size"``).

    Returns
    -------
    nibabel.Nifti1Image
    """
    return model.compute_contrast(contrast_def, stat_type=stat_type, output_type=output_type)


def compute_all_betas(model, condition_mask: np.ndarray) -> Nifti1Image:
    """Per-condition effect sizes (betas) for every voxel, as a 4D image.

    Reproduces what ``model.compute_contrast`` does per-condition, but for
    *all* kept conditions at once and without recomputing the regression -
    it reads the already-fitted per-cluster results directly, which is
    both faster and is what lets the app show a full condition-by-condition
    bar chart at any voxel without one ``compute_contrast`` call per bar.

    Parameters
    ----------
    model : nilearn FirstLevelModel
        Must already be fitted with ``minimize_memory=False`` *or* default
        settings - either way this only needs ``labels_``, ``results_`` and
        ``masker_``, which nilearn always keeps.
    condition_mask : numpy.ndarray of bool
        As returned by :func:`list_conditions`; selects which design-matrix
        columns to keep, in order.

    Returns
    -------
    nibabel.Nifti1Image
        4D image, shape ``(nx, ny, nz, n_conditions)``, in the model's
        native (masker) grid.
    """
    labels = model.labels_[0]
    results = model.results_[0]
    n_conditions = int(condition_mask.sum())

    effects = np.zeros((n_conditions, labels.size), dtype=np.float32)
    for cluster_label, result in results.items():
        effects[:, labels == cluster_label] = result.theta[condition_mask]

    return model.masker_.inverse_transform(effects)


def voxel_betas(betas: np.ndarray, ijk: tuple[int, int, int]) -> np.ndarray:
    """Per-condition beta values at one voxel of an already-aligned 4D betas array."""
    i, j, k = ijk
    return betas[i, j, k, :]


def beta_display_range(betas: np.ndarray, percentile: float = 99.5) -> tuple[float, float]:
    """A fixed, symmetric-around-zero y-axis range for the beta bar chart.

    Computed once from the whole 4D betas volume (not per-voxel), so the
    chart's scale stays stable as the user clicks between voxels instead
    of rescaling - and re-anchoring the reader's sense of magnitude - on
    every click.
    """
    finite = betas[np.isfinite(betas)]
    nonzero = finite[finite != 0]
    sample = nonzero if nonzero.size > 0 else finite
    if sample.size == 0:
        return -1.0, 1.0
    bound = float(np.percentile(np.abs(sample), percentile))
    bound = max(bound, 1e-6)
    return -bound * 1.1, bound * 1.1
