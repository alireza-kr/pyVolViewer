# coding=utf-8
"""
The measured BOLD time series at a voxel, alongside each condition's own
contribution to the GLM's fitted signal - "does the model track the data
here, and which condition is driving it?"

Design note: kept at the func image's own native resolution
-------------------------------------------------------------
``BrainData.func_data`` deliberately stays at whatever resolution the
functional run was acquired at, rather than being resampled onto the
(typically much finer) anatomical reference grid the way backgrounds and
ROIs are. A typical 2mm-anatomical / 3-4mm-functional setup would turn a
manageable ~70MB functional array into a 500MB+ one if upsampled to
anatomical resolution for no real benefit - the extra "resolution" would
be pure interpolation, not new information. Instead, a clicked voxel's
position is converted from the reference grid to millimetres and then to
the nearest native functional voxel, on demand, only when this panel needs
it.

Design note: one fresh OLS fit, decomposed per condition
-----------------------------------------------------------
The statistical maps elsewhere in the app come from the original fitted
model (whichever noise model it used - often AR(1)-whitened). Reconstructing
*that exact* per-voxel fit here would mean re-deriving nilearn's internal
per-voxel cluster/theta indexing (voxels are grouped and solved together by
shared noise properties, not stored one-by-one) - fragile to depend on
across nilearn versions and masker types for what is fundamentally a
"does this look reasonable" visualization. A fresh ordinary-least-squares
fit of the real signal against the model's own design matrix, computed
just for the clicked voxel, is standard practice for this kind of plot.

Decomposing that single fit per condition is just linear algebra, not a
second fit: with fitted coefficients ``beta`` and design matrix columns
``X``, the *whole* fitted signal is ``X @ beta`` - the sum, over every
column, of ``X[:, j] * beta[j]``. Each condition's own line adds that
condition's own term (its column times its own coefficient) *on top of*
the shared baseline/drift terms (the constant and any drift columns,
summed the same way). That shared part matters: without it, each
condition's line would show only its own modulation above baseline -
typically a handful of signal units - which is mathematically the
"contribution" but reads as flatly zero next to a baseline that can be in
the hundreds or thousands, and isn't comparable at a glance to the
measured signal. Including it means every condition's line sits on the
same absolute scale as "Measured", differing from the others only during
its own active periods - which is what makes "does this condition's fit
track the data" a plot you can actually read.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from interactive_glm.processing.alignment import mm_to_voxel, voxel_to_mm


def voxel_timeseries_and_fit(
    model,
    func_data: np.ndarray,
    func_affine: np.ndarray,
    reference_ijk: tuple[int, int, int],
    reference_affine: np.ndarray,
    condition_names: list,
) -> Optional[tuple[np.ndarray, np.ndarray, dict]]:
    """Measured signal + each condition's fitted contribution, at one voxel.

    Parameters
    ----------
    model : nilearn FirstLevelModel
        Fitted model; only ``design_matrices_`` and ``t_r`` are used.
    func_data : numpy.ndarray
        4D array, native functional resolution (``BrainData.func_data``).
    func_affine : numpy.ndarray
        The functional image's own affine (``BrainData.func_affine``) -
        generally *not* the same grid as ``reference_affine``.
    reference_ijk : tuple of int
        The clicked voxel, in the shared reference grid's coordinates.
    reference_affine : numpy.ndarray
        The reference grid's affine (``BrainData.affine``).
    condition_names : list of str
        Which design-matrix columns to break the fit out by (typically
        ``BrainData.condition_names`` - the same conditions shown in the
        beta bar chart and the design timeline, in the same order, so
        colors stay consistent - see :mod:`interactive_glm.figures.timeseries`).

    Returns
    -------
    (frame_times, measured, condition_fits) or None
        ``frame_times`` and ``measured`` are 1D arrays of length
        ``n_scans``. ``condition_fits`` is a dict mapping each name in
        ``condition_names`` that is actually a design-matrix column to its
        own 1D array - that condition's contribution *plus* the shared
        baseline/drift terms (see the module docstring), so it's on the
        same scale as ``measured`` rather than centered on zero. Returns
        None if the corresponding native-resolution voxel has no signal at
        all (e.g. it falls outside the functional field of view / brain
        mask - a real possibility since the two grids can differ in
        extent, not just resolution).
    """
    mm = voxel_to_mm(reference_ijk, reference_affine)
    i, j, k = mm_to_voxel(mm, func_affine, func_data.shape[:3])

    measured = np.asarray(func_data[i, j, k, :], dtype=np.float64)
    if not np.any(measured):
        return None

    design_matrix = model.design_matrices_[0]
    columns = list(design_matrix.columns)
    design_values = design_matrix.to_numpy() if hasattr(design_matrix, "to_numpy") else np.asarray(design_matrix)

    beta, *_ = np.linalg.lstsq(design_values, measured, rcond=None)

    condition_cols = {name: columns.index(name) for name in condition_names if name in columns}
    nuisance_cols = [c for c in range(len(columns)) if c not in condition_cols.values()]
    # Shared by every condition's line below: the constant/intercept and any
    # drift terms, i.e. everything in the design matrix that is *not* one of
    # the conditions being broken out. Without this, each condition's line
    # would show only its own modulation above baseline (tiny - a handful of
    # signal units - against a baseline that can be in the hundreds or
    # thousands), which is mathematically correct but not comparable at a
    # glance to "Measured". Adding it back means every condition's line sits
    # on the same absolute scale as the measured signal and only diverges
    # from the others during its own active periods.
    baseline_and_drift = design_values[:, nuisance_cols] @ beta[nuisance_cols] if nuisance_cols else 0.0

    condition_fits = {}
    for name, col in condition_cols.items():
        condition_fits[name] = baseline_and_drift + design_values[:, col] * beta[col]

    t_r = getattr(model, "t_r", None) or 1.0
    frame_times = np.arange(measured.shape[0]) * t_r

    return frame_times, measured, condition_fits