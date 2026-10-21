# coding=utf-8
"""
A fully offline, synthetic dataset for trying the app (``python run.py --demo``)
or writing tests without any real imaging data.

The anatomical background is the real MNI152 template that ships with
nilearn (no download needed). Everything functional - the BOLD time
series, the events, the GLM, the ROI locations - is synthetically
generated: a handful of "hotspot" regions inside the real brain mask, each
driven by an HRF-convolved response to one condition, plus independent
Gaussian noise, fit with an ordinary first-level GLM. This is enough to
exercise every panel in the app (background switch, ROI toggling, contrast
selection, thresholding, the beta bar chart, the design timeline) without
requiring any of your own files.

Nothing here is intended to represent real neuroscience - amplitudes and
locations are chosen to look clear and clickable, not to be biologically
plausible.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from nibabel.nifti1 import Nifti1Image
from nilearn.datasets import load_mni152_brain_mask, load_mni152_template
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level.hemodynamic_models import compute_regressor
from nilearn.image import resample_img, resample_to_img
from scipy.ndimage import gaussian_filter

from interactive_glm.config import BrainData, RoiSource
from interactive_glm.data.loading import load_brain_data
from interactive_glm.processing.alignment import mm_to_voxel, voxel_to_mm

#: (condition name, hotspot MNI mm coordinate, ROI display label)
_CONDITIONS = [
    ("FaceLoc", (-45, -55, -12), "Region A"),
    ("SceneLoc", (30, -75, 8), "Region B"),
    ("ObjectLoc", (0, 20, 28), "Region C"),
]
_EXTRA_ROI = ((-38, -22, 55), "Region D (no matching contrast)")


@dataclass
class DemoRunInfo:
    """A few facts about the generated run, handy for the CLI banner / docs."""

    t_r: float
    n_scans: int
    conditions: list


def build_demo_dataset(
    func_voxel_size: float = 3.0,
    n_scans: int = 128,
    t_r: float = 2.0,
    reps_per_condition: int = 8,
    seed: int = 0,
) -> BrainData:
    """Build a ready-to-use, fully offline demo :class:`~interactive_glm.config.BrainData`.

    Takes a few seconds (mostly spent loading and resampling the real MNI
    template) - there is no network access involved anywhere.

    Parameters
    ----------
    func_voxel_size : float
        Isotropic resolution (mm) of the synthetic functional run. Kept
        coarser than the anatomical template, as is typical for real data.
    n_scans : int
        Number of synthetic functional volumes.
    t_r : float
        Repetition time in seconds.
    reps_per_condition : int
        Number of events generated per condition.
    seed : int
        Seed for the synthetic noise / event timing, for reproducibility.

    Returns
    -------
    BrainData
    """
    rng = np.random.default_rng(seed)

    anat_img = load_mni152_template(resolution=2)
    brain_mask_img = load_mni152_brain_mask(resolution=2)

    func_grid = resample_img(
        anat_img,
        target_affine=np.diag([func_voxel_size] * 3),
        target_shape=None,
        interpolation="continuous",
        force_resample=True,
    )
    func_mask_img = resample_to_img(brain_mask_img, func_grid, interpolation="nearest")
    func_mask = np.asarray(func_mask_img.dataobj) > 0.5

    hotspots_mm = {
        name: _nearest_in_mask_mm(point, brain_mask_img) for name, point, _ in _CONDITIONS
    }
    conditions = [name for name, _, _ in _CONDITIONS]

    events = _generate_events(conditions, n_scans, t_r, reps_per_condition, rng)
    func_img = _generate_synthetic_bold(func_grid, func_mask, hotspots_mm, events, t_r, n_scans, rng)

    model = FirstLevelModel(
        t_r=t_r, hrf_model="glover", mask_img=func_mask_img, minimize_memory=True, noise_model="ols"
    )
    with warnings.catch_warnings():
        # nilearn is chatty (INFO-level progress) during GLM fitting; nothing
        # here indicates a problem.
        warnings.simplefilter("ignore")
        model.fit(func_img, events=events)

    rois = [
        RoiSource(img=_sphere_mask(anat_img, hotspots_mm[name], radius_mm=16.0), label=f"{roi_label} ({name})")
        for name, _, roi_label in _CONDITIONS
    ]
    extra_point_mm = _nearest_in_mask_mm(_EXTRA_ROI[0], brain_mask_img)
    rois.append(RoiSource(img=_sphere_mask(anat_img, extra_point_mm, radius_mm=14.0), label=_EXTRA_ROI[1]))

    return load_brain_data(
        anat=anat_img,
        func=func_img,
        glm_model=model,
        events=events,
        rois=rois,
    )


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------


def _nearest_in_mask_mm(center_mm: tuple, mask_img: Nifti1Image) -> tuple:
    """Snap ``center_mm`` to the nearest True voxel of ``mask_img``, if needed."""
    mask_data = np.asarray(mask_img.dataobj) > 0.5
    ijk = mm_to_voxel(center_mm, mask_img.affine, mask_img.shape[:3])
    if mask_data[ijk]:
        return center_mm
    mask_voxels = np.argwhere(mask_data)
    dists = np.sum((mask_voxels - np.array(ijk)) ** 2, axis=1)
    nearest = mask_voxels[np.argmin(dists)]
    return voxel_to_mm(tuple(nearest), mask_img.affine)


def _sphere_mask(reference_img: Nifti1Image, center_mm: tuple, radius_mm: float) -> Nifti1Image:
    """A binary sphere of the given radius (mm), on ``reference_img``'s grid."""
    shape = reference_img.shape[:3]
    affine = reference_img.affine
    ii, jj, kk = np.meshgrid(*(np.arange(s) for s in shape), indexing="ij")
    world = np.stack([ii, jj, kk, np.ones_like(ii)], axis=-1).astype(np.float64) @ affine.T
    cx, cy, cz = center_mm
    dist = np.sqrt((world[..., 0] - cx) ** 2 + (world[..., 1] - cy) ** 2 + (world[..., 2] - cz) ** 2)
    mask = (dist <= radius_mm).astype(np.float32)
    return Nifti1Image(mask, affine)


def _generate_events(
    conditions: list, n_scans: int, t_r: float, reps_per_condition: int, rng: np.random.Generator
) -> pd.DataFrame:
    """A jittered event-related design: each condition repeated ``reps_per_condition`` times."""
    duration = 2.0
    total_time = n_scans * t_r
    n_events = len(conditions) * reps_per_condition
    usable_time = total_time - 20.0  # keep a margin at the start/end of the run
    slot = usable_time / n_events

    labels = conditions * reps_per_condition
    rng.shuffle(labels)

    onsets = []
    t = 10.0
    for _ in range(n_events):
        jitter = rng.uniform(0, max(slot - duration - 2.0, 0.0))
        onsets.append(round(t + jitter, 2))
        t += slot

    return pd.DataFrame({"onset": onsets, "duration": [duration] * n_events, "trial_type": labels})


def _generate_synthetic_bold(
    func_grid: Nifti1Image,
    func_mask: np.ndarray,
    hotspots_mm: dict,
    events: pd.DataFrame,
    t_r: float,
    n_scans: int,
    rng: np.random.Generator,
    noise_sigma: float = 10.0,
    baseline: float = 1000.0,
    amplitude: float = 55.0,
    radius_mm: float = 13.0,
) -> Nifti1Image:
    """Gaussian noise + an HRF-convolved, spatially localized response per condition."""
    shape = func_grid.shape[:3]
    affine = func_grid.affine
    frame_times = np.arange(n_scans) * t_r

    data = (baseline + rng.normal(0, noise_sigma, shape + (n_scans,))).astype(np.float32)

    ii, jj, kk = np.meshgrid(*(np.arange(s) for s in shape), indexing="ij")
    world = np.stack([ii, jj, kk, np.ones_like(ii)], axis=-1).astype(np.float64) @ affine.T

    for condition, center_mm in hotspots_mm.items():
        sub = events[events["trial_type"] == condition]
        exp_condition = np.vstack(
            [sub["onset"].to_numpy(), sub["duration"].to_numpy(), np.ones(len(sub))]
        )
        regressor, _ = compute_regressor(exp_condition, "glover", frame_times)
        regressor = regressor[:, 0]
        peak = np.abs(regressor).max()
        if peak > 0:
            regressor = regressor / peak

        cx, cy, cz = center_mm
        dist2 = (world[..., 0] - cx) ** 2 + (world[..., 1] - cy) ** 2 + (world[..., 2] - cz) ** 2
        spatial = np.exp(-dist2 / (2 * radius_mm**2)).astype(np.float32)

        data += amplitude * spatial[..., None] * regressor[None, None, None, :]

    # Taper to zero smoothly outside the brain mask rather than a hard cutoff:
    # a sharp step would make later cubic-spline resampling (when this image
    # is aligned to the display grid) ring/overshoot right at the boundary.
    soft_mask = np.clip(gaussian_filter(func_mask.astype(np.float32), sigma=1.0), 0.0, 1.0)
    data *= soft_mask[..., None]
    return Nifti1Image(data, affine)
