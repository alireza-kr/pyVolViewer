# coding=utf-8
"""
Turn whatever you have on disk - an anatomical scan, a functional run, a
fitted GLM, a handful of ROI masks - into one aligned
:class:`~interactive_glm.config.BrainData` ready to hand to
:func:`interactive_glm.app.create_app`.

:func:`load_brain_data` is the one function most users need; everything
else here is a smaller piece it's built from, exposed separately so you
can reuse them (e.g. :func:`split_atlas` to turn a labeled atlas into
per-region ROIs, or :func:`load_rois` to align a list of ROIs on their
own).
"""

from __future__ import annotations

import re
import warnings
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from nibabel.nifti1 import Nifti1Image
from nilearn.image import mean_img

from interactive_glm.config import BackgroundOption, BrainData, NiimgLike, OverlayLayer, RoiSource
from interactive_glm.processing.alignment import (
    align_to_reference,
    build_reference_grid,
    ensure_img,
    make_zero_template,
)
from interactive_glm.processing.colors import distinct_palette
from interactive_glm.processing.glm import (
    NUISANCE_PATTERN,
    beta_display_range,
    compute_all_betas,
    list_conditions,
)


def load_brain_data(
    anat: Optional[NiimgLike] = None,
    func: Optional[NiimgLike] = None,
    glm_model=None,
    events=None,
    rois: Optional[list[RoiSource]] = None,
    voxel_size: Optional[float] = None,
    nuisance_pattern: re.Pattern = NUISANCE_PATTERN,
) -> BrainData:
    """Load and align everything the viewer needs into one :class:`BrainData`.

    At least one of ``anat`` or ``func`` is required to define a background
    and the reference grid. Everything else is optional and additive: no
    ROIs and no GLM gives you a plain background viewer; ROIs without a GLM
    gives you an ROI viewer; a GLM without ROIs reproduces the original
    contrast-explorer behaviour; any combination of the three works
    together, since they all end up as independent, toggleable overlay
    layers on the same grid.

    Parameters
    ----------
    anat : str or nibabel.Nifti1Image, optional
        Path to an anatomical (e.g. T1w) image, or an already-loaded image.
        Used as the reference grid's anchor if provided.
    func : str or nibabel.Nifti1Image, optional
        Path to a 4D functional image, or an already-loaded image. Its
        mean volume becomes a second background option; used as the
        reference grid's anchor if ``anat`` is not given. Its full 4D
        array is also kept, at native resolution, as
        ``BrainData.func_data`` (see that field's docstring) to power the
        measured-vs-fitted-signal panel.
    glm_model : nilearn FirstLevelModel or str, optional
        A fitted model, or a path to one saved with ``joblib.dump``.
        Enables the contrast, beta-bar and timeline panels.
    events : pandas.DataFrame or str, optional
        Either a DataFrame or a path to a ``.tsv``/``.csv`` file with
        ``onset``, ``duration`` and ``trial_type`` columns. Only used for
        the timeline panel.
    rois : list of RoiSource, optional
        ROIs to load as toggleable overlays. Build these by hand, or with
        :func:`split_atlas` from a labeled atlas image.
    voxel_size : float, optional
        Isotropic voxel size (mm) for the shared reference grid. Defaults
        to the anchor image's own finest native resolution - see
        :func:`interactive_glm.processing.alignment.build_reference_grid`.
    nuisance_pattern : re.Pattern
        Design-matrix columns matching this pattern are excluded from the
        condition list / beta bar chart.

    Returns
    -------
    BrainData

    Raises
    ------
    ValueError
        If neither ``anat`` nor ``func`` is provided.
    """
    if anat is None and func is None:
        raise ValueError("load_brain_data() needs at least one of anat= or func=.")

    anat_img = ensure_img(anat) if anat is not None else None
    func_img = ensure_img(func) if func is not None else None
    func_mean_img = mean_img(func_img) if func_img is not None else None

    anchor_key = "anat" if anat_img is not None else "func"
    anchor_img = anat_img if anat_img is not None else func_mean_img

    reference = build_reference_grid(anchor_img, voxel_size=voxel_size)
    anchor_data = np.asarray(reference.dataobj, dtype=np.float32)

    backgrounds: dict[str, BackgroundOption] = {}
    if anat_img is not None:
        data = anchor_data if anchor_key == "anat" else align_to_reference(anat_img, reference)
        vmin, vmax = _robust_intensity_range(data)
        backgrounds["anat"] = BackgroundOption(
            key="anat", label="Anatomical (T1)", data=data, vmin=vmin, vmax=vmax
        )
    if func_mean_img is not None:
        data = anchor_data if anchor_key == "func" else align_to_reference(func_mean_img, reference)
        vmin, vmax = _robust_intensity_range(data)
        backgrounds["func"] = BackgroundOption(
            key="func", label="Functional (mean)", data=data, vmin=vmin, vmax=vmax
        )

    overlays: dict[str, OverlayLayer] = {}
    if rois:
        overlays.update(load_rois(rois, reference))

    brain_data = BrainData(
        affine=reference.affine,
        shape=reference.shape[:3],
        backgrounds=backgrounds,
        overlays=overlays,
        reference_template=make_zero_template(reference),
    )

    if func_img is not None:
        # Kept at native resolution - see BrainData.func_data's docstring
        # for why this is deliberately not resampled onto the reference grid.
        brain_data.func_data = np.asarray(func_img.dataobj, dtype=np.float32)
        brain_data.func_affine = func_img.affine

    if glm_model is not None:
        model = joblib.load(glm_model) if isinstance(glm_model, str) else glm_model
        condition_names, condition_mask = list_conditions(model, exclude=nuisance_pattern)
        betas_img = compute_all_betas(model, condition_mask)

        brain_data.glm_model = model
        brain_data.condition_names = condition_names
        brain_data.betas = align_to_reference(betas_img, reference, interpolation="continuous")
        brain_data.beta_range = beta_display_range(brain_data.betas)
        brain_data.t_r = getattr(model, "t_r", None)
        if func_img is not None and len(func_img.shape) > 3:
            brain_data.n_scans = int(func_img.shape[3])
        if events is not None:
            brain_data.events = _load_events(events)

    return brain_data


def load_rois(rois: list[RoiSource], reference: Nifti1Image) -> dict[str, OverlayLayer]:
    """Align a list of :class:`RoiSource` onto ``reference`` and build overlay layers.

    Colors are auto-assigned from :func:`interactive_glm.processing.colors.distinct_palette`
    for any ROI that didn't specify one, so a long list of ROIs never
    accidentally reuses a color.

    Parameters
    ----------
    rois : list of RoiSource
    reference : nibabel.Nifti1Image
        Target grid, typically ``BrainData.reference_template`` or the
        output of :func:`~interactive_glm.processing.alignment.build_reference_grid`.

    Returns
    -------
    dict of str to OverlayLayer
    """
    palette = iter(distinct_palette(sum(1 for r in rois if r.color is None) or 1))
    layers: dict[str, OverlayLayer] = {}
    used_keys: set[str] = set()

    for index, roi in enumerate(rois):
        color = roi.color or next(palette)
        key = _unique_key(f"roi_{index}_{_slugify(roi.label)}", used_keys)
        with warnings.catch_warnings():
            # Continuous interpolation of a binary/probabilistic mask is
            # deliberate here - it gives smooth, anti-aliased ROI edges
            # instead of blocky nearest-neighbor ones. nilearn's warning
            # about this is expected and safe to suppress.
            warnings.simplefilter("ignore", UserWarning)
            data = align_to_reference(roi.img, reference, interpolation="continuous")
        layers[key] = OverlayLayer(
            key=key,
            label=roi.label,
            kind="roi",
            data=data,
            color=color,
            opacity=roi.opacity,
            threshold=roi.threshold,
        )
    return layers


def split_atlas(
    atlas: NiimgLike, labels: dict[int, str], colors: Optional[dict[int, str]] = None
) -> list[RoiSource]:
    """Turn an integer-labeled atlas image into one :class:`RoiSource` per region.

    Parameters
    ----------
    atlas : str or nibabel.Nifti1Image
        An image where each region is a distinct integer value (0 is
        treated as background and skipped).
    labels : dict of int to str
        Maps label value to a display name, e.g. ``{1: "Left FFA", 2: "Right FFA"}``.
    colors : dict of int to str, optional
        Maps label value to a hex color. Regions without an entry get an
        automatically assigned color.

    Returns
    -------
    list of RoiSource

    Examples
    --------
    >>> from nilearn.datasets import fetch_atlas_harvard_oxford
    >>> ho = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")  # doctest: +SKIP
    >>> labels = dict(enumerate(ho.labels))  # doctest: +SKIP
    >>> rois = split_atlas(ho.maps, labels)  # doctest: +SKIP
    """
    atlas_img = ensure_img(atlas)
    atlas_data = np.asarray(atlas_img.dataobj)

    sources = []
    for label_value, name in labels.items():
        if label_value == 0:
            continue
        mask = (atlas_data == label_value).astype(np.float32)
        mask_img = Nifti1Image(mask, atlas_img.affine)
        color = colors.get(label_value) if colors else None
        sources.append(RoiSource(img=mask_img, label=name, color=color))
    return sources


def _load_events(events) -> pd.DataFrame:
    """Accept a DataFrame or a path to a tsv/csv file; return a trimmed DataFrame."""
    if isinstance(events, pd.DataFrame):
        df = events
    else:
        sep = "\t" if str(events).endswith(".tsv") else None
        df = pd.read_csv(events, sep=sep, engine="python")
    return df[["onset", "duration", "trial_type"]].copy()


def _robust_intensity_range(data: np.ndarray, low: float = 1.0, high: float = 99.5) -> tuple[float, float]:
    """A display (vmin, vmax) that ignores background/near-zero voxels and outliers.

    Background voxels are excluded by a *relative* threshold (a fraction of
    the data's own peak magnitude) rather than an exact ``!= 0`` check,
    since real-world images - including nilearn's own bundled MNI
    template - often carry a sea of tiny near-zero values (interpolation
    noise, float rounding) outside the brain that would otherwise dominate
    the low percentile and wash out the display range.
    """
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    epsilon = 1e-3 * np.max(np.abs(finite))
    sample = finite[np.abs(finite) > epsilon] if epsilon > 0 else finite
    if sample.size == 0:
        sample = finite
    vmin, vmax = (float(v) for v in np.percentile(sample, [low, high]))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "roi"


def _unique_key(key: str, used: set) -> str:
    candidate, suffix = key, 1
    while candidate in used:
        suffix += 1
        candidate = f"{key}_{suffix}"
    used.add(candidate)
    return candidate