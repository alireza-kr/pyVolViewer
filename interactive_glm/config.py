# coding=utf-8
"""
Core data structures shared across the package.

This module is deliberately dependency-light (no Dash, no Plotly, no
nilearn imports beyond typing) so it can be imported from anywhere -
loaders, processing code, figure builders and callbacks - without risking
circular imports. If you add a new field that the rest of the app needs to
know about, it almost certainly belongs here.

There are two kinds of objects in this module:

* **Input specs** (:class:`RoiSource`, :class:`ViewerSettings`) - things
  *you* construct when setting up the app.
* **Runtime containers** (:class:`BackgroundOption`, :class:`OverlayLayer`,
  :class:`BrainData`) - things the loaders in :mod:`interactive_glm.data`
  build for you, already aligned to a common voxel grid, ready to be sliced
  and displayed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from nibabel.nifti1 import Nifti1Image

# A "Niimg-like" object: a path to a NIfTI file, or an already-loaded image.
NiimgLike = Union[str, Nifti1Image]

OverlayKind = Literal["roi", "stat_map"]


# --------------------------------------------------------------------------
# Input specs
# --------------------------------------------------------------------------


@dataclass
class RoiSource:
    """One region-of-interest to load, before it has been aligned to the grid.

    Pass a list of these to :func:`interactive_glm.data.loading.load_rois`.

    Parameters
    ----------
    img : str or nibabel.Nifti1Image
        Path to a NIfTI file, or an already-loaded image. Can be a binary
        mask, a probabilistic map in [0, 1], or a single region pulled out
        of an atlas with :func:`interactive_glm.data.loading.split_atlas`.
    label : str
        Display name, e.g. ``"Left FFA"``. Shown in the ROI checklist and
        in the coordinate readout when hovering.
    color : str, optional
        Hex color (e.g. ``"#e63946"``). If omitted, a color is assigned
        automatically from a perceptually-distinct palette so that
        consecutive ROIs never look alike (see
        :func:`interactive_glm.processing.colors.distinct_palette`).
    opacity : float, default=0.55
        Initial fill opacity, in [0, 1]. Adjustable at runtime from the
        control panel.
    threshold : float, default=0.5
        Voxels with a (resampled, 0-1) membership value above this are
        considered "inside" the ROI. Only relevant for probabilistic ROIs;
        binary masks are unaffected by the exact value as long as it is
        between 0 and 1.
    """

    img: NiimgLike
    label: str
    color: Optional[str] = None
    opacity: float = 0.55
    threshold: float = 0.5


@dataclass
class ViewerSettings:
    """User-tunable display and server settings.

    All fields have sensible defaults, so ``ViewerSettings()`` is a valid
    starting point; override only what you care about.

    Parameters
    ----------
    title : str
        Text shown in the browser tab and the app header.
    default_contrast : str, optional
        Contrast definition passed to ``model.compute_contrast`` on
        startup (e.g. ``"FaceHouse"`` or ``"Face - House"``). Ignored if
        ``data.glm_model`` is None. If None, the GLM stat-map layer starts
        hidden until the user picks a contrast.
    default_threshold : float
        Initial statistical threshold (in the same units as the computed
        map, e.g. z-scores).
    stat_cmap : str
        Matplotlib colormap name for the statistical map. Any diverging
        colormap works well; nilearn also registers neuroimaging-specific
        maps such as ``"cold_hot"`` on import.
    stat_opacity : float
        Initial opacity of the statistical map layer, in [0, 1].
    roi_opacity : float
        Initial opacity shared by all ROI layers, in [0, 1]. Controlled by
        a single slider in the UI to keep the control panel uncluttered;
        see the README for how to switch to per-ROI opacity.
    default_background : str, optional
        Key of the background to show on startup (``"anat"`` or ``"func"``,
        depending on what was loaded). Defaults to whichever background was
        loaded first.
    initial_voxel : tuple of int, optional
        (i, j, k) voxel coordinate to center the crosshair on at startup.
        If None, a reasonable default is chosen automatically: the peak of
        the default statistical map, the centroid of the first ROI, or the
        volume center, in that order of preference.
    show_direction_labels : bool
        Whether to draw L/R/A/P/S/I text labels on the slice views. Keeping
        this on is strongly recommended - it removes any ambiguity about
        display orientation regardless of how the source data was acquired.
    colorbar : bool
        Whether to show a colorbar for the statistical map.
    host : str
        Host to bind the Dash server to.
    port : int
        Port to serve the app on.
    debug : bool
        Passed straight to ``app.run``. Leave off (False) unless you are
        actively developing the app - Dash's debug/reloader mode starts
        the whole script twice and adds overhead on every callback.
    """

    title: str = "Interactive Brain Viewer"
    default_contrast: Optional[str] = None
    default_threshold: float = 3.1
    stat_cmap: str = "cold_hot"
    stat_opacity: float = 0.75
    roi_opacity: float = 0.55
    default_background: Optional[str] = None
    initial_voxel: Optional[tuple[int, int, int]] = None
    show_direction_labels: bool = True
    colorbar: bool = True
    host: str = "127.0.0.1"
    port: int = 8050
    debug: bool = False


# --------------------------------------------------------------------------
# Runtime containers (built by interactive_glm.data.loading / .demo)
# --------------------------------------------------------------------------


@dataclass
class BackgroundOption:
    """A background volume, already resampled onto the shared reference grid.

    Parameters
    ----------
    key : str
        Internal identifier, e.g. ``"anat"`` or ``"func"``.
    label : str
        Display name shown in the background selector, e.g.
        ``"T1 (anatomical)"``.
    data : numpy.ndarray
        3D array on the shared reference grid.
    vmin, vmax : float
        Display intensity range, pre-computed from robust percentiles so a
        few hot voxels don't wash out the whole image.
    cmap : str
        Matplotlib colormap name used to render this background in
        grayscale (or any other single-channel colormap).
    """

    key: str
    label: str
    data: np.ndarray
    vmin: float
    vmax: float
    cmap: str = "gray"


@dataclass
class OverlayLayer:
    """A statistical map or ROI mask, already resampled onto the reference grid.

    This is the runtime counterpart of :class:`RoiSource` (for ROIs) - GLM
    contrasts are turned into a ``kind="stat_map"`` layer on the fly by
    :mod:`interactive_glm.processing.glm` whenever the user picks a new
    contrast. Both kinds flow through the exact same compositing and
    figure code, which is what makes "one background + several arbitrary
    overlays" possible: the viewer never needs to know whether a layer
    started life as an ROI file or a computed contrast.

    Parameters
    ----------
    key : str
        Unique identifier, used as the ROI-checklist value / dict key.
    label : str
        Display name.
    kind : {"roi", "stat_map"}
        Determines how the layer is colored: ``"roi"`` uses a single flat
        ``color`` with alpha proportional to membership; ``"stat_map"``
        uses a diverging ``cmap`` normalized to ``[vmin, vmax]``.
    data : numpy.ndarray
        3D array on the shared reference grid. For ROIs this is typically
        in [0, 1] (soft edges from resampling); for stat maps it is in the
        map's native units (e.g. z-scores).
    color : str
        Hex color, used when ``kind == "roi"``.
    cmap : str
        Matplotlib colormap name, used when ``kind == "stat_map"``.
    opacity : float
        Layer opacity in [0, 1].
    threshold : float
        Values with ``abs(value) < threshold`` are fully transparent, i.e.
        for ROI masks values below this are outside the region, and for
        stat maps this is the usual statistical threshold.
    vmin, vmax : float, optional
        Color scaling range for ``kind == "stat_map"``. Ignored for ROIs.
        If omitted for a stat map, computed from the data at construction
        time (symmetric around zero).
    visible : bool
        Whether the layer is currently drawn. Toggled from the UI without
        recomputing or re-resampling anything.
    """

    key: str
    label: str
    kind: OverlayKind
    data: np.ndarray
    color: str = "#ef476f"
    cmap: str = "cold_hot"
    opacity: float = 0.55
    threshold: float = 0.5
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    visible: bool = True


@dataclass
class BrainData:
    """Everything the app needs, already aligned to one common voxel grid.

    Build this with :func:`interactive_glm.data.loading.load_brain_data` or
    :func:`interactive_glm.data.demo.build_demo_dataset` rather than by
    hand - the loaders take care of resampling every image onto the same
    canonical, isotropic grid so that the rest of the app can slice plain
    numpy arrays without ever touching an affine again.

    Parameters
    ----------
    affine : numpy.ndarray
        4x4 affine of the shared reference grid (RAS+ oriented, isotropic
        voxels).
    shape : tuple of int
        Shape of the reference grid, ``(nx, ny, nz)``.
    backgrounds : dict of str to BackgroundOption
        Available background images, keyed by :attr:`BackgroundOption.key`.
    overlays : dict of str to OverlayLayer
        ROI layers, keyed by :attr:`OverlayLayer.key`. The live GLM
        stat-map layer (if any) is stored separately in
        ``overlays["stat_map"]`` and replaced wholesale whenever the
        contrast or threshold changes.
    glm_model : object, optional
        A fitted ``nilearn.glm.first_level.FirstLevelModel``-like object
        (anything with ``compute_contrast``, ``labels_``, ``results_``,
        ``masker_`` and ``design_matrices_``). None disables the
        contrast / beta-bar / timeline panels.
    events : pandas.DataFrame, optional
        Columns ``onset``, ``duration``, ``trial_type``. Required for the
        experimental-timeline panel.
    condition_names : list of str, optional
        Design-matrix columns to expose as betas, with nuisance regressors
        (motion, drift, ...) already filtered out.
    betas : numpy.ndarray, optional
        4D array ``(nx, ny, nz, n_conditions)`` of per-condition effect
        sizes, on the shared reference grid, aligned with
        ``condition_names``.
    beta_range : tuple of float, optional
        A ``(min, max)`` range for the beta bar chart's y-axis, precomputed
        once from robust percentiles of the whole ``betas`` volume so the
        chart's scale stays fixed as you click between voxels instead of
        rescaling on every click.
    reference_template : nibabel.Nifti1Image, optional
        A zero-filled image carrying the shared affine/shape, used as the
        resampling target whenever a new contrast is computed. Built
        automatically by the loaders.
    t_r : float, optional
        Repetition time in seconds, for the timeline panel's x-axis.
    n_scans : int, optional
        Number of functional volumes, for the timeline panel's x-axis.
    func_data : numpy.ndarray, optional
        The functional run's own 4D array, at its *native* resolution -
        deliberately **not** resampled onto ``affine``/``shape`` like
        everything else in this class, since upsampling a full 4D series
        onto a finer anatomical grid just to read off single-voxel time
        series would multiply its memory footprint for no real benefit.
        Powers the measured-vs-fitted-signal panel; see
        :mod:`interactive_glm.processing.timeseries`. None if no ``func``
        was provided to the loader.
    func_affine : numpy.ndarray, optional
        ``func_data``'s own affine (generally a different grid than
        ``affine``). Required alongside ``func_data`` to convert a clicked
        reference-grid voxel to the corresponding native-resolution voxel.
    """

    affine: np.ndarray
    shape: tuple[int, int, int]
    backgrounds: dict[str, BackgroundOption]
    overlays: dict[str, OverlayLayer] = field(default_factory=dict)

    glm_model: Optional[object] = None
    events: Optional[pd.DataFrame] = None
    condition_names: Optional[list] = None
    betas: Optional[np.ndarray] = None
    beta_range: Optional[tuple[float, float]] = None
    reference_template: Optional[Nifti1Image] = None
    t_r: Optional[float] = None
    n_scans: Optional[int] = None
    func_data: Optional[np.ndarray] = None
    func_affine: Optional[np.ndarray] = None

    @property
    def has_glm(self) -> bool:
        """True if enough GLM data is present to enable the contrast/beta/timeline panels."""
        return self.glm_model is not None and self.condition_names is not None

    @property
    def has_timeseries(self) -> bool:
        """True if the measured-vs-fitted-signal panel can be shown."""
        return self.has_glm and self.func_data is not None and self.func_affine is not None

    @property
    def roi_layers(self) -> list[OverlayLayer]:
        """All overlay layers of kind ``"roi"``, in insertion order."""
        return [layer for layer in self.overlays.values() if layer.kind == "roi"]