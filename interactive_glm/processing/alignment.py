# coding=utf-8
"""
Everything to do with getting different images onto one shared grid, and
converting between voxel indices, millimetre coordinates, and on-screen
pixel positions.

Why a shared grid?
-------------------
The original app resampled the statistical map to the background *once*,
then recomputed a voxel-space coordinate transform on every single click.
That's easy to get subtly wrong (it's easy to swap the direction of a
transform) and it means every new overlay type needs its own plumbing.

Here we resample *every* image - every background candidate, every ROI,
every freshly-computed contrast - onto one canonical grid, once, up front.
After that, "which voxel did the user click on" is just an (i, j, k)
integer triple that indexes directly into any of these arrays. There is
no more affine algebra anywhere else in the app.

Display orientation
--------------------
The canonical grid is built to be RAS+ (voxel axes increasing in the same
direction as +Right, +Anterior, +Superior world coordinates), regardless
of how the source data was stored, using :func:`nilearn.image.resample_img`
with a diagonal target affine (nilearn computes the tightest bounding shape
automatically - see its docstring for details).

Given that, :func:`to_display` implements *one* rule that turns the 2D
slice you get by fixing one axis into a correctly oriented image - top of
screen is always the "higher" anatomical direction (Anterior for axial,
Superior for coronal/sagittal) and the right of the screen is always the
patient's right / anterior, matching neurological display convention. This
was verified against a synthetic volume with a marker at a known
Right/Anterior/Superior location before being relied on elsewhere - see
``tests/test_alignment.py``.

Regardless of the exact convention, :func:`panel_direction_labels` gives
you the correct L/R/A/P/S/I labels to paint on top of the image, computed
from the same rule, so there is never any ambiguity for the person looking
at the screen.

nilearn version compatibility
-------------------------------
``resample_img``/``resample_to_img`` gained a ``copy_header`` keyword in
nilearn 0.11.0 (see nilearn's "what's new" for that release). Older
installs (nilearn < 0.11.0, e.g. 0.10.x) raise ``TypeError`` if you pass it
at all. Since this argument only controls whether NIfTI header *metadata*
is preserved - it has no effect on the affine or voxel data this module
actually cares about - we detect support at import time and only pass it
when available, rather than forcing a hard minimum-version requirement on
every user of this package.
"""

from __future__ import annotations

import inspect
from typing import Optional

import numpy as np
from nibabel.nifti1 import Nifti1Image
from nilearn.image import load_img, resample_img, resample_to_img

from interactive_glm.config import NiimgLike

# Axis identifiers for the three orthogonal views. The value is the index,
# in an (i, j, k) voxel triple, that is *held fixed* for that view.
SAGITTAL = 0  # fixed i (left-right axis); shows the (A, S) plane
CORONAL = 1  # fixed j (anterior-posterior axis); shows the (R, S) plane
AXIAL = 2  # fixed k (superior-inferior axis); shows the (R, A) plane

VIEW_NAMES = {SAGITTAL: "sagittal", CORONAL: "coronal", AXIAL: "axial"}


def _supports_copy_header(func) -> bool:
    """Whether ``func`` (a nilearn resampling function) accepts ``copy_header``.

    See the module docstring - this is a forward/backward compatibility
    check, not a feature flag: it lets this module work unmodified across
    nilearn versions before and after 0.11.0.
    """
    return "copy_header" in inspect.signature(func).parameters


_RESAMPLE_IMG_SUPPORTS_COPY_HEADER = _supports_copy_header(resample_img)
_RESAMPLE_TO_IMG_SUPPORTS_COPY_HEADER = _supports_copy_header(resample_to_img)


def ensure_img(img: NiimgLike) -> Nifti1Image:
    """Load ``img`` if it's a path; pass it through unchanged otherwise."""
    return load_img(img) if isinstance(img, str) else img


def build_reference_grid(
    anchor: NiimgLike, voxel_size: Optional[float] = None
) -> Nifti1Image:
    """Resample ``anchor`` onto a canonical, isotropic, RAS+ grid.

    This is the single source of truth for "the grid" that every other
    image in the app gets aligned to. It is intentionally the *only* place
    in the codebase that builds a target affine from scratch; everything
    else calls :func:`align_to_reference` with the result.

    Parameters
    ----------
    anchor : str or nibabel.Nifti1Image
        The image whose field of view defines the grid - typically the
        anatomical image.
    voxel_size : float, optional
        Isotropic voxel size in mm for the reference grid. If None, uses
        the finest resolution already present in ``anchor`` (the minimum
        of its native voxel dimensions), clipped to [0.5, 4.0] mm as a
        safety net against degenerate headers.

    Returns
    -------
    nibabel.Nifti1Image
        ``anchor`` resampled onto the new grid. Its ``.affine`` and
        ``.shape`` define the reference grid used everywhere else; its
        data is also directly usable as a background layer.
    """
    anchor = ensure_img(anchor)

    if voxel_size is None:
        zooms = anchor.header.get_zooms()[:3]
        voxel_size = float(np.clip(round(float(np.min(zooms)), 2), 0.5, 4.0))

    # A 3x3 (not 4x4) target affine tells nilearn "use this orientation and
    # spacing, and figure out the offset and shape that tightly bound the
    # data yourself" - see resample_img's docstring. All-positive diagonal
    # => RAS+.
    target_affine = np.diag([voxel_size, voxel_size, voxel_size]).astype(float)

    kwargs = dict(
        target_affine=target_affine,
        target_shape=None,
        interpolation="continuous",
        force_resample=True,
    )
    if _RESAMPLE_IMG_SUPPORTS_COPY_HEADER:
        kwargs["copy_header"] = False

    return resample_img(anchor, **kwargs)


def align_to_reference(
    img: NiimgLike, reference: Nifti1Image, interpolation: str = "continuous"
) -> np.ndarray:
    """Resample ``img`` onto ``reference``'s grid and return a plain float32 array.

    Parameters
    ----------
    img : str or nibabel.Nifti1Image
        Image to align.
    reference : nibabel.Nifti1Image
        Defines the target grid (typically the output of
        :func:`build_reference_grid`, or ``BrainData.reference_template``).
    interpolation : {"continuous", "linear", "nearest"}
        Use ``"nearest"`` for label/atlas images where you must not blend
        adjacent labels; ``"continuous"`` (the default) is appropriate for
        anatomical/functional images and gives ROI masks smooth,
        anti-aliased edges instead of blocky ones.

    Returns
    -------
    numpy.ndarray
        3D float32 array with shape ``reference.shape``.
    """
    img = ensure_img(img)

    kwargs = dict(interpolation=interpolation, force_resample=True)
    if _RESAMPLE_TO_IMG_SUPPORTS_COPY_HEADER:
        kwargs["copy_header"] = False

    resampled = resample_to_img(img, reference, **kwargs)
    return np.asarray(resampled.dataobj, dtype=np.float32)


def make_zero_template(reference: Nifti1Image) -> Nifti1Image:
    """Build a lightweight zero-filled image sharing ``reference``'s affine/shape.

    Used as the ``target_img`` for :func:`nilearn.image.resample_to_img`
    when aligning images computed *after* startup (e.g. a freshly computed
    GLM contrast) without having to keep the original, potentially large,
    anchor image around.
    """
    shape = reference.shape[:3]
    return Nifti1Image(np.zeros(shape, dtype=np.float32), reference.affine)


def voxel_to_mm(ijk: tuple[int, int, int], affine: np.ndarray) -> tuple[float, float, float]:
    """Convert a voxel index to world (mm) coordinates."""
    i, j, k = ijk
    x, y, z, _ = affine @ np.array([i, j, k, 1.0])
    return float(x), float(y), float(z)


def mm_to_voxel(
    xyz: tuple[float, float, float], affine: np.ndarray, shape: tuple[int, int, int]
) -> tuple[int, int, int]:
    """Convert world (mm) coordinates to the nearest in-bounds voxel index."""
    x, y, z = xyz
    i, j, k, _ = np.linalg.inv(affine) @ np.array([x, y, z, 1.0])
    i = int(np.clip(round(i), 0, shape[0] - 1))
    j = int(np.clip(round(j), 0, shape[1] - 1))
    k = int(np.clip(round(k), 0, shape[2] - 1))
    return i, j, k


# --------------------------------------------------------------------------
# Voxel grid <-> on-screen pixel position for the three orthogonal views
# --------------------------------------------------------------------------


def get_slice_2d(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Extract the 2D slice of ``volume`` for ``axis`` held fixed at ``index``."""
    index = int(np.clip(index, 0, volume.shape[axis] - 1))
    if axis == SAGITTAL:
        return volume[index, :, :]
    if axis == CORONAL:
        return volume[:, index, :]
    return volume[:, :, index]


def to_display(slice_2d: np.ndarray) -> np.ndarray:
    """Reorient a raw 2D slice (as returned by :func:`get_slice_2d`) for display.

    One rule, used identically for all three views: transpose, then flip
    rows. Applied to a volume on an RAS+ grid this puts the higher
    anatomical direction (Anterior for axial; Superior for coronal and
    sagittal) at the top of the image, and the patient's right (axial,
    coronal) or anterior (sagittal) on the right of the image - see the
    module docstring and ``tests/test_alignment.py``.
    """
    return np.asarray(slice_2d).T[::-1, :]


def display_shape(shape: tuple[int, int, int], axis: int) -> tuple[int, int]:
    """(rows, cols) of the displayed image for ``axis``, given the volume ``shape``."""
    nx, ny, nz = shape
    if axis == SAGITTAL:
        return nz, ny
    if axis == CORONAL:
        return nz, nx
    return ny, nx


def voxel_to_panel_rc(
    ijk: tuple[int, int, int], shape: tuple[int, int, int], axis: int
) -> tuple[int, int]:
    """Where voxel ``ijk`` lands on-screen (row, col) in the ``axis`` panel."""
    i, j, k = ijk
    nx, ny, nz = shape
    if axis == SAGITTAL:
        return nz - 1 - k, j
    if axis == CORONAL:
        return nz - 1 - k, i
    return ny - 1 - j, i


def panel_rc_to_voxel(
    row: float,
    col: float,
    current_ijk: tuple[int, int, int],
    shape: tuple[int, int, int],
    axis: int,
) -> tuple[int, int, int]:
    """Inverse of :func:`voxel_to_panel_rc`: a click at (row, col) in the
    ``axis`` panel updates the two free coordinates and keeps the fixed one.
    """
    i, j, k = current_ijk
    nx, ny, nz = shape
    row, col = int(round(row)), int(round(col))
    if axis == SAGITTAL:
        j, k = col, nz - 1 - row
    elif axis == CORONAL:
        i, k = col, nz - 1 - row
    else:
        i, j = col, ny - 1 - row
    i = int(np.clip(i, 0, nx - 1))
    j = int(np.clip(j, 0, ny - 1))
    k = int(np.clip(k, 0, nz - 1))
    return i, j, k


def panel_direction_labels(axis: int) -> dict[str, str]:
    """Text labels for the four edges of an orthogonal-view panel.

    Returns a dict with keys ``"top"``, ``"bottom"``, ``"left"``, ``"right"``,
    matching the orientation produced by :func:`to_display`.
    """
    if axis == SAGITTAL:
        return {"top": "S", "bottom": "I", "left": "P", "right": "A"}
    if axis == CORONAL:
        return {"top": "S", "bottom": "I", "left": "L", "right": "R"}
    return {"top": "A", "bottom": "P", "left": "L", "right": "R"}