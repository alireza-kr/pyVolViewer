"""
interactive_glm
================

A fast, modular Dash application for interactively exploring brain imaging
data: one anatomical or functional image in the background, any number of
statistical maps / ROI masks overlaid on top, and (optionally) a first-level
GLM explorer showing per-condition betas and the experimental timeline at
the selected voxel.

The public entry point most users need is :func:`interactive_glm.app.create_app`,
which takes a :class:`~interactive_glm.config.BrainData` (built with the
helpers in :mod:`interactive_glm.data.loading` or
:mod:`interactive_glm.data.demo`) and a :class:`~interactive_glm.config.ViewerSettings`,
and returns a ready-to-run Dash app.

Example
-------
>>> from interactive_glm.data.demo import build_demo_dataset
>>> from interactive_glm.app import create_app
>>> data = build_demo_dataset()
>>> app = create_app(data)
>>> app.run(debug=False)  # doctest: +SKIP

See the project README for the full guide, including how to load your own
NIfTI files instead of the demo dataset.
"""

from __future__ import annotations

__version__ = "2.0.0"

__all__ = ["__version__"]
