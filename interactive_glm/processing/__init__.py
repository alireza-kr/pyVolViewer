"""
Pure data-processing code: alignment/resampling, GLM contrasts, colormapping
and image compositing, and ROI color assignment.

Nothing in this subpackage imports Dash or Plotly. Every function takes
plain arrays/images in and returns plain arrays/images out, which is what
makes it straightforward to unit test (see ``tests/``) and to reuse outside
the app entirely, e.g. from a notebook.
"""
