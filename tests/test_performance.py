# coding=utf-8
"""
Performance benchmarks, run as tests with generous thresholds so they
double as regression guards (e.g. if a future change accidentally
reintroduces a full-volume recomputation on every click) without being
flaky on slower machines than this project was developed on.

Reference numbers from the development machine, after JIT/import warmup
(see the docstrings below for what each one measures):

* compose_panel (one panel, background + a few overlays): ~1-2 ms
* a full click -> 3 panels rebuilt + figures serialized to JSON: ~50 ms
* recomputing a GLM contrast (compute_contrast + resample to the
  reference grid) at the demo dataset's full ~2mm resolution: ~400 ms

The ~8x gap between "redraw" and "recompute a contrast" is the whole
reason callbacks/contrast.py exists as a separate, explicitly-triggered
callback instead of folding into callbacks/viewer.py's redraw - see that
module's docstring.
"""

from __future__ import annotations

import time

from interactive_glm.processing import alignment as al
from interactive_glm.processing import compositing as comp
from interactive_glm.processing.alignment import align_to_reference
from interactive_glm.processing.glm import compute_contrast_img


def _time_it(fn, n=20):
    fn()  # warmup (first call(s) pay one-time Python/Plotly setup costs)
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def test_compose_panel_is_fast(demo_data):
    bg = demo_data.backgrounds["anat"]
    overlays = list(demo_data.overlays.values())

    elapsed = _time_it(
        lambda: comp.compose_panel(bg.data, bg.vmin, bg.vmax, bg.cmap, al.AXIAL, demo_data.shape[2] // 2, overlays)
    )
    # Generous ceiling (development machine: ~1-2ms) - this is meant to catch
    # an accidental O(n^2) or a lost vectorization, not to be a tight SLA.
    assert elapsed < 0.05, f"compose_panel took {elapsed*1000:.1f}ms/call, expected well under 50ms"


def test_full_three_panel_redraw_is_fast(demo_data):
    from interactive_glm.figures.slice_views import build_slice_figure

    bg = demo_data.backgrounds["anat"]
    overlays = list(demo_data.overlays.values())
    cursor = tuple(s // 2 for s in demo_data.shape)

    def redraw_all_panels():
        for axis in (al.SAGITTAL, al.CORONAL, al.AXIAL):
            rgb = comp.compose_panel(bg.data, bg.vmin, bg.vmax, bg.cmap, axis, cursor[axis], overlays)
            fig = build_slice_figure(rgb, axis, demo_data.shape, cursor, title="t")
            fig.to_json()  # this is the cost that actually reaches the browser

    elapsed = _time_it(redraw_all_panels, n=10)
    # Generous ceiling (development machine: ~50ms) - this is the number a
    # user actually feels on every click, so it's worth a real assertion.
    assert elapsed < 0.5, f"full redraw took {elapsed*1000:.0f}ms, expected well under 500ms"


def test_contrast_recompute_benchmark(demo_data, capsys):
    """Not asserted tightly - this documents *why* contrast recomputation
    is gated behind an explicit Apply action rather than happening on
    every control change (see callbacks/contrast.py's docstring)."""

    def recompute():
        img = compute_contrast_img(demo_data.glm_model, "FaceLoc - SceneLoc", stat_type="t", output_type="z_score")
        align_to_reference(img, demo_data.reference_template, interpolation="continuous")

    elapsed = _time_it(recompute, n=3)
    with capsys.disabled():
        print(f"\n  [benchmark] contrast recompute: {elapsed*1000:.0f} ms/call at {demo_data.shape} resolution")
    # Sanity ceiling only - this operation is *expected* to be much slower
    # than a redraw; the assertion just catches genuine pathological
    # regressions (e.g. an accidental N+1 resample).
    assert elapsed < 5.0
