# coding=utf-8
"""Shared fixtures. The demo dataset is expensive to build (~5-10s, mostly
spent loading/resampling the real MNI template), so it's session-scoped:
built once and reused read-only across every test that needs a realistic,
fully-wired BrainData instead of hand-rolled synthetic arrays."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="session")
def demo_data():
    """A fully-loaded BrainData from the offline synthetic demo dataset.

    Session-scoped and not mutated by any test: several tests in
    test_app.py exercise callbacks that mutate layer styling (opacity,
    visibility, the active contrast) as a side effect of simulating user
    interaction. If you add a test that depends on a *specific* layer
    state, don't assume a fresh/default state - set what you need
    explicitly, the same way the app's own callbacks do.
    """
    from interactive_glm.data.demo import build_demo_dataset

    return build_demo_dataset()


@pytest.fixture
def rng():
    return np.random.default_rng(0)
