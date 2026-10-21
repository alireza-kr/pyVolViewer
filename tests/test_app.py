# coding=utf-8
"""
Tests for interactive_glm.app.

The callback tests POST directly to Dash's ``/_dash-update-component``
endpoint (through Flask's test client) rather than calling the registered
callback functions in Python, because those functions read
``dash.ctx.triggered_id`` / ``dash.ctx.triggered`` internally - state that
only exists inside a real Dash request. Going through the actual endpoint
is the only way to exercise that code path faithfully, and as a bonus it
validates the exact wire format the browser would send, including
pattern-matching component ids.

Note: these tests share the session-scoped ``demo_data`` fixture with
other test modules and some of them deliberately trigger callbacks that
mutate layer styling (opacity/visibility) or replace the active
statistical map - see the note in conftest.py. None of the assertions
here or elsewhere depend on that mutable state being in any particular
starting configuration.
"""

from __future__ import annotations

import json

import pytest

from interactive_glm.app import create_app
from interactive_glm.config import ViewerSettings


@pytest.fixture(scope="module")
def app_and_client(demo_data):
    app = create_app(demo_data, ViewerSettings(default_contrast="FaceLoc - SceneLoc"))
    return app, app.server.test_client()


def _post(client, output, inputs, state=None, changed=None, outputs=None):
    body = {
        "output": output,
        "outputs": outputs
        if outputs is not None
        else {"id": output.rsplit(".", 1)[0], "property": output.rsplit(".", 1)[1]},
        "inputs": inputs,
        "state": state or [],
        "changedPropIds": changed or [],
        "parsedChangedPropsIds": [],
    }
    return client.post(
        "/_dash-update-component", data=json.dumps(body), headers={"Content-Type": "application/json"}
    )


def test_create_app_builds_a_dash_app(demo_data):
    app = create_app(demo_data, ViewerSettings(default_contrast="FaceLoc - SceneLoc"))
    assert len(app.callback_map) == 4


def test_index_page_serves(app_and_client):
    _app, client = app_and_client
    resp = client.get("/")
    assert resp.status_code == 200


def test_layout_endpoint_serves_valid_json(app_and_client):
    _app, client = app_and_client
    resp = client.get("/_dash-layout")
    assert resp.status_code == 200
    json.loads(resp.data)  # must not raise


def test_click_on_axial_panel_moves_cursor(app_and_client):
    _app, client = app_and_client
    resp = _post(
        client,
        "cursor-position.data",
        inputs=[
            [
                {"id": {"axis": 0, "type": "slice-graph"}, "property": "clickData", "value": None},
                {"id": {"axis": 1, "type": "slice-graph"}, "property": "clickData", "value": None},
                {
                    "id": {"axis": 2, "type": "slice-graph"},
                    "property": "clickData",
                    "value": {"points": [{"x": 40, "y": 30}]},
                },
            ]
        ],
        state=[{"id": "cursor-position", "property": "data", "value": {"i": 49, "j": 58, "k": 47}}],
        changed=['{"axis":2,"type":"slice-graph"}.clickData'],
    )
    assert resp.status_code == 200
    new_cursor = json.loads(resp.data)["response"]["cursor-position"]["data"]
    # axial view: click sets (i, j) from (col, row) and leaves k untouched
    assert new_cursor == {"i": 40, "j": 86, "k": 47}


def test_redraw_applies_roi_visibility_and_opacity(app_and_client, demo_data):
    _app, client = app_and_client
    roi_key = demo_data.roi_layers[0].key
    other_key = demo_data.roi_layers[1].key

    resp = _post(
        client,
        '..{"axis":0,"type":"slice-graph"}.figure...{"axis":1,"type":"slice-graph"}.figure...'
        '{"axis":2,"type":"slice-graph"}.figure...coordinate-readout.children..',
        inputs=[
            {"id": "cursor-position", "property": "data", "value": {"i": 40, "j": 86, "k": 47}},
            {"id": "background-selector", "property": "value", "value": "func"},
            {"id": "roi-checklist", "property": "value", "value": [roi_key]},
            {"id": "roi-opacity-slider", "property": "value", "value": 0.7},
            {"id": "stat-threshold-slider", "property": "value", "value": 2.5},
            {"id": "stat-cmap-dropdown", "property": "value", "value": "bwr"},
            {"id": "stat-opacity-slider", "property": "value", "value": 0.8},
            {"id": "contrast-status", "property": "children", "value": "x"},
        ],
        changed=["cursor-position.data"],
        outputs=[
            {"id": {"axis": 0, "type": "slice-graph"}, "property": "figure"},
            {"id": {"axis": 1, "type": "slice-graph"}, "property": "figure"},
            {"id": {"axis": 2, "type": "slice-graph"}, "property": "figure"},
            {"id": "coordinate-readout", "property": "children"},
        ],
    )
    assert resp.status_code == 200
    payload = json.loads(resp.data)["response"]
    assert payload['{"axis":2,"type":"slice-graph"}']["figure"]["data"][0]["type"] == "image"
    assert "voxel (40, 86, 47)" in payload["coordinate-readout"]["children"]

    assert demo_data.overlays[roi_key].visible is True
    assert demo_data.overlays[roi_key].opacity == pytest.approx(0.7)
    assert demo_data.overlays[other_key].visible is False


def test_apply_valid_contrast_updates_status_and_layer(app_and_client, demo_data):
    _app, client = app_and_client
    resp = _post(
        client,
        "contrast-status.children",
        inputs=[
            {"id": "contrast-apply", "property": "n_clicks", "value": 1},
            {"id": "contrast-input", "property": "n_submit", "value": None},
        ],
        state=[
            {"id": "contrast-input", "property": "value", "value": "ObjectLoc - FaceLoc"},
            {"id": "stat-threshold-slider", "property": "value", "value": 3.1},
            {"id": "stat-cmap-dropdown", "property": "value", "value": "cold_hot"},
            {"id": "stat-opacity-slider", "property": "value", "value": 0.75},
        ],
        changed=["contrast-apply.n_clicks"],
    )
    assert resp.status_code == 200
    status = json.loads(resp.data)["response"]["contrast-status"]["children"]
    assert "ObjectLoc - FaceLoc" in status
    assert demo_data.overlays["stat_map"].label == "ObjectLoc - FaceLoc"


def test_apply_invalid_contrast_returns_friendly_error_not_a_500(app_and_client):
    _app, client = app_and_client
    resp = _post(
        client,
        "contrast-status.children",
        inputs=[
            {"id": "contrast-apply", "property": "n_clicks", "value": 2},
            {"id": "contrast-input", "property": "n_submit", "value": None},
        ],
        state=[
            {"id": "contrast-input", "property": "value", "value": "NotARealCondition * nonsense $$"},
            {"id": "stat-threshold-slider", "property": "value", "value": 3.1},
            {"id": "stat-cmap-dropdown", "property": "value", "value": "cold_hot"},
            {"id": "stat-opacity-slider", "property": "value", "value": 0.75},
        ],
        changed=["contrast-apply.n_clicks"],
    )
    assert resp.status_code == 200
    status = json.loads(resp.data)["response"]["contrast-status"]["children"]
    assert "Could not compute" in status


def test_beta_graph_updates_with_cursor(app_and_client):
    _app, client = app_and_client
    resp = _post(
        client,
        "beta-graph.figure",
        inputs=[{"id": "cursor-position", "property": "data", "value": {"i": 10, "j": 10, "k": 10}}],
        changed=["cursor-position.data"],
    )
    assert resp.status_code == 200
    fig = json.loads(resp.data)["response"]["beta-graph"]["figure"]
    assert fig["data"][0]["type"] == "bar"
