import json
import re
from pathlib import Path

import numpy as np
import pytest

from app.viz import build_payload, viewer_html
from evaluation.rollout import RunOptions, fly, reference_pilot, summarize
from evaluation.scenarios import SCENARIOS, scenario_worlds


def test_viewer_html_embeds_valid_json():
    worlds = scenario_worlds(SCENARIOS[0], 1)
    res = fly(worlds, reference_pilot(), RunOptions(record=True))[0]
    html = viewer_html(build_payload(res, worlds[0], "forest", show_rays=False))
    raw = re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S)
    data = json.loads(raw.group(1).replace("<\\/", "</"))
    assert len(data["frames"]["t"]) == len(data["frames"]["pos"]) > 10
    assert len(data["rayDirs"]) == 64
    # the only external resource is three.js from jsDelivr
    urls = set(re.findall(r"https?://[^\s\"')]+", html))
    assert all(u.startswith("https://cdn.jsdelivr.net/npm/three@0.160.0/") for u in urls)


def test_evaluation_summary_fields():
    worlds = scenario_worlds(SCENARIOS[0], 3)
    s = summarize(fly(worlds, reference_pilot(), RunOptions()))
    assert s["episodes"] == 3
    assert 0.0 <= s["success_rate"] <= 1.0


def test_streamlit_app_runs():
    st_testing = pytest.importorskip("streamlit.testing.v1")
    app_file = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    at = st_testing.AppTest.from_file(str(app_file), default_timeout=240).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Result" in labels and "Average speed" in labels
    np.testing.assert_equal(len(at.button) >= 2, True)


def test_success_by_onboard_estimate():
    """Ideal mode: estimate == truth, so both success measures agree. Realistic mode:
    success by the estimate can only add flights that held the hover where they believed
    the target was, and the true end error is reported for them."""
    worlds = scenario_worlds(SCENARIOS[0], 12)  # open field
    ideal = fly(worlds, reference_pilot(), RunOptions())
    assert all(r["success"] == r["true_success"] for r in ideal)
    real = fly(worlds, reference_pilot(), RunOptions(realistic=True))
    s = summarize(real)
    assert s["success_rate"] >= s["true_success_rate"]
    assert all(r["true_final_error"] is not None for r in real if r["success"])