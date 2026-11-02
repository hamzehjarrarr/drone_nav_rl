"""Streamlit demo: build a scenario, press Fly, watch the drone in a 3D viewer.

Run locally from the repository root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # make `sim`, `evaluation` and `app` importable
    sys.path.insert(0, str(ROOT))

from app.policy_numpy import NumpyPolicy  # noqa: E402
from app.viz import build_payload, viewer_html  # noqa: E402
from evaluation.rollout import RunOptions, fly, policy_pilot, reference_pilot  # noqa: E402
from sim.world import (  # noqa: E402
    ARENA_HEIGHT,
    World,
    WorldSpec,
    distance_to_obstacles,
    generate_world,
)

WEIGHTS = ROOT / "app" / "weights" / "policy.npz"

SCENES = {
    "Forest · golden hour": "forest",
    "City blocks · dusk": "city",
    "Warehouse · night": "warehouse",
}
DENSITY = {
    "forest": {"low": dict(n_trees=12), "medium": dict(n_trees=24), "high": dict(n_trees=36)},
    "city": {
        "low": dict(n_blocks=4, n_trees=6),
        "medium": dict(n_blocks=7, n_trees=10),
        "high": dict(n_blocks=10, n_trees=14),
    },
    "warehouse": {
        "low": dict(n_walls=3, n_trees=4),
        "medium": dict(n_walls=5, n_trees=8),
        "high": dict(n_walls=7, n_trees=12),
    },
}

st.set_page_config(page_title="Autonomous Drone RL", page_icon="🛸", layout="wide")


@st.cache_resource
def load_policy() -> NumpyPolicy | None:
    return NumpyPolicy(WEIGHTS) if WEIGHTS.exists() else None


def make_world(theme: str, density: str, seed: int) -> World:
    spec = WorldSpec(n_waypoints=4, **DENSITY[theme][density])
    return generate_world(spec, np.random.default_rng(seed))


def with_waypoints(world: World, pts: np.ndarray) -> World:
    wps = np.zeros_like(world.waypoints)
    wps[: len(pts)] = pts
    return World(
        world.cyl,
        world.cyl_mask,
        world.box,
        world.box_mask,
        wps,
        len(pts),
        world.start,
        world.start_yaw,
    )


def check_waypoints(world: World, pts: np.ndarray) -> list[str]:
    problems = []
    for i, p in enumerate(pts):
        if np.any(np.abs(p[:2]) > 12.0):
            problems.append(f"Waypoint {i + 1}: keep x and y between -12 and 12 m.")
        if not 1.0 <= p[2] <= ARENA_HEIGHT - 1.5:
            problems.append(f"Waypoint {i + 1}: altitude must be 1 to {ARENA_HEIGHT - 1.5:g} m.")
        d = distance_to_obstacles(
            p[None], world.cyl[None], world.cyl_mask[None], world.box[None], world.box_mask[None]
        )[0]
        if d < 0.6:
            problems.append(f"Waypoint {i + 1} is inside or too close to an obstacle.")
    return problems


@st.cache_data(max_entries=32, show_spinner=False)
def run_flight(
    theme: str,
    density: str,
    seed: int,
    waypoints: tuple,
    wind: bool,
    push: bool,
    realistic: bool,
    pilot_name: str,
) -> tuple[dict, dict]:
    world = make_world(theme, density, seed)
    if waypoints:
        world = with_waypoints(world, np.asarray(waypoints, dtype=float))
    policy = load_policy()
    pilot = policy_pilot(policy) if (pilot_name == "policy" and policy) else reference_pilot()
    opt = RunOptions(
        realistic=realistic,
        wind=3.0 if wind else 0.0,
        gust=1.0 if wind else 0.0,
        push=(0.0, 12.0, 0.0) if push else None,
        seed=seed,
        record=True,
        profile=policy.profile if policy else "kingfisher",
    )
    result = fly([world], pilot, opt)[0]
    payload = build_payload(result, world, theme, show_rays=False)
    summary = {
        k: result[k]
        for k in (
            "success",
            "collision",
            "out_of_bounds",
            "timeout",
            "time",
            "path_length",
            "avg_speed",
            "min_clearance",
            "waypoints_reached",
            "n_waypoints",
        )
    }
    return payload, summary


# ------------------------------------------------------------------ header
st.markdown(
    "### Autonomous Drone RL\n"
    "A PPO policy flies a simulated quadrotor through your waypoints, avoids obstacles it has "
    "never seen, and hovers at the end. Physics and sensing are modelled on a real agile "
    "quadrotor and depth camera."
)

policy = load_policy()
if policy is None:
    st.info(
        "No trained policy found in `app/weights/policy.npz` yet, so flights use the "
        "classical straight-line reference pilot. Train and export a policy to see the "
        "learned one (see the README)."
    )

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("#### 1 · World")
    scene_label = st.selectbox("Scene", list(SCENES))
    theme = SCENES[scene_label]
    density = st.select_slider("Obstacle density", ["low", "medium", "high"], value="medium")
    if "seed" not in st.session_state:
        st.session_state.seed = 4217

    def _shuffle() -> None:
        st.session_state.seed = int(np.random.default_rng().integers(0, 999_999))

    c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
    seed = c1.number_input("Seed", min_value=0, max_value=999_999, key="seed", step=1)
    c2.button("Shuffle", width="stretch", on_click=_shuffle)

    st.markdown("#### 2 · Waypoints")
    world = make_world(theme, density, int(seed))
    custom = st.toggle("Edit waypoints", value=False)
    waypoints: tuple = ()
    if custom:
        base = world.waypoints[: world.n_waypoints]
        table = st.data_editor(
            pd.DataFrame(
                {"x": base[:, 0].round(1), "y": base[:, 1].round(1), "alt": base[:, 2].round(1)}
            ),
            num_rows="dynamic",
            width="stretch",
            key=f"wp_{theme}_{density}_{seed}",
        )
        pts = table[["x", "y", "alt"]].to_numpy(dtype=float)
        pts = pts[~np.isnan(pts).any(axis=1)][:6]
        problems = check_waypoints(world, pts) if len(pts) else ["Add at least one waypoint."]
        for p in problems:
            st.warning(p)
        if not problems:
            waypoints = tuple(map(tuple, pts.round(2)))
        st.caption("The last waypoint is where the drone stops and hovers.")

    st.markdown("#### 3 · Conditions")
    wind = st.checkbox("Wind and gusts (3 m/s)")
    push = st.checkbox("Push the drone mid-flight")
    realistic = st.checkbox(
        "Realistic sensing",
        value=False,
        help="Forward-only depth camera (85° × 58°, 3 m), drifting position estimate, "
        "latency, battery sag and randomized dynamics.",
    )

    st.markdown("#### 4 · Pilot")
    options = (["Learned policy (PPO)"] if policy else []) + ["Straight-line reference (PID)"]
    pilot_label = st.radio("Pilot", options, label_visibility="collapsed")
    pilot_name = "policy" if pilot_label.startswith("Learned") else "reference"

    fly_clicked = st.button(
        "▶  Fly", type="primary", width="stretch", disabled=custom and not waypoints
    )

# ------------------------------------------------------------------ main
key = (theme, density, int(seed), waypoints, wind, push, realistic, pilot_name)
if fly_clicked or "last" not in st.session_state:
    with st.spinner("Simulating the flight…"):
        st.session_state.last = key
        st.session_state.flight = run_flight(*key)

payload, summary = st.session_state.flight
html = viewer_html(payload, height=580)
if hasattr(st, "iframe"):  # Streamlit 1.5x+ API
    st.iframe(html, height=600)
else:  # older Streamlit versions
    import streamlit.components.v1 as components

    components.html(html, height=600)

if st.session_state.last != key:
    st.caption("Settings changed. Press **Fly** to simulate the new scenario.")

if summary["success"]:
    result = "Success"
elif summary["collision"]:
    result = "Collision"
elif summary["out_of_bounds"]:
    result = "Lost control"
else:
    result = "Timeout"
m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Result",
    result,
    f"{summary['waypoints_reached']} / {summary['n_waypoints']} waypoints",
    delta_color="off",
)
m2.metric(
    "Flight time",
    f"{summary['time']:.1f} s",
    f"path {summary['path_length']:.0f} m",
    delta_color="off",
)
m3.metric("Average speed", f"{summary['avg_speed']:.1f} m/s")
m4.metric("Min clearance", f"{max(summary['min_clearance'], 0.0):.2f} m")

frames = payload["frames"]
t = np.asarray(frames["t"])
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Speed (m/s)**")
    st.line_chart({"t": t, "speed": frames["speed"]}, x="t", y="speed", height=160)
with c2:
    st.markdown("**Altitude (m)**")
    st.line_chart(
        {"t": t, "altitude": [p[2] for p in frames["pos"]]}, x="t", y="altitude", height=160
    )
with c3:
    st.markdown("**Nearest obstacle seen (m)**")
    near = [np.nan if v is None else v for v in frames["nearest"]]
    st.line_chart({"t": t, "nearest": near}, x="t", y="nearest", height=160)

with st.expander("How it works"):
    st.markdown(
        "- **Policy:** a small neural network (2 × 256) trained with PPO and an asymmetric "
        "critic. It outputs collective thrust and body rates at 50 Hz.\n"
        "- **Senses:** a 64-sector range scan (ideal) or a forward depth camera that tilts "
        "with the drone (realistic), plus velocity, attitude and the next two waypoints. "
        "No map.\n"
        "- **Simulator:** 6-DoF rigid body, motor lag, body and rotor drag, ground effect, "
        "wind and a rate controller at 250 Hz, using the drone profile the policy was "
        "trained for (default: the open-source Agilicious quadrotor).\n"
        "- **Viewer:** the drone is drawn 2.5× larger than life so it stays visible; physics "
        "uses its real size. The **Rays** button shows what its range sensor sees.\n"
        "- **The reference pilot** flies straight lines with a classical controller and is "
        "given the true state; it has no obstacle avoidance."
    )