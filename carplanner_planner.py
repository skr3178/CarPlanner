"""
CarPlanner as a nuPlan AbstractPlanner.

Wraps the CarPlanner model so it can be driven by nuPlan's closed-loop
simulation framework (`run_simulation.py`), producing official CLS-NR /
S-CR / S-Area / S-PR / S-Comfort metrics.

Usage (Hydra):
    python nuplan-devkit/nuplan/planning/script/run_simulation.py \
        planner=carplanner \
        +simulation=closed_loop_nonreactive_agents \
        scenario_filter=test14_random

Usage (programmatic):
    from carplanner_planner import CarPlannerPlanner
    planner = CarPlannerPlanner(checkpoint_path="checkpoints/stage_b_best.pt")
    run_simulation(cfg, planners=planner)
"""

import math
import warnings
from typing import List, Optional, Type

import numpy as np
import torch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import (
    StateSE2,
    StateVector2D,
    TimePoint,
)
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)

import sys, os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config as cfg
from model import CarPlanner
from data_loader import _extract_routes

# ── Constants ─────────────────────────────────────────────────────────────────

_AGENT_TYPES = [
    TrackedObjectType.VEHICLE,
    TrackedObjectType.PEDESTRIAN,
    TrackedObjectType.BICYCLE,
]
_AGENT_CAT_NAMES = ['VEHICLE', 'PEDESTRIAN', 'BICYCLE', 'MOTORCYCLE']
_MAP_CATEGORIES = ['LANE', 'LANE_CONNECTOR', 'INTERSECTION', 'OTHER']
_POLYGON_LAYERS = [
    (SemanticMapLayer.CROSSWALK, 0),
    (SemanticMapLayer.STOP_LINE, 1),
    (SemanticMapLayer.INTERSECTION, 2),
]
_TL_STATUS_MAP = {'green': 0, 'yellow': 1, 'red': 2, 'unknown': 3}

DT = 0.1  # nuPlan sim timestep [s]
VEHICLE = get_pacifica_parameters()


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _to_ego(x, y, heading, ref_x, ref_y, ref_h):
    dx, dy = x - ref_x, y - ref_y
    cos_h, sin_h = math.cos(-ref_h), math.sin(-ref_h)
    xe = cos_h * dx - sin_h * dy
    ye = sin_h * dx + cos_h * dy
    he = (heading - ref_h + math.pi) % (2 * math.pi) - math.pi
    return xe, ye, he


def _ego_to_world(xe, ye, he, ref_x, ref_y, ref_h):
    cos_h, sin_h = math.cos(ref_h), math.sin(ref_h)
    xw = cos_h * xe - sin_h * ye + ref_x
    yw = sin_h * xe + cos_h * ye + ref_y
    hw = he + ref_h
    return xw, yw, hw


def _resample_polyline_xy(pts_xy: np.ndarray, n: int) -> np.ndarray:
    """Resample (K, 2) polyline to (n, 2) by uniform arc-length."""
    if len(pts_xy) < 2:
        return np.zeros((n, 2), dtype=np.float32)
    diffs = np.diff(pts_xy, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate([[0], np.cumsum(seg_len)])
    total = cumlen[-1]
    if total < 1e-6:
        return np.tile(pts_xy[0], (n, 1)).astype(np.float32)
    sample_d = np.linspace(0, total, n)
    out = np.zeros((n, 2), dtype=np.float32)
    for i, d in enumerate(sample_d):
        idx = max(0, min(int(np.searchsorted(cumlen, d)) - 1, len(pts_xy) - 2))
        t = (d - cumlen[idx]) / seg_len[idx] if seg_len[idx] > 1e-6 else 0.0
        out[i] = pts_xy[idx] + t * (pts_xy[idx + 1] - pts_xy[idx])
    return out


def _resample_polygon_ring(ring: np.ndarray, n: int) -> np.ndarray:
    pts = np.array(ring, dtype=np.float32)
    if len(pts) < 3:
        return np.zeros((n, 2), dtype=np.float32)
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    return _resample_polyline_xy(pts, n)


# ── Feature encoding helpers ─────────────────────────────────────────────────

def _encode_polyline_pts(pts_2d, speed_limit, cat_onehot, ref_x, ref_y, ref_h,
                         tl_onehot=None):
    """Encode (N, 2) world-frame points → (N, D_MAP_POINT=13) ego-frame features."""
    if tl_onehot is None:
        tl_onehot = np.zeros(4, dtype=np.float32)
    N = len(pts_2d)
    headings = np.zeros(N, dtype=np.float32)
    for j in range(N - 1):
        headings[j] = math.atan2(pts_2d[j + 1, 1] - pts_2d[j, 1],
                                  pts_2d[j + 1, 0] - pts_2d[j, 0])
    if N > 1:
        headings[-1] = headings[-2]
    feats = np.zeros((N, cfg.D_MAP_POINT), dtype=np.float32)
    for j in range(N):
        xe, ye, he = _to_ego(pts_2d[j, 0], pts_2d[j, 1], headings[j],
                             ref_x, ref_y, ref_h)
        feats[j] = np.concatenate([
            [xe, ye, math.sin(he), math.cos(he), speed_limit],
            cat_onehot, tl_onehot,
        ])
    return feats


def _resample_lane_path(discrete_path, n_points):
    """nuPlan discrete_path (list of StateSE2-like) → (n_points, 2) world xy."""
    pts = np.array([[p.x, p.y] for p in discrete_path], dtype=np.float32)
    return _resample_polyline_xy(pts, n_points)


# ── Feature extraction from nuPlan runtime state ─────────────────────────────

def _extract_agents(ego_state: EgoState, observation: DetectionsTracks,
                    ego_states_history: List[EgoState],
                    observations_history: List[DetectionsTracks]):
    """
    Build agents_now (N, D_AGENT=14), agents_mask (N,),
    and agents_history (T_HIST, N, D_AGENT).
    """
    ref_x = ego_state.rear_axle.x
    ref_y = ego_state.rear_axle.y
    ref_h = ego_state.rear_axle.heading

    def _encode_agents(obs, time_step=0.0):
        arr = np.zeros((cfg.N_AGENTS, cfg.D_AGENT), dtype=np.float32)
        mask = np.zeros(cfg.N_AGENTS, dtype=np.float32)
        if obs is None:
            return arr, mask
        tracked = obs.tracked_objects
        agents = tracked.get_tracked_objects_of_types(_AGENT_TYPES)
        for j, ag in enumerate(agents[:cfg.N_AGENTS]):
            xe, ye, he = _to_ego(
                ag.center.x, ag.center.y, ag.center.heading,
                ref_x, ref_y, ref_h,
            )
            vx = ag.velocity.x if hasattr(ag, 'velocity') and ag.velocity else 0.0
            vy = ag.velocity.y if hasattr(ag, 'velocity') and ag.velocity else 0.0
            bw = ag.box.width
            bl = ag.box.length
            bh = ag.box.height
            cat_name = ag.tracked_object_type.name
            cat_idx = (_AGENT_CAT_NAMES.index(cat_name)
                       if cat_name in _AGENT_CAT_NAMES
                       else len(_AGENT_CAT_NAMES) - 1)
            cat_oh = np.zeros(len(_AGENT_CAT_NAMES), dtype=np.float32)
            cat_oh[cat_idx] = 1.0
            arr[j] = np.concatenate([
                [xe, ye, math.sin(he), math.cos(he), vx, vy, bw, bl, bh, time_step],
                cat_oh,
            ])
            mask[j] = 1.0
        return arr, mask

    agents_now, agents_mask = _encode_agents(observation, time_step=0.0)

    # History: oldest first, newest (t=0) last
    agents_history = np.stack([agents_now] * cfg.T_HIST, axis=0)
    n_hist = len(observations_history)
    for i in range(min(n_hist, cfg.T_HIST)):
        hist_idx = cfg.T_HIST - n_hist + i
        if hist_idx < 0:
            continue
        t_norm = float(i - n_hist)  # negative time offset
        obs_h = observations_history[i]
        arr_h, _ = _encode_agents(obs_h, time_step=t_norm)
        agents_history[hist_idx] = arr_h
    agents_history[-1] = agents_now

    return agents_now, agents_mask, agents_history


def _extract_ego_history(ego_state: EgoState,
                         ego_states_history: List[EgoState]) -> np.ndarray:
    """Build ego_history (T_HIST, 4) — x, y, heading, speed in current ego frame.
    Mirrors data_loader.py: oldest first, last slot = current; pads oldest end
    by repeating the first valid frame if fewer than T_HIST states are available.
    """
    ref_x = ego_state.rear_axle.x
    ref_y = ego_state.rear_axle.y
    ref_h = ego_state.rear_axle.heading

    states = list(ego_states_history)[-cfg.T_HIST:] if ego_states_history else []
    if not states or states[-1] is not ego_state:
        states = (states + [ego_state])[-cfg.T_HIST:]

    ego_history = np.zeros((cfg.T_HIST, 4), dtype=np.float32)
    for i, st in enumerate(states):
        xe, ye, he = _to_ego(st.rear_axle.x, st.rear_axle.y, st.rear_axle.heading,
                             ref_x, ref_y, ref_h)
        spd = st.dynamic_car_state.rear_axle_velocity_2d.x
        idx = cfg.T_HIST - len(states) + i
        ego_history[idx] = [xe, ye, he, spd]
    if len(states) < cfg.T_HIST:
        first_valid = cfg.T_HIST - len(states)
        ego_history[:first_valid] = ego_history[first_valid]
    return ego_history


def _extract_map_lanes(map_api: AbstractMap, ref_x, ref_y, ref_h,
                       traffic_light_data=None):
    """Build (N_LANES, N_LANE_POINTS, D_POLYLINE_POINT=39) and mask."""
    point = Point2D(ref_x, ref_y)
    raw_lanes = []
    for layer in [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]:
        try:
            objs = map_api.get_proximal_map_objects(
                point, cfg.MAP_QUERY_RADIUS, [layer])
            raw_lanes.extend(objs[layer])
        except Exception:
            pass

    def _lane_dist(lane):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "invalid value encountered in cast")
                bp = lane.baseline_path
            if bp is None:
                return float('inf')
            pts = bp.discrete_path
            mid = pts[len(pts) // 2]
            return math.sqrt((mid.x - ref_x) ** 2 + (mid.y - ref_y) ** 2)
        except Exception:
            return float('inf')

    raw_lanes.sort(key=_lane_dist)

    # Build TL status lookup
    tl_status = {}
    if traffic_light_data:
        for tld in traffic_light_data:
            status_name = tld.status.name.lower() if hasattr(tld.status, 'name') else 'unknown'
            lane_connector_id = tld.lane_connector_id if hasattr(tld, 'lane_connector_id') else None
            if lane_connector_id is not None:
                tl_status[lane_connector_id] = status_name

    lanes = np.zeros((cfg.N_LANES, cfg.N_LANE_POINTS, cfg.D_POLYLINE_POINT),
                     dtype=np.float32)
    lanes_mask = np.zeros(cfg.N_LANES, dtype=np.float32)

    for i, lane in enumerate(raw_lanes[:cfg.N_LANES]):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "invalid value encountered in cast")
            bp = lane.baseline_path
        if bp is None or len(bp.discrete_path) < 2:
            continue

        pts_center = _resample_lane_path(bp.discrete_path, cfg.N_LANE_POINTS)

        lane_type = type(lane).__name__
        cat_oh = np.zeros(len(_MAP_CATEGORIES), dtype=np.float32)
        if 'LaneConnector' in lane_type:
            cat_oh[1] = 1.0
        elif 'Intersection' in lane_type:
            cat_oh[2] = 1.0
        elif 'Lane' in lane_type:
            cat_oh[0] = 1.0
        else:
            cat_oh[3] = 1.0

        sl = getattr(lane, 'speed_limit_mps', None)
        speed_limit = float(sl) if sl is not None else 0.0

        tl_oh = np.zeros(4, dtype=np.float32)
        lane_id = int(lane.id) if hasattr(lane, 'id') else -1
        if lane_id in tl_status:
            tl_idx = _TL_STATUS_MAP.get(tl_status[lane_id], 3)
            tl_oh[tl_idx] = 1.0

        feats_c = _encode_polyline_pts(pts_center, speed_limit, cat_oh,
                                       ref_x, ref_y, ref_h, tl_oh)

        # Left boundary
        try:
            lp = lane.left_boundary.discrete_path
            if len(lp) >= 2:
                pts_l = _resample_lane_path(lp, cfg.N_LANE_POINTS)
                feats_l = _encode_polyline_pts(pts_l, speed_limit, cat_oh,
                                               ref_x, ref_y, ref_h, tl_oh)
            else:
                feats_l = feats_c.copy()
        except Exception:
            feats_l = feats_c.copy()

        # Right boundary
        try:
            rp = lane.right_boundary.discrete_path
            if len(rp) >= 2:
                pts_r = _resample_lane_path(rp, cfg.N_LANE_POINTS)
                feats_r = _encode_polyline_pts(pts_r, speed_limit, cat_oh,
                                               ref_x, ref_y, ref_h, tl_oh)
            else:
                feats_r = feats_c.copy()
        except Exception:
            feats_r = feats_c.copy()

        lanes[i] = np.concatenate([feats_c, feats_l, feats_r], axis=-1)
        lanes_mask[i] = 1.0

    return lanes, lanes_mask


def _extract_map_polygons(map_api: AbstractMap, ref_x, ref_y, ref_h):
    """Build (N_POLYGONS, N_LANE_POINTS, D_POLYGON_POINT) and mask."""
    point = Point2D(ref_x, ref_y)
    collected = []

    for layer, cat_idx in _POLYGON_LAYERS:
        try:
            objs = map_api.get_proximal_map_objects(
                point, cfg.MAP_QUERY_RADIUS, [layer])
            raw = objs[layer]
        except Exception:
            continue

        cat_oh = np.zeros(4, dtype=np.float32)
        cat_oh[cat_idx] = 1.0

        for poly_obj in raw:
            try:
                ring = np.array(poly_obj.polygon.exterior.coords,
                                dtype=np.float64)[:, :2]
            except Exception:
                continue
            if len(ring) < 3:
                continue
            pts_2d = _resample_polygon_ring(ring, cfg.N_LANE_POINTS)
            feats = _encode_polyline_pts(pts_2d, 0.0, cat_oh,
                                         ref_x, ref_y, ref_h)
            cx, cy = pts_2d[:, 0].mean(), pts_2d[:, 1].mean()
            dist = math.sqrt((cx - ref_x) ** 2 + (cy - ref_y) ** 2)
            collected.append((dist, feats))

    collected.sort(key=lambda x: x[0])

    polygons = np.zeros((cfg.N_POLYGONS, cfg.N_LANE_POINTS, cfg.D_POLYGON_POINT),
                        dtype=np.float32)
    polygons_mask = np.zeros(cfg.N_POLYGONS, dtype=np.float32)
    for i, (_, feats) in enumerate(collected[:cfg.N_POLYGONS]):
        polygons[i] = feats
        polygons_mask[i] = 1.0

    return polygons, polygons_mask


# ── Trajectory conversion ────────────────────────────────────────────────────

def _pred_to_ego_states(pred_traj_ego: np.ndarray, ego_state: EgoState,
                        ) -> List[EgoState]:
    """
    Convert model output (T_FUTURE, 3) in ego frame → list of EgoState for nuPlan.

    The model emits a 1 Hz plan over `cfg.FUTURE_DT_S * cfg.T_FUTURE` seconds
    (paper §A: T=8 over 8 s). nuPlan's simulator and tracker run at DT=0.1 s,
    so before handing the plan to InterpolatedTrajectory we linearly interpolate
    the (xy, yaw) waypoints from the planner cadence to 0.1 s sub-steps. This
    matches the paper's §A note: "during testing, these trajectories are
    interpolated to 0.1-second intervals".

    pred_traj_ego: (T_FUTURE, 3)  — (x, y, yaw) in the ego frame at plan time.
    """
    ref_x = ego_state.rear_axle.x
    ref_y = ego_state.rear_axle.y
    ref_h = ego_state.rear_axle.heading
    t0_us = ego_state.time_point.time_us

    plan_dt_s = float(cfg.FUTURE_DT_S)            # 1.0 s — paper plan cadence
    sim_dt_s  = DT                                 # 0.1 s — nuPlan sim cadence
    sub_per_plan = max(1, int(round(plan_dt_s / sim_dt_s)))   # 10
    T_plan = len(pred_traj_ego)                    # 8
    n_sub = T_plan * sub_per_plan                  # 80

    # ── Linearly interpolate (x, y, yaw) from 1 Hz to 10 Hz ────────────────
    # Anchors at t = 0 (current ego, ego-frame origin) and t = k·plan_dt_s
    # for k = 1..T_plan. Interpolate to t = sim_dt_s, 2·sim_dt_s, ..., T_plan·plan_dt_s.
    anchor_t = np.concatenate([[0.0],
                                np.arange(1, T_plan + 1) * plan_dt_s])           # (T_plan+1,)
    anchor_x = np.concatenate([[0.0], pred_traj_ego[:, 0].astype(np.float64)])    # ego frame
    anchor_y = np.concatenate([[0.0], pred_traj_ego[:, 1].astype(np.float64)])
    anchor_h = np.concatenate([[0.0], pred_traj_ego[:, 2].astype(np.float64)])

    sub_t = np.arange(1, n_sub + 1) * sim_dt_s                                   # (n_sub,)
    sub_xe = np.interp(sub_t, anchor_t, anchor_x)
    sub_ye = np.interp(sub_t, anchor_t, anchor_y)
    # Unwrap heading before interp to avoid wrap-around discontinuities.
    sub_he = np.interp(sub_t, anchor_t, np.unwrap(anchor_h))

    # ── Build EgoStates at the interpolated cadence ────────────────────────
    states: List[EgoState] = []
    prev_x, prev_y = ref_x, ref_y
    for t in range(n_sub):
        xw, yw, hw = _ego_to_world(float(sub_xe[t]), float(sub_ye[t]), float(sub_he[t]),
                                    ref_x, ref_y, ref_h)

        # Velocity over the 0.1 s interval. For t == 0 use the ego's current
        # velocity so the first sub-step starts smoothly.
        if t > 0:
            vx = (xw - prev_x) / sim_dt_s
            vy = (yw - prev_y) / sim_dt_s
        else:
            vx = ego_state.dynamic_car_state.rear_axle_velocity_2d.x
            vy = ego_state.dynamic_car_state.rear_axle_velocity_2d.y
        prev_x, prev_y = xw, yw

        time_us = t0_us + int((t + 1) * sim_dt_s * 1e6)
        state = EgoState.build_from_rear_axle(
            rear_axle_pose=StateSE2(xw, yw, hw),
            rear_axle_velocity_2d=StateVector2D(vx, vy),
            rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
            tire_steering_angle=0.0,
            time_point=TimePoint(time_us),
            vehicle_parameters=VEHICLE,
        )
        states.append(state)

    return states


# ── The Planner ───────────────────────────────────────────────────────────────

class CarPlannerPlanner(AbstractPlanner):
    """
    nuPlan-compatible wrapper around the CarPlanner model.

    At each replan step the simulator calls compute_planner_trajectory().
    We extract features from the live simulation state, run model.forward_inference(),
    and return an InterpolatedTrajectory that the nuPlan LQR controller tracks.
    """

    requires_scenario: bool = False

    def __init__(
        self,
        checkpoint_path: str,
        stage: str = "b",
        map_radius: float = cfg.MAP_QUERY_RADIUS,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self._checkpoint_path = checkpoint_path
        self._stage = stage
        self._map_radius = map_radius
        self._device = torch.device(device)

        self._model: Optional[CarPlanner] = None
        self._map_api: Optional[AbstractMap] = None
        self._iteration = 0

        # Map cache: keyed by (round(x, 0), round(y, 0)) — re-query only when
        # ego moves >1 m from the last cached position.
        self._map_cache: dict = {}
        self._map_cache_pos: Optional[tuple] = None

    def _ensure_model(self):
        if self._model is not None:
            return
        self._model = CarPlanner().to(self._device)
        ckpt = torch.load(self._checkpoint_path, map_location=self._device)
        self._model.load_state_dict(ckpt['model'], strict=False)
        self._model.eval()

    def name(self) -> str:
        return "CarPlannerPlanner"

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        self._map_api = initialization.map_api
        self._iteration = 0
        self._map_cache = {}
        self._map_cache_pos = None
        self._ensure_model()

    def compute_planner_trajectory(
        self, current_input: PlannerInput
    ) -> InterpolatedTrajectory:
        ego_state, observation = current_input.history.current_state

        ref_x = ego_state.rear_axle.x
        ref_y = ego_state.rear_axle.y
        ref_h = ego_state.rear_axle.heading

        # ── History buffers ───────────────────────────────────────────────
        ego_states_hist = current_input.history.ego_states
        observations_hist = current_input.history.observations

        # ── Agents ────────────────────────────────────────────────────────
        agents_now, agents_mask, agents_history = _extract_agents(
            ego_state, observation, ego_states_hist, observations_hist,
        )

        # ── Ego history ──────────────────────────────────────────────────
        ego_history = _extract_ego_history(ego_state, ego_states_hist)

        # ── Map features with position cache ─────────────────────────────
        # Re-query map only when ego moves >1 m; map topology is stable
        # within that radius and queries dominate per-step latency.
        _pos = (ref_x, ref_y)
        _cache_miss = (
            self._map_cache_pos is None or
            math.hypot(ref_x - self._map_cache_pos[0],
                       ref_y - self._map_cache_pos[1]) > 1.0
        )
        if _cache_miss:
            tl_dict = {}
            if current_input.traffic_light_data:
                for tl in current_input.traffic_light_data:
                    tl_dict[tl.lane_connector_id] = tl.status.name.lower()
            map_lanes, map_lanes_mask = _extract_map_lanes(
                self._map_api, ref_x, ref_y, ref_h,
                traffic_light_data=current_input.traffic_light_data,
            )
            map_polygons, map_polygons_mask = _extract_map_polygons(
                self._map_api, ref_x, ref_y, ref_h,
            )
            route_polylines, route_mask, _ = _extract_routes(
                self._map_api, ref_x, ref_y, ref_h, tl_status=tl_dict,
            )
            self._map_cache = dict(
                map_lanes=map_lanes, map_lanes_mask=map_lanes_mask,
                map_polygons=map_polygons, map_polygons_mask=map_polygons_mask,
                route_polylines=route_polylines, route_mask=route_mask,
            )
            self._map_cache_pos = _pos
        else:
            map_lanes      = self._map_cache['map_lanes']
            map_lanes_mask = self._map_cache['map_lanes_mask']
            map_polygons      = self._map_cache['map_polygons']
            map_polygons_mask = self._map_cache['map_polygons_mask']
            route_polylines = self._map_cache['route_polylines']
            route_mask      = self._map_cache['route_mask']

        # ── To tensors (batch=1) ──────────────────────────────────────────
        def _t(arr):
            return torch.from_numpy(arr).unsqueeze(0).to(self._device)

        with torch.no_grad():
            _, _, best_traj, _ = self._model.forward_inference_fast(
                agents_now=_t(agents_now),
                agents_mask=_t(agents_mask),
                map_lanes=_t(map_lanes),
                map_lanes_mask=_t(map_lanes_mask),
                agents_history=_t(agents_history),
                ego_history=_t(ego_history),
                map_polygons=_t(map_polygons),
                map_polygons_mask=_t(map_polygons_mask),
                route_polylines=_t(route_polylines),
                route_mask=_t(route_mask),
            )

        # best_traj: (1, T_FUTURE, 3) in ego frame
        pred_np = best_traj[0].cpu().numpy()

        # ── Convert to nuPlan trajectory ──────────────────────────────────
        future_states = _pred_to_ego_states(pred_np, ego_state)

        # InterpolatedTrajectory needs ≥2 states; prepend current ego
        trajectory_states = [ego_state] + future_states

        self._iteration += 1
        return InterpolatedTrajectory(trajectory_states)
