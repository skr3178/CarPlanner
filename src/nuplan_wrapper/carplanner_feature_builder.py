"""
CarPlanner feature builder for nuPlan closed-loop simulation.

Converts nuPlan's PlannerInput (ego states, tracked objects, map data)
into CarPlanner's tensor format matching data_loader.py conventions.
"""

import math
import numpy as np
import torch
from typing import Dict, List, Optional

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config as cfg


# ── Coordinate helpers (same as data_loader.py) ──────────────────────────────────

def _to_ego_frame(x, y, heading, ref_x, ref_y, ref_heading):
    """Transform global (x, y, heading) into ego-centric frame."""
    dx = x - ref_x
    dy = y - ref_y
    cos_h = math.cos(-ref_heading)
    sin_h = math.sin(-ref_heading)
    x_e = cos_h * dx - sin_h * dy
    y_e = sin_h * dx + cos_h * dy
    h_e = heading - ref_heading
    h_e = (h_e + math.pi) % (2 * math.pi) - math.pi
    return x_e, y_e, h_e


def _resample_polyline(points: list, n_points: int) -> np.ndarray:
    """Resample a polyline to exactly n_points via uniform spacing."""
    if len(points) < 2:
        return np.zeros((n_points, 2), dtype=np.float32)

    pts = np.array([[p.x, p.y] for p in points], dtype=np.float32)
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_len = cumlen[-1]

    if total_len < 1e-6:
        return np.zeros((n_points, 2), dtype=np.float32)

    sample_dists = np.linspace(0, total_len, n_points)
    resampled = np.zeros((n_points, 2), dtype=np.float32)
    for i, d in enumerate(sample_dists):
        seg_idx = max(0, min(np.searchsorted(cumlen, d) - 1, len(pts) - 2))
        seg_start = cumlen[seg_idx]
        seg_len = seg_lengths[seg_idx]
        t = (d - seg_start) / seg_len if seg_len > 1e-6 else 0.0
        resampled[i] = pts[seg_idx] + t * (pts[seg_idx + 1] - pts[seg_idx])

    return resampled


def _encode_polyline_pts(pts_2d: np.ndarray, speed_limit: float,
                         cat_onehot: np.ndarray,
                         ref_x: float, ref_y: float, ref_h: float) -> np.ndarray:
    """
    Encode polyline (N×2) into per-point features (N×9).
    Features: x, y, sin(h), cos(h), speed_limit, 4×category_onehot — in ego frame.
    """
    N = len(pts_2d)
    headings = np.zeros(N, dtype=np.float32)
    for j in range(N - 1):
        dx = pts_2d[j + 1, 0] - pts_2d[j, 0]
        dy = pts_2d[j + 1, 1] - pts_2d[j, 1]
        headings[j] = math.atan2(dy, dx)
    headings[-1] = headings[-2]

    feats = np.zeros((N, cfg.D_MAP_POINT), dtype=np.float32)
    for j in range(N):
        x_e, y_e, h_e = _to_ego_frame(
            pts_2d[j, 0], pts_2d[j, 1], headings[j],
            ref_x, ref_y, ref_h
        )
        feats[j] = np.concatenate([
            [x_e, y_e, math.sin(h_e), math.cos(h_e), speed_limit],
            cat_onehot,
        ])
    return feats


# ── Map loading (adapted from data_loader._load_map_lanes) ──────────────────────

_MAP_CATEGORIES = ['LANE', 'LANE_CONNECTOR', 'INTERSECTION', 'OTHER']

_AGENT_CATEGORIES = ['VEHICLE', 'PEDESTRIAN', 'BICYCLE', 'MOTORCYCLE']

_INTERESTED_TYPES = [
    TrackedObjectType.VEHICLE,
    TrackedObjectType.PEDESTRIAN,
    TrackedObjectType.BICYCLE,
]


def _load_map_lanes(map_api: AbstractMap, ego_x: float, ego_y: float,
                    ego_h: float, ref_x: float, ref_y: float, ref_h: float,
                    radius: float = None):
    """
    Load nearby lane polylines and transform to ego-centric frame.
    Same logic as data_loader._load_map_lanes but uses live map_api.

    Returns:
        lanes:      (N_LANES, N_LANE_POINTS, 27)
        lanes_mask: (N_LANES,)
    """
    radius = radius or cfg.MAP_QUERY_RADIUS
    point = Point2D(ref_x, ref_y)
    try:
        map_objs = map_api.get_proximal_map_objects(
            point, radius, [SemanticMapLayer.LANE]
        )
        raw_lanes = map_objs[SemanticMapLayer.LANE]
    except Exception:
        raw_lanes = []

    lanes = np.zeros((cfg.N_LANES, cfg.N_LANE_POINTS, cfg.D_POLYLINE_POINT),
                     dtype=np.float32)
    lanes_mask = np.zeros(cfg.N_LANES, dtype=np.float32)

    for i, lane in enumerate(raw_lanes[:cfg.N_LANES]):
        bp = lane.baseline_path
        if bp is None:
            continue
        centerline = bp.discrete_path
        if len(centerline) < 2:
            continue

        pts_center = _resample_polyline(centerline, cfg.N_LANE_POINTS)

        _sl = getattr(lane, 'speed_limit_mps', None)
        speed_limit = float(_sl) if _sl is not None else 0.0
        lane_type = type(lane).__name__
        cat_onehot = np.zeros(len(_MAP_CATEGORIES), dtype=np.float32)
        if 'LaneConnector' in lane_type:
            cat_onehot[1] = 1.0
        elif 'Intersection' in lane_type:
            cat_onehot[2] = 1.0
        elif 'Lane' in lane_type:
            cat_onehot[0] = 1.0
        else:
            cat_onehot[3] = 1.0

        feats_center = _encode_polyline_pts(
            pts_center, speed_limit, cat_onehot, ref_x, ref_y, ref_h)

        try:
            left_path = lane.left_boundary.discrete_path
            if len(left_path) >= 2:
                pts_left = _resample_polyline(left_path, cfg.N_LANE_POINTS)
                feats_left = _encode_polyline_pts(
                    pts_left, speed_limit, cat_onehot, ref_x, ref_y, ref_h)
            else:
                feats_left = feats_center.copy()
        except Exception:
            feats_left = feats_center.copy()

        try:
            right_path = lane.right_boundary.discrete_path
            if len(right_path) >= 2:
                pts_right = _resample_polyline(right_path, cfg.N_LANE_POINTS)
                feats_right = _encode_polyline_pts(
                    pts_right, speed_limit, cat_onehot, ref_x, ref_y, ref_h)
            else:
                feats_right = feats_center.copy()
        except Exception:
            feats_right = feats_center.copy()

        lanes[i] = np.concatenate([feats_center, feats_left, feats_right], axis=-1)
        lanes_mask[i] = 1.0

    return lanes, lanes_mask


# ── Agent extraction ─────────────────────────────────────────────────────────────

def _build_agents(tracked_objects: TrackedObjects, ref_x, ref_y, ref_h,
                  time_step: float = 0.0):
    """
    Extract agent features from tracked objects in ego frame.

    Returns:
        agents: (N_AGENTS, D_AGENT=10)
        mask:   (N_AGENTS,)
    """
    agents = np.zeros((cfg.N_AGENTS, cfg.D_AGENT), dtype=np.float32)
    mask = np.zeros(cfg.N_AGENTS, dtype=np.float32)

    objs = tracked_objects.get_tracked_objects_of_types(_INTERESTED_TYPES)
    for j, ag in enumerate(objs[:cfg.N_AGENTS]):
        xe, ye, he = _to_ego_frame(
            ag.center.x, ag.center.y, ag.center.heading,
            ref_x, ref_y, ref_h
        )
        vx = ag.velocity.x if hasattr(ag, 'velocity') and ag.velocity is not None else 0.0
        vy = ag.velocity.y if hasattr(ag, 'velocity') and ag.velocity is not None else 0.0
        bw = ag.box.width if hasattr(ag, 'box') else 0.0
        bl = ag.box.length if hasattr(ag, 'box') else 0.0
        bh = ag.box.height if hasattr(ag, 'box') else 0.0
        cat_name = ag.tracked_object_type.name if hasattr(ag, 'tracked_object_type') else 'UNKNOWN'
        cat_idx = (_AGENT_CATEGORIES.index(cat_name)
                   if cat_name in _AGENT_CATEGORIES else len(_AGENT_CATEGORIES))
        agents[j] = [xe, ye, he, vx, vy, bw, bl, bh, time_step, cat_idx]
        mask[j] = 1.0

    return agents, mask


# ── Feature builder ──────────────────────────────────────────────────────────────

class CarPlannerFeatureBuilder:
    """
    Converts nuPlan PlannerInput to CarPlanner's tensor format.

    Output dict (no batch dim — the planner wrapper adds it):
        agents_now:      (N_AGENTS, D_AGENT=10)
        agents_mask:     (N_AGENTS,)
        agents_history:  (T_HIST, N_AGENTS, D_AGENT=10)
        map_lanes:       (N_LANES, N_LANE_POINTS, 27)
        map_lanes_mask:  (N_LANES,)
    """

    def __init__(self):
        self.history_horizon = cfg.T_HIST  # 10 steps

    def get_features_from_simulation(
        self,
        current_input: PlannerInput,
        initialization: PlannerInitialization,
    ) -> Dict[str, np.ndarray]:
        """
        Build CarPlanner features from a simulation step.

        Args:
            current_input: Current simulation state (ego history, observations)
            initialization: Planner init (map_api, route, goal)

        Returns:
            Dict of numpy arrays in CarPlanner's expected format.
        """
        history = current_input.history

        # Current ego state is the reference frame
        ego_state = history.ego_states[-1]
        ref_x = ego_state.rear_axle.x
        ref_y = ego_state.rear_axle.y
        ref_h = ego_state.rear_axle.heading

        # ── Agent features ───────────────────────────────────────────────────

        # Get H+1 observations (H history + current)
        observations = list(history.observations)
        H = self.history_horizon
        obs_window = observations[-(H):]  # last H observations (incl. current)

        # Build agent history: (T_HIST, N_AGENTS, D_AGENT)
        agents_history = np.zeros(
            (cfg.T_HIST, cfg.N_AGENTS, cfg.D_AGENT), dtype=np.float32)
        agents_mask = np.zeros(cfg.N_AGENTS, dtype=np.float32)

        n_obs = len(obs_window)

        for i, obs in enumerate(obs_window):
            hist_idx = (cfg.T_HIST - n_obs) + i
            t_norm = -(n_obs - 1 - i)  # negative for past, 0 for current
            ag, m = _build_agents(
                obs.tracked_objects, ref_x, ref_y, ref_h, time_step=float(t_norm))
            agents_history[hist_idx] = ag
            if i == n_obs - 1:
                agents_mask = m

        # Pad early slots with oldest available if fewer than H observations
        if n_obs < cfg.T_HIST:
            oldest = agents_history[cfg.T_HIST - n_obs]
            for k in range(cfg.T_HIST - n_obs):
                agents_history[k] = oldest

        # agents_now is the last timestep of history (t=0)
        agents_now = agents_history[-1].copy()
        agents_now[:, 8] = 0.0  # time_step = 0 for current

        # ── Map features ─────────────────────────────────────────────────────

        map_lanes, map_lanes_mask = _load_map_lanes(
            initialization.map_api,
            ego_x=ref_x, ego_y=ref_y, ego_h=ref_h,
            ref_x=ref_x, ref_y=ref_y, ref_h=ref_h,
        )

        return {
            'agents_now': agents_now,
            'agents_mask': agents_mask,
            'agents_history': agents_history,
            'map_lanes': map_lanes,
            'map_lanes_mask': map_lanes_mask,
        }
