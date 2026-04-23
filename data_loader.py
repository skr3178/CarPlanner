"""
CarPlanner baseline — data loader.
Loads nuPlan SQLite scenarios using the devkit API and produces tensors
matching the paper's data contract (phase1_report.md, submodules.md).

All positions are transformed to ego-centric frame at t=0.
"""

import os
import math
import glob
import sqlite3
import time
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import config as cfg

from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_ego_state_for_lidarpc_token_from_db,
    get_lidarpc_tokens_with_scenario_tag_from_db,
    get_tracked_objects_for_lidarpc_token_from_db,
    get_sampled_ego_states_from_db,
    get_sampled_lidarpcs_from_db,
)
from nuplan.database.nuplan_db.nuplan_db_utils import get_lidarpc_sensor_data

# Map API imports
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

# Singleton sensor source (lidar_pc / MergedPointCloud)
_SENSOR_SOURCE = get_lidarpc_sensor_data()

# Map API cache (per map name)
_MAP_API_CACHE = {}


# ── Coordinate helpers ─────────────────────────────────────────────────────────

def _to_ego_frame(x, y, heading, ref_x, ref_y, ref_heading):
    """Transform global (x, y, heading) into ego-centric frame defined by ref."""
    dx = x - ref_x
    dy = y - ref_y
    cos_h = math.cos(-ref_heading)
    sin_h = math.sin(-ref_heading)
    x_e = cos_h * dx - sin_h * dy
    y_e = sin_h * dx + cos_h * dy
    h_e = heading - ref_heading
    # normalise heading to [-pi, pi]
    h_e = (h_e + math.pi) % (2 * math.pi) - math.pi
    return x_e, y_e, h_e


# ── Map loading helpers ────────────────────────────────────────────────────────

def _get_map_name_from_db(db_path: str) -> str:
    """Query the DB to get the map version/name."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT map_version FROM log LIMIT 1")
        row = cursor.fetchone()
        if row and row[0]:
            return row[0]
    finally:
        conn.close()
    return "us-nv-las-vegas-strip"  # fallback


def _get_map_api(map_name: str):
    """Get cached map API or create new one."""
    if map_name not in _MAP_API_CACHE:
        _MAP_API_CACHE[map_name] = get_maps_api(
            map_root=cfg.MAPS_DIR,
            map_version="nuplan-maps-v1.0",
            map_name=map_name
        )
    return _MAP_API_CACHE[map_name]


def _get_traffic_light_status(db_path: str, token: str) -> dict:
    """Get traffic light status for lane connectors at the given frame.
    Returns: dict mapping lane_connector_id → status string ('green', 'red', 'yellow', 'unknown')
    """
    try:
        token_bytes = bytes.fromhex(token)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT lane_connector_id, status FROM traffic_light_status WHERE lidar_pc_token = ?",
            (token_bytes,)
        )
        result = {int(row[0]): row[1] for row in cursor.fetchall()}
        conn.close()
        return result
    except Exception:
        return {}


_TL_STATUS_TO_IDX = {'green': 0, 'yellow': 1, 'red': 2, 'unknown': 3}


def _resample_polyline(points: list, n_points: int) -> np.ndarray:
    """Resample a polyline to exactly n_points via uniform spacing."""
    if len(points) < 2:
        return np.zeros((n_points, 2), dtype=np.float32)

    pts = np.array([[p.x, p.y] for p in points], dtype=np.float32)

    # Compute cumulative arc length
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_len = cumlen[-1]

    if total_len < 1e-6:
        return np.zeros((n_points, 2), dtype=np.float32)

    # Sample at uniform arc length intervals
    sample_dists = np.linspace(0, total_len, n_points)
    resampled = np.zeros((n_points, 2), dtype=np.float32)

    for i, d in enumerate(sample_dists):
        # Find segment containing this distance
        seg_idx = np.searchsorted(cumlen, d) - 1
        seg_idx = max(0, min(seg_idx, len(pts) - 2))
        seg_start = cumlen[seg_idx]
        seg_len = seg_lengths[seg_idx]
        if seg_len > 1e-6:
            t = (d - seg_start) / seg_len
        else:
            t = 0.0
        resampled[i] = pts[seg_idx] + t * (pts[seg_idx + 1] - pts[seg_idx])

    return resampled


def _resample_polygon_ring(ring_coords: np.ndarray, n_points: int) -> np.ndarray:
    """Resample a closed polygon ring (Nx2) to n_points via arc-length (endpoint=False)."""
    if len(ring_coords) < 3:
        return np.zeros((n_points, 2), dtype=np.float32)

    pts = np.array(ring_coords, dtype=np.float32)
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_len = cumlen[-1]

    if total_len < 1e-6:
        return np.zeros((n_points, 2), dtype=np.float32)

    sample_dists = np.linspace(0, total_len, n_points, endpoint=False)
    resampled = np.zeros((n_points, 2), dtype=np.float32)
    for i, d in enumerate(sample_dists):
        seg_idx = np.searchsorted(cumlen, d) - 1
        seg_idx = max(0, min(seg_idx, len(pts) - 2))
        seg_start = cumlen[seg_idx]
        seg_len = seg_lengths[seg_idx]
        t = (d - seg_start) / seg_len if seg_len > 1e-6 else 0.0
        resampled[i] = pts[seg_idx] + t * (pts[seg_idx + 1] - pts[seg_idx])

    return resampled


def _encode_polyline_pts(pts_2d: np.ndarray, speed_limit: float,
                         cat_onehot: np.ndarray,
                         ref_x: float, ref_y: float, ref_h: float,
                         tl_onehot: np.ndarray = None) -> np.ndarray:
    """
    Encode a resampled polyline (N_LANE_POINTS × 2) into per-point features (N_LANE_POINTS × 13).
    Features: x, y, sin(h), cos(h), speed_limit, 4×category_onehot, 4×traffic_light_onehot.
    """
    if tl_onehot is None:
        tl_onehot = np.zeros(4, dtype=np.float32)
    N = len(pts_2d)
    headings = np.zeros(N, dtype=np.float32)
    for j in range(N - 1):
        dx = pts_2d[j+1, 0] - pts_2d[j, 0]
        dy = pts_2d[j+1, 1] - pts_2d[j, 1]
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
            tl_onehot,
        ])
    return feats


def _load_map_lanes(map_api, ego_x: float, ego_y: float, ego_h: float,
                    ref_x: float, ref_y: float, ref_h: float,
                    tl_status: dict = None):
    """
    Load nearby lane polylines and transform to ego-centric frame.

    Per paper §2.3: each polyline point has 3×Dm = 39 features —
    center + left_boundary + right_boundary concatenated per point.
    If a boundary is unavailable, centerline features are duplicated for that slot.

    Returns:
      lanes:      (N_LANES, N_LANE_POINTS, D_POLYLINE_POINT=39) — per-point features in ego frame
      lanes_mask: (N_LANES,) — 1 if valid lane, 0 if padding
    """
    _MAP_CATEGORIES = ['LANE', 'LANE_CONNECTOR', 'INTERSECTION', 'OTHER']

    point = Point2D(ref_x, ref_y)
    raw_lanes = []

    # Query both lanes and lane connectors
    for layer in [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]:
        try:
            map_objs = map_api.get_proximal_map_objects(
                point, cfg.MAP_QUERY_RADIUS, [layer]
            )
            raw_lanes.extend(map_objs[layer])
        except Exception:
            pass

    # Sort by centroid distance to ego so closest lanes/connectors get priority
    def _lane_dist(lane):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "invalid value encountered in cast")
                bp = lane.baseline_path
            if bp is None:
                return float('inf')
            pts = bp.discrete_path
            if len(pts) < 1:
                return float('inf')
            mid = pts[len(pts) // 2]
            return math.sqrt((mid.x - ref_x)**2 + (mid.y - ref_y)**2)
        except Exception:
            return float('inf')
    raw_lanes.sort(key=_lane_dist)

    lanes = np.zeros((cfg.N_LANES, cfg.N_LANE_POINTS, cfg.D_POLYLINE_POINT), dtype=np.float32)
    lanes_mask = np.zeros(cfg.N_LANES, dtype=np.float32)
    if tl_status is None:
        tl_status = {}

    for i, lane in enumerate(raw_lanes[:cfg.N_LANES]):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "invalid value encountered in cast")
            bp = lane.baseline_path
        if bp is None:
            continue
        centerline = bp.discrete_path
        if len(centerline) < 2:
            continue

        pts_center = _resample_polyline(centerline, cfg.N_LANE_POINTS)

        # Lane-level features (shared across center/left/right)
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

        # Traffic light status for this lane/connector
        tl_onehot = np.zeros(4, dtype=np.float32)
        lane_id = int(lane.id) if hasattr(lane, 'id') else -1
        if lane_id in tl_status:
            tl_idx = _TL_STATUS_TO_IDX.get(tl_status[lane_id], 3)
            tl_onehot[tl_idx] = 1.0

        # Encode centerline
        feats_center = _encode_polyline_pts(pts_center, speed_limit, cat_onehot,
                                            ref_x, ref_y, ref_h, tl_onehot)

        # Left boundary (fallback: duplicate centerline)
        try:
            left_path = lane.left_boundary.discrete_path
            if len(left_path) >= 2:
                pts_left = _resample_polyline(left_path, cfg.N_LANE_POINTS)
                feats_left = _encode_polyline_pts(pts_left, speed_limit, cat_onehot,
                                                  ref_x, ref_y, ref_h, tl_onehot)
            else:
                feats_left = feats_center.copy()
        except Exception:
            feats_left = feats_center.copy()

        # Right boundary (fallback: duplicate centerline)
        try:
            right_path = lane.right_boundary.discrete_path
            if len(right_path) >= 2:
                pts_right = _resample_polyline(right_path, cfg.N_LANE_POINTS)
                feats_right = _encode_polyline_pts(pts_right, speed_limit, cat_onehot,
                                                   ref_x, ref_y, ref_h, tl_onehot)
            else:
                feats_right = feats_center.copy()
        except Exception:
            feats_right = feats_center.copy()

        # Concatenate: [center(9) | left(9) | right(9)] per point → (N_LANE_POINTS, 27)
        lanes[i] = np.concatenate([feats_center, feats_left, feats_right], axis=-1)
        lanes_mask[i] = 1.0

    return lanes, lanes_mask


def _load_map_polygons(map_api, ego_x: float, ego_y: float, ego_h: float,
                       ref_x: float, ref_y: float, ref_h: float):
    """
    Load nearby polygon map elements (crosswalks, stop lines, intersections).

    Per paper: Nm,2 × Np × Dm = N_POLYGONS × N_LANE_POINTS × D_MAP_POINT
    Each polygon ring is resampled to N_LANE_POINTS points, encoded to D_MAP_POINT dims.
    Sorted by centroid distance to ego, closest N_POLYGONS kept.

    Returns:
      polygons:      (N_POLYGONS, N_LANE_POINTS, D_MAP_POINT=9)
      polygons_mask: (N_POLYGONS,) — 1 if valid, 0 if padding
    """
    _POLYGON_LAYERS = [
        (SemanticMapLayer.CROSSWALK, 0),
        (SemanticMapLayer.STOP_LINE, 1),
        (SemanticMapLayer.INTERSECTION, 2),
    ]
    _POLYGON_CATEGORIES = ['CROSSWALK', 'STOP_LINE', 'INTERSECTION', 'OTHER']

    point = Point2D(ref_x, ref_y)
    collected = []

    for layer, cat_idx in _POLYGON_LAYERS:
        try:
            map_objs = map_api.get_proximal_map_objects(
                point, cfg.MAP_QUERY_RADIUS, [layer]
            )
            raw_polys = map_objs[layer]
        except Exception:
            continue

        cat_onehot = np.zeros(len(_POLYGON_CATEGORIES), dtype=np.float32)
        cat_onehot[cat_idx] = 1.0

        for poly_obj in raw_polys:
            try:
                ring = np.array(poly_obj.polygon.exterior.coords, dtype=np.float64)[:, :2]
            except Exception:
                continue
            if len(ring) < 3:
                continue

            pts_2d = _resample_polygon_ring(ring, cfg.N_LANE_POINTS)
            feats = _encode_polyline_pts(pts_2d, speed_limit=0.0, cat_onehot=cat_onehot,
                                         ref_x=ref_x, ref_y=ref_y, ref_h=ref_h)
            cx = pts_2d[:, 0].mean()
            cy = pts_2d[:, 1].mean()
            dist = math.sqrt((cx - ref_x)**2 + (cy - ref_y)**2)
            collected.append((dist, feats))

    collected.sort(key=lambda x: x[0])

    polygons = np.zeros((cfg.N_POLYGONS, cfg.N_LANE_POINTS, cfg.D_POLYGON_POINT), dtype=np.float32)
    polygons_mask = np.zeros(cfg.N_POLYGONS, dtype=np.float32)

    for i, (_, feats) in enumerate(collected[:cfg.N_POLYGONS]):
        polygons[i] = feats
        polygons_mask[i] = 1.0

    return polygons, polygons_mask


# ── Route extraction via lane graph search (Section 3.3.2) ────────────────────

def _trace_route_forward(start_lane, max_depth: int, max_length: float) -> list:
    """
    Trace a single route forward from start_lane by following the lane graph.
    At each junction, pick the most heading-aligned successor.

    Returns list of lane objects forming the route.
    """
    route = [start_lane]
    total_length = 0.0

    current = start_lane
    visited = {current.id}

    for _ in range(max_depth - 1):
        bp = current.baseline_path
        if bp is None:
            break
        pts = bp.discrete_path
        if len(pts) >= 2:
            for j in range(len(pts) - 1):
                total_length += math.sqrt(
                    (pts[j+1].x - pts[j].x)**2 + (pts[j+1].y - pts[j].y)**2
                )
        if total_length >= max_length:
            break

        last_pt = pts[-1] if pts else None
        if last_pt is None:
            break
        end_heading = math.atan2(
            pts[-1].y - pts[-2].y, pts[-1].x - pts[-2].x
        ) if len(pts) >= 2 else 0.0

        best_lane = None
        best_align = -2.0
        for connector in current.outgoing_edges:
            for next_lane in connector.outgoing_edges:
                if next_lane.id in visited:
                    continue
                nbp = next_lane.baseline_path
                if nbp is None:
                    continue
                npts = nbp.discrete_path
                if len(npts) < 2:
                    continue
                next_heading = math.atan2(
                    npts[1].y - npts[0].y, npts[1].x - npts[0].x
                )
                align = math.cos(next_heading - end_heading)
                if align > best_align:
                    best_align = align
                    best_lane = next_lane

        if best_lane is None:
            break
        visited.add(best_lane.id)
        route.append(best_lane)
        current = best_lane

    return route


def _route_to_polyline(route_lanes: list, n_points: int,
                       ref_x: float, ref_y: float, ref_h: float,
                       tl_status: dict = None) -> np.ndarray:
    """
    Concatenate lane segments into a single polyline and encode to (n_points, D_POLYLINE_POINT).
    """
    _MAP_CATEGORIES = ['LANE', 'LANE_CONNECTOR', 'INTERSECTION', 'OTHER']
    if tl_status is None:
        tl_status = {}

    all_center_pts = []
    all_left_pts = []
    all_right_pts = []
    all_speed_limits = []
    all_cat_onehots = []
    all_tl_onehots = []

    for lane in route_lanes:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "invalid value encountered in cast")
            bp = lane.baseline_path
        if bp is None:
            continue
        center_pts = bp.discrete_path
        if len(center_pts) < 2:
            continue

        _sl = getattr(lane, 'speed_limit_mps', None)
        speed_limit = float(_sl) if _sl is not None else 0.0
        lane_type = type(lane).__name__
        cat_onehot = np.zeros(len(_MAP_CATEGORIES), dtype=np.float32)
        if 'LaneConnector' in lane_type:
            cat_onehot[1] = 1.0
        elif 'Lane' in lane_type:
            cat_onehot[0] = 1.0
        else:
            cat_onehot[3] = 1.0

        tl_onehot = np.zeros(4, dtype=np.float32)
        lane_id = int(lane.id) if hasattr(lane, 'id') else -1
        if lane_id in tl_status:
            tl_idx = _TL_STATUS_TO_IDX.get(tl_status[lane_id], 3)
            tl_onehot[tl_idx] = 1.0

        for pt in center_pts:
            all_center_pts.append([pt.x, pt.y])
            all_speed_limits.append(speed_limit)
            all_cat_onehots.append(cat_onehot.copy())
            all_tl_onehots.append(tl_onehot.copy())

        try:
            left_path = lane.left_boundary.discrete_path
            if len(left_path) >= 2:
                for pt in left_path:
                    all_left_pts.append([pt.x, pt.y])
            else:
                for pt in center_pts:
                    all_left_pts.append([pt.x, pt.y])
        except Exception:
            for pt in center_pts:
                all_left_pts.append([pt.x, pt.y])

        try:
            right_path = lane.right_boundary.discrete_path
            if len(right_path) >= 2:
                for pt in right_path:
                    all_right_pts.append([pt.x, pt.y])
            else:
                for pt in center_pts:
                    all_right_pts.append([pt.x, pt.y])
        except Exception:
            for pt in center_pts:
                all_right_pts.append([pt.x, pt.y])

    if len(all_center_pts) < 2:
        return np.zeros((n_points, cfg.D_POLYLINE_POINT), dtype=np.float32)

    center_arr = np.array(all_center_pts, dtype=np.float32)
    left_arr = np.array(all_left_pts, dtype=np.float32)
    right_arr = np.array(all_right_pts, dtype=np.float32)

    center_resampled = _resample_raw_array(center_arr, n_points)
    left_resampled = _resample_raw_array(left_arr, n_points)
    right_resampled = _resample_raw_array(right_arr, n_points)

    speed_arr = np.array(all_speed_limits, dtype=np.float32)
    cat_arr = np.array(all_cat_onehots, dtype=np.float32)
    tl_arr = np.array(all_tl_onehots, dtype=np.float32)
    speed_resampled = _resample_scalar_along_polyline(center_arr, speed_arr, center_resampled)
    cat_resampled = _resample_categorical_along_polyline(center_arr, cat_arr, center_resampled)
    tl_resampled = _resample_categorical_along_polyline(center_arr, tl_arr, center_resampled)

    feats = np.zeros((n_points, cfg.D_POLYLINE_POINT), dtype=np.float32)
    Dm = cfg.D_MAP_POINT
    for offset, pts_2d in [(0, center_resampled), (Dm, left_resampled), (2 * Dm, right_resampled)]:
        headings = np.zeros(n_points, dtype=np.float32)
        for j in range(n_points - 1):
            dx = pts_2d[j+1, 0] - pts_2d[j, 0]
            dy = pts_2d[j+1, 1] - pts_2d[j, 1]
            headings[j] = math.atan2(dy, dx)
        if n_points > 1:
            headings[-1] = headings[-2]

        for j in range(n_points):
            x_e, y_e, h_e = _to_ego_frame(
                pts_2d[j, 0], pts_2d[j, 1], headings[j],
                ref_x, ref_y, ref_h
            )
            feats[j, offset:offset+Dm] = np.concatenate([
                [x_e, y_e, math.sin(h_e), math.cos(h_e), speed_resampled[j]],
                cat_resampled[j],
                tl_resampled[j],
            ])

    return feats


def _resample_raw_array(pts: np.ndarray, n_points: int) -> np.ndarray:
    """Resample a (M, 2) array of raw global coords to n_points via uniform arc length."""
    if len(pts) < 2:
        return np.zeros((n_points, 2), dtype=np.float32)

    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_len = cumlen[-1]

    if total_len < 1e-6:
        return np.tile(pts[0], (n_points, 1))

    sample_dists = np.linspace(0, total_len, n_points)
    resampled = np.zeros((n_points, 2), dtype=np.float32)
    for i, d in enumerate(sample_dists):
        seg_idx = np.searchsorted(cumlen, d) - 1
        seg_idx = max(0, min(seg_idx, len(pts) - 2))
        seg_start = cumlen[seg_idx]
        seg_len = seg_lengths[seg_idx]
        t = (d - seg_start) / seg_len if seg_len > 1e-6 else 0.0
        resampled[i] = pts[seg_idx] + t * (pts[seg_idx + 1] - pts[seg_idx])
    return resampled


def _resample_scalar_along_polyline(orig_pts, orig_vals, resampled_pts):
    """Nearest-neighbor resample scalar values along a polyline."""
    n = len(resampled_pts)
    result = np.zeros(n, dtype=np.float32)
    for i in range(n):
        dists = np.sqrt(
            (orig_pts[:, 0] - resampled_pts[i, 0])**2 +
            (orig_pts[:, 1] - resampled_pts[i, 1])**2
        )
        result[i] = orig_vals[np.argmin(dists)]
    return result


def _resample_categorical_along_polyline(orig_pts, orig_cats, resampled_pts):
    """Nearest-neighbor resample categorical arrays along a polyline."""
    n = len(resampled_pts)
    D = orig_cats.shape[1]
    result = np.zeros((n, D), dtype=np.float32)
    for i in range(n):
        dists = np.sqrt(
            (orig_pts[:, 0] - resampled_pts[i, 0])**2 +
            (orig_pts[:, 1] - resampled_pts[i, 1])**2
        )
        result[i] = orig_cats[np.argmin(dists)]
    return result


def _extract_routes(map_api, ref_x: float, ref_y: float, ref_h: float,
                    tl_status: dict = None, gt_trajectory: np.ndarray = None):
    """
    Extract up to N_LAT connected routes via lane graph search (Section 3.3.2).

    Algorithm:
      1. Find ego's current lane via get_one_map_object
      2. Get all lanes in the current roadblock (parallel lanes = lateral options)
      3. For each starting lane, trace forward through lane graph
      4. Encode each route as a resampled polyline with D_POLYLINE_POINT features
      5. Sort by lateral offset and subsample to N_LAT
      6. Compute positive lateral mode label from GT trajectory

    Returns:
        route_polylines: (N_LAT, N_ROUTE_POINTS, D_POLYLINE_POINT)
        route_mask: (N_LAT,)
        positive_lat_idx: int in [0, N_LAT)
    """
    N_LAT = cfg.N_LAT
    N_RP = cfg.N_ROUTE_POINTS
    D_PP = cfg.D_POLYLINE_POINT

    route_polylines = np.zeros((N_LAT, N_RP, D_PP), dtype=np.float32)
    route_mask = np.zeros(N_LAT, dtype=np.float32)

    point = Point2D(ref_x, ref_y)

    try:
        ego_lane = map_api.get_one_map_object(point, SemanticMapLayer.LANE)
    except AssertionError:
        candidates = map_api.get_all_map_objects(point, SemanticMapLayer.LANE)
        ego_lane = candidates[0] if candidates else None
    if ego_lane is None:
        try:
            ego_lane = map_api.get_one_map_object(point, SemanticMapLayer.LANE_CONNECTOR)
        except AssertionError:
            candidates = map_api.get_all_map_objects(point, SemanticMapLayer.LANE_CONNECTOR)
            ego_lane = candidates[0] if candidates else None
    if ego_lane is None:
        lat_idx = _bin_lateral_offset(gt_trajectory[-1, 1]) if gt_trajectory is not None else 0
        return route_polylines, route_mask, lat_idx

    if hasattr(ego_lane, 'parent') and ego_lane.parent is not None:
        start_lanes = list(ego_lane.parent.interior_edges)
    else:
        start_lanes = [ego_lane]

    routes_with_offset = []
    for lane in start_lanes:
        route_lanes = _trace_route_forward(
            lane, cfg.ROUTE_MAX_DEPTH, cfg.ROUTE_MAX_LENGTH_M
        )
        polyline = _route_to_polyline(
            route_lanes, N_RP, ref_x, ref_y, ref_h, tl_status
        )
        if np.abs(polyline[:, :2]).sum() < 1e-6:
            continue

        first_valid = polyline[np.abs(polyline[:, 0]) + np.abs(polyline[:, 1]) > 1e-6]
        if len(first_valid) < 2:
            continue

        dists = np.sqrt(first_valid[:, 0]**2 + first_valid[:, 1]**2)
        closest_idx = int(np.argmin(dists))

        heading = math.atan2(first_valid[closest_idx, 2], first_valid[closest_idx, 3])
        if abs(heading) > math.pi / 2:
            continue

        lat_y = float(first_valid[closest_idx, 1])
        routes_with_offset.append((lat_y, polyline))

    routes_with_offset.sort(key=lambda x: x[0])

    if len(routes_with_offset) > N_LAT:
        indices = np.linspace(0, len(routes_with_offset) - 1, N_LAT).astype(int).tolist()
        routes_with_offset = [routes_with_offset[i] for i in indices]

    for i, (_, polyline) in enumerate(routes_with_offset):
        route_polylines[i] = polyline
        route_mask[i] = 1.0

    positive_lat_idx = 0
    if gt_trajectory is not None and len(routes_with_offset) > 0:
        gt_endpoint = gt_trajectory[-1, :2]
        best_dist = float('inf')
        for i, (_, polyline) in enumerate(routes_with_offset):
            route_xy = polyline[:, :2]
            valid = route_xy[np.abs(route_xy[:, 0]) + np.abs(route_xy[:, 1]) > 1e-6]
            if len(valid) == 0:
                continue
            dist = float(np.min(np.sqrt(
                (valid[:, 0] - gt_endpoint[0])**2 + (valid[:, 1] - gt_endpoint[1])**2
            )))
            if dist < best_dist:
                best_dist = dist
                positive_lat_idx = i

    return route_polylines, route_mask, positive_lat_idx


# ── Mode assignment (winner-takes-all, Algorithm 1) ────────────────────────────

def _collect_candidate_lanes(map_lanes: np.ndarray,
                             map_lanes_mask: np.ndarray):
    """
    Step 1 of route-proximity matching (paper Section 3.3.2).

    Collect candidate lanes whose heading roughly matches ego's current heading.
    Filters out oncoming lanes and cross traffic.

    Args:
        map_lanes:      (N_LANES, N_PTS, 9) in ego frame
        map_lanes_mask: (N_LANES,)
    Returns:
        candidates: list of dicts with keys:
            'valid_pts': (M, 9) — non-padded lane points
            'closest_y': float  — y-offset of the lane point nearest to ego
            'closest_dist': float — distance from ego to nearest lane point
    """
    candidates = []
    for i in range(len(map_lanes_mask)):
        if map_lanes_mask[i] < 0.5:
            continue
        lane = map_lanes[i]                                       # (N_PTS, 9)
        valid = lane[np.abs(lane[:, 0]) + np.abs(lane[:, 1]) > 1e-6]
        if len(valid) < 2:
            continue

        # Find closest point to ego (origin)
        dists = np.sqrt(valid[:, 0] ** 2 + valid[:, 1] ** 2)
        closest_idx = int(np.argmin(dists))

        # Check heading alignment — ego faces +x (heading ≈ 0 in ego frame)
        heading = math.atan2(valid[closest_idx, 2], valid[closest_idx, 3])
        if abs(heading) > math.pi / 3:                           # >60° = not a candidate
            continue

        candidates.append({
            'valid_pts':    valid,
            'closest_y':    float(valid[closest_idx, 1]),
            'closest_dist': float(dists[closest_idx]),
        })
    return candidates


def _match_endpoint_to_route(candidates: list,
                             gt_endpoint: np.ndarray) -> int:
    """
    Steps 2-3 of route-proximity matching (paper Section 3.4).

    Step 2: For each candidate lane, compute min distance from GT endpoint
            to any point on that lane's polyline. Pick the closest = matched lane.
    Step 3: Compute the lateral offset between the matched lane and ego's
            current lane (the candidate closest to ego). Bin that offset
            into LAT_BIN_EDGES to get c_lat*.

    This is robust on curved roads because we never measure raw y-offset —
    we ask "which polyline did the human end up on?" A left turn on a curved
    road correctly gets labelled "lane keep" if the human stayed in their lane.

    Args:
        candidates: output of _collect_candidate_lanes
        gt_endpoint: (3,) — GT endpoint (x, y, yaw) in ego frame
    Returns:
        lat_idx in [0, N_LAT), or -1 if no candidates found
    """
    if len(candidates) == 0:
        return -1

    gt_xy = gt_endpoint[:2]

    # Step 2: find which candidate lane the GT endpoint is closest to
    best_cand_idx = -1
    best_dist = float('inf')
    for i, cand in enumerate(candidates):
        lane_xy = cand['valid_pts'][:, :2]
        dist = float(np.min(np.sqrt(
            (lane_xy[:, 0] - gt_xy[0]) ** 2 + (lane_xy[:, 1] - gt_xy[1]) ** 2
        )))
        if dist < best_dist:
            best_dist = dist
            best_cand_idx = i

    # Step 3: label relative to ego's current lane
    # Ego's current lane = the candidate closest to the origin
    ego_lane_y = min(candidates, key=lambda c: c['closest_dist'])['closest_y']
    matched_lane_y = candidates[best_cand_idx]['closest_y']

    # Lateral offset = matched lane y − ego lane y (positive = right)
    lat_offset = matched_lane_y - ego_lane_y

    # Bin into LAT_BIN_EDGES
    lat_idx = 0
    for i, edge in enumerate(cfg.LAT_BIN_EDGES[1:]):
        if lat_offset < edge:
            lat_idx = i
            break
    else:
        lat_idx = cfg.N_LAT - 1

    return lat_idx


def _assign_mode(gt_traj: np.ndarray,
                 map_lanes: np.ndarray = None,
                 map_lanes_mask: np.ndarray = None) -> int:
    """
    Assign positive mode c* (Algorithm 1, lines 15-18).

    c_lon: from GT endpoint displacement (longitudinal speed bin).
    c_lat: by route-proximity matching (paper Section 3.3.2 + 3.4):
           - Collect candidate lanes heading-aligned with ego
           - Match GT endpoint to nearest candidate lane polyline
           - Label relative to ego's current lane
    c*:    = c_lon * N_LAT + c_lat.

    Falls back to raw GT endpoint y-offset when map data unavailable.

    Args:
        gt_traj:        (T, 3) in ego frame
        map_lanes:      (N_LANES, N_PTS, 9) in ego frame (optional)
        map_lanes_mask: (N_LANES,) validity mask (optional)
    Returns:
        c* in [0, N_MODES).
    """
    endpoint = gt_traj[-1]                                        # (x, y, yaw)

    # Longitudinal mode: total displacement distance bucketed into N_LON bins
    dist = math.sqrt(endpoint[0] ** 2 + endpoint[1] ** 2)
    lon_idx = min(
        int(dist / (cfg.MAX_SPEED * cfg.T_FUTURE * 0.1 / cfg.N_LON)),
        cfg.N_LON - 1,
    )

    # Lateral mode: route-proximity matching if map available
    if (map_lanes is not None and map_lanes_mask is not None
            and map_lanes_mask.sum() > 0):
        candidates = _collect_candidate_lanes(map_lanes, map_lanes_mask)
        lat_idx = _match_endpoint_to_route(candidates, endpoint)
        if lat_idx < 0:                                           # no candidates
            lat_idx = _bin_lateral_offset(endpoint[1])
    else:
        lat_idx = _bin_lateral_offset(endpoint[1])

    return lon_idx * cfg.N_LAT + lat_idx


def _bin_lateral_offset(lat_y: float) -> int:
    """Bin a raw y-offset into N_LAT lateral bins (fallback)."""
    lat_idx = 0
    for i, edge in enumerate(cfg.LAT_BIN_EDGES[1:]):
        if lat_y < edge:
            lat_idx = i
            break
    else:
        lat_idx = cfg.N_LAT - 1
    return lat_idx


# ── Per-sample loading ─────────────────────────────────────────────────────────

def _load_sample(db_path: str, token: str):
    """
    Load one sample (ego history, agents, GT future) for a given anchor token.
    Returns a dict of numpy arrays, or None if the frame lacks enough context.
    """
    sensor_source = _SENSOR_SOURCE

    # ── Current ego state (reference frame) ──────────────────────────────────
    try:
        current_ego = get_ego_state_for_lidarpc_token_from_db(db_path, token)
    except Exception:
        return None

    ref_x = current_ego.rear_axle.x
    ref_y = current_ego.rear_axle.y
    ref_h = current_ego.rear_axle.heading

    # ── Ego history (T_HIST frames, future=False includes anchor at idx 0) ───
    try:
        hist_states = list(get_sampled_ego_states_from_db(
            db_path, token, sensor_source,
            list(range(0, cfg.T_HIST * 2, 2)), future=False  # 10Hz: every 2nd frame of 20Hz DB
        ))
    except Exception:
        return None

    if len(hist_states) == 0:
        return None

    # Build ego_history tensor: pad oldest end if fewer than T_HIST frames
    ego_history = np.zeros((cfg.T_HIST, 4), dtype=np.float32)
    for i, state in enumerate(hist_states):
        xe, ye, he = _to_ego_frame(
            state.rear_axle.x, state.rear_axle.y, state.rear_axle.heading,
            ref_x, ref_y, ref_h
        )
        spd = state.dynamic_car_state.rear_axle_velocity_2d.x
        # hist_states is sorted ascending (oldest first), place in order
        idx = cfg.T_HIST - len(hist_states) + i
        ego_history[idx] = [xe, ye, he, spd]
    # Repeat first valid row to fill leading zeros
    if len(hist_states) < cfg.T_HIST:
        first_valid = cfg.T_HIST - len(hist_states)
        ego_history[:first_valid] = ego_history[first_valid]

    # ── GT future trajectory (T_FUTURE frames, future=True, skip anchor) ─────
    try:
        fut_states = list(get_sampled_ego_states_from_db(
            db_path, token, sensor_source,
            list(range(2, cfg.T_FUTURE * 2 + 1, 2)), future=True  # 10Hz: every 2nd frame of 20Hz DB
        ))
    except Exception:
        return None

    if len(fut_states) < cfg.T_FUTURE:
        return None  # skip frames without full future horizon

    gt_trajectory = np.zeros((cfg.T_FUTURE, 3), dtype=np.float32)
    for i, state in enumerate(fut_states):
        xe, ye, he = _to_ego_frame(
            state.rear_axle.x, state.rear_axle.y, state.rear_axle.heading,
            ref_x, ref_y, ref_h
        )
        gt_trajectory[i] = [xe, ye, he]

    # ── Agents at current frame (t=0) ─────────────────────────────────────────
    # Da=14: x, y, sin_h, cos_h, vx, vy, box_w, box_l, box_h, time_step, 4×category_onehot
    _AGENT_CATEGORIES = ['VEHICLE', 'PEDESTRIAN', 'BICYCLE', 'MOTORCYCLE']

    def _load_agents_at_token(tok: str, time_step: float = 0.0) -> tuple:
        """Returns (agents array (N, Da=14), mask (N,)) in initial ego frame."""
        try:
            raw = list(get_tracked_objects_for_lidarpc_token_from_db(db_path, tok))
        except Exception:
            raw = []
        arr  = np.zeros((cfg.N_AGENTS, cfg.D_AGENT), dtype=np.float32)
        mask = np.zeros(cfg.N_AGENTS, dtype=np.float32)
        for j, ag in enumerate(raw[:cfg.N_AGENTS]):
            xe, ye, he = _to_ego_frame(
                ag.center.x, ag.center.y, ag.center.heading,
                ref_x, ref_y, ref_h
            )
            vx = ag.velocity.x if hasattr(ag, 'velocity') and ag.velocity else 0.0
            vy = ag.velocity.y if hasattr(ag, 'velocity') and ag.velocity else 0.0
            bw = ag.box.width  if hasattr(ag, 'box') else 0.0
            bl = ag.box.length if hasattr(ag, 'box') else 0.0
            bh = ag.box.height if hasattr(ag, 'box') else 0.0
            cat_name = ag.tracked_object_type.name if hasattr(ag, 'tracked_object_type') else 'UNKNOWN'
            cat_idx = _AGENT_CATEGORIES.index(cat_name) if cat_name in _AGENT_CATEGORIES else len(_AGENT_CATEGORIES) - 1
            cat_onehot = np.zeros(len(_AGENT_CATEGORIES), dtype=np.float32)
            cat_onehot[cat_idx] = 1.0
            arr[j] = np.concatenate([
                [xe, ye, math.sin(he), math.cos(he), vx, vy, bw, bl, bh, time_step],
                cat_onehot,
            ])
            mask[j] = 1.0
        return arr, mask

    agents_now, agents_mask = _load_agents_at_token(token)

    # ── Agent history (t=-H+1..0): load real past frames from DB ─────────────
    # Paper Section 3.1: "each agent maintains poses for the past H time steps"
    # Indices: -H+1, -H+2, ..., -1, 0  (H frames total, last = current t=0)
    agents_history = np.stack([agents_now] * cfg.T_HIST, axis=0)   # fallback: repeat t=0
    try:
        past_pcs = list(get_sampled_lidarpcs_from_db(
            db_path, token, sensor_source,
            list(range(2, cfg.T_HIST * 2 + 1, 2)), future=False  # 10Hz: every 2nd frame of 20Hz DB
        ))
        # past_pcs is ordered oldest→newest (furthest past first)
        # Pad from the left with t=0 if fewer than H past frames are available
        n_past = len(past_pcs)
        for i, pc in enumerate(past_pcs):
            hist_idx = (cfg.T_HIST - n_past) + i   # place in correct slot
            t_norm   = -(n_past - i)                # time index: -n_past...-1
            arr_h, _ = _load_agents_at_token(pc.token, time_step=float(t_norm))
            agents_history[hist_idx] = arr_h
        # Last slot is always t=0 (agents_now, already set by fallback)
        agents_history[-1] = agents_now
        # Fill any remaining early slots (if fewer than H past frames) with oldest available
        if n_past < cfg.T_HIST - 1:
            oldest = agents_history[cfg.T_HIST - n_past] if n_past > 0 else agents_now
            for i in range(cfg.T_HIST - n_past):
                agents_history[i] = oldest
    except Exception:
        pass  # fallback already set above

    # ── GT agent futures (t=1..T) — used by AutoregressivePolicy IVM ──────────
    # Get future lidar_pc tokens, then load agents at each step
    agents_seq = np.zeros((cfg.T_FUTURE, cfg.N_AGENTS, cfg.D_AGENT), dtype=np.float32)
    try:
        future_pcs = list(get_sampled_lidarpcs_from_db(
            db_path, token, sensor_source,
            list(range(2, cfg.T_FUTURE * 2 + 1, 2)), future=True  # 10Hz: every 2nd frame of 20Hz DB
        ))
        for step_i, pc in enumerate(future_pcs[:cfg.T_FUTURE]):
            arr_t, _ = _load_agents_at_token(pc.token, time_step=float(step_i + 1))
            agents_seq[step_i] = arr_t
    except Exception:
        # Fallback: hold agents static (current frame repeated)
        agents_seq[:] = agents_now[np.newaxis]

    # ── BEV raster (zeros; real rasterisation deferred) ───────────────────────
    map_raster = np.zeros((cfg.BEV_C, cfg.BEV_H, cfg.BEV_W), dtype=np.float32)

    # ── Traffic light status for this frame ─────────────────────────────────
    tl_status = _get_traffic_light_status(db_path, token)

    # ── Map lane loading ──────────────────────────────────────────────────────
    try:
        map_name = _get_map_name_from_db(db_path)
        map_api = _get_map_api(map_name)
        map_lanes, map_lanes_mask = _load_map_lanes(
            map_api, ref_x, ref_y, ref_h, ref_x, ref_y, ref_h,
            tl_status=tl_status
        )
    except Exception:
        map_api = None
        map_lanes = np.zeros((cfg.N_LANES, cfg.N_LANE_POINTS, cfg.D_POLYLINE_POINT), dtype=np.float32)
        map_lanes_mask = np.zeros(cfg.N_LANES, dtype=np.float32)

    # ── Map polygon loading ──────────────────────────────────────────────────
    try:
        if map_api is None:
            map_name = _get_map_name_from_db(db_path)
            map_api = _get_map_api(map_name)
        map_polygons, map_polygons_mask = _load_map_polygons(
            map_api, ref_x, ref_y, ref_h, ref_x, ref_y, ref_h
        )
    except Exception:
        map_polygons = np.zeros((cfg.N_POLYGONS, cfg.N_LANE_POINTS, cfg.D_POLYGON_POINT), dtype=np.float32)
        map_polygons_mask = np.zeros(cfg.N_POLYGONS, dtype=np.float32)

    # ── Route extraction via lane graph search ──────────────────────────────
    try:
        if map_api is None:
            map_name = _get_map_name_from_db(db_path)
            map_api = _get_map_api(map_name)
        route_polylines, route_mask, positive_lat_idx = _extract_routes(
            map_api, ref_x, ref_y, ref_h,
            tl_status=tl_status, gt_trajectory=gt_trajectory
        )
    except Exception:
        route_polylines = np.zeros(
            (cfg.N_LAT, cfg.N_ROUTE_POINTS, cfg.D_POLYLINE_POINT), dtype=np.float32
        )
        route_mask = np.zeros(cfg.N_LAT, dtype=np.float32)
        positive_lat_idx = 0

    # ── Mode assignment (use route-based lateral label) ───────────────────
    endpoint = gt_trajectory[-1]
    dist = math.sqrt(endpoint[0] ** 2 + endpoint[1] ** 2)
    lon_idx = min(
        int(dist / (cfg.MAX_SPEED * cfg.T_FUTURE * 0.1 / cfg.N_LON)),
        cfg.N_LON - 1,
    )
    if route_mask.sum() > 0:
        lat_idx = positive_lat_idx
    else:
        lat_idx = _bin_lateral_offset(endpoint[1])
    mode_label = lon_idx * cfg.N_LAT + lat_idx

    return {
        'map_raster':          map_raster,
        'ego_history':         ego_history,
        'agents_history':      agents_history,
        'agents_history_mask': agents_mask,
        'agents_now':          agents_now,
        'agents_seq':          agents_seq,
        'gt_trajectory':       gt_trajectory,
        'mode_label':          np.int64(mode_label),
        'map_lanes':           map_lanes,
        'map_lanes_mask':      map_lanes_mask,
        'map_polygons':        map_polygons,
        'map_polygons_mask':   map_polygons_mask,
        'route_polylines':     route_polylines,
        'route_mask':          route_mask,
    }


# ── Dataset ────────────────────────────────────────────────────────────────────

class NuPlanCarPlannerDataset(Dataset):
    """
    Loads nuPlan scenario frames from .db files.
    Builds an index of (db_path, token) pairs at construction time.
    """

    def __init__(self, split: str, max_per_file: int = None):
        """
        split: 'mini' | 'train_boston' | 'train_pittsburgh' | 'train_singapore' | 'val14' | 'test14_random' | 'reduced_val14'
        max_per_file: cap on tokens loaded per .db file (None = all)
        """
        allowed_tokens = None  # set → filter index to these scenario tokens only

        if split == 'mini':
            db_dir = cfg.MINI_DIR
            if max_per_file is None:
                max_per_file = cfg.MAX_SAMPLES_PER_FILE_MINI
        elif split == 'train_boston':
            db_dir = cfg.TRAIN_DIR
            if max_per_file is None:
                max_per_file = cfg.MAX_SAMPLES_PER_FILE_TRAIN
        elif split == 'train_pittsburgh':
            db_dir = cfg.TRAIN_PITTSBURGH_DIR
            if max_per_file is None:
                max_per_file = cfg.MAX_SAMPLES_PER_FILE_TRAIN
        elif split == 'train_singapore':
            db_dir = cfg.TRAIN_SINGAPORE_DIR
            if max_per_file is None:
                max_per_file = cfg.MAX_SAMPLES_PER_FILE_TRAIN
        elif split == 'val14':
            db_dir = cfg.VAL_DIR
            max_per_file = max_per_file or 10**9  # no per-file cap for val
            import yaml
            with open(cfg.VAL14_YAML, 'r') as f:
                y = yaml.safe_load(f)
            allowed_tokens = set(str(t) for t in y.get('scenario_tokens', []))
            print(f"[DataLoader] val14: loaded {len(allowed_tokens)} tokens from {cfg.VAL14_YAML}")
        elif split == 'test14_random':
            db_dir = cfg.VAL_DIR
            max_per_file = max_per_file or 10**9
            import yaml
            with open(cfg.TEST14_RANDOM_YAML, 'r') as f:
                y = yaml.safe_load(f)
            allowed_tokens = set(str(t) for t in y.get('scenario_tokens', []))
            print(f"[DataLoader] test14_random: loaded {len(allowed_tokens)} tokens from {cfg.TEST14_RANDOM_YAML}")
        elif split == 'reduced_val14':
            db_dir = cfg.VAL_DIR
            max_per_file = max_per_file or 10**9
            import yaml
            with open(cfg.REDUCED_VAL14_YAML, 'r') as f:
                y = yaml.safe_load(f)
            allowed_tokens = set(str(t) for t in y.get('scenario_tokens', []))
            print(f"[DataLoader] reduced_val14: loaded {len(allowed_tokens)} tokens from {cfg.REDUCED_VAL14_YAML}")
        else:
            raise ValueError(f"Unknown split: {split}")

        db_files = sorted(glob.glob(os.path.join(db_dir, "*.db")))
        if not db_files:
            raise FileNotFoundError(f"No .db files found in {db_dir}")

        print(f"[DataLoader] Building index for '{split}' split ({len(db_files)} files)...")
        self._index = []  # list of (db_path, token)

        for db_path in db_files:
            try:
                seen = set()
                count = 0
                for scenario_type, token in get_lidarpc_tokens_with_scenario_tag_from_db(db_path):
                    if token in seen:
                        continue
                    seen.add(token)
                    if allowed_tokens is not None and token not in allowed_tokens:
                        continue
                    self._index.append((db_path, token))
                    count += 1
                    if count >= max_per_file:
                        break
            except Exception as e:
                print(f"  Warning: skipping {os.path.basename(db_path)}: {e}")

        print(f"[DataLoader] Index built: {len(self._index)} candidate frames.")

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx: int):
        db_path, token = self._index[idx]
        sample = _load_sample(db_path, token)

        if sample is None:
            return {
                'map_raster':          torch.zeros(cfg.BEV_C, cfg.BEV_H, cfg.BEV_W),
                'ego_history':         torch.zeros(cfg.T_HIST, 4),
                'agents_history':      torch.zeros(cfg.T_HIST, cfg.N_AGENTS, cfg.D_AGENT),
                'agents_history_mask': torch.zeros(cfg.N_AGENTS),
                'agents_now':          torch.zeros(cfg.N_AGENTS, cfg.D_AGENT),
                'agents_seq':          torch.zeros(cfg.T_FUTURE, cfg.N_AGENTS, cfg.D_AGENT),
                'gt_trajectory':       torch.zeros(cfg.T_FUTURE, 3),
                'mode_label':          torch.zeros(1, dtype=torch.long).squeeze(),
                'map_lanes':           torch.zeros(cfg.N_LANES, cfg.N_LANE_POINTS, cfg.D_POLYLINE_POINT),
                'map_lanes_mask':      torch.zeros(cfg.N_LANES),
                'map_polygons':        torch.zeros(cfg.N_POLYGONS, cfg.N_LANE_POINTS, cfg.D_POLYGON_POINT),
                'map_polygons_mask':   torch.zeros(cfg.N_POLYGONS),
            }

        return {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else torch.tensor(v)
                for k, v in sample.items()}


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0]}


def make_dataloader(split: str, batch_size: int = cfg.BATCH_SIZE,
                    shuffle: bool = True, num_workers: int = cfg.NUM_WORKERS,
                    max_per_file: int = None):
    dataset = NuPlanCarPlannerDataset(split, max_per_file=max_per_file)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=True,                        # faster CPU→GPU transfer
        persistent_workers=(num_workers > 0),   # avoid worker respawn per epoch
    )


# ── Pre-extracted fast dataset (no SQLite at training time) ────────────────────

class PreextractedDataset(Dataset):
    """
    Loads pre-extracted data from a single .pt cache file.
    Serves all fields needed by Stages A, B, and C.
    Zero SQLite queries, zero map API calls — pure tensor indexing.
    """
    def __init__(self, cache_path: str, device=None):
        print(f"[PreextractedDataset] Loading {cache_path}...")
        t0 = time.time()
        data = torch.load(cache_path, map_location=device or 'cpu')
        elapsed = time.time() - t0
        self.agents_history = data['agents_history']
        self.agents_mask = data['agents_mask']
        self.agents_seq = data['agents_seq']
        self.agents_now = data.get('agents_now', self.agents_history[:, -1, :])
        self.gt_trajectory = data.get('gt_trajectory', None)
        self.mode_label = data.get('mode_label', None)
        self.map_lanes = data['map_lanes']
        self.map_lanes_mask = data['map_lanes_mask']
        # Polygon keys — backward compat with old caches that lack them
        n = data['n_samples']
        self.map_polygons = data.get('map_polygons', torch.zeros(
            n, cfg.N_POLYGONS, cfg.N_LANE_POINTS, cfg.D_POLYGON_POINT))
        self.map_polygons_mask = data.get('map_polygons_mask', torch.zeros(
            n, cfg.N_POLYGONS))
        self.route_polylines = data.get('route_polylines', torch.zeros(
            n, cfg.N_LAT, cfg.N_ROUTE_POINTS, cfg.D_POLYLINE_POINT))
        self.route_mask = data.get('route_mask', torch.zeros(n, cfg.N_LAT))
        self.n = n
        print(f"[PreextractedDataset] {self.n} samples loaded in {elapsed:.1f}s")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        item = {
            'agents_history':      self.agents_history[idx],
            'agents_history_mask': self.agents_mask[idx],
            'agents_seq':          self.agents_seq[idx],
            'agents_now':          self.agents_now[idx],
            'map_lanes':           self.map_lanes[idx],
            'map_lanes_mask':      self.map_lanes_mask[idx],
            'map_polygons':        self.map_polygons[idx],
            'map_polygons_mask':   self.map_polygons_mask[idx],
            'route_polylines':     self.route_polylines[idx],
            'route_mask':          self.route_mask[idx],
        }
        if self.gt_trajectory is not None:
            item['gt_trajectory'] = self.gt_trajectory[idx]
        if self.mode_label is not None:
            item['mode_label'] = self.mode_label[idx]
        return item


def make_cached_dataloader(cache_path: str, batch_size: int = cfg.BATCH_SIZE,
                           shuffle: bool = True, num_workers: int = 0):
    """Create a DataLoader from a pre-extracted .pt cache."""
    dataset = PreextractedDataset(cache_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=True,
    )


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    split = sys.argv[1] if len(sys.argv) > 1 else 'mini'
    print(f"\n=== Smoke test: split='{split}' ===")

    loader = make_dataloader(split, batch_size=4, shuffle=False,
                             num_workers=0, max_per_file=20)

    for i, batch in enumerate(loader):
        if i == 0:
            print("\nShapes:")
            for k, v in batch.items():
                print(f"  {k}: {tuple(v.shape)}  dtype={v.dtype}")
        if i >= 4:
            break

    # Checks
    batch = next(iter(make_dataloader(split, batch_size=8, shuffle=False,
                                      num_workers=0, max_per_file=20)))

    assert batch['ego_history'].shape         == (8, cfg.T_HIST, 4)
    assert batch['agents_history'].shape      == (8, cfg.T_HIST, cfg.N_AGENTS, cfg.D_AGENT)
    assert batch['agents_history_mask'].shape == (8, cfg.N_AGENTS)
    assert batch['agents_now'].shape          == (8, cfg.N_AGENTS, cfg.D_AGENT)
    assert batch['agents_seq'].shape          == (8, cfg.T_FUTURE, cfg.N_AGENTS, cfg.D_AGENT)
    assert batch['map_raster'].shape          == (8, cfg.BEV_C, cfg.BEV_H, cfg.BEV_W)
    assert batch['gt_trajectory'].shape       == (8, cfg.T_FUTURE, 3)
    assert batch['mode_label'].shape          == (8,)
    assert batch['map_lanes'].shape           == (8, cfg.N_LANES, cfg.N_LANE_POINTS, cfg.D_POLYLINE_POINT)
    assert batch['map_lanes_mask'].shape      == (8, cfg.N_LANES)
    assert batch['map_polygons'].shape        == (8, cfg.N_POLYGONS, cfg.N_LANE_POINTS, cfg.D_POLYGON_POINT)
    assert batch['map_polygons_mask'].shape   == (8, cfg.N_POLYGONS)

    assert not torch.isnan(batch['ego_history']).any(), "NaN in ego_history"
    assert not torch.isnan(batch['gt_trajectory']).any(), "NaN in gt_trajectory"
    assert (batch['map_raster'] == 0).all(), "map_raster should be zeros"

    # Ego at t=0 should be near origin in ego frame (index -1 of history)
    ego_t0 = batch['ego_history'][:, -1, :2]  # (B, 2) x,y at current frame
    assert (ego_t0.abs() < 0.1).all(), f"Ego t=0 not near origin: {ego_t0}"

    mode_labels = batch['mode_label']
    assert ((mode_labels >= 0) & (mode_labels < cfg.N_MODES)).all(), "mode_label out of range"

    # Map lanes: at least some samples should have non-zero mask
    n_valid_lanes = batch['map_lanes_mask'].sum().item()
    print(f"  Valid lanes across batch: {n_valid_lanes:.0f}")

    n_valid_polys = batch['map_polygons_mask'].sum().item()
    print(f"  Valid polygons across batch: {n_valid_polys:.0f}")

    print("\n✓ All checks passed.")
    print(f"  GT trajectory range x: [{batch['gt_trajectory'][...,0].min():.2f}, "
          f"{batch['gt_trajectory'][...,0].max():.2f}]")
    print(f"  GT trajectory range y: [{batch['gt_trajectory'][...,1].min():.2f}, "
          f"{batch['gt_trajectory'][...,1].max():.2f}]")
    print(f"  Mode labels: {mode_labels.tolist()}")
