"""
Pre-extract Test14-Random scenarios into a GPU-ready cache.

Scans the 280 nuPlan val scenarios matching the test14-random filter,
extracts world-frame ego/agent/map data, and saves to a .pt file.

Run with nuplan_venv:
    cd /media/skr/storage/autoresearch/CarPlanner_Implementation
    source paper/dataset/nuplan-devkit/activate_nuplan_env.sh
    PYTHONPATH=/media/skr/storage/autoresearch/CarPlanner_Implementation \
    python scripts/preextract_test14.py \
        --split test14-random \
        --output checkpoints/test14_random_cache.pt
"""

import os
import sys
import argparse
import numpy as np
import torch
from collections import Counter
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import config as cfg

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_sequential import Sequential
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.state_representation import Point2D

# ── Constants ──────────────────────────────────────────────────────────────────
T_SIM      = 150
N_AGENTS   = cfg.N_AGENTS    # 20
N_LANES    = cfg.N_LANES     # 20
N_PTS      = cfg.N_LANE_POINTS  # 10
MAP_RADIUS = cfg.MAP_QUERY_RADIUS  # 50m

INTERESTED_TYPES = [
    TrackedObjectType.VEHICLE,
    TrackedObjectType.PEDESTRIAN,
    TrackedObjectType.BICYCLE,
]

TEST14_TYPES = [
    'starting_left_turn', 'starting_right_turn',
    'starting_straight_traffic_light_intersection_traversal',
    'stopping_with_lead', 'high_lateral_acceleration',
    'high_magnitude_speed', 'low_magnitude_speed',
    'traversing_pickup_dropoff', 'waiting_for_pedestrian_to_cross',
    'behind_long_vehicle', 'stationary_in_traffic',
    'near_multiple_vehicles', 'changing_lane', 'following_lane_with_lead',
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resample_polyline(points, n_points):
    pts = np.array([[p.x, p.y] for p in points], dtype=np.float32)
    if len(pts) < 2:
        return np.zeros((n_points, 2), dtype=np.float32)
    diffs = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate([[0], np.cumsum(seg_len)])
    total = cumlen[-1]
    if total < 1e-6:
        return np.tile(pts[0], (n_points, 1))
    sample_d = np.linspace(0, total, n_points)
    out = np.zeros((n_points, 2), dtype=np.float32)
    for i, d in enumerate(sample_d):
        idx = max(0, min(np.searchsorted(cumlen, d) - 1, len(pts) - 2))
        t = (d - cumlen[idx]) / seg_len[idx] if seg_len[idx] > 1e-6 else 0.0
        out[i] = pts[idx] + t * (pts[idx + 1] - pts[idx])
    return out


# ── Core extraction ────────────────────────────────────────────────────────────

def extract_scenario(scenario) -> dict:
    """Extract one nuPlan scenario into world-frame tensors."""
    n_iters = min(scenario.get_number_of_iterations(), T_SIM)

    # ── GT ego trajectory ──────────────────────────────────────────────────
    ego_gt = np.zeros((T_SIM, 4), dtype=np.float32)  # x, y, yaw, speed
    for i in range(n_iters):
        s = scenario.get_ego_state_at_iteration(i)
        spd = s.dynamic_car_state.speed if hasattr(s, 'dynamic_car_state') else 0.0
        ego_gt[i] = [s.rear_axle.x, s.rear_axle.y, s.rear_axle.heading, spd]
    if n_iters < T_SIM:
        ego_gt[n_iters:] = ego_gt[n_iters - 1]

    # ── Agent trajectories (world frame) ───────────────────────────────────
    agents_world = np.zeros((T_SIM, N_AGENTS, 5), dtype=np.float32)  # x,y,yaw,vx,vy
    agents_size  = np.zeros((N_AGENTS, 3), dtype=np.float32)          # w,l,h
    agents_cat   = np.zeros(N_AGENTS, dtype=np.int32)
    agents_valid = np.zeros((T_SIM, N_AGENTS), dtype=np.float32)

    cat_map = {'VEHICLE': 0, 'PEDESTRIAN': 1, 'BICYCLE': 2, 'MOTORCYCLE': 3}
    token_to_idx = {}

    # Seed agent index from t=0
    t0_objs = scenario.get_tracked_objects_at_iteration(0)\
                       .tracked_objects\
                       .get_tracked_objects_of_types(INTERESTED_TYPES)
    for j, ag in enumerate(t0_objs[:N_AGENTS]):
        token_to_idx[ag.track_token] = j
        if hasattr(ag, 'box'):
            agents_size[j] = [ag.box.width, ag.box.length, ag.box.height]
        agents_cat[j] = cat_map.get(
            ag.tracked_object_type.name if hasattr(ag, 'tracked_object_type') else 'VEHICLE', 0)

    for i in range(n_iters):
        objs = scenario.get_tracked_objects_at_iteration(i)\
                       .tracked_objects\
                       .get_tracked_objects_of_types(INTERESTED_TYPES)
        for ag in objs:
            idx = token_to_idx.get(ag.track_token)
            if idx is None and len(token_to_idx) < N_AGENTS:
                idx = len(token_to_idx)
                token_to_idx[ag.track_token] = idx
                if hasattr(ag, 'box'):
                    agents_size[idx] = [ag.box.width, ag.box.length, ag.box.height]
                agents_cat[idx] = cat_map.get(
                    ag.tracked_object_type.name if hasattr(ag, 'tracked_object_type') else 'VEHICLE', 0)
            if idx is not None and idx < N_AGENTS:
                vx = ag.velocity.x if hasattr(ag, 'velocity') and ag.velocity else 0.0
                vy = ag.velocity.y if hasattr(ag, 'velocity') and ag.velocity else 0.0
                agents_world[i, idx] = [ag.center.x, ag.center.y, ag.center.heading, vx, vy]
                agents_valid[i, idx] = 1.0

    # ── Map lanes (world-frame centerline xy) ──────────────────────────────
    ego0   = scenario.initial_ego_state
    ref_pt = Point2D(ego0.rear_axle.x, ego0.rear_axle.y)
    try:
        map_objs  = scenario.map_api.get_proximal_map_objects(
            ref_pt, MAP_RADIUS, [SemanticMapLayer.LANE])
        raw_lanes = map_objs[SemanticMapLayer.LANE]
    except Exception:
        raw_lanes = []

    map_pts_world  = np.zeros((N_LANES, N_PTS, 2), dtype=np.float32)
    map_lanes_mask = np.zeros(N_LANES, dtype=np.float32)
    for i, lane in enumerate(raw_lanes[:N_LANES]):
        bp = lane.baseline_path
        if bp is None or len(bp.discrete_path) < 2:
            continue
        map_pts_world[i]  = _resample_polyline(bp.discrete_path, N_PTS)
        map_lanes_mask[i] = 1.0

    # ── Goal ───────────────────────────────────────────────────────────────
    goal      = scenario.get_mission_goal()
    goal_world = np.array([goal.x, goal.y], dtype=np.float32) \
                 if goal is not None else np.zeros(2, dtype=np.float32)

    return {
        'scenario_type':  scenario.scenario_type,
        'scenario_token': scenario.token,
        'log_name':       scenario.log_name,
        'ego_gt':         torch.from_numpy(ego_gt),          # (T_SIM, 4)
        'agents_world':   torch.from_numpy(agents_world),    # (T_SIM, N, 5)
        'agents_size':    torch.from_numpy(agents_size),     # (N, 3)
        'agents_cat':     torch.from_numpy(agents_cat),      # (N,)
        'agents_valid':   torch.from_numpy(agents_valid),    # (T_SIM, N)
        'map_pts_world':  torch.from_numpy(map_pts_world),   # (N_L, N_P, 2)
        'map_lanes_mask': torch.from_numpy(map_lanes_mask),  # (N_L,)
        'goal_world':     torch.from_numpy(goal_world),      # (2,)
        'n_iters':        n_iters,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='test14-random',
                        choices=['test14-random', 'mini'])
    parser.add_argument('--db_files',
                        default='/home/skr/nuplan_cities/val/data/cache/val')
    parser.add_argument('--maps_root',
                        default=os.path.join(PROJECT_ROOT, 'paper/dataset/maps'))
    parser.add_argument('--output',
                        default=os.path.join(PROJECT_ROOT,
                                             'checkpoints/test14_random_cache.pt'))
    parser.add_argument('--num_per_type', type=int, default=20,
                        help='Scenarios per type for test14-random (default: 20)')
    args = parser.parse_args()

    print(f"[Preextract] Split:    {args.split}")
    print(f"[Preextract] DB files: {args.db_files}")
    print(f"[Preextract] Maps:     {args.maps_root}")
    print(f"[Preextract] Output:   {args.output}")

    os.environ.setdefault('NUPLAN_MAPS_ROOT', args.maps_root)
    os.environ.setdefault('NUPLAN_DATA_ROOT',
                          os.path.join(PROJECT_ROOT, 'paper/dataset'))

    # ── Build scenario list ────────────────────────────────────────────────
    print("[Preextract] Scanning DB files for scenarios...")
    builder = NuPlanScenarioBuilder(
        data_root=args.db_files,
        map_root=args.maps_root,
        sensor_root=args.db_files,   # not used (no cameras), just needs a valid path
        db_files=args.db_files,
        map_version='nuplan-maps-v1.0',
        verbose=True,
    )

    if args.split == 'test14-random':
        scenario_filter = ScenarioFilter(
            scenario_types=TEST14_TYPES,
            scenario_tokens=None,
            log_names=None,
            map_names=None,
            num_scenarios_per_type=args.num_per_type,
            limit_total_scenarios=None,
            timestamp_threshold_s=15,
            ego_displacement_minimum_m=None,
            ego_start_speed_threshold=None,
            ego_stop_speed_threshold=None,
            speed_noise_tolerance=None,
            expand_scenarios=False,
            remove_invalid_goals=True,
            shuffle=False,
        )
    else:  # mini
        scenario_filter = ScenarioFilter(
            scenario_types=None,
            scenario_tokens=None,
            log_names=None,
            map_names=None,
            num_scenarios_per_type=None,
            limit_total_scenarios=100,
            timestamp_threshold_s=None,
            ego_displacement_minimum_m=None,
            ego_start_speed_threshold=None,
            ego_stop_speed_threshold=None,
            speed_noise_tolerance=None,
            expand_scenarios=True,
            remove_invalid_goals=True,
            shuffle=True,
        )

    scenarios = builder.get_scenarios(scenario_filter, Sequential())
    print(f"[Preextract] Found {len(scenarios)} scenarios")

    # ── Extract ────────────────────────────────────────────────────────────
    cache, failed = [], 0
    for scenario in tqdm(scenarios, desc='Extracting'):
        try:
            cache.append(extract_scenario(scenario))
        except Exception as e:
            print(f"  WARNING: {scenario.token}: {e}")
            failed += 1

    print(f"[Preextract] Done: {len(cache)} extracted, {failed} failed")

    type_counts = Counter(d['scenario_type'] for d in cache)
    print("[Preextract] Type distribution:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t:55s}: {c}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({'split': args.split, 'scenarios': cache}, args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"[Preextract] Saved → {args.output}  ({size_mb:.0f} MB)")


if __name__ == '__main__':
    main()
