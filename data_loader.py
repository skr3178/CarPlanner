"""
CarPlanner baseline — data loader.
Loads nuPlan SQLite scenarios using the devkit API and produces tensors
matching the paper's data contract (phase1_report.md, submodules.md).

All positions are transformed to ego-centric frame at t=0.
"""

import os
import math
import glob
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

# Singleton sensor source (lidar_pc / MergedPointCloud)
_SENSOR_SOURCE = get_lidarpc_sensor_data()


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


# ── Mode assignment (winner-takes-all, Algorithm 1) ────────────────────────────

def _assign_mode(gt_traj: np.ndarray) -> int:
    """
    Assign positive mode c* from GT trajectory.
    gt_traj: (T, 3) in ego frame — (x, y, yaw)
    Returns c* in [0, N_MODES).
    """
    endpoint = gt_traj[-1]  # (x, y, yaw) at T steps ahead

    # Longitudinal mode: total displacement distance bucketed into N_LON bins
    dist = math.sqrt(endpoint[0] ** 2 + endpoint[1] ** 2)
    lon_idx = min(
        int(dist / (cfg.MAX_SPEED * cfg.T_FUTURE * 0.1 / cfg.N_LON)),
        cfg.N_LON - 1,
    )

    # Lateral mode: lateral offset (y) at endpoint bucketed into N_LAT bins
    lat_y = endpoint[1]
    lat_idx = 0
    for i, edge in enumerate(cfg.LAT_BIN_EDGES[1:]):
        if lat_y < edge:
            lat_idx = i
            break
    else:
        lat_idx = cfg.N_LAT - 1

    return lon_idx * cfg.N_LAT + lat_idx


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
            list(range(cfg.T_HIST)), future=False
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
            list(range(1, cfg.T_FUTURE + 1)), future=True
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
    def _load_agents_at_token(tok: str) -> tuple:
        """Returns (agents array (N,4), mask (N,)) in initial ego frame."""
        try:
            raw = list(get_tracked_objects_for_lidarpc_token_from_db(db_path, tok))
        except Exception:
            raw = []
        arr  = np.zeros((cfg.N_AGENTS, 4), dtype=np.float32)
        mask = np.zeros(cfg.N_AGENTS,      dtype=np.float32)
        for j, ag in enumerate(raw[:cfg.N_AGENTS]):
            xe, ye, he = _to_ego_frame(
                ag.center.x, ag.center.y, ag.center.heading,
                ref_x, ref_y, ref_h
            )
            spd = 0.0
            if hasattr(ag, 'velocity') and ag.velocity is not None:
                spd = math.sqrt(ag.velocity.x ** 2 + ag.velocity.y ** 2)
            arr[j]  = [xe, ye, he, spd]
            mask[j] = 1.0
        return arr, mask

    agents_now, agents_mask = _load_agents_at_token(token)

    # Keep history tensor for backward-compat (fill all steps with t=0 state)
    agents_history = np.stack([agents_now] * cfg.T_HIST, axis=0)   # (T_HIST, N, 4)

    # ── GT agent futures (t=1..T) — used by AutoregressivePolicy IVM ──────────
    # Get future lidar_pc tokens, then load agents at each step
    agents_seq = np.zeros((cfg.T_FUTURE, cfg.N_AGENTS, 4), dtype=np.float32)
    try:
        future_pcs = list(get_sampled_lidarpcs_from_db(
            db_path, token, sensor_source,
            list(range(1, cfg.T_FUTURE + 1)), future=True
        ))
        for step_i, pc in enumerate(future_pcs[:cfg.T_FUTURE]):
            arr_t, _ = _load_agents_at_token(pc.token)
            agents_seq[step_i] = arr_t
    except Exception:
        # Fallback: hold agents static (current frame repeated)
        agents_seq[:] = agents_now[np.newaxis]

    # ── BEV raster (zeros; real rasterisation deferred) ───────────────────────
    map_raster = np.zeros((cfg.BEV_C, cfg.BEV_H, cfg.BEV_W), dtype=np.float32)

    # ── Mode assignment ───────────────────────────────────────────────────────
    mode_label = _assign_mode(gt_trajectory)

    return {
        'map_raster':          map_raster,
        'ego_history':         ego_history,
        'agents_history':      agents_history,
        'agents_history_mask': agents_mask,
        'agents_now':          agents_now,       # (N, 4)  — t=0 agents
        'agents_seq':          agents_seq,        # (T_FUTURE, N, 4) — GT future agents
        'gt_trajectory':       gt_trajectory,
        'mode_label':          np.int64(mode_label),
    }


# ── Dataset ────────────────────────────────────────────────────────────────────

class NuPlanCarPlannerDataset(Dataset):
    """
    Loads nuPlan scenario frames from .db files.
    Builds an index of (db_path, token) pairs at construction time.
    """

    def __init__(self, split: str, max_per_file: int = None):
        """
        split: 'mini' | 'train_boston'
        max_per_file: cap on tokens loaded per .db file (None = all)
        """
        if split == 'mini':
            db_dir = cfg.MINI_DIR
            if max_per_file is None:
                max_per_file = cfg.MAX_SAMPLES_PER_FILE_MINI
        elif split == 'train_boston':
            db_dir = cfg.TRAIN_DIR
            if max_per_file is None:
                max_per_file = cfg.MAX_SAMPLES_PER_FILE_TRAIN
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
                'agents_history':      torch.zeros(cfg.T_HIST, cfg.N_AGENTS, 4),
                'agents_history_mask': torch.zeros(cfg.N_AGENTS),
                'agents_now':          torch.zeros(cfg.N_AGENTS, 4),
                'agents_seq':          torch.zeros(cfg.T_FUTURE, cfg.N_AGENTS, 4),
                'gt_trajectory':       torch.zeros(cfg.T_FUTURE, 3),
                'mode_label':          torch.zeros(1, dtype=torch.long).squeeze(),
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
    assert batch['agents_history'].shape      == (8, cfg.T_HIST, cfg.N_AGENTS, 4)
    assert batch['agents_history_mask'].shape == (8, cfg.N_AGENTS)
    assert batch['agents_now'].shape          == (8, cfg.N_AGENTS, 4)
    assert batch['agents_seq'].shape          == (8, cfg.T_FUTURE, cfg.N_AGENTS, 4)
    assert batch['map_raster'].shape          == (8, cfg.BEV_C, cfg.BEV_H, cfg.BEV_W)
    assert batch['gt_trajectory'].shape       == (8, cfg.T_FUTURE, 3)
    assert batch['mode_label'].shape          == (8,)

    assert not torch.isnan(batch['ego_history']).any(), "NaN in ego_history"
    assert not torch.isnan(batch['gt_trajectory']).any(), "NaN in gt_trajectory"
    assert (batch['map_raster'] == 0).all(), "map_raster should be zeros"

    # Ego at t=0 should be near origin in ego frame (index -1 of history)
    ego_t0 = batch['ego_history'][:, -1, :2]  # (B, 2) x,y at current frame
    assert (ego_t0.abs() < 0.1).all(), f"Ego t=0 not near origin: {ego_t0}"

    mode_labels = batch['mode_label']
    assert ((mode_labels >= 0) & (mode_labels < cfg.N_MODES)).all(), "mode_label out of range"

    print("\n✓ All checks passed.")
    print(f"  GT trajectory range x: [{batch['gt_trajectory'][...,0].min():.2f}, "
          f"{batch['gt_trajectory'][...,0].max():.2f}]")
    print(f"  GT trajectory range y: [{batch['gt_trajectory'][...,1].min():.2f}, "
          f"{batch['gt_trajectory'][...,1].max():.2f}]")
    print(f"  Mode labels: {mode_labels.tolist()}")
