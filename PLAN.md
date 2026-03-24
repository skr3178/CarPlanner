# CarPlanner Baseline Implementation Plan

## Context

The goal is a rough but runnable baseline for **CarPlanner** (Consistent Auto-regressive Trajectory Planning) in `/media/skr/storage/autoresearch/CarPlanner_Implementation/`, running under the `autoresearch-paper/.venv` (Python 3.10, torch 2.9.1+cu128, nuplan-devkit 1.2.2 installed).

An existing stub implementation lives in `autoresearch-paper/implementation/` but `data_loader.py` returns all-zeros (no real data), and all models are minimal parameter-table stubs. The baseline here will use real nuplan data and a functional IL (Stage B only) training pipeline.

**Scope**: Stage B (IL pre-training) end-to-end. Skip Stage A (transition model) and Stage C (PPO/RL) — those are left for later phases. The target metric is open-loop **L2 displacement error (best-of-K)**.

---

## What Already Exists

| File | Location | Status |
|------|----------|--------|
| `data_loader.py` | `autoresearch-paper/implementation/` | Stub — returns zeros |
| `autoregressive_policy.py` | same | Stub — bias table only |
| `critic.py` | same | Exists but unused for baseline |
| `mode_selector.py` | same | Exists |
| `consistency_module.py` | same | Exists |
| `transition_model.py` | same | Exists |
| nuplan-devkit | `autoresearch-paper/paper/dataset/nuplan-devkit/` | Installed in `.venv`, tested working |
| Dataset | `autoresearch-paper/paper/dataset/nuplan-extracted/data/cache/{mini,train_boston}/` | 64 mini + 1647 train_boston `.db` files |

---

## Files to Create

All files go in `/media/skr/storage/autoresearch/CarPlanner_Implementation/`:

| File | Purpose |
|------|---------|
| `config.py` | All hyperparams and absolute dataset paths |
| `data_loader.py` | Real nuplan data loading — ego history, agent history, GT trajectory |
| `model.py` | BEV encoder + ModeSelector + TrajectoryGenerator (combined) |
| `train.py` | IL training loop (Stage B) |
| `evaluate.py` | Open-loop L2/L1 best-of-K evaluation |

---

## Architecture (Baseline, not full IVM)

**Simplifications justified for rough baseline:**
- BEV raster: zero tensor (real rasterization deferred; vector features carry signal)
- No autoregressive step dependency (decode all T steps in one pass via MLP)
- No rule selector (use mode with highest predicted score)
- No PPO/RL

### Modules

**`BEVEncoder`** (`model.py`)
- Input: `(B, C, H, W)` BEV raster
- Output: `(B, D_bev)` embedding
- Impl: small CNN (3 conv layers + adaptive pool) → 128-dim

**`StateEncoder`** (`model.py`)
- Input: ego_history `(B, T_hist, 4)` + agents_history `(B, T_hist, N, 4)` masked mean-pooled
- Output: `(B, D_state)` 128-dim via MLP

**`ModeSelector`** (`model.py`)
- Input: fused `(B, D_bev + D_state)`
- Output: logits `(B, M)` where M=60 modes
- Impl: 2-layer MLP

**`TrajectoryGenerator`** (`model.py`)
- Input: fused features + mode embedding `(B, D_bev + D_state + D_mode)`
- Output: `(B, T_future, 3)` predicted trajectory (x, y, yaw) per mode
- Impl: For each of M modes, independent head `(B, T*3)` via MLP
- At inference: generate K=M trajectories (one per mode), select best

### Training (IL Stage B)

**Best-mode assignment** (winner-takes-all):
- For each sample, compute L1 between each mode's predicted trajectory and GT
- Assign label = argmin mode (closest to GT)
- CE loss on mode selector with these labels

**Losses:**
- `L_traj = L1(predicted_traj[best_mode], gt_trajectory)` — trajectory regression
- `L_mode = CrossEntropy(mode_logits, best_mode_label)` — mode classification
- `L_total = L_traj + L_mode`

**Optimizer:** AdamW, lr=1e-4, weight_decay=1e-4
**Schedule:** ReduceLROnPlateau patience=3, factor=0.5
**Batch size:** 32 (fits on 4090 with current arch)
**Epochs:** 50

---

## Data Loading (`data_loader.py`)

Uses nuplan devkit API (tested working). Key classes:

```python
from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_ego_state_for_lidarpc_token_from_db,
    get_lidarpc_tokens_with_scenario_tag_from_db,
    get_tracked_objects_for_lidarpc_token_from_db,
    get_sampled_ego_states_from_db,
)
from nuplan.database.nuplan_db.nuplan_db_utils import get_lidarpc_sensor_data
# sensor_source = get_lidarpc_sensor_data()  # SensorDataSource('lidar_pc','lidar','lidar_token','MergedPointCloud')
```

**Per-frame loading** (anchor token = current frame `t=0`):
- History: `get_sampled_ego_states_from_db(db, token, sensor_source, list(range(T_hist)), future=False)` → T_hist EgoState objects (past, sorted ascending)
- GT future: `get_sampled_ego_states_from_db(db, token, sensor_source, list(range(1, T+1)), future=True)` → T EgoState objects
- Agents at current frame: `get_tracked_objects_for_lidarpc_token_from_db(db, token)` → TrackedObject list

**Coordinate frame**: ego-centric at t=0. Transform all (x, y, heading) by subtracting ego position at t=0 and rotating by -ego_heading.

**EgoState fields used:**
- `ego.rear_axle.x`, `ego.rear_axle.y`, `ego.rear_axle.heading`
- `ego.dynamic_car_state.rear_axle_velocity_2d.x` (longitudinal speed)

**Agent fields used:**
- `agent.center.x`, `agent.center.y`, `agent.center.heading`
- Velocity from `agent.velocity` if available, else 0

**Output tensors per sample:**
- `ego_history`: `(T_hist=10, 4)` — (x, y, yaw, speed) in ego frame
- `agents_history`: `(T_hist=10, N_agents=20, 4)` — current-frame agent states repeated for simplicity (full history deferred)
- `agents_history_mask`: `(N_agents=20,)` — 1 = valid agent
- `map_raster`: `(C=3, 224, 224)` — zeros for baseline
- `gt_trajectory`: `(T=8, 3)` — future (x, y, yaw) in ego frame

**Mode assignment** (winner-takes-all, per Algorithm 1):
- Longitudinal (12 modes): bucket GT endpoint speed into 12 equal intervals over [0, max_speed=15 m/s]
- Lateral (5 modes): discretize GT lateral offset at horizon end into 5 bins [-4m, -2m, 0m, 2m, 4m]
- Positive mode index: `c* = lon_bin * N_lat + lat_bin`, shape `()` int64

---

## Evaluation (`evaluate.py`)

For each sample in the mini split:
1. Run model to get M trajectory predictions
2. Compute L2 displacement for each: `mean over T of ||pred_xy - gt_xy||_2`
3. Best-of-K (K=M=60): take trajectory with lowest L2
4. Report: mean best-of-K L2 over all samples

---

## Environment

```
Python: /media/skr/storage/autoresearch/autoresearch-paper/.venv/bin/python
Venv: autoresearch-paper/.venv (named "autoresearch", Python 3.10, torch 2.9.1+cu128)
```

Note: the **root** `.venv` (also named "autoresearch") does NOT have nuplan — must use `autoresearch-paper/.venv`.

---

## Critical File Paths Referenced

| Path | Role |
|------|------|
| `autoresearch-paper/paper/dataset/nuplan-extracted/data/cache/mini/` | Smoke test / eval |
| `autoresearch-paper/paper/dataset/nuplan-extracted/data/cache/train_boston/` | Training |
| `autoresearch-paper/paper/dataset/nuplan-extracted/nuplan-maps-v1.0/` | Maps |
| `autoresearch-paper/paper/dataset/nuplan-devkit/nuplan/database/nuplan_db/nuplan_scenario_queries.py` | Key devkit queries |
| `autoresearch-paper/implementation/` | Reference stubs (do NOT import from here) |

---

## Implementation Order (mini-first validation)

**Step 0**: Copy this plan to `/media/skr/storage/autoresearch/CarPlanner_Implementation/PLAN.md`

1. **`config.py`** — constants, paths, hyperparams
2. **`data_loader.py`** — real nuplan loading; run standalone smoke test: print shapes, check no NaN, check ego at origin
3. **`model.py`** — BEVEncoder + StateEncoder + ModeSelector + TrajectoryGenerator; run forward pass shape test on random input
4. **`train.py`** — IL training loop; first run `--split mini --epochs 5 --batch_size 4` to verify loss decreases
5. **`evaluate.py`** — L2 best-of-K on mini; run after training

## Verification

1. `python data_loader.py` — smoke test on mini: prints shapes, no NaN, ego near origin
2. `python model.py` — forward pass shape assertions (runs on dummy data, no nuplan needed)
3. `python train.py --split mini --epochs 5 --batch_size 4` — loss should drop within 5 epochs on mini
4. `python evaluate.py --split mini` — reports best-of-60 L2 on mini split
5. `python train.py --split train_boston --epochs 50` — full training (only after mini validates)
