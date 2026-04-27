"""
CarPlanner baseline — configuration.
All hyperparameters from paper (carplanner_equations.md, hyperparameters.md).
"""

import os

# ── Dataset paths ─────────────────────────────────────────────────────────────
BASE_DIR = "/media/skr/storage/autoresearch/CarPlanner_Implementation/paper/dataset/nuplan-extracted"
MINI_DIR = os.path.join(BASE_DIR, "data/cache/mini")
TRAIN_DIR = os.path.join(BASE_DIR, "data/cache/train_boston")
TRAIN_PITTSBURGH_DIR = "/home/skr/nuplan_cities/pittsburgh/data/cache/train_pittsburgh"
TRAIN_SINGAPORE_DIR = "/home/skr/nuplan_cities/singapore/data/cache/train_singapore"
TRAIN_VEGAS_DIR = "/media/skr/storage/autoresearch/CarPlanner_Implementation/data/cache/train_vegas_5"
VAL_DIR = "/home/skr/nuplan_cities/val/data/cache/val"
MAPS_DIR = os.path.join(BASE_DIR, "nuplan-maps-v1.0")

# PDM val14 split: 1,118 scenario tokens from tuplan_garage/val14_split.yaml
_SPLIT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "tuplan_garage/tuplan_garage/planning/script/config/common/scenario_filter",
)
VAL14_YAML = os.path.join(_SPLIT_DIR, "val14_split.yaml")
TEST14_RANDOM_YAML = os.path.join(_SPLIT_DIR, "test14_random_split.yaml")
REDUCED_VAL14_YAML = os.path.join(_SPLIT_DIR, "reduced_val14_split.yaml")

# ── Data cadence ──────────────────────────────────────────────────────────────
# nuPlan logs are at 20 Hz internally; planning horizon in paper is 8 s @ 1 Hz.
DB_FREQ_HZ    = 20
HISTORY_DT_S  = 0.1   # 10 Hz history
FUTURE_DT_S   = 1.0   # 1 Hz future (paper: 8 future steps over 8 s)
HISTORY_STRIDE = int(DB_FREQ_HZ * HISTORY_DT_S)  # 2 frames per history step
FUTURE_STRIDE  = int(DB_FREQ_HZ * FUTURE_DT_S)   # 20 frames per future step

# ── Data dimensions ───────────────────────────────────────────────────────────
T_HIST = 10           # history frames (~1s at 10Hz)
T_FUTURE = 8          # horizon steps (paper default: 8 s @ 1 Hz)
N_AGENTS = 20         # max tracked agents per frame (padded/truncated)
BEV_C = 3             # BEV raster channels (placeholder zeros)
BEV_H = 224
BEV_W = 224

# ── Mode structure (Section 4.1) ──────────────────────────────────────────────
N_LON = 12            # longitudinal modes
N_LAT = 5             # lateral modes
N_MODES = N_LON * N_LAT  # 60 total candidate trajectories

# Mode assignment bins
MAX_SPEED = 15.0      # m/s — max longitudinal speed for bin edges
# Lon bins: [0, 1.25, 2.5, ..., 15.0] — 12 intervals
LON_BIN_EDGES = [i * MAX_SPEED / N_LON for i in range(N_LON + 1)]
# Lat bins: lateral offset at T steps in ego frame (y-axis)
LAT_BIN_EDGES = [-float('inf'), -3.0, -1.0, 1.0, 3.0, float('inf')]  # 5 bins

# ── Feature dimensions ────────────────────────────────────────────────────────
D_AGENT = 14          # per-agent: x, y, sin_h, cos_h, vx, vy, box_w, box_l, box_h, time_step, 4×category_onehot
D_MAP_POINT = 13      # per-map-point: x, y, sin_h, cos_h, speed_limit, 4×category_onehot, 4×traffic_light_onehot

# Per-feature normalization stds (from training data stats, valid agents only)
# Dims: x, y, sin_h, cos_h, vx, vy, box_w, box_l, box_h, time_step, cat(4)
AGENT_FEATURE_STD = [34.55, 16.12, 1.0, 1.0, 1.90, 1.58, 0.76, 2.71, 0.66, 10.0, 1.0, 1.0, 1.0, 1.0]
D_POLYLINE_POINT = 3 * D_MAP_POINT  # = 39: center + left_boundary + right_boundary per point (paper §2.3: Np × 3Dm)

# ── Architecture ──────────────────────────────────────────────────────────────
D_ACTION = 3          # (x, y, yaw) per step
D_HIDDEN = 256
D_MODE_EMBED = 64     # mode embedding dimension
D_BEV = 128           # BEV encoder output dim
D_STATE = 128         # state encoder output dim

# ── Training (Section 4.1, IL Stage B) ───────────────────────────────────────
BATCH_SIZE = 64
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 50
LR_PATIENCE = 5       # ReduceLROnPlateau — bigger model + batch256 needs more exploration time
LR_FACTOR = 0.3

# ── Ablation flags (Table 4) ─────────────────────────────────────────────────
# Defaults = IL-best.  Call set_stage('c') before model creation for RL-best.
STAGE = 'b'
MODE_DROPOUT = True
SELECTOR_SIDE_TASK = True
EGO_HISTORY_DROPOUT = True     # IL best: True,  RL best: False
BACKBONE_SHARING = True        # IL best: True,  RL best: False
MODE_DROPOUT_P = 0.1


def set_stage(stage: str):
    """Set ablation flags for the given training stage."""
    import config as _self
    stage = stage.lower()
    _self.STAGE = stage
    if stage in ('a', 'b'):
        _self.EGO_HISTORY_DROPOUT = True
        _self.BACKBONE_SHARING = True
    elif stage == 'c':
        _self.EGO_HISTORY_DROPOUT = False
        _self.BACKBONE_SHARING = False
    else:
        raise ValueError(f"Unknown stage: {stage!r}  (expected 'a', 'b', or 'c')")

# ── RL coefficients (Eq 8-10, paper §A "loss magnitudes value:1e3 policy:1 entropy:1e-3")
LAMBDA_POLICY = 1.0
LAMBDA_VALUE = 1000.0
LAMBDA_ENTROPY = 0.001
PPO_CLIP = 0.2
GAMMA = 0.1
GAE_LAMBDA = 0.9

# ── Map encoding ──────────────────────────────────────────────────────────────
MAP_QUERY_RADIUS = 50.0          # meters — radius to query nearby lanes
N_LANES = 20                     # max lanes to encode (padded/truncated)
N_LANE_POINTS = 10               # points per lane polyline (resampled)
D_LANE = 256                     # lane encoder output dim (= D, paper §3.1)

# ── Polygon map encoding ──────────────────────────────────────────────────────
N_POLYGONS = 10                      # max polygons per scene (crosswalks, intersections, stop lines)
D_POLYGON_POINT = D_MAP_POINT        # per-polygon point: same 13-dim encoding as map points (TL dims = 0 for polygons)

# ── Route extraction (Section 3.3.2: lane graph search) ──────────────────────
N_ROUTE_POINTS = N_LANE_POINTS       # points per route polyline (resampled from concatenated lanes)
ROUTE_MAX_DEPTH = 6                  # max lane segments to trace forward per route
ROUTE_MAX_LENGTH_M = 150.0           # max route length in meters (~10s at 15 m/s)

# ── Misc ──────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = os.path.join(
    os.path.dirname(__file__), "checkpoints"
)
MAX_SAMPLES_PER_FILE_MINI = 200   # cap per DB file for mini split
MAX_SAMPLES_PER_FILE_TRAIN = 50   # cap per DB file for train_boston
NUM_WORKERS = 4
