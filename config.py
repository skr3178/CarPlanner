"""
CarPlanner baseline — configuration.
All hyperparameters from paper (carplanner_equations.md, hyperparameters.md).
"""

import os

# ── Dataset paths ─────────────────────────────────────────────────────────────
BASE_DIR = "/media/skr/storage/autoresearch/autoresearch-paper/paper/dataset/nuplan-extracted"
MINI_DIR = os.path.join(BASE_DIR, "data/cache/mini")
TRAIN_DIR = os.path.join(BASE_DIR, "data/cache/train_boston")
MAPS_DIR = os.path.join(BASE_DIR, "nuplan-maps-v1.0")

# ── Data dimensions ───────────────────────────────────────────────────────────
T_HIST = 10           # history frames (~1s at 10Hz)
T_FUTURE = 8          # horizon steps (paper default, Fig 4)
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

# ── Architecture ──────────────────────────────────────────────────────────────
D_ACTION = 3          # (x, y, yaw) per step
D_HIDDEN = 256
D_MODE_EMBED = 64     # mode embedding dimension
D_BEV = 128           # BEV encoder output dim
D_STATE = 128         # state encoder output dim

# ── Training (Section 4.1, IL Stage B) ───────────────────────────────────────
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 50
LR_PATIENCE = 3       # ReduceLROnPlateau (paper uses 0, we use 3 for stability)
LR_FACTOR = 0.5

# ablation flags (IL best config: all True; RL best: dropout=True, side_task=True, rest=False)
MODE_DROPOUT = True
SELECTOR_SIDE_TASK = True
EGO_HISTORY_DROPOUT = True
BACKBONE_SHARING = True
MODE_DROPOUT_P = 0.1  # probability of zeroing mode embedding during training

# ── RL coefficients (Eq 8-10, unused in IL-only baseline) ────────────────────
LAMBDA_POLICY = 100.0
LAMBDA_VALUE = 3.0
LAMBDA_ENTROPY = 0.001
PPO_CLIP = 0.2
GAMMA = 0.1
GAE_LAMBDA = 0.9

# ── Misc ──────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = os.path.join(
    os.path.dirname(__file__), "checkpoints"
)
MAX_SAMPLES_PER_FILE_MINI = 200   # cap per DB file for mini split
MAX_SAMPLES_PER_FILE_TRAIN = 50   # cap per DB file for train_boston
NUM_WORKERS = 4
