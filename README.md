# CarPlanner

Faithful PyTorch implementation of **CarPlanner: Consistent Auto-regressive Trajectory Planning for Large-scale Reinforcement Learning in Autonomous Driving**.

## Architecture

Implements the three-stage system from Figure 2 of the paper:

```
s_0 ──► NR Transition Model ──────────────────────────────► agent futures
         │                                                        │
         ├──► Mode Selector ──► logits (60 modes) + side traj    │
         │         c*                                             │
         └──────────────────► Autoregressive Policy (IVM × T) ◄──┘
                                    a_0, a_1, ..., a_{T-1}
                                         │
                                    Rule Selector
                                         │
                                  ego-planned trajectory
```

**Key modules:**
- `PointNetEncoder` — shared MLP + max-pool over agent point sets
- `IVMBlock` — Invariant-View Module (Algorithm 3): K-NN selection → ego-centric transform → Transformer decoder with mode `c` as query
- `ModeSelector` — predicts mode distribution (Eq 6) + side-task trajectory (Eq 7)
- `AutoregressivePolicy` — T=8 steps of IVM-based decoding, consistent mode across all steps
- `TransitionModel` — non-reactive agent future prediction (Eq 5), frozen after Stage A
- `RuleSelector` — inference-time best-of-60 selection by comfort + progress + mode scores

## Training

Three-stage training (Algorithm 1):

| Stage | What trains | Loss |
|-------|------------|------|
| A | Transition model | L1 on agent futures (Eq 5) |
| B (IL) | Mode selector + Policy | L_CE + L_SideTask + L_generator (Eq 6+7+11) |
| C (RL) | Policy + Value | PPO + L_generator (Eq 8–11) — *deferred* |

## Modes

60 modes = 12 longitudinal × 5 lateral (Section 4.1). Mode index `c` is fixed across all T autoregressive steps (consistent AR, Figure 1c).

## Setup

```bash
# Requires nuplan-devkit installed in your environment
pip install -e /path/to/nuplan-devkit --no-deps
pip install torch scipy geopandas pyarrow shapely sqlalchemy pyquaternion
```

Set dataset paths in `config.py`.

## Usage

```bash
# Smoke test data loader
python data_loader.py mini

# Verify model shapes
python model.py

# Train IL baseline (Stage B) on mini split
python train.py --split mini --epochs 5 --batch_size 8

# Full training
python train.py --split train_boston --epochs 50 --batch_size 32

# Evaluate (best-of-60 L2 displacement)
python evaluate.py --split mini --checkpoint checkpoints/best.pt
```

## Hyperparameters

All values from Section 4.1 / `paper/hyperparameters.md`:

| Parameter | Value |
|-----------|-------|
| Modes (N_lon × N_lat) | 12 × 5 = 60 |
| Horizon T | 8 steps |
| History H | 10 frames |
| Hidden dim D | 256 |
| Optimizer | AdamW, lr=1e-4 |
| LR schedule | ReduceLROnPlateau |
| Batch size | 32 |
| Epochs | 50 |

## Ablation flags (`config.py`)

| Flag | IL best | RL best |
|------|---------|---------|
| `MODE_DROPOUT` | ✓ | ✓ |
| `SELECTOR_SIDE_TASK` | ✓ | ✓ |
| `EGO_HISTORY_DROPOUT` | ✓ | ✗ |
| `BACKBONE_SHARING` | ✓ | ✗ |

## Status

- [x] Stage A: Transition model
- [x] Stage B: IL pre-training (PointNet + IVM + autoregressive decoding)
- [ ] Stage C: RL fine-tuning (PPO) — in progress
- [ ] Map encoding (nuplan lane polylines) — in progress
