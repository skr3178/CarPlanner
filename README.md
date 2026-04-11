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
python train_stage_b.py --split mini --epochs 5 --batch_size 8

# Full training
python train_stage_b.py --split train_boston --epochs 50 --batch_size 32

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

## Key concepts

Current drawbacks:
1. Causal Confusion: Imitation learning suffers from distribution shift and causal confusion. Learning co-relations instead of actual causations

Model sees dashboard + road
When braking, brake light turns ON
Model learns: "Brake when brake light is ON"

2. 


## Issues with auto-regressive planners and trajectories

Classical autoregressive : predict next pose -> Feed it back --> Predict next pose- allowing for interactive dynamics and behaviors


1. Long term inconsistency

2. Multi model explosion- exploration space explosion exponentially

## Special kind of AutoRegressive models- Expert guided and task-oriented terms

1. Universal and diverse reward functions- eliminate scenario specific reward functions

2. 

Questions that need answering:
- Perform an ablation study on these with and without Autoregressive IVM. 
- Mode selector- Modes score - what are the different modes and how are they defined
- Vanilla vs autoregressive 
- Non reactive component- other cars do not changes shape- is this then a partial world model? 
- Rule selector
- Design of IVM (Invariant-view module)


## Planning over an Horizon is important: Short, Medium and Long Horizon tasks

Short Horizon: Pros: very reactive,  Cons: very efficient, needs massive data 

Long horizons: Pros: reactive enough, efficient enough, Cons: Less reactive, say child comes in front or pedestrian comes over. 

What is it trying to solve for? To optimize

π max ​   E  s  t ​   ∼P  τ ​   ,a  t ​   ∼π ​   [  t=0 ∑ T ​   γ  t  R(s  t ​   ,a  t ​   )] 

we have pi. 

sum_{t=0 to T} gamma^t * R(s_t, a_t)

Is this the reward function? Trying to optimize for? 
R(s_t, a_t) =
  w1 * forward_progress
- w2 * collision_penalty
- w3 * jerk_penalty
- w4 * lane_deviation


## Change standard RL formulation to autoregressive type: Leads to faster simulation and efficient training 
Before:
s_{t+1} ~ P(s_{t+1} | s_t, a_t)
After match trick:
s_{t+1} ~ P_hat(s_{t+1} | s_t, a_t)

Advantages: 
- simulate multiple trajectories
- GPU based
- Without real env

Planner architecture components
1) the nonreactive transition model, 
2) the mode selector, 
3) the trajectory generator, and 
4) the rule-augmented selector.

Initial state (s_0), possible N_modes
trajectory selector evaluates and assignes score to each mode. 

Creating N_mode parallel worlds.
Policy rollout, trajectory predictor acts as state transition model.
Generating future poses of traffic agents across time horizons. 


Reward function defination: 
Negative displacement error (DE) between ego future pose and ground truth as a universal reward.
Trajectory quality improvement terms: 
1. collision rate 
2. Drivable area compliance

Loss function: PPO - Policy improvement, value estimation entropy, cross-entropy loss estimator

IVM (invariant view module)

## Benchmarks 
- Test14 Random
- Reduced-Val14

CLS: closed loop scores- nuPlan dev kit
1. reactive CLS-R
2. non reactive CLS-NR

Why carplanners refernce PDM (Policy Driven Motion Planning):
- Define a policy over a set of candidate trajectories- score and select best one
- Trajectory (safety, comfort, progress, rule compliance)


## Use of 2 different loss terms: 
IL: Imitation learning
RL: Reinforcement learning

Total Loss = RL loss + IL loss

IL loss → learn from expert data
RL loss → optimize behavior using reward
This is called Hybrid RL + Imitation Learning.

Pure RL has problems:
- Very slow learning
- Needs lots of exploration
- Dangerous for driving
- Hard to converge
So RL alone is unstable and inefficient.

Why Not Just Use Imitation Learning?
- Pure IL also has problems:
- Distribution shift
- Causal confusion
- Cannot exceed expert performance and generalizes
- So IL alone is limited.


## Rewards

- Quality rewards
- DE rewards (negative displacement error): between the ego future pose and the ground truth as a universal reward

## Dim=9

### Agent Features (Da=10)

| # | Feature | Source | Status |
|---|---------|--------|--------|
| 1 | x | center.x | ✅ |
| 2 | y | center.y | ✅ |
| 3 | heading | center.heading | ✅ |
| 4 | vx | velocity.x | ❌ (using scalar speed) |
| 5 | vy | velocity.y | ❌ |
| 6 | box_w | box.width | ❌ |
| 7 | box_l | box.length | ❌ |
| 8 | box_h | box.height | ❌ |
| 9 | time_step | relative timestamp | ❌ |
| 10 | category | tracked_object_type | ❌ |

We currently only use 4 of 10. Missing: vx, vy, box dims, time_step, category.

### Map Point Features (Dm=9)

The paper says x, y, heading, speed_limit, category per point, but that's 5 not 9. The most likely 9-dim encoding:

| # | Feature | Source |
|---|---------|--------|
| 1-2 | x, y | discrete_path[i].x/.y |
| 3-4 | sin(heading), cos(heading) | discrete_path[i].heading |
| 5 | speed_limit | lane.speed_limit_mps |
| 6-9 | category one-hot | LANE / LANE_CONNECTOR / CROSSWALK / STOP_LINE |

## Paper vs Implementation: Algorithm-by-Algorithm

### Algorithm 1 — Training Procedure

#### Step 1: Train Transition Model (Alg 1, lines 3–8)

**Paper:** Loop over D, run β(s_0), compute L_tm (Eq 5), backprop, update β, then freeze.

**Implementation:** `TransitionModel` exists (`model.py:385`) but no training loop calls it. `train_stage_b.py` only implements Stage B. β is permanently random and never frozen because it was never trained.

**Gap:** Stage A entirely missing (known, in-scope to fix later).

---

#### Step 2, lines 13–14 — Agent simulation via β during IL

**Paper (line 14):** s^{1:N}_{1:T} ← β(s_0) — even in IL, the transition model β generates the agent futures used by the policy.

**Implementation:** `train_stage_b.py:108` passes `agents_seq = batch['agents_seq']` — GT future agent states loaded from the DB, not β output. `forward_train` never calls `self.transition_model`.

**Gap:** IL training uses GT agent oracle instead of β. Correct for teacher-forcing baseline but diverges from the paper's algorithm.

---

#### Step 2, lines 15–18 — Mode assignment

**Paper (line 16):** c_lat determined from s_0 (current state / route). Line 17: concatenate c_lat + c_lon → c. Line 18: determine c* from GT trajectory and c.

**Implementation (`data_loader.py:174`):** Both c_lat (lateral bin) and c_lon are derived purely from `gt_trajectory[-1]` (endpoint position/speed). There is no route or s_0 reference for c_lat.

**Gap:** c_lat should come from route proximity to s_0, not GT endpoint lateral offset.

---

#### Step 2, lines 19–21 — Mode Selector call signature

**Paper (line 20):** σ, s̄ ← f_selector(s_0, c) — mode c is an input to the selector.

**Implementation (`model.py:221`):**
```python
def forward(self, global_feat: torch.Tensor):  # only takes s_0 encoding
```
ModeSelector takes only the s_0 global feature. Mode c is not passed in.

**Gap:** Mode selector doesn't condition on c.

---

#### Step 2, lines 28–31 — IL generator loss ✓

**Paper:** Use π, s_0, c*, s^{1:N}_{1:T} → collect a_{0:T-1} → stack → L1Loss vs GT.

**Implementation (`train_stage_b.py:117`, `train_stage_b.py:44`):**
```python
pred_traj = model.forward_train(agents_seq, ..., mode_c=c, gt_ego=gt_traj)
L_gen = (pred_traj - gt_traj).abs().sum(dim=-1).mean()
```
Match. ✓

---

#### Step 2, lines 33–35 — Overall loss ✓

**Paper:** L = L_selector + L_generator, update f_selector and π.

**Implementation (`train_stage_b.py:46`):** `L_total = L_CE + L_side + L_gen` where `L_selector = L_CE + L_side`. ✓

---

#### Step 2, lines 36–40 — π_old update every I steps

**Paper:** PPO delayed policy update.

**Implementation:** Not implemented. RL-only, intentionally skipped. ✓ (out of scope)

---

### Algorithm 2 — Inference

#### Step 1 — Encode s_0

**Paper:** "PointNet + Transformer" to encode s_0.

**Implementation (`model.py:512`):** `s0_encoder = PointNetEncoder` only — no Transformer at the s_0 encoding stage. The Transformer is inside IVM (per-step), not at initial encoding.

**Minor gap:** No global Transformer over s_0 features.

---

#### Step 2 — Mode Selector scores ✓

`mode_logits, _ = self.mode_selector(s0_global)` at `model.py:611`. Match. ✓

---

#### Steps 3–4 — Autoregressive decode per mode ✓

Loop over 60 modes at `model.py:619–631`, IVM applied inside `AutoregressivePolicy`. Match. ✓

**Minor gap:** Paper says "use mean of Gaussian" for deterministic inference. The action head at `model.py:271` outputs a deterministic 3-vector — no distribution is modelled at all. No Gaussian, no sampling.

---

#### Step 5 — Rule Selector

**Paper:** Scores = collision + drivable area + comfort + progress, combined with mode scores.

**Implementation (`model.py:441–480`):** Only comfort (jerk) and progress (final x) are implemented. Collision and drivable area checks are missing.

**Gap:** Rule selector is incomplete — 2 of 4 rule criteria absent.

---

### Algorithm 3 — IVM

#### Step 1 — K-nearest agent selection ✓

`IVMBlock` at `model.py:176`, `k_nn = N_AGENTS // 2`. Match. ✓

---

#### Step 2 — Route filtering (lateral mode)

**Paper:** Filter route segments where the closest point to ego is the starting point; retain K_r = N_r / 4 points per route.

**Implementation:** No route concept at all. Map lanes are treated uniformly — all lanes within 50m radius, no route-specific filtering, no K_r logic.

**Gap:** Route filtering entirely absent from IVM.

---

#### Step 3 — Transform to ego frame at each step t

**Paper:** All agent, map, and route poses transformed to ego's current frame at time t.

**Implementation (`model.py:336`):** Agent poses are re-transformed per step ✓. But map lanes are encoded once at t=0 (`model.py:316–317`) and reused for all T steps — never re-transformed to the updated ego frame at t.

**Gap:** Map features are static across timesteps; should be re-transformed per step.

---

#### Step 4 — Time normalisation (t-H:t → -H:0)

**Paper:** Subtract history window so time indices are always in [-H, 0].

**Implementation:** Only the current-step agent state is used per timestep. No agent history window is maintained or time-normalised inside IVM. `agents_seq` is indexed as a single frame per step with no H-length window.

**Gap:** No temporal history in IVM — algorithm expects H past frames at each step.

---

#### Transformer decoder — value head

**Paper (bottom of Alg 3):** "Decoded through MLP → policy head (action distribution params) + value head (scalar)"

**Implementation:** `action_head` exists (`model.py:271`). No value head anywhere in `AutoregressivePolicy` or `IVMBlock`.

**Gap:** Value head missing (needed for RL; irrelevant for IL-only baseline).

---

### Summary Table

| Algorithm step | Match? | Gap |
|---|---|---|
| Alg 1 Step 1 — Transition model training | No | Stage A not coded |
| Alg 1 line 14 — β generates agent futures | No | GT agents used instead |
| Alg 1 line 16 — c_lat from s_0/route | No | c_lat from GT endpoint y-offset |
| Alg 1 line 20 — f_selector(s_0, c) | No | c not passed to selector |
| Alg 1 lines 28–31 — IL L_gen | ✓ | |
| Alg 1 lines 33–35 — total loss | ✓ | |
| Alg 2 step 1 — s_0 encoding | Partial | No global Transformer |
| Alg 2 step 4 — Gaussian policy | No | Deterministic head, no distribution |
| Alg 2 step 5 — Rule selector | Partial | Collision + drivable area missing |
| Alg 3 step 1 — K-NN agents | ✓ | |
| Alg 3 step 2 — Route filtering | No | No route concept |
| Alg 3 step 3 — Map re-transform per step | No | Map encoded once at t=0 |
| Alg 3 step 4 — Time history H | No | Single frame per step, no H window |
| Alg 3 decoder — value head | No | Missing (RL only) |

**Most impactful gaps for IL quality:** c_lat from route (not GT), map not re-transformed per IVM step, and no agent time history in IVM.

## IVM 

IVM strips out everything that would make the policy time-dependent — absolute position, absolute time, distant irrelevant agents — so the same learned policy
generalises across all timesteps in the rollout.

Techniques:
 
1. K-nearest neighbor filtering (reduce noise)
Why: Removes irrelevant context, focuses attention on what actually matters for the next action.

2. Ego-centric coordinate transform (remove absolute position)
Why: The policy never sees global coordinates. It only sees "what's around me right now." This makes it position-invariant.

3. Time normalisation (remove absolute time)
Why: The policy always sees "the last H steps leading to now" with the same indices. It never knows whether it's at absolute step 3 or step 7.



### nuPlan Database Tables

| Table                | Rows      | Purpose                                            |
|----------------------|-----------|----------------------------------------------------|
| log                  | 1         | Scenario metadata (vehicle, date, location, map)   |
| ego_pose             | 51,152    | AV pose at every IMU sample (~100Hz)               |
| lidar_pc             | 10,200    | LiDAR sweep timestamps + file refs (10Hz)          |
| lidar_box            | 1,280,922 | Detected bounding boxes per LiDAR sweep            |
| track                | 6,460     | Unique tracked objects (vehicle/ped/bike dims)     |
| scenario_tag         | 13,812    | Labels marking which frames belong to scenario types |
| traffic_light_status | 96,994    | Red/green per lane connector per frame             |
| scene                | 26        | Scene-level goal poses + roadblock IDs             |
| category             | 7         | Object class definitions                           |

---

### ego_pose columns and ranges

| Column             | Range                                          | Unit          |
|--------------------|------------------------------------------------|---------------|
| x, y               | [664,425→664,656], [3,996,370→3,999,268]       | m (global UTM)|
| vx, vy             | [-0.02→14.05], [-0.32→0.04]                    | m/s           |
| acceleration_x     | [-1.77→3.07]                                   | m/s²          |
| qw/qx/qy/qz        | quaternion                                     | —             |
| angular_rate_x/y/z | —                                              | rad/s         |

---

### lidar_box columns and ranges (detected agents)

| Column     | Range                                          | Unit          |
|------------|------------------------------------------------|---------------|
| x, y       | [664,348→664,711], [3,996,297→3,999,330]       | m (global UTM)|
| vx, vy     | [-17.4→15.8], [-21.2→20.6]                     | m/s           |
| width      | [0.19→4.48]                                    | m             |
| length     | [0.17→16.17]                                   | m             |
| yaw        | —                                              | rad           |
| confidence | [0.32→1.00]                                    | —             |

---

### Category types

| Name           | Description                  |
|----------------|------------------------------|
| vehicle        | Cars, trucks, trailers       |
| bicycle        | Bikes, motorcycles           |
| pedestrian     | All pedestrians              |
| traffic_cone   | Temporary cones              |
| barrier        | Permanent/temporary barriers |
| czone_sign     | Construction zone signs      |
| generic_object | Animals, debris, poles       |

---

### scenario_tag types (frequency)

| Type                          | Count |
|-------------------------------|-------|
| stationary                    | 3,020 |
| stationary_in_traffic         | 2,034 |
| traversing_intersection       | 1,667 |
| on_traffic_light_intersection | 1,291 |
| high_magnitude_speed          | 1,122 |
| near_pedestrian_on_crosswalk  | 250   |
| traversing_crosswalk          | 65    |
| ...                           | ...   |                                                                                                   

---

### Traffic Light Status (mini split)

| Status | Count  |
|--------|--------|
| Green  | 41,241 |
| Red    | 55,753 |

---

### Stage B IL Training Results (mini split, 5 epochs)

| Epoch | L_total | L_CE   | L_side | L_gen  | Time |
|-------|---------|--------|--------|--------|------|
| 1     | 2.9378  | 1.8570 | 0.9831 | 0.0977 | 167s |
| 2     | 1.7210  | 1.1855 | 0.4749 | 0.0606 | 167s |
| 3     | 1.3520  | 0.9520 | 0.3414 | 0.0587 | 168s |
| 4     | 1.1497  | 0.8126 | 0.2813 | 0.0559 | 167s |
| 5     | 1.0062  | 0.7132 | 0.2437 | 0.0493 | 167s |

#### Loss term interpretation

| Loss   | Supervises                        | High value means                          |
|--------|-----------------------------------|-------------------------------------------|
| L_CE   | Mode classifier                   | Can't identify the right behavioral intention |
| L_side | Mode selector's trajectory sketch | Mode choice doesn't match trajectory shape |
| L_gen  | Autoregressive policy rollout     | Policy can't predict next waypoints accurately |

The dominant signal is `L_CE` — mode classification is the hardest task and drives most of the learning.

---

### Stage A — TransitionModel I/O

**Inputs:**

| Tensor        | Shape              | Description |
|---------------|--------------------|-------------|
| `agents_now`  | `(B, N=20, Da=10)` | Agent states at t=0 in ego frame: x, y, heading, vx, vy, box_w, box_l, box_h, time_step, category |
| `agents_mask` | `(B, N=20)`        | Validity mask (1=valid agent, 0=padding) |

**Output:**

| Tensor        | Shape               | Description |
|---------------|---------------------|-------------|
| `agents_pred` | `(B, T=8, N=20, Da=10)` | Predicted future agent states for all 8 steps |

**Ground truth label (loss only):**

| Tensor       | Shape               | Description |
|--------------|---------------------|-------------|
| `agents_seq` | `(B, T=8, N=20, Da=10)` | GT future agent states from dataset |

**Loss (Eq 5):**

```
L_tm = masked L1(agents_pred, agents_seq)
     = mean over (B, T, valid N) of ||pred - gt||_1

Mask ensures padding agent slots (mask=0) don't contribute to the gradient.
```

Architectural components (what the model IS):
1. Transition Model — predicts future agent positions
2. Trajectory Generator = ModeSelector + AutoregressivePolicy — predicts ego trajectory
3. Rule-Augmented Selector — inference-only, picks best trajectory from 60 candidates using rules (no learning)

Training stages (how it's TRAINED):
- Stage A — pre-train the Transition Model alone (Eq 5)
- Stage B — IL training of the Trajectory Generator (ModeSelector + AutoregressivePolicy) using GT data
- Stage C — RL fine-tuning of the Trajectory Generator using the trained Transition Model as simulator

Here's a full breakdown of every symbol in Algorithm 1:

---

## Algorithm 1 Notation

### Inputs / Requires

| Symbol | Name | Meaning |
|---|---|---|
| $\mathcal{D}$ | Dataset | nuPlan scenarios, each a tuple of initial state + GT trajectories |
| $s_0$ | Initial state | Scene at t=0: ego pose, all agent states, map — the full observation |
| $s^{0:N,\text{gt}}_{1:T}$ | GT trajectories | Ground-truth future states for agents 0..N across timesteps 1..T. Superscript = agent index (0=ego, 1..N=others). Subscript = time range |
| $c_{\text{lon}}$ | Longitudinal mode label | Pre-computed speed bin from GT — which of 12 speed intervals the ego lands in |
| $\gamma$ | Discount factor | RL reward discounting (=0.1 per paper) |
| $\lambda$ | GAE parameter | GAE advantage estimation smoothing (=0.9 per paper) |
| $I$ | Update interval | How many steps between copying π → π_old (PPO delayed update) |
| $\beta$ | Transition model | The non-reactive model that predicts agent futures given s_0 |
| $f_{\text{selector}}$ | Mode selector | Predicts mode scores σ and side-task trajectory s̄ |
| $\pi$ | Policy (current) | Autoregressive trajectory generator, updated each step |
| $\pi_{\text{old}}$ | Policy (old) | Frozen copy of π used for PPO importance ratio — updated every I steps |

---

### Stage 1 (lines 3–8) — Transition Model Training

| Symbol | Meaning |
|---|---|
| $s^{1:N}_{1:T}$ | Predicted agent futures — agents 1..N (NOT ego) at times 1..T, output of β |
| $s^{1:N,\text{gt}}_{1:T}$ | GT agent futures — same shape, from dataset |
| $L_{\text{tm}}$ | Transition model loss — L1 between predicted and GT agent states (Eq 5) |

---

### Stage 2 (lines 9–41) — Selector + Generator Training

| Symbol | Meaning |
|---|---|
| $s^{0,\text{gt}}_{1:T}$ | GT **ego** trajectory (agent index 0 = ego), timesteps 1..T |
| $s^{1:N}_{1:T}$ | Agent futures from β (now frozen) — used as the simulated world during rollout |
| $c_{\text{lat}}$ | Lateral mode — which of 5 lateral bins (determined from route proximity to s_0) |
| $c$ | Full mode = concatenate(c_lat, c_lon) → one of 60 modes |
| $c^*$ | Positive mode — the single GT-assigned mode label used for CE loss supervision |
| $\sigma$ | Mode logits — (N_modes,) scores output by f_selector |
| $\bar{s}^0_{1:T}$ | Side-task trajectory — rough ego trajectory predicted by f_selector conditioned on c |
| $L_{\text{selector}}$ | Selector loss = CE(σ, c*) + SideTaskLoss(s̄, s_gt) |

**RL branch only (lines 23–27):**

| Symbol | Meaning |
|---|---|
| $s_{0:T-1}$ | State sequence collected during rollout |
| $a_{0:T-1}$ | Action sequence (waypoints) output by π during rollout |
| $d_{0:T-1}$ | Policy distribution parameters at each step (Gaussian mean/var for PPO ratio) |
| $V_{0:T-1}$ | Value estimates from value head at each step |
| $R_{0:T-1}$ | Rewards at each step (from nuPlan scoring: collision, drivable, comfort, progress) |
| $A_{0:T-1}$ | GAE advantages computed from R and V |
| $\hat{R}_{0:T-1}$ | GAE returns (discounted reward targets for value loss) |
| $d_{0:T-1,\text{new}}$ | New policy distribution from updated π (used for PPO clipped ratio) |
| $V_{0:T-1,\text{new}}$ | New value estimates from updated π |
| $L_{\text{generator}}$ (RL) | = ValueLoss + PolicyLoss − Entropy |

**IL branch only (lines 28–31):**

| Symbol | Meaning |
|---|---|
| $a_{0:T-1}$ | Action sequence from π under teacher forcing |
| $s^0_{1:T}$ | Stacked actions = predicted ego trajectory |
| $L_{\text{generator}}$ (IL) | = L1(s^0_{1:T}, s^{0,gt}_{1:T}) — Eq 11 |

---

### Key indexing convention

- **Superscript** = agent index: `0` = ego, `1:N` = other agents
- **Subscript** = time range: `1:T` = future steps, `0:T-1` = current+future steps (0-indexed for actions)
- $s^{1:N}_{1:T}$ from β vs $s^{0,\text{gt}}_{1:T}$ from dataset — same shape structure, different agent/source


CarPlanner uses the Advantage actor critic in a revised form. 

A2C v/s CarPlanner (PPO + GAE)
- Biggest difference- Stability. 
- The problem is nothing stops the policy from updating too aggressively on a lucky batch, which can cause catastrophic forgetting or collapse.
- rt = π_new(at|st) / π_old(at|st)
- L = -min( rt · At,  clip(rt, 1-ε, 1+ε) · At )

- CarPlanner's RL training is actor-critic at its core — the trajectory generator is the actor, and there's a value head on the same network acting as the critic. 
- PPO+GAE is best understood as A2C with two important engineering improvements bolted on: better advantage estimates (GAE) and a safety constraint on how much the policy can change per update (clipping). 
- Those two additions are what make it practical for large-scale real-world training.

Reference of earier implementaion of Hierarchical Actor Critic:
```
https://github.com/skr3178/reinforcement_learning/blob/main/HAC/README.md
```

![explainations](paper/claude_explainations/c_star_intersection.svg)

## All Neural Networks trained

1. β — the transition model
Takes the initial scene s₀ and predicts where all other traffic agents (cars, pedestrians) will be at every future timestep. It models the world, not the ego car. Crucially it is non-reactive — it ignores whatever the ego car does. Think of it as a trajectory predictor for everyone else.
- Inputs
s0 = {agents_now, ego_history, map, masks}
Outputs: agents_future = (T_future, N_agents, features)

where 
agents_now = {N_agents, features:[[x, y, vx, vy, heading, length, width]]}
ego_history = {T_history, features}
map_lanes = (N_agents, lane_features:[lane centerlines, drivable area, road topology])

- Outputs
agents_future = (T_future, N_agents, features)
predict future trajectory for every other agent
L_tm = L1(pred, gt); backprop, update beta
ego plan + predicted traffic, simulation without simulator

beta is frozen for subsequent steps as it determines the future world, it becomes instable, RL is unstable.

2. f_selector — the mode selector
Given the initial scene, it scores all possible driving modes (e.g. "turn left slowly", "go straight fast"). It answers: which of these modes is most appropriate right now? This is trained with supervised learning using the ground truth trajectory to identify the correct mode.


3. π — the policy (trajectory generator)
The actual planner. Given the current state and a chosen mode c, it autoregressively outputs the next ego waypoint at each step. This is the actor in the actor-critic sense. π_old is a frozen copy of π used for PPO's stability constraint


For details check view of what one sample `(s_0, s^{0,\text{gt}}_{1:T})` looks like.


Longitudinal mode (c_lon) answers "how fast?" — the GT endpoint's speed tells us which of the 12 speed buckets the human driver was in.

Lateral mode (c_lat) answers "which direction?" — the GT endpoint's position tells us which available route the human was following.


To train Stage A:
```
python3 train_stage_a.py --split mini --epochs 5 --batch_size 4
```
Then use in Stage B:
```
python3 train_stage_b.py --split mini --epochs 50 --transition_ckpt checkpoints/stage_a_best.pt
```

To train Stage A:
```
python3 train_stage_a.py --split mini --epochs 5 --batch_size 4
```
Then use in Stage B:
```
python3 train_stage_b.py --split mini --epochs 50 --transition_ckpt checkpoints/stage_a_best.pt
```

CrossEntropy  →  trains selector to pick the RIGHT mode
SideTaskLoss  →  trains selector to UNDERSTAND the scene deeply


SideTaskLoss = (1/T) Σ_t  | ego_pred[t] - ego_gt[t] |
                t=1..8

where each timestep:
  ego_pred[t] = [x, y, yaw]   ← what selector thinks ego will do
  ego_gt[t]   = [x, y, yaw]   ← what expert actually did

Different losses: 

1. L_tm → β. β's job is purely predictive — given the scene, predict where other agents physically go. L1 on positions is the natural fit: penalise the distance between predicted and actual poses at every timestep, averaged over all agents.
2. L_selector → f_selector. The selector has to answer two questions simultaneously — which mode is correct (classification), and can it also roughly predict the ego trajectory (regression). CrossEntropy handles the classification part. The L1 side task is a bonus signal that forces the selector's internal features to be rich enough to reason about the future, not just assign labels.
3. L_generator → π. This is where IL and RL diverge. In IL, the generator is just told "copy the expert" — L1 between predicted and GT waypoints. In RL, the generator is told "maximise reward" — PPO loss which includes the policy gradient term, a value regression term, and an entropy bonus to prevent the policy from collapsing to a single deterministic action too early.


### The value of c*:
A single integer pointing to one cell in the 4×12 grid

### Weight of entropy

ValueLoss   magnitude ~ 10³   weight = 3      → contribution ~ 3,000
PolicyLoss  magnitude ~ 10⁰   weight = 100    → contribution ~ 100
Entropy     magnitude ~ 10⁻³  weight = 0.001  → contribution ~ tiny

Entropy of gaussian distribution is the H(X)=21​ln(2πeσ2), which is dependent on the variance or the spread of the distribution. 
High σ → distribution is wide → high entropy → policy is uncertain
Low σ  → distribution is narrow → low entropy → policy is confident

The minus sign on entropy. Minimising L_gen means maximising entropy. This is deliberate — it is called an entropy bonus and it serves one purpose: preventing the policy from collapsing too early.

### Check with actual loss term values

Term          Raw magnitude    Weight    Contribution
──────────────────────────────────────────────────────
ValueLoss         ~10³           3          ~3,000
PolicyLoss        ~10⁰          100          ~100
Entropy           ~10⁻³        0.001        ~tiny


ValueLoss dominates because accurate value estimation is the foundation of PPO. If V is wrong, the advantage estimates A_t are wrong, and the policy gradient signal is meaningless noise. You need the critic to be well-calibrated before the actor can improve.

PolicyLoss is secondary because the actual policy update should be conservative — PPO is designed to take small, stable steps. A contribution of ~100 versus ~3,000 for value means the policy updates are restrained relative to critic updates. This matches the PPO philosophy of trust region — do not let the policy change too fast.

Entropy is a whisper because it is purely a regulariser. It should never compete with the actual learning signal. 0.001 weight means it only matters when ValueLoss and PolicyLoss are both near zero — i.e. when the policy is already good and in danger of collapsing. At that point the entropy term nudges it to stay exploratory.



  ┌──────────────────────────┬───────────────────────────────────┬────────────────────────────────────────────────────────────┐                           
  │        Component         │            Input shape            │                        Output shape                        │                           
  ├──────────────────────────┼───────────────────────────────────┼────────────────────────────────────────────────────────────┤                           
  │ LaneEncoder              │ (2, 20, 10, **27**)               │ (2, 20, 64)                                                │                           
  ├──────────────────────────┼───────────────────────────────────┼────────────────────────────────────────────────────────────┤                           
  │ DecomposedModeEncoder    │ routes (2, 5, 10, **27**)         │ lon(12,256) lat(2,5,256) tensor(2,5,12,256) query(2,1,256) │                           
  ├──────────────────────────┼───────────────────────────────────┼────────────────────────────────────────────────────────────┤                           
  │ ModeSelector             │ global(2,256) + map(2,20,10,27)   │ logits(2,60) side(2,8,3)                                   │                           
  ├──────────────────────────┼───────────────────────────────────┼────────────────────────────────────────────────────────────┤                           
  │ TransitionModel          │ agents(2,20,10) + map(2,20,10,27) │ (2,8,20,10)                                                │                           
  ├──────────────────────────┼───────────────────────────────────┼────────────────────────────────────────────────────────────┤                           
  │ AutoregressivePolicy     │ agents_seq(2,8,20,10)             │ (2,8,3)                                                    │                           
  ├──────────────────────────┼───────────────────────────────────┼────────────────────────────────────────────────────────────┤                           
  │ CarPlanner.forward_train │ full inputs                       │ logits(2,60) side(2,8,3) pred(2,8,3)                       │  


## commands to start training: 

# Stage B — IL training (run after Stage A finishes)
nohup python -u train_stage_b.py --split mini --epochs 50 --batch_size 4 \
     --transition_ckpt checkpoints/stage_a_best.pt \
     > checkpoints/stage_b_train.log 2>&1 &
echo "Stage B PID: $!"

# Stage C — RL fine-tuning (run after Stage B finishes)
nohup python -u train_stage_c.py --split mini --epochs 50 --batch_size 4 \
     --stage_a_ckpt checkpoints/stage_a_best.pt \
     --stage_b_ckpt checkpoints/best.pt \
     > checkpoints/stage_c_train.log 2>&1 &
echo "Stage C PID: $!"

claude:
claude --resume c9c246ec-6d4a-48b2-9aa6-bae8fc225246 

claude-glm:
claude --resume "stage-c-rl-finetuning"


## commands: 

Stage A (50 epochs) → eval_stage_a.py → Stage B (50 epochs) → Stage C (50 epochs)                                               
                                                                                                                                  
  Stage A — train transition model β on real agent history. Checkpoint: stage_a_best.pt                                           
                                                                                                                                  
  Eval Stage A — run eval_stage_a.py to confirm model beats stationary baseline before proceeding. If it fails, retrain with more 
  epochs.                                                                                                                         
                                                                                                                                  
  Stage B — IL training. Loads frozen β, trains mode selector + autoregressive policy on GT ego trajectories. Checkpoint: best.pt 
                                                                                                                                  
  Stage C — RL fine-tuning. Loads frozen β + Stage B weights, fine-tunes with PPO using reward signal. Checkpoint: stage_c_best.pt
                                                                                                                                  
# Commands in order:                                                                                                              
                                                                                                                        
## 1. Stage A                                                                                                                    
nohup python -u train_stage_a.py --split mini --epochs 50 --batch_size 1024 > checkpoints/stage_a_train.log 2>&1 &              
                                                                                                                        
## 2. Eval Stage A (after Stage A finishes)                                                                                      
python -u eval_stage_a.py --ckpt checkpoints/stage_a_best.pt --split mini --n_batches 50                                        
                                                                                                                        
## 3. Stage B (after Stage A passes eval)                                                                                        
nohup python -u train_stage_b.py --split mini --epochs 50 --batch_size 64 --transition_ckpt checkpoints/stage_a_best.pt >       
checkpoints/stage_b_train.log 2>&1 &                                                                                            
                                                                                                                        
## 4. Stage C (after Stage B finishes)                                                                                           
nohup python -u train_stage_c.py --split mini --epochs 50 --batch_size 64 --stage_a_ckpt checkpoints/stage_a_best.pt            
--stage_b_ckpt checkpoints/best.pt > checkpoints/stage_c_train.log 2>&1 &                                                       



## Wrapper

● All checks pass. The model accepts the exact tensor shapes that the feature builder produces:

- agents_now: (1, 20, 10)
- agents_mask: (1, 20)
- map_lanes: (1, 20, 10, 27)
- agents_history: (1, 10, 20, 10)
- Output best_traj: (1, 8, 3) — the 8-step trajectory in ego frame

● Here's a summary of everything created:

Files Created (no existing files modified)

src/nuplan_wrapper/
├── __init__.py                          # Package init
├── carplanner_feature_builder.py        # nuPlan → CarPlanner tensor conversion
└── carplanner_planner.py                # AbstractPlanner wrapper

config/nuplan/
└── carplanner.yaml                      # Planner config

scripts/
└── eval_nuplan.py                       # Evaluation launcher script

How It Works

1. CarPlannerFeatureBuilder — At each simulation step, takes nuPlan's PlannerInput and converts:
- history.observations → agents_now (N, 10) + agents_history (H, N, 10)
- initialization.map_api → map_lanes (20, 10, 27) with center/left/right boundaries
- All coordinates transformed to ego-centric frame (same as data_loader.py)
2. CarPlannerPlanner — Wraps the model and handles:
- Loading checkpoints for stages A/B/C
- Calling model.forward_inference()
- Converting output (T, 3) ego-centric trajectory to global coordinates
- Returning InterpolatedTrajectory for the nuPlan simulator
3. eval_nuplan.py — Launch script:
python scripts/eval_nuplan.py \
     --checkpoint checkpoints/stage_c_best.pt \
     --stage c --split test14-random

Before Running

You'll need cv2 installed for nuPlan devkit imports:
pip install opencv-python

And the nuPlan devkit environment must be fully set up (the simulation requires the full devkit stack).
Now I have a **much more complete picture**, and my earlier answer was incomplete. Two big new findings:

## Finding 1: The model is **worse than a trivial copy baseline**

```
Naive baseline "predict s_0 for all T"  →  L_tm = 14.4423
Trained model (epoch 182, best)          →  L_tm = 15.3507
Naive baseline "predict zeros"           →  L_tm = 52.5947
```

**The trained model never even reached the copy-s_0 baseline.** It started at ~50 (close to predict-zeros), descended, and got stuck at 15.35 — *above* 14.44. So the LR decay isn't the only problem; even with a healthy LR, the model found a local minimum worse than a trivial copy.

## Finding 2: Raw, unnormalized features with huge dynamic range

Per-feature stats of `agents_seq` (valid agents only):

| Dim | Feature | abs_mean | std | range |
|---|---|---|---|---|
| 0 | x | **29.58** | 34.55 | [-82, +91] |
| 1 | y | **12.25** | 16.12 | [-80, +81] |
| 2 | heading | 1.42 | 1.84 | [-π, π] |
| 3 | vx | 0.63 | 1.90 | [-17.7, 17.8] |
| 4 | vy | 0.52 | 1.58 | [-15.4, 13.0] |
| 5 | box_w | 1.71 | 0.76 | [0, 5.8] |
| 6 | box_l | 3.78 | 2.71 | [0, 20.0] |
| 7 | box_h | 1.78 | 0.66 | [0, 9.1] |
| 8 | **time_step** | **0.00** | **0.00** | **[0, 0]** |
| 9 | category | 0.93 | 1.56 | [0, 4] |

**Dim 8 (time_step) is dead — always zero.** That's a bug in `data_loader.py` extraction. Not critical but a loose end.

## Revised root-cause summary (ranked)

### Cause A: Model has no residual/copy inductive bias — **primary**
The architecture max-pools over the H=10 history frames (`model.py:1059`), collapsing all temporal information into one per-agent vector. The decoder then has to **reconstruct s_0 from scratch** for the output. There's no skip connection from s_0 to the output.

Consequence: the model can't easily express "predict current state" which is the optimal zero-effort baseline. It gets stuck in a local minimum *worse* than copy.

**Fix:** Make the model predict a **delta from s_0**:
```python
pred_delta = self.per_agent_head(...)  # (B, N, T, Da)
pred = s_0.unsqueeze(1) + pred_delta   # residual: easy init ≈ copy s_0
```
With this, zero initialization gives loss 14.44 from epoch 0, and the model only learns the hard part (the motion).

### Cause B: Raw unnormalized features — **contributing**
x has mean-abs 29.6m, y has 12.2m. The network regresses absolute meter values directly. Without normalization, gradients are dominated by far-away agents and box dims (constant but large). Standard fix: normalize per-feature to unit std.

### Cause C: LR collapse (confirmed) — **secondary**
At checkpoint: `lr = 6.56e-09` (8 reductions from 1e-4). Happened because `patience=0` was applied to noisy training loss instead of smooth validation loss. This **locked in** the bad local minimum of cause A — once stuck, LR couldn't recover.

### Cause D: PointNet max-pool over time destroys ordering — **architectural**
With `max(dim=1)` over H frames, the per-agent feature doesn't distinguish "moving right" from "moving left" in the temporal sense — only "which frame had the largest value per channel." A GRU or per-time-step attention would preserve ordering. But this may be secondary if the residual connection (Cause A fix) gives enough signal.

---

## What to actually fix (priority order)

1. **Add residual connection** in `TransitionModel.forward` — `pred = s_0 + delta`. This alone should drop loss below 14.44.
2. **Validation-based scheduler** — add val split, use val_loss for `scheduler.step()`.
3. **Normalize features** — at minimum, divide x/y by ~30m before feeding in and after predicting.
4. **Fix `time_step` extraction bug** in `data_loader.py` (low priority, doesn't affect plateau).

Without fix #1, fixing the LR alone will likely just let the model continue to its bad local minimum more slowly. The core bug is architectural, not just a schedule issue.

## Fixes needed:


Now I have a **much more complete picture**, and my earlier answer was incomplete. Two big new findings:

## Finding 1: The model is **worse than a trivial copy baseline**

```
Naive baseline "predict s_0 for all T"  →  L_tm = 14.4423
Trained model (epoch 182, best)          →  L_tm = 15.3507
Naive baseline "predict zeros"           →  L_tm = 52.5947
```

**The trained model never even reached the copy-s_0 baseline.** It started at ~50 (close to predict-zeros), descended, and got stuck at 15.35 — *above* 14.44. So the LR decay isn't the only problem; even with a healthy LR, the model found a local minimum worse than a trivial copy.

## Finding 2: Raw, unnormalized features with huge dynamic range

Per-feature stats of `agents_seq` (valid agents only):

| Dim | Feature | abs_mean | std | range |
|---|---|---|---|---|
| 0 | x | **29.58** | 34.55 | [-82, +91] |
| 1 | y | **12.25** | 16.12 | [-80, +81] |
| 2 | heading | 1.42 | 1.84 | [-π, π] |
| 3 | vx | 0.63 | 1.90 | [-17.7, 17.8] |
| 4 | vy | 0.52 | 1.58 | [-15.4, 13.0] |
| 5 | box_w | 1.71 | 0.76 | [0, 5.8] |
| 6 | box_l | 3.78 | 2.71 | [0, 20.0] |
| 7 | box_h | 1.78 | 0.66 | [0, 9.1] |
| 8 | **time_step** | **0.00** | **0.00** | **[0, 0]** |
| 9 | category | 0.93 | 1.56 | [0, 4] |

**Dim 8 (time_step) is dead — always zero.** That's a bug in `data_loader.py` extraction. Not critical but a loose end.

## Revised root-cause summary (ranked)

### Cause A: Model has no residual/copy inductive bias — **primary**
The architecture max-pools over the H=10 history frames (`model.py:1059`), collapsing all temporal information into one per-agent vector. The decoder then has to **reconstruct s_0 from scratch** for the output. There's no skip connection from s_0 to the output.

Consequence: the model can't easily express "predict current state" which is the optimal zero-effort baseline. It gets stuck in a local minimum *worse* than copy.

**Fix:** Make the model predict a **delta from s_0**:
```python
pred_delta = self.per_agent_head(...)  # (B, N, T, Da)
pred = s_0.unsqueeze(1) + pred_delta   # residual: easy init ≈ copy s_0
```
With this, zero initialization gives loss 14.44 from epoch 0, and the model only learns the hard part (the motion).

### Cause B: Raw unnormalized features — **contributing**
x has mean-abs 29.6m, y has 12.2m. The network regresses absolute meter values directly. Without normalization, gradients are dominated by far-away agents and box dims (constant but large). Standard fix: normalize per-feature to unit std.

### Cause C: LR collapse (confirmed) — **secondary**
At checkpoint: `lr = 6.56e-09` (8 reductions from 1e-4). Happened because `patience=0` was applied to noisy training loss instead of smooth validation loss. This **locked in** the bad local minimum of cause A — once stuck, LR couldn't recover.

### Cause D: PointNet max-pool over time destroys ordering — **architectural**
With `max(dim=1)` over H frames, the per-agent feature doesn't distinguish "moving right" from "moving left" in the temporal sense — only "which frame had the largest value per channel." A GRU or per-time-step attention would preserve ordering. But this may be secondary if the residual connection (Cause A fix) gives enough signal.

---

## What to actually fix (priority order)

1. **Add residual connection** in `TransitionModel.forward` — `pred = s_0 + delta`. This alone should drop loss below 14.44.
2. **Validation-based scheduler** — add val split, use val_loss for `scheduler.step()`.
3. **Normalize features** — at minimum, divide x/y by ~30m before feeding in and after predicting.
4. **Fix `time_step` extraction bug** in `data_loader.py` (low priority, doesn't affect plateau).

Without fix #1, fixing the LR alone will likely just let the model continue to its bad local minimum more slowly. The core bug is architectural, not just a schedule issue.

Want me to implement the residual connection and validation split?

data_loader.py — add polygon extraction alongside lanes:                                                                                   
# Currently only queries this:
map_objs = map_api.get_proximal_map_objects(point, cfg.MAP_QUERY_RADIUS, [SemanticMapLayer.LANE])                                          
                                                                                                                                       
# Needs to also query:                                                                                                                     
SemanticMapLayer.INTERSECTION                                                                                                              
SemanticMapLayer.CROSSWALK                                                                                                                 
SemanticMapLayer.STOP_LINE                                                                                                                 
                    
Each polygon needs to be resampled to Np points and encoded into its own tensor (N_POLY, N_PTS, D_POLY_POINT) — similar to lanes but for
closed polygons rather than open polylines.                                                                                                

model.py — add a second PointNet for polygons and concatenate with polyline features.                                                               
config.py — add N_POLYGONS, D_POLYGON_POINT constants.

┌───────────────────────────────┬──────────────────────┐
│            Metric             │        Value         │
├───────────────────────────────┼──────────────────────┤
│ Model L_tm                    │ 9.92                 │
├───────────────────────────────┼──────────────────────┤
│ Stationary baseline (copy-s₀) │ 13.25                │
├───────────────────────────────┼──────────────────────┤
│ Improvement over baseline     │ +25.2%               │
├───────────────────────────────┼──────────────────────┤
│ Non-reactivity ratio          │ 0.022 (target: <0.1) │
└───────────────────────────────┴──────────────────────┘
Both tests pass:
- The transition model beats the stationary baseline by 25% — it's learning meaningful agent dynamics
- The non-reactivity ratio is 0.022 — predicted agent futures barely change when ego position is perturbed, confirming the model correctly
ignores ego


# Need to add

Summary

┌────────────────────────────────┬───────────────────────────┬───────────────────────────────────────────┐
│              Fix               │          Affects          │            Requires retraining            │
├────────────────────────────────┼───────────────────────────┼───────────────────────────────────────────┤
│ IVM: 1 → 3 layers              │ Stage B, Stage C          │ Stage B + C (Stage A checkpoint reusable) │
├────────────────────────────────┼───────────────────────────┼───────────────────────────────────────────┤
│ TransitionModel: 2L/4H → 3L/8H │ Stage A                   │ Stage A from scratch, then B + C          │
├────────────────────────────────┼───────────────────────────┼───────────────────────────────────────────┤
│ Polygon encoder                │ All stages (data + model) │ Full re-extraction + all stages           │
├────────────────────────────────┼───────────────────────────┼───────────────────────────────────────────┤
│ s0 global Transformer          │ Stage B, Stage C          │ Stage B + C                               │
└────────────────────────────────┴───────────────────────────┴───────────────────────────────────────────┘

# model size: 

┌───────┬─────────────────────┬───────────┬─────────────┐
│ Stage │        Model        │  Params   │ Size (fp32) │
├───────┼─────────────────────┼───────────┼─────────────┤
│ A     │ TransitionModel (β) │ 2,587,024 │ 10.3 MB     │
├───────┼─────────────────────┼───────────┼─────────────┤
│ B     │ CarPlanner (IL)     │ 4,888,627 │ 19.6 MB     │
├───────┼─────────────────────┼───────────┼─────────────┤
│ C     │ CarPlanner (RL)     │ 4,888,627 │ 19.6 MB     │
└───────┴─────────────────────┴───────────┴─────────────┘


┌──────────────────┬───────────────────────────────────────────────────────┬───────────────────────────────────────────────────────┐   
│                  │             Option 3 (derived from cache)             │            Option 2 (real nuPlan polygons)            │
├──────────────────┼───────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤   
│ Geometry         │ Lane centerlines repurposed as "polygons" — these are │ Real closed polygons from GeoPackage — crosswalks,    │
│                  │  open polylines, not closed shapes                    │ stop lines, intersections                             │
├──────────────────┼───────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Categories       │ Only cat0=LANE (all derived from lanes) — no          │ 3 distinct categories: crosswalks (17%),              │
│                  │ diversity                                             │ stop_polygons (33%), intersections (50%)              │   
├──────────────────┼───────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Spatial          │ x: [-75, 133]m, y: [-117, 143]m — spread across large │ x: [-21, 42]m, y: [-4, 25]m — tighter, more local     │   
│ distribution     │  area (lane geometry)                                 │ (polygon geometry near intersections)                 │
├──────────────────┼───────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Valid per sample │ 3.1 avg (limited by how many lanes have valid data)   │ 6.0 avg (consistent — polygons cluster near           │
│                  │                                                       │ intersections)                                        │
├──────────────────┼───────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Semantic meaning │ None — lanes treated as polygons don't represent      │ Correct — represents what the paper describes:        │
│                  │ crosswalks/stop lines                                 │ intersections, crosswalks, stop lines                 │   
├──────────────────┼───────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ Speed limit      │ Non-zero (1.18 std) — inherited from lanes            │ Always 0 — polygons don't have speed limits (correct) │   
└──────────────────┴───────────────────────────────────────────────────────┴───────────────────────────────────────────────────────┘

Bottom line: Option 3 only validates that the architecture works on real tensor shapes. Option 2 validates the architecture on actual
polygon geometry with correct semantics and category diversity. For integration into the training pipeline, Option 2 is the right
choice.


## improvement plan

The plan at `paper/test_paper_encoders_plan.md` is ready but awaiting approval to implement. Before continuing there, here's the prioritized speedup analysis for the currently-running Stage B.

# Stage B speedup — ranked by ROI

β is already pre-computed and cached on CPU — that path is done. The remaining cost is the **autoregressive policy loop in `model.py` (~lines 667–794), which runs T_FUTURE=8 times per batch**. Fix the loop, not the data path.

## Tier 1 — biggest wins, low risk

**1. AMP mixed precision (bf16)** — expect 1.7–2.2× end-to-end
```python
scaler = None  # bf16 doesn't need a scaler
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    mode_logits, side_traj, pred_traj = model.forward_train(...)
    L_total, loss_dict = compute_il_loss(...)
L_total.backward()
optimizer.step()
```
bf16 is safer than fp16 on your 40-series GPU — no loss scaling, no NaN risk from the L1 losses. Applies to the whole forward pass including all 8 IVM steps. **Do this first.**

**2. Vectorize the H=10 frame-transform Python loop (line 688)** — expect 1.3–1.8×
Currently a Python `for` loop running 8×batch_size times per step. Replace with a single batched matmul: stack the 10 history frames into `(B, N, H, 3)`, build a single `(B, 2, 2)` rotation matrix per step, and apply it with `einsum`. Zero semantic change.

**3. Cache LaneEncoder output once per sample** — expect 1.2–1.4×
The lane *coordinates* change each step (ego frame moves), but you can instead:
- Encode lanes **once** in the global/initial frame → `(B, N_LANES, D_LANE)`
- At each step, apply a learned or analytic pose delta to the encoded features, OR add the pose delta as a conditioning token to IVM

The cheaper paper-faithful version: encode lanes once in the step-0 ego frame, then for each step pass the ego-delta (Δx, Δy, Δθ) as an extra key to IVM. This preserves the autoregressive structure without rerunning LaneEncoder 8 times.

If that feels too invasive, the safer variant: cache `LaneEncoder(map_lanes)` for step 0, and only re-transform+re-encode lanes that fall inside a tight radius at each step — typically only a few change.

## Tier 2 — material gains, moderate effort

**4. Increase batch size to 96 or 128** — expect 1.2–1.5×
With β now on CPU and the model at 6.5M params, your GPU memory headroom should allow larger batches. Bigger batches amortize the per-step Python overhead of the autoregressive loop. Watch `nvidia-smi` — target ~85% memory util.

**5. Pin β transfers**
```python
beta_seq_cpu = beta_seq_cpu.pin_memory()
# then:
agents_seq_batch = beta_seq_cpu[idx.cpu()].to(device, non_blocking=True)
```
Small but free. Overlaps H2D transfer with compute.

**6. Route re-filtering (line ~770)** — expect 1.05–1.15×
Precompute the per-sample route lane mask **once** per batch (route doesn't change across the 8 steps). Index into it per step instead of re-filtering.

## Tier 3 — only if still not fast enough

**7. `torch.compile(model, mode='reduce-overhead')`** — expect 1.1–1.3× on top of the above
Worth trying AFTER the structural fixes above. The autoregressive loop is dynamic-shape-friendly since T_FUTURE is fixed. Compile the `AutoregressivePolicy.forward` specifically; compiling the whole `CarPlanner` is more fragile. Expect 1–2 min compile time per rank.

**8. Reduce logging/checkpoint frequency**
You're already checkpointing every epoch. Fine. But the `% (n_batches_total // 5)` print is cheap — no change needed.

## What NOT to do

- **Don't drop to GT `agents_seq`** — you already rejected this; it breaks paper faithfulness and loses the point of Stage A.
- **Don't skip the IVM 3-layer stack** — that's a paper hyperparameter; reverting to 1 layer would be the single biggest speedup (~2.5×) but invalidates the run.
- **Don't use fp16** — the L1 losses + long autoregressive chain are fp16-hostile; bf16 is the right call.

## Recommended order of operations

1. **Stop current run**, add bf16 autocast (Tier 1 #1) → restart. Measure new epoch time.
2. If still slow: vectorize frame transforms (Tier 1 #2). Low-risk, localized edit in `model.py` around line 688.
3. If still slow: bump batch size (Tier 2 #4) + pin β (#5). Zero risk.
4. If still not enough: tackle LaneEncoder caching (Tier 1 #3) — this is the biggest structural win but also the most code.
5. Last resort: `torch.compile`.

**Expected cumulative speedup from steps 1+2+3+5: ~3–4×**, which should bring a Stage B epoch from its current pace down to something tolerable without touching the IVM/autoregressive structure at all.

Want me to implement the bf16 autocast + vectorized frame transform as a first pass? Those are ~30 lines of change total and don't touch model semantics.

# claude
claude --resume 4fbb0ac8-946b-428b-9ad8-38316cbe1c55  
## claude-glm
claude-glm --resume 602e77c9-4ad3-42cf-865f-f2aeb75d6c9f                                                                                                     

### Stage B training restart
```
  nohup python train_stage_b.py --split train_boston \                                                                        
    --batch_size 512 --pin_gpu \                                                                                              
    --resume checkpoints/best.pt \                                                                                            
    >> checkpoints/stage_b_boston.log 2>&1 &    
```
Yes exactly. Here's the concept:

**Ground truth:** nuPlan recorded a real human driver completing a scenario — that's the GT trajectory (a sequence of (x, y, yaw) waypoints over ~8 seconds future).

**What the model does:** Given the current scene (agents, map lanes, ego history), it generates **N_MODES trajectories** — one per driving mode (e.g. lane-keep, left turn, right turn, stop, etc.). Each is a candidate "plan" for what the ego should do next.

**Best-of-K:** Instead of committing to one trajectory, we ask — *of all the modes the model generated, how close is the best one to what the human actually did?* It picks the mode with minimum L2 distance to GT and reports that distance.

```
GT:    ──────────────►  (what human did)
Mode 1: ──────────────►  (lane keep)   ← closest → L2 = 0.8m  ✓ picked
Mode 2: ────────────/    (left turn)              L2 = 3.2m
Mode 3: ────────────\    (right turn)             L2 = 4.1m
```

**Why best-of-K and not just the selected mode?**
- It measures the model's *coverage* — can it at least generate a trajectory close to GT, regardless of which mode it picks?
- Mode accuracy separately measures whether the selector correctly identifies *which* mode matches GT
- Together they tell you: does the model generate good trajectories (L2), and does it know which one to use (accuracy)?

**Limitation:** This is open-loop — the model predicts once and we compare. It doesn't simulate what happens if the car actually follows the predicted path (that's closed-loop eval). A trajectory can look close to GT in L2 but still be unsafe in practice.


1. Stage A checkpoint (stage_a_best.pt) — the transition model β, which predicts where other agents will be in the future. This is frozen — Stage C never      
updates it, just uses it to simulate the scene.                                                                                                              
2. Stage B checkpoint (best.pt) — the policy π and mode selector, used as the warm start. Stage C inherits all of Stage B's learned IL behavior and fine-tunes 
it with RL.                                                                                                                                                    

---                                                                                                                                                            
What happens each training step:                          
                                                                                                                                                                
Frozen β (Stage A)
     ↓                                                                                                                                                        
agent_futures = β(scene)          ← "where will other cars go?"
     ↓                                                                                                                                                        
policy_old rollout → traj_old     ← "what would the current policy do?"                                                                                      
     ↓                                                                                                                                                        
compute_rewards(traj_old)         ← "was it good? collision? on-road?"                                                                                       
     ↓                                                                                                                                                        
GAE advantages                    ← "which timesteps were better than expected?"
     ↓                                                                                                                                                        
policy_new forward (with grad)    ← "can we do better than policy_old?"                                                                                      
     ↓                                                                                                                                                        
PPO update                        ← nudge policy toward higher-reward trajectories                                                                           
                                                                                                                                                           
---                                                                                                                                                            
In plain terms:                                                                                                                                              
- Stage A taught the model how the world works (agent motion)                                                                                                  
- Stage B taught the model what humans do (imitation)                                                                                                        
- Stage C asks "can we improve on what humans do?" by rewarding safety, staying on-road, and smoothness — using RL to fine-tune without forgetting Stage B's   
knowledge                                                                                                                                                      
                                                                                                                                                                
The L_gen + L_selector losses in Stage C act as a leash to prevent RL from wandering too far from Stage B's behavior.                                          
                                                                                                                        
=== Evaluation Results (mini) ===
  Best-of-60 L2 displacement:  0.4756 m
  Mode prediction accuracy:             5.94%
  Samples evaluated:                    12567         


# Stage C architecture
  ┌─────────────────────────────────────────────────────────────────┐
  │                        STAGE C INPUTS                           │
  │  agents_now (B,N,10)  │  map_lanes (B,20,10,27)  │  gt_traj     │
  └──────────┬────────────────────────┬───────────────────┬─────────┘
             │                        │                   │
             ▼                        │                   │
  ┌──────────────────────┐            │                   │
  │   TRANSITION MODEL β │  (FROZEN)  │                   │
  │   Stage A weights    │            │                   │
  │   transformer ×2     │            │                   │
  └──────────┬───────────┘            │                   │
             │ agent_futures          │                   │
             │ (B, T, N, Da)          │                   │
             ▼                        ▼                   │
  ┌──────────────────────────────────────────────┐        │
  │              S0 ENCODER                      │        │
  │   MLP → s0_per_agent (B,N,256)               │        │
  │         s0_global    (B,256)                 │        │
  └──────────────────────┬───────────────────────┘        │
                         │                                │
             ┌───────────┴───────────┐                    │
             ▼                       ▼                    │
  ┌─────────────────┐   ┌────────────────────────────┐    │
  │  MODE SELECTOR  │   │       POLICY π             │    │
  │  map_encoder    │   │  lane_encoder (PointNet)   │    │
  │  mode_transform │   │  decomposed_mode_embed     │    │
  │  mode_ffn       │   │  IVM cross-attn ×2         │    │
  │       ↓         │   │       ↓                    │    │
  │  logits (B,60)  │   │  action_mean_head → traj   │    │
  │  side_traj      │   │  action_log_std  → σ       │    │
  └────────┬────────┘   │  value_head      → V(s)    │    │
           │            └────────┬─────────┬─────────┘    │
           │                     │         │              │
           │              traj_old       V(s)             │
           │                     │         │              │
           │                     ▼         │              │
           │            ┌────────────────┐ │              │
           │            │ REWARD ENGINE  │◄───────────────┘
           │            │ R_displacement │  gt_traj
           │            │ R_collision    │
           │            │ R_drivable     │
           │            │ R_comfort      │
           │            └───────┬────────┘
           │                    │ rewards (B,T)
           │                    ▼
           │            ┌───────────────┐
           │            │  GAE (γ=0.1,  │
           │            │   λ=0.9)      │
           │            │  advantages   │
           │            │  returns      │
           │            └───────┬───────┘
           │                    │
           ▼                    ▼
  ┌─────────────────────────────────────────────────────┐
  │                    PPO LOSS                         │
  │                                                     │
  │  L_total = 100·L_policy  (PPO clip ε=0.2)           │
  │           +  3·L_value   (MSE vs returns)           │
  │           - 0.001·L_entropy (exploration bonus)     │
  │           +  1·L_gen     (L1 vs gt_traj)            │
  │           +  1·L_selector (CE + side task)          │
  └─────────────────────────────────────────────────────┘
           │
           ▼  AdamW lr=1e-4
  ┌─────────────────────┐
  │  UPDATE every batch │  policy_old refreshed every 10 batches
  │  (policy + selector)│  transition_model stays FROZEN
  └─────────────────────┘

# Eval and training models to be used: 
                                                       
1. Training / open-loop eval venv:                                                                                                     
/media/skr/storage/autoresearch/CarPlanner_Implementation/paper/.venv
- Has: torch, our CarPlanner model code                                                                                                
- Used for: train_stage_a/b/c.py, evaluate.py, preextract_features.py
- Does NOT have nuPlan simulation deps (pytorch_lightning, tensorboard, etc.)                                                          
                                                                                                                                       
2. nuPlan closed-loop eval venv:                                                                                                       
/media/skr/storage/autoresearch/CarPlanner_Implementation/paper/dataset/nuplan-devkit/nuplan_venv                                      
- Has: full nuPlan devkit deps (pytorch_lightning 1.3.8, tensorboard, ray, shapely, etc.)                                              
- Used for: scripts/eval_nuplan.py (CLS-NR, CLS-R metrics)                                                                             
- Activate via: source paper/dataset/nuplan-devkit/activate_nuplan_env.sh   