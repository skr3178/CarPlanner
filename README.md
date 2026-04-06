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

**Implementation:** `TransitionModel` exists (`model.py:385`) but no training loop calls it. `train.py` only implements Stage B. β is permanently random and never frozen because it was never trained.

**Gap:** Stage A entirely missing (known, in-scope to fix later).

---

#### Step 2, lines 13–14 — Agent simulation via β during IL

**Paper (line 14):** s^{1:N}_{1:T} ← β(s_0) — even in IL, the transition model β generates the agent futures used by the policy.

**Implementation:** `train.py:108` passes `agents_seq = batch['agents_seq']` — GT future agent states loaded from the DB, not β output. `forward_train` never calls `self.transition_model`.

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

**Implementation (`train.py:117`, `train.py:44`):**
```python
pred_traj = model.forward_train(agents_seq, ..., mode_c=c, gt_ego=gt_traj)
L_gen = (pred_traj - gt_traj).abs().sum(dim=-1).mean()
```
Match. ✓

---

#### Step 2, lines 33–35 — Overall loss ✓

**Paper:** L = L_selector + L_generator, update f_selector and π.

**Implementation (`train.py:46`):** `L_total = L_CE + L_side + L_gen` where `L_selector = L_CE + L_side`. ✓

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



Tables                                                                                                                                          
                                                                                                                                       
┌──────────────────────┬───────────┬──────────────────────────────────────────────────────┐                                                     
│        Table         │   Rows    │                       Purpose                        │                                                     
├──────────────────────┼───────────┼──────────────────────────────────────────────────────┤                                                     
│ log                  │ 1         │ Scenario metadata (vehicle, date, location, map)     │                                                 
├──────────────────────┼───────────┼──────────────────────────────────────────────────────┤                                                     
│ ego_pose             │ 51,152    │ AV pose at every IMU sample (~100Hz)                 │                                                     
├──────────────────────┼───────────┼──────────────────────────────────────────────────────┤                                                     
│ lidar_pc             │ 10,200    │ LiDAR sweep timestamps + file refs (10Hz)            │                                                     
├──────────────────────┼───────────┼──────────────────────────────────────────────────────┤                                                     
│ lidar_box            │ 1,280,922 │ Detected bounding boxes per LiDAR sweep              │                                                     
├──────────────────────┼───────────┼──────────────────────────────────────────────────────┤                                                     
│ track                │ 6,460     │ Unique tracked objects (vehicle/ped/bike dimensions) │                                                     
├──────────────────────┼───────────┼──────────────────────────────────────────────────────┤                                                     
│ scenario_tag         │ 13,812    │ Labels marking which frames belong to scenario types │                                                     
├──────────────────────┼───────────┼──────────────────────────────────────────────────────┤                                                     
│ traffic_light_status │ 96,994    │ Red/green per lane connector per frame               │                                                     
├──────────────────────┼───────────┼──────────────────────────────────────────────────────┤                                                     
│ scene                │ 26        │ Scene-level goal poses + roadblock IDs               │                                                     
├──────────────────────┼───────────┼──────────────────────────────────────────────────────┤                                                     
│ category             │ 7         │ Object class definitions                             │                                                     
└──────────────────────┴───────────┴──────────────────────────────────────────────────────┘                                                     
                                                                                                                                       
---                                                                                                                                         
ego_pose columns and ranges                                                                                                                     
                                                                                                                                       
┌────────────────────┬──────────────────────────────────────────────┬────────────────┐                                                          
│       Column       │                    Range                     │      Unit      │                                                          
├────────────────────┼──────────────────────────────────────────────┼────────────────┤                                                          
│ x, y               │ [664,425 → 664,656], [3,996,370 → 3,999,268] │ m (global UTM) │                                                      
├────────────────────┼──────────────────────────────────────────────┼────────────────┤                                                          
│ vx, vy             │ [-0.02 → 14.05], [-0.32 → 0.04]              │ m/s            │                                                          
├────────────────────┼──────────────────────────────────────────────┼────────────────┤                                                          
│ acceleration_x     │ [-1.77 → 3.07]                               │ m/s²           │                                                          
├────────────────────┼──────────────────────────────────────────────┼────────────────┤                                                          
│ qw/qx/qy/qz        │ quaternion                                   │ —              │                                                          
├────────────────────┼──────────────────────────────────────────────┼────────────────┤                                                          
│ angular_rate_x/y/z │ —                                            │ rad/s          │                                                          
└────────────────────┴──────────────────────────────────────────────┴────────────────┘                                                          
                                                                                                                                       
---                                                                                                                                         
lidar_box columns and ranges (detected agents)                                                                                                  
                                                                                                                                       
┌────────────┬──────────────────────────────────────────────┬────────────────┐                                                                  
│   Column   │                    Range                     │      Unit      │                                                                  
├────────────┼──────────────────────────────────────────────┼────────────────┤                                                                  
│ x, y       │ [664,348 → 664,711], [3,996,297 → 3,999,330] │ m (global UTM) │                                                              
├────────────┼──────────────────────────────────────────────┼────────────────┤                                                                  
│ vx, vy     │ [-17.4 → 15.8], [-21.2 → 20.6]               │ m/s            │                                                                  
├────────────┼──────────────────────────────────────────────┼────────────────┤                                                                  
│ width      │ [0.19 → 4.48]                                │ m              │                                                                  
├────────────┼──────────────────────────────────────────────┼────────────────┤                                                                  
│ length     │ [0.17 → 16.17]                               │ m              │                                                                  
├────────────┼──────────────────────────────────────────────┼────────────────┤                                                                  
│ yaw        │ —                                            │ rad            │                                                                  
├────────────┼──────────────────────────────────────────────┼────────────────┤                                                              
│ confidence │ [0.32 → 1.00]                                │ —              │                                                                  
└────────────┴──────────────────────────────────────────────┴────────────────┘                                                                  
                                                                                                                                       
---                                                                                                                                             
category types                                                                                                                                  
                                                                                                                                  
┌────────────────┬──────────────────────────────┐
│      Name      │         Description          │
├────────────────┼──────────────────────────────┤
│ vehicle        │ Cars, trucks, trailers       │
├────────────────┼──────────────────────────────┤                                                                                               
│ bicycle        │ Bikes, motorcycles           │                                                                                               
├────────────────┼──────────────────────────────┤                                                                                               
│ pedestrian     │ All pedestrians              │                                                                                               
├────────────────┼──────────────────────────────┤                                                                                               
│ traffic_cone   │ Temporary cones              │                                                                                               
├────────────────┼──────────────────────────────┤                                                                                               
│ barrier        │ Permanent/temporary barriers │                                                                                               
├────────────────┼──────────────────────────────┤                                                                                               
│ czone_sign     │ Construction zone signs      │                                                                                               
├────────────────┼──────────────────────────────┤                                                                                           
│ generic_object │ Animals, debris, poles       │
└────────────────┴──────────────────────────────┘

---
scenario_tag types (frequency)                                                                                                                  
                                                                                                                                       
┌───────────────────────────────┬───────┐                                                                                                       
│             Type              │ Count │                                                                                                       
├───────────────────────────────┼───────┤                                                                                                       
│ stationary                    │ 3,020 │                                                                                                   
├───────────────────────────────┼───────┤
│ stationary_in_traffic         │ 2,034 │
├───────────────────────────────┼───────┤
│ traversing_intersection       │ 1,667 │                                                                                                       
├───────────────────────────────┼───────┤                                                                                                       
│ on_traffic_light_intersection │ 1,291 │                                                                                                       
├───────────────────────────────┼───────┤                                                                                                       
│ high_magnitude_speed          │ 1,122 │                                                                                                       
├───────────────────────────────┼───────┤                                                                                                   
│ near_pedestrian_on_crosswalk  │ 250   │                                                                                                       
├───────────────────────────────┼───────┤                                                                                                       
│ traversing_crosswalk          │ 65    │                                                                                                       
├───────────────────────────────┼───────┤                                                                                                       
│ ...                           │ ...   │                                                                                                       
└───────────────────────────────┴───────┘                                                                                                   

---
traffic_light_status                                                                                                                            
                                                                                                                                       
- green: 41,241 entries                                                                                                                         
- red: 55,753 entries  