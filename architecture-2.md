# CarPlanner — Paper Architecture Reference

Authoritative specification from  
**"CarPlanner: Consistent Auto-regressive Trajectory Planning for Large-scale Reinforcement Learning in Autonomous Driving"**

This file is the single reference for what the paper requires. Use it for implementation comparison instead of re-reading the paper.  
Implementation deviations from this spec are noted in `§9 Known Implementation Gaps`.

---

## 1. System Overview

CarPlanner is a three-module pipeline applied once per planning step given initial state **s₀**:

```
s₀  ──►  Mode Selector  ──────────────────────────────────────►  σ  (N_mode scores)
  │                                                                       │
  └──►  Transition Model β  ──►  s^{1:N}_{1:T}  ──►  Trajectory Generator π (×N_mode)
                                                         ↑ c fixed per rollout        │
                                                                          Rule Selector
                                                                               │
                                                                     best trajectory (output)
```

**Parallel worlds:** s₀ is replicated N_mode times. Each copy is associated with one mode.  
The policy rollout executes inside all N_mode parallel worlds simultaneously (batched, not sequential).

| Module | Role | Trained in |
|--------|------|-----------|
| Transition Model β | Predicts future positions of all other agents (non-reactive to ego) | Stage A — then frozen |
| Mode Selector f_selector | Scores N_mode candidate driving modes given s₀ | Stage B (IL) + Stage C (RL, selector loss only) |
| Trajectory Generator π | Auto-regressively decodes ego waypoints conditioned on a fixed mode c | Stage B (IL) + Stage C (RL) |
| Rule Selector | Inference-only: re-scores N_mode trajectories with safety/comfort rules | No parameters |

**Key invariant:** mode c is **fixed** across all T autoregressive steps (consistent AR).  
The mode query is *updated* by cross-attention at each step, but the mode index never changes during a rollout.

---

## 2. State and Feature Dimensions

### 2.1 Agent State — Dₐ = 10

Each agent at one time step:

| Index | Field | Notes |
|-------|-------|-------|
| 0 | x | ego-relative |
| 1 | y | ego-relative |
| 2 | heading | radians, ego-relative |
| 3 | vx | velocity x |
| 4 | vy | velocity y |
| 5 | box width | metres |
| 6 | box length | metres |
| 7 | box height | metres |
| 8 | time step | normalised to −H…0 inside IVM |
| 9 | category | encoded (vehicle/pedestrian/cyclist/…) |

Agent history tensor shape per scene: **(N × H × Dₐ)**

### 2.2 Map Point — D_m = 9

Each point within a lane polyline or polygon:

| Index | Field | Notes |
|-------|-------|-------|
| 0 | x | ego-relative |
| 1 | y | ego-relative |
| 2 | sin(heading) | **Inferred** — paper states "heading" as a single field. sin/cos decomposition avoids angle discontinuity but is not explicitly stated |
| 3 | cos(heading) | **Inferred** — see above |
| 4 | speed limit | m/s; 0.0 if unknown |
| 5–8 | category one-hot | lane / connector / intersection / other |

### 2.3 Map Elements

**Polylines** (lane centers + boundaries):
- Each polyline contains **N_p points**, each with **3D_m = 27 features**: center point features + left boundary features + right boundary features **concatenated along the feature dimension** per point.
- Total per-polyline tensor: **N_p × 3D_m** (paper: "Nm,1 × Np × 3Dm").
- Count: N_{m,1} polylines.
- Encoded by a **PointNet with in_dim=27** over the N_p points → single vector of dim D per polyline.
- Output: N_{m,1} × D.
- If a boundary is unavailable (e.g. connector lanes), duplicate centerline features for that slot.

**Polygons** (intersections, crosswalks, stop lines):
- Each polygon: N_p points × D_m.
- Count: N_{m,2} polygons.
- Encoded by a **separate PointNet** → N_{m,2} × D.

**Concatenate** both outputs → **N_m × D** total map features, where N_m = N_{m,1} + N_{m,2}.

---

## 3. Module Specifications

### 3.1 Transition Model β

**Purpose:** Given s₀, predict all agent futures s^{1:N}_{1:T}. Non-reactive to ego (ignores ego action).

**Loss (Eq. 5):**
```
L_tm = (1/T) Σ_t Σ_n ‖ s_t^n − s_t^{n,gt} ‖₁
```
Masked: sum over Da dimensions, mean over valid (B, T, N) entries.

**Architecture:**
1. **Agent encoder** — PointNet over agent history **(N × H × Dₐ)** → per-agent feats N × D + global feat D. Each agent's H historical poses are encoded together — not just t=0. The paper states: *"each agent maintains poses for the past H time steps"* with tensor shape N × H × Dₐ.
2. **Map encoder** — PointNet(polylines) + PointNet(polygons) → concatenate → N_m × D
3. **Self-attention Transformer encoder** — fuses agent and map features:
   - Concatenate: (N + N_m) × D
   - Standard Transformer encoder (self-attention + FFN) → same shape
4. **Per-agent decoder** — MLP per agent: [per-agent feat; global feat] → T × Dₐ future states

**Output shape:** (B, T, N, Dₐ)  
Frozen after Stage A. Provides agent_seq for Stages B and C.

---

### 3.2 Mode Representation

Modes are **longitudinal–lateral decomposed**. They are constructed per-scene at each forward pass, not stored as fixed learned embeddings.

#### Longitudinal modes — N_lon = 12

Each longitudinal mode represents a **target average speed** for the trajectory. Mode j encodes the intent "drive at j/N_lon of maximum speed on average":

```
c_lon,j = j / N_lon   for j = 0…N_lon−1
```
Each scalar is **repeated across D dimensions** → row tensor of shape (D,).  
Stack all N_lon rows → shape: **N_lon × D** (no learned parameters).

> **Mode meaning vs assignment rule — these are two different things:**
> - The mode scalar j/N_lon encodes the *intended average speed* of the trajectory (what the mode represents to the policy at rollout time).
> - The assignment rule for c_lon* during training uses the **endpoint speed** of the GT trajectory to find which bucket j the human expert's final speed falls in. Endpoint speed is used as a proxy for the trajectory's average speed intent. These diverge when the ego accelerates or decelerates significantly — a known approximation.

#### Lateral modes — N_lat ≤ 5

Identified by **graph search** from the ego's current lane in the HD map:
1. Enumerate up to N_lat candidate routes (lane sequences the ego could follow).
2. Each route: N_r sampled points × D_m = 9.
3. Apply **PointNet** over the N_r points per route → one vector of dim D per route.
4. Stack → shape: **N_lat × D** (learned PointNet).

#### Combined mode tensor

```
cat([lon_modes, lat_modes], dim=-1)  →  N_lat × N_lon × 2D
Linear(2D → D)                       →  N_lat × N_lon × D   (= N_mode × D)
```

Total modes: **N_mode = N_lat × N_lon = 5 × 12 = 60**

#### Mode assignment (training only)

Given ground-truth trajectory s^{0,gt}_{1:T}:
- **c_lon**: bin index of the **endpoint speed** of the GT trajectory: j such that endpoint_speed ∈ [j/N_lon, (j+1)/N_lon) × MAX_SPEED. Note: this is the speed at t=T (the final waypoint), **not** the mean speed across all timesteps. Mean and endpoint speed diverge whenever the ego accelerates or decelerates significantly over the horizon.
- **c_lat**: index of the candidate route whose **endpoint is closest to the GT endpoint** (route proximity; not GT lateral y-offset)
- **c*** = (c_lat × N_lon + c_lon) → integer in [0, N_mode)

---

### 3.3 Mode Selector f_selector

**Inputs:**
- Encoded s₀: global agent feature (B, D) + map features (B, N_m, D)
- Full mode tensor c: **(B, N_mode, D)** — constructed per scene (§3.2)

**Architecture:**

1. Encode s₀:
   - Agent PointNet → global_feat **(B, D)**, per-agent feats **(B, N, D)**
   - Map PointNet(polylines) + PointNet(polygons) → map_feat **(B, N_m, D)**

2. **Query-based Transformer decoder:**
   - Query: mode features **(B, N_mode, D)** — all 60 modes simultaneously
   - Keys/Values: agent + map features concatenated **(B, N + N_m, D)**
   - Standard cross-attention + FFN + LayerNorm
   - Output: updated mode features **(B, N_mode, D)**

3. **MLP** applied per mode → per-mode scalar **(B, N_mode)**

4. **Softmax** → mode probabilities σ **(B, N_mode)**

**Side task head** (conditioned on selected mode c*):
- Concatenate updated mode feature for c* with global_feat → MLP → predicted ego trajectory s̄⁰_{1:T} **(B, T, 3)**
- Used only during training (Eq. 7)
- > **Note:** This head architecture (concatenate + MLP) is a reasonable reconstruction. The paper does not describe the side task head's internal structure explicitly — only that it predicts the ego trajectory as a side task of the selector. Treat this as inferred.

**Losses:**

Eq. 6 — Mode classification:
```
L_CE = CrossEntropy(logits, c*)
```

Eq. 7 — Side task:
```
L_SideTask = (1/T) Σ_t ‖ s̄_t^0 − s_t^{0,gt} ‖₁
```

**Mode dropout (training only, p=0.1):**
- During training, randomly replace the lateral route input with a zero/masked tensor with probability p=0.1
- Purpose: prevents the selector (and downstream policy) from over-relying on route information. When no routes are available at inference (e.g. complex intersections), the model must still produce stable outputs
- Applied in both Stage B (IL) and Stage C (RL)
- Without mode dropout, mode collapse occurs — the model learns to always predict the same mode regardless of scene context

Algorithm 1 line 20: `σ, s̄ ← f_selector(s₀, c)` — the mode tensor c is an explicit input, not just an index.

---

### 3.4 Trajectory Generator π (Autoregressive Policy)

Generates one ego trajectory conditioned on a **fixed** mode c.  
At inference: called N_mode=60 times in parallel (batched over mode dimension).

#### 3.4.1 Invariant-View Module (IVM) — Algorithm 3

Applied at **every timestep t**. Makes the policy time-agnostic by re-centering the scene view.

**Step 1 — K-Nearest Neighbor selection:**
- Agents: K = N / 2 (half of total agent count, selected by distance at current step t)
- Map elements: K_m = N_m / 2 (half of total map features, selected by distance)

**Step 2 — Route segment filtering** (per lateral route, per step t):
- Find the point on the route polyline closest to ego's current pose at step t
- **Discard** that route if the closest point is the route's starting point (ego has already passed it)
- Retain only **K_r = N_r / 4** points **ahead** of the closest point per route
- This ensures only forward-looking route context is used at each step

**Step 3 — Coordinate transform to ego frame at step t:**
- Translate and rotate all agent (x, y, heading), map, and route point positions into the ego vehicle's frame at **current** time step t
- Applied at every step — features are **not** static across timesteps

**Step 4 — Time normalisation:**
- Maintain a history buffer of H most recent frames per agent
- Replace absolute time indices t−H…t with normalised indices **−H…0**
- PointNet encodes all H frames per agent → feature per agent-history-step → shape: (K × H) × D

**Query-based Transformer decoder** (same backbone architecture as Mode Selector):
- Query: mode_query **(B, 1, D)** — mode embedding, carried and updated across steps
- Keys/Values: selected agent feats + map feats **(B, K×H + K_m, D)**
- Standard cross-attention + FFN + LayerNorm
- Output: updated mode_query **(B, 1, D)**

Note: different modes yield different ego poses at step t → different IVM-transformed views → **map and agent features cannot be shared across modes**. Each mode is processed independently, but all N_mode modes run in parallel via the batch dimension.

#### 3.4.2 Policy and Value Heads

Both heads operate on the **updated mode_query (B, 1, D)** at each step:

**Policy head (Gaussian):**
- MLP → (μ, log σ) per action dimension **(B, 3)** — action = (Δx, Δy, Δyaw)
- Training: sample a_t ~ N(μ, σ)
- Inference: use mean μ only (deterministic, no sampling)

**Value head:**
- Separate MLP → scalar V(s_t) **(B, 1)**
- Used for GAE advantage estimation in Stage C

#### 3.4.3 Generator Loss (Eq. 11 — IL and RL)

```
L_generator = (1/T) Σ_t ‖ s_t^0 − s_t^{0,gt} ‖₁
```

Present in both IL (Stage B) and RL (Stage C) — acts as an imitation anchor in RL.

---

### 3.5 Rule Selector (Inference Only)

Scores all N_mode=60 candidate trajectories on four criteria:

| Criterion | Formula | Direction |
|-----------|---------|-----------|
| Collision | fraction of timesteps within collision_radius of any agent | lower = better |
| Drivable area | fraction of timesteps farther than max_dist from any lane centerline | lower = better |
| Comfort | mean jerk (3rd finite diff of position) | lower = better |
| Progress | final forward displacement (x in ego frame) | higher = better |

Combined score (Algorithm 2, step 5):
```
score_i = w_mode × σ_i + w_comfort × comfort_i + w_progress × progress_i
        − w_collision × collision_i − w_drivable × drivable_i
```

Select trajectory with **argmax** score. Exact scalar weights not stated in paper.

---

## 4. Training Stages

### Stage A — Transition Model

**Trains:** β only  
**Loss:** Eq. 5 (masked L1 on agent trajectories)  
**Outcome:** β frozen, checkpoint saved for Stages B and C

### Stage B — Imitation Learning (IL)

**Trains:** f_selector + π jointly  
**β:** frozen, called at start of each batch to generate agent_seq  

**Loss:**
```
L_IL = L_CE + L_SideTask + L_generator   (equal weights, all three)
```

**Mode c input to π:** positive mode c* (GT-assigned)  
**Ego frame for IVM:** teacher-forcing — use GT ego pose at step t for coordinate transforms  

**Best config (Table 4):** Mode Dropout ✓ · Selector Side Task ✓ · Ego-history Dropout ✓ · Backbone Sharing ✓ → CLS-NR 93.41

**Ablation notes:**
- **Mode dropout (p=0.1):** randomly replace c* with a random mode — prevents mode collapse, critical for both IL and RL
- **Selector side task:** training the side-task head substantially improves S-PR (progress metric)
- **Ego-history dropout:** randomly zero ego's historical poses — improves IL robustness but **hurts RL**
- **Backbone sharing:** share the s₀ encoder between f_selector and the per-step IVM encoder inside π — helps IL but **hurts RL**

### Stage C — RL Fine-tuning (PPO)

**Trains:** f_selector (selector loss only) + π (RL loss)  
**β:** frozen, used for world simulation  
**Best config (Table 4):** Mode Dropout ✓ · Selector Side Task ✓ · Ego-history Dropout ✗ · Backbone Sharing ✗ → CLS-NR 94.07

**Rollout collection (per scenario):**
1. Sample mode c* for each scenario
2. Replicate s₀ for N_mode parallel worlds
3. For t = 0…T−1: run π_old → sample action a_t → step world with β
4. Collect (s_{0:T−1}, a_{0:T−1}, log π_old(a_t), V(s_t), R_t)

**Rewards (Section 3.4):**
- Displacement error: negative L1 to GT trajectory (dense, per step)
- Collision penalty: binary, per step
- Drivable area penalty: binary, per step

**Advantage:** GAE with γ=0.1, λ=0.9

**RL total loss (Eq. 8–11):**
```
L_RL = λ_π · L_policy + λ_V · L_value − λ_H · H(π) + L_generator
```

> **Note:** The weight of L_generator in Stage C is not explicitly stated in the paper. The formulation above (weight=1, same as Stage B) is inferred from the Table 4 structure and the paper's statement that the imitation term acts as a "behavioural anchor during RL fine-tuning." Treat this as an assumption, not a stated fact.

**PPO policy loss (Eq. 8):**
```
r_t = π_new(a_t) / π_old(a_t)
L_policy = −(1/T) Σ_t min( r_t · A_t,  clip(r_t, 1−ε, 1+ε) · A_t )
```

**π_old update:** every I=8 steps (delayed policy update, Table 5)  
**PPO clip ε=0.2** (not stated in paper; standard value)

**Loss weights (Table 5):** λ_π=100, λ_V=3, λ_H=0.001

---

## 5. Inference Procedure — Algorithm 2

1. Encode s₀ → agent features, map features
2. Construct mode tensor (§3.2): lon scalars + lateral route PointNets → N_mode × D
3. Run Mode Selector → σ (N_mode scores)
4. Run β(s₀) → agent_seq **(T × N × Dₐ)**
5. For each of N_mode=60 modes **in parallel (batched)**:
   - Initialise mode_query from mode embedding for mode c
   - For t = 0…T−1:
     - Apply IVM: K-NN select, route filter (Step 2), ego-centric transform, time normalise
     - Cross-attend mode_query over selected agents + map
     - Policy head → **μ only** (no sampling at inference)
     - Decode waypoint, accumulate trajectory
6. Run Rule Selector → combined score per mode → argmax → select trajectory
7. Return selected ego trajectory

---

## 6. Hyperparameters

| Symbol | Value | Source |
|--------|-------|--------|
| N_lon | 12 | Paper |
| N_lat | 5 | Paper |
| N_mode | 60 | Paper (= N_lat × N_lon) |
| T | 8 | Paper |
| H | ~10–20 | **Not stated.** nuPlan standard = 20 frames (2s × 10Hz) |
| D | ~256 | **Not stated.** Inferred from Transformer practice |
| Dₐ | 10 | Paper |
| D_m | 9 | Paper |
| K (agents) | N/2 | Paper |
| K_m (map) | N_m/2 | Paper |
| K_r (route pts) | N_r/4 | Paper |
| N_r | — | **Not stated** |
| Optimizer | AdamW | Paper |
| LR | 1e-4 | Paper |
| LR schedule | ReduceLROnPlateau | Paper |
| Epochs | 50 | Paper |
| Batch size | 64/GPU | Paper (2× RTX 3090) |
| γ (discount) | 0.1 | Paper |
| λ (GAE) | 0.9 | Paper |
| ε (PPO clip) | 0.2 | **Not stated.** Standard value |
| λ_π | 100 | Paper Table 5 |
| λ_V | 3 | Paper Table 5 |
| λ_H | 0.001 | Paper Table 5 |
| I (π_old interval) | 8 | Paper Table 5 |
| Mode dropout p | 0.1 | Paper |

---

## 7. Known Ambiguities in the Paper

| Item | Ambiguity | Impact |
|------|-----------|--------|
| H (history length) | Not stated | Medium — tensor shapes change |
| D (feature dim) | Not stated | Low — tune freely |
| N_r (route points) | Not stated | Low — tune freely |
| N (max agents) | Not stated. nuPlan: typically 20–80 | Low — truncate/pad |
| PPO ε | Not stated | Low |
| Map encoder depth (β) | Type/depth of map encoder in TransitionModel unspecified | Medium |
| Rule selector weights | Qualitative description only; no scalar values given | Medium |
| Polyline 3× structure | Paper states 3×N_p (center + left + right). Boundary features may differ from centerline features | Medium |
| Graph search for routes | Algorithm for enumerating N_lat routes from HD map not specified | High — determines c_lat quality |

---

## 8. Key Design Decisions (From Ablations — Table 4)

1. **Consistent AR over vanilla AR** (core contribution): mode c fixed across all T steps. Vanilla AR re-samples c each step — significantly worse.

2. **Mode dropout is non-negotiable:** without it, mode collapse occurs. Use p=0.1 for both IL and RL.

3. **IL and RL best configs diverge on two components:**
   - Ego-history dropout: ON for IL, OFF for RL
   - Backbone sharing: ON for IL, OFF for RL
   These must be controlled by separate flags per training stage.

4. **Short training horizons disproportionately hurt:** T=1 during training → CLS-NR=75.79 even at test T=1. Use T=8 throughout all stages.

5. **Generator loss (Eq. 11) remains in Stage C:** the L1 imitation term is added alongside PPO losses (weight=1), providing a behavioural anchor during RL fine-tuning.

---

## 9. Known Implementation Gaps (vs. This Spec)

Gaps between the current codebase and this specification, in priority order:

| # | Component | Paper Spec | Current Code | Priority |
|---|-----------|-----------|--------------|----------|
| 0 | Agent state dimension D_a (§2.1) | D_a = 10: x, y, heading, vx, vy, box_w, box_l, box_h, time_step, category | cfg.D_AGENT likely = 4 (x, y, heading, v only) — missing bounding box dims, category, and separate vx/vy | **High** — affects every module that ingests agent features: β, IVM, and mode selector all receive impoverished agent representations. Bounding box dims are essential for collision reasoning; category distinguishes pedestrians from vehicles |
| 1 | Mode encoding (§3.2) | Decomposed lon scalar×D + lat PointNet(route) → 2D→linear D | `nn.Embedding(60, D)` flat lookup | High |
| 2 | Mode Selector decoder (§3.3) | Transformer decoder with all 60 modes as queries | MLP on global_feat only; no cross-attention | High |
| 3 | TransitionModel map encoder (§3.1) | PointNet(polylines) + PointNet(polygons) + self-attention Transformer | PointNet(agents) + MLP only; no map, no Transformer | High |
| 4 | IVM Step 2 — route filtering (§3.4.1) | Discard passed routes; retain K_r=N_r/4 forward points | Not implemented | Medium |
| 5 | IVM K_m — map K-NN (§3.4.1) | K_m = N_m/2 nearest map elements | All map features used unfiltered | Medium |
| 6 | Map structure (§2.3) | Polylines: N_p × 3Dm (center + left + right concatenated per point); Polygons: separate PointNet | Polyline boundaries now loaded and concatenated (D_POLYLINE_POINT=27). Polygon encoder still absent. | Low |
| 7 | Backbone sharing (§4 Stage B) | s₀ encoder shared between mode selector and IVM backbone in Stage B; disabled in Stage C | BACKBONE_SHARING=True flag exists but wiring is broken — s0_encoder output feeds mode selector but is **not** routed into IVM; policy always re-encodes independently regardless of flag. Stage B path produces incorrect behaviour silently. | **Medium** — broken config flag is worse than a missing feature; code appears to implement sharing but does not |
| 8 | c_lat assignment (§3.2) | Closest candidate route to GT endpoint (route proximity) | Route-proximity via `_route_lateral_offset` — correct approach, but graph-search routes not enumerated; uses encoded lane centroids instead | **Medium** — on curved roads ego-frame y mixes forward progress with lateral offset; a left turn looks like large positive y even if ego is following its lane. Proper graph-search routes required to correctly distinguish lane-keep from lane-change on bends |
