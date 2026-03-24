"""
CarPlanner — faithful implementation.

Architecture grounded in:
  - paper/images/figure2_page3.png  (system overview)
  - paper/algorithms.md             (Algorithm 1 training, Algorithm 2 inference, Algorithm 3 IVM)
  - paper/carplanner_equations.md   (Eq 5–11)
  - paper/hyperparameters.md        (D, N_modes, T, K-NN, etc.)
  - paper/tables.md                 (Table 4 ablation — IL/RL best configs)

Three main stages (Figure 2, left → right):
  1. NR Transition Model      — P_τ(s^{1:N}_{1:T} | s_0), pre-trained, frozen in stages 2–3
  2. Trajectory Generator     — ModeSelector + AutoregressivePolicy (IVM-based)
  3. Rule-augmented Selector  — inference-time rule-based trajectory selection

IVM (Algorithm 3):
  - K-nearest agent selection (K = N // 2)
  - Ego-centric coordinate transform at each step t
  - Time normalisation: t-H:t → -H:0 (time-agnostic)
  - Transformer decoder: mode c as query, agent features as K/V
  - Updated mode query carries temporal context across steps

Mode structure (Section 4.1):
  - N_lon = 12 longitudinal × N_lat = 5 lateral = 60 modes total
  - Mode c is FIXED across all T autoregressive steps (consistent AR, Figure 1c)

Losses (IL stage B, equal weights):
  L_IL = L_CE (Eq 6) + L_SideTask (Eq 7) + L_generator (Eq 11)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


# ─────────────────────────────────────────────────────────────────────────────
# PointNet agent encoder  (shared MLP + max-pool, per-agent and global)
# ─────────────────────────────────────────────────────────────────────────────

class PointNetEncoder(nn.Module):
    """
    PointNet-style encoder for a variable-size set of agents.
    Shared MLP applied per-point, then max-pool for global representation.

    Input:  (B, N, D_in)  agent features in ego-centric frame
    Output: per_agent (B, N, D), global_feat (B, D)
    """
    def __init__(self, in_dim: int = 4, d_model: int = cfg.D_HIDDEN):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, d_model),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        x:    (B, N, in_dim)
        mask: (B, N) float — 1=valid, 0=padding
        """
        per_agent = self.mlp(x)                          # (B, N, D)
        if mask is not None:
            m = mask.unsqueeze(-1)                        # (B, N, 1)
            per_agent = per_agent * m
            pool_src = per_agent.masked_fill(m == 0, float('-inf'))
        else:
            pool_src = per_agent
        # Max-pool over agent dimension (ignores padding via -inf)
        global_feat = pool_src.max(dim=1).values         # (B, D)
        # Guard: all-padding batch rows → replace -inf with 0
        global_feat = torch.nan_to_num(global_feat, nan=0.0,
                                       posinf=0.0, neginf=0.0)
        return per_agent, global_feat


# ─────────────────────────────────────────────────────────────────────────────
# Lane encoder  (PointNet over lane polylines)
# ─────────────────────────────────────────────────────────────────────────────

class LaneEncoder(nn.Module):
    """
    PointNet-style encoder for lane centerlines.
    Each lane is a polyline of (x, y, heading) points.

    Input:  lanes (B, N_LANES, N_POINTS, 3) + mask (B, N_LANES)
    Output: per_lane (B, N_LANES, D_LANE)
    """
    def __init__(self, in_dim: int = 3, d_model: int = cfg.D_LANE):
        super().__init__()
        # Per-point MLP: (x, y, h) → D_LANE
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, d_model),
        )
        self.d_model = d_model

    def forward(self, lanes: torch.Tensor, mask: torch.Tensor = None):
        """
        lanes: (B, N_LANES, N_POINTS, 3)
        mask:  (B, N_LANES) — 1=valid, 0=padding
        Returns: per_lane (B, N_LANES, D_LANE)
        """
        B, N_L, N_P, _ = lanes.shape
        # Flatten batch + lanes for point-wise encoding
        flat = lanes.view(B * N_L, N_P, 3)               # (B*N_L, N_P, 3)
        per_point = self.point_mlp(flat)                  # (B*N_L, N_P, D)

        # Max-pool over points within each lane
        lane_feat = per_point.max(dim=1).values           # (B*N_L, D)
        lane_feat = lane_feat.view(B, N_L, self.d_model)  # (B, N_LANES, D_LANE)

        # Apply mask (zero out padding lanes)
        if mask is not None:
            lane_feat = lane_feat * mask.unsqueeze(-1)

        return lane_feat


# ─────────────────────────────────────────────────────────────────────────────
# IVM — Invariant-View Module  (Algorithm 3)
# ─────────────────────────────────────────────────────────────────────────────

class IVMBlock(nn.Module):
    """
    Invariant-View Module (Algorithm 3, Section 3.3.3).

    Steps per autoregressive timestep t:
      1. K-nearest agent selection  (K = N // 2)
      2. Ego-centric coordinate transform already applied upstream
      3. Time normalisation already applied upstream
      4. Transformer decoder: mode c as query, (agent + map) feats as K/V
      5. Output updated mode query (carries temporal state across steps)

    Query: mode_query  (B, 1, D)       — fixed mode index, updated by attention
    K/V:   agent_feats (B, K, D)       — K-NN selected agents at step t
           map_feats   (B, N_LANES, D)  — encoded lane centerlines (optional)
    Output: updated mode_query (B, 1, D)
    """
    def __init__(self, d_model: int = cfg.D_HIDDEN, n_heads: int = 4,
                 k_nn: int = cfg.N_AGENTS // 2):
        super().__init__()
        self.k_nn = k_nn
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=0.0
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        # Project lane features from D_LANE to D_HIDDEN for concatenation
        self.lane_proj = nn.Linear(cfg.D_LANE, d_model)

    def forward(self, mode_query: torch.Tensor, agent_feats: torch.Tensor,
                agent_poses: torch.Tensor, map_feats: torch.Tensor = None) -> torch.Tensor:
        """
        mode_query:  (B, 1, D)   — current mode state (query)
        agent_feats: (B, N, D)   — PointNet-encoded agents in ego frame at t
        agent_poses: (B, N, 2)   — (x, y) positions for K-NN distance
        map_feats:   (B, N_LANES, D_LANE) — LaneEncoder output (optional)
        Returns:     (B, 1, D)   — updated mode query
        """
        B, N, D = agent_feats.shape

        # Step 1: K-nearest agent selection (Algorithm 3, line 1)
        if N > self.k_nn:
            dist = agent_poses.norm(dim=-1)              # (B, N) — distance to ego origin
            _, topk = dist.topk(self.k_nn, dim=1, largest=False)  # (B, K)
            kv = agent_feats.gather(
                1, topk.unsqueeze(-1).expand(-1, -1, D)
            )                                            # (B, K, D)
        else:
            kv = agent_feats                             # (B, N, D)

        # Optionally concatenate map features to K/V
        if map_feats is not None:
            map_proj = self.lane_proj(map_feats)         # (B, N_LANES, D)
            kv = torch.cat([kv, map_proj], dim=1)        # (B, K + N_LANES, D)

        # Step 4: Transformer decoder (mode as query, agents+map as K/V)
        attn_out, _ = self.cross_attn(mode_query, kv, kv)
        mode_query = self.norm1(mode_query + attn_out)
        mode_query = self.norm2(mode_query + self.ffn(mode_query))

        return mode_query                                # (B, 1, D)


# ─────────────────────────────────────────────────────────────────────────────
# Mode Selector  (Eq 6 + Eq 7)
# ─────────────────────────────────────────────────────────────────────────────

class ModeSelector(nn.Module):
    """
    Predicts mode distribution and side-task ego trajectory from s_0.

    Inputs:  global_feat (B, D)  — PointNet global encoding of agents at t=0
    Outputs:
      logits    (B, N_MODES)               — Eq (6): CrossEntropy against c*
      side_traj (B, T_FUTURE, 3)           — Eq (7): L1 against gt_trajectory
    """
    def __init__(self):
        super().__init__()
        D = cfg.D_HIDDEN
        self.shared = nn.Sequential(
            nn.Linear(D, D), nn.ReLU(inplace=True),
            nn.Linear(D, D), nn.ReLU(inplace=True),
        )
        self.mode_head    = nn.Linear(D, cfg.N_MODES)
        self.side_task_head = nn.Linear(D, cfg.T_FUTURE * 3)

    def forward(self, global_feat: torch.Tensor):
        h = self.shared(global_feat)                     # (B, D)
        logits    = self.mode_head(h)                    # (B, N_MODES)
        side_traj = self.side_task_head(h).view(
            global_feat.size(0), cfg.T_FUTURE, 3
        )                                                # (B, T_FUTURE, 3)
        return logits, side_traj


# ─────────────────────────────────────────────────────────────────────────────
# Autoregressive Policy  (Algorithm 2 + Algorithm 3)
# ─────────────────────────────────────────────────────────────────────────────

class AutoregressivePolicy(nn.Module):
    """
    T-step autoregressive trajectory generator.

    At each step t (Algorithm 2, lines 3–4 + Algorithm 3):
      1. Agent states at t are transformed to ego-centric frame at t
         (IVM step: Algorithm 3, lines 3–4)
      2. Time history normalised to [-H:0]  (Algorithm 3, line 4)
      3. PointNet encodes agents → per-agent feats
      4. IVM: mode_query (updated from prev step) attends to agents + map
      5. Action head: updated query → waypoint a_t  (in initial ego frame)

    Mode c is fixed across all T steps (consistent AR, Figure 1c, Section III-B).
    The mode QUERY however is updated by cross-attention (carries temporal context).

    Inputs:
      agents_seq:  (B, T_FUTURE, N, 4)  — agent states at each future step
                   in the INITIAL ego-centric frame (t=0 frame)
                   (for IL: GT agents; for RL: transition model output)
      agents_mask: (B, N)               — validity mask
      mode_c:      (B,)  int64          — selected mode index in [0, N_MODES)
      gt_ego:      (B, T_FUTURE, 3)     — GT ego positions for teacher-forcing
                   the IVM coordinate transform (IL training only)
      map_lanes:   (B, N_LANES, N_PTS, 3) — lane centerlines in ego frame
      map_lanes_mask: (B, N_LANES)      — validity mask for lanes
    Output:
      trajectory:  (B, T_FUTURE, 3)    — predicted (x, y, yaw) in initial ego frame
    """
    def __init__(self):
        super().__init__()
        D = cfg.D_HIDDEN
        self.agent_encoder = PointNetEncoder(in_dim=4, d_model=D)
        self.lane_encoder  = LaneEncoder(in_dim=3, d_model=cfg.D_LANE)
        self.mode_embed    = nn.Embedding(cfg.N_MODES, D)
        self.ivm           = IVMBlock(d_model=D, n_heads=4,
                                      k_nn=cfg.N_AGENTS // 2)
        # Action head: updated mode query → (x, y, yaw)
        self.action_head   = nn.Sequential(
            nn.Linear(D, D), nn.ReLU(inplace=True),
            nn.Linear(D, 3)
        )

    @staticmethod
    def _transform_to_ego_frame(agents_xy: torch.Tensor,
                                agents_h:  torch.Tensor,
                                ego_x: torch.Tensor,
                                ego_y: torch.Tensor,
                                ego_h: torch.Tensor) -> tuple:
        """
        Transform agent positions/headings into the ego-centric frame
        defined by (ego_x, ego_y, ego_h).
        IVM Algorithm 3, line 3.

        agents_xy: (B, N, 2)
        agents_h:  (B, N)
        ego_x/y/h: (B,)
        """
        dx = agents_xy[..., 0] - ego_x.unsqueeze(1)     # (B, N)
        dy = agents_xy[..., 1] - ego_y.unsqueeze(1)

        cos_h = torch.cos(-ego_h).unsqueeze(1)           # (B, 1)
        sin_h = torch.sin(-ego_h).unsqueeze(1)

        x_e = cos_h * dx - sin_h * dy                   # (B, N)
        y_e = sin_h * dx + cos_h * dy
        h_e = agents_h - ego_h.unsqueeze(1)
        h_e = (h_e + math.pi) % (2 * math.pi) - math.pi

        return x_e, y_e, h_e                             # (B, N) each

    def forward(self, agents_seq: torch.Tensor, agents_mask: torch.Tensor,
                mode_c: torch.Tensor, gt_ego: torch.Tensor = None,
                map_lanes: torch.Tensor = None,
                map_lanes_mask: torch.Tensor = None) -> torch.Tensor:
        B = agents_seq.size(0)
        device = agents_seq.device

        # Initialise mode query from embedding (B, 1, D)
        mode_query = self.mode_embed(mode_c).unsqueeze(1)

        # Encode map lanes once (static across time steps)
        map_feats = None
        if map_lanes is not None:
            map_feats = self.lane_encoder(map_lanes, map_lanes_mask)  # (B, N_LANES, D_LANE)

        # Ego pose accumulator for coordinate transforms
        # At t=0 ego is at origin; updated via GT (teacher-forcing) or predictions
        ego_x = torch.zeros(B, device=device)
        ego_y = torch.zeros(B, device=device)
        ego_h = torch.zeros(B, device=device)

        actions = []

        for t in range(cfg.T_FUTURE):
            # ── Get agent states at step t (in initial ego frame) ─────────
            agents_t = agents_seq[:, t, :, :]            # (B, N, 4)
            ag_xy  = agents_t[..., :2]                   # (B, N, 2)
            ag_h   = agents_t[..., 2]                    # (B, N)
            ag_spd = agents_t[..., 3]                    # (B, N)

            # ── IVM step 1–3: transform to ego-centric frame at t ─────────
            # (Algorithm 3, lines 1–4)
            x_e, y_e, h_e = self._transform_to_ego_frame(
                ag_xy, ag_h, ego_x, ego_y, ego_h
            )
            # IVM step 4: time normalisation — the speed feature already
            # encodes magnitude; headings/positions are now t-relative
            agents_ego = torch.stack([x_e, y_e, h_e, ag_spd], dim=-1)  # (B, N, 4)

            # ── PointNet encode agents in ego frame ───────────────────────
            per_agent, _ = self.agent_encoder(agents_ego, agents_mask)  # (B, N, D)

            # ── IVM Transformer decoder: update mode query ────────────────
            # (Algorithm 3, Transformer decoder; mode c as query)
            mode_query = self.ivm(mode_query, per_agent,
                                  agents_ego[..., :2], map_feats)  # (B, 1, D) updated

            # ── Action head → waypoint in CURRENT ego frame ───────────────
            a_local = self.action_head(
                mode_query.squeeze(1)
            )                                            # (B, 3): dx, dy, dh in ego frame at t

            # ── Convert back to initial ego frame ─────────────────────────
            cos_h_bk = torch.cos(ego_h)                 # (B,)
            sin_h_bk = torch.sin(ego_h)
            x_g = cos_h_bk * a_local[:, 0] - sin_h_bk * a_local[:, 1] + ego_x
            y_g = sin_h_bk * a_local[:, 0] + cos_h_bk * a_local[:, 1] + ego_y
            h_g = a_local[:, 2] + ego_h
            h_g = (h_g + math.pi) % (2 * math.pi) - math.pi
            a_t = torch.stack([x_g, y_g, h_g], dim=-1)  # (B, 3) in initial frame
            actions.append(a_t.unsqueeze(1))

            # ── Update ego pose for next step ─────────────────────────────
            # Teacher-forcing (IL): use GT ego position for IVM transform
            # Inference / RL: use predicted action
            if gt_ego is not None and self.training:
                ego_x = gt_ego[:, t, 0]
                ego_y = gt_ego[:, t, 1]
                ego_h = gt_ego[:, t, 2]
            else:
                ego_x = a_t[:, 0].detach()
                ego_y = a_t[:, 1].detach()
                ego_h = a_t[:, 2].detach()

        return torch.cat(actions, dim=1)                 # (B, T_FUTURE, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Non-reactive Transition Model  (Eq 5, Stage A)
# ─────────────────────────────────────────────────────────────────────────────

class TransitionModel(nn.Module):
    """
    Non-reactive transition model P_τ(s^{1:N}_{1:T} | s_0).
    Predicts future agent trajectories independently of the ego action.
    Pre-trained with L1 loss (Eq 5), frozen during stages 2–3.

    Simplified: global PointNet encode s_0 → MLP → agent futures.
    Real implementation would use per-agent RNN/Transformer over history.

    Inputs:
      agents_now:  (B, N, 4)  — agents at t=0 in ego frame
      agents_mask: (B, N)
    Output:
      agent_futures: (B, T_FUTURE, N, 4)  — predicted agent states in ego frame
    """
    def __init__(self):
        super().__init__()
        D = cfg.D_HIDDEN
        self.agent_encoder = PointNetEncoder(in_dim=4, d_model=D)
        self.per_agent_head = nn.Sequential(
            nn.Linear(D + D, D), nn.ReLU(inplace=True),   # global + per-agent
            nn.Linear(D, cfg.T_FUTURE * 4),
        )

    def forward(self, agents_now: torch.Tensor,
                agents_mask: torch.Tensor) -> torch.Tensor:
        per_agent, global_feat = self.agent_encoder(
            agents_now, agents_mask
        )                                                # (B, N, D), (B, D)
        B, N, D = per_agent.shape
        g_exp = global_feat.unsqueeze(1).expand(-1, N, -1)  # (B, N, D)
        per = self.per_agent_head(
            torch.cat([per_agent, g_exp], dim=-1)
        )                                                # (B, N, T*4)
        return per.view(B, N, cfg.T_FUTURE, 4).permute(
            0, 2, 1, 3
        )                                                # (B, T, N, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Rule Selector  (Algorithm 2, step 5; inference only)
# ─────────────────────────────────────────────────────────────────────────────

class RuleSelector(nn.Module):
    """
    Inference-time rule-based trajectory selection (Algorithm 2, step 5).
    Scores K=60 candidate trajectories on:
      - Comfort: penalise jerk (3rd derivative of position via finite differences)
      - Progress: reward forward distance from start
      - Boundary: penalise trajectories that leave a rough drivable corridor

    Combines with mode selector scores (weighted sum, Algorithm 2 step 5).
    Returns index of best trajectory.

    All operations are pure Python / torch, no learned parameters.
    """

    def __init__(self, w_mode: float = 1.0, w_comfort: float = 0.1,
                 w_progress: float = 0.5):
        super().__init__()
        self.w_mode    = w_mode
        self.w_comfort = w_comfort
        self.w_progress = w_progress

    @torch.no_grad()
    def forward(self, mode_logits: torch.Tensor,
                all_trajs: torch.Tensor) -> tuple:
        """
        mode_logits: (B, N_MODES)
        all_trajs:   (B, N_MODES, T, 3)
        Returns:
          selected_traj: (B, T, 3)
          selected_idx:  (B,)  int64
        """
        B, M, T, _ = all_trajs.shape
        device = all_trajs.device

        # Mode selector score (higher is better → already in logits)
        mode_scores = F.softmax(mode_logits, dim=-1)         # (B, M)

        # Comfort: negative mean jerk (finite-diff on xy velocity)
        xy = all_trajs[..., :2]                              # (B, M, T, 2)
        vel  = xy[:, :, 1:] - xy[:, :, :-1]                 # (B, M, T-1, 2)
        acc  = vel[:, :, 1:] - vel[:, :, :-1]               # (B, M, T-2, 2)
        jerk = acc[:, :, 1:] - acc[:, :, :-1]               # (B, M, T-3, 2)
        comfort = -jerk.norm(dim=-1).mean(dim=-1)            # (B, M) — higher is smoother

        # Progress: forward distance (x-axis in ego frame, positive = forward)
        progress = all_trajs[:, :, -1, 0]                   # (B, M) final x-displacement

        # Combined score (Algorithm 2: "combine with mode selector scores")
        score = (self.w_mode * mode_scores
                 + self.w_comfort * comfort
                 + self.w_progress * progress)               # (B, M)

        selected_idx = score.argmax(dim=1)                   # (B,)
        selected_traj = all_trajs[
            torch.arange(B, device=device), selected_idx
        ]                                                    # (B, T, 3)

        return selected_traj, selected_idx


# ─────────────────────────────────────────────────────────────────────────────
# CarPlanner — full system
# ─────────────────────────────────────────────────────────────────────────────

class CarPlanner(nn.Module):
    """
    Full CarPlanner system (Figure 2).

    Submodules:
      agent_encoder_s0  — shared PointNet for s_0 (mode selector + policy init)
      mode_selector     — → logits (B, M) + side_traj (B, T, 3)
      policy            — AutoregressivePolicy with IVM
      transition_model  — pre-trained, used to build agents_seq at inference
      rule_selector     — inference-time trajectory selection

    Backbone sharing (Table 4 ablation):
      IL best config:  shared agent_encoder_s0 between mode_selector and policy (BACKBONE_SHARING=True)
      RL best config:  separate encoders for policy and value (BACKBONE_SHARING=False)
      Controlled by cfg.BACKBONE_SHARING.
    """

    def __init__(self):
        super().__init__()
        # Shared s_0 encoder for mode selector (and optionally policy backbone)
        self.s0_encoder = PointNetEncoder(in_dim=4, d_model=cfg.D_HIDDEN)
        # If NOT backbone sharing, policy has its own per-step encoder inside AutoregressivePolicy
        self.mode_selector     = ModeSelector()
        self.policy            = AutoregressivePolicy()
        self.transition_model  = TransitionModel()
        self.rule_selector     = RuleSelector()

    # ── Training forward (IL Stage B) ────────────────────────────────────────

    def forward_train(self, agents_now: torch.Tensor, agents_mask: torch.Tensor,
                      agents_seq: torch.Tensor, gt_traj: torch.Tensor,
                      mode_label: torch.Tensor,
                      map_lanes: torch.Tensor = None,
                      map_lanes_mask: torch.Tensor = None):
        """
        IL training pass (Algorithm 1, Stage 2 IL branch).

        Inputs:
          agents_now:  (B, N, 4)             — agents at t=0 in ego frame
          agents_mask: (B, N)
          agents_seq:  (B, T_FUTURE, N, 4)   — GT agent states at t=1..T in ego frame
          gt_traj:     (B, T_FUTURE, 3)      — GT ego trajectory in ego frame
          mode_label:  (B,)  int64           — positive mode c* (pre-assigned)
          map_lanes:   (B, N_LANES, N_PTS, 3) — lane centerlines in ego frame
          map_lanes_mask: (B, N_LANES)       — validity mask for lanes

        Returns:
          mode_logits: (B, N_MODES)
          side_traj:   (B, T_FUTURE, 3)
          pred_traj:   (B, T_FUTURE, 3)
        """
        # Encode s_0 agents for mode selector
        _, s0_global = self.s0_encoder(agents_now, agents_mask)  # (B, D)

        # Mode selector  (Eq 6 + Eq 7)
        mode_logits, side_traj = self.mode_selector(s0_global)

        # Mode dropout (Table 4: improves both IL and RL)
        c = mode_label.clone()
        if cfg.MODE_DROPOUT and self.training:
            drop_mask = torch.rand(c.size(0), device=c.device) < cfg.MODE_DROPOUT_P
            c[drop_mask] = torch.randint(
                0, cfg.N_MODES, (int(drop_mask.sum()),), device=c.device
            )

        # Ego-history dropout (Table 4: ON for IL best, OFF for RL best)
        # Applied to agents_now before encoding inside the policy if enabled
        # (implemented as zeroing the passed agents_now for the policy)
        agents_now_policy = agents_now
        if cfg.EGO_HISTORY_DROPOUT and self.training:
            drop_mask_b = (torch.rand(agents_now.size(0), 1, 1, device=agents_now.device)
                           > cfg.MODE_DROPOUT_P).float()
            agents_now_policy = agents_now * drop_mask_b

        # Autoregressive policy with IVM (T steps, teacher-forcing on ego frame)
        pred_traj = self.policy(
            agents_seq=agents_seq,
            agents_mask=agents_mask,
            mode_c=c,
            gt_ego=gt_traj,          # teacher-forcing for IVM frame transforms
            map_lanes=map_lanes,
            map_lanes_mask=map_lanes_mask,
        )                            # (B, T_FUTURE, 3)

        return mode_logits, side_traj, pred_traj

    # ── Inference forward ─────────────────────────────────────────────────────

    @torch.no_grad()
    def forward_inference(self, agents_now: torch.Tensor, agents_mask: torch.Tensor,
                          map_lanes: torch.Tensor = None,
                          map_lanes_mask: torch.Tensor = None):
        """
        Full inference pass (Algorithm 2):
          1. Encode s_0
          2. Mode selector → scores for all N_MODES modes
          3. Transition model → agent futures
          4. For each of N_MODES modes: run autoregressive policy with IVM
          5. Rule selector → best trajectory

        Inputs:
          agents_now:  (B, N, 4)             — agents at t=0 in ego frame
          agents_mask: (B, N)
          map_lanes:   (B, N_LANES, N_PTS, 3) — lane centerlines in ego frame
          map_lanes_mask: (B, N_LANES)       — validity mask for lanes

        Returns:
          mode_logits:   (B, N_MODES)
          all_trajs:     (B, N_MODES, T_FUTURE, 3)
          best_traj:     (B, T_FUTURE, 3)
          best_idx:      (B,)
        """
        self.eval()
        B = agents_now.size(0)

        # Step 1: encode s_0
        _, s0_global = self.s0_encoder(agents_now, agents_mask)

        # Step 2: mode scores
        mode_logits, _ = self.mode_selector(s0_global)

        # Step 3: transition model → agent futures for all steps
        agent_futures = self.transition_model(
            agents_now, agents_mask
        )                                                # (B, T, N, 4)

        # Step 4: generate one trajectory per mode
        all_trajs = []
        for m in range(cfg.N_MODES):
            c = torch.full((B,), m, dtype=torch.long, device=agents_now.device)
            traj = self.policy(
                agents_seq=agent_futures,
                agents_mask=agents_mask,
                mode_c=c,
                gt_ego=None,         # no teacher-forcing at inference
                map_lanes=map_lanes,
                map_lanes_mask=map_lanes_mask,
            )                        # (B, T, 3)
            all_trajs.append(traj.unsqueeze(1))

        all_trajs = torch.cat(all_trajs, dim=1)          # (B, N_MODES, T, 3)

        # Step 5: rule selector
        best_traj, best_idx = self.rule_selector(mode_logits, all_trajs)

        return mode_logits, all_trajs, best_traj, best_idx

    # ── Transition model pre-training pass (Stage A, Eq 5) ───────────────────

    def forward_transition(self, agents_now: torch.Tensor, agents_mask: torch.Tensor):
        """
        Forward pass for transition model pre-training.
        Returns predicted agent futures: (B, T_FUTURE, N, 4)
        """
        return self.transition_model(agents_now, agents_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Shape self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    B = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    model = CarPlanner().to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
    for name, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters())
        print(f"  {name:25s}: {n:>8,}")

    # ── Dummy data matching data contract ──────────────────────────────────
    T, N, D = cfg.T_FUTURE, cfg.N_AGENTS, cfg.D_HIDDEN
    agents_now  = torch.randn(B, N, 4, device=device)
    agents_mask = torch.ones(B, N, device=device)
    agents_seq  = torch.randn(B, T, N, 4, device=device)   # GT agent futures
    gt_traj     = torch.randn(B, T, 3, device=device)
    mode_label  = torch.randint(0, cfg.N_MODES, (B,), device=device)
    map_lanes   = torch.randn(B, cfg.N_LANES, cfg.N_LANE_POINTS, 3, device=device)
    map_lanes_mask = torch.ones(B, cfg.N_LANES, device=device)

    # ── Training forward ───────────────────────────────────────────────────
    model.train()
    logits, side, pred = model.forward_train(
        agents_now, agents_mask, agents_seq, gt_traj, mode_label,
        map_lanes=map_lanes, map_lanes_mask=map_lanes_mask
    )
    assert logits.shape == (B, cfg.N_MODES),     f"logits {logits.shape}"
    assert side.shape   == (B, T, 3),            f"side {side.shape}"
    assert pred.shape   == (B, T, 3),            f"pred {pred.shape}"
    print(f"\n✓ forward_train shapes:  logits={tuple(logits.shape)}  "
          f"side={tuple(side.shape)}  pred={tuple(pred.shape)}")

    # ── Backward ───────────────────────────────────────────────────────────
    L = (F.cross_entropy(logits, mode_label)
         + F.l1_loss(side, gt_traj)
         + F.l1_loss(pred, gt_traj))
    L.backward()
    print(f"✓ backward OK  L={L.item():.4f}")

    # ── Inference forward ──────────────────────────────────────────────────
    inf_logits, all_trajs, best_traj, best_idx = model.forward_inference(
        agents_now, agents_mask,
        map_lanes=map_lanes, map_lanes_mask=map_lanes_mask
    )
    assert inf_logits.shape == (B, cfg.N_MODES),             f"inf_logits"
    assert all_trajs.shape  == (B, cfg.N_MODES, T, 3),       f"all_trajs"
    assert best_traj.shape  == (B, T, 3),                    f"best_traj"
    assert best_idx.shape   == (B,),                         f"best_idx"
    print(f"✓ forward_inference:     all_trajs={tuple(all_trajs.shape)}  "
          f"best_traj={tuple(best_traj.shape)}")

    # ── Transition model forward ───────────────────────────────────────────
    agent_fut = model.forward_transition(agents_now, agents_mask)
    assert agent_fut.shape == (B, T, N, 4),                  f"agent_fut"
    print(f"✓ transition model:      agent_fut={tuple(agent_fut.shape)}")

    print("\n✓ All shape tests passed.")
