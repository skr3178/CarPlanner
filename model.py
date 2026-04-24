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
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, d_model),
        )
        self.d_model = d_model

    def forward(self, lanes: torch.Tensor, mask: torch.Tensor = None):
        """
        lanes: (B, N_LANES, N_POINTS, D_MAP_POINT)
        mask:  (B, N_LANES) — 1=valid, 0=padding
        Returns: per_lane (B, N_LANES, D_LANE)
        """
        B, N_L, N_P, D = lanes.shape
        # Flatten batch + lanes for point-wise encoding
        flat = lanes.view(B * N_L, N_P, D)               # (B*N_L, N_P, D_MAP_POINT)
        per_point = self.point_mlp(flat)                  # (B*N_L, N_P, D_LANE)

        # Max-pool over points within each lane
        lane_feat = per_point.max(dim=1).values           # (B*N_L, D)
        lane_feat = lane_feat.view(B, N_L, self.d_model)  # (B, N_LANES, D_LANE)

        # Apply mask (zero out padding lanes)
        if mask is not None:
            lane_feat = lane_feat * mask.unsqueeze(-1)

        return lane_feat


# ─────────────────────────────────────────────────────────────────────────────
# Polygon encoder  (PointNet over polygon rings — separate weights per paper)
# ─────────────────────────────────────────────────────────────────────────────

class PolygonEncoder(nn.Module):
    """
    PointNet-style encoder for polygon map elements (crosswalks, stop lines, intersections).
    Same structure as LaneEncoder but separate weights per paper's "another PointNet".

    Input:  polygons (B, N_POLYGONS, N_POINTS, D_POLYGON_POINT) + mask (B, N_POLYGONS)
    Output: per_polygon (B, N_POLYGONS, D_LANE)
    """
    def __init__(self, in_dim: int = cfg.D_POLYGON_POINT, d_model: int = cfg.D_LANE):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, d_model),
        )
        self.d_model = d_model

    def forward(self, polygons: torch.Tensor, mask: torch.Tensor = None):
        B, N_P, N_Pt, D = polygons.shape
        flat = polygons.view(B * N_P, N_Pt, D)
        per_point = self.point_mlp(flat)
        poly_feat = per_point.max(dim=1).values
        poly_feat = poly_feat.view(B, N_P, self.d_model)
        if mask is not None:
            poly_feat = poly_feat * mask.unsqueeze(-1)
        return poly_feat


# ─────────────────────────────────────────────────────────────────────────────
# IVM — Invariant-View Module  (Algorithm 3)
# ─────────────────────────────────────────────────────────────────────────────

class IVM(nn.Module):
    """
    Invariant-View Module (§3.3.3) — parameter-less preprocessing.

    Per autoregressive timestep t (after upstream coord transform + time norm):
      1. KNN agent selection (K = N // 2)
      2. KNN map element selection (K_m = N_m // 2)
      3. KNN polygon selection (K_p = N_p // 2)
      4. Concatenate filtered agents + map + routes + polygons into K/V tensor

    Route trim (K_r = N_r/4) is applied upstream before encoding.
    Coord transform and time normalization are applied upstream in the AR loop.
    """
    def __init__(self, k_nn: int = cfg.N_AGENTS // 2):
        super().__init__()
        self.k_nn = k_nn

    def forward(self, agent_feats: torch.Tensor, agent_poses: torch.Tensor,
                map_feats: torch.Tensor = None, map_poses: torch.Tensor = None,
                route_feats: torch.Tensor = None,
                polygon_feats: torch.Tensor = None,
                polygon_poses: torch.Tensor = None) -> torch.Tensor:
        """
        Returns kv: (B, K_total, D) — filtered and concatenated context for decoder.
        """
        B, HN, D = agent_feats.shape
        N = agent_poses.size(1)

        H = HN // N
        if N > self.k_nn:
            dist = agent_poses.norm(dim=-1)              # (B, N)
            _, topk = dist.topk(self.k_nn, dim=1, largest=False)  # (B, K)
            kv_list = []
            for h in range(H):
                idx = topk + h * N
                kv_list.append(agent_feats.gather(
                    1, idx.unsqueeze(-1).expand(-1, -1, D)
                ))
            kv = torch.cat(kv_list, dim=1)               # (B, K*H, D)
        else:
            kv = agent_feats                             # (B, H*N, D)

        if map_feats is not None:
            if map_poses is not None:
                N_m = map_feats.size(1)
                K_m = max(1, N_m // 2)
                if N_m > K_m:
                    map_dist = map_poses.norm(dim=-1)
                    _, map_topk = map_dist.topk(K_m, dim=1, largest=False)
                    map_feats = map_feats.gather(
                        1, map_topk.unsqueeze(-1).expand(-1, -1, map_feats.size(-1))
                    )
            kv = torch.cat([kv, map_feats], dim=1)

        if route_feats is not None:
            kv = torch.cat([kv, route_feats], dim=1)

        if polygon_feats is not None:
            if polygon_poses is not None:
                N_p = polygon_feats.size(1)
                K_p = max(1, N_p // 2)
                if N_p > K_p:
                    poly_dist = polygon_poses.norm(dim=-1)
                    _, poly_topk = poly_dist.topk(K_p, dim=1, largest=False)
                    polygon_feats = polygon_feats.gather(
                        1, poly_topk.unsqueeze(-1).expand(-1, -1, polygon_feats.size(-1))
                    )
            kv = torch.cat([kv, polygon_feats], dim=1)

        return kv                                        # (B, K_total, D)


class PolicyDecoderLayer(nn.Module):
    """
    Transformer decoder layer (Table 5: 3 layers).
    Cross-attention: mode query attends to IVM-filtered context (agents + map + routes).
    """
    def __init__(self, d_model: int = cfg.D_HIDDEN, n_heads: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=0.1
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, mode_query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        mode_query: (B, 1, D)       — current mode state
        kv:         (B, K_total, D)  — IVM-filtered context
        Returns:    (B, 1, D)       — updated mode query
        """
        attn_out, _ = self.cross_attn(mode_query, kv, kv)
        mode_query = self.norm1(mode_query + attn_out)
        mode_query = self.norm2(mode_query + self.ffn(mode_query))
        return mode_query


# ─────────────────────────────────────────────────────────────────────────────
# Decomposed Mode Encoder  (Section 3.2: lon/lat mode construction)
# ─────────────────────────────────────────────────────────────────────────────

def select_candidate_routes(map_lanes: torch.Tensor,
                            map_lanes_mask: torch.Tensor,
                            n_routes: int = cfg.N_LAT) -> tuple:
    """
    Select up to n_routes heading-aligned candidate lanes from map data.

    Uses the same filtering logic as data_loader._collect_candidate_lanes
    but operates on torch tensors and returns selected lane polylines.

    Args:
        map_lanes:      (B, N_LANES, N_PTS, 9) — all lane polylines in ego frame
        map_lanes_mask: (B, N_LANES) — validity mask
        n_routes:       number of candidate routes to select (default N_LAT=5)

    Returns:
        route_polylines: (B, n_routes, N_PTS, 9) — selected candidate lanes
        route_mask:      (B, n_routes) — validity mask for selected lanes
    """
    B, N_L, N_P, D = map_lanes.shape
    device = map_lanes.device

    route_polylines = torch.zeros(B, n_routes, N_P, D, device=device,
                                   dtype=map_lanes.dtype)
    route_mask = torch.zeros(B, n_routes, device=device, dtype=map_lanes_mask.dtype)

    for b in range(B):
        candidates = []
        for i in range(N_L):
            if map_lanes_mask[b, i] < 0.5:
                continue
            lane = map_lanes[b, i]                              # (N_P, 9)
            valid = lane[torch.abs(lane[:, 0]) + torch.abs(lane[:, 1]) > 1e-6]
            if len(valid) < 2:
                continue

            # Closest point to ego (origin)
            dists = torch.sqrt(valid[:, 0] ** 2 + valid[:, 1] ** 2)
            closest_idx = int(torch.argmin(dists))

            # Heading alignment (ego faces +x, heading ≈ 0)
            heading = torch.atan2(valid[closest_idx, 2], valid[closest_idx, 3])
            if abs(heading.item()) > math.pi / 3:
                continue

            candidates.append((i, dists[closest_idx].item(), valid[closest_idx, 1].item()))

        # Sort by lateral offset to spread across left/center/right
        candidates.sort(key=lambda c: c[2])

        # Pick up to n_routes, spread evenly
        if len(candidates) <= n_routes:
            selected = candidates
        else:
            indices = torch.linspace(0, len(candidates) - 1, n_routes).long().tolist()
            selected = [candidates[i] for i in indices]

        for j, (lane_idx, _, _) in enumerate(selected[:n_routes]):
            route_polylines[b, j] = map_lanes[b, lane_idx]
            route_mask[b, j] = 1.0

    return route_polylines, route_mask


class DecomposedModeEncoder(nn.Module):
    """
    Paper-faithful decomposed mode construction (Section 3.2).

    Longitudinal: c_lon,j = j/N_lon scalar, repeated across D → N_lon × D
    Lateral:      PointNet over N_lat route polylines → N_lat × D (per scene)
    Combined:     concat lon + lat → N_lat × N_lon × 2D → Linear → N_lat × N_lon × D
    """

    def __init__(self):
        super().__init__()
        D = cfg.D_HIDDEN

        # Lateral: PointNet over route polylines — in_dim=27 (3×Dm: center+left+right per point)
        self.route_pointnet = LaneEncoder(
            in_dim=cfg.D_POLYLINE_POINT, d_model=cfg.D_LANE
        )

        # Combined: linear projection 2D → D
        self.combine = nn.Linear(2 * D, D)

    def encode_longitudinal(self, device: torch.device) -> torch.Tensor:
        """
        Construct N_lon × D longitudinal mode tensor.
        c_lon,j = j/N_lon repeated across D dimensions.
        No learned parameters — deterministic construction.
        """
        scalars = torch.arange(cfg.N_LON, dtype=torch.float32, device=device) / cfg.N_LON
        return scalars.unsqueeze(1).expand(cfg.N_LON, cfg.D_HIDDEN)  # (N_lon, D)

    def encode_lateral(self, route_polylines: torch.Tensor,
                       route_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode N_lat candidate routes via PointNet (per scene).

        Args:
            route_polylines: (B, N_lat, N_PTS, 9)
            route_mask:      (B, N_lat)
        Returns:
            lat_modes: (B, N_lat, D) — scene-dependent lateral mode features
        """
        # LaneEncoder: (B, N_lat, N_PTS, 9) → (B, N_lat, D)
        return self.route_pointnet(route_polylines, route_mask)

    def construct_mode_tensor(self, lon_modes: torch.Tensor,
                              lat_modes: torch.Tensor) -> torch.Tensor:
        """
        Combine lon + lat → N_lat × N_lon × 2D → Linear → N_lat × N_lon × D.

        Args:
            lon_modes: (N_lon, D) or (B, N_lon, D)
            lat_modes: (B, N_lat, D)
        Returns:
            mode_tensor: (B, N_lat, N_lon, D)
        """
        B = lat_modes.size(0)
        N_lon, N_lat = cfg.N_LON, cfg.N_LAT
        D = cfg.D_HIDDEN

        # Ensure lon_modes has batch dim
        if lon_modes.dim() == 2:
            lon_exp = lon_modes.unsqueeze(0).expand(B, -1, -1)  # (B, N_lon, D)
        else:
            lon_exp = lon_modes

        # Expand for cartesian product: (B, N_lat, N_lon, D)
        lon_exp = lon_exp.unsqueeze(1).expand(-1, N_lat, -1, -1)  # (B, N_lat, N_lon, D)
        lat_exp = lat_modes.unsqueeze(2).expand(-1, -1, N_lon, -1)  # (B, N_lat, N_lon, D)

        # Concatenate + linear projection: (B, N_lat, N_lon, 2D) → (B, N_lat, N_lon, D)
        combined = torch.cat([lon_exp, lat_exp], dim=-1)
        return self.combine(combined)

    def get_mode_query(self, mode_c: torch.Tensor,
                       mode_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract single mode embedding for IVM query initialisation.

        Args:
            mode_c:        (B,) int64 — flat mode index in [0, N_MODES)
            mode_tensor:   (B, N_lat, N_lon, D)
        Returns:
            mode_query: (B, 1, D)
        """
        lon_idx = mode_c // cfg.N_LAT                      # (B,)
        lat_idx = mode_c % cfg.N_LAT                        # (B,)
        B = mode_c.size(0)

        # Gather: for each batch element, pick mode_tensor[b, lat_idx[b], lon_idx[b], :]
        mode_embed = mode_tensor[
            torch.arange(B, device=mode_c.device),
            lat_idx, lon_idx
        ]                                                    # (B, D)
        return mode_embed.unsqueeze(1)                       # (B, 1, D)


# ─────────────────────────────────────────────────────────────────────────────
# Mode Selector  (Eq 6 + Eq 7)
# ─────────────────────────────────────────────────────────────────────────────

class ModeSelector(nn.Module):
    """
    Mode selector with decomposed lon/lat mode construction (Section 3.2 + 3.3).

    Architecture:
      1. Construct mode tensor: lon scalar × D + lat PointNet(route) → 2D → Linear D
      2. Transformer decoder: mode features as query, scene features as K/V
      3. MLP per mode → logits → softmax
      4. Side task head conditioned on selected mode

    Inputs:
      global_feat:     (B, D)  — s0 PointNet global feature
      s0_per_agent:    (B, N, D) — s0 PointNet per-agent features (for K/V)
      map_lanes:       (B, N_LANES, N_PTS, 9) — lane polylines for route encoding
      map_lanes_mask:  (B, N_LANES)
      mode_c:          (B,) int64 — positive mode c* (for side-task conditioning)

    Outputs:
      logits:    (B, N_MODES)
      side_traj: (B, T_FUTURE, 3)
    """
    def __init__(self):
        super().__init__()
        D = cfg.D_HIDDEN

        # Decomposed mode encoder
        self.decomposed_mode = DecomposedModeEncoder()

        # s0 shared encoder (processes global_feat)
        self.shared = nn.Sequential(
            nn.Linear(D, D), nn.ReLU(inplace=True),
            nn.Linear(D, D), nn.ReLU(inplace=True),
        )

        # Map encoder for K/V
        self.map_encoder = LaneEncoder(in_dim=cfg.D_POLYLINE_POINT, d_model=cfg.D_LANE)
        self.polygon_encoder = PolygonEncoder()

        # Transformer decoder: mode features as query, scene features as K/V
        self.mode_transformer = nn.MultiheadAttention(
            D, num_heads=8, batch_first=True, dropout=0.1
        )
        self.mode_norm = nn.LayerNorm(D)
        self.mode_ffn = nn.Sequential(
            nn.Linear(D, D * 4), nn.ReLU(inplace=True),
            nn.Linear(D * 4, D),
        )
        self.mode_norm2 = nn.LayerNorm(D)

        # Score head: per-mode scalar → (B, N_MODES)
        self.mode_head = nn.Linear(D, 1)

        # Side-task conditioned on selected mode c*
        self.side_proj = nn.Linear(D, cfg.D_MODE_EMBED)
        self.side_task_head = nn.Linear(D + cfg.D_MODE_EMBED, cfg.T_FUTURE * 3)

    def forward(self, global_feat: torch.Tensor,
                s0_per_agent: torch.Tensor = None,
                map_lanes: torch.Tensor = None,
                map_lanes_mask: torch.Tensor = None,
                mode_c: torch.Tensor = None,
                map_polygons: torch.Tensor = None,
                map_polygons_mask: torch.Tensor = None,
                route_polylines: torch.Tensor = None,
                route_mask: torch.Tensor = None):
        """
        Args:
            global_feat:      (B, D)
            s0_per_agent:     (B, N, D) — per-agent features for Transformer K/V
            map_lanes:        (B, N_LANES, N_PTS, D_POLYLINE_POINT) — for K/V scene context
            map_lanes_mask:   (B, N_LANES)
            mode_c:           (B,) int64 — for side-task conditioning (None at inference)
            map_polygons:     (B, N_POLYGONS, N_PTS, D_POLYGON_POINT) — polygon map elements
            map_polygons_mask:(B, N_POLYGONS) — polygon validity mask
            route_polylines:  (B, N_LAT, N_PTS, D_POLYLINE_POINT) — precomputed route polylines
            route_mask:       (B, N_LAT) — route validity mask
        Returns:
            logits:    (B, N_MODES)
            side_traj: (B, T_FUTURE, 3)
        """
        B = global_feat.size(0)
        D = cfg.D_HIDDEN
        device = global_feat.device

        # 1. Encode s0 features
        h = self.shared(global_feat)                          # (B, D)

        # 2. Construct decomposed mode tensor
        lon_modes = self.decomposed_mode.encode_longitudinal(device)  # (N_lon, D)

        lat_modes = None
        if route_polylines is not None and route_mask is not None:
            lat_modes = self.decomposed_mode.encode_lateral(
                route_polylines, route_mask
            )                                                # (B, N_lat, D)

        if lat_modes is None:
            # Fallback: zero lateral modes (no map data)
            lat_modes = torch.zeros(B, cfg.N_LAT, D, device=device)

        mode_tensor = self.decomposed_mode.construct_mode_tensor(
            lon_modes, lat_modes
        )                                                    # (B, N_lat, N_lon, D)

        # Flatten mode grid: (B, N_lat, N_lon, D) → permute to (B, N_lon, N_lat, D)
        # so flat index = lon * N_LAT + lat — matches data_loader._assign_mode ordering
        mode_feats = mode_tensor.permute(0, 2, 1, 3).contiguous().view(B, cfg.N_MODES, D)

        # 3. Transformer decoder: modes as query, scene as K/V
        #    K/V = per-agent features + map features + global feature
        kv_parts = [h.unsqueeze(1)]                            # (B, 1, D) — global
        if s0_per_agent is not None:
            kv_parts.append(s0_per_agent)                      # (B, N, D) — per-agent
        if map_lanes is not None and map_lanes_mask is not None:
            map_feats = self.map_encoder(map_lanes, map_lanes_mask)  # (B, N_L, D)
            kv_parts.append(map_feats)
        if map_polygons is not None and map_polygons_mask is not None:
            poly_feats = self.polygon_encoder(map_polygons, map_polygons_mask)  # (B, N_P, D)
            kv_parts.append(poly_feats)
        scene_kv = torch.cat(kv_parts, dim=1)                  # (B, 1 + N + N_m + N_p, D)

        attn_out, _ = self.mode_transformer(
            query=mode_feats, key=scene_kv, value=scene_kv
        )
        mode_updated = self.mode_norm(mode_feats + attn_out)
        mode_updated = self.mode_norm2(mode_updated + self.mode_ffn(mode_updated))

        # 4. Score each mode → logits
        logits = self.mode_head(mode_updated).squeeze(-1)     # (B, N_MODES)

        # 5. Side task: conditioned on c* mode feature from the updated tensor
        if mode_c is not None and mode_tensor is not None:
            mode_embed = self.decomposed_mode.get_mode_query(mode_c, mode_tensor)
            c_proj = self.side_proj(mode_embed.squeeze(1))    # (B, D_MODE_EMBED)
            h_side = torch.cat([h, c_proj], dim=-1)
        else:
            h_side = torch.cat([h, torch.zeros(
                h.size(0), cfg.D_MODE_EMBED, device=device
            )], dim=-1)
        side_traj = self.side_task_head(h_side).view(B, cfg.T_FUTURE, 3)

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
        # Time-aware encoder: 4 agent features + 1 normalised time index
        self.agent_encoder_time = PointNetEncoder(in_dim=cfg.D_AGENT, d_model=D)
        self.lane_encoder  = LaneEncoder(in_dim=cfg.D_POLYLINE_POINT, d_model=cfg.D_LANE)
        self.polygon_encoder = PolygonEncoder()
        # Decomposed mode encoder (Section 3.2) — replaces flat nn.Embedding
        self.decomposed_mode = DecomposedModeEncoder()
        # IVM: parameter-less KNN filtering (§3.3.3)
        self.ivm = IVM(k_nn=cfg.N_AGENTS * cfg.T_HIST // 2)
        # Transformer decoder: 3 layers (Table 5)
        self.decoder_layers = nn.ModuleList([
            PolicyDecoderLayer(d_model=D, n_heads=8)
            for _ in range(3)
        ])
        # Action head: Gaussian policy — shared between Stage B (IL) and Stage C (RL).
        # Stage B applies L1 on the mean output; Stage C uses mean + std for PPO.
        self.action_mean_head = nn.Sequential(
            nn.Linear(D, D), nn.ReLU(inplace=True),
            nn.Linear(D, 3)
        )
        self.action_log_std = nn.Parameter(torch.zeros(3))

        # Value head (Stage C only, unused in Stage B)
        self.value_head = nn.Sequential(
            nn.Linear(D, D // 2), nn.ReLU(inplace=True),
            nn.Linear(D // 2, 1)
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

    def forward(self, agents_now: torch.Tensor, agents_seq: torch.Tensor,
                agents_mask: torch.Tensor, mode_c: torch.Tensor,
                gt_ego: torch.Tensor = None,
                map_lanes: torch.Tensor = None,
                map_lanes_mask: torch.Tensor = None,
                map_polygons: torch.Tensor = None,
                map_polygons_mask: torch.Tensor = None,
                route_polylines: torch.Tensor = None,
                route_mask: torch.Tensor = None) -> torch.Tensor:
        """
        T-step autoregressive rollout with IVM (Algorithm 2 + Algorithm 3).

       agents_now:  (B, N, Da)  — agents at t=0 in initial ego frame
        agents_seq:  (B, T_FUTURE, N, Da) — agent states at t=1..T in initial ego frame
        """
        B = agents_seq.size(0)
        device = agents_seq.device
        H = cfg.T_HIST  # history window length for IVM time normalisation

        # Initialise mode query from decomposed mode tensor (Section 3.2)
        lon_modes = self.decomposed_mode.encode_longitudinal(device)  # (N_lon, D)
        if route_polylines is not None and route_mask is not None:
            lat_modes = self.decomposed_mode.encode_lateral(route_polylines, route_mask)
        else:
            lat_modes = torch.zeros(B, cfg.N_LAT, cfg.D_HIDDEN, device=device)
        mode_tensor = self.decomposed_mode.construct_mode_tensor(lon_modes, lat_modes)
        mode_query = self.decomposed_mode.get_mode_query(mode_c, mode_tensor)  # (B, 1, D)

        # Ego pose accumulator for coordinate transforms
        ego_x = torch.zeros(B, device=device)
        ego_y = torch.zeros(B, device=device)
        ego_h = torch.zeros(B, device=device)

        # Agent history buffer: H most recent frames for IVM time normalisation
        # (Alg 3 step 4: t-H:t → -H:0)
        # Initialise with t=0 agents repeated H times (no real history available)
        agent_buffer = torch.stack(
            [agents_now] * H, dim=1
        )  # (B, H, N, Da)

        actions = []

        for t in range(cfg.T_FUTURE):
            # ── Get agent states at step t (in initial ego frame) ─────────
            agents_t = agents_seq[:, t, :, :]            # (B, N, Da)

            # ── Update history buffer: append current, drop oldest ─────────
            # (Alg 3 step 4: maintain H-length window, time-normalised)
            agent_buffer = torch.cat(
                [agent_buffer[:, 1:, :, :], agents_t.unsqueeze(1)], dim=1
            )  # (B, H, N, 4)

            # Time normalisation: assign indices [-H, ..., -1, 0]
            # Last slot (index H-1) = current step t → normalised time 0
            # Earlier slots → normalised time -H..-1
            time_indices = torch.arange(-(H - 1), 1, dtype=torch.float, device=device)  # (H,)

            # ── IVM step 1–3: transform buffer to ego-centric frame at t ──
            # Da=14: x, y, sin_h, cos_h, vx, vy, box_w, box_l, box_h, time_step, cat_onehot(4)
            Da = cfg.D_AGENT
            buffer_ego = torch.zeros(B, H, cfg.N_AGENTS, Da, device=device,
                                     dtype=agent_buffer.dtype)
            for h_idx in range(H):
                ag_xy_h  = agent_buffer[:, h_idx, :, :2]    # (B, N, 2) — x, y
                ag_sin_h = agent_buffer[:, h_idx, :, 2]     # (B, N) — sin(heading)
                ag_cos_h = agent_buffer[:, h_idx, :, 3]     # (B, N) — cos(heading)
                ag_h_h   = torch.atan2(ag_sin_h, ag_cos_h)  # (B, N) — reconstruct heading
                ag_vx_h  = agent_buffer[:, h_idx, :, 4]     # (B, N) — vx
                ag_vy_h  = agent_buffer[:, h_idx, :, 5]     # (B, N) — vy
                ag_bw_h  = agent_buffer[:, h_idx, :, 6]     # (B, N) — box_w
                ag_bl_h  = agent_buffer[:, h_idx, :, 7]     # (B, N) — box_l
                ag_bh_h  = agent_buffer[:, h_idx, :, 8]     # (B, N) — box_h
                ag_cat_h = agent_buffer[:, h_idx, :, 10:14]  # (B, N, 4) — category one-hot
                x_e, y_e, h_e = self._transform_to_ego_frame(
                    ag_xy_h, ag_h_h, ego_x, ego_y, ego_h
                )
                cos_h = torch.cos(-ego_h).unsqueeze(1)
                sin_h = torch.sin(-ego_h).unsqueeze(1)
                vx_e = cos_h * ag_vx_h - sin_h * ag_vy_h
                vy_e = sin_h * ag_vx_h + cos_h * ag_vy_h
                t_norm = time_indices[h_idx].expand(B, cfg.N_AGENTS)
                buffer_ego[:, h_idx] = torch.cat([
                    x_e.unsqueeze(-1), y_e.unsqueeze(-1),
                    torch.sin(h_e).unsqueeze(-1), torch.cos(h_e).unsqueeze(-1),
                    vx_e.unsqueeze(-1), vy_e.unsqueeze(-1),
                    ag_bw_h.unsqueeze(-1), ag_bl_h.unsqueeze(-1), ag_bh_h.unsqueeze(-1),
                    t_norm.unsqueeze(-1),
                    ag_cat_h,
                ], dim=-1)  # (B, N, Da=14)

            # Flatten history into agent dimension for PointNet
            # (B, H*N, Da) — PointNet sees all H frames as a point cloud
            buffer_flat = buffer_ego.view(B, H * cfg.N_AGENTS, Da)
            mask_flat = agents_mask.unsqueeze(1).expand(-1, H, -1).reshape(
                B, H * cfg.N_AGENTS
            )  # (B, H*N)

            # ── PointNet encode agents with time-normalised features ───────
            per_agent, _ = self.agent_encoder_time(buffer_flat, mask_flat)  # (B, H*N, D)

            # ── IVM K-NN selection: use only CURRENT step (last H slot) ────
            # K-NN should be on spatial positions only (current agents)
            agents_ego_now = buffer_ego[:, -1]            # (B, N, Da)

            # ── Map re-transform per step (Alg 3 step 3) ──────────────────
            map_feats = None
            lanes_ego = None
            if map_lanes is not None:
                lanes_t = map_lanes.clone()
                cos_hl = torch.cos(-ego_h).unsqueeze(1).unsqueeze(1)
                sin_hl = torch.sin(-ego_h).unsqueeze(1).unsqueeze(1)
                ego_x_exp = ego_x.unsqueeze(1).unsqueeze(1)
                ego_y_exp = ego_y.unsqueeze(1).unsqueeze(1)
                ego_h_exp = ego_h.unsqueeze(1).unsqueeze(1)

                Dm = cfg.D_MAP_POINT
                parts = []
                for offset in (0, Dm, 2 * Dm):  # center, left, right
                    poly = lanes_t[..., offset:offset+Dm]
                    px, py = poly[..., 0], poly[..., 1]
                    ph = torch.atan2(poly[..., 2], poly[..., 3])

                    dx_l = px - ego_x_exp
                    dy_l = py - ego_y_exp
                    x_el = cos_hl * dx_l - sin_hl * dy_l
                    y_el = sin_hl * dx_l + cos_hl * dy_l
                    h_el = (ph - ego_h_exp + math.pi) % (2 * math.pi) - math.pi

                    parts.append(torch.cat([
                        x_el.unsqueeze(-1), y_el.unsqueeze(-1),
                        torch.sin(h_el).unsqueeze(-1), torch.cos(h_el).unsqueeze(-1),
                        poly[..., 4:],
                    ], dim=-1))

                lanes_ego = torch.cat(parts, dim=-1)  # (B, N_L, P, D_POLYLINE_POINT)
                map_feats = self.lane_encoder(lanes_ego, map_lanes_mask)

            # ── Polygon re-transform per step ─────────────────────────────
            poly_feats = None
            poly_poses = None
            if map_polygons is not None and map_polygons_mask is not None:
                polys_t = map_polygons.clone()
                cos_hp = torch.cos(-ego_h).unsqueeze(1).unsqueeze(1)
                sin_hp = torch.sin(-ego_h).unsqueeze(1).unsqueeze(1)
                ego_xp = ego_x.unsqueeze(1).unsqueeze(1)
                ego_yp = ego_y.unsqueeze(1).unsqueeze(1)
                ego_hp = ego_h.unsqueeze(1).unsqueeze(1)

                ppx, ppy = polys_t[..., 0], polys_t[..., 1]
                pph = torch.atan2(polys_t[..., 2], polys_t[..., 3])
                dx_p = ppx - ego_xp
                dy_p = ppy - ego_yp
                x_ep = cos_hp * dx_p - sin_hp * dy_p
                y_ep = sin_hp * dx_p + cos_hp * dy_p
                h_ep = (pph - ego_hp + math.pi) % (2 * math.pi) - math.pi

                polys_ego = torch.cat([
                    x_ep.unsqueeze(-1), y_ep.unsqueeze(-1),
                    torch.sin(h_ep).unsqueeze(-1), torch.cos(h_ep).unsqueeze(-1),
                    polys_t[..., 4:],
                ], dim=-1)  # (B, N_POLYGONS, N_PTS, D_POLYGON_POINT)
                poly_feats = self.polygon_encoder(polys_ego, map_polygons_mask)
                poly_poses = polys_ego[..., :2].mean(dim=2)  # (B, N_POLYGONS, 2)

            # ── Route trim (K_r = N_r/4 forward points from closest) ────────
            route_feats_t = None
            if route_polylines is not None:
                rp_xy = route_polylines[..., :2]                # (B, N_lat, N_r, 2)
                dx_r = rp_xy[..., 0] - ego_x.unsqueeze(1).unsqueeze(1)
                dy_r = rp_xy[..., 1] - ego_y.unsqueeze(1).unsqueeze(1)
                cos_hr = torch.cos(-ego_h).unsqueeze(1).unsqueeze(1)
                sin_hr = torch.sin(-ego_h).unsqueeze(1).unsqueeze(1)
                rx = cos_hr * dx_r - sin_hr * dy_r
                ry = sin_hr * dx_r + cos_hr * dy_r
                route_dists = torch.sqrt(rx**2 + ry**2)
                closest_idx = route_dists.argmin(dim=-1)        # (B, N_lat)
                active_mask = (closest_idx > 0).float()         # discard if ego passed start

                N_r = route_polylines.size(2)
                K_r = max(1, N_r // 4)
                B_r, N_lat_r = closest_idx.shape
                trimmed = torch.zeros(B_r, N_lat_r, K_r, route_polylines.size(-1),
                                      device=device, dtype=route_polylines.dtype)
                for b in range(B_r):
                    for r in range(N_lat_r):
                        start = int(closest_idx[b, r].item())
                        end = min(start + K_r, N_r)
                        length = end - start
                        trimmed[b, r, :length] = route_polylines[b, r, start:end]

                route_feats_t = self.decomposed_mode.route_pointnet(
                    trimmed, route_mask
                ) * active_mask.unsqueeze(-1)                   # (B, N_lat, D)

            # ── IVM KNN filter → decoder K/V ─────────────────────────────
            map_poses = None
            if lanes_ego is not None:
                map_poses = lanes_ego[..., :2].mean(dim=2)

            kv = self.ivm(
                per_agent, agents_ego_now[..., :2],
                map_feats, map_poses=map_poses,
                route_feats=route_feats_t,
                polygon_feats=poly_feats, polygon_poses=poly_poses,
            )  # (B, K_total, D)

            for layer in self.decoder_layers:
                mode_query = layer(mode_query, kv)         # (B, 1, D)

            # ── Action head → waypoint in CURRENT ego frame ───────────────
            a_local = self.action_mean_head(mode_query.squeeze(1))  # (B, 3)

            # ── Convert back to initial ego frame ─────────────────────────
            cos_h_bk = torch.cos(ego_h)
            sin_h_bk = torch.sin(ego_h)
            x_g = cos_h_bk * a_local[:, 0] - sin_h_bk * a_local[:, 1] + ego_x
            y_g = sin_h_bk * a_local[:, 0] + cos_h_bk * a_local[:, 1] + ego_y
            h_g = a_local[:, 2] + ego_h
            h_g = (h_g + math.pi) % (2 * math.pi) - math.pi
            a_t = torch.stack([x_g, y_g, h_g], dim=-1)  # (B, 3) in initial frame
            actions.append(a_t.unsqueeze(1))

            # ── Update ego pose for next step ─────────────────────────────
            if gt_ego is not None and self.training:
                ego_x = gt_ego[:, t, 0]
                ego_y = gt_ego[:, t, 1]
                ego_h = gt_ego[:, t, 2]
            else:
                ego_x = a_t[:, 0].detach()
                ego_y = a_t[:, 1].detach()
                ego_h = a_t[:, 2].detach()

        return torch.cat(actions, dim=1)                 # (B, T_FUTURE, 3)

    def forward_rl(self, agents_now: torch.Tensor, agents_seq: torch.Tensor,
                   agents_mask: torch.Tensor, mode_c: torch.Tensor,
                   map_lanes: torch.Tensor = None,
                   map_lanes_mask: torch.Tensor = None,
                   stored_actions: torch.Tensor = None,
                   map_polygons: torch.Tensor = None,
                   map_polygons_mask: torch.Tensor = None,
                   route_polylines: torch.Tensor = None,
                   route_mask: torch.Tensor = None) -> tuple:
        """
        RL forward pass with Gaussian policy and value estimation.

        Two modes:
          - stored_actions=None:   Sample from N(mean, std)   [collect mode]
          - stored_actions given:  Compute log_prob of those actions [eval mode]

        Returns:
            trajectory: (B, T_FUTURE, 3) — predicted (x, y, yaw) in initial ego frame
            log_probs:  (B, T_FUTURE)   — log probability of each action
            values:     (B, T_FUTURE)   — value estimates at each step
            entropies:  (B, T_FUTURE)   — entropy of policy at each step
        """
        B = agents_seq.size(0)
        device = agents_seq.device
        H = cfg.T_HIST

        # Initialise mode query from decomposed mode tensor
        lon_modes = self.decomposed_mode.encode_longitudinal(device)
        if route_polylines is not None and route_mask is not None:
            lat_modes = self.decomposed_mode.encode_lateral(route_polylines, route_mask)
        else:
            lat_modes = torch.zeros(B, cfg.N_LAT, cfg.D_HIDDEN, device=device)
        mode_tensor = self.decomposed_mode.construct_mode_tensor(lon_modes, lat_modes)
        mode_query = self.decomposed_mode.get_mode_query(mode_c, mode_tensor)  # (B, 1, D)

        # Ego pose accumulator
        ego_x = torch.zeros(B, device=device)
        ego_y = torch.zeros(B, device=device)
        ego_h = torch.zeros(B, device=device)

        # Agent history buffer (same initialisation as forward())
        agent_buffer = torch.stack([agents_now] * H, dim=1)  # (B, H, N, Da)

        actions = []
        log_probs_list = []
        values_list = []
        entropies_list = []

        for t in range(cfg.T_FUTURE):
            # ── Get agent states at step t ────────────────────────────────
            agents_t = agents_seq[:, t, :, :]            # (B, N, 4)

            # ── Update history buffer ─────────────────────────────────────
            agent_buffer = torch.cat(
                [agent_buffer[:, 1:, :, :], agents_t.unsqueeze(1)], dim=1
            )

            # Time normalisation indices
            time_indices = torch.arange(-(H - 1), 1, dtype=torch.float, device=device)

            # ── IVM coordinate transform (same as forward()) ──────────────
            Da = cfg.D_AGENT
            buffer_ego = torch.zeros(B, H, cfg.N_AGENTS, Da, device=device,
                                     dtype=agent_buffer.dtype)
            for h_idx in range(H):
                ag_xy_h  = agent_buffer[:, h_idx, :, :2]
                ag_sin_h = agent_buffer[:, h_idx, :, 2]
                ag_cos_h = agent_buffer[:, h_idx, :, 3]
                ag_h_h   = torch.atan2(ag_sin_h, ag_cos_h)
                ag_vx_h  = agent_buffer[:, h_idx, :, 4]
                ag_vy_h  = agent_buffer[:, h_idx, :, 5]
                ag_bw_h  = agent_buffer[:, h_idx, :, 6]
                ag_bl_h  = agent_buffer[:, h_idx, :, 7]
                ag_bh_h  = agent_buffer[:, h_idx, :, 8]
                ag_cat_h = agent_buffer[:, h_idx, :, 10:14]

                x_e, y_e, h_e = self._transform_to_ego_frame(
                    ag_xy_h, ag_h_h, ego_x, ego_y, ego_h
                )
                cos_h = torch.cos(-ego_h).unsqueeze(1)
                sin_h = torch.sin(-ego_h).unsqueeze(1)
                vx_e = cos_h * ag_vx_h - sin_h * ag_vy_h
                vy_e = sin_h * ag_vx_h + cos_h * ag_vy_h
                t_norm = time_indices[h_idx].expand(B, cfg.N_AGENTS)
                buffer_ego[:, h_idx] = torch.cat([
                    x_e.unsqueeze(-1), y_e.unsqueeze(-1),
                    torch.sin(h_e).unsqueeze(-1), torch.cos(h_e).unsqueeze(-1),
                    vx_e.unsqueeze(-1), vy_e.unsqueeze(-1),
                    ag_bw_h.unsqueeze(-1), ag_bl_h.unsqueeze(-1), ag_bh_h.unsqueeze(-1),
                    t_norm.unsqueeze(-1),
                    ag_cat_h,
                ], dim=-1)

            # PointNet encode
            buffer_flat = buffer_ego.view(B, H * cfg.N_AGENTS, Da)
            mask_flat = agents_mask.unsqueeze(1).expand(-1, H, -1).reshape(
                B, H * cfg.N_AGENTS
            )
            per_agent, _ = self.agent_encoder_time(buffer_flat, mask_flat)

            # K-NN current step
            agents_ego_now = buffer_ego[:, -1]

            # Map re-transform per step (Alg 3 step 3)
            map_feats = None
            lanes_ego = None
            if map_lanes is not None:
                lanes_t = map_lanes.clone()
                cos_hl = torch.cos(-ego_h).unsqueeze(1).unsqueeze(1)
                sin_hl = torch.sin(-ego_h).unsqueeze(1).unsqueeze(1)
                ego_x_exp = ego_x.unsqueeze(1).unsqueeze(1)
                ego_y_exp = ego_y.unsqueeze(1).unsqueeze(1)
                ego_h_exp = ego_h.unsqueeze(1).unsqueeze(1)

                Dm = cfg.D_MAP_POINT
                parts = []
                for offset in (0, Dm, 2 * Dm):
                    poly = lanes_t[..., offset:offset+Dm]
                    px, py = poly[..., 0], poly[..., 1]
                    ph = torch.atan2(poly[..., 2], poly[..., 3])

                    dx_l = px - ego_x_exp
                    dy_l = py - ego_y_exp
                    x_el = cos_hl * dx_l - sin_hl * dy_l
                    y_el = sin_hl * dx_l + cos_hl * dy_l
                    h_el = (ph - ego_h_exp + math.pi) % (2 * math.pi) - math.pi

                    parts.append(torch.cat([
                        x_el.unsqueeze(-1), y_el.unsqueeze(-1),
                        torch.sin(h_el).unsqueeze(-1), torch.cos(h_el).unsqueeze(-1),
                        poly[..., 4:],
                    ], dim=-1))

                lanes_ego = torch.cat(parts, dim=-1)
                map_feats = self.lane_encoder(lanes_ego, map_lanes_mask)

            # Polygon re-transform per step
            poly_feats = None
            poly_poses = None
            if map_polygons is not None and map_polygons_mask is not None:
                polys_t = map_polygons.clone()
                cos_hp = torch.cos(-ego_h).unsqueeze(1).unsqueeze(1)
                sin_hp = torch.sin(-ego_h).unsqueeze(1).unsqueeze(1)
                ego_xp = ego_x.unsqueeze(1).unsqueeze(1)
                ego_yp = ego_y.unsqueeze(1).unsqueeze(1)
                ego_hp = ego_h.unsqueeze(1).unsqueeze(1)

                ppx, ppy = polys_t[..., 0], polys_t[..., 1]
                pph = torch.atan2(polys_t[..., 2], polys_t[..., 3])
                dx_p = ppx - ego_xp
                dy_p = ppy - ego_yp
                x_ep = cos_hp * dx_p - sin_hp * dy_p
                y_ep = sin_hp * dx_p + cos_hp * dy_p
                h_ep = (pph - ego_hp + math.pi) % (2 * math.pi) - math.pi

                polys_ego = torch.cat([
                    x_ep.unsqueeze(-1), y_ep.unsqueeze(-1),
                    torch.sin(h_ep).unsqueeze(-1), torch.cos(h_ep).unsqueeze(-1),
                    polys_t[..., 4:],
                ], dim=-1)
                poly_feats = self.polygon_encoder(polys_ego, map_polygons_mask)
                poly_poses = polys_ego[..., :2].mean(dim=2)

            # Route trim (K_r = N_r/4 forward points from closest)
            route_feats_t = None
            if route_polylines is not None:
                rp_xy = route_polylines[..., :2]
                dx_r = rp_xy[..., 0] - ego_x.unsqueeze(1).unsqueeze(1)
                dy_r = rp_xy[..., 1] - ego_y.unsqueeze(1).unsqueeze(1)
                cos_hr = torch.cos(-ego_h).unsqueeze(1).unsqueeze(1)
                sin_hr = torch.sin(-ego_h).unsqueeze(1).unsqueeze(1)
                rx = cos_hr * dx_r - sin_hr * dy_r
                ry = sin_hr * dx_r + cos_hr * dy_r
                closest_idx = torch.sqrt(rx**2 + ry**2).argmin(dim=-1)
                active_mask = (closest_idx > 0).float()

                N_r = route_polylines.size(2)
                K_r = max(1, N_r // 4)
                B_r, N_lat_r = closest_idx.shape
                trimmed = torch.zeros(B_r, N_lat_r, K_r, route_polylines.size(-1),
                                      device=device, dtype=route_polylines.dtype)
                for b in range(B_r):
                    for r in range(N_lat_r):
                        start = int(closest_idx[b, r].item())
                        end = min(start + K_r, N_r)
                        length = end - start
                        trimmed[b, r, :length] = route_polylines[b, r, start:end]

                route_feats_t = self.decomposed_mode.route_pointnet(
                    trimmed, route_mask
                ) * active_mask.unsqueeze(-1)

            map_poses = None
            if lanes_ego is not None:
                map_poses = lanes_ego[..., :2].mean(dim=2)

            kv = self.ivm(
                per_agent, agents_ego_now[..., :2],
                map_feats, map_poses=map_poses,
                route_feats=route_feats_t,
                polygon_feats=poly_feats, polygon_poses=poly_poses,
            )
            for layer in self.decoder_layers:
                mode_query = layer(mode_query, kv)

            # ── Gaussian policy head ──────────────────────────────────────
            action_mean = self.action_mean_head(mode_query.squeeze(1))  # (B, 3)
            action_std = self.action_log_std.exp().expand_as(action_mean)  # (B, 3)

            # Value estimation
            value = self.value_head(mode_query.squeeze(1)).squeeze(-1)  # (B,)

            dist = torch.distributions.Normal(action_mean, action_std)

            if stored_actions is None:
                # Collect mode: sample from Gaussian
                a_local = dist.sample()                             # (B, 3)
                log_prob = dist.log_prob(a_local).sum(dim=-1)       # (B,)
                entropy = dist.entropy().sum(dim=-1)                # (B,)
            else:
                # Eval mode: compute log_prob of stored actions
                a_global = stored_actions[:, t, :]                  # (B, 3)
                # Transform from initial ego frame to current local frame
                cos_h_bk = torch.cos(ego_h)
                sin_h_bk = torch.sin(ego_h)
                dx = a_global[:, 0] - ego_x
                dy = a_global[:, 1] - ego_y
                a_local_x =  cos_h_bk * dx + sin_h_bk * dy
                a_local_y = -sin_h_bk * dx + cos_h_bk * dy
                a_local_h = a_global[:, 2] - ego_h
                a_local = torch.stack([a_local_x, a_local_y, a_local_h], dim=-1)

                log_prob = dist.log_prob(a_local).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)

            # Convert local action back to initial ego frame
            cos_h_bk = torch.cos(ego_h)
            sin_h_bk = torch.sin(ego_h)
            x_g = cos_h_bk * a_local[:, 0] - sin_h_bk * a_local[:, 1] + ego_x
            y_g = sin_h_bk * a_local[:, 0] + cos_h_bk * a_local[:, 1] + ego_y
            h_g = a_local[:, 2] + ego_h
            h_g = (h_g + math.pi) % (2 * math.pi) - math.pi
            a_t = torch.stack([x_g, y_g, h_g], dim=-1)

            actions.append(a_t.unsqueeze(1))
            log_probs_list.append(log_prob.unsqueeze(1))
            values_list.append(value.unsqueeze(1))
            entropies_list.append(entropy.unsqueeze(1))

            # Update ego pose (always use predicted action, no teacher-forcing)
            ego_x = a_t[:, 0].detach()
            ego_y = a_t[:, 1].detach()
            ego_h = a_t[:, 2].detach()

        trajectory = torch.cat(actions, dim=1)           # (B, T, 3)
        log_probs  = torch.cat(log_probs_list, dim=1)    # (B, T)
        values     = torch.cat(values_list, dim=1)       # (B, T)
        entropies  = torch.cat(entropies_list, dim=1)    # (B, T)

        return trajectory, log_probs, values, entropies


# ─────────────────────────────────────────────────────────────────────────────
# Non-reactive Transition Model  (Eq 5, Stage A)
# ─────────────────────────────────────────────────────────────────────────────

class TransitionModel(nn.Module):
    """
    Non-reactive transition model P_τ(s^{1:N}_{1:T} | s_0) (Section 3.1).

    Architecture (architecture-2.md §3.1):
      1. Agent encoder — PointNet over N × H × Da history:
           each agent's H historical poses treated as a point cloud → N × D
      2. Map encoder — PointNet(polylines, D_POLYLINE_POINT=27) → N_m × D
      3. Self-attention Transformer — fuses (N + N_m) × D agent+map features
      4. Per-agent decoder — MLP per agent: [per-agent; global] → T × Da

    Inputs:
      agents_history: (B, H, N, Da) — H historical poses per agent in ego frame
      agents_mask:    (B, N)
      map_lanes:      (B, N_LANES, N_PTS, 27) — lane polylines (optional)
      map_lanes_mask: (B, N_LANES) — validity mask (optional)
    Output:
      agent_futures: (B, T_FUTURE, N, Da)  — predicted agent states in ego frame
    """
    def __init__(self):
        super().__init__()
        D = cfg.D_HIDDEN

        # Feature normalization constants (registered buffer for device tracking)
        self.register_buffer(
            'feat_std',
            torch.tensor(cfg.AGENT_FEATURE_STD, dtype=torch.float32).reshape(1, 1, 1, -1)
        )  # (1, 1, 1, Da) — broadcastable to (B, H, N, Da)

        # 1. Agent encoder: shared MLP over Da-dim poses, max-pool over H per agent
        self.agent_encoder = PointNetEncoder(in_dim=cfg.D_AGENT, d_model=D)

        # 2. Map encoder — PointNet over lane polylines — in_dim=D_POLYLINE_POINT (3×Dm)
        self.map_encoder = LaneEncoder(in_dim=cfg.D_POLYLINE_POINT, d_model=cfg.D_LANE)
        self.polygon_encoder = PolygonEncoder()

        # 3. Self-attention Transformer encoder (fuses agents + map + polygons)
        # Paper Table 5: 3 layers, 8 attention heads, dropout 0.1
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=8, dim_feedforward=D * 4,
            dropout=0.1, activation='relu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layer, num_layers=3
        )

        # 4. Per-agent decoder: [per-agent feat; global feat] → T × D_a (DELTA from s_0)
        self.per_agent_head = nn.Sequential(
            nn.Linear(D + D, D), nn.ReLU(inplace=True),
            nn.Linear(D, cfg.T_FUTURE * cfg.D_AGENT),
        )

    def forward(self, agents_history: torch.Tensor,
                agents_mask: torch.Tensor,
                map_lanes: torch.Tensor = None,
                map_lanes_mask: torch.Tensor = None,
                map_polygons: torch.Tensor = None,
                map_polygons_mask: torch.Tensor = None) -> torch.Tensor:
        # 1. Per-agent PointNet over H historical poses (Section 3.1)
        # agents_history: (B, H, N, Da) — H poses per agent
        B, H, N, Da = agents_history.shape

        # Normalize features to unit scale for stable gradients
        hist_norm = agents_history / self.feat_std            # (B, H, N, Da)

        # Extract s_0 = current-frame state (last timestep in history, normalized)
        s_0_norm = hist_norm[:, -1, :, :]                   # (B, N, Da)

        # Reshape to (B*N, H, Da) to apply shared MLP per-point over H
        hist = hist_norm.permute(0, 2, 1, 3).reshape(B * N, H, Da)
        per_point = self.agent_encoder.mlp(hist)             # (B*N, H, D)
        per_agent = per_point.max(dim=1).values              # (B*N, D) — pool over H
        per_agent = per_agent.reshape(B, N, -1)              # (B, N, D)
        D = per_agent.size(-1)

        # Zero out padding agents
        per_agent = per_agent * agents_mask.unsqueeze(-1)

        # 2. Encode map (if available)
        if map_lanes is not None and map_lanes_mask is not None:
            map_feats = self.map_encoder(map_lanes, map_lanes_mask)  # (B, N_L, D)
        else:
            map_feats = torch.zeros(B, 0, D, device=agents_history.device)

        # Encode polygons (if available)
        if map_polygons is not None and map_polygons_mask is not None:
            poly_feats = self.polygon_encoder(map_polygons, map_polygons_mask)
        else:
            poly_feats = torch.zeros(B, 0, D, device=agents_history.device)

        # 3. Self-attention Transformer over agents + map + polygons
        combined = torch.cat([per_agent, map_feats, poly_feats], dim=1)

        # Build padding mask
        N_m = map_feats.size(1)
        N_p = poly_feats.size(1)
        mask_parts = [agents_mask]
        if N_m > 0 and map_lanes_mask is not None:
            mask_parts.append(map_lanes_mask)
        elif N_m > 0:
            mask_parts.append(torch.zeros(B, N_m, device=agents_history.device))
        if N_p > 0 and map_polygons_mask is not None:
            mask_parts.append(map_polygons_mask)
        elif N_p > 0:
            mask_parts.append(torch.zeros(B, N_p, device=agents_history.device))
        combined_mask = torch.cat(mask_parts, dim=1)
        padding_mask = (combined_mask < 0.5)

        fused = self.transformer(combined, src_key_padding_mask=padding_mask)

        # Extract per-agent features (first N positions)
        per_agent_fused = fused[:, :N, :]                   # (B, N, D)

        # Global feature: mean over valid agents from fused representation
        valid_sum = per_agent_fused * agents_mask.unsqueeze(-1)
        valid_count = agents_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        global_fused = valid_sum.sum(dim=1) / valid_count    # (B, D)

        # 4. Per-agent decoder: predict delta from s_0 (in normalized space)
        g_exp = global_fused.unsqueeze(1).expand(-1, N, -1)  # (B, N, D)
        delta_norm = self.per_agent_head(
            torch.cat([per_agent_fused, g_exp], dim=-1)
        )                                                    # (B, N, T*Da)
        delta_norm = delta_norm.view(B, N, cfg.T_FUTURE, cfg.D_AGENT)
        # Residual in normalized space: prediction = current state + learned delta
        pred_norm = s_0_norm.unsqueeze(2) + delta_norm       # (B, N, T, Da) normalized
        # Denormalize: (B, N, T, Da) * (1, 1, 1, Da) → raw space
        pred = pred_norm * self.feat_std
        pred = pred.permute(0, 2, 1, 3)                      # (B, T, N, Da)
        return pred


# ─────────────────────────────────────────────────────────────────────────────
# Rule Selector  (Algorithm 2, step 5; inference only)
# ─────────────────────────────────────────────────────────────────────────────

class RuleSelector(nn.Module):
    """
    Inference-time rule-based trajectory selection (Algorithm 2, step 5).
    Scores K=60 candidate trajectories on 4 criteria:
      1. Collision: penalise proximity to predicted agent positions
      2. Drivable area: penalise trajectories far from lane centerlines
      3. Comfort: penalise jerk (3rd derivative of position via finite differences)
      4. Progress: reward forward distance from start

    Combines with mode selector scores (weighted sum, Algorithm 2 step 5).
    Returns index of best trajectory.

    All operations are pure Python / torch, no learned parameters.
    """

    def __init__(self, w_mode: float = 1.0, w_comfort: float = 0.1,
                 w_progress: float = 0.5, w_collision: float = 1.0,
                 w_drivable: float = 0.3, collision_radius: float = 2.0,
                 lane_max_dist: float = 3.0):
        super().__init__()
        self.w_mode      = w_mode
        self.w_comfort    = w_comfort
        self.w_progress   = w_progress
        self.w_collision  = w_collision
        self.w_drivable   = w_drivable
        self.collision_radius = collision_radius
        self.lane_max_dist = lane_max_dist

    @torch.no_grad()
    def forward(self, mode_logits: torch.Tensor,
                all_trajs: torch.Tensor,
                agents_now: torch.Tensor = None,
                agents_mask: torch.Tensor = None,
                map_lanes: torch.Tensor = None,
                map_lanes_mask: torch.Tensor = None) -> tuple:
        """
        mode_logits: (B, N_MODES)
        all_trajs:   (B, N_MODES, T, 3)
        agents_now:  (B, N, 4)   — current agent positions for collision check
        agents_mask: (B, N)      — agent validity mask
        map_lanes:   (B, N_LANES, N_PTS, 3) — lane centerlines for drivable area
        map_lanes_mask: (B, N_LANES) — lane validity mask
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

        # Collision: penalise trajectories that come within collision_radius of agents
        collision = torch.zeros(B, M, device=device)
        if agents_now is not None:
            agent_xy = agents_now[:, :, :2]                  # (B, N, 2)
            agent_valid = agents_mask.unsqueeze(1)           # (B, 1, N)
            # For each trajectory point, compute min distance to any valid agent
            ego_xy = all_trajs[..., :2]                      # (B, M, T, 2)
            # (B, M, T, 1, 2) - (B, 1, 1, N, 2) → (B, M, T, N)
            dist_to_agents = (ego_xy.unsqueeze(3) - agent_xy.unsqueeze(1).unsqueeze(1)).norm(dim=-1)
            # Mask invalid agents with large distance
            dist_to_agents = dist_to_agents + (1 - agent_valid.unsqueeze(2)) * 1e6
            min_dist, _ = dist_to_agents.min(dim=-1)         # (B, M, T) — closest agent at each step
            # Penalise steps where min_dist < collision_radius
            violation = (min_dist < self.collision_radius).float()
            collision = -violation.mean(dim=-1)               # (B, M) — more violations → lower score

        # Drivable area: penalise trajectory points far from any lane centerline
        drivable = torch.zeros(B, M, device=device)
        if map_lanes is not None:
            # lane center points: (B, N_LANES, N_PTS, 2)
            lane_pts = map_lanes[:, :, :, :2]                # (B, N_LANES, N_PTS, 2)
            lane_valid = map_lanes_mask                       # (B, N_LANES)
            # Reshape for distance computation
            lane_flat = lane_pts.reshape(B, -1, 2)           # (B, N_LANES*N_PTS, 2)
            lane_valid_flat = lane_valid.unsqueeze(2).expand(-1, -1, cfg.N_LANE_POINTS).reshape(B, -1)
            # (B, M, T, 1, 2) - (B, 1, 1, N_L*N_P, 2) → (B, M, T, N_L*N_P)
            dist_to_lanes = (ego_xy.unsqueeze(3) - lane_flat.unsqueeze(1).unsqueeze(1)).norm(dim=-1)
            # Mask invalid lane points
            dist_to_lanes = dist_to_lanes + (1 - lane_valid_flat.unsqueeze(1).unsqueeze(2)) * 1e6
            min_lane_dist, _ = dist_to_lanes.min(dim=-1)     # (B, M, T)
            # Penalise points farther than lane_max_dist from any lane
            off_road = (min_lane_dist > self.lane_max_dist).float()
            drivable = -off_road.mean(dim=-1)                 # (B, M)

        # Combined score (Algorithm 2: "combine with mode selector scores")
        score = (self.w_mode * mode_scores
                 + self.w_comfort * comfort
                 + self.w_progress * progress
                 + self.w_collision * collision
                 + self.w_drivable * drivable)                # (B, M)

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
        self.s0_encoder = PointNetEncoder(in_dim=cfg.D_AGENT, d_model=cfg.D_HIDDEN)
        # If NOT backbone sharing, policy has its own per-step encoder inside AutoregressivePolicy
        self.mode_selector     = ModeSelector()
        self.policy            = AutoregressivePolicy()
        self.transition_model  = TransitionModel()
        self.rule_selector     = RuleSelector()
        self._transition_loaded = False

        # Backbone sharing (Table 4): share s0 encoder with policy's per-step encoder
        # ON for IL best, OFF for RL best — controlled by cfg.BACKBONE_SHARING
        if cfg.BACKBONE_SHARING:
            self.policy.agent_encoder_time = self.s0_encoder

    # ── Training forward (IL Stage B) ────────────────────────────────────────

    def forward_train(self, agents_now: torch.Tensor, agents_mask: torch.Tensor,
                      agents_seq: torch.Tensor, gt_traj: torch.Tensor,
                      mode_label: torch.Tensor,
                      map_lanes: torch.Tensor = None,
                      map_lanes_mask: torch.Tensor = None,
                      agents_history: torch.Tensor = None,
                      map_polygons: torch.Tensor = None,
                      map_polygons_mask: torch.Tensor = None,
                      route_polylines: torch.Tensor = None,
                      route_mask: torch.Tensor = None):
        """
        IL training pass (Algorithm 1, Stage 2 IL branch).
        """
        if self._transition_loaded:
            with torch.no_grad():
                hist = agents_history if agents_history is not None else agents_now.unsqueeze(1)
                agents_seq = self.transition_model(
                    hist, agents_mask, map_lanes, map_lanes_mask,
                    map_polygons=map_polygons, map_polygons_mask=map_polygons_mask,
                )
        s0_per_agent, s0_global = self.s0_encoder(agents_now, agents_mask)

        mode_logits, side_traj = self.mode_selector(
            s0_global,
            s0_per_agent=s0_per_agent,
            map_lanes=map_lanes,
            map_lanes_mask=map_lanes_mask,
            mode_c=mode_label,
            map_polygons=map_polygons,
            map_polygons_mask=map_polygons_mask,
            route_polylines=route_polylines,
            route_mask=route_mask,
        )

        c = mode_label.clone()
        if cfg.MODE_DROPOUT and self.training:
            drop_mask = torch.rand(c.size(0), device=c.device) < cfg.MODE_DROPOUT_P
            c[drop_mask] = torch.randint(
                0, cfg.N_MODES, (int(drop_mask.sum()),), device=c.device
            )

        agents_now_policy = agents_now
        if cfg.EGO_HISTORY_DROPOUT and self.training:
            drop_mask_b = (torch.rand(agents_now.size(0), 1, 1, device=agents_now.device)
                           > cfg.MODE_DROPOUT_P).float()
            agents_now_policy = agents_now * drop_mask_b

        pred_traj = self.policy(
            agents_now=agents_now_policy,
            agents_seq=agents_seq,
            agents_mask=agents_mask,
            mode_c=c,
            gt_ego=gt_traj,
            map_lanes=map_lanes,
            map_lanes_mask=map_lanes_mask,
            map_polygons=map_polygons,
            map_polygons_mask=map_polygons_mask,
            route_polylines=route_polylines,
            route_mask=route_mask,
        )

        return mode_logits, side_traj, pred_traj

    # ── RL training forward (Stage C) ──────────────────────────────────────────

    def forward_rl_train(self, agents_now: torch.Tensor, agents_mask: torch.Tensor,
                         agents_seq: torch.Tensor, gt_traj: torch.Tensor,
                         mode_label: torch.Tensor,
                         map_lanes: torch.Tensor = None,
                         map_lanes_mask: torch.Tensor = None,
                         stored_actions: torch.Tensor = None,
                         agents_history: torch.Tensor = None,
                         map_polygons: torch.Tensor = None,
                         map_polygons_mask: torch.Tensor = None,
                         route_polylines: torch.Tensor = None,
                         route_mask: torch.Tensor = None) -> tuple:
        """
        RL training forward pass (Stage C).
        """
        s0_per_agent, s0_global = self.s0_encoder(agents_now, agents_mask)

        mode_logits, side_traj = self.mode_selector(
            s0_global,
            s0_per_agent=s0_per_agent,
            map_lanes=map_lanes,
            map_lanes_mask=map_lanes_mask,
            mode_c=mode_label,
            map_polygons=map_polygons,
            map_polygons_mask=map_polygons_mask,
            route_polylines=route_polylines,
            route_mask=route_mask,
        )

        c = mode_label.clone()
        if cfg.MODE_DROPOUT and self.training:
            drop_mask = torch.rand(c.size(0), device=c.device) < cfg.MODE_DROPOUT_P
            c[drop_mask] = torch.randint(
                0, cfg.N_MODES, (int(drop_mask.sum()),), device=c.device
            )

        trajectory, log_probs, values, entropies = self.policy.forward_rl(
            agents_now=agents_now,
            agents_seq=agents_seq,
            agents_mask=agents_mask,
            mode_c=c,
            map_lanes=map_lanes,
            map_lanes_mask=map_lanes_mask,
            stored_actions=stored_actions,
            map_polygons=map_polygons,
            map_polygons_mask=map_polygons_mask,
            route_polylines=route_polylines,
            route_mask=route_mask,
        )

        return mode_logits, side_traj, trajectory, log_probs, values, entropies

    # ── Inference forward ─────────────────────────────────────────────────────

    @torch.no_grad()
    def forward_inference(self, agents_now: torch.Tensor, agents_mask: torch.Tensor,
                          map_lanes: torch.Tensor = None,
                          map_lanes_mask: torch.Tensor = None,
                          agents_history: torch.Tensor = None,
                          map_polygons: torch.Tensor = None,
                          map_polygons_mask: torch.Tensor = None,
                          route_polylines: torch.Tensor = None,
                          route_mask: torch.Tensor = None):
        """
        Full inference pass (Algorithm 2).
        """
        self.eval()
        B = agents_now.size(0)

        s0_per_agent, s0_global = self.s0_encoder(agents_now, agents_mask)

        mode_logits, _ = self.mode_selector(
            s0_global,
            s0_per_agent=s0_per_agent,
            map_lanes=map_lanes,
            map_lanes_mask=map_lanes_mask,
            map_polygons=map_polygons,
            map_polygons_mask=map_polygons_mask,
            route_polylines=route_polylines,
            route_mask=route_mask,
        )

        hist = agents_history if agents_history is not None else agents_now.unsqueeze(1)
        agent_futures = self.transition_model(
            hist, agents_mask, map_lanes, map_lanes_mask,
            map_polygons=map_polygons, map_polygons_mask=map_polygons_mask,
        )

        all_trajs = []
        for m in range(cfg.N_MODES):
            c = torch.full((B,), m, dtype=torch.long, device=agents_now.device)
            traj = self.policy(
                agents_now=agents_now,
                agents_seq=agent_futures,
                agents_mask=agents_mask,
                mode_c=c,
                gt_ego=None,
                map_lanes=map_lanes,
                map_lanes_mask=map_lanes_mask,
                map_polygons=map_polygons,
                map_polygons_mask=map_polygons_mask,
                route_polylines=route_polylines,
                route_mask=route_mask,
            )
            all_trajs.append(traj.unsqueeze(1))

        all_trajs = torch.cat(all_trajs, dim=1)

        best_traj, best_idx = self.rule_selector(
            mode_logits, all_trajs,
            agents_now=agents_now, agents_mask=agents_mask,
            map_lanes=map_lanes, map_lanes_mask=map_lanes_mask,
        )

        return mode_logits, all_trajs, best_traj, best_idx

    def forward_inference_fast(self, agents_now: torch.Tensor, agents_mask: torch.Tensor,
                               map_lanes: torch.Tensor = None,
                               map_lanes_mask: torch.Tensor = None,
                               agents_history: torch.Tensor = None,
                               map_polygons: torch.Tensor = None,
                               map_polygons_mask: torch.Tensor = None,
                               route_polylines: torch.Tensor = None,
                               route_mask: torch.Tensor = None):
        """
        Batched inference: tiles all N_MODES into one policy forward pass
        instead of a Python for-loop. ~60x faster than forward_inference.
        """
        self.eval()
        B = agents_now.size(0)
        M = cfg.N_MODES
        device = agents_now.device

        s0_per_agent, s0_global = self.s0_encoder(agents_now, agents_mask)
        mode_logits, _ = self.mode_selector(
            s0_global, s0_per_agent=s0_per_agent,
            map_lanes=map_lanes, map_lanes_mask=map_lanes_mask,
            map_polygons=map_polygons, map_polygons_mask=map_polygons_mask,
            route_polylines=route_polylines, route_mask=route_mask,
        )

        hist = agents_history if agents_history is not None else agents_now.unsqueeze(1)
        agent_futures = self.transition_model(
            hist, agents_mask, map_lanes, map_lanes_mask,
            map_polygons=map_polygons, map_polygons_mask=map_polygons_mask,
        )

        # Tile each input (B, ...) → (B*M, ...) so all modes run in one call
        def tile(t):
            return t.repeat_interleave(M, dim=0) if t is not None else None

        # mode_c: [0,1,...,M-1, 0,1,...,M-1, ...] length B*M
        mode_c = torch.arange(M, device=device).repeat(B)

        all_trajs_flat = self.policy(
            agents_now=tile(agents_now),
            agents_seq=tile(agent_futures),
            agents_mask=tile(agents_mask),
            mode_c=mode_c,
            gt_ego=None,
            map_lanes=tile(map_lanes),
            map_lanes_mask=tile(map_lanes_mask),
            map_polygons=tile(map_polygons),
            map_polygons_mask=tile(map_polygons_mask),
            route_polylines=tile(route_polylines),
            route_mask=tile(route_mask),
        )
        # (B*M, T, D) → (B, M, T, D)
        T, D = all_trajs_flat.size(1), all_trajs_flat.size(2)
        all_trajs = all_trajs_flat.reshape(B, M, T, D)

        best_traj, best_idx = self.rule_selector(
            mode_logits, all_trajs,
            agents_now=agents_now, agents_mask=agents_mask,
            map_lanes=map_lanes, map_lanes_mask=map_lanes_mask,
        )
        return mode_logits, all_trajs, best_traj, best_idx

    # ── Transition model pre-training pass (Stage A, Eq 5) ───────────────────

    def forward_transition(self, agents_history: torch.Tensor, agents_mask: torch.Tensor,
                           map_lanes: torch.Tensor = None,
                           map_lanes_mask: torch.Tensor = None,
                           map_polygons: torch.Tensor = None,
                           map_polygons_mask: torch.Tensor = None):
        """
        Forward pass for transition model pre-training.
        agents_history: (B, H, N, Da) — H historical poses per agent
        Returns predicted agent futures: (B, T_FUTURE, N, Da)
        """
        return self.transition_model(
            agents_history, agents_mask, map_lanes, map_lanes_mask,
            map_polygons=map_polygons, map_polygons_mask=map_polygons_mask,
        )

    def load_transition_model(self, ckpt_path: str, freeze: bool = True):
        """
        Load pre-trained transition model from Stage A checkpoint.

        Args:
            ckpt_path: Path to stage_a_*.pt checkpoint
            freeze: If True, set requires_grad=False and eval mode
                    (paper: frozen during stages 2-3)
        """
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.transition_model.load_state_dict(ckpt['model'])
        self._transition_loaded = True
        if freeze:
            for p in self.transition_model.parameters():
                p.requires_grad = False
            self.transition_model.eval()


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
    agents_now  = torch.randn(B, N, cfg.D_AGENT, device=device)
    agents_mask = torch.ones(B, N, device=device)
    agents_seq  = torch.randn(B, T, N, cfg.D_AGENT, device=device)
    gt_traj     = torch.randn(B, T, 3, device=device)
    mode_label  = torch.randint(0, cfg.N_MODES, (B,), device=device)
    map_lanes   = torch.randn(B, cfg.N_LANES, cfg.N_LANE_POINTS, cfg.D_POLYLINE_POINT, device=device)
    map_lanes_mask = torch.ones(B, cfg.N_LANES, device=device)
    map_polygons = torch.randn(B, cfg.N_POLYGONS, cfg.N_LANE_POINTS, cfg.D_POLYGON_POINT, device=device)
    map_polygons_mask = torch.ones(B, cfg.N_POLYGONS, device=device)

    # ── Training forward ───────────────────────────────────────────────────
    model.train()
    logits, side, pred = model.forward_train(
        agents_now, agents_mask, agents_seq, gt_traj, mode_label,
        map_lanes=map_lanes, map_lanes_mask=map_lanes_mask,
        map_polygons=map_polygons, map_polygons_mask=map_polygons_mask,
    )
    assert logits.shape == (B, cfg.N_MODES),     f"logits {logits.shape}"
    assert side.shape   == (B, T, 3),            f"side {side.shape}"
    assert pred.shape   == (B, T, 3),            f"pred {pred.shape}"
    print(f"\n✓ forward_train shapes:  logits={tuple(logits.shape)}  "
          f"side={tuple(side.shape)}  pred={tuple(pred.shape)}")

    # ── Backward ───────────────────────────────────────────────────────────
    L = (F.cross_entropy(logits, mode_label)
         + (side - gt_traj).abs().sum(dim=-1).mean()
         + (pred - gt_traj).abs().sum(dim=-1).mean())
    L.backward()
    print(f"✓ backward OK  L={L.item():.4f}")

    # ── Inference forward ──────────────────────────────────────────────────
    inf_logits, all_trajs, best_traj, best_idx = model.forward_inference(
        agents_now, agents_mask,
        map_lanes=map_lanes, map_lanes_mask=map_lanes_mask,
        map_polygons=map_polygons, map_polygons_mask=map_polygons_mask,
    )
    assert inf_logits.shape == (B, cfg.N_MODES),             f"inf_logits"
    assert all_trajs.shape  == (B, cfg.N_MODES, T, 3),       f"all_trajs"
    assert best_traj.shape  == (B, T, 3),                    f"best_traj"
    assert best_idx.shape   == (B,),                         f"best_idx"
    print(f"✓ forward_inference:     all_trajs={tuple(all_trajs.shape)}  "
          f"best_traj={tuple(best_traj.shape)}")

    # ── Transition model forward ───────────────────────────────────────────
    agents_history = torch.randn(B, cfg.T_HIST, N, cfg.D_AGENT, device=device)
    agent_fut = model.forward_transition(
        agents_history, agents_mask,
        map_lanes=map_lanes, map_lanes_mask=map_lanes_mask,
        map_polygons=map_polygons, map_polygons_mask=map_polygons_mask,
    )
    assert agent_fut.shape == (B, T, N, cfg.D_AGENT),        f"agent_fut {agent_fut.shape}"
    print(f"✓ transition model:      agent_fut={tuple(agent_fut.shape)}")

    print("\n✓ All shape tests passed.")
