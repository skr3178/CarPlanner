"""
Reward functions for Stage C RL fine-tuning.

3-component per-step reward (paper §3.4):
  R_displacement: -L2(ego, gt)           — proximity to ground truth
  R_collision:    -indicator(d < 2m)     — collision penalty
  R_drivable:     -indicator(d > 3m)     — off-road penalty

Plus GAE computation and advantage normalization for PPO.
"""

import torch
import torch.nn.functional as F

import config as cfg


def compute_rewards(ego_traj: torch.Tensor,
                    gt_traj: torch.Tensor,
                    agent_futures: torch.Tensor,
                    agents_mask: torch.Tensor,
                    map_lanes: torch.Tensor = None,
                    map_lanes_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Compute per-step rewards for RL training.

    Args:
        ego_traj:       (B, T, 3)      — policy rollout (x, y, yaw)
        gt_traj:        (B, T, 3)      — GT ego trajectory
        agent_futures:  (B, T, N, Da)  — predicted agent states from frozen β
        agents_mask:    (B, N)         — agent validity mask
        map_lanes:      (B, N_LANES, N_PTS, 9) — lane centerlines (optional)
        map_lanes_mask: (B, N_LANES)   — lane validity mask (optional)

    Returns:
        rewards: (B, T) — per-step reward scalar
    """
    # Component 1: Displacement reward (negative L2 distance to GT)
    R_displacement = -torch.norm(
        ego_traj[..., :2] - gt_traj[..., :2], dim=-1
    )                                                      # (B, T)

    # Component 2: Collision reward (penalty for agent proximity < 2m)
    ego_xy = ego_traj[..., :2].unsqueeze(2)                # (B, T, 1, 2)
    agent_xy = agent_futures[..., :2]                      # (B, T, N, 2)
    dist_to_agents = torch.norm(ego_xy - agent_xy, dim=-1) # (B, T, N)

    # Mask invalid agents with large distance
    mask_exp = agents_mask.unsqueeze(1)                    # (B, 1, N)
    dist_to_agents = dist_to_agents + (1 - mask_exp) * 1e6
    min_agent_dist, _ = dist_to_agents.min(dim=-1)         # (B, T)

    R_collision = -(min_agent_dist < 2.0).float()          # (B, T)

    # Component 3: Drivable area reward
    R_drivable = torch.zeros_like(R_displacement)
    if map_lanes is not None:
        R_drivable = _compute_drivable_penalty(
            ego_traj[..., :2], map_lanes, map_lanes_mask
        )

    return R_displacement + R_collision + R_drivable


def _compute_drivable_penalty(ego_xy: torch.Tensor,
                              map_lanes: torch.Tensor,
                              map_lanes_mask: torch.Tensor) -> torch.Tensor:
    """
    Penalty for trajectory points far from lane centerlines (> 3m).

    Args:
        ego_xy:         (B, T, 2)
        map_lanes:      (B, N_LANES, N_PTS, 9)
        map_lanes_mask: (B, N_LANES)

    Returns:
        penalty: (B, T) — -1 if off-road, 0 otherwise
    """
    B = ego_xy.size(0)

    lane_pts = map_lanes[:, :, :, :2]                      # (B, N_L, N_P, 2)
    lane_flat = lane_pts.reshape(B, -1, 2)                 # (B, N_L*N_P, 2)
    lane_valid_flat = map_lanes_mask.unsqueeze(2).expand(
        -1, -1, cfg.N_LANE_POINTS
    ).reshape(B, -1)                                       # (B, N_L*N_P)

    # (B, T, 1, 2) - (B, 1, N_L*N_P, 2) → (B, T, N_L*N_P)
    dist_to_lanes = torch.norm(
        ego_xy.unsqueeze(2) - lane_flat.unsqueeze(1), dim=-1
    )
    dist_to_lanes = dist_to_lanes + (1 - lane_valid_flat.unsqueeze(1)) * 1e6
    min_lane_dist, _ = dist_to_lanes.min(dim=-1)           # (B, T)

    return -(min_lane_dist > 3.0).float()


def compute_gae(rewards: torch.Tensor,
                values: torch.Tensor,
                gamma: float = cfg.GAMMA,
                lambda_gae: float = cfg.GAE_LAMBDA) -> tuple:
    """
    Generalized Advantage Estimation (GAE).

    Args:
        rewards:     (B, T) — per-step rewards
        values:      (B, T) — value estimates
        gamma:       discount factor
        lambda_gae:  GAE parameter

    Returns:
        advantages: (B, T)
        returns:    (B, T)
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    last_adv = torch.zeros(B, device=rewards.device)
    last_ret = torch.zeros(B, device=rewards.device)

    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t < T - 1 else torch.zeros(B, device=rewards.device)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_adv = delta + gamma * lambda_gae * last_adv
        last_ret = rewards[:, t] + gamma * last_ret
        advantages[:, t] = last_adv
        returns[:, t] = last_ret

    return advantages, returns


def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """Normalize advantages to zero mean, unit std."""
    std = advantages.std()
    if std > 1e-8:
        return (advantages - advantages.mean()) / (std + 1e-8)
    return advantages
