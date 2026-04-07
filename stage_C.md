# Stage C — RL Fine-tuning Plan

## What's new vs Stage B

Stage B is supervised (GT data does all the work). Stage C adds:
1. The policy must **explore** — output a probability distribution, not just a point prediction
2. A **reward signal** replaces GT supervision for the generator
3. **PPO** — importance-weighted policy gradient with clipping
4. **β generates agent futures** — no more GT oracle for agents

---

## New components needed

### 1. `model.py` — 3 additions to `AutoregressivePolicy`

**Gaussian policy head** (replaces deterministic `action_head`):
```python
# Replace:
self.action_head = nn.Sequential(Linear(D,D), ReLU, Linear(D,3))

# With:
self.action_mean_head = nn.Sequential(Linear(D,D), ReLU, Linear(D,3))
self.action_log_std   = nn.Parameter(torch.zeros(3))  # learnable per-dim std
```

**Value head** (new):
```python
self.value_head = nn.Sequential(Linear(D, D//2), ReLU, Linear(D//2, 1))
```

**New `forward_rl()` method** — handles both collect and eval in one pass:
```python
def forward_rl(self, ..., stored_actions=None):
    # stored_actions=None  → sample from Gaussian (collect mode)
    # stored_actions given → compute log_prob of those actions (eval mode)
    # Returns: (trajectory, log_probs (B,T), values (B,T), entropies (B,T))
```

---

### 2. `rewards.py` — 4-component reward (new file)

```python
def compute_rewards(ego_traj, gt_traj, agent_futures, agents_mask, map_lanes, map_lanes_mask):
    # ego_traj:      (B, T, 3) — policy rollout
    # gt_traj:       (B, T, 3) — GT ego
    # agent_futures: (B, T, N, Da) — from frozen β
    # Returns: R (B, T) — per-step reward scalar

    R_displacement = -L2(ego_traj, gt_traj)           # per step
    R_collision    = -indicator(min_agent_dist < 2m)   # per step
    R_drivable     = -indicator(min_lane_dist > 3m)    # per step
    R_comfort      = -jerk_magnitude(ego_traj)         # per step
    return R_displacement + R_collision + R_drivable + R_comfort
```

---

### 3. `train_stage_c.py` — RL training loop (new file)

```
startup:
  load stage_a_best.pt → freeze β
  load stage_b best.pt → warm-start π, f_selector
  π_old ← deepcopy(π)

per batch:
  # Step 1: simulate agents
  agent_futures ← β(agents_now)          # frozen

  # Step 2: rollout with π_old (no grad)
  traj, log_probs_old, values, _ ← π_old.forward_rl(agents_now, agent_futures, mode_c*)

  # Step 3: compute rewards + GAE
  R ← compute_rewards(traj, gt_traj, agent_futures, ...)   # (B, T)
  A, R_hat ← GAE(R, values, γ=0.1, λ=0.9)                 # (B, T)
  A ← normalize(A)                                         # zero mean, unit std

  # Step 4: recompute with current π (with grad)
  traj_new, log_probs_new, values_new, entropy ← π.forward_rl(..., stored_actions=traj)

  # Step 5: losses
  r_t = exp(log_probs_new - log_probs_old)    # importance ratio (B, T)
  L_policy  = PPO_clip(r_t, A, ε=0.2)        # Eq 8
  L_value   = MSE(values_new, R_hat)          # Eq 9
  L_entropy = mean(entropy)                   # Eq 10
  L_gen     = L1(traj_new, gt_traj)           # Eq 11
  L_selector = L_CE + L_side                  # same as Stage B

  L_RL = 100*L_policy + 3*L_value - 0.001*L_entropy + L_gen + L_selector

  # Step 6: update π_old every I=10 steps
  if step % 10 == 0:
      π_old ← deepcopy(π)
```

---

## Files to create / modify

| File | Action | What changes |
|---|---|---|
| `model.py` | EDIT | Add Gaussian head + value head + `forward_rl()` to `AutoregressivePolicy` |
| `rewards.py` | CREATE | 4-component per-step reward function |
| `train_stage_c.py` | CREATE | PPO training loop, loads Stage A + B checkpoints |
| `config.py` | none | RL params already defined |
| `data_loader.py` | none | Same dataloader used |

---

## Resolved design decisions

| Question | Decision |
|---|---|
| PPO clip ε | 0.2 — already in `config.py` as `PPO_CLIP` |
| Update interval I | 10 steps |
| Reward scaling | Normalize advantages (zero mean, unit std) before PPO loss |
| Value head sharing | Shares IVM encoder output (`mode_query`) — same as policy head |
| Backbone sharing | `cfg.BACKBONE_SHARING=False` for RL (paper Table 4 ablation) |

---

## RL hyperparameters (already in `config.py`)

| Parameter | Value |
|---|---|
| `LAMBDA_POLICY` | 100.0 |
| `LAMBDA_VALUE` | 3.0 |
| `LAMBDA_ENTROPY` | 0.001 |
| `PPO_CLIP` | 0.2 |
| `GAMMA` | 0.1 |
| `GAE_LAMBDA` | 0.9 |

---

## Checkpoint loading

```
train_stage_c.py --stage_a_ckpt checkpoints/stage_a_best.pt \
                 --stage_b_ckpt checkpoints/best.pt \
                 --split mini --epochs 50
```
