"""
GPU-native closed-loop evaluation using pre-extracted Test14 cache.

Runs CarPlanner autoregressively on all scenarios simultaneously in one
batched forward pass per replan step. Fully vectorised — no Python loops
over agents or map points.

Run with CarPlanner venv:
    cd /media/skr/storage/autoresearch/CarPlanner_Implementation
    source paper/.venv/bin/activate
    python scripts/eval_closedloop_gpu.py \
        --cache   checkpoints/test14_random_cache.pt \
        --checkpoint checkpoints/stage_c_best.pt \
        --stage c

For Stage B comparison:
    python scripts/eval_closedloop_gpu.py \
        --cache   checkpoints/test14_random_cache.pt \
        --checkpoint checkpoints/best.pt \
        --stage b
"""

import os
import sys
import argparse
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config as cfg
from model import CarPlanner

# ── Constants ──────────────────────────────────────────────────────────────────
T_SIM        = 150
DT           = 0.1
REPLAN_EVERY = cfg.T_FUTURE   # 8 steps = 0.8 s
N_AGENTS     = cfg.N_AGENTS   # 20
N_LANES      = cfg.N_LANES    # 20
N_PTS        = cfg.N_LANE_POINTS  # 10
T_HIST       = cfg.T_HIST     # 10
D_AGENT      = cfg.D_AGENT    # 10
D_MAP        = cfg.D_POLYLINE_POINT  # 27

COLL_THRESH  = 2.0
ROAD_THRESH  = 3.0

W_COLLISION  = 0.5
W_DRIVABLE   = 0.2
W_COMFORT    = 0.1
W_PROGRESS   = 0.2


# ── Pre-stack all scenario tensors ─────────────────────────────────────────────

def prestack(scenarios):
    """Stack list of scenario dicts into a single dict of batched tensors (CPU)."""
    return {
        'agents_world':   torch.stack([s['agents_world']   for s in scenarios]),  # (B,T,N,5)
        'agents_valid':   torch.stack([s['agents_valid']   for s in scenarios]),  # (B,T,N)
        'agents_size':    torch.stack([s['agents_size']    for s in scenarios]),  # (B,N,3)
        'agents_cat':     torch.stack([s['agents_cat']     for s in scenarios]),  # (B,N)
        'map_pts_world':  torch.stack([s['map_pts_world']  for s in scenarios]),  # (B,NL,NP,2)
        'map_lanes_mask': torch.stack([s['map_lanes_mask'] for s in scenarios]),  # (B,NL)
        'goal_world':     torch.stack([s['goal_world']     for s in scenarios]),  # (B,2)
        'ego_gt':         torch.stack([s['ego_gt']         for s in scenarios]),  # (B,T,4)
        'n_iters':        torch.tensor([s['n_iters']       for s in scenarios]),  # (B,)
    }


# ── Vectorised coordinate helpers ─────────────────────────────────────────────

def rotate_to_ego(xy_world, ego_x, ego_y, ego_yaw):
    """
    Batch-rotate world-frame xy into per-scenario ego frames.

    xy_world : (B, ..., 2)
    ego_x    : (B,)
    ego_y    : (B,)
    ego_yaw  : (B,)
    Returns  : (B, ..., 2)
    """
    # Build broadcast shape (B, 1, ..., 1)  for scalars
    extra = xy_world.dim() - 2          # number of middle dims
    view = (-1,) + (1,) * extra

    cos_h = torch.cos(-ego_yaw).view(*view)   # (B, 1..., 1) → broadcasts
    sin_h = torch.sin(-ego_yaw).view(*view)

    dx = xy_world[..., 0] - ego_x.view(*view).squeeze(-1) if extra == 0 \
         else xy_world[..., 0] - ego_x.view(*view[:-1])
    dy = xy_world[..., 1] - ego_y.view(*view).squeeze(-1) if extra == 0 \
         else xy_world[..., 1] - ego_y.view(*view[:-1])

    # Simpler: just expand dims manually
    sh = [ego_x.size(0)] + [1] * (xy_world.dim() - 2)
    cos_h = torch.cos(-ego_yaw).view(*sh)
    sin_h = torch.sin(-ego_yaw).view(*sh)
    dx = xy_world[..., 0] - ego_x.view(*sh)
    dy = xy_world[..., 1] - ego_y.view(*sh)

    xe = cos_h * dx - sin_h * dy
    ye = sin_h * dx + cos_h * dy
    return torch.stack([xe, ye], dim=-1)


def ego_to_world_batch(ego_xy, ego_states):
    """
    ego_xy     : (B, T, 2) in ego frame
    ego_states : (B, 3) — x, y, yaw
    Returns    : (B, T, 2) in world frame
    """
    ref_x   = ego_states[:, 0]
    ref_y   = ego_states[:, 1]
    ref_yaw = ego_states[:, 2]
    cos_h = torch.cos(ref_yaw)   # (B,)
    sin_h = torch.sin(ref_yaw)
    xw = cos_h[:, None] * ego_xy[:, :, 0] - sin_h[:, None] * ego_xy[:, :, 1] + ref_x[:, None]
    yw = sin_h[:, None] * ego_xy[:, :, 0] + cos_h[:, None] * ego_xy[:, :, 1] + ref_y[:, None]
    return torch.stack([xw, yw], dim=-1)   # (B, T, 2)


# ── Vectorised input builder ────────────────────────────────────────────────────

def build_batch_inputs(batch, t, ego_states, device):
    """
    Build model inputs for all B scenarios at step t, given current ego poses.

    batch      : prestack() output (CPU tensors)
    t          : int — current simulation step
    ego_states : (B, 3) — x, y, yaw for each scenario  (CPU)
    Returns    : dict of (B, ...) tensors on device
    """
    B = ego_states.size(0)
    ego_x   = ego_states[:, 0]   # (B,)
    ego_y   = ego_states[:, 1]
    ego_yaw = ego_states[:, 2]

    cos_h = torch.cos(-ego_yaw)   # (B,)
    sin_h = torch.sin(-ego_yaw)

    # ── agents_now & history ───────────────────────────────────────────────
    sz  = batch['agents_size']          # (B, N, 3)
    cat = batch['agents_cat'].float()   # (B, N)

    def encode_agents_at(t_step, t_offset):
        aw = batch['agents_world'][:, t_step]   # (B, N, 5)
        av = batch['agents_valid'][:, t_step]   # (B, N)

        dx = aw[:, :, 0] - ego_x[:, None]   # (B, N)
        dy = aw[:, :, 1] - ego_y[:, None]
        xe = cos_h[:, None] * dx - sin_h[:, None] * dy
        ye = sin_h[:, None] * dx + cos_h[:, None] * dy
        h_e  = aw[:, :, 2] - ego_yaw[:, None]  # (B, N)
        vxe  = cos_h[:, None] * aw[:, :, 3] - sin_h[:, None] * aw[:, :, 4]
        vye  = sin_h[:, None] * aw[:, :, 3] + cos_h[:, None] * aw[:, :, 4]
        t_off = torch.full((B, N_AGENTS), t_offset)

        feat = torch.stack([xe, ye, h_e, vxe, vye,
                            sz[:, :, 0], sz[:, :, 1], sz[:, :, 2],
                            t_off, cat], dim=-1)  # (B, N, D_AGENT)
        return feat * av.unsqueeze(-1), av   # mask invalid agents to 0

    agents_now, agents_mask = encode_agents_at(t, 0.0)   # (B, N, D), (B, N)

    agents_hist = torch.zeros(B, T_HIST, N_AGENTS, D_AGENT)
    for hi in range(T_HIST):
        t_h = max(0, t - (T_HIST - 1 - hi))
        t_off = float(hi - (T_HIST - 1))
        agents_hist[:, hi], _ = encode_agents_at(t_h, t_off)

    # ── map lanes ─────────────────────────────────────────────────────────
    mp = batch['map_pts_world']    # (B, NL, NP, 2)
    ml = batch['map_lanes_mask']   # (B, NL)

    dx_m = mp[:, :, :, 0] - ego_x[:, None, None]   # (B, NL, NP)
    dy_m = mp[:, :, :, 1] - ego_y[:, None, None]
    xe_m = cos_h[:, None, None] * dx_m - sin_h[:, None, None] * dy_m
    ye_m = sin_h[:, None, None] * dx_m + cos_h[:, None, None] * dy_m

    dx_seg = xe_m[:, :, 1:] - xe_m[:, :, :-1]   # (B, NL, NP-1)
    dy_seg = ye_m[:, :, 1:] - ye_m[:, :, :-1]
    headings = torch.atan2(dy_seg, dx_seg)        # (B, NL, NP-1)
    headings = torch.cat([headings, headings[:, :, -1:]], dim=2)  # (B, NL, NP)

    map_lanes = torch.zeros(B, N_LANES, N_PTS, D_MAP)
    map_lanes[:, :, :, 0] = xe_m
    map_lanes[:, :, :, 1] = ye_m
    map_lanes[:, :, :, 2] = torch.sin(headings)
    map_lanes[:, :, :, 3] = torch.cos(headings)
    # feature 4 = speed limit (unknown = 0), feature 5 = LANE category flag
    map_lanes[:, :, :, 5] = ml.unsqueeze(-1).expand(-1, -1, N_PTS)

    return {
        'agents_now':     agents_now.to(device),        # (B, N, D)
        'agents_history': agents_hist.to(device),       # (B, TH, N, D)
        'agents_mask':    agents_mask.to(device),       # (B, N)
        'map_lanes':      map_lanes.to(device),         # (B, NL, NP, 27)
        'map_lanes_mask': ml.to(device),                # (B, NL)
    }


# ── Vectorised metric computation ──────────────────────────────────────────────

def compute_metrics_batch(ego_trajs, batch, device):
    """
    ego_trajs : (B, T_SIM, 2) on device
    batch     : prestack() output (CPU)
    Returns   : dict of (B,) tensors (CPU)
    """
    B, T = ego_trajs.size(0), ego_trajs.size(1)

    agent_xy = batch['agents_world'][:, :T, :, :2].to(device)  # (B, T, N, 2)
    agent_v  = batch['agents_valid'][:, :T].to(device)          # (B, T, N)

    # ── Collision ─────────────────────────────────────────────────────────
    ego_exp = ego_trajs.unsqueeze(2)                            # (B, T, 1, 2)
    dist    = torch.norm(ego_exp - agent_xy, dim=-1)            # (B, T, N)
    dist    = dist + (1 - agent_v) * 1e6
    min_dist = dist.min(dim=-1).values                          # (B, T)
    coll_steps = (min_dist < COLL_THRESH).float()
    no_collision = 1.0 - coll_steps.mean(dim=1)                 # (B,)

    # ── Drivable ──────────────────────────────────────────────────────────
    map_pts  = batch['map_pts_world'].to(device)                # (B, NL, NP, 2)
    map_mask = batch['map_lanes_mask'].to(device)               # (B, NL)
    map_flat = map_pts.reshape(B, -1, 2)                        # (B, NL*NP, 2)
    mask_flat = map_mask.unsqueeze(2).expand(-1, -1, N_PTS).reshape(B, -1)  # (B, NL*NP)

    d_lane = torch.norm(
        ego_trajs.unsqueeze(2) - map_flat.unsqueeze(1), dim=-1)  # (B, T, NL*NP)
    d_lane = d_lane + (1 - mask_flat.unsqueeze(1)) * 1e6
    min_lane = d_lane.min(dim=-1).values                         # (B, T)
    drivable = 1.0 - (min_lane > ROAD_THRESH).float().mean(dim=1)  # (B,)

    # ── Comfort (jerk) ────────────────────────────────────────────────────
    vel  = ego_trajs[:, 1:] - ego_trajs[:, :-1]     # (B, T-1, 2)
    acc  = vel[:, 1:] - vel[:, :-1]                 # (B, T-2, 2)
    jerk = acc[:, 1:] - acc[:, :-1]                 # (B, T-3, 2)
    mean_jerk = torch.norm(jerk, dim=-1).mean(dim=1) / (DT ** 3)
    mean_jerk = torch.nan_to_num(mean_jerk, nan=5.0, posinf=5.0)
    comfort = torch.clamp(1.0 - mean_jerk / 5.0, 0.0, 1.0)     # (B,)

    # ── Progress ──────────────────────────────────────────────────────────
    goals = batch['goal_world'].to(device)           # (B, 2)
    d0 = torch.norm(ego_trajs[:, 0]  - goals, dim=-1)
    dT = torch.norm(ego_trajs[:, -1] - goals, dim=-1)
    progress = torch.clamp((d0 - dT) / (d0 + 1e-6), 0.0, 1.0)
    progress = torch.nan_to_num(progress, nan=0.0)               # (B,)

    # ── CLS-NR ────────────────────────────────────────────────────────────
    cls_nr = (W_COLLISION * no_collision + W_DRIVABLE * drivable
              + W_COMFORT * comfort + W_PROGRESS * progress)

    return {
        'cls_nr':       cls_nr.cpu(),
        'no_collision': no_collision.cpu(),
        'drivable':     drivable.cpu(),
        'comfort':      comfort.cpu(),
        'progress':     progress.cpu(),
        'coll_steps':   coll_steps.sum(dim=1).cpu(),
    }


# ── Batched simulation loop ─────────────────────────────────────────────────────

def run_batch(model, batch, device):
    """
    Run all B scenarios simultaneously, replanning every REPLAN_EVERY steps.

    batch  : prestack() output
    Returns: dict of (B,) metric tensors
    """
    B = batch['ego_gt'].size(0)
    n_iters = batch['n_iters']                           # (B,)

    # Initialise ego states from GT step 0
    ego_states = batch['ego_gt'][:, 0, :3].clone()      # (B, 3) x,y,yaw — CPU

    ego_trajs = torch.zeros(B, T_SIM, 2)
    ego_trajs[:, 0] = ego_states[:, :2]

    t = 0
    while t < T_SIM - 1:
        steps_to_fill = min(REPLAN_EVERY, T_SIM - t - 1)

        # Build batched inputs and run model
        inp = build_batch_inputs(batch, t, ego_states, device)
        with torch.no_grad():
            _, _, best_traj, _ = model.forward_inference(
                agents_now=inp['agents_now'],
                agents_mask=inp['agents_mask'],
                map_lanes=inp['map_lanes'],
                map_lanes_mask=inp['map_lanes_mask'],
                agents_history=inp['agents_history'],
            )
        # best_traj: (B, T_FUTURE, 3) in ego frame

        # Transform to world frame: (B, T_FUTURE, 2)
        pred_world = ego_to_world_batch(
            best_traj[:, :, :2].cpu(), ego_states)

        # Fill trajectory buffer
        active = (t < n_iters - 1)                       # (B,) bool
        for k in range(steps_to_fill):
            si = t + 1 + k
            if si < T_SIM:
                ego_trajs[:, si] = torch.where(
                    active.unsqueeze(1).expand(-1, 2),
                    pred_world[:, k],
                    ego_trajs[:, si - 1],
                )

        # Advance ego position; keep GT yaw
        last_k = steps_to_fill - 1
        next_step = min(t + steps_to_fill, T_SIM - 1)
        ego_states[:, :2] = torch.where(
            active.unsqueeze(1).expand(-1, 2),
            pred_world[:, last_k],
            ego_states[:, :2],
        )
        ego_states[:, 2] = batch['ego_gt'][:, next_step, 2]

        t += steps_to_fill

    # Fill any scenarios that ended early
    for b in range(B):
        ni = n_iters[b].item()
        if ni < T_SIM:
            ego_trajs[b, ni:] = ego_trajs[b, ni - 1]

    # Replace NaN/Inf in trajectories with last valid position
    for b in range(B):
        for t_s in range(1, T_SIM):
            if torch.isnan(ego_trajs[b, t_s]).any() or torch.isinf(ego_trajs[b, t_s]).any():
                ego_trajs[b, t_s] = ego_trajs[b, t_s - 1]

    ego_trajs = ego_trajs.to(device)
    return compute_metrics_batch(ego_trajs, batch, device)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache',      required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--stage',      default='c', choices=['a', 'b', 'c'])
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Scenarios per forward pass (default: 64)')
    parser.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[GPU Eval] Device:      {device}")
    print(f"[GPU Eval] Cache:       {args.cache}")
    print(f"[GPU Eval] Checkpoint:  {args.checkpoint}")
    print(f"[GPU Eval] Batch size:  {args.batch_size}")

    # ── Load cache ─────────────────────────────────────────────────────────
    data      = torch.load(args.cache, map_location='cpu')
    scenarios = data['scenarios']
    print(f"[GPU Eval] {len(scenarios)} scenarios loaded")

    # ── Load model ─────────────────────────────────────────────────────────
    model = CarPlanner().to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    print(f"[GPU Eval] Model loaded (epoch {ckpt.get('epoch', '?')})")

    # ── Run in mini-batches ────────────────────────────────────────────────
    all_m   = defaultdict(list)
    type_m  = defaultdict(lambda: defaultdict(list))
    stypes  = [s['scenario_type'] for s in scenarios]
    t0      = time.time()

    n_batches = (len(scenarios) + args.batch_size - 1) // args.batch_size
    print(f"[GPU Eval] Running {n_batches} batch(es) of up to {args.batch_size} scenarios", flush=True)

    for bi in range(n_batches):
        sl      = slice(bi * args.batch_size, (bi + 1) * args.batch_size)
        scens_b = scenarios[sl]
        types_b = stypes[sl]

        print(f"[Batch {bi+1}/{n_batches}] scenarios {sl.start}–{min(sl.stop, len(scenarios))-1}  ...", flush=True)
        t_batch = time.time()

        try:
            batch   = prestack(scens_b)
            metrics = run_batch(model, batch, device)

            B_cur = len(scens_b)
            for b in range(B_cur):
                for k, v in metrics.items():
                    val = v[b].item()
                    all_m[k].append(val)
                    type_m[types_b[b]][k].append(val)

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  WARNING batch {bi}: {e}", flush=True)

        n_done = len(all_m['cls_nr'])
        if n_done:
            elapsed  = time.time() - t0
            b_time   = time.time() - t_batch
            eta      = (elapsed / (bi + 1)) * (n_batches - bi - 1)
            cls_now  = sum(all_m['cls_nr']) / n_done * 100
            nocol    = sum(all_m['no_collision']) / n_done * 100
            prog     = sum(all_m['progress']) / n_done * 100
            driv     = sum(all_m['drivable']) / n_done * 100
            print(f"  done in {b_time:.1f}s  |  "
                  f"CLS-NR={cls_now:.1f}  noCol={nocol:.1f}%  "
                  f"driv={driv:.1f}%  prog={prog:.1f}%  "
                  f"ETA={eta:.0f}s  [{n_done}/{len(scenarios)}]", flush=True)

    elapsed = time.time() - t0

    # ── Report ─────────────────────────────────────────────────────────────
    def mean(lst): return sum(lst) / len(lst) if lst else 0.0

    print(f"\n{'='*65}")
    print(f"  GPU Closed-Loop Eval — Stage {args.stage.upper()}")
    print(f"{'='*65}")
    print(f"  Scenarios:            {len(all_m['cls_nr'])}")
    print(f"  Time:                 {elapsed:.0f}s")
    print(f"")
    print(f"  CLS-NR (composite):   {mean(all_m['cls_nr'])*100:.2f} / 100")
    print(f"  No-collision:         {mean(all_m['no_collision'])*100:.2f}%")
    print(f"  Drivable compliance:  {mean(all_m['drivable'])*100:.2f}%")
    print(f"  Comfort score:        {mean(all_m['comfort'])*100:.2f}%")
    print(f"  Progress score:       {mean(all_m['progress'])*100:.2f}%")
    print(f"{'='*65}")

    print(f"\n  Per-type CLS-NR:")
    for stype in sorted(type_m.keys()):
        n = len(type_m[stype]['cls_nr'])
        print(f"    {stype:55s}  {mean(type_m[stype]['cls_nr'])*100:.2f}  (n={n})")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()
