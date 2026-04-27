"""
CarPlanner — Stage B (IL) comprehensive evaluation.

Computes all open-loop metrics reported in the paper tables:
  - L_gen, L_sel (Table 4 open-loop columns)
  - ADE, FDE
  - Mode accuracy (top-1, top-5)
  - Consistent Ratio Lat/Lon (Table 8)
  - Open-loop Collision & Drivable Area metrics (Table 10)

Prints a comparison table against paper benchmarks.

For closed-loop metrics (CLS-NR, S-CR, S-Area, S-PR, S-Comfort from
Tables 1, 2, 4, 7, 9, 10, 11), run:
    python scripts/eval_nuplan.py --checkpoint <ckpt> --split test14-random

Usage:
    python eval_stage_b.py --checkpoint checkpoints/stage_b_best.pt
    python eval_stage_b.py --checkpoint checkpoints/stage_b_best.pt --split test14_random
    python eval_stage_b.py --checkpoint checkpoints/stage_b_best.pt --split reduced_val14
"""

import os
import sys
import argparse
import time

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset

import config as cfg
from data_loader import (PreextractedDataset,
                         _collect_candidate_lanes,
                         _match_endpoint_to_route,
                         _bin_lateral_offset)
from model import CarPlanner


# ── Paper benchmarks ─────────────────────────────────────────────────────────
# Table 4 IL-best row (all 4 flags ON), Table 8 IL consistent, Table 10 IL
PAPER_IL = {
    'L_gen': 174.3, 'L_sel': 1.04,
    'CLS_NR': 93.41, 'S_CR': 98.85, 'S_Area': 98.85,
    'S_PR': 93.87, 'S_Comfort': 96.15,
    'consistent_lat': 68.26, 'consistent_lon': 43.01,
    'col_mean': 0.15, 'col_min': 0.00, 'col_max': 0.44,
    'area_mean': 0.09, 'area_min': 0.00, 'area_max': 0.40,
}
# Table 4 RL-best row (Mode Drop + Side only), Table 8 RL, Table 10 RL
PAPER_RL = {
    'L_gen': 1624.5, 'L_sel': 1.03,
    'CLS_NR': 94.07, 'S_CR': 99.22, 'S_Area': 99.22,
    'S_PR': 95.06, 'S_Comfort': 91.09,
    'consistent_lat': 79.58, 'consistent_lon': 43.03,
    'col_mean': 0.12, 'col_min': 0.00, 'col_max': 0.39,
    'area_mean': 0.05, 'area_min': 0.00, 'area_max': 0.22,
}


def _mode_to_bins(m):
    return int(m) // cfg.N_LAT, int(m) % cfg.N_LAT


def _compute_consistent_ratio(all_trajs_np, map_lanes_np, map_lanes_mask_np):
    """
    Table 8: Consistent Ratio. For each mode index m = lon_idx · N_LAT + lat_idx,
    check whether the generated trajectory's endpoint actually lands in the
    (lon, lat) bin assigned to that mode index.

    The eval mirrors data_loader.py's label-assignment logic exactly so that
    paper Table 8 numbers compare apples-to-apples:

      Longitudinal — `max(traj[-1, 0], 0.0)` (forward ego-frame x at horizon
        end), binned via `cfg.MAX_SPEED · T_FUTURE · FUTURE_DT_S / N_LON`.
        Matches data_loader.py:1132-1138.

      Lateral     — route-proximity matching: collect candidate lanes from
        `map_lanes`, find which candidate the trajectory endpoint is closest
        to, bin the lateral offset between the matched lane and the ego lane
        via `LAT_BIN_EDGES` (paper §3.3.2). Falls back to a y-offset bin only
        when no candidates exist, matching the trainer's fallback at
        data_loader.py:1140-1145.

    all_trajs_np:       (N_MODES, T, 3)
    map_lanes_np:       (N_LANES, N_PTS, D_MAP) in ego frame
    map_lanes_mask_np:  (N_LANES,)
    Returns: (lat_ratio, lon_ratio) as percentages
    """
    n_modes = all_trajs_np.shape[0]
    lon_step = cfg.MAX_SPEED * cfg.T_FUTURE * cfg.FUTURE_DT_S / cfg.N_LON

    candidates = _collect_candidate_lanes(map_lanes_np, map_lanes_mask_np)

    lat_ok = 0
    lon_ok = 0
    for m in range(n_modes):
        lon_idx, lat_idx = _mode_to_bins(m)
        traj = all_trajs_np[m]  # (T, 3)

        # Longitudinal: forward (ego x) endpoint displacement → bin.
        lon_dist = max(float(traj[-1, 0]), 0.0)
        speed_bin = min(int(lon_dist / lon_step), cfg.N_LON - 1)
        if speed_bin == lon_idx:
            lon_ok += 1

        # Lateral: route-proximity match against actual lane candidates.
        # Falls back to y-offset binning iff no candidates were found
        # (matches the trainer's fallback exactly).
        y_bin = _match_endpoint_to_route(candidates, traj[-1])
        if y_bin < 0:
            y_bin = _bin_lateral_offset(float(traj[-1, 1]))
        if y_bin == lat_idx:
            lat_ok += 1

    return lat_ok / n_modes * 100, lon_ok / n_modes * 100


def _compute_ol_col_area(all_trajs_np, agents_now_np, agents_mask_np,
                         map_lanes_np, map_lanes_mask_np,
                         route_polylines_np=None, route_mask_np=None):
    """
    Table 10 open-loop: Collision and Drivable Area metrics across all
    candidate trajectories.

    Collision: fraction of trajectory timesteps where ego is within 2m of
    any valid agent position (at t=0, static proxy).

    Area: fraction of trajectory timesteps where ego is >3.5m from any
    drivable-corridor reference point (off drivable area). The reference
    pool is lane centerlines + route polyline centers (up to 150 m forward),
    needed to cover the 8 s × ~15 m/s ≈ 120 m horizon. Without route points
    GT trajectories register ~26% off-road purely because lanes are sorted
    by t=0 ego-distance and far-future lanes are missing from the cache.

    Returns: (col_mean, col_min, col_max, area_mean, area_min, area_max)
    """
    n_modes = all_trajs_np.shape[0]

    # Agent positions at t=0
    valid_agents = []
    for n in range(len(agents_mask_np)):
        if agents_mask_np[n] < 0.5:
            continue
        ax, ay = agents_now_np[n, 0], agents_now_np[n, 1]
        if abs(ax) < 1e-6 and abs(ay) < 1e-6:
            continue
        valid_agents.append([ax, ay])
    agent_pts = np.array(valid_agents) if valid_agents else np.zeros((0, 2))

    # Drivable-corridor reference points: lane centerlines + route polylines
    lane_pts_list = []
    for i in range(len(map_lanes_mask_np)):
        if map_lanes_mask_np[i] < 0.5:
            continue
        lane = map_lanes_np[i]
        valid = np.abs(lane[:, 0]) + np.abs(lane[:, 1]) > 1e-6
        if valid.sum() >= 2:
            lane_pts_list.append(lane[valid, :2])
    if route_polylines_np is not None and route_mask_np is not None:
        for i in range(len(route_mask_np)):
            if route_mask_np[i] < 0.5:
                continue
            poly = route_polylines_np[i]
            valid = np.abs(poly[:, 0]) + np.abs(poly[:, 1]) > 1e-6
            if valid.sum() >= 2:
                lane_pts_list.append(poly[valid, :2])
    lane_pts = np.concatenate(lane_pts_list, axis=0) if lane_pts_list else np.zeros((0, 2))

    col_scores = []
    area_scores = []

    for m in range(n_modes):
        traj_xy = all_trajs_np[m, :, :2]  # (T, 2)
        T = len(traj_xy)

        # Collision: min distance to any agent at each timestep
        if len(agent_pts) > 0:
            # (T, 1, 2) - (1, N_agents, 2) → (T, N_agents)
            dists = np.sqrt(((traj_xy[:, None, :] - agent_pts[None, :, :])**2).sum(axis=2))
            min_agent_dist = dists.min(axis=1)  # (T,)
            col_frac = float((min_agent_dist < 2.0).mean())
        else:
            col_frac = 0.0

        # Area: min distance to any lane centerline at each timestep
        if len(lane_pts) > 0:
            dists = np.sqrt(((traj_xy[:, None, :] - lane_pts[None, :, :])**2).sum(axis=2))
            min_lane_dist = dists.min(axis=1)  # (T,)
            area_frac = float((min_lane_dist > 3.5).mean())
        else:
            area_frac = 1.0

        col_scores.append(col_frac)
        area_scores.append(area_frac)

    col = np.array(col_scores)
    area = np.array(area_scores)
    return (float(col.mean()), float(col.min()), float(col.max()),
            float(area.mean()), float(area.min()), float(area.max()))


# ── Main ─────────────────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Eval Stage B] Device: {device}")

    dataset = PreextractedDataset(args.cache, device=device)
    n_total = len(dataset)
    n_eval = min(args.n_eval, n_total) if args.n_eval > 0 else n_total
    print(f"[Eval Stage B] Dataset: {args.cache} ({n_total} samples, evaluating {n_eval})")

    model = CarPlanner().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    epoch = ckpt.get('epoch', '?')
    print(f"[Eval Stage B] Checkpoint: {args.checkpoint} (epoch {epoch}, {n_params:,} params)")
    print(f"[Eval Stage B] Config: MODE_DROPOUT={cfg.MODE_DROPOUT}, "
          f"SELECTOR_SIDE_TASK={cfg.SELECTOR_SIDE_TASK}, "
          f"EGO_HISTORY_DROPOUT={cfg.EGO_HISTORY_DROPOUT}, "
          f"BACKBONE_SHARING={cfg.BACKBONE_SHARING}")
    print(f"[Eval Stage B] N_MODES={cfg.N_MODES} (N_LON={cfg.N_LON} x N_LAT={cfg.N_LAT})")

    indices = np.linspace(0, n_total - 1, n_eval, dtype=int).tolist()
    subset = Subset(dataset, indices) if n_eval < n_total else dataset
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    stats = {
        'ade': [], 'fde': [],
        # Two L_gen flavours: paper trains generator on the GT-mode-conditioned
        # rollout (l_gen_train); we previously only reported the inference-time
        # rule-selected trajectory's L1 (l_gen_inference). Reporting both makes
        # paper comparisons unambiguous — Table 4's "L_gen" is the training one.
        'l_gen_train': [], 'l_gen_inference': [],
        'l_sel': [],
        'mode_correct': [], 'top5_correct': [],
        'gt_prob': [], 'gt_rank': [],
        'consistent_lat': [], 'consistent_lon': [],
        'col_mean': [], 'col_min': [], 'col_max': [],
        'area_mean': [], 'area_min': [], 'area_max': [],
    }

    t0 = time.time()
    done = 0

    with torch.no_grad():
        for batch in loader:
            mode_logits, all_trajs, best_traj, best_idx = model.forward_inference_fast(
                agents_now=batch['agents_now'],
                agents_mask=batch['agents_history_mask'],
                map_lanes=batch['map_lanes'],
                map_lanes_mask=batch['map_lanes_mask'],
                agents_history=batch['agents_history'],
                ego_history=batch.get('ego_history'),
                map_polygons=batch.get('map_polygons'),
                map_polygons_mask=batch.get('map_polygons_mask'),
                route_polylines=batch.get('route_polylines'),
                route_mask=batch.get('route_mask'),
            )

            probs = F.softmax(mode_logits, dim=-1)          # (B, M)
            log_probs = F.log_softmax(mode_logits, dim=-1)  # (B, M)
            gt_traj = batch['gt_trajectory']                 # (B, T, D)
            mode_labels = batch['mode_label']                # (B,)
            B_cur = mode_logits.size(0)

            for i in range(B_cur):
                pred = best_traj[i].cpu().numpy()
                gt   = gt_traj[i].cpu().numpy()
                mode_label   = mode_labels[i].item()
                chosen_mode  = best_idx[i].item()
                all_trajs_np = all_trajs[i].cpu().numpy()

                diff = pred[:, :2] - gt[:, :2]
                stats['ade'].append(float(np.mean(np.linalg.norm(diff, axis=1))))
                stats['fde'].append(float(np.linalg.norm(pred[-1, :2] - gt[-1, :2])))
                # Inference-time L_gen (rule-selector pick vs GT) — sum-reduced
                # per-sample to match the previously-reported scalar magnitude.
                stats['l_gen_inference'].append(float(np.sum(np.abs(pred - gt))))
                # Training-time L_gen (Eq 11): GT-mode-conditioned rollout vs
                # GT, mean-reduced over T then summed over feature dims (matches
                # train_stage_b.py compute_il_loss).
                pred_gt_mode = all_trajs_np[mode_label]
                stats['l_gen_train'].append(
                    float(np.mean(np.sum(np.abs(pred_gt_mode - gt), axis=-1)))
                )
                stats['l_sel'].append(float(-log_probs[i, mode_label]))

                order = torch.argsort(probs[i], descending=True)
                gt_rank = int((order == mode_label).nonzero(as_tuple=True)[0].item()) + 1
                stats['mode_correct'].append(chosen_mode == mode_label)
                stats['top5_correct'].append(gt_rank <= 5)
                stats['gt_prob'].append(float(probs[i, mode_label]))
                stats['gt_rank'].append(gt_rank)

                agents_now_np  = batch['agents_now'][i].cpu().numpy()
                agents_mask_np = batch['agents_history_mask'][i].cpu().numpy()
                ml_np          = batch['map_lanes'][i].cpu().numpy()
                mlm_np         = batch['map_lanes_mask'][i].cpu().numpy()

                lat_r, lon_r = _compute_consistent_ratio(all_trajs_np, ml_np, mlm_np)
                stats['consistent_lat'].append(lat_r)
                stats['consistent_lon'].append(lon_r)
                rp_np          = batch['route_polylines'][i].cpu().numpy() if 'route_polylines' in batch else None
                rm_np          = batch['route_mask'][i].cpu().numpy() if 'route_mask' in batch else None
                cm, cmin, cmax, am, amin, amax = _compute_ol_col_area(
                    all_trajs_np, agents_now_np, agents_mask_np, ml_np, mlm_np,
                    route_polylines_np=rp_np, route_mask_np=rm_np)
                stats['col_mean'].append(cm);  stats['col_min'].append(cmin);  stats['col_max'].append(cmax)
                stats['area_mean'].append(am); stats['area_min'].append(amin); stats['area_max'].append(amax)

            done += B_cur
            if done % 100 < args.batch_size or done >= n_eval:
                elapsed = time.time() - t0
                print(f"  [{done}/{n_eval}]  {done/elapsed:.1f} samples/s  "
                      f"ADE={np.mean(stats['ade']):.3f}  "
                      f"L_gen(train)={np.mean(stats['l_gen_train']):.2f}  "
                      f"L_sel={np.mean(stats['l_sel']):.2f}")

    elapsed = time.time() - t0
    print(f"\n[Eval Stage B] Done in {elapsed:.1f}s ({n_eval/elapsed:.1f} samples/s)\n")

    # ── Aggregate ────────────────────────────────────────────────────────────
    r = {}
    for k, v in stats.items():
        if k in ('mode_correct', 'top5_correct'):
            r[k] = float(np.mean(v)) * 100
        else:
            r[k] = float(np.mean(v))
    r['median_gt_rank'] = int(np.median(stats['gt_rank']))
    # Col/Area min/max are per-sample averages of per-mode min/max
    r['col_min'] = float(np.mean(stats['col_min']))
    r['col_max'] = float(np.mean(stats['col_max']))
    r['area_min'] = float(np.mean(stats['area_min']))
    r['area_max'] = float(np.mean(stats['area_max']))

    il = PAPER_IL
    rl = PAPER_RL

    # ── Print results ────────────────────────────────────────────────────────
    W = 78
    print("=" * W)
    print(f"  STAGE B (IL) EVALUATION — {n_eval} samples on {os.path.basename(args.cache)}")
    print(f"  Checkpoint: {args.checkpoint} (epoch {epoch})")
    print("=" * W)

    def _row(label, ours, paper_il, paper_rl):
        print(f"  {label:<35} {ours:>12} {paper_il:>15} {paper_rl:>15}")

    def _sep():
        print(f"  {'─'*74}")

    # Table 4 open-loop
    print(f"\n  Table 4 — Open-loop metrics")
    _sep()
    _row("Metric", "Ours", "Paper IL-best", "Paper RL-best")
    _sep()
    # Paper Table 4 "L_gen" column = training-objective generator loss → use l_gen_train.
    _row("L_gen (Eq 11, GT-mode)",  f"{r['l_gen_train']:.2f}",     f"{il['L_gen']:.1f}", f"{rl['L_gen']:.1f}")
    _row("L_gen (inference, sum)",  f"{r['l_gen_inference']:.1f}", "—",                  "—")
    _row("L_sel (selector CE loss)", f"{r['l_sel']:.2f}", f"{il['L_sel']:.2f}", f"{rl['L_sel']:.2f}")
    _row("ADE (m)", f"{r['ade']:.3f}", "—", "—")
    _row("FDE (m)", f"{r['fde']:.3f}", "—", "—")
    _row("Mode accuracy top-1 (%)", f"{r['mode_correct']:.2f}", "—", "—")
    _row("Mode accuracy top-5 (%)", f"{r['top5_correct']:.2f}", "—", "—")
    _row("Median GT rank", f"{r['median_gt_rank']}", f"— / {cfg.N_MODES}", f"— / {cfg.N_MODES}")

    # Table 8
    print(f"\n  Table 8 — Consistent Ratio")
    _sep()
    _row("Metric", "Ours", "Paper IL-best", "Paper RL-best")
    _sep()
    _row("Consistent Ratio Lat (%)", f"{r['consistent_lat']:.2f}", f"{il['consistent_lat']:.2f}", f"{rl['consistent_lat']:.2f}")
    _row("Consistent Ratio Lon (%)", f"{r['consistent_lon']:.2f}", f"{il['consistent_lon']:.2f}", f"{rl['consistent_lon']:.2f}")

    # Table 10 open-loop
    print(f"\n  Table 10 — Open-loop Collision & Area")
    _sep()
    _row("Metric", "Ours", "Paper IL-best", "Paper RL-best")
    _sep()
    _row("Col Mean", f"{r['col_mean']:.2f}", f"{il['col_mean']:.2f}", f"{rl['col_mean']:.2f}")
    _row("Col [Min, Max]", f"[{r['col_min']:.2f}, {r['col_max']:.2f}]",
         f"[{il['col_min']:.2f}, {il['col_max']:.2f}]", f"[{rl['col_min']:.2f}, {rl['col_max']:.2f}]")
    _row("Area Mean", f"{r['area_mean']:.2f}", f"{il['area_mean']:.2f}", f"{rl['area_mean']:.2f}")
    _row("Area [Min, Max]", f"[{r['area_min']:.2f}, {r['area_max']:.2f}]",
         f"[{il['area_min']:.2f}, {il['area_max']:.2f}]", f"[{rl['area_min']:.2f}, {rl['area_max']:.2f}]")

    # Closed-loop reminder
    print(f"\n  Tables 1, 2, 4, 7, 9, 11 — Closed-loop (requires nuPlan simulator)")
    _sep()
    print(f"  Run:")
    print(f"    python scripts/eval_nuplan.py --checkpoint {args.checkpoint} --split test14-random")
    print(f"    python scripts/eval_nuplan.py --checkpoint {args.checkpoint} --split val14")
    _sep()
    print(f"\n  Paper IL-best closed-loop (Table 4): CLS-NR={il['CLS_NR']}, "
          f"S-CR={il['S_CR']}, S-Area={il['S_Area']}, S-PR={il['S_PR']}, S-Comfort={il['S_Comfort']}")
    print("=" * W)


SPLIT_TO_CACHE = {
    'val14':          'checkpoints/stage_cache_val14.pt',
    'test14_random':  'checkpoints/stage_cache_test14_random.pt',
    'reduced_val14':  'checkpoints/stage_cache_reduced_val14.pt',
}


def parse_args():
    p = argparse.ArgumentParser(description="CarPlanner Stage B (IL) evaluation")
    p.add_argument('--checkpoint', required=True, help='Path to Stage B checkpoint')
    p.add_argument('--split', default='val14',
                   choices=list(SPLIT_TO_CACHE.keys()),
                   help='Benchmark split (default: val14)')
    p.add_argument('--cache', default=None,
                   help='Override cache path (default: auto from --split)')
    p.add_argument('--n_eval', type=int, default=0,
                   help='Max samples to evaluate (0 = all)')
    p.add_argument('--batch_size', type=int, default=32,
                   help='Inference batch size (default: 32)')
    args = p.parse_args()
    if args.cache is None:
        args.cache = SPLIT_TO_CACHE[args.split]
    return args


if __name__ == '__main__':
    evaluate(parse_args())
