"""
CarPlanner — Stage C (RL) comprehensive evaluation.

Same open-loop metrics as eval_stage_b.py (Tables 4, 8, 10) plus
RL-specific diagnostics: reward components, policy entropy, value
estimates, and action std.

The primary comparison target is the paper's RL-best row.

For closed-loop metrics (CLS-NR, S-CR, S-Area, S-PR, S-Comfort from
Tables 1, 2, 4, 7, 9, 10, 11), run:
    python scripts/eval_nuplan.py --checkpoint <ckpt> --split test14-random

Usage:
    python eval_stage_c.py --checkpoint checkpoints/stage_c_best.pt
    python eval_stage_c.py --checkpoint checkpoints/stage_c_best.pt --split test14_random
    python eval_stage_c.py --checkpoint checkpoints/stage_c_best.pt --split reduced_val14
"""

import os
import sys
import argparse
import time

import torch
import torch.nn.functional as F
import numpy as np

import config as cfg
from data_loader import PreextractedDataset
from torch.utils.data import DataLoader, Subset
from model import CarPlanner
from rewards import compute_rewards


# ── Paper benchmarks ─────────────────────────────────────────────────────────
PAPER_IL = {
    'L_gen': 174.3, 'L_sel': 1.04,
    'CLS_NR': 93.41, 'S_CR': 98.85, 'S_Area': 98.85,
    'S_PR': 93.87, 'S_Comfort': 96.15,
    'consistent_lat': 68.26, 'consistent_lon': 43.01,
    'col_mean': 0.15, 'col_min': 0.00, 'col_max': 0.44,
    'area_mean': 0.09, 'area_min': 0.00, 'area_max': 0.40,
}
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


def _compute_consistent_ratio(all_trajs_np, gt_np):
    n_modes = all_trajs_np.shape[0]
    T = all_trajs_np.shape[1]
    dt = 0.5

    lat_ok = 0
    lon_ok = 0

    for m in range(n_modes):
        lon_idx, lat_idx = _mode_to_bins(m)
        traj = all_trajs_np[m]

        displacements = np.sqrt(np.sum(np.diff(traj[:, :2], axis=0)**2, axis=1))
        avg_speed = float(displacements.sum()) / (T * dt)
        speed_bin = min(int(avg_speed / (cfg.MAX_SPEED / cfg.N_LON)), cfg.N_LON - 1)
        if speed_bin == lon_idx:
            lon_ok += 1

        endpoint_y = float(traj[-1, 1])
        lane_width = 3.7
        half_range = (cfg.N_LAT / 2) * lane_width
        y_bin = int((endpoint_y + half_range) / lane_width)
        y_bin = max(0, min(y_bin, cfg.N_LAT - 1))
        if y_bin == lat_idx:
            lat_ok += 1

    return lat_ok / n_modes * 100, lon_ok / n_modes * 100


def _compute_ol_col_area(all_trajs_np, agents_now_np, agents_mask_np,
                         map_lanes_np, map_lanes_mask_np):
    n_modes = all_trajs_np.shape[0]

    valid_agents = []
    for n in range(len(agents_mask_np)):
        if agents_mask_np[n] < 0.5:
            continue
        ax, ay = agents_now_np[n, 0], agents_now_np[n, 1]
        if abs(ax) < 1e-6 and abs(ay) < 1e-6:
            continue
        valid_agents.append([ax, ay])
    agent_pts = np.array(valid_agents) if valid_agents else np.zeros((0, 2))

    lane_pts_list = []
    for i in range(len(map_lanes_mask_np)):
        if map_lanes_mask_np[i] < 0.5:
            continue
        lane = map_lanes_np[i]
        valid = np.abs(lane[:, 0]) + np.abs(lane[:, 1]) > 1e-6
        if valid.sum() >= 2:
            lane_pts_list.append(lane[valid, :2])
    lane_pts = np.concatenate(lane_pts_list, axis=0) if lane_pts_list else np.zeros((0, 2))

    col_scores = []
    area_scores = []

    for m in range(n_modes):
        traj_xy = all_trajs_np[m, :, :2]
        T = len(traj_xy)

        if len(agent_pts) > 0:
            dists = np.sqrt(((traj_xy[:, None, :] - agent_pts[None, :, :])**2).sum(axis=2))
            min_agent_dist = dists.min(axis=1)
            col_frac = float((min_agent_dist < 2.0).mean())
        else:
            col_frac = 0.0

        if len(lane_pts) > 0:
            dists = np.sqrt(((traj_xy[:, None, :] - lane_pts[None, :, :])**2).sum(axis=2))
            min_lane_dist = dists.min(axis=1)
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
    print(f"[Eval Stage C] Device: {device}")

    dataset = PreextractedDataset(args.cache, device=device)
    n_total = len(dataset)
    n_eval = min(args.n_eval, n_total) if args.n_eval > 0 else n_total
    print(f"[Eval Stage C] Dataset: {args.cache} ({n_total} samples, evaluating {n_eval})")

    cfg.set_stage('c')
    model = CarPlanner().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    epoch = ckpt.get('epoch', '?')
    print(f"[Eval Stage C] Checkpoint: {args.checkpoint} (epoch {epoch}, {n_params:,} params)")
    print(f"[Eval Stage C] Config: MODE_DROPOUT={cfg.MODE_DROPOUT}, "
          f"SELECTOR_SIDE_TASK={cfg.SELECTOR_SIDE_TASK}, "
          f"EGO_HISTORY_DROPOUT={cfg.EGO_HISTORY_DROPOUT}, "
          f"BACKBONE_SHARING={cfg.BACKBONE_SHARING}")
    print(f"[Eval Stage C] N_MODES={cfg.N_MODES} (N_LON={cfg.N_LON} x N_LAT={cfg.N_LAT})")

    # Report learned action std
    if hasattr(model.policy, 'action_log_std'):
        action_std = model.policy.action_log_std.exp().detach().cpu().numpy()
        print(f"[Eval Stage C] Learned action_std: x={action_std[0]:.4f}, "
              f"y={action_std[1]:.4f}, yaw={action_std[2]:.4f}")

    # Load Stage A transition model if available (for reward computation)
    has_transition = False
    if args.stage_a_ckpt and os.path.isfile(args.stage_a_ckpt):
        model.load_transition_model(args.stage_a_ckpt, freeze=True)
        has_transition = True
        print(f"[Eval Stage C] Loaded transition model for reward eval: {args.stage_a_ckpt}")
    else:
        print(f"[Eval Stage C] No --stage_a_ckpt provided; skipping reward diagnostics")

    indices = np.linspace(0, n_total - 1, n_eval, dtype=int).tolist()
    subset = Subset(dataset, indices) if n_eval < n_total else dataset
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    stats = {
        'ade': [], 'fde': [],
        'l_gen': [], 'l_sel': [],
        'mode_correct': [], 'top5_correct': [],
        'gt_prob': [], 'gt_rank': [],
        'consistent_lat': [], 'consistent_lon': [],
        'col_mean': [], 'col_min': [], 'col_max': [],
        'area_mean': [], 'area_min': [], 'area_max': [],
        # RL-specific
        'entropy': [], 'value_mean': [],
        'reward_total': [], 'reward_displacement': [],
        'reward_collision': [], 'reward_drivable': [], 'reward_comfort': [],
    }

    t0 = time.time()
    done = 0

    with torch.no_grad():
        for batch in loader:
            B_cur = batch['agents_now'].size(0)

            # ── Batched deterministic inference ───────────────────────
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

            probs     = F.softmax(mode_logits, dim=-1)
            log_probs = F.log_softmax(mode_logits, dim=-1)
            gt_traj   = batch['gt_trajectory']
            mode_labels = batch['mode_label']

            # ── RL-specific: batched stochastic rollout ────────────────
            if has_transition:
                agent_futures_rl = model.transition_model(
                    batch['agents_history'], batch['agents_history_mask'],
                    batch['map_lanes'], batch['map_lanes_mask'],
                    map_polygons=batch.get('map_polygons'),
                    map_polygons_mask=batch.get('map_polygons_mask'),
                )
                traj_rl, lp, values, entropies = model.policy.forward_rl(
                    agents_now=batch['agents_now'],
                    agents_seq=agent_futures_rl,
                    agents_mask=batch['agents_history_mask'],
                    mode_c=batch['mode_label'],
                    map_lanes=batch['map_lanes'],
                    map_lanes_mask=batch['map_lanes_mask'],
                    map_polygons=batch.get('map_polygons'),
                    map_polygons_mask=batch.get('map_polygons_mask'),
                    route_polylines=batch.get('route_polylines'),
                    route_mask=batch.get('route_mask'),
                )
                rewards = compute_rewards(
                    ego_traj=traj_rl, gt_traj=gt_traj,
                    agent_futures=agent_futures_rl,
                    agents_mask=batch['agents_history_mask'],
                    map_lanes=batch['map_lanes'],
                    map_lanes_mask=batch['map_lanes_mask'],
                )
                # Batch-level RL stats
                ego_xy = traj_rl[..., :2]
                gt_xy  = gt_traj[..., :2]
                r_disp = -torch.norm(ego_xy - gt_xy, dim=-1).mean(dim=-1)

                agent_xy  = agent_futures_rl[..., :2]
                d_agents  = torch.norm(ego_xy.unsqueeze(2) - agent_xy, dim=-1)
                mask_exp  = batch['agents_history_mask'].unsqueeze(1)
                d_agents  = d_agents + (1 - mask_exp) * 1e6
                min_d, _  = d_agents.min(dim=-1)
                r_col     = -(min_d < 2.0).float().mean(dim=-1)

                lane_flat      = batch['map_lanes'][:, :, :, :2].reshape(B_cur, -1, 2)
                d_lanes        = torch.norm(ego_xy.unsqueeze(2) - lane_flat.unsqueeze(1), dim=-1)
                lane_mask_flat = batch['map_lanes_mask'].unsqueeze(2).expand(
                    -1, -1, cfg.N_LANE_POINTS).reshape(B_cur, -1)
                d_lanes        = d_lanes + (1 - lane_mask_flat.unsqueeze(1)) * 1e6
                min_dl, _      = d_lanes.min(dim=-1)
                r_area         = -(min_dl > 3.0).float().mean(dim=-1)

                vel      = ego_xy[:, 1:] - ego_xy[:, :-1]
                acc      = vel[:, 1:] - vel[:, :-1]
                jerk     = acc[:, 1:] - acc[:, :-1]
                r_comfort = -torch.norm(jerk, dim=-1).mean(dim=-1) if jerk.numel() > 0 else torch.zeros(B_cur, device=ego_xy.device)

            # ── Per-sample metric accumulation ────────────────────────
            for i in range(B_cur):
                pred         = best_traj[i].cpu().numpy()
                gt           = gt_traj[i].cpu().numpy()
                mode_label   = mode_labels[i].item()
                chosen_mode  = best_idx[i].item()
                all_trajs_np = all_trajs[i].cpu().numpy()

                diff = pred[:, :2] - gt[:, :2]
                stats['ade'].append(float(np.mean(np.linalg.norm(diff, axis=1))))
                stats['fde'].append(float(np.linalg.norm(pred[-1, :2] - gt[-1, :2])))
                stats['l_gen'].append(float(np.sum(np.abs(pred - gt))))
                stats['l_sel'].append(float(-log_probs[i, mode_label]))

                order   = torch.argsort(probs[i], descending=True)
                gt_rank = int((order == mode_label).nonzero(as_tuple=True)[0].item()) + 1
                stats['mode_correct'].append(chosen_mode == mode_label)
                stats['top5_correct'].append(gt_rank <= 5)
                stats['gt_prob'].append(float(probs[i, mode_label]))
                stats['gt_rank'].append(gt_rank)

                lat_r, lon_r = _compute_consistent_ratio(all_trajs_np, gt)
                stats['consistent_lat'].append(lat_r)
                stats['consistent_lon'].append(lon_r)

                agents_now_np  = batch['agents_now'][i].cpu().numpy()
                agents_mask_np = batch['agents_history_mask'][i].cpu().numpy()
                ml_np          = batch['map_lanes'][i].cpu().numpy()
                mlm_np         = batch['map_lanes_mask'][i].cpu().numpy()
                cm, cmin, cmax, am, amin, amax = _compute_ol_col_area(
                    all_trajs_np, agents_now_np, agents_mask_np, ml_np, mlm_np)
                stats['col_mean'].append(cm);  stats['col_min'].append(cmin);  stats['col_max'].append(cmax)
                stats['area_mean'].append(am); stats['area_min'].append(amin); stats['area_max'].append(amax)

                if has_transition:
                    stats['entropy'].append(float(entropies[i].mean()))
                    stats['value_mean'].append(float(values[i].mean()))
                    stats['reward_total'].append(float(rewards[i].mean()))
                    stats['reward_displacement'].append(float(r_disp[i]))
                    stats['reward_collision'].append(float(r_col[i]))
                    stats['reward_drivable'].append(float(r_area[i]))
                    stats['reward_comfort'].append(float(r_comfort[i]))

            done += B_cur
            if done % 100 < args.batch_size or done >= n_eval:
                elapsed = time.time() - t0
                rl_str = ""
                if has_transition and stats['reward_total']:
                    rl_str = (f"  R={np.mean(stats['reward_total']):.3f}  "
                              f"H={np.mean(stats['entropy']):.3f}")
                print(f"  [{done}/{n_eval}]  {done/elapsed:.1f} samples/s  "
                      f"ADE={np.mean(stats['ade']):.3f}  "
                      f"L_gen={np.mean(stats['l_gen']):.1f}  "
                      f"L_sel={np.mean(stats['l_sel']):.2f}{rl_str}")

    elapsed = time.time() - t0
    print(f"\n[Eval Stage C] Done in {elapsed:.1f}s ({n_eval/elapsed:.1f} samples/s)\n")

    # ── Aggregate ────────────────────────────────────────────────────────────
    r = {}
    for k, v in stats.items():
        if not v:
            continue
        if k in ('mode_correct', 'top5_correct'):
            r[k] = float(np.mean(v)) * 100
        else:
            r[k] = float(np.mean(v))
    r['median_gt_rank'] = int(np.median(stats['gt_rank']))
    r['col_min'] = float(np.mean(stats['col_min']))
    r['col_max'] = float(np.mean(stats['col_max']))
    r['area_min'] = float(np.mean(stats['area_min']))
    r['area_max'] = float(np.mean(stats['area_max']))

    il = PAPER_IL
    rl = PAPER_RL

    # ── Print results ────────────────────────────────────────────────────────
    W = 78
    print("=" * W)
    print(f"  STAGE C (RL) EVALUATION — {n_eval} samples on {os.path.basename(args.cache)}")
    print(f"  Checkpoint: {args.checkpoint} (epoch {epoch})")
    print("=" * W)

    def _row(label, ours, paper_rl, paper_il):
        print(f"  {label:<35} {ours:>12} {paper_rl:>15} {paper_il:>15}")

    def _sep():
        print(f"  {'─'*74}")

    # Table 4 open-loop
    print(f"\n  Table 4 — Open-loop metrics")
    _sep()
    _row("Metric", "Ours", "Paper RL-best", "Paper IL-best")
    _sep()
    _row("L_gen (generator loss)", f"{r['l_gen']:.1f}", f"{rl['L_gen']:.1f}", f"{il['L_gen']:.1f}")
    _row("L_sel (selector CE loss)", f"{r['l_sel']:.2f}", f"{rl['L_sel']:.2f}", f"{il['L_sel']:.2f}")
    _row("ADE (m)", f"{r['ade']:.3f}", "—", "—")
    _row("FDE (m)", f"{r['fde']:.3f}", "—", "—")
    _row("Mode accuracy top-1 (%)", f"{r['mode_correct']:.2f}", "—", "—")
    _row("Mode accuracy top-5 (%)", f"{r['top5_correct']:.2f}", "—", "—")
    _row("Median GT rank", f"{r['median_gt_rank']}", f"— / {cfg.N_MODES}", f"— / {cfg.N_MODES}")

    # Table 8
    print(f"\n  Table 8 — Consistent Ratio")
    _sep()
    _row("Metric", "Ours", "Paper RL-best", "Paper IL-best")
    _sep()
    _row("Consistent Ratio Lat (%)", f"{r['consistent_lat']:.2f}", f"{rl['consistent_lat']:.2f}", f"{il['consistent_lat']:.2f}")
    _row("Consistent Ratio Lon (%)", f"{r['consistent_lon']:.2f}", f"{rl['consistent_lon']:.2f}", f"{il['consistent_lon']:.2f}")

    # Table 10 open-loop
    print(f"\n  Table 10 — Open-loop Collision & Area")
    _sep()
    _row("Metric", "Ours", "Paper RL-best", "Paper IL-best")
    _sep()
    _row("Col Mean", f"{r['col_mean']:.2f}", f"{rl['col_mean']:.2f}", f"{il['col_mean']:.2f}")
    _row("Col [Min, Max]", f"[{r['col_min']:.2f}, {r['col_max']:.2f}]",
         f"[{rl['col_min']:.2f}, {rl['col_max']:.2f}]", f"[{il['col_min']:.2f}, {il['col_max']:.2f}]")
    _row("Area Mean", f"{r['area_mean']:.2f}", f"{rl['area_mean']:.2f}", f"{il['area_mean']:.2f}")
    _row("Area [Min, Max]", f"[{r['area_min']:.2f}, {r['area_max']:.2f}]",
         f"[{rl['area_min']:.2f}, {rl['area_max']:.2f}]", f"[{il['area_min']:.2f}, {il['area_max']:.2f}]")

    # RL diagnostics
    if has_transition:
        print(f"\n  RL Diagnostics (stochastic rollout with GT mode)")
        _sep()
        print(f"  {'Metric':<35} {'Value':>12}")
        _sep()
        print(f"  {'Policy entropy (mean)' :<35} {r.get('entropy', 0)  :>12.4f}")
        print(f"  {'Value estimate (mean)' :<35} {r.get('value_mean', 0):>12.4f}")
        print(f"  {'Reward total (mean)'   :<35} {r.get('reward_total', 0):>12.4f}")
        _sep()
        print(f"  {'R_displacement'        :<35} {r.get('reward_displacement', 0):>12.4f}")
        print(f"  {'R_collision'           :<35} {r.get('reward_collision', 0):>12.4f}")
        print(f"  {'R_drivable'            :<35} {r.get('reward_drivable', 0):>12.4f}")
        print(f"  {'R_comfort'             :<35} {r.get('reward_comfort', 0):>12.4f}")
        if hasattr(model.policy, 'action_log_std'):
            std = model.policy.action_log_std.exp().detach().cpu().numpy()
            print(f"  {'Action std (x, y, yaw)':<35} {std[0]:>5.4f}, {std[1]:.4f}, {std[2]:.4f}")

    # Closed-loop reminder
    print(f"\n  Tables 1, 2, 4, 7, 9, 11 — Closed-loop (requires nuPlan simulator)")
    _sep()
    print(f"  Run:")
    print(f"    python scripts/eval_nuplan.py --checkpoint {args.checkpoint} --split test14-random")
    print(f"    python scripts/eval_nuplan.py --checkpoint {args.checkpoint} --split val14")
    _sep()
    print(f"\n  Paper RL-best closed-loop (Table 4): CLS-NR={rl['CLS_NR']}, "
          f"S-CR={rl['S_CR']}, S-Area={rl['S_Area']}, S-PR={rl['S_PR']}, S-Comfort={rl['S_Comfort']}")
    print(f"  Paper IL-best closed-loop (Table 4): CLS-NR={il['CLS_NR']}, "
          f"S-CR={il['S_CR']}, S-Area={il['S_Area']}, S-PR={il['S_PR']}, S-Comfort={il['S_Comfort']}")
    print("=" * W)


SPLIT_TO_CACHE = {
    'val14':          'checkpoints/stage_cache_val14.pt',
    'test14_random':  'checkpoints/stage_cache_test14_random.pt',
    'reduced_val14':  'checkpoints/stage_cache_reduced_val14.pt',
}


def parse_args():
    p = argparse.ArgumentParser(description="CarPlanner Stage C (RL) evaluation")
    p.add_argument('--checkpoint', required=True, help='Path to Stage C checkpoint')
    p.add_argument('--split', default='val14',
                   choices=list(SPLIT_TO_CACHE.keys()),
                   help='Benchmark split (default: val14)')
    p.add_argument('--cache', default=None,
                   help='Override cache path (default: auto from --split)')
    p.add_argument('--stage_a_ckpt', default='checkpoints/stage_a_best.pt',
                   help='Stage A checkpoint for reward computation (default: checkpoints/stage_a_best.pt)')
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
