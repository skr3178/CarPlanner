"""
Coordinate-system sanity check for Stage-B-seed predictions.

Tests three things on a handful of val14 scenes:
  1. Scale: are predicted positions in metres and per-step displacements
     plausible for driving (typically 1-6 m/step at 0.4s/step)?
  2. Yaw convention: does the predicted yaw at each step match the heading
     implied by atan2(dy, dx) of consecutive predicted positions?
  3. Frame: does ego start at ~(0,0) and do nearby lanes have headings
     roughly aligned with ego at t=0?

Runs on CPU only — does not interfere with a GPU training process.
"""
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

import config as cfg
from data_loader import PreextractedDataset
from model import CarPlanner

DT = 0.1  # closed-loop sim step time (Hz=10), per scripts/eval_closedloop_gpu.py


def normalize_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def draw_scene_with_coords(ax, sample, pred, gt, idx):
    """Detailed BEV with scale grid, yaw arrows, lane headings."""
    map_lanes = sample['map_lanes'].cpu().numpy()
    map_mask = sample['map_lanes_mask'].cpu().numpy()
    agents = sample['agents_now'].cpu().numpy()
    ag_mask = sample['agents_history_mask'].cpu().numpy()

    # Lanes
    for i in range(len(map_mask)):
        if map_mask[i] < 0.5:
            continue
        lane = map_lanes[i]
        valid = np.abs(lane[:, 0]) + np.abs(lane[:, 1]) > 1e-6
        if valid.sum() < 2:
            continue
        c = lane[valid, 0:2]
        ax.plot(c[:, 0], c[:, 1], '-', color='#909090', linewidth=1.0, alpha=0.85)
        # Lane edges if present
        if lane.shape[1] >= 21:
            l = lane[valid, 9:11]
            r = lane[valid, 18:20]
            if np.abs(l).sum() > 1e-3:
                ax.plot(l[:, 0], l[:, 1], '--', color='#bbbbbb', linewidth=0.6, alpha=0.6)
            if np.abs(r).sum() > 1e-3:
                ax.plot(r[:, 0], r[:, 1], '--', color='#bbbbbb', linewidth=0.6, alpha=0.6)

    # Other agents
    for i in range(len(ag_mask)):
        if ag_mask[i] < 0.5:
            continue
        ax.plot(agents[i, 0], agents[i, 1], 's', color='#3b6ea8',
                markersize=4, alpha=0.7)

    # GT trajectory with yaw arrows
    ax.plot(gt[:, 0], gt[:, 1], '-', color='#1f9e3c', linewidth=2.2, label='GT', zorder=5)
    for t in range(0, gt.shape[0], 2):
        x, y, yaw = gt[t, 0], gt[t, 1], gt[t, 2]
        ax.arrow(x, y, 0.8 * np.cos(yaw), 0.8 * np.sin(yaw),
                 head_width=0.3, head_length=0.3, fc='#1f9e3c', ec='#1f9e3c',
                 alpha=0.6, zorder=6)

    # Predicted trajectory with yaw arrows
    ax.plot(pred[:, 0], pred[:, 1], '-', color='#d23a2c', linewidth=1.8, label='pred', zorder=7)
    for t in range(0, pred.shape[0], 2):
        x, y, yaw = pred[t, 0], pred[t, 1], pred[t, 2]
        ax.arrow(x, y, 0.8 * np.cos(yaw), 0.8 * np.sin(yaw),
                 head_width=0.3, head_length=0.3, fc='#d23a2c', ec='#d23a2c',
                 alpha=0.6, zorder=8)

    # Ego at origin
    ax.plot(0, 0, '^', color='black', markersize=10, zorder=10)
    ax.arrow(0, 0, 1.5, 0, head_width=0.5, head_length=0.5,
             fc='black', ec='black', zorder=11)  # ego heading = +x axis

    # 5m grid + axis labels
    pts = np.concatenate([gt[:, :2], pred[:, :2], np.array([[0, 0]])], axis=0)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    rng = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 20.0) / 2 + 5.0
    ax.set_xlim(cx - rng, cx + rng)
    ax.set_ylim(cy - rng, cy + rng)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x [m] (ego forward)')
    ax.set_ylabel('y [m] (ego left)')
    ax.set_title(f'val14 #{idx}', fontsize=10)


def numeric_check(pred, gt, tag, lines):
    """Print per-step displacements and yaw vs heading consistency."""
    pred_xy = pred[:, :2]
    gt_xy = gt[:, :2]

    pred_steps = np.linalg.norm(np.diff(pred_xy, axis=0), axis=-1)
    gt_steps = np.linalg.norm(np.diff(gt_xy, axis=0), axis=-1)

    pred_total = np.linalg.norm(pred_xy[-1] - pred_xy[0])
    gt_total = np.linalg.norm(gt_xy[-1] - gt_xy[0])

    # Yaw consistency: predicted yaw[t] vs atan2(pred[t+1]-pred[t])
    pred_dxy = np.diff(pred_xy, axis=0)
    pred_heading_implied = np.arctan2(pred_dxy[:, 1], pred_dxy[:, 0])
    pred_yaw = pred[:-1, 2]
    yaw_err = normalize_angle(pred_yaw - pred_heading_implied)
    yaw_err_deg = np.degrees(np.abs(yaw_err))

    lines.append(f'  [{tag}]')
    lines.append(f'    step displacements (m): ' + ' '.join(f'{s:.2f}' for s in pred_steps))
    lines.append(f'    GT step displacements:  ' + ' '.join(f'{s:.2f}' for s in gt_steps))
    lines.append(f'    total displacement:    pred={pred_total:.2f}m  gt={gt_total:.2f}m')
    lines.append(f'    mean per-step speed:   pred={pred_steps.mean()/DT:.2f} m/s  gt={gt_steps.mean()/DT:.2f} m/s')
    lines.append(f'    yaw vs heading-from-Δxy mean error: {yaw_err_deg.mean():.2f}°  '
                 f'max: {yaw_err_deg.max():.2f}°')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='checkpoints/stage_b_20260424_140710/stage_b_epoch_020.pt')
    p.add_argument('--cache', default='checkpoints/stage_cache_val14.pt')
    p.add_argument('--n_samples', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', default='check_coord_system.png')
    args = p.parse_args()

    device = torch.device('cpu')
    print(f'[CoordCheck] Device: {device}  (CPU-only, no GPU contention)')

    cfg.set_stage('b')
    dataset = PreextractedDataset(args.cache)
    print(f'[CoordCheck] Loaded {len(dataset)} val14 samples')

    rng = np.random.default_rng(args.seed)
    indices = sorted(rng.choice(len(dataset), size=args.n_samples, replace=False).tolist())
    print(f'[CoordCheck] Picked indices: {indices}')

    model = CarPlanner().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    model._transition_loaded = True
    print(f'[CoordCheck] Stage-B loaded (epoch {ckpt.get("epoch", "?")})  '
          f'param count: {sum(p.numel() for p in model.parameters()):,}')
    print(f'[CoordCheck] DT = {DT}s, T_FUTURE = {cfg.T_FUTURE} steps  →  horizon {DT * cfg.T_FUTURE:.1f}s')

    n = len(indices)
    fig, axes = plt.subplots(n, 1, figsize=(10, 8 * n))
    if n == 1:
        axes = [axes]
    report_lines = []

    for r, idx in enumerate(indices):
        sample = dataset[idx]
        batch = {kk: v.unsqueeze(0).to(device) for kk, v in sample.items()}
        with torch.no_grad():
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
        pred = best_traj[0].cpu().numpy()
        gt = sample['gt_trajectory'].cpu().numpy()

        report_lines.append(f'val14 #{idx}  (mode pred={best_idx.item()}, GT={int(sample["mode_label"].item())})')
        numeric_check(pred, gt, 'PRED', report_lines)
        numeric_check(gt, gt, 'GT  ', report_lines)
        report_lines.append('')

        draw_scene_with_coords(axes[r], sample, pred, gt, idx)
        if r == 0:
            axes[r].legend(loc='upper left', fontsize=9)

    fig.suptitle(
        'Coordinate-system check — Stage-B-seed on val14\n'
        'Green=GT, Red=pred, arrows=yaw, ▲=ego at t=0 (heading +x), 5m grid',
        fontsize=11, y=0.997)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(args.output, dpi=140, bbox_inches='tight')
    print(f'[CoordCheck] Saved: {args.output}')

    print()
    print('\n'.join(report_lines))


if __name__ == '__main__':
    main()
