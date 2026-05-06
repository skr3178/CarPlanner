"""Phase 0 diagnostic: visualize and quantify how the policy consumes route
information across the 60 candidate trajectories on val14 samples.

Per sample, saves:
  diag_outputs/sample_<idx>.png      4 panels (cached routes, trimmed at t=0,
                                     trimmed at t=7, candidate trajectories)
  diag_outputs/sample_<idx>.json     numeric checks (endpoint spread, route
                                     distinctness, lat-bin diversity)

Usage:
    paper/.venv/bin/python scripts/diag_route_usage.py \\
        --checkpoint checkpoints/stage_b_best.pt \\
        --indices 0,200,500,800,1100 \\
        --out diag_outputs

Numeric checks paired with each picture (the visualization-isn't-enough part):
  - Cached route y-range per lat bin (Panel A)
  - Trimmed route y-range per lat bin at step 0 vs step 7 (Panels B, C)
  - Endpoint y-spread across lat bins for the GT-lon row (Panel D)
  - Whether route polylines themselves change between step 0 and step 7
    (paper §3.3 says they should be re-transformed to current ego frame; the
    current code only uses the trim *index* in current frame, the features
    stay in initial-ego frame)

Reads:
  checkpoints/stage_cache_val14.pt
  any Stage B checkpoint (.pt with 'model' key)
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from model import CarPlanner

LAT_COLORS = ['#1f77b4', '#2ca02c', '#bbbbbb', '#ff7f0e', '#d62728']  # left → right
LAT_LABELS = ['lat 0 (far-L)', 'lat 1 (L)', 'lat 2 (lane-keep)', 'lat 3 (R)', 'lat 4 (far-R)']


# ───────────────────────────────────────────────────────────────────────────
# Replicate the policy's route-trim logic externally so we can show the
# trimmed segment without instrumenting model.py.

def trim_routes_at_pose(route_polylines: torch.Tensor,
                        route_mask: torch.Tensor,
                        ego_x: float, ego_y: float, ego_h: float):
    """Mirrors model.py:849-872 exactly.

    route_polylines: (N_lat, N_r, D)         in initial ego frame (cache state)
    route_mask:      (N_lat,)
    ego pose:        scalar — current ego pose in initial ego frame

    Returns:
      trimmed: (N_lat, K_r, D) — same logic as the policy: trim index uses
               the current ego pose, but the features are still in INITIAL
               ego frame (this is the bug — the route features are never
               re-transformed).
      closest_idx: (N_lat,)
    """
    rp_xy = route_polylines[..., :2]                       # (N_lat, N_r, 2)
    dx = rp_xy[..., 0] - ego_x
    dy = rp_xy[..., 1] - ego_y
    cos_h, sin_h = math.cos(-ego_h), math.sin(-ego_h)
    rx = cos_h * dx - sin_h * dy
    ry = sin_h * dx + cos_h * dy
    dist = torch.sqrt(rx**2 + ry**2)
    closest_idx = dist.argmin(dim=-1)                       # (N_lat,)

    N_r = route_polylines.size(1)
    K_r = max(1, N_r // 4)
    trimmed = torch.zeros(route_polylines.size(0), K_r,
                          route_polylines.size(-1),
                          dtype=route_polylines.dtype)
    for r in range(route_polylines.size(0)):
        if route_mask[r] < 0.5:
            continue
        s = int(closest_idx[r].item())
        e = min(s + K_r, N_r)
        trimmed[r, :e - s] = route_polylines[r, s:e]
    return trimmed, closest_idx


# ───────────────────────────────────────────────────────────────────────────
# Numeric checks

def compute_metrics(routes, route_mask, all_trajs, gt_traj, mode_label):
    """All distance / spread metrics for a single sample.

    routes:      (N_LAT, N_r, D)    initial ego frame
    route_mask:  (N_LAT,)
    all_trajs:   (N_MODES, T, 3)    initial ego frame
    gt_traj:     (T, 3)
    mode_label:  int — assigned (lon, lat) at training
    """
    valid = route_mask > 0.5
    routes_xy = routes[..., :2]                            # (N_LAT, N_r, 2)

    # 1) Cached route y-range per lat bin
    y_ranges = {}
    for r in range(int(route_mask.size(0))):
        if not valid[r]:
            continue
        ys = routes_xy[r, :, 1]
        nonpad = (routes_xy[r, :, 0].abs() + routes_xy[r, :, 1].abs()) > 1e-6
        if nonpad.any():
            yvals = ys[nonpad]
            y_ranges[int(r)] = [float(yvals.min()), float(yvals.max())]

    # 2) Pairwise route distinctness (mean nearest-point distance between bins)
    pair_dist = {}
    valid_bins = [int(r) for r in range(int(route_mask.size(0))) if valid[r]]
    for i in range(len(valid_bins)):
        for j in range(i + 1, len(valid_bins)):
            a, b = valid_bins[i], valid_bins[j]
            pa = routes_xy[a]
            pb = routes_xy[b]
            nonpad_a = (pa.abs().sum(dim=-1) > 1e-6)
            nonpad_b = (pb.abs().sum(dim=-1) > 1e-6)
            if nonpad_a.any() and nonpad_b.any():
                pa_, pb_ = pa[nonpad_a], pb[nonpad_b]
                d = torch.cdist(pa_.unsqueeze(0), pb_.unsqueeze(0))[0]
                pair_dist[f"{a}-{b}"] = float(d.min(dim=1).values.mean())

    # 3) Trimmed route y-range at step 0 vs step 7 (using GT ego pose at each
    #    step as the anchor)
    trimmed_step0, _ = trim_routes_at_pose(routes, route_mask, 0., 0., 0.)
    gt_x7, gt_y7, gt_h7 = float(gt_traj[-1, 0]), float(gt_traj[-1, 1]), float(gt_traj[-1, 2])
    trimmed_step7, _ = trim_routes_at_pose(routes, route_mask, gt_x7, gt_y7, gt_h7)

    def _yrange_per_bin(trimmed):
        out = {}
        for r in range(int(route_mask.size(0))):
            if not valid[r]:
                continue
            ys = trimmed[r, :, 1]
            nonpad = (trimmed[r, :, 0].abs() + trimmed[r, :, 1].abs()) > 1e-6
            if nonpad.any():
                yvals = ys[nonpad]
                out[int(r)] = [float(yvals.min()), float(yvals.max())]
        return out

    trimmed_y_step0 = _yrange_per_bin(trimmed_step0)
    trimmed_y_step7 = _yrange_per_bin(trimmed_step7)

    # 4) Endpoint y per (lon, lat) and y-spread across lat bins for GT-lon row
    N_LON, N_LAT = cfg.N_LON, cfg.N_LAT
    endpoints = all_trajs[:, -1, :2].cpu().numpy()         # (N_MODES, 2)
    end_x = endpoints[:, 0].reshape(N_LON, N_LAT)
    end_y = endpoints[:, 1].reshape(N_LON, N_LAT)
    gt_lon_idx = mode_label // N_LAT
    gt_lat_idx = mode_label % N_LAT

    end_y_at_gt_lon = end_y[gt_lon_idx, :].tolist()        # 5 values
    spread_lat_for_gt_lon = float(np.std(end_y[gt_lon_idx, :]))

    # Mean lat-bin spread averaged over all lon bins
    mean_spread = float(np.mean([np.std(end_y[lon_idx, :]) for lon_idx in range(N_LON)]))

    return {
        'mode_label': int(mode_label),
        'gt_lon': int(gt_lon_idx), 'gt_lat': int(gt_lat_idx),
        'cache_y_range_per_lat':       y_ranges,
        'route_pair_min_distance':     pair_dist,
        'trimmed_y_range_step0':       trimmed_y_step0,
        'trimmed_y_range_step7':       trimmed_y_step7,
        'endpoint_y_at_gt_lon_row':    end_y_at_gt_lon,
        'endpoint_y_lat_spread_gt_lon': spread_lat_for_gt_lon,
        'endpoint_y_lat_spread_mean':  mean_spread,
        'gt_endpoint_xy':              [float(gt_traj[-1, 0]), float(gt_traj[-1, 1])],
    }


# ───────────────────────────────────────────────────────────────────────────
# Visualization

def plot_sample(sample_idx, routes, route_mask, agents_now, agents_mask,
                gt_traj, all_trajs, metrics, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    fig.suptitle(
        f"Sample {sample_idx}    GT mode = (lon={metrics['gt_lon']}, "
        f"lat={metrics['gt_lat']})    "
        f"endpoint y-spread across lat (GT-lon row) = {metrics['endpoint_y_lat_spread_gt_lon']:.3f}",
        fontsize=11)

    routes_np = routes.cpu().numpy()
    rmask_np  = route_mask.cpu().numpy()
    gt_np     = gt_traj.cpu().numpy()
    agt_np    = agents_now.cpu().numpy()
    am_np     = agents_mask.cpu().numpy()
    at_np     = all_trajs.cpu().numpy()                    # (N_MODES, T, 3)

    def _draw_agents_and_ego(ax):
        ax.scatter([0], [0], marker='s', s=120, c='black', zorder=5, label='ego (t=0)')
        ax.plot(gt_np[:, 0], gt_np[:, 1], 'k-', lw=2.5, zorder=4, label='GT')
        for i in range(agt_np.shape[0]):
            if am_np[i] > 0.5:
                ax.scatter(agt_np[i, 0], agt_np[i, 1],
                           marker='o', s=40, c='gray', alpha=0.6, zorder=3)

    # Panel A — full cached routes
    ax = axes[0, 0]
    ax.set_title("Panel A — cached route polylines (initial ego frame)")
    _draw_agents_and_ego(ax)
    for r in range(int(rmask_np.shape[0])):
        if rmask_np[r] < 0.5:
            continue
        xy = routes_np[r, :, :2]
        nonpad = (np.abs(xy[:, 0]) + np.abs(xy[:, 1])) > 1e-6
        if nonpad.any():
            ax.plot(xy[nonpad, 0], xy[nonpad, 1], '-o',
                    color=LAT_COLORS[r], lw=1.8, markersize=4,
                    label=LAT_LABELS[r], alpha=0.85)
    ax.set_xlabel('x [m] (forward)'); ax.set_ylabel('y [m]'); ax.grid(True, alpha=0.3)
    ax.set_aspect('equal'); ax.legend(loc='upper left', fontsize=8)

    # Panel B — trimmed routes at step 0
    trimmed_0, _ = trim_routes_at_pose(routes, route_mask, 0., 0., 0.)
    t0_np = trimmed_0.cpu().numpy()
    ax = axes[0, 1]
    ax.set_title("Panel B — trimmed routes at step 0 (this is what the policy uses at t=0)")
    _draw_agents_and_ego(ax)
    for r in range(t0_np.shape[0]):
        if rmask_np[r] < 0.5:
            continue
        xy = t0_np[r, :, :2]
        nonpad = (np.abs(xy[:, 0]) + np.abs(xy[:, 1])) > 1e-6
        if nonpad.any():
            ax.plot(xy[nonpad, 0], xy[nonpad, 1], '-o',
                    color=LAT_COLORS[r], lw=2.5, markersize=6,
                    label=LAT_LABELS[r], alpha=0.95)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.grid(True, alpha=0.3)
    ax.set_aspect('equal'); ax.legend(loc='upper left', fontsize=8)

    # Panel C — trimmed routes at step 7 (using GT pose as anchor)
    trimmed_7, _ = trim_routes_at_pose(
        routes, route_mask,
        float(gt_np[-1, 0]), float(gt_np[-1, 1]), float(gt_np[-1, 2]),
    )
    t7_np = trimmed_7.cpu().numpy()
    ax = axes[1, 0]
    ax.set_title("Panel C — trimmed routes at step 7 (anchor = GT ego pose at t=7)")
    _draw_agents_and_ego(ax)
    ax.scatter([gt_np[-1, 0]], [gt_np[-1, 1]],
               marker='s', s=120, c='red', zorder=5, label='ego (t=7, GT)')
    for r in range(t7_np.shape[0]):
        if rmask_np[r] < 0.5:
            continue
        xy = t7_np[r, :, :2]
        nonpad = (np.abs(xy[:, 0]) + np.abs(xy[:, 1])) > 1e-6
        if nonpad.any():
            ax.plot(xy[nonpad, 0], xy[nonpad, 1], '-o',
                    color=LAT_COLORS[r], lw=2.5, markersize=6,
                    label=LAT_LABELS[r], alpha=0.95)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.grid(True, alpha=0.3)
    ax.set_aspect('equal'); ax.legend(loc='upper left', fontsize=8)

    # Panel D — 5 candidate trajectories at GT-lon row, plus full 60 in light gray
    ax = axes[1, 1]
    gt_lon = metrics['gt_lon']
    gt_lat = metrics['gt_lat']
    ax.set_title(
        f"Panel D — candidate trajectories: GT-lon row only "
        f"(lon={gt_lon}, all 5 lat bins); GT in black"
    )
    # background: all 60 candidates faint
    for m in range(at_np.shape[0]):
        ax.plot(at_np[m, :, 0], at_np[m, :, 1], '-', color='lightgray',
                lw=0.6, alpha=0.5, zorder=1)
    # foreground: GT-lon row
    for lat_idx in range(cfg.N_LAT):
        m = gt_lon * cfg.N_LAT + lat_idx
        ax.plot(at_np[m, :, 0], at_np[m, :, 1], '-o',
                color=LAT_COLORS[lat_idx], lw=2.0, markersize=4,
                label=f"{LAT_LABELS[lat_idx]} (m={m})", zorder=3, alpha=0.95)
        ax.scatter(at_np[m, -1, 0], at_np[m, -1, 1],
                   color=LAT_COLORS[lat_idx], s=60, marker='X', zorder=4)
    # GT
    ax.plot(gt_np[:, 0], gt_np[:, 1], 'k-', lw=2.5, zorder=5, label='GT')
    ax.scatter([0], [0], marker='s', s=120, c='black', zorder=6)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.grid(True, alpha=0.3)
    ax.set_aspect('equal'); ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints/stage_b_best.pt')
    p.add_argument('--cache',      default='checkpoints/stage_cache_val14.pt')
    p.add_argument('--indices',    default='0,200,500,800,1100',
                   help='Comma-separated sample indices in val14 cache')
    p.add_argument('--out',        default='diag_outputs')
    p.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = [int(s) for s in args.indices.split(',')]

    device = torch.device(args.device)
    cfg.set_stage('b')

    print(f'[diag] device={device}  cache={args.cache}  ckpt={args.checkpoint}')
    data = torch.load(args.cache, map_location='cpu', weights_only=False)
    n_total = data['mode_label'].size(0)
    print(f'[diag] val14 size: {n_total}')

    model = CarPlanner().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    print(f"[diag] loaded model (epoch {ckpt.get('epoch', '?')})")

    # Helpers
    def _slice(name, idx):
        return data[name][idx:idx+1].to(device) if name in data else None

    summary = {}
    for sidx in indices:
        if sidx < 0 or sidx >= n_total:
            print(f'[diag] skipping out-of-range index {sidx}')
            continue
        # Run forward_inference_fast on a single sample to get all 60 trajectories
        with torch.no_grad():
            mode_logits, all_trajs, _, _ = model.forward_inference_fast(
                agents_now        = _slice('agents_now',        sidx),
                agents_mask       = _slice('agents_mask',       sidx),
                map_lanes         = _slice('map_lanes',         sidx),
                map_lanes_mask    = _slice('map_lanes_mask',    sidx),
                agents_history    = _slice('agents_history',    sidx),
                ego_history       = _slice('ego_history',       sidx),
                map_polygons      = _slice('map_polygons',      sidx),
                map_polygons_mask = _slice('map_polygons_mask', sidx),
                route_polylines   = _slice('route_polylines',   sidx),
                route_mask        = _slice('route_mask',        sidx),
            )
        all_trajs = all_trajs[0].cpu()                     # (N_MODES, T, 3)
        routes    = data['route_polylines'][sidx]          # (N_LAT, N_r, D)
        rmask     = data['route_mask'][sidx]               # (N_LAT,)
        gt_traj   = data['gt_trajectory'][sidx]            # (T, 3)
        agents_now = data['agents_now'][sidx]              # (N_AGENTS, D_AGENT)
        agents_mask = data['agents_mask'][sidx] if 'agents_mask' in data else (
            data['agents_history_mask'][sidx])
        mode_label = int(data['mode_label'][sidx].item())

        m = compute_metrics(routes, rmask, all_trajs, gt_traj, mode_label)
        png_path  = out_dir / f'sample_{sidx:04d}.png'
        json_path = out_dir / f'sample_{sidx:04d}.json'
        plot_sample(sidx, routes, rmask, agents_now, agents_mask,
                    gt_traj, all_trajs, m, png_path)
        json_path.write_text(json.dumps(m, indent=2))
        print(f"  sample {sidx}: GT mode=(lon={m['gt_lon']}, lat={m['gt_lat']})  "
              f"lat-spread@gt-lon={m['endpoint_y_lat_spread_gt_lon']:.3f}  "
              f"lat-spread mean={m['endpoint_y_lat_spread_mean']:.3f}  "
              f"valid bins={list(m['cache_y_range_per_lat'].keys())}  "
              f"→ {png_path.name}")
        summary[sidx] = {
            'gt_mode':                m['mode_label'],
            'gt_lon':                 m['gt_lon'],
            'gt_lat':                 m['gt_lat'],
            'lat_spread_gt_lon':      m['endpoint_y_lat_spread_gt_lon'],
            'lat_spread_mean':        m['endpoint_y_lat_spread_mean'],
            'route_pair_dist':        m['route_pair_min_distance'],
        }

    # Aggregate report
    if summary:
        spreads = [s['lat_spread_mean'] for s in summary.values()]
        gt_spreads = [s['lat_spread_gt_lon'] for s in summary.values()]
        print()
        print('=== aggregate ===')
        print(f"  endpoint_y_lat_spread_mean      mean over samples = {np.mean(spreads):.3f}")
        print(f"  endpoint_y_lat_spread_gt_lon    mean over samples = {np.mean(gt_spreads):.3f}")
        print(f"  per-sample summary saved to    {out_dir}/aggregate.json")
        (out_dir / 'aggregate.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
