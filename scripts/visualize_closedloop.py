"""
BEV (bird's-eye view) visualization for CarPlanner closed-loop evaluation.

Produces paper-style Figure 5 renderings: ego vehicle, traffic agents,
map lanes, and planned trajectory at selected simulation timesteps.

Usage:
    cd /media/skr/storage/autoresearch/CarPlanner_Implementation
    python scripts/visualize_closedloop.py \
        --cache checkpoints/test14_random_cache.pt \
        --checkpoint checkpoints/stage_c_best.pt \
        --scenarios 0,1,2 \
        --timesteps 0,50,100,120,130
"""

import os
import sys
import argparse
import math

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Affine2D

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

import config as cfg
from model import CarPlanner
from eval_closedloop_gpu import (
    prestack, build_batch_inputs, ego_to_world_batch, compute_metrics_batch,
)

# Constants (mirror eval script)
T_SIM        = 150
DT           = 0.1
REPLAN_EVERY = cfg.T_FUTURE
N_AGENTS     = cfg.N_AGENTS
N_LANES      = cfg.N_LANES
N_PTS        = cfg.N_LANE_POINTS

# Metric weights
W_COLLISION  = 0.5
W_DRIVABLE   = 0.2
W_COMFORT    = 0.1
W_PROGRESS   = 0.2

# Default ego box (meters)
EGO_LENGTH = 4.5
EGO_WIDTH  = 2.0


# ---------------------------------------------------------------------------
# Simulation with trajectory capture
# ---------------------------------------------------------------------------

def run_scenario_with_viz(model, scenario, device):
    """
    Run closed-loop simulation for a single scenario, capturing the planned
    trajectory at every replan step.

    Returns:
        ego_trajs_world: (T_SIM, 2) — simulated ego xy in world frame
        planned_trajs:   dict mapping replan_step -> (T_FUTURE, 2) world-frame
        batch:           prestack output for metric computation
    """
    batch = prestack([scenario])
    B = 1
    n_iters = batch['n_iters']  # (1,)

    ego_states = batch['ego_gt'][:, 0, :3].clone()  # (1, 3)
    ego_trajs = torch.zeros(B, T_SIM, 2)
    ego_trajs[:, 0] = ego_states[:, :2]

    planned_trajs = {}  # step -> (T_FUTURE, 2) world

    t = 0
    while t < T_SIM - 1:
        steps_to_fill = min(REPLAN_EVERY, T_SIM - t - 1)

        inp = build_batch_inputs(batch, t, ego_states, device)
        with torch.no_grad():
            _, _, best_traj, _ = model.forward_inference(
                agents_now=inp['agents_now'],
                agents_mask=inp['agents_mask'],
                map_lanes=inp['map_lanes'],
                map_lanes_mask=inp['map_lanes_mask'],
                agents_history=inp['agents_history'],
            )

        pred_world = ego_to_world_batch(best_traj[:, :, :2].cpu(), ego_states)
        planned_trajs[t] = pred_world[0].clone().numpy()  # (T_FUTURE, 2)

        active = (t < n_iters - 1)
        for k in range(steps_to_fill):
            si = t + 1 + k
            if si < T_SIM:
                ego_trajs[:, si] = torch.where(
                    active.unsqueeze(1).expand(-1, 2),
                    pred_world[:, k],
                    ego_trajs[:, si - 1],
                )

        last_k = steps_to_fill - 1
        next_step = min(t + steps_to_fill, T_SIM - 1)
        ego_states[:, :2] = torch.where(
            active.unsqueeze(1).expand(-1, 2),
            pred_world[:, last_k],
            ego_states[:, :2],
        )
        ego_states[:, 2] = batch['ego_gt'][:, next_step, 2]

        t += steps_to_fill

    # Fill early-ending scenarios
    ni = n_iters[0].item()
    if ni < T_SIM:
        ego_trajs[0, ni:] = ego_trajs[0, ni - 1]

    # Replace NaN/Inf
    for t_s in range(1, T_SIM):
        if torch.isnan(ego_trajs[0, t_s]).any() or torch.isinf(ego_trajs[0, t_s]).any():
            ego_trajs[0, t_s] = ego_trajs[0, t_s - 1]

    return ego_trajs[0].numpy(), planned_trajs, batch


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_rotated_box(ax, cx, cy, yaw, length, width, color, alpha=0.8,
                     edgecolor='k', linewidth=0.6, zorder=5):
    """Draw a rotated rectangle centered at (cx, cy) with given yaw."""
    rect = patches.Rectangle(
        (-length / 2, -width / 2), length, width,
        linewidth=linewidth, edgecolor=edgecolor, facecolor=color, alpha=alpha,
    )
    t = (Affine2D()
         .rotate(yaw)
         .translate(cx, cy)
         + ax.transData)
    rect.set_transform(t)
    ax.add_patch(rect)

    # Small heading indicator (triangle at front)
    tri_len = length * 0.25
    tri_pts = np.array([
        [length / 2, 0],
        [length / 2 - tri_len, width * 0.3],
        [length / 2 - tri_len, -width * 0.3],
    ])
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    tri_rot = (rot @ tri_pts.T).T + np.array([cx, cy])
    tri_patch = patches.Polygon(tri_rot, closed=True, facecolor=color,
                                edgecolor=edgecolor, linewidth=0.4, alpha=alpha,
                                zorder=zorder + 1)
    ax.add_patch(tri_patch)


def transform_to_ego_frame(xy_world, ego_x, ego_y, ego_yaw):
    """Transform world-frame points to ego-centered, ego-heading-up frame."""
    dx = xy_world[..., 0] - ego_x
    dy = xy_world[..., 1] - ego_y
    cos_h = np.cos(-ego_yaw)
    sin_h = np.sin(-ego_yaw)
    xe = cos_h * dx - sin_h * dy
    ye = sin_h * dx + cos_h * dy
    return np.stack([xe, ye], axis=-1)


def transform_yaw_to_ego(yaw_world, ego_yaw):
    """Rotate yaw to ego frame."""
    return yaw_world - ego_yaw


def render_bev_frame(ax, t_step, scenario, ego_trajs_world, planned_trajs,
                     ego_gt, view_radius):
    """
    Render a single BEV frame at simulation step t_step.

    The view is ego-centered with ego heading pointing up (+y).
    """
    # Ego world state at this step
    ego_x = ego_trajs_world[t_step, 0]
    ego_y = ego_trajs_world[t_step, 1]
    ego_yaw = ego_gt[t_step, 2]  # use GT yaw

    # --- Map lanes ---
    map_center = scenario['map_center_world'].numpy()  # (NL, NP, 2)
    map_left   = scenario['map_left_world'].numpy()
    map_right  = scenario['map_right_world'].numpy()
    map_mask   = scenario['map_lanes_mask'].numpy()     # (NL,)

    for li in range(N_LANES):
        if map_mask[li] < 0.5:
            continue
        left_ego  = transform_to_ego_frame(map_left[li], ego_x, ego_y, ego_yaw)
        right_ego = transform_to_ego_frame(map_right[li], ego_x, ego_y, ego_yaw)
        center_ego = transform_to_ego_frame(map_center[li], ego_x, ego_y, ego_yaw)

        # Filled road surface between left and right boundaries
        road_poly = np.concatenate([left_ego, right_ego[::-1]], axis=0)
        road_patch = patches.Polygon(road_poly, closed=True,
                                     facecolor='#D8D8D0', edgecolor='none',
                                     alpha=0.6, zorder=0)
        ax.add_patch(road_patch)

        # Lane boundaries (solid lines)
        ax.plot(left_ego[:, 0], left_ego[:, 1], color='#666666',
                linewidth=1.5, linestyle='-', zorder=1)
        ax.plot(right_ego[:, 0], right_ego[:, 1], color='#666666',
                linewidth=1.5, linestyle='-', zorder=1)
        # Centerline (dashed)
        ax.plot(center_ego[:, 0], center_ego[:, 1], color='#999999',
                linewidth=1.0, linestyle='--', zorder=1)

    # --- GT ego trajectory (dashed) ---
    gt_xy = ego_gt[:, :2].numpy()
    gt_ego = transform_to_ego_frame(gt_xy, ego_x, ego_y, ego_yaw)
    ax.plot(gt_ego[:, 0], gt_ego[:, 1], color='#444444', linewidth=1.8,
            linestyle=':', alpha=0.5, zorder=2, label='GT')

    # --- Simulated ego trajectory up to this step ---
    sim_ego = transform_to_ego_frame(ego_trajs_world[:t_step + 1], ego_x, ego_y, ego_yaw)
    ax.plot(sim_ego[:, 0], sim_ego[:, 1], color='#2ca02c', linewidth=2.0,
            linestyle='-', alpha=0.8, zorder=3, label='Sim')

    # --- Planned trajectory at this replan step ---
    # Find the most recent replan step at or before t_step
    replan_steps = sorted(planned_trajs.keys())
    recent_replan = None
    for rs in replan_steps:
        if rs <= t_step:
            recent_replan = rs
    if recent_replan is not None:
        plan_world = planned_trajs[recent_replan]  # (T_FUTURE, 2)
        plan_ego = transform_to_ego_frame(plan_world, ego_x, ego_y, ego_yaw)
        # Prepend current ego position for visual continuity
        plan_full = np.concatenate([[[0.0, 0.0]], plan_ego], axis=0)
        ax.plot(plan_full[:, 0], plan_full[:, 1], color='#1f77b4',
                linewidth=2.5, linestyle='-', zorder=4, label='Plan')
        # Dots at plan waypoints
        ax.scatter(plan_ego[:, 0], plan_ego[:, 1], color='#1f77b4',
                   s=20, zorder=4, edgecolors='white', linewidths=0.5)

    # --- Traffic agents ---
    agents_world = scenario['agents_world'][t_step].numpy()  # (N, 5)
    agents_valid = scenario['agents_valid'][t_step].numpy()   # (N,)
    agents_size  = scenario['agents_size'].numpy()            # (N, 3)

    for ai in range(N_AGENTS):
        if agents_valid[ai] < 0.5:
            continue
        ax_w, ay_w, ayaw_w = agents_world[ai, 0], agents_world[ai, 1], agents_world[ai, 2]
        a_ego = transform_to_ego_frame(
            np.array([[ax_w, ay_w]]), ego_x, ego_y, ego_yaw)[0]
        a_yaw_ego = transform_yaw_to_ego(ayaw_w, ego_yaw)
        a_w = max(agents_size[ai, 0], 0.5)   # width
        a_l = max(agents_size[ai, 1], 0.5)   # length

        # Only draw agents within view
        if abs(a_ego[0]) > view_radius * 1.1 or abs(a_ego[1]) > view_radius * 1.1:
            continue

        draw_rotated_box(ax, a_ego[0], a_ego[1], a_yaw_ego,
                         length=a_l, width=a_w,
                         color='#87CEEB', alpha=0.7, zorder=5)

    # --- Ego vehicle (always at origin, heading up = pi/2 in ego frame) ---
    draw_rotated_box(ax, 0.0, 0.0, np.pi / 2,
                     length=EGO_LENGTH, width=EGO_WIDTH,
                     color='#2ca02c', alpha=0.9, edgecolor='darkgreen',
                     linewidth=1.0, zorder=10)

    # --- Goal marker ---
    goal_world = scenario['goal_world'].numpy()  # (2,)
    goal_ego = transform_to_ego_frame(
        goal_world.reshape(1, 2), ego_x, ego_y, ego_yaw)[0]
    if abs(goal_ego[0]) < view_radius and abs(goal_ego[1]) < view_radius:
        ax.scatter(goal_ego[0], goal_ego[1], marker='*', s=120,
                   color='red', edgecolors='darkred', linewidths=0.5,
                   zorder=11, label='Goal')

    # --- Formatting ---
    ax.set_xlim(-view_radius, view_radius)
    ax.set_ylim(-view_radius, view_radius)
    ax.set_aspect('equal')
    ax.set_facecolor('#F5F5F0')
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.tick_params(labelsize=6)
    ax.set_title(f't = {t_step * DT:.1f}s (step {t_step})', fontsize=9, fontweight='bold')


def compute_metrics_at_step(ego_trajs_world, batch, t_step, device):
    """Compute cumulative metrics up to step t_step."""
    if t_step < 4:
        return {}
    ego_t = torch.tensor(ego_trajs_world[:t_step + 1]).unsqueeze(0).float().to(device)
    try:
        m = compute_metrics_batch(ego_t, batch, device)
        return {k: v[0].item() for k, v in m.items()}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Main visualization pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='BEV visualization for CarPlanner closed-loop')
    parser.add_argument('--cache', default=os.path.join(PROJECT_ROOT, 'checkpoints/test14_random_cache.pt'))
    parser.add_argument('--checkpoint', default=os.path.join(PROJECT_ROOT, 'checkpoints/stage_c_best.pt'))
    parser.add_argument('--stage', default='c', choices=['a', 'b', 'c'])
    parser.add_argument('--scenarios', default='0,1,2',
                        help='Comma-separated indices or "random:N"')
    parser.add_argument('--timesteps', default='0,50,100,120,130',
                        help='Comma-separated sim steps to render')
    parser.add_argument('--output_dir', default=os.path.join(PROJECT_ROOT, 'checkpoints/viz'))
    parser.add_argument('--view_radius', type=float, default=60.0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Parse timesteps
    timesteps = [int(x.strip()) for x in args.timesteps.split(',')]

    # Load cache
    print(f'[Viz] Loading cache: {args.cache}')
    data = torch.load(args.cache, map_location='cpu')
    scenarios = data['scenarios']
    print(f'[Viz] {len(scenarios)} scenarios in cache')

    # Parse scenario indices
    if args.scenarios.startswith('random:'):
        n_rand = int(args.scenarios.split(':')[1])
        import random
        indices = random.sample(range(len(scenarios)), min(n_rand, len(scenarios)))
    else:
        indices = [int(x.strip()) for x in args.scenarios.split(',')]
    print(f'[Viz] Selected scenario indices: {indices}')

    # Load model
    print(f'[Viz] Loading model: {args.checkpoint}')
    model = CarPlanner().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    print(f'[Viz] Model loaded (epoch {ckpt.get("epoch", "?")})')

    n_scenarios = len(indices)
    n_cols = len(timesteps) + 1  # +1 for metric summary column

    fig, axes = plt.subplots(
        n_scenarios, n_cols,
        figsize=(4.2 * n_cols, 4.2 * n_scenarios),
        squeeze=False,
    )
    fig.patch.set_facecolor('white')

    for row_idx, scen_idx in enumerate(indices):
        scenario = scenarios[scen_idx]
        stype = scenario.get('scenario_type', 'unknown')
        stoken = scenario.get('scenario_token', '?')[:8]
        print(f'[Viz] Scenario {scen_idx} ({stype}, token={stoken}): simulating...', flush=True)

        ego_gt = scenario['ego_gt']  # (T_SIM, 4)
        ego_trajs_world, planned_trajs, batch = run_scenario_with_viz(
            model, scenario, device)

        # Compute full-simulation metrics
        full_metrics = compute_metrics_at_step(ego_trajs_world, batch, T_SIM - 1, device)

        # Render BEV frames
        for col_idx, t_step in enumerate(timesteps):
            t_step = min(t_step, T_SIM - 1)
            ax = axes[row_idx, col_idx]
            render_bev_frame(ax, t_step, scenario, ego_trajs_world,
                             planned_trajs, ego_gt, args.view_radius)

            # Per-step metrics overlay
            step_m = compute_metrics_at_step(ego_trajs_world, batch, t_step, device)
            if step_m:
                txt = (f"CLS-NR: {step_m.get('cls_nr', 0)*100:.0f}\n"
                       f"NoColl: {step_m.get('no_collision', 0)*100:.0f}%\n"
                       f"Driv:   {step_m.get('drivable', 0)*100:.0f}%")
                ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                        fontsize=8, verticalalignment='top',
                        fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  alpha=0.9, edgecolor='gray', linewidth=0.5))

            if col_idx == 0:
                ax.set_ylabel(f'Scen {scen_idx}\n{stype}', fontsize=8, fontweight='bold')

        # Metric summary column (last column)
        ax_summary = axes[row_idx, -1]
        ax_summary.axis('off')
        ax_summary.set_facecolor('white')

        summary_lines = [
            f'Scenario {scen_idx}',
            f'Type: {stype}',
            f'Token: {stoken}',
            '',
            'Final Metrics:',
            f'  CLS-NR:       {full_metrics.get("cls_nr", 0)*100:.1f}',
            f'  No-Collision:  {full_metrics.get("no_collision", 0)*100:.1f}%',
            f'  Drivable:      {full_metrics.get("drivable", 0)*100:.1f}%',
            f'  Comfort:       {full_metrics.get("comfort", 0)*100:.1f}%',
            f'  Progress:      {full_metrics.get("progress", 0)*100:.1f}%',
            f'  Coll steps:    {full_metrics.get("coll_steps", 0):.0f}',
        ]
        ax_summary.text(0.1, 0.95, '\n'.join(summary_lines),
                        transform=ax_summary.transAxes,
                        fontsize=8, verticalalignment='top',
                        fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0',
                                  edgecolor='gray', linewidth=0.5))

    # Legend in first panel
    axes[0, 0].legend(loc='lower right', fontsize=6, framealpha=0.8)

    fig.suptitle(
        f'CarPlanner Stage {args.stage.upper()} — Closed-Loop BEV Visualization',
        fontsize=13, fontweight='bold', y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # Save combined figure
    combined_path = os.path.join(output_dir, f'bev_stage{args.stage}_combined.png')
    fig.savefig(combined_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'[Viz] Saved combined figure: {combined_path}')

    # Also save individual scenario strips
    for row_idx, scen_idx in enumerate(indices):
        scenario = scenarios[scen_idx]
        stype = scenario.get('scenario_type', 'unknown')
        ego_gt = scenario['ego_gt']

        print(f'[Viz] Rendering individual strip for scenario {scen_idx}...', flush=True)
        ego_trajs_world, planned_trajs, batch = run_scenario_with_viz(
            model, scenario, device)
        full_metrics = compute_metrics_at_step(ego_trajs_world, batch, T_SIM - 1, device)

        fig_s, axes_s = plt.subplots(
            1, len(timesteps),
            figsize=(4.2 * len(timesteps), 4.2),
            squeeze=False,
        )
        fig_s.patch.set_facecolor('white')

        for col_idx, t_step in enumerate(timesteps):
            t_step = min(t_step, T_SIM - 1)
            ax = axes_s[0, col_idx]
            render_bev_frame(ax, t_step, scenario, ego_trajs_world,
                             planned_trajs, ego_gt, args.view_radius)

            step_m = compute_metrics_at_step(ego_trajs_world, batch, t_step, device)
            if step_m:
                txt = (f"CLS-NR: {step_m.get('cls_nr', 0)*100:.0f}\n"
                       f"NoColl: {step_m.get('no_collision', 0)*100:.0f}%\n"
                       f"Driv:   {step_m.get('drivable', 0)*100:.0f}%")
                ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                        fontsize=8, verticalalignment='top',
                        fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  alpha=0.9, edgecolor='gray', linewidth=0.5))

        axes_s[0, 0].legend(loc='lower right', fontsize=6, framealpha=0.8)
        fig_s.suptitle(
            f'Scenario {scen_idx}: {stype}  |  CLS-NR={full_metrics.get("cls_nr", 0)*100:.1f}',
            fontsize=11, fontweight='bold',
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        strip_path = os.path.join(output_dir, f'bev_stage{args.stage}_scen{scen_idx}.png')
        fig_s.savefig(strip_path, dpi=180, bbox_inches='tight', facecolor='white')
        plt.close(fig_s)
        print(f'[Viz] Saved: {strip_path}')

    print(f'[Viz] All outputs saved to {output_dir}')


if __name__ == '__main__':
    main()
