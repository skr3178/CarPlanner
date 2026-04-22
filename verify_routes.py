"""
Verify route extraction: extract a few samples, print shapes/values, and plot routes.
"""

import sys
import os
import math
import warnings
import numpy as np
import sqlite3
import glob

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from data_loader import (
    _load_sample, _get_map_name_from_db, _get_map_api,
    _extract_routes, _get_traffic_light_status
)
from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_ego_state_for_lidarpc_token_from_db,
    get_lidarpc_tokens_with_scenario_tag_from_db,
)


def get_sample_tokens(db_dir, n_samples=5):
    """Get a few sample tokens from the DB files."""
    db_files = sorted(glob.glob(os.path.join(db_dir, "*.db")))
    tokens = []
    for db_path in db_files[:10]:
        try:
            for scenario_type, token in get_lidarpc_tokens_with_scenario_tag_from_db(db_path):
                tokens.append((db_path, token, scenario_type))
                if len(tokens) >= n_samples:
                    return tokens
        except Exception:
            continue
    return tokens


def main():
    # Try mini split first, then train_boston
    db_dir = cfg.MINI_DIR
    if not os.path.isdir(db_dir):
        db_dir = cfg.TRAIN_DIR
    print(f"Using DB dir: {db_dir}")

    tokens = get_sample_tokens(db_dir, n_samples=5)
    print(f"Found {len(tokens)} sample tokens\n")

    for i, (db_path, token, scenario_type) in enumerate(tokens):
        print(f"{'='*70}")
        print(f"Sample {i+1}: type={scenario_type}")
        print(f"  DB: {os.path.basename(db_path)}")
        print(f"  Token: {token[:16]}...")

        sample = _load_sample(db_path, token)
        if sample is None:
            print("  SKIPPED (load failed)")
            continue

        # Print all tensor shapes
        print(f"\n  Tensor shapes:")
        for k, v in sorted(sample.items()):
            if isinstance(v, np.ndarray):
                print(f"    {k:25s} {str(v.shape):20s} dtype={v.dtype}")
            else:
                print(f"    {k:25s} = {v}")

        # Route details
        rp = sample['route_polylines']
        rm = sample['route_mask']
        n_valid = int(rm.sum())
        print(f"\n  Routes: {n_valid}/{cfg.N_LAT} valid")

        for r in range(n_valid):
            route = rp[r]  # (N_ROUTE_POINTS, D_POLYLINE_POINT)
            # Extract center polyline (first D_MAP_POINT features)
            center_xy = route[:, :2]  # x, y in ego frame
            center_heading = route[:, 2:4]  # sin_h, cos_h
            speed_limit = route[:, 4]
            cat_onehot = route[:, 5:9]
            tl_onehot = route[:, 9:13]

            # Route statistics
            valid_pts = center_xy[np.abs(center_xy[:, 0]) + np.abs(center_xy[:, 1]) > 1e-6]
            if len(valid_pts) > 1:
                total_len = np.sum(np.sqrt(np.sum(np.diff(valid_pts, axis=0)**2, axis=1)))
                x_range = (valid_pts[:, 0].min(), valid_pts[:, 0].max())
                y_range = (valid_pts[:, 1].min(), valid_pts[:, 1].max())
                avg_speed = speed_limit[speed_limit > 0].mean() if (speed_limit > 0).any() else 0
                print(f"\n  Route {r}:")
                print(f"    Points: {len(valid_pts)}/{cfg.N_ROUTE_POINTS}")
                print(f"    Length: {total_len:.1f}m")
                print(f"    X range: [{x_range[0]:.1f}, {x_range[1]:.1f}]m")
                print(f"    Y range: [{y_range[0]:.1f}, {y_range[1]:.1f}]m")
                print(f"    Avg speed limit: {avg_speed:.1f} m/s")
                print(f"    Category: {cat_onehot[0]}")
                print(f"    Traffic light: {tl_onehot[0]}")
                print(f"    First 3 pts (ego frame): {center_xy[:3]}")

        # GT trajectory info
        gt = sample['gt_trajectory']
        print(f"\n  GT trajectory:")
        print(f"    Endpoint: ({gt[-1, 0]:.2f}, {gt[-1, 1]:.2f}, {gt[-1, 2]:.3f})")
        print(f"    Total displacement: {math.sqrt(gt[-1, 0]**2 + gt[-1, 1]**2):.2f}m")
        print(f"    Mode label: {sample['mode_label']} "
              f"(lon={sample['mode_label'] // cfg.N_LAT}, "
              f"lat={sample['mode_label'] % cfg.N_LAT})")

        # Map lanes for comparison
        ml = sample['map_lanes']
        mlm = sample['map_lanes_mask']
        print(f"\n  Map lanes: {int(mlm.sum())}/{cfg.N_LANES} valid")

    # Try to plot if matplotlib available
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        print(f"\n{'='*70}")
        print("Generating route visualization...")

        fig, axes = plt.subplots(1, min(len(tokens), 3), figsize=(6*min(len(tokens), 3), 6))
        if min(len(tokens), 3) == 1:
            axes = [axes]

        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

        for ax_idx, (db_path, token, scenario_type) in enumerate(tokens[:3]):
            ax = axes[ax_idx]
            sample = _load_sample(db_path, token)
            if sample is None:
                continue

            # Plot map lanes (gray)
            ml = sample['map_lanes']
            mlm = sample['map_lanes_mask']
            for j in range(cfg.N_LANES):
                if mlm[j] < 0.5:
                    continue
                lane_xy = ml[j, :, :2]  # center x, y
                valid = lane_xy[np.abs(lane_xy[:, 0]) + np.abs(lane_xy[:, 1]) > 1e-6]
                if len(valid) > 1:
                    ax.plot(valid[:, 0], valid[:, 1], 'gray', alpha=0.3, linewidth=1)

            # Plot routes (colored)
            rp = sample['route_polylines']
            rm = sample['route_mask']
            for r in range(cfg.N_LAT):
                if rm[r] < 0.5:
                    continue
                route_xy = rp[r, :, :2]
                valid = route_xy[np.abs(route_xy[:, 0]) + np.abs(route_xy[:, 1]) > 1e-6]
                if len(valid) > 1:
                    ax.plot(valid[:, 0], valid[:, 1], color=colors[r],
                            linewidth=2.5, label=f'Route {r}', alpha=0.8)
                    ax.plot(valid[0, 0], valid[0, 1], 'o', color=colors[r], markersize=6)

            # Plot GT trajectory (black dashed)
            gt = sample['gt_trajectory']
            ax.plot(gt[:, 0], gt[:, 1], 'k--', linewidth=2, label='GT', alpha=0.8)
            ax.plot(gt[-1, 0], gt[-1, 1], 'k*', markersize=10)

            # Plot ego
            ax.plot(0, 0, 'r^', markersize=12, label='Ego')

            lat_idx = sample['mode_label'] % cfg.N_LAT
            ax.set_title(f'{scenario_type}\nlat_mode={lat_idx}, routes={int(rm.sum())}',
                         fontsize=9)
            ax.set_xlabel('x (ego frame, m)')
            ax.set_ylabel('y (ego frame, m)')
            ax.legend(fontsize=7, loc='upper left')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(os.path.dirname(__file__), 'route_verification.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")

    except ImportError:
        print("\nmatplotlib not available — skipping visualization")


if __name__ == '__main__':
    main()
