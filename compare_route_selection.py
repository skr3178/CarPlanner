"""
Apples-to-apples comparison of OLD vs NEW route-selection algorithm.

Both columns use the SAME data_loader output (same map, same GT, same ego).
Only the route-selection step differs:
  - LEFT  (OLD): select_candidate_routes() — proximity filter over map_lanes
  - RIGHT (NEW): _extract_routes() — lane-graph BFS, cached in sample['route_polylines']
"""
import os, sys, warnings
import numpy as np
import torch
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from data_loader import _load_sample
from model import select_candidate_routes  # OLD proximity-based

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCENES = [
    ("on_intersection",                                        "2021.05.12.22.00.38_veh-35_01008_01518.db", "69e109f6e2a85ab4"),
    ("on_stopline_traffic_light",                              "2021.05.12.22.00.38_veh-35_01008_01518.db", "eb0cbe31a10d54aa"),
    ("starting_straight_stop_sign_intersection_traversal",     "2021.05.12.22.00.38_veh-35_01008_01518.db", "bb61b22a2b17570d"),
    ("starting_straight_traffic_light_intersection_traversal", "2021.05.12.22.00.38_veh-35_01008_01518.db", "69e109f6e2a85ab4"),
    ("accelerating_at_crosswalk",                              "2021.05.12.22.28.35_veh-35_00620_01164.db", "3d8fc8455f4558c3"),
    ("following_lane_without_lead",                            "2021.05.12.22.28.35_veh-35_00620_01164.db", "da221ff877a551ea"),
]

COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]


def plot_routes(ax, sample, routes_xy, routes_mask, title):
    """Plot a scene panel with map lanes + routes + GT."""
    ml  = sample["map_lanes"]
    mlm = sample["map_lanes_mask"]
    for j in range(cfg.N_LANES):
        if mlm[j] < 0.5:
            continue
        lane_xy = ml[j, :, :2]
        valid = lane_xy[np.abs(lane_xy[:, 0]) + np.abs(lane_xy[:, 1]) > 1e-6]
        if len(valid) > 1:
            ax.plot(valid[:, 0], valid[:, 1], "gray", alpha=0.25, linewidth=0.8)

    for r in range(len(routes_mask)):
        if routes_mask[r] < 0.5:
            continue
        route_xy = routes_xy[r, :, :2]
        valid = route_xy[np.abs(route_xy[:, 0]) + np.abs(route_xy[:, 1]) > 1e-6]
        if len(valid) > 1:
            ax.plot(valid[:, 0], valid[:, 1], color=COLORS[r],
                    linewidth=2.5, label=f"Route {r}", alpha=0.85)
            ax.plot(valid[0, 0],  valid[0, 1],  "o", color=COLORS[r], markersize=5)
            ax.plot(valid[-1, 0], valid[-1, 1], "s", color=COLORS[r], markersize=4)

    gt = sample["gt_trajectory"]
    ax.plot(gt[:, 0], gt[:, 1], "k--", linewidth=2.5, label="GT", alpha=0.9)
    ax.plot(gt[-1, 0], gt[-1, 1], "k*", markersize=12)
    ax.plot(0, 0, "r^", markersize=14, zorder=10)

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (ego frame, m)")
    ax.set_ylabel("y (ego frame, m)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def main():
    db_dir = cfg.MINI_DIR if os.path.isdir(cfg.MINI_DIR) else cfg.TRAIN_DIR
    n = len(SCENES)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4.5 * n))

    for row, (stype, db_name, token) in enumerate(SCENES):
        db_path = os.path.join(db_dir, db_name)
        sample = _load_sample(db_path, token)
        if sample is None:
            axes[row, 0].set_title(f"{stype}\n(load failed)")
            continue

        # NEW routes: already cached in sample dict
        new_rp = sample["route_polylines"]       # (5, 10, 39)
        new_rm = sample["route_mask"]            # (5,)

        # OLD routes: compute from map_lanes via select_candidate_routes
        ml_t  = torch.from_numpy(sample["map_lanes"]).unsqueeze(0).float()
        mlm_t = torch.from_numpy(sample["map_lanes_mask"]).unsqueeze(0).float()
        old_rp_t, old_rm_t = select_candidate_routes(ml_t, mlm_t)
        old_rp = old_rp_t[0].numpy()             # (5, 10, 39)
        old_rm = old_rm_t[0].numpy()             # (5,)

        plot_routes(
            axes[row, 0], sample, old_rp, old_rm,
            f"OLD — select_candidate_routes (proximity)\n"
            f"{stype}  |  routes={int(old_rm.sum())}/{cfg.N_LAT}"
        )
        plot_routes(
            axes[row, 1], sample, new_rp, new_rm,
            f"NEW — _extract_routes (lane-graph BFS)\n"
            f"{stype}  |  routes={int(new_rm.sum())}/{cfg.N_LAT}"
        )

    plt.suptitle("Route-Selection Comparison (same scene, same data_loader)",
                 fontsize=14, y=1.001)
    plt.tight_layout()
    out = "compare_route_selection.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
