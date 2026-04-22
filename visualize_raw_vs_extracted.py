"""
Side-by-side comparison: raw map geometry from nuPlan API vs encoded/extracted cache.
Shows that the extraction pipeline faithfully captures the map structure.
"""
import os, sys, math, warnings
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import config as cfg
from data_loader import (
    NuPlanCarPlannerDataset, _get_map_name_from_db, _get_map_api,
    _to_ego_frame, _resample_polyline, _resample_polygon_ring,
)
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_ego_state_for_lidarpc_token_from_db,
)

CACHE_PATH = os.path.join(cfg.CHECKPOINT_DIR, "stage_cache_mini.pt")
OUT_DIR = os.path.join(os.path.dirname(__file__), "viz_raw_vs_extracted")
os.makedirs(OUT_DIR, exist_ok=True)


def get_raw_map_geometry(db_path, token):
    """Query raw lane and polygon geometry from the nuPlan map API."""
    ego = get_ego_state_for_lidarpc_token_from_db(db_path, token)
    ref_x, ref_y, ref_h = ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading

    map_name = _get_map_name_from_db(db_path)
    map_api = _get_map_api(map_name)
    point = Point2D(ref_x, ref_y)

    # ── Raw lanes ────────────────────────────────────────────────────────────
    raw_lanes = []
    try:
        map_objs = map_api.get_proximal_map_objects(
            point, cfg.MAP_QUERY_RADIUS, [SemanticMapLayer.LANE]
        )
        for lane in map_objs[SemanticMapLayer.LANE][:cfg.N_LANES]:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "invalid value encountered in cast")
                bp = lane.baseline_path
            if bp is None:
                continue
            centerline = bp.discrete_path
            if len(centerline) < 2:
                continue
            # Raw global coords → ego frame
            pts_global = np.array([[p.x, p.y] for p in centerline], dtype=np.float64)
            pts_ego = np.zeros_like(pts_global)
            for j in range(len(pts_global)):
                x_e, y_e, _ = _to_ego_frame(pts_global[j, 0], pts_global[j, 1], 0,
                                              ref_x, ref_y, ref_h)
                pts_ego[j] = [x_e, y_e]
            raw_lanes.append(pts_ego)
    except Exception:
        pass

    # ── Raw polygons ─────────────────────────────────────────────────────────
    polygon_layers = [
        (SemanticMapLayer.CROSSWALK, 'Crosswalk'),
        (SemanticMapLayer.STOP_LINE, 'Stop Line'),
        (SemanticMapLayer.INTERSECTION, 'Intersection'),
    ]
    raw_polygons = []
    for layer, name in polygon_layers:
        try:
            map_objs = map_api.get_proximal_map_objects(
                point, cfg.MAP_QUERY_RADIUS, [layer]
            )
            for obj in map_objs[layer]:
                ring = np.array(obj.polygon.exterior.coords, dtype=np.float64)[:, :2]
                if len(ring) < 3:
                    continue
                # Transform to ego frame
                ring_ego = np.zeros_like(ring)
                for j in range(len(ring)):
                    x_e, y_e, _ = _to_ego_frame(ring[j, 0], ring[j, 1], 0,
                                                  ref_x, ref_y, ref_h)
                    ring_ego[j] = [x_e, y_e]
                raw_polygons.append((name, ring_ego))
        except Exception:
            continue

    return raw_lanes, raw_polygons


def plot_comparison(idx, db_path, token, cached_data, save_path):
    """Plot raw geometry vs extracted/encoded for one sample."""
    raw_lanes, raw_polygons = get_raw_map_geometry(db_path, token)

    lanes = cached_data['map_lanes'][idx].numpy()
    lanes_mask = cached_data['map_lanes_mask'][idx].numpy()
    polys = cached_data['map_polygons'][idx].numpy()
    polys_mask = cached_data['map_polygons_mask'][idx].numpy()

    poly_colors = {'Crosswalk': 'green', 'Stop Line': 'red', 'Intersection': 'orange'}

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # ── Left: Raw geometry (full resolution, all points) ────────────────────
    ax = axes[0]
    ax.set_title(f"Sample {idx}: RAW Map Geometry\n"
                 f"({len(raw_lanes)} lanes, {len(raw_polygons)} polygons, full resolution)")
    for pts in raw_lanes:
        ax.plot(pts[:, 0], pts[:, 1], 'b-', linewidth=0.8, alpha=0.6)

    seen_cats = set()
    for name, ring in raw_polygons:
        seen_cats.add(name)
        color = poly_colors.get(name, 'gray')
        ring_closed = np.vstack([ring, ring[0]])
        ax.fill(ring_closed[:, 0], ring_closed[:, 1], color=color, alpha=0.2)
        ax.plot(ring_closed[:, 0], ring_closed[:, 1], color=color, linewidth=1.2, alpha=0.7)

    ax.plot(0, 0, 'r*', markersize=15, zorder=10)
    handles = [mpatches.Patch(color='blue', alpha=0.5, label=f'Lanes ({len(raw_lanes)})')]
    handles += [mpatches.Patch(color=poly_colors[c], alpha=0.5,
                label=f'{c} ({sum(1 for n,_ in raw_polygons if n==c)})')
                for c in sorted(seen_cats)]
    ax.legend(handles=handles, loc='upper right', fontsize=8)
    ax.set_xlabel("x (m, ego frame)")
    ax.set_ylabel("y (m, ego frame)")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # ── Right: Extracted/encoded (resampled to 10 pts, top-N selected) ──────
    ax = axes[1]
    n_valid_lanes = int(lanes_mask.sum())
    n_valid_polys = int(polys_mask.sum())
    ax.set_title(f"Sample {idx}: EXTRACTED (Encoded Cache)\n"
                 f"({n_valid_lanes}/{cfg.N_LANES} lanes, "
                 f"{n_valid_polys}/{cfg.N_POLYGONS} polygons, "
                 f"{cfg.N_LANE_POINTS} pts each)")

    for i in range(len(lanes_mask)):
        if lanes_mask[i] < 0.5:
            continue
        cx = lanes[i, :, 0]
        cy = lanes[i, :, 1]
        ax.plot(cx, cy, 'b-o', linewidth=0.8, markersize=2, alpha=0.6)

    cat_names_map = {0: 'Crosswalk', 1: 'Stop Line', 2: 'Intersection', 3: 'Other'}
    ext_type_counts = {}
    for i in range(len(polys_mask)):
        if polys_mask[i] < 0.5:
            continue
        px = polys[i, :, 0]
        py = polys[i, :, 1]
        cat_idx = int(polys[i, 0, 5:9].argmax())
        name = cat_names_map[cat_idx]
        ext_type_counts[name] = ext_type_counts.get(name, 0) + 1
        color = poly_colors.get(name, 'gray')
        px_closed = np.append(px, px[0])
        py_closed = np.append(py, py[0])
        ax.fill(px_closed, py_closed, color=color, alpha=0.2)
        ax.plot(px_closed, py_closed, color=color, linewidth=1.2, alpha=0.7,
                marker='o', markersize=2)

    ax.plot(0, 0, 'r*', markersize=15, zorder=10)
    handles = [mpatches.Patch(color='blue', alpha=0.5, label=f'Lanes ({n_valid_lanes})')]
    handles += [mpatches.Patch(color=poly_colors.get(c, 'gray'), alpha=0.5,
                label=f'{c} ({ext_type_counts[c]})')
                for c in sorted(ext_type_counts)]
    ax.legend(handles=handles, loc='upper right', fontsize=8)
    ax.set_xlabel("x (m, ego frame)")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Match axis limits
    all_x, all_y = [], []
    for a in axes:
        xl = a.get_xlim()
        yl = a.get_ylim()
        all_x.extend(xl)
        all_y.extend(yl)
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    for a in axes:
        a.set_xlim(xmin, xmax)
        a.set_ylim(ymin, ymax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    print("Loading dataset index...")
    ds = NuPlanCarPlannerDataset('mini', max_per_file=5)

    print(f"Loading cache: {CACHE_PATH}")
    cached = torch.load(CACHE_PATH, map_location='cpu', weights_only=True)

    sample_indices = [0, 50, 100, 150, 200]
    sample_indices = [i for i in sample_indices if i < len(ds)]

    for idx in sample_indices:
        db_path, token = ds._index[idx]
        print(f"\nSample {idx}: {os.path.basename(db_path)}")
        save_path = os.path.join(OUT_DIR, f"raw_vs_extracted_{idx:04d}.png")
        plot_comparison(idx, db_path, token, cached, save_path)

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
