"""
Visualize polygons and lanes from the extracted .pt cache.
Reconstructs ego-frame map elements from the cached 9-dim features.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import config as cfg

CACHE_PATH = os.path.join(cfg.CHECKPOINT_DIR, "stage_cache_mini.pt")
OUT_DIR = os.path.join(os.path.dirname(__file__), "viz_extracted_polygons")
os.makedirs(OUT_DIR, exist_ok=True)


def plot_sample(idx, data, save_path):
    """Plot lanes and polygons for a single sample from the cache."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    lanes = data['map_lanes'][idx].numpy()          # (N_LANES, N_PTS, 27)
    lanes_mask = data['map_lanes_mask'][idx].numpy()  # (N_LANES,)
    polys = data['map_polygons'][idx].numpy()         # (N_POLYGONS, N_PTS, 9)
    polys_mask = data['map_polygons_mask'][idx].numpy()  # (N_POLYGONS,)

    # ── Panel 1: Lanes only ──────────────────────────────────────────────────
    ax = axes[0]
    ax.set_title(f"Sample {idx}: Lane Polylines")
    for i in range(len(lanes_mask)):
        if lanes_mask[i] < 0.5:
            continue
        # Center line is first 9 dims per point
        cx = lanes[i, :, 0]
        cy = lanes[i, :, 1]
        ax.plot(cx, cy, 'b-', linewidth=1, alpha=0.7)
        # Left boundary (dims 9-17)
        lx = lanes[i, :, 9]
        ly = lanes[i, :, 10]
        ax.plot(lx, ly, 'c-', linewidth=0.5, alpha=0.4)
        # Right boundary (dims 18-26)
        rx = lanes[i, :, 18]
        ry = lanes[i, :, 19]
        ax.plot(rx, ry, 'c-', linewidth=0.5, alpha=0.4)
    ax.plot(0, 0, 'r*', markersize=15, zorder=10)
    ax.set_xlabel("x (m, ego frame)")
    ax.set_ylabel("y (m, ego frame)")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Polygons only ───────────────────────────────────────────────
    ax = axes[1]
    ax.set_title(f"Sample {idx}: Polygon Map Elements")
    poly_colors = {0: 'green', 1: 'red', 2: 'orange', 3: 'purple'}
    poly_names = {0: 'Crosswalk', 1: 'Stop Line', 2: 'Intersection', 3: 'Other'}
    seen_cats = set()
    for i in range(len(polys_mask)):
        if polys_mask[i] < 0.5:
            continue
        px = polys[i, :, 0]
        py = polys[i, :, 1]
        cat_onehot = polys[i, 0, 5:9]
        cat_idx = int(cat_onehot.argmax())
        seen_cats.add(cat_idx)
        color = poly_colors.get(cat_idx, 'gray')
        # Close the polygon for display
        px_closed = np.append(px, px[0])
        py_closed = np.append(py, py[0])
        ax.fill(px_closed, py_closed, color=color, alpha=0.3)
        ax.plot(px_closed, py_closed, color=color, linewidth=1.5, alpha=0.8)
    ax.plot(0, 0, 'r*', markersize=15, zorder=10)
    handles = [mpatches.Patch(color=poly_colors[c], alpha=0.5, label=poly_names[c])
               for c in sorted(seen_cats)]
    if handles:
        ax.legend(handles=handles, loc='upper right', fontsize=8)
    ax.set_xlabel("x (m, ego frame)")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Combined ────────────────────────────────────────────────────
    ax = axes[2]
    ax.set_title(f"Sample {idx}: Lanes + Polygons Combined")
    for i in range(len(lanes_mask)):
        if lanes_mask[i] < 0.5:
            continue
        cx = lanes[i, :, 0]
        cy = lanes[i, :, 1]
        ax.plot(cx, cy, 'b-', linewidth=1, alpha=0.5)
    for i in range(len(polys_mask)):
        if polys_mask[i] < 0.5:
            continue
        px = polys[i, :, 0]
        py = polys[i, :, 1]
        cat_onehot = polys[i, 0, 5:9]
        cat_idx = int(cat_onehot.argmax())
        color = poly_colors.get(cat_idx, 'gray')
        px_closed = np.append(px, px[0])
        py_closed = np.append(py, py[0])
        ax.fill(px_closed, py_closed, color=color, alpha=0.25)
        ax.plot(px_closed, py_closed, color=color, linewidth=1.5, alpha=0.8)
    ax.plot(0, 0, 'r*', markersize=15, zorder=10)
    handles = [mpatches.Patch(color='blue', alpha=0.5, label='Lanes')]
    handles += [mpatches.Patch(color=poly_colors[c], alpha=0.5, label=poly_names[c])
                for c in sorted(seen_cats)]
    ax.legend(handles=handles, loc='upper right', fontsize=8)
    ax.set_xlabel("x (m, ego frame)")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    print(f"Loading cache: {CACHE_PATH}")
    data = torch.load(CACHE_PATH, map_location='cpu', weights_only=True)
    n = data['map_polygons'].shape[0]
    print(f"  {n} samples, polygons shape: {tuple(data['map_polygons'].shape)}")
    print(f"  Avg valid polygons: {data['map_polygons_mask'].sum(1).mean():.1f}")

    # Plot a selection of samples with varying polygon counts
    poly_counts = data['map_polygons_mask'].sum(1)
    indices = [0, 50, 100, 150, 200]
    indices = [i for i in indices if i < n]

    for idx in indices:
        n_polys = int(poly_counts[idx].item())
        n_lanes = int(data['map_lanes_mask'][idx].sum().item())
        print(f"\nSample {idx}: {n_lanes} lanes, {n_polys} polygons")
        save_path = os.path.join(OUT_DIR, f"sample_{idx:04d}.png")
        plot_sample(idx, data, save_path)

    # Summary plot: polygon count distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(poly_counts.numpy(), bins=range(0, cfg.N_POLYGONS + 2), align='left',
            color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_xlabel("Valid polygons per sample")
    ax.set_ylabel("Count")
    ax.set_title(f"Polygon Count Distribution ({n} samples)")
    ax.set_xticks(range(0, cfg.N_POLYGONS + 1))
    plt.tight_layout()
    dist_path = os.path.join(OUT_DIR, "polygon_count_distribution.png")
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved distribution: {dist_path}")


if __name__ == '__main__':
    main()
