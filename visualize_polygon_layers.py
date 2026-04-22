"""
Visualize all polygon layers from the nuPlan GeoPackage map.

Produces:
  1. Full-map overview of all polygon layers (color-coded)
  2. Zoomed ego-centric views for sample scenes showing lanes + polygons
  3. Per-layer detail panels with geometry statistics
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import math
import sqlite3
import glob
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from shapely.geometry import box as shapely_box
import fiona

import config as cfg

GPKG_PATH = os.path.join(cfg.MAPS_DIR, "us-ma-boston", "9.12.1817", "map.gpkg")
OUT_DIR = os.path.join(os.path.dirname(__file__), "viz_polygons")
os.makedirs(OUT_DIR, exist_ok=True)

# Layers we care about for driving context
POLYGON_LAYERS = {
    'crosswalks':           {'color': '#e74c3c', 'alpha': 0.6, 'label': 'Crosswalks'},
    'stop_polygons':        {'color': '#f39c12', 'alpha': 0.6, 'label': 'Stop Polygons'},
    'intersections':        {'color': '#3498db', 'alpha': 0.4, 'label': 'Intersections'},
    'generic_drivable_areas': {'color': '#2ecc71', 'alpha': 0.3, 'label': 'Drivable Areas'},
    'walkways':             {'color': '#9b59b6', 'alpha': 0.4, 'label': 'Walkways'},
    'carpark_areas':        {'color': '#1abc9c', 'alpha': 0.4, 'label': 'Carpark Areas'},
}

LANE_LAYERS = {
    'lanes_polygons':       {'color': '#bdc3c7', 'alpha': 0.25, 'label': 'Lane Polygons'},
    'lane_group_connectors': {'color': '#95a5a6', 'alpha': 0.2, 'label': 'Lane Group Connectors'},
}


def load_all_polygon_layers(gpkg_path):
    """Load all polygon layers into UTM-projected GeoDataFrames."""
    layers = {}
    for name in list(POLYGON_LAYERS.keys()) + list(LANE_LAYERS.keys()):
        try:
            gdf = gpd.read_file(gpkg_path, layer=name)
            gdf_utm = gdf.to_crs("EPSG:32619")
            layers[name] = gdf_utm
            print(f"  {name:40s}: {len(gdf_utm):5d} polygons")
        except Exception as e:
            print(f"  {name:40s}: FAILED ({e})")
    return layers


def get_boston_ego_poses(n_poses=20):
    """Get sample ego poses from Boston mini DBs in UTM coords."""
    db_files = sorted(glob.glob(os.path.join(cfg.MINI_DIR, "*.db")))
    poses = []
    for db in db_files:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute("SELECT location FROM log LIMIT 1")
        loc = str(cur.fetchone()[0])
        if 'boston' not in loc.lower():
            conn.close()
            continue
        cur.execute("SELECT x, y, qw, qx, qy, qz FROM ego_pose ORDER BY timestamp")
        rows = cur.fetchall()
        conn.close()
        # Subsample: take every 50th pose for variety
        for i in range(0, len(rows), 50):
            x, y, qw, qx, qy, qz = rows[i]
            heading = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
            poses.append((x, y, heading))
            if len(poses) >= n_poses:
                return poses
    return poses


def plot_full_map(layers):
    """Plot 1: Full-map overview of all polygon layers."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    ax.set_facecolor('#1a1a2e')

    # Plot lane background first
    for name, style in LANE_LAYERS.items():
        if name in layers:
            layers[name].plot(ax=ax, color=style['color'], alpha=style['alpha'],
                              edgecolor='none')

    # Plot polygon layers
    for name, style in POLYGON_LAYERS.items():
        if name in layers:
            layers[name].plot(ax=ax, color=style['color'], alpha=style['alpha'],
                              edgecolor=style['color'], linewidth=0.3)

    legend_patches = []
    for name, style in POLYGON_LAYERS.items():
        if name in layers:
            n = len(layers[name])
            legend_patches.append(mpatches.Patch(
                color=style['color'], alpha=style['alpha'],
                label=f"{style['label']} ({n})"
            ))
    for name, style in LANE_LAYERS.items():
        if name in layers:
            n = len(layers[name])
            legend_patches.append(mpatches.Patch(
                color=style['color'], alpha=style['alpha'],
                label=f"{style['label']} ({n})"
            ))

    ax.legend(handles=legend_patches, loc='upper right', fontsize=9,
              facecolor='#16213e', edgecolor='white', labelcolor='white')
    ax.set_title("nuPlan Boston Map — All Polygon Layers", fontsize=14, color='white')
    ax.tick_params(colors='white', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.set_xlabel("UTM Easting (m)", color='white')
    ax.set_ylabel("UTM Northing (m)", color='white')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "01_full_map_overview.png")
    fig.savefig(path, dpi=150, facecolor='#1a1a2e')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_per_layer_details(layers):
    """Plot 2: One subplot per polygon layer with geometry stats."""
    poly_layers = {k: v for k, v in layers.items() if k in POLYGON_LAYERS}
    n = len(poly_layers)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    fig.patch.set_facecolor('#1a1a2e')
    axes = axes.flatten() if n > 1 else [axes]

    for idx, (name, gdf) in enumerate(poly_layers.items()):
        ax = axes[idx]
        ax.set_facecolor('#16213e')
        style = POLYGON_LAYERS[name]

        gdf.plot(ax=ax, color=style['color'], alpha=0.7,
                 edgecolor=style['color'], linewidth=0.5)

        # Geometry stats
        areas = gdf.geometry.area
        perims = gdf.geometry.length
        n_pts = [len(g.exterior.coords) if hasattr(g, 'exterior') else 0 for g in gdf.geometry]

        stats = (f"Count: {len(gdf)}\n"
                 f"Area: {areas.mean():.1f} ± {areas.std():.1f} m²\n"
                 f"  min={areas.min():.1f}, max={areas.max():.1f}\n"
                 f"Perimeter: {perims.mean():.1f} ± {perims.std():.1f} m\n"
                 f"Pts/polygon: {np.mean(n_pts):.1f} ± {np.std(n_pts):.1f}")

        ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        ax.set_title(f"{style['label']} ({len(gdf)})", fontsize=11, color='white')
        ax.tick_params(colors='white', labelsize=6)
        for spine in ax.spines.values():
            spine.set_color('white')

    # Hide unused axes
    for idx in range(len(poly_layers), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "02_per_layer_details.png")
    fig.savefig(path, dpi=150, facecolor='#1a1a2e')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_ego_scenes(layers, ego_poses, n_scenes=6):
    """Plot 3: Zoomed ego-centric views showing lanes + polygons around the ego."""
    radius = cfg.MAP_QUERY_RADIUS  # 50m
    n_scenes = min(n_scenes, len(ego_poses))

    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.patch.set_facecolor('#1a1a2e')
    axes = axes.flatten()

    for i in range(n_scenes):
        ax = axes[i]
        ax.set_facecolor('#16213e')
        ego_x, ego_y, ego_h = ego_poses[i]

        # Clip region
        clip_box = shapely_box(ego_x - radius, ego_y - radius,
                               ego_x + radius, ego_y + radius)

        # Draw lanes background
        for lname, lstyle in LANE_LAYERS.items():
            if lname in layers:
                clipped = gpd.clip(layers[lname], clip_box)
                if len(clipped) > 0:
                    clipped.plot(ax=ax, color=lstyle['color'], alpha=0.3,
                                 edgecolor='#7f8c8d', linewidth=0.3)

        # Draw baseline paths (centerlines) if available
        if 'baseline_paths' in layers:
            clipped_bp = gpd.clip(layers['baseline_paths'], clip_box)
            if len(clipped_bp) > 0:
                clipped_bp.plot(ax=ax, color='#ecf0f1', linewidth=0.8, alpha=0.5)

        # Draw polygon layers
        nearby_counts = {}
        for pname, pstyle in POLYGON_LAYERS.items():
            if pname in layers:
                clipped = gpd.clip(layers[pname], clip_box)
                nearby_counts[pname] = len(clipped)
                if len(clipped) > 0:
                    clipped.plot(ax=ax, color=pstyle['color'], alpha=pstyle['alpha'],
                                 edgecolor=pstyle['color'], linewidth=1.0)

        # Draw ego position and heading
        arrow_len = 5.0
        ax.plot(ego_x, ego_y, 'w*', markersize=15, zorder=10)
        ax.annotate('', xy=(ego_x + arrow_len * math.cos(ego_h),
                            ego_y + arrow_len * math.sin(ego_h)),
                     xytext=(ego_x, ego_y),
                     arrowprops=dict(arrowstyle='->', color='white', lw=2),
                     zorder=10)

        # Query circle
        circle = plt.Circle((ego_x, ego_y), radius, fill=False,
                             edgecolor='yellow', linewidth=1.0, linestyle='--', alpha=0.5)
        ax.add_patch(circle)

        ax.set_xlim(ego_x - radius * 1.1, ego_x + radius * 1.1)
        ax.set_ylim(ego_y - radius * 1.1, ego_y + radius * 1.1)
        ax.set_aspect('equal')

        # Info text
        info = f"Ego: ({ego_x:.0f}, {ego_y:.0f}), h={math.degrees(ego_h):.0f}°\n"
        info += "\n".join(f"  {POLYGON_LAYERS[k]['label']}: {v}"
                          for k, v in nearby_counts.items() if v > 0)
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=7,
                verticalalignment='top', color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        ax.set_title(f"Scene {i+1} (r={radius}m)", fontsize=10, color='white')
        ax.tick_params(colors='white', labelsize=6)
        for spine in ax.spines.values():
            spine.set_color('white')

    for idx in range(n_scenes, len(axes)):
        axes[idx].set_visible(False)

    # Shared legend
    legend_patches = [mpatches.Patch(color=s['color'], alpha=s['alpha'], label=s['label'])
                      for s in POLYGON_LAYERS.values()]
    legend_patches.append(mpatches.Patch(color='#bdc3c7', alpha=0.3, label='Lane Polygons'))
    fig.legend(handles=legend_patches, loc='lower center', ncol=4, fontsize=9,
               facecolor='#16213e', edgecolor='white', labelcolor='white')

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(OUT_DIR, "03_ego_scenes_with_polygons.png")
    fig.savefig(path, dpi=150, facecolor='#1a1a2e')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_polygon_size_distributions(layers):
    """Plot 4: Histograms of area and vertex count per layer."""
    poly_layers = {k: v for k, v in layers.items() if k in POLYGON_LAYERS}
    n = len(poly_layers)

    fig, axes = plt.subplots(n, 2, figsize=(14, 3.5 * n))
    fig.patch.set_facecolor('#1a1a2e')
    if n == 1:
        axes = axes.reshape(1, -1)

    for idx, (name, gdf) in enumerate(poly_layers.items()):
        style = POLYGON_LAYERS[name]
        areas = gdf.geometry.area.values
        n_verts = np.array([len(g.exterior.coords) if hasattr(g, 'exterior') else 0
                            for g in gdf.geometry])

        # Area histogram
        ax = axes[idx, 0]
        ax.set_facecolor('#16213e')
        ax.hist(areas, bins=50, color=style['color'], alpha=0.8, edgecolor='white', linewidth=0.3)
        ax.set_title(f"{style['label']} — Area (m²)", fontsize=10, color='white')
        ax.axvline(np.median(areas), color='white', linestyle='--', linewidth=1, label=f'median={np.median(areas):.1f}')
        ax.legend(fontsize=7, facecolor='black', labelcolor='white')
        ax.tick_params(colors='white', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('white')

        # Vertex count histogram
        ax = axes[idx, 1]
        ax.set_facecolor('#16213e')
        ax.hist(n_verts, bins=50, color=style['color'], alpha=0.8, edgecolor='white', linewidth=0.3)
        ax.set_title(f"{style['label']} — Vertices per Polygon", fontsize=10, color='white')
        ax.axvline(np.median(n_verts), color='white', linestyle='--', linewidth=1, label=f'median={np.median(n_verts):.0f}')
        ax.legend(fontsize=7, facecolor='black', labelcolor='white')
        ax.tick_params(colors='white', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('white')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "04_polygon_size_distributions.png")
    fig.savefig(path, dpi=150, facecolor='#1a1a2e')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_ego_frame_polygons(layers, ego_poses, n_scenes=4):
    """Plot 5: Polygons transformed into ego-centric frame (what the model would see)."""
    radius = cfg.MAP_QUERY_RADIUS

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.patch.set_facecolor('#1a1a2e')
    axes = axes.flatten()

    for i in range(min(n_scenes, len(ego_poses))):
        ax = axes[i]
        ax.set_facecolor('#16213e')
        ego_x, ego_y, ego_h = ego_poses[i]
        cos_h = math.cos(-ego_h)
        sin_h = math.sin(-ego_h)

        clip_box = shapely_box(ego_x - radius, ego_y - radius,
                               ego_x + radius, ego_y + radius)

        def to_ego(coords):
            """Transform UTM coords to ego frame."""
            arr = np.array(coords)
            dx = arr[:, 0] - ego_x
            dy = arr[:, 1] - ego_y
            x_e = cos_h * dx - sin_h * dy
            y_e = sin_h * dx + cos_h * dy
            return np.column_stack([x_e, y_e])

        # Draw lane polygons in ego frame
        for lname in ['lanes_polygons', 'lane_group_connectors']:
            if lname not in layers:
                continue
            clipped = gpd.clip(layers[lname], clip_box)
            for _, row in clipped.iterrows():
                geom = row.geometry
                if hasattr(geom, 'exterior'):
                    pts = to_ego(geom.exterior.coords)
                    poly = plt.Polygon(pts, closed=True, facecolor='#bdc3c7',
                                       alpha=0.15, edgecolor='#7f8c8d', linewidth=0.3)
                    ax.add_patch(poly)

        # Draw polygon layers in ego frame
        total_nearby = 0
        layer_info = []
        for pname, pstyle in POLYGON_LAYERS.items():
            if pname not in layers:
                continue
            clipped = gpd.clip(layers[pname], clip_box)
            count = 0
            for _, row in clipped.iterrows():
                geom = row.geometry
                if hasattr(geom, 'exterior'):
                    pts = to_ego(geom.exterior.coords)
                    poly = plt.Polygon(pts, closed=True, facecolor=pstyle['color'],
                                       alpha=pstyle['alpha'], edgecolor=pstyle['color'],
                                       linewidth=1.0)
                    ax.add_patch(poly)
                    # Mark centroid
                    cx, cy = pts.mean(axis=0)
                    ax.plot(cx, cy, 'o', color=pstyle['color'], markersize=3, alpha=0.8)
                    count += 1
            if count > 0:
                layer_info.append(f"{pstyle['label']}: {count}")
                total_nearby += count

        # Ego marker at origin
        ax.plot(0, 0, 'w*', markersize=18, zorder=10)
        ax.annotate('', xy=(5, 0), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color='white', lw=2.5), zorder=10)

        # Query circle
        circle = plt.Circle((0, 0), radius, fill=False,
                             edgecolor='yellow', linewidth=1.0, linestyle='--', alpha=0.5)
        ax.add_patch(circle)

        ax.set_xlim(-radius * 1.1, radius * 1.1)
        ax.set_ylim(-radius * 1.1, radius * 1.1)
        ax.set_aspect('equal')

        info = f"Ego-centric frame\n{total_nearby} polygons within {radius}m\n"
        info += "\n".join(f"  {l}" for l in layer_info)
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        ax.set_title(f"Scene {i+1} — Ego Frame", fontsize=11, color='white')
        ax.set_xlabel("x (forward, m)", color='white', fontsize=9)
        ax.set_ylabel("y (left, m)", color='white', fontsize=9)
        ax.tick_params(colors='white', labelsize=7)
        ax.axhline(0, color='white', alpha=0.15, linewidth=0.5)
        ax.axvline(0, color='white', alpha=0.15, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color('white')

    legend_patches = [mpatches.Patch(color=s['color'], alpha=s['alpha'], label=s['label'])
                      for s in POLYGON_LAYERS.values()]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=9,
               facecolor='#16213e', edgecolor='white', labelcolor='white')

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(OUT_DIR, "05_ego_frame_polygons.png")
    fig.savefig(path, dpi=150, facecolor='#1a1a2e')
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print("nuPlan Polygon Layer Visualization")
    print("=" * 60)

    # Load baseline_paths too for centerline display
    print("\nLoading polygon layers...")
    layers = load_all_polygon_layers(GPKG_PATH)

    # Also load baseline_paths for centerline display
    try:
        bp = gpd.read_file(GPKG_PATH, layer='baseline_paths')
        layers['baseline_paths'] = bp.to_crs("EPSG:32619")
        print(f"  {'baseline_paths':40s}: {len(layers['baseline_paths']):5d} lines")
    except Exception:
        pass

    print(f"\nGetting ego poses from Boston DBs...")
    ego_poses = get_boston_ego_poses(n_poses=20)
    print(f"  Got {len(ego_poses)} ego poses")

    print(f"\nGenerating visualizations in {OUT_DIR}/")

    print("\n[1/5] Full map overview...")
    plot_full_map(layers)

    print("[2/5] Per-layer details...")
    plot_per_layer_details(layers)

    print("[3/5] Ego scene views (global frame)...")
    plot_ego_scenes(layers, ego_poses, n_scenes=6)

    print("[4/5] Size distributions...")
    plot_polygon_size_distributions(layers)

    print("[5/5] Ego-frame polygon views...")
    plot_ego_frame_polygons(layers, ego_poses, n_scenes=4)

    # Summary table
    print("\n" + "=" * 60)
    print("LAYER SUMMARY")
    print("=" * 60)
    print(f"{'Layer':<35s} {'Count':>6s} {'Median Area':>12s} {'Med Vertices':>13s}")
    print("-" * 66)
    for name, style in POLYGON_LAYERS.items():
        if name in layers:
            gdf = layers[name]
            areas = gdf.geometry.area
            n_v = [len(g.exterior.coords) if hasattr(g, 'exterior') else 0 for g in gdf.geometry]
            print(f"{style['label']:<35s} {len(gdf):>6d} {np.median(areas):>11.1f}m² {np.median(n_v):>12.0f}")
    print("=" * 60)
    print(f"\nAll outputs saved to: {OUT_DIR}/")


if __name__ == '__main__':
    main()
