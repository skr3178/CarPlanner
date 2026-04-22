"""
Visualize the two map encoding sets from the CarPlanner paper:
  Set 1: POLYLINES — lane centers + left/right boundaries (Nm,1 × Np × 3Dm)
  Set 2: POLYGONS  — intersections, crosswalks, stop lines (Nm,2 × Np × Dm)

Shows both on the full Boston map and in ego-centric scenes.
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
from shapely.geometry import box as shapely_box
import fiona

import config as cfg

GPKG_PATH = os.path.join(cfg.MAPS_DIR, "us-ma-boston", "9.12.1817", "map.gpkg")
OUT_DIR = os.path.join(os.path.dirname(__file__), "viz_polygons")
os.makedirs(OUT_DIR, exist_ok=True)

# ── The two sets from the paper ──────────────────────────────────────────────

# Set 1: POLYLINES (lane network) — what feeds into LaneEncoder
POLYLINE_LAYERS = {
    'baseline_paths': {'color': '#3498db', 'alpha': 0.8, 'lw': 1.0,
                       'label': 'Lane Centerlines (baseline_paths)'},
    'boundaries':     {'color': '#1abc9c', 'alpha': 0.5, 'lw': 0.5,
                       'label': 'Lane Boundaries (left/right edges)'},
    'lane_connectors': {'color': '#9b59b6', 'alpha': 0.6, 'lw': 0.8,
                        'label': 'Lane Connectors (turn paths)'},
}

# Set 2: POLYGONS (semantic regions) — what feeds into PolygonEncoder
POLYGON_LAYERS = {
    'intersections':  {'color': '#e74c3c', 'alpha': 0.45, 'label': 'Intersections'},
    'crosswalks':     {'color': '#f39c12', 'alpha': 0.6,  'label': 'Crosswalks'},
    'stop_polygons':  {'color': '#2ecc71', 'alpha': 0.6,  'label': 'Stop Polygons'},
}

# Background context (not encoded, just for spatial reference)
BACKGROUND_LAYERS = {
    'lanes_polygons': {'color': '#2c3e50', 'alpha': 0.15, 'label': 'Lane Areas (background)'},
}


def load_layers(gpkg_path):
    """Load all relevant layers into UTM GeoDataFrames."""
    layers = {}
    all_names = (list(POLYLINE_LAYERS.keys()) + list(POLYGON_LAYERS.keys()) +
                 list(BACKGROUND_LAYERS.keys()))
    for name in all_names:
        try:
            gdf = gpd.read_file(gpkg_path, layer=name)
            gdf_utm = gdf.to_crs("EPSG:32619")
            layers[name] = gdf_utm
            print(f"  {name:40s}: {len(gdf_utm):5d} features")
        except Exception as e:
            print(f"  {name:40s}: FAILED ({e})")
    return layers


def get_ego_poses(n_poses=20):
    """Get sample ego poses from Boston mini DBs."""
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
        for i in range(0, len(rows), 50):
            x, y, qw, qx, qy, qz = rows[i]
            heading = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
            poses.append((x, y, heading))
            if len(poses) >= n_poses:
                return poses
    return poses


def plot_side_by_side_full_map(layers):
    """Plot 1: Full Boston map — polylines on left, polygons on right."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 14))
    fig.patch.set_facecolor('#0d1117')

    for ax in (ax1, ax2):
        ax.set_facecolor('#161b22')
        # Draw lane background on both
        if 'lanes_polygons' in layers:
            layers['lanes_polygons'].plot(ax=ax, color='#2c3e50', alpha=0.12,
                                          edgecolor='none')

    # ── Left panel: POLYLINES ────────────────────────────────────────────────
    for name, style in POLYLINE_LAYERS.items():
        if name in layers:
            layers[name].plot(ax=ax1, color=style['color'], alpha=style['alpha'],
                              linewidth=style['lw'])

    legend1 = [mpatches.Patch(color=s['color'], alpha=s['alpha'],
               label=f"{s['label']} ({len(layers[n])})")
               for n, s in POLYLINE_LAYERS.items() if n in layers]
    ax1.legend(handles=legend1, loc='upper right', fontsize=9,
               facecolor='#21262d', edgecolor='#30363d', labelcolor='white')
    ax1.set_title("SET 1: POLYLINES  (Nm,1 × Np × 3Dm)\nLane centers + left/right boundaries → LaneEncoder",
                  fontsize=13, color='white', pad=12)

    # ── Right panel: POLYGONS ────────────────────────────────────────────────
    for name, style in POLYGON_LAYERS.items():
        if name in layers:
            layers[name].plot(ax=ax2, color=style['color'], alpha=style['alpha'],
                              edgecolor=style['color'], linewidth=0.5)

    legend2 = [mpatches.Patch(color=s['color'], alpha=s['alpha'],
               label=f"{s['label']} ({len(layers[n])})")
               for n, s in POLYGON_LAYERS.items() if n in layers]
    ax2.legend(handles=legend2, loc='upper right', fontsize=9,
               facecolor='#21262d', edgecolor='#30363d', labelcolor='white')
    ax2.set_title("SET 2: POLYGONS  (Nm,2 × Np × Dm)\nIntersections + crosswalks + stop lines → PolygonEncoder",
                  fontsize=13, color='white', pad=12)

    for ax in (ax1, ax2):
        ax.tick_params(colors='#8b949e', labelsize=7)
        ax.set_xlabel("UTM Easting (m)", color='#8b949e', fontsize=9)
        ax.set_ylabel("UTM Northing (m)", color='#8b949e', fontsize=9)
        for spine in ax.spines.values():
            spine.set_color('#30363d')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "06_polylines_vs_polygons_full_map.png")
    fig.savefig(path, dpi=150, facecolor='#0d1117')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_combined_full_map(layers):
    """Plot 2: Both sets overlaid on a single map."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    # Background
    if 'lanes_polygons' in layers:
        layers['lanes_polygons'].plot(ax=ax, color='#2c3e50', alpha=0.12, edgecolor='none')

    # Polylines first (underneath)
    for name, style in POLYLINE_LAYERS.items():
        if name in layers:
            layers[name].plot(ax=ax, color=style['color'], alpha=style['alpha'] * 0.6,
                              linewidth=style['lw'] * 0.7)

    # Polygons on top
    for name, style in POLYGON_LAYERS.items():
        if name in layers:
            layers[name].plot(ax=ax, color=style['color'], alpha=style['alpha'],
                              edgecolor=style['color'], linewidth=0.5)

    # Combined legend
    legend_items = []
    legend_items.append(mpatches.Patch(color='none', label='── SET 1: POLYLINES ──'))
    for n, s in POLYLINE_LAYERS.items():
        if n in layers:
            legend_items.append(mpatches.Patch(color=s['color'], alpha=s['alpha'],
                                label=f"  {s['label']} ({len(layers[n])})"))
    legend_items.append(mpatches.Patch(color='none', label='── SET 2: POLYGONS ──'))
    for n, s in POLYGON_LAYERS.items():
        if n in layers:
            legend_items.append(mpatches.Patch(color=s['color'], alpha=s['alpha'],
                                label=f"  {s['label']} ({len(layers[n])})"))

    ax.legend(handles=legend_items, loc='upper right', fontsize=9,
              facecolor='#21262d', edgecolor='#30363d', labelcolor='white')
    ax.set_title("Boston Map — POLYLINES + POLYGONS Combined\n"
                 "Concatenated → (Nm,1 + Nm,2) × D map features",
                 fontsize=14, color='white', pad=12)
    ax.tick_params(colors='#8b949e', labelsize=7)
    ax.set_xlabel("UTM Easting (m)", color='#8b949e', fontsize=9)
    ax.set_ylabel("UTM Northing (m)", color='#8b949e', fontsize=9)
    for spine in ax.spines.values():
        spine.set_color('#30363d')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "07_combined_polylines_polygons.png")
    fig.savefig(path, dpi=150, facecolor='#0d1117')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_ego_scenes_dual(layers, ego_poses, n_scenes=6):
    """Plot 3: Ego-centric scenes — each scene has polylines (left) and polygons (right)."""
    radius = cfg.MAP_QUERY_RADIUS
    n_scenes = min(n_scenes, len(ego_poses))

    fig, axes = plt.subplots(n_scenes, 2, figsize=(16, 5 * n_scenes))
    fig.patch.set_facecolor('#0d1117')
    if n_scenes == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_scenes):
        ego_x, ego_y, ego_h = ego_poses[i]
        cos_h = math.cos(-ego_h)
        sin_h = math.sin(-ego_h)
        clip_box = shapely_box(ego_x - radius, ego_y - radius,
                               ego_x + radius, ego_y + radius)

        def to_ego(coords):
            arr = np.array(coords)
            dx = arr[:, 0] - ego_x
            dy = arr[:, 1] - ego_y
            x_e = cos_h * dx - sin_h * dy
            y_e = sin_h * dx + cos_h * dy
            return np.column_stack([x_e, y_e])

        for col, (layer_set, set_name) in enumerate([
            (POLYLINE_LAYERS, "POLYLINES"),
            (POLYGON_LAYERS, "POLYGONS"),
        ]):
            ax = axes[i, col]
            ax.set_facecolor('#161b22')

            # Lane background in ego frame
            if 'lanes_polygons' in layers:
                clipped = gpd.clip(layers['lanes_polygons'], clip_box)
                for _, row in clipped.iterrows():
                    geom = row.geometry
                    if hasattr(geom, 'exterior'):
                        pts = to_ego(geom.exterior.coords)
                        poly = plt.Polygon(pts, closed=True, facecolor='#2c3e50',
                                           alpha=0.12, edgecolor='#3d4450', linewidth=0.2)
                        ax.add_patch(poly)

            counts = {}
            for name, style in layer_set.items():
                if name not in layers:
                    continue
                clipped = gpd.clip(layers[name], clip_box)
                counts[name] = len(clipped)
                for _, row in clipped.iterrows():
                    geom = row.geometry
                    if geom.geom_type in ('Polygon', 'MultiPolygon'):
                        if hasattr(geom, 'exterior'):
                            pts = to_ego(geom.exterior.coords)
                            poly = plt.Polygon(pts, closed=True,
                                               facecolor=style['color'],
                                               alpha=style['alpha'],
                                               edgecolor=style['color'],
                                               linewidth=0.8)
                            ax.add_patch(poly)
                    elif geom.geom_type in ('LineString', 'MultiLineString'):
                        pts = to_ego(geom.coords)
                        ax.plot(pts[:, 0], pts[:, 1], color=style['color'],
                                alpha=style['alpha'], linewidth=style.get('lw', 1.0))

            # Ego marker
            ax.plot(0, 0, 'w*', markersize=14, zorder=10)
            ax.annotate('', xy=(5, 0), xytext=(0, 0),
                         arrowprops=dict(arrowstyle='->', color='white', lw=2), zorder=10)

            # Query circle
            circle = plt.Circle((0, 0), radius, fill=False,
                                 edgecolor='#f0c674', linewidth=0.8, linestyle='--', alpha=0.4)
            ax.add_patch(circle)

            ax.set_xlim(-radius * 1.1, radius * 1.1)
            ax.set_ylim(-radius * 1.1, radius * 1.1)
            ax.set_aspect('equal')

            info_lines = [f"{layer_set[k]['label'].split('(')[0].strip()}: {v}"
                          for k, v in counts.items() if v > 0]
            info = "\n".join(info_lines) if info_lines else "None in range"
            ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=7,
                    verticalalignment='top', color='white',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

            if i == 0:
                ax.set_title(f"SET {'1' if col == 0 else '2'}: {set_name}",
                             fontsize=12, color='white', pad=8)

            ax.tick_params(colors='#8b949e', labelsize=6)
            ax.set_xlabel("x (forward, m)", color='#8b949e', fontsize=8)
            ax.set_ylabel("y (left, m)", color='#8b949e', fontsize=8)
            ax.axhline(0, color='white', alpha=0.08, linewidth=0.5)
            ax.axvline(0, color='white', alpha=0.08, linewidth=0.5)
            for spine in ax.spines.values():
                spine.set_color('#30363d')

        # Scene label on left side
        axes[i, 0].text(-0.12, 0.5, f"Scene {i+1}", transform=axes[i, 0].transAxes,
                         fontsize=11, color='white', rotation=90,
                         verticalalignment='center', fontweight='bold')

    # Shared legends
    leg1 = [mpatches.Patch(color=s['color'], alpha=s['alpha'],
            label=s['label'].split('(')[0].strip())
            for s in POLYLINE_LAYERS.values()]
    leg2 = [mpatches.Patch(color=s['color'], alpha=s['alpha'],
            label=s['label'])
            for s in POLYGON_LAYERS.values()]

    fig.legend(handles=leg1 + [mpatches.Patch(color='none', label='')] + leg2,
               loc='lower center', ncol=3, fontsize=9,
               facecolor='#21262d', edgecolor='#30363d', labelcolor='white')

    plt.tight_layout(rect=[0.02, 0.05, 1, 1])
    path = os.path.join(OUT_DIR, "08_ego_scenes_polylines_vs_polygons.png")
    fig.savefig(path, dpi=150, facecolor='#0d1117')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_ego_combined(layers, ego_poses, n_scenes=6):
    """Plot 4: Ego-centric with both sets overlaid — what the model sees."""
    radius = cfg.MAP_QUERY_RADIUS
    n_scenes = min(n_scenes, len(ego_poses))

    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.patch.set_facecolor('#0d1117')
    axes = axes.flatten()

    for i in range(n_scenes):
        ax = axes[i]
        ax.set_facecolor('#161b22')
        ego_x, ego_y, ego_h = ego_poses[i]
        cos_h = math.cos(-ego_h)
        sin_h = math.sin(-ego_h)
        clip_box = shapely_box(ego_x - radius, ego_y - radius,
                               ego_x + radius, ego_y + radius)

        def to_ego(coords):
            arr = np.array(coords)
            dx = arr[:, 0] - ego_x
            dy = arr[:, 1] - ego_y
            x_e = cos_h * dx - sin_h * dy
            y_e = sin_h * dx + cos_h * dy
            return np.column_stack([x_e, y_e])

        # Background lanes
        if 'lanes_polygons' in layers:
            clipped = gpd.clip(layers['lanes_polygons'], clip_box)
            for _, row in clipped.iterrows():
                geom = row.geometry
                if hasattr(geom, 'exterior'):
                    pts = to_ego(geom.exterior.coords)
                    poly = plt.Polygon(pts, closed=True, facecolor='#2c3e50',
                                       alpha=0.12, edgecolor='#3d4450', linewidth=0.2)
                    ax.add_patch(poly)

        polyline_count = 0
        polygon_count = 0

        # Draw polylines
        for name, style in POLYLINE_LAYERS.items():
            if name not in layers:
                continue
            clipped = gpd.clip(layers[name], clip_box)
            polyline_count += len(clipped)
            for _, row in clipped.iterrows():
                geom = row.geometry
                if geom.geom_type in ('LineString',):
                    pts = to_ego(geom.coords)
                    ax.plot(pts[:, 0], pts[:, 1], color=style['color'],
                            alpha=style['alpha'] * 0.7, linewidth=style['lw'] * 0.8)

        # Draw polygons on top
        for name, style in POLYGON_LAYERS.items():
            if name not in layers:
                continue
            clipped = gpd.clip(layers[name], clip_box)
            polygon_count += len(clipped)
            for _, row in clipped.iterrows():
                geom = row.geometry
                if hasattr(geom, 'exterior'):
                    pts = to_ego(geom.exterior.coords)
                    poly = plt.Polygon(pts, closed=True, facecolor=style['color'],
                                       alpha=style['alpha'], edgecolor=style['color'],
                                       linewidth=1.0)
                    ax.add_patch(poly)

        # Ego
        ax.plot(0, 0, 'w*', markersize=16, zorder=10)
        ax.annotate('', xy=(6, 0), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color='white', lw=2.5), zorder=10)
        circle = plt.Circle((0, 0), radius, fill=False,
                             edgecolor='#f0c674', linewidth=0.8, linestyle='--', alpha=0.4)
        ax.add_patch(circle)

        ax.set_xlim(-radius * 1.1, radius * 1.1)
        ax.set_ylim(-radius * 1.1, radius * 1.1)
        ax.set_aspect('equal')

        info = f"Polylines: {polyline_count}\nPolygons: {polygon_count}"
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        ax.set_title(f"Scene {i+1}", fontsize=11, color='white')
        ax.tick_params(colors='#8b949e', labelsize=6)
        ax.set_xlabel("x (forward, m)", color='#8b949e', fontsize=8)
        ax.set_ylabel("y (left, m)", color='#8b949e', fontsize=8)
        ax.axhline(0, color='white', alpha=0.08, linewidth=0.5)
        ax.axvline(0, color='white', alpha=0.08, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color('#30363d')

    for idx in range(n_scenes, len(axes)):
        axes[idx].set_visible(False)

    # Legend
    all_items = []
    all_items.append(mpatches.Patch(color='none', label='Set 1: Polylines'))
    for s in POLYLINE_LAYERS.values():
        all_items.append(mpatches.Patch(color=s['color'], alpha=s['alpha'],
                         label=f"  {s['label'].split('(')[0].strip()}"))
    all_items.append(mpatches.Patch(color='none', label='Set 2: Polygons'))
    for s in POLYGON_LAYERS.values():
        all_items.append(mpatches.Patch(color=s['color'], alpha=s['alpha'],
                         label=f"  {s['label']}"))

    fig.legend(handles=all_items, loc='lower center', ncol=4, fontsize=9,
               facecolor='#21262d', edgecolor='#30363d', labelcolor='white')

    fig.suptitle("Ego-Centric View — Both Map Encoding Sets Combined\n"
                 "Polylines (LaneEncoder) + Polygons (PolygonEncoder) → concat → Nm × D",
                 fontsize=14, color='white', y=1.01)
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    path = os.path.join(OUT_DIR, "09_ego_combined_both_sets.png")
    fig.savefig(path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def print_summary(layers):
    """Print summary table matching paper notation."""
    print("\n" + "=" * 70)
    print("PAPER MAP ENCODING SUMMARY")
    print("=" * 70)

    print("\nSET 1: POLYLINES → LaneEncoder (PointNet)")
    print(f"  Paper shape: Nm,1 × Np × 3Dm = {cfg.N_LANES} × {cfg.N_LANE_POINTS} × {cfg.D_POLYLINE_POINT}")
    print(f"  Output:      Nm,1 × D = {cfg.N_LANES} × {cfg.D_LANE}")
    total_polylines = 0
    for n, s in POLYLINE_LAYERS.items():
        if n in layers:
            c = len(layers[n])
            total_polylines += c
            print(f"    {s['label']:45s}: {c:5d}")
    print(f"    {'TOTAL':45s}: {total_polylines:5d}")

    print(f"\nSET 2: POLYGONS → PolygonEncoder (PointNet)")
    print(f"  Paper shape: Nm,2 × Np × Dm = {cfg.N_POLYGONS} × {cfg.N_LANE_POINTS} × {cfg.D_POLYGON_POINT}")
    print(f"  Output:      Nm,2 × D = {cfg.N_POLYGONS} × {cfg.D_LANE}")
    total_polygons = 0
    for n, s in POLYGON_LAYERS.items():
        if n in layers:
            c = len(layers[n])
            total_polygons += c
            print(f"    {s['label']:45s}: {c:5d}")
    print(f"    {'TOTAL':45s}: {total_polygons:5d}")

    print(f"\nCOMBINED MAP FEATURES:")
    print(f"  Concat: (Nm,1 + Nm,2) × D = ({cfg.N_LANES} + {cfg.N_POLYGONS}) × {cfg.D_LANE}")
    print(f"        = {cfg.N_LANES + cfg.N_POLYGONS} × {cfg.D_LANE}")
    print(f"\n  Status: Polylines ✅ IMPLEMENTED  |  Polygons ❌ NOT YET")
    print("=" * 70)


def main():
    print("=" * 60)
    print("Polylines vs Polygons — Boston Map Visualization")
    print("=" * 60)

    print("\nLoading layers...")
    layers = load_layers(GPKG_PATH)

    print(f"\nGetting ego poses...")
    ego_poses = get_ego_poses(n_poses=20)
    print(f"  Got {len(ego_poses)} poses")

    print(f"\nGenerating visualizations...")

    print("\n[1/4] Side-by-side full map...")
    plot_side_by_side_full_map(layers)

    print("[2/4] Combined full map...")
    plot_combined_full_map(layers)

    print("[3/4] Ego scenes — polylines vs polygons...")
    plot_ego_scenes_dual(layers, ego_poses, n_scenes=6)

    print("[4/4] Ego scenes — both combined...")
    plot_ego_combined(layers, ego_poses, n_scenes=6)

    print_summary(layers)


if __name__ == '__main__':
    main()
