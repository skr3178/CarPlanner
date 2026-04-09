"""
Test Option 2: PolygonEncoder with REAL nuPlan polygon geometry.

Extracts crosswalks, stop_polygons, and intersections from the nuPlan
GeoPackage map file, resamples to N_PTS points, encodes into ego-frame
features (D_POLYGON_POINT=9), and runs through PolygonEncoder.

This uses real polygon geometry — not derived from lanes.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import math
import numpy as np
import torch
import config as cfg
from test_polygon_encoder import PolygonEncoder, CombinedMapEncoder

GPKG_PATH = os.path.join(
    cfg.MAPS_DIR, "us-ma-boston", "9.12.1817", "map.gpkg"
)
CACHE_PATH = "/media/skr/storage/autoresearch/CarPlanner_Implementation/checkpoints/stage_cache_train_boston.pt"

# Polygon layers to extract (from GeoPackage)
POLYGON_LAYERS = ['crosswalks', 'stop_polygons', 'intersections']


def _resample_polygon_ring(ring_coords, n_points):
    """Resample a polygon ring (Nx2 array) to exactly n_points via arc-length."""
    if len(ring_coords) < 3:
        return np.zeros((n_points, 2), dtype=np.float32)

    pts = np.array(ring_coords, dtype=np.float32)

    # Close the ring if not already closed
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_len = cumlen[-1]

    if total_len < 1e-6:
        return np.zeros((n_points, 2), dtype=np.float32)

    sample_dists = np.linspace(0, total_len, n_points, endpoint=False)
    resampled = np.zeros((n_points, 2), dtype=np.float32)
    for i, d in enumerate(sample_dists):
        seg_idx = np.searchsorted(cumlen, d) - 1
        seg_idx = max(0, min(seg_idx, len(pts) - 2))
        seg_start = cumlen[seg_idx]
        seg_len = seg_lengths[seg_idx]
        t = (d - seg_start) / seg_len if seg_len > 1e-6 else 0.0
        resampled[i] = pts[seg_idx] + t * (pts[seg_idx + 1] - pts[seg_idx])

    return resampled


def _to_ego_frame(x, y, ref_x, ref_y, ref_heading):
    """Transform global (x, y) to ego-centric frame."""
    dx = x - ref_x
    dy = y - ref_y
    cos_h = math.cos(-ref_heading)
    sin_h = math.sin(-ref_heading)
    x_e = cos_h * dx - sin_h * dy
    y_e = sin_h * dx + cos_h * dy
    return x_e, y_e


def _encode_polygon_pts(pts_2d, category_idx, ref_x, ref_y, ref_heading):
    """
    Encode resampled polygon points (N_PTS × 2) into features (N_PTS × 9).

    Features: x, y, sin(h), cos(h), 0 (no speed limit), 4×category_onehot
    """
    N = len(pts_2d)
    headings = np.zeros(N, dtype=np.float32)
    for j in range(N - 1):
        dx = pts_2d[j+1, 0] - pts_2d[j, 0]
        dy = pts_2d[j+1, 1] - pts_2d[j, 1]
        headings[j] = math.atan2(dy, dx)
    headings[-1] = headings[-2]

    cat_onehot = np.zeros(4, dtype=np.float32)
    cat_onehot[min(category_idx, 3)] = 1.0

    feats = np.zeros((N, cfg.D_POLYGON_POINT), dtype=np.float32)
    for j in range(N):
        x_e, y_e = _to_ego_frame(pts_2d[j, 0], pts_2d[j, 1], ref_x, ref_y, ref_heading)
        h_e = headings[j] - ref_heading
        h_e = (h_e + math.pi) % (2 * math.pi) - math.pi
        feats[j] = np.concatenate([
            [x_e, y_e, math.sin(h_e), math.cos(h_e), 0.0],  # no speed_limit for polygons
            cat_onehot,
        ])
    return feats


def extract_polygons_for_samples(n_samples=256):
    """
    Extract real polygons from GeoPackage, transform to ego frame for each sample.

    Uses ego poses from the Boston cache to define the reference frame.
    Returns polygon tensor and mask ready for PolygonEncoder.
    """
    import geopandas as gpd
    from pyproj import Transformer

    print(f"Loading GeoPackage: {GPKG_PATH}")
    all_layers = {}
    layer_cat_idx = {'crosswalks': 0, 'stop_polygons': 1, 'intersections': 2}
    for layer_name in POLYGON_LAYERS:
        gdf = gpd.read_file(GPKG_PATH, layer=layer_name)
        print(f"  {layer_name}: {len(gdf)} polygons")
        all_layers[layer_name] = gdf

    # GeoPackage coords are in WGS84 (lon/lat), need to convert to UTM for distance queries
    # nuPlan uses UTM zone 19N for Boston (EPSG:32619)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32619", always_xy=True)

    # Convert all polygon geometries to UTM
    utm_polygons = {}
    for layer_name, gdf in all_layers.items():
        gdf_utm = gdf.to_crs("EPSG:32619")
        utm_polygons[layer_name] = gdf_utm

    # Load ego poses from Boston mini DBs (not cache — cache is in ego frame)
    print(f"\nLoading ego poses from Boston mini DBs...")
    import sqlite3, glob

    db_files = sorted(glob.glob(os.path.join(cfg.MINI_DIR, "*.db")))
    # Filter for Boston DBs only
    boston_dbs = []
    for db in db_files:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute("SELECT location FROM log LIMIT 1")
        loc = str(cur.fetchone()[0])
        conn.close()
        if 'boston' in loc.lower():
            boston_dbs.append(db)
    print(f"  Found {len(boston_dbs)} Boston DB files")

    # Collect global ego poses from Boston DBs (UTM coords)
    global_poses = []  # (x, y, heading) in UTM
    for db in boston_dbs:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute("SELECT x, y, qw, qx, qy, qz FROM ego_pose ORDER BY timestamp")
        for row in cur.fetchall():
            x, y, qw, qx, qy, qz = row
            heading = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy*qy + qz*qz))
            global_poses.append((x, y, heading))
        conn.close()
        if len(global_poses) >= n_samples:
            break
    global_poses = global_poses[:n_samples]
    print(f"  Got {len(global_poses)} Boston global ego poses")
    if len(global_poses) == 0:
        print("  ERROR: No Boston ego poses found!")
        return None, None, layer_counts

    # For each sample, find nearby polygons and encode them
    N_PTS = cfg.N_LANE_POINTS
    polygons = np.zeros((n_samples, cfg.N_POLYGONS, N_PTS, cfg.D_POLYGON_POINT), dtype=np.float32)
    polygons_mask = np.zeros((n_samples, cfg.N_POLYGONS), dtype=np.float32)

    query_radius = cfg.MAP_QUERY_RADIUS

    for i in range(n_samples):
        if i >= len(global_poses):
            break
        ego_x, ego_y, ego_h = global_poses[i]

        collected = []
        for layer_name, gdf_utm in utm_polygons.items():
            cat_idx = layer_cat_idx[layer_name]
            for _, row in gdf_utm.iterrows():
                geom = row.geometry
                # Quick distance filter: check if polygon centroid is within radius
                cx, cy = geom.centroid.x, geom.centroid.y
                dist = math.sqrt((cx - ego_x)**2 + (cy - ego_y)**2)
                if dist > query_radius:
                    continue

                # Get exterior ring coords
                ring = np.array(geom.exterior.coords, dtype=np.float64)[:, :2]
                # Resample to N_PTS
                ring_resampled = _resample_polygon_ring(ring, N_PTS)
                # Encode in ego frame
                feats = _encode_polygon_pts(ring_resampled, cat_idx, ego_x, ego_y, ego_h)
                collected.append(feats)

        # Take up to N_POLYGONS closest
        if collected:
            # Sort by distance to ego (mean of point distances)
            def mean_dist(f):
                return np.sqrt(f[:, 0]**2 + f[:, 1]**2).mean()
            collected.sort(key=mean_dist)
            n_take = min(len(collected), cfg.N_POLYGONS)
            for j in range(n_take):
                polygons[i, j] = collected[j]
                polygons_mask[i, j] = 1.0

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{n_samples} samples")

    return (torch.from_numpy(polygons), torch.from_numpy(polygons_mask),
            {k: len(v) for k, v in utm_polygons.items()})


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # ─── Extract real polygons ───────────────────────────────────────────────
    polygons, polygon_mask, layer_counts = extract_polygons_for_samples(n_samples=256)
    polygons = polygons.to(device)
    polygon_mask = polygon_mask.to(device)

    print(f"\nExtracted polygons: {tuple(polygons.shape)}")
    print(f"Polygon mask:       {tuple(polygon_mask.shape)}")
    valid_per_sample = polygon_mask.sum(dim=1)
    print(f"Valid polygons/sample: mean={valid_per_sample.mean():.1f}  "
          f"min={valid_per_sample.min():.0f}  max={valid_per_sample.max():.0f}")
    print(f"Samples with 0 polygons: {(valid_per_sample == 0).sum().item()}")
    print(f"Layer counts: {layer_counts}")

    # ─── Statistics on real polygon features ─────────────────────────────────
    valid_mask = polygon_mask.bool()
    valid_pts = polygons[valid_mask.unsqueeze(-1).unsqueeze(-1).expand_as(polygons)]
    valid_pts = valid_pts.view(-1, cfg.D_POLYGON_POINT)
    if len(valid_pts) > 0:
        print(f"\nReal polygon feature stats ({valid_pts.shape[0]} valid points):")
        for i, name in enumerate(['x', 'y', 'sin_h', 'cos_h', 'speed_limit', 'cat0', 'cat1', 'cat2', 'cat3']):
            col = valid_pts[:, i]
            print(f"  {name:12s}: mean={col.mean():.4f}  std={col.std():.4f}  "
                  f"min={col.min():.4f}  max={col.max():.4f}")

    # ─── Test 1: PolygonEncoder on real polygon data ─────────────────────────
    print("\n" + "=" * 60)
    print("TEST 1: PolygonEncoder on real polygon geometry")
    print("=" * 60)
    encoder = PolygonEncoder().to(device)
    poly_out = encoder(polygons, polygon_mask)
    assert poly_out.shape == (256, cfg.N_POLYGONS, cfg.D_LANE)
    print(f"  Input:  {tuple(polygons.shape)}")
    print(f"  Output: {tuple(poly_out.shape)}")
    print(f"  Output range: [{poly_out.min():.4f}, {poly_out.max():.4f}]")
    # Check padded are zero
    if (valid_per_sample < cfg.N_POLYGONS).any():
        padded = poly_out[polygon_mask == 0]
        assert (padded == 0).all(), "Padded polygon features not zero"
        print(f"  Padded entries zeroed: OK")
    loss = poly_out.sum()
    loss.backward()
    print(f"  Backward OK, loss={loss.item():.2f}")
    print("  PASSED\n")

    # ─── Test 2: Combined with real polylines from cache ─────────────────────
    print("=" * 60)
    print("TEST 2: CombinedMapEncoder — real polylines + real polygons")
    print("=" * 60)
    cache = torch.load(CACHE_PATH, map_location=device, weights_only=True)
    # Use corresponding samples from cache
    # Note: polygon samples come from mini DB, cache is train_boston — different scenes.
    # So we just validate the pipeline works with real data shapes.
    map_lanes = cache['map_lanes'][:256]
    map_lanes_mask = cache['map_lanes_mask'][:256]

    combined = CombinedMapEncoder().to(device)
    map_feats, pf, gf = combined(map_lanes, map_lanes_mask, polygons, polygon_mask)
    N_total = cfg.N_LANES + cfg.N_POLYGONS
    assert map_feats.shape == (256, N_total, cfg.D_HIDDEN)
    print(f"  Polylines (real): {tuple(map_lanes.shape)}")
    print(f"  Polygons (real):  {tuple(polygons.shape)}")
    print(f"  Combined out:     {tuple(map_feats.shape)}")
    print(f"  Map feat range:   [{map_feats.min():.4f}, {map_feats.max():.4f}]")
    loss = map_feats.pow(2).mean()
    loss.backward()
    print(f"  Backward OK, loss={loss.item():.6f}")
    print("  PASSED\n")

    # ─── Test 3: Training loop simulation ────────────────────────────────────
    print("=" * 60)
    print("TEST 3: Training loop with real polygons (5 steps)")
    print("=" * 60)
    combined = CombinedMapEncoder().to(device)
    optimizer = torch.optim.Adam(combined.parameters(), lr=1e-3)
    batch_size = 32

    for step in range(5):
        idx = torch.randint(0, min(256, polygons.size(0)), (batch_size,))
        ml = map_lanes[idx]
        mm = map_lanes_mask[idx]
        pg = polygons[idx]
        pm = polygon_mask[idx]

        mf, _, _ = combined(ml, mm, pg, pm)
        loss = mf.pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"  Step {step+1}: loss={loss.item():.6f}")
    print("  PASSED\n")

    # ─── Summary ─────────────────────────────────────────────────────────────
    print("=" * 60)
    print("ALL OPTION 2 TESTS PASSED")
    print(f"Real polygon layers used: {list(layer_counts.keys())}")
    print(f"Total polygons in map:    {sum(layer_counts.values())}")
    print("=" * 60)


if __name__ == '__main__':
    main()
