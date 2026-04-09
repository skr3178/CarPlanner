"""
Test Option 3: PolygonEncoder on Boston cache data (derived polygon-like inputs).

Takes the first D_MAP_POINT=9 dims from existing map_lanes (centerline features)
and treats them as polygon-like inputs. Tests the full pipeline:
  - Load real cache data
  - Derive polygon tensors from map_lanes[:, :, :, :9]
  - Run through PolygonEncoder + CombinedMapEncoder
  - Validate shapes, statistics, gradient flow on real data
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import config as cfg
from test_polygon_encoder import PolygonEncoder, CombinedMapEncoder

CACHE_PATH = "/media/skr/storage/autoresearch/CarPlanner_Implementation/checkpoints/stage_cache_train_boston.pt"
N_SAMPLES = 256   # subset of cache to test


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Loading cache: {CACHE_PATH}")

    cache = torch.load(CACHE_PATH, map_location='cpu', weights_only=True)
    map_lanes = cache['map_lanes'][:N_SAMPLES].to(device)
    map_lanes_mask = cache['map_lanes_mask'][:N_SAMPLES].to(device)
    agents_now = cache['agents_now'][:N_SAMPLES].to(device)
    agents_mask = cache['agents_mask'][:N_SAMPLES].to(device)
    agents_seq = cache['agents_seq'][:N_SAMPLES].to(device)
    gt_traj = cache['gt_trajectory'][:N_SAMPLES].to(device)
    mode_label = cache['mode_label'][:N_SAMPLES].to(device)
    del cache  # free CPU memory

    print(f"Cache loaded — {N_SAMPLES} samples")
    print(f"  map_lanes:      {tuple(map_lanes.shape)}")
    print(f"  map_lanes_mask: {tuple(map_lanes_mask.shape)}")

    # ─── Derive polygon data from map_lanes ──────────────────────────────────
    # Take first 9 dims (centerline: x, y, sin_h, cos_h, speed_limit, 4×cat)
    # Select a subset of lanes as "polygons" — e.g. last N_POLYGONS lanes
    polygon_feats = map_lanes[:, -cfg.N_POLYGONS:, :, :cfg.D_POLYGON_POINT]
    polygon_mask = map_lanes_mask[:, -cfg.N_POLYGONS:]
    print(f"\n  Derived polygons: {tuple(polygon_feats.shape)}")
    print(f"  Derived poly mask: {tuple(polygon_mask.shape)}")

    # Polylines = remaining lanes (first N_LANES - N_POLYGONS)
    n_polyline_lanes = cfg.N_LANES - cfg.N_POLYGONS
    polyline_feats = map_lanes[:, :n_polyline_lanes, :, :]
    polyline_mask = map_lanes_mask[:, :n_polyline_lanes]
    print(f"  Polylines (remaining): {tuple(polyline_feats.shape)}")

    # ─── Statistics on real data ─────────────────────────────────────────────
    valid_poly = polygon_feats[polygon_mask.bool().unsqueeze(-1).unsqueeze(-1).expand_as(polygon_feats)]
    valid_poly = valid_poly.view(-1, cfg.D_POLYGON_POINT)
    print(f"\n  Polygon feature stats (valid only, {valid_poly.shape[0]} points):")
    for i, name in enumerate(['x', 'y', 'sin_h', 'cos_h', 'speed_limit', 'cat0', 'cat1', 'cat2', 'cat3']):
        col = valid_poly[:, i]
        print(f"    {name:12s}: mean={col.mean():.4f}  std={col.std():.4f}  "
              f"min={col.min():.4f}  max={col.max():.4f}")

    valid_count = polygon_mask.sum(dim=1)
    print(f"\n  Valid polygons per sample: mean={valid_count.mean():.1f}  "
          f"min={valid_count.min():.0f}  max={valid_count.max():.0f}")

    # ─── Test 1: PolygonEncoder on real data ─────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 1: PolygonEncoder on derived polygon data")
    print("=" * 60)
    encoder = PolygonEncoder().to(device)
    poly_out = encoder(polygon_feats, polygon_mask)
    assert poly_out.shape == (N_SAMPLES, cfg.N_POLYGONS, cfg.D_LANE)
    print(f"  Input:  {tuple(polygon_feats.shape)}")
    print(f"  Output: {tuple(poly_out.shape)}")
    print(f"  Output range: [{poly_out.min():.4f}, {poly_out.max():.4f}]")
    print(f"  Non-zero entries: {(poly_out != 0).sum().item()}/{poly_out.numel()}")
    loss = poly_out.sum()
    loss.backward()
    print(f"  Backward OK, loss={loss.item():.2f}")
    print("  PASSED\n")

    # ─── Test 2: CombinedMapEncoder on real data ─────────────────────────────
    print("=" * 60)
    print("TEST 2: CombinedMapEncoder on real polyline + derived polygon data")
    print("=" * 60)
    combined = CombinedMapEncoder().to(device)
    map_feats, pf, gf = combined(
        polyline_feats, polyline_mask,
        polygon_feats, polygon_mask
    )
    N_total = n_polyline_lanes + cfg.N_POLYGONS
    assert map_feats.shape == (N_SAMPLES, N_total, cfg.D_HIDDEN)
    print(f"  Polylines in:  {tuple(polyline_feats.shape)}")
    print(f"  Polygons in:   {tuple(polygon_feats.shape)}")
    print(f"  Combined out:  {tuple(map_feats.shape)}")
    print(f"  Map feat range: [{map_feats.min():.4f}, {map_feats.max():.4f}]")

    loss = map_feats.pow(2).mean()
    loss.backward()
    # Check gradients on both encoders
    poly_grads = sum(1 for p in combined.polyline_encoder.parameters()
                     if p.grad is not None and p.grad.abs().sum() > 0)
    gon_grads = sum(1 for p in combined.polygon_encoder.parameters()
                    if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  Polyline grads: {poly_grads} params")
    print(f"  Polygon grads:  {gon_grads} params")
    print(f"  Backward OK, loss={loss.item():.6f}")
    print("  PASSED\n")

    # ─── Test 3: Mini-batch training loop simulation ─────────────────────────
    print("=" * 60)
    print("TEST 3: Mini-batch training loop (5 steps)")
    print("=" * 60)
    combined = CombinedMapEncoder().to(device)
    optimizer = torch.optim.Adam(combined.parameters(), lr=1e-3)

    batch_size = 32
    for step in range(5):
        idx = torch.randint(0, N_SAMPLES, (batch_size,))
        pl = polyline_feats[idx]
        pm = polyline_mask[idx]
        pg = polygon_feats[idx]
        pgm = polygon_mask[idx]

        mf, _, _ = combined(pl, pm, pg, pgm)
        loss = mf.pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"  Step {step+1}: loss={loss.item():.6f}  "
              f"map_feat_norm={mf.norm(dim=-1).mean():.4f}")
    print("  PASSED\n")

    # ─── Test 4: Masking correctness on real data ────────────────────────────
    print("=" * 60)
    print("TEST 4: Masking correctness on real data")
    print("=" * 60)
    encoder = PolygonEncoder().to(device)
    # Use a sample where some polygons are masked
    for i in range(N_SAMPLES):
        m = polygon_mask[i]
        if 0 < m.sum() < cfg.N_POLYGONS:
            pg = polygon_feats[i:i+1]
            pm = polygon_mask[i:i+1]
            out = encoder(pg, pm)
            padded = out[0, m == 0]
            valid = out[0, m == 1]
            assert (padded == 0).all(), "Padded polygon features not zero"
            assert (valid != 0).any(), "All valid features are zero"
            print(f"  Sample {i}: {int(m.sum())} valid / {cfg.N_POLYGONS - int(m.sum())} padded")
            print(f"  Padded all-zero: True")
            print(f"  Valid non-zero:  {(valid != 0).sum().item()}/{valid.numel()}")
            break
    print("  PASSED\n")

    print("=" * 60)
    print("ALL OPTION 3 TESTS PASSED")
    print("=" * 60)


if __name__ == '__main__':
    main()
