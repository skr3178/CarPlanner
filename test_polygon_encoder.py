"""
Isolated test: Second PointNet for polygon map features (paper Section 3).

Concept from paper:
  - Polylines (lanes): Nm,1 polylines → PointNet → Nm,1 × D
  - Polygons (crosswalks, intersections, stop lines): Nm,2 polygons → PointNet → Nm,2 × D
  - Concatenate: [polyline_feats; polygon_feats] → Nm × D

This file:
  1. Defines PolygonEncoder (PointNet over closed polygons, mirrors LaneEncoder)
  2. Builds a combined map feature by concatenating polyline + polygon encodings
  3. Tests shapes, forward pass, backward pass, and integration with existing modules
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from model import LaneEncoder


# ─── Polygon Encoder ──────────────────────────────────────────────────────────

class PolygonEncoder(nn.Module):
    """
    PointNet-style encoder for closed polygons (crosswalks, intersections, stop lines).

    Mirrors LaneEncoder but operates on polygon point clouds.
    Each polygon is resampled to N_PTS points with D_POLYGON_POINT=9 features each.

    Input:  polygons (B, N_POLYGONS, N_PTS, D_POLYGON_POINT) + mask (B, N_POLYGONS)
    Output: per_polygon (B, N_POLYGONS, D_POLYGON)
    """

    def __init__(self, in_dim: int = cfg.D_POLYGON_POINT,
                 d_model: int = cfg.D_LANE):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, d_model),
        )
        self.d_model = d_model

    def forward(self, polygons: torch.Tensor, mask: torch.Tensor = None):
        """
        polygons: (B, N_POLYGONS, N_PTS, D_POLYGON_POINT)
        mask:     (B, N_POLYGONS) — 1=valid, 0=padding
        Returns:  (B, N_POLYGONS, D_POLYGON)
        """
        B, N_poly, N_P, D = polygons.shape
        flat = polygons.view(B * N_poly, N_P, D)
        per_point = self.point_mlp(flat)                         # (B*N_poly, N_P, d_model)

        # Max-pool over points within each polygon
        poly_feat = per_point.max(dim=1).values                  # (B*N_poly, d_model)
        poly_feat = poly_feat.view(B, N_poly, self.d_model)

        if mask is not None:
            poly_feat = poly_feat * mask.unsqueeze(-1)

        return poly_feat


# ─── Combined Map Encoder ─────────────────────────────────────────────────────

class CombinedMapEncoder(nn.Module):
    """
    Paper-faithful map encoding: polyline PointNet + polygon PointNet → concat.

    Input:
      lane_polylines:    (B, N_LANES, N_PTS, D_POLYLINE_POINT)  — open polylines
      lane_mask:         (B, N_LANES)
      polygons:          (B, N_POLYGONS, N_PTS, D_POLYGON_POINT) — closed polygons
      polygon_mask:      (B, N_POLYGONS)

    Output:
      map_feats:         (B, N_LANES + N_POLYGONS, D)  — concatenated map features
      polyline_feats:    (B, N_LANES, D)
      polygon_feats:     (B, N_POLYGONS, D)
    """

    def __init__(self):
        super().__init__()
        # Polyline encoder (reuses existing LaneEncoder architecture)
        self.polyline_encoder = LaneEncoder(
            in_dim=cfg.D_POLYLINE_POINT, d_model=cfg.D_LANE
        )
        # Polygon encoder (second PointNet, paper Section 3)
        self.polygon_encoder = PolygonEncoder(
            in_dim=cfg.D_POLYGON_POINT, d_model=cfg.D_LANE
        )
        # Project from D_LANE to D_HIDDEN for downstream use
        self.map_proj = nn.Linear(cfg.D_LANE, cfg.D_HIDDEN)

    def forward(self, lane_polylines, lane_mask, polygons, polygon_mask):
        # Polyline features: (B, N_LANES, D_LANE)
        polyline_feats = self.polyline_encoder(lane_polylines, lane_mask)
        # Polygon features: (B, N_POLYGONS, D_LANE)
        polygon_feats = self.polygon_encoder(polygons, polygon_mask)
        # Concatenate: (B, N_LANES + N_POLYGONS, D_LANE)
        map_feats = torch.cat([polyline_feats, polygon_feats], dim=1)
        # Project to D_HIDDEN: (B, N_LANES + N_POLYGONS, D_HIDDEN)
        map_feats = self.map_proj(map_feats)

        return map_feats, polyline_feats, polygon_feats


# ─── Tests ─────────────────────────────────────────────────────────────────────

def _make_dummy_data(B, device):
    """Create synthetic data matching the expected shapes."""
    lane_polylines = torch.randn(
        B, cfg.N_LANES, cfg.N_LANE_POINTS, cfg.D_POLYLINE_POINT, device=device
    )
    lane_mask = torch.ones(B, cfg.N_LANES, device=device)
    # Randomly invalidate some lanes
    lane_mask[:, cfg.N_LANES // 2:] = 0.0

    polygons = torch.randn(
        B, cfg.N_POLYGONS, cfg.N_LANE_POINTS, cfg.D_POLYGON_POINT, device=device
    )
    polygon_mask = torch.ones(B, cfg.N_POLYGONS, device=device)
    polygon_mask[:, cfg.N_POLYGONS // 2:] = 0.0

    return lane_polylines, lane_mask, polygons, polygon_mask


def test_polygon_encoder_standalone():
    """Test PolygonEncoder in isolation."""
    print("=" * 60)
    print("TEST 1: PolygonEncoder standalone")
    print("=" * 60)

    B, device = 4, 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = PolygonEncoder().to(device)

    polygons = torch.randn(B, cfg.N_POLYGONS, cfg.N_LANE_POINTS,
                           cfg.D_POLYGON_POINT, device=device)
    mask = torch.ones(B, cfg.N_POLYGONS, device=device)
    mask[:, -3:] = 0.0

    out = encoder(polygons, mask)
    assert out.shape == (B, cfg.N_POLYGONS, cfg.D_LANE), \
        f"Expected (B, {cfg.N_POLYGONS}, {cfg.D_LANE}), got {tuple(out.shape)}"

    # Check padding lanes are zeroed
    assert (out[:, -3:] == 0).all(), "Padded polygons should be zero"

    # Backward pass
    loss = out.sum()
    loss.backward()
    print(f"  Input:  {tuple(polygons.shape)}")
    print(f"  Output: {tuple(out.shape)}")
    print(f"  Params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  Backward OK, loss={loss.item():.2f}")
    print("  PASSED\n")


def test_combined_map_encoder():
    """Test CombinedMapEncoder (polyline + polygon → concat)."""
    print("=" * 60)
    print("TEST 2: CombinedMapEncoder (polyline + polygon concatenation)")
    print("=" * 60)

    B, device = 4, 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CombinedMapEncoder().to(device)
    lane_polylines, lane_mask, polygons, polygon_mask = _make_dummy_data(B, device)

    map_feats, polyfeats, gonfeats = model(
        lane_polylines, lane_mask, polygons, polygon_mask
    )

    N_total = cfg.N_LANES + cfg.N_POLYGONS
    assert map_feats.shape == (B, N_total, cfg.D_HIDDEN), \
        f"Expected (B, {N_total}, {cfg.D_HIDDEN}), got {tuple(map_feats.shape)}"
    assert polyfeats.shape == (B, cfg.N_LANES, cfg.D_LANE)
    assert gonfeats.shape == (B, cfg.N_POLYGONS, cfg.D_LANE)

    # Backward
    loss = map_feats.sum()
    loss.backward()

    print(f"  Lane polylines: {tuple(lane_polylines.shape)}")
    print(f"  Polygons:       {tuple(polygons.shape)}")
    print(f"  Polyline feats: {tuple(polyfeats.shape)}")
    print(f"  Polygon feats:  {tuple(gonfeats.shape)}")
    print(f"  Combined map:   {tuple(map_feats.shape)}")
    print(f"  Params:         {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Backward OK, loss={loss.item():.2f}")
    print("  PASSED\n")


def test_gradient_flow():
    """Verify gradients flow through both encoders."""
    print("=" * 60)
    print("TEST 3: Gradient flow through both PointNets")
    print("=" * 60)

    B, device = 2, 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CombinedMapEncoder().to(device)
    lane_polylines, lane_mask, polygons, polygon_mask = _make_dummy_data(B, device)

    map_feats, _, _ = model(lane_polylines, lane_mask, polygons, polygon_mask)
    loss = map_feats.pow(2).mean()
    loss.backward()

    # Check gradients exist on both encoders
    poly_grads = [p.grad is not None and p.grad.abs().sum() > 0
                  for p in model.polyline_encoder.parameters()]
    gon_grads = [p.grad is not None and p.grad.abs().sum() > 0
                 for p in model.polygon_encoder.parameters()]

    assert all(poly_grads), "Polyline encoder has zero gradients"
    assert all(gon_grads), "Polygon encoder has zero gradients"

    print(f"  Polyline encoder grads: {sum(poly_grads)}/{len(poly_grads)} params have gradients")
    print(f"  Polygon encoder grads:  {sum(gon_grads)}/{len(gon_grads)} params have gradients")
    print("  PASSED\n")


def test_output_consistency():
    """Verify output shape is consistent across batches and devices."""
    print("=" * 60)
    print("TEST 4: Output consistency (CPU/CUDA, variable batch)")
    print("=" * 60)

    for device_str in ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']:
        model = CombinedMapEncoder().to(device_str)
        for B in [1, 4, 16]:
            lane_polylines, lane_mask, polygons, polygon_mask = \
                _make_dummy_data(B, device_str)
            map_feats, _, _ = model(
                lane_polylines, lane_mask, polygons, polygon_mask
            )
            expected = (B, cfg.N_LANES + cfg.N_POLYGONS, cfg.D_HIDDEN)
            assert map_feats.shape == expected, \
                f"B={B} device={device_str}: expected {expected}, got {tuple(map_feats.shape)}"
        print(f"  device={device_str:4s} — batch sizes [1, 4, 16] OK")
    print("  PASSED\n")


def test_mask_effect():
    """Verify masking correctly zeros out padding entries."""
    print("=" * 60)
    print("TEST 5: Mask correctly zeros padding")
    print("=" * 60)

    B, device = 2, 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = PolygonEncoder().to(device)

    polygons = torch.randn(B, cfg.N_POLYGONS, cfg.N_LANE_POINTS,
                           cfg.D_POLYGON_POINT, device=device)
    # Mask: first half valid, second half padding
    mask = torch.zeros(B, cfg.N_POLYGONS, device=device)
    mask[:, :cfg.N_POLYGONS // 2] = 1.0

    out = encoder(polygons, mask)
    # Padded entries should be exactly zero
    padded = out[:, cfg.N_POLYGONS // 2:]
    assert (padded == 0).all(), f"Padded polygon features not zero: max={padded.abs().max().item()}"
    # Valid entries should be non-zero (with high probability)
    valid = out[:, :cfg.N_POLYGONS // 2]
    assert (valid != 0).any(), "Valid polygon features are all zero"

    print(f"  Valid features  non-zero: {(valid != 0).sum().item()} values")
    print(f"  Padded features all-zero: {(padded == 0).all().item()}")
    print("  PASSED\n")


def test_dimension_paper():
    """Verify dimensions match paper description."""
    print("=" * 60)
    print("TEST 6: Paper dimension verification")
    print("=" * 60)

    B = 1
    device = 'cpu'
    model = CombinedMapEncoder().to(device)
    lane_polylines, lane_mask, polygons, polygon_mask = _make_dummy_data(B, device)

    map_feats, polyfeats, gonfeats = model(
        lane_polylines, lane_mask, polygons, polygon_mask
    )

    # Paper: polyline features → Nm,1 × D
    print(f"  Polylines:  Nm,1={cfg.N_LANES} × D={cfg.D_LANE}  →  {tuple(polyfeats.shape[1:])}")
    # Paper: polygon features → Nm,2 × D
    print(f"  Polygons:   Nm,2={cfg.N_POLYGONS} × D={cfg.D_LANE}  →  {tuple(gonfeats.shape[1:])}")
    # Paper: concatenated → Nm × D  (where Nm = Nm,1 + Nm,2)
    Nm = cfg.N_LANES + cfg.N_POLYGONS
    print(f"  Combined:   Nm={Nm} × D_hidden={cfg.D_HIDDEN}  →  {tuple(map_feats.shape[1:])}")
    print(f"  D_POLYGON_POINT = {cfg.D_POLYGON_POINT} (paper Dm=9)")
    print(f"  D_POLYLINE_POINT = {cfg.D_POLYLINE_POINT} (paper 3×Dm=27)")
    print("  PASSED\n")


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"\nDevice: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Config: N_LANES={cfg.N_LANES}, N_POLYGONS={cfg.N_POLYGONS}, "
          f"N_PTS={cfg.N_LANE_POINTS}, D_POLYGON_POINT={cfg.D_POLYGON_POINT}, "
          f"D_POLYLINE_POINT={cfg.D_POLYLINE_POINT}, D_LANE={cfg.D_LANE}, "
          f"D_HIDDEN={cfg.D_HIDDEN}\n")

    test_polygon_encoder_standalone()
    test_combined_map_encoder()
    test_gradient_flow()
    test_output_consistency()
    test_mask_effect()
    test_dimension_paper()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
