"""Standalone test for the per-step route re-transform.

Two implementations are compared:
  current  — what model.py:874-897 does today (in-place rotation of x/y/sin_h/
             cos_h channels using cos_e3=cos(-ego_h), sin_e3=sin(-ego_h))
  clean    — the same pattern map_lanes uses at model.py:802-817 (extract
             heading via atan2, subtract ego_h, recompute sin/cos)

If the two agree on x/y/sin_h/cos_h within float tolerance for arbitrary ego
poses, the existing model fix is correct. If they disagree, the diff tells us
exactly which channel(s) need to change.

Run:
    paper/.venv/bin/python scripts/test_route_transform.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg


# ───────────────────────────────────────────────────────────────────────────
# Reference: analytical ground truth — compose the world-to-current-ego
# rotation for an arbitrary route point given in initial-ego frame.

def gt_transform(x0, y0, h0, ego_x, ego_y, ego_h):
    """Initial-ego-frame point (x0, y0, h0) → current-ego-frame point.

    With ego at (ego_x, ego_y, ego_h) in the initial-ego frame, a point
    transforms by:  (rotation by -ego_h after subtracting ego origin).
    """
    dx = x0 - ego_x
    dy = y0 - ego_y
    cos_h, sin_h = math.cos(ego_h), math.sin(ego_h)
    x_e =  cos_h * dx + sin_h * dy        # = cos(-h)*dx - sin(-h)*dy
    y_e = -sin_h * dx + cos_h * dy        # = sin(-h)*dx + cos(-h)*dy
    h_e = (h0 - ego_h + math.pi) % (2 * math.pi) - math.pi
    return x_e, y_e, h_e


# ───────────────────────────────────────────────────────────────────────────
# Implementation A: current model.py:874-897 (in-place after trim)

def current_inplace(trimmed: torch.Tensor, ego_x, ego_y, ego_h):
    """Mirrors model.py lines 881-897. Operates on a 39-dim per-point layout
    = 3 × 13 (center | left_boundary | right_boundary), each 13-dim block
    = [x, y, sin_h, cos_h, ...11 side-info channels].
    """
    cos_hr = torch.cos(-ego_h).unsqueeze(1).unsqueeze(1)
    sin_hr = torch.sin(-ego_h).unsqueeze(1).unsqueeze(1)
    cos_e3 = cos_hr.squeeze(-1)        # (B, 1, 1)
    sin_e3 = sin_hr.squeeze(-1)
    ex = ego_x.unsqueeze(1).unsqueeze(2)
    ey = ego_y.unsqueeze(1).unsqueeze(2)
    cos_e3 = cos_e3.unsqueeze(-1)
    sin_e3 = sin_e3.unsqueeze(-1)

    out = trimmed.clone()
    for off in (0, 13, 26):
        rxx = out[..., off + 0]
        ryy = out[..., off + 1]
        rs  = out[..., off + 2]
        rc  = out[..., off + 3]
        dxr = rxx - ex
        dyr = ryy - ey
        out[..., off + 0] = cos_e3 * dxr - sin_e3 * dyr
        out[..., off + 1] = sin_e3 * dxr + cos_e3 * dyr
        out[..., off + 2] = cos_e3 * rs  - sin_e3 * rc
        out[..., off + 3] = cos_e3 * rc  + sin_e3 * rs
    return out


# ───────────────────────────────────────────────────────────────────────────
# Implementation B: clean — same pattern as map_lanes (model.py:802-817)

def clean_recompute(trimmed: torch.Tensor, ego_x, ego_y, ego_h):
    """Extract heading via atan2, subtract ego_h, recompute sin/cos."""
    ego_xr = ego_x.unsqueeze(1).unsqueeze(2)
    ego_yr = ego_y.unsqueeze(1).unsqueeze(2)
    ego_hr = ego_h.unsqueeze(1).unsqueeze(2)
    cos_h  = torch.cos(-ego_hr)
    sin_h  = torch.sin(-ego_hr)

    blocks = []
    for off in (0, 13, 26):
        block = trimmed[..., off:off+13]
        px, py = block[..., 0], block[..., 1]
        ph = torch.atan2(block[..., 2], block[..., 3])

        dx = px - ego_xr
        dy = py - ego_yr
        x_e = cos_h * dx - sin_h * dy
        y_e = sin_h * dx + cos_h * dy
        h_e = (ph - ego_hr + math.pi) % (2 * math.pi) - math.pi

        blocks.append(torch.cat([
            x_e.unsqueeze(-1), y_e.unsqueeze(-1),
            torch.sin(h_e).unsqueeze(-1), torch.cos(h_e).unsqueeze(-1),
            block[..., 4:],
        ], dim=-1))
    return torch.cat(blocks, dim=-1)


# ───────────────────────────────────────────────────────────────────────────

def test_synthetic():
    """Build a known-input scenario and check both implementations against
    the analytical ground truth."""
    print("\n=== Synthetic test ===")
    # Single route, single point at (10, 3, π/4) in initial-ego frame.
    # Side-info channels filled with sentinels so we can verify they pass through.
    B, N_lat, N_r = 1, 1, 1
    pt = torch.zeros(B, N_lat, N_r, 39)
    h0 = math.pi / 4
    pt[..., 0]  = 10.0          # x
    pt[..., 1]  = 3.0           # y
    pt[..., 2]  = math.sin(h0)  # sin(h)
    pt[..., 3]  = math.cos(h0)  # cos(h)
    pt[..., 4:13] = 0.5         # center side-info
    # Mirror the layout: left + right blocks at offsets 13, 26 (use simple values)
    pt[..., 13] = 11.0; pt[..., 14] = 4.0; pt[..., 15] = math.sin(h0); pt[..., 16] = math.cos(h0)
    pt[..., 26] =  9.0; pt[..., 27] = 2.0; pt[..., 28] = math.sin(h0); pt[..., 29] = math.cos(h0)

    cases = [
        ("ego at origin, h=0 (no-op)",         (0.0, 0.0, 0.0)),
        ("ego moved forward 5 m, h=0",          (5.0, 0.0, 0.0)),
        ("ego rotated 90° at origin",           (0.0, 0.0, math.pi / 2)),
        ("ego forward 5 + rotated 30°",         (5.0, 1.0, math.pi / 6)),
        ("ego moved into lane left, rotated 45°", (3.0, 2.0, math.pi / 4)),
    ]
    for label, (ex, ey, eh) in cases:
        ego_x = torch.tensor([ex])
        ego_y = torch.tensor([ey])
        ego_h = torch.tensor([eh])

        cur   = current_inplace(pt.clone(), ego_x, ego_y, ego_h)
        clean = clean_recompute(pt.clone(), ego_x, ego_y, ego_h)

        # GT for the center block
        gx, gy, gh = gt_transform(10.0, 3.0, h0, ex, ey, eh)
        gt_sin, gt_cos = math.sin(gh), math.cos(gh)

        cur_xy   = (float(cur[0, 0, 0, 0]), float(cur[0, 0, 0, 1]))
        clean_xy = (float(clean[0, 0, 0, 0]), float(clean[0, 0, 0, 1]))
        cur_sc   = (float(cur[0, 0, 0, 2]), float(cur[0, 0, 0, 3]))
        clean_sc = (float(clean[0, 0, 0, 2]), float(clean[0, 0, 0, 3]))

        print(f"\n  Case: {label}")
        print(f"    GT     center: x={gx:7.3f}  y={gy:7.3f}  sin={gt_sin:7.3f}  cos={gt_cos:7.3f}")
        print(f"    current center: x={cur_xy[0]:7.3f}  y={cur_xy[1]:7.3f}  "
              f"sin={cur_sc[0]:7.3f}  cos={cur_sc[1]:7.3f}")
        print(f"    clean   center: x={clean_xy[0]:7.3f}  y={clean_xy[1]:7.3f}  "
              f"sin={clean_sc[0]:7.3f}  cos={clean_sc[1]:7.3f}")

        cur_xy_err   = math.hypot(cur_xy[0]-gx, cur_xy[1]-gy)
        clean_xy_err = math.hypot(clean_xy[0]-gx, clean_xy[1]-gy)
        cur_h_err    = math.hypot(cur_sc[0]-gt_sin, cur_sc[1]-gt_cos)
        clean_h_err  = math.hypot(clean_sc[0]-gt_sin, clean_sc[1]-gt_cos)
        print(f"    Δ(xy):  current={cur_xy_err:.3e}  clean={clean_xy_err:.3e}")
        print(f"    Δ(sc):  current={cur_h_err:.3e}  clean={clean_h_err:.3e}")

        # Verify side-info channels pass through unchanged in both
        assert torch.allclose(cur[..., 4:13], pt[..., 4:13]), "current zeroed center side-info"
        assert torch.allclose(clean[..., 4:13], pt[..., 4:13]), "clean zeroed center side-info"


# ───────────────────────────────────────────────────────────────────────────

def test_val14_sample(idx=500):
    """Apply both implementations to a real val14 sample and quantify divergence."""
    print(f"\n=== val14 sample {idx} (route shape, divergence per channel) ===")
    cache = torch.load('checkpoints/stage_cache_val14.pt', map_location='cpu', weights_only=False)
    routes = cache['route_polylines'][idx:idx+1]      # (1, N_lat, N_r, 39)
    rmask  = cache['route_mask'][idx:idx+1]           # (1, N_lat)
    gt     = cache['gt_trajectory'][idx]              # (T, 3)

    # Try ego at step 0 (origin) and step 7 (GT pose)
    for label, (ex, ey, eh) in [
        ("step 0 (origin, h=0)",     (0.0, 0.0, 0.0)),
        ("step 7 (GT pose)",         (float(gt[-1, 0]), float(gt[-1, 1]), float(gt[-1, 2]))),
    ]:
        ego_x = torch.tensor([ex])
        ego_y = torch.tensor([ey])
        ego_h = torch.tensor([eh])

        cur   = current_inplace(routes.clone(), ego_x, ego_y, ego_h)
        clean = clean_recompute(routes.clone(), ego_x, ego_y, ego_h)

        # Per-channel max abs diff between current and clean
        diff = (cur - clean).abs()
        # Channels of interest per sub-block: x, y, sin_h, cos_h
        print(f"\n  {label}  ego=({ex:.2f}, {ey:.2f}, {eh:.3f})")
        print(f"    Max |cur - clean| per channel (across all valid route points):")
        for off, name in [(0, "center"), (13, "left"), (26, "right")]:
            mx = diff[..., off + 0].max().item()
            my = diff[..., off + 1].max().item()
            ms = diff[..., off + 2].max().item()
            mc = diff[..., off + 3].max().item()
            print(f"      {name:<6s}  Δx={mx:.3e}  Δy={my:.3e}  Δsin={ms:.3e}  Δcos={mc:.3e}")


# ───────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(0)
    test_synthetic()
    test_val14_sample(500)
    test_val14_sample(800)


if __name__ == '__main__':
    main()
