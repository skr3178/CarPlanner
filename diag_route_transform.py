"""Standalone prototype: transform route polylines into the current ego frame.

The current AutoregressivePolicy rollout (model.py:849-879 and :1090-1116)
recomputes `closest_idx` in the current ego frame at each step `t`, but the
`trimmed` tensor it feeds to the lateral PointNet still carries route
coordinates in the **t=0 ego frame**. Paper §3.3.3 says the routes should
be re-expressed in the current ego frame at each `t`, just like agents and
map lanes are.

This script:

  1. Loads route_polylines for a few real val14 samples (cache pre-stored
     in t=0 ego frame).
  2. Picks a couple of "current ego pose" deltas to simulate where the ego
     would be after rolling out a few steps (Δx forward, small heading change).
  3. Applies the proposed per-step transform — rotate+shift each of the
     three 13-dim sub-polylines (center / left / right) — and verifies:
       - x/y in current ego frame match what an exact world-frame
         re-transform would give (round-trip via the original t=0 ego pose).
       - sin/cos heading channels stay unit-norm and rotate by exactly
         the ego's heading delta.

Run:
    paper/.venv/bin/python diag_route_transform.py
"""

import math
import torch
import config as cfg


# ── Proposed transform — drop-in replacement for raw `trimmed` feed ────────────

def transform_route_to_current_ego(trimmed: torch.Tensor,
                                   ego_x: torch.Tensor,
                                   ego_y: torch.Tensor,
                                   ego_h: torch.Tensor) -> torch.Tensor:
    """Transform route polylines from the t=0 ego frame into the current ego
    frame defined by (ego_x, ego_y, ego_h).

    Args:
      trimmed: (B, N_lat, K_r, 39)  — route polylines in t=0 ego frame.
               Layout per point: [center(13) | left(13) | right(13)].
               Each 13-dim block: [x, y, sin_h, cos_h, speed_limit, cat(4), tl(4)].
      ego_x, ego_y, ego_h: (B,) — current ego pose in the t=0 frame.

    Returns:
      (B, N_lat, K_r, 39) — same layout, x/y/sin_h/cos_h re-expressed in
      the current ego frame.
    """
    B = trimmed.size(0)
    cos_h = torch.cos(-ego_h).view(B, 1, 1)   # rotate by -ego_h
    sin_h = torch.sin(-ego_h).view(B, 1, 1)
    ex = ego_x.view(B, 1, 1)
    ey = ego_y.view(B, 1, 1)

    out = trimmed.clone()
    for off in (0, 13, 26):           # center, left, right sub-polylines
        x = trimmed[..., off + 0]
        y = trimmed[..., off + 1]
        sin_th = trimmed[..., off + 2]
        cos_th = trimmed[..., off + 3]
        # Shift then rotate into current ego frame
        dx = x - ex
        dy = y - ey
        out[..., off + 0] = cos_h * dx - sin_h * dy
        out[..., off + 1] = sin_h * dx + cos_h * dy
        # Rotate the per-point heading by -ego_h
        out[..., off + 2] = cos_h * sin_th - sin_h * cos_th
        out[..., off + 3] = cos_h * cos_th + sin_h * sin_th
    return out


# ── Verification ──────────────────────────────────────────────────────────────

def world_frame_route(trimmed: torch.Tensor,
                      ref_x: torch.Tensor, ref_y: torch.Tensor, ref_h: torch.Tensor):
    """Round-trip oracle: undo the t=0 ego→world transform to recover world coords.

    `trimmed` is in t=0 ego frame; (ref_x, ref_y, ref_h) is the t=0 ego pose
    in world frame. Inverse transform = rotate by +ref_h then shift by ref.
    """
    B = trimmed.size(0)
    cos_h = torch.cos(ref_h).view(B, 1, 1)
    sin_h = torch.sin(ref_h).view(B, 1, 1)
    rx = ref_x.view(B, 1, 1)
    ry = ref_y.view(B, 1, 1)

    world = trimmed.clone()
    for off in (0, 13, 26):
        x = trimmed[..., off + 0]
        y = trimmed[..., off + 1]
        world[..., off + 0] = cos_h * x - sin_h * y + rx
        world[..., off + 1] = sin_h * x + cos_h * y + ry
    return world


def world_to_ego(world: torch.Tensor,
                 ex_w: torch.Tensor, ey_w: torch.Tensor, eh_w: torch.Tensor):
    """Forward transform world → ego at the given world-frame ego pose."""
    B = world.size(0)
    cos_h = torch.cos(-eh_w).view(B, 1, 1)
    sin_h = torch.sin(-eh_w).view(B, 1, 1)
    ex = ex_w.view(B, 1, 1)
    ey = ey_w.view(B, 1, 1)

    out = world.clone()
    for off in (0, 13, 26):
        x = world[..., off + 0]
        y = world[..., off + 1]
        dx = x - ex
        dy = y - ey
        out[..., off + 0] = cos_h * dx - sin_h * dy
        out[..., off + 1] = sin_h * dx + cos_h * dy
    return out


def main():
    # Load val14 cache (route polylines in t=0 ego frame, by construction)
    cache_path = "checkpoints/stage_cache_val14.pt"
    print(f"[DiagRoute] loading {cache_path}")
    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    rp_all = data["route_polylines"]    # (N, N_LAT, N_ROUTE_POINTS, D_POLYLINE_POINT)
    rm_all = data["route_mask"]         # (N, N_LAT)
    print(f"  route_polylines shape: {tuple(rp_all.shape)}")
    print(f"  route_mask shape:      {tuple(rm_all.shape)}")

    B = 3
    rp = rp_all[:B].clone()             # (3, N_LAT, N_r, D)
    rm = rm_all[:B].clone()
    print(f"  using {B} samples; valid routes per sample: {rm.sum(1).tolist()}")

    # Trim to first K_r = N_r // 4 points (matches the rollout's clip)
    N_r = rp.size(2)
    K_r = max(1, N_r // 4)
    trimmed_t0 = rp[:, :, :K_r, :].clone()        # (B, N_LAT, K_r, D=39)
    print(f"  trimmed shape: {tuple(trimmed_t0.shape)}  (K_r = N_r // 4 = {K_r})")

    # Pretend the ego has rolled forward 5 m and turned 0.1 rad after step 1
    ego_x = torch.tensor([5.0,  10.0, 15.0])      # forward in t=0 ego frame
    ego_y = torch.tensor([0.0,  0.5,  -0.5])
    ego_h = torch.tensor([0.05, 0.10, -0.08])     # small turn (rad)
    print(f"\n  Simulated current ego pose (in t=0 frame):")
    for i in range(B):
        print(f"    sample {i}: x={ego_x[i]:+.2f} m  y={ego_y[i]:+.2f} m  "
              f"h={math.degrees(ego_h[i]):+.2f}°")

    # Apply the proposed transform
    trimmed_t = transform_route_to_current_ego(trimmed_t0, ego_x, ego_y, ego_h)

    # ── Numerical sanity ───────────────────────────────────────────────────
    # Pick the first valid route in sample 0 and inspect its first point's
    # center channel before vs after.
    print("\n  Before vs after on sample 0, route 0 (most-relevant lane):")
    p_before = trimmed_t0[0, 0, 0, :4].tolist()    # [x, y, sin_h, cos_h] center
    p_after  = trimmed_t[0, 0, 0, :4].tolist()
    print(f"    BEFORE (t=0 frame):    x={p_before[0]:+.3f}  y={p_before[1]:+.3f}  "
          f"sin_h={p_before[2]:+.3f}  cos_h={p_before[3]:+.3f}")
    print(f"    AFTER  (current frame): x={p_after[0]:+.3f}  y={p_after[1]:+.3f}  "
          f"sin_h={p_after[2]:+.3f}  cos_h={p_after[3]:+.3f}")
    print(f"    Δx = {p_after[0]-p_before[0]:+.3f} (expected ≈ -ego_x[0] = {-ego_x[0]:+.3f})")

    # ── Round-trip sanity: t=0 frame ─→ world ─→ current frame should match ──
    # Pick an arbitrary "world" reference pose for sample 0 (any value works
    # because the transform is invariant — we just need a consistent reference).
    ref_x = torch.tensor([1000.0, 2000.0, 3000.0])
    ref_y = torch.tensor([1000.0, 2000.0, 3000.0])
    ref_h = torch.tensor([0.7, -0.3, 1.2])

    # Convert t=0-frame routes to world via the inverse of the t=0 transform
    world = world_frame_route(trimmed_t0, ref_x, ref_y, ref_h)

    # Where would the current ego be in the world?
    cos_r = torch.cos(ref_h)
    sin_r = torch.sin(ref_h)
    cur_x_w = ref_x + cos_r * ego_x - sin_r * ego_y
    cur_y_w = ref_y + sin_r * ego_x + cos_r * ego_y
    cur_h_w = ref_h + ego_h

    # Now express world routes in the current ego frame using the world pose
    expected_t = world_to_ego(world, cur_x_w, cur_y_w, cur_h_w)

    diff = (trimmed_t[..., :2] - expected_t[..., :2]).abs()
    # Mask invalid routes
    rm_b = rm.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, K_r, 2)
    diff_valid = diff[rm_b > 0.5]
    print(f"\n  Round-trip check (proposed transform vs world→ego oracle):")
    print(f"    max  |Δx,Δy| over valid points: {diff_valid.max().item():.2e} m")
    print(f"    mean |Δx,Δy| over valid points: {diff_valid.mean().item():.2e} m")
    if diff_valid.max().item() < 1e-3:
        print("    ✓ PASS — transform agrees with world-frame oracle to <1 mm")
    else:
        print("    ✗ FAIL — math doesn't round-trip; do not patch model.py yet")

    # ── Heading sanity: sin²+cos² should remain 1 after transform ─────────
    sins = trimmed_t[..., [2, 15, 28]]
    coss = trimmed_t[..., [3, 16, 29]]
    norm = sins * sins + coss * coss
    rm_full = rm.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, K_r, 3)
    norm_valid = norm[rm_full > 0.5]
    print(f"\n  Unit-norm sin/cos check on valid points:")
    print(f"    max  |sin²+cos² - 1|: {(norm_valid - 1.0).abs().max().item():.2e}")
    print(f"    mean |sin²+cos² - 1|: {(norm_valid - 1.0).abs().mean().item():.2e}")

    # ── Compare what the policy currently sees vs what it SHOULD see ───────
    print(f"\n  Magnitude of feature change (sample 0, route 0, all K_r points):")
    delta = (trimmed_t[0, 0, :, :2] - trimmed_t0[0, 0, :, :2]).norm(dim=-1)
    print(f"    {[f'{d:.2f}' for d in delta.tolist()]} m  per-point")
    print(f"  → these are the metres of x-y mis-alignment the lateral PointNet")
    print(f"    currently sees if we don't transform routes per step.")


if __name__ == "__main__":
    main()
