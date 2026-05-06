# Plan: Lateral-Collapse Fix and Validation

Author: Apr 27 2026
Status: planning — not yet executed

## Background

After the Apr 27 fix-and-retrain session, the open-loop and closed-loop pipelines
are paper-faithful in their definitions and metrics. The remaining gap is
**lateral mode collapse** in the trajectory generator:

- top-1 / top-5 = 56.4 / 90.4 % (selector head healthy)
- L_sel = 1.37 (vs paper IL-best 1.04)
- lon consistency = 44.16 % (matches paper 43.01 %)
- **lat consistency = 20.20 %** flat across 30 retrain epochs

The model is learning the right speed / longitudinal mode structure, but the 60
generated trajectories collapse onto essentially one lateral family. The recipe
change (batch 1280→64, patience 5→0) does not move the number — confirmed
empirically.

## Most likely cause

Reviewer findings (verified against `model.py`, `data_loader.py`,
`config.py`):

1. **Per-step route re-transform is missing.** The policy trims routes using
   the ego pose at autoregressive step `t`, but feeds the original t=0-frame
   route features into `route_pointnet`. The trim index uses current frame; the
   features themselves do not. Map lanes are transformed correctly. **This is a
   real paper deviation and the strongest code-level remaining bug.**
2. **Route resolution is too low.** `config.py` has `N_ROUTE_POINTS = 10`, and
   the IVM keeps `K_r = N_r / 4`, so the decoder sees only ~2-3 trimmed route
   points per lateral mode per step over an 8 s, ~120 m horizon.
3. **Routes themselves are distinct.** Per the reviewer's empirical check on
   val14: cache averages 2.3 valid route bins per sample, and route polylines
   for distinct lat bins occupy distinct y ranges. The policy ignores this
   diversity — it is not a route-extraction problem.

## Plan

Ordered cheapest-to-most-expensive with explicit exit criteria so we don't pay
for cache re-extraction unless needed.

### Phase 0 — Diagnose before changing anything (1-2 hours)

Build `scripts/diag_route_usage.py`. For ~5 val14 samples, save:

**Pictures (PNG, 4 panels per sample):**

1. Current ego frame at step 0: ego at origin, GT trajectory, agents.
2. The 5 cached route polylines (after route_mask), color-coded by lat bin.
3. The trimmed segment per lat bin (post-IVM, what the policy actually consumes).
4. Five candidate trajectories at a fixed lon bin, one per lat bin, plus the
   GT trajectory.

**Numeric checks (JSON sidecar — these are what tell us *where* the collapse is):**

- Route feature L2 norm per lat bin (before and after `route_pointnet`).
- Initial `mode_query` pairwise cosine distance across the 5 lat bins
  (post-route-injection).
- Endpoint y-coord spread across the 5 lat bins for the fixed lon bin.
- All four numbers at autoregressive step 0 *and* step 7 (tests whether the
  trimmed route view actually changes as ego moves — the reviewer's per-step
  transform hypothesis).

**Decision tree from Phase 0 output:**

| Picture | Numeric | Interpretation |
|---|---|---|
| Routes overlap visually | route feature norms ~equal across bins | Route extraction broken — fix in `_extract_routes` |
| Routes distinct | norms differ, mode_query collapsed | Route injection in decoder broken — fix in route attention path |
| Routes distinct, mode_query distinct | endpoint spread still ~0 | Action head collapse — needs per-mode L_gen |
| Step 7 routes ≡ step 0 routes (no change with ego motion) | trimmed segment unchanged | **Per-step route transform missing** — Phase 1 fix |

### Phase 1 — Per-step route re-transform in model.py (no cache regen)

**Time:** ~30 min code + ~2 h retrain + ~10 min eval

1. Read `AutoregressivePolicy.forward()` and `forward_rl()`. Locate the route
   trim block.
2. Confirm: trim index uses current ego pose, but route features fed to
   `route_pointnet` are in initial-ego frame.
3. Add re-transform of route xy/yaw/heading channels to current ego frame
   *before* `route_pointnet`, identical to how `map_lanes` are re-transformed
   per step.
4. Unit test:
   - Step 0 route features unchanged from current behavior.
   - Step 7 route features measurably different (ego has moved ~80 m forward).
5. Retrain Stage B at paper recipe (batch=64, patience=0, same Stage A frozen
   checkpoint, `train_4city_paper_balanced` cache).
6. Re-evaluate.

**Exit criterion:** lat consistency **≥ 30 %** post-retrain. If yes → Phase 4.
If no → Phase 2.

### Phase 2 — Higher route resolution (cache regen + retrain)

**Time:** ~6-8 h re-extract + ~2 h retrain + ~10 min eval

1. `config.py`: `N_ROUTE_POINTS = 10 → 20`. `K_r = N_r/4` automatically rises
   from 2-3 to 5 trimmed route points per lat bin per step.
2. Re-extract per-city caches: boston, vegas, pittsburgh, singapore.
3. Re-merge 4-city paper-balanced cache.
4. Re-extract val14, test14_random, reduced_val14.
5. Retrain Stage B same recipe.

**Exit criterion:** lat consistency **≥ 50 %**. If yes → Phase 4. If no →
Phase 3.

### Phase 3 — Per-mode L_gen (only if Phases 1+2 leave lat collapsed)

**Time:** ~30 min code + ~2 h retrain + ~10 min eval

`compute_il_loss` in `train_stage_b.py`: apply
`L1(all_trajs[:, m], gt_traj)` for each `m ≠ mode_label` with small weight
(e.g. 0.1). Breaks the symmetry that lets the 59 non-GT modes collapse without
loss penalty.

We save this for last because it changes the loss surface for *every* training
run; we want to know how far the cleaner geometric/representational fixes go
first.

### Phase 4 — Verify (~10 min after each retrain)

Track on the same val14 / closed-loop subset we've been using:

```
Open-loop:           lat, lon, top1, top5, L_sel, area mean, ADE / FDE
Closed-loop GPU CL:  CLS-NR, no-coll %, drivable %, comfort %, progress %
```

Plus re-run `scripts/diag_route_usage.py` on the same 5 samples, side-by-side
with Phase 0 output.

## Time budget summary

| Phase | Wall-clock | What changes |
|---|---|---|
| 0 — Diagnose | 1-2 h | new debug script, no model change |
| 1 — Per-step transform | ~3 h total | `model.py` only |
| 2 — Higher route res | ~10 h total | `config.py`, all caches |
| 3 — Per-mode L_gen | ~3 h total | `train_stage_b.py` `compute_il_loss` |

- Best case: Phase 1 alone fixes it → ~3 hours.
- Worst case: all four phases → ~16-18 hours (mostly cache re-extraction in
  Phase 2).

## Why this order

1. **Phase 0 first** — cheapest possible signal on whether the per-step
   transform is the bug. A 5-minute look at the diagnostic can save us from
   spending hours on fixes for the wrong cause.
2. **Phase 1 alone second** — no cache regen, fast retrain. If this works, we
   stop here.
3. **Phase 2 only if Phase 1 isn't enough** — cache re-extraction is the most
   expensive thing in this plan.
4. **Phase 3 saved for last** — touching the loss interacts with everything
   else; do it once, when the geometric/representational fixes have had their
   chance.

## Cross-reference: visualization-as-diagnostic

Pictures alone are not sufficient. They tell you what the geometry looks like,
but not whether the decoder token actually changes across lateral modes. Phase
0 deliberately pairs each picture with a numeric check so the failure mode is
identifiable.

Specifically:

| What pictures show | What numerics add |
|---|---|
| Whether routes look distinct across lat bins | Whether route feature norms differ post-PointNet |
| Whether trimmed segments change between step 0 and step 7 | Confirms the per-step transform hypothesis |
| Whether the 5 candidate trajectories spread laterally | Endpoint y-spread quantifies the collapse magnitude |
| GT trajectory + the 60 candidates | mode_query distance tells us if the collapse starts in the decoder embedding or in the action head |

## How to Fix It (plain language)

A non-technical recap of the same plan, for clarity:

**1. Update the lane picture every step (like we already do for the map and other cars).**
Every time the car moves, recalculate the lane dots so they're always seen from
the car's new position. This makes the lane picture sharp and current, just
like the car's own view. (Phase 1 above — the per-step route re-transform in
`model.py`.)

**2. Give more dots for the lane.**
Increase the number of dots per lane from 10 to 20. Then when we pick the part
ahead of the car, we'll keep about 5 dots instead of 2. That's a much clearer
picture of whether the lane curves left or right. (Phase 2 above — bumping
`N_ROUTE_POINTS` and re-extracting caches.)

**3. (Optional) Check if the model still collapses.**
If after the two fixes above the model still ignores the lanes, test whether
the model's internal code (the "query" that represents the lane) distinguishes
left from right.
- If the query *does* distinguish but the trajectory still collapses → the
  problem is in how the model turns that query into a path; that's the
  per-mode L_gen / action-head fix (Phase 3).
- If the query *does not* distinguish → the lane picture needs to be fed
  more strongly into the decoder (e.g. attend only to the lane we care about
  for each mode, not all 5 lanes at once).

## Phase 0 results (Apr 27 2026)

Ran `scripts/diag_route_usage.py` on 5 val14 samples (indices 0, 200, 500, 800,
1100) against `stage_b_best.pt`. Outputs in `diag_outputs/`.

**Aggregate numerics (5 samples):**

```
endpoint_y_lat_spread_gt_lon    mean = 0.115 m
endpoint_y_lat_spread_mean      mean = 0.340 m
```

Lat bins are 2 m wide (`LAT_BIN_EDGES`). Paper-level lat consistency (68 %)
would require ~2-4 m endpoint spread across the 5 lat candidates. We're seeing
0.1-0.3 m. **Severe collapse, confirmed.**

**Sample 500 (gt_lon=3, gt_lat=4) is illustrative:**

```
Cached routes — DISTINCT:
  lat 2 y range = [-0.2, 143]   (lane keep)
  lat 4 y range = [+3.6, 177]   (far-right, ~7 m apart)

Trimmed at step 0 — STILL DISTINCT:
  lat 2 y range = [-0.2, 3.2]
  lat 4 y range = [+3.6, 9.1]

Trimmed at step 7 (GT pose anchor at ~40 m forward) — OVERLAP:
  lat 2 y range = [52, 72]      ← features still in INITIAL ego frame,
  lat 4 y range = [46, 68]         so trimming gives overlapping windows

Endpoint y across the 5 lat-bin candidates (GT-lon row):
  lat 0  lat 1  lat 2  lat 3  lat 4
  33.38  33.38  33.44  33.38  33.62
  → spread = 0.096 m  (vs 7 m route spread)
```

**Two findings:**

1. **Per-step transform bug confirmed** (Phase 1 hypothesis). Step-0 trimmed
   routes are clearly separated; step-7 trimmed ranges overlap because
   features are still in initial ego frame after the anchor has moved.

2. **Action head also collapses** (Phase 3 territory). Even at step 0, when
   route inputs are distinct, the policy outputs nearly-identical trajectories
   (0.1 m endpoint spread vs 7 m route spread). The decoder is failing to
   convert distinct route inputs into distinct trajectory outputs.

### Implication for plan ordering

Phase 1 is necessary (step-7 overlap is a real bug) but **probably not
sufficient on its own**. The step-0 collapse — when route distinctness is
healthy — argues that **Phases 1 and 3 should ship together** rather than
Phase 1 alone followed by deciding whether Phase 3 is needed.

Updated ordering recommendation:
- Phase 1 (per-step transform) + Phase 3 (per-mode L_gen) in one combined
  retrain pass.
- Phase 2 (route resolution bump) only if 1+3 still leave lat below ~50 %.

## Reference: paper's intended route-conditioning path

(Per `paper/algorithms.md` and `model.py` review)

1. Build `N_lat` candidate routes from the map / lane graph.
2. **At each autoregressive step**, transform agent, map, *and route* poses
   into the current ego frame.
3. Trim each route from the closest point forward and keep `K_r = N_r / 4`
   points.
4. Encode those trimmed route points and feed them into the IVM / decoder as
   part of the K/V context.

The current implementation does steps 1, 3, and 4. **Step 2 is partial: only
the trim index lives in the current ego frame; the route features themselves
do not.** That gap is the Phase 1 fix.
