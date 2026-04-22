# Polygon Map Encoding Integration Plan

## Context

CarPlanner's paper specifies two map encoding sets: polylines (lanes) and polygons (crosswalks, intersections, stop lines). Lane polylines are already implemented and trained. Polygon encoding is NOT — it's the missing piece before the map encoding matches the paper. The goal is to add polygon extraction, encoding, and integration across the full pipeline, then test on the mini split.

Paper spec: `Nm,2 × Np × Dm = 10 × 10 × 9` → PolygonEncoder (PointNet) → `10 × 256`, concatenated with lane features as additional K/V tokens in attention.

Config already has: `N_POLYGONS=10`, `D_POLYGON_POINT=9`, `D_LANE=256`.

---

## Step 1: `data_loader.py` — Polygon extraction from nuPlan maps

**1a. Add `_resample_polygon_ring()` helper** (after `_resample_polyline`, ~line 117)
- Port from `test_polygon_option2_real.py:29-58`
- Takes raw Nx2 coords, closes the ring, resamples to `N_LANE_POINTS` via arc-length with `endpoint=False`

**1b. Add `_load_map_polygons()` function** (after `_load_map_lanes`, ~line 233)
- Query 3 layers: `SemanticMapLayer.CROSSWALK` (cat=0), `STOP_LINE` (cat=1), `INTERSECTION` (cat=2)
- For each object: `obj.polygon.exterior.coords` → resample ring → `_encode_polyline_pts()` with speed_limit=0
- Sort all collected polygons by centroid distance to ego, take closest `N_POLYGONS`
- Returns: `(N_POLYGONS, N_LANE_POINTS, 9)` + `(N_POLYGONS,)` mask

**1c. Update `_load_sample()`** (~line 538)
- Call `_load_map_polygons()` alongside `_load_map_lanes()`, with try/except fallback to zeros
- Add `map_polygons` and `map_polygons_mask` to return dict

**1d. Update `__getitem__()` fallback** (~line 630)
- Add zero-tensor fallbacks for both new keys

**1e. Update `PreextractedDataset`** (~line 675)
- Use `data.get('map_polygons', None)` with zero-tensor fallback for backward compat with old caches
- Return in `__getitem__()`

---

## Step 2: `model.py` — PolygonEncoder + wiring

**2a. Add `PolygonEncoder` class** (after `LaneEncoder`, ~line 125)
- MLP: `9 → 64 → 128 → 256`, max-pool over points, mask zeroing
- Same structure as `LaneEncoder` but separate weights per paper's "another PointNet"

**2b. Update `IVMBlock.forward()`** (~line 164)
- Add `polygon_feats=None, polygon_poses=None` params
- After map_feats K-NN + append, add parallel block for polygon_feats (K-NN filter + concat to kv)

**2c. Update `ModeSelector`** (~line 389)
- `__init__`: add `self.polygon_encoder = PolygonEncoder()`
- `forward()`: add `map_polygons=None, map_polygons_mask=None` params; encode and append to `kv_parts`

**2d. Update `AutoregressivePolicy`** (~line 528)
- `__init__`: add `self.polygon_encoder = PolygonEncoder()`
- `forward()`: add polygon params; per-step ego-frame transform (simpler than lanes — single 9-dim block, not 3×9); encode via `polygon_encoder`; pass `polygon_feats` + `polygon_poses` to IVM layers
- `forward_rl()`: same changes

**2e. Update `TransitionModel`** (~line 1024)
- `__init__`: add `self.polygon_encoder = PolygonEncoder()`
- `forward()`: add polygon params; encode polygons; concat with `[per_agent, map_feats, poly_feats]`; update padding mask

**2f. Update `CarPlanner` methods**
- `forward_train()`, `forward_rl_train()`, `forward_inference()`, `forward_transition()`: add polygon params, pass through to submodules

---

## Step 3: `extract_stage_a.py` — Cache extraction

- Add `'map_polygons'` and `'map_polygons_mask'` to `keys` list (~line 63)
- Add shape print in sanity checks
- No loop changes needed — generic key-based buffering picks up new keys automatically

---

## Step 4: `merge_caches.py` — Cache merging

- Add polygon keys to `keys` list (~line 41)

---

## Step 5: `train_stage_a.py` — Stage A training

- Add polygon keys to `gpu_cache` dict with `.get()` fallback
- Pass `map_polygons`, `map_polygons_mask` to `model()` in both pinned and non-pinned paths
- Update validation loop similarly

---

## Step 6: `train_stage_b.py` — Stage B training

- Add polygon keys to `gpu_cache` dict
- Update `get_batch()` to return polygon tensors
- Pass to `model.forward_train()`

---

## Step 7: `train_stage_c.py` — Stage C training

- Extract polygon tensors from batch
- Pass to `model.forward_rl_train()`

---

## Step 8 (defer): Evaluation & planner scripts

- `evaluate.py`, `carplanner_feature_builder.py`, `carplanner_planner.py`, `scripts/eval_closedloop_gpu.py`
- Can be done after training pipeline is verified

---

## Backward Compatibility

All new params are `Optional = None`. Key safety mechanisms:
- `PreextractedDataset`: `data.get()` with zero fallback → old caches still load
- All model modules: `if map_polygons is None` → skip encoding, zero-length tensor
- Old checkpoints: load with `strict=False` → new polygon_encoder weights init randomly

---

## Verification

1. **Extract mini cache**: `python extract_stage_a.py --split mini` — verify `map_polygons: (N, 10, 10, 9)` in output
2. **Model smoke test**: `python -c "from model import CarPlanner; ..."` — forward + backward with polygon inputs
3. **Stage B mini train**: `python train_stage_b.py --split mini --epochs 2 --batch_size 8` — verify loss decreases
4. **Backward compat**: Load old cache through `PreextractedDataset` — verify zero fallbacks work

---

## Key files

| File | Change scope |
|---|---|
| `data_loader.py` | Add `_resample_polygon_ring()`, `_load_map_polygons()`, update sample/fallback/cache |
| `model.py` | Add `PolygonEncoder`, update 6 classes (IVMBlock, ModeSelector, AutoregressivePolicy, TransitionModel, CarPlanner, RuleSelector) |
| `extract_stage_a.py` | Add 2 keys to list |
| `merge_caches.py` | Add 2 keys to list |
| `train_stage_a.py` | Pass polygon tensors through cache → model |
| `train_stage_b.py` | Pass polygon tensors through cache → model |
| `train_stage_c.py` | Pass polygon tensors through cache → model |

## Existing code to reuse

- `_resample_polygon_ring()` from `test_polygon_option2_real.py:29-58`
- `_encode_polyline_pts()` already in `data_loader.py:120-145` — works for polygon points too
- `PolygonEncoder` class from `test_polygon_encoder.py:27-67` (update MLP widths to match new D_LANE=256)
- nuPlan map API: `map_api.get_proximal_map_objects(point, radius, [SemanticMapLayer.CROSSWALK])` — confirmed working
