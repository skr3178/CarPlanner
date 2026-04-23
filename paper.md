## Explaination of the eq 2-4


The paper is answering: "How do we write down the probability of a whole driving rollout, and what does that tell us about how to build the planner?"
They do this in three steps (Eq 2, 3, 4), each one making the formula cleaner and more useful.
Setup: two key definitions
Before the equations, remember two things the paper defines:

Action = the ego's next position. So "a_t" (action at time t) is the same thing as "s^0 at time t+1" (ego's pose at the next step). The planner's job is to pick where ego goes next.
State = map + recent history of every agent. So "s_t" bundles together the map, and the last H seconds of positions for ego and all nearby vehicles/pedestrians.

Equation 2: Drop the actions
We start with the full sequence:
P(s_0, a_0, s_1, a_1, s_2, a_2, ..., s_T)
The insight: a_0 is just ego's position at time 1, which is already part of s_1. The action is redundant — it's already living inside the next state.
So we can just cross the actions out:
P(s_0, s_1, s_2, ..., s_T)
Simpler takeaway: Instead of tracking "state, action, state, action…", we only need to track the sequence of states. One thing instead of two.
Equation 3: Break it into step-by-step pieces
Now we use a standard probability trick (the chain rule): the probability of a long sequence = probability of the first thing × probability of each next thing given the previous one.
P(s_0, s_1, ..., s_T) = P(s_0) × P(s_1 | s_0) × P(s_2 | s_1) × ...
Then for each step, we split what changes into two parts:

What ego does next → call this the Policy
What other cars/pedestrians do next → call this the Transition Model

So each step becomes:
P(next state | current state) = Policy(ego's next move) × TransitionModel(others' next moves)
Simpler takeaway: A driving rollout is just: at each step, ego picks a move, and the world moves around ego. Repeat.
The problem with Equation 3
The policy is random — it's a probability distribution, like "60% go straight, 30% turn left, 10% turn right." If you sample from it fresh at every single step, you get schizophrenic behavior: turn left now, straight next, right after that. No coherent plan.
This is the "vanilla auto-regressive" problem shown in Figure 1(b). The car wiggles instead of committing.
Equation 4: Add a "mode" to keep ego consistent
The fix: commit to a high-level plan once, at the start, and stick with it.
They call this high-level plan the mode, written as c. A mode is basically "which route do I take + roughly how fast do I go." You pick c once at time 0 and never change it.
The math trick: introduce c into the formula and integrate over all possible modes (this is just a standard way of saying "average over all possible plans"):
P(s_0, ..., s_T) = P(s_0) × ∫ over all possible modes c [ P(c | s_0) × P(rollout | c) ]
Breaking down the pieces:

P(c | s_0) → given the current scene, how good is each possible plan? This becomes the Mode Selector.
P(rollout | c) → given we've committed to plan c, what does the rollout look like?

And when we expand that rollout piece the same way as before, we get:
Policy(ego's next move, GIVEN mode c) × TransitionModel(others' next moves)
The crucial change: the policy now depends on c. Since c is fixed across all time steps, ego's decisions stay coherent — "I'm taking the left route at moderate speed" shapes every step consistently.


## Paper excerpt on encoders (lines 29-55): Describes how CarPlanner's "vectorized" state s_0 is built:                                     
                                                                                               
┌────────────────────────────────────────────────┬───────────────────────────────────┬────────────────────────┐                             
│                     Source                     │               Shape               │        Encoder         │                             
├────────────────────────────────────────────────┼───────────────────────────────────┼────────────────────────┤                             
│ Map polylines (lanes: center+left+right)       │ N_{m,1} × N_p × 3·D_m where D_m=9 │ PointNet → N_{m,1} × D │
├────────────────────────────────────────────────┼───────────────────────────────────┼────────────────────────┤
│ Map polygons (intersections, crosswalks, etc.) │ N_{m,2} × N_p × D_m               │ PointNet → N_{m,2} × D │                             
├────────────────────────────────────────────────┼───────────────────────────────────┼────────────────────────┤                             
│ Agent history                                  │ N × H × D_a where D_a=10          │ PointNet → N × D       │                             
└────────────────────────────────────────────────┴───────────────────────────────────┴────────────────────────┘                             

All three feature tensors get concatenated → overall (N_m + N) × D feature set = s_0.                                                       

That's why config.py has D_AGENT=10, D_MAP_POINT=9, and D_POLYLINE_POINT=27 (= 3 × D_m, the concatenation of center + left + right per      
polyline point). The notes file is essentially a pasted copy of paper §2 that maps directly onto those config constants — looks like
in-progress reference material, not something being actively imported or run.                                                               

❯ why are these start 2 starting out as seperate? polylines and polygons? are they not from the same map?                                     
                                                                                     
● They are from the same map, but they represent fundamentally different types of geometry and different semantics, so the encoders treat them
separately:                                                                                                                                
                                                                                               
Polylines = 1D directional curves (lanes)                                                                                                   
- Lane centerlines, left/right boundaries                                                                                                   
- Have a direction of travel — the order of points matters                                                                                  
- Answer "where can I drive and in what direction?"       
- Come in a natural triplet: for each lane, you have center + left boundary + right boundary, which is why the code stacks them into a      
27-dim point (3 × 9)                                                                                                                        
                                                                                               
Polygons = 2D closed regions (intersections, crosswalks, stop lines)                                                                        
- Areas, not paths — points trace a closed loop                                                                                             
- No direction of travel; you just need to know "am I inside this region?"                                                                  
- Answer "what kind of zone am I in?"                                                                                                       
                                                                                               
Why separate encoders?
                                                                                               
1. Different shape: polylines pack as N_{m,1} × N_p × 3·D_m (=27), polygons pack as N_{m,2} × N_p × D_m (=9). You literally can't feed both 
through the same tensor op.                                                                                                                 
2. Different semantics per point: on a lane centerline, the heading at point k means "direction you'd drive at k." On a crosswalk polygon,  
that same "heading" field is the edge direction of the region — not comparable. A shared PointNet would blur the meaning.                   
3. Different learned invariances: lane features need to preserve order (upstream vs. downstream matters); polygon features should be roughly
permutation-invariant over their points (a square is a square regardless of where you start tracing). Separate PointNets let each head     
learn the right symmetry.
4. Different counts: N_{m,1} (lanes) and N_{m,2} (polygons) are different sizes per scene, so they'd need separate padding/masking anyway.  
                                                                                               
After encoding, both get projected to the same feature dim D and concatenated into one set of N_m × D map tokens — at that point the        
transformer/attention layers treat them uniformly. The separation is only at the geometric-feature-extraction stage, where it matters.      
                                                                                               
❯ why is the dimension as this? - Come in a natural triplet: for each lane, you have center + left boundary + right boundary, which is why    
the code stacks them into a                                                                                                                 
27-dim point (3 × 9)                                                                                                                      
                                                                                               
● The 27 comes from two independent expansions:                                                                                               
                                                                                               
The 9 (per-point features, D_m=9):                                                                                                          

Each single point on a lane carries 9 scalar features describing what's at that point:                                                      

┌─────┬────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────┐       
│  #  │        Feature         │                                                Why                                                 │
├─────┼────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 1   │ x                      │ position                                                                                           │
├─────┼────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 2   │ y                      │ position                                                                                           │       
├─────┼────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 3   │ sin(heading)           │ split heading into sin/cos so 0° and 360° are numerically close (avoids wrap-around discontinuity) │       
├─────┼────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤       
│ 4   │ cos(heading)           │ same                                                                                               │
├─────┼────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤       
│ 5   │ speed_limit            │ local speed rule                                                                                   │
├─────┼────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 6-9 │ 4-way category one-hot │ lane type (e.g., regular lane vs. merge vs. on-ramp vs. off-ramp)                                  │
└─────┴────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────┘       

So 2 (xy) + 2 (heading) + 1 (speed) + 4 (category one-hot) = 9. This matches D_MAP_POINT = 9 in config.py.                                  

The 3 (three parallel polylines per lane):                                                                                                  

A "lane" in the data isn't just a centerline — it's three co-sampled polylines that share the same point index:                             

point k of lane i:                                                                                                                          
center_k       ← 9 features
left_bound_k   ← 9 features                                                                                                              
right_bound_k  ← 9 features                                                                                                              
                                                                                               
The three are sampled at the same longitudinal position along the lane, so point k on the centerline corresponds geometrically to point k on
the left edge and point k on the right edge (roughly perpendicular-across).
                                                                                               
Stacking them gives 3 × 9 = 27 features per point index, which is D_POLYLINE_POINT = 27 in config.py.                                       

Why store all three as one "point" instead of as 3 separate polylines?                                                                      

- Preserves the cross-lane geometry: the PointNet sees the centerline sample together with the lane width on both sides at that location.   
Width + curvature of the boundaries tells the model things the centerline alone can't (e.g., "this lane narrows here" or "the shoulder is
wide").                                                                                                                                     
- Saves sequence length: 1 polyline of length N_p instead of 3 polylines of length N_p each — fewer tokens for attention later.
- Keeps the "lane" as a single object so masking/batching stays simple.                                                                     
                                                                                               
Polygons don't need this triplet structure (no left/right to speak of — they're closed regions), which is why they stay at D_m = 9 per point
rather than 3 × 9 = 27.                                                                                                                    
                                                                                               
※ recap: We improved eval_sanity.py to produce readable BEV plots with map context and a paper-benchmark comparison table, and regenerated    
eval_stage_b_sanity.png and eval_stage_c_sanity.png. Next action: waiting for your next request. (disable recaps in /config)
                                                                                               
❯ explain the shape for each type:  │ Map polylines (lanes: center+left+right)       │ N_{m,1} × N_p × 3·D_m where D_m=9 │ PointNet → N_{m,1} 
× D │                                                                                                                                      
├────────────────────────────────────────────────┼───────────────────────────────────┼────────────────────────┤                           
│ Map polygons (intersections, crosswalks, etc.) │ N_{m,2} × N_p × D_m               │ PointNet → N_{m,2} × D │                           
                                                                                               
├────────────────────────────────────────────────┼───────────────────────────────────┼────────────────────────┤                           
                                                                                               
│ Agent history                                  │ N × H × D_a where D_a=10          │ PointNet → N × D       │                           

● Here's what each axis means. I'll use concrete numbers from config.py so it's easier to picture.

---                                                                                                                                         
1. Lane polylines: N_{m,1} × N_p × 3·D_m
                                                                                               
Think of this as a 3D tensor, one lane per outer slice.
                                                                                               
outer axis = lane index       ──────────►  N_{m,1}  lanes  (e.g. 20)                                                                     
│                                                                                             
middle axis = point index along the lane   ▼  N_p points  (e.g. 20 samples per lane)                                                     
          │                                                                                 
inner axis = per-point feature vector                  ▼  3·D_m = 27 floats
                                                                                               
┌──────┬─────────┬─────────────────────────────────────────────────────┬────────────────────────┐
│ Axis │  Name   │                       Meaning                       │      Example size      │                                           
├──────┼─────────┼─────────────────────────────────────────────────────┼────────────────────────┤                                           
│ 0    │ N_{m,1} │ how many lane polylines are in the scene            │ 20 (N_LANES in config) │
├──────┼─────────┼─────────────────────────────────────────────────────┼────────────────────────┤                                           
│ 1    │ N_p     │ how many points were sampled along one lane         │ 20                     │                                           
├──────┼─────────┼─────────────────────────────────────────────────────┼────────────────────────┤                                           
│ 2    │ 3·D_m   │ features at that point: [center_9, left_9, right_9] │ 27                     │                                           
└──────┴─────────┴─────────────────────────────────────────────────────┴────────────────────────┘                                           

One row along axis 1 is one sample location along a lane, holding the centerline point + the matching left-edge point + the matching        
right-edge point all concatenated.
                                                                                               
PointNet collapses axis 1 (all the points along a lane) into a single fixed vector, producing N_{m,1} × D — one embedding per lane.         

---                                                                                                                                         
2. Map polygons: N_{m,2} × N_p × D_m
                                                                                               
Same idea, but no triplet in the inner axis.
                                                                                               
outer axis = polygon index     ──────────►  N_{m,2}  polygons  (e.g. 10)
│                                                                                             
middle axis = point index around polygon   ▼  N_p points
          │                                                                                 
inner axis = per-point feature vector                  ▼  D_m = 9 floats
                                                                                               
┌──────┬─────────┬──────────────────────────────────────────────────────────────────────┬─────────┐                                         
│ Axis │  Name   │                               Meaning                                │ Example │
├──────┼─────────┼──────────────────────────────────────────────────────────────────────┼─────────┤                                         
│ 0    │ N_{m,2} │ number of polygons (intersections, crosswalks, stop lines)           │ 10      │
├──────┼─────────┼──────────────────────────────────────────────────────────────────────┼─────────┤
│ 1    │ N_p     │ sampled points around the polygon's outline                          │ 20      │                                         
├──────┼─────────┼──────────────────────────────────────────────────────────────────────┼─────────┤
│ 2    │ D_m     │ features at that point: x, y, sin_h, cos_h, speed_limit, 4× category │ 9       │                                         
└──────┴─────────┴──────────────────────────────────────────────────────────────────────┴─────────┘                                         

Inner axis is only 9 (not 27) because polygons are closed regions — there's no "left bound vs right bound" pairing to co-sample.            

PointNet collapses axis 1, giving N_{m,2} × D — one embedding per polygon.                                                                  

---
3. Agent history: N × H × D_a
                                                                                               
outer axis = agent index          ──────────►  N  agents  (e.g. 20, inc. ego)
     │                                                                                         
middle axis = history time step                ▼  H time steps  (20 = 2s @ 10 Hz)
               │                                                                             
inner axis = pose at that time step                        ▼  D_a = 10 floats
                                                                                               
┌──────┬──────┬─────────────────────────────────────────────────────────────────────────────┬───────────────────────────────────────────┐
│ Axis │ Name │                                   Meaning                                   │                  Example                  │   
├──────┼──────┼─────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤
│ 0    │ N    │ number of tracked agents in the scene                                       │ 20 (N_AGENTS)                             │
├──────┼──────┼─────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤
│ 1    │ H    │ past time steps                                                             │ 20 (T_HIST — though config.py uses 10     │   
│      │      │                                                                             │ here)                                     │   
├──────┼──────┼─────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────┤   
│ 2    │ D_a  │ per-pose features: x, y, heading, velocity, bounding-box, time step,        │ 10                                        │   
│      │      │ category                                                                    │                                           │
└──────┴──────┴─────────────────────────────────────────────────────────────────────────────┴───────────────────────────────────────────┘   

One row along axis 1 is one snapshot in time for one agent — exactly the s^i_t pose vector from the notation in vectorized.md.              

PointNet collapses axis 1 (the time history of one agent) into a single vector, producing N × D — one embedding per agent summarizing its   
2-second history.
                                                                                               
---             
The common pattern
                                                                                               
All three tensors share the same recipe:
                                                                                               
[ entity × inner-structure × features ]   ──PointNet over inner──►   [ entity × D ]
                                                                                               
- For lanes, the "inner structure" is sampled points along the lane.                                                                        
- For polygons, the "inner structure" is sampled points around the boundary.                                                                
- For agents, the "inner structure" is a sequence of past time steps.                                                                       
                                                                                               
After encoding, you have three same-shaped tensors: (N_{m,1}, D), (N_{m,2}, D), (N, D). They get concatenated along the entity axis into one
big token set of shape (N_m + N) × D that downstream attention layers operate on uniformly.                                                
                                                                                               
❯ so we have to train an encoder/pointnet? or does it already exist?                                                                          
                         
Searched for 1 pattern, read 1 file (ctrl+o to expand)                                                                                      
                                                                                               
● You train it — it's part of the model, not something pre-existing.                                                                          
                                                                                               
"PointNet" is a neural network architecture (from Qi et al. 2017), not a specific pretrained model you download. It's defined right in      
model.py:                                                                                                                                   
                                                                                               
- PointNetEncoder (model.py:43) — used for agents                                                                                           
- LaneEncoder (model.py:85) — used for map polylines                                                                                        
                                                                                               
Each is a tiny MLP (3-layer: e.g. 9 → 32 → 64 → 256) applied independently to every point, followed by a max-pool across the set. That's the
whole "PointNet" recipe:
                                                                                               
for each point in the set:
feat_i = MLP(point_i)       # shared weights, per-point
global_feat = max_over_i(feat_i)  # symmetric aggregation                                                                                   
                                                                                               
The MLP weights are learned end-to-end with the rest of CarPlanner (selector, generator, IVM) via the Stage B IL loss — there's no separate 
pretraining step and no external PointNet checkpoint.                                                                                       
                                                                                               
Why not use a pretrained encoder?

- The input isn't generic 3D point clouds (like ShapeNet, where pretrained PointNets exist). Here each "point" is a domain-specific 9-dim or
10-dim feature vector with fields like speed_limit and category_one-hot. Pretrained weights for that don't exist.
- The encoder output dim (D = D_HIDDEN = 256 in config) and the downstream task are both task-specific, so random init + joint training is  
standard practice in this family of models (VectorNet, SceneTransformer, PlanTF, etc. all do the same).                                     

The parameter count is small. Each PointNet is ~20-30k params — it's essentially free to train compared to the attention backbone and       
autoregressive generator. All three encoders (agents, lanes, polygons) train together from the first gradient step.
                                                                                               
So when you ran train_stage_b.py earlier, those PointNets were being trained from scratch as part of the same loss — nothing pre-existing to
load.


All 3 cities extracted. Now merging.                                                                                                      
               
Extraction Summary:                                                                                                                       
               
┌────────────┬──────────┬───────────┬─────────┬─────────┬─────────┐                                                                       
│    City    │ Selected │ Extracted │ Dropped │  Size   │  Speed  │
├────────────┼──────────┼───────────┼─────────┼─────────┼─────────┤                                                                       
│ Boston     │ 36,737   │ 36,550    │ 1       │ 2.12 GB │ 157 s/s │
├────────────┼──────────┼───────────┼─────────┼─────────┼─────────┤
│ Pittsburgh │ 41,913   │ 41,045    │ 797     │ 2.38 GB │ 132 s/s │                                                                       
├────────────┼──────────┼───────────┼─────────┼─────────┼─────────┤                                                                       
│ Singapore  │ 38,695   │ 38,346    │ 240     │ 2.22 GB │ 163 s/s │                                                                       
├────────────┼──────────┼───────────┼─────────┼─────────┼─────────┤                                                                       
│ Total      │ 117,345  │ 115,941   │ 1,038   │ 6.72 GB │         │
└────────────┴──────────┴───────────┴─────────┴─────────┴─────────┘ 


Here's the issue. The loss is summed across 8 dims, not averaged, and computed in raw physical units     
(meters, m/s):                                                                                           
                                                                      
Loss ≈ |Δx| + |Δy| + |Δsin_h| + |Δcos_h| + |Δvx| + |Δvy| + |Δbox_w| + |Δbox_l|                           
                                                                      
A loss of 10 means ~1.25 average error per dimension. For positions in meters, that's ~1.25m position    
error predicting where 20 agents will be 0.8s into the future — actually reasonable. The loss won't go to
~1 unless the model gets sub-12cm accuracy on every dimension.

---

# Stage A — Transition Model Training Pipeline

**Goal:** predict next 8 future steps of 20 agents' states from a rich vectorized scene (agents + lanes + polygons).

## 1. Batch inputs (from `stage_cache_train_all_balanced.pt`)

```
agents_history   (B, 10, 20, 14)   ← 10 past frames × 20 agents × 14 feats
agents_now       (B,     20, 14)   ← "s_0" — current state (for residual base)
agents_mask      (B,     20)       ← 1 = valid agent
map_lanes        (B, 20, 10, 39)   ← 20 lanes × 10 pts × (3·13) polyline triplets
map_lanes_mask   (B,     20)
map_polygons     (B, 10, 10, 13)   ← 10 polygons × 10 pts × 13 feats
map_polygons_mask(B,     10)
agents_seq       (B,  8, 20, 14)   ← ground-truth future  ★ TARGET
```

## 2. Three PointNet encoders (shared per-point MLP + max-pool)

```
agents_history ─► PointNet_A ─► per_agent_feat  (B, 20, D=256)
│             Linear→ReLU→Linear over each (t, a) point,
│             max-pool across the 10 time steps

map_lanes      ─► PointNet_L ─► per_lane_feat   (B, 20, D=256)
│             Linear→ReLU→Linear over each (lane, pt) of 39-d,
│             max-pool across the 10 polyline points

map_polygons   ─► PointNet_P ─► per_poly_feat   (B, 10, D=256)
same pattern, max-pool across 10 points
```

## 3. Token concatenation → Transformer encoder

```
tokens = cat[ agents(20) | lanes(20) | polys(10) ]    (B, 50, 256)
mask   = cat[ agents_mask | lanes_mask | polys_mask ] (B, 50)

┌──────────────────────────┐
│  TransformerEncoder      │
│   3 layers × 8 heads     │   ← paper Table 5
│   d_model=256, FF=1024   │
│   dropout=0.1            │
└──────────────┬───────────┘
│
fused tokens (B, 50, 256)
│
┌──────────────┴───────────────┐
│                              │
per_agent_fused                  mean-pool → global
(B, 20, 256)                     (B, 256)
```

Self-attention fuses every agent with every lane / polygon / other agent — one global context pass.

## 4. Decoder (per-agent MLP regression head)

```
for each of 20 agents:
x = cat[ per_agent_fused , global ]        (B, 20, 512)
Δ = Linear(512→256) → ReLU → Linear(256 → 8·14)   (B, 20, 112)
Δ = reshape → (B, 20, 8, 14)                       ← normalized delta

pred_norm = s_0_norm + Δ                           ← RESIDUAL
pred      = pred_norm * AGENT_FEATURE_STD          ← denormalize
pred      = transpose to (B, 8, 20, 14)             ★ OUTPUT
```

One-shot prediction — all 8 future steps emitted in parallel, non-reactive.

## 5. Loss (train_stage_a.py:54-56)

```
diff = |pred[..., :8] - gt[..., :8]|.sum(-1)      ← L1 on motion dims only
(x, y, sinh, cosh, vx, vy, box_w, box_l)
loss = (diff * agents_mask).sum() / valid_count
```

Excludes box_h, time_step, 4 category one-hots — those are constants per agent.

## 6. Training loop

```
┌────────────┐   batch=256   ┌────────────┐
│ train_all_ │──────────────►│ T ransModel │
│ balanced   │               │  3.45 M p  │
│ 115,941 s  │◄── masked L1 ─│            │
└────────────┘   loss        └────────────┘
│
AdamW  lr=1e-4  wd=1e-4
ReduceLROnPlateau  patience=5  factor=0.3
50 epochs   → save best on val loss
│
checkpoints/stage_a_best.pt
```

## Quick shape trace (single sample)

```
s_0 (20, 14)  +  Δ (20, 8, 14)          → pred_norm (20, 8, 14)
└── from fused scene ──┘
───────────────────────────────────────►  pred (8, 20, 14)  vs  gt (8, 20, 14)
          L1 on dims 0-7
```

The frozen weights from this stage become the reactive world-model used by Stage B's IL training and Stage C's PPO rollouts.   

HF Key takeaways:                                   
- Vegas raw diversity: 50 scenario types, ~5.6M samples available                        
- We kept 60,113 (1.08%) — a surgical sample, not a bulk download                        
- All 21 rare types fully preserved (≤ budget, 100% kept — starting_left_turn,           
high_magnitude_jerk, stationary_at_crosswalk, etc.)                                      
- 5 previously-missing types filled to ~4000: on_pickup_dropoff,                         
near_pedestrian_at_pickup_dropoff, near_construction_zone_sign,                          
traversing_traffic_light_intersection, near_multiple_pedestrians — these were absent from
Boston/Pgh/Singapore, so Vegas provides the only coverage                               
- 24 common types topped up (~1,345 each) — just enough so the 4-city merge hits the     
paper's 176,218 target                                                              
- NaN filter cost: 99.7% survival — the "99.58%"/"99.6%"/"99.09%" rows are rare types    
that lost 1 sample each to NaN                                                       


## pape results

Here's the comparison — our results vs the paper's reported numbers:

Paper's Results (Table 1 — Test14-Random, 261 scenarios)

┌──────┬────────────────────┬────────┬───────┐
│ Type │      Planner       │ CLS-NR │ CLS-R │
├──────┼────────────────────┼────────┼───────┤
│ Rule │ PDM-Closed         │ 90.05  │ 91.64 │
├──────┼────────────────────┼────────┼───────┤
│ IL   │ PLUTO              │ 91.92  │ 90.03 │
├──────┼────────────────────┼────────┼───────┤
│ RL   │ CarPlanner (Paper) │ 94.07  │ 91.1  │
└──────┴────────────────────┴────────┴───────┘

Our Results (Test14-Random, 258 scenarios)

┌────────────────────────┬────────┬──────────────┬──────────┬──────────┬─────────┐
│          Run           │ CLS-NR │ No-collision │ Drivable │ Progress │ Comfort │
├────────────────────────┼────────┼──────────────┼──────────┼──────────┼─────────┤
│ eval_closedloop_stagec │ 48.07  │ 94.73%       │ 0.00%    │ 3.54%    │ 0.00%   │
├────────────────────────┼────────┼──────────────┼──────────┼──────────┼─────────┤
│ eval_test14_v3         │ 54.62  │ 96.87%       │ 22.30%   │ 8.60%    │ 0.00%   │
├────────────────────────┼────────┼──────────────┼──────────┼──────────┼─────────┤
│ eval_test14_v4         │ 54.59  │ 96.72%       │ 22.83%   │ 8.32%    │ 0.00%   │
└────────────────────────┴────────┴──────────────┴──────────┴──────────┴─────────┘

The gap and why

Our CLS-NR ~54 vs paper's 94 — a huge gap. The reasons:

1. Drivable compliance is near 0% — this is the killer. The car leaves the drivable area almost immediately. The paper gets
~99% (Table 3). This suggests our route selection / lateral mode encoding isn't producing trajectories that stay on-road.
The heuristic lane selection vs paper's graph search matters here.
2. Progress ~8% vs paper's ~95% — the car barely moves forward. Likely a consequence of leaving drivable area (score zeroed)
or the planner generating near-stationary trajectories.
3. Comfort 0% — trajectory jerk/acceleration violations, likely from poor trajectory quality.
4. No-collision ~97% — actually comparable to the paper (~99%). The car avoids hitting things — it just doesn't go anywhere
useful.

Root causes
- Trained on Boston-only (previous run) with a weaker Stage A — the current 3-city/4-city runs haven't been evaluated yet
- 2 IVM layers instead of 3 — less representational capacity for the autoregressive policy
- Heuristic route selection instead of proper lane graph search — bad lateral modes = bad trajectory anchors
- No drivable area reward during RL — the paper's reward explicitly penalizes leaving the drivable area


● The polylines and polygons help the model's scene understanding in the Transformer attention (the model sees lanes,
crosswalks, intersections as K/V context), but they don't directly fix the route selection problem. These are two separate
things:

1. Map encoding (what we have): Polylines + polygons encoded via PointNet → fed as K/V tokens to the mode selector and
policy Transformers. This helps the model understand the road layout. Yes, this is good and matches the paper.
2. Route selection for lateral modes (the gap): The paper says "we identify N_lat possible routes from the map using a graph
search algorithm." This means traversing the lane connectivity graph to find actual drivable paths forward. Let me check
what our implementation does instead.

Read 1 file (ctrl+o to expand)

● Here's the difference:

Our approach (lines 300-333): Pick individual lane segments that are heading-aligned with ego and spread across
left/center/right by lateral offset. These are disconnected lane fragments, not connected paths.

Paper's approach: Graph search on lane connectivity → finds actual multi-lane drivable paths (e.g., "go straight through
this intersection then turn left"). These are coherent route plans.

The impact:
- Our lateral modes represent "which nearby lane segment am I closest to" — local
- The paper's lateral modes represent "which drivable route do I follow for the next 8 steps" — global

This is why drivable compliance is 0% in our eval — the lateral modes don't encode actual routes, so the planner can't
commit to a coherent path and drives off-road.

The polyline/polygon encoding helps within each mode (the policy sees nearby lanes and intersections), but the mode itself
(which route to follow) is poorly defined because we're just picking nearby lane fragments instead of searching for
connected routes.

To fix this properly, we'd need to use nuPlan's lane graph connectivity (which lanes connect to which) to build actual route
candidates via BFS/DFS from the ego's current lane. This is the biggest architectural gap to close.


The route selection only affects Stage B and Stage C — not data prep or Stage A.

Here's why:

┌──────────────────────────┬───────────┬───────────────────────────────────────────────────────────────────────────────┐
│          Stage           │   Uses    │                                      Why                                      │
│                          │  routes?  │                                                                               │
├──────────────────────────┼───────────┼───────────────────────────────────────────────────────────────────────────────┤
│ Data prep / extraction   │ No        │ Extracts raw lanes, polygons, agents. Routes are computed at model forward    │
│                          │           │ time, not cached.                                                             │
├──────────────────────────┼───────────┼───────────────────────────────────────────────────────────────────────────────┤
│ Stage A                  │ No        │ Predicts other agents' futures. Doesn't use modes or routes at all — just     │
│ (TransitionModel)        │           │ agents + map context via self-attention.                                      │
├──────────────────────────┼───────────┼───────────────────────────────────────────────────────────────────────────────┤
│ Stage B (IL:             │ Yes       │ ModeSelector.forward() calls select_candidate_routes() to build lateral       │
│ ModeSelector + Policy)   │           │ modes. AutoregressivePolicy uses mode-conditioned IVM with route features.    │
├──────────────────────────┼───────────┼───────────────────────────────────────────────────────────────────────────────┤
│ Stage C (RL: PPO)        │ Yes       │ Same model, same forward pass — routes flow through the same ModeSelector +   │
│                          │           │ Policy path.                                                                  │
└──────────────────────────┴───────────┴───────────────────────────────────────────────────────────────────────────────┘

The fix is in model.py only — specifically replacing select_candidate_routes() with a proper lane graph search. The function
takes map_lanes tensors and returns (B, N_lat, N_PTS, 9) route polylines. The interface stays the same, just the selection
logic changes.

However, there's a catch: the current map_lanes tensor is just the 20 nearest lane segments with no connectivity info. A
proper graph search needs to know which lanes connect to which. That means we'd either need to:

1. Query the lane graph at extraction time and cache the route candidates alongside the existing data (changes
extract_stage_a.py + cache format), or
2. Store lane connectivity in the cache (adjacency list) and do graph search in the model forward pass

Option 1 is cleaner — precompute the N_lat=5 best routes per scenario once and cache them. No model.py changes needed beyond
using the cached routes instead of computing them on the fly.


## issues

                                                       
The paper says "graph search" without pinning down:                                                                                       
                                                       
┌────────────────────┬────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────┐    
│      Decision      │                     My choice                      │                           Note                           │ 
├────────────────────┼────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤    
│ Starting lane set  │ ego_lane.parent.interior_edges (parallel lanes in  │ Alternative: only ego's single lane + its successors;    │ 
│                    │ current roadblock)                                 │ paper ambiguous                                          │ 
├────────────────────┼────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤    
│ Branching at       │ pick most heading-aligned successor, not enumerate │ Alternative: DFS over all branches, subsample to N_lat;  │ 
│ junctions          │  all branches                                      │ would yield more diverse turning routes                  │    
├────────────────────┼────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤ 
│ Max depth / length │ 6 lanes OR 150 m                                   │ Paper doesn't specify; 150 m covers ~10 s at 15 m/s      │    
├────────────────────┼────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤    
│ Feature dim per    │ D_POLYLINE_POINT = 39 (center + left + right × 13) │ Paper says D_m but this repo has historically used 3×D_m │ 
│ route point        │                                                    │  everywhere; kept consistent                             │    
└────────────────────┴────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────┘    

These are reasonable readings but not uniquely implied by the paper. If you wanted to tune for tighter paper-faithfulness you could flip  
the branching strategy.                                   
                                                                                                                                       
What this does NOT solve                                  

1. The checkpoint on disk is still old-weights — trained against the broken routes. Mode-selector probabilities will stay poor until you  
re-extract caches and retrain Stage B. Architecture correctness ≠ trained performance.
2. Drivable-compliance 0% → ? — my original review flagged that bad routes probably don't fully explain 0% drivable compliance. Other     
possible causes (route points ignored in policy's coordinate reference frame, route encoding not reaching the action head, etc.) are not  
addressed by this fix. You'd want to verify drivable-compliance jumps significantly after retraining before declaring that fully resolved.
3. Other paper sections — 3.3.1 (transition model), 3.3.3 (rule-augmented selector), 3.4 (mode-assignment c*), the IVM coordinate         
transforms, etc. — those weren't touched by this fix. The mode-assignment (Sec 3.4) was partially affected: positive_lat_idx is now       
computed by route-proximity matching inside _extract_routes, which aligns with paper Section 3.4 step 2-3.
                                                                                                                                       
Short answer                                                                                                                              

Architecturally, the mode-selector side of Section 3.3.2 is now paper-faithful. The lateral modes are connected, graph-searched routes,   
passed through the same PointNet + decomposition the code already had. What's left is operational — re-extract caches, retrain, and verify
metrics actually improve.


## Evaluation Pipeline

Two completely separate evaluation pipelines exist.

### Pipeline 1: Open-loop evaluation (fast, minutes)

Single forward pass per sample — no simulation loop, no replanning. Uses pre-extracted
.pt caches with GT trajectories and mode labels already baked in.

**What it measures:** L_gen (generator loss), L_sel (mode selector CE loss), ADE, FDE, mode accuracy.
These correspond to the open-loop columns in Table 4.

**Which dataset:** Val14 only (1,116 samples in `stage_cache_val14.pt`).
Val14 is the standard validation split (1,118 scenarios, 100 per type × 14 types, from PDM).

**Scripts:**
```bash
# Open-loop eval on Val14 — matches Table 4 open-loop columns (L_gen, L_sel)
python eval_sanity.py --checkpoint checkpoints/stage_b_best.pt \
    --cache checkpoints/stage_cache_val14.pt --n_eval 1116

# Quick sanity on mini split
python evaluate.py --checkpoint checkpoints/stage_b_best.pt --cache --split mini
```

**Run this FIRST.** If L_gen is far from the paper's 174.3 (IL-best), there's no point
running the expensive closed-loop simulation.

### Pipeline 2: Official closed-loop evaluation (slow, hours)

nuPlan's full simulation framework. At each 0.1s step, the simulator calls
`CarPlannerPlanner.compute_planner_trajectory()`, the LQR controller tracks the output,
and the official metric engine scores the full 15s rollout.

**What it measures:** CLS-NR, S-CR (collision rate), S-Area (drivable area compliance),
S-PR (progress), S-Comfort, S-TTC (time to collision), speed limit compliance,
driving direction compliance. These are the closed-loop columns in Tables 1, 2, and 4.

**Which datasets (per paper Section 4.1 "Benchmarks and metrics"):**
- Test14-Random: 261 scenarios (20 per type × 14 types, from PlanTF)
- Reduced-Val14: 318 scenarios (from PDM)

**This reads raw DB files at runtime** — it does NOT use our pre-extracted .pt caches.
The simulator needs to step through time, update agent positions (log-replay or IDM),
and let the planner react to the evolving scene. Pre-extraction can't capture this
because the ego trajectory changes every run depending on the planner's decisions.

**Script:**
```bash
PYTHONPATH=. python paper/dataset/nuplan-devkit/nuplan/planning/script/run_simulation.py \
    planner=carplanner \
    +simulation=closed_loop_nonreactive_agents \
    scenario_filter=test14_random \
    scenario_builder.db_files=/home/skr/nuplan_cities/val/data/cache/val \
    scenario_builder.map_root=paper/dataset/nuplan-extracted/nuplan-maps-v1.0
```

**Planner wrapper:** `carplanner_planner.py` (project root) — subclass of AbstractPlanner.
**Hydra config:** `tuplan_garage/.../planner/carplanner.yaml`.

### What about the pre-extracted scenario caches?

| Cache file                        | Format              | Used by                    | Purpose                          |
|-----------------------------------|---------------------|----------------------------|----------------------------------|
| `stage_cache_train_*.pt`          | Training tensors    | train_stage_a/b/c.py       | Training data (essential)        |
| `stage_cache_val14.pt`            | Training tensors    | eval_sanity.py, evaluate.py| **Open-loop eval** (has GT traj + mode labels) |
| `stage_cache_mini.pt`             | Training tensors    | eval_sanity.py, evaluate.py| Quick open-loop sanity check     |
| `test14_random_cache.pt` (258 sc) | Scenario tensors    | eval_closedloop_gpu.py     | Fast approximate closed-loop (dev only, NOT paper metrics) |
| `val14_cache.pt` (1118 sc)        | Scenario tensors    | eval_closedloop_gpu.py     | Fast approximate closed-loop (dev only) |
| `reduced_val14_cache.pt` (318 sc) | Scenario tensors    | eval_closedloop_gpu.py     | Fast approximate closed-loop (dev only) |
| `test14_hard_cache.pt`            | Scenario tensors    | eval_closedloop_gpu.py     | **EMPTY — extraction failed**, skipping for now |

The scenario caches (test14_random_cache.pt, etc.) were built for `eval_closedloop_gpu.py` —
a fast but approximate GPU-native eval. They are useful for quick development iteration
but do NOT produce official paper metrics. Official metrics come only from Pipeline 2.

### Evaluation order

1. **Open-loop on Val14** (Pipeline 1) — minutes. Validates model produces reasonable
   trajectories and mode predictions before investing in closed-loop.
2. **Closed-loop via nuPlan simulator** (Pipeline 2) — hours. Produces the official
   CLS-NR, S-CR, S-Area, S-PR, S-Comfort that match paper Tables 1/2/4.

### Table 4 ablation config (IL-best vs RL-best)

| Ablation flag        | Table 4 IL-best | Table 4 RL-best | Current config.py |
|----------------------|-----------------|-----------------|-------------------|
| MODE_DROPOUT         | True            | True            | True              |
| SELECTOR_SIDE_TASK   | True            | True            | True              |
| EGO_HISTORY_DROPOUT  | True            | False           | **False** (RL)    |
| BACKBONE_SHARING     | True            | False           | **False** (RL)    |

Current config set to IL-best (both True) for Stage B training.
Switching doesn't break anything — tested: same state dict keys, forward pass OK,
param count drops by ~42K due to shared backbone.

## Stage B Evaluation Pipeline

### Open-loop eval (fast, ~2 min)

Computes metrics from paper Tables 4 (OL columns), 8, and 10 (OL columns):

```bash
python eval_stage_b.py --checkpoint checkpoints/stage_b_best.pt
```

| Metric | Table | Source |
|--------|-------|--------|
| L_gen (generator loss) | Table 4 OL | sum \|pred - gt\| over dims & time |
| L_sel (selector CE) | Table 4 OL | -log P(gt_mode) |
| ADE, FDE | — | displacement errors |
| Mode accuracy (top-1, top-5) | — | mode selector accuracy |
| Consistent Ratio Lat/Lon | Table 8 | fraction of trajectories in assigned mode bin |
| Col mean/min/max | Table 10 OL | proximity to agents across all candidates |
| Area mean/min/max | Table 10 OL | off-drivable-area fraction across all candidates |

For BEV visualization, use `eval_sanity.py` instead:
```bash
python eval_sanity.py --checkpoint checkpoints/stage_b_best.pt \
    --cache checkpoints/stage_cache_val14.pt --n_eval 1116
```

### Closed-loop eval (slow, hours)

Computes metrics from Tables 1, 2, 4 (CL columns), 7, 9, 11 via nuPlan simulator:

```bash
# Test14-Random (261 scenarios, paper Table 1)
python scripts/eval_nuplan.py --checkpoint checkpoints/stage_b_best.pt --split test14-random

# Reduced-Val14 (318 scenarios, paper Table 2)
python scripts/eval_nuplan.py --checkpoint checkpoints/stage_b_best.pt --split val14
```

Produces: CLS-NR, CLS-R, S-CR, S-Area, S-PR, S-Comfort.

### Eval order

1. Open-loop first — if L_gen is far from 174.3 (IL-best), don't bother with closed-loop
2. Closed-loop on test14-random — main benchmark (Table 1)
3. Closed-loop on reduced-val14 — secondary benchmark (Table 2)

---

## Current Run Results (Apr 23 2026)

### Stage B — Open-loop eval on val14 (1116 samples)

Checkpoint: `stage_b_best.pt` (epoch 11, 3-city train_all_balanced, route polylines, IL-best config)

**Table 4 — Open-loop metrics**

| Metric | Ours | Paper IL-best | Paper RL-best |
|--------|------|---------------|---------------|
| L_gen | 12.6 | 174.3 | 1624.5 |
| L_sel (CE) | 2.31 | 1.04 | 1.03 |
| ADE (m) | 1.499 | — | — |
| FDE (m) | 2.664 | — | — |
| Mode acc top-1 (%) | 9.32 | — | — |
| Mode acc top-5 (%) | 72.04 | — | — |
| Median GT rank | 3 / 60 | — | — |

**Table 8 — Consistent Ratio**

| Metric | Ours | Paper IL-best | Paper RL-best |
|--------|------|---------------|---------------|
| Consistent Ratio Lat (%) | 20.00 | 68.26 | 79.58 |
| Consistent Ratio Lon (%) | 8.38 | 43.01 | 43.03 |

**Table 10 — Open-loop Collision & Area**

| Metric | Ours | Paper IL-best | Paper RL-best |
|--------|------|---------------|---------------|
| Col Mean | 0.00 | 0.15 | 0.12 |
| Area Mean | 0.10 | 0.09 | 0.05 |

**Notes:**
- L_gen (12.6) is much lower than paper IL-best (174.3) — different scale/units, lower is better
- L_sel (2.31) worse than paper (1.04) — mode selector not yet accurately predicting GT mode
- Consistent Ratio far below paper — model not committing to coherent lateral/longitudinal bins
- Zero collisions is good; area off-road fraction similar to paper

### Stage C — Open-loop eval on val14 (1116 samples)

Checkpoint: `stage_c_best.pt` (epoch 8, RL-best config: EGO_HISTORY_DROPOUT=False, BACKBONE_SHARING=False)

**Table 4 — Open-loop metrics**

| Metric | Stage C (ours) | Stage B (ours) | Paper RL-best | Paper IL-best |
|--------|---------------|---------------|---------------|---------------|
| L_gen | 14.4 | 12.6 | 1624.5 | 174.3 |
| L_sel (CE) | 2.47 | 2.31 | 1.03 | 1.04 |
| ADE (m) | **1.210** | 1.499 | — | — |
| FDE (m) | **2.364** | 2.664 | — | — |
| Mode acc top-1 (%) | **23.75** | 9.32 | — | — |
| Mode acc top-5 (%) | 71.42 | 72.04 | — | — |
| Median GT rank | 3 / 60 | 3 / 60 | — | — |

**Table 8 — Consistent Ratio**

| Metric | Stage C (ours) | Paper RL-best | Paper IL-best |
|--------|---------------|---------------|---------------|
| Consistent Ratio Lat (%) | 20.04 | 79.58 | 68.26 |
| Consistent Ratio Lon (%) | 8.33 | 43.03 | 43.01 |

**Table 10 — Open-loop Collision & Area**

| Metric | Stage C (ours) | Paper RL-best |
|--------|---------------|---------------|
| Col Mean | 0.00 | 0.12 |
| Area Mean | 0.10 | 0.05 |

**RL Diagnostics**

| Metric | Value |
|--------|-------|
| Policy entropy | 1.662 |
| Value estimate (mean) | -2.007 |
| Reward total (mean) | -1.727 |
| R_displacement | -0.932 |
| R_collision | -0.008 |
| R_drivable | -0.133 |
| R_comfort | -1.045 |
| Action std (x, y, yaw) | 0.305, 0.306, 0.800 |

**Notes:**
- RL improved ADE by 19% (1.499 → 1.210) and top-1 mode accuracy by +14pp (9.3% → 23.8%)
- Consistent Ratio unchanged from stage B — RL did not improve mode coherence
- Negative reward on val set expected (trained on train split, val is out-of-distribution)
- R_comfort is the largest penalty (-1.045) — trajectory jerk is high
- Consistent Ratio gap vs paper (20% vs 79%) is the primary signal that route/mode encoding needs improvement