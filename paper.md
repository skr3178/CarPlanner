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