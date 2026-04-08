### One meta-rule

Always test on a **fixed batch of 8-16 scenarios** that you save to disk before any training. Every module test runs on this same fixed batch. This means any regression — a change that breaks a previously passing test — is immediately visible because the baseline numbers are stable. If you test on random batches each time, a test that passed yesterday might fail today just due to sampling variance, not a real bug.

---

### The general principle

Every submodule test has three layers, applied in order:

**Shape test** — does the output tensor have exactly the right dimensions? This catches wiring errors immediately without needing to understand the output's meaning.

**Behaviour test** — does the output have the right mathematical properties, independent of training? These are properties that must hold true even at random initialisation — things like permutation invariance, symmetry, or monotonicity. If these fail, the architecture itself is wrong.

**Loss test** — does the loss decrease on a small fixed batch? This catches gradient flow issues — dead ReLUs, detached tensors, wrong loss reduction. You fix the batch, run 100-200 steps, and verify the loss goes down. If it does not, the learning signal is broken somewhere.

---

### Module 0 — Agent Encoder (PointNet)

**Shape test.** Feed a batch of agent tensors with shape (B, N, H, Da=10). Verify output is per-agent features (B, N, D) and a global feature (B, D). Check D matches your config.

**Behaviour tests.**
- Permutation invariance: shuffle the agent order (dim=1) and verify the global feature is unchanged. PointNet must be order-agnostic — if it is not, you have accidentally used an architecture that depends on agent ordering.
- Masked agents: set some agent slots to zero and mark them as padding. Verify those agents do not influence the global feature. This is critical because nuPlan scenes have variable numbers of agents padded to a fixed N.
- Two identical agents at different positions should produce different per-agent features but the global feature should reflect both.

**Loss test.** Supervise the global feature to predict a simple scalar property of the scene — for example, the number of valid agents. Loss should decrease within 100 steps on a fixed batch of 8 scenarios.

---

### Module 1 — Map Encoder (two PointNets)

**Shape test.** Polyline input (B, N_m1, Np, 3×Dm) → output (B, N_m1, D). Polygon input (B, N_m2, Np, Dm) → output (B, N_m2, D). Concatenated map features (B, N_m, D) where N_m = N_m1 + N_m2.

**Behaviour tests.**
- Permutation invariance over the Np point axis within one polyline — shuffling the points of a single lane should not change its feature vector.
- Two geometrically identical polylines should produce identical feature vectors.
- A zero-padded polyline (all points zero) should produce a consistent output — not NaN or inf. NaN here usually means a division by zero in the pooling operation.
- The polygon encoder and polyline encoder should produce features in the same range — if one outputs values in [-1, 1] and the other in [-100, 100], concatenation will be dominated by one side.

**Loss test.** Supervise map features to predict presence of an intersection polygon. Loss should decrease on a fixed batch.

---

### Module 2 — Transition Model β

**Shape test.** Input is s0 (agents + map). Output is (B, T, N, Da) — future poses of agents 1 to N. Verify ego agent (index 0) is not in the output.

**Behaviour tests.**
- Static scene test: if all agents have zero velocity in s0, predicted future positions should stay near their t=0 positions. They will not be perfect at random init, but after brief training on this synthetic case they should converge.
- Non-reactivity test: modify the ego's current position in s0 while keeping all other agents identical — the predicted agent futures should be unchanged. β must be blind to ego.
- Plausibility test: predicted agent speeds should be in a physically reasonable range. An agent moving at 200 m/s per step is a sign of unnormalised outputs.

**Loss test.** On a fixed batch of 10 scenarios, L_tm should decrease from random initialisation. Watch for the degenerate solution where the model predicts all agents stay stationary — this minimises L_tm for parked cars but fails for moving agents. Check that predicted velocities are nonzero when GT velocities are nonzero.

---

### Module 3 — Mode Representation

**Shape test.** Longitudinal modes output (N_lon=12, D). Lateral modes output (N_lat, D). Combined after linear projection (N_lat, N_lon, D) — equivalently (N_mode=60, D) when flattened.

**Behaviour tests.**
- Longitudinal ordering: the scalar values before the repeat operation should be strictly increasing — mode 0 < mode 6 < mode 11. This is structural, not learned, so it should always hold.
- Lateral distinctiveness: two different routes (geometrically different polylines) should produce different lateral mode vectors. If two distinct routes produce identical features, the route PointNet has collapsed — common early in training.
- Combined tensor: verify that mode (lat=0, lon=0) and mode (lat=0, lon=1) differ only in their longitudinal component before the linear projection. After the projection they will mix, but the inputs should be distinguishable.

**Loss test.** The mode representation itself has no standalone loss. Test it as part of the mode selector (Module 4 below).

---

### Module 4 — Mode Selector f_selector

**Shape test.** Input: s0 features (B, N+N_m, D) and mode tensor (B, N_mode, D). Output: scores (B, N_mode) after softmax summing to 1.0, and ego_pred (B, T, 3) from the side task head.

**Behaviour tests.**
- Softmax sum: scores must sum to exactly 1.0 per batch item. Numerical errors above 1e-5 indicate a softmax implementation problem.
- Scene sensitivity: two different scenes should produce different score distributions. If the selector assigns the same scores regardless of s0, the cross-attention is not functioning — the mode features are not actually attending to the scene.
- Mode sensitivity: the same scene with two different mode tensors should produce different updated mode features out of the Transformer decoder. If modes are ignored and scores depend only on s0, the query mechanism is broken.
- Confidence calibration: before training, scores should be near uniform (1/60 ≈ 0.017 per mode). A heavily peaked distribution at random init suggests the linear projection after the Transformer has a large bias.

**Loss test.** L_CE should decrease on a fixed batch where c* is always the same index. The selector should learn to assign high probability to that index. Additionally verify L_SideTask (ego trajectory regression) decreases independently — if only L_CE goes down but L_SideTask stays flat, the side task head is detached or not receiving gradients.

---

### Module 5 — IVM (Invariant-View Module)

This is the hardest module to test because it is a preprocessing step with no learned parameters of its own (except the downstream Transformer). Test each of its four steps separately.

**Step 1 — K-NN agent selection.** Given N=20 agents, verify exactly K=10 are returned. Verify the selected agents are the K closest to ego's current position — place one agent at 1m and one at 100m, confirm the distant one is excluded when K is tight.

**Step 2 — Route filtering.** Place a route polyline behind the ego (all points have negative x in ego frame). Verify this route is discarded. Place a route ahead — verify it is retained and exactly K_r = N_r/4 points are returned.

**Step 3 — Coordinate transform.** After transformation, ego's own position should be exactly (0, 0, 0) in the output. An agent 5m directly ahead of ego should appear at approximately (5, 0, heading_relative) in the transformed output. Test this at multiple timesteps — at t=3, the transform should use ego's pose at t=3, not t=0.

**Step 4 — Time normalisation.** Verify that time indices in the output are always in range [-H, 0] regardless of the absolute timestep. At t=5, a history of frames [t=1,2,3,4,5] should appear as [-4,-3,-2,-1,0] in the output.

**Overall IVM behaviour test — time agnosticism.** This is the key property the IVM exists to provide. Run the policy on a trajectory at t=0 and at t=4. Feed the raw state at each step through IVM, then check that the transformed features at the two steps are statistically similar in distribution — not identical (the scene has changed) but in the same range and format. If features at t=4 are systematically larger or shifted, the transform is not working correctly.

---

### Module 6 — Autoregressive Policy π

**Shape test.** At each step t, input is the IVM-processed state and mode query (B, 1, D). Output is action (B, 3) representing (Δx, Δy, Δyaw), and value estimate (B, 1).

**Behaviour tests.**
- Mode conditioning: given the same state but two different mode vectors (e.g. left turn mode vs right turn mode), the policy should produce different actions. If actions are identical regardless of mode, the cross-attention is not using the mode query.
- Training vs inference: in training mode, actions should be sampled from the Gaussian and vary across calls. In inference mode, actions should be deterministic (always the mean) and identical across calls given the same input.
- Value head independence: gradients from the value head should not flow back through the policy head parameters and vice versa. Verify with a gradient inspection — this matters for PPO stability.
- Autoregressive consistency: run a full 8-step rollout. The ego trajectory produced should be spatially smooth — large discontinuities between consecutive waypoints indicate the coordinate transform in IVM is not being applied correctly per step.

**Loss test.** Under IL training, L_generator (L1 against GT trajectory) should decrease on a fixed batch. Under a mock RL setup with a fixed positive reward, the policy entropy should increase slightly — the policy should explore more when rewarded consistently.

---

### Integration checkpoints — when to wire modules together

The order to wire them up and what to verify at each join:

```
1. Agent Encoder + Map Encoder → verify combined (N + N_m) × D
   features have matching scale before feeding Transformer

2. Both encoders → Transition Model → verify β loss decreases
   with full map context vs without (should improve)

3. Mode Representation → Mode Selector → verify L_CE decreases
   and score distribution is not uniform after 200 steps

4. IVM → Policy (single step) → verify action output is in
   a physically reasonable range (not 50m per step)

5. IVM → Policy (full 8-step rollout) → verify trajectory
   is spatially coherent

6. Full Stage B pipeline → verify L_IL = L_CE + L_SideTask
   + L_generator all decrease together and none flatlines
```

---

