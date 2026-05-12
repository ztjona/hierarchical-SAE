# Series P — Frozen-trunk select-head test

Diagnostic: ME(2) trunk frozen, only `fc2_select` trainable. Result: Q_select saturation is representational, not head-level. See parent: `Research-status.md`.

## Pa_frozenTrunkSelect — Full Trunk Freeze, fc2_select Only (PLANNED)

**Hypothesis:** Q_select saturation is a *head*-level problem, not a representational one. The ME(2) trunk already encodes everything needed to ground select Q-values — it just learned the wrong head mapping under the joint loss. If we freeze the entire trunk (conv1, conv2, fc1, fc_in_aux, fc2_place) at ME(2)_E_5000 and train ONLY `fc2_select` with `td_place_mc_select` targets at the same data regime that produced ME(2) (N=3, `ENDGAME_FRACTION=0.5`), the select head should escape −1 saturation without losing the place-head competence that powers the 73% WR. This is the cheapest possible test of "head vs trunk" as the locus of the saturation pathology.

This is also the inverse of `Ne_freezePlace` (which froze the place head): together the two experiments triangulate whether the open problem lives in the representation (both should fail), the joint optimization (both should succeed), or one specific head (asymmetric outcomes).

**Code change:** new train script `train_scripts/Pa_frozenTrunkSelect(1)0511.py`. Loads `ME_endgame(2)_E_5000.pt` into a `QuartoCNNAutoreg`, sets `requires_grad=False` on all parameters, then re-enables exactly `fc2_select.weight` and `fc2_select.bias`; the optimizer only receives trainable parameters. A sanity assertion verifies that the set of trainable parameter names equals `{"fc2_select.weight", "fc2_select.bias"}` — anything else raises before training starts. Schema, target style, data recipe (`ENDGAME_FRACTION=0.5`, `N_LAST_STATES_INIT=N_LAST_STATES_FINAL=3`, `N_LAST_STATES_ENDGAME=2`), and hyperparameters (`LR=7e-4`, `TAU=0.01`, `BATCH_SIZE=32`, `MATCHES_PER_EPOCH=32`, `NUM_EPOCHs_BUFFER=8`, `EPOCHS=1000`) all mirror ME(2).

**Fixed:** `STARTING_NET=ME_endgame(2)_E_5000.pt`, `ARCHITECTURE=QuartoCNNAutoreg`, `TRANSITION_SCHEMA="decoupled_autoreg"`, `DECOUPLED_TARGET_STYLE="td_place_mc_select"`, `REWARD_FUNCTION="final"`, all trunk + `fc2_place` parameters frozen.

| Run | N_LAST_STATES | Epochs | Trainable params |
|-----|---------------|--------|------------------|
| Pa_frozenTrunkSelect(1) | 3 | 1000 | `fc2_select.weight`, `fc2_select.bias` only |

**Decision gate:**
- Q_select mean of winners (`Outcome=+1`) increases by ≥ 0.40 over the run, AND
- WR vs `bot_random` ≥ 90%, AND
- WR vs `ME_endgame(2)_E_5000` ≥ 45% (not catastrophically worse than the starting checkpoint).

Combinations:
- All three pass → head-level bug confirmed; the trunk was fine all along. Next step: same recipe with `Ne_freezePlace` for direct triangulation.
- Q_select recovers but WR collapses → head retraining destabilises the rest of the policy; need joint LR asymmetry instead.
- Q_select stays at −1 → representational problem, not head problem. Move on to the QC architecture (already designed) without further freeze experiments.

**Result (2026-05-12, 1000 epochs):**

| Metric | Gate | Observed | Pass? |
|---|---|---|---|
| Q_select mean of winners, Δ over run | ≥ +0.40 | −0.71 → −0.51 (Δ ≈ +0.20) | **no** |
| WR vs `bot_random` (last 100, smoothed) | ≥ 90% | 87.0% (peak 100%) | marginal no |
| WR vs `ME_endgame(2)_E_5000` (last 100) | ≥ 45% | 53.9% (peak 86.7%) | yes |
| Q_place winners / losers (last 100) | sanity | +0.15 / −0.17 | yes (place head intact) |

The Q_select head moved a fifth of the required distance off saturation and then stalled; the place head, identical to ME(2) by construction, kept the policy alive (53.9% vs the source checkpoint, peak 86.7%). This matches the third row of the gate table almost exactly: **`fc2_select` alone cannot escape −1 even when given a fully trained trunk and 1000 epochs of `td_place_mc_select` targets**.

**Conclusion:** Q_select saturation is **not** a head-level pathology. The mapping from trunk features to a useful select value does not exist in the ME(2) representation — `fc2_select` had nothing to learn from. Combined with the asymmetric `Nb_asymLR` result (10× LR on the select head also failed), Pa closes the head-vs-trunk triangulation on the trunk side. Do not run `Ne_freezePlace`; the next step is to change what the trunk sees / how it is supervised, which is what QC was designed to test in parallel.

