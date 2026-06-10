# Series Y — select-pressure into the trunk (auxiliary hot-piece head)

Parent: [`Research-status.md`](../../Research-status.md). Motivating evidence:
[`series-X.md`](series-X.md) (the select wall is **trunk allocation, not
capacity** — reshaping the same trunk select-only cut held-out blunder
16.7→6.6%) + [`analysis/competence_audit/METRICS.md`](../../analysis/competence_audit/METRICS.md).

**Code change (new code letter).** `QuartoCNNAutoregUnifiedS4Hot` (subclass of
S4) adds an auxiliary `fc_hot` head (16-d) predicting the depth-1 hot-piece mask;
trained with a masked `BCEWithLogits` (`LAMBDA_HOT`) on SELECT rows against the
frozen `target_hot_piece` (`EMIT_HOT_MASK` is now true when `LAMBDA_HOT>0`).
`forward()` is unchanged — returns `(q_place, q_select)` — so the triplet stays
`unified_autoreg` × `QuartoCNNAutoregUnifiedS4Hot` × `Quarto_unified_bot`. The hot
head is a **training-time scaffold** (not consulted at inference; discardable).
Loss in `hot_head_aux_loss` (QuartoRL/RL_functions.py), wired in `trainRL.py`
(`LAMBDA_HOT`, logged as `loss_data["loss_hot_values"]` = weighted λ·BCE).

## Ya_hotHead — does forcing the trunk to encode hotness cut select blunders?

**Hypothesis.** The select wall is the trunk under-allocating capacity to
piece-safety. A dense aux BCE hot head reshapes the trunk to encode hotness,
which the (unchanged) `fc2_select` reads → lower in-play avoidable rate **vs a
punishing opponent**, with the place head (`LAMBDA_PLACE_WIN=0.5`) intact. This
is **stage A**; stage B (wire `σ(hot_logits)` into `q_select`) is contingent on a
re-probe showing the trunk enriched but `q_select` still under-using it.

**Design — 4-arm `λ_hot` sweep on the X(1) recipe** (`S4Hot`, depth-2 minimax
oracle always on, `N_LAST_STATES_INIT=4`, `LAMBDA_PLACE_WIN=0.5`, 6000 epochs):

| Arm | `LAMBDA_HOT` |
|---|---|
| Ya(1) | 0.03 |
| Ya(2) | 0.1 |
| Ya(3) | 0.3 |
| Ya(4) | 1.0 |

λ log-spaced to **bracket the loss-balance point** (raw BCE ≈ 0.7 vs `L_select` ≈
0.05–0.1, so balance ≈ λ∈[0.1,0.3]). The sweep is **self-calibrating**: each arm
logs `loss_hot` (weighted λ·BCE) vs `loss_select` and `grad_norm`/clip-rate — a
λ where the aux dominates the gradient (clip-rate high **and** place metric
regressing) is the X(3) failure signature and is read off post-hoc (no abort
logic). NB the grad-norm/clip behaviour is a **loss-scale** effect, not a clip
artefact — see `series-X.md` → 2026-06-09 correction and
[[aux-loss-scale-vs-grad-clip]].

**Decision gate.** Punishing-opponent autopsy (`loss_autopsy.py` default
opponent) — avoidable rate ↓ from ~12% **with missed-win ~1.3% intact** — plus
`audit.py` Test-B hot-give rate and inline D1/WR. Promote the λ that most cuts the
punishing avoidable rate without a place/WR regression to a tuned 10k run.

## Result — pending
