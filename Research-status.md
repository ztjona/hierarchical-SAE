# Quarto RL — Research status

Slim ledger. Long-form per-series content lives in [`docs/diary/`](docs/diary/).
Add a new series entry there when starting a new code-version letter; append
result paragraphs here under "Recent results" when a sweep concludes.

---

## Champion

- **`ME_endgame(2)0429_ENDGAME_FRACTION_0.5`** — 73.7% WR vs `bot_loss-BT`,
  85.8% WR vs `bot_random`, loss 0.207. Uses the M-series decoupled-autoreg
  trunk (`QuartoCNNAutoreg`) with `td_place_mc_select` targets,
  N_LAST_STATES_INIT=2 curriculum to N=4, ENDGAME_FRACTION=0.5,
  N_LAST_STATES_ENDGAME=2, LR=7e-4, TAU=0.01, 5000 epochs.
- **Interpretability substrate of record:** `OA_unifiedAux(1)0509_N_LAST_STATES_INIT_2`
  — 66.3% WR vs `bot_loss-BT`. Same loss / schema family as ME(2) but with the
  unified 32-d phase-stable aux (`[offered_one_hot ; available_mask]`) and no
  phase embedding. 7pp behind ME(2) on WR; standing trade for SAE-able activations.
- **Candidate replacement (pending confirmation):**
  `Ta_minimaxSelect(1)0514_DEPTH_2` — **80.3% WR vs `bot_loss-BT`, 90.2%
  vs `bot_random`** (peaks 81.5% / 90.9%, trend still rising at +3.6 pp/1000ep
  when training was truncated at epoch 4000). Sa(3) substrate + minimax
  depth=2 supervised target on Q_select with `Q_place` unchanged. Q_select
  remains "saturated" under the match-outcome-conditioned Δ metric (Δ ≈
  +0.02); WR gain runs through the place head. Promotion contingent on a
  clean 5000-epoch re-run and a 10k confirmation. Full write-up:
  `docs/diary/series-T.md`.
- **Prior candidate (now superseded):** `Sa_archScan(3)0512_ARCH_S4_uniform512`
  — 73.5% WR; held as interpretability substrate of record under Ta(1)
  since Ta(1) uses the Sa(3) trunk.

## Current Open Problem — Q_select "saturation"

> **[AI-REASONED PROVISIONAL FRAMING — 2026-05-18]** The framing of this
> section was revised after the Ta-series results. Direct measurements
> (numbers, gate pass/fail) are reliable; mechanism stories are inferential
> and may be biased. Future readers: verify with
> [`analysis/qselect_diagnostics/PLAN.md`](analysis/qselect_diagnostics/PLAN.md)
> before treating the inferred conclusions as load-bearing.

`Q_select` shows ~0 winners-minus-losers Δ across every architecture and
target style we have tried (M, N, O, Q, R, S, T) — the place head carries
WR while the select head contributes ~0 to value separation **under the
current match-outcome-conditioned metric** (`QuartoRL/results_io.py:192-214`).

Whether this represents a genuine learning failure or a measurement
artefact is now open. Triangulation as of 2026-05-18:

- **Head only (Pa_frozenTrunkSelect):** freezing the ME(2) trunk and retraining
  only `fc2_select` lifts the winners-mean by ≈ +0.20, not the ≥ +0.40 the gate
  required. **Head retraining alone is insufficient.**
- **Asymmetric LR (Nb_asymLR):** 10× LR on the select head produced no separation.
- **Wider `fc1` + aux legality + no inference mask (QC_unifiedNoMask):** the
  aux BCE collapsed (0.69 → 0.15) but `invalid_argmax_rate` stayed at 0.32 and
  `Q_select` reproduced the same −0.73 saturation.
- **Static per-head loss reweighting (Ra_lossWeight):** `loss_select` is
  pinned at ~0.24 across a 30× swing in α_select / α_place. Reweighting
  buys a marginal Q_select Δ (≤ +0.04) at a one-way cost in WR.
  **Gradient-starvation rejected.**
- **Structural trunk variants (Sa_archScan):** deeper conv (Sa(1)) produced
  the largest numeric Q_select Δ (+0.170) but **no visible plate separation
  in the qv heatmaps**, with a 30pp WR cost. Re-interpreted 2026-05-18 as
  a likely artefact of a failing run rather than evidence that deeper conv
  helps Q_select. Wider/uniform FC (Sa(3)) matched ME(2) WR with the
  unified-aux substrate; no Q_select gain.
- **Minimax-oracle distillation (Ta-series):** clean per-piece minimax
  targets at depth=2 drove `loss_select` to **0.055** (well below the 0.24
  R/S floor — proving the floor was a target-noise artefact) **and** lifted
  WR to **80.3%** (new champion candidate). But Q_select Δ stayed ≈0 under
  the existing metric. Three live hypotheses for why (see below).

Triangulated **direct** observations:

1. `loss_select` floor under MC return targets (0.24) was indeed target
   noise. Confirmed by Ta: clean targets are fittable.
2. WR can be lifted to ≥80% via clean select-side gradient even when the
   Q_select head output remains uninformative under the current Δ metric.
   The trunk benefits from supervised auxiliary signal even when the head
   output looks flat.

**[INFERENTIAL]** Three live hypotheses for the Δ ≈ 0 observation:

- **H1 — metric artefact.** Most Quarto select decisions are functionally
  neutral; match-outcome-conditioned Δ averages over them. A network
  correctly outputting "≈0 except in forcing positions" looks saturated.
- **H2 — signal sparsity.** Only a small fraction of buffer select rows
  carry non-trivial oracle targets; α reweighting (Ra) cannot help per-row.
- **H3 — multitask interference.** Shared trunk converges to dense-gradient
  Q_place features; Q_select head cannot invert them on sparse-signal
  events.

Next direction: **diagnostic suite, not another architecture swing.** See
[`analysis/qselect_diagnostics/PLAN.md`](analysis/qselect_diagnostics/PLAN.md).
Sb_hybridTrunk and similar architectural hedges are deprioritised pending
at least one of D1/D2/D3 returning evidence for an architectural mechanism.

## Code-version letters (active and queued)

| Letter | Series | Architecture / Schema | Status |
|---|---|---|---|
| A–E | early / combined_avg baseline | joint | done — see `docs/diary/series-A.md` |
| F | adversarial sign flip | joint | failed/reverted — `series-FG.md` |
| G | separate_bellman | joint | done — `series-FG.md` |
| H–K | terminal mask / unbound / final / coupled | joint | done — `series-HIJK.md` |
| L | mc_select | joint | done — `series-L.md` |
| M | decoupled_autoreg | `QuartoCNNAutoreg*` + `Quarto_autoreg_bot` | **champion** — `series-M.md` |
| N | shared-trunk diagnostics | M-series | done — `series-N.md` |
| O | unified_autoreg | `QuartoCNNAutoregUnified*` + `Quarto_unified_bot` | done — `series-O.md`, `docs/diary/2026-05-08_unified-aux-trunk.md` |
| P | frozen-trunk select | M-series | done (negative) — `series-P.md` |
| Q | unified + aux legality + no mask | `QuartoCNNUnifiedNoMask` + `Quarto_unified_nomask_bot` | done (gate failed) — `series-Q.md`, `docs/diary/2026-05-11_qc-no-mask.md` |
| R | per-head loss weighting | OA-family | done (gate failed) — `series-R.md` |
| S | structural trunk variants | OA-family | done (mixed) — `series-S.md`; Sa(1) "+0.170 Δ" re-interpreted as artefact 2026-05-18 |
| T | minimax-oracle select target | Sa(3) | done — `series-T.md`; Ta(1) = **WR champion candidate**, but Q_select metric still flat |
| **U?** | **Q_select diagnostics (not a training series)** | n/a | **active** — `analysis/qselect_diagnostics/PLAN.md` (D1/D2/D3) |

## Recent results

### Ta_minimaxSelect — 2026-05-18 (3 runs)

| Run | Param | Epochs | `loss_select` | Q_sel Δ | Q_pl Δ | WR vs BT | WR vs random |
|---|---|---|---|---|---|---|---|
| Ta(1) | DEPTH=2, 16-d masked | 4000 (truncated) | **0.055** | +0.019 | +0.514 | **80.3%** | 90.2% |
| Ta(2) | DEPTH=1, 16-d masked | 5000 | **5e-6** | +3e-5 | +0.335 | 71.5% | 85.0% |
| Ta(3) | DEPTH=2, scalar | 4000 (truncated) | 0.044 | +0.002 | +0.591 | 74.9% | 87.0% |

Gates: `loss_select` ≤ 0.10 **PASS everywhere** (target-noise floor broken).
Q_select Δ ≥ +0.40 **FAIL everywhere** (≈0). WR ≥ 70% **PASS everywhere**;
Ta(1) is new champion candidate. **[INFERENTIAL]** Pre-registered "trunk
can't represent piece × board" routing to Sb hybrid re-examined and
deprioritised — see `series-T.md` → Result section. Tb_depth4 and
Td_oracleCache dropped.

### Sa_archScan — 2026-05-12 (5000 epochs each, 3 runs)

| Run | Trunk | Q_select Δ | Q_place Δ | WR vs BT (final / peak) |
|---|---|---|---|---|
| Sa(1) | deepConv | **+0.170** (largest on record, numeric only — no plate separation in heatmaps) | +0.231 | 44.3% / 47.8% (still rising) |
| Sa(2) | wideFC | +0.072 | +0.332 | 63.7% / 66.6% |
| Sa(3) | uniform512 | +0.037 | +0.457 | **73.5% / 75.2%** |

Sa(3) matches the M-series champion on WR while keeping the unified-aux
substrate (+7pp over OA(1) interp champion at no WR cost) — promoted to
**interpretability substrate of record (pending confirmation run)**. Sa(1)
moved Q_select Δ numerically but not visibly; WR curve still rising at
5000 epochs — direction not retired. Full write-up in
[`docs/diary/series-S.md`](docs/diary/series-S.md).

### Ra_lossWeight — 2026-05-12 (5000 epochs each, 4 runs)

| α_place / α_select | loss_select | Q_select Δ | WR vs BT (final / peak) |
|---|---|---|---|
| 1.0 / 1.0 | 0.246 | +0.010 | 71.3% / 72.7% |
| 1.0 / 3.0 | 0.245 | +0.030 | 64.2% / 66.9% (rising) |
| 0.3 / 1.0 | 0.256 | +0.033 | 65.1% / 66.8% (rising) |
| 0.1 / 1.0 | 0.242 | +0.044 | 48.1% / 49.7% (rising) |

`loss_select` floor flat at ~0.24 across the grid; WR trades 1:1 against
α_select. **Gradient starvation rejected** as the mechanism. Full write-up
in [`docs/diary/series-R.md`](docs/diary/series-R.md).

### QC_unifiedNoMask(1) — 2026-05-12 (5000 epochs)

| Metric | Gate | Observed | Pass? |
|---|---|---|---|
| WR vs `bot_random` | ≥ 95% **hard** | 85.0% (peak 86.1%) | **no** |
| WR vs `ME(2)_E_5000` | ≥ 50% | 46.2% (peak 48.7%) | **no** |
| `invalid_argmax_rate` (last 100) | < 0.05 | 0.324 (start 0.32, no movement) | **no** |
| `legality_loss` (BCE) | qualitative | 0.69 → 0.15 | aux head fits cleanly |

Aux head decoupled from the Q heads at λ_legality=0.05; do not iterate on Q3/Q4
from QC. Full write-up in [`docs/diary/series-Q.md`](docs/diary/series-Q.md).

### Pa_frozenTrunkSelect(1) — 2026-05-12 (1000 epochs)

| Metric | Gate | Observed | Pass? |
|---|---|---|---|
| Q_select winners Δ | ≥ +0.40 | +0.20 (−0.71 → −0.51) | **no** |
| WR vs `bot_random` | ≥ 90% | 87.0% | marginal |
| WR vs `ME(2)` | ≥ 45% | 53.9% | yes |

Saturation is **not** a head-level pathology. Full write-up in
[`docs/diary/series-P.md`](docs/diary/series-P.md).

## Forward queue

1. **Q_select diagnostic suite (D1/D2/D3)** — *active.* Three non-training
   diagnostics under `analysis/qselect_diagnostics/` that distinguish
   H1 (metric artefact) / H2 (signal sparsity) / H3 (multitask interference)
   for the long-standing "Q_select saturation" observation. **D1 first**
   (cheapest, falsifies the most). Implementation plan:
   [`analysis/qselect_diagnostics/PLAN.md`](analysis/qselect_diagnostics/PLAN.md).
2. **Ta(1) clean re-run + 10k confirmation** — the existing Ta(1) checkpoint
   was truncated at epoch 4000 with WR still rising at +3.6 pp/1000ep.
   Re-run to 5000 (clean) then 10k (confirmation) before formally promoting
   over ME(2) as overall champion. Cheap; reuses the existing train script.
3. **Sa(3) confirmation run** — re-run `Sa_archScan(3)0512_ARCH_S4_uniform512`
   at 10k epochs to lock the interpretability-substrate-of-record claim
   independently of Ta(1). Lower priority since Ta(1) already uses Sa(3).
4. **Rb_schedAlpha** — time-varying (α_place, α_select). Tests whether a
   warm Q_place trunk can absorb select-loss pressure that the cold-start
   Ra grid could not. Lowest priority — Ra's `loss_select` floor was
   targets-side evidence (now reframed by Ta), so Rb is an
   optimisation-side hedge with less specific motivation than before.

**Dropped 2026-05-18:**

- **Tb_depth4** — Ta(1) at depth=2 already produced fittable targets and
  Δ ≈ 0 remained; deeper lookahead does not address H1/H2/H3.
- **Td_oracleCache** — Ta(2) showed depth=1 collapses to zero, so caching
  at that depth is pointless; depth=2 ran without it.
- **Sb_hybridTrunk** — built on the Sa(1) "+0.170 Δ" inference that the
  Ta-series re-reading classifies as likely artefact. Revisit only if a
  diagnostic returns evidence for an architectural mechanism.

All runs use the per-head loss / grad logging and JSONL emission from
2026-05-12; runs trained after 2026-05-14 also receive `wr_trend` slope
statistics in their JSONL final record.

---

## Experiment Naming Convention

Two-part name: **`XY_description`**.

- **First letter (X)** = code version / significant algorithm change. Currently
  in use: A–N, O (unified aux), P (frozen trunk), Q (aux legality + no mask),
  R (loss weighting — queued), S (structural — queued).
- **Second letter (Y)** = hyperparameter sweep within the same code version:
  `a`, `b`, `c`, ...

`run_trains.py` appends `(<idx>)<MMDD>_<PARAM>_<VALUE>` to produce the per-run name (e.g. `MB_final(3)0426_N_LAST_STATES_INIT_4`).

## Operational notes (load-bearing)

- **`mode_2x2=True` is the default** across training and evaluation.
- **`TRANSITION_SCHEMA` is a triplet contract** (schema × model class × bot class). See `CLAUDE.md` for the valid combinations.
- **Pickles may contain CUDA tensors** — use `CPUUnpickler` (see `tools/view_qv.py` or `QuartoRL/results_io.py`) on CPU-only machines.
- **`-1` sentinels in the experience tuple are load-bearing** (`action_place=-1` first move, `action_sel=-1` terminal). Don't filter them out.
- **JSONL results format** is the new source of truth for cross-experiment comparison; see `tools/results_compare.py`. Legacy pickles are backfilled via `tools/pkl_to_jsonl.py`.

## Key takeaways (consolidated)

1. **N=2 starves Q_select of positive samples.** The decoupled-autoreg N=2 window contains only `{loser_place, loser_select, winner_terminal_place}` — zero winner-side select transitions. N≥3 fixes availability, not learning.
2. **Q_select saturation persists across joint/decoupled/unified schemas and across head-only / trunk-wider / aux-legality fixes.** See Open Problem above.
3. **Curriculum jumps from N=2 to higher N catastrophically forget** (Ac_fineShallow). Replay-buffer distribution shift, not LR.
4. **Endgame anchor buffer (ENDGAME_FRACTION=0.5, N_LAST_STATES_ENDGAME=2)** is the single change that unlocked 73% WR (M-series → ME).
5. **Loss reweighting (R) is rejected as a fix for Q_select saturation.** `loss_select` has a floor near 0.24 invariant to a 30× swing in α_select / α_place; reweighting buys Q_select Δ ≤ +0.04 at a one-way WR cost. Gradient starvation is not the mechanism.
6. **Structural trunk changes (S) close the unified-aux WR tax but not Q_select saturation.** Sa(3) matches ME(2) WR with the unified-aux substrate (+7pp over OA(1)) — new interpretability champion candidate. Sa(1) deepConv produced the largest numeric Q_select Δ on record (+0.170) but with no visible plate separation and a WR collapse. The select-side gap is not where deeper trunks alone close it.
7. **Target noise (T-series) was half the story.** Clean minimax targets drove `loss_select` from the 0.24 R/S floor down to 0.055 — confirming the floor was target noise. But `Q_select` Δ stayed ≈0 under the match-outcome metric, and Ta(1) became a new WR champion candidate (80.3%) *via the place head*, not via a now-discriminative Q_select. The "Q_select saturation" open problem is now **under-determined between metric artefact (H1), buffer signal sparsity (H2), and multitask interference (H3)** — see `analysis/qselect_diagnostics/PLAN.md`. **[INFERENTIAL — written 2026-05-18 by an AI agent revisiting its own prior claims; verify with diagnostics before treating as load-bearing.]**

For each of these, the originating series file in `docs/diary/` carries the full evidence.
