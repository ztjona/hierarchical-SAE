# Quarto RL — Research status

Slim ledger. Long-form per-series content lives in [`docs/diary/`](docs/diary/).
Add a new series entry there when starting a new code-version letter; append
result paragraphs here under "Recent results" when a sweep concludes.

---

## Champion

- **`Ve_oracleAblation(4)0522_DISABLE_NEVER_10k`** — **87.2% WR vs `bot_loss-BT`
  (peak 88.9%), 93.8% vs `bot_random` (peak 94.8%)**, loss 0.053,
  `loss_select` 0.041. Same recipe as Ve(1) — Sa(3) unified-autoreg trunk
  (`QuartoCNNAutoregUnifiedS4`) with `td_place_minimax_select` targets
  at depth=2, oracle always on — extended to 10000 epochs. Trends still
  rising at +0.65 / +0.44 pp/1000ep against BT / random at run end.
  **D1: best on record — `safe_piece_recall=0.846`,
  `forcing_loss_bottom_recall=0.968`, ρ̄=0.610** (vs Ve(1) @6k: 0.821,
  0.957, 0.578). Both heads improve with more oracle-supervised training.
  Supersedes ME(2) by **+13.5 pp WR vs BT** and Ve(1) by +2.0 pp WR / +2.5
  pp safe-piece recall. See `docs/diary/series-V.md`.
- **Prior champion (superseded 2026-05-24):**
  `ME_endgame(2)0429_ENDGAME_FRACTION_0.5` — 73.7% WR vs BT, 85.8% vs
  random. Held the champion slot 2026-04-29 → 2026-05-24. M-series
  decoupled-autoreg trunk, no oracle supervision. Retained as the
  reference point for any non-oracle recipe.
- **Interpretability substrate of record:**
  `OA_unifiedAux(1)0509_N_LAST_STATES_INIT_2` — 66.3% WR vs BT. Unified
  32-d phase-stable aux (`[offered_one_hot ; available_mask]`); SAE-able
  activations. Now 20.9 pp behind Ve(4) on WR — gap widened by the
  oracle recipe; revisit as a candidate for "oracle + unified-aux"
  substrate after Wa_oracleStates lands.
- **Prior candidates (superseded by Ve(4)):**
  - `Ve_oracleAblation(1)0519_DISABLE_NEVER` — 85.2% / 87.1% peak @6000,
    +2.9 pp/1000ep. Ve(4) is the clean 10k continuation; same recipe.
  - `Ta_minimaxSelect(1)0514_DEPTH_2` — 80.3% WR, truncated at 4000.
    Ve series is the clean continuation modulo training length.
- **Earlier candidate:** `Sa_archScan(3)0512_ARCH_S4_uniform512` — 73.5%
  WR; trunk used by all T/V runs.

## Current Open Problem — Q_select "saturation" [**RESOLVED as metric artefact, 2026-05-24**]

> **[2026-05-24 — closed.]** D1 has now scanned every relevant family
> (T, V, O, S). Pre-T runs (O, S — never oracle-supervised) land at
> `safe_piece_recall = 0.35–0.52` (chance 0.11) with Sa(3) S4 leading
> (0.516). Oracle-disabled runs Ve(2/3) sit at 0.54–0.58. Oracle-always-on
> runs Ve(1/4) sit at 0.82–0.85. The select head was always doing
> *some* position-structure work; the legacy match-outcome-conditioned
> Δ averaged it away. **The "Q_select saturation" framing is retired.**
> Future readings should use the inline `d1_*` JSONL fields shipped
> since 2026-05-24 (see `CLAUDE.md` operational rules). Pre-T scan
> table + analysis: `analysis/qselect_diagnostics/REPORT.md` → "Pre-T
> scan". M-series D1 still pending — needs adapter work for the joint
> schema, but is low-priority since OA/Sa already locked the headline.
>
> Below preserved for historical context only.

> **[2026-05-19 — diagnostic suite returned: H1 (metric artefact) supported,
> H2 and H3 falsified.]** See
> [`analysis/qselect_diagnostics/REPORT.md`](analysis/qselect_diagnostics/REPORT.md).
> The Q_select head at Ta(1) epoch 4350 encodes position structure
> (`safe_piece_recall = 0.81` vs chance 0.11, Spearman ρ̄ = 0.55); the
> match-outcome-conditioned Δ metric was averaging it away. Per-row
> oversampling (H2) and structural trunk decoupling (H3) are deprioritised.
> **[2026-05-22 — Ve_oracleAblation extends H1 across runs.]** D1 reproduces
> on Ve(1)@6000 (0.82 / 0.96 / 0.58) and shows expected decay on
> Ve(2)/Ve(3) when the oracle is disabled (0.54–0.58 / 0.83–0.85 /
> 0.25–0.27), all still above chance. The recall metric is oracle-supervision
> sensitive *and* generalises across checkpoints, not just a Ta(1)
> idiosyncrasy. See [`docs/diary/series-V.md`](docs/diary/series-V.md).
> The text below is preserved as the framing that motivated the diagnostic
> — the diagnostic now supersedes it.

> **[AI-REASONED PROVISIONAL FRAMING — 2026-05-18, superseded by 2026-05-19 diagnostic]** The framing of this
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
| M | decoupled_autoreg | `QuartoCNNAutoreg*` + `Quarto_autoreg_bot` | prior champion (2026-04-29 → 2026-05-24) — `series-M.md` |
| N | shared-trunk diagnostics | M-series | done — `series-N.md` |
| O | unified_autoreg | `QuartoCNNAutoregUnified*` + `Quarto_unified_bot` | done — `series-O.md`, `docs/diary/2026-05-08_unified-aux-trunk.md` |
| P | frozen-trunk select | M-series | done (negative) — `series-P.md` |
| Q | unified + aux legality + no mask | `QuartoCNNUnifiedNoMask` + `Quarto_unified_nomask_bot` | done (gate failed) — `series-Q.md`, `docs/diary/2026-05-11_qc-no-mask.md` |
| R | per-head loss weighting | OA-family | done (gate failed) — `series-R.md` |
| S | structural trunk variants | OA-family | done (mixed) — `series-S.md`; Sa(1) "+0.170 Δ" re-interpreted as artefact 2026-05-18 |
| T | minimax-oracle select target | Sa(3) | done — `series-T.md`; Ta(1) = **WR champion candidate**, but Q_select metric still flat |
| **U?** | **Q_select diagnostics (not a training series)** | n/a | **done 2026-05-19** — `analysis/qselect_diagnostics/REPORT.md` — H1 supported, H2/H3 falsified |
| V | minimax-oracle ablation mid-training | Sa(3) + T-recipe | **done 2026-05-24** — `series-V.md`; Ve(4) @10k = **new champion (87.2% WR, D1 best-on-record)** — supersedes ME(2); Ve(1)/(2)/(3) D1 confirmed oracle is persistent driver not warmup |

## Recent results

### Wa_oracleStates + loss autopsy — 2026-06-04

**Wa (N_LAST_STATES sweep, N∈{8,12,16} on the Ve recipe).** Full write-up:
[`docs/diary/series-W.md`](docs/diary/series-W.md). D1 is an inverted-U in N,
peaking at **N=8** (best-on-record: safe-piece 0.89 back-half / 0.92 final,
ρ̄ 0.67) — beating the champion Ve(4) (N=4: 0.846 / 0.610) at WR statistically
tied (87.0% vs 87.2%). N≥12 is dominated (worse WR *and* worse D1; only "wins"
on the misleading `loss_select`, which falls with N as the buffer fills with
easy early-game states — a dilution artefact, not quality). N=8 is a candidate
**interpretability substrate** (confirmation seed advised); Ve(4) stays WR
champion by a whisker. N axis exhausted for WR.

**Loss autopsy of Ve(4) vs random** (`analysis/competence_audit/REPORT.md`).
The argmax policy loses **~4.2%** (the 5.7% in `champion-results.jsonl` was the
temp=0.1 *sampling* agent). Of those losses, only ~⅓ are avoidable `Q_select`
blunders (**~1.4%** of games); ~⅔ are **forced** positions the agent walks into
mid-game (~2.7%). The place head also misses **6.7%** of immediate wins. **No
first-player effect** (P1/P2 symmetric, |z|<1.6). Consequence: a perfect select
head caps the WR gain at ~1.4 pp — so the next training direction is **place-side
win-taking supervision**, then deeper-oracle planning for the forced floor, with
the `Q_select` margin loss demoted. See forward queue.

### Ve_oracleAblation(4) — 10k confirmation, 2026-05-24

Same recipe as Ve(1) (Sa(3) + minimax depth=2, oracle always on) at
EPOCHS=10000. Full write-up: [`docs/diary/series-V.md`](docs/diary/series-V.md).

| Metric | Ve(4) @10k | Ve(1) @6k | Δ |
|---|---|---|---|
| WR vs BT (final / peak) | **87.2% / 88.9%** | 85.2% / 87.1% | +2.0 / +1.8 |
| WR vs random (final / peak) | **93.8% / 94.8%** | 92.0% / 92.9% | +1.8 / +1.9 |
| WR trend vs BT (pp/1000ep, last 5k) | +0.65 ↑ (CI 0.51–0.78) | +2.9 ↑ | slowing but still positive |
| `loss_select` | **0.041** | 0.048 | −0.007 |
| D1 `safe_piece_recall` | **0.846** | 0.821 | +0.025 |
| D1 `forcing_loss_bottom_recall` | **0.968** | 0.957 | +0.011 |
| D1 `spearman_rho_mean` (n) | **0.610** (916) | 0.578 (924) | +0.032 |

**Reading.** Longer oracle-supervised training keeps improving both the
place head (WR) and the select head (D1 position-structure encoding).
Trend slope on WR has decelerated by ~4× from Ve(1)'s window but
remains positive and significant — model is not saturated at 10k. D1
deltas are smaller than the Ve(1)→Ve(2/3) gap (~30 pp recall) but all
positive, with the safe-piece-recall ceiling now within 0.15 of the
oracle's argmax.

**Action.** Ve(4) **formally promoted to champion** 2026-05-24,
superseding ME(2). Next-up training axis: N_LAST_STATES sweep on the
Ve recipe (Wa_oracleStates), since N directly modulates how many
oracle-supervised select transitions per game enter the buffer.

### Ve_oracleAblation — 2026-05-21 (3 runs @ 6000 epochs) + D1 follow-up 2026-05-22

Knob `MINIMAX_DISABLE_AFTER_EPOCH ∈ {None, 2000, 4000}` on the Ta(1) recipe.

| Run | Disable @ | `loss_select` | WR vs BT (final / peak) | Trend pp/1000ep | D1 `safe_piece_recall` | D1 `spearman_rho_mean` |
|---|---|---|---|---|---|---|
| Ve(1) NEVER | — | **0.048** | **85.2% / 87.1%** | **+2.9 ↑** | **0.821** | **0.578** |
| Ve(2) DIS 2000 | 2000 | 0.240 | 78.3% / 80.3% | +2.1 ↑ | 0.541 | 0.247 |
| Ve(3) DIS 4000 | 4000 | 0.213 | 79.1% / 81.5% | +0.1 (p=0.79) | 0.580 | 0.272 |
| Ta(1) ref @ 4350 | — | 0.055 | 80.3% / 81.5% | +3.6 ↑ | 0.807 | 0.547 |
| chance | — | — | — | — | 0.114 | 0 |

**Reading.** The oracle is a *persistent driver*, not a removable warmup.
Ve(1) reproduces and slightly exceeds Ta(1) on every D1 metric. Disabling
the oracle (Ve(2)/(3)) causes Q_select position-structure recall to
*decay* under MC supervision — safe-piece recall drops ~30pp, ρ̄ halves —
and WR caps ~6–7pp below Ve(1). Late disable ≈ early disable, suggesting
MC pulls Q_select toward a fixed equilibrium below the oracle-supervised
one. `loss_select` snaps from the minimax 0.05 floor back to the R/S 0.24
floor within a single checkpoint of the disable epoch, independently
re-confirming that the 0.24 floor is a target-noise property.

### Q_select diagnostic suite — 2026-05-19 (on Ta(1) @ epoch 4350)

D1/D2/D3 per [`analysis/qselect_diagnostics/PLAN.md`](analysis/qselect_diagnostics/PLAN.md);
full write-up in [`analysis/qselect_diagnostics/REPORT.md`](analysis/qselect_diagnostics/REPORT.md).

| Diagnostic | Metric | Joint net | Decoupled net (D3) | Chance | Verdict |
|---|---|---|---|---|---|
| D1 | `safe_piece_recall` | **0.807** | 0.620 | 0.115 | H1 supported |
| D1 | `forcing_loss_bottom_recall` | **0.946** | 0.928 | 0.703 | H1 supported |
| D1 | Spearman ρ̄ (n=896) | **0.547** | 0.365 | — | H1 supported |
| D2 | `nonzero_frac_p5` (cheap path) | **0.953** overall (N=4: 0.94, N=2: 1.00) | — | — | H2 falsified |

The strict `forcing_loss_recall` (n=1 forcing piece) is uninformative —
mean forcing-set size is 5.83 of ~14 available pieces, so single-forcing
states only fire ~0.1% of the time. Use the relaxed bottom-set recall
instead.

**Action:** retire α-reweighting (H2) and structural trunk decoupling (H3)
from the forward queue. Replace the match-outcome-conditioned Q_select
gate in `QuartoRL/results_io.py:192-214` with the D1 position-structure
recalls. Open follow-up: re-evaluate older non-T checkpoints under D1.

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

**Standing constraint (user, 2026-06-04):** the champion must be a **pure
learned policy — no inference-time tactical search.** All WR fixes land in the
weights.

1. **Place-side win-taking supervision (next training series, new code letter)**
   — the 2026-06-04 loss autopsy (`analysis/competence_audit/REPORT.md`) shows
   the argmax champion loses ~4.2% to random, of which only ~1.4% (a third) is
   avoidable `Q_select` blunders — the rest is *forced* positions the agent
   walks into mid-game (~2.7%) — and the place head misses **6.7%** of immediate
   wins. So the select-margin idea is demoted; the highest WR-per-effort lever is
   place-side: mask-supervise `Q_place` toward the winning cell on every PLACE
   state where one exists (cf. `competence_audit/PLAN.md` → Vf). Cheap, reuses
   the win-check machinery, and trims forced exposure indirectly.
2. **Deeper oracle / planning (depth-3)** — the only lever that attacks the
   dominant ~2.7% *forced* floor (locally unavoidable at the give, but the agent
   could avoid *reaching* those positions with lookahead). Expensive; now
   justified by the autopsy as the dominant loss class. Sequence after the
   place-side series.
3. **`Q_select` margin/ranking loss** — for the avoidable ~1.4% only. Worth
   doing for the interpretability substrate (pushes `safe_piece_recall`→1.0 in
   the weights), but no longer the headline WR fix.
4. **Sa(3) 10k confirmation run** — re-run
   `Sa_archScan(3)0512_ARCH_S4_uniform512` at 10k epochs to lock the
   interpretability-substrate-of-record claim. **Priority raised** —
   the pre-T D1 scan (forward-queue #3, done 2026-05-24) shows Sa(3)
   leads the pre-T pack on D1 (`safe_piece_recall=0.516` vs OA ~0.36),
   reversing the prior series-S verdict that Sa(3) "no Q_select gain".
   Worth confirming the +0.15 recall lead is stable across training
   length before promoting Sa(3) as a *non-oracle* interp substrate.
5. **D1 on M-series (joint schema)** — requires adapter work in
   `analysis/qselect_diagnostics/_common.py` to support `Quarto_autoreg_bot`
   + the joint-schema `gen_experience` path. Low priority — M is the
   displaced champion, and the OA/Sa scan already locked the headline
   finding that pre-oracle runs sit at 0.35–0.52 safe-piece recall.
6. **Faithful-path D2 (buffer-dump hook)** — optional config-gated hook in
   `trainRL.py` to log the actual replay buffer at fixed checkpoints. The
   2026-05-19 D2 verdict was on the cheap path; faithful path would lock
   the H2-falsified finding for early-training phases too.

**Dropped 2026-05-24 (after Ve(4) + D1 ship):**

- **"Ve(4) 10k confirmation"** — done; Ve(4) promoted to champion.
- **"Replace Q_select gate with D1 recalls (item #1 of 2026-05-22 queue)"**
  — done; `COMPUTE_D1_INLINE = True` in `trainRL.py`, `build_checkpoint_record`
  and `build_final_record` merge `d1_safe_piece_recall`,
  `d1_forcing_loss_bottom_recall`, `d1_spearman_rho_mean` into every
  future checkpoint event. Gated on `TRANSITION_SCHEMA == "unified_autoreg"`.
  See `CLAUDE.md` operational rules + [[d1-jsonl-contract]] memory.
- **"D1 on O/S pre-T runs (item #3 of 2026-05-22 queue)"** — done.
  Pre-T runs land 0.35–0.52 safe-piece recall (chance 0.11); Sa(3)
  leads (0.516), OA(1) N=2 and OA(4) N=12 at chance on forcing-loss
  detection. Full table in `analysis/qselect_diagnostics/REPORT.md`
  → "Pre-T scan".

**Dropped 2026-05-22 (after Ve(1/2/3)):**

- **"Oracle as warmup that can be removed"** — Ve(2)/Ve(3) refute this on
  both WR (~6–7pp cap below Ve(1)) and D1 (recall drops ~30pp post-disable).
  Production training recipes must keep the oracle on.

**Dropped 2026-05-19 (after diagnostic):**

- **Rb_schedAlpha** — H2 (signal sparsity) was falsified; α-reweighting
  in any form (static or scheduled) is now without motivation.
- **Sb_hybridTrunk / structural select-trunk decoupling** — H3 was
  falsified; the shared trunk helps Q_select, not hurts it.

**Dropped 2026-05-18:**

- **Tb_depth4** — Ta(1) at depth=2 already produced fittable targets and
  Δ ≈ 0 remained; deeper lookahead does not address H1/H2/H3.
- **Td_oracleCache** — Ta(2) showed depth=1 collapses to zero, so caching
  at that depth is pointless; depth=2 ran without it.

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
6. **Structural trunk changes (S) close the unified-aux WR tax AND lift Q_select position-structure (2026-05-24 D1 update).** Sa(3) S4 matches ME(2) WR with the unified-aux substrate (+7pp over OA(1)) — interpretability champion candidate. **Pre-T D1 scan now shows Sa(3) also leads the pre-T pack on `safe_piece_recall` (0.516 vs OA ~0.36) and ρ̄ (0.195 vs ~0.05) — reversing the prior series-S "no Q_select gain" verdict, which was a legacy-Δ artefact.** Sa(1) deepConv stays flat on D1 (0.024 ρ̄ ≈ OA baseline), so its +0.170 numeric Q_select Δ from 2026-05-12 really was a metric artefact as suspected.
7. **Target noise (T-series) was half the story.** Clean minimax targets drove `loss_select` from the 0.24 R/S floor down to 0.055 — confirming the floor was target noise. But `Q_select` Δ stayed ≈0 under the match-outcome metric, and Ta(1) became a new WR champion candidate (80.3%) *via the place head*, not via a now-discriminative Q_select. The "Q_select saturation" open problem is now **under-determined between metric artefact (H1), buffer signal sparsity (H2), and multitask interference (H3)** — see `analysis/qselect_diagnostics/PLAN.md`. **[INFERENTIAL — written 2026-05-18 by an AI agent revisiting its own prior claims; verify with diagnostics before treating as load-bearing.]**
8. **"Q_select saturation" was largely a metric artefact (2026-05-19).** The D1 diagnostic on Ta(1) @ epoch 4350 shows `safe_piece_recall = 0.81` vs chance 0.11, `forcing_loss_bottom_recall = 0.95` vs chance 0.70, and Spearman ρ̄ = 0.55 — the Q_select head ranks pieces correctly. The Δ-on-match-outcome metric averaged over many functionally neutral states and hid this. H2 (sparsity) and H3 (interference) were both falsified by D2 (95% buffer nonzero) and D3 (decoupled select-only net underperforms the joint net at 8× training data scale). Full write-up: [`analysis/qselect_diagnostics/REPORT.md`](analysis/qselect_diagnostics/REPORT.md). [DIRECT for numbers; interpretation AI-drafted.]
9. **The minimax oracle is a *persistent driver*, not a removable warmup (2026-05-22, Ve series).** Ve(1) NEVER (oracle on throughout) reaches 85.2% WR and D1 `safe_piece_recall=0.82`. Disabling the oracle mid-training (Ve(2) @ 2000, Ve(3) @ 4000) caps WR at ~78–79% and decays D1 safe-piece recall to ~0.55 — well above chance 0.11 (MC holds *some* of the imprint) but ~30pp below the always-on run. Late disable ≈ early disable, indicating MC pulls Q_select toward a fixed sub-oracle equilibrium. Plan for ongoing oracle cost in production training recipes. Full write-up: [`docs/diary/series-V.md`](docs/diary/series-V.md).
10. **Longer oracle-supervised training keeps lifting *both* heads (2026-05-24, Ve(4) @10k).** Extending the Ve(1) recipe 6k → 10k epochs raises WR vs BT 85.2% → **87.2%** (peak 87.1% → **88.9%**) and D1 `safe_piece_recall` 0.821 → **0.846** (chance 0.11), `forcing_loss_bottom_recall` 0.957 → **0.968**, ρ̄ 0.578 → **0.610**. WR trend slope decelerates ~4× (Ve(1) +2.9 → Ve(4) +0.65 pp/1000ep) but stays significantly positive — no ceiling at 10k. **Ve(4) is the new overall champion**, superseding ME(2) by +13.5 pp WR.

For each of these, the originating series file in `docs/diary/` carries the full evidence.
