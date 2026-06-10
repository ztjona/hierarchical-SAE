# Series X — competence-lever screen (place-win / deeper oracle / select-margin)

First training series driven by the **loss autopsy** rather than a metric
sweep. Parent: [`Research-status.md`](../../Research-status.md); motivating
evidence: [`analysis/competence_audit/REPORT.md`](../../analysis/competence_audit/REPORT.md)
("Loss autopsy", 2026-06-04).

## Xa_levers — which lever moves WR-vs-random?

**Background.** The autopsy of champion Ve(4) found: the argmax policy loses
~4.2% to random; of those losses ~⅔ are *forced* positions the agent walks
into mid-game (~2.7% of games) and only ~⅓ are *avoidable* `Q_select` blunders
(~1.4%); separately, the place head misses **6.7%** of immediate wins. No
first-player effect. The select-margin idea (originally the headline) is
therefore capped at ~1.4 pp; place-side and planning levers may matter more.

**Constraint of record (user, 2026-06-04):** the champion must be a **pure
learned policy — no inference-time tactical search.** Every lever lands in the
weights.

**Hypothesis (per arm, vs Ve(1)@6k baseline).**

- **X(1) PLACE_WIN** — an auxiliary place-win ranking hinge raises Test-A
  (immediate-win taking) and trims the missed-win rate, and *indirectly* the
  forced rate (fewer prolonged games → fewer self-inflicted forced positions).
- **X(2) DEPTH_3** — a depth-3 select oracle teaches the agent to avoid
  *reaching* forced positions, lowering the dominant ~2.7% forced rate (the
  only lever that can touch it). Cost wildcard.
- **X(3) SEL_MARGIN** — an auxiliary select-margin ranking hinge pushes
  `safe_piece_recall`→1.0, removing the avoidable ~1.4% (and improving the D1
  interpretability substrate).

This is a **screen** (rank the levers in one overnight run), not a confirmation;
the winner gets a tuned 10k deep run afterwards.

**Design — one variable each, on the Ve(4) champion recipe.**

| Arm | `MINIMAX_SELECT_DEPTH` | `LAMBDA_PLACE_WIN` | `LAMBDA_SEL_MARGIN` | else |
|---|---|---|---|---|
| X(1) PLACE_WIN | 2 | **0.5** | 0.0 | champion |
| X(2) DEPTH_3 | **3** | 0.0 | 0.0 | champion |
| X(3) SEL_MARGIN | 2 | 0.0 | **0.5** | champion |
| baseline = Ve(1)@6k | 2 | 0.0 | 0.0 | (on disk) |

**Fixed (base `trainRL.py`):** `QuartoCNNAutoregUnifiedS4`, `unified_autoreg`,
`USE_MINIMAX_SELECT_TARGET=True`, oracle always on, `N_LAST_STATES_INIT=4`,
`ENDGAME_FRACTION=0.5`, `N_LAST_STATES_ENDGAME=2`, `REWARD_FUNCTION="final"`,
`LR=7e-4`, `TAU=0.01`, `GAMMA=0.99`, `BATCH_SIZE=32`, `EPOCHS=6000`,
`WIN_MARGIN=0.5`. λ/margin are **screen defaults** — tune on the winner.

**Code surface.**
- `QuartoRL/RL_functions.py` — `_winning_cell_mask` / `_hot_piece_mask`
  (1-ply primitives, undo via `board.board[r][c]=0` since `remove_piece` is
  storage-only); `gen_experience_unified_autoreg` gains `emit_place_win` /
  `emit_hot_mask`, emitting frozen `target_place_win` / `target_hot_piece`
  buffer keys (zeros when off — forward-only schema add); `win_margin_aux_loss`
  computes both ranking hinges (its own forward; place-win → `fc2_place`,
  select-margin → `fc2_select`, both into the trunk).
- `trainRL.py` — `LAMBDA_PLACE_WIN` / `LAMBDA_SEL_MARGIN` / `WIN_MARGIN`
  (default 0; `EMIT_*` derived), threaded into both `gen_experience` calls and
  added to the loss before `backward()`; logged + dumped to JSONL config.
- `run_trains.py` — `MULTI_PARAMS` Xa_levers block (3 arms above).

**Decision gate.** Evaluate each arm at the common checkpoint with
`analysis/competence_audit/loss_autopsy.py` (loss-rate vs random, avoidable /
forced split, missed-win rate) + inline WR-vs-BT/random + D1. Promote the lever
with the largest loss-rate reduction *without* a WR/D1 regression to a tuned
10k deep run.

**Validation done (2026-06-04, pre-launch).** Unit smoke test
(`/tmp/x_smoke.py`): masks emit correctly (win⊆legal, hot⊆available, zero on
the wrong phase); `win_margin_aux_loss` is finite, ≥0, differentiable, routes
place→`fc2_place` / select→`fc2_select`, and is an exact `0.0` no-op at λ=0.
All edited files `py_compile`-clean. Base recipe verified == Ve(4). Full
end-to-end few-epoch run + thread-scaling timing left to the operator.

## Result — 2026-06-08 (3 arms @ 6000 epochs; autopsy decision gate)

All three arms ran to 6000 epochs on the Ve(4) recipe + the S4 trunk
(`QuartoCNNAutoregUnifiedS4`). Decision gate = `loss_autopsy.py` at the common
checkpoint, vs a uniform-random opponent, **2500 games/direction (5000 total)**,
argmax agent, seed 1234. The same-epoch baseline `Ve_oracleAblation(1)@6000` was
autopsied under the identical harness (it predates the inline-D1/autopsy era).

**Autopsy + inline metrics (Δ vs Ve(1)@6k baseline in the last column).**

| Arm | loss% | avoid% | forced% | missed-win% | WR-BT | WR-rand | D1 safe/forc/ρ̄ | train loss | Δ loss% |
|---|---|---|---|---|---|---|---|---|---|
| **Ve(1) base @6k** | 5.00 | 1.54 | 3.46 | 9.32 | 85.2 | 92.0 | 0.821 / 0.957 / 0.578 | 0.056 | — |
| **X(1) PLACE_WIN** | **3.54** | 1.46 | **2.08** | **1.33** | **88.4** | **94.7** | 0.813 / 0.952 / 0.572 | **0.054** | **−1.46** |
| X(2) DEPTH_3 | 4.78 | 1.82 | 2.96 | 8.78 | 85.1 | 92.6 | 0.812 / 0.952 / 0.568 | 0.059 | −0.22 |
| X(3) SEL_MARGIN | 7.52 | 1.50 | 6.02 | 26.97 | 80.9 | 90.0 | **0.857 / 0.978 / 0.622** | 0.123 | +2.52 |

(`avoid%`/`forced%`/`missed-win%` from the autopsy; `WR-*` final from the inline
JSONL; `D1` final; baseline D1 from the 2026-05-22 follow-up. Champion anchor:
`Ve(4)@10k` autopsies at loss 4.36 / avoid 1.42 / forced 2.94 / missed-win 6.73.)

**X(1) PLACE_WIN — winner; passes the gate.** Largest loss-rate cut
(**5.00→3.54%, −1.46pp / −29% rel**) with **no WR or D1 regression** (WR +3.2/+2.7pp
vs BT/random; D1 flat within sampling noise). Mechanism is exactly the hypothesis,
on *both* paths: (a) **direct** — missed-immediate-win rate **9.32→1.33%, −8.0pp**;
the place-win hinge took ~86% of the wins the baseline was walking past. (b)
**indirect** — forced rate **3.46→2.08%, −1.38pp**; taking wins earlier shortens
games, so the agent self-inflicts fewer forced positions. Avoidable rate is flat
(−0.08pp), as expected — the place hinge does not touch the select-side blunder
class. Lowest train loss of the screen (0.054). At 6k it already edges the
`Ve(4)@10k` champion on autopsy loss-rate (3.54 vs 4.36%).

**X(2) DEPTH_3 — reject; dominated.** Loss-rate barely moves (−0.22pp, ~noise);
forced rate −0.50pp — and the whole rationale ("the *only* lever that can touch
the forced floor") is **falsified**: PLACE_WIN beat depth-3 on the forced rate
itself (2.08 vs 2.96%) at lower compute. Missed-win and avoidable both ~flat-to-
worse; WR/D1 flat. A depth-3 oracle buys nothing the cheap place hinge doesn't
buy more of.

**X(3) SEL_MARGIN — reject as a policy lever; interp-only.** Loss-rate **worse**
(+2.52pp). The select-margin hinge (into `fc2_select` + trunk) corrupted the place
head — missed-win **9.32→26.97%** (train place loss 0.136, ~3×) — inflating both
forced (+2.56pp) and total loss, and costing WR (−4.3/−2.0pp vs BT/random). The
decisive falsification: it pushed D1 `safe_piece_recall` to **0.857 (best on
record)** yet the **in-game avoidable rate did not move (1.54→1.50%)**. Better
offline safe-piece recall ≠ fewer real avoidable losses; the substrate gain did
not transfer, and it failed the no-regression clause hard.

**[Correction, 2026-06-09 — X(3)'s place corruption is a loss-SCALE artifact, not
proof the lever is bad.]** X(3) ran at `λ_sel_margin=0.5`, where the hinge loss
(~0.5) dwarfs the DQN place/select losses (~0.05–0.1) by ~5–10×. The global
`clip_grad_norm_(…, MAX_GRAD_NORM=1.0)` is direction-preserving (a scalar rescale
= a smaller LR that step), so it does *not* cause the corruption — but the
*pre-clip* gradient direction is already hinge-dominated, so AdamW steps mostly in
the hinge direction and the place TD signal is throttled. Net: the place-head
damage is a consequence of **λ being mis-scaled (aux ≫ DQN)**, observable as the
X-series grad-norm crossing the 1.0 clip threshold (`comparison_grad_norm_Xa_levers.png`).
So "select-margin is fundamentally place-harmful / interp-only" is **not
established** — only `λ=0.5` was tested. The clean, scale-balanced test of
select-side pressure is the **Ya_hotHead** screen (aux hot-piece BCE at a λ set to
balance the loss magnitudes); a balanced-λ *margin* arm is a later follow-up if the
BCE head underperforms. (X(1)/X(2) are unaffected: X(1) won despite the same
regime — scale imbalance can only understate an arm — and X(2) added no aux loss.)

**Decision.** Promote **X(1) PLACE_WIN** to a tuned 10k deep run (λ_place_win=0.5
was a screen default — optionally bracket 0.25/0.5/1.0). Both WR trends are still
strictly rising at 6k (+1.7 BT / +0.8 random pp/1000ep), so 10k should extend the
gains. This rewrites the autopsy's reducible-loss framing: the dominant reducible
chunk was **not** the select-side avoidable ~1.4% (capped, and SEL_MARGIN couldn't
even move it) but the **place-side** missed wins + the forced positions long games
manufacture — both reachable by one cheap place lever. Depth-3 and select-margin
are demoted: depth-3 dominated, select-margin retained only as an interpretability
track (best D1, but it trades WR for it).

Autopsy JSONLs: `analysis/competence_audit/results/Xa_levers(*)/loss_autopsy.jsonl`
and `.../Ve_oracleAblation(1)0519_DISABLE_NEVER/loss_autopsy.jsonl`.

## Post-screen probe — the autopsy `avoidable_rate` is opponent-diluted (2026-06-08)

Following up on "X(3) pushed `safe_piece_recall` to 0.857 but the in-game
avoidable rate didn't move," I ran the **static** `audit.py` probe (Tests A/B,
depth-1, 2000 sampled states) on X(1) and the Ve(1)@6k baseline:

| Static test | Ve(1) base @6k | X(1) PLACE_WIN |
|---|---|---|
| A — winning placement (place head) | 0.873 (n=822) | **0.975** (n=727) |
| B — losing-piece avoidance (depth-1 select) | 0.798 (n=1218) | **0.838** (n=1216) |

Two findings:

1. **X(1)'s mechanism cross-validates statically.** Test A 0.873→0.975 (+10pp)
   is the static twin of the autopsy missed-win 9.32→1.33%. Test B even rose
   +4pp despite X(1) adding *no* select lever — the place hinge helped the
   select head too (shared trunk / shorter games). X(1) is clean.

2. **`avoidable_rate` massively understates select blunders.** Test B = 0.838
   ⇒ the agent **still hands over an immediately-losing piece ~16% of the time**
   when a safe one exists — yet the autopsy `avoidable_rate` is only ~1.46% of
   games. The gap is **opponent dilution**: a *random* opponent handed a hot
   piece usually places it in the wrong cell and fails to punish, so a 16%
   depth-1 blunder rate collapses to a ~1.5% loss rate (~10×). So the earlier
   "didn't transfer" conclusion was an **artefact** — measured on a corrupted
   model (X(3)) over a shifted distribution with ~77 noisy events. The real
   select-blunder bucket is ~10× larger than the autopsy implied, just masked.

**Implications.** (a) The select-margin lever (X(3)'s depth-1 hot-piece hinge)
is *more* motivated than the diluted 1.46% suggested — but gate it on **Test B**
(punishment-independent, ~1200 events), not the noisy in-game `avoidable_rate`.
(b) The "make the agent deterministic" idea aims at the wrong actor — the agent
is *already* argmax in the autopsy; the fix is a **punishing opponent**. (c)
Legacy D1 `safe_piece_recall` is demoted to secondary; the canonical evaluation
suite + metric definitions now live in
[`analysis/competence_audit/METRICS.md`](../../analysis/competence_audit/METRICS.md).
Probe JSONLs: `analysis/competence_audit/results/{Xa_levers(1)…,Ve_oracleAblation(1)…}/audit.jsonl`.

## Punishing-opponent autopsy — the select head is the wall (2026-06-08)

Implemented `loss_autopsy.py --opponent punishing` (uniform-random except it
always takes an immediate win). Re-ran @6k, 5000 games:

| Opponent | X(1) avoid / forced / loss | Ve(1) avoid / forced / loss | X(1) missed-win |
|---|---|---|---|
| uniform | 1.46 / 2.08 / **3.54** | 1.54 / 3.46 / **5.00** | 1.33% |
| **punishing** | 12.36 / 10.10 / **22.46** | 10.74 / 11.06 / **21.80** | 1.40% |

Three takeaways:

1. **Validates the dilution thesis quantitatively.** Avoidable jumps ~7–8×
   (1.46→12.36%), landing in the same range as the Test-B hot-give rate (16%).
   The diluted uniform `avoidable_rate` was hiding the bulk of select errors.
2. **X(1)'s win evaporates vs a competent opponent.** Against `punishing`, X(1)
   (22.46%) and the Ve(1) baseline (21.80%) are **statistically tied** — X(1) is
   even marginally *worse* on avoidable (12.36 vs 10.74). The PLACE_WIN edge was
   almost entirely the place head + shorter games **vs random**; it keeps its
   place advantage (missed-win 1.40 vs 9.22%) but that barely matters once the
   opponent punishes, because **~half of all losses are now avoidable SELECT
   blunders**. The select head — untouched by the place lever — is the wall.
3. **Reframes the champion metric.** WR-vs-random rewards the place head and
   game-length and *masks* a ~12% in-play select-blunder rate. Future champion
   comparisons should report the **punishing-opponent autopsy**, not just
   WR-vs-random. Punishing JSONLs appended to the same `loss_autopsy.jsonl`s
   (`opponent: random_punishing`). **Punishing is now the `loss_autopsy.py`
   default opponent.**

## Oracle-target audit — the select wall is a FIT failure, not a target failure (2026-06-08)

New suite tool `oracle_target_audit.py` (saved under `competence_audit/`).
On 600 decisive SELECT states it compares the model's `Q_select` argmax to the
**actual training target** (`_minimax_select_target`, the same MinimaxBot the
trainer uses), depth 2:

| | X(1) PLACE_WIN | Ve(1) base |
|---|---|---|
| `oracle_separates_rate` (target ranks all hot < all safe) | **100%** | 100% |
| `oracle_blunder_rate` (target's own argmax is hot) | **0%** | 0% |
| `model_blunder_rate` (≈ 1 − Test B) | 15.5% | 20.3% |
| residual `|q−target|` correct / blunder | 0.288 / 0.353 | 0.291 / 0.347 |
| on blunders: mean target of chosen hot piece | **−0.990** | −0.990 |
| on blunders: target margin thrown away | **0.99** | 0.99 |
| on blunders: model's own q-margin for the hot pick | +0.135 | +0.147 |

**Verdict: the target is essentially perfect** (clean 100% separation, the hot
piece sits at the −1 floor with a ~0.99 margin) **and the model fails to fit the
ranking on the tail.** The residual is only ~22% larger on blunder states, so
this isn't gross misfit — `Q_select` is **too compressed near the decision
boundary** to preserve a 0.99-margin argmax, and noise flips ~16–20% of them.
Pointwise SmoothL1 supervision is not enforcing the ranking.

**Consequence for the next run.** This is the textbook case for a **ranking /
margin loss on `Q_select`** — i.e. the de-risked select-margin hinge (X(3)'s
lever, which already nudged Test B 0.798→0.838). It is *indicated by the
diagnostic*, not just plausible. **Punishing-opponent rollouts are demoted**:
the select head here is pure oracle-supervision (not TD), and the target is
already correct, so more rollout coverage attacks the wrong thing — the loss
*function* (no ranking term), not the data. Diagnostic JSONL:
`competence_audit/results/<exp>/oracle_target_audit.jsonl`.

## Capacity probe — the wall is the TRUNK representation, not head capacity (2026-06-08)

User asked whether the select wall is a *capacity* limit (the head is a single
linear layer `fc2_select` on the shared `fc1`). New suite tool
`select_capacity_probe.py` freezes the trunk, captures the exact 512-d features
the select head reads (forward-pre-hook on `fc2_select`), and trains read-out
probes of increasing depth on the **oracle target** with a proper train/val/test
split + early stopping, measuring held-out blunder rate.

| run (decisive) | deployed | linear | mlp1 | mlp2 | best MLP − deployed |
|---|---|---|---|---|---|
| X(1) seed42 (5000) | 18.6 | 21.7 | 17.4 | 17.1 | **−1.5pp** |
| X(1) seed7 (3000) | 17.3 | 25.1 | 16.9 | 16.5 | **−0.8pp** |
| Ve(1) seed42 (3000) | 20.8 | 31.9 | 21.5 | 19.9 | **−0.9pp** |

(First X(1) run reported a spurious "+27pp / CAPACITY-LIMITED" — an artefact of an
underfit linear baseline + unregularised, memorising MLPs; fixed with
val/early-stopping/weight-decay. The *linear* probe overfits 512-d features and
is regularisation-fragile, so it is reported but **not** the signal — hence the
two "INCONCLUSIVE" auto-flags. The robust comparison is **best MLP vs deployed**.)

**Verdict: REPRESENTATION-LIMITED, not capacity-limited.** Across seeds and both
models, a fresh head of *any* capacity on the frozen trunk beats the deployed
head by only **~1pp**, and nothing — linear or 75k-param MLP — pushes held-out
blunder below ~16–20%. The deployed (in-loop-trained, linear) head is already at
the floor set by what `fc1` encodes. A **deeper select-specific head is therefore
NOT the fix (~1pp ceiling).** This also unifies with the oracle-target audit:
the head "fails to fit the ranking" *because the frozen features don't separate
hot-vs-safe* on ~17% of decisive states — a representation gap, not a head gap.

**Correction to the earlier plan.** The select-margin loss must flow **through
the trunk** (small λ, balanced against place), *not* be detached into
`fc2_select` only — detaching is a head-only change, which the probe shows is
capped at ~1pp. The bottleneck is the representation, so the lever has to
*reshape* it. (X(3)'s margin-through-trunk did move Test B +4pp — consistent;
its only failure was λ=0.5 corrupting the place head.) The probe freezes the
trunk so it can only *rule out* head-only fixes; the positive case for
margin-through-trunk rests on X(3)'s Test-B gain. Alternatives if small-λ
margin-through-trunk stalls: a select branch off richer/earlier (pre-`fc1`)
features, or more trunk capacity. Probe JSONL:
`competence_audit/results/<exp>/select_capacity_probe.jsonl`.

## Safety learnability — is the TRUNK too small? No: capacity sufficient, allocation-limited (2026-06-08)

User then asked whether growing the trunk (more conv/linear layers) is the fix,
and proposed the right control: **same architecture, same input, drop the place
head** — train only the select side, so the whole trunk trains for select. New
suite tool `select_safety_learnability.py` does this with cheap depth-1 hot-mask
labels (`_piece_is_losing`, no minimax), 10k states, two arms + the live head:

| arm | held-out blunder | train | meaning |
|---|---|---|---|
| deployed (live head) | 16.7% | — | the champion's select head |
| scratch (random init, select-only) | 20.9% | 15.8% | **underfit** (didn't fit train) — from-scratch optimisation failed; uninformative |
| **champion_init (reshape the trunk, select-only)** | **6.6%** | 1.6% | reshaping the SAME trunk for select **>halves** the blunder rate |

**Verdict: CAPACITY SUFFICIENT — the wall is ALLOCATION, not trunk size.** Taking
the champion's trunk and fine-tuning it select-only (same architecture, no place
loss) drops held-out blunder **16.7% → 6.6%** — a **>10pp** gain unlocked purely
by *reshaping* the existing trunk toward select. So the architecture already has
the capacity; the deployed 16.7% is the joint head **under-allocating** it.
**Growing the trunk (more conv/linear layers) is NOT the lever.** (The `scratch`
arm's 20.9% is a from-scratch optimisation failure — train blunder 15.8% — not a
representability ceiling; ignore it. Note also `champion_init` 6.6% ≪ `scratch`
20.9%: the place co-training builds features that *transfer* usefully to select —
the place head isn't destructively stealing capacity, the select side just
doesn't exploit the shared features.)

**The three SELECT diagnostics now converge:**
1. `oracle_target_audit` — the target is perfect; the head under-ranks.
2. `select_capacity_probe` — on the *frozen* trunk a better head buys only ~1pp.
3. `select_safety_learnability` — *reshaping* the trunk for select buys >10pp.

⇒ The lever is **select pressure that flows into the trunk** to make it *use* its
capacity for ranking — **not** a deeper head (~1pp) and **not** a bigger trunk
(capacity already sufficient). Two candidates, both reshape the trunk:
(a) an **auxiliary hot-piece head** (`BCEWithLogits` on the depth-1 hot mask, à la
the QC-series legality head) — dense, well-posed; (b) the small-λ
margin-through-trunk hinge. The `scratch` failure (still 20.9%/train 15.8% at 250
epochs) is a real optimisation-hardness signal — piece-hotness is
*attribute-conjunctive* (threat-line attribute → per-piece match, piece = output
index), a bad basin from random init — so the lever wants a **strong, dense
gradient** (favours the aux BCE head over the weaker margin) on the **warm** trunk,
never from scratch. Learnability JSONL:
`competence_audit/results/<exp>/select_safety_learnability.jsonl`.

**Caveats on the 6.6% (added 2026-06-08, after review).** (i) 6.6% is **not** a
rigorous bound on the combine — it is a *demonstrated optimistic reference*; the
`champion_init` model sacrifices the place head and is **not deployable**, and the
real combine number must be **measured** (autopsy vs punishing), not bracketed.
(ii) Crucially, the select lever attacks only the **avoidable** half of the
punishing loss. X(1) vs punishing = avoidable 12.4% + **forced 10.1%**; a better
give-decision cannot touch forced positions (every available piece already hot).
So the realistic best case takes punishing loss from ~22% toward the **~10% forced
floor**, not toward zero. Beating a punisher *strongly* needs a **second lever** —
reducing forced *exposure* (mid-game planning / not walking into zugzwang), the
real content of the demoted depth-3 idea — which becomes the next wall after the
select-pressure lever lands.
