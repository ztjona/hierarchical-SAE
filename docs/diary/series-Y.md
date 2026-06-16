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

## Result — 2026-06-12 — **the select wall breaks** (gate PASS, monotonic in λ)

**Headline.** The aux hot-piece BCE head, gradient flowing **through the trunk**,
cuts the punishing-opponent **avoidable rate ~12% → 1.64%** (λ=1.0) — a ~7×
reduction — **with the place head intact** (missed-win ≈1.3–1.8% vs X(1)'s 1.40%).
Every axis improves monotonically with λ. The Q_select "saturation" wall, open
since the M-series, is resolved: it was **trunk allocation**, exactly as the
`series-X` `select_safety_learnability` probe predicted, and the lever is dense
select-pressure into the trunk.

**Decision gate — punishing-opponent autopsy** (`loss_autopsy.py`, 10k games/arm,
argmax) + **Test-B hot-give** (`audit.py`, n-states=2000) + inline D1/WR. Baselines
re-quoted from `Ve(1)@6k` / `X(1)@6k` (`competence_audit/results/`).

| arm | λ_hot | punish loss% | **avoidable%** | forced% | missed-win% | Test-A place | **Test-B hot-give** | WR vs BT | WR vs rand | D1 safe-recall | D1 ρ̄ | loss_select |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ve(1) base | — | 21.80 | 10.74 | 11.06 | — | 87.4% | 20.2% | 85.2% | 92.0% | 0.821 | 0.578 | 0.048 |
| X(1) base | — | 22.46 | 12.36 | 10.10 | 1.40 | 97.5% | 16.2% | (X1) | — | 0.857 | — | — |
| Ya(1) | 0.03 | 21.83 | 11.29 | 10.54 | 2.59 | 98.1% | 15.1% | 88.3% | 94.6% | 0.835 | 0.609 | 0.0446 |
| Ya(2) | 0.1 | 19.44 | 9.05 | 10.39 | 1.55 | 98.1% | 12.8% | 90.9% | 95.5% | 0.860 | 0.634 | 0.0329 |
| Ya(3) | 0.3 | 12.12 | **3.36** | 8.76 | 1.27 | 96.5% | **6.1%** | 93.3% | 96.7% | 0.947 | 0.718 | 0.0167 |
| Ya(4) | 1.0 | 10.36 | **1.64** | 8.72 | 1.76 | 95.6% | **4.5%** | 94.4% | 97.0% | 0.958 | 0.727 | 0.0129 |

(WR/D1/loss_select = inline JSONL final record; autopsy/Test-A/B re-run at epoch
6000 with `--architecture QuartoCNNAutoregUnifiedS4Hot`. Raw loss-cause counts at
10k games/arm: avoidable losses **1129→905→336→164**, forced **1054→1039→876→872**,
wins **7817→8056→8788→8964** — both drops are absolute, not denominator artefacts.)

**Reading.**

1. **Select wall resolved.** Avoidable (in-play oracle-blunder) rate collapses
   monotonically with λ, and the static Test-B hot-give corroborates it
   (16.2%→4.5%). The deployed `fc2_select` reads the reshaped trunk *better* than
   the `series-X` `select_safety_learnability` optimistic static ceiling (6.6%
   `champion_init`) — so **stage B (wiring `σ(hot_logits)` into `q_select`) is
   unmotivated**: `q_select` is already using the enriched trunk.
2. **No place corruption — opposite of X(3).** missed-win stays ≈1.3–1.8% and
   Test-A place accuracy stays 95.6–98.1% (X(1) 97.5%). A *mild* place tax appears
   only at λ=1.0 (Test-A 98.1%→95.6%, missed-win 1.27%→1.76%) — the first hint of
   the loss-scale regime, but nowhere near X(3)'s collapse. The difference from
   X(3): the BCE hot head is a **dense, well-posed** signal aligned with the
   trunk's need, so even at a pre-clip grad norm ~3× the clip (λ=1.0, the highest
   on record) the *direction* helps both heads. This is the clean confirmation of
   [[aux-loss-scale-vs-grad-clip]]: **clip-rate alone ≠ reject** — the
   highest-grad-norm arm has the best WR.
3. **Bonus — forced floor also moved.** Forced losses fell 1054→872 (~17%
   absolute; rate 11.06%→8.72%). The hot-encoding trunk steers placement away from
   zugzwang too, not just the immediate give. Forced is now the dominant residual
   (872 of 1036 losses = **84%** at λ=1.0) — the planning lever is the clear next
   frontier.
4. **WR side-effect, not the gate, but striking.** All four arms beat the 10k
   champion `Ve(4)` (87.2% BT) at only 6k epochs, still rising; λ=1.0 hits 94.4% BT
   / 97.0% random.

**Grad-norm / loss-scale (self-calibration readout).** Pre-clip grad norm scales
with λ (≈1.0/1.3/2.0/3.0 for λ=0.03/0.1/0.3/1.0; `comparison_grad_norm`), all above
`MAX_GRAD_NORM=1.0`. Per the design note this is the **expected loss-scale** effect,
not a clip artefact, and — unlike X(3) — it is **benign**: WR and place rise with
grad norm. `loss_final` rising with λ is just the weighted aux term inflating the
reported total; `loss_select` itself *falls* 0.045→0.013.

**Decision.** Gate **PASS** at λ≥0.1, decisively at λ∈{0.3,1.0}.
- **λ=1.0** dominates the gate (avoidable 1.64%, hot-give 4.5%, D1 0.958) and WR
  (94.4%/97.0%) at a marginal place tax.
- **λ=0.3** is the cleanest gate-phrasing hit (missed-win **1.27%** ≈ "1.3% intact",
  Test-A 96.5%) while still cutting avoidable to 3.36% / hot-give to 6.1%.

Promote to a **tuned 10k run, bracketing λ∈{0.3, 1.0}** — the place tax at λ=1.0 is
tiny at 6k but could compound over 10k under heavy aux pressure, so confirm Test-A
holds at the longer length before crowning. Do **not** push λ>1.0 without watching
Test-A (the X(3) regime lies beyond). The select-margin-through-trunk hinge
(forward-queue alternative) is **not needed** — the BCE head won outright.

## Yb_hotChamp — 10k promotion — 2026-06-16 — **NEW CHAMPION (λ=1.0)**

**Design.** Promote the Ya winners to 10000 epochs: λ∈{0.3, 1.0} (the bracket) +
a **λ=2.0** push-past arm + a **λ=1.0 seedB** reproducibility check. Same X(1)
recipe (`S4Hot`, depth-2 minimax oracle always on, `N_LAST_STATES_INIT=4`,
`LAMBDA_PLACE_WIN=0.5`). Gate = same as Ya — punishing-opponent autopsy
(`loss_autopsy.py`, **10k games/direction**) + `audit.py` Test-A/B, run with
`--architecture QuartoCNNAutoregUnifiedS4Hot`.

| arm | λ_hot | punish loss% | **avoidable%** | forced% | missed-win% | Test-A | **hot-give** (1−B) | WR vs BT (f/peak) | WR vs rand | loss_select | D1 safe | D1 ρ̄ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ve(4) champ@10k | — | 21.80 | 10.74 | 11.06 | — | 87.4% | 20.2% | 87.2 / 88.9 | 93.8 | 0.041 | 0.846 | 0.610 |
| Ya λ=1.0 @6k | 1.0 | 10.36 | 1.64 | 8.72 | 1.76 | 95.6% | 4.5% | 94.4 | 97.0 | 0.0129 | 0.958 | 0.727 |
| Yb λ=0.3 | 0.3 | 12.02 | 1.66 | 10.36 | **0.98** | **97.5%** | 4.1% | 94.5 / 95.6 | 97.1 | 0.0152 | 0.973 | 0.754 |
| **Yb λ=1.0** | 1.0 | 9.99 | **0.99** | 9.00 | 1.58 | 96.5% | 3.4% | **95.1 / 96.2** | 97.4 | 0.0105 | 0.962 | 0.745 |
| Yb λ=1.0 seedB | 1.0 | 9.14 | **0.95** | 8.19 | 1.45 | 97.7% | **2.0%** | **95.1 / 96.2** | 97.5 | 0.0108 | 0.976 | 0.750 |
| Yb λ=2.0 | 2.0 | 9.44 | 1.43 | 8.01 | **3.93** | **93.1%** | 3.3% | 94.3 / 95.3 | 97.1 | 0.0132 | 0.967 | 0.748 |

(WR/loss_select/D1 = inline JSONL final record; autopsy/Test-A/B re-run at epoch
10000. Avoidable rate = "of games", un-diluted punishing opponent. Raw counts at
20000 games/arm, λ=1.0: avoidable 198, forced 1800, wins 17999.)

**Reading.**

1. **Gate PASS — the select fix held and improved at 10k.** λ=1.0's punishing
   avoidable rate fell 1.64% (6k) → **0.99%** (10k), hot-give 4.5% → 3.4%; against
   the Ve baseline (10.74% / 20.2%) that's a **~11× reduction** in the in-play
   select-blunder rate. The wall stays broken at the longer length.
2. **The λ=1.0 place tax did NOT compound — it recovered.** This was *the* open
   crowning question. At 6k λ=1.0 showed a mild tax (Test-A 95.6%, missed-win
   1.76%); at 10k it is **Test-A 96.5%, missed-win 1.58%** — both *better* than at
   6k. The extra 4k epochs let the place head reconverge under the aux pressure.
3. **λ=1.0 is reproducible.** seedB matches or slightly beats seed-A on every gate
   metric (avoidable 0.95 vs 0.99, Test-A 0.977 vs 0.965, hot-give 2.0 vs 3.4, WR
   identical 95.1%). Not a lucky seed.
4. **λ=2.0 rejected — confirms the ">1.0 watch Test-A" warning.** The only arm with
   real place corruption (Test-A 93.1%, missed-win 3.93% ≈ 2.5× the λ=1.0 arms),
   with no select or WR benefit — the X(3) loss-scale regime starting to bite. The
   bracket {0.3, 1.0} was correct.
5. **λ=0.3 cleanest place but dominated on the gate.** Best Test-A (97.5%) /
   missed-win (0.98%), but pays ~0.7pp avoidable and ~0.6pp WR; at 10k λ=1.0 wins
   the select gate outright. Keep λ=0.3 as the place-purity reference.
6. **Forced floor barely moved — the new wall.** Forced rate 11.06% (Ve) → **9.0%**
   (Yb λ=1.0): the avoidable half is essentially solved (now ~10% of losses), so
   **forced losses are ~90% of the residual** (1800 of 1998 at λ=1.0). The hot
   trunk's side-effect on forced exposure (Ya's −17%) does **not** continue to scale
   with λ at 10k — λ does not buy forced-floor reduction. This needs a **dedicated
   strategy**, not more select pressure (see Research-status open problem).

**Decision.** **Crown `Yb_hotChamp(3)0612_HOT_1.0_seedB` as the new overall
champion** (2026-06-16): 95.1% WR vs BT (+7.9pp over Ve(4)), avoidable 0.95%, place
intact (Test-A 97.7%). Chosen over seed A (`Yb(2)`, avoidable 0.99% / Test-A 96.5% /
hot-give 3.4% / D1 0.962) which it dominates on nearly every gate axis at WR tied —
the two seeds bracket *expected* performance; seedB is the deployed artifact. λ=2.0
rejected (place corruption). The select lever is now
**exhausted** — the next frontier is the **forced-exposure / mid-game planning
floor**, which neither oracle-select-distillation nor the hot trunk meaningfully
touches.

### Hot-head-at-inference diagnostic — 2026-06-16 — **Stage B empirically closed**

Probe: could the aux hot head *replace* `q_select` at the give, and does the gap
reveal whether `fc2_select` under-reads the trunk? Added `audit.py --use-hot-head`
(scores Test B with `argmin(hot_logits)` alongside the deployed `argmax(q_select)`
and reports their agreement). Run on the three λ∈{0.3,1.0,1.0-seedB} 10k checkpoints:

| arm | Test-B `argmax(q_select)` | Test-B `argmin(hot)` | select↔hot agreement |
|---|---|---|---|
| Yb(3) seedB (champ) | 0.980 | 0.978 | 73.6% |
| Yb(2) seed A | 0.966 | 0.966 | 73.4% |
| Yb(1) λ=0.3 | 0.959 | 0.965 | 66.4% |

**Reading.**
1. **The hot head ties the select head on Test-B** (differences are a handful of
   states). So `q_select` is **not under-reading** the trunk's safety encoding — a
   direct linear hot probe does no better. **Replacing select with hot would not
   help, and Stage-B wiring (`σ(hot)`→`q_select`) / a hot-veto is confirmed
   unmotivated — now by measurement, not assertion.**
2. **The ~26–34% disagreement is the signal:** the heads reach the *same* Test-B
   accuracy while choosing a *different* give a quarter of the time ⇒ they diverge
   almost entirely among the **safe** pieces, where depth-2 `q_select` adds ranking
   structure the depth-1 hot head lacks. Positive evidence that select carries
   information **beyond** hotness (the depth-2 superset), not a noisy copy of it.
3. **Agreement rises monotonically with λ_hot** (66%→74%): more hot-pressure ⇒ more
   hotness-shaped trunk ⇒ `q_select` aligns more with the probe. Clean sanity check.

**Caveat (motivates the next diagnostic).** Test-B scores only the *binary*
immediate-loser avoidance; it is blind to choice quality *among safe pieces* —
exactly where depth-2 lookahead and forced-exposure avoidance live. So "tied on
Test-B, 26% disagreement" is consistent with select being strictly better in ways
Test-B can't see. A **depth-2 Test-B** (avoid a give that is safe now but forces a
loss next ply) is the natural probe for the forced-floor work.
