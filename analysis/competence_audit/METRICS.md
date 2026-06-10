# Competence-evaluation suite — metric definitions (canonical)

**Status:** authoritative as of 2026-06-08. This is the **single evaluation
suite** going forward. The older Q_select diagnostic suite
(`analysis/qselect_diagnostics/`, "D1/D2/D3") is **legacy** — its code and
results are retained for provenance but are no longer the gate (see
[Legacy: D1/D2/D3](#legacy-d1d2d3) below).

Two complementary instruments, one rule set:

| Instrument | File | What it measures | State source |
|---|---|---|---|
| **Static audit** (Tests A–E) | `audit.py` | per-position competence of the heads, in isolation | sampled SELECT/PLACE states |
| **Loss autopsy** | `loss_autopsy.py` | game-outcome competence (why losses happen) | full games vs an opponent |
| **Oracle-target audit** | `oracle_target_audit.py` | is a SELECT failure a *target* failure or a *fit* failure? compares `Q_select` argmax to the actual `_minimax_select_target` | sampled decisive SELECT states |
| **Capacity probe** | `select_capacity_probe.py` | head-capacity vs trunk-representation limit: trains frozen-feature read-out probes of increasing depth vs the deployed head | sampled decisive SELECT states |
| **Safety learnability** | `select_safety_learnability.py` | trunk *capacity* vs *allocation*: trains the same arch with the place head removed (scratch + champion-init) on depth-1 hot labels | sampled SELECT states |

Both import the **same depth-1 rule helpers** from `audit.py`, so there is no
rule drift between them. All runs use `mode_2x2=True` and the agent plays
**deterministic argmax** unless `--stochastic` is passed.

---

## Shared rule helpers (depth-1, `mode_2x2=True`)

These define "winning"/"losing"/"safe" everywhere in the suite. **All are
depth-1** (single-ply) — they ask only about the *immediate* next placement.

- `_placing_wins(board, piece, r, c)` — placing `piece` at `(r,c)` completes a
  line *now*.
- `_winning_cells(board, piece, empties)` — cells where `piece` wins now.
- `_piece_is_losing(board, piece, empties)` — `piece` is **losing to give** iff
  it has *any* immediate winning placement (the opponent could win next ply).
  A piece is **safe** iff it is not losing-to-give.

> Depth note. "Safe" here is **depth-1**. The legacy D1 `safe_piece_recall`
> used a **depth-2 minimax** oracle — a *different, stronger* notion. Do not
> equate them.

---

## Static audit — `audit.py` (Tests A–E)

`python analysis/competence_audit/audit.py --exp '<name>' --epoch N \
  --architecture QuartoCNNAutoregUnifiedS4 --n-states 2000`
→ one JSONL record per (exp, epoch) at `results/<exp>/audit.jsonl`.
Cost: minutes on a 2 000-state sample. `unified_autoreg` (32-d aux) only.

| Test | Key | Definition | Reading |
|---|---|---|---|
| **A** Winning placement | `test_A_winning_placement.accuracy` | Over PLACE states where the offered piece has ≥1 winning cell: does `argmax Q_place[legal]` land on a winning cell? | **place-head win-taking.** 1.0 = always takes the immediate win. |
| **B** Losing-piece avoidance | `test_B_losing_piece_avoidance.accuracy` | Over SELECT states where **both** safe and losing pieces exist in storage: does `argmax Q_select[available]` avoid every losing piece? | **select-head depth-1 competence.** `1 − accuracy` = **hot-give rate** (the headline select gate; see below). |
| **C** Offered-piece sensitivity | `test_C_piece_sensitivity` | Fraction of *distinct* argmax-place cells across all offered-piece options. | Does the place head condition on the offered piece at all? |
| **D** Q-occupancy gap | `test_D_q_occupancy_gap` (mean, frac>0) | `mean Q_place(empty) − mean Q_place(occupied)`. | `>0` ⇒ learned legality. Often negative — see D′. |
| **D′** Counterfactual occupancy gap | `test_Dprime_counterfactual_occupancy` | Re-forward with occupied piece-channels zeroed; re-measure the gap at formerly-occupied cells. | `gap_cleared ≈ 0` ⇒ D's negative gap is an **input-driven representation artefact** (ignorable at interp time); `≈ gap_orig` ⇒ genuine head/position bias. |
| **E** Phase-stratified entropy | `test_E_phase_entropy.by_phase` | Entropy of `softmax(Q_place[legal])`, bucketed by piece count. | Lower late-game entropy ⇒ more decisive endgame. |

### The headline select gate: **hot-give rate** = `1 − test_B_accuracy`

Test B is **depth-1 and punishment-independent** — it reads the select head's
true blunder rate directly, without depending on whether an opponent happens to
convert the blunder. This is the select-side gate of record. (As of 2026-06-08
the champions sit at hot-give rate ≈ 16–20%, far above what the diluted in-game
`avoidable_rate` implied — see the caveat under the autopsy.)

---

## Loss autopsy — `loss_autopsy.py`

`python analysis/competence_audit/loss_autopsy.py --exp '<name>' --epoch N \
  --architecture QuartoCNNAutoregUnifiedS4 --opponent uniform --n-games 2500`
→ one JSONL record per (exp, epoch, opponent) at `results/<exp>/loss_autopsy.jsonl`.
`--n-games` is **per direction** (agent as P1, then P2); 2500 ⇒ 5000 games.

Opponent modes:
- **`uniform`** (`random_bot`) — uniform-random placement and selection. The
  historical default; its `avoidable_rate` is opponent-diluted (see caveat).
- **`punishing`** — uniform-random **except it always takes an immediate win**
  (places a handed winning piece in the winning cell). The fatal-give invariant
  still holds (it can only win with a piece the agent gave it), so the
  classification is unchanged, but `avoidable_rate` is now **un-diluted** and
  tracks the agent's true depth-1 select-blunder rate. Use this for select
  competence in play.
- **`benchmark`** — the `champion_config` epoch-0 CNN "Random Baseline" (for
  comparability with `champion-results.jsonl`).

**Why every loss-vs-random is a SELECT event.** You never place the opponent's
pieces; a random opponent can only win by *placing a piece the agent handed it*
into a completing line. So the fatal give is **always** a hot give, and the
only question is whether a safe alternative existed at that give.

| Key | Definition |
|---|---|
| `loss_rate` / `win_rate` | losses / wins over all games. |
| `losses_avoidable` / `avoidable_rate` | fatal give was hot **and a safe piece existed** ; rate = `/games`. |
| `losses_forced` / `forced_rate` | fatal give was hot and **every** available piece was losing (irreducible) ; rate = `/games`. |
| `losses_anomalous` | lost without a hot fatal give (should be ~0). |
| `avoidable_fraction_of_losses` | `avoidable / losses`. |
| `n_safe_at_blunder_mean` | mean # safe pieces available at an avoidable blunder. |
| `n_placed_at_loss_mean` / `_hist` | pieces on board at the fatal give. |
| `place_audit.missed_win_rate` | live **Test A**: `missed_wins / win_opportunities` over the played games. |
| `by_direction.agent_p1/p2` | the same split per starting side (no first-player effect observed). |

### ⚠ Caveat — `avoidable_rate` is **opponent-diluted** (2026-06-08 finding)

The autopsy docstring historically called `avoidable_rate` "the residual
`1 − safe_piece_recall`." **That is wrong.** A *random* opponent handed an
immediately-winning piece usually places it in the wrong cell and **fails to
punish**, so the agent's true depth-1 blunder rate (Test B hot-give rate ≈ 16%)
collapses to an in-game `avoidable_rate` ≈ 1.5% — a **~10× dilution**.

Consequences:
- The fix is the **`--opponent punishing`** mode (above), not the default. Run
  vs `uniform` for comparability with historical numbers and WR-vs-random; run
  vs `punishing` for the true in-play select-blunder rate. Measured 2026-06-08
  @6k: avoidable_rate **1.46% → 12.36%** (X(1)) and **1.54% → 10.74%** (Ve(1))
  going uniform → punishing — a ~7–8× de-dilution that lands in the same range
  as the Test-B hot-give rate, confirming both instruments agree.
- At ~1.5% vs uniform (~75 events / 5000 games), `avoidable_rate` is also **too
  underpowered** to resolve sub-0.2pp changes; `punishing` (~600 events) is not.
- `forced_rate` vs `uniform` is a cleaner in-game signal than `avoidable_rate`
  there (it reflects reaching unavoidable positions, which the agent controls).

---

## Oracle-target audit — `oracle_target_audit.py`

`python analysis/competence_audit/oracle_target_audit.py --exp '<name>' --epoch N \
  --architecture QuartoCNNAutoregUnifiedS4 [--n-states 1500] [--max-oracle 600] [--depth 2]`
→ `results/<exp>/oracle_target_audit.jsonl`.

Diagnostic, not a gate: when the select head blunders, decide whether the
**training target** is wrong or the **model fails to fit** a correct target. On
decisive SELECT states it compares `argmax Q_select` to the actual training
target `_minimax_select_target` (the same MinimaxBot the trainer uses).

| Key | Reading |
|---|---|
| `oracle_separates_rate` | target ranks **all** hot pieces below **all** safe — ≈1.0 ⇒ target is correct. |
| `oracle_blunder_rate` | the target's own argmax is a hot give — ≈0 ⇒ target is correct. |
| `model_blunder_rate` | ≈ `1 − Test_B` (sanity cross-check). |
| `resid_mean_on_{correct,blunder}` | `|Q_select − target|`; concentration on blunders ⇒ tail-fit problem. |
| `on_blunder.mean_target_of_chosen_hot` / `mean_target_margin_lost` | how bad the chosen piece is *per the target* (≈ −1 / ≈ 1 ⇒ model inverts a near-maximal margin). |

2026-06-08 finding (X(1) & Ve(1) @6k): `oracle_separates_rate = 100%`,
`oracle_blunder_rate = 0%`, blunders throw away a ~0.99 target margin ⇒ **fit
failure, not target failure** → the indicated fix is a **ranking/margin loss on
`Q_select`**, not more rollout coverage.

---

## Capacity probe — `select_capacity_probe.py`

`python analysis/competence_audit/select_capacity_probe.py --exp '<name>' --epoch N \
  --architecture QuartoCNNAutoregUnifiedS4 [--n-states 9000] [--max-oracle 5000]`
→ `results/<exp>/select_capacity_probe.jsonl`.

Diagnostic, not a gate: is the SELECT failure a **head-capacity** limit or a
**trunk-representation** limit? Freezes the trunk, captures the exact 512-d
features `fc2_select` reads (forward-pre-hook), and trains read-out probes
(`linear` / `mlp1` / `mlp2`) on the oracle target with a train/val/test split +
early stopping. Compares held-out blunder rate to the live `deployed` head.

| Key | Reading |
|---|---|
| `probes.{deployed,linear,mlp1,mlp2}.test_blunder` | held-out blunder rate per head. **`deployed` is the robust reference.** |
| `head_headroom` (= deployed − best MLP) | how much a better head on the *same* features buys. ≥5pp ⇒ capacity-limited; ≈0 ⇒ representation-limited. |
| `linear_unreliable` | the linear probe overfits 512-d features (regularisation-fragile); reported, not used for the verdict. |

2026-06-08 finding (X(1)±seed, Ve(1) @6k): best MLP beats the deployed head by
only **~1pp**, floor ~16–20% regardless of capacity ⇒ **REPRESENTATION-limited**.
A deeper select head won't help (~1pp); the fix must reshape the **trunk** (e.g.
a small select-margin loss flowing *through* the trunk), not the head.

---

## Safety learnability — `select_safety_learnability.py`

`python analysis/competence_audit/select_safety_learnability.py --exp '<name>' --epoch N \
  --architecture QuartoCNNAutoregUnifiedS4 [--n-states 10000] [--epochs 250]`
→ `results/<exp>/select_safety_learnability.jsonl`.

Diagnostic, not a gate: is the trunk **too small** (capacity) or just
**mis-allocated**? Trains the **same architecture with the place head removed**
(only `fc2_select` gets gradient → the trunk trains purely for select) on cheap
**depth-1 hot-mask** labels (`_piece_is_losing`, no minimax). Two arms vs the
live head:

| Key | Reading |
|---|---|
| `probes.deployed.test_blunder` | the live champion head (reference). |
| `probes.scratch` | random-init, select-only. Prone to from-scratch **optimisation failure** — check `train_blunder`; if it didn't fit train, ignore its test (flagged by `scratch_underfit`). |
| `probes.champion_init` | champion trunk reshaped select-only — the **decisive capacity signal**. |

2026-06-08 finding (X(1) @6k): `champion_init` cuts held-out blunder
**16.7%→6.6%** (>10pp) reshaping the *same* trunk ⇒ **capacity is sufficient, the
wall is allocation** ⇒ pressure the trunk (aux hot-piece head / margin),
**don't grow it**. (`scratch` underfit at 20.9% — optimisation, not a ceiling.)

---

## Legacy: D1/D2/D3

The **Q_select diagnostic suite** (`analysis/qselect_diagnostics/`), created
2026-05-18 to test whether "Q_select saturation" was real or a metric artefact:

| | Script | Hypothesis | Verdict (2026-05-19) |
|---|---|---|---|
| **D1** | `position_structure.py` | H1 — metric artefact | **supported** (the head *does* encode position structure) |
| **D2** | `buffer_signal_density.py` | H2 — signal sparsity | falsified |
| **D3** | `decoupled_select.py` | H3 — multitask interference | falsified |

"D" = Diagnostic; the digit indexes the hypothesis. D1's three recalls
(`safe_piece_recall` [depth-2], `forcing_loss_bottom_recall`, `spearman_rho`)
were promoted into the inline checkpoint JSONL via `COMPUTE_D1_INLINE` (see
`CLAUDE.md`) and served as the Q_select gate 2026-05-24 → 2026-06-08.

**Why now secondary, not the gate.** `safe_piece_recall` is a **depth-2 offline
mean**: (i) it can *rise while in-play behavior collapses* (Xa X(3): recall hit
0.857 best-on-record while WR fell and the place head was corrupted), (ii) it
hides the loss-producing tail, and (iii) its supposed in-game counterpart
(`avoidable_rate`) is opponent-diluted ~10×, so the "residual `1−recall`"
identity never held. The `d1_*` keys **continue to be emitted** (code unchanged)
but are **secondary/diagnostic**; the gate is the competence suite above. See
[[d1-jsonl-contract]].
