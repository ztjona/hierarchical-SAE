# Competence-audit suite — implementation plan

**Status:** plan only, no scripts yet. Successor to
[`analysis/qselect_diagnostics/`](../qselect_diagnostics/) (D1/D2/D3
returned 2026-05-19 → H1 supported / H2-H3 falsified). This plan extends
that line of work from "is the diagnostic-metric right?" to "given the
right metrics, what does each champion actually do?".

**Origin.** Triggered 2026-05-19 by an AI agent + user pair after noticing
that the games-interp model_competence_audit at
`/mnt/mydrive/SAE-research/games-interp/scripts/model_competence_audit.py`
contains five independent behavioural probes that don't depend on the
match-outcome-conditioned Δ metric. Test B is the same logic as our D1
`safe_piece_recall` (cheaper, no oracle); Tests A / C / D / E close blind
spots we have on Q_place and trunk-conditioning that the qselect suite
did not address.

---

## Why this matters

The qselect diagnostic falsified the framing of the long-standing
"Q_select saturation" problem. It did **not** address whether the
*matched* Q_place metric is also a measurement artefact, nor whether the
WR gain in Ta(1) actually routes through Q_place as we've inferred. The
project's headline numbers (Δ-on-match-outcome for Q_place, Δ-on-match
outcome for Q_select, WR vs `bot_loss-BT`) are now half-trusted; we
need an audit pass before committing more training compute.

---

## What the games-interp audit gives us

Each test is a single scalar per (model, position-set). All five run on a
**held-out** position set; none depend on the training distribution.

| Test | Quantity | Probes |
|---|---|---|
| **A** Winning placement | `argmax Q_place ∈ winning_cells` when at least one exists | Q_place — "if a win exists, does the bot take it?" |
| **B** Losing-piece avoidance | `argmax Q_select ∉ losing_pieces` when both safe and losing exist | Q_select — *same as our D1 `safe_piece_recall`, cheaper (depth-1 vs depth-2)* |
| **C** Offered-piece sensitivity | Fraction of distinct argmax cells across the 16 possible offered pieces | Trunk — does the place head actually condition on `offered`? |
| **D** Q-occupancy gap | `mean Q_place(empty) − mean Q_place(occupied)` | Trunk — has the model learned legality? |
| **E** Phase-stratified Q entropy | `H(softmax(Q_place[empty]))` bucketed by piece count | Trunk — does decisiveness grow late game? |

**Critically, the rules are exact** (immediate-win check, occupancy from
the board encoding). No minimax oracle → ~50× faster than D1. This makes
it cheap enough to run *inside* the training loop.

The audit was written for the joint schema (one Q_board head, one
Q_piece head, no phase). Porting to our unified-autoreg schema requires
the (`state_board`, `state_aux`, `phase`) input shape and routing
through `q_values_phase` / `forward` with `phase=PHASE_PLACE` or
`PHASE_SELECT` per test.

---

## Phase 1 — Tooling

### V0 — Port the audit to `unified_autoreg`

**File:** `analysis/competence_audit/audit.py` (one script, no per-test
files; the audit is short).

- Reuse `_common.load_checkpoint`, `_common.sample_states` for the
  held-out position set (no need for a separate `positions.pt` —
  self-play generates fresh positions on demand).
- Per state, produce `(state_board, state_aux, phase)` for both phases.
  Test A queries `phase=PLACE`; Test B queries `phase=SELECT`; Tests C / D / E
  query `phase=PLACE` over the place-side input shape.
- Replace the `placing_wins` rule with a vectorized version over the
  16-d Q_place output — `quartopy.Board.check_win(mode_2x2=True)` exists
  already; the rule definition is unchanged.
- Output: one JSONL record per (exp, epoch) under
  `analysis/competence_audit/results/<exp>/audit.jsonl`. Schema mirrors
  the games-interp JSON (Tests A / B / C / D / E) plus the metadata block
  the qselect suite emits (exp_name, epoch, checkpoint_path,
  architecture, config).

**Reusability.** Any `unified_autoreg` checkpoint that loads via
`_common.load_checkpoint`. No oracle, no training. Runs in **minutes**
on a 2000-position set.

### V0b — Random-init baseline

Include a control run on epoch-0 weights of each experiment (already
exists for most; `CHECKPOINTS/<exp>/*_E_0000.pt`). The games-interp
audit established that the random→trained delta on each test is
informative; we want the same.

---

## Phase 2 — Run the audit on what we have

### Va — Current champion candidates

| Run | Why |
|---|---|
| `Ta_minimaxSelect(1)0514_DEPTH_2` (latest) | new WR champion candidate; the suspect (does WR really route through place?) |
| `Ta_minimaxSelect(2)0515_DEPTH_1` | depth-1 collapse → useful control for Tests A/B |
| `Ta_minimaxSelect(3)0515_SCALAR` | scalar-target variant — does Test C separate from Ta(1)? |
| `Sa_archScan(3)0512_ARCH_S4_uniform512` | interpretability substrate of record; first non-T audit |
| `OA_unifiedAux(1)0509_N_LAST_STATES_INIT_2` | original unified-aux baseline |
| `ME_endgame(2)0429_ENDGAME_FRACTION_0.5` | current overall champion (decoupled schema — different bot/arch path; may need a separate audit branch) |

**Decision gate.** What we expect, written before running, so the audit
can refute us:

- Tests A and B should **both be high** (say ≥0.70) on all four
  unified_autoreg champions. If Test A is also at chance-baseline-ish
  the way the Δ metric reported, then Q_place is also mismeasured and
  the project's WR-driven champion ranking is on shaky ground.
- Test C should be **higher on Ta(1) than on OA(1)** if the minimax
  oracle helped piece-conditioning in the trunk (the inferred
  mechanism for Ta(1)'s WR gain).
- Test D (legality) should be high everywhere (sanity).
- Test E (entropy) should drop with piece count on all champions.

### Vb — Historical re-audit (if Va surprises us)

If Va reveals that Test A is *also* a near-zero gain over random (the
parallel to the Q_select Δ artefact), audit the M, N, O, Q, R, S
champions too. This is the "did past 'no progress' verdicts also
mismeasure?" question already queued in
[`Research-status.md`](../../Research-status.md) → forward queue item 2.
The Vb result decides whether to issue retractions to the diary entries
that called those runs "Q_select-flat".

---

## Phase 3 — Training experiments gated on the audit

These are *candidate* training series; which one (if any) we actually
run depends on what Va returns. Pre-registering them so the audit can
adjudicate cleanly.

### Vc — Live competence audit during training (tooling, not a sweep)

Add Tests A and B (cheapest, most-informative) to the per-epoch JSONL
summary produced by `trainRL.py`. Run on a fixed held-out 200-state
sample every K=50 epochs. Cost: ≈ 100 ms / call. Replaces the
match-outcome-conditioned Q_place / Q_select Δ as the headline gate.

**Deliverable:** the next champion-candidate training run ships with
`safe_piece_recall_trajectory` and `winning_placement_trajectory` in its
JSONL. This is a one-time `trainRL.py` change, default-on.

### Vd — Ta(1) 10k confirmation with live audit

The forward-queue item 3 in Research-status (10k confirmation) becomes
informative once Vc lands. Re-run Ta(1)'s recipe to 10k epochs with the
live audit; if Tests A / B keep climbing past 0.90, the run is solid and
we promote Ta(1) over ME(2). If they plateau early (despite WR rising),
that's a different story — possibly WR climbs by exploiting `bot_loss-BT`
weaknesses rather than by becoming objectively stronger.

### Ve — Mechanism ablation for Ta(1)'s WR gain *(conditional on Va)*

The qselect diagnostic implies WR gain in Ta(1) routes through Q_place,
not Q_select. If Va confirms Test A rises in Ta(1) over OA(1), this is
the clean experiment: train Ta(1)'s recipe but **disable the oracle
after epoch K=2000**. Does WR plateau immediately (oracle is a continual
regulariser), continue to rise (it was about feature shape, not
gradient), or regress (oracle holds Q_place's trunk in place)? Each
outcome implies a different next direction. Cost: one new variant under
`Te_oracleAblation` letter.

### Vf — Place-side oracle distillation *(conditional on Ve outcome)*

If Ve shows the oracle's main effect is via Q_place trunk features, the
natural follow-up is to distill a place-side oracle directly: for every
PLACE state where a winning cell exists, mask-supervise Q_place towards
the win. Skips Q_select entirely. **Only worth running if Ve says
oracle helps via trunk features**; otherwise it's a different
hypothesis.

---

## Execution order and stop conditions

1. **V0 / V0b — port audit + random baseline.** Hours. Stops only if the
   audit can't be ported cleanly (unlikely; the bot infrastructure
   already exists).
2. **Va — run on current champions.** Single afternoon. Decides whether
   Vb is needed and which (Ve, Vf, neither) candidate is on the table.
3. **Vb (conditional) — historical re-audit.** Single afternoon. Mostly
   for narrative integrity if Va shows multiple metric artefacts.
4. **Vc — wire audit into trainRL.py.** Half a day. Required infra for
   any further training-series.
5. **Vd or Ve — first training run with new gates.** Days. Outcome
   dictates what (if anything) gets a Vf.

**Cumulative stop condition.** If Va + Vb show that Tests A and B are
high on every champion we have, *and* the WR-vs-baseline ranking
matches the audit ranking, the project's "open problem" framing
collapses entirely. At that point the work is no longer "fix the broken
head" — it's interpretability per se (SAE work on the existing models)
and incremental engineering. Be ready for that outcome; it's the most
likely one given the qselect verdict.

---

## What this plan deliberately does NOT do

- Does **not** propose new architectural changes. The qselect suite
  already de-prioritised those, and the audit is meant to clean up the
  metric story, not chase a new mechanism.
- Does **not** depend on a separate `positions.pt` corpus. Self-play
  sampling from `_common.sample_states` is sufficient.
- Does **not** retire the match-outcome Δ outputs from
  `summarize_q_outcome` (yet). Vc adds the new metrics; the old ones can
  stay for backwards-compatibility through one more champion-candidate
  cycle.
- Does **not** import code from the games-interp repo as a dependency.
  The audit is short enough to re-implement; the shared logic is the
  *test definitions*, not the *code*.
