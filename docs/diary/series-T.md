# Series T — Minimax-oracle distillation on the SELECT head

OA-family substrate with the Sa(3) `QuartoCNNAutoregUnifiedS4` trunk. Q_place
continues to learn from `td_place_mc_select` Bellman bootstrap; Q_select is
**supervised** by per-piece minimax-oracle scores captured at the moment
each transition is generated. Targets are frozen in the buffer — the policy
of that moment — and never recomputed on replay. See parent:
`Research-status.md` and design note
[`docs/diary/2026-05-14_qselect-target-rethink.md`](2026-05-14_qselect-target-rethink.md).

## Ta_minimaxSelect(1) — depth-2 oracle, Sa(3) substrate

**Hypothesis.** `loss_select` floors near 0.24 across R/S because the MC
return target on SELECT actions is structurally noisy: `Q_select=−1`
correctly labels piece-gives that *caused* a loss, but `Q_select=+1` labels
selects up to 6+ moves before the win — credit buried under many
subsequent decisions. Swapping the noisy MC label for `MinimaxBot.depth=2`
per-piece scores should:

- drive `loss_select` substantially below the 0.24 floor (clean targets
  have a clean Bayes optimum),
- lift `Q_select` winners-minus-losers Δ above the +0.40 long-standing
  gate, and
- leave `Q_place` and WR alone (place head untouched).

**Code surface.**

- `bot/minimax_bot.py` — defensive patch around `check_win()[0]`: the
  installed `quartopy.Board.check_win` returns `bool`, the file's comment
  expects `tuple[bool, coords]`. Three sites patched to tolerate both.
- `QuartoRL/RL_functions.py` —
  - new constant `DECOUPLED_TARGET_TD_PLACE_MINIMAX_SELECT = "td_place_minimax_select"`,
  - new helper `_minimax_select_target(oracle, board_serial, available_pieces, mode_2x2)` returning per-piece `(target, mask)` vectors. Scores are normalised by `100 + oracle.depth` and **negated** so higher target = better SELECT (the raw minimax convention is lower=better for the selector — see `best_action_set`),
  - `gen_experience_unified_autoreg` (and the dispatcher) take a new `select_oracle=` kwarg; when provided, each SELECT row gets `(target_sel_minimax, target_sel_minimax_mask)` captured before the available-piece set is mutated. PLACE rows get zero-filled placeholders so the TensorDict schema is uniform.
  - `DQN_training_step_decoupled_autoreg` accepts the new target style and **returns a 5-tuple** `(q_place, target_place, q_select_full, target_select_full, sel_loss_mask)` on the SELECT side. `q_select_full` is the full 16-d head output (not scalar-at-chosen-action like other styles).
- `trainRL.py` — new flag block:
  ```python
  USE_MINIMAX_SELECT_TARGET = False
  MINIMAX_SELECT_DEPTH = 2
  ```
  When True, overrides `DECOUPLED_TARGET_STYLE` and instantiates `SELECT_ORACLE = MinimaxBot(depth=MINIMAX_SELECT_DEPTH)`. The loss-composition branch detects the 5-tuple return and applies a **masked SmoothL1** on the full Q_select vector: `((l1_elt * sel_mask).sum() / sel_mask.sum())`. Backward-compatible: with the flag off, every existing training script behaves identically (the new TensorDict fields are present but contain zeros, and the 4-tuple return path is unchanged).

**Fixed:** `STARTING_NET=None`, `ARCHITECTURE=QuartoCNNAutoregUnifiedS4`
(the Sa(3) substrate — pending its 10k confirmation but already the
interp-champion candidate), `TRANSITION_SCHEMA="unified_autoreg"`,
`DECOUPLED_TARGET_STYLE="td_place_minimax_select"` (set by the flag),
`USE_MINIMAX_SELECT_TARGET=True`, `MINIMAX_SELECT_DEPTH=2`,
`REWARD_FUNCTION="final"`, `N_LAST_STATES_INIT = N_LAST_STATES_FINAL = 4`,
`ENDGAME_FRACTION=0.5`, `N_LAST_STATES_ENDGAME=2`, `LR=7e-4`, `TAU=0.01`,
`(α_place, α_select) = (1.0, 1.0)`, `EPOCHS=5000`.

| Run | Oracle depth | Target shape | Substrate | Epochs |
|-----|--------------|--------------|-----------|--------|
| Ta(1) | 2 | full 16-d (masked) | `QuartoCNNAutoregUnifiedS4` (Sa(3)) | 5000 |
| Ta(2) | 1 | full 16-d (masked) | `QuartoCNNAutoregUnifiedS4` (Sa(3)) | 5000 |
| Ta(3) | 2 | scalar at chosen piece | `QuartoCNNAutoregUnifiedS4` (Sa(3)) | 5000 |

The 3-pack is launched together (cost is dominated by the oracle, comparable
across variants) to triangulate the mechanism in one shot rather than serially:

- **Ta(1) vs Ta(2)** — depth ladder. If WR / Q_select Δ match, the cache (Td)
  becomes much more attractive and we learn the useful signal lives only in
  immediate-force positions. If Ta(1) dominates, depth=2 lookahead is doing
  real work and Tb_depth4 is justified.
- **Ta(1) vs Ta(3)** — signal-density ablation (= Tc). Same depth, same
  oracle, only difference is whether the head sees per-piece supervision on
  all 16 pieces (Ta(1)) or only on the chosen piece (Ta(3), like every other
  scalar target style). Isolates "oracle replaces MC noise" from "oracle
  gives 16× more signal per state". Without Ta(3), a positive Ta(1) is
  ambiguous between the two mechanisms.

Ta(3) requires a new target-style constant
`DECOUPLED_TARGET_TD_PLACE_MINIMAX_SELECT_SCALAR` in `QuartoRL/RL_functions.py`
that reuses the existing `target_sel_minimax` field but gathers the chosen
piece's score, returning the standard scalar 4-tuple. Train scripts gate on
a new `MINIMAX_SELECT_SCALAR_TARGET` flag.

**Compare to:** `Sa_archScan(3)0512_ARCH_S4_uniform512` (substrate, no
oracle), `ME_endgame(2)_E_5000` (current overall champion),
`OA_unifiedAux(1)` (prior interpretability champion), `bot_random`.

**Decision gate (pre-registered).**
- `loss_select` final ≤ 0.10 (substantially below the 0.24 floor R/S
  bounded). Necessary condition: the new targets are fittable.
- `Q_select` winners-minus-losers Δ ≥ +0.40 (the open-problem gate).
- WR vs `bot_loss-BT` ≥ 70% (non-regression on place-driven WR; Sa(3)
  reached 73.5%/peak 75.2%, with a still-rising trend at 5000 epochs).

Combinations:
- All three pass → **the open problem is target noise**, not
  representation or optimisation. Q_place RL + Q_select minimax-distillation
  becomes the new recipe; sweep depth (Tb), substrate (Sa(3) confirmation
  vs OA(1)), and ablate to scalar-target (Tc) for diagnostics.
- `loss_select` drops cleanly but Δ stays small → trunk *cannot* represent
  the piece × board interaction even with clean supervision; the failure is
  representational after all and we re-prioritise Sa(1) / Sb hybrid.
- WR collapses but `loss_select` drops → supervised + RL co-training has a
  balance problem; fix is α-scheduling (Rb) on the select term, not
  rejecting the idea.
- `loss_select` does **not** drop → the oracle's targets are themselves
  too noisy at depth=2 (mostly 0 in early game), or the trunk representation
  cannot route them to the Q_select head. Move depth up (Tb_depth4) before
  abandoning.

**Cost.** Oracle adds ≈ 7 s/epoch at the main N=4 buffer (32 matches/epoch,
depth=2); with the N=2 endgame anchor buffer also generating select rows,
total oracle overhead is roughly 10–20 hours on a 5000-epoch run. Mitigations
ready if it becomes painful:
- **Cache** by `(board_serial, frozenset(available_pieces), mode_2x2)` —
  many states recur across self-play matches, especially early in training.
- **Drop to depth=1** — much faster, validated to ~98% WR in
  `projects/supervised-cloning` at depth=2 so depth=1 should still be a
  meaningful teacher signal.
- **Throttling** — only oracle on a configurable fraction of SELECT rows;
  others retain MC return. Adds another knob; not in v1.

**Risks / open questions.**
- *Depth-2 minimax in early Quarto is mostly 0.* Most non-forced positions
  return neutral scores at depth=2. The select head will see near-zero
  targets for most early-game decisions, which is informative as "no
  forced win/loss" but doesn't grade between non-forcing offers. If the
  loss floor drops but Δ stays small, this is the most likely diagnosis.
- *Replay-buffer staleness.* Targets are frozen at the policy of the moment
  the state was visited, not the current policy. With `NUM_EPOCHs_BUFFER=8`
  the same transition is sampled ~8 times across epochs. This is the same
  staleness pattern as Q_place's TD target — accepted by convention; flagged
  here because the oracle target is *exact* and so doesn't auto-correct
  the way bootstrapped targets do.
- *Player-perspective and turn semantics.* The reconstruction helper sets
  `game.pick=True` and reuses placeholder `Quarto_bot()`s for the
  `QuartoGame` constructor. `score_all_moves` reads only board, storage,
  pick, and mode_2x2 — the placeholder bots are not invoked. The minimax
  recursion is symmetric so `game.turn` does not affect scoring.

**Next variants (status updated 2026-05-18 after Ta(1)/(2)/(3) results):**

- **Tb_depth4 — DROPPED.** Ta(1) showed depth=2 already produces fittable
  targets (`loss_select`=0.055). Deeper lookahead does not address the
  observed failure mode (Q_select Δ stays ≈0 even with cleanly-fitted clean
  targets). Prior project evidence had depth=4 ~4× slower with no WR gain.
- **Td_oracleCache — DROPPED.** Ta(2) showed depth=1 targets collapse to
  zero almost everywhere (`loss_select`=5×10⁻⁶), so caching at that depth is
  pointless. Depth=2 ran within budget without a cache.

## Result — 2026-05-18

> **[AI-REASONED PROVISIONAL ANALYSIS]** Numbers in the table below are
> direct measurements from the JSONL summaries and are reliable.
> Interpretation paragraphs are inferential; they may be biased and were
> specifically flagged for re-examination because previous series
> interpretations (also AI-generated) may have been over-confident. **Future
> readers — including future AI agents: if you find yourself agreeing with
> a provisional claim without independent evidence, that is the failure
> mode this caveat exists for.** Verify via the diagnostic suite
> [`analysis/qselect_diagnostics/PLAN.md`](../../analysis/qselect_diagnostics/PLAN.md)
> before acting on the inferred conclusions.

### Numbers [DIRECT — from JSONL summaries]

| Run | Epochs | `loss_select` | Q_select Δ | Q_place Δ | WR vs BT (final / peak) | WR vs random (final / peak) | WR trend vs BT |
|---|---|---|---|---|---|---|---|
| Ta(1) DEPTH=2, 16-d masked | 4000 (truncated) | **0.055** | +0.019 | +0.514 | **80.3% / 81.5%** | 90.2% / 90.9% | +3.6↑ |
| Ta(2) DEPTH=1, 16-d masked | 5000 | **5×10⁻⁶** | +3×10⁻⁵ | +0.335 | 71.5% / 73.2% | 85.0% / 86.7% | +2.2↑ |
| Ta(3) DEPTH=2, scalar at chosen | 4000 (truncated) | 0.044 | +0.002 | +0.591 | 74.9% / 77.3% | 87.0% / 88.2% | +2.9↑ |

Pre-registered gate outcomes (gates defined above at lines 82–87):

- **`loss_select` ≤ 0.10** — PASS on all three. The 0.24 floor across R/S
  was a target-noise artefact, not a Bayes limit on a representable function.
- **Q_select Δ ≥ +0.40** — FAIL on all three; Δ ≈ 0 everywhere.
- **WR vs `bot_loss-BT` ≥ 70%** — PASS on all three. Ta(1) at 80.3%
  surpasses ME(2) (73.7%) and Sa(3) (73.5%).

Ta(1) and Ta(3) were truncated short of the planned 5000 epochs; WR trends
were still rising at termination. Their WR figures are therefore conservative.

### Provisional interpretation [INFERENTIAL — read sceptically]

The pre-registered decision tree (lines 82–103) routes this outcome to
"trunk cannot represent piece × board interaction → Sa(1)/Sb hybrid." On
re-reading after the results, that inference is **too quick**. Three reasons:

1. **The winners-minus-losers Δ metric
   (`QuartoRL/results_io.py:192-214`) conditions on match outcome, not on
   position structure.** In Quarto most select decisions are functionally
   neutral — only positions with a forcing piece available carry a
   discriminable signal. Match outcome is mostly determined by the place
   head and opponent mistakes. A network that correctly outputs
   "≈0 everywhere except forcing positions" will look saturated under this
   metric. **Ta(2) is the cleanest evidence:** with depth=1 targets that
   are nearly all zero, the network correctly learned a flat-zero output
   (loss = 5×10⁻⁶, Δ = 0). That is the Bayes optimum of the target
   distribution, not a representational failure.
2. **The "deepConv helps Q_select" inference from Sa(1) was weak
   evidence.** Sa(1)'s +0.170 Δ came with a 30pp WR collapse and no
   visible heatmap plate separation. Building Sb on that extrapolation
   does not have a mechanism story; prior null evidence from series P, Q,
   and S for capacity / trunk shape as Q_select fixes remains the
   load-bearing prior.
3. **Ta(1) is a new WR champion candidate even though Q_select is silent
   under the current metric.** The supervised select gradient is helping
   the trunk / place head in a way that is not visible at the select
   head output. Consistent with either H1 below ("metric is wrong") or
   "co-training de-noises representation" (auxiliary clean targets as a
   regulariser). Neither is "representational failure."

Three live hypotheses, none privileged. Each maps to one diagnostic in
[`analysis/qselect_diagnostics/PLAN.md`](../../analysis/qselect_diagnostics/PLAN.md):

- **H1 (metric artefact).** Q_select already encodes forcing structure;
  the match-outcome mean obscures it. → **D1: position-structure metric.**
- **H2 (signal sparsity).** Only a small fraction of buffer rows have
  non-trivial oracle targets. Loss reweighting (Ra) cannot help per-row.
  → **D2: buffer signal-density audit.**
- **H3 (multitask interference).** Shared trunk converges to Q_place
  features (dense gradient); Q_select head cannot fully invert them
  on its sparse signal events. → **D3: decoupled select-only network.**

**Sa(1)/Sb hybrid is deprioritised** until at least one of D1/D2/D3
returns evidence for an architectural mechanism. Promoting Ta(1) to
champion is the cheap follow-up regardless of which hypothesis survives.

### Champion candidate flag

Ta(1) WR (80.3% vs `bot_loss-BT`, 90.2% vs `bot_random`) surpasses ME(2)
and the Sa(3) candidate. Promotion contingent on a clean 5000-epoch
re-run (the existing one was truncated) plus a 10k confirmation. See
`Research-status.md` → Champion.
