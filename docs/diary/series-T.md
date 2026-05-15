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

| Run | Oracle depth | Substrate | Epochs |
|-----|--------------|-----------|--------|
| Ta(1) | 2 | `QuartoCNNAutoregUnifiedS4` (Sa(3)) | 5000 |

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

**Next variants (queued, not run):**

- **Tb_depth4** — same recipe as Ta(1) but `MINIMAX_SELECT_DEPTH=4`. Test
  whether deeper lookahead breaks the depth-2 information ceiling.
- **Tc_scalarTarget** — Ta(1) with scalar-at-chosen-piece target instead
  of full 16-d vector. Diagnostic: isolates "oracle replaces MC" from
  "oracle gives 16× more signal per state".
- **Td_oracleCache** — adds an LRU cache to `_minimax_select_target`
  keyed by `(board_serial, frozenset(available_pieces), mode_2x2)`. Pure
  perf, no semantic change.

Result section to be filled in after the run lands.
