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

## Result — pending
