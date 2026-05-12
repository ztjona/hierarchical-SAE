# Series Q — Auxiliary legality + no inference mask

`QuartoCNNUnifiedNoMask` + auxiliary BCE legality head + no inference legality mask in the bot. See parent: `Research-status.md` and `docs/diary/2026-05-11_qc-no-mask.md`.

## QC_unifiedNoMask — New Architecture, Q-series Start

**Hypothesis (WR-first, with a substrate fix):** The M/N/O series have plateaued near 66–73% WR while sharing a structural defect that the `games-interp` audit on `AA(2)` made concrete:

(Full design notes: [`docs/diary/2026-05-11_qc-no-mask.md`](docs/diary/2026-05-11_qc-no-mask.md).)

1. The network does **not** learn cell-legality (`games-interp` Test D Q-gap = −0.47 mean; legality is enforced entirely by the bot's inference-time validity filter).
2. `fc1` destroys the threat features that `conv2` still carries (LP F1 collapses 0.50 → 0.02 across the fc1 bottleneck).

QC targets both of those simultaneously without changing the RL algorithm:

- **Auxiliary legality head** `fc_aux_legality: Linear(n_neurons, 16)` trained with `BCEWithLogits(logits, is_empty(cell))`, supervised every batch from `state_board`. λ_legality = 0.05 starting point. This forces `fc1` to preserve per-cell legality information explicitly, instead of relying on the inference mask.
- **Wider `fc1` (128 → 256).** Cheap capacity check on the bottleneck where threats are getting lost.
- **No inference legality mask.** The paired `Quarto_unified_nomask_bot` picks `argmax(Q_place)` directly. If the top-ranked cell is invalid the bot falls back to the next valid rank so training can complete, and records `invalid_argmax_rate` as the legality-learning metric. The success criterion is that this rate → 0 over training, *not* zero from epoch 1.
- **Unified 32-d phase-stable aux + no phase embedding.** Inherited from OA. Single activation distribution per layer for SAE work.

QC starts the **Q-series** (per `Experiment Naming Convention`, a new code-version letter is mandated by the new architecture class, new bot class, and new auxiliary loss term).

**Code change (additive — no existing class touched):**

- `models/CNN_unified_nomask.py`: new class `QuartoCNNUnifiedNoMask` with `forward(x_board, x_aux, phase=...)` → `(q_place, q_select)`, `forward_with_aux(...)` → `(q_place, q_select, legality_logits)`, plus the helper `legality_target_from_board(x_board) -> (B, 16)`.
- `bot/CNN_unified_nomask_bot.py`: subclasses `Quarto_unified_bot` and overrides `_choose_board_position` to pick `argmax(Q_place)` without validity filtering; exposes `invalid_argmax_rate()` and `reset_legality_counters()`.
- `train_scripts/QC_unifiedNoMask(1)0511.py`: from-scratch training, ME(2) recipe (N=2→4 curriculum, `ENDGAME_FRACTION=0.5`, `N_LAST_STATES_ENDGAME=2`, `EPOCHS=5000`, `LR=7e-4`, `TAU=0.01`). After every `DQN_training_step` call the loop additionally computes `legality_loss = BCEWithLogits(policy_net.legality_logits(state_board, state_aux), legality_target_from_board(state_board))` and minimises `L = L_DQN + λ_legality · L_legality` with a single `backward()`.
- `tests/test_qc_unified_nomask.py`: 14 pytest cases covering the legality-target helper (empty / half-full / row-major / bad-shape), `forward` shapes and tanh range, `forward_with_aux` three-tuple, `legality_logits` shape, aux-dim validation, numpy input acceptance, state-dict round-trip, `q_values_phase` routing, and `name` property. All pass as of 2026-05-11.

**Fixed:** `ARCHITECTURE=QuartoCNNUnifiedNoMask`, `BOT=Quarto_unified_nomask_bot`, `TRANSITION_SCHEMA="unified_autoreg"`, `DECOUPLED_TARGET_STYLE="td_place_mc_select"`, `REWARD_FUNCTION="final"`, `STARTING_NET=None`, `λ_legality=0.05`.

| Run | ENDGAME_FRACTION | Epochs |
|-----|-------------------|--------|
| QC_unifiedNoMask(1) | 0.5 | 5000 |

**Compare to:** `ME_endgame(2)_E_5000` (current champion, 73% WR vs `bot_loss-BT`), `OA_unifiedAux(1)` (66.3% at N=2, matched MB), `bot_random`.

**Decision gate (pre-registered):**
- **WR vs `bot_random` ≥ 95%** (hard requirement — anything weaker means the new architecture failed to learn the game).
- **WR vs `ME_endgame(2)_E_5000` ≥ 50%** (non-regression on the WR-first criterion).
- **`invalid_argmax_rate` < 0.05 averaged over the last 100 epochs** (legality has been learned, not masked).
- **Optional, deferred to `games-interp`:** Test D Q-gap (`Q(empty) − Q(occupied)`) ≥ +0.3 on `QC_unifiedNoMask(1)_E_5000`. Confirms the legality concept landed in the Q values, not just the auxiliary head.

Combinations:
- All four pass → QC is the new champion AND the new SAE substrate. Q2 is then the platform on which Q3 (selection-reward shaping) and Q4 (extended N curriculum / larger endgame buffer) build.
- WR passes but `invalid_argmax_rate` stays high → the aux head learned legality in isolation (gradient through `fc_aux_legality`) but `fc1` is still legality-blind in the Q heads. Increase λ_legality, or freeze `fc_aux_legality` and back-propagate only the BCE through `fc1`.
- WR vs ME(2) < 50% → the architecture change cost performance; do NOT proceed to Q3/Q4 from QC. Diagnose whether the wider `fc1` or the absent mask is the regression driver before iterating.

**Q-series roadmap (forward references, not yet started):**

- **Q3 — selection-reward shaping.** From QC(1), add intermediate signal on `Q_select` (e.g. `+0.1` for offering a piece that cannot complete a line on the opponent's turn, `−0.1` otherwise). Tests whether dense select feedback can shake the head off saturation that the trunk fix alone failed to address.
- **Q4 — extended N curriculum.** From QC(1), push `N_LAST_STATES_FINAL` to 6 or 8 with a larger endgame buffer. Tests whether more horizon, on the legality-aware substrate, finally produces a bot competent at mid-game offensive play.

**Result (2026-05-12, 5000 epochs):**

| Metric | Gate | Observed | Pass? |
|---|---|---|---|
| WR vs `bot_random` (last 10% of epochs) | ≥ 95% **hard** | 85.0% (peak 86.1%) | **no** |
| WR vs `ME_endgame(2)_E_5000` (last 10%) | ≥ 50% | 46.2% (peak 48.7%) | **no** (just) |
| `invalid_argmax_rate` (last 100 ep, smoothed) | < 0.05 | **0.324** (start 0.32, no movement) | **no** |
| `legality_loss` (BCE) | qualitative | 0.69 → 0.15 | aux head fits cleanly |
| Q_select winners (last 100) | — | −0.73 (saturated) | unchanged from M/N/O |
| Final DQN loss | — | 0.204 | higher than ME(2) (0.03–0.11) |

Two failure modes confirmed simultaneously:

1. **WR regression.** QC underperforms ME(2) and every relevant N=2 baseline (LA(1) 80.9%, Aa(2) 81.0% vs `bot_random` — QC 85.0% is in the bottom half of the N=2 cohort and far short of the 95% hard gate). The wider `fc1` + dropped phase embedding + dropped inference mask cost performance.
2. **Auxiliary-head decoupling.** The BCE on `fc_aux_legality` collapsed from 0.69 (random) to 0.15 — the aux head learned legality. The Q heads did not: `invalid_argmax_rate` stayed at ~0.32 for the entire run, identical to its epoch-1 value. λ_legality=0.05 produced a side-head that fits supervised legality without forcing `fc1` to expose legality to `fc2_place`.

Q_select once again saturates at ≈ −0.73, identical to the M/N/O pattern, despite a wider trunk and an explicit legality signal upstream. This is the third architecture in a row to reproduce the saturation, confirming Pa's diagnosis: the open problem is *not* an `fc1` capacity issue or an inference-mask artefact — it is in how select-Q targets relate to trunk activations under `td_place_mc_select`.

**Conclusion:** QC fails its decision gate on every hard criterion. Per the pre-registered logic, **do not proceed to Q3 (select-shaping) or Q4 (extended N curriculum) from QC(1)**. The auxiliary-legality + no-mask design is rejected at λ_legality=0.05 in this configuration. The two follow-ups indicated by the gate ("WR passes but invalid_argmax_rate stays high") are still on the table for diagnosing the aux-head decoupling — raise λ_legality, or freeze `fc_aux_legality` and route BCE only through `fc1` — but neither is a champion candidate; both are subordinate to a separate WR-fixing direction. `ME_endgame(2)_E_5000` remains champion (73% WR vs `bot_loss-BT`).

