# Series O — Unified-aux variant of decoupled-autoreg trunk

`QuartoCNNAutoregUnified` + `Quarto_unified_bot`. 32-d phase-stable aux (`[offered_one_hot ; available_mask]`). Designed as the interpretability substrate for SAE work. See parent: `Research-status.md` and `docs/diary/2026-05-08_unified-aux-trunk.md`.

## Pending: OA_unifiedAux — Unified-Aux Variant of Decoupled Autoreg

**Hypothesis:** The two-mode `state_aux` (one-hot offered piece during place / available-piece mask during select) is unnecessary. A single phase-stable 32-d aux `[offered_one_hot ; available_pieces_mask]` carries all relevant context at every step, lets the trunk be phase-agnostic (no phase embedding), and pumps a single distribution through every hooked layer. **Targets are unchanged** (TD on place, MC on select); only the input semantics change. The conjecture is that the matched-WR run will be cleanly comparable to MB/ME and produce a single, pool-able activation distribution suitable for one SAE per layer — eliminating the per-phase split that decoupled-autoreg would force on `games-interp`.

**Design rationale:** see [`docs/diary/2026-05-08_unified-aux-trunk.md`](docs/diary/2026-05-08_unified-aux-trunk.md).

**Code change from M (added 2026-05-08, additive — no existing class touched):**

- New model classes `QuartoCNNAutoregUnified` (tanh) and `QuartoCNNAutoregUnifiedUnbound` (no tanh) in `models/CNN_autoreg.py`. 32-d aux input (16 offered + 16 available), no phase embedding, otherwise identical trunk shape to `QuartoCNNAutoreg` (`trunk_in_channels=18`).
- New `TRANSITION_SCHEMA="unified_autoreg"` in `QuartoRL/RL_functions.py` with `gen_experience_unified_autoreg`. Reuses `DQN_training_step_decoupled_autoreg` for targets (per-phase masked TD/MC; aux dimensionality is consumed inside the model).
- New bot `Quarto_unified_bot` (`bot/CNN_unified_bot.py`) builds 32-d aux at every forward pass and tracks `last_placed_piece` between this player's `place_piece(...)` and the next `select(...)`.

**Sweep:** `N_LAST_STATES_INIT` ∈ {2, 3, 4} — same range as the informative MB N-values. Mirrors the MB_final sweep so head-to-head comparison isolates the input-semantics change.

| Run | N_LAST_STATES_INIT | Epochs |
|-----|---------------------|--------|
| OA_unifiedAux(1) | 2 | 5000 |
| OA_unifiedAux(2) | 3 | 5000 |
| OA_unifiedAux(3) | 4 | 5000 |

**Fixed:** `ARCHITECTURE=QuartoCNNAutoregUnified`, `TRANSITION_SCHEMA="unified_autoreg"`, `DECOUPLED_TARGET_STYLE="td_place_mc_select"`, `REWARD_FUNCTION="final"`, `STARTING_NET=None`, `LR=7e-4`, `TAU=0.01`, `MATCHES_PER_EPOCH=32`, `EPOCHS=5000`. Identical to the corresponding MB_final config except for `TRANSITION_SCHEMA` and `ARCHITECTURE`.

**Compare to:** MB_final(1) (N=2, 66.9% WR), MB_final(2) (N=3, 45.8% WR), MB_final(3) (N=4, 56.2% WR) — same WR-vs-{bot_loss-BT, bot_random} comparison. The headline question is whether OA matches MB on WR. Secondary: whether the Q_select panel changes shape under unified aux (trunk now sees richer input; if Q_select was input-starved this would show).

**Decision gate:**
- If OA WR ≥ MB WR within 3pp at every N → the unified-aux trunk is the new interp target. Proceed to a place-only `games-interp` adapter on the OA(1) checkpoint.
- If OA WR loses ≥5pp at any N → the input redesign costs something; either keep the phase embedding (cheap variant) or fall back to ME(2) for interp work.
- Independent of WR, also check the Q_select Outcome=+1 panel: any improvement here would corroborate the schema-bug hypothesis from the open problem above (winner-side select samples may be reaching the loss with cleaner labels through unified aux than they did through the phase-conditional schema).

