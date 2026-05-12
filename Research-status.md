# Quarto RL — Research status

Slim ledger. Long-form per-series content lives in [`docs/diary/`](docs/diary/).
Add a new series entry there when starting a new code-version letter; append
result paragraphs here under "Recent results" when a sweep concludes.

---

## Champion

- **`ME_endgame(2)0429_ENDGAME_FRACTION_0.5`** — 73.7% WR vs `bot_loss-BT`,
  85.8% WR vs `bot_random`, loss 0.207. Uses the M-series decoupled-autoreg
  trunk (`QuartoCNNAutoreg`) with `td_place_mc_select` targets,
  N_LAST_STATES_INIT=2 curriculum to N=4, ENDGAME_FRACTION=0.5,
  N_LAST_STATES_ENDGAME=2, LR=7e-4, TAU=0.01, 5000 epochs.
- **Interpretability substrate of record:** `OA_unifiedAux(1)0509_N_LAST_STATES_INIT_2`
  — 66.3% WR vs `bot_loss-BT`. Same loss / schema family as ME(2) but with the
  unified 32-d phase-stable aux (`[offered_one_hot ; available_mask]`) and no
  phase embedding. 7pp behind ME(2) on WR; standing trade for SAE-able activations.

## Current Open Problem — Q_select saturation

`Q_select` collapses near −1 across every architecture and target style we
have tried (M, N, O, Q) — the place head carries WR while the select head
contributes ~0 to value separation. Triangulation as of 2026-05-12:

- **Head only (Pa_frozenTrunkSelect):** freezing the ME(2) trunk and retraining
  only `fc2_select` lifts the winners-mean by ≈ +0.20, not the ≥ +0.40 the gate
  required. **Head retraining alone is insufficient.**
- **Asymmetric LR (Nb_asymLR):** 10× LR on the select head produced no separation.
- **Wider `fc1` + aux legality + no inference mask (QC_unifiedNoMask):** the
  aux BCE collapsed (0.69 → 0.15) but `invalid_argmax_rate` stayed at 0.32 and
  `Q_select` reproduced the same −0.73 saturation. The fix did not transfer
  into the Q heads.

Conclusion: the locus is the **trunk representation of the (offered piece × board)
interaction** under `td_place_mc_select`. Next directions on deck: per-head
loss reweighting (Ra), structural trunk changes (Sa).

## Code-version letters (active and queued)

| Letter | Series | Architecture / Schema | Status |
|---|---|---|---|
| A–E | early / combined_avg baseline | joint | done — see `docs/diary/series-A.md` |
| F | adversarial sign flip | joint | failed/reverted — `series-FG.md` |
| G | separate_bellman | joint | done — `series-FG.md` |
| H–K | terminal mask / unbound / final / coupled | joint | done — `series-HIJK.md` |
| L | mc_select | joint | done — `series-L.md` |
| M | decoupled_autoreg | `QuartoCNNAutoreg*` + `Quarto_autoreg_bot` | **champion** — `series-M.md` |
| N | shared-trunk diagnostics | M-series | done — `series-N.md` |
| O | unified_autoreg | `QuartoCNNAutoregUnified*` + `Quarto_unified_bot` | done — `series-O.md`, `docs/diary/2026-05-08_unified-aux-trunk.md` |
| P | frozen-trunk select | M-series | done (negative) — `series-P.md` |
| Q | unified + aux legality + no mask | `QuartoCNNUnifiedNoMask` + `Quarto_unified_nomask_bot` | done (gate failed) — `series-Q.md`, `docs/diary/2026-05-11_qc-no-mask.md` |
| **R** | **per-head loss weighting** | OA-family | **queued** |
| **S** | **structural trunk variants** | OA-family | **queued** |

## Recent results

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

1. **Ra_lossWeight** — Ra sweep over `(LOSS_ALPHA_PLACE, LOSS_ALPHA_SELECT)` from scratch on the OA aux family at N=4, 5000 epochs. Tests whether the select head is gradient-starved vs mis-targeted. Grid: (1,1), (1,3), (0.3,1), (0.1,1).
2. **Sa_archScan** — three structural trunk variants on the OA family, also N=4, 5000 epochs: S1 deeper conv, S2 wide+deep `fc1`, S4 uniform 512-d trunk. Per-line attention (S3) deferred.

Both use the new per-head loss / grad logging and JSONL emission added on 2026-05-12.

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
5. **Loss reweighting and structural trunk changes have not been tested yet** at the same recipe / horizon as the M-series. R and S series fill that gap.

For each of these, the originating series file in `docs/diary/` carries the full evidence.
