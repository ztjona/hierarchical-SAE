# Series W — N_LAST_STATES sweep on the Ve champion recipe

Champion (Ve(4)) substrate held fixed — `unified_autoreg` +
`QuartoCNNAutoregUnifiedS4` (Sa(3) trunk) + depth-2 minimax select target,
oracle always on — with the single knob `N_LAST_STATES_INIT` swept *upward*
from the champion's value of 4. Parent: [`Research-status.md`](../../Research-status.md);
recipe lineage in [`series-V.md`](series-V.md) and
[`series-T.md`](series-T.md).

## Wa_oracleStates — does more oracle-supervised buffer depth help?

**Hypothesis.** `N_LAST_STATES_INIT` controls how many of each game's final
transitions enter the replay buffer. Larger N ⇒ more oracle-supervised
SELECT transitions per game. The pre-registered bet (forward-queue item #1):
**larger N raises both D1 position-structure recall and WR** by giving the
minimax oracle more opportunities per game to imprint piece-safety structure.

**Stated risk (from the queue).** Replay-buffer distribution shift à la
`Ac_fineShallow`, and dilution — the pre-T D1 scan showed OA(4) N=12 landed
at *chance* on `forcing_loss_bottom_recall` **without** an oracle. The queue
flagged that N=12 may dilute per-state signal even with the oracle on, and
recommended capping at N=8. **This sweep deliberately tested the upper regime
({8, 12, 16}) to characterise the dilution edge; the lower arm (N<4, N=6) was
not run.**

**Code surface.** `run_trains.py:107-137` — `MULTI_PARAMS` block sweeping
`N_LAST_STATES_INIT ∈ {8, 12, 16}` at `EPOCHS=10000`, everything else at the
Ve(4) recipe. No `trainRL.py` change needed; the knob already existed.

**Fixed:** `STARTING_NET=None`, `ARCHITECTURE=QuartoCNNAutoregUnifiedS4`,
`TRANSITION_SCHEMA="unified_autoreg"`, `USE_MINIMAX_SELECT_TARGET=True`,
`MINIMAX_SELECT_DEPTH=2`, `MINIMAX_DISABLE_AFTER_EPOCH=None`,
`DECOUPLED_TARGET_STYLE="td_place_minimax_select"`, `REWARD_FUNCTION="final"`,
`N_LAST_STATES_ENDGAME=2`, `ENDGAME_FRACTION=0.5`, `LR=7e-4`, `TAU=0.01`,
`GAMMA=0.99`, `BATCH_SIZE=32`, `NUM_EPOCHs_BUFFER=8`, `EPOCHS=10000`.
D1 metrics computed inline per checkpoint (`COMPUTE_D1_INLINE=True`,
~260 decisive states/checkpoint).

| Run | `N_LAST_STATES_INIT` | Epochs |
|---|---|---|
| Wa(1) | 8 | 10000 |
| Wa(2) | 12 | 10000 |
| Wa(3) | 16 | 10000 |
| Ve(4) ref | 4 | 10000 |

**Decision gate (post-hoc, the queue did not pre-register one).** Treat the
hypothesis as supported only if some N>4 beats Ve(4) on **both** WR and D1.
Promote a new interpretability substrate if any N beats Ve(4) on D1 while
matching WR within noise (~1pp).

## Result — 2026-06-04 (10000 epochs each)

### WR and loss [DIRECT — from JSONL summaries]

| Run | N | `loss_select` | WR vs BT (final / peak) | WR vs random (final / peak) | WR trend vs BT (pp/1000ep) |
|---|---|---|---|---|---|
| Ve(4) ref | 4 | 0.041 | **87.2% / 88.9%** | **93.8% / 94.8%** | +0.65 ↑ |
| Wa(1) | 8 | 0.029 | 87.0% / 88.3% | 93.3% / 94.5% | +0.6 ↑ |
| Wa(2) | 12 | 0.0167 | 85.7% / 87.4% | 92.2% / 93.4% | +0.5 ↑ |
| Wa(3) | 16 | 0.0153 | 85.9% / 86.9% | 92.5% / 94.1% | +0.8 ↑ |

### D1 position-structure recalls [DIRECT — inline JSONL, back-half pooled]

Back-half = epochs 5k–10k, ~1,500 decisive states pooled per run. Ve(4)
column is the offline diagnostic reading (n=916, single final checkpoint)
cited in [`series-V.md`](series-V.md) — measurement protocol differs (offline
large-n vs inline ~260/checkpoint), so the N=4↔N=8 step is cross-protocol.

| Metric | Ve(4) N=4 | Wa(1) N=8 | Wa(2) N=12 | Wa(3) N=16 | Chance |
|---|---|---|---|---|---|
| `safe_piece_recall` (back-half / final) | — / 0.846 | **0.891 / 0.916** | 0.874 / 0.904 | 0.856 / 0.876 | 0.114 |
| `forcing_loss_bottom_recall` | 0.968 | **0.970** | 0.963 | 0.962 | ~0.66 |
| `spearman_rho_mean` (back-half) | 0.610 | **0.667** | 0.653 | 0.629 | 0 |

### Reading [INFERENTIAL — interpretation; numbers above are DIRECT]

1. **D1 is an inverted-U in N, peaking at N=8.** N=4→8 lifts every D1 metric
   to a new best-on-record (safe-piece recall 0.891 back-half / 0.916 final
   vs Ve(4)'s 0.846; ρ̄ 0.667 vs 0.610). N=8→12→16 then walks all three
   metrics monotonically *back down* toward the N=4 level. The falling arm
   (8>12>16) is within-protocol and consistent across all three metrics —
   solid. The rising arm (4→8) is a single cross-protocol pairwise
   comparison (~2–3σ) — suggestive, not conclusive.

2. **WR is flat-to-down — the hypothesis is refuted for play strength.** WR
   is tied at N=4≈8 (87.2 vs 87.0, inside noise) then drops ~1.3–1.5pp at
   N=12/16. More oracle-supervised transitions per game did **not** buy WR.
   All trends remain mildly positive (+0.5–0.8 pp/1000ep) so nothing has
   plateaued, but the ordering is set by N, not by training length.

3. **`loss_select` falls monotonically with N — and it is a dilution
   artefact, not a quality signal.** loss_select drops 0.041→0.029→0.017→0.015
   (N=4→16) while D1, the recall on *decisive* states, gets *worse* past
   N=8. Larger N pulls earlier-game states into the buffer; their minimax
   targets are low-variance (most pieces still safe ⇒ flat target vector ⇒
   trivially fittable), so they dominate the average and depress loss while
   the decisive-state encoding bleeds away. **loss_select is anti-correlated
   with D1 across N — do not use it as a cross-N quality proxy.**

4. **The dilution the queue feared is real but the oracle floors it.** OA(4)
   N=12 hit chance on `forcing_loss_bottom_recall` *without* an oracle. Here,
   with the oracle on, forcing recall holds at ~0.96 even at N=16 (chance
   ~0.66) and safe-piece recall declines *gracefully* rather than collapsing.
   Dilution under oracle supervision is graded, not catastrophic.

### Conclusions

- **Hypothesis partially confirmed for D1, refuted for WR, reversed past
  N=8.** N=8 is the D1 optimum; N≥12 is strictly dominated (worse WR *and*
  worse D1, "winning" only on the misleading loss metric).
- **Wa(1) N=8 is a candidate interpretability substrate** — best-on-record
  D1 (safe-piece 0.89–0.92, ρ̄ 0.67) at WR statistically tied with Ve(4).
  For raw WR, **Ve(4) (N=4) stays champion** by a whisker (marginally ahead
  on every WR measure). Single seed per N; the N=4→8 D1 gain is cross-protocol
  — confirmation seed advisable before formal promotion.
- **N≥12 retired.** N=6 not pursued — the N=8 plateau makes a further point
  between 4 and 8 a minimal-upside probe (user call, 2026-06-04).
- **Forward-queue item #1 (Wa sweep) closed.** The N axis is exhausted for
  WR purposes; the headline open problem — Ve(4) still loses ~5.7% vs random
  (57/1000 games, near-symmetric by side; see `champion-results.jsonl`) —
  is *not* an N problem. Next direction shifts to a **loss autopsy** of the
  vs-random defeats (every loss vs random is a SELECT-side event: the agent
  handed random a piece it then completed a line with), to separate avoidable
  hot-gives (the 1−`safe_piece_recall`≈15% residual) from forced-give floor
  positions before the next training swing. See `Research-status.md` →
  forward queue.
- **Autopsy result (2026-06-04, `analysis/competence_audit/REPORT.md`).** The
  deterministic argmax policy loses ~4.2% (the 5.7% was the temp=0.1 sampling
  agent). Of those losses, only ~⅓ are avoidable hot-gives (~1.4% of games);
  ~⅔ are *forced* positions the agent walked into mid-game (~2.7%). A perfect
  select head caps the WR gain at ~1.4 pp — the select side is **not** where
  most of the WR-vs-random loss lives. The place head also misses 6.7% of
  immediate wins. **No first-player effect** (loss/avoidable/forced all
  symmetric P1 vs P2, |z|<1.6). The next direction is therefore place-side /
  planning, not the originally-implied select-margin loss.
