# Series S — Structural trunk variants

Three structural trunk changes on the OA-family substrate
(`QuartoCNNAutoregUnified*`, `unified_autoreg`, `td_place_mc_select`), N=4
constant, otherwise identical ME(2) recipe. See parent: `Research-status.md`.

## Sa_archScan — Three trunk variants

**Hypothesis:** Q_select saturation persists across joint/decoupled/unified
schemas and is not rescued by reweighting (`series-R.md`). The remaining
locus is the **trunk representation of the (offered piece × board)
interaction** — the place head can reduce its loss by exploiting board-only
features, but the select head needs to grade *future-piece × current-board*
combinations and the OA trunk may not factor that out cleanly. The S
variants probe three structural axes that touch that interaction:

- **S1 deepConv** — additional conv block before `fc1`. Hypothesis: more
  spatial receptive field on the board lets the trunk represent
  threat-windows the offered piece either fills or doesn't.
- **S2 wideFC** — wider+deeper `fc1`. Hypothesis: the (board, aux)
  combination is fundamentally an FC mixing problem; capacity at the mixer is
  what's missing.
- **S4 uniform512** — flat 512-d trunk throughout. Hypothesis: bottlenecks
  destroy the piece × board interaction; remove them.

(S3 per-line attention deferred — separate design surface.)

**Fixed:** `STARTING_NET=None`,
`TRANSITION_SCHEMA="unified_autoreg"`,
`DECOUPLED_TARGET_STYLE="td_place_mc_select"`,
`REWARD_FUNCTION="final"`, `N_LAST_STATES_INIT = N_LAST_STATES_FINAL = 4`,
`ENDGAME_FRACTION=0.5`, `N_LAST_STATES_ENDGAME=2`, `LR=7e-4`, `TAU=0.01`,
`(α_place, α_select) = (1.0, 1.0)`, `EPOCHS=5000`.

| Run | Architecture | Epochs |
|-----|--------------|--------|
| Sa(1) | `QuartoCNNAutoregUnifiedS1` (deepConv) | 5000 |
| Sa(2) | `QuartoCNNAutoregUnifiedS2` (wideFC) | 5000 |
| Sa(3) | `QuartoCNNAutoregUnifiedS4` (uniform512) | 5000 |

**Decision gate (pre-registered):**
- WR vs `bot_loss-BT` ≥ 70% on at least one cell (non-regression vs ME(2)
  while keeping the unified aux substrate), AND/OR
- Q_select winners-minus-losers Δ ≥ +0.40 (the open-problem gate).

**Result (2026-05-12, 5000 epochs each):**

| Run | Trunk | loss_select | Q_select Δ | Q_place Δ | WR vs BT (final) | WR vs BT (peak) | WR vs random (final) |
|---|---|---|---|---|---|---|---|
| Sa(1) | deepConv | 0.225 | **+0.170** | +0.231 | 44.3% | 47.8% | 66.9% |
| Sa(2) | wideFC | 0.228 | +0.072 | +0.332 | 63.7% | 66.6% | 80.4% |
| Sa(3) | uniform512 | 0.248 | +0.037 | **+0.457** | **73.5%** | 75.2% | 86.4% |
| ME(2) | M-trunk (ref) | 0.21–0.24 | ≈ +0.04 | ≈ +0.55 | 73.7% | 75.0% | 85.8% |
| OA(1) | OA-trunk (ref) | — | — | — | 66.3% | — | — |

**Sa(3) — interpretability champion candidate.** Sa(3) matches ME(2) on WR vs
both baselines (73.5% vs 73.7%, 86.4% vs 85.8%) while keeping the unified
32-d phase-stable aux required by the SAE pipeline. That is +7pp over the
prior OA interpretability champion at no WR cost. Q_select Δ is unchanged
from the M/N/O pattern (+0.037 ≈ +0.04), so it does not solve the open
problem — but it removes the standing 7pp interpretability tax on the
substrate. Pending a confirmation run, Sa(3) replaces OA(1) as
*interpretability substrate of record*. WR curves still rising at epoch
5000; a 10k extension is cheap and would tighten the comparison to ME(2).

**Sa(1) — first numeric movement on Q_select, no visible plate separation.**
The +0.170 winners-vs-losers gap on Q_select is the largest in any
architecture we have run — roughly 4× the typical ≈ +0.04 the open problem
has been stuck at. **Caveat:** this is a *numeric* statistic on the means of
the two outcome populations, **not** visible separation in the qv heatmaps.
Both `Outcome=−1` and `Outcome=+1` panels of `Sa_archScan(1)_qv.png` still
show the bulk of Q_select density piled at the −1 floor; the +0.170 number
reflects a small upward shift in the upper-tail mean, not two distinguishable
populations. Sa(1) WR is also the worst of the three (44.3% final, peak
47.8%) — but the WR curve is **still rising at epoch 5000** and the trend
across the back half is shallow but not flat, so the WR collapse may be
slow convergence rather than capacity loss. Do not retire deepConv as a
direction on the 5000-epoch run alone.

**Sa(2) wideFC.** Intermediate on every metric. Neither structural axis on
its own (deeper conv, wider FC) breaks the open problem; both individually
keep WR in the OA/M range without producing a select-head improvement worth
recording.

**Conclusion.**

1. **Sa(3) is promoted to interpretability substrate of record (pending
   confirmation run).** Same WR as M-series champion, with the unified aux
   substrate. ME(2) remains overall WR champion; Sa(3) is the
   SAE-pipeline-eligible champion.
2. **The OA trunk is the bottleneck for the unified aux WR tax.** Replacing
   it with a flat 512-d trunk (no bottleneck FC) closes the 7pp gap to ME(2).
3. **Sa(1)'s deepConv produced the largest Q_select Δ on record but is not a
   visible separation.** Combined with Ra showing a `loss_select` floor,
   this points the same direction: the targets the select head is asked to
   fit (MC return on select actions in `td_place_mc_select`) are noisy enough
   that even an architecture that *can* fit them slightly better hits a
   target-quality wall before producing a usable plate split.

**Queued (not run):**

- **Sb_hybridTrunk** — Sa(3) uniform-512 trunk with Sa(1)'s deeper conv
  stack on top. Sa(3) carries WR through Q_place; Sa(1) carries the
  numeric Q_select Δ. If the two effects are orthogonal, the combination
  beats ME(2) on Q_select Δ while matching it on WR.
- **Sc_deepConvLong** — Sa(1) at 10k epochs (or LR=3e-4 over 5k). The
  rising-but-low WR trend is the only direct evidence we have that deepConv
  is capacity-limited rather than slow-converging. Cheap to settle.
- **S3 per-line attention** — still deferred; separate design note required
  before queueing.

Ranking: confirmation runs for Sa(3) first (interpretability champion is a
hard-won win), then Sb. Sc is contingent on the target-rethink
(`2026-05-14_qselect-target-rethink.md`) — if select targets switch from MC
return to minimax oracle, the deepConv WR collapse may evaporate, making a
plain extension less informative.
