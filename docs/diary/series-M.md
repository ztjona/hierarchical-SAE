# Series M — Decoupled-autoregressive schema (`td_place_mc_select`)

Introduced the `decoupled_autoreg` transition schema with separate place / select transitions. ME_endgame(2) is the canonical champion from this series (73% WR vs `bot_loss-BT`). See parent: `Research-status.md` and `docs/diary/2026-05-08_unified-aux-trunk.md`.

## MA_tempRegresive — Decoupled-Autoregressive Schema (N_LAST_STATES sweep)

> **Naming note:** this should have been `MA_*` per the convention (first sweep of code version `M`). The `Z_*` prefix was an oversight; subsequent decoupled-autoreg sweeps follow the `M*_*` scheme.

**Question:** Does training place and select on **separate transitions** — instead of bundling them into one joint experience tuple — give Q_select an independent, well-anchored learning signal and remove the cross-head interference observed in every prior code version?

**Code change:**
- New `TRANSITION_SCHEMA="decoupled_autoreg"` in `gen_experience` / `DQN_training_step` (`QuartoRL/RL_functions.py`).
- New architecture `QuartoCNNAutoreg` (`models/CNN_autoreg.py`) with a shared trunk + phase embedding + place/select heads, called twice per turn (one phase per pass).
- New bot `Quarto_autoreg_bot` (`bot/CNN_autoreg_bot.py`) using `predict_phase` / `q_values_phase` for inference.
- `DECOUPLED_TARGET_STYLE="td_place_mc_select"`: place transitions use `R + γ · max_a' Q_target(s', next_phase=select, a')`; select transitions use `γ^steps_to_terminal · outcome`.

| Run | N_LAST_STATES_INIT |
|-----|---------------------|
| MA_tempRegresive(1) | 2 |
| MA_tempRegresive(2) | 3 |
| MA_tempRegresive(3) | 4 |
| MA_tempRegresive(4) | 6 |
| MA_tempRegresive(5) | 12 |
| MA_tempRegresive(6) | 16 |

**Fixed:** `STARTING_NET=None`, `EPOCHS=5000`, `NUM_EPOCHs_BUFFER=8`, `LR=7e-4`, `TAU=0.01`, `GAMMA=0.99`, `ARCHITECTURE=QuartoCNNAutoreg`, `TRANSITION_SCHEMA="decoupled_autoreg"`, `DECOUPLED_TARGET_STYLE="td_place_mc_select"`, `LOSS_APPROACH="mc_select"`, `REWARD_FUNCTION="propagate"`.

**Result (5000 epochs, full sweep):**

| N | Final loss | Final WR vs bot_loss-BT | Final WR vs bot_random |
|---|---|---|---|
| 2 | 0.314 | 56.0% | 72.8% |
| 3 | 0.388 | 41.8% | 62.5% |
| 4 | 0.420 | 49.7% | 70.7% |
| 6 | 0.424 | 41.0% | 61.8% |
| 12 | 0.413 | 34.5% | 54.9% |
| 16 | 0.392 | 32.7% | 53.7% |

Compared to the joint `LA_mcSelect` baseline:

| N | Joint LA_mcSelect WR vs bot_loss-BT | Decoupled MA_tempRegresive WR vs bot_loss-BT | Δ |
|---|---|---|---|
| 2 | 65.2% | 56.0% | **−9.2pp** |
| 3 | 30.0% | 41.8% | **+11.8pp** |

**Key diagnostic findings (qv panels):**

1. **N=2 starves Q_select of positive samples.** With cutoff = 3 transitions per match in the decoupled schema, the kept window is always `{loser_place, loser_select, winner_terminal_place}` — the only select transition is the loser's, so Q_select sees `outcome=−1` exclusively. The N=2 Q_select Outcome=+1 panel has effectively no samples (colorbar maxes at 0.10%). N=3 finally adds the winner's earlier select to the window, populating both signs.
2. **N=3 confirms Q_select gets data, but does not learn to separate.** At N=4 both Q_select Outcome=−1 and Outcome=+1 collapse to a single ~−0.5 band; at N=16 both collapse to ~0. The schema fixed *availability* of positive samples but not the *learning* problem on the select head.
3. **Q_place develops a bimodal pathology across every N.** Outcome=−1 panels show a bright band at **+1** (loser states predicted as winning) alongside the expected band at −1. At N≥6 both heads collapse toward a diffuse band near 0. The model is partly fitting the degenerate "predict +1 always" solution — which is enough to win 56% vs `bot_loss-BT` at N=2 but is not value learning.
4. **Loss is stuck at 0.30–0.42 across every N**, with no decay over 5000 epochs (vs `LA_mcSelect(1)` reaching 0.033 at N=2). The floor matches what `propagate + tanh + bootstrap` predicts: place targets ≈ ±2 (immediate ±1 reward + γ·±1 bootstrap from Q_select MC) are unreachable under `tanh` and leave a permanent SmoothL1 residual — the same double-counting pathology as `FA_Bellman`, mediated through Q_select instead of an explicit sign flip.

**Conclusion:**
- ✓ Hypothesis partially confirmed: decoupling gives the schema a softer N-cliff (N=3 beats joint by +12pp; N=4 beats N=3 — non-monotonic, real). It does spread value learning across deeper windows.
- ✗ Hypothesis partially falsified: Q_select still does not separate by outcome even when given positive samples, so decoupling alone is not the missing ingredient.
- ✗ Decoupled-autoreg with `propagate` loses the N=2 sweet spot and cannot match the best joint baseline. The dominant failure mode shifts from "Q_select has no signal" (joint) to "Q_place targets are unreachable under tanh" (decoupled).
- **Next move:** keep the schema, change reward to `final` (`MB_final`). This is the single change most likely to fix the unreachable-target floor without re-coupling the heads.

---

## Literature Note — Branching DQN

**Reference:** Tavakoli, Pardo, Kormushev, *Action Branching Architectures for Deep Reinforcement Learning*, AAAI 2018.

**Main idea:** Use one shared state trunk and one action-value branch per action dimension, so output size grows linearly rather than combinatorially with a factored action space. The proposed BDQ agent combines a shared state-value stream with per-branch advantages.

**Why it is relevant here:**
- Quarto also has a factored decision structure: `place` and `select`.
- The paper supports the general architectural idea of a shared representation plus per-action heads.
- It suggests that branch coordination benefits from a shared state-value signal, not fully independent heads.

**Why it is not a direct solution:**
- In Branching DQN the action dimensions are chosen together at the same step and share the same temporal credit assignment.
- In Quarto, `place` and `select` are sequential and asymmetric: `place` can have immediate terminal consequences, while `select` affects the opponent's next turn and only indirectly affects future return.
- So the main Quarto difficulty is not just factorized actions; it is **asymmetric credit assignment across the two branches**.

**Usefulness for future work:**
- Still useful after `JA_final`, but lower priority than testing a coupled architecture first.
- Most promising takeaway: introduce a shared value anchor or a dueling-style decomposition instead of purely independent Bellman targets per head.
- A broad literature review is not necessary yet; a **targeted** review around Branching DQN, dueling DQN, factored action spaces, and hierarchical / semi-MDP credit assignment will be more useful once the next reward-design experiment is run.

---

## Diagnostic Plot Ideas (deferred)

**Bellman residual by horizon** (`|Q(s,a) − target|` vs `steps_to_terminal`):
- For Q_select with `mc_select`, the target is `γ^steps · outcome` — fully deterministic from stored fields. The residual is a linear transformation of the existing horizon QV data and adds no new information in this configuration.
- For Q_place the target requires `max_a' Q_target(s', a')` (not stored), so Q_place residual would be genuinely new — but Q_place is the already-working head, so priority is low.
- **Verdict:** Skip until a configuration where the Q_select target is non-trivial (e.g. Bellman bootstrap for select).

**Action gap** (`max_a Q(s,a) − Q(s, taken_a)` vs steps):
- Measures how far the logged policy is from greedy. Would require returning `qav_place/select.max(dim=1)` from `evaluate()` alongside the taken-action values — no extra forward passes.
- Confounded by `TEMPERATURE_EXPLORE`: non-zero gap can simply mean a non-greedy sample was drawn. Not informative about policy quality independently of temperature.
- **Verdict:** Defer. Note the idea if a low-temperature evaluation buffer is introduced.

---

## MB_final — Decoupled-Autoreg with Final Reward (N_LAST_STATES sweep)

**Question:** Does `REWARD_FUNCTION="final"` remove the unreachable-target loss floor seen in `MA_tempRegresive`? Under `propagate`, the place TD target is `±1 + γ · max_a' Q_select(s', a') ≈ ±2`, unreachable under `tanh`. Under `final` it becomes `0 + γ · max_a' Q_select(s', a')`, bounded in `[−γ, +γ]`.

**Code change vs MA:** only `REWARD_FUNCTION` changes from `"propagate"` to `"final"`.

| Run | N_LAST_STATES_INIT | Epochs |
|-----|---------------------|--------|
| MB_final(1) | 2 | 5000 |
| MB_final(2) | 3 | 5000 |
| MB_final(3) | 4 | 5000 |
| MB_final(4) | 6 | 4050 |
| MB_final(5) | 12 | 2500 |
| MB_final(6) | 16 | 2000 |

**Fixed:** `STARTING_NET=None`, `NUM_EPOCHs_BUFFER=8`, `LR=7e-4`, `TAU=0.01`, `GAMMA=0.99`, `ARCHITECTURE=QuartoCNNAutoreg`, `TRANSITION_SCHEMA="decoupled_autoreg"`, `DECOUPLED_TARGET_STYLE="td_place_mc_select"`, `LOSS_APPROACH="mc_select"`, `REWARD_FUNCTION="final"`.

**Result:**

| N | Final loss | Final WR vs bot_loss-BT | Final WR vs bot_random | MA Δ vs bot_loss-BT |
|---|---|---|---|---|
| 2 | 0.112 | 66.9% | 80.5% | **+10.9pp vs MA** |
| 3 | 0.221 | 45.8% | 64.8% | +4.0pp |
| 4 | 0.237 | 56.2% | 74.0% | +6.5pp |
| 6 | 0.216 | 43.8% | 65.8% | +2.8pp |
| 12 | 0.192 | 34.2% | 54.7% | −0.3pp |
| 16 | 0.179 | 32.3% | 54.3% | −0.4pp |

The reward gain is monotonically larger as N decreases — exactly the signature predicted if the unreachable-target pathology dominated at small N and was washed out by other failures at large N. **N=2 fully matches the best joint baseline `LA_mcSelect(1)` (65.2% / 80.9%) within noise**, recovering the N=2 sweet spot that decoupled-autoreg lost in MA.

**Key diagnostic findings (qv panels):**

1. **MA's bimodal Q_place pathology is gone.** No more spurious `+1` band on Outcome=−1 panels. The `propagate + tanh + bootstrap` double-counting failure is fully resolved.
2. **At N=2, Q_place is correctly bimodal**: Outcome=−1 → ~−1, Outcome=+1 → ~+1. Direct evidence the place head learns terminal value when given reachable targets.
3. **At N≥3, Q_place collapses to a single horizon-invariant band**, drifting from ~+0.10 (N=4) up to ~+0.30 (N=16). Both Outcome=−1 and Outcome=+1 panels overlap. Not bimodal anymore — a *different* failure mode than MA.
4. **Q_select stays flat at ~0 for both outcomes at N≥3** — same as MA. Decoupling + reward fix did not unlock the select head.
5. **Horizon plots (qv_horizon)** show no horizon-conditional shape at N≥3. A correctly-trained MC value function should show |Q| ≈ 1 at terminal step decaying toward 0 with horizon. MB shows a flat band with mild *variance* growth at deep horizons but no shift in the band's center. The model has converged to a horizon-marginal expected return rather than learning a temporal value function. Confirmed `steps_to_terminal` is not fed into the trunk (only `state_board`, `state_aux`, and the phase embedding) — but board fullness is a coarse proxy and the network is not making use of it.
6. **Grad norm telemetry** (`plot_grad_norm`) shows total gradient norm staying well below the `MAX_GRAD_NORM=1.0` clip threshold across all runs — unlike MA(N=2) which occasionally activated clipping. Consistent with select-head saturation: the small select-head gradient is invisible in the summed total.

**Conclusion:**
- ✓ Hypothesis confirmed at N=2: reward design was the dominant failure mode there. Loss floor reduced ~3× (0.314 → 0.112) and WR fully matches the best joint baseline.
- ✗ N-cliff persists for N≥3. MB shows that even with reachable targets, Q_place cannot separate by outcome at non-terminal-only depths. The post-MA bimodal pathology was a symptom of one specific double-counting issue, not the underlying credit-assignment problem.
- **Next move (per pre-registered decision gate):** if the floor persists, the issue is bootstrap-through-Q_select coupling. Next step is `MC_*` with MC targets on both heads (no place TD), to sever Q_place's dependency on Q_select.

---

## MC_MCboth — Full Monte Carlo on Both Heads (N_LAST_STATES sweep)

**Question:** Is Q_place's flat-band failure at N≥3 caused by being supervised through a poorly-anchored Q_select via the place TD bootstrap target? If so, replacing the Bellman bootstrap with a direct MC target (`γ^k · outcome`) on Q_place — same formula already used for Q_select — should let Q_place separate by outcome regardless of what Q_select does.

**Code change vs MB:**
- New `DECOUPLED_TARGET_STYLE="mc_both"` in `QuartoRL/RL_functions.py` → `DQN_training_step_decoupled_autoreg`.
- When active, place target = `γ^steps_to_terminal · outcome` (same formula as the select target). The target net is no longer read by the loss (still updated, just unused — minimal-diff).

| Run | N_LAST_STATES_INIT | Epochs |
|-----|---------------------|--------|
| MC_MCboth(1) | 2 | 5000 |
| MC_MCboth(2) | 3 | 4000 |
| MC_MCboth(3) | 4 | 4000 |
| MC_MCboth(4) | 6 | 3000 |
| MC_MCboth(5) | 12 | 2000 |
| MC_MCboth(6) | 16 | 1000 |

**Fixed:** Identical to MB except `DECOUPLED_TARGET_STYLE="mc_both"`.

**Result vs MB head-to-head:**

| N | MB Final WR vs BT | MC Final WR vs BT | Δ | MB loss | MC loss |
|---|---|---|---|---|---|
| 2 | 66.9% | 66.7% | ~tied | 0.112 | 0.105 |
| 3 | 45.8% | 44.7% | ~tied | 0.221 | 0.377 |
| 4 | 56.2% | 47.1% | **−9.1pp** | 0.237 | 0.384 |
| 6 | 43.8% | 39.5% | −4.3pp | 0.216 | 0.397 |
| 12 | 34.2% | 34.4% | ~tied | 0.192 | 0.369 |
| 16 | 32.3% | 32.5% | ~tied | 0.179 | 0.349 |

Removing the Q_place bootstrap **hurts at N=4 / N=6** and ties everywhere else.

**Key diagnostic findings:**

1. **N=2 matches MB exactly** (66.7% vs 66.9%, identical qv plots). Expected: at N=2 the formulations are mathematically equivalent — only terminal place transitions exist, where `R + γ · 0 = outcome = γ^0 · outcome`. Sanity check passes.
2. **At N≥3, Q_place still collapses to a horizon-invariant band**, visually identical to MB. Direct MC supervision did NOT unlock outcome separation on Q_place. The bootstrap-dependency hypothesis is **falsified**: Q_place fails to fit the outcome at deep horizons even when given the MC target directly.
3. **Q_select stays flat at ~0 across N≥3**, same as MB. Removing Q_place's dependency on Q_select did not free Q_select either — its failure is intrinsic, not a coupling artifact.
4. **MC's higher loss at N≥3 is more truthful, not worse learning.** Under MB, when both heads collapse to ~0, the bootstrap target `0 + γ · 0 = 0` becomes a self-fulfilling fixed point (loss looks small because the model agrees with its own broken estimate). Under MC, the target is `γ^k · outcome` ∈ [±0.85, ±1] and the model is asked to predict large signed values, fails, and loss exposes that failure. **MB's lower loss should not be read as a positive signal in retrospect.**
5. **The bootstrap was doing weak useful regularization** at intermediate N: when both heads sit near 0, MB's bootstrap target stays near 0 and SmoothL1 says "stay put", preserving the slight partial signal. MC removes that and the partial signal degrades into noise — explaining the N=4 / N=6 regression.

**Conclusion:**
- ✗ Bootstrap-dependency hypothesis falsified. Q_place's flat-band failure at N≥3 is not caused by being supervised through Q_select. The pathology survives direct MC supervision unchanged.
- ✓ The failure is signal/representation, not target design. The model cannot fit `γ^k · outcome` targets at non-zero horizons even when handed them directly. This is consistent with the prior gradient-magnitude diagnostic (Nexus note on a different checkpoint): tanh saturation collapses select-trunk gradient ~280× relative to place; the MC switch was *expected* to revive select-head gradient but the qv plots show it did not.
- **Next move:** the only major candidate left is the saturation hypothesis itself. Test unbounded heads (`MD_unbound`), forking from MB (the better baseline), with per-head trunk grad-norm telemetry added so we can *measure* whether unbounded heads revive the select-side gradient.

---

## MD_unbound — Unbounded Decoupled-Autoreg Heads (N_LAST_STATES sweep)

**Question:** Does removing `tanh` from both heads allow the select-trunk gradient to reach the shared trunk, unlocking outcome separation at N≥3?

**Code change vs MB:** `ARCHITECTURE=QuartoCNNAutoregUnbound` (identity output, no tanh). Everything else identical to MB.

| Run | N_LAST_STATES_INIT | Epochs |
|-----|---------------------|--------|
| MD_unbound(1) | 2 | 5000 |
| MD_unbound(2) | 3 | 4000 |
| MD_unbound(3) | 4 | 4000 |
| MD_unbound(4) | 6 | 3000 |
| MD_unbound(5) | 12 | 2000 |
| MD_unbound(6) | 16 | 1000 |

**Fixed:** `TRANSITION_SCHEMA="decoupled_autoreg"`, `DECOUPLED_TARGET_STYLE="td_place_mc_select"`, `REWARD_FUNCTION="final"`, `LR=7e-4`, `TAU=0.01`, `GAMMA=0.99`.

**Result vs MB head-to-head:**

| N | MB WR vs BT | MD WR vs BT | Δ |
|---|---|---|---|
| 2 | 66.9% | 61.6% / 78.6% | **−5.3pp** |
| 3 | 45.8% | ~tied | ~0 |
| 4 | 56.2% | ~tied | ~0 |
| 6–16 | 43.8%–32.3% | ~tied | ~0 |

**Key diagnostic findings:**
1. **No divergence** — stability criterion passed.
2. **N=2 regressed ~5pp** — unbounded heads cost performance at the known-good regime.
3. **N=4 Q_place qv shows weak outcome separation** (bands ~+0.1 vs ~−0.1) — only N where unbounding had a visible positive effect on Q_place.
4. **Q_select flat in every N panel** — unbounding did not revive the select head. The tanh-saturation hypothesis for Q_select is falsified.
5. **Per-head trunk grad-norm split was not captured** — the comparison plot logged total norm only; the load-bearing measurement was incomplete.

**Conclusion:**
- ✗ Decision gate failed on WR: N=2 lost ~5pp; no N improved.
- ✗ Q_select stays flat even unbounded: tanh saturation is not the Q_select bottleneck.
- ✓ Stability holds under `final` reward (unlike `IA_unbound` with `propagate`).
- **Next moves (0429 sweep):** `ME_endgame`, `MF_dataScale`, `NA_dropout`, `NB_asymLR`.

---

## ME_endgame — Endgame Anchor Buffer (ENDGAME_FRACTION sweep)

**Hypothesis:** Maintaining a separate endgame replay buffer (N=2 experience) alongside a curriculum buffer expanding from N=2 to N=4 prevents catastrophic forgetting of the terminal-state Q_place anchor while simultaneously exposing the model to mid-game states.

**Renamed from "Ad_endgame"** — the pending section was registered pre-M-series. Since this runs on the decoupled-autoreg code (M), the correct series letter is **ME** (next after MD_unbound).

**Sweep:** `ENDGAME_FRACTION` ∈ {0.25, 0.5, 0.75} — fraction of each training batch drawn from the N=2 endgame buffer. Remaining batch fraction comes from the curriculum buffer (N=2→4, linearly).

| Run | ENDGAME_FRACTION | N_LAST_STATES_INIT | N_LAST_STATES_FINAL | Epochs |
|-----|---|----|-----|--------|
| ME_endgame(1) | 0.25 | 2 | 4 | 5000 |
| ME_endgame(2) | 0.50 | 2 | 4 | 5000 |
| ME_endgame(3) | 0.75 | 2 | 4 | 5000 |

**Fixed:** `ARCHITECTURE=QuartoCNNAutoreg`, `TRANSITION_SCHEMA="decoupled_autoreg"`, `DECOUPLED_TARGET_STYLE="td_place_mc_select"`, `REWARD_FUNCTION="final"`, `STARTING_NET=None`, `LR=7e-4`, `TAU=0.01`, `EPOCHS=5000`, `MATCHES_PER_EPOCH=32`.

**Compare to:** MB_final(1) (N=2, 66.9% WR) and MB_final(3) (N=4, 56.2% WR).

**Result:**

| Run | ENDGAME_FRACTION | last-100 WR vs bot_loss-BT | last-100 WR vs bot_random | peak WR vs BT |
|-----|---|---|---|---|
| ME(1) | 0.25 | **69.8%** | 85.7% | 93.3% |
| **ME(2)** | **0.50** | **73.4%** | **85.8%** | **96.7%** |
| ME(3) | 0.75 | 68.6% | 83.9% | 93.3% |

ME(2) is the **new WR champion** — first genuine improvement over MB(1) since the M-series began (+6.5pp last-100 BT, +5.1pp random).

**Key diagnostic findings (qv panels):**

1. **Q_place at N=2 (endgame buffer): clearly bimodal throughout.** Outcome=−1 band stays near −1; Outcome=+1 band stays near +1. The endgame anchor successfully maintains the terminal-state value anchor that catastrophic forgetting destroyed in Ac_fineShallow.
2. **Q_place in the curriculum (N=2→4): transitions but does not fully separate.** Beyond epoch ~1300 when N begins to grow, Q_place loses the tight bimodal structure. Still better than MB at the same N.
3. **Q_select: both Outcome=−1 and Outcome=+1 panels show data, but Q_select never separates by outcome.** Both classes cluster in [−0.75, −0.55] (dark on the colormap), making them visually indistinguishable. The "No valid samples" text in the figure appears only for the Outcome=0 column (draws), which is expected — no draws occur vs a random bot. Schema confirmed correct: empirical check at N=3 gives 200/400 winner-select transitions; at N=4, 200/600. The head is learning a weak signal (Δ≈0.18 between classes) but converges to a negative attractor rather than the expected ±1 bimodal distribution.
4. **The horizon plot confirms Q_place carries the WR lift.** Q_place Outcome=−1 bands at steps=0/1 near −1 and Outcome=+1 bands near +1 — the cleanest horizon separation in the M-series. Q_select Outcome=+1 column renders but both win/lose bands sit between −0.75 and −0.55, with no outcome-conditional structure.
5. **Grad norm:** stays below clip throughout. The endgame anchor slightly reduces the high-N instability seen in MA.

**Conclusion:**
- ✓ Endgame anchoring works. FRACTION=0.5 is the sweet spot — 25% is too weak to prevent forgetting, 75% dilutes curriculum signal too much.
- ✓ New champion: ME(2) at 73.4% last-100 / 96.7% peak vs bot_loss-BT, 85.8% last-100 vs random. **ME(2) becomes the new baseline for the Ne series.**
- ✗ Q_select still weak. Win rate gain is entirely Q_place-driven.
- 🔎 **Root cause of Q_select weakness: target class imbalance.** "No valid samples" in the figures is for Outcome=0 (draws) only — expected. Q_select+Outcome=+1 DOES render data but both outcome classes cluster in [−0.75, −0.55], visually dark and indistinguishable. Direct Q-value measurement on ME(2) at E_5000: `sel_neg mean=−0.74`, `sel_pos mean=−0.56` (Δ≈+0.18 — weak but nonzero; Q_place: place_neg mean=−0.25 vs place_pos mean=+0.33). Cause: (i) the entire N=2 curriculum phase (~2500 epochs) and permanent endgame buffer supply only loser-select transitions; (ii) at N=4, 2 loser selects per 1 winner select per match; (iii) winner selects are further from terminal (steps=3 vs 1) → MC target γ³≈0.97 vs −γ¹≈−0.99, weaker signal per sample.
- 📊 **Known `plot_Qv_progress` limitation:** the function clips all epochs to `min_size_across_epochs` (epoch-0 batch size ≈ 96 at N=2). In later epochs with N=4 (224 samples), only the first 96 positions are tracked. Moreover, position index i refers to a different physical sample in each epoch (exp is regenerated each epoch), so individual trajectories are not meaningful — only the per-group mean is informative. This limitation does not affect training correctness.
- **Checkpoint:** `CHECKPOINTS/ME_endgame(2)0429_ENDGAME_FRACTION_0.5/20260507_0829-ME_endgame(2)0429_ENDGAME_FRACTION_0.5_E_5000.pt`

---

## MF_dataScale — Data Scaling at N=4 and N=6

**Hypothesis:** Q_place's flat-band failure at N≥3 is variance-limited estimation: the MC return target `γ^k · outcome` has high variance with only 32 matches/epoch, and more data per epoch narrows the gradient noise enough for the model to find the outcome-conditional bands.

**Code change vs MB:** only `MATCHES_PER_EPOCH` changes (×4 = 128, ×10 = 320). Epochs reduced proportionally to fit a 3-day run window.

| Run | N | MATCHES_PER_EPOCH | Epochs |
|-----|---|---|--------|
| MF_dataScale(1) | 4 | 128 (×4) | 3000 |
| MF_dataScale(2) | 4 | 320 (×10) | 1500 |
| MF_dataScale(3) | 6 | 128 (×4) | 2000 |
| MF_dataScale(4) | 6 | 320 (×10) | 1000 |

**Fixed:** Identical to corresponding MB runs except `MATCHES_PER_EPOCH`. `ARCHITECTURE=QuartoCNNAutoreg`, `REWARD_FUNCTION="final"`, `LR=7e-4`, `TAU=0.01`, `ENDGAME_FRACTION=0`.

**Compare to:** MB_final(3) (N=4, 32 matches, 56.2% WR) and MB_final(4) (N=6, 32 matches, 43.8% WR).

**Result:**

| Run | N | MATCHES | last-100 WR vs BT | last-100 WR vs rand | MB equivalent |
|-----|---|---|---|---|---|
| MF(1) | 4 | 128 (×4) | 64.8% | 80.1% | MB(3): 56.2% |
| **MF(2)** | **4** | **320 (×10)** | **65.5%** | **82.0%** | MB(3): 56.2% |
| MF(3) | 6 | 128 (×4) | 45.1% | 65.9% | MB(4): 43.8% |
| MF(4) | 6 | 320 (×10) | 47.7% | 68.1% | MB(4): 43.8% |

**Key diagnostic findings:**

1. **N=4 — variance hypothesis partially confirmed.** MF(1) and MF(2) both gain ~8–9pp vs MB(3), with faster convergence (~500 epochs vs ~2000 for the 32-match baseline). More data narrows the MC gradient noise enough to accelerate learning at N=4. MF(2) (320 matches, 1500 epochs) converges best and earliest.
2. **N=4 qv_horizon: Q_place bands narrow but do not diverge by outcome.** Both Outcome=−1 and Outcome=+1 bands are near 0, stable, and overlapping. The WR gain at N=4 is driven by reduced variance in the Q_place training signal (cleaner gradients per epoch), not by the model actually learning to separate winning from losing states. The network exploits local patterns better but does not develop temporal value understanding.
3. **N=6 — variance scaling does not help.** MF(3) and MF(4) gain only 1–4pp over MB(4), well within noise. The N-cliff is structural at N=6: more data per epoch cannot overcome the credit-assignment problem at this horizon depth.
4. **Q_select: flat in all four panels, all N values.** Confirmed: data scaling does not revive the select head.
5. **Ceiling: MF(2) at 65.5% trails ME(2) at 73.4%** — endgame anchoring with the curriculum (ME) outperforms pure data scaling at fixed N. The endgame buffer approach is the more efficient intervention.

**Conclusion:**
- ✓ Variance-limited estimation at N=4 is real — 10× data gives a meaningful convergence speedup and ~9pp WR lift.
- ✗ Not the root cause: Q_place bands remain horizon-flat even with 320 matches; the model learns faster but does not learn temporal values.
- ✗ N=6 cliff is structural: more data buys nothing at this horizon.
- ✗ Q_select unaffected by data scaling: target-quality problem, not a noise/sample problem.

---

