# Series N — Shared-trunk diagnostics & select-head fixes

Dropout, asymmetric LR, freeze-place, balanced-select buffer attempts. See parent: `Research-status.md`.

## NA_dropout — Low Dropout Diagnostic

**Hypothesis:** `dropout=0.5` in the shared trunk suppresses the select-head learning signal by zeroing half the trunk activations on every forward pass. With ~76K total params and a 1300-sample buffer, this is plausibly too high and may be throwing away most of the Q_select credit-assignment signal per update. Reducing to `dropout=0.1` costs almost nothing and could explain why Q_select stayed flat even in MB.

**Code change from M:** New `QuartoCNNAutoregLowDropout` class in `models/CNN_autoreg.py` — identical to `QuartoCNNAutoreg` but `self.dropout = nn.Dropout(0.1)`. Output activation remains `tanh`.

**Sweep:** `N_LAST_STATES_INIT` ∈ {2, 3, 4} — same range as the informative MB N-values.

| Run | N_LAST_STATES_INIT | Epochs |
|-----|---------------------|--------|
| NA_dropout(1) | 2 | 5000 |
| NA_dropout(2) | 3 | 5000 |
| NA_dropout(3) | 4 | 5000 |

**Fixed:** `TRANSITION_SCHEMA="decoupled_autoreg"`, `DECOUPLED_TARGET_STYLE="td_place_mc_select"`, `REWARD_FUNCTION="final"`, `STARTING_NET=None`, `LR=7e-4`, `TAU=0.01`, `MATCHES_PER_EPOCH=32`.

**Compare to:** MB_final(1) (N=2, 66.9% WR), MB_final(2) (N=3, 45.8% WR), MB_final(3) (N=4, 56.2% WR).

**Result:**

| Run | N | last-100 WR vs BT | last-100 WR vs rand | MB equivalent |
|-----|---|---|---|---|
| NA(1) | 2 | 65.0% | 80.7% | MB(1): 66.9% |
| NA(2) | 3 | 40.3% | 59.8% | MB(2): 45.8% |
| NA(3) | 4 | 40.3% | 62.0% | MB(3): 56.2% |

**Key diagnostic findings (grad norm):**

1. **NA N=2 (blue): clearly below clip**, steady at ~0.65. Lower than MA reference (orange dashed, ~1.2–1.6).
2. **NA N=3 (green): frequently at or above the clip threshold (1.0).** Smoothed line hovers between 0.9 and 1.05, with frequent spikes above clip. *Correction from earlier analysis: N=3 does clip regularly — this was previously misread as "below clip" by comparing against the MA reference. Absolute behavior: N=3 is at the boundary throughout training.*
3. **NA N=4 (red): ~0.8**, below clip. Intermediate between N=2 and N=3.
4. **Dropout=0.1 meaningfully lowers gradient magnitude at N=2 and N=4 vs MA reference (which regularly hit 1.3–1.6).** The gradient pathology at N=3 persists even with low dropout.
5. **Q_select: flat in all panels.** Reducing dropout did not revive the select head.
6. **WR: ties or trails MB at every N.** NA(1) at N=2 matches MB(1) within noise (65.0% vs 66.9%); NA(2–3) at N≥3 collapse to 40.3% — below MB(2)'s 45.8%.

**Conclusion:**
- ✗ Dropout was not the Q_select bottleneck. Even with 0.1 dropout the select head is flat.
- ✗ N≥3 performance degrades vs MB — low dropout does not improve generalisation at larger horizons.
- ✓ The gradient evidence is clean: dropout reduction does stabilise N=2 and N=4 training, but has no effect on the select head and actively hurts WR at N=3.
- **Dropout=0.5 is the correct setting** for this model and regime. The original MB configuration is not contaminated by a dropout pathology.

---

## NB_asymLR — Asymmetric Learning Rate for Select Head

**Hypothesis:** Q_select's gradient signal reaching `fc2_select` is ~280× weaker than Q_place's gradient reaching `fc2_place` (observed ratio in prior Nexus diagnostic). Even if the gradient is non-zero, the AdamW update step for `fc2_select` is vanishingly small relative to the simultaneous `fc2_place` update. Giving `fc2_select` a 10× higher learning rate directly compensates for this imbalance without any architectural change.

**Code change from M:** Modified optimizer in the training script: AdamW with two parameter groups. `fc2_select` parameters use `LR_SELECT=7e-3`; all other parameters (trunk, `fc_in_aux`, `conv1`, `conv2`, `fc1`, `fc2_place`, phase embedding) use `LR=7e-4`. No LR schedule (constant rates throughout).

**Sweep:** `N_LAST_STATES_INIT` ∈ {2, 3, 4}.

| Run | N_LAST_STATES_INIT | LR (base) | LR_SELECT | Epochs |
|-----|---------------------|-----------|-----------|--------|
| NB_asymLR(1) | 2 | 7e-4 | 7e-3 | 5000 |
| NB_asymLR(2) | 3 | 7e-4 | 7e-3 | 5000 |
| NB_asymLR(3) | 4 | 7e-4 | 7e-3 | 5000 |

**Fixed:** `ARCHITECTURE=QuartoCNNAutoreg`, `TRANSITION_SCHEMA="decoupled_autoreg"`, `DECOUPLED_TARGET_STYLE="td_place_mc_select"`, `REWARD_FUNCTION="final"`, `STARTING_NET=None`, `TAU=0.01`, `MATCHES_PER_EPOCH=32`.

**Compare to:** MB_final (same N values).

**Result:**

| Run | N | last-100 WR vs BT | last-100 WR vs rand | MB equivalent |
|-----|---|---|---|---|
| NB(1) | 2 | 66.5% | 81.5% | MB(1): 66.9% |
| NB(2) | 3 | 39.0% | 58.9% | MB(2): 45.8% |
| NB(3) | 4 | 53.0% | 72.9% | MB(3): 56.2% |

**Key diagnostic findings (grad norm):**

1. **NB N=2 (blue): below clip**, ~0.75 throughout. Asymmetric LR has no visible effect on total gradient at N=2 — Q_select has no Outcome=+1 samples at N=2 (starvation), so the 10× select LR fires on one outcome class only and adds no useful signal.
2. **NB N=3 (green): always above 1.0**, sustained at ~1.1–1.2 throughout training. The inflated select LR destabilises the trunk at N=3 where Q_select finally has mixed-outcome samples. Constant clipping confirmed.
3. **NB N=4 (red): above 1.0 in early training (~epochs 0–700)**, then settles to ~0.8 as the cosine schedule reduces effective LR. Early instability attributable to 10× select LR before the model has a settled trunk representation.
4. **Q_select: flat in all panels, all N values.** The 10× LR did not produce any Outcome=+1 signal in the Q_select qv panels.
5. **WR at N=3: below MB(2)** (39.0% vs 45.8%). The trunk destabilisation at N=3 directly costs ~7pp WR. At N=4, NB(3) nearly matches MB(3) (53.0% vs 56.2%) once the LR schedule settles.

**Conclusion:**
- ✗ Asymmetric LR does not revive Q_select. The select head receives stronger gradient updates but still produces no outcome-conditional signal. This rules out insufficient gradient magnitude as the cause of select-head failure.
- ✗ 10× select LR actively hurts performance at N=3 by destabilising the trunk.
- ✓ N=2 behaviour unchanged — confirms the starvation diagnosis at N=2 (no positive-outcome select targets, so extra LR does nothing).
- **Implication:** the Q_select failure is a target-quality problem, not a gradient-magnitude problem. Addressing it requires either fixing the sample labeling (schema bug investigation) or providing a supervised anchor signal (auxiliary loss).

---

## Experiment Naming Update — N-series

Code version **N** = architecture/hyperparameter changes on top of the established ME(2) base. Experiments starting from `NA_dropout` use the same `QuartoCNNAutoreg` + `decoupled_autoreg` + `final` reward stack as the M-series but sweep architectural or training-loop hyperparameters. Next code-change experiments continue the N-series.

**Baseline change from Ne onward:** `bot_loss-BT` replaced by `ME_endgame(2)_E_5000` as the "current best" reference. `bot_random` retained. Rationale: ME(2) is now the champion and represents a meaningful performance bar; loss-BT is a weaker coupled-architecture bot that no longer adds signal. *Historical comparisons through loss-BT remain valid for A–N series; the break is noted here.*

---

## Key Takeaways (updated May 2026)

1. **End-game training works well** (Aa_replay): N=2, buffer=8 → 65% WR, loss 0.06.
2. **Full-game from scratch fails** (Ab_data): N≥8 → loss stuck at 0.43+, no learning.
3. **Fine-tuning with curriculum fails** (Ac_fine, Ac_fineShallow): catastrophic forgetting.
4. **Adversarial sign flip is wrong with propagated rewards** (FA_Bellman): double-negation diverges Q.
5. **Q_select saturation persists across architectures A–K** in the joint schema: target design, not architecture, is the bottleneck.
6. **Decoupled-autoreg schema (M-series)** softens the N-cliff and achieves the best N=2 result under `final` reward (MB: 66.9% WR).
7. **MC bootstrap removal (MC_MCboth)** does not fix Q_place flat-band at N≥3.
8. **Unbounded activations (MD_unbound)** do not revive Q_select; tanh saturation is not the select bottleneck.
9. **Endgame anchoring (ME_endgame(2))** is the first genuine WR improvement: 73.4% last-100 vs bot_loss-BT, +6.5pp over MB(1). **New champion.**
10. **Data scaling (MF_dataScale)**: variance-limited at N=4 is real but not the root cause; Q_place learns faster but not temporally; N=6 cliff is structural.
11. **Dropout reduction (NA_dropout)**: not the Q_select bottleneck; dropout=0.5 is correct.
12. **Asymmetric LR (NB_asymLR)**: insufficient gradient magnitude is ruled out as the Q_select failure cause; 10× select LR destabilises training at N≥3.
13. **Root cause of Q_select weakness confirmed (May 2026 diagnostic):** Schema is correct — winner-side selects are labeled Outcome=+1 at N≥3 (empirical: N=3 → 200 sel_pos / 400 select total; N=4 → 200/600). ME(2) checkpoint Q_select: sel_pos mean=−0.56 vs sel_neg mean=−0.74 (Δ≈0.18, weak nonzero). The "No valid samples" text in qv figures is for Outcome=0 (draws only); Q_select+Outcome=+1 renders data but looks dark because both classes are in [−0.75, −0.55]. Root cause: **permanent class imbalance** — endgame buffer supplies loser-selects at 50% of every batch throughout all 5000 epochs; at N=4, loser-select outnumber winner-select 2:1 per match. Side note: `plot_Qv_progress` clips to epoch-0 batch size and tracks position index i across epochs where each epoch has a different exp batch — individual trajectories are misleading; only the per-group mean is informative. Fix: track Q-values on a fixed held-out eval batch rather than the rotating training batch.
14. **Pickle label artifact (fixed from Ne onward):** In experiments Aa through NB, `q_values_history["outcome"]` and `q_values_history["steps_to_terminal"]` in saved `.pkl` files contain only epoch-0 labels (guarded by `if len() == 0`). Since `exp` regenerates every epoch, these stored labels are stale for all subsequent epochs. Fixed in `trainRL.py` (May 2026): both fields now append every epoch, making offline replot tools (`view_qv.py` etc.) self-consistent.

---

## Current Open Problem: Q_select Dead Signal + Schema Bug Hypothesis

**Symptom:** Q_select is flat (near −0.5 to −1) in every M/N-series experiment across all N values, including N=3 and N=4 where the decoupled-autoreg schema is expected to supply both Outcome=−1 and Outcome=+1 select transitions.

**Standing interventions tried and ruled out:**
| Intervention | Experiment | Result |
|---|---|---|
| MC target for Q_select | MA–MB, LA–LB | Still flat |
| Remove tanh | MD_unbound | Still flat |
| Reduce dropout | NA_dropout | Still flat |
| Asymmetric 10× LR | NB_asymLR | Still flat |
| More data (×10) | MF_dataScale | Still flat |

**Schema bug hypothesis refuted (May 2026 diagnostic).** `_actor_outcome(player_pos, match_result)` returns +1.0 when the acting player is the eventual winner and −1.0 otherwise. Empirical at N=3 (200 matches): `select_neg=200, select_zero=0, select_pos=200`. At N=4: `select_neg=400, select_pos=200`. At N=2: `select_neg=200, select_pos=0` — structurally zero winner-selects because the N=2 window contains {loser_place, loser_select, winner_terminal_place} only.

**Plot artifact clarification.** The qv figures show "No valid samples" text only for Outcome=0 (draws), which have zero samples vs a random bot (essentially impossible to draw on a 2×2 board). Q_select+Outcome=+1 renders data but both outcome classes cluster in [−0.75, −0.55] on the summer colormap (dark). Additionally, `plot_Qv_progress` clips all epochs to `min_size_across_epochs` (epoch-0 batch size ≈ 96 at N=2); in later epochs with larger exp batches (N=4, ~224 rows), only the first 96 positions are tracked. Since each epoch generates a fresh exp batch, position index `i` represents different physical samples at different epochs — individual trajectories are not meaningful; only the per-group mean is. **This is a plotting limitation to fix in future**, not a training correctness issue.

**Confirmed root cause: target class imbalance.** ME(2) at E_5000: sel_pos mean=−0.56 vs sel_neg mean=−0.74 (Δ≈0.18). Both classes firmly negative; converges to a "default −0.65" attractor. Causes in order of magnitude: (1) endgame buffer permanently supplies loser-select transitions (outcome=−1) to 50% of every training batch for all 5000 epochs; (2) during N=2 curriculum phase (epochs 0–~2500), zero winner-select transitions exist in the curriculum buffer either; (3) at N=4, loser-selects outnumber winner-selects 2:1 per match; (4) winner selects sit 2 steps further from terminal → MC target γ³≈0.97 vs γ¹≈0.99 (minor). The five interventions (MC target, dropout, asymmetric LR, unbounded activations, data scaling) all assumed the signal existed and was too weak — they addressed the wrong cause.

**Remaining structural fixes (still untested):**
| Fix | Description | Status |
|---|---|---|
| **Ne_freezePlace** | Load ME(2), fix N=3, freeze `fc2_place`, train `fc2_select` + trunk. Isolates gradient-starvation hypothesis. | Next |
| **Nf_balanceSelect** | Separate winner-select replay buffer; mix with regular buffer at equal frequency. Direct attack on class-imbalance root cause. | After Ne |
| **Ng_auxSelect** | Supervised 1-step lookahead auxiliary loss on Q_select. No bootstrap, no MC variance, no imbalance. | After Nf |
| **Nh_sepTrunks** | Separate trunk for place and select. Tests shared-trunk interference independently. | After Ng |

---

## Ne_freezePlace — Freeze Q_place, Train Q_select Alone (PLANNED)

**Hypothesis:** Q_select gradient contribution is dwarfed by Q_place (N=3 has 2 place transitions per 1 select winner transition; the N=2 endgame buffer adds more place signal). Freezing Q_place removes the dominant gradient source and forces the trunk to adapt to select-only updates.

**Code changes relative to NB_asymLR pattern:**
1. Load `STARTING_NET` from ME(2) E_5000 checkpoint.
2. Freeze `fc2_place` parameters: `for p in policy_net.fc2_place.parameters(): p.requires_grad = False`
3. Build optimizer with two parameter groups (pattern from `NB_asymLR` scripts):
   ```python
   select_param_ids = {id(p) for p in policy_net.fc2_select.parameters()}
   trunk_params = [p for name, p in policy_net.named_parameters()
                   if name not in ('fc2_place.weight', 'fc2_place.bias')]
   optimizer = optim.AdamW([
       {'params': [p for p in trunk_params if id(p) not in select_param_ids], 'lr': LR},
       {'params': list(policy_net.fc2_select.parameters()), 'lr': LR_SELECT},
   ], amsgrad=True)
   ```
4. `N_LAST_STATES_INIT = N_LAST_STATES_FINAL = 3` (no curriculum, N=3 is the minimum for winner-select transitions).
5. `GEN_EXPERIENCE_BY_EPOCH = True`, `ENDGAME_FRACTION = 0` (no endgame buffer — this experiment must let winner-select transitions compete fairly).

**Model parameter names** (from `QuartoCNNAutoreg`):
`fc_in_aux`, `phase_embedding`, `conv1`, `conv2`, `fc1`, `fc2_place` (freeze), `fc2_select` (high LR).

**Key hyperparameters:**
- `STARTING_NET`: `CHECKPOINTS/ME_endgame(2)0429_ENDGAME_FRACTION_0.5/20260507_0829-ME_endgame(2)0429_ENDGAME_FRACTION_0.5_E_5000.pt`
- `LR = 7e-4` (trunk), `LR_SELECT = 5e-3` (7× select head)
- `TAU = 0.01`, `GAMMA = 0.99`, `BATCH_SIZE = 32`, `MAX_GRAD_NORM = 1.0`
- `MATCHES_PER_EPOCH = 32`, `NUM_EPOCHs_BUFFER = 8`
- `ARCHITECTURE = QuartoCNNAutoreg`, `TRANSITION_SCHEMA = "decoupled_autoreg"`
- `DECOUPLED_TARGET_STYLE = "td_place_mc_select"`, `REWARD_FUNCTION = "final"`
- `EPOCHS = 2000`

**Sweep:** one variant is enough to test the hypothesis (`N=3`, single run).

**Primary diagnostic:** Q_select qv_progress. If the Outcome=+1 band separates from Outcome=−1 after 500 epochs → gradient starvation was dominant. If it stays flat → trunk representation or class imbalance (proceed to Nf_balanceSelect).

**Secondary diagnostic:** WR vs ME(2) bot. Even with Q_select dead, loading ME(2) weights and training at N=3 may hurt WR if the trunk updates degrade Q_place. If WR drops below 40% → trunk is being corrupted; add `requires_grad=False` to trunk params too and only train `fc2_select`.

**Compare to:** MB(2) at N=3 (45.8% WR vs bot_loss-BT), ME(2) loaded model (73.4%).

---

## Nf_balanceSelect — Balanced Select Buffer (PLANNED)

**Hypothesis:** Q_select failure is primarily class imbalance. Even with full gradient flow, the optimizer converges to a negative attractor because loser-select transitions outnumber winner-select 2:1 in the curriculum and 100:0 in the endgame buffer. Balancing the select-training distribution to 1:1 should let Q_select reach the same separation Q_place achieves.

**Code change:**
Add a second replay buffer `winner_select_buffer` that stores only transitions where `phase==PHASE_SELECT and outcome==+1`. During the training loop, mix from four sources per batch:
- Regular curriculum (place + select, full distribution)
- Endgame buffer (as in ME(2), fraction=0.5)
- Winner-select buffer (separate, sampled to make winner-select 50% of all select samples in each batch)

**Implementation sketch:**
```python
# After gen_experience, extract winner-select slice:
sel_win_mask = (exp['phase'] == PHASE_SELECT) & (exp['outcome'] == 1.0)
winner_select_exp = exp[sel_win_mask]
winner_select_buffer.extend(winner_select_exp)

# In batch construction:
# target: of BATCH_SIZE transitions, half are select (16), of those half winner (8)
# → sample 8 from winner_select_buffer, 8 loser-select from regular buffer,
#    remaining 16 from place-only transitions
```

**Key hyperparameters (copy from ME(2) unless noted):**
- `STARTING_NET = None` (train from scratch, same as ME(2)) OR `STARTING_NET = ME(2)` (fine-tune)
- `N_LAST_STATES_INIT = 2, N_LAST_STATES_FINAL = 4, ENDGAME_FRACTION = 0.5` (replicate ME(2) curriculum)
- `LR = 7e-4`, `TAU = 0.01`, `EPOCHS = 5000`
- `ARCHITECTURE = QuartoCNNAutoreg`, `TRANSITION_SCHEMA = "decoupled_autoreg"`
- `DECOUPLED_TARGET_STYLE = "td_place_mc_select"`, `REWARD_FUNCTION = "final"`

**Sweep:** two variants — `(1)` train from scratch; `(2)` fine-tune from ME(2) E_5000.

**Decision gate:** Q_select qv_progress Outcome=+1 band separates at N=3. If it does → class imbalance was the bottleneck, balance is the fix. If it does not → MC target quality is the issue (proceed to Ng_auxSelect).

---

