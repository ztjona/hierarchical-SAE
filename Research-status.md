# Quarto RL — Experiment Results

## Experiment Naming Convention

Experiments use a two-part name: **`XY_description`**

- **First letter (X)** — Code version / significant algorithm change:
  - `A`–`E`: early iterations (combined_avg baseline)
  - `F`: adversarial sign flip in Bellman (FA_Bellman — failed, reverted)
  - `G`: separate_bellman loss approach (GA_Bellman)
  - `H`: separate_bellman + terminal state masking (HA_mask)
  - `I`: unbounded Q-values — no tanh (IA_unbound)
  - `J`: final-only rewards with standard bounded uncoupled DQN (JA_final)
  - `K`: final-only rewards with coupled place→select architecture (KA_coupled)
  - `L`: Monte Carlo return target on Q_select in the joint pipeline (LA_mcSelect, LB_mcSelect)
  - `M`: decoupled-autoregressive transition schema (`QuartoCNNAutoreg` + `Quarto_autoreg_bot`) with `td_place_mc_select` targets — TD/Bellman on Q_place, Monte Carlo on Q_select. First sweep `MA_tempRegresive` was mis-named (should have been `MA_*`); subsequent sweeps follow the convention (`MB_final`, ...).
- **Second letter (Y)** — Hyperparameter sweep within the same code version:
  - `a`, `b`, `c`, ... for successive sweeps (e.g., Aa_replay, Ab_data, Ac_fine)

Runs within a sweep are numbered: `HA_mask(1)0416_N_LAST_STATES_INIT_2`, where `(1)` is the run index, `0416` is the date (MMDD), and the suffix is the swept parameter value.

---

## Overview

All experiments train a `QuartoCNN_uncoupled` DQN agent for the Quarto board game via self-play.
Evaluation is against two baselines: **bot_loss-BT** (strong) and **bot_random** (weak).

**Shared config across all A-series:**
`BATCH_SIZE=32`, `MATCHES_PER_EPOCH=32`, `TEMPERATURE_EXPLORE=2`, `TEMPERATURE_EXPLOIT=0.1`, `LOSS_APPROACH="combined_avg"`, `REWARD_FUNCTION="propagate"`, `GEN_EXPERIENCE_BY_EPOCH=True`, `GAMMA=0.99`.

---

## Aa_replay — Replay Buffer Size Sweep

**Question:** How many epochs of experience should the replay buffer retain?

| Run | NUM_EPOCHs_BUFFER | REPLAY_SIZE |
|-----|-------------------|-------------|
| Aa_replay(1) | 2 | 128 |
| **Aa_replay(2)** | **8** | **512** |
| Aa_replay(3) | 128 | 8192 |
| Aa_replay(4) | 512 | 32768 |
| Aa_replay(5) | 1024 | 65536 |
| Aa_replay(6) | 64 | 4096 |

**Fixed:** `STARTING_NET=None`, `EPOCHS=5000`, `N_LAST_STATES=2` (no curriculum), `LR=7e-4`, `TAU=0.01`.

**Result:** `NUM_EPOCHs_BUFFER=8` (Aa_replay(2)) achieved the best loss (~0.06) and highest win rates (~65% vs bot_loss-BT, ~80% vs bot_random). Very large buffers (128-1024) converged to ~0.15 loss — stale data dilutes learning. Buffer=2 was too small.

**Conclusion:** Buffer=8 is optimal. **Aa_replay(2) became the base model for fine-tuning experiments.**

---

## Ab_data — Training Data Distribution Sweep

**Question:** Can we train from scratch using deeper game states (not just end-game)?

| Run | N_LAST_STATES_INIT | N_LAST_STATES_FINAL |
|-----|---------------------|---------------------|
| Ab_data(1) | 4 | 4 |
| Ab_data(2) | 12 | 12 |
| Ab_data(3) | 16 | 16 |
| Ab_data(4) | 8 | 8 |

**Fixed:** `STARTING_NET=None`, `EPOCHS=5000`, `NUM_EPOCHs_BUFFER=8`, `LR=7e-4`, `TAU=0.01`. No curriculum (INIT=FINAL).

**Result:** Sharp performance cliff as N_LAST_STATES increases:
- **N=4:** Loss ~0.34, WR ~45% vs bot_loss-BT (learns slowly but improves)
- **N=8:** Loss ~0.43, WR ~30% (stuck)
- **N=12, 16:** Loss ~0.45-0.47, WR ~30% (no learning)

Reference runs (B02replicate, B03_verLR, Aa_replay) with N=2 reached 0.06-0.18 loss and 55-65% WR.

**Conclusion:** Training from scratch with full game data doesn't work — the Bellman bootstrap chain through 8-16 early/mid-game states generates too noisy Q-value targets. End-game-only training (N=2) is far more effective.

---

## Ac_fine — Fine-tuning LR Sweep (Aggressive Curriculum)

**Question:** Can we fine-tune the best Aa_replay(2) model to learn deeper game states?

| Run | LR |
|-----|----|
| Ac_fine(1) | 1e-4 |
| Ac_fine(2) | 3.5e-4 |
| Ac_fine(3) | 5e-4 |
| Ac_fine(4) | 1e-5 |
| Ac_fine(5) | 5e-5 |
| Ac_fine(6) | 5e-6 |
| Ac_fine(7) | 7e-4 |

**Fixed:** `STARTING_NET=Aa_replay(2)` (pretrained on N=2), `EPOCHS=5000`, `N_LAST_STATES_INIT=4→FINAL=16` (curriculum), `TAU=0.01`.

**Result:** All LRs failed. Loss rose from pretrained ~0.06 to 0.40-0.50. Win rates collapsed to ~30% vs bot_loss-BT. Higher LRs collapsed faster; even LR=5e-6 eventually diverged.

**Conclusion:** The curriculum jump from the pretrained distribution (N=2) to N=4→16 was too aggressive. The target network produces garbage Q-values for unseen early-game states, corrupting gradients regardless of LR. This is a **data distribution mismatch** problem, not a learning rate problem.

---

## Ac_fineShallow — Conservative Fine-tuning (Shallow Curriculum)

**Question:** Does starting the curriculum from the pretrained distribution (N=2) with a gentler target (→8) prevent collapse?

| Run | LR |
|-----|----|
| Ac_fineShallow(1) | 1e-4 |
| Ac_fineShallow(2) | 3.5e-4 |
| Ac_fineShallow(3) | 5e-4 |
| Ac_fineShallow(4) | 1e-5 |
| Ac_fineShallow(5) | 5e-5 |
| Ac_fineShallow(6) | 5e-6 |
| Ac_fineShallow(7) | 7e-4 |

**Fixed:** `STARTING_NET=Aa_replay(2)`, `EPOCHS=10000`, `N_LAST_STATES_INIT=2→FINAL=8`, `TAU=0.005` (lower than Ac_fine).

**Result:** Better than Ac_fine but still fails — **every LR eventually catastrophically collapses.** The pattern is a sharp phase transition (loss jumps 0.12→0.35, WR drops 60%→30% abruptly). Timing varies by LR: conservative LRs (5e-6) delay collapse to ~700 epochs, higher LRs collapse immediately. Some runs (LR=1e-4, 5e-4) show temporary recoveries before re-collapsing, indicating catastrophic forgetting of end-game knowledge.

**Conclusion:** Even a gentle curriculum with matching initial distribution causes **catastrophic forgetting**. The problem is not LR, TAU, or curriculum aggressiveness — as the buffer fills with mid-game states, end-game experience gets evicted, and the model loses its foundational knowledge. A mechanism to **retain end-game experience** during curriculum expansion is needed.

---

## FA_Bellman — Adversarial Sign Flip Experiment (FAILED)

**Question:** Does fixing the Bellman equation to account for adversarial self-play (`r - γV` instead of `r + γV`) improve learning across different N_LAST_STATES?

**Commit:** `51b59ba` made two changes to `DQN_training_step`:
1. **Independent max per head** (combined_avg): changed `max((pos + piece) / 2)` → `(max(pos) + max(piece)) / 2`
2. **Adversarial sign flip**: changed `r + γ * max Q(s')` → `r - γ * max Q(s')`

| Run | N_LAST_STATES_INIT |
|-----|---------------------|
| FA_Bellman(1) | 2 |
| FA_Bellman(2) | 3 |
| FA_Bellman(3) | 4 |
| FA_Bellman(4) | 5 |
| FA_Bellman(5) | 8 |
| FA_Bellman(6) | 12 |
| FA_Bellman(7) | 16 |

**Fixed:** `STARTING_NET=None`, `EPOCHS=5000`, `NUM_EPOCHs_BUFFER=8`, `LR=7e-4`, `TAU=0.01`, `REWARD_FUNCTION="propagate"`, `LOSS_APPROACH="combined_avg"`.

**Result:** N=2 performed similarly to previous baselines (~65% WR vs bot_loss-BT, ~80% vs bot_random). All N≥3 progressively worsened — same pattern as Ab_data. Loss scales with N (0.08 at N=2 → 0.48 at N=16).

### Post-mortem: Sign flip was wrong

**Change 1 (independent max) was correct.** Position and piece are independent action spaces; taking max over the averaged logits is mathematically wrong.

**Change 2 (sign flip) was incorrect** given `REWARD_FUNCTION="propagate"`. The "propagate" reward scheme **already encodes the adversarial sign**:

```python
# R=+1 for P1 win, R_2=-R=-1 for P2
reward = [R if i % 2 == 0 else R_2 for i in range(num_states)]
```

P1 states get `+R`, P2 states get `-R`. Adding a second sign flip in the Bellman equation **double-negates**, causing Q-values to diverge.

**Proof** (P1 wins, R=1, γ=0.99):

| State | Original `r + γV` | With sign flip `r − γV` |
|-------|-------------------|------------------------|
| T (terminal) | +1 | +1 |
| T−1 (opponent) | −1 + 0.99(1) = **−0.01** | −1 − 0.99(1) = **−1.99** |
| T−2 (player) | +1 + 0.99(−0.01) = **+0.99** | +1 − 0.99(−1.99) = **+2.97** |
| T−3 (opponent) | −1 + 0.99(0.99) = **−0.02** | −1 − 0.99(2.97) = **−3.94** |

Original: Q-values stay bounded ≈ ±1. Sign-flipped: Q-values diverge as ≈ R/(1−γ) = ±100.

This explains why N=2 still worked (only 2 Bellman steps, limited divergence) while N≥3 failed increasingly.

**Fix applied:** Reverted sign flip back to `r + γV`, kept the correct independent max fix. Re-running FA_Bellman to validate.

---

---

## GA_Bellman — Separate Bellman Loss (N_LAST_STATES sweep)

**Question:** Does training each head with its own independent Bellman target solve the "lazy Q_select head" problem observed across all prior experiments?

**Commit:** `9edad2f` introduced `LOSS_APPROACH="separate_bellman"` in `DQN_training_step`.

| Run | N_LAST_STATES_INIT |
|-----|---------------------|
| GA_Bellman(1) | 2 |
| GA_Bellman(2) | 3 |
| GA_Bellman(3) | 6 |
| GA_Bellman(4) | 12 |
| GA_Bellman(5) | 16 |

**Fixed:** `STARTING_NET=None`, `EPOCHS=5000`, `NUM_EPOCHs_BUFFER=8`, `LR=7e-4`, `TAU=0.01`, `REWARD_FUNCTION="propagate"`, `LOSS_APPROACH="separate_bellman"`.

**Result:** Worse than `combined_avg` baselines. Best run (N=2) reached ~70% WR vs bot_loss-BT and ~85% vs bot_random but with higher loss (~0.10 vs 0.06). N≥3 degraded progressively (same pattern as Ab_data). Q_select still saturates at -1 — the separate loss did not fix the fundamental problem.

**Why it failed:** The separate Bellman approach actually _amplifies_ the Q_select saturation problem:

1. **Terminal state poison:** At terminal states, `action_sel=-1` so `state_sel_action_values = 0` (never set). But `expected_select = reward` (±1). This creates a persistent `SmoothL1(0, ±1)` loss every single epoch — a constant force pushing Q_select outputs toward tanh boundaries.

2. **Tanh gradient vanishing:** The network uses `tanh` activation, so targets at ±1 push logits toward ±∞ where `tanh'(x) ≈ 0`. The head loses ability to learn once saturated.

3. **No gradient masking:** In `combined_avg`, Q_select is implicitly shielded at terminal states (averaged with Q_place which dominates). With `separate_bellman`, Q_select gets the full unmasked ±1 target directly.

---

## Key Takeaways

1. **End-game training works well** (Aa_replay): N=2, buffer=8 → 65% WR, loss 0.06
2. **Full-game from scratch fails** (Ab_data): N≥8 → loss stuck at 0.43+, no learning
3. **Fine-tuning with curriculum fails** (Ac_fine, Ac_fineShallow): catastrophic forgetting of end-game knowledge as new states enter the buffer
4. **Root cause identified:** the replay buffer evicts end-game experience as curriculum expands, destroying the Q-value anchor for Bellman targets
5. **Adversarial sign flip is wrong with propagated rewards** (FA_Bellman): double-negation causes Q-value divergence proportional to chain length. The "propagate" reward function already handles adversarial perspective.
6. **Q_select saturation at -1 is a persistent problem** across all experiments — the select head never learns meaningful Q-values.
7. **Decoupled-autoreg schema needs N≥3 to feed Q_select positive samples** (MA_tempRegresive). N=2 in this schema keeps only `{loser_place, loser_select, winner_terminal_place}`, so Q_select sees `outcome=−1` exclusively. Decoupling softens the N-cliff (N=3 beats joint mc_select by +12pp WR), but Q_select still does not separate by outcome and Q_place develops a `propagate + tanh` unreachable-target floor at ~0.35 loss.

---

## Current Open Problem: Q_select Saturation

**Symptom:** Across ALL experiments (Aa, Ab, FA, GA), the Q_select head saturates at -1 and never produces useful Q-values. Only Q_place learns. The bot effectively selects pieces randomly.

**Root causes identified:**

1. **Terminal state training signal is destructive.** Terminal states have `action_sel=-1` (no selection happened), so `Q_select_pred = 0`. But the Bellman target is `reward = ±1`. This `SmoothL1(0, ±1)` loss occurs every epoch and pushes the tanh activation to saturation, killing gradients.

2. **Temporal credit assignment asymmetry.** Placing a piece on a winning square gets immediate reward. Selecting a piece for the opponent has an effect 2+ turns later. The Bellman chain for Q_select is inherently longer and noisier.

3. **Tanh at the boundary kills learning.** With "propagate" rewards always being ±1, targets land exactly at the tanh saturation boundaries. Once logits drift to ±3, `tanh'(x) ≈ 0` and the head can no longer learn.

4. **The experience tuple couples two different decision types.** Place and select are fundamentally different transitions with different temporal horizons, but they share one reward signal.

**Candidate fixes:**

| Fix | Description | Effort | Status |
|-----|-------------|--------|--------|
| **Mask terminal states from Q_select loss** | Exclude states where `action_sel=-1` from Q_select loss. Similarly mask first-move from Q_place loss. | Low | **HA_mask** (in progress) |
| **Remove tanh from Q_select** | Use unbounded Q-values (standard DQN) or clamp. Prevents gradient vanishing at ±1 boundary. | Low | Pending |
| **Decouple transitions** | Restructure experience: separate place and select transitions with independent Bellman equations. | High | **Implementation in progress** (`exp/decoupled-autoreg`) |
| **Monte Carlo returns for Q_select** | Use actual game outcome as Q_select target instead of bootstrapping through the noisy Q_select head. | Medium | **LA_mcSelect** (in progress) |
| **Asymmetric learning rates** | Higher LR or more gradient steps for Q_select to compensate for weaker signal. | Low | Pending |

---

## HA_mask — Terminal State Masking (N_LAST_STATES sweep)

**Question:** Does masking invalid actions from per-head loss fix Q_select saturation and enable learning from deeper game states?

**Code change:** In `DQN_training_step`, for `separate_bellman` mode, set Bellman target = prediction (0) for masked entries so they contribute zero loss:
- `expected_place[first_move_mask] = 0.0` (no placement on first move)
- `expected_select[final_move_mask] = 0.0` (no selection on terminal states)

This eliminates the persistent `SmoothL1(0, ±1)` signal that was pushing Q_select into tanh saturation.

| Run | N_LAST_STATES_INIT |
|-----|---------------------|
| HA_mask(1) | 2 |
| HA_mask(2) | 3 |
| HA_mask(3) | 4 |
| HA_mask(4) | 6 |
| HA_mask(5) | 10 |
| HA_mask(6) | 16 |

**Fixed:** `STARTING_NET=None`, `EPOCHS=5000`, `NUM_EPOCHs_BUFFER=8`, `LR=7e-4`, `TAU=0.01`, `REWARD_FUNCTION="propagate"`, `LOSS_APPROACH="separate_bellman"`.

**Expected outcome:** Q_select should produce a distribution of values instead of collapsing to -1. N=2 should match or beat Aa_replay(2) baseline (65% WR). N=3–4 should improve over GA_Bellman equivalents.

**Result (partial, 500 epochs due to bug):**
- Loss improved vs GA_Bellman across all N (the mask removed the constant SmoothL1(0,±1) noise).
- N=2: ~48% vs bot_loss-BT, ~72% vs bot_random — learning, but Q_select still at -1. Q_place carries all performance.
- N=4: Some Q_select learning visible — best case among the sweep.
- N≥6: Loss stuck at 0.35+, flat WR.
- N=16: Both Q_place and Q_select saturate at +1 — tanh ceiling. Bellman targets `R + γ*max_Q` exceed [-1,1] range, pushing logits to ±∞.

**Conclusion:** The mask fixed the terminal-state poison signal but exposed the deeper problem: **tanh output activation constrains Q-values to [-1,1], but Bellman targets naturally exceed that range.** Standard DQN uses unbounded Q-values. The N=2 Q_select issue is a broken bootstrap: Q_select at terminal states is masked (never trained), so the penultimate Q_select bootstraps from random/drifting target-net values.

---

## IA_unbound — Unbounded Q-values (N_LAST_STATES sweep)

**Question:** Does removing tanh activation from both heads (unbounded Q-values, standard DQN practice) fix Q_select saturation and enable learning from deeper game states?

**Code change:** New architecture `QuartoCNN_unbound` in `models/CNN_unbound.py` — identical to `QuartoCNN_uncoupled` but with raw linear outputs instead of tanh. Keeps the terminal state masking from HA_mask.

| Run | N_LAST_STATES_INIT |
|-----|---------------------|
| IA_unbound(1) | 2 |
| IA_unbound(2) | 3 |
| IA_unbound(3) | 4 |
| IA_unbound(4) | 6 |

**Fixed:** `STARTING_NET=None`, `EPOCHS=5000`, `NUM_EPOCHs_BUFFER=8`, `LR=7e-4`, `TAU=0.01`, `REWARD_FUNCTION="propagate"`, `LOSS_APPROACH="separate_bellman"`, `ARCHITECTURE=QuartoCNN_unbound`.

**Expected outcome:** Q-values can freely match Bellman targets without gradient vanishing. Q_select should finally show meaningful variation. May need gradient clipping attention since Q-values are unbounded.

**Result:** Failed badly. `comparison_loss_IA_unbound.png` shows strong divergence, especially at `N=2`, and `comparison_win_rate_IA_unbound.png` stays near ~30% against both baselines.

**Interpretation:** The tanh removal was implemented correctly in both theory and code, but it exposed a deeper instability in the learning target rather than solving the task.

- The implementation is correct: `CNN_unbound.py` returns raw linear outputs from `fc2_board` and `fc2_piece`, so this is standard unbounded Q-learning behavior.
- The failure is not evidence that an extra activation layer after `fc2_*` is needed. Adding another nonlinearity before a final linear layer would change optimization dynamics, but it would not fix the target-definition problem.
- With `REWARD_FUNCTION="propagate"`, every state receives `R=±1`, so the Bellman targets remain noisy and can grow under repeated bootstrap/max overestimation.
- More importantly, **Q_select still has no clean supervised anchor**. Q_place is grounded by terminal placement states; Q_select mostly bootstraps from itself.

**Conclusion:** Tanh saturation was a real symptom, but not the main cause. The main issue is the reward / target design for the select branch.

---

## JA_final — Final-Only Reward with Combined DQN (N_LAST_STATES sweep)

**Question:** Does switching from `propagate` to `final` rewards stabilize the Bellman targets and restore useful learning, while keeping the previously best-performing bounded uncoupled architecture?

**Code change:** Revert to `QuartoCNN_uncoupled`, `LOSS_APPROACH="combined_avg"`, and use `REWARD_FUNCTION="final"`.

Rationale:
- `final` gives only the terminal transition a non-zero reward and lets value propagate through the Bellman chain naturally.
- This avoids injecting `±1` into every state, which appears to destabilize both heads under longer horizons.
- `combined_avg` is retained because it was the only regime that previously produced useful behavior at `N=2`, likely by letting Q_select benefit indirectly from Q_place learning.

| Run | N_LAST_STATES_INIT |
|-----|---------------------|
| JA_final(1) | 2 |
| JA_final(2) | 3 |
| JA_final(3) | 4 |
| JA_final(4) | 6 |
| JA_final(5) | 12 |
| JA_final(6) | 16 |

**Fixed:** `STARTING_NET=None`, `EPOCHS=5000`, `NUM_EPOCHs_BUFFER=8`, `LR=7e-4`, `TAU=0.01`, `ARCHITECTURE=QuartoCNN_uncoupled`, `LOSS_APPROACH="combined_avg"`, `REWARD_FUNCTION="final"`.

**Why include `12` and `16`?** They are worth running. The cheaper sweep `{2,3,4,6}` was enough to localize the transition zone, but `12` and `16` tell us whether `final` reward actually changes the long-horizon failure regime or only delays the same collapse.

**Expected outcome:**
- Better-behaved targets than `propagate`
- Q-values should remain bounded by the discounted terminal reward chain
- If the real issue is reward design rather than architecture, `N=3` and `N=4` should improve first

**Result:** Clear improvement over `Ab_data` and `IA_unbound`, but only partial success.

From the summary and Q-value plots:
- `N=2` recovered the known-good regime: final WR `66.4%` vs `bot_loss-BT`, `80.2%` vs `bot_random`, close to `Aa_replay(2)`.
- `N=3` remained viable: `52.7%` vs `bot_loss-BT`, `71.9%` vs `bot_random`.
- `N>=4` still degraded monotonically with horizon; `N=12` and `N=16` had the lowest losses but the worst win rates.
- Q-value plots show `q_select` is still effectively dead at `N=2` and `N=4` (collapsed near `-1` regardless of reward class).
- At `N=16`, `q_select` escapes saturation but still does not separate reward classes meaningfully, so the branch is no longer saturated but is still not informative.

**Conclusion:** `final` reward fixed a real target-instability problem, but did **not** solve select-head learning. The mismatch between low loss and poor win rate at large `N` means the model is fitting easy targets rather than learning a useful policy. This shifts the main hypothesis from reward design alone to **state / architecture mismatch for the select decision**.

---

## KA_coupled — Final Reward with Coupled Place→Select Architecture (N_LAST_STATES sweep)

**Question:** Does conditioning the select head on the board-placement head restore a useful learning signal for piece selection?

**Code change:** Switch from `QuartoCNN_uncoupled` to `QuartoCNN` in `models/CNN1.py`, keeping `LOSS_APPROACH="combined_avg"` and `REWARD_FUNCTION="final"` unchanged.

Rationale:
- In Quarto, selecting the next piece happens **after** placement, on a changed board.
- `QuartoCNN_uncoupled` predicts `q_select` from the same pre-placement latent as `q_place`, which appears structurally misaligned with the game.
- `QuartoCNN` is still approximate, but it at least conditions the piece head on `qav_board`, making the select decision depend on the placement decision.

| Run | N_LAST_STATES_INIT |
|-----|---------------------|
| KA_coupled(1) | 2 |
| KA_coupled(2) | 3 |
| KA_coupled(3) | 4 |
| KA_coupled(4) | 6 |
| KA_coupled(5) | 12 |
| KA_coupled(6) | 16 |

**Fixed:** `STARTING_NET=None`, `EPOCHS=5000`, `NUM_EPOCHs_BUFFER=8`, `LR=7e-4`, `TAU=0.01`, `ARCHITECTURE=QuartoCNN`, `LOSS_APPROACH="combined_avg"`, `REWARD_FUNCTION="final"`.

**Expected outcome:**
- `q_select` should stop being reward-agnostic at small `N`
- If coupling is the missing ingredient, `N=3` and `N=4` should improve first, while `N=12` and `N=16` remain hard
- If there is no improvement, the next step should be a more principled two-stage architecture where the select head sees the post-placement board state directly

**Result:** No meaningful improvement over `JA_final`. Final win rates vs `bot_loss-BT` track `JA_final` to within ~1–2pp at every `N` (see `results/KA_coupled/summary_KA_coupled.md`):

| N | Final WR vs bot_loss-BT | Final WR vs bot_random |
|---|-------------------------|------------------------|
| 2 | 66.7% | 80.1% |
| 3 | 54.0% | 74.4% |
| 4 | 45.2% | 67.6% |
| 6 | 41.1% | 63.4% |
| 12 | 34.2% | 57.4% |
| 16 | 33.9% | 55.7% |

Same monotonic `N`-cliff as every prior experiment. Same low-loss / bad-WR inversion at large `N` (N=16 has the lowest final loss, 0.048, and the lowest win rate). The Q-value plots are diagnostically identical to `JA_final`:

- **N=2:** `q_place` separates reward classes cleanly; `q_select` is a bright band pegged at −1 across R=−1, R=0, and R=+1. Dead, tanh-saturated.
- **N=4:** `q_place` learns weakly; `q_select` still pegged at −1.
- **N=16:** `q_place` collapses to a near-constant ~0.7 regardless of reward class (fitting the mean). `q_select` escapes −1 saturation but sits as a near-constant ~0–0.3 band across all reward classes — no longer saturated, but still uninformative.

**Conclusion:** Coupling the select head on `qav_board` in `QuartoCNN` was not enough to change `q_select`'s behaviour — the head still only sees the pre-placement board through the shared trunk, and, more importantly, **the Q_select pathology reproduces across code versions A, F, G, H, I, J, K**. That is strong evidence the bottleneck is the **target/credit design for the select branch**, not the architecture of its input. Next step: replace the Bellman target on Q_select with a Monte Carlo return (`LA_mcSelect`).

---

## LA_mcSelect — Monte Carlo Return Target on Q_select (N_LAST_STATES sweep)

**Question:** Does supervising Q_select with the actual discounted game outcome — instead of bootstrapping through the noisy Q_select head — restore a useful learning signal for piece selection?

**Code change:**
- New `LOSS_APPROACH="mc_select"` in `QuartoRL/RL_functions.py`:
  - Q_place: standard Bellman target `R + γ · max_a' Q_place(s')`.
  - Q_select: Monte Carlo return `γ^steps_to_terminal · outcome`, with no bootstrap through Q_select.
  - Masking from `HA_mask` preserved (first-move zeroes Q_place loss; terminal zeroes Q_select loss).
- Experience tuple extended with two new fields, computed independently of `REWARD_FUNCTION_TYPE`:
  - `outcome`: ±1 (or 0 for ties) from this state's player's perspective.
  - `steps_to_terminal`: distance from state `i` to the terminal state `T-1`.
- Architecture reverted from `QuartoCNN` (coupled) back to `QuartoCNN_uncoupled`: the `KA_coupled` coupling hypothesis was falsified, and the MC hypothesis is orthogonal to architecture — isolating the target-design change requires the best-known architecture.

Rationale (from `Current Open Problem` → Q_select saturation):
- `q_select` has no clean supervised anchor; under every Bellman variant tried so far it mostly bootstraps from itself through the target net, which stays near-constant and never produces a useful gradient on reward separation.
- Under `mc_select` the Q_select target becomes exactly `γ^k · outcome`, which is bounded in `[−1, 1]` (safe for tanh), independent of bootstrap noise, and carries the reward-class signal to every non-terminal state. In these runs the reward function still remained `propagate`, so the change isolates the select-head target while keeping the place-head Bellman target on the established reward scheme.

| Run | N_LAST_STATES_INIT |
|-----|---------------------|
| LA_mcSelect(1) | 2 |
| LA_mcSelect(2) | 3 |
| LA_mcSelect(3) | 4 |
| LA_mcSelect(4) | 6 |
| LA_mcSelect(5) | 12 |
| LA_mcSelect(6) | 16 |

**Fixed:** `STARTING_NET=None`, `EPOCHS=5000`, `NUM_EPOCHs_BUFFER=8`, `LR=7e-4`, `TAU=0.01`, `GAMMA=0.99`, `ARCHITECTURE=QuartoCNN_uncoupled`, `LOSS_APPROACH="mc_select"`, `REWARD_FUNCTION="propagate"`.

**Expected outcome / decision gate:**
- **Primary diagnostic:** the qv plots at `N=2`. If the R=+1 band of `q_select` sits measurably above the R=−1 band, the hypothesis is confirmed — Q_select's failure across A/F/G/H/I/J/K was a self-bootstrap / anchoring problem, not a representational one.
- If confirmed at small `N`, check whether the `N`-cliff is also softened. `N=3` and `N=4` should improve first; `N=12`/`N=16` likely remain hard (those are the regime `Ad_endgame` is meant to attack, not LA).
- If `q_select` stays flat / unseparated even at `N=2`, the problem is structural to the select transition itself (place and select shouldn't share one reward tuple), and the next move is to decouple the transitions rather than keep iterating on targets.

---

## LB_mcSelect — Joint mc_select Rerun After Transition-Schema Refactor

**Question:** Did the transition-schema refactor and added horizon-plot metadata change the behaviour of the existing joint `mc_select` experiment family?

**Code change vs LA_mcSelect:**
- Added `TRANSITION_SCHEMA` / `DECOUPLED_TARGET_STYLE` plumbing and decoupled-autoreg imports, but left `TRANSITION_SCHEMA="joint"`.
- Kept `ARCHITECTURE=QuartoCNN_uncoupled`, `PLAYER_BOT_CLASS=Quarto_bot`, `LOSS_APPROACH="mc_select"`, and `REWARD_FUNCTION="propagate"`.
- Added `outcome` / `steps_to_terminal` tracking and the horizon Q-value plot.

**What it is not:**
- Not an autoregressive run: the decoupled-autoregressive design note still lists `joint` as the default, and these runs never switched away from it.
- Not a Bellman-select run: `q_select` still uses the Monte Carlo target `γ^k · outcome` in `DQN_training_step`; only `q_place` remains Bellman.

| Run | N_LAST_STATES_INIT |
|-----|---------------------|
| LB_mcSelect(1) | 2 |
| LB_mcSelect(2) | 3 |
| LB_mcSelect(3) | 4 |
| LB_mcSelect(4) | 6 |
| LB_mcSelect(5) | 12 |
| LB_mcSelect(6) | 16 |

**Fixed:** `STARTING_NET=None`, `EPOCHS=5000`, `NUM_EPOCHs_BUFFER=8`, `LR=7e-4`, `TAU=0.01`, `GAMMA=0.99`, `ARCHITECTURE=QuartoCNN_uncoupled`, `TRANSITION_SCHEMA="joint"`, `LOSS_APPROACH="mc_select"`, `REWARD_FUNCTION="propagate"`.

**Result:** Behaviour matches `LA_mcSelect` within small run-to-run noise. N=2 remains the only clearly successful setting, and the same N-cliff reappears for deeper state windows. Any metric differences between the two sweeps are minor and do not suggest a learning-rule change.

**Conclusion:** The old name `MA_autoregresive` was misleading. This family is best understood as a **joint mc_select rerun on refactored training/plumbing code**, not as a decoupled-autoregressive or Bellman-select experiment.

---

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

## Pending: ME_endgame — Endgame Anchor Buffer (ENDGAME_FRACTION sweep)

**Hypothesis:** Maintaining a separate endgame replay buffer (N=2 experience) alongside a curriculum buffer expanding from N=2 to N=4 prevents catastrophic forgetting of the terminal-state Q_place anchor while simultaneously exposing the model to mid-game states.

**Renamed from "Ad_endgame"** — the pending section was registered pre-M-series. Since this runs on the decoupled-autoreg code (M), the correct series letter is **ME** (next after MD_unbound).

**Sweep:** `ENDGAME_FRACTION` ∈ {0.25, 0.5, 0.75} — fraction of each training batch drawn from the N=2 endgame buffer. Remaining batch fraction comes from the curriculum buffer (N=2→4, linearly).

| Run | ENDGAME_FRACTION | N_LAST_STATES_INIT | N_LAST_STATES_FINAL | Epochs |
|-----|---|----|-----|--------|
| ME_endgame(1) | 0.25 | 2 | 4 | 5000 |
| ME_endgame(2) | 0.50 | 2 | 4 | 5000 |
| ME_endgame(3) | 0.75 | 2 | 4 | 5000 |

**Fixed:** `ARCHITECTURE=QuartoCNNAutoreg`, `TRANSITION_SCHEMA="decoupled_autoreg"`, `DECOUPLED_TARGET_STYLE="td_place_mc_select"`, `REWARD_FUNCTION="final"`, `STARTING_NET=None`, `LR=7e-4`, `TAU=0.01`, `EPOCHS=5000`, `MATCHES_PER_EPOCH=32`.

**Compare to:** MB_final(3) (N=4, no endgame anchor, 56.2% WR vs bot_loss-BT). If endgame anchoring works, the N=4 run here should stay closer to MB(1)'s 66.9%.

---

## Pending: MF_dataScale — Data Scaling at N=4 and N=6

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

---

## Pending: NA_dropout — Low Dropout Diagnostic

**Hypothesis:** `dropout=0.5` in the shared trunk suppresses the select-head learning signal by zeroing half the trunk activations on every forward pass. With ~76K total params and a 1300-sample buffer, this is plausibly too high and may be throwing away most of the Q_select credit-assignment signal per update. Reducing to `dropout=0.1` costs almost nothing and could explain why Q_select stayed flat even in MB.

**Code change from M:** New `QuartoCNNAutoregLowDropout` class in `models/CNN_autoreg.py` — identical to `QuartoCNNAutoreg` but `self.dropout = nn.Dropout(0.1)`. Output activation remains `tanh`.

**Sweep:** `N_LAST_STATES_INIT` ∈ {2, 3, 4} — same range as the informative MB N-values.

| Run | N_LAST_STATES_INIT | Epochs |
|-----|---------------------|--------|
| NA_dropout(1) | 2 | 5000 |
| NA_dropout(2) | 3 | 5000 |
| NA_dropout(3) | 4 | 5000 |

**Fixed:** `TRANSITION_SCHEMA="decoupled_autoreg"`, `DECOUPLED_TARGET_STYLE="td_place_mc_select"`, `REWARD_FUNCTION="final"`, `STARTING_NET=None`, `LR=7e-4`, `TAU=0.01`, `MATCHES_PER_EPOCH=32`.

**Compare to:** MB_final(1) (N=2, 66.9% WR), MB_final(2) (N=3, 45.8% WR), MB_final(3) (N=4, 56.2% WR). The qv panel at N=2 is the primary diagnostic: if Q_select shows any outcome separation under low dropout, the high-dropout hypothesis is confirmed for the N=2 regime.

---

## Pending: NB_asymLR — Asymmetric Learning Rate for Select Head

**Hypothesis:** Q_select's gradient signal reaching `fc2_select` is ~280× weaker than Q_place's gradient reaching `fc2_place` (observed ratio in prior Nexus diagnostic). Even if the gradient is non-zero, the AdamW update step for `fc2_select` is vanishingly small relative to the simultaneous `fc2_place` update. Giving `fc2_select` a 10× higher learning rate directly compensates for this imbalance without any architectural change.

**Code change from M:** Modified optimizer in the training script: AdamW with two parameter groups. `fc2_select` parameters use `LR_SELECT=7e-3`; all other parameters (trunk, `fc_in_aux`, `conv1`, `conv2`, `fc1`, `fc2_place`, phase embedding) use `LR=7e-4`. No LR schedule (constant rates throughout).

**Sweep:** `N_LAST_STATES_INIT` ∈ {2, 3, 4}.

| Run | N_LAST_STATES_INIT | LR (base) | LR_SELECT | Epochs |
|-----|---------------------|-----------|-----------|--------|
| NB_asymLR(1) | 2 | 7e-4 | 7e-3 | 5000 |
| NB_asymLR(2) | 3 | 7e-4 | 7e-3 | 5000 |
| NB_asymLR(3) | 4 | 7e-4 | 7e-3 | 5000 |

**Fixed:** `ARCHITECTURE=QuartoCNNAutoreg`, `TRANSITION_SCHEMA="decoupled_autoreg"`, `DECOUPLED_TARGET_STYLE="td_place_mc_select"`, `REWARD_FUNCTION="final"`, `STARTING_NET=None`, `TAU=0.01`, `MATCHES_PER_EPOCH=32`.

**Compare to:** MB_final (same N values). If Q_select shows outcome separation in the N=2 qv panel → gradient starvation at the output layer was the dominant cause. If Q_select stays flat even at 10× LR → the starvation is deeper in the trunk, and the next hypothesis is separate trunks for place and select.
