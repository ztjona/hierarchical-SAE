# Series H, I, J, K — terminal masking / unbounded / final-only / coupled

Sequential code-version iterations on the joint pipeline before the M-series autoreg refactor. See parent: `Research-status.md`.

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

