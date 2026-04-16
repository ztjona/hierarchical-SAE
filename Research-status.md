# Quarto RL — Experiment Results

## Experiment Naming Convention

Experiments use a two-part name: **`XY_description`**

- **First letter (X)** — Code version / significant algorithm change:
  - `A`–`E`: early iterations (combined_avg baseline)
  - `F`: adversarial sign flip in Bellman (FA_Bellman — failed, reverted)
  - `G`: separate_bellman loss approach (GA_Bellman)
  - `H`: separate_bellman + terminal state masking (HA_mask)
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
| **Decouple transitions** | Restructure experience: separate place and select transitions with independent Bellman equations. | High | Pending |
| **Monte Carlo returns for Q_select** | Use actual game outcome as Q_select target instead of bootstrapping through the noisy Q_select head. | Medium | Pending |
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

**Result:** *(pending)*

---

## Pending: Ad_endgame

**Hypothesis:** Maintaining a separate endgame replay buffer (N=2 experience) alongside the curriculum buffer will prevent catastrophic forgetting.

**Sweep:** `ENDGAME_FRACTION` ∈ {0.25, 0.5, 0.75} — fraction of each training batch drawn from the endgame buffer.
