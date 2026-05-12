# Series F & G — Bellman variants (combined_avg → separate_bellman)

`FA_Bellman` introduced an adversarial sign flip that diverged; reverted. `GA_Bellman` switched to a per-head Bellman loss. See parent: `Research-status.md`.

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

