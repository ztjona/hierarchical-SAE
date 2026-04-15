# Quarto RL — Experiment Results (A-Series)

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

## Key Takeaways

1. **End-game training works well** (Aa_replay): N=2, buffer=8 → 65% WR, loss 0.06
2. **Full-game from scratch fails** (Ab_data): N≥8 → loss stuck at 0.43+, no learning
3. **Fine-tuning with curriculum fails** (Ac_fine, Ac_fineShallow): catastrophic forgetting of end-game knowledge as new states enter the buffer
4. **Root cause identified:** the replay buffer evicts end-game experience as curriculum expands, destroying the Q-value anchor for Bellman targets
5. **Adversarial sign flip is wrong with propagated rewards** (FA_Bellman): double-negation causes Q-value divergence proportional to chain length. The "propagate" reward function already handles adversarial perspective.

## Next Experiment: FA_Bellman (re-run)

**Hypothesis:** With the independent max fix (correct) and sign flip reverted (was wrong), training with N>2 should match or improve over the original Ab_data results, since the independent max is a genuine improvement.

**Sweep:** `N_LAST_STATES_INIT` ∈ {2, 3, 4, 5, 8, 12, 16}

## Pending: Ad_endgame

**Hypothesis:** Maintaining a separate endgame replay buffer (N=2 experience) alongside the curriculum buffer will prevent catastrophic forgetting.

**Sweep:** `ENDGAME_FRACTION` ∈ {0.25, 0.5, 0.75} — fraction of each training batch drawn from the endgame buffer.
