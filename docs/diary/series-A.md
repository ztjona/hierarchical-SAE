# Series A — Replay/Data/Fine-tuning (combined_avg baseline)

Earliest sweeps establishing the baseline. Champion of the series: `Aa_replay(2)0226_NUM_EPOCHs_BUFFER_8` (~65% WR vs `bot_loss-BT`) — used as the starting net for fine-tuning experiments. See `Research-status.md` → naming convention for code-version letters.

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

