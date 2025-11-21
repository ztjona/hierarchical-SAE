# E05 Hyperparameter Search - Experiment Plan

## Executive Summary

**Goal:** Find optimal `MATCHES_PER_EPOCH` and `BATCH_SIZE` for stable, improving Quarto RL training.

**Problem:** E04b experiments showed performance degradation over time, even with stable loss (N=2).

**Root Cause:** Starting from pre-trained checkpoint (epoch 22) + pure self-play creates "strategy bubble"

**Solution:** Start from random weights (AlphaZero approach) + test different game coverage levels

---

## Theoretical Foundation

### 1. **AlphaZero Self-Play Paradigm**

**Citation:** Silver, D., et al. (2017). "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm." *arXiv preprint arXiv:1712.01815*.

**Key principles:**
- ✓ Start from **random initialization** (no pre-training)
- ✓ Pure self-play (both players use same evolving network)
- ✓ Massive scale (millions of games, thousands of gradient updates)
- ✓ MCTS for move selection (we use temperature sampling instead)

**Why it works:**
- Both players start equally weak → co-evolve together
- No pre-existing biases to reinforce
- Symmetric improvement prevents strategy collapse

---

### 2. **Deep Q-Networks (DQN) Framework**

**Citation:** Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*, 518(7540), 529-533.

**Components:**
- Experience replay buffer (decorrelates samples)
- Target network (stabilizes training via soft updates)
- Gradient clipping (prevents exploding gradients)
- Huber loss (robust to outliers)

**Our implementation:**
```python
TAU = 0.01          # Soft update coefficient
GAMMA = 0.99        # Discount factor
MAX_GRAD_NORM = 1.0 # Gradient clipping threshold
LR = 5e-5 → 1e-5    # Cosine annealing schedule
```

---

### 3. **State Space Coverage Analysis**

**Quarto Complexity:**
- Total states: ~10^8 (much smaller than Go's ~10^170)
- Average game: 8 moves = 16 half-moves
- With n_last_states=2: **2 states per game**

**Coverage calculation:**
```
States_per_epoch = MATCHES_PER_EPOCH × 2 states/game × 2 players
Coverage_ratio = States_per_epoch / 10^8
```

| Config | Matches | States/Epoch | Coverage | Grad Updates |
|--------|---------|--------------|----------|--------------|
| E05a   | 500     | 2,000        | 0.002%   | 39           |
| E05b   | 1,000   | 4,000        | 0.004%   | 39           |
| E05c   | 2,000   | 8,000        | 0.008%   | 31           |

**Insight:** Even E05c covers only 0.008% per epoch → 125 epochs for 1% coverage

---

### 4. **Batch Size Theory**

**Rule of thumb:** `BATCH_SIZE >> episode_length` for decorrelation

**Citation:** Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction*. MIT press.

- Episode length (n_last_states=2): 2 states
- Batch sizes: 128/256/512 → all provide 64x-256x decorrelation
- Trade-off: Larger batch = more stable gradients, but fewer updates per epoch

**Gradient updates per epoch:**
```python
ITER_PER_EPOCH = STEPS_PER_EPOCH / BATCH_SIZE
                = (10 × MATCHES_PER_EPOCH) / BATCH_SIZE
```

- E05a/E05b: 39 updates (both have same ratio due to proportional scaling)
- E05c: 31 updates (slightly fewer, but more diverse data)

---

### 5. **Related Work on Multi-Agent Training**

**Population-Based Training:**
- **Vinyals, O., et al. (2019).** "Grandmaster level in StarCraft II using multi-agent reinforcement learning." *Nature*, 575(7782), 350-354.
  - Uses league of past agents to prevent strategy collapse
  - Relevant if E05 experiments still show degradation

**Curriculum Learning Failures:**
- **Bengio, Y., et al. (2009).** "Curriculum learning." *ICML*.
- Our E04b finding: Curriculum (2→16 states) causes loss divergence
- Reason: Network architecture not suited for variable-length inputs
- Solution: Constant n_last_states=2

**Model-Based RL (Future Direction):**
- **Schrittwieser, J., et al. (2020).** "Mastering Atari, Go, chess and shogi by planning with a learned model." *Nature*, 588(7839), 604-609.
  - MuZero: Combines planning with learning
  - Could improve sample efficiency if E05 requires too many games

---

## E05 Experiment Configuration

### Key Changes from E04b:

| Parameter | E04b (old) | E05 (new) | Rationale |
|-----------|-----------|-----------|-----------|
| `STARTING_NET` | Epoch 22 checkpoint | `None` (random) | AlphaZero approach, avoid bias |
| `N_LAST_STATES_FINAL` | 2/4/8/12/16 | 2 (constant) | Proven stable, no curriculum |
| `MATCHES_PER_EPOCH` | 310 (fixed) | 500/1000/2000 | **Main experiment variable** |
| `BATCH_SIZE` | 512 (fixed) | 128/256/512 | Scaled proportionally |
| `EXPERIMENT_NAME` | E04b | E05 | New experiment series |

### Experiment Variants:

**E05_MATCHES_PER_EPOCH500:**
- 500 games/epoch, batch size 128
- 2,000 states/epoch
- 39 gradient updates/epoch
- **Hypothesis:** Fast iterations, might plateau due to limited diversity

**E05_MATCHES_PER_EPOCH1000:** ⭐ **RECOMMENDED**
- 1,000 games/epoch, batch size 256
- 4,000 states/epoch
- 39 gradient updates/epoch
- **Hypothesis:** Best balance of coverage and speed

**E05_MATCHES_PER_EPOCH2000:**
- 2,000 games/epoch, batch size 512
- 8,000 states/epoch
- 31 gradient updates/epoch
- **Hypothesis:** Maximum coverage, slower but best final performance

---

## How to Run

### Step 1: Generate Training Scripts
```bash
cd d:/ZJONA/hierarchical-SAE
python run_trains.py
```

This creates 3 files:
- `trainRL_MATCHES_PER_EPOCH500.py`
- `trainRL_MATCHES_PER_EPOCH1000.py`
- `trainRL_MATCHES_PER_EPOCH2000.py`

### Step 2: Run in Parallel (3 terminals)

**Terminal 1:**
```bash
python trainRL_MATCHES_PER_EPOCH500.py | tee E05_MATCHES_PER_EPOCH500.log &
```

**Terminal 2:**
```bash
python trainRL_MATCHES_PER_EPOCH1000.py | tee E05_MATCHES_PER_EPOCH1000.log &
```

**Terminal 3:**
```bash
python trainRL_MATCHES_PER_EPOCH2000.py | tee E05_MATCHES_PER_EPOCH2000.log &
```

### Step 3: Monitor Progress

Check logs periodically:
```bash
tail -f E05_MATCHES_PER_EPOCH1000.log
```

Or create monitoring script:
```bash
watch -n 60 'tail -n 20 E05_MATCHES_PER_EPOCH*.log'
```

---

## Success Criteria

### Primary (Must Achieve):
1. ✅ **Win rate improves over time** (not degrades like E04b)
2. ✅ **Loss remains stable** (no divergence)
3. ✅ **Reaches ≥50% vs bot_good_WR_B** by epoch 500

### Secondary (Nice to Have):
4. ✅ Fastest learning speed (E05a predicted)
5. ✅ Best final performance (E05c predicted)
6. ✅ Best stability (E05b predicted)

### Decision Tree:

**If ALL variants still degrade:**
→ Problem is pure self-play itself (not hyperparameters)
→ Next: Implement population-based training (keep past checkpoints as opponents)

**If ONE variant improves:**
→ Use that configuration for production
→ Run longer (2000+ epochs) to verify convergence

**If E05b (moderate) wins:**
→ Confirms hypothesis, adopt as standard
→ Focus on other improvements (early stopping, architecture, etc.)

---

## Timeline

- **Epoch 100:** First checkpoint - check for improvement trend
- **Epoch 500:** Mid-point analysis - compare learning curves
- **Epoch 1000:** Final comparison - determine winner

**Expected runtime per variant:**
- E05a: ~2-3 days (fastest)
- E05b: ~4-5 days (moderate)
- E05c: ~8-10 days (slowest)

**Total experiment:** ~10 days if run in parallel

---

## Analysis Plan

After training completes, create analysis notebook:

1. **Win rate trajectories** - Is performance improving?
2. **Loss curves** - Is training stable?
3. **Learning speed** - Which reaches thresholds fastest?
4. **Variance analysis** - Which is most reliable?
5. **Statistical tests** - Is improvement significant?

Compare against E04b baseline to quantify improvement from:
- Random initialization vs pre-trained
- Optimal game coverage

---

## Next Steps (After E05)

**If successful:**
- Early stopping to save best checkpoint
- Tournament against broader baseline set
- Architecture improvements (attention mechanisms?)

**If still failing:**
- Population-based training (opponent diversity)
- MCTS-enhanced move selection
- Increase exploration (higher temperature schedule)

**Research questions:**
- Is Quarto too simple? (Converges to rock-paper-scissors cycles)
- Do we need opponent diversity for stable learning?
- Would model-based RL (planning) help?

---

## References

1. Silver, D., et al. (2017). Mastering Chess and Shogi by Self-Play. *arXiv:1712.01815*.
2. Mnih, V., et al. (2015). Human-level control through DRL. *Nature*, 518(7540), 529-533.
3. Vinyals, O., et al. (2019). StarCraft II Multi-Agent RL. *Nature*, 575(7782), 350-354.
4. Schrittwieser, J., et al. (2020). MuZero. *Nature*, 588(7839), 604-609.
5. Bengio, Y., et al. (2009). Curriculum learning. *ICML*.
6. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction*. MIT press.
