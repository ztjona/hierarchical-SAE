# E05 Hyperparameter Search - Theoretical Foundation

## Objective
Find optimal `MATCHES_PER_EPOCH` and `BATCH_SIZE` for Quarto based on game complexity.

## Theoretical Framework

### 1. State Space Coverage

**Quarto State Space:**
- Total states: ~10^8
- Average game length: 8 moves (16 half-moves)
- States generated per game (with n_last_states=2): 2

**Coverage Formula:**
```
States_per_epoch = MATCHES_PER_EPOCH × avg_states_per_game × 2_players
Coverage_ratio = States_per_epoch / Total_state_space
```

**Current E04b (baseline):**
- 310 matches × 2 states × 2 = 1,240 states/epoch
- Coverage: 0.0000124% per epoch

**Proposed experiments:**
- E05a: 500 × 2 × 2 = 2,000 states/epoch (1.6x baseline)
- E05b: 1000 × 2 × 2 = 4,000 states/epoch (3.2x baseline)
- E05c: 2000 × 2 × 2 = 8,000 states/epoch (6.5x baseline)

### 2. Batch Size Selection

**Rule of Thumb:** Batch size should decorrelate experiences while maintaining gradient stability.

**Episode length consideration:**
- With n_last_states=2: only 2 states per game
- Need batch >> episode length to decorrelate
- Too large: slower iterations, less frequent updates

**Gradient updates per epoch:**
```
ITER_PER_EPOCH = STEPS_PER_EPOCH / BATCH_SIZE
             = (10 × MATCHES_PER_EPOCH) / BATCH_SIZE
```

**Proposed configurations:**

| Variant | Matches | Batch | Steps | Iters | States | Grad/State Ratio |
|---------|---------|-------|-------|-------|--------|------------------|
| E05a    | 500     | 128   | 5000  | 39    | 2000   | 1.95%            |
| E05b    | 1000    | 256   | 10000 | 39    | 4000   | 0.98%            |
| E05c    | 2000    | 512   | 16000 | 31    | 8000   | 0.39%            |

### 3. AlphaZero Comparison

**AlphaZero (for reference):**
- Games per iteration: ~5,000-25,000
- Batch size: 4096
- Gradient updates: ~1000 per iteration
- Training epochs: ~700,000 iterations

**Scaled to Quarto's complexity:**
- Quarto is ~10^8 states vs Go's ~10^170
- Reasonable scaling: 1,000-2,000 games/epoch
- Batch size: 128-512 (smaller game = smaller batch)

## Expected Outcomes

### E05a (500 games, BS=128)
**Pros:**
- Fastest iterations (39 gradient updates/epoch)
- More frequent model updates
- Good for rapid prototyping

**Cons:**
- Lower state space coverage
- Might underfit (not seeing enough diversity)

**Prediction:** Fast learning initially, might plateau early due to limited coverage.

---

### E05b (1000 games, BS=256)
**Pros:**
- Balanced coverage vs speed
- 39 gradient updates/epoch (same as E05a due to proportional scaling)
- 2x state coverage of E05a

**Cons:**
- 2x computation per epoch vs E05a

**Prediction:** Best balance - should show steady improvement with good generalization.

---

### E05c (2000 games, BS=512)
**Pros:**
- Highest state space coverage (8,000 states/epoch)
- Best diversity in replay buffer
- Closest to AlphaZero proportions

**Cons:**
- Slowest (2x computation vs E05b)
- Only 31 gradient updates/epoch (fewer than E05a/E05b)
- Might be overkill for Quarto's simplicity

**Prediction:** Slower learning initially, but best final performance and generalization.

## Evaluation Metrics

**Primary:**
1. **Win rate vs baselines over time** - Is it improving or degrading?
2. **Loss stability** - Does loss stay stable or diverge?
3. **Learning speed** - How fast does it reach 50% win rate?

**Secondary:**
4. **Computational efficiency** - Time per epoch
5. **Convergence** - Does it plateau or keep improving?

## Decision Criteria

**Choose E05a if:**
- Need rapid iteration
- Computational budget is limited
- Prototyping/debugging

**Choose E05b if:**
- Want balanced approach
- Standard production training
- Good default choice

**Choose E05c if:**
- Need maximum robustness
- Want best final performance
- Computational resources available
- Research/publication quality

## How to Run

```bash
# Terminal 1: Run E05a
cd d:\ZJONA\hierarchical-SAE
python exe/trainRL_E05_hyperparams.py  # Set VARIANT = "E05a"

# Terminal 2: Run E05b (parallel)
python exe/trainRL_E05_hyperparams.py  # Set VARIANT = "E05b"

# Terminal 3: Run E05c (parallel)
python exe/trainRL_E05_hyperparams.py  # Set VARIANT = "E05c"
```

**Monitor:** After 100 epochs, check which shows:
1. Increasing win rate (not degrading)
2. Stable loss
3. Best performance vs bot_good_WR_B

## Next Steps After E05

Once optimal hyperparameters are found:
1. Run longer (2000+ epochs) with best config
2. Consider population-based training if pure self-play still degrades
3. Implement early stopping to save best checkpoint
