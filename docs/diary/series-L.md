# Series L — Monte Carlo Q_select target (joint pipeline)

LA introduced `LOSS_APPROACH="mc_select"`. LB is the same run after the transition-schema plumbing refactor (same config). See parent: `Research-status.md`.

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

