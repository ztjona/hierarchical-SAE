# Summary — Z_tempRegresive

Generated: 2026-04-26 11:01
Parameter varied: `N_LAST_STATES_INIT`
Runs: 6 (+ 3 baselines)

Final metrics = mean over the last 10% of epochs. `Peak` = max of the smoothed win-rate curve reached at any point during training.

## Experiment runs

| Run                                          | Param                 | Epochs | Final loss | Final vs bot_loss-BT | Peak vs bot_loss-BT | Final vs bot_random | Peak vs bot_random |
|----------------------------------------------|-----------------------|--------|------------|----------------------|---------------------|---------------------|--------------------|
| Z_tempRegresive(1)0424_N_LAST_STATES_INIT_2  | N_LAST_STATES_INIT=2  | 5000   | 0.3143     | 56.0%                | 58.7%               | 72.8%               | 74.4%              |
| Z_tempRegresive(2)0424_N_LAST_STATES_INIT_3  | N_LAST_STATES_INIT=3  | 5000   | 0.3879     | 41.8%                | 44.6%               | 62.5%               | 65.5%              |
| Z_tempRegresive(3)0424_N_LAST_STATES_INIT_4  | N_LAST_STATES_INIT=4  | 5000   | 0.4200     | 49.7%                | 51.9%               | 70.7%               | 72.9%              |
| Z_tempRegresive(4)0424_N_LAST_STATES_INIT_6  | N_LAST_STATES_INIT=6  | 5000   | 0.4241     | 41.0%                | 43.2%               | 61.8%               | 64.8%              |
| Z_tempRegresive(5)0424_N_LAST_STATES_INIT_12 | N_LAST_STATES_INIT=12 | 5000   | 0.4127     | 34.5%                | 36.4%               | 54.9%               | 57.6%              |
| Z_tempRegresive(6)0424_N_LAST_STATES_INIT_16 | N_LAST_STATES_INIT=16 | 5000   | 0.3919     | 32.7%                | 34.9%               | 53.7%               | 55.9%              |

## Baselines

| Run                                     | Param                | Epochs | Final loss | Final vs bot_loss-BT | Peak vs bot_loss-BT | Final vs bot_random | Peak vs bot_random |
|-----------------------------------------|----------------------|--------|------------|----------------------|---------------------|---------------------|--------------------|
| LA_mcSelect(1)0422_N_LAST_STATES_INIT_2 | N_LAST_STATES_INIT=2 | 5000   | 0.0326     | 65.2%                | 68.3%               | 80.9%               | 82.5%              |
| LA_mcSelect(2)0422_N_LAST_STATES_INIT_3 | N_LAST_STATES_INIT=3 | 5000   | 0.3220     | 30.0%                | 33.4%               | 51.5%               | 55.3%              |
| Aa_replay(2)0226_NUM_EPOCHs_BUFFER_8    | NUM_EPOCHs_BUFFER=8  | 5000   | 0.1094     | 65.7%                | 67.1%               | 81.0%               | 82.5%              |

## Best runs (experiments only)

- Lowest final loss: **Z_tempRegresive(1)0424_N_LAST_STATES_INIT_2** — 0.3143
- Highest final WR vs bot_loss-BT: **Z_tempRegresive(1)0424_N_LAST_STATES_INIT_2** — 56.0%
- Highest final WR vs bot_random: **Z_tempRegresive(1)0424_N_LAST_STATES_INIT_2** — 72.8%
