# Summary — NB_asymLR

Generated: 2026-05-07 10:57
Parameter varied: `N_LAST_STATES_INIT`
Runs: 3 (+ 6 baselines)

Final metrics = mean over the last 10% of epochs. `Peak` = max of the smoothed win-rate curve reached at any point during training.

## Experiment runs

| Run                                   | Param                | Epochs | Final loss | Final vs bot_loss-BT | Peak vs bot_loss-BT | Final vs bot_random | Peak vs bot_random |
|---------------------------------------|----------------------|--------|------------|----------------------|---------------------|---------------------|--------------------|
| NB_asymLR(1)0429_N_LAST_STATES_INIT_2 | N_LAST_STATES_INIT=2 | 5000   | 0.1088     | 66.2%                | 68.0%               | 81.5%               | 82.7%              |
| NB_asymLR(2)0429_N_LAST_STATES_INIT_3 | N_LAST_STATES_INIT=3 | 5000   | 0.2113     | 39.1%                | 40.5%               | 59.4%               | 61.0%              |
| NB_asymLR(3)0429_N_LAST_STATES_INIT_4 | N_LAST_STATES_INIT=4 | 5000   | 0.2186     | 51.5%                | 53.7%               | 71.5%               | 72.9%              |

## Baselines

| Run                                          | Param                | Epochs | Final loss | Final vs bot_loss-BT | Peak vs bot_loss-BT | Final vs bot_random | Peak vs bot_random |
|----------------------------------------------|----------------------|--------|------------|----------------------|---------------------|---------------------|--------------------|
| LA_mcSelect(1)0422_N_LAST_STATES_INIT_2      | N_LAST_STATES_INIT=2 | 5000   | 0.0326     | 65.2%                | 68.3%               | 80.9%               | 82.5%              |
| MA_tempRegresive(1)0424_N_LAST_STATES_INIT_2 | N_LAST_STATES_INIT=2 | 5000   | 0.3143     | 56.0%                | 58.7%               | 72.8%               | 74.4%              |
| JA_final(2)0420_N_LAST_STATES_INIT_3         | N_LAST_STATES_INIT=3 | 5000   | 0.1149     | 52.7%                | 54.9%               | 71.9%               | 74.0%              |
| LA_mcSelect(2)0422_N_LAST_STATES_INIT_3      | N_LAST_STATES_INIT=3 | 5000   | 0.3220     | 30.0%                | 33.4%               | 51.5%               | 55.3%              |
| MA_tempRegresive(3)0424_N_LAST_STATES_INIT_4 | N_LAST_STATES_INIT=4 | 5000   | 0.4200     | 49.7%                | 51.9%               | 70.7%               | 72.9%              |
| Aa_replay(2)0226_NUM_EPOCHs_BUFFER_8         | NUM_EPOCHs_BUFFER=8  | 5000   | 0.1094     | 65.7%                | 67.1%               | 81.0%               | 82.5%              |

## Best runs (experiments only)

- Lowest final loss: **NB_asymLR(1)0429_N_LAST_STATES_INIT_2** — 0.1088
- Highest final WR vs bot_loss-BT: **NB_asymLR(1)0429_N_LAST_STATES_INIT_2** — 66.2%
- Highest final WR vs bot_random: **NB_asymLR(1)0429_N_LAST_STATES_INIT_2** — 81.5%
