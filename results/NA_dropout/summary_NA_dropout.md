# Summary — NA_dropout

Generated: 2026-05-07 10:56
Parameter varied: `N_LAST_STATES_INIT`
Runs: 3 (+ 6 baselines)

Final metrics = mean over the last 10% of epochs. `Peak` = max of the smoothed win-rate curve reached at any point during training.

## Experiment runs

| Run                                    | Param                | Epochs | Final loss | Final vs bot_loss-BT | Peak vs bot_loss-BT | Final vs bot_random | Peak vs bot_random |
|----------------------------------------|----------------------|--------|------------|----------------------|---------------------|---------------------|--------------------|
| NA_dropout(1)0429_N_LAST_STATES_INIT_2 | N_LAST_STATES_INIT=2 | 5000   | 0.0897     | 65.6%                | 67.2%               | 80.4%               | 82.2%              |
| NA_dropout(2)0429_N_LAST_STATES_INIT_3 | N_LAST_STATES_INIT=3 | 5000   | 0.1864     | 39.8%                | 42.1%               | 60.3%               | 62.5%              |
| NA_dropout(3)0429_N_LAST_STATES_INIT_4 | N_LAST_STATES_INIT=4 | 5000   | 0.1823     | 40.3%                | 43.0%               | 61.3%               | 62.8%              |

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

- Lowest final loss: **NA_dropout(1)0429_N_LAST_STATES_INIT_2** — 0.0897
- Highest final WR vs bot_loss-BT: **NA_dropout(1)0429_N_LAST_STATES_INIT_2** — 65.6%
- Highest final WR vs bot_random: **NA_dropout(1)0429_N_LAST_STATES_INIT_2** — 80.4%
