# Summary — OA_unifiedAux

Generated: 2026-05-10 09:48
Parameter varied: `N_LAST_STATES_INIT`
Runs: 4 (+ 6 baselines)

Final metrics = mean over the last 10% of epochs. `Peak` = max of the smoothed win-rate curve reached at any point during training.

## Experiment runs

| Run                                        | Param                 | Epochs | Final loss | Final vs bot_loss-BT | Peak vs bot_loss-BT | Final vs bot_random | Peak vs bot_random |
|--------------------------------------------|-----------------------|--------|------------|----------------------|---------------------|---------------------|--------------------|
| OA_unifiedAux(1)0509_N_LAST_STATES_INIT_2  | N_LAST_STATES_INIT=2  | 5000   | 0.1126     | 65.0%                | 67.8%               | 80.9%               | 82.7%              |
| OA_unifiedAux(2)0509_N_LAST_STATES_INIT_3  | N_LAST_STATES_INIT=3  | 3000   | 0.2114     | 36.9%                | 39.2%               | 56.7%               | 59.5%              |
| OA_unifiedAux(3)0509_N_LAST_STATES_INIT_4  | N_LAST_STATES_INIT=4  | 3000   | 0.2308     | 37.9%                | 39.9%               | 59.2%               | 60.9%              |
| OA_unifiedAux(4)0509_N_LAST_STATES_INIT_12 | N_LAST_STATES_INIT=12 | 1000   | 0.1866     | 33.7%                | 34.8%               | 53.2%               | 56.0%              |

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

- Lowest final loss: **OA_unifiedAux(1)0509_N_LAST_STATES_INIT_2** — 0.1126
- Highest final WR vs bot_loss-BT: **OA_unifiedAux(1)0509_N_LAST_STATES_INIT_2** — 65.0%
- Highest final WR vs bot_random: **OA_unifiedAux(1)0509_N_LAST_STATES_INIT_2** — 80.9%
