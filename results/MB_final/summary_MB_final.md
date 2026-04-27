# Summary — MB_final

Generated: 2026-04-27 12:03
Parameter varied: `N_LAST_STATES_INIT`
Runs: 6 (+ 6 baselines)

Final metrics = mean over the last 10% of epochs. `Peak` = max of the smoothed win-rate curve reached at any point during training.

## Experiment runs

| Run                                   | Param                 | Epochs | Final loss | Final vs bot_loss-BT | Peak vs bot_loss-BT | Final vs bot_random | Peak vs bot_random |
|---------------------------------------|-----------------------|--------|------------|----------------------|---------------------|---------------------|--------------------|
| MB_final(1)0426_N_LAST_STATES_INIT_2  | N_LAST_STATES_INIT=2  | 5000   | 0.1117     | 66.9%                | 68.9%               | 80.5%               | 81.6%              |
| MB_final(2)0426_N_LAST_STATES_INIT_3  | N_LAST_STATES_INIT=3  | 5000   | 0.2210     | 45.8%                | 48.0%               | 64.8%               | 66.6%              |
| MB_final(3)0426_N_LAST_STATES_INIT_4  | N_LAST_STATES_INIT=4  | 5000   | 0.2367     | 56.2%                | 57.9%               | 74.0%               | 75.7%              |
| MB_final(4)0426_N_LAST_STATES_INIT_6  | N_LAST_STATES_INIT=6  | 4050   | 0.2164     | 43.8%                | 45.1%               | 65.8%               | 67.8%              |
| MB_final(5)0426_N_LAST_STATES_INIT_12 | N_LAST_STATES_INIT=12 | 2500   | 0.1920     | 34.2%                | 36.5%               | 54.7%               | 57.1%              |
| MB_final(6)0426_N_LAST_STATES_INIT_16 | N_LAST_STATES_INIT=16 | 2000   | 0.1785     | 32.3%                | 34.7%               | 54.3%               | 55.3%              |

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

- Lowest final loss: **MB_final(1)0426_N_LAST_STATES_INIT_2** — 0.1117
- Highest final WR vs bot_loss-BT: **MB_final(1)0426_N_LAST_STATES_INIT_2** — 66.9%
- Highest final WR vs bot_random: **MB_final(1)0426_N_LAST_STATES_INIT_2** — 80.5%
