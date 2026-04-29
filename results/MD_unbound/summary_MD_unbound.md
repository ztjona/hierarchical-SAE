# Summary — MD_unbound

Generated: 2026-04-29 11:46
Parameter varied: `N_LAST_STATES_INIT`
Runs: 6 (+ 6 baselines)

Final metrics = mean over the last 10% of epochs. `Peak` = max of the smoothed win-rate curve reached at any point during training.

## Experiment runs

| Run                                     | Param                 | Epochs | Final loss | Final vs bot_loss-BT | Peak vs bot_loss-BT | Final vs bot_random | Peak vs bot_random |
|-----------------------------------------|-----------------------|--------|------------|----------------------|---------------------|---------------------|--------------------|
| MD_unbound(1)0428_N_LAST_STATES_INIT_2  | N_LAST_STATES_INIT=2  | 5000   | 0.1282     | 61.6%                | 63.4%               | 78.6%               | 79.9%              |
| MD_unbound(2)0428_N_LAST_STATES_INIT_3  | N_LAST_STATES_INIT=3  | 5000   | 0.2162     | 45.8%                | 48.5%               | 64.0%               | 66.2%              |
| MD_unbound(3)0428_N_LAST_STATES_INIT_4  | N_LAST_STATES_INIT=4  | 5000   | 0.2354     | 53.2%                | 55.3%               | 73.1%               | 74.8%              |
| MD_unbound(4)0428_N_LAST_STATES_INIT_6  | N_LAST_STATES_INIT=6  | 3000   | 0.2190     | 41.1%                | 44.1%               | 62.8%               | 64.1%              |
| MD_unbound(5)0428_N_LAST_STATES_INIT_12 | N_LAST_STATES_INIT=12 | 2000   | 0.1865     | 33.4%                | 35.2%               | 55.6%               | 57.1%              |
| MD_unbound(6)0428_N_LAST_STATES_INIT_16 | N_LAST_STATES_INIT=16 | 1000   | 0.1716     | 31.8%                | 34.0%               | 53.6%               | 55.4%              |

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

- Lowest final loss: **MD_unbound(1)0428_N_LAST_STATES_INIT_2** — 0.1282
- Highest final WR vs bot_loss-BT: **MD_unbound(1)0428_N_LAST_STATES_INIT_2** — 61.6%
- Highest final WR vs bot_random: **MD_unbound(1)0428_N_LAST_STATES_INIT_2** — 78.6%
