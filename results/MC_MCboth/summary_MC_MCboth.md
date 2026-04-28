# Summary — MC_MCboth

Generated: 2026-04-28 10:35
Parameter varied: `N_LAST_STATES_INIT`
Runs: 6 (+ 6 baselines)

Final metrics = mean over the last 10% of epochs. `Peak` = max of the smoothed win-rate curve reached at any point during training.

## Experiment runs

| Run                                    | Param                 | Epochs | Final loss | Final vs bot_loss-BT | Peak vs bot_loss-BT | Final vs bot_random | Peak vs bot_random |
|----------------------------------------|-----------------------|--------|------------|----------------------|---------------------|---------------------|--------------------|
| MC_MCboth(1)0427_N_LAST_STATES_INIT_2  | N_LAST_STATES_INIT=2  | 5000   | 0.1049     | 66.7%                | 68.2%               | 81.9%               | 83.8%              |
| MC_MCboth(2)0427_N_LAST_STATES_INIT_3  | N_LAST_STATES_INIT=3  | 4000   | 0.3773     | 44.7%                | 47.1%               | 64.6%               | 66.8%              |
| MC_MCboth(3)0427_N_LAST_STATES_INIT_4  | N_LAST_STATES_INIT=4  | 4000   | 0.3842     | 47.1%                | 48.6%               | 68.0%               | 70.4%              |
| MC_MCboth(4)0427_N_LAST_STATES_INIT_6  | N_LAST_STATES_INIT=6  | 3000   | 0.3973     | 39.5%                | 43.4%               | 61.5%               | 63.8%              |
| MC_MCboth(5)0427_N_LAST_STATES_INIT_12 | N_LAST_STATES_INIT=12 | 2000   | 0.3687     | 34.4%                | 36.7%               | 57.0%               | 58.7%              |
| MC_MCboth(6)0427_N_LAST_STATES_INIT_16 | N_LAST_STATES_INIT=16 | 1000   | 0.3491     | 32.5%                | 35.0%               | 54.1%               | 57.1%              |

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

- Lowest final loss: **MC_MCboth(1)0427_N_LAST_STATES_INIT_2** — 0.1049
- Highest final WR vs bot_loss-BT: **MC_MCboth(1)0427_N_LAST_STATES_INIT_2** — 66.7%
- Highest final WR vs bot_random: **MC_MCboth(1)0427_N_LAST_STATES_INIT_2** — 81.9%
