# Summary — LA_mcSelect

Generated: 2026-04-23 12:11
Parameter varied: `N_LAST_STATES_INIT`
Runs: 6 (+ 5 baselines)

Final metrics = mean over the last 10% of epochs. `Peak` = max of the smoothed win-rate curve reached at any point during training.

## Experiment runs

| Run                                      | Param                 | Epochs | Final loss | Final vs bot_loss-BT | Peak vs bot_loss-BT | Final vs bot_random | Peak vs bot_random |
|------------------------------------------|-----------------------|--------|------------|----------------------|---------------------|---------------------|--------------------|
| LA_mcSelect(1)0422_N_LAST_STATES_INIT_2  | N_LAST_STATES_INIT=2  | 5000   | 0.0326     | 65.2%                | 68.3%               | 80.9%               | 82.5%              |
| LA_mcSelect(2)0422_N_LAST_STATES_INIT_3  | N_LAST_STATES_INIT=3  | 5000   | 0.3220     | 30.0%                | 33.4%               | 51.5%               | 55.3%              |
| LA_mcSelect(3)0422_N_LAST_STATES_INIT_4  | N_LAST_STATES_INIT=4  | 5000   | 0.3128     | 39.7%                | 42.5%               | 59.6%               | 63.5%              |
| LA_mcSelect(4)0422_N_LAST_STATES_INIT_6  | N_LAST_STATES_INIT=6  | 5000   | 0.3759     | 33.0%                | 34.9%               | 53.7%               | 55.9%              |
| LA_mcSelect(5)0422_N_LAST_STATES_INIT_12 | N_LAST_STATES_INIT=12 | 4500   | 0.4109     | 30.2%                | 33.5%               | 51.1%               | 54.3%              |
| LA_mcSelect(6)0422_N_LAST_STATES_INIT_16 | N_LAST_STATES_INIT=16 | 3500   | 0.4057     | 31.2%                | 33.0%               | 51.5%               | 53.8%              |

## Baselines

| Run                                     | Param                  | Epochs | Final loss | Final vs bot_loss-BT | Peak vs bot_loss-BT | Final vs bot_random | Peak vs bot_random |
|-----------------------------------------|------------------------|--------|------------|----------------------|---------------------|---------------------|--------------------|
| B03_verLR(5)0210_LR_0.002               | LR=0.002               | 5173   | 0.1605     | 58.8%                | 60.8%               | 76.9%               | 78.2%              |
| JA_final(2)0420_N_LAST_STATES_INIT_3    | N_LAST_STATES_INIT=3   | 5000   | 0.1149     | 52.7%                | 54.9%               | 71.9%               | 74.0%              |
| Aa_replay(2)0226_NUM_EPOCHs_BUFFER_8    | NUM_EPOCHs_BUFFER=8    | 5000   | 0.1094     | 65.7%                | 67.1%               | 81.0%               | 82.5%              |
| Ab_data(4)0302_N_LAST_STATES_INIT_8     | N_LAST_STATES_INIT=8   | 5000   | 0.4181     | 31.5%                | 34.3%               | 51.3%               | 54.5%              |
| Aa_replay(5)0226_NUM_EPOCHs_BUFFER_1024 | NUM_EPOCHs_BUFFER=1024 | 5000   | 0.0583     | 69.4%                | 71.0%               | 81.4%               | 82.8%              |

## Best runs (experiments only)

- Lowest final loss: **LA_mcSelect(1)0422_N_LAST_STATES_INIT_2** — 0.0326
- Highest final WR vs bot_loss-BT: **LA_mcSelect(1)0422_N_LAST_STATES_INIT_2** — 65.2%
- Highest final WR vs bot_random: **LA_mcSelect(1)0422_N_LAST_STATES_INIT_2** — 80.9%
