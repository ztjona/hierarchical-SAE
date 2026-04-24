# Summary — LB_mcSelect

Generated: 2026-04-24 10:41
Parameter varied: `N_LAST_STATES_INIT`
Runs: 6 (+ 6 baselines)

Final metrics = mean over the last 10% of epochs. `Peak` = max of the smoothed win-rate curve reached at any point during training.

## Experiment runs

| Run                                      | Param                 | Epochs | Final loss | Final vs bot_loss-BT | Peak vs bot_loss-BT | Final vs bot_random | Peak vs bot_random |
|------------------------------------------|-----------------------|--------|------------|----------------------|---------------------|---------------------|--------------------|
| LB_mcSelect(1)0423_N_LAST_STATES_INIT_2  | N_LAST_STATES_INIT=2  | 5000   | 0.0326     | 65.2%                | 68.3%               | 80.9%               | 82.5%              |
| LB_mcSelect(2)0423_N_LAST_STATES_INIT_3  | N_LAST_STATES_INIT=3  | 5000   | 0.3220     | 30.0%                | 33.4%               | 51.5%               | 55.3%              |
| LB_mcSelect(3)0423_N_LAST_STATES_INIT_4  | N_LAST_STATES_INIT=4  | 5000   | 0.3128     | 39.7%                | 42.5%               | 59.6%               | 63.5%              |
| LB_mcSelect(4)0423_N_LAST_STATES_INIT_6  | N_LAST_STATES_INIT=6  | 4500   | 0.3752     | 33.1%                | 34.9%               | 53.7%               | 55.7%              |
| LB_mcSelect(5)0423_N_LAST_STATES_INIT_12 | N_LAST_STATES_INIT=12 | 3000   | 0.4110     | 29.9%                | 33.5%               | 50.9%               | 54.3%              |
| LB_mcSelect(6)0423_N_LAST_STATES_INIT_16 | N_LAST_STATES_INIT=16 | 2500   | 0.4053     | 29.6%                | 33.0%               | 50.4%               | 53.8%              |

## Baselines

| Run                                      | Param                 | Epochs | Final loss | Final vs bot_loss-BT | Peak vs bot_loss-BT | Final vs bot_random | Peak vs bot_random |
|------------------------------------------|-----------------------|--------|------------|----------------------|---------------------|---------------------|--------------------|
| LA_mcSelect(1)0422_N_LAST_STATES_INIT_2  | N_LAST_STATES_INIT=2  | 5000   | 0.0326     | 65.2%                | 68.3%               | 80.9%               | 82.5%              |
| LA_mcSelect(2)0422_N_LAST_STATES_INIT_3  | N_LAST_STATES_INIT=3  | 5000   | 0.3220     | 30.0%                | 33.4%               | 51.5%               | 55.3%              |
| LA_mcSelect(3)0422_N_LAST_STATES_INIT_4  | N_LAST_STATES_INIT=4  | 5000   | 0.3128     | 39.7%                | 42.5%               | 59.6%               | 63.5%              |
| LA_mcSelect(4)0422_N_LAST_STATES_INIT_6  | N_LAST_STATES_INIT=6  | 5000   | 0.3759     | 33.0%                | 34.9%               | 53.7%               | 55.9%              |
| LA_mcSelect(5)0422_N_LAST_STATES_INIT_12 | N_LAST_STATES_INIT=12 | 4500   | 0.4109     | 30.2%                | 33.5%               | 51.1%               | 54.3%              |
| LA_mcSelect(6)0422_N_LAST_STATES_INIT_16 | N_LAST_STATES_INIT=16 | 3500   | 0.4057     | 31.2%                | 33.0%               | 51.5%               | 53.8%              |

## Best runs (experiments only)

- Lowest final loss: **LB_mcSelect(1)0423_N_LAST_STATES_INIT_2** — 0.0326
- Highest final WR vs bot_loss-BT: **LB_mcSelect(1)0423_N_LAST_STATES_INIT_2** — 65.2%
- Highest final WR vs bot_random: **LB_mcSelect(1)0423_N_LAST_STATES_INIT_2** — 80.9%
