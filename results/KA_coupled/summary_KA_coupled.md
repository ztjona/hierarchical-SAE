# Summary — KA_coupled

Generated: 2026-04-22 13:30
Parameter varied: `N_LAST_STATES_INIT`
Runs: 6 (+ 6 baselines)

Final metrics = mean over the last 10% of epochs. `Peak` = max of the smoothed win-rate curve reached at any point during training.

## Experiment runs

| Run                                     | Param                 | Epochs | Final loss | Final vs bot_loss-BT | Peak vs bot_loss-BT | Final vs bot_random | Peak vs bot_random |
|-----------------------------------------|-----------------------|--------|------------|----------------------|---------------------|---------------------|--------------------|
| KA_coupled(1)0421_N_LAST_STATES_INIT_2  | N_LAST_STATES_INIT=2  | 5000   | 0.0994     | 66.7%                | 68.0%               | 80.1%               | 82.1%              |
| KA_coupled(2)0421_N_LAST_STATES_INIT_3  | N_LAST_STATES_INIT=3  | 5000   | 0.1147     | 54.0%                | 55.5%               | 74.4%               | 76.8%              |
| KA_coupled(3)0421_N_LAST_STATES_INIT_4  | N_LAST_STATES_INIT=4  | 5000   | 0.0989     | 45.2%                | 47.8%               | 67.6%               | 69.5%              |
| KA_coupled(4)0421_N_LAST_STATES_INIT_6  | N_LAST_STATES_INIT=6  | 4000   | 0.0718     | 41.5%                | 43.3%               | 63.1%               | 65.0%              |
| KA_coupled(5)0421_N_LAST_STATES_INIT_12 | N_LAST_STATES_INIT=12 | 2500   | 0.0510     | 34.2%                | 35.9%               | 57.4%               | 59.0%              |
| KA_coupled(6)0421_N_LAST_STATES_INIT_16 | N_LAST_STATES_INIT=16 | 2000   | 0.0478     | 33.9%                | 36.7%               | 55.7%               | 57.0%              |

## Baselines

| Run                                   | Param                 | Epochs | Final loss | Final vs bot_loss-BT | Peak vs bot_loss-BT | Final vs bot_random | Peak vs bot_random |
|---------------------------------------|-----------------------|--------|------------|----------------------|---------------------|---------------------|--------------------|
| JA_final(1)0420_N_LAST_STATES_INIT_2  | N_LAST_STATES_INIT=2  | 5000   | 0.1062     | 66.4%                | 68.1%               | 80.2%               | 81.3%              |
| JA_final(2)0420_N_LAST_STATES_INIT_3  | N_LAST_STATES_INIT=3  | 5000   | 0.1149     | 52.7%                | 54.9%               | 71.9%               | 74.0%              |
| JA_final(3)0420_N_LAST_STATES_INIT_4  | N_LAST_STATES_INIT=4  | 5000   | 0.0983     | 44.8%                | 47.4%               | 67.2%               | 69.1%              |
| JA_final(4)0420_N_LAST_STATES_INIT_6  | N_LAST_STATES_INIT=6  | 5000   | 0.0719     | 40.3%                | 42.0%               | 63.4%               | 65.7%              |
| JA_final(5)0420_N_LAST_STATES_INIT_12 | N_LAST_STATES_INIT=12 | 5000   | 0.0515     | 35.6%                | 37.7%               | 57.5%               | 59.8%              |
| JA_final(6)0420_N_LAST_STATES_INIT_16 | N_LAST_STATES_INIT=16 | 5000   | 0.0486     | 34.6%                | 36.6%               | 56.0%               | 58.1%              |

## Best runs (experiments only)

- Lowest final loss: **KA_coupled(6)0421_N_LAST_STATES_INIT_16** — 0.0478
- Highest final WR vs bot_loss-BT: **KA_coupled(1)0421_N_LAST_STATES_INIT_2** — 66.7%
- Highest final WR vs bot_random: **KA_coupled(1)0421_N_LAST_STATES_INIT_2** — 80.1%
