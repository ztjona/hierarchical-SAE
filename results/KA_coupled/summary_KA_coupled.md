# Summary — KA_coupled

Generated: 2026-04-22 14:09
Parameter varied: `N_LAST_STATES_INIT`
Runs: 6 (+ 4 baselines)

Final metrics = mean over the last 10% of epochs. `Peak` = max of the smoothed win-rate curve reached at any point during training.

## Experiment runs

| Run                                     | Param                 | Epochs | Final loss | Final vs bot_loss-BT | Peak vs bot_loss-BT | Final vs bot_random | Peak vs bot_random |
|-----------------------------------------|-----------------------|--------|------------|----------------------|---------------------|---------------------|--------------------|
| KA_coupled(1)0421_N_LAST_STATES_INIT_2  | N_LAST_STATES_INIT=2  | 5000   | 0.0994     | 66.7%                | 68.0%               | 80.1%               | 82.1%              |
| KA_coupled(2)0421_N_LAST_STATES_INIT_3  | N_LAST_STATES_INIT=3  | 5000   | 0.1147     | 54.0%                | 55.5%               | 74.4%               | 76.8%              |
| KA_coupled(3)0421_N_LAST_STATES_INIT_4  | N_LAST_STATES_INIT=4  | 5000   | 0.0989     | 45.2%                | 47.8%               | 67.6%               | 69.5%              |
| KA_coupled(4)0421_N_LAST_STATES_INIT_6  | N_LAST_STATES_INIT=6  | 4500   | 0.0717     | 41.1%                | 43.3%               | 63.4%               | 65.1%              |
| KA_coupled(5)0421_N_LAST_STATES_INIT_12 | N_LAST_STATES_INIT=12 | 2500   | 0.0510     | 34.2%                | 35.9%               | 57.4%               | 59.0%              |
| KA_coupled(6)0421_N_LAST_STATES_INIT_16 | N_LAST_STATES_INIT=16 | 2000   | 0.0478     | 33.9%                | 36.7%               | 55.7%               | 57.0%              |

## Baselines

| Run                                     | Param                  | Epochs | Final loss | Final vs bot_loss-BT | Peak vs bot_loss-BT | Final vs bot_random | Peak vs bot_random |
|-----------------------------------------|------------------------|--------|------------|----------------------|---------------------|---------------------|--------------------|
| B03_verLR(5)0210_LR_0.002               | LR=0.002               | 5173   | 0.1605     | 58.8%                | 60.8%               | 76.9%               | 78.2%              |
| Aa_replay(2)0226_NUM_EPOCHs_BUFFER_8    | NUM_EPOCHs_BUFFER=8    | 5000   | 0.1094     | 65.7%                | 67.1%               | 81.0%               | 82.5%              |
| Ab_data(4)0302_N_LAST_STATES_INIT_8     | N_LAST_STATES_INIT=8   | 5000   | 0.4181     | 31.5%                | 34.3%               | 51.3%               | 54.5%              |
| Aa_replay(5)0226_NUM_EPOCHs_BUFFER_1024 | NUM_EPOCHs_BUFFER=1024 | 5000   | 0.0583     | 69.4%                | 71.0%               | 81.4%               | 82.8%              |

## Best runs (experiments only)

- Lowest final loss: **KA_coupled(6)0421_N_LAST_STATES_INIT_16** — 0.0478
- Highest final WR vs bot_loss-BT: **KA_coupled(1)0421_N_LAST_STATES_INIT_2** — 66.7%
- Highest final WR vs bot_random: **KA_coupled(1)0421_N_LAST_STATES_INIT_2** — 80.1%
