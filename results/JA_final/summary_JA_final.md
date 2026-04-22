# Summary — JA_final

Generated: 2026-04-21 21:44
Parameter varied: `N_LAST_STATES_INIT`
Runs: 6 (+ 5 baselines)

Final metrics = mean over the last 10% of epochs. `Peak` = max of the smoothed win-rate curve reached at any point during training.

## Experiment runs

| Run                                   | Param                 | Epochs | Final loss | Final vs bot_loss-BT | Peak vs bot_loss-BT | Final vs bot_random | Peak vs bot_random |
|---------------------------------------|-----------------------|--------|------------|----------------------|---------------------|---------------------|--------------------|
| JA_final(1)0420_N_LAST_STATES_INIT_2  | N_LAST_STATES_INIT=2  | 5000   | 0.1062     | 66.4%                | 68.1%               | 80.2%               | 81.3%              |
| JA_final(2)0420_N_LAST_STATES_INIT_3  | N_LAST_STATES_INIT=3  | 5000   | 0.1149     | 52.7%                | 54.9%               | 71.9%               | 74.0%              |
| JA_final(3)0420_N_LAST_STATES_INIT_4  | N_LAST_STATES_INIT=4  | 5000   | 0.0983     | 44.8%                | 47.4%               | 67.2%               | 69.1%              |
| JA_final(4)0420_N_LAST_STATES_INIT_6  | N_LAST_STATES_INIT=6  | 5000   | 0.0719     | 40.3%                | 42.0%               | 63.4%               | 65.8%              |
| JA_final(5)0420_N_LAST_STATES_INIT_12 | N_LAST_STATES_INIT=12 | 5000   | 0.0515     | 35.6%                | 37.7%               | 57.5%               | 59.8%              |
| JA_final(6)0420_N_LAST_STATES_INIT_16 | N_LAST_STATES_INIT=16 | 5000   | 0.0486     | 34.6%                | 36.6%               | 56.0%               | 58.1%              |

## Baselines

| Run                                     | Param                  | Epochs | Final loss | Final vs bot_loss-BT | Peak vs bot_loss-BT | Final vs bot_random | Peak vs bot_random |
|-----------------------------------------|------------------------|--------|------------|----------------------|---------------------|---------------------|--------------------|
| B02replicate(6)0121_LR_0.0005           | LR=5e-04               | 10000  | 0.1574     | 62.7%                | 65.0%               | 76.3%               | 77.8%              |
| B03_verLR(5)0210_LR_0.002               | LR=0.002               | 5173   | 0.1605     | 58.8%                | 60.8%               | 76.9%               | 78.2%              |
| Aa_replay(2)0226_NUM_EPOCHs_BUFFER_8    | NUM_EPOCHs_BUFFER=8    | 5000   | 0.1094     | 65.7%                | 67.1%               | 81.0%               | 82.5%              |
| Ab_data(4)0302_N_LAST_STATES_INIT_8     | N_LAST_STATES_INIT=8   | 5000   | 0.4181     | 31.5%                | 34.3%               | 51.3%               | 54.5%              |
| Aa_replay(5)0226_NUM_EPOCHs_BUFFER_1024 | NUM_EPOCHs_BUFFER=1024 | 5000   | 0.0583     | 69.4%                | 71.0%               | 81.4%               | 82.8%              |

## Best runs (experiments only)

- Lowest final loss: **JA_final(6)0420_N_LAST_STATES_INIT_16** — 0.0486
- Highest final WR vs bot_loss-BT: **JA_final(1)0420_N_LAST_STATES_INIT_2** — 66.4%
- Highest final WR vs bot_random: **JA_final(1)0420_N_LAST_STATES_INIT_2** — 80.2%
