# Summary — MF_dataScale

Generated: 2026-05-06 10:20
Parameter varied: `MATCHES`
Runs: 4 (+ 6 baselines)

Final metrics = mean over the last 10% of epochs. `Peak` = max of the smoothed win-rate curve reached at any point during training.

## Experiment runs

| Run                                 | Param       | Epochs | Final loss | Final vs bot_loss-BT | Peak vs bot_loss-BT | Final vs bot_random | Peak vs bot_random |
|-------------------------------------|-------------|--------|------------|----------------------|---------------------|---------------------|--------------------|
| MF_dataScale(3)0429_MATCHES_128_N_6 | MATCHES=128 | 2000   | 0.2162     | 44.5%                | 45.9%               | 65.3%               | 67.2%              |
| MF_dataScale(1)0429_MATCHES_128_N_4 | MATCHES=128 | 3000   | 0.2271     | 63.9%                | 65.3%               | 80.4%               | 81.9%              |
| MF_dataScale(4)0429_MATCHES_320_N_6 | MATCHES=320 | 1000   | 0.2197     | 47.7%                | 48.4%               | 68.1%               | 68.9%              |
| MF_dataScale(2)0429_MATCHES_320_N_4 | MATCHES=320 | 1500   | 0.2248     | 65.5%                | 67.2%               | 81.4%               | 82.1%              |

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

- Lowest final loss: **MF_dataScale(3)0429_MATCHES_128_N_6** — 0.2162
- Highest final WR vs bot_loss-BT: **MF_dataScale(2)0429_MATCHES_320_N_4** — 65.5%
- Highest final WR vs bot_random: **MF_dataScale(2)0429_MATCHES_320_N_4** — 81.4%
