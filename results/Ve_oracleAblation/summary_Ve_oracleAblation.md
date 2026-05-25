# Summary — Ve_oracleAblation

Generated: 2026-05-24 19:18
Parameter varied: `DISABLE`, `DISABLE_NEVER`, `Sweep`
Runs: 4 (+ 7 baselines)

Final metrics = mean over the last 10% of epochs. `Peak` = max of the smoothed win-rate curve reached at any point during training. `Trend` = OLS slope (pp per 1000 epochs) on the smoothed back half of the WR curve, decimated to the smoothing window for approximate independence; `↑` marks slopes whose 95% CI is strictly positive (i.e. the curve was still climbing at the end of training).

## Experiment runs

| Run                                        | Param               | Epochs | Final loss | Final vs bot_loss-BT | Peak vs bot_loss-BT | Trend vs bot_loss-BT | Final vs bot_random | Peak vs bot_random | Trend vs bot_random |
|--------------------------------------------|---------------------|--------|------------|----------------------|---------------------|----------------------|---------------------|--------------------|---------------------|
| Ve_oracleAblation(2)0519_DISABLE_2000      | DISABLE=2000        | 6000   | 0.1589     | 78.3%                | 80.3%               | +2.1↑                | 89.3%               | 91.1%              | +1.2↑               |
| Ve_oracleAblation(3)0519_DISABLE_4000      | DISABLE=4000        | 6000   | 0.1588     | 79.1%                | 81.5%               | +0.1                 | 89.5%               | 91.5%              | -0.0                |
| Ve_oracleAblation(4)0522_DISABLE_NEVER_10k | DISABLE_NEVER=10k   | 10000  | 0.0495     | 87.2%                | 88.9%               | +0.6↑                | 93.8%               | 94.8%              | +0.4↑               |
| Ve_oracleAblation(1)0519_DISABLE_NEVER     | Sweep=DISABLE_NEVER | 6000   | 0.0602     | 85.2%                | 87.1%               | +2.9↑                | 92.0%               | 92.9%              | +1.2↑               |

## Baselines

| Run                                          | Param                | Epochs | Final loss | Final vs bot_loss-BT | Peak vs bot_loss-BT | Trend vs bot_loss-BT | Final vs bot_random | Peak vs bot_random | Trend vs bot_random |
|----------------------------------------------|----------------------|--------|------------|----------------------|---------------------|----------------------|---------------------|--------------------|---------------------|
| ME_endgame(2)0429_ENDGAME_FRACTION_0.5       | ENDGAME_FRACTION=0.5 | 5000   | 0.1987     | 73.7%                | 75.0%               | +2.5↑                | 85.8%               | 86.9%              | +1.8↑               |
| LA_mcSelect(1)0422_N_LAST_STATES_INIT_2      | N_LAST_STATES_INIT=2 | 5000   | 0.0326     | 65.2%                | 68.3%               | +1.3↑                | 80.9%               | 82.5%              | +0.8↑               |
| MA_tempRegresive(1)0424_N_LAST_STATES_INIT_2 | N_LAST_STATES_INIT=2 | 5000   | 0.3143     | 56.0%                | 58.7%               | +1.0↑                | 72.8%               | 74.4%              | +0.8↑               |
| Ta_minimaxSelect(1)0514_DEPTH_2              | DEPTH=2              | 4000   | 0.0739     | 80.3%                | 81.5%               | +3.6↑                | 90.2%               | 90.9%              | +2.1↑               |
| JA_final(2)0420_N_LAST_STATES_INIT_3         | N_LAST_STATES_INIT=3 | 5000   | 0.1149     | 52.7%                | 54.9%               | +1.9↑                | 71.9%               | 74.0%              | +1.0↑               |
| MA_tempRegresive(3)0424_N_LAST_STATES_INIT_4 | N_LAST_STATES_INIT=4 | 5000   | 0.4200     | 49.7%                | 51.9%               | +1.2↑                | 70.7%               | 72.9%              | +0.7↑               |
| Aa_replay(2)0226_NUM_EPOCHs_BUFFER_8         | NUM_EPOCHs_BUFFER=8  | 5000   | 0.1094     | 65.7%                | 67.1%               | +1.3↑                | 81.0%               | 82.5%              | +1.1↑               |

## Best runs (experiments only)

- Lowest final loss: **Ve_oracleAblation(4)0522_DISABLE_NEVER_10k** — 0.0495
- Highest final WR vs bot_loss-BT: **Ve_oracleAblation(4)0522_DISABLE_NEVER_10k** — 87.2%
- Highest final WR vs bot_random: **Ve_oracleAblation(4)0522_DISABLE_NEVER_10k** — 93.8%
