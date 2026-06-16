# Summary — Yb_hotChamp

Generated: 2026-06-15 13:53
Parameter varied: `HOT`
Runs: 4 (+ 8 baselines)

Final metrics = mean over the last 10% of epochs. `Peak` = max of the smoothed win-rate curve reached at any point during training. `Trend` = OLS slope (pp per 1000 epochs) on the smoothed back half of the WR curve, decimated to the smoothing window for approximate independence; `↑` marks slopes whose 95% CI is strictly positive (i.e. the curve was still climbing at the end of training).

## Experiment runs

| Run                              | Param         | Epochs | Final loss | Final vs bot_loss-BT | Peak vs bot_loss-BT | Trend vs bot_loss-BT | Final vs bot_random | Peak vs bot_random | Trend vs bot_random |
|----------------------------------|---------------|--------|------------|----------------------|---------------------|----------------------|---------------------|--------------------|---------------------|
| Yb_hotChamp(1)0612_HOT_0.3       | HOT=0.3       | 10000  | 0.0750     | 94.5%                | 95.6%               | +0.3↑                | 97.1%               | 97.7%              | +0.1↑               |
| Yb_hotChamp(2)0612_HOT_1.0       | HOT=1         | 10000  | 0.1241     | 95.1%                | 96.2%               | +0.2↑                | 97.4%               | 98.1%              | +0.1↑               |
| Yb_hotChamp(4)0612_HOT_2.0       | HOT=2         | 10000  | 0.2301     | 94.3%                | 95.3%               | +0.7↑                | 97.1%               | 98.0%              | +0.3↑               |
| Yb_hotChamp(3)0612_HOT_1.0_seedB | HOT=1.0_seedB | 10000  | 0.1266     | 95.1%                | 96.2%               | +0.2↑                | 97.5%               | 98.3%              | +0.1↑               |

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
| Wa_oracleStates(1)0525_N_LAST_STATES_8       | N_LAST_STATES=8      | 10000  | 0.0356     | 87.0%                | 88.3%               | +0.6↑                | 93.3%               | 94.5%              | +0.4↑               |

## Best runs (experiments only)

- Lowest final loss: **Yb_hotChamp(1)0612_HOT_0.3** — 0.0750
- Highest final WR vs bot_loss-BT: **Yb_hotChamp(2)0612_HOT_1.0** — 95.1%
- Highest final WR vs bot_random: **Yb_hotChamp(3)0612_HOT_1.0_seedB** — 97.5%
