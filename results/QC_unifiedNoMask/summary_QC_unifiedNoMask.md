# Summary — QC_unifiedNoMask

Generated: 2026-05-12 12:33
Parameter varied: `N_LAST_STATES_INIT`
Runs: 1 (+ 6 baselines)

Final metrics = mean over the last 10% of epochs. `Peak` = max of the smoothed win-rate curve reached at any point during training.

## Experiment runs

| Run                     | Param | Epochs | Final loss | Final vs ME_endgame(2)_E_5000 | Peak vs ME_endgame(2)_E_5000 | Final vs bot_random | Peak vs bot_random |
|-------------------------|-------|--------|------------|-------------------------------|------------------------------|---------------------|--------------------|
| QC_unifiedNoMask(1)0511 | run=1 | 5000   | 0.2043     | 46.2%                         | 48.7%                        | 85.0%               | 86.1%              |

## Baselines

| Run                                          | Param                | Epochs | Final loss | Final vs ME_endgame(2)_E_5000 | Peak vs ME_endgame(2)_E_5000 | Final vs bot_random | Peak vs bot_random |
|----------------------------------------------|----------------------|--------|------------|-------------------------------|------------------------------|---------------------|--------------------|
| LA_mcSelect(1)0422_N_LAST_STATES_INIT_2      | N_LAST_STATES_INIT=2 | 5000   | 0.0326     | —                             | —                            | 80.9%               | 82.5%              |
| MA_tempRegresive(1)0424_N_LAST_STATES_INIT_2 | N_LAST_STATES_INIT=2 | 5000   | 0.3143     | —                             | —                            | 72.8%               | 74.4%              |
| JA_final(2)0420_N_LAST_STATES_INIT_3         | N_LAST_STATES_INIT=3 | 5000   | 0.1149     | —                             | —                            | 71.9%               | 74.0%              |
| LA_mcSelect(2)0422_N_LAST_STATES_INIT_3      | N_LAST_STATES_INIT=3 | 5000   | 0.3220     | —                             | —                            | 51.5%               | 55.3%              |
| MA_tempRegresive(3)0424_N_LAST_STATES_INIT_4 | N_LAST_STATES_INIT=4 | 5000   | 0.4200     | —                             | —                            | 70.7%               | 72.9%              |
| Aa_replay(2)0226_NUM_EPOCHs_BUFFER_8         | NUM_EPOCHs_BUFFER=8  | 5000   | 0.1094     | —                             | —                            | 81.0%               | 82.5%              |

## Best runs (experiments only)

- Lowest final loss: **QC_unifiedNoMask(1)0511** — 0.2043
- Highest final WR vs ME_endgame(2)_E_5000: **QC_unifiedNoMask(1)0511** — 46.2%
- Highest final WR vs bot_random: **QC_unifiedNoMask(1)0511** — 85.0%
