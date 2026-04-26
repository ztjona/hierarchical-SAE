# Summary — Zpre_tempRegresive

Generated: 2026-04-24 19:02
Parameter varied: ``
Runs: 1 (+ 3 baselines)

Final metrics = mean over the last 10% of epochs. `Peak` = max of the smoothed win-rate curve reached at any point during training.

## Experiment runs

| Run             | Param | Epochs | Final loss | Final vs bot_loss-BT | Peak vs bot_loss-BT | Final vs bot_random | Peak vs bot_random |
|-----------------|-------|--------|------------|----------------------|---------------------|---------------------|--------------------|
| Zpre_tempRegresive | =1    | 2130   | 0.3480     | 50.0%                | 51.3%               | 69.5%               | 72.0%              |

## Baselines

| Run                                     | Param                | Epochs | Final loss | Final vs bot_loss-BT | Peak vs bot_loss-BT | Final vs bot_random | Peak vs bot_random |
|-----------------------------------------|----------------------|--------|------------|----------------------|---------------------|---------------------|--------------------|
| LA_mcSelect(1)0422_N_LAST_STATES_INIT_2 | N_LAST_STATES_INIT=2 | 5000   | 0.0326     | 65.2%                | 68.3%               | 80.9%               | 82.5%              |
| LA_mcSelect(2)0422_N_LAST_STATES_INIT_3 | N_LAST_STATES_INIT=3 | 5000   | 0.3220     | 30.0%                | 33.4%               | 51.5%               | 55.3%              |
| Aa_replay(2)0226_NUM_EPOCHs_BUFFER_8    | NUM_EPOCHs_BUFFER=8  | 5000   | 0.1094     | 65.7%                | 67.1%               | 81.0%               | 82.5%              |

## Best runs (experiments only)

- Lowest final loss: **Zpre_tempRegresive** — 0.3480
- Highest final WR vs bot_loss-BT: **Zpre_tempRegresive** — 50.0%
- Highest final WR vs bot_random: **Zpre_tempRegresive** — 69.5%
