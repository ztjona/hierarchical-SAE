# Training Summary — C1_unifiedAux

**Date**: 2026-05-14 12:31

## Config
| Key | Value |
|-----|-------|
| model | QuartoCNNAutoregUnified |
| data | `projects\supervised-cloning\data\C1_unified.npz` |
| epochs | 150 |
| batch | 256 |
| lr | 0.001 |
| λ (select weight) | 1.0 |
| soft_weight | 1.0 |
| val_split | 0.15 |
| seed | 42 |
| device | cpu |

## Dataset
| | Samples |
|--|--|
| train (after aug) | 329,312 |
| val (no aug) | 7,274 |

## Results
| Metric | Train | Val |
|--------|-------|-----|
| Best epoch | — | 2 |
| Best val acc (avg heads) | — | 17.34% |
| Final loss | 4.1251 | 4.2901 |
| PLACE top-1 (final) | 8.60% | 18.15% |
| PLACE top-3 (final) | 23.81% | 25.47% |
| SELECT top-1 (final) | 10.89% | 6.17% |
| SELECT top-3 (final) | 23.83% | 24.14% |

## Win-rate evaluation (best checkpoint, 50 matches each)
| Baseline | Win rate |
|----------|----------|
| random | 72.00% |
| minimax_d2 | 0.00% |

![Training curves](training_curves.png)

## Notes
