from .NN_abstract import NN_abstract

# Autoregressive phase-aware models
from .CNN_autoreg import (
    QuartoCNNAutoreg,
    QuartoCNNAutoregUnbound,
    QuartoCNNAutoregSepTrunks,
    QuartoCNNAutoregUnified,
    QuartoCNNAutoregUnifiedUnbound,
)

from .CNN_autoreg_sa import (
    QuartoCNNAutoregUnifiedS1,
    QuartoCNNAutoregUnifiedS2,
    QuartoCNNAutoregUnifiedS4,
    QuartoCNNAutoregUnifiedS4Hot,
)

# from .CNN1 import QuartoCNN
