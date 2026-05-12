"""QC (series Q) — Unified-aux trunk WITHOUT inference legality mask.

Architectural rationale (see Research-status.md → QC_unifiedNoMask):

This is a deliberate departure from ME(2). The interp work in
`games-interp` revealed (Phase 1G audit) that:
  - The current network does NOT learn cell-legality (Test D mean gap = −0.47);
    the bot's inference-time validity filter does all the work.
  - fc1 destroys the threat information conv2 still carries (LP F1 drops
    0.50 → 0.02).

QC changes three things on top of `QuartoCNNAutoregUnified` (the OA trunk):

  1. **Wider fc1 (128 → 256).** Cheap capacity check on the fc1 bottleneck.
  2. **Auxiliary legality head** ``fc_aux_legality: Linear(n_neurons, 16)``
     producing 16 logits trained with BCEWithLogits against
     ``is_empty(cell)`` derived from ``x_board``. This forces fc1 to
     preserve legality information — a substrate fix, independent of how
     the bot uses Q_place at inference.
  3. **No phase embedding.** Inherited from the unified-aux design — the
     32-d aux ``[offered_one_hot ; available_mask]`` is phase-stable, so
     the trunk stays phase-agnostic. Same SAE pipeline, single
     activation distribution per layer.

What does NOT change relative to OA:
  - Two-head output (``fc2_place``, ``fc2_select``), max independently.
  - ``tanh`` cap on both heads (final reward is in [-1, 1]).
  - Same trunk shape downstream of fc1.

The "no validity mask" half of the architecture lives in the matching
bot ``bot/CNN_unified_nomask_bot.py``: that bot picks argmax-Q_place
directly without filtering occupied cells. Combined with the legality
aux loss, this lets us *measure* whether the network has learned
legality, instead of papering over the failure at inference.

See ``docs/diary/2026-05-08_unified-aux-trunk.md`` for the 32-d aux
contract that QC inherits, and ``docs/diary/2026-05-11_qc-no-mask.md``
for QC's own design notes.
"""

from models.NN_abstract import NN_abstract

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

PHASE_PLACE = 0
PHASE_SELECT = 1


def legality_target_from_board(x_board: torch.Tensor) -> torch.Tensor:
    """Compute the per-cell legality label from a board encoding.

    Parameters
    ----------
    x_board : (B, 16, 4, 4)
        One-hot board state (one channel per piece). A cell is occupied iff
        any of the 16 piece channels is set at that (row, col).

    Returns
    -------
    legality : (B, 16) float tensor, 1.0 = empty (legal place target),
        0.0 = occupied. Flattened in row-major order to match ``Q_place``.
    """
    if x_board.dim() != 4 or x_board.shape[1:] != (16, 4, 4):
        raise ValueError(
            f"Expected x_board of shape (B, 16, 4, 4); got {tuple(x_board.shape)}."
        )
    occupied = x_board.sum(dim=1) > 0  # (B, 4, 4)
    legality = (~occupied).reshape(x_board.shape[0], 16).to(x_board.dtype)
    return legality


def _normalize_phase_tensor(
    phase, batch_size: int, device: torch.device
) -> torch.Tensor:
    """Coerce ``phase`` into a 1-D long tensor of shape (batch_size,).

    Accepted: ``"place"`` / ``"select"`` strings, an int, a 0-d or 1-d
    tensor / array / list of ints.
    """
    if isinstance(phase, str):
        phase_idx = PHASE_PLACE if phase == "place" else PHASE_SELECT
        return torch.full((batch_size,), phase_idx, dtype=torch.long, device=device)
    if isinstance(phase, int):
        return torch.full((batch_size,), phase, dtype=torch.long, device=device)
    if isinstance(phase, list):
        phase_tensor = torch.tensor(phase, dtype=torch.long, device=device)
    elif isinstance(phase, np.ndarray):
        phase_tensor = torch.from_numpy(phase).to(device=device, dtype=torch.long)
    elif isinstance(phase, torch.Tensor):
        phase_tensor = phase.to(device=device, dtype=torch.long)
    else:
        raise TypeError(f"Unsupported phase type: {type(phase)}")
    if phase_tensor.ndim == 0:
        phase_tensor = phase_tensor.repeat(batch_size)
    phase_tensor = phase_tensor.reshape(-1)
    if phase_tensor.shape[0] != batch_size:
        raise ValueError(
            f"Phase batch size mismatch. Expected {batch_size}, got {phase_tensor.shape[0]}."
        )
    return phase_tensor


class QuartoCNNUnifiedNoMask(NN_abstract):
    """Unified-aux trunk with wider fc1 and aux legality head.

    Architecture summary
    --------------------
    Input
      x_board : (B, 16, 4, 4) one-hot piece-channel encoding
      x_aux   : (B, 32)        offered_one_hot ⊕ available_mask (phase-stable)
    Trunk (phase-agnostic — no phase embedding)
      fc_in_aux : Linear(32 → 32) + ReLU  → reshape (B, 2, 4, 4)
      cat       : (B, 18, 4, 4)
      conv1     : Conv2d(18 → 16, 3×3, pad=1) + ReLU
      conv2     : Conv2d(16 → 32, 3×3, pad=1) + ReLU
      flatten   : (B, 512)
      fc1       : Linear(512 → 256) + ReLU + Dropout(0.5)        ← wider
    Heads
      fc2_place        : Linear(256 → 16) + tanh
      fc2_select       : Linear(256 → 16) + tanh
      fc_aux_legality  : Linear(256 → 16)  (raw logits; BCEWithLogits in loss)
    """

    AUX_DIM: int = 32
    N_NEURONS: int = 256  # wider than the 128-d fc1 in OA/ME

    def __init__(self):
        super().__init__()

        fc_aux_size = 32
        assert fc_aux_size % 16 == 0, "fc_aux_size must be a multiple of 16"

        self.fc_in_aux = nn.Linear(self.AUX_DIM, fc_aux_size)

        trunk_in_channels = 16 + (fc_aux_size // 16)
        k1_size = 16
        k2_size = 32

        self.conv1 = nn.Conv2d(trunk_in_channels, k1_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(k1_size, k2_size, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(k2_size * 4 * 4, self.N_NEURONS)

        self.fc2_place = nn.Linear(self.N_NEURONS, 16)
        self.fc2_select = nn.Linear(self.N_NEURONS, 16)
        self.fc_aux_legality = nn.Linear(self.N_NEURONS, 16)

        self.dropout = nn.Dropout(0.5)

    @property
    def name(self) -> str:
        return "QuartoCNN_unified_nomask"

    # ──────────────────────────────────────────────────────────────────
    def _normalize_phase(self, phase, batch_size, device):
        return _normalize_phase_tensor(phase, batch_size, device)

    def _shared_trunk(self, x_board, x_aux):
        if isinstance(x_board, np.ndarray):
            x_board = torch.from_numpy(x_board).float()
        if isinstance(x_aux, np.ndarray):
            x_aux = torch.from_numpy(x_aux).float()
        x_board = x_board.to(self.device).float()
        x_aux = x_aux.to(self.device).float()

        if x_aux.shape[-1] != self.AUX_DIM:
            raise ValueError(
                f"QC expects aux of size {self.AUX_DIM} (offered⊕available); "
                f"got {tuple(x_aux.shape)}."
            )

        aux_feat = F.relu(self.fc_in_aux(x_aux))
        aux_channels = aux_feat.shape[-1] // 16
        aux_map = aux_feat.view(-1, aux_channels, 4, 4)

        x = torch.cat([x_board, aux_map], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return x

    # ──────────────────────────────────────────────────────────────────
    def forward(self, x_board, x_aux, phase="place"):
        """Return (q_place, q_select) for API parity with the unified trunk.

        ``phase`` is accepted and ignored — the trunk is phase-agnostic.
        """
        del phase
        x = self._shared_trunk(x_board, x_aux)
        q_place = torch.tanh(self.fc2_place(x))
        q_select = torch.tanh(self.fc2_select(x))
        return q_place, q_select

    def forward_with_aux(self, x_board, x_aux):
        """Forward pass returning all three heads plus the legality logits.

        Used by the training loop to compute the auxiliary BCE legality loss
        without re-running the trunk.
        """
        x = self._shared_trunk(x_board, x_aux)
        q_place = torch.tanh(self.fc2_place(x))
        q_select = torch.tanh(self.fc2_select(x))
        legality_logits = self.fc_aux_legality(x)
        return q_place, q_select, legality_logits

    def legality_logits(self, x_board, x_aux) -> torch.Tensor:
        """Return raw 16-d legality logits (no sigmoid)."""
        x = self._shared_trunk(x_board, x_aux)
        return self.fc_aux_legality(x)

    def q_values_phase(self, x_board, x_aux, phase) -> torch.Tensor:
        q_place, q_select = self.forward(x_board, x_aux, phase=phase)
        phase_tensor = self._normalize_phase(phase, q_place.shape[0], q_place.device)
        select_mask = phase_tensor == PHASE_SELECT
        return torch.where(select_mask.unsqueeze(1), q_select, q_place)

    def predict_phase(
        self,
        x_board,
        x_aux,
        *,
        phase,
        TEMPERATURE: float = 1.0,
        DETERMINISTIC: bool = True,
    ) -> torch.Tensor:
        assert x_board.shape[0] == 1, "Batch size of 1 is required for prediction"
        self.eval()
        with torch.no_grad():
            q_values = self.q_values_phase(x_board, x_aux, phase=phase)
            if DETERMINISTIC:
                return torch.argsort(q_values, descending=True, dim=1)
            probs = F.softmax(q_values / TEMPERATURE, dim=1)
            return torch.multinomial(probs, probs.shape[1], replacement=False)
