# -*- coding: utf-8 -*-

"""
Autoregressive CNN models for Quarto.

These models preserve the familiar shared-trunk/two-head layout, but change the
contract: inference happens in two phase-aware passes, one for placement and one
for piece selection.
"""

"""
Python 3
23 / 04 / 2026
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""

from models.NN_abstract import NN_abstract

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


PHASE_PLACE = 0
PHASE_SELECT = 1


class _QuartoCNNAutoregBase(NN_abstract):
    def __init__(self):
        super().__init__()

        fc_aux_size = 16
        assert fc_aux_size % 16 == 0, "fc_aux_size must be a multiple of 16"

        self.fc_in_aux = nn.Linear(16, fc_aux_size)
        self.phase_embedding = nn.Embedding(2, 16)

        trunk_in_channels = 16 + (fc_aux_size // 16) + 1
        k1_size = 16
        k2_size = 32
        n_neurons = 128

        self.conv1 = nn.Conv2d(trunk_in_channels, k1_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(k1_size, k2_size, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(k2_size * 4 * 4, n_neurons)
        self.fc2_place = nn.Linear(n_neurons, 16)
        self.fc2_select = nn.Linear(n_neurons, 16)
        self.dropout = nn.Dropout(0.5)

    def _normalize_phase(
        self,
        phase: str | int | torch.Tensor | np.ndarray | list[int],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
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

    def _shared_trunk(
        self,
        x_board: torch.Tensor | np.ndarray,
        x_aux: torch.Tensor | np.ndarray,
        phase: str | int | torch.Tensor | np.ndarray | list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x_board, np.ndarray):
            x_board = torch.from_numpy(x_board).float()
        if isinstance(x_aux, np.ndarray):
            x_aux = torch.from_numpy(x_aux).float()

        x_board = x_board.to(self.device)
        x_aux = x_aux.to(self.device)
        batch_size = x_board.shape[0]

        phase_tensor = self._normalize_phase(phase, batch_size, x_board.device)
        aux_feat = F.relu(self.fc_in_aux(x_aux))
        aux_map = aux_feat.view(-1, 1, 4, 4)

        phase_feat = self.phase_embedding(phase_tensor)
        phase_map = phase_feat.view(-1, 1, 4, 4)

        x = torch.cat([x_board, aux_map, phase_map], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return x, phase_tensor

    def _apply_output_activation(self, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        x_board: torch.Tensor | np.ndarray,
        x_aux: torch.Tensor | np.ndarray,
        phase: str | int | torch.Tensor | np.ndarray | list[int] = "place",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, _ = self._shared_trunk(x_board, x_aux, phase)
        logits_place = self.fc2_place(x)
        logits_select = self.fc2_select(x)
        q_place = self._apply_output_activation(logits_place)
        q_select = self._apply_output_activation(logits_select)
        return q_place, q_select

    def q_values_phase(
        self,
        x_board: torch.Tensor,
        x_aux: torch.Tensor,
        phase: str | int | torch.Tensor | np.ndarray | list[int],
    ) -> torch.Tensor:
        q_place, q_select = self.forward(x_board, x_aux, phase=phase)
        phase_tensor = self._normalize_phase(phase, q_place.shape[0], q_place.device)
        select_mask = phase_tensor == PHASE_SELECT
        return torch.where(select_mask.unsqueeze(1), q_select, q_place)

    def predict_phase(
        self,
        x_board: torch.Tensor,
        x_aux: torch.Tensor,
        *,
        phase: str | int,
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
            return torch.multinomial(
                probs,
                probs.shape[1],
                replacement=False,
            )


class QuartoCNNAutoreg(_QuartoCNNAutoregBase):
    @property
    def name(self) -> str:
        return "QuartoCNN_autoreg"

    def _apply_output_activation(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.tanh(logits)


class QuartoCNNAutoregUnbound(_QuartoCNNAutoregBase):
    @property
    def name(self) -> str:
        return "QuartoCNN_autoreg_unbound"

    def _apply_output_activation(self, logits: torch.Tensor) -> torch.Tensor:
        return logits
