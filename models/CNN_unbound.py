# -*- coding: utf-8 -*-

"""
CNN_unbound - Same architecture as CNN_uncoupled but with unbounded Q-value outputs
(no tanh). Standard DQN practice — lets Bellman targets exceed [-1, 1].
"""

"""
Python 3
17 / 04 / 2026
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""
from models.NN_abstract import NN_abstract

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ----------------------------- logging --------------------------


class QuartoCNN(NN_abstract):
    """
    QuartoCNN with unbounded Q-value outputs (no tanh activation on heads).

    Same architecture as QuartoCNN_uncoupled:
        * Shared CNN backbone
        * Two independent output heads for placement and selection
        * But outputs are raw logits (unbounded) instead of tanh-squashed to [-1, 1]

    # Input:
    * batchx16x4x4 input tensors representing different positions of the game board.
    * batchx16 dims for each piece

    # Output:
    * batch-16 unbounded tensor: Q-values for board position placement
    * batch-16 unbounded tensor: Q-values for piece selection
    """

    @property
    def name(self) -> str:
        return "QuartoCNN_unbound"

    def __init__(self):
        super().__init__()
        # Input shape: (batch_size, 16, 4, 4)
        # (batch_size, dims, height, width)
        fc_inpiece_size = 16  # must be multiple of 16

        assert fc_inpiece_size % 16 == 0, "fc_inpiece_size must be a multiple of 16"
        self.fc_in_piece = nn.Linear(
            16, fc_inpiece_size
        )  # Input layer for piece features

        k1_size = 16
        self.conv1 = nn.Conv2d(
            16 + fc_inpiece_size // 16, k1_size, kernel_size=3, padding=1
        )
        k2_size = 32
        self.conv2 = nn.Conv2d(k1_size, k2_size, kernel_size=3, padding=1)
        n_neurons = 128
        self.fc1 = nn.Linear(k2_size * 4 * 4, n_neurons)

        # Predicts piece selection (now independent of board prediction)
        self.fc2_board = nn.Linear(n_neurons, 4 * 4)

        # piece: predicts piece to give
        self.fc2_piece = nn.Linear(n_neurons, 4 * 4)
        self.dropout = nn.Dropout(0.5)  # 0.3 before

    def forward(
        self, x_board: torch.Tensor | np.ndarray, x_piece: torch.Tensor | np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.
        Args:
            ``x_board``: Input tensor of the board with placed pieces (batch_size, 16, 4, 4).
            ``x_piece``: Input tensor of selected piece to place (batch_size, 16).
        Returns:
            qav_board: Unbounded Q-values for board position placement (batch_size, 16).
            qav_piece: Unbounded Q-values for piece selection (batch_size, 16).
        """
        piece_feat = F.relu(self.fc_in_piece(x_piece))
        piece_map = piece_feat.view(-1, 1, 4, 4)
        x = torch.cat([x_board, piece_map], dim=1)  # type: ignore
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Bx128

        # output 1: board position (batch, 16) — no activation
        qav_board = self.fc2_board(x)

        # output 2: selected piece (batch, 16) — no activation
        qav_piece = self.fc2_piece(x)
        return qav_board, qav_piece
