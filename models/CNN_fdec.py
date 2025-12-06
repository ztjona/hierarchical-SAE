"""
Adaptador para cargar modelos CNN de diferentes arquitecturas.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple
from models.NN_abstract import NN_abstract


class QuartoCNNExtended(NN_abstract):
    """
    Arquitectura CNN extendida para Quarto con BatchNorm y capas adicionales.
    Soporta: fc_in_piece, conv1, bn1, conv2, bn2, fc1, bn_fc1, fc1b, bn_fc1b,
             fc1c, bn_fc1c, fc1d, bn_fc1d, fc2_board, fc2_piece
    """

    @property
    def name(self) -> str:
        return "Quarto_Fdec"

    def __init__(
        self,
        weights_path="CHECKPOINTS//Francis//20251204_0932-ba_increasing_n_last_states_epoch_0505.pt",
    ):
        super().__init__()
        state_dict = torch.load(weights_path, weights_only=True)
        self._crear_capas_desde_state_dict(state_dict)

    def _crear_capas_desde_state_dict(self, state_dict: dict):
        """Crea todas las capas basándose en el state_dict."""

        # fc_in_piece
        if "fc_in_piece.weight" in state_dict:
            w = state_dict["fc_in_piece.weight"]
            self.fc_in_piece = nn.Linear(w.shape[1], w.shape[0])
        else:
            self.fc_in_piece = nn.Linear(16, 16)

        # conv1
        if "conv1.weight" in state_dict:
            w = state_dict["conv1.weight"]
            self.conv1 = nn.Conv2d(
                w.shape[1], w.shape[0], kernel_size=w.shape[2], padding=w.shape[2] // 2
            )
            conv1_out = w.shape[0]
        else:
            self.conv1 = nn.Conv2d(17, 16, kernel_size=3, padding=1)
            conv1_out = 16

        # bn1 (BatchNorm después de conv1)
        if "bn1.weight" in state_dict:
            self.bn1 = nn.BatchNorm2d(conv1_out)
            self.has_bn1 = True
        else:
            self.has_bn1 = False

        # conv2
        if "conv2.weight" in state_dict:
            w = state_dict["conv2.weight"]
            self.conv2 = nn.Conv2d(
                w.shape[1], w.shape[0], kernel_size=w.shape[2], padding=w.shape[2] // 2
            )
            conv2_out = w.shape[0]
        else:
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            conv2_out = 32

        # bn2 (BatchNorm después de conv2)
        if "bn2.weight" in state_dict:
            self.bn2 = nn.BatchNorm2d(conv2_out)
            self.has_bn2 = True
        else:
            self.has_bn2 = False

        # fc1
        if "fc1.weight" in state_dict:
            w = state_dict["fc1.weight"]
            self.fc1 = nn.Linear(w.shape[1], w.shape[0])
            fc1_out = w.shape[0]
        else:
            self.fc1 = nn.Linear(conv2_out * 16, 256)
            fc1_out = 256

        # bn_fc1
        if "bn_fc1.weight" in state_dict:
            self.bn_fc1 = nn.BatchNorm1d(fc1_out)
            self.has_bn_fc1 = True
        else:
            self.has_bn_fc1 = False

        # fc1b (capa adicional)
        if "fc1b.weight" in state_dict:
            w = state_dict["fc1b.weight"]
            self.fc1b = nn.Linear(w.shape[1], w.shape[0])
            fc1b_out = w.shape[0]
            self.has_fc1b = True
        else:
            self.has_fc1b = False
            fc1b_out = fc1_out

        # bn_fc1b
        if "bn_fc1b.weight" in state_dict:
            self.bn_fc1b = nn.BatchNorm1d(fc1b_out)
            self.has_bn_fc1b = True
        else:
            self.has_bn_fc1b = False

        # fc1c
        if "fc1c.weight" in state_dict:
            w = state_dict["fc1c.weight"]
            self.fc1c = nn.Linear(w.shape[1], w.shape[0])
            fc1c_out = w.shape[0]
            self.has_fc1c = True
        else:
            self.has_fc1c = False
            fc1c_out = fc1b_out

        # bn_fc1c
        if "bn_fc1c.weight" in state_dict:
            self.bn_fc1c = nn.BatchNorm1d(fc1c_out)
            self.has_bn_fc1c = True
        else:
            self.has_bn_fc1c = False

        # fc1d
        if "fc1d.weight" in state_dict:
            w = state_dict["fc1d.weight"]
            self.fc1d = nn.Linear(w.shape[1], w.shape[0])
            fc1d_out = w.shape[0]
            self.has_fc1d = True
        else:
            self.has_fc1d = False
            fc1d_out = fc1c_out

        # bn_fc1d
        if "bn_fc1d.weight" in state_dict:
            self.bn_fc1d = nn.BatchNorm1d(fc1d_out)
            self.has_bn_fc1d = True
        else:
            self.has_bn_fc1d = False

        # fc2_board
        if "fc2_board.weight" in state_dict:
            w = state_dict["fc2_board.weight"]
            self.fc2_board = nn.Linear(w.shape[1], w.shape[0])
        else:
            self.fc2_board = nn.Linear(fc1d_out, 16)

        # fc2_piece
        if "fc2_piece.weight" in state_dict:
            w = state_dict["fc2_piece.weight"]
            self.fc2_piece = nn.Linear(w.shape[1], w.shape[0])
        else:
            self.fc2_piece = nn.Linear(fc1d_out, 16)

        # Dropout
        self.dropout = nn.Dropout(0.2)

    def forward(self, x_board, x_piece=None):
        """Forward pass con arquitectura extendida."""
        device = next(self.parameters()).device
        batch_size = 1

        # Preparar tablero como 16 canales one-hot
        if (
            isinstance(x_board, torch.Tensor)
            and x_board.dim() == 4
            and x_board.shape[1] == 16
        ):
            x_board = x_board.to(device)
            batch_size = x_board.shape[0]
        else:
            if isinstance(x_board, torch.Tensor):
                tablero_flat = x_board.view(-1).tolist()
            elif isinstance(x_board, list):
                if isinstance(x_board[0], list):
                    tablero_flat = [item for row in x_board for item in row]
                else:
                    tablero_flat = x_board
            else:
                tablero_flat = list(x_board)

            x_board_tensor = torch.zeros(1, 16, 4, 4, device=device)
            for pos in range(16):
                pieza_en_pos = int(tablero_flat[pos])
                if 0 <= pieza_en_pos < 16:
                    row = pos // 4
                    col = pos % 4
                    x_board_tensor[0, pieza_en_pos, row, col] = 1.0
            x_board = x_board_tensor

        # Preparar pieza como one-hot
        if x_piece is None:
            x_piece_oh = torch.zeros(batch_size, 16, device=device)
        elif isinstance(x_piece, int):
            x_piece_oh = torch.zeros(batch_size, 16, device=device)
            if 0 <= x_piece < 16:
                x_piece_oh[:, x_piece] = 1.0
        elif isinstance(x_piece, torch.Tensor):
            if x_piece.dim() == 0:
                x_piece_oh = torch.zeros(batch_size, 16, device=device)
                idx = int(x_piece.item())
                if 0 <= idx < 16:
                    x_piece_oh[:, idx] = 1.0
            elif x_piece.shape[-1] == 16:
                x_piece_oh = x_piece.float().to(device)
                if x_piece_oh.dim() == 1:
                    x_piece_oh = x_piece_oh.unsqueeze(0)
            else:
                x_piece_oh = torch.zeros(batch_size, 16, device=device)
                idx = int(x_piece.view(-1)[0].item())
                if 0 <= idx < 16:
                    x_piece_oh[:, idx] = 1.0
        else:
            x_piece_oh = torch.zeros(batch_size, 16, device=device)

        # Procesar pieza
        piece_feat = F.relu(self.fc_in_piece(x_piece_oh))
        piece_map = piece_feat.view(batch_size, 1, 4, 4)

        # Concatenar tablero + piece_map
        x = torch.cat([x_board, piece_map], dim=1)

        # Conv1 + BN1
        x = self.conv1(x)
        if self.has_bn1:
            x = self.bn1(x)
        x = F.relu(x)

        # Conv2 + BN2
        x = self.conv2(x)
        if self.has_bn2:
            x = self.bn2(x)
        x = F.relu(x)

        # Flatten
        x = x.flatten(start_dim=1)

        # FC1 + BN_FC1
        x = self.fc1(x)
        if self.has_bn_fc1:
            x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # FC1B + BN_FC1B
        if self.has_fc1b:
            x = self.fc1b(x)
            if self.has_bn_fc1b:
                x = self.bn_fc1b(x)
            x = F.relu(x)
            x = self.dropout(x)

        # FC1C + BN_FC1C
        if self.has_fc1c:
            x = self.fc1c(x)
            if self.has_bn_fc1c:
                x = self.bn_fc1c(x)
            x = F.relu(x)
            x = self.dropout(x)

        # FC1D + BN_FC1D
        if self.has_fc1d:
            x = self.fc1d(x)
            if self.has_bn_fc1d:
                x = self.bn_fc1d(x)
            x = F.relu(x)

        x = self.dropout(x)

        # Salidas
        logits_board = self.fc2_board(x)
        logits_piece = self.fc2_piece(x)

        return logits_board, logits_piece
