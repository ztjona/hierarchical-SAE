# -*- coding: utf-8 -*-

"""
CNN_autoreg_bot - Autoregressive CNN bot for Quarto.

This bot keeps the constructor and public gameplay methods close to CNN_bot,
but it expects a phase-aware model interface for separate place/select passes.
"""

"""
Python 3
23 / 04 / 2026
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""

from models.CNN_autoreg import QuartoCNNAutoreg
from models.NN_abstract import NN_abstract

from quartopy import BotAI, Piece, QuartoGame

from utils.logger import logger
import numpy as np
import torch
import os
from tensordict import TensorDict

logger.debug("Loading CNN_autoreg_bot...")


class Quarto_bot(BotAI):
    @property
    def name(self) -> str:
        if hasattr(self, "model_path"):
            return f"CNN_autoreg_bot|{self.model_path}"
        elif hasattr(self, "model_name"):
            return f"CNN_autoreg_bot|{self.model_name}"
        else:
            return "CNN_autoreg_bot|random_weights"

    def __init__(
        self,
        *,
        model_path: str | None = None,
        model: NN_abstract | None = None,
        model_class: type[NN_abstract] = QuartoCNNAutoreg,
        deterministic: bool = True,
        temperature: float = 0.1,
    ):
        """Initialize the autoregressive CNN bot.

        The interface intentionally mirrors bot/CNN_bot.py. The difference is
        that the underlying model must expose a phase-aware prediction method:

            predict_phase(x_board, x_aux, phase, TEMPERATURE, DETERMINISTIC)

        Optionally, future decoupled-transition training can use:

            q_values_phase(x_board, x_aux, phase)
        """
        super().__init__()
        logger.debug("CNN_autoreg_bot initialized")

        assert not (
            model_path is not None and model is not None
        ), "Cannot provide both ``model_path`` and ``model`` instance. Choose one."

        if model_path:
            logger.debug(
                f"Loading autoregressive model from {model_path} using {model_class.__name__}"
            )
            self.model = model_class.from_file(model_path)
            self.model_path = os.path.basename(model_path)
        elif model:
            assert isinstance(
                model, NN_abstract
            ), "Provided model must be a derived class of ``NN_abstract``."
            self.model = model
            self.model_name = model.name
            logger.debug(
                f"Using provided autoregressive model instance {self.model_name}"
            )
        else:
            logger.debug("Loading autoregressive model with random weights")
            self.model = model_class()

        self.model.to(self.model.device)
        logger.debug(
            f"Autoregressive model loaded successfully on device: {self.model.device}"
        )

        self.DETERMINISTIC: bool = deterministic
        self.TEMPERATURE: float = temperature

    def _require_phase_predictor(self):
        predictor = getattr(self.model, "predict_phase", None)
        if predictor is None or not callable(predictor):
            raise TypeError(
                "CNN_autoreg_bot requires a model implementing "
                "predict_phase(x_board, x_aux, phase, TEMPERATURE, DETERMINISTIC)."
            )
        return predictor

    def _encode_board(self, game: QuartoGame) -> torch.Tensor:
        board_matrix = game.game_board.encode()
        return torch.from_numpy(board_matrix).float().to(self.model.device)

    def _encode_incoming_piece(self, game: QuartoGame) -> torch.Tensor:
        if isinstance(game.selected_piece, Piece):
            piece_onehot = game.selected_piece.vectorize_onehot().reshape(1, -1)
        else:
            piece_onehot = np.zeros((1, 16), dtype=float)
        return torch.from_numpy(piece_onehot).float().to(self.model.device)

    def _encode_available_pieces(self, game: QuartoGame) -> torch.Tensor:
        available_mask = np.zeros((1, 16), dtype=float)
        valid_pieces = game.storage_board.get_valid_pieces()
        for piece in valid_pieces:
            available_mask += piece.vectorize_onehot().reshape(1, -1)
        return torch.from_numpy(available_mask).float().to(self.model.device)

    def _predict_order(
        self,
        board_tensor: torch.Tensor,
        aux_tensor: torch.Tensor,
        *,
        phase: str,
    ) -> torch.Tensor:
        predictor = self._require_phase_predictor()
        ordered_indices = predictor(
            board_tensor,
            aux_tensor,
            phase=phase,
            TEMPERATURE=self.TEMPERATURE,
            DETERMINISTIC=self.DETERMINISTIC,
        )
        batch_size = ordered_indices.shape[0]
        assert batch_size == 1, f"Expected batch size of 1, got {batch_size}."
        return ordered_indices

    def _valid_piece_indices(self, game: QuartoGame) -> set[int]:
        valid_pieces = game.storage_board.get_valid_pieces()
        return {int(np.argmax(piece.vectorize_onehot())) for piece in valid_pieces}

    def _choose_piece(
        self, game: QuartoGame, ordered_indices: torch.Tensor, ith_option: int
    ) -> Piece:
        valid_indices = self._valid_piece_indices(game)
        valid_ranked = [
            int(idx)
            for idx in ordered_indices[0].detach().cpu().tolist()
            if int(idx) in valid_indices
        ]
        assert (
            valid_ranked
        ), "No valid pieces were predicted by the autoregressive model."

        option_idx = min(ith_option, len(valid_ranked) - 1)
        return Piece.from_index(valid_ranked[option_idx])

    def _choose_board_position(
        self,
        game: QuartoGame,
        ordered_indices: torch.Tensor,
        ith_option: int,
    ) -> tuple[int, int]:
        valid_moves = set(game.game_board.get_valid_moves())
        valid_ranked: list[tuple[int, int]] = []
        for idx in ordered_indices[0].detach().cpu().tolist():
            board_position = game.game_board.get_position_index(int(idx))
            if board_position in valid_moves:
                valid_ranked.append(board_position)

        assert (
            valid_ranked
        ), "No valid board positions were predicted by the autoregressive model."
        option_idx = min(ith_option, len(valid_ranked) - 1)
        return valid_ranked[option_idx]

    def select(
        self,
        game: QuartoGame,
        ith_option: int = 0,
        *args,
        **kwargs,
    ) -> Piece:
        """Select a piece for the other player using a select-phase forward pass."""
        board_tensor = self._encode_board(game)
        available_tensor = self._encode_available_pieces(game)
        ordered_indices = self._predict_order(
            board_tensor,
            available_tensor,
            phase="select",
        )
        return self._choose_piece(game, ordered_indices, ith_option)

    def place_piece(
        self,
        game: QuartoGame,
        piece: Piece,
        ith_option: int = 0,
        *args,
        **kwargs,
    ) -> tuple[int, int]:
        """Place the currently selected piece using a place-phase forward pass."""
        board_tensor = self._encode_board(game)
        piece_tensor = self._encode_incoming_piece(game)
        ordered_indices = self._predict_order(
            board_tensor,
            piece_tensor,
            phase="place",
        )
        return self._choose_board_position(game, ordered_indices, ith_option)

    def evaluate(self, exp_batch: TensorDict):
        """Evaluate a decoupled autoregressive batch.

        This method intentionally keeps the same public name as CNN_bot, but it
        is only meaningful for the decoupled-transition batch format. The
        expected batch keys are:

        - state_board
        - state_aux
        - phase
        - action

        Returns one tensor per head with length equal to the batch size. Only
        the active phase is populated on each row; the inactive head is filled
        with NaN so downstream diagnostics do not confuse cross-phase values.
        """
        q_values_phase = getattr(self.model, "q_values_phase", None)
        if q_values_phase is None or not callable(q_values_phase):
            raise TypeError(
                "CNN_autoreg_bot.evaluate requires a model implementing "
                "q_values_phase(x_board, x_aux, phase)."
            )

        required_keys = {"state_board", "state_aux", "phase", "action"}
        missing_keys = sorted(required_keys.difference(exp_batch.keys()))
        if missing_keys:
            raise ValueError(
                "CNN_autoreg_bot.evaluate expects a decoupled batch with keys "
                f"{sorted(required_keys)}. Missing: {missing_keys}."
            )

        state_board: torch.Tensor = exp_batch["state_board"].to(self.model.device)
        state_aux: torch.Tensor = exp_batch["state_aux"].to(self.model.device)
        phase = exp_batch["phase"].to(device=self.model.device, dtype=torch.int64)
        action = exp_batch["action"].to(device=self.model.device, dtype=torch.int64)

        valid_mask = exp_batch.get("valid_mask")
        if valid_mask is not None:
            valid_mask = valid_mask.to(self.model.device)
            action_validity = valid_mask.gather(1, action.unsqueeze(1)).squeeze(1)
            if not (action_validity > 0).all():
                raise ValueError(
                    "CNN_autoreg_bot.evaluate encountered a row with an invalid chosen action."
                )

        self.model.eval()
        with torch.no_grad():
            qav_place, qav_select = self.model.forward(
                state_board,
                state_aux,
                phase=phase,
            )

        batch_size = state_board.shape[0]
        batch_indices = torch.arange(batch_size, device=self.model.device)
        place_mask = phase == 0
        select_mask = phase == 1

        q_place = torch.full(
            (batch_size,),
            float("nan"),
            dtype=qav_place.dtype,
            device=self.model.device,
        )
        q_select = torch.full(
            (batch_size,),
            float("nan"),
            dtype=qav_select.dtype,
            device=self.model.device,
        )

        if place_mask.any():
            q_place[place_mask] = qav_place[
                batch_indices[place_mask], action[place_mask]
            ]

        if select_mask.any():
            q_select[select_mask] = qav_select[
                batch_indices[select_mask], action[select_mask]
            ]

        return q_place, q_select
