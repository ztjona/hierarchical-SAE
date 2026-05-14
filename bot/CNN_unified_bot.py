# -*- coding: utf-8 -*-

"""
CNN_unified_bot - Autoregressive CNN bot with phase-stable unified aux.

Sibling to ``bot.CNN_autoreg_bot``. The constructor and public gameplay
methods are identical; the only difference is that this bot constructs a
32-d aux vector ``[offered_one_hot ; available_pieces_mask]`` at every
forward pass, regardless of phase. The trunk of ``QuartoCNNAutoregUnified``
is phase-agnostic — phase only routes which output head is read.

See ``docs/diary/2026-05-08_unified-aux-trunk.md``.
"""

"""
Python 3
2026-05-08
@author: z_tjona
"""

from models.CNN_autoreg import QuartoCNNAutoregUnified
from models.NN_abstract import NN_abstract

from quartopy import BotAI, Piece, QuartoGame

from utils.logger import logger
import numpy as np
import torch
import os
from tensordict import TensorDict


logger.debug("Loading CNN_unified_bot...")


UNIFIED_AUX_DIM = 32  # offered_one_hot (16) ⊕ available_pieces_mask (16)


class Quarto_bot(BotAI):
    @property
    def name(self) -> str:
        if hasattr(self, "model_path"):
            return f"CNN_unified_bot|{self.model_path}"
        elif hasattr(self, "model_name"):
            return f"CNN_unified_bot|{self.model_name}"
        else:
            return "CNN_unified_bot|random_weights"

    def __init__(
        self,
        *,
        model_path: str | None = None,
        model: NN_abstract | None = None,
        model_class: type[NN_abstract] = QuartoCNNAutoregUnified,
        deterministic: bool = True,
        temperature: float = 0.1,
    ):
        """Initialize the unified-aux autoregressive CNN bot.

        The interface mirrors ``bot.CNN_autoreg_bot.Quarto_bot``. The model
        must implement ``predict_phase``/``q_values_phase`` and accept aux
        of shape ``(batch, 32)``.
        """
        super().__init__()
        logger.debug("CNN_unified_bot initialized")

        assert not (
            model_path is not None and model is not None
        ), "Cannot provide both ``model_path`` and ``model`` instance. Choose one."

        if model_path:
            logger.debug(
                f"Loading unified model from {model_path} using {model_class.__name__}"
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
                f"Using provided unified model instance {self.model_name}"
            )
        else:
            logger.debug("Loading unified model with random weights")
            self.model = model_class()

        self.model.to(self.model.device)
        logger.debug(
            f"Unified model loaded successfully on device: {self.model.device}"
        )

        self.DETERMINISTIC: bool = deterministic
        self.TEMPERATURE: float = temperature

        # ``last_placed_piece`` tracks the piece this player most recently
        # placed in the current game. It feeds the ``offered`` block of the
        # 32-d aux during the select phase. Reset to -1 between games.
        self._last_placed_piece: int = -1

    # ──────────────────────────────────────────────────────────────────
    # encoding helpers
    # ──────────────────────────────────────────────────────────────────

    def _require_phase_predictor(self):
        predictor = getattr(self.model, "predict_phase", None)
        if predictor is None or not callable(predictor):
            raise TypeError(
                "CNN_unified_bot requires a model implementing "
                "predict_phase(x_board, x_aux, phase, TEMPERATURE, DETERMINISTIC)."
            )
        return predictor

    def _encode_board(self, game: QuartoGame) -> torch.Tensor:
        board_matrix = game.game_board.encode()
        return torch.from_numpy(board_matrix).float().to(self.model.device)

    def _piece_one_hot(self, piece_index: int) -> np.ndarray:
        oh = np.zeros(16, dtype=np.float32)
        if 0 <= piece_index < 16:
            oh[piece_index] = 1.0
        return oh

    def _available_pieces_mask(self, game: QuartoGame) -> np.ndarray:
        mask = np.zeros(16, dtype=np.float32)
        for piece in game.storage_board.get_valid_pieces():
            mask += piece.vectorize_onehot().reshape(-1)
        return mask

    def _build_unified_aux(
        self, game: QuartoGame, *, offered_index: int
    ) -> torch.Tensor:
        """Compose the 32-d phase-stable aux: offered ⊕ available."""
        offered = self._piece_one_hot(offered_index)
        available = self._available_pieces_mask(game)
        aux = np.concatenate([offered, available]).astype(np.float32).reshape(1, -1)
        return torch.from_numpy(aux).to(self.model.device)

    def _selected_piece_index(self, game: QuartoGame) -> int:
        if isinstance(game.selected_piece, Piece):
            return int(np.argmax(game.selected_piece.vectorize_onehot()))
        return -1

    # ──────────────────────────────────────────────────────────────────
    # ranking / option selection
    # ──────────────────────────────────────────────────────────────────

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
        assert valid_ranked, "No valid pieces were predicted by the unified model."

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
        ), "No valid board positions were predicted by the unified model."
        option_idx = min(ith_option, len(valid_ranked) - 1)
        return valid_ranked[option_idx]

    # ──────────────────────────────────────────────────────────────────
    # public gameplay API
    # ──────────────────────────────────────────────────────────────────

    def select(
        self,
        game: QuartoGame,
        ith_option: int = 0,
        *args,
        **kwargs,
    ) -> Piece:
        """Select a piece for the other player using a select-phase forward pass.

        Aux semantics: offered = the piece I just placed this turn (zero-vec
        on the very first select of a game), available = current storage pool.
        """
        # Reset cross-game state if quartopy has not yet placed anything this
        # game (storage at full 16 pieces, board empty, no selected piece).
        if (
            self._last_placed_piece != -1
            and len(game.storage_board.get_valid_pieces()) == 16
            and self._selected_piece_index(game) == -1
        ):
            self._last_placed_piece = -1

        board_tensor = self._encode_board(game)
        aux_tensor = self._build_unified_aux(
            game, offered_index=self._last_placed_piece
        )
        ordered_indices = self._predict_order(
            board_tensor,
            aux_tensor,
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
        """Place the currently selected piece using a place-phase forward pass.

        Aux semantics: offered = the piece in my hand (the one I am about to
        place), available = current storage pool.
        """
        offered_index = int(np.argmax(piece.vectorize_onehot()))

        board_tensor = self._encode_board(game)
        aux_tensor = self._build_unified_aux(game, offered_index=offered_index)
        ordered_indices = self._predict_order(
            board_tensor,
            aux_tensor,
            phase="place",
        )

        # Remember the piece for the upcoming select-phase aux.
        self._last_placed_piece = offered_index

        return self._choose_board_position(game, ordered_indices, ith_option)

    # ──────────────────────────────────────────────────────────────────
    # batch evaluation (for end-of-epoch Q-value diagnostics)
    # ──────────────────────────────────────────────────────────────────

    def evaluate(self, exp_batch: TensorDict):
        """Evaluate a unified-autoregressive batch.

        Mirrors ``CNN_autoreg_bot.Quarto_bot.evaluate``: returns one tensor
        per head with length equal to the batch size, populated only on rows
        whose phase matches the head; the inactive head is filled with NaN.

        Required batch keys: ``state_board``, ``state_aux`` (32-d), ``phase``,
        ``action``.
        """
        q_values_phase = getattr(self.model, "q_values_phase", None)
        if q_values_phase is None or not callable(q_values_phase):
            raise TypeError(
                "CNN_unified_bot.evaluate requires a model implementing "
                "q_values_phase(x_board, x_aux, phase)."
            )

        required_keys = {"state_board", "state_aux", "phase", "action"}
        missing_keys = sorted(required_keys.difference(exp_batch.keys()))
        if missing_keys:
            raise ValueError(
                "CNN_unified_bot.evaluate expects a unified batch with keys "
                f"{sorted(required_keys)}. Missing: {missing_keys}."
            )

        state_board: torch.Tensor = exp_batch["state_board"].to(self.model.device)
        state_aux: torch.Tensor = exp_batch["state_aux"].to(self.model.device)
        phase = exp_batch["phase"].to(device=self.model.device, dtype=torch.int64)
        action = exp_batch["action"].to(device=self.model.device, dtype=torch.int64)

        if state_aux.shape[-1] != UNIFIED_AUX_DIM:
            raise ValueError(
                f"CNN_unified_bot.evaluate expects state_aux of size "
                f"{UNIFIED_AUX_DIM}, got {tuple(state_aux.shape)}."
            )

        valid_mask = exp_batch.get("valid_mask")
        if valid_mask is not None:
            valid_mask = valid_mask.to(self.model.device)
            action_validity = valid_mask.gather(1, action.unsqueeze(1)).squeeze(1)
            if not (action_validity > 0).all():
                raise ValueError(
                    "CNN_unified_bot.evaluate encountered a row with an invalid chosen action."
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
