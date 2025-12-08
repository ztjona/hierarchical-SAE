# -*- coding: utf-8 -*-

"""
CNN_bot - Bot based in CNN to play Quarto
"""

"""
Python 3
26 / 05 / 2025
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""

from models.CNN1 import QuartoCNN
from models.NN_abstract import NN_abstract

from quartopy import BotAI, Piece, QuartoGame

from utils.logger import logger
import numpy as np
import torch
import os
from tensordict import TensorDict

logger.debug("Loading CNN_bot...")


class Quarto_bot(BotAI):
    @property
    def name(self) -> str:
        if hasattr(self, "model_path"):
            return f"CNN_bot|{self.model_path}"
        elif hasattr(self, "model_name"):
            return f"CNN_bot|{self.model_name}"
        else:
            return "CNN_bot|random_weights"

    def __init__(
        self,
        *,
        model_path: str | None = None,
        model: NN_abstract | None = None,
        model_class: type[NN_abstract] = QuartoCNN,
        deterministic: bool = True,
        temperature: float = 0.1,
    ):
        """
        Initializes the CNN bot.
        ## Parameters
        ``model_path``: str | None
            Path to the pre-trained model. If None and ``model`` is None, random weights are loaded.

        ``model``: NN_abstract | None
            An instance of a model derived from NN_abstract. If provided, it will be used instead of loading from a file.
            Cannot be used together with ``model_path``.

        ``model_class``: type[NN_abstract] | None
            The model class to instantiate when loading from ``model_path``.
            Defaults to QuartoCNN if not provided.
            Only used when ``model_path`` is provided.

        ``deterministic``: bool
            If True, the model will select the most probable action. Default is True.

        ``temperature``: float
            Controls the randomness of the selection. Higher values lead to more exploration.
            Only applicable if ``deterministic`` is False. Default is 0.1.

        ## Attributes
        ``DETERMINISTIC``: bool
            If True, the model will select the most probable action.
        ``TEMPERATURE``: float
            Controls the randomness of the selection. Higher values lead to more exploration. Only applicable if ``DETERMINISTIC`` is False.
        """
        super().__init__()  # aunque no hace nada
        logger.debug(f"CNN_bot initialized")

        # Validate input parameters
        assert not (
            model_path is not None and model is not None
        ), "Cannot provide both ``model_path`` and ``model`` instance. Choose one."

        if model_path:
            # Use provided model_class
            logger.debug(
                f"Loading model from {model_path} using {model_class.__name__}"
            )
            self.model = model_class.from_file(model_path)
            self.model_path = os.path.basename(model_path)

        elif model:
            assert isinstance(
                model, NN_abstract
            ), "Provided model must be a derived class of ``NN_abstract``."

            self.model = model
            self.model_name = model.name
            logger.debug(f"Using provided model instance {self.model_name}")

        else:
            logger.debug("Loading model with random weights")
            self.model = model_class()

        # Move model to appropriate device
        self.model.to(self.model.device)
        logger.debug(f"Model loaded successfully on device: {self.model.device}")

        self._recalculate = True  # Recalculate the model on each turn
        self.selected_piece: Piece
        self.board_position: tuple[int, int]
        # If True, the model will select the most probable action
        self.DETERMINISTIC: bool = deterministic

        # Controls the randomness of the selection. Higher values lead to more exploration.
        # Only applicable if ``DETERMINISTIC`` is False.
        self.TEMPERATURE: float = temperature

    # ####################################################################
    def calculate(
        self,
        game: QuartoGame,
        ith_try: int = 0,
    ):
        """Calculates the move for the bot based on the current board state and selected piece.
        ## Parameters
        ``game``: QuartoGame
            The current game instance.
        ``ith_try``: int
            The index of the current attempt to select or place a piece.
        ## Returns

        """
        if self._recalculate:
            board_matrix = game.game_board.encode()
            if isinstance(game.selected_piece, Piece):
                piece_onehot = game.selected_piece.vectorize_onehot()
                piece_onehot = piece_onehot.reshape(1, -1)  # Reshape to (1, 16)
            else:
                piece_onehot = np.zeros((1, 16), dtype=float)

            # Create tensors and move to model's device
            board_tensor = torch.from_numpy(board_matrix).float().to(self.model.device)
            piece_tensor = torch.from_numpy(piece_onehot).float().to(self.model.device)

            self.board_pos_onehot_cached, self.select_piece_onehot_cached = (
                self.model.predict(
                    board_tensor,
                    piece_tensor,
                    TEMPERATURE=self.TEMPERATURE,
                    DETERMINISTIC=self.DETERMINISTIC,
                )
            )
            batch_size = self.board_pos_onehot_cached.shape[0]
            assert batch_size == 1, f"Expected batch size of 1, got {batch_size}."

            self._recalculate = False  # Do not recalculate until the next turn

        # load from cached values
        # 0 for batch size is 1
        _idx_piece: int = self.select_piece_onehot_cached[0, ith_try].item()  # type: ignore
        selected_piece = Piece.from_index(_idx_piece)

        _idx_board_pos: int = self.board_pos_onehot_cached[0, ith_try].item()  # type: ignore
        board_position = game.game_board.get_position_index(_idx_board_pos)

        return board_position, selected_piece

    def select(
        self,
        game: QuartoGame,
        ith_option: int = 0,
        *args,
        **kwargs,
    ) -> Piece:
        """Selects a piece for the other player."""

        _, selected_piece = self.calculate(game, ith_option)

        return selected_piece

    def place_piece(
        self,
        game: QuartoGame,
        piece: Piece,
        ith_option: int = 0,
        *args,
        **kwargs,
    ) -> tuple[int, int]:
        """Places the selected piece on the game board at a random valid position."""
        if ith_option == 0:
            self._recalculate = True
        board_position, _ = self.calculate(game, ith_option)
        return board_position

    def evaluate(self, exp_batch: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluates a batch of experiences and returns the Q-values for the taken actions.
        This is useful for tracking how action-values evolve during training.

        ## Parameters
        ``exp_batch``: TensorDict
            Batch of experiences containing:
            - state_board: Board states (batch_size, 16, 4, 4)
            - state_piece: Piece states (batch_size, 16)
            - action_place: Placement actions taken (batch_size,). -1 for first move.
            - action_sel: Selection actions taken (batch_size,). -1 for terminal states.

        ## Returns
        ``q_place``: torch.Tensor
            Q-values for the placement actions taken (batch_size,)
        ``q_select``: torch.Tensor
            Q-values for the selection actions taken (batch_size,)

        ## Note
        Terminal states (winning moves) have action_sel=-1, which will cause
        incorrect indexing. The caller should handle this appropriately.
        """
        state_board: torch.Tensor = exp_batch["state_board"]
        state_piece: torch.Tensor = exp_batch["state_piece"]
        action_place = exp_batch["action_place"].to(torch.int64)
        action_sel = exp_batch["action_sel"].to(torch.int64)

        # Get Q-values for all actions
        self.model.eval()
        with torch.no_grad():
            # Move tensors to model device
            state_board = state_board.to(self.model.device)
            state_piece = state_piece.to(self.model.device)
            qav_board, qav_piece = self.model.forward(state_board, state_piece)

        # Extract Q-values for the actual actions taken
        batch_indices = torch.arange(state_board.shape[0])
        q_place = qav_board[batch_indices, action_place]
        q_select = qav_piece[batch_indices, action_sel]

        return q_place, q_select
