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

from models.CNNfrancis import QuartoCNN
from utils.logger import logger
import numpy as np
import torch


def _validate_and_import_quartopy():
    """
    Validates and imports quartopy dependencies with clear error messages.

    Returns:
        tuple: (BotAI, Piece, QuartoGame) classes from quartopy

    Raises:
        ImportError: If quartopy cannot be imported with helpful instructions
    """
    try:
        from quartopy import BotAI, Piece, QuartoGame

        logger.debug("✅ Quartopy imported successfully")
        return BotAI, Piece, QuartoGame

    except ImportError as initial_error:
        logger.warning(
            "⚠️ Initial quartopy import failed, attempting dependency setup..."
        )

        # Attempt fallback with setup_dependencies
        try:
            import sys
            from pathlib import Path

            # Add parent directory to path for setup_dependencies
            parent_dir = Path(__file__).parent.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))

            # Import and run dependency setup
            import setup_dependencies

            setup_dependencies.setup_quartopy(silent=False)

            # Retry import after setup
            from quartopy import BotAI, Piece, QuartoGame

            logger.info("✅ Quartopy imported successfully after dependency setup")
            return BotAI, Piece, QuartoGame

        except ImportError as final_error:
            # Create comprehensive error message with troubleshooting steps
            error_msg = (
                "❌ DEPENDENCY ERROR: Cannot import quartopy\n\n"
                "🔧 TROUBLESHOOTING STEPS:\n"
                "1. Ensure the 'quartopy' project is available in your environment\n"
                "2. Check if quartopy is in one of these locations:\n"
                "   - ../quartopy (relative to this project)\n"
                "   - ~/Documents/GitHub/Quartopy\n"
                "   - C:/Users/bravo/Documents/quartopy\n"
                "3. If quartopy is elsewhere, create a .env file with:\n"
                "   QUARTOPY_PATH=/path/to/your/quartopy/project\n"
                "4. Or install quartopy as a package: pip install quartopy\n\n"
                f"📋 Original error: {initial_error}\n"
                f"📋 Setup attempt error: {final_error}\n\n"
                "💡 For more help, check the project documentation or setup_dependencies.py"
            )

            logger.error(error_msg)
            raise ImportError(error_msg) from final_error

        except Exception as unexpected_error:
            # Handle any unexpected errors during setup
            error_msg = (
                f"❌ UNEXPECTED ERROR during quartopy setup: {unexpected_error}\n\n"
                "🔧 SUGGESTED ACTIONS:\n"
                "1. Check that setup_dependencies.py exists and is valid\n"
                "2. Verify file permissions in the project directory\n"
                "3. Try running the project with administrator privileges\n"
                "4. Check the utils/logger.py for any issues\n\n"
                "💡 Consider manually adding quartopy to your Python path"
            )

            logger.error(error_msg)
            raise ImportError(error_msg) from unexpected_error


# Import quartopy with validation
BotAI, Piece, QuartoGame = _validate_and_import_quartopy()

logger.debug("Loading CNN_bot...")


class Quarto_bot(BotAI):
    @property
    def name(self) -> str:
        return "CNN_bot"

    def __init__(
        self, *, model_path: str | None = None, model: QuartoCNN | None = None
    ):
        """
        Initializes the CNN bot.
        ## Parameters
        ``model_path``: str | None
            Path to the pre-trained model. If None, random weights are loaded.

        ``model``: QuartoCNN | None
            An instance of QuartoCNN. If provided, it will be used instead of loading from a file.

        ## Attributes
        ``DETERMINISTIC``: bool
            If True, the model will select the most probable action.
        ``TEMPERATURE``: float
            Controls the randomness of the selection. Higher values lead to more exploration. Only applicable if ``DETERMINISTIC`` is False.
        """
        try:
            super().__init__()  # aunque no hace nada
            logger.debug("CNN_bot initialized")

            # Validate input parameters with proper exception handling
            if model_path is not None and model is not None:
                error_msg = "❌ PARAMETER ERROR: Either 'model_path' or 'model' must be provided, but not both."
                logger.error(error_msg)
                raise ValueError(error_msg)

            if model_path:
                try:
                    logger.debug(f"Loading model from {model_path}")
                    self.model = QuartoCNN.from_file(model_path)
                    # logger.info(f"✅ Model loaded successfully from {model_path}")
                except FileNotFoundError as e:
                    error_msg = (
                        f"❌ MODEL FILE NOT FOUND: {model_path}\n\n"
                        "🔧 TROUBLESHOOTING STEPS:\n"
                        "1. Check if the file path is correct\n"
                        "2. Verify the file exists in the specified location\n"
                        "3. Ensure you have read permissions for the file\n"
                        f"📋 Original error: {e}"
                    )
                    logger.error(error_msg)
                    raise FileNotFoundError(error_msg) from e
                except Exception as e:
                    error_msg = (
                        f"❌ MODEL LOADING ERROR: Failed to load model from {model_path}\n\n"
                        "🔧 POSSIBLE CAUSES:\n"
                        "1. Corrupted model file\n"
                        "2. Model was saved with different PyTorch version\n"
                        "3. Model architecture mismatch\n"
                        "4. Insufficient memory to load model\n"
                        f"📋 Original error: {e}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

            elif model:
                if not isinstance(model, QuartoCNN):
                    error_msg = (
                        f"❌ MODEL TYPE ERROR: Provided model must be an instance of QuartoCNN.\n"
                        f"Got: {type(model).__name__}\n"
                        f"Expected: QuartoCNN"
                    )
                    logger.error(error_msg)
                    raise TypeError(error_msg)

                self.model = model
                logger.debug("✅ Using provided model instance")

            else:
                try:
                    logger.debug("Loading model with random weights")
                    self.model = QuartoCNN()
                    logger.info("✅ Model initialized with random weights")
                except Exception as e:
                    error_msg = (
                        f"❌ MODEL INITIALIZATION ERROR: Failed to initialize QuartoCNN\n\n"
                        "🔧 POSSIBLE CAUSES:\n"
                        "1. Missing dependencies (torch, etc.)\n"
                        "2. Insufficient memory\n"
                        "3. CUDA/GPU configuration issues\n"
                        f"📋 Original error: {e}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

            # Initialize bot attributes
            self.recalculate = True  # Recalculate the model on each turn
            self.selected_piece: Piece
            self.board_position: tuple[int, int]
            # If True, the model will select the most probable action
            self.DETERMINISTIC: bool = True

            # Controls the randomness of the selection. Higher values lead to more exploration.
            # Only applicable if ``DETERMINISTIC`` is False.
            self.TEMPERATURE: float = 0.1

            logger.debug("CNN_bot initialization completed successfully")

        except Exception as e:
            logger.error(f"❌ CRITICAL ERROR: Failed to initialize Quarto_bot: {e}")
            raise

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
        if self.recalculate:
            board_matrix = game.game_board.encode()
            if isinstance(game.selected_piece, Piece):
                piece_onehot = game.selected_piece.vectorize_onehot()
                piece_onehot = piece_onehot.reshape(1, -1)  # Reshape to (1, 16)
            else:
                piece_onehot = np.zeros((1, 16), dtype=float)

            # Crear tensores
            board_tensor = torch.from_numpy(board_matrix).float()
            piece_tensor = torch.from_numpy(piece_onehot).float()

            # Mover a GPU si el modelo está en GPU
            if hasattr(self, "_device"):
                board_tensor = board_tensor.to(self._device)
                piece_tensor = piece_tensor.to(self._device)
            elif hasattr(self.model, "_device"):
                board_tensor = board_tensor.to(self.model._device)
                piece_tensor = piece_tensor.to(self.model._device)

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

            self.recalculate = False  # Do not recalculate until the next turn

        # load from cached values
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
            self.recalculate = True
        board_position, _ = self.calculate(game, ith_option)
        return board_position
