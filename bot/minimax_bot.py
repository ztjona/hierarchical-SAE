"""
Python 3
10 / 01 / 2026
@author: jairo

"I know that I know nothing."
-Sócrates
"""

import copy
import random
from quartopy import BotAI, Piece, QuartoGame


class MinimaxBot(BotAI):
    """
    A bot that uses the Minimax algorithm with Alpha-Beta pruning to play Quarto.
    """

    def __init__(self, name: str = "MinimaxBot", depth: int = 2, **kwargs):
        """
        Initializes the MinimaxBot.
        :param name: The name of the bot.
        :param depth: The search depth for the Minimax algorithm. A depth of 2 is a good balance of performance and foresight.
        """
        self._name = name
        self.depth = depth
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return f"{self._name}|depth={self.depth}"

    def select(self, game: QuartoGame, ith_option: int = 0, *args, **kwargs) -> Piece:
        """
        Selects a piece for the opponent to place. It simulates giving each available piece
        and chooses the one that leads to the best outcome for the bot after the opponent's move.
        This is a 'minimizing' action from the opponent's perspective.
        """
        # We want to find the piece that MINIMIZES the opponent's score.
        eval, best_piece = self._minimax(
            game, self.depth, float("-inf"), float("inf"), False
        )  # False = Minimizing phase (select piece)

        if best_piece:
            return best_piece

        # Fallback in case minimax fails, which shouldn't happen in a valid game state.
        valid_pieces = game.storage_board.get_valid_pieces()
        return (
            random.choice(valid_pieces)
            if valid_pieces
            else game.storage_board.get_valid_moves()[0]
        )

    def place_piece(
        self, game: QuartoGame, piece: Piece, ith_option: int = 0, *args, **kwargs
    ) -> tuple[int, int]:
        """
        Places the given piece on the board. It simulates placing the piece in all valid
        positions and chooses the one that leads to the best outcome for the bot.
        This is a 'maximizing' action.
        """
        # We want to find the position that MAXIMIZES our score.
        eval, best_move = self._minimax(
            game, self.depth, float("-inf"), float("inf"), True
        )  # True = Maximizing phase (place piece)

        if best_move:
            return best_move

        # Fallback
        return random.choice(game.game_board.get_valid_moves())

    def _minimax(
        self,
        game_state: QuartoGame,
        depth: int,
        alpha: float,
        beta: float,
        is_maximizing_player: bool,
    ) -> tuple[float, Piece | tuple[int, int] | None]:
        """
        The core Minimax algorithm with Alpha-Beta pruning.
        - is_maximizing_player: True when placing a piece, False when selecting a piece.
        """
        # --- Base Cases: Game is Over or Depth Limit Reached ---
        # NOTE: ``check_win`` returns ``tuple[bool, coords | None]``. The bare
        # call evaluates as truthy for *any* tuple (including ``(False, None)``)
        # — must subscript ``[0]`` to read the actual win flag.
        if (
            game_state.game_board.last_move is not None
            and game_state.game_board.check_win(game_state.mode_2x2)[0]
        ):
            # If a win is detected, the player who made the LAST move won.
            # 'is_maximizing_player' is for the CURRENT turn. The winner is the player from the PREVIOUS turn.
            if is_maximizing_player:
                # The minimizer (opponent) must have just moved to create this state, which is a win for them.
                return -100 - depth, None
            else:
                # The maximizer (our bot) must have just moved and won.
                return 100 + depth, None

        if game_state.game_board.is_full():
            return 0, None  # Draw

        if depth == 0:
            return 0, None  # Depth limit reached, neutral score

        # --- Recursive Step ---
        if is_maximizing_player:  # Placing a piece to maximize our score
            max_eval = float("-inf")
            best_move = None

            valid_moves = game_state.game_board.get_valid_moves()
            for r_g, c_g in valid_moves:
                new_game_state = copy.deepcopy(game_state)
                piece_to_place = new_game_state.selected_piece
                new_game_state.game_board.put_piece(piece_to_place, r_g, c_g)

                # If this move is a winning move, it's the best possible move.
                # ``check_win`` returns a tuple — subscript [0] for the flag.
                if new_game_state.game_board.check_win(new_game_state.mode_2x2)[0]:
                    return 100 + depth, (r_g, c_g)

                new_game_state.pick = True  # Switch to selection phase

                # Now it's the opponent's turn to select a piece (minimizing phase)
                eval, _ = self._minimax(new_game_state, depth - 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_move = (r_g, c_g)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Prune
            return max_eval, best_move

        else:  # Selecting a piece for the opponent to minimize their score
            min_eval = float("inf")
            best_piece = None

            possible_pieces = game_state.storage_board.get_valid_pieces()
            for piece_to_give in possible_pieces:
                new_game_state = copy.deepcopy(game_state)
                # Replicate select_and_remove_piece (not present in all quartopy versions)
                coord = new_game_state.storage_board.find_piece(piece_to_give)
                if coord is None:
                    continue
                new_game_state.storage_board.remove_piece(*coord)
                new_game_state.selected_piece = piece_to_give
                new_game_state.pick = False  # Switch to placing phase
                new_game_state.turn = (
                    not new_game_state.turn
                )  # Opponent's turn to place

                # Now it's the opponent's turn to place a piece (maximizing phase for them)
                eval, _ = self._minimax(new_game_state, depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_piece = piece_to_give
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Prune
            return min_eval, best_piece
