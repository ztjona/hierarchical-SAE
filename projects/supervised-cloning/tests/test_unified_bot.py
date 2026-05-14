"""Tests for the bot/CNN_unified_bot inference wrapper.

Validates that:
- _build_unified_aux returns a 32-d vector with the expected block structure.
- _last_placed_piece is correctly updated after place_piece(...) so the
  next select(...) can use it as offered.
- A full play loop against RandomBot completes without raising.
"""

from __future__ import annotations
import numpy as np
import torch

from quartopy import QuartoGame, Piece, play_games
from bot.CNN_unified_bot import Quarto_bot as UnifiedBot
from bot.random_bot import Quarto_bot as RandomBot
from models.CNN_autoreg import QuartoCNNAutoregUnified


def test_build_unified_aux_block_structure():
    torch.manual_seed(0)
    model = QuartoCNNAutoregUnified()
    bot = UnifiedBot(model=model)
    game = QuartoGame(player1=RandomBot(), player2=RandomBot(), mode_2x2=True)

    aux = bot._build_unified_aux(game, offered_index=4)
    assert aux.shape == (1, 32)
    aux_np = aux.detach().cpu().numpy().reshape(-1)
    # offered = one-hot at index 4
    assert aux_np[4] == 1.0 and aux_np[:16].sum() == 1.0
    # available = all 16 pieces in storage at game start
    assert aux_np[16:32].sum() == 16.0


def test_last_placed_piece_tracked_through_place():
    torch.manual_seed(0)
    model = QuartoCNNAutoregUnified()
    bot = UnifiedBot(model=model)
    assert bot._last_placed_piece == -1

    game = QuartoGame(player1=RandomBot(), player2=RandomBot(), mode_2x2=True)
    # Advance to a PLACE phase manually: pick piece 9.
    p9 = Piece.from_index(9)
    coord = game.storage_board.find_piece(p9)
    game.storage_board.remove_piece(*coord)
    game.selected_piece = p9
    game.pick = False

    bot.place_piece(game, p9)
    assert bot._last_placed_piece == 9, (
        f"Expected _last_placed_piece=9 after placing piece 9, "
        f"got {bot._last_placed_piece}"
    )


def test_select_uses_last_placed_piece_in_offered_block():
    """After placing piece 7, the next select-phase aux must have offered=one-hot(7)."""
    torch.manual_seed(0)
    model = QuartoCNNAutoregUnified()
    bot = UnifiedBot(model=model)

    game = QuartoGame(player1=RandomBot(), player2=RandomBot(), mode_2x2=True)
    p7 = Piece.from_index(7)
    coord = game.storage_board.find_piece(p7)
    game.storage_board.remove_piece(*coord)
    game.selected_piece = p7
    game.pick = False
    bot.place_piece(game, p7)

    # Now manually call the aux builder with the bot's tracked piece.
    aux = bot._build_unified_aux(
        game, offered_index=bot._last_placed_piece
    )
    aux_np = aux.detach().cpu().numpy().reshape(-1)
    assert aux_np[7] == 1.0
    assert aux_np[:16].sum() == 1.0


def test_full_game_against_random_completes():
    """End-to-end sanity check — the bot must complete a real game."""
    torch.manual_seed(0)
    model = QuartoCNNAutoregUnified()
    bot = UnifiedBot(model=model, deterministic=True)
    rival = RandomBot()
    # 2 matches should run quickly and exercise both P1 and P2 sides.
    _, stats = play_games(
        matches=2,
        player1=bot,
        player2=rival,
        verbose=False,
        save_match=False,
        mode_2x2=True,
        PROGRESS_MESSAGE="",
    )
    total = stats["Player 1"] + stats["Player 2"] + stats["Tie"]
    assert total == 2
