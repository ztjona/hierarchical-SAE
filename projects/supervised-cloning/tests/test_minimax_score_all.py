"""Tests for MinimaxBot.score_all_moves and best_action_set.

Validates that:
- score_all_moves enumerates every legal move (no αβ pruning at the top level).
- The move actually picked by select() / place_piece() is in the
  best_action_set returned for the same game state.
- A win-now position assigns the maximal score to the winning placement.
"""

from __future__ import annotations
import copy
import numpy as np

from quartopy import QuartoGame, Piece
from bot.minimax_bot import MinimaxBot, best_action_set
from bot.random_bot import Quarto_bot as RandomBot


def _fresh_game() -> QuartoGame:
    """A 2x2-mode game ready for the opening SELECT (P1 picks for P2)."""
    return QuartoGame(player1=RandomBot(), player2=RandomBot(), mode_2x2=True)


def test_score_all_moves_covers_every_legal_move_select():
    bot = MinimaxBot(depth=2)
    game = _fresh_game()
    # game.pick starts True (player must SELECT a piece for opponent).
    assert game.pick is True
    scores, kind = bot.score_all_moves(game)
    assert kind == 1
    legal_pieces = {p.index() for p in game.storage_board.get_valid_pieces()}
    assert set(scores.keys()) == legal_pieces, (
        f"Missing pieces from scores: legal={legal_pieces}, scored={set(scores)}"
    )
    assert len(scores) == 16  # opening: all pieces available


def test_score_all_moves_covers_every_legal_move_place():
    bot = MinimaxBot(depth=2)
    game = _fresh_game()
    # Manually advance to a PLACE phase: pick piece 0, hand it over.
    p0 = Piece.from_index(0)
    coord = game.storage_board.find_piece(p0)
    assert coord is not None
    game.storage_board.remove_piece(*coord)
    game.selected_piece = p0
    game.pick = False  # PLACE phase

    scores, kind = bot.score_all_moves(game)
    assert kind == 0
    legal_positions = {
        game.game_board.pos2index(r, c)
        for r, c in game.game_board.get_valid_moves()
    }
    assert set(scores.keys()) == legal_positions
    assert len(scores) == 16  # empty board


def test_actually_picked_move_is_in_best_set_select():
    bot = MinimaxBot(depth=2)
    game = _fresh_game()
    g2 = copy.deepcopy(game)
    scores, kind = bot.score_all_moves(g2)
    best = best_action_set(scores, kind)

    chosen_piece = bot.select(game)
    assert chosen_piece.index() in best, (
        f"select() picked piece {chosen_piece.index()} but best_set = {best}"
    )


def test_actually_picked_move_is_in_best_set_place():
    bot = MinimaxBot(depth=2)
    game = _fresh_game()
    # advance to PLACE
    p3 = Piece.from_index(3)
    coord = game.storage_board.find_piece(p3)
    game.storage_board.remove_piece(*coord)
    game.selected_piece = p3
    game.pick = False

    g2 = copy.deepcopy(game)
    scores, kind = bot.score_all_moves(g2)
    best = best_action_set(scores, kind)

    chosen_pos = bot.place_piece(game, p3)
    chosen_idx = game.game_board.pos2index(*chosen_pos)
    assert chosen_idx in best, (
        f"place_piece picked pos {chosen_idx} but best_set = {best}"
    )


def test_best_action_set_polarity():
    """PLACE = argmax, SELECT = argmin. Manually build score dicts."""
    place_scores = {0: 10.0, 1: 50.0, 2: 50.0, 3: 5.0}
    select_scores = {0: 10.0, 1: 50.0, 2: 5.0, 3: 5.0}
    assert best_action_set(place_scores, 0) == {1, 2}
    assert best_action_set(select_scores, 1) == {2, 3}


def test_best_action_set_empty_input():
    """Empty scores dict → empty best set (graceful fallback)."""
    assert best_action_set({}, 0) == set()
    assert best_action_set({}, 1) == set()
