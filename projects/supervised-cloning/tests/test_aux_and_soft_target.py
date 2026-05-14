"""Tests for the unified-aux composition and soft-target build helpers in
collect_data.py.

Validates that:
- aux is a 32-d vector with offered (one-hot) in [0:16] and available mask
  in [16:32].
- offered = zero block when offered_index = -1 (semantic "no piece offered").
- soft_target sums to 1 on legal positions and 0 elsewhere.
- soft_target is uniform over best_action_set.
- Fallback: empty scores → uniform over legal mask.
"""

from __future__ import annotations
import numpy as np

import bc_collect

_build_aux32 = bc_collect._build_aux32
_soft_target_from_scores = bc_collect._soft_target_from_scores
_piece_one_hot = bc_collect._piece_one_hot


def test_aux_composition_place_phase():
    """PLACE: offered = one-hot of selected piece, available = storage mask."""
    available = np.zeros(16, dtype=bool)
    available[[1, 3, 5, 7]] = True
    aux = _build_aux32(offered_index=5, available_mask_bool=available)
    assert aux.shape == (32,)
    # offered block
    assert aux[5] == 1.0
    assert aux[:16].sum() == 1.0
    # available block (after 16)
    np.testing.assert_array_almost_equal(aux[16:32], available.astype(np.float32))


def test_aux_composition_select_phase_first_select_is_zero_block():
    """First SELECT of a game: offered_index = -1 → zero offered block."""
    available = np.ones(16, dtype=bool)
    aux = _build_aux32(offered_index=-1, available_mask_bool=available)
    assert aux[:16].sum() == 0.0
    np.testing.assert_array_almost_equal(aux[16:32], np.ones(16, dtype=np.float32))


def test_aux_offered_index_out_of_range_is_zero_block():
    """Sentinel offered indices (-1 / 16 / 99) should produce a zero offered block."""
    available = np.zeros(16, dtype=bool)
    for bad in (-1, 16, 99):
        aux = _build_aux32(offered_index=bad, available_mask_bool=available)
        assert aux[:16].sum() == 0.0, f"offered_index={bad}: block not zero"


def test_soft_target_uniform_over_best_set_place():
    """PLACE: argmax. Scores with three tied maxes → 1/3 mass each."""
    scores = {0: 1.0, 1: 5.0, 2: 5.0, 3: 5.0, 4: 2.0}
    legal_mask = np.zeros(16, dtype=bool)
    legal_mask[[0, 1, 2, 3, 4]] = True
    soft = _soft_target_from_scores(scores, action_kind=0, legal_mask=legal_mask)
    assert soft.shape == (16,)
    np.testing.assert_array_almost_equal(soft[[1, 2, 3]], [1 / 3, 1 / 3, 1 / 3])
    assert soft[0] == 0.0 and soft[4] == 0.0
    assert abs(soft.sum() - 1.0) < 1e-6


def test_soft_target_uniform_over_best_set_select():
    """SELECT: argmin. Scores with two tied mins → 0.5 mass each."""
    scores = {2: 10.0, 5: -3.0, 7: -3.0, 9: 0.0}
    legal_mask = np.zeros(16, dtype=bool)
    legal_mask[[2, 5, 7, 9]] = True
    soft = _soft_target_from_scores(scores, action_kind=1, legal_mask=legal_mask)
    np.testing.assert_array_almost_equal(soft[[5, 7]], [0.5, 0.5])
    assert soft[2] == 0.0 and soft[9] == 0.0
    assert abs(soft.sum() - 1.0) < 1e-6


def test_soft_target_zero_on_illegal_positions():
    """No prob mass should ever land on illegal indices."""
    scores = {0: 1.0, 5: 1.0}
    legal_mask = np.zeros(16, dtype=bool)
    legal_mask[[0, 5]] = True
    soft = _soft_target_from_scores(scores, action_kind=0, legal_mask=legal_mask)
    illegal = np.where(~legal_mask)[0]
    assert (soft[illegal] == 0).all()


def test_soft_target_fallback_uniform_over_legal_when_no_scores():
    """If score_all_moves returned nothing (shouldn't happen, but defensive),
    soft target falls back to uniform over the legal mask.
    """
    legal_mask = np.zeros(16, dtype=bool)
    legal_mask[[1, 3, 5, 7]] = True
    soft = _soft_target_from_scores({}, action_kind=0, legal_mask=legal_mask)
    np.testing.assert_array_almost_equal(
        soft[legal_mask], np.full(4, 0.25, dtype=np.float32)
    )
    assert (soft[~legal_mask] == 0).all()


def test_piece_one_hot_basics():
    oh = _piece_one_hot(7)
    assert oh.shape == (16,) and oh.dtype == np.float32
    assert oh[7] == 1.0 and oh.sum() == 1.0
    # Out-of-range stays all-zero.
    assert _piece_one_hot(-1).sum() == 0.0
    assert _piece_one_hot(16).sum() == 0.0
