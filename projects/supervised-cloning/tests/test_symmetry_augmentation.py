"""Tests for the D4 symmetry augmentation in train.py.

Validates that:
- Permutation tables _POS_PERMS_FWD and _POS_PERMS_INV are mutual inverses.
- A single piece placed on the board rotates consistently with its PLACE label.
- aux (32-d) is rotation-invariant and passes through unchanged.
- PLACE legal_mask and soft_target permute via the same gather as labels.
- SELECT samples are NOT transformed on any action-indexed field.
"""

from __future__ import annotations
import numpy as np
import pytest

import bc_train  # loaded by conftest.py from projects/supervised-cloning/train.py

_POS_PERMS_FWD = bc_train._POS_PERMS_FWD
_POS_PERMS_INV = bc_train._POS_PERMS_INV
augment_symmetries = bc_train.augment_symmetries
ACTION_PLACE = bc_train.ACTION_PLACE
ACTION_SELECT = bc_train.ACTION_SELECT


# ----- forward / inverse permutation sanity ---------------------------------


def test_perm_fwd_inv_are_mutual_inverses():
    for t in range(8):
        fwd = _POS_PERMS_FWD[t]
        inv = _POS_PERMS_INV[t]
        assert (fwd[inv] == np.arange(16)).all(), f"t={t}: fwd∘inv ≠ identity"
        assert (inv[fwd] == np.arange(16)).all(), f"t={t}: inv∘fwd ≠ identity"


def test_identity_transform_is_arange():
    # t=0 is the identity transform.
    assert (_POS_PERMS_FWD[0] == np.arange(16)).all()
    assert (_POS_PERMS_INV[0] == np.arange(16)).all()


# ----- board / label rotation consistency -----------------------------------


def _make_single_piece_board(pos: int) -> np.ndarray:
    """(16, 4, 4) board with a single non-zero channel-0 entry at pos."""
    board = np.zeros((1, 16, 4, 4), dtype=np.float32)
    r, c = divmod(pos, 4)
    board[0, 0, r, c] = 1.0
    return board


def test_place_label_follows_board_rotation():
    """If a piece sits at PLACE label `pos`, after each symmetry the piece's
    new (r, c) on the rotated board must equal divmod(new_label, 4).
    """
    for old_pos in range(16):
        boards = _make_single_piece_board(old_pos)
        aux = np.zeros((1, 32), dtype=np.float32)
        labels = np.array([old_pos], dtype=np.int16)
        actions = np.array([ACTION_PLACE], dtype=np.uint8)
        legal_masks = np.zeros((1, 16), dtype=bool)
        legal_masks[0, old_pos] = True
        soft = np.zeros((1, 16), dtype=np.float32)
        soft[0, old_pos] = 1.0

        b, a, lbl, act, m, s = augment_symmetries(
            boards, aux, labels, actions, legal_masks, soft
        )

        assert b.shape == (8, 16, 4, 4)
        for t in range(8):
            new_label = int(lbl[t])
            new_r, new_c = divmod(new_label, 4)
            # Piece must end up on channel 0 at the rotated cell.
            assert b[t, 0, new_r, new_c] == 1.0, (
                f"t={t}, old_pos={old_pos}: label says ({new_r},{new_c}) "
                f"but board[{t},0,{new_r},{new_c}] = {b[t,0,new_r,new_c]}"
            )
            # And nowhere else on channel 0.
            total = b[t, 0].sum()
            assert total == 1.0, f"t={t}: channel-0 mass = {total} (≠ 1)"


# ----- aux is rotation-invariant --------------------------------------------


def test_aux_unchanged_under_all_transforms():
    rng = np.random.default_rng(0)
    boards = rng.normal(size=(3, 16, 4, 4)).astype(np.float32)
    aux = rng.normal(size=(3, 32)).astype(np.float32)
    labels = np.array([0, 5, 9], dtype=np.int16)
    actions = np.array([ACTION_PLACE, ACTION_PLACE, ACTION_SELECT], dtype=np.uint8)
    legal_masks = rng.integers(0, 2, size=(3, 16)).astype(bool)
    soft = rng.uniform(size=(3, 16)).astype(np.float32)
    soft /= soft.sum(axis=1, keepdims=True)

    _, a, _, _, _, _ = augment_symmetries(
        boards, aux, labels, actions, legal_masks, soft
    )
    # 8 transforms × 3 samples = 24 rows. Each block of 3 rows should equal aux.
    assert a.shape == (24, 32)
    for t in range(8):
        np.testing.assert_array_equal(a[t * 3 : (t + 1) * 3], aux)


# ----- PLACE mask / soft_target use the same gather as labels ---------------


def test_place_mask_and_soft_target_permute_consistently():
    """For a PLACE sample with all-True mask and soft mass at exactly the label
    position, the post-transform soft target must be a one-hot at the new label,
    and the mask must still cover every cell.
    """
    for old_pos in [0, 3, 7, 12]:
        boards = _make_single_piece_board(old_pos)
        aux = np.zeros((1, 32), dtype=np.float32)
        labels = np.array([old_pos], dtype=np.int16)
        actions = np.array([ACTION_PLACE], dtype=np.uint8)
        legal_masks = np.ones((1, 16), dtype=bool)
        soft = np.zeros((1, 16), dtype=np.float32)
        soft[0, old_pos] = 1.0

        b, _, lbl, _, m, s = augment_symmetries(
            boards, aux, labels, actions, legal_masks, soft
        )
        for t in range(8):
            new_label = int(lbl[t])
            assert s[t, new_label] == 1.0, (
                f"t={t}, old_pos={old_pos}: soft target lost its mass"
            )
            assert s[t].sum() == 1.0
            assert m[t].all(), "Full-True mask should remain full after rotation"


def test_select_samples_are_invariant():
    """SELECT samples must have their label / mask / soft_target untouched
    by board rotations (piece indices are rotation-invariant).
    """
    rng = np.random.default_rng(1)
    boards = rng.normal(size=(2, 16, 4, 4)).astype(np.float32)
    aux = rng.normal(size=(2, 32)).astype(np.float32)
    labels = np.array([3, 11], dtype=np.int16)
    actions = np.array([ACTION_SELECT, ACTION_SELECT], dtype=np.uint8)
    legal_masks = rng.integers(0, 2, size=(2, 16)).astype(bool)
    soft = rng.uniform(size=(2, 16)).astype(np.float32)
    soft /= soft.sum(axis=1, keepdims=True)

    _, _, lbl, act, m, s = augment_symmetries(
        boards, aux, labels, actions, legal_masks, soft
    )
    for t in range(8):
        np.testing.assert_array_equal(lbl[t * 2 : (t + 1) * 2], labels)
        np.testing.assert_array_equal(act[t * 2 : (t + 1) * 2], actions)
        np.testing.assert_array_equal(m[t * 2 : (t + 1) * 2], legal_masks)
        np.testing.assert_array_almost_equal(s[t * 2 : (t + 1) * 2], soft)


def test_augmentation_preserves_sample_count_8x():
    rng = np.random.default_rng(2)
    n = 5
    boards = rng.normal(size=(n, 16, 4, 4)).astype(np.float32)
    aux = rng.normal(size=(n, 32)).astype(np.float32)
    labels = rng.integers(0, 16, size=n).astype(np.int16)
    actions = rng.integers(0, 2, size=n).astype(np.uint8)
    legal_masks = np.ones((n, 16), dtype=bool)
    soft = np.zeros((n, 16), dtype=np.float32)
    for i, lbl in enumerate(labels):
        soft[i, int(lbl)] = 1.0
    out = augment_symmetries(boards, aux, labels, actions, legal_masks, soft)
    for arr in out:
        assert arr.shape[0] == 8 * n
