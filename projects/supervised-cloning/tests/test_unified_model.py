"""Tests for QuartoCNNAutoregUnified: shapes, aux-dim assertion, tanh range, head routing."""

from __future__ import annotations
import numpy as np
import pytest
import torch

from models.CNN_autoreg import (
    QuartoCNNAutoregUnified,
    QuartoCNNAutoregUnifiedUnbound,
    PHASE_PLACE,
    PHASE_SELECT,
)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return QuartoCNNAutoregUnified()


def _rand_inputs(batch: int = 4):
    board = torch.randn(batch, 16, 4, 4)
    aux = torch.randn(batch, 32)
    return board, aux


def test_forward_returns_two_heads_with_correct_shape(model):
    board, aux = _rand_inputs(4)
    q_place, q_select = model(board, aux)
    assert q_place.shape == (4, 16)
    assert q_select.shape == (4, 16)


def test_forward_output_is_in_tanh_range(model):
    board, aux = _rand_inputs(8)
    q_place, q_select = model(board, aux)
    assert q_place.abs().max().item() <= 1.0 + 1e-6
    assert q_select.abs().max().item() <= 1.0 + 1e-6


def test_raw_logits_helper_is_unbounded():
    """The trainer's project-local ``raw_logits`` helper must bypass tanh."""
    import bc_train  # loaded by conftest from projects/supervised-cloning/train.py

    torch.manual_seed(0)
    model = QuartoCNNAutoregUnified()
    # Force a wide range by initialising the last linear with a large scale.
    with torch.no_grad():
        model.fc2_place.weight.mul_(50.0)
        model.fc2_place.bias.mul_(50.0)
    board, aux = _rand_inputs(4)
    logits_place, _ = bc_train.raw_logits(model, board, aux)
    # If tanh were applied, |x| ≤ 1. We expect a much larger magnitude.
    assert logits_place.abs().max().item() > 1.5


def test_aux_dim_mismatch_raises(model):
    board = torch.randn(2, 16, 4, 4)
    bad_aux = torch.randn(2, 16)  # 16 instead of 32
    with pytest.raises(ValueError, match="aux of size 32"):
        model(board, bad_aux)


def test_q_values_phase_routes_to_correct_head(model):
    # Dropout would make two forward passes non-deterministic; eval mode
    # disables it so the routed Q-values match the direct forward output.
    model.eval()
    board, aux = _rand_inputs(4)
    q_place, q_select = model(board, aux)
    routed_p = model.q_values_phase(board, aux, phase=PHASE_PLACE)
    routed_s = model.q_values_phase(board, aux, phase=PHASE_SELECT)
    assert torch.allclose(routed_p, q_place)
    assert torch.allclose(routed_s, q_select)
    phases = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    routed_mixed = model.q_values_phase(board, aux, phase=phases)
    expected = torch.stack(
        [q_place[0], q_select[1], q_place[2], q_select[3]], dim=0
    )
    assert torch.allclose(routed_mixed, expected)


def test_trunk_is_phase_agnostic(model):
    """The unified trunk must ignore the ``phase`` argument entirely.
    Eval mode is required to silence dropout, otherwise two passes diverge.
    """
    model.eval()
    board, aux = _rand_inputs(3)
    with torch.no_grad():
        a = model(board, aux, phase="place")
        b = model(board, aux, phase="select")
    assert torch.allclose(a[0], b[0])
    assert torch.allclose(a[1], b[1])


def test_unbound_variant_has_same_shape():
    torch.manual_seed(0)
    m = QuartoCNNAutoregUnifiedUnbound()
    board, aux = _rand_inputs(2)
    qp, qs = m(board, aux)
    assert qp.shape == (2, 16)
    assert qs.shape == (2, 16)


def test_numpy_inputs_accepted(model):
    board = np.random.randn(2, 16, 4, 4).astype(np.float32)
    aux = np.random.randn(2, 32).astype(np.float32)
    qp, qs = model(board, aux)
    assert qp.shape == (2, 16) and qs.shape == (2, 16)
