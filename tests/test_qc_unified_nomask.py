"""Smoke tests for QC (Q-series) — `models.CNN_unified_nomask` and the
matching bot. Goal: catch shape / API regressions before launching the
5000-epoch training run, not to exhaustively test learning behaviour.

Run with:
    pytest tests/test_qc_unified_nomask.py -q
"""

import io
import numpy as np
import pytest
import torch

from models.CNN_unified_nomask import (
    QuartoCNNUnifiedNoMask,
    legality_target_from_board,
)


@pytest.fixture
def empty_board() -> torch.Tensor:
    return torch.zeros(1, 16, 4, 4)


@pytest.fixture
def half_full_board() -> torch.Tensor:
    # Place 3 distinct pieces on 3 distinct cells.
    b = torch.zeros(2, 16, 4, 4)
    b[0, 0, 0, 0] = 1.0
    b[0, 5, 1, 2] = 1.0
    b[0, 11, 3, 3] = 1.0
    b[1, 2, 2, 2] = 1.0
    return b


@pytest.fixture
def model() -> QuartoCNNUnifiedNoMask:
    torch.manual_seed(0)
    m = QuartoCNNUnifiedNoMask()
    m.eval()
    return m


# ─────────────── legality_target_from_board ──────────────────────────
class TestLegalityTarget:
    def test_empty_board_all_legal(self, empty_board):
        leg = legality_target_from_board(empty_board)
        assert leg.shape == (1, 16)
        assert leg.sum().item() == 16.0
        assert torch.all((leg == 0) | (leg == 1))

    def test_half_full_board_counts(self, half_full_board):
        leg = legality_target_from_board(half_full_board)
        assert leg.shape == (2, 16)
        assert leg[0].sum().item() == 16 - 3
        assert leg[1].sum().item() == 16 - 1

    def test_row_major_order(self):
        # Place a piece at (row=0, col=2) -> flat index 2 should be 0.
        b = torch.zeros(1, 16, 4, 4)
        b[0, 7, 0, 2] = 1.0
        leg = legality_target_from_board(b)
        assert leg[0, 2].item() == 0.0
        assert leg[0, 0].item() == 1.0

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            legality_target_from_board(torch.zeros(1, 4, 4))
        with pytest.raises(ValueError):
            legality_target_from_board(torch.zeros(1, 8, 4, 4))


# ─────────────── model forward ───────────────────────────────────────
class TestModelForward:
    def test_forward_shapes_and_range(self, model):
        x_board = torch.zeros(4, 16, 4, 4)
        x_aux = torch.zeros(4, 32)
        q_place, q_select = model(x_board, x_aux, phase="place")
        assert q_place.shape == (4, 16)
        assert q_select.shape == (4, 16)
        assert torch.all((q_place >= -1) & (q_place <= 1))
        assert torch.all((q_select >= -1) & (q_select <= 1))

    def test_forward_with_aux_returns_three_tensors(self, model):
        x_board = torch.zeros(3, 16, 4, 4)
        x_aux = torch.zeros(3, 32)
        q_place, q_select, leg_logits = model.forward_with_aux(x_board, x_aux)
        assert q_place.shape == (3, 16)
        assert q_select.shape == (3, 16)
        assert leg_logits.shape == (3, 16)
        # Logits are NOT capped to [-1, 1] — they go through BCEWithLogits.
        # Check they are at least finite.
        assert torch.isfinite(leg_logits).all()

    def test_phase_argument_is_ignored(self, model):
        x_board = torch.zeros(2, 16, 4, 4)
        x_aux = torch.zeros(2, 32)
        qp_a, qs_a = model(x_board, x_aux, phase="place")
        qp_b, qs_b = model(x_board, x_aux, phase="select")
        assert torch.allclose(qp_a, qp_b)
        assert torch.allclose(qs_a, qs_b)

    def test_legality_logits_shape(self, model):
        x_board = torch.zeros(5, 16, 4, 4)
        x_aux = torch.zeros(5, 32)
        out = model.legality_logits(x_board, x_aux)
        assert out.shape == (5, 16)

    def test_wrong_aux_dim_raises(self, model):
        x_board = torch.zeros(1, 16, 4, 4)
        x_aux = torch.zeros(1, 16)  # OA had 16 — QC requires 32
        with pytest.raises(ValueError):
            model(x_board, x_aux)

    def test_numpy_inputs_accepted(self, model):
        x_board = np.zeros((2, 16, 4, 4), dtype=np.float32)
        x_aux = np.zeros((2, 32), dtype=np.float32)
        q_place, q_select = model(x_board, x_aux)
        assert q_place.shape == (2, 16)


# ─────────────── state_dict round-trip ───────────────────────────────
class TestSerialization:
    def test_state_dict_roundtrip(self, model):
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)
        sd = torch.load(buf, weights_only=True)
        m2 = QuartoCNNUnifiedNoMask()
        m2.load_state_dict(sd)
        m2.eval()
        x_board = torch.zeros(1, 16, 4, 4)
        x_aux = torch.zeros(1, 32)
        q1 = model(x_board, x_aux)
        q2 = m2(x_board, x_aux)
        assert torch.allclose(q1[0], q2[0])
        assert torch.allclose(q1[1], q2[1])

    def test_name_property(self, model):
        assert model.name == "QuartoCNN_unified_nomask"


# ─────────────── q_values_phase routes correctly ─────────────────────
class TestQValuesPhase:
    def test_place_returns_place_head(self, model):
        x_board = torch.zeros(3, 16, 4, 4)
        x_aux = torch.zeros(3, 32)
        q_phase = model.q_values_phase(x_board, x_aux, phase="place")
        q_place, _ = model(x_board, x_aux)
        assert torch.allclose(q_phase, q_place)

    def test_select_returns_select_head(self, model):
        x_board = torch.zeros(3, 16, 4, 4)
        x_aux = torch.zeros(3, 32)
        q_phase = model.q_values_phase(x_board, x_aux, phase="select")
        _, q_select = model(x_board, x_aux)
        assert torch.allclose(q_phase, q_select)
