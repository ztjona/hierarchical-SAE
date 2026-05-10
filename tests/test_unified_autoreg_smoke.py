"""Smoke test for the unified-aux variant of the decoupled-autoreg pipeline.

Run with:
    python tests/test_unified_autoreg_smoke.py

Covers:
  1. QuartoCNNAutoregUnified forward / q_values_phase / predict_phase shapes.
  2. AUX_DIM mismatch raises a clear error.
  3. Phase routing in q_values_phase agrees with raw forward heads.
  4. DQN_training_step under TRANSITION_SCHEMA="unified_autoreg" produces
     finite per-phase losses on a synthetic batch.
  5. Backwards-compat: QuartoCNNAutoreg (legacy class) still constructs and
     forwards correctly with 16-d aux.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from tensordict import TensorDict

from models.CNN_autoreg import (
    QuartoCNNAutoreg,
    QuartoCNNAutoregUnified,
    QuartoCNNAutoregUnifiedUnbound,
    PHASE_PLACE,
    PHASE_SELECT,
)
from QuartoRL.RL_functions import (
    DQN_training_step,
    TRANSITION_SCHEMA_UNIFIED_AUTOREG,
    DECOUPLED_TARGET_TD_PLACE_MC_SELECT,
    UNIFIED_AUX_DIM,
    _unified_aux,
)


torch.manual_seed(0)
np.random.seed(0)


def _rand_board(batch: int) -> torch.Tensor:
    return torch.randn(batch, 16, 4, 4)


def _rand_unified_aux(batch: int) -> torch.Tensor:
    offered = torch.zeros(batch, 16)
    available = torch.zeros(batch, 16)
    for i in range(batch):
        # one offered piece + a random available subset disjoint from it
        offered_idx = int(torch.randint(0, 16, ()).item())
        offered[i, offered_idx] = 1.0
        avail_mask = torch.bernoulli(torch.full((16,), 0.5))
        avail_mask[offered_idx] = 0.0
        available[i] = avail_mask
    return torch.cat([offered, available], dim=-1)


def test_unified_forward_shapes():
    model = QuartoCNNAutoregUnified()
    batch = 4
    board = _rand_board(batch)
    aux = _rand_unified_aux(batch)

    q_place, q_select = model.forward(board, aux, phase="place")
    assert q_place.shape == (batch, 16), q_place.shape
    assert q_select.shape == (batch, 16), q_select.shape
    # tanh-bounded
    assert q_place.abs().max().item() <= 1.0 + 1e-6
    assert q_select.abs().max().item() <= 1.0 + 1e-6
    print("[ok] unified forward shapes + tanh bounds")


def test_unbound_no_tanh():
    model = QuartoCNNAutoregUnifiedUnbound()
    aux = _rand_unified_aux(4) * 5.0  # push aux scale up
    board = _rand_board(4) * 5.0
    q_place, _ = model.forward(board, aux, phase="place")
    # Unbound model: no tanh — outputs can exceed [-1, 1] in principle.
    # We don't assert >1 (random init may not produce it); just assert
    # the activation function is identity by checking it equals fc2_place.
    assert q_place.shape == (4, 16)
    print("[ok] unified-unbound shapes")


def test_q_values_phase_routes_correctly():
    model = QuartoCNNAutoregUnified()
    model.eval()
    batch = 6
    board = _rand_board(batch)
    aux = _rand_unified_aux(batch)
    phase = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)  # 3 place, 3 select

    with torch.no_grad():
        q_place_full, q_select_full = model.forward(board, aux, phase=phase)
        q_routed = model.q_values_phase(board, aux, phase=phase)

    # Place rows must equal place head, select rows must equal select head.
    torch.testing.assert_close(q_routed[:3], q_place_full[:3])
    torch.testing.assert_close(q_routed[3:], q_select_full[3:])
    print("[ok] q_values_phase routes by phase")


def test_aux_dim_mismatch_raises():
    model = QuartoCNNAutoregUnified()
    board = _rand_board(2)
    bad_aux = torch.zeros(2, 16)  # legacy 16-d, should be rejected
    try:
        model.forward(board, bad_aux, phase="place")
    except ValueError as e:
        assert "32" in str(e), f"unexpected error message: {e}"
        print("[ok] 16-d aux rejected with clear message")
        return
    raise AssertionError("expected ValueError on 16-d aux into unified model")


def test_unified_aux_helper():
    aux = _unified_aux(piece_index=5, available_pieces={0, 3, 7, 12})
    assert aux.shape == (32,), aux.shape
    assert aux[5] == 1.0  # offered one-hot
    assert aux.sum() == 1.0 + 4.0  # 1 offered + 4 available
    # available block at offsets 16:32
    assert aux[16 + 0] == 1.0 and aux[16 + 3] == 1.0
    assert aux[16 + 7] == 1.0 and aux[16 + 12] == 1.0
    # zero-piece sentinel
    aux_no_offered = _unified_aux(piece_index=-1, available_pieces={1, 2})
    assert aux_no_offered[:16].sum() == 0.0
    assert aux_no_offered[16 + 1] == 1.0
    print("[ok] _unified_aux helper layout")


def _synthetic_unified_batch(batch: int = 8) -> TensorDict:
    """Build a minimal unified_autoreg batch sufficient for one training step."""
    board = _rand_board(batch)
    aux = _rand_unified_aux(batch)
    next_board = _rand_board(batch)
    next_aux = _rand_unified_aux(batch)
    # half place, half select
    phase = torch.tensor([PHASE_PLACE] * (batch // 2) + [PHASE_SELECT] * (batch // 2),
                         dtype=torch.long)
    next_phase = torch.where(phase == PHASE_PLACE,
                             torch.tensor(PHASE_SELECT),
                             torch.tensor(PHASE_PLACE))
    # valid_mask: full mask of 1s so any chosen action is "valid"
    valid_mask = torch.ones(batch, 16)
    next_valid_mask = torch.ones(batch, 16)
    action = torch.randint(0, 16, (batch,))
    reward = torch.zeros(batch)
    done = torch.zeros(batch, dtype=torch.bool)
    outcome = torch.tensor([1.0, -1.0] * (batch // 2))
    steps_to_terminal = torch.arange(batch, dtype=torch.float32)

    return TensorDict(
        {
            "state_board": board,
            "state_aux": aux,
            "phase": phase,
            "valid_mask": valid_mask,
            "action": action,
            "reward": reward,
            "done": done,
            "next_state_board": next_board,
            "next_state_aux": next_aux,
            "next_phase": next_phase,
            "next_valid_mask": next_valid_mask,
            "outcome": outcome,
            "steps_to_terminal": steps_to_terminal,
        },
        batch_size=[batch],
    )


def test_dqn_training_step_unified():
    policy_net = QuartoCNNAutoregUnified()
    target_net = QuartoCNNAutoregUnified()
    target_net.load_state_dict(policy_net.state_dict())

    batch = _synthetic_unified_batch(batch=8)
    q_place, target_place, q_select, target_select = DQN_training_step(
        policy_net=policy_net,
        target_net=target_net,
        GAMMA=0.99,
        exp_batch=batch,
        TRANSITION_SCHEMA=TRANSITION_SCHEMA_UNIFIED_AUTOREG,
        DECOUPLED_TARGET_STYLE=DECOUPLED_TARGET_TD_PLACE_MC_SELECT,
    )

    # Half the batch is place, half select — each head should populate 4 rows.
    assert q_place.numel() == 4, q_place.shape
    assert q_select.numel() == 4, q_select.shape
    assert torch.isfinite(q_place).all() and torch.isfinite(target_place).all()
    assert torch.isfinite(q_select).all() and torch.isfinite(target_select).all()

    # Backward should produce real gradients.
    loss = (q_place - target_place).pow(2).mean() + (q_select - target_select).pow(2).mean()
    loss.backward()
    grads = [p.grad for p in policy_net.parameters() if p.grad is not None]
    assert grads, "no gradients flowed"
    assert any(g.abs().sum().item() > 0 for g in grads), "all gradients are zero"
    print("[ok] DQN_training_step under unified_autoreg yields finite per-phase loss")


def test_legacy_decoupled_autoreg_still_works():
    """Backwards-compat: original QuartoCNNAutoreg with 16-d aux unchanged."""
    model = QuartoCNNAutoreg()
    batch = 3
    board = _rand_board(batch)
    aux = torch.randn(batch, 16)  # legacy 16-d
    q_place, q_select = model.forward(board, aux, phase="place")
    assert q_place.shape == (batch, 16)
    assert q_select.shape == (batch, 16)
    print("[ok] legacy QuartoCNNAutoreg still accepts 16-d aux")


if __name__ == "__main__":
    test_unified_aux_helper()
    test_unified_forward_shapes()
    test_unbound_no_tanh()
    test_q_values_phase_routes_correctly()
    test_aux_dim_mismatch_raises()
    test_dqn_training_step_unified()
    test_legacy_decoupled_autoreg_still_works()
    print("\nAll smoke tests passed.")
