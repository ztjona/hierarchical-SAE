"""Tests for soft_masked_ce / hard_masked_ce in train.py.

Validates that:
- One-hot soft target reproduces hard CE exactly.
- Uniform soft target over k legal positions yields loss = log(k).
- Illegal-position masking does not produce NaN gradients.
- Gradient flows only through legal positions.
"""

from __future__ import annotations
import math
import torch

import bc_train

soft_masked_ce = bc_train.soft_masked_ce
hard_masked_ce = bc_train.hard_masked_ce


def test_soft_target_one_hot_equals_hard_ce():
    torch.manual_seed(0)
    logits = torch.randn(4, 16)
    mask = torch.ones(4, 16, dtype=torch.bool)
    hard = torch.tensor([3, 7, 0, 15], dtype=torch.long)

    soft = torch.zeros(4, 16)
    soft[torch.arange(4), hard] = 1.0

    loss_soft = soft_masked_ce(logits, soft, mask)
    loss_hard = hard_masked_ce(logits, hard, mask)
    assert torch.allclose(loss_soft, loss_hard, atol=1e-6)


def test_uniform_soft_target_yields_log_k():
    """If logits are zero on legal positions, uniform soft target gives log(k)."""
    k = 4
    logits = torch.zeros(1, 16)
    mask = torch.zeros(1, 16, dtype=torch.bool)
    mask[0, :k] = True
    soft = torch.zeros(1, 16)
    soft[0, :k] = 1.0 / k

    loss = soft_masked_ce(logits, soft, mask)
    assert math.isclose(loss.item(), math.log(k), abs_tol=1e-5)


def test_illegal_mask_no_nan():
    torch.manual_seed(1)
    logits = torch.randn(2, 16, requires_grad=True)
    mask = torch.zeros(2, 16, dtype=torch.bool)
    # Only 3 legal positions; rest are masked.
    mask[:, :3] = True
    soft = torch.zeros(2, 16)
    soft[:, :3] = 1.0 / 3
    loss = soft_masked_ce(logits, soft, mask)
    assert torch.isfinite(loss), "Loss must be finite with masked illegals"
    loss.backward()
    assert torch.isfinite(logits.grad).all(), "Gradients must be finite everywhere"


def test_gradient_on_illegal_positions_is_negligible():
    """Mask of -1e9 should drive softmax mass on illegal positions ≈ 0,
    so the gradient on illegal logits is ~0 (not exactly 0 due to float math).
    """
    torch.manual_seed(2)
    logits = torch.randn(1, 16, requires_grad=True)
    mask = torch.zeros(1, 16, dtype=torch.bool)
    mask[0, :2] = True
    soft = torch.zeros(1, 16)
    soft[0, :2] = 0.5
    loss = soft_masked_ce(logits, soft, mask)
    loss.backward()
    legal_grad_norm = logits.grad[0, :2].abs().sum().item()
    illegal_grad_norm = logits.grad[0, 2:].abs().sum().item()
    # Illegal gradient should be many orders of magnitude smaller than legal.
    assert illegal_grad_norm < 1e-6, (
        f"Illegal positions still receive gradient: "
        f"illegal={illegal_grad_norm}, legal={legal_grad_norm}"
    )


def test_hard_ce_matches_torch_cross_entropy_on_full_legal():
    """When all positions are legal, hard_masked_ce should match
    F.cross_entropy up to numerical precision.
    """
    import torch.nn.functional as F

    torch.manual_seed(3)
    logits = torch.randn(8, 16)
    mask = torch.ones(8, 16, dtype=torch.bool)
    targets = torch.randint(0, 16, (8,))
    expected = F.cross_entropy(logits, targets)
    got = hard_masked_ce(logits, targets, mask)
    assert torch.allclose(got, expected, atol=1e-5)
