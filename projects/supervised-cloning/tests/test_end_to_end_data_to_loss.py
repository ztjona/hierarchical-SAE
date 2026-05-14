"""End-to-end smoke test: collect a tiny dataset → augment → forward pass → loss.

Validates that the full data → model → loss path works without errors and
produces a finite, differentiable loss. Catches integration breaks that
unit tests in isolation would miss.
"""

from __future__ import annotations
import tempfile
import numpy as np
from pathlib import Path
import torch

import bc_train
import bc_collect


def test_collect_and_train_smoke(tmp_path: Path):
    """Run a 5-game collection, load it through the trainer's pipeline,
    augment, and compute one training-step loss + backward.
    """
    # 1. Collect a tiny dataset by calling collect_data.collect_game directly.
    from quartopy import QuartoGame
    from bot.minimax_bot import MinimaxBot
    from bot.random_bot import Quarto_bot as RandomBot

    teacher = MinimaxBot(depth=2)
    opponent = RandomBot()
    all_records = []
    for gid in range(3):
        recs = bc_collect.collect_game(
            teacher=teacher,
            opponent=opponent,
            mode_2x2=True,
            game_id=gid,
            teacher_is_p1=(gid % 2 == 0),
        )
        all_records.extend(recs)
    assert len(all_records) > 0, "No records collected — collection pipeline is broken"

    boards = np.stack([r["board"] for r in all_records])
    aux = np.stack([r["aux"] for r in all_records])
    labels = np.array([r["label"] for r in all_records], dtype=np.int16)
    actions = np.array([r["action"] for r in all_records], dtype=np.uint8)
    legal_masks = np.stack([r["legal_mask"] for r in all_records])
    soft_targets = np.stack([r["soft_target"] for r in all_records])
    game_ids = np.array([r["game_id"] for r in all_records], dtype=np.int32)

    # 2. Verify schema invariants
    assert aux.shape == (len(all_records), 32)
    assert legal_masks.dtype == bool
    # Every soft_target should sum to ~1 (over legal positions).
    sums = soft_targets.sum(axis=1)
    np.testing.assert_array_almost_equal(sums, np.ones_like(sums), decimal=5)
    # And it should be zero on illegal positions.
    assert ((soft_targets > 0) & (~legal_masks)).sum() == 0, (
        "Soft target placed prob mass on illegal positions"
    )

    # 3. Run the augmentation
    out = bc_train.augment_symmetries(
        boards, aux, labels, actions, legal_masks, soft_targets
    )
    for arr in out:
        assert arr.shape[0] == 8 * len(all_records)
    aug_b, aug_aux, aug_lbl, aug_act, aug_msk, aug_soft = out

    # Soft targets must still sum to 1 after augmentation.
    sums_aug = aug_soft.sum(axis=1)
    np.testing.assert_array_almost_equal(sums_aug, np.ones_like(sums_aug), decimal=5)

    # 4. Forward pass + loss + backward
    torch.manual_seed(0)
    model = bc_train.QuartoCNNAutoregUnified()
    # Pick a small batch
    n = min(32, aug_b.shape[0])
    batch = (
        torch.from_numpy(aug_b[:n]),
        torch.from_numpy(aug_aux[:n]),
        torch.from_numpy(aug_lbl[:n].astype(np.int64)),
        torch.from_numpy(aug_act[:n].astype(np.int64)),
        torch.from_numpy(aug_msk[:n].astype(np.bool_)),
        torch.from_numpy(aug_soft[:n].astype(np.float32)),
    )
    logits_p, logits_s = bc_train.raw_logits(model, batch[0], batch[1])
    assert logits_p.shape == (n, 16) and logits_s.shape == (n, 16)

    # Compose a tiny per-head loss similar to run_epoch
    place_idx = batch[3] == bc_train.ACTION_PLACE
    select_idx = batch[3] == bc_train.ACTION_SELECT
    loss = torch.tensor(0.0)
    if place_idx.any():
        loss = loss + bc_train.soft_masked_ce(
            logits_p[place_idx], batch[5][place_idx], batch[4][place_idx]
        )
    if select_idx.any():
        loss = loss + bc_train.soft_masked_ce(
            logits_s[select_idx], batch[5][select_idx], batch[4][select_idx]
        )
    assert torch.isfinite(loss), f"Smoke loss is non-finite: {loss.item()}"
    loss.backward()
    # at least one parameter must have a non-zero gradient
    has_grad = any(
        (p.grad is not None and p.grad.abs().sum().item() > 0)
        for p in model.parameters()
    )
    assert has_grad
