# -*- coding: utf-8 -*-
"""D1 — position-structure Q_select metric.

See ``analysis/qselect_diagnostics/PLAN.md`` → "D1".

Falsifies (or supports) H1: that the existing match-outcome-conditioned Δ
metric hides per-position structure already learned by Q_select.

Usage:
    python analysis/qselect_diagnostics/position_structure.py \
        --exp 'Ta_minimaxSelect(1)0514_DEPTH_2' [--epoch 4000] [--n-states 500]
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np


def spearmanr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Minimal Spearman ρ for 1-D arrays (scipy-free).

    Returns ``(rho, nan)``; the p-value slot is unused but kept for parity.
    Handles ties via average ranking (pandas-style midrank).
    """

    def _rankdata(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
        # tie-correction: average ranks within equal-value groups
        s = x[order]
        i = 0
        while i < len(s):
            j = i + 1
            while j < len(s) and s[j] == s[i]:
                j += 1
            if j > i + 1:
                avg = (i + j + 1) / 2.0  # average rank in 1-based
                ranks[order[i:j]] = avg
            i = j
        return ranks

    ra = _rankdata(a)
    rb = _rankdata(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if denom == 0:
        return (float("nan"), float("nan"))
    return (float((ra * rb).sum() / denom), float("nan"))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from analysis.qselect_diagnostics._common import (  # noqa: E402
    SelectState,
    emit_jsonl,
    load_checkpoint,
    qselect_predict,
    sample_states,
)


# Target normalisation: terminal-force values map to ~±1; non-terminal
# minimax aggregates fall close to 0. 0.5 cleanly separates the two regimes
# for any oracle depth ≥ 1.
FORCE_THRESHOLD = 0.5


def _classify_state(
    target: np.ndarray, mask: np.ndarray
) -> tuple[set[int], set[int], int]:
    """Identify forcing-loss pieces (target ≤ -thr) and safe pieces.

    Returns ``(forcing_loss_set, safe_set, n_available)`` over the masked
    piece indices. "Safe" = available ∧ target > -threshold.
    """
    avail_idx = np.where(mask > 0.5)[0]
    forcing_loss = {int(i) for i in avail_idx if target[i] <= -FORCE_THRESHOLD}
    safe = {int(i) for i in avail_idx if target[i] > -FORCE_THRESHOLD}
    return forcing_loss, safe, len(avail_idx)


def _argmin_masked(q: np.ndarray, mask: np.ndarray) -> int:
    q_masked = np.where(mask > 0.5, q, np.inf)
    return int(np.argmin(q_masked))


def _argmax_masked(q: np.ndarray, mask: np.ndarray) -> int:
    q_masked = np.where(mask > 0.5, q, -np.inf)
    return int(np.argmax(q_masked))


def _state_spearman(
    q: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> float | None:
    avail_idx = np.where(mask > 0.5)[0]
    if len(avail_idx) < 3:
        return None
    q_avail = q[avail_idx]
    t_avail = target[avail_idx]
    # If the oracle target is constant across the available set, ρ is
    # undefined; skip these (they don't carry ranking information).
    if np.allclose(t_avail, t_avail[0]):
        return None
    if np.allclose(q_avail, q_avail[0]):
        return 0.0
    rho, _p = spearmanr(q_avail, t_avail)
    if not np.isfinite(rho):
        return None
    return float(rho)


def _aggregate_metrics(raw_states: list[SelectState], q_all: np.ndarray) -> dict:
    """Classify each (state, q_predictions) pair and aggregate D1 metrics.

    Shared between the CLI ``run()`` and the in-loop
    ``compute_position_structure_record`` helper, so the training-loop
    summary and the standalone diagnostic compute the same numbers.
    """
    n_total = len(raw_states)
    n_decisive = 0
    n_single_force = 0
    forcing_loss_hits = 0
    n_any_force = 0
    forcing_loss_bottom_hits = 0
    chance_baseline_bottom = []
    n_safe_eval = 0
    safe_piece_hits = 0
    rhos: list[float] = []
    chance_baselines: list[float] = []
    forcing_set_sizes: list[int] = []

    for s, q in zip(raw_states, q_all):
        forcing, safe, n_avail = _classify_state(
            s.target_sel_minimax, s.target_sel_minimax_mask
        )
        if n_avail < 2:
            continue

        rho = _state_spearman(q, s.target_sel_minimax, s.target_sel_minimax_mask)
        if rho is not None:
            rhos.append(rho)

        # Decisive states: at least one safe AND at least one forcing-loss piece.
        decisive = len(forcing) >= 1 and len(safe) >= 1
        if decisive:
            n_decisive += 1
            chance_baselines.append(1.0 / n_avail)
            argmax_idx = _argmax_masked(q, s.target_sel_minimax_mask)
            n_safe_eval += 1
            if argmax_idx in safe:
                safe_piece_hits += 1

        # Strict forcing-loss-recall: exactly one forcing-loss piece (rare).
        if len(forcing) == 1 and n_avail >= 2:
            n_single_force += 1
            argmin_idx = _argmin_masked(q, s.target_sel_minimax_mask)
            if argmin_idx in forcing:
                forcing_loss_hits += 1

        # Relaxed: at least one forcing-loss piece. Does argmin(Q_select) over
        # available pieces fall inside the forcing-loss set? Chance baseline =
        # |forcing| / n_avail (random pick).
        if len(forcing) >= 1 and n_avail >= 2:
            n_any_force += 1
            forcing_set_sizes.append(len(forcing))
            chance_baseline_bottom.append(len(forcing) / n_avail)
            argmin_idx = _argmin_masked(q, s.target_sel_minimax_mask)
            if argmin_idx in forcing:
                forcing_loss_bottom_hits += 1

    forcing_loss_recall = (
        forcing_loss_hits / n_single_force if n_single_force else None
    )
    forcing_loss_bottom_recall = (
        forcing_loss_bottom_hits / n_any_force if n_any_force else None
    )
    forcing_loss_bottom_chance = (
        float(np.mean(chance_baseline_bottom)) if chance_baseline_bottom else None
    )
    safe_piece_recall = safe_piece_hits / n_safe_eval if n_safe_eval else None
    chance_baseline = float(np.mean(chance_baselines)) if chance_baselines else None
    forcing_set_size_mean = (
        float(np.mean(forcing_set_sizes)) if forcing_set_sizes else None
    )

    rho_arr = np.array(rhos, dtype=np.float64)
    rho_summary = {
        "n": int(len(rho_arr)),
        "mean": float(rho_arr.mean()) if len(rho_arr) else None,
        "p25": float(np.percentile(rho_arr, 25)) if len(rho_arr) else None,
        "p50": float(np.percentile(rho_arr, 50)) if len(rho_arr) else None,
        "p75": float(np.percentile(rho_arr, 75)) if len(rho_arr) else None,
    }

    return {
        "n_states_total": int(n_total),
        "n_states_decisive": int(n_decisive),
        "n_states_single_forcing": int(n_single_force),
        "n_states_any_forcing": int(n_any_force),
        "forcing_set_size_mean": forcing_set_size_mean,
        "forcing_loss_recall": forcing_loss_recall,
        "forcing_loss_bottom_recall": forcing_loss_bottom_recall,
        "forcing_loss_bottom_chance": forcing_loss_bottom_chance,
        "safe_piece_recall": safe_piece_recall,
        "chance_baseline_recall": chance_baseline,
        "spearman_rho": rho_summary,
    }


def compute_position_structure_record(
    net,
    *,
    n_states: int = 200,
    n_pieces_min: int = 2,
    n_last_states: int = 4,
    oracle_depth: int = 2,
    seed: int = 1234,
) -> dict | None:
    """In-process D1 for the training loop. Lighter than the CLI ``run()``.

    Returns the same per-state-aggregation dict produced by
    ``_aggregate_metrics`` (so JSONL fields match the standalone
    diagnostic), or ``None`` if sampling returned no usable states. No
    JSONL emission, no checkpoint metadata; the caller injects fields
    into the training-loop record.

    Lighter default ``n_states`` (200 vs 500) keeps per-checkpoint cost
    ~10s on Sa(3); std-error of recall stays under ±0.02 at the typical
    ~0.8 operating point.
    """
    raw_states: list[SelectState] = sample_states(
        net=net,
        n_states=n_states * 3,
        n_pieces_min=n_pieces_min,
        mode_2x2=True,
        n_last_states=n_last_states,
        oracle_depth=oracle_depth,
        require_nonzero_oracle=False,
        seed=seed,
    )
    if not raw_states:
        return None
    q_all = qselect_predict(net, raw_states)
    return _aggregate_metrics(raw_states, q_all)


def run(
    exp_name: str,
    epoch: int | None,
    n_states: int,
    n_pieces_min: int,
    n_last_states: int,
    oracle_depth: int,
    seed: int,
) -> dict:
    net, cfg = load_checkpoint(exp_name, epoch=epoch)

    # We over-sample so the post-filter set still hits n_states.
    raw_states: list[SelectState] = sample_states(
        net=net,
        n_states=n_states * 3,
        n_pieces_min=n_pieces_min,
        mode_2x2=True,
        n_last_states=n_last_states,
        oracle_depth=oracle_depth,
        require_nonzero_oracle=False,
        seed=seed,
    )
    if not raw_states:
        raise RuntimeError("No states sampled.")

    q_all = qselect_predict(net, raw_states)
    metrics = _aggregate_metrics(raw_states, q_all)
    n_total = metrics["n_states_total"]
    n_decisive = metrics["n_states_decisive"]
    n_single_force = metrics["n_states_single_forcing"]
    n_any_force = metrics["n_states_any_forcing"]
    forcing_set_size_mean = metrics["forcing_set_size_mean"]
    forcing_loss_recall = metrics["forcing_loss_recall"]
    forcing_loss_bottom_recall = metrics["forcing_loss_bottom_recall"]
    forcing_loss_bottom_chance = metrics["forcing_loss_bottom_chance"]
    safe_piece_recall = metrics["safe_piece_recall"]
    chance_baseline = metrics["chance_baseline_recall"]
    rho_summary = metrics["spearman_rho"]

    record = {
        "diagnostic": "position_structure",
        "schema_version": 1,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "exp_name": cfg["exp_name"],
        "epoch": cfg["epoch"],
        "architecture": cfg["architecture"],
        "checkpoint_path": cfg["checkpoint_path"],
        "config": {
            "n_states_requested": n_states,
            "n_pieces_min": n_pieces_min,
            "n_last_states": n_last_states,
            "oracle_depth": oracle_depth,
            "force_threshold": FORCE_THRESHOLD,
            "seed": seed,
        },
        "n_states_total": int(n_total),
        "n_states_decisive": int(n_decisive),
        "n_states_single_forcing": int(n_single_force),
        "n_states_any_forcing": int(n_any_force),
        "forcing_set_size_mean": forcing_set_size_mean,
        "forcing_loss_recall": forcing_loss_recall,
        "forcing_loss_bottom_recall": forcing_loss_bottom_recall,
        "forcing_loss_bottom_chance": forcing_loss_bottom_chance,
        "safe_piece_recall": safe_piece_recall,
        "chance_baseline_recall": chance_baseline,
        "spearman_rho": rho_summary,
        "network": "policy_net",
    }

    out = emit_jsonl(cfg["exp_name"], "position_structure", record)
    print(f"Wrote {out}")
    print(f"  n_states_total                = {n_total}")
    print(f"  n_states_decisive             = {n_decisive}")
    print(f"  n_states_any_forcing          = {n_any_force}")
    print(f"  n_states_single_forcing       = {n_single_force}")
    print(f"  forcing_set_size_mean         = {forcing_set_size_mean}")
    print(f"  forcing_loss_recall (n=1)     = {forcing_loss_recall}")
    print(f"  forcing_loss_bottom_recall    = {forcing_loss_bottom_recall}")
    print(f"  forcing_loss_bottom_chance    = {forcing_loss_bottom_chance}")
    print(f"  safe_piece_recall             = {safe_piece_recall}")
    print(f"  chance_baseline_recall        = {chance_baseline}")
    print(f"  spearman_rho_mean             = {rho_summary['mean']}  (n={rho_summary['n']})")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True, help="Experiment folder under CHECKPOINTS/")
    parser.add_argument("--epoch", type=int, default=None, help="Specific epoch; latest if omitted")
    parser.add_argument("--n-states", type=int, default=500)
    parser.add_argument("--n-pieces-min", type=int, default=2)
    parser.add_argument("--n-last-states", type=int, default=4)
    parser.add_argument("--oracle-depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    run(
        exp_name=args.exp,
        epoch=args.epoch,
        n_states=args.n_states,
        n_pieces_min=args.n_pieces_min,
        n_last_states=args.n_last_states,
        oracle_depth=args.oracle_depth,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
