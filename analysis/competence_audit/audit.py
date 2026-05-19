# -*- coding: utf-8 -*-
"""Competence audit for unified_autoreg checkpoints (V0 / V0b).

Five behavioral tests adapted from
``games-interp/scripts/model_competence_audit.py`` to the
``(state_board, state_aux, phase)`` input shape of the ``unified_autoreg``
schema.  No oracle, no training.  Runs in minutes on a 2 000-state sample.

Tests
-----
A. Winning placement       — When a winning cell exists for the offered
                             piece, does argmax Q_place[legal] land on it?
B. Losing-piece avoidance  — When safe AND losing pieces both exist in
                             storage, does argmax Q_select[available] avoid
                             the losing pieces?
C. Offered-piece sensitivity
                           — Fraction of distinct argmax-place cells across
                             all available piece options.
D. Q-occupancy gap         — Mean Q_place(empty) − Mean Q_place(occupied).
                             >0 means the model has learned legality.
E. Phase-stratified entropy
                           — Entropy of softmax(Q_place[legal]) bucketed by
                             piece count.  Lower entropy late-game → more
                             decisive.
D'. Counterfactual Q-occupancy gap
                           — Decomposes Test D's negative gap.  Hypothesis:
                             the gap is *input-driven* (the trunk reads off
                             non-zero piece channels at occupied cells).
                             Re-forwards the same board with all piece
                             channels zeroed at occupied cells (synthesising
                             an "empty everywhere" board) and re-measures
                             the gap at the formerly-occupied cells.
                             - If gap_cleared ≈ 0  → input-driven (i);
                               Q_place(occ) is a representation artefact
                               and can be ignored at interp time.
                             - If gap_cleared ≈ gap_orig → position/head
                               bias (ii); the head prefers specific cells
                               regardless of content.
                             Also reports per-piece-count stratification of
                             the original Test D gap.

Supports only the ``unified_autoreg`` schema (32-d aux).  The
``decoupled_autoreg`` series (e.g. ME_endgame) requires a separate branch;
pass ``--skip-decoupled`` (the default) or ``--force-decoupled`` to override.

Output
------
One JSONL record per (exp, epoch) appended to:
    analysis/competence_audit/results/<exp>/audit.jsonl

Usage
-----
    python analysis/competence_audit/audit.py \\
        --exp 'Ta_minimaxSelect(1)0514_DEPTH_2' \\
        [--epoch N] \\
        [--include-epoch-0] \\
        [--n-states N] \\
        [--seed N] \\
        [--batch-size N]

    # Run the full Va champion set at once:
    python analysis/competence_audit/audit.py --va-champions
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from quartopy import Board, Piece  # noqa: E402

from QuartoRL.RL_functions import (  # noqa: E402
    PHASE_PLACE,
    PHASE_SELECT,
    UNIFIED_AUX_DIM,
    gen_experience_unified_autoreg,
)
from bot.CNN_unified_bot import Quarto_bot as Quarto_unified_bot  # noqa: E402

# Reuse checkpoint loading + architecture registry from the qselect suite.
sys.path.insert(0, os.path.join(ROOT, "analysis", "qselect_diagnostics"))
from _common import load_checkpoint, infer_architecture  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON-serializable: {type(obj)}")


def _emit_jsonl(exp_name: str, record: dict) -> str:
    out_dir = os.path.join(RESULTS_DIR, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "audit.jsonl")
    with open(out_path, "a") as f:
        f.write(json.dumps(record, default=_json_default) + "\n")
    return out_path

# Va champion candidates from PLAN.md § Phase 2.  ME_endgame uses the
# decoupled_autoreg schema; it is skipped unless --force-decoupled is set.
VA_CHAMPIONS: list[str] = [
    "Ta_minimaxSelect(1)0514_DEPTH_2",
    "Ta_minimaxSelect(2)0515_DEPTH_1",
    "Ta_minimaxSelect(3)0515_SCALAR",
    "Sa_archScan(3)0512_ARCH_S4_uniform512",
    "OA_unifiedAux(1)0509_N_LAST_STATES_INIT_2",
]

# ME_endgame uses decoupled_autoreg (16-d aux) — separate branch needed.
VA_DECOUPLED_CHAMPIONS: list[str] = [
    "ME_endgame(2)0429_ENDGAME_FRACTION_0.5",
]


# ──────────────────────────────────────────────────────────────────────
# Internal state dataclasses (lightweight wrappers around raw arrays)
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _PlaceState:
    state_board: np.ndarray  # (16, 4, 4) float32
    state_aux: np.ndarray    # (32,)  float32  [offered_onehot | available_mask]
    valid_mask: np.ndarray   # (16,)  float32  legal placement mask
    n_pieces_on_board: int


@dataclass
class _SelectState:
    state_board: np.ndarray  # (16, 4, 4) float32
    state_aux: np.ndarray    # (32,)  float32  [zeros(16) | available_mask]
    valid_mask: np.ndarray   # (16,)  float32  legal piece selection mask
    n_pieces_on_board: int


# ──────────────────────────────────────────────────────────────────────
# State sampling
# ──────────────────────────────────────────────────────────────────────


def sample_states(
    net: torch.nn.Module,
    n_states: int = 2000,
    n_pieces_min: int = 2,
    mode_2x2: bool = True,
    n_last_states: int = 4,
    temperature: float = 0.5,
    seed: int = 42,
) -> tuple[list[_PlaceState], list[_SelectState]]:
    """Self-play to collect held-out PLACE and SELECT states.

    Returns ``(place_states, select_states)`` each capped at ``n_states``.
    No oracle needed — this is evaluation only.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    p1 = Quarto_unified_bot(model=net, deterministic=False, temperature=temperature)
    p2 = Quarto_unified_bot(model=net, deterministic=False, temperature=temperature)

    # Overshoot 3× to leave headroom for the n_pieces_min filter.
    per_match = max(1, n_last_states)
    n_matches = max(64, 3 * n_states // per_match)

    exp = gen_experience_unified_autoreg(
        p1_bot=p1,
        p2_bot=p2,
        n_last_states=n_last_states,
        number_of_matches=n_matches,
        verbose=False,
        PROGRESS_MESSAGE="Sampling states",
        mode_2x2=mode_2x2,
        REWARD_FUNCTION_TYPE="final",
        COLLECT_BOARDS=False,
        select_oracle=None,
    )

    phase_np = exp["phase"].cpu().numpy()
    sb_np = exp["state_board"].cpu().numpy()       # (N, 16, 4, 4)
    sa_np = exp["state_aux"].cpu().numpy()          # (N, 32)
    vm_np = exp["valid_mask"].cpu().numpy()         # (N, 16)

    place_states: list[_PlaceState] = []
    select_states: list[_SelectState] = []

    for i in range(len(phase_np)):
        occ = sb_np[i].any(axis=0)          # (4,4) bool
        n_placed = int(occ.sum())
        if n_placed < n_pieces_min:
            continue

        if phase_np[i] == PHASE_PLACE and len(place_states) < n_states:
            place_states.append(
                _PlaceState(
                    state_board=sb_np[i].astype(np.float32),
                    state_aux=sa_np[i].astype(np.float32),
                    valid_mask=vm_np[i].astype(np.float32),
                    n_pieces_on_board=n_placed,
                )
            )
        elif phase_np[i] == PHASE_SELECT and len(select_states) < n_states:
            select_states.append(
                _SelectState(
                    state_board=sb_np[i].astype(np.float32),
                    state_aux=sa_np[i].astype(np.float32),
                    valid_mask=vm_np[i].astype(np.float32),
                    n_pieces_on_board=n_placed,
                )
            )

        if len(place_states) >= n_states and len(select_states) >= n_states:
            break

    return place_states, select_states


# ──────────────────────────────────────────────────────────────────────
# Board reconstruction helpers
# ──────────────────────────────────────────────────────────────────────


def _board_from_encoding(
    state_board: np.ndarray,
) -> tuple[Board, list[tuple[int, int]]]:
    """Reconstruct a Board from the (16, 4, 4) encoding.

    Returns ``(board, empties)`` where ``empties`` is a list of (row, col)
    pairs for cells that have no piece.
    """
    board = Board("audit", storage=False, rows=4, cols=4)
    empties: list[tuple[int, int]] = []
    for r in range(4):
        for c in range(4):
            channels = np.flatnonzero(state_board[:, r, c])
            if len(channels):
                board.put_piece(Piece.from_index(int(channels[0])), r, c)
            else:
                empties.append((r, c))
    return board, empties


def _placing_wins(board: Board, piece: Piece, r: int, c: int) -> bool:
    """Return True if placing ``piece`` at (r, c) on a copy of ``board`` wins."""
    test = Board("test", storage=False, rows=4, cols=4)
    for rr in range(4):
        for cc in range(4):
            cell = board.board[rr][cc]
            if isinstance(cell, Piece):
                test.put_piece(cell, rr, cc)
    test.put_piece(piece, r, c)
    won, _ = test.check_win(mode_2x2=True)
    return won


def _winning_cells(
    board: Board, piece: Piece, empties: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    return [(r, c) for (r, c) in empties if _placing_wins(board, piece, r, c)]


def _piece_is_losing(
    board: Board, piece: Piece, empties: list[tuple[int, int]]
) -> bool:
    """A piece is *losing* if placing it in any empty cell wins the opponent."""
    return any(_placing_wins(board, piece, r, c) for (r, c) in empties)


# ──────────────────────────────────────────────────────────────────────
# Model query helpers
# ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def _q_place_single(
    net: torch.nn.Module,
    state_board: np.ndarray,
    state_aux: np.ndarray,
) -> np.ndarray:
    """Return Q_place (shape 16,) for a single PLACE-phase state."""
    device = next(net.parameters()).device
    sb = torch.from_numpy(state_board[None]).to(device)   # (1, 16, 4, 4)
    sa = torch.from_numpy(state_aux[None]).to(device)     # (1, 32)
    q_place, _ = net.forward(sb, sa)
    return q_place[0].cpu().numpy()


@torch.no_grad()
def _q_select_single(
    net: torch.nn.Module,
    state_board: np.ndarray,
    state_aux: np.ndarray,
) -> np.ndarray:
    """Return Q_select (shape 16,) for a single SELECT-phase state."""
    device = next(net.parameters()).device
    sb = torch.from_numpy(state_board[None]).to(device)   # (1, 16, 4, 4)
    sa = torch.from_numpy(state_aux[None]).to(device)     # (1, 32)
    _, q_select = net.forward(sb, sa)
    return q_select[0].cpu().numpy()


def _argmax_legal_cell(
    q_place: np.ndarray, empties: list[tuple[int, int]]
) -> tuple[int, int]:
    flat_idx = max(
        (r * 4 + c for (r, c) in empties),
        key=lambda i: q_place[i],
    )
    return flat_idx // 4, flat_idx % 4


def _argmax_legal_piece(
    q_select: np.ndarray, available: list[Piece]
) -> Piece:
    return max(available, key=lambda p: q_select[p.index()])


# ──────────────────────────────────────────────────────────────────────
# Test A / B / C  (per-position; rely on game simulation)
# ──────────────────────────────────────────────────────────────────────


def run_tests_abc(
    net: torch.nn.Module,
    place_states: list[_PlaceState],
    select_states: list[_SelectState],
) -> dict:
    """Run Tests A (PLACE states), B (SELECT states), C (PLACE states)."""

    # ── Test A ────────────────────────────────────────────────────────
    a_total = a_correct = 0
    for ps in tqdm(place_states, desc="Test A (winning placement)"):
        board, empties = _board_from_encoding(ps.state_board)
        if not empties:
            continue
        offered_vec = ps.state_aux[:16]
        if not offered_vec.any():
            continue
        offered = Piece.from_index(int(np.argmax(offered_vec)))
        win_cells = _winning_cells(board, offered, empties)
        if not win_cells:
            continue
        a_total += 1
        q_place = _q_place_single(net, ps.state_board, ps.state_aux)
        chosen = _argmax_legal_cell(q_place, empties)
        if chosen in win_cells:
            a_correct += 1

    # ── Test B ────────────────────────────────────────────────────────
    # Use SELECT-phase states directly: board is post-placement, aux has
    # available pieces (no offered piece since SELECT is in progress).
    b_total = b_correct = b_forced_loss = 0
    for ss in tqdm(select_states, desc="Test B (losing-piece avoidance)"):
        board, empties = _board_from_encoding(ss.state_board)
        available_mask = ss.state_aux[16:]
        available = [Piece.from_index(i) for i in range(16) if available_mask[i] > 0.5]
        if not available or not empties:
            continue

        losing = [p for p in available if _piece_is_losing(board, p, empties)]
        safe = [p for p in available if p not in losing]

        if losing and not safe:
            b_forced_loss += 1
            continue
        if not losing:
            continue  # No losing pieces — question is vacuous.

        b_total += 1
        q_select = _q_select_single(net, ss.state_board, ss.state_aux)
        chosen_piece = _argmax_legal_piece(q_select, available)
        if chosen_piece not in losing:
            b_correct += 1

    # ── Test C ────────────────────────────────────────────────────────
    c_fractions: list[float] = []
    for ps in tqdm(place_states, desc="Test C (piece sensitivity)"):
        board, empties = _board_from_encoding(ps.state_board)
        if not empties:
            continue
        offered_vec = ps.state_aux[:16]
        available_mask = ps.state_aux[16:]
        available_idxs = [i for i in range(16) if available_mask[i] > 0.5]
        # Candidate piece indices = offered (if any) ∪ available
        if offered_vec.any():
            offered_idx = int(np.argmax(offered_vec))
            candidates = list(dict.fromkeys([offered_idx] + available_idxs))
        else:
            candidates = available_idxs
        if len(candidates) < 2:
            continue
        chosen_cells: set[tuple[int, int]] = set()
        for piece_idx in candidates:
            # Rebuild aux with this candidate as the offered piece.
            alt_aux = ps.state_aux.copy()
            alt_aux[:16] = 0.0
            alt_aux[piece_idx] = 1.0
            q_place = _q_place_single(net, ps.state_board, alt_aux)
            chosen_cells.add(_argmax_legal_cell(q_place, empties))
        c_fractions.append(len(chosen_cells) / len(candidates))

    return {
        "test_A_winning_placement": {
            "n_winnable": a_total,
            "n_correct": a_correct,
            "accuracy": (a_correct / a_total) if a_total else None,
        },
        "test_B_losing_piece_avoidance": {
            "n_auditable": b_total,
            "n_correct": b_correct,
            "accuracy": (b_correct / b_total) if b_total else None,
            "n_forced_loss_skipped": b_forced_loss,
        },
        "test_C_offered_piece_sensitivity": {
            "n_positions": len(c_fractions),
            "mean_distinct_fraction": float(np.mean(c_fractions)) if c_fractions else None,
            "median_distinct_fraction": float(np.median(c_fractions)) if c_fractions else None,
        },
    }


# ──────────────────────────────────────────────────────────────────────
# Tests D / E  (vectorised batch pass over PLACE states)
# ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def run_tests_de(
    net: torch.nn.Module,
    place_states: list[_PlaceState],
    batch_size: int = 1024,
) -> dict:
    """Run Tests D and E — both are vectorisable over the PLACE state set."""
    if not place_states:
        return {
            "test_D_q_occupancy_gap": {"n_positions": 0},
            "test_E_phase_entropy": {"by_phase": {}},
        }

    device = next(net.parameters()).device
    n = len(place_states)

    sb_all = torch.from_numpy(
        np.stack([ps.state_board for ps in place_states])
    ).float()   # (N, 16, 4, 4)
    sa_all = torch.from_numpy(
        np.stack([ps.state_aux for ps in place_states])
    ).float()   # (N, 32)

    occ_flat = sb_all.any(dim=1).view(n, 16)   # (N, 16) bool — occupied cells
    n_pieces = occ_flat.sum(dim=1)              # (N,) long

    qp_all = torch.empty((n, 16), dtype=torch.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        qb, _ = net.forward(sb_all[start:end].to(device), sa_all[start:end].to(device))
        qp_all[start:end] = qb.cpu()

    # ── Test D ────────────────────────────────────────────────────────
    empty_mask = ~occ_flat  # (N, 16)
    q_empty_mean = (qp_all * empty_mask.float()).sum(1) / empty_mask.sum(1).clamp(min=1).float()
    q_occ_mean = (qp_all * occ_flat.float()).sum(1) / occ_flat.sum(1).clamp(min=1).float()
    has_both = occ_flat.any(dim=1) & empty_mask.any(dim=1)
    gap = (q_empty_mean - q_occ_mean)[has_both].numpy()

    test_d = {
        "n_positions": int(has_both.sum().item()),
        "mean_gap": float(gap.mean()),
        "median_gap": float(np.median(gap)),
        "std_gap": float(gap.std()),
        "fraction_positive_gap": float((gap > 0).mean()),
    }

    # ── Test E ────────────────────────────────────────────────────────
    masked_q = qp_all.clone()
    masked_q[occ_flat] = -1e9
    log_probs = F.log_softmax(masked_q, dim=1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=1).numpy()   # (N,) nats
    n_pieces_np = n_pieces.cpu().numpy()
    has_legal = empty_mask.any(dim=1).cpu().numpy()

    buckets = {
        "all": np.ones(n, dtype=bool),
        "early_0_4": n_pieces_np <= 4,
        "mid_5_10": (n_pieces_np >= 5) & (n_pieces_np <= 10),
        "late_11_15": n_pieces_np >= 11,
    }
    by_phase: dict[str, dict] = {}
    for bname, bmask in buckets.items():
        sel = bmask & has_legal
        if not sel.any():
            by_phase[bname] = {"n": 0}
            continue
        e = entropy[sel]
        by_phase[bname] = {
            "n": int(sel.sum()),
            "mean_entropy_nats": float(np.mean(e)),
            "median_entropy_nats": float(np.median(e)),
        }

    test_e = {"by_phase": by_phase}

    # ── Test D' (counterfactual occupancy gap) ───────────────────────
    # (i) Per-piece-count stratification of the original gap.
    n_pieces_np_full = n_pieces.cpu().numpy()
    by_phase_d: dict[str, dict] = {}
    has_both_np = has_both.cpu().numpy()
    gap_full = np.full(n, np.nan, dtype=np.float32)
    gap_full[has_both_np] = gap
    for bname, bmask in {
        "early_0_4": n_pieces_np_full <= 4,
        "mid_5_10": (n_pieces_np_full >= 5) & (n_pieces_np_full <= 10),
        "late_11_15": n_pieces_np_full >= 11,
    }.items():
        sel = bmask & has_both_np
        if not sel.any():
            by_phase_d[bname] = {"n": 0}
            continue
        g = gap_full[sel]
        by_phase_d[bname] = {
            "n": int(sel.sum()),
            "mean_gap": float(np.mean(g)),
            "fraction_positive_gap": float((g > 0).mean()),
        }

    # (ii) Counterfactual: zero all piece channels at occupied cells, leaving
    # aux unchanged. Re-forward and compare Q_place at the *was-occupied*
    # cells against Q_place at the (originally) empty cells. If the trained
    # model has merely been reading off feature magnitude, this gap should
    # collapse to ≈ 0.
    sb_cleared_all = sb_all.clone()
    # Zero out every channel at every occupied (r, c). occ_flat is (N, 16) over
    # the spatial flatten; reshape so we can broadcast to (N, 16_chan, 4, 4).
    occ_spatial = occ_flat.view(n, 1, 4, 4).float()       # (N, 1, 4, 4)
    sb_cleared_all = sb_cleared_all * (1.0 - occ_spatial)  # zero those cells

    qp_cleared = torch.empty((n, 16), dtype=torch.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        qb, _ = net.forward(
            sb_cleared_all[start:end].to(device), sa_all[start:end].to(device)
        )
        qp_cleared[start:end] = qb.cpu()

    # Compare Q at the *originally-occupied* cells (now zeroed in input) vs
    # Q at originally-empty cells, on the cleared forward.  Both regions of
    # the cleared board look identical to the trunk (all-zero channels), so
    # any remaining gap is positional/head bias.
    q_wasocc_mean = (qp_cleared * occ_flat.float()).sum(1) / occ_flat.sum(1).clamp(min=1).float()
    q_empty_cleared_mean = (qp_cleared * empty_mask.float()).sum(1) / empty_mask.sum(1).clamp(min=1).float()
    gap_cleared = (q_wasocc_mean - q_empty_cleared_mean)[has_both].numpy()

    # Diagnostic ratio: how much of the original gap survives the clearing?
    mean_gap_orig = float(np.mean(gap))
    mean_gap_cleared = float(np.mean(gap_cleared))
    if abs(mean_gap_orig) > 1e-6:
        survival_ratio = float(mean_gap_cleared / mean_gap_orig)
    else:
        survival_ratio = None

    test_d_prime = {
        "n_positions": int(has_both.sum().item()),
        "mean_gap_orig": mean_gap_orig,
        "mean_gap_cleared": mean_gap_cleared,
        "std_gap_cleared": float(gap_cleared.std()),
        "fraction_positive_gap_cleared": float((gap_cleared > 0).mean()),
        "survival_ratio": survival_ratio,
        "interpretation_hint": (
            "|survival_ratio| << 1 → input/feature-magnitude driven (artefact); "
            "|survival_ratio| ≈ 1 → position/head bias (real)"
        ),
        "by_piece_count_original_gap": by_phase_d,
    }

    return {
        "test_D_q_occupancy_gap": test_d,
        "test_E_phase_entropy": test_e,
        "test_Dprime_counterfactual_occupancy": test_d_prime,
    }


# ──────────────────────────────────────────────────────────────────────
# Single-experiment driver
# ──────────────────────────────────────────────────────────────────────


def audit_experiment(
    exp_name: str,
    *,
    epoch: int | None = None,
    n_states: int = 2000,
    seed: int = 42,
    batch_size: int = 1024,
    include_epoch_0: bool = False,
    architecture: str | None = None,
) -> list[dict]:
    """Audit one experiment; returns list of JSONL records emitted."""
    print(f"\n{'='*60}")
    print(f"  Auditing: {exp_name}")
    print(f"  epoch={epoch if epoch is not None else 'latest'}  "
          f"n_states={n_states}  seed={seed}")
    print(f"{'='*60}")

    epochs_to_run: list[int | None] = [epoch]  # None = latest
    if include_epoch_0:
        epochs_to_run = [0] + epochs_to_run

    records = []
    for ep in epochs_to_run:
        t0 = time.time()
        net, cfg = load_checkpoint(exp_name, epoch=ep, architecture=architecture)
        resolved_epoch = cfg["epoch"]
        print(f"\n-- epoch {resolved_epoch} ({cfg['architecture']}) --")

        print("Sampling states …")
        place_states, select_states = sample_states(
            net,
            n_states=n_states,
            seed=seed,
        )
        print(f"  PLACE: {len(place_states)}  SELECT: {len(select_states)}")

        print("Running Tests A / B / C …")
        abc = run_tests_abc(net, place_states, select_states)

        print("Running Tests D / E …")
        de = run_tests_de(net, place_states, batch_size=batch_size)

        record = {
            "schema_version": 1,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "exp_name": exp_name,
            "epoch": resolved_epoch,
            "checkpoint_path": cfg["checkpoint_path"],
            "architecture": cfg["architecture"],
            "n_place_states": len(place_states),
            "n_select_states": len(select_states),
            "n_states_requested": n_states,
            "seed": seed,
            "elapsed_seconds": round(time.time() - t0, 1),
            **abc,
            **de,
        }

        out_path = _emit_jsonl(exp_name, record)
        out_path_display = os.path.relpath(out_path, ROOT)
        print(f"  -> {out_path_display}")
        records.append(record)
        _print_summary(record)

    return records


# ──────────────────────────────────────────────────────────────────────
# Console summary table
# ──────────────────────────────────────────────────────────────────────


def _fmt(val) -> str:
    if val is None:
        return "  — "
    if isinstance(val, float):
        return f"{val:.3f}"
    return str(val)


def _print_summary(record: dict) -> None:
    a = record.get("test_A_winning_placement", {})
    b = record.get("test_B_losing_piece_avoidance", {})
    c = record.get("test_C_offered_piece_sensitivity", {})
    d = record.get("test_D_q_occupancy_gap", {})
    dp = record.get("test_Dprime_counterfactual_occupancy", {})
    e_all = record.get("test_E_phase_entropy", {}).get("by_phase", {}).get("all", {})
    e_late = record.get("test_E_phase_entropy", {}).get("by_phase", {}).get("late_11_15", {})

    print(f"\n  Summary for epoch {record['epoch']}:")
    print(f"    A  winning placement  accuracy : {_fmt(a.get('accuracy'))}"
          f"  (n={a.get('n_winnable', 0)})")
    print(f"    B  losing-piece avoid accuracy : {_fmt(b.get('accuracy'))}"
          f"  (n={b.get('n_auditable', 0)})")
    print(f"    C  piece sensitivity (mean frac): {_fmt(c.get('mean_distinct_fraction'))}"
          f"  (n={c.get('n_positions', 0)})")
    print(f"    D  Q-occ gap  (mean / frac>0)  : "
          f"{_fmt(d.get('mean_gap'))} / {_fmt(d.get('fraction_positive_gap'))}")
    print(f"    D' counterfactual gap (cleared): "
          f"{_fmt(dp.get('mean_gap_cleared'))}  "
          f"survival_ratio={_fmt(dp.get('survival_ratio'))}")
    print(f"    E  entropy all / late           : "
          f"{_fmt(e_all.get('mean_entropy_nats'))} / {_fmt(e_late.get('mean_entropy_nats'))}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Competence audit for unified_autoreg checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--exp", metavar="EXP", nargs="+",
                   help="Experiment name(s) under CHECKPOINTS/")
    p.add_argument("--va-champions", action="store_true",
                   help="Run the full Va champion set defined in PLAN.md")
    p.add_argument("--epoch", type=int, default=None,
                   help="Specific epoch to audit (default: latest)")
    p.add_argument("--include-epoch-0", action="store_true",
                   help="Also audit the epoch-0 random-init checkpoint (V0b baseline)")
    p.add_argument("--n-states", type=int, default=2000,
                   help="Target number of PLACE and SELECT states each")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=1024,
                   help="Forward-pass batch size for Tests D / E")
    p.add_argument("--architecture", default=None,
                   help="Override architecture (default: auto-inferred from exp name)")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    exps: list[str] = []
    if args.va_champions:
        exps.extend(VA_CHAMPIONS)
    if args.exp:
        exps.extend(args.exp)
    if not exps:
        parser.error("Provide --exp EXP [EXP ...] or --va-champions")

    # Warn about decoupled experiments.
    for exp in list(exps):
        for skip in VA_DECOUPLED_CHAMPIONS:
            if skip in exp or exp in skip:
                print(
                    f"[SKIP] {exp} — uses decoupled_autoreg schema (32-d→16-d aux)."
                    " A separate audit branch is needed.  Remove from --exp to suppress."
                )
                exps.remove(exp)
                break

    all_records: list[dict] = []
    for exp in exps:
        try:
            recs = audit_experiment(
                exp,
                epoch=args.epoch,
                n_states=args.n_states,
                seed=args.seed,
                batch_size=args.batch_size,
                include_epoch_0=args.include_epoch_0,
                architecture=args.architecture,
            )
            all_records.extend(recs)
        except Exception as exc:
            print(f"[ERROR] {exp}: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Done.  {len(all_records)} record(s) emitted.")


if __name__ == "__main__":
    main()
