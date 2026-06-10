# -*- coding: utf-8 -*-
"""Shared helpers for the Q_select diagnostic suite.

See ``analysis/qselect_diagnostics/PLAN.md`` for context and the rationale
behind exposing exactly these three primitives. Keep convention bugs
quarantined here — every diagnostic depends on this module reusing the
training-time conventions verbatim.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bot.CNN_unified_bot import Quarto_bot as Quarto_unified_bot  # noqa: E402
from bot.minimax_bot import MinimaxBot  # noqa: E402
from models.CNN_autoreg import (  # noqa: E402
    QuartoCNNAutoregUnified,
    QuartoCNNAutoregUnifiedUnbound,
)
from models.CNN_autoreg_sa import (  # noqa: E402
    QuartoCNNAutoregUnifiedS1,
    QuartoCNNAutoregUnifiedS2,
    QuartoCNNAutoregUnifiedS4,
    QuartoCNNAutoregUnifiedS4Hot,
)
from QuartoRL.RL_functions import (  # noqa: E402
    PHASE_PLACE,
    PHASE_SELECT,
    UNIFIED_AUX_DIM,
    _minimax_select_target,
    gen_experience_unified_autoreg,
)


CHECKPOINTS_DIR = os.path.join(ROOT, "CHECKPOINTS")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ──────────────────────────────────────────────────────────────────────
# Checkpoint discovery & loading
# ──────────────────────────────────────────────────────────────────────


_EPOCH_RE = re.compile(r"_E_(\d+)\.pt$")


def _list_checkpoints(exp_name: str) -> list[tuple[int, str]]:
    folder = os.path.join(CHECKPOINTS_DIR, exp_name)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"No CHECKPOINTS folder for experiment: {exp_name}")
    out: list[tuple[int, str]] = []
    for p in glob.glob(os.path.join(folder, "*.pt")):
        m = _EPOCH_RE.search(p)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda t: t[0])
    return out


_ARCHITECTURE_REGISTRY: dict[str, type] = {
    "QuartoCNNAutoregUnified": QuartoCNNAutoregUnified,
    "QuartoCNNAutoregUnifiedUnbound": QuartoCNNAutoregUnifiedUnbound,
    "QuartoCNNAutoregUnifiedS1": QuartoCNNAutoregUnifiedS1,
    "QuartoCNNAutoregUnifiedS2": QuartoCNNAutoregUnifiedS2,
    "QuartoCNNAutoregUnifiedS4": QuartoCNNAutoregUnifiedS4,
    "QuartoCNNAutoregUnifiedS4Hot": QuartoCNNAutoregUnifiedS4Hot,
}


# Map experiment-name prefix → architecture used. T-series ran on the Sa(3)
# substrate (S4). Extend as new series come online.
_EXP_ARCH_HINTS: list[tuple[str, str]] = [
    ("Ta_minimaxSelect", "QuartoCNNAutoregUnifiedS4"),
    ("Ve_oracleAblation", "QuartoCNNAutoregUnifiedS4"),
    ("S4", "QuartoCNNAutoregUnifiedS4"),
    ("S1", "QuartoCNNAutoregUnifiedS1"),
    ("S2", "QuartoCNNAutoregUnifiedS2"),
]


def infer_architecture(exp_name: str) -> str:
    for prefix, arch in _EXP_ARCH_HINTS:
        if prefix in exp_name:
            return arch
    return "QuartoCNNAutoregUnified"


def load_checkpoint(
    exp_name: str,
    epoch: int | None = None,
    architecture: str | None = None,
) -> tuple[torch.nn.Module, dict]:
    """Load a unified-autoreg policy network in eval mode.

    Returns ``(net, config)`` where ``config`` carries the resolved path,
    epoch, and architecture name. The caller is responsible for the schema
    triplet contract (see ``CLAUDE.md``): this loader assumes the unified
    autoreg model class.
    """
    if architecture is None:
        architecture = infer_architecture(exp_name)
    if architecture not in _ARCHITECTURE_REGISTRY:
        raise ValueError(
            f"Unknown architecture {architecture!r}. "
            f"Known: {sorted(_ARCHITECTURE_REGISTRY)}"
        )
    cls = _ARCHITECTURE_REGISTRY[architecture]

    ckpts = _list_checkpoints(exp_name)
    if not ckpts:
        raise FileNotFoundError(f"No .pt checkpoints under CHECKPOINTS/{exp_name}/")
    if epoch is None:
        chosen_epoch, chosen_path = ckpts[-1]
    else:
        match = [c for c in ckpts if c[0] == epoch]
        if not match:
            available = ", ".join(str(c[0]) for c in ckpts[:5]) + " ..."
            raise FileNotFoundError(
                f"Epoch {epoch} not found for {exp_name}. Have: {available}"
            )
        chosen_epoch, chosen_path = match[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = cls()
    state = torch.load(chosen_path, map_location=device, weights_only=True)
    net.load_state_dict(state)
    net.to(device)
    net.eval()

    config = {
        "exp_name": exp_name,
        "epoch": chosen_epoch,
        "checkpoint_path": os.path.relpath(chosen_path, ROOT),
        "architecture": architecture,
        "device": str(device),
    }
    return net, config


# ──────────────────────────────────────────────────────────────────────
# Self-play sampling
# ──────────────────────────────────────────────────────────────────────


@dataclass
class SelectState:
    """A SELECT-phase snapshot ready for Q_select evaluation."""

    state_board: np.ndarray  # (16, 4, 4) float32 — already deserialized
    state_aux: np.ndarray  # (32,) float32 — unified aux
    available_mask: np.ndarray  # (16,) float32 — legal piece mask (==valid_mask)
    target_sel_minimax: np.ndarray  # (16,) float32 — oracle target (higher=better)
    target_sel_minimax_mask: np.ndarray  # (16,) float32 — same as available_mask
    action_taken: int  # piece index actually selected (for reference)
    n_pieces_on_board: int  # for filtering / bucketing
    n_last_states_bucket: int  # which experience bucket this row came from
    metadata: dict = field(default_factory=dict)


def sample_states(
    *,
    net: torch.nn.Module,
    n_states: int = 500,
    n_pieces_min: int = 2,
    mode_2x2: bool = True,
    n_last_states: int = 4,
    oracle_depth: int = 2,
    number_of_matches: int | None = None,
    temperature: float = 0.5,
    deterministic: bool = False,
    require_nonzero_oracle: bool = False,
    seed: int | None = 1234,
) -> list[SelectState]:
    """Generate a held-out evaluation set of SELECT-phase states.

    Self-plays the supplied ``net`` against itself via
    ``gen_experience_unified_autoreg`` (with a MinimaxBot oracle wired in so
    we get oracle targets for free), then filters to SELECT rows.

    The "n_pieces_min" filter is applied to the number of placed pieces on
    the state board — states with fewer placed pieces are usually too early
    for any piece to be forcing.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    p1 = Quarto_unified_bot(model=net, deterministic=deterministic, temperature=temperature)
    p2 = Quarto_unified_bot(model=net, deterministic=deterministic, temperature=temperature)
    oracle = MinimaxBot(depth=oracle_depth)

    if number_of_matches is None:
        # Each match contributes ~n_last_states SELECT rows pre-filter.
        # Overshoot 4× to leave room for filtering.
        per_match = max(1, n_last_states)
        number_of_matches = max(32, 4 * n_states // per_match)

    exp = gen_experience_unified_autoreg(
        p1_bot=p1,
        p2_bot=p2,
        n_last_states=n_last_states,
        number_of_matches=number_of_matches,
        verbose=False,
        PROGRESS_MESSAGE="Sampling SELECT states",
        mode_2x2=mode_2x2,
        REWARD_FUNCTION_TYPE="final",
        COLLECT_BOARDS=False,
        select_oracle=oracle,
    )

    phase = exp["phase"].cpu().numpy()
    select_idx = np.where(phase == PHASE_SELECT)[0]

    out: list[SelectState] = []
    sb = exp["state_board"].cpu().numpy()
    sa = exp["state_aux"].cpu().numpy()
    vm = exp["valid_mask"].cpu().numpy()
    tgt = exp["target_sel_minimax"].cpu().numpy()
    tmask = exp["target_sel_minimax_mask"].cpu().numpy()
    act = exp["action"].cpu().numpy()

    for i in select_idx:
        # Number of placed pieces = 16 minus available (off-diagonal of board encoding
        # is the storage; use available_mask sum from aux which is the cleaner proxy).
        available_block = sa[i, 16:]
        n_available = int(available_block.sum())
        n_placed = 16 - n_available
        if n_placed < n_pieces_min:
            continue
        if require_nonzero_oracle:
            masked = tgt[i] * tmask[i]
            if not np.any(np.abs(masked) > 1e-6):
                continue
        out.append(
            SelectState(
                state_board=sb[i].astype(np.float32),
                state_aux=sa[i].astype(np.float32),
                available_mask=vm[i].astype(np.float32),
                target_sel_minimax=tgt[i].astype(np.float32),
                target_sel_minimax_mask=tmask[i].astype(np.float32),
                action_taken=int(act[i]),
                n_pieces_on_board=n_placed,
                n_last_states_bucket=int(n_last_states),
            )
        )
        if len(out) >= n_states:
            break
    return out


# ──────────────────────────────────────────────────────────────────────
# Oracle scoring (standalone — for D3 etc.)
# ──────────────────────────────────────────────────────────────────────


def oracle_scores(
    *,
    oracle: MinimaxBot,
    board_serial: str,
    available_pieces: Iterable[int],
    mode_2x2: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Standalone wrapper around ``_minimax_select_target``.

    Returns ``(target_16d, mask_16d)`` with the same ``(100 + depth)``
    normalisation and sign-flip convention used in training.
    """
    return _minimax_select_target(
        oracle,
        board_serial,
        set(int(p) for p in available_pieces),
        mode_2x2=mode_2x2,
    )


# ──────────────────────────────────────────────────────────────────────
# Q_select forward pass over a batch of SelectState
# ──────────────────────────────────────────────────────────────────────


def qselect_predict(
    net: torch.nn.Module,
    states: list[SelectState],
    batch_size: int = 256,
) -> np.ndarray:
    """Run Q_select on every state. Returns ``(N, 16)`` float32 ndarray."""
    if not states:
        return np.zeros((0, 16), dtype=np.float32)
    device = next(net.parameters()).device
    out_chunks: list[np.ndarray] = []
    net.eval()
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            chunk = states[start : start + batch_size]
            sb = torch.from_numpy(np.stack([s.state_board for s in chunk])).to(device)
            sa = torch.from_numpy(np.stack([s.state_aux for s in chunk])).to(device)
            phase = torch.full((len(chunk),), PHASE_SELECT, dtype=torch.long, device=device)
            _q_place, q_select = net.forward(sb, sa, phase=phase)
            out_chunks.append(q_select.cpu().numpy().astype(np.float32))
    return np.concatenate(out_chunks, axis=0)


# ──────────────────────────────────────────────────────────────────────
# JSONL output
# ──────────────────────────────────────────────────────────────────────


def emit_jsonl(exp_name: str, diagnostic: str, record: dict) -> str:
    """Append one record to ``results/<exp_name>/<diagnostic>.jsonl``."""
    out_dir = os.path.join(RESULTS_DIR, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{diagnostic}.jsonl")
    with open(out_path, "a") as f:
        f.write(json.dumps(record, default=_json_default) + "\n")
    return out_path


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON-serializable: {type(obj)}")


__all__ = [
    "PHASE_PLACE",
    "PHASE_SELECT",
    "UNIFIED_AUX_DIM",
    "SelectState",
    "load_checkpoint",
    "sample_states",
    "oracle_scores",
    "qselect_predict",
    "emit_jsonl",
    "MinimaxBot",
]
