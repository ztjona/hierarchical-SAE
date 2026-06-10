# -*- coding: utf-8 -*-
"""Oracle-target audit — is the SELECT bottleneck a *target* failure or a *fit* failure?

The unified_autoreg champions are trained with a depth-2 minimax oracle target
on the full Q_select vector (``td_place_minimax_select``), yet they still hand
over an immediately-losing piece ~16% of the time on decisive states (audit.py
Test B) — and ~12% of games are avoidable SELECT blunders against a punishing
opponent (see ``METRICS.md``). This script decides *why*, cheaply, before we
spend a training run:

* **Target failure** — the minimax target itself fails to rank the hot (losing)
  piece below the safe ones on these states. Then the fix is the *target*.
* **Fit failure** — the target is correct (ranks hot lowest) but the model's
  ``Q_select`` deviates from it on exactly the states it blunders. Then the fix
  is *coverage / signal* (punishing-opponent rollouts, an aux hinge, more
  training), not the target.

Method. Self-play–sample SELECT states (reusing ``audit.py``), keep the Test-B
*decisive* ones (both safe and losing pieces available), and for each compute
both (a) the model's ``Q_select`` argmax and (b) the **actual training target**
via ``QuartoRL.RL_functions._minimax_select_target`` (the same MinimaxBot the
trainer uses). Aggregate where they agree/disagree and the regression residual
``|Q_select − target|`` on blunder vs non-blunder states.

Output
------
One JSONL record per (exp, epoch) appended to:
    analysis/competence_audit/results/<exp>/oracle_target_audit.jsonl

Usage
-----
    python analysis/competence_audit/oracle_target_audit.py \
        --exp 'Xa_levers(1)0604_PLACE_WIN' --epoch 6000 \
        --architecture QuartoCNNAutoregUnifiedS4 \
        [--n-states 1500] [--max-oracle 600] [--depth 2] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(ROOT, "analysis", "qselect_diagnostics"))

from quartopy import Piece  # noqa: E402
from bot.minimax_bot import MinimaxBot  # noqa: E402
from QuartoRL.RL_functions import _minimax_select_target  # noqa: E402
from _common import load_checkpoint  # noqa: E402
from audit import (  # noqa: E402
    sample_states,
    _board_from_encoding,
    _q_select_single,
    _piece_is_losing,
)

RESULTS_DIR = os.path.join(THIS_DIR, "results")
MAX_EXAMPLES = 15


def _decisive_select_states(select_states):
    """Yield (ss, board, empties, available_idx, losing_idx, safe_idx) for the
    Test-B-decisive states (both a safe and a losing give exist)."""
    for ss in select_states:
        board, empties = _board_from_encoding(ss.state_board)
        if not empties:
            continue
        avail = [Piece.from_index(i) for i in range(16) if ss.state_aux[16 + i] > 0.5]
        if not avail:
            continue
        losing = {p.index() for p in avail if _piece_is_losing(board, p, empties)}
        if not losing:
            continue                       # vacuous — no hot piece to avoid
        safe = {p.index() for p in avail} - losing
        if not safe:
            continue                       # forced — no safe give exists
        yield ss, board, [p.index() for p in avail], losing, safe


def audit_oracle_target(
    exp_name: str,
    *,
    epoch: int | None,
    architecture: str | None,
    n_states: int,
    max_oracle: int,
    depth: int,
    seed: int,
) -> dict:
    t0 = time.time()
    net, cfg = load_checkpoint(exp_name, epoch=epoch, architecture=architecture)
    _, select_states = sample_states(net, n_states=n_states, seed=seed)
    oracle = MinimaxBot(depth=depth)

    print(f"\n{'='*64}")
    print(f"  Oracle-target audit: {exp_name}  (epoch {cfg['epoch']}, {cfg['architecture']})")
    print(f"  select states sampled: {len(select_states)}   minimax depth: {depth}")
    print(f"{'='*64}")

    agg = {
        "n_decisive": 0,
        "n_model_blunder": 0,        # model argmax is a hot piece (Test-B failure)
        "n_oracle_blunder": 0,       # TARGET argmax is a hot piece (target failure)
        "n_oracle_separates": 0,     # target ranks ALL hot below ALL safe
        "resid_all": [], "resid_blunder": [], "resid_correct": [],
        # on model-blunder states:
        "t_model_pick": [],          # target value the oracle gives the chosen hot piece
        "target_margin_lost": [],    # best-safe target − chosen-hot target
        "q_margin_for_hot": [],      # model q(chosen hot) − model q(best safe): how strongly it prefers the blunder
        "oracle_agrees_hot_is_worst_on_blunder": 0,  # on a model blunder, did target rank that hot piece below all safe?
    }
    examples: list[dict] = []
    n_oracle = 0

    for ss, board, avail_idx, losing_idx, safe_idx in _decisive_select_states(select_states):
        if n_oracle >= max_oracle:
            break
        q = _q_select_single(net, ss.state_board, ss.state_aux)
        target, mask = _minimax_select_target(
            oracle, board.serialize(), set(avail_idx), mode_2x2=True
        )
        masked = [i for i in avail_idx if mask[i] > 0.5]
        if not masked or not (losing_idx & set(masked)) or not (safe_idx & set(masked)):
            continue
        n_oracle += 1
        agg["n_decisive"] += 1

        model_pick = max(avail_idx, key=lambda i: q[i])
        oracle_pick = max(masked, key=lambda i: target[i])
        model_blunder = model_pick in losing_idx
        oracle_blunder = oracle_pick in losing_idx

        hot_masked = [i for i in masked if i in losing_idx]
        safe_masked = [i for i in masked if i in safe_idx]
        separates = max(target[i] for i in hot_masked) < min(target[i] for i in safe_masked)

        resid = float(np.mean([abs(q[i] - target[i]) for i in masked]))
        agg["resid_all"].append(resid)
        agg["n_model_blunder"] += int(model_blunder)
        agg["n_oracle_blunder"] += int(oracle_blunder)
        agg["n_oracle_separates"] += int(separates)

        if model_blunder:
            agg["resid_blunder"].append(resid)
            best_safe_t = max(target[i] for i in safe_masked)
            best_safe_q = max(q[i] for i in safe_masked)
            agg["t_model_pick"].append(float(target[model_pick]))
            agg["target_margin_lost"].append(float(best_safe_t - target[model_pick]))
            agg["q_margin_for_hot"].append(float(q[model_pick] - best_safe_q))
            agg["oracle_agrees_hot_is_worst_on_blunder"] += int(
                target[model_pick] < min(target[i] for i in safe_masked)
            )
            if len(examples) < MAX_EXAMPLES:
                examples.append({
                    "n_pieces": ss.n_pieces_on_board,
                    "model_pick": int(model_pick), "oracle_pick": int(oracle_pick),
                    "q_model_pick": round(float(q[model_pick]), 3),
                    "t_model_pick": round(float(target[model_pick]), 3),
                    "best_safe_target": round(float(best_safe_t), 3),
                    "resid": round(resid, 3),
                })
        else:
            agg["resid_correct"].append(resid)

    n = max(1, agg["n_decisive"])
    nb = max(1, agg["n_model_blunder"])

    def _m(xs):
        return float(np.mean(xs)) if xs else None

    record = {
        "schema_version": 1,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "exp_name": exp_name,
        "epoch": cfg["epoch"],
        "checkpoint_path": cfg["checkpoint_path"],
        "architecture": cfg["architecture"],
        "minimax_depth": depth,
        "seed": seed,
        "elapsed_seconds": round(time.time() - t0, 1),
        "n_decisive": agg["n_decisive"],
        # --- TARGET correctness (is the oracle target right?) ---
        "oracle_blunder_rate": agg["n_oracle_blunder"] / n,       # target's argmax is hot
        "oracle_separates_rate": agg["n_oracle_separates"] / n,   # target ranks all hot < all safe
        # --- MODEL behaviour (does it fit the target?) ---
        "model_blunder_rate": agg["n_model_blunder"] / n,         # ≈ Test-B (1−accuracy)
        "resid_mean_all": _m(agg["resid_all"]),
        "resid_mean_on_blunder": _m(agg["resid_blunder"]),
        "resid_mean_on_correct": _m(agg["resid_correct"]),
        # --- on the states the model blunders ---
        "on_blunder": {
            "oracle_agrees_hot_is_worst_rate": agg["oracle_agrees_hot_is_worst_on_blunder"] / nb,
            "mean_target_of_chosen_hot": _m(agg["t_model_pick"]),
            "mean_target_margin_lost": _m(agg["target_margin_lost"]),
            "mean_model_q_margin_for_hot": _m(agg["q_margin_for_hot"]),
        },
        "examples": examples,
    }
    _emit(exp_name, record)
    _print_summary(record)
    return record


def _emit(exp_name: str, record: dict) -> str:
    out_dir = os.path.join(RESULTS_DIR, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "oracle_target_audit.jsonl")
    with open(out_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return out_path


def _pct(x):
    return f"{100*x:.1f}%" if isinstance(x, float) else " — "


def _print_summary(rec: dict) -> None:
    ob = rec["on_blunder"]
    print(f"\n  decisive SELECT states scored: {rec['n_decisive']}")
    print(f"  TARGET correctness:")
    print(f"    oracle_separates_rate (target ranks all hot < all safe): {_pct(rec['oracle_separates_rate'])}")
    print(f"    oracle_blunder_rate   (target's own argmax is a hot give): {_pct(rec['oracle_blunder_rate'])}")
    print(f"  MODEL fit:")
    print(f"    model_blunder_rate (≈ 1 − Test-B): {_pct(rec['model_blunder_rate'])}")
    print(f"    residual |q−target|  all/correct/blunder: "
          f"{rec['resid_mean_all']:.3f} / {rec['resid_mean_on_correct']:.3f} / {rec['resid_mean_on_blunder']:.3f}")
    print(f"  ON THE MODEL'S BLUNDER STATES:")
    print(f"    target agrees the chosen piece is the worst: {_pct(ob['oracle_agrees_hot_is_worst_rate'])}")
    print(f"    mean target of the chosen hot piece: {ob['mean_target_of_chosen_hot']:.3f}  "
          f"(target margin thrown away: {ob['mean_target_margin_lost']:.3f})")
    print(f"    model's own q-margin for the hot pick over best safe: {ob['mean_model_q_margin_for_hot']:.3f}")
    # one-line verdict
    target_ok = rec["oracle_separates_rate"] > 0.9 and rec["oracle_blunder_rate"] < 0.05
    fit_gap = (rec["resid_mean_on_blunder"] or 0) > 1.3 * (rec["resid_mean_on_correct"] or 1e9)
    verdict = ("FIT failure (target correct, model deviates on blunder states)"
               if target_ok and fit_gap else
               "TARGET failure (oracle target itself mis-ranks hot pieces)"
               if not target_ok else
               "FIT failure (target correct; residual not concentrated — coverage/optimisation)")
    print(f"  → likely {verdict}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Oracle-target audit for unified_autoreg SELECT heads.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--exp", required=True, nargs="+", help="Experiment name(s) under CHECKPOINTS/")
    p.add_argument("--epoch", type=int, default=None)
    p.add_argument("--architecture", default=None)
    p.add_argument("--n-states", type=int, default=1500, help="SELECT states to self-play sample")
    p.add_argument("--max-oracle", type=int, default=600, help="Cap on decisive states scored by the oracle")
    p.add_argument("--depth", type=int, default=2, help="MinimaxBot depth (match MINIMAX_SELECT_DEPTH)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    for exp in args.exp:
        try:
            audit_oracle_target(
                exp, epoch=args.epoch, architecture=args.architecture,
                n_states=args.n_states, max_oracle=args.max_oracle,
                depth=args.depth, seed=args.seed,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {exp}: {e}")


if __name__ == "__main__":
    main()
