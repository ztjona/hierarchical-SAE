# -*- coding: utf-8 -*-
"""D2 — buffer signal-density audit (cheap path).

See ``analysis/qselect_diagnostics/PLAN.md`` → "D2".

Falsifies (or supports) H2: that only a small fraction of replay-buffer
SELECT rows carry a non-trivial oracle target.

Cheap path implementation: regenerates one epoch's worth of unified-autoreg
experience with the trained policy_net as the behaviour policy, using a
MinimaxBot oracle (depth configurable, default 2 to match training).

Usage:
    python analysis/qselect_diagnostics/buffer_signal_density.py \
        --exp 'Ta_minimaxSelect(1)0514_DEPTH_2' [--epoch 4000]
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from analysis.qselect_diagnostics._common import (  # noqa: E402
    MinimaxBot,
    PHASE_SELECT,
    Quarto_unified_bot,
    emit_jsonl,
    gen_experience_unified_autoreg,
    load_checkpoint,
)


EPS_THRESHOLDS = (1e-3, 0.1, 0.5)
HIST_BINS = np.linspace(0.0, 1.0, 21)


def _aggregate(target_block: np.ndarray, mask_block: np.ndarray) -> dict:
    """Compute fraction-above-ε and magnitude histogram on a set of rows.

    ``target_block`` and ``mask_block`` are both ``(N, 16)`` float32.
    """
    n_rows = int(target_block.shape[0])
    if n_rows == 0:
        return {
            "n_select_rows": 0,
            "max_abs_target": [],
            **{f"nonzero_frac_{_label(eps)}": None for eps in EPS_THRESHOLDS},
            "magnitude_hist": [0] * (len(HIST_BINS) - 1),
        }
    masked = target_block * mask_block
    max_abs = np.max(np.abs(masked), axis=1)  # (N,)
    result: dict = {"n_select_rows": n_rows}
    for eps in EPS_THRESHOLDS:
        result[f"nonzero_frac_{_label(eps)}"] = float((max_abs > eps).mean())
    nonzero_vals = np.abs(masked[mask_block > 0.5])
    nonzero_vals = nonzero_vals[nonzero_vals > 0]
    hist, _edges = np.histogram(nonzero_vals, bins=HIST_BINS)
    result["magnitude_hist"] = hist.astype(int).tolist()
    result["magnitude_hist_bin_edges"] = HIST_BINS.tolist()
    result["max_abs_target_mean"] = float(max_abs.mean())
    result["max_abs_target_p50"] = float(np.percentile(max_abs, 50))
    result["max_abs_target_p90"] = float(np.percentile(max_abs, 90))
    return result


def _label(eps: float) -> str:
    if eps == 1e-3:
        return "e3"
    if eps == 0.1:
        return "p1"
    if eps == 0.5:
        return "p5"
    return f"{eps:g}"


def _collect_one_pass(
    *,
    net: torch.nn.Module,
    n_last_states: int,
    n_matches: int,
    mode_2x2: bool,
    temperature: float,
    oracle: MinimaxBot,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    p1 = Quarto_unified_bot(model=net, deterministic=False, temperature=temperature)
    p2 = Quarto_unified_bot(model=net, deterministic=False, temperature=temperature)
    exp = gen_experience_unified_autoreg(
        p1_bot=p1,
        p2_bot=p2,
        n_last_states=n_last_states,
        number_of_matches=n_matches,
        verbose=False,
        PROGRESS_MESSAGE=f"D2 buffer regen ({label}, N={n_last_states})",
        mode_2x2=mode_2x2,
        REWARD_FUNCTION_TYPE="final",
        COLLECT_BOARDS=False,
        select_oracle=oracle,
    )
    phase = exp["phase"].cpu().numpy()
    sel_idx = np.where(phase == PHASE_SELECT)[0]
    tgt = exp["target_sel_minimax"].cpu().numpy()[sel_idx]
    msk = exp["target_sel_minimax_mask"].cpu().numpy()[sel_idx]
    return tgt, msk


def run(
    exp_name: str,
    epoch: int | None,
    n_matches: int,
    n_last_states_curriculum: int,
    n_last_states_endgame: int,
    mode_2x2: bool,
    oracle_depth: int,
    temperature: float,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    net, cfg = load_checkpoint(exp_name, epoch=epoch)
    oracle = MinimaxBot(depth=oracle_depth)

    curric_tgt, curric_msk = _collect_one_pass(
        net=net,
        n_last_states=n_last_states_curriculum,
        n_matches=n_matches,
        mode_2x2=mode_2x2,
        temperature=temperature,
        oracle=oracle,
        label="curriculum",
    )
    endgame_tgt, endgame_msk = _collect_one_pass(
        net=net,
        n_last_states=n_last_states_endgame,
        n_matches=n_matches,
        mode_2x2=mode_2x2,
        temperature=temperature,
        oracle=oracle,
        label="endgame",
    )

    full_tgt = np.concatenate([curric_tgt, endgame_tgt], axis=0)
    full_msk = np.concatenate([curric_msk, endgame_msk], axis=0)

    record = {
        "diagnostic": "buffer_signal_density",
        "schema_version": 1,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "exp_name": cfg["exp_name"],
        "epoch": cfg["epoch"],
        "architecture": cfg["architecture"],
        "checkpoint_path": cfg["checkpoint_path"],
        "source": "cheap",
        "config": {
            "n_matches_per_bucket": n_matches,
            "n_last_states_curriculum": n_last_states_curriculum,
            "n_last_states_endgame": n_last_states_endgame,
            "mode_2x2": mode_2x2,
            "oracle_depth": oracle_depth,
            "temperature": temperature,
            "seed": seed,
            "eps_thresholds": list(EPS_THRESHOLDS),
        },
        "overall": _aggregate(full_tgt, full_msk),
        "by_n_last_states": {
            str(n_last_states_curriculum): _aggregate(curric_tgt, curric_msk),
            str(n_last_states_endgame): _aggregate(endgame_tgt, endgame_msk),
        },
    }
    out = emit_jsonl(cfg["exp_name"], "buffer_signal_density", record)
    print(f"Wrote {out}")
    overall = record["overall"]
    print(f"  total_select_rows         = {overall['n_select_rows']}")
    print(f"  nonzero_frac_e3 (overall) = {overall['nonzero_frac_e3']}")
    print(f"  nonzero_frac_p1 (overall) = {overall['nonzero_frac_p1']}")
    print(f"  nonzero_frac_p5 (overall) = {overall['nonzero_frac_p5']}")
    for bucket, stats in record["by_n_last_states"].items():
        print(
            f"  N={bucket}: n={stats['n_select_rows']}  "
            f"frac>p1={stats['nonzero_frac_p1']}  "
            f"frac>p5={stats['nonzero_frac_p5']}"
        )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--n-matches", type=int, default=32,
                        help="matches per bucket — matches MATCHES_PER_EPOCH default")
    parser.add_argument("--n-last-states-curriculum", type=int, default=4)
    parser.add_argument("--n-last-states-endgame", type=int, default=2)
    parser.add_argument("--no-2x2", action="store_true")
    parser.add_argument("--oracle-depth", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="behaviour-policy temperature (TEMPERATURE_EXPLORE default)")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    run(
        exp_name=args.exp,
        epoch=args.epoch,
        n_matches=args.n_matches,
        n_last_states_curriculum=args.n_last_states_curriculum,
        n_last_states_endgame=args.n_last_states_endgame,
        mode_2x2=not args.no_2x2,
        oracle_depth=args.oracle_depth,
        temperature=args.temperature,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
