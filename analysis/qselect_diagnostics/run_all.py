#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the Q_select diagnostic suite end-to-end on one experiment.

Order (per PLAN.md): D1 → D2 → D3 (D3 only if --include-d3 is passed).

D1 and D2 are cheap (minutes). D3 trains a small network for several
thousand epochs over the collected select rows (~tens of minutes on CPU,
much faster on GPU).

Usage:
    ./analysis/qselect_diagnostics/run_all.py \
        --exp 'Ta_minimaxSelect(1)0514_DEPTH_2'

To run D3 too:
    ./analysis/qselect_diagnostics/run_all.py \
        --exp 'Ta_minimaxSelect(1)0514_DEPTH_2' --include-d3

JSONL records land under analysis/qselect_diagnostics/results/<exp>/, and
the console output is also tee'd to results/<exp>/run_all.log.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from analysis.qselect_diagnostics import position_structure  # noqa: E402
from analysis.qselect_diagnostics import buffer_signal_density  # noqa: E402
from analysis.qselect_diagnostics import decoupled_select  # noqa: E402
from analysis.qselect_diagnostics._common import RESULTS_DIR  # noqa: E402


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


@contextmanager
def tee_stdout(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    f = open(log_path, "a")
    f.write(f"\n========== run_all started {datetime.utcnow().isoformat()}Z ==========\n")
    f.flush()
    real_stdout = sys.stdout
    sys.stdout = Tee(real_stdout, f)
    try:
        yield
    finally:
        sys.stdout = real_stdout
        f.write(f"========== run_all finished {datetime.utcnow().isoformat()}Z ==========\n")
        f.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True, help="Experiment folder under CHECKPOINTS/")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--n-states", type=int, default=500, help="D1 sample size")
    parser.add_argument("--n-matches", type=int, default=32, help="D2 matches per bucket")
    parser.add_argument("--include-d3", action="store_true", help="Also run D3 (slow)")
    parser.add_argument("--d3-epochs", type=int, default=2000)
    parser.add_argument("--d3-n-matches", type=int, default=64)
    parser.add_argument("--oracle-depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    log_path = os.path.join(RESULTS_DIR, args.exp, "run_all.log")
    with tee_stdout(log_path):
        print(f"# exp = {args.exp}")
        print(f"# epoch = {args.epoch if args.epoch is not None else 'latest'}")
        print(f"# include_d3 = {args.include_d3}")
        print("")

        # ---- D1 ----
        print("=" * 60)
        print("D1 — position_structure")
        print("=" * 60)
        t = time.time()
        position_structure.run(
            exp_name=args.exp,
            epoch=args.epoch,
            n_states=args.n_states,
            n_pieces_min=2,
            n_last_states=4,
            oracle_depth=args.oracle_depth,
            seed=args.seed,
        )
        print(f"[D1 took {time.time() - t:.1f}s]\n")

        # ---- D2 ----
        print("=" * 60)
        print("D2 — buffer_signal_density")
        print("=" * 60)
        t = time.time()
        buffer_signal_density.run(
            exp_name=args.exp,
            epoch=args.epoch,
            n_matches=args.n_matches,
            n_last_states_curriculum=4,
            n_last_states_endgame=2,
            mode_2x2=True,
            oracle_depth=args.oracle_depth,
            temperature=2.0,
            seed=args.seed,
        )
        print(f"[D2 took {time.time() - t:.1f}s]\n")

        # ---- D3 ----
        if args.include_d3:
            print("=" * 60)
            print("D3 — decoupled_select")
            print("=" * 60)
            t = time.time()
            decoupled_select.run(
                exp_name=args.exp,
                epoch=args.epoch,
                n_matches=args.d3_n_matches,
                n_matches_eval=args.d3_n_matches,
                train_epochs=args.d3_epochs,
                batch_size=64,
                lr=3e-4,
                n_last_states_curriculum=4,
                n_last_states_endgame=2,
                mode_2x2=True,
                oracle_depth=args.oracle_depth,
                temperature=2.0,
                seed=args.seed,
            )
            print(f"[D3 took {time.time() - t:.1f}s]\n")
        else:
            print("(D3 skipped — pass --include-d3 to enable)\n")


if __name__ == "__main__":
    main()
