# -*- coding: utf-8 -*-
"""Generate JSONL summaries from legacy ``<exp>.pkl`` result files.

The new training pipeline writes ``<EXPERIMENT_NAME>.summary.jsonl`` under
``results/<series>/``. For experiments completed before that change, this
helper re-reads the pickle (which still lives in ``CHECKPOINTS/<exp>/``)
and emits a single ``type=final`` JSONL record so the results-compare CLI
can list them uniformly.

Usage:
    python tools/pkl_to_jsonl.py                  # backfill every legacy pickle
    python tools/pkl_to_jsonl.py <exp1> <exp2>    # specific experiments
    python tools/pkl_to_jsonl.py --force          # overwrite existing JSONL
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from os import path

sys.path.insert(0, path.join(path.dirname(__file__), ".."))

from QuartoRL.results_io import (  # noqa: E402
    SUMMARY_SUFFIX,
    build_final_record,
    load_pickle_results,
    write_records,
)


REPO_ROOT = path.normpath(path.join(path.dirname(__file__), ".."))
CHECKPOINTS_DIR = path.join(REPO_ROOT, "CHECKPOINTS")
RESULTS_DIR = path.join(REPO_ROOT, "results")


def _series_root(exp: str) -> str:
    return re.split(r"\(", exp, maxsplit=1)[0] or exp


def backfill_one(pkl_path: str, force: bool = False) -> str:
    exp = path.basename(path.dirname(pkl_path))
    series_dir = path.join(RESULTS_DIR, _series_root(exp))
    jsonl_path = path.join(series_dir, f"{exp}{SUMMARY_SUFFIX}")
    if path.exists(jsonl_path) and not force:
        return f"[skip] {exp} (JSONL exists)"
    try:
        d = load_pickle_results(pkl_path)
    except Exception as e:  # pragma: no cover
        return f"[error] {exp}: {e}"
    wr = d.get("win_rate", {}) or {}
    epochs = max((len(v) for v in wr.values()), default=0)
    final_rec = build_final_record(
        exp_name=exp,
        epochs=epochs,
        config={"_backfilled_from_pkl": True},
        loss_data=d.get("loss_values", {}) or {},
        grad_norm_data=d.get("grad_norm_data", {}) or {},
        win_rate=wr,
        q_values_history=d.get("q_values_history", {}) or {},
    )
    write_records(jsonl_path, [final_rec])
    return f"[ok]   {exp} -> {path.relpath(jsonl_path, REPO_ROOT)}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiments", nargs="*", help="experiment folder names")
    p.add_argument("--force", action="store_true", help="overwrite existing JSONL")
    args = p.parse_args()

    if args.experiments:
        pkls = []
        for exp in args.experiments:
            cand = glob.glob(path.join(CHECKPOINTS_DIR, exp, "*.pkl"))
            if not cand:
                print(f"[warn] no pickle for {exp}")
                continue
            pkls.extend(cand)
    else:
        pkls = sorted(glob.glob(path.join(CHECKPOINTS_DIR, "*", "*.pkl")))

    for pkl in pkls:
        print(backfill_one(pkl, force=args.force))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
