# -*- coding: utf-8 -*-
"""Compare training runs from their JSONL summary files.

Source of truth: ``results/<series>/<exp>.summary.jsonl`` (see
``QuartoRL/results_io.py``). If the JSONL is absent, falls back to
``<exp>.pkl`` for backwards compatibility — but please backfill with
``tools/pkl_to_jsonl.py`` so the CLI is fast.

Subcommands
-----------
- ``list``                List every experiment, one row each, sortable.
- ``show <exp>``          Pretty-print the final record of one experiment.
- ``diff <a> <b>``        Side-by-side config + metrics for two experiments.
- ``rank --metric KEY``   Leaderboard sorted by a chosen metric.
- ``gates <exp>``         Pass/fail of pre-registered gates (best-effort).

All commands accept ``--filter SUBSTR`` to limit to experiments whose name
contains the substring.

Usage:
    python tools/results_compare.py list [--filter SUBSTR] [--sort KEY]
    python tools/results_compare.py show <exp>
    python tools/results_compare.py diff <a> <b>
    python tools/results_compare.py rank --metric KEY [--top N] [--filter SUBSTR]
    python tools/results_compare.py gates <exp>
"""
from __future__ import annotations

import argparse
import glob
import sys
from os import path
from typing import Any

sys.path.insert(0, path.join(path.dirname(__file__), ".."))

from QuartoRL.results_io import (  # noqa: E402
    SUMMARY_SUFFIX,
    build_final_record,
    final_record,
    load_pickle_results,
    read_records,
)

CHECKPOINTS_DIR = path.join(path.dirname(__file__), "..", "CHECKPOINTS")
RESULTS_DIR = path.join(path.dirname(__file__), "..", "results")


# ---------- discovery ----------


def _series_root(exp: str) -> str:
    import re
    return re.split(r"\(", exp, maxsplit=1)[0] or exp


def discover_experiments(filter_substr: str | None = None) -> list[str]:
    """Return a sorted list of experiment names with either a JSONL summary
    (under ``results/<series>/``) or a legacy pickle (under ``CHECKPOINTS/<exp>/``)."""
    found: set[str] = set()
    for p in glob.glob(path.join(RESULTS_DIR, "*", f"*{SUMMARY_SUFFIX}")):
        exp = path.basename(p)[: -len(SUMMARY_SUFFIX)]
        found.add(exp)
    for p in glob.glob(path.join(CHECKPOINTS_DIR, "*", "*.pkl")):
        exp = path.basename(path.dirname(p))
        found.add(exp)
    out = sorted(found)
    if filter_substr:
        out = [e for e in out if filter_substr.lower() in e.lower()]
    return out


def _exp_folder(exp: str) -> str:
    return path.join(CHECKPOINTS_DIR, exp)


def _jsonl_path(exp: str) -> str:
    return path.join(RESULTS_DIR, _series_root(exp), f"{exp}{SUMMARY_SUFFIX}")


def _pkl_path(exp: str) -> str | None:
    candidates = glob.glob(path.join(_exp_folder(exp), "*.pkl"))
    return candidates[0] if candidates else None


def load_final(exp: str) -> dict | None:
    """Load the final record for one experiment, JSONL first, pickle fallback."""
    jsonl = _jsonl_path(exp)
    if path.exists(jsonl):
        rec = final_record(jsonl)
        if rec is not None:
            return rec
    pkl = _pkl_path(exp)
    if pkl is None:
        return None
    try:
        d = load_pickle_results(pkl)
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"[warn] failed to read {pkl}: {e}\n")
        return None
    # Build a final record on the fly (no config — legacy pickles do not carry it).
    return build_final_record(
        exp_name=exp,
        epochs=len(next(iter((d.get("win_rate") or {}).values()), [])) or 0,
        config={"_backfilled_from_pkl": True},
        loss_data=d.get("loss_values", {}) or {},
        grad_norm_data=d.get("grad_norm_data", {}) or {},
        win_rate=d.get("win_rate", {}) or {},
        q_values_history=d.get("q_values_history", {}) or {},
    )


def load_checkpoints(exp: str) -> list[dict]:
    jsonl = _jsonl_path(exp)
    if not path.exists(jsonl):
        return []
    return [r for r in read_records(jsonl) if r.get("type") == "checkpoint"]


# ---------- value access ----------


def get_metric(rec: dict, key: str) -> Any:
    """Look up a metric by dotted key, e.g. 'wr_final.ME_endgame(2)_E_5000' or
    'q_select_winners'. Returns None if missing."""
    if rec is None:
        return None
    parts = key.split(".")
    cur: Any = rec
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur


def _fmt(v: Any, width: int = 0) -> str:
    if v is None:
        s = "—"
    elif isinstance(v, float):
        s = f"{v:.3f}"
    elif isinstance(v, bool):
        s = "yes" if v else "no"
    elif isinstance(v, dict) and "slope_pp_per_1000ep" in v:
        s = _fmt_trend(v)
    else:
        s = str(v)
    if width:
        return s.ljust(width)
    return s


def _fmt_trend(t: Any) -> str:
    """Format a wr_trend dict as 'slope±halfCI (★ if rising)' in pp/1k ep."""
    if not isinstance(t, dict):
        return "—"
    slope = t.get("slope_pp_per_1000ep")
    lo = t.get("ci_low_pp_per_1000ep")
    hi = t.get("ci_high_pp_per_1000ep")
    rising = t.get("still_rising")
    if slope is None:
        return "—"
    half = None
    if lo is not None and hi is not None:
        half = (hi - lo) / 2.0
    marker = "↑" if rising else " "
    if half is not None:
        return f"{slope:+.1f}±{half:.1f}{marker}"
    return f"{slope:+.1f}{marker}"


def _common_rivals(recs: list[dict]) -> list[str]:
    rivals: dict[str, int] = {}
    for r in recs:
        for rv in (r.get("wr_final") or {}).keys():
            rivals[rv] = rivals.get(rv, 0) + 1
    return sorted(rivals.keys(), key=lambda r: (-rivals[r], r))


# ---------- subcommands ----------


def cmd_list(args: argparse.Namespace) -> int:
    exps = discover_experiments(args.filter)
    if not exps:
        print("No experiments found.")
        return 1
    recs = [(e, load_final(e)) for e in exps]
    rivals = _common_rivals([r for _, r in recs if r])

    header = ["experiment", "epochs", "loss"]
    for rv in rivals:
        header.append(f"WR_final[{rv}]")
        header.append(f"WR_trend[{rv}]")
    header += ["q_sel_win", "q_pl_win"]

    # Sort
    sort_key = args.sort
    def sort_val(item):
        e, r = item
        if r is None:
            return (1, e)
        v = get_metric(r, sort_key) if sort_key else None
        if v is None:
            return (1, e)
        return (0, -v if isinstance(v, (int, float)) else v)

    if sort_key:
        recs.sort(key=sort_val)

    widths = [max(len(h), 12) for h in header]
    for e, _ in recs:
        widths[0] = max(widths[0], len(e))

    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))
    for e, r in recs:
        if r is None:
            row = [e, "—", "—"] + ["—", "—"] * len(rivals) + ["—", "—"]
        else:
            row = [
                e,
                _fmt(r.get("epochs")),
                _fmt(r.get("loss_final")),
            ]
            for rv in rivals:
                row.append(_fmt((r.get("wr_final") or {}).get(rv)))
                row.append(_fmt_trend((r.get("wr_trend") or {}).get(rv)))
            row.append(_fmt(r.get("q_select_winners")))
            row.append(_fmt(r.get("q_place_winners")))
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    rec = load_final(args.exp)
    if rec is None:
        print(f"No results found for {args.exp}")
        return 1
    import json

    print(json.dumps(rec, indent=2, ensure_ascii=False))
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    ra = load_final(args.a)
    rb = load_final(args.b)
    if ra is None or rb is None:
        miss = args.a if ra is None else args.b
        print(f"No results found for {miss}")
        return 1

    def all_keys(d: dict, prefix: str = "") -> list[str]:
        out: list[str] = []
        for k, v in d.items():
            kp = f"{prefix}{k}"
            if isinstance(v, dict):
                out.extend(all_keys(v, kp + "."))
            else:
                out.append(kp)
        return sorted(out)

    # Metrics: top-level keys excluding config + gates
    metric_keys = sorted(
        k for k in (set(ra) | set(rb))
        if k not in ("config", "gates", "type", "exp_name")
        and not isinstance(ra.get(k), dict)
        and not isinstance(rb.get(k), dict)
    )
    # Nested dict metrics
    dict_keys = sorted(
        k for k in (set(ra) | set(rb))
        if k in ("wr_final", "wr_peak", "wr_trend")
    )
    config_keys = sorted(set((ra.get("config") or {}).keys()) | set((rb.get("config") or {}).keys()))

    def line(label, va, vb):
        diff_marker = "  " if va == vb else "* "
        print(f"  {diff_marker}{label:<40s}  {_fmt(va):<14s}  {_fmt(vb):<14s}")

    print(f"== {args.a}  vs  {args.b} ==")
    print(f"  {'METRIC':<42s}  {args.a:<14s}  {args.b:<14s}")
    print(f"  {'-'*42}  {'-'*14}  {'-'*14}")
    for k in metric_keys:
        line(k, ra.get(k), rb.get(k))
    for k in dict_keys:
        da = ra.get(k) or {}
        db = rb.get(k) or {}
        for rv in sorted(set(da) | set(db)):
            line(f"{k}[{rv}]", da.get(rv), db.get(rv))
    print()
    print(f"  {'CONFIG':<42s}  {args.a:<14s}  {args.b:<14s}")
    print(f"  {'-'*42}  {'-'*14}  {'-'*14}")
    ca = ra.get("config") or {}
    cb = rb.get("config") or {}
    for k in config_keys:
        line(k, ca.get(k), cb.get(k))
    return 0


def cmd_rank(args: argparse.Namespace) -> int:
    exps = discover_experiments(args.filter)
    rows: list[tuple[float, str, dict]] = []
    for e in exps:
        rec = load_final(e)
        if rec is None:
            continue
        v = get_metric(rec, args.metric)
        if v is None or not isinstance(v, (int, float)):
            continue
        rows.append((float(v), e, rec))
    rows.sort(reverse=not args.ascending)
    rows = rows[: args.top] if args.top else rows
    print(f"Rank by {args.metric}{' (asc)' if args.ascending else ' (desc)'}:")
    print(f"  {'#':<4s}{'experiment':<40s}{'value':<12s}{'epochs':<10s}{'loss_final':<12s}")
    print(f"  {'-'*4}{'-'*40}{'-'*12}{'-'*10}{'-'*12}")
    for i, (v, e, rec) in enumerate(rows, 1):
        print(
            f"  {i:<4d}{e:<40s}{_fmt(v):<12s}"
            f"{_fmt(rec.get('epochs')):<10s}{_fmt(rec.get('loss_final')):<12s}"
        )
    return 0


def cmd_gates(args: argparse.Namespace) -> int:
    rec = load_final(args.exp)
    if rec is None:
        print(f"No results found for {args.exp}")
        return 1
    gates = rec.get("gates") or {}
    if not gates:
        print(f"No pre-registered gates recorded for {args.exp}.")
        return 1
    print(f"Gates for {args.exp}:")
    for k, v in gates.items():
        print(f"  {'PASS' if v else 'FAIL':<5s}  {k}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="results_compare", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="list experiments with headline metrics")
    pl.add_argument("--filter", default=None, help="substring filter on exp name")
    pl.add_argument(
        "--sort", default=None,
        help="dotted metric key to sort by (e.g. 'wr_final.bot_random')",
    )
    pl.set_defaults(func=cmd_list)

    ps = sub.add_parser("show", help="show final record as JSON")
    ps.add_argument("exp")
    ps.set_defaults(func=cmd_show)

    pd = sub.add_parser("diff", help="side-by-side comparison of two experiments")
    pd.add_argument("a")
    pd.add_argument("b")
    pd.set_defaults(func=cmd_diff)

    pr = sub.add_parser("rank", help="leaderboard by a metric")
    pr.add_argument("--metric", required=True, help="dotted metric key")
    pr.add_argument("--top", type=int, default=0, help="limit to top N (0 = all)")
    pr.add_argument("--filter", default=None)
    pr.add_argument("--ascending", action="store_true", help="ascending sort")
    pr.set_defaults(func=cmd_rank)

    pg = sub.add_parser("gates", help="show pass/fail of pre-registered gates")
    pg.add_argument("exp")
    pg.set_defaults(func=cmd_gates)
    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
