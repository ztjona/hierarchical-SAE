# -*- coding: utf-8 -*-
"""SELECT-head capacity probe — is the blunder a head-capacity limit or a loss/optimisation limit?

Companion to ``oracle_target_audit.py``. That tool showed the minimax target is
*perfect* yet the deployed ``Q_select`` (a single linear layer ``fc2_select`` on
the shared ``fc1`` features) inverts the ranking on ~16-20% of decisive states.
Two explanations remain, indistinguishable from the autopsy:

* **capacity** — the ranking is not *linearly* decodable from ``fc1``; a single
  linear head cannot express it. Fix = a deeper, select-specific head.
* **loss / optimisation** — the ranking *is* linearly decodable but RL training
  (SmoothL1 + shared-trunk multitask + replay) didn't fit the linear head to it.
  Fix = a ranking/margin loss, not more capacity.

Method. Freeze the trunk. Capture the exact 512-d features the select head reads
(forward-pre-hook on ``fc2_select``) over self-play-sampled **decisive** SELECT
states, with the **oracle minimax target** as labels. Train read-out *probes* of
increasing depth on a train split and measure the held-out **blunder rate**
(argmax over legal pieces picks a hot give):

* ``deployed``  — the live ``Q_select`` (reference; ≈ 1 − Test B).
* ``linear``    — best a linear head could do on these features (the ceiling for
  the current architecture). ``deployed − linear`` = optimisation slack.
* ``mlp1`` / ``mlp2`` — one / two hidden layers. ``linear − mlp_best`` = the gain
  from nonlinearity = the case for a deeper select head.

Verdict
-------
* ``mlp_best ≪ linear``           → CAPACITY: add select-specific layers.
* ``linear ≪ deployed`` (mlp ≈ linear) → LOSS/OPT: ranking loss, not capacity.
* all ≈ deployed (high)            → TRUNK: ``fc1`` lacks the info.

Output: ``analysis/competence_audit/results/<exp>/select_capacity_probe.jsonl``.

Usage
-----
    python analysis/competence_audit/select_capacity_probe.py \
        --exp 'Xa_levers(1)0604_PLACE_WIN' --epoch 6000 \
        --architecture QuartoCNNAutoregUnifiedS4 \
        [--n-states 4000] [--max-oracle 3000] [--depth 2] \
        [--probe-epochs 400] [--seed 42]
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
import torch.nn as nn

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
from audit import sample_states, _board_from_encoding, _piece_is_losing  # noqa: E402

RESULTS_DIR = os.path.join(THIS_DIR, "results")


# ──────────────────────────────────────────────────────────────────────
# Dataset construction: frozen features + oracle targets on decisive states
# ──────────────────────────────────────────────────────────────────────


def _collect_dataset(net, select_states, oracle, max_oracle):
    """Return parallel arrays over decisive SELECT states:
    features (N,512), target (N,16), legal-mask (N,16), available/losing index
    lists, and the deployed Q_select (N,16)."""
    device = next(net.parameters()).device
    sb = torch.from_numpy(np.stack([s.state_board for s in select_states])).to(device)
    sa = torch.from_numpy(np.stack([s.state_aux for s in select_states])).to(device)

    captured = {}
    h = net.fc2_select.register_forward_pre_hook(
        lambda module, inp: captured.__setitem__("x", inp[0].detach())
    )
    with torch.no_grad():
        _, q_select = net.forward(sb, sa)
    h.remove()
    feats_all = captured["x"].cpu().numpy()          # (M, 512) — exactly what fc2_select reads
    qsel_all = q_select.cpu().numpy()                # (M, 16)  — deployed head

    feats, targets, masks, avails, losings, qsels = [], [], [], [], [], []
    n = 0
    for i, ss in enumerate(select_states):
        if n >= max_oracle:
            break
        board, empties = _board_from_encoding(ss.state_board)
        if not empties:
            continue
        avail = [Piece.from_index(j) for j in range(16) if ss.state_aux[16 + j] > 0.5]
        if not avail:
            continue
        losing = {p.index() for p in avail if _piece_is_losing(board, p, empties)}
        if not losing:
            continue
        safe = {p.index() for p in avail} - losing
        if not safe:
            continue
        t, m = _minimax_select_target(oracle, board.serialize(),
                                      {p.index() for p in avail}, mode_2x2=True)
        masked = [j for j in (p.index() for p in avail) if m[j] > 0.5]
        if not (losing & set(masked)) or not (safe & set(masked)):
            continue
        feats.append(feats_all[i]); targets.append(t); masks.append(m)
        avails.append([p.index() for p in avail]); losings.append(losing)
        qsels.append(qsel_all[i])
        n += 1

    return (np.asarray(feats, np.float32), np.asarray(targets, np.float32),
            np.asarray(masks, np.float32), avails, losings, np.asarray(qsels, np.float32))


# ──────────────────────────────────────────────────────────────────────
# Probes
# ──────────────────────────────────────────────────────────────────────


def _build_probe(kind: str, d_in: int = 512) -> nn.Module:
    if kind == "linear":
        body: list[nn.Module] = [nn.Linear(d_in, 16)]
    elif kind == "mlp1":
        body = [nn.Linear(d_in, 128), nn.ReLU(), nn.Linear(128, 16)]
    elif kind == "mlp2":
        body = [nn.Linear(d_in, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 16)]
    else:
        raise ValueError(kind)
    return nn.Sequential(*body, nn.Tanh())


def _blunder_rate(out: np.ndarray, avails, losings) -> float:
    """Fraction of states whose argmax over legal pieces is a losing give."""
    bl = 0
    for i in range(len(avails)):
        pick = max(avails[i], key=lambda j: out[i, j])
        bl += int(pick in losings[i])
    return bl / max(1, len(avails))


def _masked_sl1(out, y, m):
    return (nn.functional.smooth_l1_loss(out, y, reduction="none") * m).sum() / m.sum()


def _train_probe(kind, Xtr, Ytr, Mtr, Xval, Yval, Mval, Xte,
                 epochs, lr, weight_decay, patience, seed):
    """Train a probe with early stopping on a held-out validation split.

    Returns (train_pred, test_pred, n_params, best_epoch) at the epoch with the
    lowest masked-SmoothL1 validation loss — so generalisation, not memorisation,
    is what we read off.
    """
    torch.manual_seed(seed)
    probe = _build_probe(kind, Xtr.shape[1])
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    xtr, ytr, mtr = (torch.from_numpy(a) for a in (Xtr, Ytr, Mtr))
    xval, yval, mval = (torch.from_numpy(a) for a in (Xval, Yval, Mval))

    best_val, best_state, best_epoch, bad = float("inf"), None, 0, 0
    for e in range(epochs):
        probe.train(); opt.zero_grad()
        _masked_sl1(probe(xtr), ytr, mtr).backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            vloss = float(_masked_sl1(probe(xval), yval, mval))
        if vloss < best_val - 1e-5:
            best_val, best_epoch, bad = vloss, e, 0
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        tr = probe(xtr).numpy()
        te = probe(torch.from_numpy(Xte)).numpy()
    return tr, te, sum(p.numel() for p in probe.parameters()), best_epoch


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────


def probe_experiment(exp_name, *, epoch, architecture, n_states, max_oracle,
                     depth, probe_epochs, lr, weight_decay, patience, seed):
    t0 = time.time()
    net, cfg = load_checkpoint(exp_name, epoch=epoch, architecture=architecture)
    net.eval()
    _, select_states = sample_states(net, n_states=n_states, seed=seed)
    oracle = MinimaxBot(depth=depth)

    print(f"\n{'='*64}")
    print(f"  SELECT capacity probe: {exp_name}  (epoch {cfg['epoch']}, {cfg['architecture']})")
    print(f"  sampled SELECT states: {len(select_states)}   minimax depth: {depth}")
    print(f"{'='*64}")

    feats, targets, masks, avails, losings, qsels = _collect_dataset(
        net, select_states, oracle, max_oracle)
    N = len(feats)
    if N < 400:
        raise RuntimeError(f"only {N} decisive states collected; raise --n-states")

    # train / val / test split (test 25%, val 15% of the remainder)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_te = N // 4
    te_idx, rest = perm[:n_te], perm[n_te:]
    n_va = max(100, len(rest) // 6)
    va_idx, tr_idx = rest[:n_va], rest[n_va:]

    mu, sd = feats[tr_idx].mean(0), feats[tr_idx].std(0) + 1e-6
    Xtr = (feats[tr_idx] - mu) / sd
    Xva = (feats[va_idx] - mu) / sd
    Xte = (feats[te_idx] - mu) / sd

    av_tr, lo_tr = [avails[i] for i in tr_idx], [losings[i] for i in tr_idx]
    av_te, lo_te = [avails[i] for i in te_idx], [losings[i] for i in te_idx]

    deployed_te = _blunder_rate(qsels[te_idx], av_te, lo_te)
    rows = {"deployed": {"test_blunder": deployed_te, "n_params": None,
                         "train_blunder": _blunder_rate(qsels[tr_idx], av_tr, lo_tr),
                         "best_epoch": None}}
    for kind in ("linear", "mlp1", "mlp2"):
        tr_out, te_out, nparams, best_ep = _train_probe(
            kind, Xtr, targets[tr_idx], masks[tr_idx],
            Xva, targets[va_idx], masks[va_idx], Xte,
            probe_epochs, lr, weight_decay, patience, seed)
        rows[kind] = {
            "train_blunder": _blunder_rate(tr_out, av_tr, lo_tr),
            "test_blunder": _blunder_rate(te_out, av_te, lo_te),
            "n_params": int(nparams), "best_epoch": int(best_ep),
        }

    lin, m1, m2 = (rows[k]["test_blunder"] for k in ("linear", "mlp1", "mlp2"))
    mlp_best = min(m1, m2)
    # Decision-relevant comparison: best head on the SAME frozen features vs the
    # live head. Robust across seeds; the linear probe is regularisation-fragile
    # (overfits 512-d features) so it is reported but NOT used for the verdict.
    head_headroom = deployed_te - mlp_best     # >0 ⇒ a better head on frozen feats helps
    opt_slack = deployed_te - lin              # secondary (linear probe unreliable)
    capacity_gain = lin - mlp_best             # secondary
    linear_unreliable = lin > deployed_te + 0.03

    if head_headroom >= 0.05:
        verdict = ("HEAD/CAPACITY-LIMITED — a better head on the same frozen features cuts "
                   "%.1fpp ⇒ a deeper select-specific head helps." % (100 * head_headroom))
    else:
        verdict = ("REPRESENTATION-LIMITED — the best head on the frozen trunk beats the "
                   "deployed head by only %.1fpp; fc1 caps the held-out blunder rate. "
                   "Head capacity/loss is ~exhausted ⇒ reshape the TRUNK (margin loss "
                   "through the trunk / richer select feature path / bigger trunk), not the head."
                   % (100 * head_headroom))

    record = {
        "schema_version": 1,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "exp_name": exp_name, "epoch": cfg["epoch"],
        "checkpoint_path": cfg["checkpoint_path"], "architecture": cfg["architecture"],
        "minimax_depth": depth, "seed": seed,
        "n_decisive": N, "n_train": len(tr_idx), "n_val": len(va_idx), "n_test": len(te_idx),
        "probe_epochs": probe_epochs, "lr": lr, "weight_decay": weight_decay, "patience": patience,
        "elapsed_seconds": round(time.time() - t0, 1),
        "probes": rows,
        "head_headroom": head_headroom,
        "optimisation_slack": opt_slack, "capacity_gain": capacity_gain,
        "linear_unreliable": linear_unreliable, "verdict": verdict,
    }
    _emit(exp_name, record)
    _print_summary(record)
    return record


def _emit(exp_name, record):
    out_dir = os.path.join(RESULTS_DIR, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "select_capacity_probe.jsonl")
    with open(out_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return out_path


def _print_summary(rec):
    print(f"\n  decisive states: {rec['n_decisive']}  "
          f"(train {rec['n_train']} / val {rec['n_val']} / test {rec['n_test']})")
    print(f"  held-out blunder rate (argmax picks a hot give):")
    for k in ("deployed", "linear", "mlp1", "mlp2"):
        r = rec["probes"][k]
        npar = f"{r['n_params']:>7,}p" if r["n_params"] else "  (live)"
        ep = f"@ep{r['best_epoch']}" if r["best_epoch"] is not None else ""
        print(f"    {k:<9} {100*r['test_blunder']:5.1f}%   (train {100*r['train_blunder']:4.1f}%) {npar} {ep}")
    print(f"  head headroom (deployed − best MLP): {100*rec['head_headroom']:+.1f} pp"
          f"   [linear probe {'UNRELIABLE/overfit' if rec['linear_unreliable'] else 'ok'}]")
    print(f"  → {rec['verdict']}")


def main():
    p = argparse.ArgumentParser(
        description="Frozen-feature capacity probe for the unified_autoreg SELECT head.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--exp", required=True, nargs="+")
    p.add_argument("--epoch", type=int, default=None)
    p.add_argument("--architecture", default=None)
    p.add_argument("--n-states", type=int, default=9000)
    p.add_argument("--max-oracle", type=int, default=5000, help="Cap on decisive states labelled by the oracle")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--probe-epochs", type=int, default=6000, help="Max epochs (early-stopped on val)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=400, help="Early-stop patience on val loss")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    for exp in args.exp:
        try:
            probe_experiment(exp, epoch=args.epoch, architecture=args.architecture,
                             n_states=args.n_states, max_oracle=args.max_oracle,
                             depth=args.depth, probe_epochs=args.probe_epochs,
                             lr=args.lr, weight_decay=args.weight_decay,
                             patience=args.patience, seed=args.seed)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            print(f"[ERROR] {exp}: {e}")


if __name__ == "__main__":
    main()
