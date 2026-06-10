# -*- coding: utf-8 -*-
"""SELECT-safety learnability — is the trunk too SMALL, or just mis-allocated?

Third design diagnostic after ``oracle_target_audit.py`` (target is perfect) and
``select_capacity_probe.py`` (the deployed head is at the floor of the FROZEN
trunk → representation-limited). The open question that remains: would a *bigger*
trunk help, or does the current architecture already have the capacity and simply
fails to allocate it to select (because the shared trunk is split with place)?

Test (the user's design). Take the **same architecture**, same input, **drop the
place head** (train only ``fc2_select`` → the whole trunk trains purely for
select), and learn piece-safety supervised, from scratch. If this exact trunk —
when dedicated to select — learns to avoid hot gives, then capacity is *not* the
limit and growing the trunk is the wrong lever; the champion's ~17% is
allocation/training. If even a dedicated same-arch net plateaus near ~17%, the
architecture/input genuinely can't represent it → a bigger/redesigned trunk is
justified.

This is a clean, well-powered redo of the qselect D3 "decoupled select" result
(which found select-only *worse*, but on only 2048 rows / depth-2 recall).

Labels. The depth-1 **hot mask** (``_piece_is_losing``) — a cheap rule, no
minimax oracle — so we can label 10k+ states fast. Target convention matches the
deployed head: ``+1`` for a safe give, ``−1`` for a hot give, masked to legal
pieces, regressed with SmoothL1 against the tanh output. Metric = held-out
**blunder rate** (argmax over legal pieces picks a hot give) on decisive states,
directly comparable to ``select_capacity_probe.py`` / Test B.

Arms: ``scratch`` (random init — the capacity test) and ``champion_init``
(start from the champion trunk, select-only fine-tune — the adaptability test).
``deployed`` = the live champion head on the same test states (reference).

Output: ``analysis/competence_audit/results/<exp>/select_safety_learnability.jsonl``.

Usage
-----
    python analysis/competence_audit/select_safety_learnability.py \
        --exp 'Xa_levers(1)0604_PLACE_WIN' --epoch 6000 \
        --architecture QuartoCNNAutoregUnifiedS4 \
        [--n-states 10000] [--epochs 120] [--batch 512] [--seed 42]
"""

from __future__ import annotations

import argparse
import copy
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
from _common import load_checkpoint  # noqa: E402
from audit import sample_states, _board_from_encoding, _piece_is_losing  # noqa: E402

RESULTS_DIR = os.path.join(THIS_DIR, "results")


# ──────────────────────────────────────────────────────────────────────
# Labels: depth-1 hot mask (rule-based, no oracle)
# ──────────────────────────────────────────────────────────────────────


def _build_labels(select_states):
    """Return target (N,16) [+1 safe / −1 hot / 0 illegal], legal-mask (N,16),
    and per-state available / hot index lists + a decisive flag."""
    N = len(select_states)
    target = np.zeros((N, 16), np.float32)
    mask = np.zeros((N, 16), np.float32)
    avails, hots, decisive = [], [], np.zeros(N, bool)
    for i, ss in enumerate(select_states):
        board, empties = _board_from_encoding(ss.state_board)
        available = [j for j in range(16) if ss.state_aux[16 + j] > 0.5]
        if not available or not empties:
            avails.append([]); hots.append(set()); continue
        hot = {j for j in available if _piece_is_losing(board, Piece.from_index(j), empties)}
        for j in available:
            mask[i, j] = 1.0
            target[i, j] = -1.0 if j in hot else 1.0
        avails.append(available); hots.append(hot)
        decisive[i] = (0 < len(hot) < len(available))
    return target, mask, avails, hots, decisive


def _blunder_rate(qsel, avails, hots, idx):
    """Blunder rate over DECISIVE states in ``idx`` (argmax legal piece is hot)."""
    n = bl = 0
    for i in idx:
        if not (0 < len(hots[i]) < len(avails[i])):
            continue
        n += 1
        pick = max(avails[i], key=lambda j: qsel[i, j])
        bl += int(pick in hots[i])
    return (bl / n) if n else None, n


# ──────────────────────────────────────────────────────────────────────
# Select-only training of the full architecture (place head gets no gradient)
# ──────────────────────────────────────────────────────────────────────


def _fresh_copy(net, reset: bool):
    m = copy.deepcopy(net)
    if reset:
        for mod in m.modules():
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters()
    return m


def _masked_sl1(out, y, m):
    return (nn.functional.smooth_l1_loss(out, y, reduction="none") * m).sum() / m.sum()


def _train_select_only(net, sb, sa, T, M, tr, va, *, epochs, lr, wd, batch, patience, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    tr_t = torch.as_tensor(tr)
    best_val, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        perm = tr_t[torch.randperm(len(tr_t))]
        for b0 in range(0, len(perm), batch):
            ids = perm[b0:b0 + batch]
            _, qs = net(sb[ids], sa[ids])
            opt.zero_grad()
            _masked_sl1(qs, T[ids], M[ids]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            _, qv = net(sb[va], sa[va])
            vloss = float(_masked_sl1(qv, T[va], M[va]))
        if vloss < best_val - 1e-5:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────


def learnability(exp_name, *, epoch, architecture, n_states, epochs, batch,
                 lr, wd, patience, seed):
    t0 = time.time()
    champ, cfg = load_checkpoint(exp_name, epoch=epoch, architecture=architecture)
    champ.eval()
    device = next(champ.parameters()).device
    _, select_states = sample_states(champ, n_states=n_states, seed=seed)

    print(f"\n{'='*64}")
    print(f"  SELECT-safety learnability: {exp_name}  ({cfg['architecture']}, ep {cfg['epoch']})")
    print(f"  sampled SELECT states: {len(select_states)}   (depth-1 hot-mask labels)")
    print(f"{'='*64}")

    T, M, avails, hots, decisive = _build_labels(select_states)
    sb = torch.as_tensor(np.stack([s.state_board for s in select_states])).to(device)
    sa = torch.as_tensor(np.stack([s.state_aux for s in select_states])).to(device)
    Tt, Mt = torch.as_tensor(T).to(device), torch.as_tensor(M).to(device)

    N = len(select_states)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_te = N // 4
    te, rest = perm[:n_te], perm[n_te:]
    n_va = max(200, len(rest) // 6)
    va, tr = rest[:n_va], rest[n_va:]

    # deployed champion head on the decisive test states (reference)
    with torch.no_grad():
        _, qs_dep = champ(sb[te], sa[te])
    # remap test-local preds back to global index space for _blunder_rate
    qs_dep_full = np.zeros((N, 16), np.float32); qs_dep_full[te] = qs_dep.cpu().numpy()
    dep_bl, n_dec_te = _blunder_rate(qs_dep_full, avails, hots, te)

    rows = {"deployed": {"test_blunder": dep_bl, "n_params": sum(p.numel() for p in champ.parameters())}}
    for arm, reset in (("scratch", True), ("champion_init", False)):
        net = _fresh_copy(champ, reset=reset).to(device)
        net = _train_select_only(net, sb, sa, Tt, Mt, tr, va,
                                 epochs=epochs, lr=lr, wd=wd, batch=batch,
                                 patience=patience, seed=seed)
        with torch.no_grad():
            qfull = np.zeros((N, 16), np.float32)
            qfull[te] = net(sb[te], sa[te])[1].cpu().numpy()
            qtr = np.zeros((N, 16), np.float32)
            qtr[tr] = net(sb[tr], sa[tr])[1].cpu().numpy()
        te_bl, _ = _blunder_rate(qfull, avails, hots, te)
        tr_bl, _ = _blunder_rate(qtr, avails, hots, tr)
        rows[arm] = {"test_blunder": te_bl, "train_blunder": tr_bl,
                     "n_params": sum(p.numel() for p in net.parameters())}

    scratch = rows["scratch"]["test_blunder"]
    scratch_tr = rows["scratch"]["train_blunder"]
    champ_init = rows["champion_init"]["test_blunder"]
    # The from-scratch arm is prone to optimisation failure (it must learn the whole
    # trunk on a few k states); if it can't fit TRAIN it underfit and its test number
    # is uninformative about the ceiling. champion_init (same arch, trunk reshaped from
    # the champion init) is the decisive capacity signal.
    scratch_underfit = scratch_tr is not None and scratch_tr > 0.10
    cand = [v for v in (scratch, champ_init) if v is not None]
    best_arm = min(cand) if cand else None
    if best_arm is not None and dep_bl is not None and best_arm <= dep_bl - 0.05:
        verdict = ("CAPACITY SUFFICIENT / ALLOCATION-LIMITED — reshaping the SAME-arch trunk "
                   "for select cuts held-out blunder %.1f%%→%.1f%%; the trunk has the capacity, "
                   "the joint head under-allocates ⇒ pressure the trunk (aux hot-head / margin), "
                   "don't grow it." % (100 * dep_bl, 100 * best_arm))
        if scratch_underfit:
            verdict += (" [from-scratch arm underfit (train %.1f%%) ⇒ optimisation, ignore its test]"
                        % (100 * scratch_tr))
    else:
        verdict = ("REPRESENTABILITY CEILING — no same-arch variant (dedicated or reshaped) "
                   "beats deployed (%.1f%%) by >5pp ⇒ the architecture/input caps it ⇒ a "
                   "bigger/redesigned trunk IS justified." % (100 * (dep_bl or 0)))

    record = {
        "schema_version": 1,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "exp_name": exp_name, "epoch": cfg["epoch"],
        "checkpoint_path": cfg["checkpoint_path"], "architecture": cfg["architecture"],
        "label": "depth1_hot_mask", "seed": seed,
        "n_states": N, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "n_decisive_test": n_dec_te,
        "epochs_max": epochs, "batch": batch, "lr": lr, "weight_decay": wd,
        "elapsed_seconds": round(time.time() - t0, 1),
        "probes": rows, "scratch_underfit": scratch_underfit, "verdict": verdict,
    }
    _emit(exp_name, record)
    _print_summary(record)
    return record


def _emit(exp_name, record):
    out_dir = os.path.join(RESULTS_DIR, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "select_safety_learnability.jsonl")
    with open(out_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return out_path


def _pct(x):
    return f"{100*x:.1f}%" if isinstance(x, float) else "  — "


def _print_summary(rec):
    print(f"\n  states {rec['n_states']} (train {rec['n_train']} / val {rec['n_val']} / "
          f"test {rec['n_test']}; decisive test {rec['n_decisive_test']})")
    print(f"  held-out blunder rate (argmax legal piece is a hot give):")
    for k in ("deployed", "scratch", "champion_init"):
        r = rec["probes"][k]
        tr = f"(train {_pct(r['train_blunder'])})" if "train_blunder" in r else "(live)"
        print(f"    {k:<13} {_pct(r['test_blunder'])}   {tr}")
    print(f"  → {rec['verdict']}")


def main():
    p = argparse.ArgumentParser(
        description="Same-arch select-only learnability test (trunk capacity vs allocation).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--exp", required=True, nargs="+")
    p.add_argument("--epoch", type=int, default=None)
    p.add_argument("--architecture", default=None)
    p.add_argument("--n-states", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=120, help="Max epochs (early-stopped on val)")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    for exp in args.exp:
        try:
            learnability(exp, epoch=args.epoch, architecture=args.architecture,
                         n_states=args.n_states, epochs=args.epochs, batch=args.batch,
                         lr=args.lr, wd=args.weight_decay, patience=args.patience, seed=args.seed)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc(); print(f"[ERROR] {exp}: {e}")


if __name__ == "__main__":
    main()
