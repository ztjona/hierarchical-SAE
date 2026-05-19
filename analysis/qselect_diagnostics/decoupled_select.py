# -*- coding: utf-8 -*-
"""D3 — decoupled select-only network.

See ``analysis/qselect_diagnostics/PLAN.md`` → "D3".

Trains a small, self-contained CNN purely on the minimax-oracle select
targets — no trunk sharing with Q_place, no place loss. The same D1 metric
is then computed on the resulting network. If the decoupled net materially
outperforms the joint network under D1, shared-trunk gradient interference
(H3) is implicated.

The architecture is deliberately defined inside this script — the PLAN says
"do not subclass from QuartoRL.architectures" because we are intentionally
breaking the shared-trunk contract for diagnostic purposes.

Usage:
    python analysis/qselect_diagnostics/decoupled_select.py \
        --exp 'Ta_minimaxSelect(1)0514_DEPTH_2' \
        [--epoch 4000] [--epochs 2000] [--n-matches 64]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from analysis.qselect_diagnostics._common import (  # noqa: E402
    PHASE_SELECT,
    Quarto_unified_bot,
    UNIFIED_AUX_DIM,
    emit_jsonl,
    gen_experience_unified_autoreg,
    load_checkpoint,
    MinimaxBot,
)
from analysis.qselect_diagnostics.position_structure import (  # noqa: E402
    FORCE_THRESHOLD,
    _argmax_masked,
    _argmin_masked,
    _classify_state,
    _state_spearman,
)


class SelectOnlyNet(nn.Module):
    """A minimal CNN that maps (board, aux) → 16-d Q_select.

    Matches the unified trunk's input shape — board (B, 16, 4, 4) and
    aux (B, 32) — but is fully independent: no shared weights, no place
    head.
    """

    def __init__(self, aux_fc_size: int = 32, n_neurons: int = 256):
        super().__init__()
        assert aux_fc_size % 16 == 0
        self.fc_in_aux = nn.Linear(UNIFIED_AUX_DIM, aux_fc_size)
        trunk_in_channels = 16 + (aux_fc_size // 16)
        self.conv1 = nn.Conv2d(trunk_in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, n_neurons)
        self.fc_out = nn.Linear(n_neurons, 16)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x_board: torch.Tensor, x_aux: torch.Tensor) -> torch.Tensor:
        aux_feat = F.relu(self.fc_in_aux(x_aux))
        aux_channels = aux_feat.shape[-1] // 16
        aux_map = aux_feat.view(-1, aux_channels, 4, 4)
        x = torch.cat([x_board, aux_map], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.tanh(self.fc_out(x))


def _collect_select_rows(
    *,
    net: torch.nn.Module,
    n_matches: int,
    n_last_states: int,
    mode_2x2: bool,
    temperature: float,
    oracle: MinimaxBot,
    label: str,
) -> tuple[np.ndarray, ...]:
    p1 = Quarto_unified_bot(model=net, deterministic=False, temperature=temperature)
    p2 = Quarto_unified_bot(model=net, deterministic=False, temperature=temperature)
    exp = gen_experience_unified_autoreg(
        p1_bot=p1,
        p2_bot=p2,
        n_last_states=n_last_states,
        number_of_matches=n_matches,
        verbose=False,
        PROGRESS_MESSAGE=f"D3 row collection ({label}, N={n_last_states})",
        mode_2x2=mode_2x2,
        REWARD_FUNCTION_TYPE="final",
        COLLECT_BOARDS=False,
        select_oracle=oracle,
    )
    phase = exp["phase"].cpu().numpy()
    sel = np.where(phase == PHASE_SELECT)[0]
    return (
        exp["state_board"].cpu().numpy()[sel].astype(np.float32),
        exp["state_aux"].cpu().numpy()[sel].astype(np.float32),
        exp["target_sel_minimax"].cpu().numpy()[sel].astype(np.float32),
        exp["target_sel_minimax_mask"].cpu().numpy()[sel].astype(np.float32),
        exp["valid_mask"].cpu().numpy()[sel].astype(np.float32),
    )


def _masked_smoothl1(
    q: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    diff = q - target
    abs_diff = diff.abs()
    elt = torch.where(abs_diff < 1.0, 0.5 * diff.pow(2), abs_diff - 0.5)
    denom = mask.sum().clamp_min(1.0)
    return (elt * mask).sum() / denom


def run(
    exp_name: str,
    epoch: int | None,
    n_matches: int,
    n_matches_eval: int,
    train_epochs: int,
    batch_size: int,
    lr: float,
    n_last_states_curriculum: int,
    n_last_states_endgame: int,
    mode_2x2: bool,
    oracle_depth: int,
    temperature: float,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net, cfg = load_checkpoint(exp_name, epoch=epoch)
    oracle = MinimaxBot(depth=oracle_depth)

    # ---- collect training rows
    sb1, sa1, tgt1, tmsk1, vm1 = _collect_select_rows(
        net=policy_net, n_matches=n_matches, n_last_states=n_last_states_curriculum,
        mode_2x2=mode_2x2, temperature=temperature, oracle=oracle, label="train-curric",
    )
    sb2, sa2, tgt2, tmsk2, vm2 = _collect_select_rows(
        net=policy_net, n_matches=n_matches, n_last_states=n_last_states_endgame,
        mode_2x2=mode_2x2, temperature=temperature, oracle=oracle, label="train-endgame",
    )
    train_sb = np.concatenate([sb1, sb2], axis=0)
    train_sa = np.concatenate([sa1, sa2], axis=0)
    train_tgt = np.concatenate([tgt1, tgt2], axis=0)
    train_tmsk = np.concatenate([tmsk1, tmsk2], axis=0)

    # ---- held-out eval rows (separate seed, separate generation)
    torch.manual_seed(seed + 99)
    np.random.seed(seed + 99)
    eval_sb, eval_sa, eval_tgt, eval_tmsk, eval_vm = _collect_select_rows(
        net=policy_net, n_matches=n_matches_eval, n_last_states=n_last_states_curriculum,
        mode_2x2=mode_2x2, temperature=temperature, oracle=oracle, label="eval",
    )

    n_train = train_sb.shape[0]
    print(f"D3: train rows = {n_train}, eval rows = {eval_sb.shape[0]}")

    # ---- model + optimizer
    model = SelectOnlyNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, amsgrad=True)

    sb_t = torch.from_numpy(train_sb).to(device)
    sa_t = torch.from_numpy(train_sa).to(device)
    tgt_t = torch.from_numpy(train_tgt).to(device)
    tmsk_t = torch.from_numpy(train_tmsk).to(device)

    loss_log: list[dict] = []
    t0 = time.time()
    for ep in range(train_epochs):
        model.train()
        # full sweep, mini-batches
        perm = torch.randperm(n_train, device=device)
        ep_losses: list[float] = []
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            q = model(sb_t[idx], sa_t[idx])
            loss = _masked_smoothl1(q, tgt_t[idx], tmsk_t[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_losses.append(float(loss.item()))
        if (ep + 1) % max(1, train_epochs // 20) == 0 or ep == 0:
            mean_loss = float(np.mean(ep_losses))
            loss_log.append({"epoch": ep + 1, "loss": mean_loss})
            print(f"  ep {ep + 1:4d}/{train_epochs}  loss={mean_loss:.6f}")
    train_seconds = time.time() - t0

    # ---- evaluate on held-out set using D1 metric
    model.eval()
    with torch.no_grad():
        eval_sb_t = torch.from_numpy(eval_sb).to(device)
        eval_sa_t = torch.from_numpy(eval_sa).to(device)
        q_eval = model(eval_sb_t, eval_sa_t).cpu().numpy()

    n_total = eval_sb.shape[0]
    n_decisive = 0
    n_single_force = 0
    forcing_loss_hits = 0
    n_any_force = 0
    forcing_loss_bottom_hits = 0
    chance_baseline_bottom: list[float] = []
    n_safe_eval = 0
    safe_piece_hits = 0
    rhos: list[float] = []
    chance_baselines: list[float] = []

    for i in range(n_total):
        target = eval_tgt[i]
        mask = eval_tmsk[i]
        forcing, safe, n_avail = _classify_state(target, mask)
        if n_avail < 2:
            continue
        rho = _state_spearman(q_eval[i], target, mask)
        if rho is not None:
            rhos.append(rho)
        if len(forcing) >= 1 and len(safe) >= 1:
            n_decisive += 1
            chance_baselines.append(1.0 / n_avail)
            argmax_idx = _argmax_masked(q_eval[i], mask)
            n_safe_eval += 1
            if argmax_idx in safe:
                safe_piece_hits += 1
        if len(forcing) == 1 and n_avail >= 2:
            n_single_force += 1
            argmin_idx = _argmin_masked(q_eval[i], mask)
            if argmin_idx in forcing:
                forcing_loss_hits += 1
        if len(forcing) >= 1 and n_avail >= 2:
            n_any_force += 1
            chance_baseline_bottom.append(len(forcing) / n_avail)
            argmin_idx = _argmin_masked(q_eval[i], mask)
            if argmin_idx in forcing:
                forcing_loss_bottom_hits += 1

    rho_arr = np.array(rhos, dtype=np.float64)
    rho_summary = {
        "n": int(len(rho_arr)),
        "mean": float(rho_arr.mean()) if len(rho_arr) else None,
        "p25": float(np.percentile(rho_arr, 25)) if len(rho_arr) else None,
        "p50": float(np.percentile(rho_arr, 50)) if len(rho_arr) else None,
        "p75": float(np.percentile(rho_arr, 75)) if len(rho_arr) else None,
    }

    record = {
        "diagnostic": "decoupled_select",
        "schema_version": 1,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "exp_name": cfg["exp_name"],
        "epoch": cfg["epoch"],
        "architecture": cfg["architecture"],
        "checkpoint_path": cfg["checkpoint_path"],
        "network": "decoupled_select_only",
        "config": {
            "train_epochs": train_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "n_train_rows": int(n_train),
            "n_eval_rows": int(eval_sb.shape[0]),
            "n_matches_train": n_matches,
            "n_matches_eval": n_matches_eval,
            "n_last_states_curriculum": n_last_states_curriculum,
            "n_last_states_endgame": n_last_states_endgame,
            "mode_2x2": mode_2x2,
            "oracle_depth": oracle_depth,
            "temperature": temperature,
            "force_threshold": FORCE_THRESHOLD,
            "seed": seed,
            "train_seconds": train_seconds,
        },
        "loss_log": loss_log,
        "n_states_total": int(n_total),
        "n_states_decisive": int(n_decisive),
        "n_states_any_forcing": int(n_any_force),
        "n_states_single_forcing": int(n_single_force),
        "forcing_loss_recall": (
            forcing_loss_hits / n_single_force if n_single_force else None
        ),
        "forcing_loss_bottom_recall": (
            forcing_loss_bottom_hits / n_any_force if n_any_force else None
        ),
        "forcing_loss_bottom_chance": (
            float(np.mean(chance_baseline_bottom)) if chance_baseline_bottom else None
        ),
        "safe_piece_recall": (
            safe_piece_hits / n_safe_eval if n_safe_eval else None
        ),
        "chance_baseline_recall": (
            float(np.mean(chance_baselines)) if chance_baselines else None
        ),
        "spearman_rho": rho_summary,
    }
    out = emit_jsonl(cfg["exp_name"], "decoupled_select", record)
    print(f"Wrote {out}")
    print(
        f"  decoupled safe_piece_recall = {record['safe_piece_recall']}  "
        f"forcing_loss_bottom_recall = {record['forcing_loss_bottom_recall']} "
        f"(chance={record['forcing_loss_bottom_chance']})  "
        f"rho_mean = {rho_summary['mean']}"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--n-matches", type=int, default=64,
                        help="self-play matches per bucket for training data")
    parser.add_argument("--n-matches-eval", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-last-states-curriculum", type=int, default=4)
    parser.add_argument("--n-last-states-endgame", type=int, default=2)
    parser.add_argument("--no-2x2", action="store_true")
    parser.add_argument("--oracle-depth", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    run(
        exp_name=args.exp,
        epoch=args.epoch,
        n_matches=args.n_matches,
        n_matches_eval=args.n_matches_eval,
        train_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_last_states_curriculum=args.n_last_states_curriculum,
        n_last_states_endgame=args.n_last_states_endgame,
        mode_2x2=not args.no_2x2,
        oracle_depth=args.oracle_depth,
        temperature=args.temperature,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
