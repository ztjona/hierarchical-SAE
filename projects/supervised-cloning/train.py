"""train.py – Train a supervised clone of MinimaxBot (unified-aux + soft labels).

Uses the ``QuartoCNNAutoregUnified`` architecture: a phase-stable 32-d aux
``[offered_one_hot ; available_pieces_mask]`` per sample, no phase embedding,
two output heads routed by sample action. PLACE samples train the PLACE head;
SELECT samples train the SELECT head.

Loss: soft-target cross-entropy (KL up to a constant) with the multi-hot
uniform-over-tied-optimal soft targets emitted by collect_data.py. Illegal
positions are masked to -1e9 before log_softmax so the unmasked simplex is
the set of legal moves. With ``--soft-weight 0.0`` the loss collapses to
ordinary CE on the hard label (for comparison runs).

Train/val split is at the GAME level. Augmentation applies the 8 D4
symmetries only on train.

Usage:
    train.py [options]

Options:
    --data <paths>           One or more input .npz files, comma-separated. game_ids
                             in later files are offset to stay unique.
                             [default: projects/supervised-cloning/data/collected.npz]
    --exp <name>             Experiment name. Output → projects/supervised-cloning/experiments/<name>/
    --out <path>             Checkpoint dir (ignored when --exp is set)
                             [default: projects/supervised-cloning/checkpoints]
    --epochs <int>           Training epochs  [default: 150]
    --batch <int>            Batch size       [default: 256]
    --lr <float>             Learning rate    [default: 1e-3]
    --val-split <float>      Val fraction     [default: 0.15]
    --lam <float>            SELECT loss weight relative to PLACE  [default: 1.0]
    --soft-weight <float>    Mix between soft and hard targets in [0,1].
                             1.0 = pure soft-target CE on multi-hot uniform;
                             0.0 = pure CE on the hard argmax label.
                             [default: 1.0]
    --seed <int>             Random seed      [default: 42]
    --n-matches-eval <int>   Matches per baseline for final win-rate eval  [default: 50]
    --no-eval                Skip win-rate evaluation after training.
    -h, --help               Show this help.
"""

from __future__ import annotations

import os
import sys
import random
import numpy as np
from pathlib import Path

_here = Path(__file__).resolve().parent
_root = _here
while not (_root / "bot").is_dir():
    _root = _root.parent
    if _root == _root.parent:
        raise RuntimeError("Could not find project root containing 'bot/'")

os.chdir(_root)
sys.path.insert(0, str(_root))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from docopt import docopt

from models.CNN_autoreg import QuartoCNNAutoregUnified
from bot.CNN_unified_bot import Quarto_bot as UnifiedBot
from bot.random_bot import Quarto_bot as RandomBot
from bot.minimax_bot import MinimaxBot
from quartopy import play_games

ACTION_PLACE = 0
ACTION_SELECT = 1


def raw_logits(model, x_board, x_aux):
    """Compute pre-activation logits from a unified-aux model.

    The trainer needs raw logits so that illegal-position masking via
    ``-1e9`` followed by ``log_softmax`` produces a clean simplex over
    legal moves. The model's ``forward(...)`` applies ``tanh``, which
    would saturate the mask and break the loss; bypass it by calling
    the model's ``_shared_trunk`` and the head linears directly.

    This helper lives here (not on the model class) so the shared
    ``models/CNN_autoreg.py`` stays byte-identical with the sibling
    hierarchical-SAE branch — avoiding a merge conflict on that file.
    """
    x = model._shared_trunk(x_board, x_aux)
    return model.fc2_place(x), model.fc2_select(x)


# ── board symmetry augmentation ───────────────────────────────────────────────
# The 4x4 Quarto board has 8 D4 symmetries. Under the unified-aux schema:
#   - boards (16,4,4)  : rotate / flip the spatial dims
#   - aux (32,)        : UNCHANGED — both 16-d blocks (offered, available) are
#                        rotation-invariant (piece identities, not positions).
#   - PLACE labels     : forward-permuted via perm_fwd (old_pos -> new_pos).
#   - PLACE legal_mask : gather via perm_inv (new_mask[i] = old_mask[perm_inv[i]]).
#   - PLACE soft_target: same gather as legal_mask (per-position prob mass).
#   - SELECT labels / masks / soft_targets: UNCHANGED (piece-indexed).


def _pos_inv(idx: int, transform_id: int) -> int:
    """For transform t, return the OLD position that ended up at NEW position idx.

    The board pipeline applies ``rot90_CCW^k`` then ``flip_cols`` (where
    k = t % 4 and the flip happens iff t ≥ 4). Inverting that composition
    requires applying the inverse operations in REVERSE order — flip first,
    then CW (inverse of CCW). Doing them in the other order silently mislabels
    half the augmented copies (transforms t ≥ 4).
    """
    r, c = divmod(idx, 4)
    if transform_id >= 4:
        c = 3 - c  # invert flip FIRST (flip is self-inverse)
    for _ in range(transform_id % 4):
        r, c = c, 3 - r  # then invert each CCW step with one CW step
    return r * 4 + c


_POS_PERMS_INV: list[np.ndarray] = [
    np.array([_pos_inv(i, t) for i in range(16)], dtype=np.int64) for t in range(8)
]
_POS_PERMS_FWD: list[np.ndarray] = [
    np.argsort(p).astype(np.int64) for p in _POS_PERMS_INV
]


def augment_symmetries(
    boards: np.ndarray,
    aux: np.ndarray,
    labels: np.ndarray,
    actions: np.ndarray,
    legal_masks: np.ndarray,
    soft_targets: np.ndarray,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Expand the dataset 8× by applying all dihedral symmetries.

    aux is rotation-invariant and passes through untouched. PLACE samples
    have their label / legal_mask / soft_target permuted; SELECT samples are
    geometry-invariant on all action-indexed fields.
    """
    out_b, out_aux, out_lbl, out_act, out_msk, out_soft = [], [], [], [], [], []

    for t in range(8):
        perm_inv = _POS_PERMS_INV[t]
        perm_fwd = _POS_PERMS_FWD[t]

        b = boards
        for _ in range(t % 4):
            b = np.rot90(b, k=1, axes=(2, 3))
        if t >= 4:
            b = np.flip(b, axis=3)
        b = b.copy()

        place_bool = actions == ACTION_PLACE

        m = legal_masks.copy()
        m[place_bool] = legal_masks[place_bool][:, perm_inv]

        s = soft_targets.copy()
        s[place_bool] = soft_targets[place_bool][:, perm_inv]

        lbl = labels.copy()
        lbl[place_bool] = perm_fwd[lbl[place_bool]]

        out_b.append(b)
        out_aux.append(aux)  # rotation-invariant
        out_lbl.append(lbl)
        out_act.append(actions)
        out_msk.append(m)
        out_soft.append(s)

    return (
        np.concatenate(out_b, axis=0),
        np.concatenate(out_aux, axis=0),
        np.concatenate(out_lbl, axis=0),
        np.concatenate(out_act, axis=0),
        np.concatenate(out_msk, axis=0),
        np.concatenate(out_soft, axis=0),
    )


# ── dataset ────────────────────────────────────────────────────────────────────


class QuartoDataset(Dataset):
    def __init__(self, boards, aux, labels, actions, legal_masks, soft_targets):
        self.boards = torch.from_numpy(boards)
        self.aux = torch.from_numpy(aux)
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.actions = torch.from_numpy(actions.astype(np.int64))
        self.legal_masks = torch.from_numpy(legal_masks.astype(np.bool_))
        self.soft_targets = torch.from_numpy(soft_targets.astype(np.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.boards[idx],
            self.aux[idx],
            self.labels[idx],
            self.actions[idx],
            self.legal_masks[idx],
            self.soft_targets[idx],
        )


def _load_npz_list(npz_paths: list[Path]):
    boards_l, aux_l, labels_l, actions_l, masks_l, soft_l, gids_l = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    gid_offset = 0
    for p in npz_paths:
        d = np.load(p)
        gids = d["game_ids"].astype(np.int64) + gid_offset
        gid_offset = int(gids.max()) + 1
        boards_l.append(d["boards"])
        aux_l.append(d["aux"])
        labels_l.append(d["labels"])
        actions_l.append(d["actions"])
        masks_l.append(d["legal_masks"])
        soft_l.append(d["soft_targets"])
        gids_l.append(gids.astype(np.int32))
        print(f"  loaded {p}  N={len(d['labels']):,}")
    return (
        np.concatenate(boards_l, axis=0),
        np.concatenate(aux_l, axis=0),
        np.concatenate(labels_l, axis=0),
        np.concatenate(actions_l, axis=0),
        np.concatenate(masks_l, axis=0),
        np.concatenate(soft_l, axis=0),
        np.concatenate(gids_l, axis=0),
    )


def load_split(npz_paths: list[Path], val_split: float, seed: int):
    boards, aux, labels, actions, legal_masks, soft_targets, game_ids = _load_npz_list(
        npz_paths
    )

    unique_games = np.unique(game_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_games)

    n_val = max(1, int(len(unique_games) * val_split))
    val_games = set(unique_games[:n_val].tolist())
    train_games = set(unique_games[n_val:].tolist())

    tr_idx = np.where(np.isin(game_ids, list(train_games)))[0]
    va_idx = np.where(np.isin(game_ids, list(val_games)))[0]

    def subset(idx):
        return (
            boards[idx],
            aux[idx],
            labels[idx],
            actions[idx],
            legal_masks[idx],
            soft_targets[idx],
        )

    tr_data = augment_symmetries(*subset(tr_idx))
    va_data = subset(va_idx)
    print(
        f"Symmetry augmentation: {len(subset(tr_idx)[0]):,} → {len(tr_data[0]):,} train samples"
    )
    return QuartoDataset(*tr_data), QuartoDataset(*va_data)


# ── masked soft/hard cross-entropy ────────────────────────────────────────────


def _masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """log_softmax with illegal positions clamped to -1e9 (avoids -inf · 0 NaNs)."""
    masked = logits.masked_fill(~mask, -1e9)
    return F.log_softmax(masked, dim=1)


def soft_masked_ce(
    logits: torch.Tensor,
    soft_target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy with a soft target distribution; illegal positions masked.

    soft_target is expected to have zero mass on illegal positions, so the
    product ``soft_target * log_p`` cannot pull on the masked tail.
    """
    log_p = _masked_log_softmax(logits, mask)
    return -(soft_target * log_p).sum(dim=1).mean()


def hard_masked_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Standard CE with illegal-position mask, computed via the same path
    as ``soft_masked_ce`` so the two are numerically comparable.
    """
    log_p = _masked_log_softmax(logits, mask)
    return F.nll_loss(log_p, targets)


# ── win-rate evaluation ────────────────────────────────────────────────────────

EVAL_BASELINES = [
    ("random", lambda: RandomBot()),
    ("minimax_d2", lambda: MinimaxBot(depth=2)),
]


def run_win_rate_eval(
    model: QuartoCNNAutoregUnified,
    n_matches: int,
    mode_2x2: bool,
) -> dict[str, float]:
    player = UnifiedBot(model=model, deterministic=True, temperature=0.1)
    win_rates: dict[str, float] = {}

    for rival_name, rival_factory in EVAL_BASELINES:
        rival = rival_factory()
        wins = losses = draws = 0

        _, stats = play_games(
            matches=n_matches // 2,
            player1=player,
            player2=rival,
            verbose=False,
            save_match=False,
            mode_2x2=mode_2x2,
            PROGRESS_MESSAGE="",
        )
        wins += stats["Player 1"]
        losses += stats["Player 2"]
        draws += stats["Tie"]

        rival = rival_factory()
        _, stats = play_games(
            matches=n_matches // 2,
            player1=rival,
            player2=player,
            verbose=False,
            save_match=False,
            mode_2x2=mode_2x2,
            PROGRESS_MESSAGE="",
        )
        wins += stats["Player 2"]
        losses += stats["Player 1"]
        draws += stats["Tie"]

        total = wins + losses + draws
        win_rates[rival_name] = (
            (wins + draws * 0.5) / total if total > 0 else float("nan")
        )

    return win_rates


# ── one epoch ─────────────────────────────────────────────────────────────────


def _topk_correct(logits: torch.Tensor, targets: torch.Tensor, k: int = 3) -> int:
    topk = logits.topk(k, dim=1).indices
    return (topk == targets.unsqueeze(1)).any(dim=1).sum().item()


def run_epoch(
    model: QuartoCNNAutoregUnified,
    loader: DataLoader,
    device: torch.device,
    optimizer=None,
    lam: float = 1.0,
    soft_weight: float = 1.0,
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    place_correct = place_top3 = place_total = 0
    sel_correct = sel_top3 = sel_total = 0

    with torch.set_grad_enabled(training):
        for boards, aux, labels, actions, masks, soft_targets in loader:
            boards = boards.to(device)
            aux = aux.to(device)
            labels = labels.to(device)
            actions = actions.to(device)
            masks = masks.to(device)
            soft_targets = soft_targets.to(device)

            logits_place, logits_select = raw_logits(model, boards, aux)

            place_idx = actions == ACTION_PLACE
            select_idx = actions == ACTION_SELECT

            loss = torch.tensor(0.0, device=device)

            if place_idx.any():
                lb = logits_place[place_idx]
                lp_lbl = labels[place_idx]
                lp_msk = masks[place_idx]
                lp_soft = soft_targets[place_idx]
                loss_p = (
                    soft_weight * soft_masked_ce(lb, lp_soft, lp_msk)
                    + (1.0 - soft_weight) * hard_masked_ce(lb, lp_lbl, lp_msk)
                )
                loss = loss + loss_p
                place_correct += (lb.argmax(dim=1) == lp_lbl).sum().item()
                place_top3 += _topk_correct(lb, lp_lbl, k=3)
                place_total += place_idx.sum().item()

            if select_idx.any():
                ls = logits_select[select_idx]
                ls_lbl = labels[select_idx]
                ls_msk = masks[select_idx]
                ls_soft = soft_targets[select_idx]
                loss_s = (
                    soft_weight * soft_masked_ce(ls, ls_soft, ls_msk)
                    + (1.0 - soft_weight) * hard_masked_ce(ls, ls_lbl, ls_msk)
                )
                loss = loss + lam * loss_s
                sel_correct += (ls.argmax(dim=1) == ls_lbl).sum().item()
                sel_top3 += _topk_correct(ls, ls_lbl, k=3)
                sel_total += select_idx.sum().item()

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)

    n = len(loader.dataset)
    return {
        "loss": total_loss / n,
        "place_acc": place_correct / place_total if place_total else float("nan"),
        "place_top3": place_top3 / place_total if place_total else float("nan"),
        "select_acc": sel_correct / sel_total if sel_total else float("nan"),
        "select_top3": sel_top3 / sel_total if sel_total else float("nan"),
    }


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    args = docopt(__doc__)
    epochs = int(args["--epochs"])
    batch = int(args["--batch"])
    lr = float(args["--lr"])
    val_spl = float(args["--val-split"])
    lam = float(args["--lam"])
    soft_weight = float(args["--soft-weight"])
    assert 0.0 <= soft_weight <= 1.0, "--soft-weight must be in [0, 1]"
    seed = int(args["--seed"])
    n_matches_eval = int(args["--n-matches-eval"])
    do_eval = not args["--no-eval"]
    data_paths = [Path(s.strip()) for s in args["--data"].split(",") if s.strip()]
    exp_name = args["--exp"]
    if exp_name:
        out_dir = Path("projects/supervised-cloning/experiments") / exp_name
    else:
        out_dir = Path(args["--out"])

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Data        : {[str(p) for p in data_paths]}")
    print(f"Output      : {out_dir}")
    print(
        f"Epochs      : {epochs}  |  Batch: {batch}  |  LR: {lr}  "
        f"|  lam: {lam}  |  soft_weight: {soft_weight}\n"
    )

    train_ds, val_ds = load_split(data_paths, val_spl, seed)
    print(f"Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}\n")

    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0)

    model = QuartoCNNAutoregUnified().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_place_acc": [],
        "val_place_acc": [],
        "train_place_top3": [],
        "val_place_top3": [],
        "train_sel_acc": [],
        "val_sel_acc": [],
        "train_sel_top3": [],
        "val_sel_top3": [],
    }

    best_val_acc = -1.0
    best_path = out_dir / "best.pt"

    for epoch in tqdm(range(1, epochs + 1), desc="Training", unit="epoch"):
        tr = run_epoch(model, train_dl, device, optimizer, lam, soft_weight)
        va = run_epoch(model, val_dl, device, lam=lam, soft_weight=soft_weight)
        scheduler.step()

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["train_place_acc"].append(tr["place_acc"])
        history["val_place_acc"].append(va["place_acc"])
        history["train_place_top3"].append(tr["place_top3"])
        history["val_place_top3"].append(va["place_top3"])
        history["train_sel_acc"].append(tr["select_acc"])
        history["val_sel_acc"].append(va["select_acc"])
        history["train_sel_top3"].append(tr["select_top3"])
        history["val_sel_top3"].append(va["select_top3"])

        val_acc = (va["place_acc"] + va["select_acc"]) / 2
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

        if epoch % 10 == 0 or epoch == 1:
            tqdm.write(
                f"Ep {epoch:4d}  "
                f"loss={tr['loss']:.4f}/{va['loss']:.4f}  "
                f"place={tr['place_acc']:.2%}(top3={tr['place_top3']:.2%})"
                f"/{va['place_acc']:.2%}(top3={va['place_top3']:.2%})  "
                f"sel={tr['select_acc']:.2%}(top3={tr['select_top3']:.2%})"
                f"/{va['select_acc']:.2%}(top3={va['select_top3']:.2%})"
            )

    final_path = out_dir / "final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nBest val accuracy : {best_val_acc:.2%}  →  {best_path}")
    print(f"Final checkpoint  : {final_path}")

    win_rates: dict[str, float] = {}
    if do_eval:
        print(
            f"\nEvaluating best checkpoint against baselines ({n_matches_eval} matches each)…"
        )
        eval_model = QuartoCNNAutoregUnified().to(device)
        eval_model.load_state_dict(torch.load(best_path, map_location=device))
        eval_model.eval()
        win_rates = run_win_rate_eval(eval_model, n_matches_eval, mode_2x2=True)
        print("\nWin-rate results (best checkpoint):")
        for rival_name, wr in win_rates.items():
            print(f"  vs {rival_name}: {wr:.2%}")

    # ── training curves ────────────────────────────────────────────────────────
    epochs_range = range(1, epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(epochs_range, history["train_loss"], label="train")
    ax.plot(epochs_range, history["val_loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Soft-target CE Loss")
    ax.legend()

    ax2 = axes[1]
    ax2.plot(epochs_range, history["train_place_acc"], label="tr PLACE-1", color="tab:blue")
    ax2.plot(epochs_range, history["val_place_acc"], label="va PLACE-1", color="tab:blue", linestyle="--")
    ax2.plot(epochs_range, history["train_place_top3"], label="tr PLACE-3", color="tab:cyan")
    ax2.plot(epochs_range, history["val_place_top3"], label="va PLACE-3", color="tab:cyan", linestyle="--")
    ax2.plot(epochs_range, history["train_sel_acc"], label="tr SEL-1", color="tab:orange")
    ax2.plot(epochs_range, history["val_sel_acc"], label="va SEL-1", color="tab:orange", linestyle="--")
    ax2.plot(epochs_range, history["train_sel_top3"], label="tr SEL-3", color="tab:red")
    ax2.plot(epochs_range, history["val_sel_top3"], label="va SEL-3", color="tab:red", linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy per Head (top-1 and top-3)")
    ax2.legend(fontsize=7)

    plt.tight_layout()
    plot_path = out_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=120)
    print(f"Training curves   : {plot_path}")

    # ── write summary .md ─────────────────────────────────────────────────────
    from datetime import datetime

    final_tr = history["train_loss"][-1]
    final_va = history["val_loss"][-1]
    best_ep = (
        int(
            np.argmax(
                [
                    (p + s) / 2
                    for p, s in zip(history["val_place_acc"], history["val_sel_acc"])
                ]
            )
        )
        + 1
    )
    md_lines = [
        f"# Training Summary — {exp_name or out_dir.name}",
        f"",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"",
        f"## Config",
        f"| Key | Value |",
        f"|-----|-------|",
        f"| model | QuartoCNNAutoregUnified |",
        f"| data | {', '.join(f'`{p}`' for p in data_paths)} |",
        f"| epochs | {epochs} |",
        f"| batch | {batch} |",
        f"| lr | {lr} |",
        f"| λ (select weight) | {lam} |",
        f"| soft_weight | {soft_weight} |",
        f"| val_split | {val_spl} |",
        f"| seed | {seed} |",
        f"| device | {device} |",
        f"",
        f"## Dataset",
        f"| | Samples |",
        f"|--|--|",
        f"| train (after aug) | {len(train_ds):,} |",
        f"| val (no aug) | {len(val_ds):,} |",
        f"",
        f"## Results",
        f"| Metric | Train | Val |",
        f"|--------|-------|-----|",
        f"| Best epoch | — | {best_ep} |",
        f"| Best val acc (avg heads) | — | {best_val_acc:.2%} |",
        f"| Final loss | {final_tr:.4f} | {final_va:.4f} |",
        f"| PLACE top-1 (final) | {history['train_place_acc'][-1]:.2%} | {history['val_place_acc'][-1]:.2%} |",
        f"| PLACE top-3 (final) | {history['train_place_top3'][-1]:.2%} | {history['val_place_top3'][-1]:.2%} |",
        f"| SELECT top-1 (final) | {history['train_sel_acc'][-1]:.2%} | {history['val_sel_acc'][-1]:.2%} |",
        f"| SELECT top-3 (final) | {history['train_sel_top3'][-1]:.2%} | {history['val_sel_top3'][-1]:.2%} |",
        f"",
    ]
    if win_rates:
        md_lines += [
            f"## Win-rate evaluation (best checkpoint, {n_matches_eval} matches each)",
            f"| Baseline | Win rate |",
            f"|----------|----------|",
        ]
        for rival_name, wr in win_rates.items():
            md_lines.append(f"| {rival_name} | {wr:.2%} |")
        md_lines.append(f"")
    md_lines += [
        f"![Training curves](training_curves.png)",
        f"",
        f"## Notes",
        f"",
    ]
    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Summary           : {md_path}")


if __name__ == "__main__":
    main()
