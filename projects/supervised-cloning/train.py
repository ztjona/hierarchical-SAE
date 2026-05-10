"""train.py – Train a supervised clone of MinimaxBot using collected game data.

Uses the existing QuartoCNN architecture.  Both the PLACE head (board position)
and the SELECT head (piece to give) are trained jointly with cross-entropy loss.
Illegal moves are masked out from the logits before computing the loss.

Train/val split is done at the game level (not sample level) to avoid data
leakage between consecutive positions of the same game.

Usage:
    train.py [options]

Options:
    --data <path>         Input .npz file   [default: projects/supervised-cloning/data/collected_5k.npz]
    --exp <name>          Experiment name, e.g. A1_baseline_cnn.  Output goes to
                          projects/supervised-cloning/experiments/<name>/
                          Overrides --out when provided.
    --out <path>          Checkpoint dir (ignored when --exp is set)
                          [default: projects/supervised-cloning/checkpoints]
    --epochs <int>        Training epochs   [default: 150]
    --batch <int>         Batch size        [default: 256]
    --lr <float>          Learning rate     [default: 1e-3]
    --val-split <float>   Val fraction      [default: 0.15]
    --lam <float>         SELECT loss weight relative to PLACE  [default: 1.0]
    --seed <int>          Random seed       [default: 42]
    -h, --help            Show this help.

Examples:
    python train.py
    python train.py --epochs 200 --lr 5e-4 --out projects/supervised-cloning/checkpoints
"""

from __future__ import annotations

import os
import sys
import random
import numpy as np
from pathlib import Path

# ── resolve project root ───────────────────────────────────────────────────────
_here = Path(__file__).resolve().parent
_root = _here
while not (_root / "bot").is_dir():
    _root = _root.parent
    if _root == _root.parent:
        raise RuntimeError("Could not find project root containing 'bot/'")

os.chdir(_root)
sys.path.insert(0, str(_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from docopt import docopt

from models.CNN1 import QuartoCNN

# ── action constants ───────────────────────────────────────────────────────────
ACTION_PLACE = 0
ACTION_SELECT = 1


# ── board symmetry augmentation ───────────────────────────────────────────────
# The 4x4 Quarto board has 8 symmetries (dihedral group D4):
# 4 rotations × 2 reflections.  Each transforms both the board tensor
# (16,4,4) and the PLACE label (position index).  SELECT labels are
# piece indices and are unaffected by board geometry.


def _rot90_board(board: np.ndarray) -> np.ndarray:
    """Rotate board (16,4,4) by 90° counter-clockwise."""
    return np.rot90(board, k=1, axes=(1, 2)).copy()


def _flip_board(board: np.ndarray) -> np.ndarray:
    """Flip board (16,4,4) horizontally (left-right)."""
    return np.flip(board, axis=2).copy()


def _pos_inv(idx: int, transform_id: int) -> int:
    """Inverse map: given a NEW position index, return the ORIGINAL position.

    numpy rot90(k=1, axes=(2,3)) rotates CCW, so the inverse is CW:
      new (r,c) came from old (c, 3-r).
    Used with gather indexing for masks: new_mask[i] = old_mask[perm_inv[i]].
    """
    r, c = divmod(idx, 4)
    for _ in range(transform_id % 4):
        r, c = c, 3 - r  # CW: inverse of CCW
    if transform_id >= 4:
        c = 3 - c  # flip is self-inverse
    return r * 4 + c


# Inverse permutation tables (new_pos -> old_pos): used for masks (gather).
_POS_PERMS_INV: list[np.ndarray] = [
    np.array([_pos_inv(i, t) for i in range(16)], dtype=np.int64) for t in range(8)
]
# Forward permutation tables (old_pos -> new_pos): used for PLACE labels.
_POS_PERMS_FWD: list[np.ndarray] = [
    np.argsort(p).astype(np.int64) for p in _POS_PERMS_INV
]


def augment_symmetries(
    boards: np.ndarray,
    pieces: np.ndarray,
    labels: np.ndarray,
    actions: np.ndarray,
    legal_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand the dataset 8× by applying all dihedral symmetries.

    SELECT samples (action==1): board and legal_mask are transformed;
        label (piece index) is unchanged.
    PLACE samples (action==0): board, legal_mask, AND label are transformed
        via the same permutation.
    """
    aug_boards, aug_pieces, aug_labels, aug_actions, aug_masks = [], [], [], [], []

    for t in range(8):
        perm_inv = _POS_PERMS_INV[t]  # new_pos → old_pos  (for masks)
        perm_fwd = _POS_PERMS_FWD[t]  # old_pos → new_pos  (for labels)

        # Transform boards: rotate / flip the spatial dims (CCW rotation)
        b = boards  # (N, 16, 4, 4)
        for _ in range(t % 4):
            b = np.rot90(b, k=1, axes=(2, 3))
        if t >= 4:
            b = np.flip(b, axis=3)
        b = b.copy()

        # Mask transform: only PLACE masks are board-position-based.
        # SELECT masks are piece-availability masks — board rotation doesn't change them.
        m = legal_masks.copy()
        place_bool = actions == ACTION_PLACE
        m[place_bool] = legal_masks[place_bool][:, perm_inv]  # (N_place, 16)

        # PLACE labels: forward map — new_label = perm_fwd[old_label]
        lbl = labels.copy()
        place_mask = actions == ACTION_PLACE
        lbl[place_mask] = perm_fwd[lbl[place_mask]]

        aug_boards.append(b)
        aug_pieces.append(pieces)
        aug_labels.append(lbl)
        aug_actions.append(actions)
        aug_masks.append(m)

    return (
        np.concatenate(aug_boards, axis=0),
        np.concatenate(aug_pieces, axis=0),
        np.concatenate(aug_labels, axis=0),
        np.concatenate(aug_actions, axis=0),
        np.concatenate(aug_masks, axis=0),
    )


# ── model wrapper ──────────────────────────────────────────────────────────────


class QuartoCNNLogits(QuartoCNN):
    """QuartoCNN that returns raw logits (pre-tanh) for supervised CE loss.

    Architecture is identical to QuartoCNN.  The only difference is that
    forward() returns (logits_board, logits_piece) instead of tanh'd values.
    The piece head still receives the tanh'd board output as context input,
    preserving the sequential dependency from the original design.
    """

    @property
    def name(self) -> str:
        return "QuartoCNNLogits"

    def forward(
        self,
        x_board: torch.Tensor,
        x_piece: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        piece_feat = F.relu(self.fc_in_piece(x_piece))
        piece_map = piece_feat.view(-1, 1, 4, 4)
        x = torch.cat([x_board, piece_map], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        logits_board = self.fc2_board(x)  # (B, 16)
        qav_board = torch.tanh(logits_board)  # for piece-head context
        x_qav = torch.cat([x, qav_board], dim=1)
        logits_piece = self.fc2_piece(x_qav)  # (B, 16)

        return logits_board, logits_piece  # raw logits


# ── dataset ────────────────────────────────────────────────────────────────────


class QuartoDataset(Dataset):
    def __init__(self, boards, pieces, labels, actions, legal_masks):
        self.boards = torch.from_numpy(boards)
        self.pieces = torch.from_numpy(pieces)
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.actions = torch.from_numpy(actions.astype(np.int64))
        self.legal_masks = torch.from_numpy(legal_masks)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.boards[idx],
            self.pieces[idx],
            self.labels[idx],
            self.actions[idx],
            self.legal_masks[idx],
        )


def load_split(npz_path: Path, val_split: float, seed: int):
    """Load .npz and split by game_id to avoid leakage."""
    d = np.load(npz_path)
    boards, pieces, labels = d["boards"], d["pieces"], d["labels"]
    actions, legal_masks, game_ids = d["actions"], d["legal_masks"], d["game_ids"]

    unique_games = np.unique(game_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_games)

    n_val = max(1, int(len(unique_games) * val_split))
    val_games = set(unique_games[:n_val].tolist())
    train_games = set(unique_games[n_val:].tolist())

    tr_idx = np.where(np.isin(game_ids, list(train_games)))[0]
    va_idx = np.where(np.isin(game_ids, list(val_games)))[0]

    def subset(idx):
        return (boards[idx], pieces[idx], labels[idx], actions[idx], legal_masks[idx])

    # Augment only training split — val stays clean (original positions)
    tr_data = augment_symmetries(*subset(tr_idx))
    va_data = subset(va_idx)
    print(
        f"Symmetry augmentation: {len(subset(tr_idx)[0]):,} → {len(tr_data[0]):,} train samples"
    )
    return QuartoDataset(*tr_data), QuartoDataset(*va_data)


# ── masked cross-entropy ───────────────────────────────────────────────────────


def masked_ce(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy with illegal moves masked to -inf."""
    logits = logits.clone()
    logits[~mask] = float("-inf")
    return F.cross_entropy(logits, targets)


# ── one epoch ─────────────────────────────────────────────────────────────────


def _topk_correct(logits: torch.Tensor, targets: torch.Tensor, k: int = 3) -> int:
    """Count samples where the true label is among the top-k predictions."""
    topk = logits.topk(k, dim=1).indices  # (B, k)
    return (topk == targets.unsqueeze(1)).any(dim=1).sum().item()


def run_epoch(model, loader, device, optimizer=None, lam=1.0):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    place_correct = place_top3 = place_total = 0
    sel_correct = sel_top3 = sel_total = 0

    with torch.set_grad_enabled(training):
        for boards, pieces, labels, actions, masks in loader:
            boards = boards.to(device)
            pieces = pieces.to(device)
            labels = labels.to(device)
            actions = actions.to(device)
            masks = masks.to(device)

            logits_board, logits_piece = model(boards, pieces)

            place_idx = actions == ACTION_PLACE
            select_idx = actions == ACTION_SELECT

            loss = torch.tensor(0.0, device=device)

            if place_idx.any():
                lb = logits_board[place_idx]
                lp_lbl = labels[place_idx]
                lp_msk = masks[place_idx]
                loss_place = masked_ce(lb, lp_lbl, lp_msk)
                loss = loss + loss_place
                place_correct += (lb.argmax(dim=1) == lp_lbl).sum().item()
                place_top3 += _topk_correct(lb, lp_lbl, k=3)
                place_total += place_idx.sum().item()

            if select_idx.any():
                lp = logits_piece[select_idx]
                ls_lbl = labels[select_idx]
                ls_msk = masks[select_idx]
                loss_select = masked_ce(lp, ls_lbl, ls_msk)
                loss = loss + lam * loss_select
                sel_correct += (lp.argmax(dim=1) == ls_lbl).sum().item()
                sel_top3 += _topk_correct(lp, ls_lbl, k=3)
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
    seed = int(args["--seed"])
    data_p = Path(args["--data"])
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
    print(f"Device   : {device}")
    print(f"Data     : {data_p}")
    print(f"Output   : {out_dir}")
    print(f"Epochs   : {epochs}  |  Batch: {batch}  |  LR: {lr}  |  lam: {lam}\n")

    train_ds, val_ds = load_split(data_p, val_spl, seed)
    print(f"Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}\n")

    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0)

    model = QuartoCNNLogits().to(device)
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
        tr = run_epoch(model, train_dl, device, optimizer, lam)
        va = run_epoch(model, val_dl, device, lam=lam)
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

        # combined val accuracy (simple average of both heads)
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

    # ── save final checkpoint ──────────────────────────────────────────────────
    final_path = out_dir / "final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nBest val accuracy : {best_val_acc:.2%}  →  {best_path}")
    print(f"Final checkpoint  : {final_path}")

    # ── training curves ────────────────────────────────────────────────────────
    epochs_range = range(1, epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(epochs_range, history["train_loss"], label="train")
    ax.plot(epochs_range, history["val_loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Cross-Entropy Loss")
    ax.legend()

    ax2 = axes[1]
    ax2.plot(
        epochs_range,
        history["train_place_acc"],
        label="train PLACE top-1",
        color="tab:blue",
    )
    ax2.plot(
        epochs_range,
        history["val_place_acc"],
        label="val PLACE top-1",
        color="tab:blue",
        linestyle="--",
    )
    ax2.plot(
        epochs_range,
        history["train_place_top3"],
        label="train PLACE top-3",
        color="tab:cyan",
    )
    ax2.plot(
        epochs_range,
        history["val_place_top3"],
        label="val PLACE top-3",
        color="tab:cyan",
        linestyle="--",
    )
    ax2.plot(
        epochs_range,
        history["train_sel_acc"],
        label="train SELECT top-1",
        color="tab:orange",
    )
    ax2.plot(
        epochs_range,
        history["val_sel_acc"],
        label="val SELECT top-1",
        color="tab:orange",
        linestyle="--",
    )
    ax2.plot(
        epochs_range,
        history["train_sel_top3"],
        label="train SELECT top-3",
        color="tab:red",
    )
    ax2.plot(
        epochs_range,
        history["val_sel_top3"],
        label="val SELECT top-3",
        color="tab:red",
        linestyle="--",
    )
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
        f"| data | `{data_p}` |",
        f"| epochs | {epochs} |",
        f"| batch | {batch} |",
        f"| lr | {lr} |",
        f"| λ (select weight) | {lam} |",
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
