"""train.py – Train a supervised clone of MinimaxBot using collected game data.

Uses the existing QuartoCNN architecture.  Both the PLACE head (board position)
and the SELECT head (piece to give) are trained jointly with cross-entropy loss.
Illegal moves are masked out from the logits before computing the loss.

Train/val split is done at the game level (not sample level) to avoid data
leakage between consecutive positions of the same game.

Usage:
    train.py [options]

Options:
    --data <path>         Input .npz file   [default: projects/supervised-cloning/data/collected.npz]
    --out <path>          Checkpoint dir    [default: projects/supervised-cloning/checkpoints]
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
ACTION_PLACE  = 0
ACTION_SELECT = 1


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
        piece_map  = piece_feat.view(-1, 1, 4, 4)
        x = torch.cat([x_board, piece_map], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        logits_board = self.fc2_board(x)                           # (B, 16)
        qav_board    = torch.tanh(logits_board)                    # for piece-head context
        x_qav        = torch.cat([x, qav_board], dim=1)
        logits_piece = self.fc2_piece(x_qav)                       # (B, 16)

        return logits_board, logits_piece                          # raw logits


# ── dataset ────────────────────────────────────────────────────────────────────

class QuartoDataset(Dataset):
    def __init__(self, boards, pieces, labels, actions, legal_masks):
        self.boards      = torch.from_numpy(boards)
        self.pieces      = torch.from_numpy(pieces)
        self.labels      = torch.from_numpy(labels.astype(np.int64))
        self.actions     = torch.from_numpy(actions.astype(np.int64))
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

    n_val  = max(1, int(len(unique_games) * val_split))
    val_games  = set(unique_games[:n_val].tolist())
    train_games = set(unique_games[n_val:].tolist())

    tr_idx = np.where(np.isin(game_ids, list(train_games)))[0]
    va_idx = np.where(np.isin(game_ids, list(val_games)))[0]

    def subset(idx):
        return (boards[idx], pieces[idx], labels[idx],
                actions[idx], legal_masks[idx])

    return QuartoDataset(*subset(tr_idx)), QuartoDataset(*subset(va_idx))


# ── masked cross-entropy ───────────────────────────────────────────────────────

def masked_ce(logits: torch.Tensor, targets: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with illegal moves masked to -inf."""
    logits = logits.clone()
    logits[~mask] = float("-inf")
    return F.cross_entropy(logits, targets)


# ── one epoch ─────────────────────────────────────────────────────────────────

def run_epoch(model, loader, device, optimizer=None, lam=1.0):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    place_correct = place_total = 0
    sel_correct   = sel_total   = 0

    with torch.set_grad_enabled(training):
        for boards, pieces, labels, actions, masks in loader:
            boards  = boards.to(device)
            pieces  = pieces.to(device)
            labels  = labels.to(device)
            actions = actions.to(device)
            masks   = masks.to(device)

            logits_board, logits_piece = model(boards, pieces)

            place_idx  = (actions == ACTION_PLACE)
            select_idx = (actions == ACTION_SELECT)

            loss = torch.tensor(0.0, device=device)

            if place_idx.any():
                loss_place = masked_ce(
                    logits_board[place_idx],
                    labels[place_idx],
                    masks[place_idx],
                )
                loss = loss + loss_place
                preds = logits_board[place_idx].argmax(dim=1)
                place_correct += (preds == labels[place_idx]).sum().item()
                place_total   += place_idx.sum().item()

            if select_idx.any():
                loss_select = masked_ce(
                    logits_piece[select_idx],
                    labels[select_idx],
                    masks[select_idx],
                )
                loss = loss + lam * loss_select
                preds = logits_piece[select_idx].argmax(dim=1)
                sel_correct += (preds == labels[select_idx]).sum().item()
                sel_total   += select_idx.sum().item()

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)

    n = len(loader.dataset)
    return {
        "loss":       total_loss / n,
        "place_acc":  place_correct / place_total  if place_total  else float("nan"),
        "select_acc": sel_correct   / sel_total    if sel_total    else float("nan"),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args    = docopt(__doc__)
    epochs  = int(args["--epochs"])
    batch   = int(args["--batch"])
    lr      = float(args["--lr"])
    val_spl = float(args["--val-split"])
    lam     = float(args["--lam"])
    seed    = int(args["--seed"])
    data_p  = Path(args["--data"])
    out_dir = Path(args["--out"])

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device   : {device}")
    print(f"Data     : {data_p}")
    print(f"Output   : {out_dir}")
    print(f"Epochs   : {epochs}  |  Batch: {batch}  |  LR: {lr}  |  λ: {lam}\n")

    train_ds, val_ds = load_split(data_p, val_spl, seed)
    print(f"Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}\n")

    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0)

    model     = QuartoCNNLogits().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [],
               "train_place_acc": [], "val_place_acc": [],
               "train_sel_acc":   [], "val_sel_acc":   []}

    best_val_acc = -1.0
    best_path    = out_dir / "best.pt"

    for epoch in tqdm(range(1, epochs + 1), desc="Training", unit="epoch"):
        tr = run_epoch(model, train_dl, device, optimizer, lam)
        va = run_epoch(model, val_dl,   device, lam=lam)
        scheduler.step()

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["train_place_acc"].append(tr["place_acc"])
        history["val_place_acc"].append(va["place_acc"])
        history["train_sel_acc"].append(tr["select_acc"])
        history["val_sel_acc"].append(va["select_acc"])

        # combined val accuracy (simple average of both heads)
        val_acc = (va["place_acc"] + va["select_acc"]) / 2

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

        if epoch % 10 == 0 or epoch == 1:
            tqdm.write(
                f"Ep {epoch:4d}  "
                f"loss={tr['loss']:.4f}/{va['loss']:.4f}  "
                f"place_acc={tr['place_acc']:.2%}/{va['place_acc']:.2%}  "
                f"sel_acc={tr['select_acc']:.2%}/{va['select_acc']:.2%}"
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
    ax.plot(epochs_range, history["val_loss"],   label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Cross-Entropy Loss")
    ax.legend()

    ax2 = axes[1]
    ax2.plot(epochs_range, history["train_place_acc"], label="train PLACE")
    ax2.plot(epochs_range, history["val_place_acc"],   label="val PLACE")
    ax2.plot(epochs_range, history["train_sel_acc"],   label="train SELECT", linestyle="--")
    ax2.plot(epochs_range, history["val_sel_acc"],     label="val SELECT",   linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Top-1 Accuracy")
    ax2.set_title("Accuracy per Head")
    ax2.legend()

    plt.tight_layout()
    plot_path = out_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=120)
    print(f"Training curves   : {plot_path}")


if __name__ == "__main__":
    main()
