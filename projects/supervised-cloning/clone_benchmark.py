"""clone_benchmark.py – Evaluate a supervised-cloning experiment against standard baselines.

Loads the best (or final) checkpoint from a supervised-cloning experiment and
plays it against RandomBot and MinimaxBot, reporting win rates.

Usage:
    clone_benchmark.py <experiment> [options]
    clone_benchmark.py -h | --help

Arguments:
    <experiment>          Experiment name (folder under
                          projects/supervised-cloning/experiments/) OR an
                          absolute / relative path to a .pt weights file.

Options:
    --checkpoint <ckpt>   Which checkpoint to load: "best" or "final".
                          Ignored when <experiment> is a direct .pt path.
                          [default: best]
    --matches <int>       Matches per baseline (split evenly as P1 / P2).
                          [default: 100]
    --no-2x2              Disable 2×2 square win detection (default: enabled).
    -h, --help            Show this help.

Examples:
    python projects/supervised-cloning/clone_benchmark.py A1_baseline_cnn
    python projects/supervised-cloning/clone_benchmark.py A1_baseline_cnn --matches 200
    python projects/supervised-cloning/clone_benchmark.py A1_baseline_cnn --checkpoint final
    python projects/supervised-cloning/clone_benchmark.py path/to/weights.pt
"""

from __future__ import annotations

import os
import sys
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
import torch.nn.functional as F
from docopt import docopt
from quartopy import play_games

from bot.CNN_bot import Quarto_bot
from bot.random_bot import Quarto_bot as RandomBot
from bot.minimax_bot import MinimaxBot
from models.CNN1 import QuartoCNN
from models.CNN_uncoupled import QuartoCNN as QuartoCNN_uncoupled

# ── model ─────────────────────────────────────────────────────────────────────


class QuartoCNNLogits(QuartoCNN):
    """QuartoCNN that returns raw logits (pre-tanh).  Matches train.py exactly."""

    @property
    def name(self) -> str:
        return "QuartoCNNLogits"

    def forward(self, x_board, x_piece):
        piece_feat = F.relu(self.fc_in_piece(x_piece))
        piece_map = piece_feat.view(-1, 1, 4, 4)
        x = torch.cat([x_board, piece_map], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits_board = self.fc2_board(x)
        qav_board = torch.tanh(logits_board)
        x_qav = torch.cat([x, qav_board], dim=1)
        logits_piece = self.fc2_piece(x_qav)
        return logits_board, logits_piece


# ── baselines ─────────────────────────────────────────────────────────────────
#: List of (display_name, factory_fn).  Add or remove entries here.
_LOSS_BT_PATH = (
    "CHECKPOINTS/LOSS_APPROACHs_1212-2_only_select"
    "/20251212_2206-LOSS_APPROACHs_1212-2_only_select_E_1034.pt"
)
BASELINES: list[tuple[str, callable]] = [
    ("random", lambda: RandomBot()),
    ("minimax_d2", lambda: MinimaxBot(depth=2)),
    (
        "loss-BT",
        lambda: Quarto_bot(
            model_path=_LOSS_BT_PATH,
            model_class=QuartoCNN_uncoupled,
            deterministic=False,
            temperature=0.1,
        ),
    ),
]


# ── evaluation ────────────────────────────────────────────────────────────────


def run_benchmark(
    weights_path: Path,
    n_matches: int,
    mode_2x2: bool,
) -> dict[str, float]:
    """Load weights and play against every baseline.  Returns win rates."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = QuartoCNNLogits()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    player = Quarto_bot(model=model, deterministic=False, temperature=0.1)

    win_rates: dict[str, float] = {}
    for rival_name, rival_factory in BASELINES:
        wins = losses = draws = 0

        # play as P1
        rival = rival_factory()
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

        # play as P2
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


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = docopt(__doc__)

    exp_arg = args["<experiment>"]
    checkpoint = args["--checkpoint"]
    n_matches = int(args["--matches"])
    mode_2x2 = not args["--no-2x2"]

    # Resolve weights path
    exp_path = Path(exp_arg)
    if exp_path.suffix == ".pt":
        # Direct path to a .pt file
        weights_path = exp_path
        if not weights_path.is_absolute():
            weights_path = _root / weights_path
    else:
        # Experiment name → look under experiments/
        exp_dir = _root / "projects" / "supervised-cloning" / "experiments" / exp_arg
        if not exp_dir.is_dir():
            print(f"Error: experiment directory not found: {exp_dir}", file=sys.stderr)
            sys.exit(1)
        weights_path = exp_dir / f"{checkpoint}.pt"

    if not weights_path.is_file():
        print(f"Error: checkpoint not found: {weights_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Checkpoint : {weights_path}")
    print(f"Matches    : {n_matches} per baseline  (mode_2x2={mode_2x2})")
    print(f"Baselines  : {[name for name, _ in BASELINES]}\n")

    win_rates = run_benchmark(weights_path, n_matches, mode_2x2)

    col_w = max(len(n) for n, _ in BASELINES) + 2
    print(f"{'Baseline':<{col_w}}  Win rate")
    print("-" * (col_w + 12))
    for rival_name, wr in win_rates.items():
        bar = "█" * int(wr * 20)
        print(f"{rival_name:<{col_w}}  {wr:6.2%}  {bar}")


if __name__ == "__main__":
    main()
