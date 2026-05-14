"""clone_benchmark.py – Evaluate a unified-aux supervised clone against baselines.

Loads the best (or final) checkpoint of a ``QuartoCNNAutoregUnified`` clone
and plays it against RandomBot, MinimaxBot(d=2), and the loss-BT CNN baseline.

Usage:
    clone_benchmark.py <experiment> [options]
    clone_benchmark.py -h | --help

Arguments:
    <experiment>          Experiment name (folder under
                          projects/supervised-cloning/experiments/) OR a
                          path to a .pt weights file.

Options:
    --checkpoint <ckpt>   "best" or "final". Ignored when <experiment> is a .pt path.
                          [default: best]
    --matches <int>       Matches per baseline (split evenly P1/P2).  [default: 100]
    --no-2x2              Disable 2×2 win detection (default: enabled).
    -h, --help            Show this help.

Examples:
    python projects/supervised-cloning/clone_benchmark.py C1_unifiedAux
    python projects/supervised-cloning/clone_benchmark.py C1_unifiedAux --matches 200
"""

from __future__ import annotations

import os
import sys
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
from docopt import docopt
from quartopy import play_games

from bot.CNN_bot import Quarto_bot as CNNBot
from bot.CNN_unified_bot import Quarto_bot as UnifiedBot
from bot.random_bot import Quarto_bot as RandomBot
from bot.minimax_bot import MinimaxBot
from models.CNN_autoreg import QuartoCNNAutoregUnified
from models.CNN_uncoupled import QuartoCNN as QuartoCNN_uncoupled

_LOSS_BT_PATH = (
    "CHECKPOINTS/LOSS_APPROACHs_1212-2_only_select"
    "/20251212_2206-LOSS_APPROACHs_1212-2_only_select_E_1034.pt"
)
BASELINES: list[tuple[str, callable]] = [
    ("random", lambda: RandomBot()),
    ("minimax_d2", lambda: MinimaxBot(depth=2)),
    (
        "loss-BT",
        lambda: CNNBot(
            model_path=_LOSS_BT_PATH,
            model_class=QuartoCNN_uncoupled,
            deterministic=False,
            temperature=0.1,
        ),
    ),
]


def run_benchmark(
    weights_path: Path,
    n_matches: int,
    mode_2x2: bool,
) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuartoCNNAutoregUnified()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    player = UnifiedBot(model=model, deterministic=False, temperature=0.01)

    win_rates: dict[str, float] = {}
    for rival_name, rival_factory in BASELINES:
        wins = losses = draws = 0

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


def main() -> None:
    args = docopt(__doc__)

    exp_arg = args["<experiment>"]
    checkpoint = args["--checkpoint"]
    n_matches = int(args["--matches"])
    mode_2x2 = not args["--no-2x2"]

    exp_path = Path(exp_arg)
    if exp_path.suffix == ".pt":
        weights_path = exp_path
        if not weights_path.is_absolute():
            weights_path = _root / weights_path
    else:
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
