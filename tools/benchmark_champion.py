# -*- coding: utf-8 -*-

"""Benchmark the current champion against previous champions and baselines.

Usage:
  benchmark_champion.py [--matches=<n>] [--output=<file>] [--opponents=<list>] [--verbose]
  benchmark_champion.py -h | --help

Options:
  --matches=<n>       Number of matches per direction [default: 500].
  --output=<file>     Output JSONL file path [default: champion-results.jsonl].
  --opponents=<list>  Comma-separated opponents to benchmark against
                      [default: random,loss_BT,Aa_replay,ME_endgame].
  --verbose           Print detailed output during games.
  -h --help           Show this screen.

Available opponents: random, loss_BT, Aa_replay, ME_endgame
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from docopt import docopt

# Ensure workspace root is on path when running from tools/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quartopy import play_games
from bot.CNN_bot import Quarto_bot
from bot.CNN_unified_bot import Quarto_bot as Quarto_unified_bot
from bot.CNN_autoreg_bot import Quarto_bot as Quarto_autoreg_bot

from models.CNN1 import QuartoCNN
from models.CNN_uncoupled import QuartoCNN as QuartoCNN_uncoupled
from models.CNN_autoreg import QuartoCNNAutoreg
from models.CNN_autoreg_sa import QuartoCNNAutoregUnifiedS4

# Checkpoint paths
CHECKPOINTS = {
    "champion": {
        "path": "CHECKPOINTS//Sa_archScan(3)0512_ARCH_S4_uniform512//20260514_0815-Sa_archScan(3)0512_ARCH_S4_uniform512_E_5000.pt",
        "class": QuartoCNNAutoregUnifiedS4,
        "bot_class": Quarto_unified_bot,
        "name": "Sa_archScan(3) [S4]",
    },
    "random": {
        "path": "CHECKPOINTS//EXP_id03//20250922_1247-EXP_id03_epoch_0000.pt",
        "class": QuartoCNN,
        "bot_class": Quarto_bot,
        "name": "Random Baseline",
    },
    "loss_BT": {
        "path": "CHECKPOINTS//LOSS_APPROACHs_1212-2_only_select//20251212_2206-LOSS_APPROACHs_1212-2_only_select_E_1034.pt",
        "class": QuartoCNN_uncoupled,
        "bot_class": Quarto_bot,
        "name": "Loss_BT",
    },
    "Aa_replay": {
        "path": "CHECKPOINTS//Aa_replay(2)0226_NUM_EPOCHs_BUFFER_8//20260227_1103-Aa_replay(2)0226_NUM_EPOCHs_BUFFER_8_E_5000.pt",
        "class": QuartoCNN_uncoupled,
        "bot_class": Quarto_bot,
        "name": "Aa_replay(2)",
    },
    "ME_endgame": {
        "path": ".//CHECKPOINTS//ME_endgame(2)0429_ENDGAME_FRACTION_0.5//20260507_0829-ME_endgame(2)0429_ENDGAME_FRACTION_0.5_E_5000.pt",
        "class": QuartoCNNAutoreg,
        "bot_class": Quarto_autoreg_bot,
        "name": "ME_endgame(2)",
    },
}


def load_bot(checkpoint_key: str) -> tuple:
    """Load a bot from checkpoint."""
    config = CHECKPOINTS[checkpoint_key]
    bot = config["bot_class"](
        model_path=config["path"],
        model_class=config["class"],
        deterministic=False,
        temperature=0.1,
    )
    return bot, config["name"]


def benchmark_champion(
    matches: int = 500,
    opponents: list = None,
    output_path: str = "champion-results.jsonl",
    verbose: bool = False,
) -> None:
    """
    Benchmark the champion against specified opponents.

    Args:
        matches: Number of matches to play in each direction
        opponents: List of opponent keys (default: all except champion)
        output_path: Path to save results
        verbose: Print detailed output
    """
    if opponents is None:
        opponents = ["random", "loss_BT", "Aa_replay", "ME_endgame"]

    # Load champion
    champion_bot, champion_name = load_bot("champion")
    print(f"\n{'='*60}")
    print(f"Benchmarking Champion: {champion_name}")
    print(f"{'='*60}")
    print(f"Number of matches per pairing: {matches}")
    print(f"Output: {output_path}")
    print(f"Opponents: {opponents}")
    print(f"{'='*60}\n")

    results = []

    # Benchmark against each opponent
    for opp_key in opponents:
        opponent_bot, opponent_name = load_bot(opp_key)

        print(f"Playing vs {opponent_name}...")

        # Champion as Player 1, Opponent as Player 2
        res_p1, wr_p1 = play_games(
            matches=matches,
            player1=champion_bot,
            player2=opponent_bot,
            verbose=False,
            save_match=False,
            mode_2x2=True,
        )

        # Champion as Player 2, Opponent as Player 1
        res_p2, wr_p2 = play_games(
            matches=matches,
            player1=opponent_bot,
            player2=champion_bot,
            verbose=False,
            save_match=False,
            mode_2x2=True,
        )

        # Aggregate results
        champion_wins_p1 = wr_p1["Player 1"]
        opponent_wins_p1 = wr_p1["Player 2"]
        champion_wins_p2 = wr_p2["Player 2"]
        opponent_wins_p2 = wr_p2["Player 1"]

        total_champion_wins = champion_wins_p1 + champion_wins_p2
        total_opponent_wins = opponent_wins_p1 + opponent_wins_p2
        total_games = total_champion_wins + total_opponent_wins
        win_rate = total_champion_wins / total_games if total_games > 0 else 0

        result = {
            "timestamp": datetime.now().isoformat(),
            "champion": champion_name,
            "opponent": opponent_name,
            "matches_per_direction": matches,
            "champion_as_p1_wins": int(champion_wins_p1),
            "opponent_as_p1_wins": int(opponent_wins_p1),
            "champion_as_p2_wins": int(champion_wins_p2),
            "opponent_as_p2_wins": int(opponent_wins_p2),
            "total_champion_wins": int(total_champion_wins),
            "total_opponent_wins": int(total_opponent_wins),
            "total_games": int(total_games),
            "champion_win_rate": float(win_rate),
        }

        results.append(result)

        # Print summary
        print(
            f"  P1 (Champion): {champion_wins_p1}/{matches} ({100*champion_wins_p1/matches:.1f}%)"
        )
        print(
            f"  P2 (Opponent): {opponent_wins_p1}/{matches} ({100*opponent_wins_p1/matches:.1f}%)"
        )
        print(
            f"  P1 (Opponent): {opponent_wins_p2}/{matches} ({100*opponent_wins_p2/matches:.1f}%)"
        )
        print(
            f"  P2 (Champion): {champion_wins_p2}/{matches} ({100*champion_wins_p2/matches:.1f}%)"
        )
        print(
            f"  Overall: {champion_name} wins {total_champion_wins}/{total_games} ({100*win_rate:.1f}%)"
        )
        print()

    # Save results to JSONL
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"{'='*60}")
    print(f"Results saved to: {output_file.absolute()}")
    print(f"Total matchups: {len(results)}")
    print(f"{'='*60}\n")

    # Print summary table
    print("SUMMARY:")
    print(f"{'Opponent':<25} {'Wins':<15} {'Win Rate':<12}")
    print("-" * 52)
    for result in results:
        opp_name = result["opponent"][:22]
        wins = f"{result['total_champion_wins']}/{result['total_games']}"
        wr = f"{100*result['champion_win_rate']:.1f}%"
        print(f"{opp_name:<25} {wins:<15} {wr:<12}")
    print()


def main():
    args = docopt(__doc__)

    matches = int(args["--matches"])
    output_path = args["--output"]
    opponents = [o.strip() for o in args["--opponents"].split(",")]
    verbose = args["--verbose"]

    # Validate opponents
    valid_opponents = set(CHECKPOINTS.keys()) - {"champion"}
    for opp in opponents:
        if opp not in valid_opponents:
            print(f"Error: Unknown opponent '{opp}'")
            print(f"Available opponents: {', '.join(sorted(valid_opponents))}")
            return

    benchmark_champion(
        matches=matches,
        opponents=opponents,
        output_path=output_path,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
