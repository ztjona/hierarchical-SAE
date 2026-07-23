# -*- coding: utf-8 -*-

"""Benchmark champion(s) against previous champions and baselines.

Single champion (default) writes one champion's row of the head-to-head matrix.
Pass --champions to run a full round-robin in one invocation (every listed
champion vs every other listed champion + the baseline opponents), which is what
the Atlas head-to-head matrix needs to be dense. See
CHAMPION-HEAD-TO-HEAD.md in the repo root.

Usage:
  benchmark_champion.py [options]
  benchmark_champion.py -h | --help

Options:
  --config=<file>      Path to champion config JSON. Defaults to champion_config.json
                       in the same directory as this script.
  --matches=<n>        Number of matches per direction [default: 500].
  --output=<file>      Output JSONL file path [default: champion-results.jsonl].
  --opponents=<list>   Comma-separated opponent keys from the config to benchmark against.
                       Defaults to the 'default_opponents' list in the config file.
  --champion=<key>     Override config["champion"] for this run (single champion).
  --round-robin=<list> Comma-separated champion keys to run as a full round-robin: each
                       plays every other listed champion plus the baseline opponents,
                       all written to one --output file. Overrides --champion/--opponents.
  --append             Append to --output instead of overwriting (implied by --round-robin).
  --verbose            Print detailed output during games.
  -h --help            Show this screen.
"""

# Examples (kept OUT of the docstring: docopt parses the docstring, and a
# continuation line starting with "--" is misread as an option definition):
#   Single champion (overwrites champion-results.jsonl):
#     python tools/benchmark_champion.py --champion=Ve_oracleAblation_4
#   Full round-robin across all five champions into one file:
#     python tools/benchmark_champion.py --output=champion-results.jsonl --round-robin=Aa_replay,Sa_archScan_3,Ta_minimaxSelect_1,Ve_oracleAblation_4,Yb_hotChamp_3

import sys
import json
from pathlib import Path
from datetime import datetime
from docopt import docopt

# Ensure workspace root is on path when running from tools/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quartopy import play_games
from bot.CNN_bot import Quarto_bot
from bot.CNN_unified_bot import Quarto_bot as Quarto_unified_bot
from bot.CNN_autoreg_bot import Quarto_bot as Quarto_autoreg_bot
from bot.minimax_bot import MinimaxBot

from models.CNN1 import QuartoCNN
from models.CNN_uncoupled import QuartoCNN as QuartoCNN_uncoupled
from models.CNN_autoreg import QuartoCNNAutoreg
from models.CNN_autoreg_sa import (
    QuartoCNNAutoregUnifiedS4,
    QuartoCNNAutoregUnifiedS4Hot,
)

# Registry mapping string names used in champion_config.json to actual classes.
# Add a new entry here whenever a new model or bot class is introduced.
# NB QuartoCNNAutoregUnifiedS4Hot's fc_hot head is a training-time scaffold:
# forward() is unchanged (returns q_place, q_select), so it plays through
# Quarto_unified_bot exactly like S4 — the hot params just sit unused at inference.
_MODEL_REGISTRY = {
    "QuartoCNN": QuartoCNN,
    "QuartoCNN_uncoupled": QuartoCNN_uncoupled,
    "QuartoCNNAutoreg": QuartoCNNAutoreg,
    "QuartoCNNAutoregUnifiedS4": QuartoCNNAutoregUnifiedS4,
    "QuartoCNNAutoregUnifiedS4Hot": QuartoCNNAutoregUnifiedS4Hot,
}

_BOT_REGISTRY = {
    "Quarto_bot": Quarto_bot,
    "Quarto_unified_bot": Quarto_unified_bot,
    "Quarto_autoreg_bot": Quarto_autoreg_bot,
}

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "champion_config.json"


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return json.load(f)


def load_bot(bot_key: str, config: dict) -> tuple:
    """Load a bot from a config dict entry."""
    bot_cfg = config["bots"][bot_key]
    if bot_cfg.get("type") == "minimax":
        bot = MinimaxBot(depth=bot_cfg.get("depth", 2))
        return bot, bot_cfg["name"]
    model_class = _MODEL_REGISTRY[bot_cfg["model_class"]]
    bot_class = _BOT_REGISTRY[bot_cfg["bot_class"]]
    bot = bot_class(
        model_path=bot_cfg["path"],
        model_class=model_class,
        deterministic=False,
        temperature=0.1,
    )
    return bot, bot_cfg["name"]


def benchmark_champion(
    config: dict,
    matches: int = 500,
    opponents: list | None = None,
    output_path: str = "champion-results.jsonl",
    verbose: bool = False,
    champion_key: str | None = None,
    append: bool = False,
) -> None:
    resolved_opponents: list = (
        opponents if opponents is not None else config.get("default_opponents", [])
    )

    champion_key = champion_key or config["champion"]
    champion_bot, champion_name = load_bot(champion_key, config)
    print(f"\n{'='*60}")
    print(f"Benchmarking Champion: {champion_name}")
    print(f"{'='*60}")
    print(f"Number of matches per pairing: {matches}")
    print(f"Output: {output_path}")
    print(f"Opponents: {resolved_opponents}")
    print(f"{'='*60}\n")

    results = []

    for opp_key in resolved_opponents:
        opponent_bot, opponent_name = load_bot(opp_key, config)

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

    with open(output_file, "a" if append else "w") as f:
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


def benchmark_round_robin(
    config: dict,
    champion_keys: list,
    matches: int = 500,
    output_path: str = "champion-results.jsonl",
    verbose: bool = False,
) -> None:
    """Run every listed champion vs every other listed champion + baselines.

    Baseline opponents = the config's default_opponents that are NOT themselves
    in champion_keys (so champion-vs-champion pairs are not double-counted with a
    different a/b orientation). All rows land in one --output file; it is
    truncated once up front, then each champion's block is appended.
    """
    all_keys = set(config["bots"].keys())
    for k in champion_keys:
        if k not in all_keys:
            raise SystemExit(
                f"Error: unknown champion key '{k}'. "
                f"Available: {', '.join(sorted(all_keys))}"
            )

    default_opponents = config.get("default_opponents", [])
    baseline_opponents = [o for o in default_opponents if o not in champion_keys]

    # Truncate once so re-running the round-robin is idempotent.
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("")

    print(f"\n{'#'*60}")
    print(f"ROUND-ROBIN: {len(champion_keys)} champions -> {output_path}")
    print(f"Champions : {champion_keys}")
    print(f"Baselines : {baseline_opponents}")
    print(f"{'#'*60}")

    for k in champion_keys:
        opponents = [c for c in champion_keys if c != k] + baseline_opponents
        benchmark_champion(
            config=config,
            matches=matches,
            opponents=opponents,
            output_path=output_path,
            verbose=verbose,
            champion_key=k,
            append=True,
        )

    print(f"{'#'*60}")
    print(f"Round-robin complete -> {out.absolute()}")
    print(f"{'#'*60}\n")


def main():
    args = docopt(__doc__)

    config_path = Path(args["--config"]) if args["--config"] else DEFAULT_CONFIG_PATH
    config = load_config(config_path)

    matches = int(args["--matches"])
    output_path = args["--output"]
    verbose = args["--verbose"]

    # Round-robin mode short-circuits the single-champion path.
    if args["--round-robin"]:
        champion_keys = [c.strip() for c in args["--round-robin"].split(",")]
        benchmark_round_robin(
            config=config,
            champion_keys=champion_keys,
            matches=matches,
            output_path=output_path,
            verbose=verbose,
        )
        return

    champion_key = args["--champion"] or config["champion"]
    if champion_key not in config["bots"]:
        print(f"Error: Unknown champion '{champion_key}'")
        print(f"Available bots: {', '.join(sorted(config['bots'].keys()))}")
        return

    if args["--opponents"]:
        opponents = [o.strip() for o in args["--opponents"].split(",")]
        valid_opponents = set(config["bots"].keys()) - {champion_key}
        for opp in opponents:
            if opp not in valid_opponents:
                print(f"Error: Unknown opponent '{opp}'")
                print(f"Available opponents: {', '.join(sorted(valid_opponents))}")
                return
    else:
        opponents = None  # will use default_opponents from config

    benchmark_champion(
        config=config,
        matches=matches,
        opponents=opponents,
        output_path=output_path,
        verbose=verbose,
        champion_key=champion_key,
        append=bool(args["--append"]),
    )


if __name__ == "__main__":
    main()
