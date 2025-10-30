# -*- coding: utf-8 -*-

"""
Python 3
28 / 10 / 2025
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

"Either mathematics is too big for the human mind or the human mind is more than a machine."
-Kurt Godël
"""


# ----------------------------- MAIN CONFIGS --------------------------
from bot.CNN_bot import Quarto_bot


FOLDER_CHECKPOINTS = "CHECKPOINTS//E02_win_rate//"
# FOLDER_CHECKPOINTS = "CHECKPOINTS//EXP_id03//"  # debugging
BOTs_CLASS = Quarto_bot
BOTs_PARAMs = {"deterministic": False, "temperature": 0.1}
NUM_ROUNDS = 200
# NUM_ROUNDS = 1000 # 5->10pm
DOUBLE_SWISS = True  # Each bot plays once as player 1 and once as player 2
MODE_McMahon = True  # Initial points based on estimated strength "Epoch"


# ----------------------------- logging config --------------------------
import logging
from sys import stdout
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    stream=stdout,
    datefmt="%m-%d %H:%M:%S",
)
logging.info(datetime.now())


def run_swiss_tournament():
    """
    Runs a Swiss-system tournament between all checkpoints found in FOLDER_CHECKPOINTS.

    Swiss system pairs players with similar scores each round.
    Optional McMahon scoring gives initial points based on model epoch (estimated strength).
    Optional double Swiss where each pairing plays twice (swapping colors).
    """
    from pathlib import Path
    from quartopy import play_games
    import re
    import pickle
    from datetime import datetime
    from tqdm.auto import tqdm
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objs as go

    # ----------------------------- LOAD CHECKPOINTS --------------------------
    logging.info("Loading checkpoints from %s", FOLDER_CHECKPOINTS)
    checkpoint_path = Path(FOLDER_CHECKPOINTS)

    if not checkpoint_path.exists():
        logging.error("Checkpoint folder not found: %s", FOLDER_CHECKPOINTS)
        return

    checkpoint_files = sorted(list(checkpoint_path.glob("*.pt")))

    if len(checkpoint_files) == 0:
        logging.error("No checkpoint files found in %s", FOLDER_CHECKPOINTS)
        return

    logging.info("Found %d checkpoint files", len(checkpoint_files))

    # ----------------------------- PARSE EPOCHS --------------------------
    # Extract epoch numbers from filenames for McMahon scoring
    def extract_epoch(filename):
        """Extract epoch number from checkpoint filename"""
        match = re.search(r"epoch[_\s]+(\d+)", filename.stem, re.IGNORECASE)
        return int(match.group(1)) if match else 0

    # Create bot information
    bots_info = []
    for ckpt_file in checkpoint_files:
        epoch = extract_epoch(ckpt_file)
        bots_info.append(
            {
                "name": ckpt_file.stem,
                "path": str(ckpt_file),
                "epoch": epoch,
                "score": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "matches_played": 0,
            }
        )
    logging.info("Loaded %d bots", len(bots_info))

    # ----------------------------- SWISS TOURNAMENT --------------------------
    results_history = []  # Store all match results

    for round_num in tqdm(range(1, NUM_ROUNDS + 1)):
        logging.info("=" * 60)
        logging.info("ROUND %d / %d", round_num, NUM_ROUNDS)
        logging.info("=" * 60)

        if MODE_McMahon and round_num == 1:
            # Initial scoring based on epoch
            logging.info("Applied McMahon initial scoring based on epoch")
            bots_info.sort(key=lambda x: -x["epoch"])
        else:
            # Sort by score (and tiebreakers: wins, then epoch)
            bots_info.sort(key=lambda x: (-x["score"], -x["wins"], -x["epoch"]))

        # Swiss pairing: pair consecutive bots in sorted list
        pairings = []
        paired_indices = set()

        for i in range(len(bots_info)):
            if i in paired_indices:
                continue

            # Find opponent: next unpaired bot
            for j in range(i + 1, len(bots_info)):
                if j not in paired_indices:
                    pairings.append((i, j))
                    paired_indices.add(i)
                    paired_indices.add(j)
                    break

        # Handle odd number of bots (bye)
        if len(bots_info) % 2 == 1:
            for i in range(len(bots_info)):
                if i not in paired_indices:
                    logging.info("Bot %s gets a BYE", bots_info[i]["name"])
                    bots_info[i]["score"] += 1  # Bye = 1 point
                    break

        logging.info("Pairings for round %d: %d matches", round_num, len(pairings))

        # ----------------------------- PLAY MATCHES --------------------------
        round_results = []

        for pair_idx, (idx1, idx2) in enumerate(
            tqdm(pairings, desc=f"Round {round_num} matches", position=0, leave=True)
        ):
            bot1_info = bots_info[idx1]
            bot2_info = bots_info[idx2]

            # logging.info(
            #     "Match %d/%d: %s (score: %.1f) vs %s (score: %.1f)",
            #     pair_idx + 1,
            #     len(pairings),
            #     bot1_info["name"],
            #     bot1_info["score"],
            #     bot2_info["name"],
            #     bot2_info["score"],
            # )

            # Create bot instances
            bot1 = BOTs_CLASS(model_path=bot1_info["path"], **BOTs_PARAMs)
            bot2 = BOTs_CLASS(model_path=bot2_info["path"], **BOTs_PARAMs)

            # Play first game
            _, win_rate1 = play_games(
                matches=1,
                player1=bot1,
                player2=bot2,
                verbose=False,
                save_match=False,
                mode_2x2=True,
            )

            result1 = {
                "round": round_num,
                "match_id": f"R{round_num}_M{pair_idx + 1}_G1",
                "player1": bot1_info["name"],
                "player2": bot2_info["name"],
                "player1_epoch": bot1_info["epoch"],
                "player2_epoch": bot2_info["epoch"],
                "p1_wins": win_rate1["Player 1"],
                "p2_wins": win_rate1["Player 2"],
                "draws": win_rate1["Tie"],
            }
            round_results.append(result1)

            # Update scores for game 1
            if win_rate1["Player 1"] > win_rate1["Player 2"]:
                bot1_info["wins"] += 1
                bot2_info["losses"] += 1
                bot1_info["score"] += 1
            elif win_rate1["Player 1"] < win_rate1["Player 2"]:
                bot2_info["wins"] += 1
                bot1_info["losses"] += 1
                bot2_info["score"] += 1
            else:
                bot1_info["draws"] += 1
                bot2_info["draws"] += 1
                bot1_info["score"] += 0.5
                bot2_info["score"] += 0.5

            bot1_info["matches_played"] += 1
            bot2_info["matches_played"] += 1

            # Double Swiss: play second game with colors swapped
            if DOUBLE_SWISS:
                _, win_rate2 = play_games(
                    matches=1,
                    player1=bot2,
                    player2=bot1,
                    verbose=False,
                    save_match=False,
                    mode_2x2=True,
                )

                result2 = {
                    "round": round_num,
                    "match_id": f"R{round_num}_M{pair_idx + 1}_G2",
                    "player1": bot2_info["name"],
                    "player2": bot1_info["name"],
                    "player1_epoch": bot2_info["epoch"],
                    "player2_epoch": bot1_info["epoch"],
                    "p1_wins": win_rate2["Player 1"],
                    "p2_wins": win_rate2["Player 2"],
                    "draws": win_rate2["Tie"],
                }
                round_results.append(result2)

                # Update scores for game 2
                if win_rate2["Player 1"] > win_rate2["Player 2"]:
                    bot2_info["wins"] += 1
                    bot1_info["losses"] += 1
                    bot2_info["score"] += 1
                elif win_rate2["Player 1"] < win_rate2["Player 2"]:
                    bot1_info["wins"] += 1
                    bot2_info["losses"] += 1
                    bot1_info["score"] += 1
                else:
                    bot1_info["draws"] += 1
                    bot2_info["draws"] += 1
                    bot1_info["score"] += 0.5
                    bot2_info["score"] += 0.5

                bot1_info["matches_played"] += 1
                bot2_info["matches_played"] += 1

        results_history.extend(round_results)

        # Display standings after round
        logging.info("\nStandings after Round %d:", round_num)
        standings = sorted(
            bots_info, key=lambda x: (-x["score"], -x["wins"], -x["epoch"])
        )
        for rank, bot in enumerate(standings[:10], 1):  # Top 10
            logging.info(
                "%2d. %s | Score: %.1f | W:%d D:%d L:%d | Epoch: %d",
                rank,
                bot["name"][-10:],
                bot["score"],
                bot["wins"],
                bot["draws"],
                bot["losses"],
                bot["epoch"],
            )

    # ----------------------------- FINAL RESULTS --------------------------
    logging.info("=" * 60)
    logging.info("FINAL STANDINGS")
    logging.info("=" * 60)

    final_standings = sorted(
        bots_info, key=lambda x: (-x["score"], -x["wins"], -x["epoch"])
    )

    for rank, bot in enumerate(final_standings, 1):
        logging.info(
            "%2d. %s | Score: %.1f | W:%d D:%d L:%d | Matches: %d | Epoch: %d",
            rank,
            bot["name"][:50],
            bot["score"],
            bot["wins"],
            bot["draws"],
            bot["losses"],
            bot["matches_played"],
            bot["epoch"],
        )

    # ----------------------------- SAVE RESULTS --------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"swiss_tournament_{timestamp}.pkl"

    tournament_data = {
        "config": {
            "num_rounds": NUM_ROUNDS,
            "double_swiss": DOUBLE_SWISS,
            "mcmahon": MODE_McMahon,
            "folder": FOLDER_CHECKPOINTS,
        },
        "final_standings": final_standings,
        "match_results": results_history,
        "timestamp": timestamp,
    }

    with open(results_file, "wb") as f:
        pickle.dump(tournament_data, f)

    logging.info("Results saved to %s", results_file)
    logging.info("Tournament completed!")

    return tournament_data


# ----------------------------- MAIN EXECUTION --------------------------
if __name__ == "__main__":
    run_swiss_tournament()
