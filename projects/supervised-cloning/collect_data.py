"""collect_data.py – Collect supervised training data from MinimaxBot games.

Records every decision made by the teacher bot (MinimaxBot at a fixed depth)
across many games.  Two action types are captured per turn:
  * PLACE (0): teacher places the selected piece → label = board position index [0-15]
  * SELECT (1): teacher selects a piece to give to opponent → label = piece index [0-15]

Output is a compressed .npz file with arrays:
  boards   (N, 16, 4, 4)  float32   – board encoding before the action
  pieces   (N, 16)         float32   – piece one-hot (zeros for SELECT turns)
  labels   (N,)            int16     – target class
  actions  (N,)            uint8     – 0=PLACE, 1=SELECT
  game_ids (N,)            int32     – game index

Usage:
    collect_data.py [options]

Options:
    -o <path>, --output <path>      Output .npz file  [default: projects/supervised-cloning/data/collected.npz]
    -g <int>, --games <int>         Total games to play  [default: 500]
    -d <int>, --depth <int>         Teacher MinimaxBot search depth  [default: 2]
    --opponent-mix <spec>           Comma-separated name:fraction pairs.
                                    Supported names: random, minimax_d1, minimax_d2, minimax_d3.
                                    Fractions are auto-normalised.
                                    [default: random:0.3,minimax_d1:0.3,minimax_d2:0.4]
    --no-mode-2x2                   Disable 2x2 win condition (default: enabled).
    --seed <int>                    Random seed  [default: 42]
    -h, --help                      Show this help.

Examples:
    # 500 games, teacher=d2, default mix, mode_2x2 enabled
    python collect_data.py -g 500 -d 2

    # 1000 games with custom mix, mode_2x2 disabled, output to a specific path
    python collect_data.py -g 1000 --opponent-mix random:0.5,minimax_d3:0.5 --no-mode-2x2 -o data/big.npz
"""

from __future__ import annotations

import os
import sys
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── resolve project root (folder that contains bot/) ──────────────────────────
_here = Path(__file__).resolve().parent
_root = _here
while not (_root / "bot").is_dir():
    _root = _root.parent
    if _root == _root.parent:
        raise RuntimeError("Could not find project root containing 'bot/' folder")

os.chdir(_root)
sys.path.insert(0, str(_root))

from docopt import docopt  # noqa: E402 (after path setup)
from quartopy import QuartoGame, Piece  # noqa: E402
from bot.minimax_bot import MinimaxBot  # noqa: E402
from bot.random_bot import Quarto_bot as RandomBot  # noqa: E402

# ── constants ──────────────────────────────────────────────────────────────────
ACTION_PLACE = np.uint8(0)   # teacher places the selected piece
ACTION_SELECT = np.uint8(1)  # teacher selects the next piece for opponent


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_bot(name: str):
    """Instantiate a bot by short name."""
    if name == "random":
        return RandomBot()
    if name.startswith("minimax_d"):
        depth = int(name.removeprefix("minimax_d"))
        return MinimaxBot(depth=depth)
    raise ValueError(f"Unknown opponent name: {name!r}.  "
                     "Supported: random, minimax_d1, minimax_d2, minimax_d3")


def _parse_mix(spec: str) -> list[tuple[str, float]]:
    """Parse 'random:0.3,minimax_d2:0.7' into [(name, normalised_frac), ...]."""
    pairs: list[tuple[str, float]] = []
    for item in spec.split(","):
        name, frac = item.strip().split(":")
        pairs.append((name.strip(), float(frac)))
    total = sum(f for _, f in pairs)
    return [(n, f / total) for n, f in pairs]


def _board_legal_mask(game: QuartoGame) -> np.ndarray:
    """Return a (16,) bool mask of empty positions on the game board."""
    mask = np.zeros(16, dtype=bool)
    for r, c in game.game_board.get_valid_moves():
        mask[game.game_board.pos2index(r, c)] = True
    return mask


def _piece_legal_mask(game: QuartoGame) -> np.ndarray:
    """Return a (16,) bool mask of pieces still available in storage."""
    mask = np.zeros(16, dtype=bool)
    for r, c in game.storage_board.get_valid_moves():
        piece = game.storage_board.get_piece(r, c)
        mask[piece.index()] = True
    return mask


# ── core data collection ──────────────────────────────────────────────────────

def collect_game(
    teacher: MinimaxBot,
    opponent,
    mode_2x2: bool,
    game_id: int,
    teacher_is_p1: bool,
) -> list[dict]:
    """Play one full game and return data records for each teacher action.

    Parameters
    ----------
    teacher:        The MinimaxBot whose decisions are recorded as labels.
    opponent:       The other player (any BotAI).
    mode_2x2:       Passed to QuartoGame.
    game_id:        Integer identifier stored in every record.
    teacher_is_p1:  If True teacher plays as player1, else player2.
    """
    if teacher_is_p1:
        game = QuartoGame(player1=teacher, player2=opponent, mode_2x2=mode_2x2)
    else:
        game = QuartoGame(player1=opponent, player2=teacher, mode_2x2=mode_2x2)

    records: list[dict] = []

    while not game.player_won and not game.game_board.is_full():
        current_player = game.get_current_player()
        is_teacher_turn = current_player is teacher

        if is_teacher_turn:
            # ── snapshot state BEFORE the action ──────────────────────────
            board_enc = game.game_board.encode().squeeze(0).astype(np.float32)   # (16,4,4)

            if game.pick:   # SELECT phase: teacher chooses a piece for opponent
                piece_onehot = np.zeros(16, dtype=np.float32)
                legal_mask = _piece_legal_mask(game)
                action = ACTION_SELECT
            else:           # PLACE phase: teacher places the current selected piece
                assert isinstance(game.selected_piece, Piece), (
                    "Expected a Piece in selected_piece during PLACE phase"
                )
                piece_onehot = game.selected_piece.vectorize_onehot().astype(np.float32)
                legal_mask = _board_legal_mask(game)
                action = ACTION_PLACE

            prev_len = len(game.move_history)

        # ── execute the turn ───────────────────────────────────────────────
        game.play_turn()

        if is_teacher_turn:
            # ── extract label from move_history ───────────────────────────
            new_moves = game.move_history[prev_len:]
            assert len(new_moves) == 1, (
                f"Expected exactly 1 new history entry, got {len(new_moves)}"
            )
            move = new_moves[0]

            if action == ACTION_SELECT:
                label = np.int16(move["piece_index"])
            else:
                label = np.int16(move["position_index"])

            records.append({
                "board": board_enc,
                "piece": piece_onehot,
                "label": label,
                "action": action,
                "legal_mask": legal_mask,
                "game_id": np.int32(game_id),
            })

        game.cambiar_turno()

    return records


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = docopt(__doc__)

    n_games = int(args["--games"])
    depth = int(args["--depth"])
    output_path = Path(args["--output"])
    mode_2x2 = not bool(args["--no-mode-2x2"])
    seed = int(args["--seed"])
    mix = _parse_mix(args["--opponent-mix"])

    random.seed(seed)
    np.random.seed(seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    teacher = MinimaxBot(depth=depth)

    opponent_names = [n for n, _ in mix]
    opponent_weights = [w for _, w in mix]
    opponents = {n: _make_bot(n) for n in opponent_names}

    print(f"Teacher   : MinimaxBot depth={depth}")
    print(f"Opponents : {mix}")
    print(f"Games     : {n_games}  (teacher plays both sides alternately)")
    print(f"mode_2x2  : {mode_2x2}")
    print(f"Output    : {output_path}\n")

    all_records: list[dict] = []

    for game_id in tqdm(range(n_games), desc="Collecting games", unit="game"):
        opp_name = random.choices(opponent_names, weights=opponent_weights, k=1)[0]
        opponent = opponents[opp_name]
        # Alternate sides to avoid positional bias
        teacher_is_p1 = (game_id % 2 == 0)
        records = collect_game(teacher, opponent, mode_2x2, game_id, teacher_is_p1)
        all_records.extend(records)

    if not all_records:
        print("No records collected — exiting.")
        return

    # ── stack into arrays ──────────────────────────────────────────────────────
    boards      = np.stack([r["board"]       for r in all_records])   # (N, 16, 4, 4)
    pieces      = np.stack([r["piece"]       for r in all_records])   # (N, 16)
    labels      = np.array([r["label"]       for r in all_records], dtype=np.int16)
    actions     = np.array([r["action"]      for r in all_records], dtype=np.uint8)
    legal_masks = np.stack([r["legal_mask"]  for r in all_records])   # (N, 16)
    game_ids    = np.array([r["game_id"]     for r in all_records], dtype=np.int32)

    np.savez_compressed(
        output_path,
        boards=boards,
        pieces=pieces,
        labels=labels,
        actions=actions,
        legal_masks=legal_masks,
        game_ids=game_ids,
    )

    n_place  = int((actions == ACTION_PLACE).sum())
    n_select = int((actions == ACTION_SELECT).sum())
    print(f"\nSaved {len(all_records):,} samples  "
          f"({n_place:,} PLACE + {n_select:,} SELECT)  →  {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
