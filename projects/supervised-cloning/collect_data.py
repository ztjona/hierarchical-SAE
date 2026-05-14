"""collect_data.py – Collect supervised training data from MinimaxBot games.

Records every decision made by the teacher bot (MinimaxBot at a fixed depth)
across many games. Two action types are captured per turn:
  * PLACE (0): teacher places the selected piece → label = board position index [0-15]
  * SELECT (1): teacher selects a piece to give to opponent → label = piece index [0-15]

Output is a compressed .npz file with arrays:
  boards        (N, 16, 4, 4) float32   – board encoding before the action
  aux           (N, 32)        float32   – [offered_one_hot(16) ; available_pieces(16)]
  labels        (N,)           int16     – hard target class
  actions       (N,)           uint8     – 0=PLACE, 1=SELECT
  legal_masks   (N, 16)        bool      – per-phase legal action mask
  soft_targets  (N, 16)        float32   – uniform over teacher-tied-optimal moves
                                           (sums to 1 on legal positions, 0 elsewhere)
  game_ids      (N,)           int32     – game index

The ``aux`` semantics are phase-stable (matches QuartoCNNAutoregUnified):
  - offered[0:16] = PLACE: one-hot of the piece in hand;
                    SELECT: one-hot of the piece just placed this turn
                    (zeros for the very first select of a game).
  - available[16:32] = mask of pieces still in storage (i.e. neither on the
                       board nor in the player's hand).

``soft_targets`` is built from MinimaxBot.score_all_moves: every legal action
gets its minimax value at the teacher's depth, and the optimal set (argmax
for PLACE, argmin for SELECT) receives equal probability mass; non-optimal
legal actions receive 0. This handles tie-breaking noise from the teacher's
deterministic argmax fallback.

Usage:
    collect_data.py [options]

Options:
    -o <path>, --output <path>      Output .npz file  [default: projects/supervised-cloning/data/collected.npz]
    -g <int>, --games <int>         Total games to play  [default: 500]
    -d <int>, --depth <int>         Teacher MinimaxBot search depth  [default: 2]
    --opponent-mix <spec>           Comma-separated name:fraction pairs.
                                    Supported names: random, minimax_d1, minimax_d2,
                                    minimax_d3, loss_BT.  Fractions are auto-normalised.
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
from bot.minimax_bot import MinimaxBot, best_action_set  # noqa: E402
from bot.random_bot import Quarto_bot as RandomBot  # noqa: E402
from bot.CNN_bot import Quarto_bot as CNNBot  # noqa: E402
from models.CNN_uncoupled import QuartoCNN as QuartoCNN_uncoupled  # noqa: E402

_LOSS_BT_PATH = (
    "CHECKPOINTS/LOSS_APPROACHs_1212-2_only_select"
    "/20251212_2206-LOSS_APPROACHs_1212-2_only_select_E_1034.pt"
)

ACTION_PLACE = np.uint8(0)
ACTION_SELECT = np.uint8(1)


def _make_bot(name: str):
    if name == "random":
        return RandomBot()
    if name.startswith("minimax_d"):
        depth = int(name.removeprefix("minimax_d"))
        return MinimaxBot(depth=depth)
    if name == "loss_BT":
        return CNNBot(
            model_path=_LOSS_BT_PATH,
            model_class=QuartoCNN_uncoupled,
            deterministic=False,
            temperature=0.1,
        )
    raise ValueError(
        f"Unknown opponent name: {name!r}.  "
        "Supported: random, minimax_d1, minimax_d2, minimax_d3, loss_BT"
    )


def _parse_mix(spec: str) -> list[tuple[str, float]]:
    pairs: list[tuple[str, float]] = []
    for item in spec.split(","):
        name, frac = item.strip().split(":")
        pairs.append((name.strip(), float(frac)))
    total = sum(f for _, f in pairs)
    return [(n, f / total) for n, f in pairs]


def _board_legal_mask(game: QuartoGame) -> np.ndarray:
    """(16,) bool mask of empty board positions."""
    mask = np.zeros(16, dtype=bool)
    for r, c in game.game_board.get_valid_moves():
        mask[game.game_board.pos2index(r, c)] = True
    return mask


def _piece_available_mask(game: QuartoGame) -> np.ndarray:
    """(16,) bool mask of pieces still in storage."""
    mask = np.zeros(16, dtype=bool)
    for piece in game.storage_board.get_valid_pieces():
        mask[piece.index()] = True
    return mask


def _piece_one_hot(piece_index: int) -> np.ndarray:
    oh = np.zeros(16, dtype=np.float32)
    if 0 <= piece_index < 16:
        oh[piece_index] = 1.0
    return oh


def _build_aux32(
    offered_index: int, available_mask_bool: np.ndarray
) -> np.ndarray:
    """Compose the 32-d phase-stable aux: offered ⊕ available."""
    offered = _piece_one_hot(offered_index)
    available = available_mask_bool.astype(np.float32)
    return np.concatenate([offered, available]).astype(np.float32)


def _soft_target_from_scores(
    scores: dict[int, float], action_kind: int, legal_mask: np.ndarray
) -> np.ndarray:
    """Uniform over teacher-tied-optimal moves; zero on non-optimal & illegal."""
    soft = np.zeros(16, dtype=np.float32)
    best = best_action_set(scores, action_kind)
    if not best:
        # Fallback: uniform over legal mask if scoring returned nothing.
        legal_idx = np.flatnonzero(legal_mask)
        if legal_idx.size:
            soft[legal_idx] = 1.0 / legal_idx.size
        return soft
    w = 1.0 / len(best)
    for idx in best:
        soft[idx] = w
    return soft


def collect_game(
    teacher: MinimaxBot,
    opponent,
    mode_2x2: bool,
    game_id: int,
    teacher_is_p1: bool,
) -> list[dict]:
    """Play one full game and return data records for each teacher action."""
    if teacher_is_p1:
        game = QuartoGame(player1=teacher, player2=opponent, mode_2x2=mode_2x2)
    else:
        game = QuartoGame(player1=opponent, player2=teacher, mode_2x2=mode_2x2)

    records: list[dict] = []
    # Tracks the piece the teacher most recently placed (for SELECT-phase aux).
    teacher_last_placed: int = -1

    while not game.player_won and not game.game_board.is_full():
        current_player = game.get_current_player()
        is_teacher_turn = current_player is teacher

        if is_teacher_turn:
            board_enc = (
                game.game_board.encode().squeeze(0).astype(np.float32)
            )  # (16,4,4)

            available_mask = _piece_available_mask(game)

            if game.pick:  # SELECT phase
                legal_mask = available_mask  # available == legal for SELECT
                action = ACTION_SELECT
                # offered for SELECT = the piece the teacher just placed.
                aux = _build_aux32(teacher_last_placed, available_mask)
            else:  # PLACE phase
                assert isinstance(
                    game.selected_piece, Piece
                ), "Expected a Piece in selected_piece during PLACE phase"
                offered_idx = int(game.selected_piece.index())
                legal_mask = _board_legal_mask(game)
                action = ACTION_PLACE
                aux = _build_aux32(offered_idx, available_mask)

            # Soft target via top-level no-pruning enumeration of teacher moves.
            scores, action_kind = teacher.score_all_moves(game)
            soft_target = _soft_target_from_scores(scores, action_kind, legal_mask)

            prev_len = len(game.move_history)

        game.play_turn()

        if is_teacher_turn:
            new_moves = game.move_history[prev_len:]
            assert (
                len(new_moves) == 1
            ), f"Expected exactly 1 new history entry, got {len(new_moves)}"
            move = new_moves[0]

            if action == ACTION_SELECT:
                label = np.int16(move["piece_index"])
            else:
                label = np.int16(move["position_index"])
                teacher_last_placed = int(label)  # remember for next SELECT

            records.append(
                {
                    "board": board_enc,
                    "aux": aux,
                    "label": label,
                    "action": action,
                    "legal_mask": legal_mask,
                    "soft_target": soft_target,
                    "game_id": np.int32(game_id),
                }
            )

        game.cambiar_turno()

    return records


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
    print(f"Schema    : unified-aux (32-d) + soft_targets (multi-hot uniform)")
    print(f"Output    : {output_path}\n")

    all_records: list[dict] = []

    for game_id in tqdm(range(n_games), desc="Collecting games", unit="game"):
        opp_name = random.choices(opponent_names, weights=opponent_weights, k=1)[0]
        opponent = opponents[opp_name]
        teacher_is_p1 = game_id % 2 == 0
        records = collect_game(teacher, opponent, mode_2x2, game_id, teacher_is_p1)
        all_records.extend(records)

    if not all_records:
        print("No records collected — exiting.")
        return

    boards = np.stack([r["board"] for r in all_records])
    aux = np.stack([r["aux"] for r in all_records])
    labels = np.array([r["label"] for r in all_records], dtype=np.int16)
    actions = np.array([r["action"] for r in all_records], dtype=np.uint8)
    legal_masks = np.stack([r["legal_mask"] for r in all_records])
    soft_targets = np.stack([r["soft_target"] for r in all_records])
    game_ids = np.array([r["game_id"] for r in all_records], dtype=np.int32)

    np.savez_compressed(
        output_path,
        boards=boards,
        aux=aux,
        labels=labels,
        actions=actions,
        legal_masks=legal_masks,
        soft_targets=soft_targets,
        game_ids=game_ids,
    )

    n_place = int((actions == ACTION_PLACE).sum())
    n_select = int((actions == ACTION_SELECT).sum())
    n_tied = int((soft_targets > 0).sum(axis=1).mean()) if len(soft_targets) else 0
    avg_tied = float((soft_targets > 0).sum(axis=1).mean()) if len(soft_targets) else 0.0
    print(
        f"\nSaved {len(all_records):,} samples  "
        f"({n_place:,} PLACE + {n_select:,} SELECT)  →  {output_path}"
    )
    print(f"Avg #tied-optimal moves per sample: {avg_tied:.2f}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
