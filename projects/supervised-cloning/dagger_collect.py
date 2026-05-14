"""dagger_collect.py - DAgger-style data collection.

Plays games where the supervised clone (loaded from a checkpoint) acts on one
side and a varied opponent acts on the other.  At every CLONE turn, the state
is recorded and labeled with the move minimax_d2 would have chosen.  This
closes the BC coverage gap: clones trained on teacher-only trajectories never
see the states they themselves visit at deploy time.

Output schema is identical to ``collect_data.py`` so the two .npz files can be
concatenated for retraining.

Usage:
    dagger_collect.py <clone_path> [options]

Arguments:
    <clone_path>                    Path to clone .pt OR experiment name
                                    (resolved under projects/supervised-cloning/experiments/).

Options:
    -o <path>, --output <path>      Output .npz file
                                    [default: projects/supervised-cloning/data/dagger.npz]
    -g <int>, --games <int>         Total games to play  [default: 500]
    -d <int>, --depth <int>         Expert MinimaxBot search depth  [default: 2]
    --opponent-mix <spec>           Same syntax as collect_data.py.
                                    [default: minimax_d2:0.4,loss_BT:0.4,minimax_d1:0.1,random:0.1]
    --checkpoint <ckpt>             "best" or "final" if <clone_path> is an experiment name
                                    [default: best]
    --no-mode-2x2                   Disable 2x2 win condition (default: enabled).
    --seed <int>                    Random seed  [default: 42]
    -h, --help                      Show this help.

Example:
    python projects/supervised-cloning/dagger_collect.py A1_baseline_cnn -g 1000
"""

from __future__ import annotations

import os
import sys
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Resolve project root.
_here = Path(__file__).resolve().parent
_root = _here
while not (_root / "bot").is_dir():
    _root = _root.parent
    if _root == _root.parent:
        raise RuntimeError("Could not find project root containing 'bot/' folder")
os.chdir(_root)
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_here))  # for `import train`

from docopt import docopt  # noqa: E402
from quartopy import QuartoGame, Piece  # noqa: E402
import torch  # noqa: E402
from bot.minimax_bot import MinimaxBot  # noqa: E402
from bot.CNN_bot import Quarto_bot as CNNBot  # noqa: E402

# Reuse the diversified opponent factory.
from collect_data import _make_bot, _parse_mix, _board_legal_mask, _piece_legal_mask  # noqa: E402
from collect_data import ACTION_PLACE, ACTION_SELECT  # noqa: E402

# Pull the same logits wrapper the training script uses, so the loaded clone
# behaves identically to what trained it.
from train import QuartoCNNLogits  # noqa: E402


# ── clone loading ─────────────────────────────────────────────────────────────


def _resolve_weights(arg: str, checkpoint: str) -> Path:
    p = Path(arg)
    if p.suffix == ".pt":
        return p if p.is_absolute() else _root / p
    exp_dir = _root / "projects" / "supervised-cloning" / "experiments" / arg
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"experiment dir not found: {exp_dir}")
    return exp_dir / f"{checkpoint}.pt"


def _load_clone(weights_path: Path) -> CNNBot:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuartoCNNLogits()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    # deterministic=True is fine here — variation comes from opponent diversity,
    # not from the clone's own stochasticity.  See feedback_tanh_temperature.md.
    return CNNBot(model=model, deterministic=True, temperature=0.1)


# ── core: play one game, record clone-visited states ─────────────────────────


def collect_game(
    clone: CNNBot,
    opponent,
    expert: MinimaxBot,
    mode_2x2: bool,
    game_id: int,
    clone_is_p1: bool,
) -> list[dict]:
    """Play one game and return one record per CLONE turn, labeled with the
    expert (minimax_d2) action at that state."""
    import copy

    if clone_is_p1:
        game = QuartoGame(player1=clone, player2=opponent, mode_2x2=mode_2x2)
    else:
        game = QuartoGame(player1=opponent, player2=clone, mode_2x2=mode_2x2)

    records: list[dict] = []

    while not game.player_won and not game.game_board.is_full():
        current = game.get_current_player()
        if current is not clone:
            game.play_turn()
            game.cambiar_turno()
            continue

        # Snapshot state BEFORE the clone moves.
        board_enc = game.game_board.encode().squeeze(0).astype(np.float32)

        if game.pick:
            piece_onehot = np.zeros(16, dtype=np.float32)
            legal_mask = _piece_legal_mask(game)
            action = ACTION_SELECT
            # Ask the expert what it would do at this same state.
            shadow = copy.deepcopy(game)
            expert_piece = expert.select(shadow)
            label = np.int16(expert_piece.index())
        else:
            assert isinstance(game.selected_piece, Piece)
            piece_onehot = game.selected_piece.vectorize_onehot().astype(np.float32)
            legal_mask = _board_legal_mask(game)
            action = ACTION_PLACE
            shadow = copy.deepcopy(game)
            r, c = expert.place_piece(shadow, shadow.selected_piece)
            label = np.int16(r * 4 + c)

        records.append(
            {
                "board": board_enc,
                "piece": piece_onehot,
                "label": label,
                "action": action,
                "legal_mask": legal_mask,
                "game_id": np.int32(game_id),
            }
        )

        # Let the clone actually play (its choice may differ from expert label).
        game.play_turn()
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
    weights_path = _resolve_weights(args["<clone_path>"], args["--checkpoint"])

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Clone     : {weights_path}")
    print(f"Expert    : MinimaxBot depth={depth}")
    print(f"Opponents : {mix}")
    print(f"Games     : {n_games}")
    print(f"mode_2x2  : {mode_2x2}")
    print(f"Output    : {output_path}\n")

    clone = _load_clone(weights_path)
    expert = MinimaxBot(depth=depth)

    opp_names = [n for n, _ in mix]
    opp_weights = [w for _, w in mix]
    opponents = {n: _make_bot(n) for n in opp_names}

    all_records: list[dict] = []
    for game_id in tqdm(range(n_games), desc="DAgger games", unit="game"):
        opp_name = random.choices(opp_names, weights=opp_weights, k=1)[0]
        opp = opponents[opp_name]
        clone_is_p1 = game_id % 2 == 0
        recs = collect_game(clone, opp, expert, mode_2x2, game_id, clone_is_p1)
        all_records.extend(recs)

    if not all_records:
        print("No records collected — exiting.")
        return

    boards = np.stack([r["board"] for r in all_records])
    pieces = np.stack([r["piece"] for r in all_records])
    labels = np.array([r["label"] for r in all_records], dtype=np.int16)
    actions = np.array([r["action"] for r in all_records], dtype=np.uint8)
    legal_masks = np.stack([r["legal_mask"] for r in all_records])
    game_ids = np.array([r["game_id"] for r in all_records], dtype=np.int32)

    np.savez_compressed(
        output_path,
        boards=boards,
        pieces=pieces,
        labels=labels,
        actions=actions,
        legal_masks=legal_masks,
        game_ids=game_ids,
    )

    n_place = int((actions == ACTION_PLACE).sum())
    n_select = int((actions == ACTION_SELECT).sum())
    print(
        f"\nSaved {len(all_records):,} samples  "
        f"({n_place:,} PLACE + {n_select:,} SELECT)  ->  {output_path}"
    )
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
