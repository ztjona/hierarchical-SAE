# -*- coding: utf-8 -*-
"""Loss autopsy — why does a unified_autoreg champion lose to a random opponent?

Game-trajectory counterpart to the *static* probes in
``analysis/competence_audit/audit.py`` (Tests A / B). Where Test B asks
"over sampled SELECT positions, does ``argmax Q_select`` avoid losing pieces?"
this script asks the *game-outcome* version: **in the actual games the agent
loses to random play, was the fatal give an avoidable blunder or a forced
position?**

Why every loss-vs-random is a SELECT-side event
-----------------------------------------------
In Quarto you never place the opponent's pieces. The random opponent can
only win by *placing a piece the agent handed it* into a completing line.
So every loss decomposes at the agent's final give into exactly one of:

* **avoidable** — the given piece had an immediate winning placement *while a
  safe (non-completing) piece was still available in storage*. A pure
  ``Q_select`` blunder; the residual ``1 − safe_piece_recall``. Fixable in the
  weights.
* **forced**   — *every* available piece completes some line. The agent was
  already lost; random punished it probabilistically. The irreducible floor.

There is no third class against a random opponent (it executes no multi-move
forcing lines). The fatal give is *always* a hot give by construction — the
only question this autopsy answers is whether a safe piece existed at that
give.

The classification rules (``_placing_wins`` / ``_piece_is_losing`` /
``_winning_cells``) are imported verbatim from ``audit.py`` so there is no
rule drift versus Tests A / B.

Output
------
One JSONL record per (exp, epoch) appended to:
    analysis/competence_audit/results/<exp>/loss_autopsy.jsonl

Usage
-----
    python analysis/competence_audit/loss_autopsy.py \\
        --exp 'Ve_oracleAblation(4)0522_DISABLE_NEVER_10k' \\
        [--epoch N] \\
        [--n-games 1000]          # per direction (agent as P1, then as P2)
        [--opponent uniform|benchmark]
        [--stochastic --temperature 0.1]   # default: deterministic argmax
        [--seed 1234]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import torch
from tqdm.auto import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from quartopy import Board, Piece, QuartoGame, BotAI  # noqa: E402

from bot.CNN_unified_bot import Quarto_bot as Quarto_unified_bot  # noqa: E402
from bot.random_bot import Quarto_bot as RandomBot  # noqa: E402

# Reuse checkpoint loading from the qselect suite and the rule helpers from
# the static audit, so classification is byte-for-byte identical to Tests A/B.
sys.path.insert(0, os.path.join(ROOT, "analysis", "qselect_diagnostics"))
from _common import load_checkpoint  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
from audit import _placing_wins, _winning_cells, _piece_is_losing  # noqa: E402,F401

RESULTS_DIR = os.path.join(THIS_DIR, "results")
MAX_EXAMPLES = 20


# ──────────────────────────────────────────────────────────────────────
# Recorder wrapper — delegates to the agent bot, snapshots its decisions
# ──────────────────────────────────────────────────────────────────────


class _RecorderBot(BotAI):
    """Thin pass-through around the agent bot that snapshots each decision.

    * ``select``      → records the board snapshot + storage pool + chosen
      piece for *the most recent give* (overwritten each turn; at game end it
      holds the fatal give).
    * ``place_piece`` → tallies missed immediate wins (game-level Test A).
    """

    def __init__(self, inner: BotAI):
        self.inner = inner
        self.reset()

    @property
    def name(self) -> str:
        return self.inner.name

    def reset(self) -> None:
        self.last_give: dict | None = None
        self.win_opportunities = 0
        self.missed_wins = 0

    # -- snapshot helpers ------------------------------------------------

    @staticmethod
    def _board_cells(board: Board) -> tuple[list[tuple[int, int, int]], list[tuple[int, int]]]:
        """Return (occupied as (piece_idx, r, c), empties as (r, c))."""
        occupied: list[tuple[int, int, int]] = []
        empties: list[tuple[int, int]] = []
        for r in range(4):
            for c in range(4):
                cell = board.board[r][c]
                if isinstance(cell, Piece):
                    occupied.append((cell.index(), r, c))
                else:
                    empties.append((r, c))
        return occupied, empties

    # -- gameplay API ----------------------------------------------------

    def select(self, game: QuartoGame, ith_option: int = 0, *args, **kwargs) -> Piece:
        piece = self.inner.select(game, ith_option, *args, **kwargs)
        # Snapshot on every call; the last call of the turn (the used one) wins,
        # and later turns overwrite, so at game end this is the fatal give.
        occupied, empties = self._board_cells(game.game_board)
        self.last_give = {
            "occupied": occupied,
            "empties": empties,
            "available_idx": [p.index() for p in game.storage_board.get_valid_pieces()],
            "chosen_idx": piece.index(),
            "n_placed": len(occupied),
        }
        return piece

    def place_piece(
        self, game: QuartoGame, piece: Piece, ith_option: int = 0, *args, **kwargs
    ) -> tuple[int, int]:
        rc = self.inner.place_piece(game, piece, ith_option, *args, **kwargs)
        if ith_option == 0:
            # Board here is pre-placement (QuartoGame.put_piece runs after this).
            _, empties = self._board_cells(game.game_board)
            wins = _winning_cells(game.game_board, piece, empties)
            if wins:
                self.win_opportunities += 1
                if tuple(rc) not in wins:
                    self.missed_wins += 1
        return rc


# ──────────────────────────────────────────────────────────────────────
# Classification of a single fatal give
# ──────────────────────────────────────────────────────────────────────


def _rebuild_board(occupied: list[tuple[int, int, int]]) -> Board:
    b = Board("autopsy", storage=False, rows=4, cols=4)
    for piece_idx, r, c in occupied:
        b.put_piece(Piece.from_index(piece_idx), r, c)
    return b


def classify_give(give: dict) -> dict:
    """Classify the agent's fatal give. Mirrors Test-B rule definitions."""
    board = _rebuild_board(give["occupied"])
    empties = [tuple(rc) for rc in give["empties"]]
    available = [Piece.from_index(i) for i in give["available_idx"]]
    losing = [p for p in available if _piece_is_losing(board, p, empties)]
    safe = [p for p in available if p not in losing]
    chosen = Piece.from_index(give["chosen_idx"])
    chosen_is_losing = _piece_is_losing(board, chosen, empties)

    return {
        "chosen_is_losing": chosen_is_losing,
        "n_available": len(available),
        "n_safe": len(safe),
        "n_losing": len(losing),
        "n_placed": give["n_placed"],
        # Against a random opponent a real loss implies the chosen piece was
        # hot; "avoidable" iff a safe piece existed at that give.
        "avoidable": bool(chosen_is_losing and len(safe) > 0),
        "forced": bool(chosen_is_losing and len(safe) == 0),
        "anomalous": bool(not chosen_is_losing),
        "board_serial": board.serialize(),
    }


# ──────────────────────────────────────────────────────────────────────
# Play one direction (agent fixed as P1 or P2) and autopsy its losses
# ──────────────────────────────────────────────────────────────────────


def _run_direction(
    agent_bot: BotAI,
    opp_bot: BotAI,
    *,
    n_games: int,
    agent_is_p1: bool,
    mode_2x2: bool,
    examples: list[dict],
) -> dict:
    rec = _RecorderBot(agent_bot)
    agent_pos = "Player 1" if agent_is_p1 else "Player 2"
    stats = {
        "games": 0, "wins": 0, "losses": 0, "ties": 0,
        "avoidable": 0, "forced": 0, "anomalous": 0,
        "win_opportunities": 0, "missed_wins": 0,
        "n_safe_at_blunder": [], "n_placed_at_loss": [],
        "n_placed_hist": Counter(),
    }
    desc = f"Autopsy (agent={agent_pos})"
    for _ in tqdm(range(n_games), desc=desc, mininterval=0.3, leave=False):
        rec.reset()
        if agent_is_p1:
            game = QuartoGame(player1=rec, player2=opp_bot, mode_2x2=mode_2x2)
        else:
            game = QuartoGame(player1=opp_bot, player2=rec, mode_2x2=mode_2x2)

        while not game.player_won and not game.game_board.is_full():
            game.play_turn()
            game.cambiar_turno()

        stats["games"] += 1
        stats["win_opportunities"] += rec.win_opportunities
        stats["missed_wins"] += rec.missed_wins

        if game.winner_pos == game.TIE:
            stats["ties"] += 1
        elif game.winner_pos == agent_pos:
            stats["wins"] += 1
        else:
            stats["losses"] += 1
            if rec.last_give is None:  # opponent won on the agent's first placement? impossible, but guard
                stats["anomalous"] += 1
                continue
            verdict = classify_give(rec.last_give)
            if verdict["avoidable"]:
                stats["avoidable"] += 1
                stats["n_safe_at_blunder"].append(verdict["n_safe"])
            elif verdict["forced"]:
                stats["forced"] += 1
            else:
                stats["anomalous"] += 1
            stats["n_placed_at_loss"].append(verdict["n_placed"])
            stats["n_placed_hist"][verdict["n_placed"]] += 1
            if len(examples) < MAX_EXAMPLES:
                examples.append({"direction": agent_pos, **verdict})
    return stats


def _merge(a: dict, b: dict) -> dict:
    out: dict = {}
    for k in ("games", "wins", "losses", "ties", "avoidable", "forced",
              "anomalous", "win_opportunities", "missed_wins"):
        out[k] = a[k] + b[k]
    out["n_safe_at_blunder"] = a["n_safe_at_blunder"] + b["n_safe_at_blunder"]
    out["n_placed_at_loss"] = a["n_placed_at_loss"] + b["n_placed_at_loss"]
    out["n_placed_hist"] = a["n_placed_hist"] + b["n_placed_hist"]
    return out


# ──────────────────────────────────────────────────────────────────────
# Opponent construction
# ──────────────────────────────────────────────────────────────────────


def _build_opponent(kind: str) -> tuple[BotAI, str]:
    if kind == "uniform":
        return RandomBot(), "random_uniform"
    if kind == "benchmark":
        # Reuse the exact "Random Baseline" the champion benchmark uses
        # (epoch-0 CNN, sampled) so the loss rate is comparable to
        # champion-results.jsonl.
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import benchmark_champion as bc  # noqa: E402

        cfg = bc.load_config(bc.DEFAULT_CONFIG_PATH)
        bot, name = bc.load_bot("random", cfg)
        return bot, f"benchmark:{name}"
    raise ValueError(f"Unknown opponent kind {kind!r}")


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Counter):
        return dict(obj)
    raise TypeError(f"Not JSON-serializable: {type(obj)}")


def _emit_jsonl(exp_name: str, record: dict) -> str:
    out_dir = os.path.join(RESULTS_DIR, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "loss_autopsy.jsonl")
    with open(out_path, "a") as f:
        f.write(json.dumps(record, default=_json_default) + "\n")
    return out_path


def autopsy(
    exp_name: str,
    *,
    epoch: int | None,
    n_games: int,
    opponent: str,
    deterministic: bool,
    temperature: float,
    mode_2x2: bool,
    seed: int,
    architecture: str | None,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    t0 = time.time()
    net, cfg = load_checkpoint(exp_name, epoch=epoch, architecture=architecture)
    agent_bot = Quarto_unified_bot(
        model=net, deterministic=deterministic, temperature=temperature
    )
    opp_bot, opp_name = _build_opponent(opponent)

    print(f"\n{'='*64}")
    print(f"  Loss autopsy: {exp_name}  (epoch {cfg['epoch']}, {cfg['architecture']})")
    print(f"  agent: {'argmax' if deterministic else f'sampled@T={temperature}'}"
          f"   opponent: {opp_name}   games: {n_games}/direction   seed: {seed}")
    print(f"{'='*64}")

    examples: list[dict] = []
    p1 = _run_direction(agent_bot, opp_bot, n_games=n_games, agent_is_p1=True,
                        mode_2x2=mode_2x2, examples=examples)
    p2 = _run_direction(agent_bot, opp_bot, n_games=n_games, agent_is_p1=False,
                        mode_2x2=mode_2x2, examples=examples)
    total = _merge(p1, p2)

    losses = total["losses"]
    avoidable = total["avoidable"]
    forced = total["forced"]
    games = total["games"]

    def _safe_div(a, b):
        return (a / b) if b else None

    record = {
        "schema_version": 1,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "exp_name": exp_name,
        "epoch": cfg["epoch"],
        "checkpoint_path": cfg["checkpoint_path"],
        "architecture": cfg["architecture"],
        "opponent": opp_name,
        "agent_deterministic": deterministic,
        "agent_temperature": temperature,
        "mode_2x2": mode_2x2,
        "seed": seed,
        "n_games_per_direction": n_games,
        "elapsed_seconds": round(time.time() - t0, 1),
        # outcomes
        "games": games,
        "wins": total["wins"],
        "losses": losses,
        "ties": total["ties"],
        "loss_rate": _safe_div(losses, games),
        "win_rate": _safe_div(total["wins"], games),
        # loss decomposition
        "losses_avoidable": avoidable,
        "losses_forced": forced,
        "losses_anomalous": total["anomalous"],
        "avoidable_fraction_of_losses": _safe_div(avoidable, losses),
        "avoidable_rate": _safe_div(avoidable, games),     # the reducible floor
        "forced_rate": _safe_div(forced, games),           # the irreducible floor
        "n_safe_at_blunder_mean": (
            float(np.mean(total["n_safe_at_blunder"])) if total["n_safe_at_blunder"] else None
        ),
        "n_placed_at_loss_mean": (
            float(np.mean(total["n_placed_at_loss"])) if total["n_placed_at_loss"] else None
        ),
        "n_placed_at_loss_hist": dict(sorted(total["n_placed_hist"].items())),
        # place-side (game-level Test A)
        "place_audit": {
            "win_opportunities": total["win_opportunities"],
            "missed_wins": total["missed_wins"],
            "missed_win_rate": _safe_div(total["missed_wins"], total["win_opportunities"]),
        },
        "by_direction": {
            "agent_p1": {k: p1[k] for k in ("games", "wins", "losses", "ties", "avoidable", "forced")},
            "agent_p2": {k: p2[k] for k in ("games", "wins", "losses", "ties", "avoidable", "forced")},
        },
        "examples": examples,
    }

    out_path = _emit_jsonl(exp_name, record)
    _print_summary(record, os.path.relpath(out_path, ROOT))
    return record


def _pct(x):
    return f"{100*x:.2f}%" if isinstance(x, float) else " — "


def _print_summary(rec: dict, out_path_display: str) -> None:
    print(f"\n  games={rec['games']}  wins={rec['wins']}  losses={rec['losses']}"
          f"  ties={rec['ties']}   loss_rate={_pct(rec['loss_rate'])}")
    print(f"  losses split: avoidable={rec['losses_avoidable']}  "
          f"forced={rec['losses_forced']}  anomalous={rec['losses_anomalous']}")
    print(f"    → {_pct(rec['avoidable_fraction_of_losses'])} of losses were AVOIDABLE "
          f"(safe piece existed at the fatal give)")
    print(f"    → reducible (avoidable) rate = {_pct(rec['avoidable_rate'])} of games;  "
          f"irreducible (forced) floor = {_pct(rec['forced_rate'])}")
    if rec["n_safe_at_blunder_mean"] is not None:
        print(f"    mean safe pieces available at an avoidable blunder: "
              f"{rec['n_safe_at_blunder_mean']:.2f}")
    if rec["n_placed_at_loss_mean"] is not None:
        print(f"    mean pieces on board at the fatal give: {rec['n_placed_at_loss_mean']:.1f}")
    pa = rec["place_audit"]
    print(f"  place-side (Test A live): missed {pa['missed_wins']}/{pa['win_opportunities']} "
          f"immediate wins  (miss rate {_pct(pa['missed_win_rate'])})")
    bd = rec["by_direction"]
    print(f"  by side: P1 {bd['agent_p1']['losses']}L (av {bd['agent_p1']['avoidable']}/"
          f"fo {bd['agent_p1']['forced']})   "
          f"P2 {bd['agent_p2']['losses']}L (av {bd['agent_p2']['avoidable']}/"
          f"fo {bd['agent_p2']['forced']})")
    print(f"  -> {out_path_display}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Loss autopsy for unified_autoreg checkpoints vs a random opponent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--exp", required=True, help="Experiment name under CHECKPOINTS/")
    p.add_argument("--epoch", type=int, default=None, help="Epoch to load (default: latest)")
    p.add_argument("--n-games", type=int, default=1000,
                   help="Games per direction (agent as P1, then as P2)")
    p.add_argument("--opponent", choices=["uniform", "benchmark"], default="uniform",
                   help="uniform = bot/random_bot; benchmark = champion_config 'random' epoch-0 CNN")
    p.add_argument("--stochastic", action="store_true",
                   help="Sample the agent's actions (default: deterministic argmax)")
    p.add_argument("--temperature", type=float, default=0.1,
                   help="Agent sampling temperature (only used with --stochastic)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--architecture", default=None,
                   help="Override architecture (default: auto-inferred from exp name)")
    p.add_argument("--no-2x2", action="store_true", help="Disable 2x2 win mode (NOT default)")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    try:
        autopsy(
            args.exp,
            epoch=args.epoch,
            n_games=args.n_games,
            opponent=args.opponent,
            deterministic=not args.stochastic,
            temperature=args.temperature,
            mode_2x2=not args.no_2x2,
            seed=args.seed,
            architecture=args.architecture,
        )
    except Exception as exc:
        print(f"[ERROR] {args.exp}: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
