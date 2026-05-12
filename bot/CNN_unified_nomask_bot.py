"""CNN_unified_nomask_bot — unified-aux bot that does NOT filter occupied
cells at inference.

Companion to ``models.CNN_unified_nomask.QuartoCNNUnifiedNoMask``. The model
exposes a separate legality head trained via BCE, so legality is supposed to
be encoded in Q_place itself — this bot exists to make that test honest by
picking argmax-Q_place directly, instead of letting the bot's filter paper
over a Q-head that hasn't learned the constraint.

Behaviour
---------
- ``select`` (piece phase): identical to ``CNN_unified_bot`` — the
  available-pieces constraint is a legitimate game rule, not a learnable
  concept.
- ``place_piece`` (board phase): picks the top-ranked move from
  ``predict_phase``; if that move is occupied, records the failure as an
  ``invalid_argmax`` event and falls back to the next valid rank so the
  game can continue. The fallback is needed during training (random init
  has ~0 legality knowledge) but is also what we want to ASYMPTOTE TO
  ZERO — that's the success criterion for the legality aux head.

The bot exposes ``invalid_argmax_count`` and ``valid_argmax_count`` counters
on the instance so a training-loop observer can compute the invalid-rate
metric for the Q3 / Q4 decision gate.
"""

"""
Python 3
2026-05-11
@author: z_tjona
"""

from models.CNN_unified_nomask import QuartoCNNUnifiedNoMask
from bot.CNN_unified_bot import Quarto_bot as Quarto_unified_bot
from models.NN_abstract import NN_abstract

from quartopy import QuartoGame

from utils.logger import logger
import torch


class Quarto_bot(Quarto_unified_bot):
    """Unified-aux bot without the inference-time legality filter."""

    @property
    def name(self) -> str:
        if hasattr(self, "model_path"):
            return f"CNN_unified_nomask_bot|{self.model_path}"
        elif hasattr(self, "model_name"):
            return f"CNN_unified_nomask_bot|{self.model_name}"
        else:
            return "CNN_unified_nomask_bot|random_weights"

    def __init__(
        self,
        *,
        model_path: str | None = None,
        model: NN_abstract | None = None,
        model_class: type[NN_abstract] = QuartoCNNUnifiedNoMask,
        deterministic: bool = True,
        temperature: float = 0.1,
    ):
        super().__init__(
            model_path=model_path,
            model=model,
            model_class=model_class,
            deterministic=deterministic,
            temperature=temperature,
        )
        # Counters for the "did the model learn legality?" metric. Reset
        # externally between evaluations as desired.
        self.invalid_argmax_count: int = 0
        self.valid_argmax_count: int = 0

    # ──────────────────────────────────────────────────────────────────
    # The only override: place-phase board selection without legality filter.
    # ──────────────────────────────────────────────────────────────────
    def _choose_board_position(
        self,
        game: QuartoGame,
        ordered_indices: torch.Tensor,
        ith_option: int,
    ) -> tuple[int, int]:
        ranked = ordered_indices[0].detach().cpu().tolist()
        valid_moves = set(game.game_board.get_valid_moves())

        # Top-ranked argmax (independent of validity)
        top_idx = int(ranked[0])
        top_position = game.game_board.get_position_index(top_idx)
        if top_position in valid_moves:
            self.valid_argmax_count += 1
        else:
            self.invalid_argmax_count += 1

        # ith_option > 0 means "second-best, third-best, ..." — preserve the
        # existing semantics. Resolve to the ith VALID rank so the game can
        # continue.
        valid_ranked: list[tuple[int, int]] = []
        for idx in ranked:
            board_position = game.game_board.get_position_index(int(idx))
            if board_position in valid_moves:
                valid_ranked.append(board_position)
        assert valid_ranked, "No valid board positions exist — game is malformed."
        option_idx = min(ith_option, len(valid_ranked) - 1)
        return valid_ranked[option_idx]

    def invalid_argmax_rate(self) -> float:
        """Fraction of place-phase calls where the top-ranked move was occupied.

        Lower = the network has learned legality. Reported as the headline
        metric for the QC legality-aux experiment.
        """
        total = self.invalid_argmax_count + self.valid_argmax_count
        if total == 0:
            return 0.0
        return self.invalid_argmax_count / total

    def reset_legality_counters(self) -> None:
        self.invalid_argmax_count = 0
        self.valid_argmax_count = 0
