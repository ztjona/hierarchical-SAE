# -*- coding: utf-8 -*-

"""
Python 3
01 / 06 / 2025
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""

from utils.logger import logger

from tensordict import TensorDict

from quartopy import play_games, BotAI, Board

import torch
import numpy as np
import pandas as pd
from models import NN_abstract

import numpy as np

import numpy as np
import pandas as pd


TRANSITION_SCHEMA_JOINT = "joint"
TRANSITION_SCHEMA_DECOUPLED_AUTOREG = "decoupled_autoreg"
TRANSITION_SCHEMA_UNIFIED_AUTOREG = "unified_autoreg"

DECOUPLED_TARGET_TD_PLACE_MC_SELECT = "td_place_mc_select"
DECOUPLED_TARGET_MC_BOTH = "mc_both"
DECOUPLED_TARGET_TD_PLACE_TD_SELECT = "td_place_td_select"
# T-series: Q_place uses TD/Bellman; Q_select is supervised by per-piece
# minimax-oracle scores captured at experience generation time (the policy
# of that moment — targets are frozen in the buffer, not recomputed on
# replay). Requires `target_sel_minimax` and `target_sel_minimax_mask`
# fields in the batch.
DECOUPLED_TARGET_TD_PLACE_MINIMAX_SELECT = "td_place_minimax_select"
# T-series Tc variant: same minimax oracle, but supervise only the scalar
# Q_select value at the chosen piece (matches the standard scalar 4-tuple
# return shape used by all other styles). Diagnostic — isolates "oracle
# replaces MC noise" from "oracle gives 16× more signal per state".
DECOUPLED_TARGET_TD_PLACE_MINIMAX_SELECT_SCALAR = "td_place_minimax_select_scalar"
DISCOUNT_REWARD_GAMMA = 0.8

PHASE_PLACE = 0
PHASE_SELECT = 1

UNIFIED_AUX_DIM = 32  # offered_one_hot (16) ⊕ available_mask (16)

JOINT_TENSORDICT_KEYS = (
    "state_board",
    "state_piece",
    "action_place",
    "action_sel",
    "reward",
    "done",
    "next_state_board",
    "next_state_piece",
    "outcome",
    "steps_to_terminal",
)

DECOUPLED_AUTOREG_TENSORDICT_KEYS = (
    "state_board",
    "state_aux",
    "phase",
    "valid_mask",
    "action",
    "reward",
    "done",
    "next_state_board",
    "next_state_aux",
    "next_phase",
    "next_valid_mask",
    "outcome",
    "steps_to_terminal",
)

# unified_autoreg shares the decoupled key set; only state_aux/next_state_aux
# change shape (16 → 32). Kept as a separate constant so call sites can verify
# the schema name independently of the keys it carries.
UNIFIED_AUTOREG_TENSORDICT_KEYS = DECOUPLED_AUTOREG_TENSORDICT_KEYS


def _piece_index_to_vector(piece_index: int) -> np.ndarray:
    if piece_index == -1:
        return np.zeros(16, dtype=np.float32)
    vector = Board.pos_index2vector(piece_index)
    return np.asarray(vector, dtype=np.float32)


def _available_pieces_mask(available_pieces: set[int]) -> np.ndarray:
    mask = np.zeros(16, dtype=np.float32)
    if available_pieces:
        mask[sorted(available_pieces)] = 1.0
    return mask


def _unified_aux(piece_index: int, available_pieces: set[int]) -> np.ndarray:
    """Build the 32-d phase-stable aux vector for the unified-autoreg schema.

    Layout: ``[offered_one_hot (16) ; available_pieces_mask (16)]``.
    ``piece_index == -1`` (no offered piece, e.g. the very first select of a
    game) maps to a zero block.
    """
    offered = _piece_index_to_vector(piece_index)
    available = _available_pieces_mask(available_pieces)
    return np.concatenate([offered, available]).astype(np.float32)


def _minimax_select_target(
    oracle,
    board_serial: str,
    available_pieces: set[int],
    mode_2x2: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Score every legal SELECT action with a minimax oracle.

    Builds a transient ``QuartoGame`` from the serialized board + available
    storage state and queries ``oracle.score_all_moves(game)`` at the SELECT
    phase. Returns ``(target_vec, mask_vec)`` both shape ``(16,)``:

    - ``target_vec[piece_idx]``: minimax value normalized to roughly ``[-1, 1]``,
      with the sign flipped so that **higher target = better SELECT** (matches
      the Q_select convention, since the raw minimax score uses lower=better
      for the selector).
    - ``mask_vec[piece_idx]``: 1.0 for legal pieces with a valid score, 0.0
      elsewhere.

    The transient ``QuartoGame`` reuses the project's ``Quarto_bot`` (random)
    as placeholder players — ``score_all_moves`` does not invoke them; it
    only reads ``game.game_board``, ``game.storage_board``, ``game.pick``,
    and ``game.mode_2x2``.
    """
    from quartopy import QuartoGame
    from bot.random_bot import Quarto_bot
    from bot.minimax_bot import MinimaxBot

    assert isinstance(oracle, MinimaxBot), (
        f"select_oracle must be a MinimaxBot, got {type(oracle).__name__}"
    )

    game = QuartoGame(
        player1=Quarto_bot(), player2=Quarto_bot(), mode_2x2=mode_2x2
    )
    # Replace the empty game_board with the actual state.
    game.game_board = Board.serialized_2_board(board_serial)
    # Prune storage_board down to ``available_pieces``. The default
    # storage starts with all 16 pieces; remove the ones already taken.
    for piece_obj in list(game.storage_board.get_valid_pieces()):
        if int(piece_obj.index()) not in available_pieces:
            coord = game.storage_board.find_piece(piece_obj)
            if coord is not None:
                game.storage_board.remove_piece(*coord)
    game.pick = True  # SELECT phase
    game.selected_piece = None

    scores, action_kind = oracle.score_all_moves(game)
    assert action_kind == 1, (
        f"Expected SELECT action_kind=1 from oracle, got {action_kind}"
    )

    target = np.zeros(16, dtype=np.float32)
    mask = np.zeros(16, dtype=np.float32)
    # Normalize so terminal +/-(100+depth) maps to ~+/-1. The selector
    # convention is "lower minimax score = better piece to give", so we
    # negate to align with Q_select where higher = better.
    scale = 100.0 + float(oracle.depth)
    for piece_idx, score in scores.items():
        target[piece_idx] = float(np.clip(-score / scale, -1.0, 1.0))
        mask[piece_idx] = 1.0
    return target, mask


def _valid_position_mask(board_state: str) -> np.ndarray:
    board = Board.serialized_2_board(board_state)
    valid_moves = set(board.get_valid_moves())
    mask = np.zeros(16, dtype=np.float32)
    for idx in range(16):
        if board.get_position_index(idx) in valid_moves:
            mask[idx] = 1.0
    return mask


def _actor_outcome(player_pos: str, match_result: str) -> float:
    if match_result == "Tie":
        return 0.0
    return 1.0 if player_pos == match_result else -1.0


def _phase_reward(
    *,
    REWARD_FUNCTION_TYPE: str,
    phase: int,
    outcome: float,
    done: bool,
    steps_to_terminal: int,
) -> float:
    if REWARD_FUNCTION_TYPE == "final":
        return float(outcome) if phase == PHASE_PLACE and done else 0.0
    if REWARD_FUNCTION_TYPE == "propagate":
        return float(outcome)
    if REWARD_FUNCTION_TYPE == "discount":
        return float((DISCOUNT_REWARD_GAMMA**steps_to_terminal) * outcome)
    raise ValueError(f"Unknown REWARD_FUNCTION_TYPE {REWARD_FUNCTION_TYPE}")


def _masked_max(q_values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    masked_q = q_values.masked_fill(valid_mask <= 0, float("-inf"))
    max_values = masked_q.max(dim=1).values
    if not torch.isfinite(max_values).all():
        raise ValueError("Encountered a next-state row with no valid actions.")
    return max_values


def convert_2_state_action_reward(match_data, REWARD_FUNCTION_TYPE: str = "propagate"):
    """
    Convert match data to state, action, reward format for RL training.

    Args:
        match_data (dict): Match data from Quartopy.
        REWARD_FUNCTION_TYPE (str): Type of reward function to use. Options are "propagate", "final", "discount".

    Returns:
        pd.DataFrame: DataFrame with columns:
            - board_state: Board state before action (str).
            - board_next_state: Board state after action (str).
            - piece_state: Piece given by opponent in previous turn (int).
            - piece_next_state: Piece given to opponent (int).
            - mov_description: Description of the move (str).
            - action_place: Position index to place the piece (int).
            - action_sel: Piece index to give to opponent (int).
            - reward: Reward for the state (int/float).
            - done: Whether the state is terminal (bool).
    """
    # When ``REWARD_FUNCTION_TYPE`` == "propagate", the reward of the final state is propagated to all previous states.
    # When "final", only the final state has reward, and all previous states have reward 0.
    # When "discount", rewards are discounted over time.
    # The reward of the final state is 1 (P1 win), -1 (P1 lose) or 0 (draw).

    m_h = match_data["move_history"]
    # Actions place and select are combined in only one action, state, reward
    # State is represented in ``state_boards`` and ``state_piece``
    # Action is represented in ``action_place`` and ``action_sel``
    # Assuming SELF PLAY: the bot plays both players
    # Rewards are from the perspective of the player taking the action (alternating P1/P2)
    # NOTE: ``board_next_state`` includes the final board after placing the piece and thus has the same size as ``board_state`` in winning or drawing.
    # NOTE: ``piece_next_state`` is -1 for terminal states (winning moves), because there is no piece selection after winning.

    _current_board = "0"  # first state board is empty
    # The state and next_board include empty board, because:
    # Turn 1 of player 1 sees empty board
    # And, at the start of Turn 1 of player 2 it also sees empty board
    board_state = ["0"]  # board before taking action
    board_next_state = ["0"]  # board after taking action

    # pos_index2vector returns all zeros if piece=-1.
    piece_state = [-1]  # piece given by the opponent ni previous turn
    piece_next_state = []  # piece given to the opponent

    # position to put the piece #index from 0 to 15
    # -1 means no piece placed yet
    action_place = [-1]
    action_sel = []  # piece to give to the opponent #index from 0 to 15

    reward = []
    done = []
    mov_description: list[str] = []

    for i, move in enumerate(m_h):
        # --- INDEX
        if move["action"] == "selected":
            mov_description.append(f"{i}|{move['action']}|{move['player_pos']}")

        elif move["action"] == "placed":
            pass

        # --- BOARD
        if move["action"] == "selected":
            pass

        elif move["action"] == "placed":
            _next_board = move["board_after"]

            board_state.append(_current_board)
            board_next_state.append(_next_board)
            _current_board = _next_board
        else:
            raise ValueError(f"Unknown action {move['action']}")

        # --- PIECE
        if move["action"] == "selected":
            _current_piece = move["piece_index"]
            piece_next_state.append(_current_piece)

        elif move["action"] == "placed":
            piece_state.append(_current_piece)

        # --- ACTION
        if move["action"] == "selected":
            action_sel.append(move["piece_index"])

        elif move["action"] == "placed":
            action_place.append(move["position_index"])

    assert len(piece_state) == len(piece_next_state) + 1

    if len(mov_description) != len(board_state):
        mov_description.append(f"{i}|{move['action']}|{move['player_pos']}")

    action_sel.append(-1)  # no piece selected after the last move
    piece_next_state.append(-1)  # no piece given after the last move

    _num_states = len(piece_state)
    _num_non_terminal_states = _num_states - 1

    # --- REWARD
    match match_data["result"]:
        case "Player 1":
            R = 1
        case "Player 2":
            R = -1
        case "Tie":
            R = 0
        case _:
            raise ValueError(f"Unknown result {match_data['result']}")

    R_2 = -R  # reward from perspective of player 2

    # Apply reward function
    match REWARD_FUNCTION_TYPE:
        case "final":
            reward = [0] * (_num_states - 2)
            if R == 0:
                reward.extend([0, 0])
            else:
                # last move is winning move for a player
                reward.extend([-1, 1])
        case "propagate":
            reward = [R if i % 2 == 0 else R_2 for i in range(_num_states)]
        case "discount":
            gamma = 0.8
            reward = [
                1 * (gamma**i) * (-1) ** (i % 2 == 1)
                for i in reversed(range(_num_states))
            ]
        case _:
            raise ValueError(f"Unknown REWARD_FUNCTION_TYPE {REWARD_FUNCTION_TYPE}")

    # ---- DONE
    done = [False] * _num_non_terminal_states
    done.append(True)  # only last state is the terminal state

    # ---- OUTCOME (per-state player perspective, independent of REWARD_FUNCTION_TYPE)
    # +1 if this state's player eventually wins, -1 if loses, 0 if tie.
    # P1 plays at i%2==0, P2 at i%2==1. Final result R is from P1's perspective.
    outcome = [R if i % 2 == 0 else R_2 for i in range(_num_states)]

    # ---- STEPS_TO_TERMINAL (distance from state i to the terminal state T-1)
    # Used to form Monte Carlo targets: G_i = gamma^steps_to_terminal_i * outcome_i
    steps_to_terminal = [_num_states - 1 - i for i in range(_num_states)]

    df = pd.DataFrame(
        {
            # but board_state and board_next_state are str
            "board_state": board_state,
            "board_next_state": board_next_state,
            "mov_description": mov_description,
            # the rest are int
            "piece_state": piece_state,
            "piece_next_state": piece_next_state,
            "action_place": action_place,
            "action_sel": action_sel,
            "reward": reward,
            "done": done,
            "outcome": outcome,
            "steps_to_terminal": steps_to_terminal,
        }
    )
    return df


# ####################################################################
def gen_experience(
    *,
    p1_bot: BotAI,
    p2_bot: BotAI,
    n_last_states: int = 16,
    number_of_matches: int = 1000,
    verbose: bool = False,
    PROGRESS_MESSAGE: str = "Generating experience",
    mode_2x2: bool = False,
    REWARD_FUNCTION_TYPE: str = "propagate",
    TRANSITION_SCHEMA: str = TRANSITION_SCHEMA_JOINT,
    COLLECT_BOARDS: bool = False,
    select_oracle=None,
) -> TensorDict | tuple[TensorDict, list[tuple[Board, Board]]]:
    """
    Generates experience by having two bots play against each other. The experience is returned as a TensorDict.
    ## Parameters
    ``p1_bot``: BotAI
        The first bot to play.

    ``p2_bot``: BotAI
        The second bot to play.

    ``n_last_states``: int
        Number of last states to consider for each match. Default is 16, i.e. all states of the match.

    ``number_of_matches``: int
        Number of matches to be played between the two bots. Default is 1000.

    ``mode_2x2``: bool
        If True, activates the 2x2 victory mode. Default is False.
    ``REWARD_FUNCTION_TYPE``: str
        Type of reward function to use. Options are "propagate", "final", "discount". Default is "propagate".
    ``COLLECT_BOARDS``: bool
        If True, collects the board states during the matches. Default is False.

    ## Returns
    TensorDict or tuple[TensorDict, list[tuple[Board, Board]]]
        If COLLECT_BOARDS is False (default):
            Returns only TensorDict with experience data
        If COLLECT_BOARDS is True:
            Returns tuple of (TensorDict, list of board pairs)

    TensorDict contains:
        - state_board: Board states (N, 16, 4, 4)
        - state_piece: Piece one-hot vectors (N, 16)
        - action_place: Placement actions (N,). -1 for first moves.
        - action_sel: Selection actions (N,). -1 for terminal states.
        - reward: Rewards (N,)
        - done: Terminal state flags (N,)
        - next_state_board: Next board states (N, 16, 4, 4)
        - next_state_piece: Next piece vectors (N, 16)

    Where N is the total number of states collected (varies by matches).
    """
    if TRANSITION_SCHEMA == TRANSITION_SCHEMA_DECOUPLED_AUTOREG:
        return gen_experience_decoupled_autoreg(
            p1_bot=p1_bot,
            p2_bot=p2_bot,
            n_last_states=n_last_states,
            number_of_matches=number_of_matches,
            verbose=verbose,
            PROGRESS_MESSAGE=PROGRESS_MESSAGE,
            mode_2x2=mode_2x2,
            REWARD_FUNCTION_TYPE=REWARD_FUNCTION_TYPE,
            COLLECT_BOARDS=COLLECT_BOARDS,
        )
    if TRANSITION_SCHEMA == TRANSITION_SCHEMA_UNIFIED_AUTOREG:
        return gen_experience_unified_autoreg(
            p1_bot=p1_bot,
            p2_bot=p2_bot,
            n_last_states=n_last_states,
            number_of_matches=number_of_matches,
            verbose=verbose,
            PROGRESS_MESSAGE=PROGRESS_MESSAGE,
            mode_2x2=mode_2x2,
            REWARD_FUNCTION_TYPE=REWARD_FUNCTION_TYPE,
            COLLECT_BOARDS=COLLECT_BOARDS,
            select_oracle=select_oracle,
        )
    if select_oracle is not None and TRANSITION_SCHEMA == TRANSITION_SCHEMA_JOINT:
        raise NotImplementedError(
            "select_oracle is only supported for TRANSITION_SCHEMA='unified_autoreg' "
            "(and would need parallel wiring through decoupled_autoreg)."
        )
    if TRANSITION_SCHEMA != TRANSITION_SCHEMA_JOINT:
        raise ValueError(f"Unknown TRANSITION_SCHEMA {TRANSITION_SCHEMA}")

    logger.debug("Generating experience...")

    matches_data, _ = play_games(  # _ winrate
        matches=number_of_matches,
        player1=p1_bot,
        player2=p2_bot,
        delay=0,
        verbose=verbose,
        PROGRESS_MESSAGE=PROGRESS_MESSAGE,
        save_match=False,
        mode_2x2=mode_2x2,
    )

    logger.debug(f"Generated experience. Matches played: {number_of_matches}.")

    exp_all = []

    if COLLECT_BOARDS:
        boards: list[tuple[Board, Board]] = []

    c_matches_shorter = 0
    exp_sizes = []
    for match_data in matches_data:
        exp = convert_2_state_action_reward(
            match_data, REWARD_FUNCTION_TYPE=REWARD_FUNCTION_TYPE
        )

        if n_last_states <= exp.shape[0]:
            exp = exp.iloc[-n_last_states:]
        elif n_last_states > exp.shape[0]:
            exp_sizes.append(exp.shape[0])
            c_matches_shorter += 1
        exp_all.append(exp)

        if COLLECT_BOARDS:
            # Collect final boards
            for _, b in exp.iterrows():
                boards.append(
                    (
                        Board.serialized_2_board(
                            b["board_state"],
                            name=f"{b['mov_description']} | R={b['reward']:.2f}",
                        ),
                        Board.serialized_2_board(
                            b["board_next_state"],
                            name=f"{b['mov_description']} | R={b['reward']:.2f}",
                        ),
                    )
                )

    if c_matches_shorter > 0:
        logger.warning(
            f"n_last_states ({n_last_states}) is greater than the number of states in the match (avg:{np.mean(exp_sizes)}) ({c_matches_shorter} times). Using all states."
        )
    p_all = pd.concat(exp_all, ignore_index=True)  # just for easy concat

    logger.debug(f"Total states collected: {p_all.shape[0]}")

    # conversion to NN input-output format
    # must be float32 for torch...
    experience = TensorDict(
        {
            "state_board": torch.tensor(
                np.stack(p_all["board_state"].apply(Board.deserialize)),  # type: ignore
                dtype=torch.float32,
            ),
            "next_state_board": torch.tensor(
                np.stack(p_all["board_next_state"].apply(Board.deserialize)),  # type: ignore
                dtype=torch.float32,
            ),
            "state_piece": torch.tensor(
                np.stack(p_all["piece_state"].apply(Board.pos_index2vector)),  # type: ignore
                dtype=torch.float32,
            ),
            "next_state_piece": torch.tensor(
                np.stack(p_all["piece_next_state"].apply(Board.pos_index2vector)),  # type: ignore
                dtype=torch.float32,
            ),
            "action_place": torch.tensor(
                p_all["action_place"].to_numpy(), dtype=torch.float32
            ),  # -1 means no action
            "action_sel": torch.tensor(
                p_all["action_sel"].to_numpy(), dtype=torch.float32
            ),  # -1 means no action
            "reward": torch.tensor(p_all["reward"].to_numpy(), dtype=torch.float32),
            "done": torch.tensor(p_all["done"].to_numpy(), dtype=torch.bool),
            "outcome": torch.tensor(p_all["outcome"].to_numpy(), dtype=torch.float32),
            "steps_to_terminal": torch.tensor(
                p_all["steps_to_terminal"].to_numpy(), dtype=torch.float32
            ),
        },
        batch_size=[p_all.shape[0]],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    if COLLECT_BOARDS:
        return experience, boards
    else:
        return experience


def gen_experience_decoupled_autoreg(
    *,
    p1_bot: BotAI,
    p2_bot: BotAI,
    n_last_states: int = 16,
    number_of_matches: int = 1000,
    verbose: bool = False,
    PROGRESS_MESSAGE: str = "Generating experience",
    mode_2x2: bool = False,
    REWARD_FUNCTION_TYPE: str = "propagate",
    COLLECT_BOARDS: bool = False,
):
    """Generate phase-decoupled autoregressive transitions from move history.

    Each stored row corresponds to exactly one action phase:
      - place: board before placing + incoming piece one-hot
      - select: board after placing + available-piece mask

    ``n_last_states`` preserves the original joint-turn semantics by keeping only
    the transitions that belong to the last N combined turn states.
    """
    logger.debug("Generating decoupled-autoregressive experience...")

    matches_data, _ = play_games(
        matches=number_of_matches,
        player1=p1_bot,
        player2=p2_bot,
        delay=0,
        verbose=verbose,
        PROGRESS_MESSAGE=PROGRESS_MESSAGE,
        save_match=False,
        mode_2x2=mode_2x2,
    )

    rows: list[dict] = []
    boards: list[tuple[Board, Board]] = []

    for match_data in matches_data:
        match_rows: list[dict] = []
        move_history = match_data["move_history"]
        match_result = match_data["result"]

        current_board = "0"
        available_pieces = set(range(16))
        pending_piece = -1
        joint_state_index = 0

        for move_idx, move in enumerate(move_history):
            action_type = move["action"]
            player_pos = move["player_pos"]
            mov_description = f"{move_idx}|{action_type}|{player_pos}"
            outcome = _actor_outcome(player_pos, match_result)

            if action_type == "selected":
                state_board = current_board
                state_aux = _available_pieces_mask(available_pieces)
                valid_mask = state_aux.copy()
                action = int(move["piece_index"])

                if action not in available_pieces:
                    raise ValueError(
                        f"Selected piece {action} not available in storage. {mov_description}"
                    )

                available_pieces.remove(action)
                pending_piece = action

                next_state_board = current_board
                next_state_aux = _piece_index_to_vector(pending_piece)
                next_phase = PHASE_PLACE
                next_valid_mask = _valid_position_mask(current_board)

                match_rows.append(
                    {
                        "joint_state_index": joint_state_index,
                        "mov_description": mov_description,
                        "board_state": state_board,
                        "board_next_state": next_state_board,
                        "state_aux": state_aux,
                        "valid_mask": valid_mask,
                        "action": action,
                        "phase": PHASE_SELECT,
                        "done": False,
                        "next_state_aux": next_state_aux,
                        "next_phase": next_phase,
                        "next_valid_mask": next_valid_mask,
                        "outcome": outcome,
                    }
                )
                joint_state_index += 1

            elif action_type == "placed":
                if pending_piece == -1:
                    raise ValueError(
                        f"Encountered a place action without a pending selected piece. {mov_description}"
                    )

                state_board = current_board
                state_aux = _piece_index_to_vector(pending_piece)
                valid_mask = _valid_position_mask(current_board)
                action = int(move["position_index"])

                next_state_board = move["board_after"]
                current_board = next_state_board
                done = move_idx == len(move_history) - 1

                if done:
                    next_state_aux = np.zeros(16, dtype=np.float32)
                    next_phase = PHASE_SELECT
                    next_valid_mask = np.zeros(16, dtype=np.float32)
                else:
                    next_state_aux = _available_pieces_mask(available_pieces)
                    next_phase = PHASE_SELECT
                    next_valid_mask = next_state_aux.copy()

                match_rows.append(
                    {
                        "joint_state_index": joint_state_index,
                        "mov_description": mov_description,
                        "board_state": state_board,
                        "board_next_state": next_state_board,
                        "state_aux": state_aux,
                        "valid_mask": valid_mask,
                        "action": action,
                        "phase": PHASE_PLACE,
                        "done": done,
                        "next_state_aux": next_state_aux,
                        "next_phase": next_phase,
                        "next_valid_mask": next_valid_mask,
                        "outcome": outcome,
                    }
                )
                pending_piece = -1
            else:
                raise ValueError(f"Unknown action {action_type}")

        if not match_rows:
            continue

        max_joint_state_index = max(row["joint_state_index"] for row in match_rows)
        if n_last_states <= max_joint_state_index + 1:
            joint_cutoff = max_joint_state_index - n_last_states + 1
            match_rows = [
                row for row in match_rows if row["joint_state_index"] >= joint_cutoff
            ]

        total_transitions = len(match_rows)
        for idx, row in enumerate(match_rows):
            steps_to_terminal = total_transitions - 1 - idx
            row["steps_to_terminal"] = steps_to_terminal
            row["reward"] = _phase_reward(
                REWARD_FUNCTION_TYPE=REWARD_FUNCTION_TYPE,
                phase=row["phase"],
                outcome=row["outcome"],
                done=row["done"],
                steps_to_terminal=steps_to_terminal,
            )

            if COLLECT_BOARDS:
                boards.append(
                    (
                        Board.serialized_2_board(
                            row["board_state"],
                            name=(
                                f"{row['mov_description']} | phase={row['phase']} | "
                                f"R={row['reward']:.2f}"
                            ),
                        ),
                        Board.serialized_2_board(
                            row["board_next_state"],
                            name=(
                                f"{row['mov_description']} | next_phase={row['next_phase']} | "
                                f"R={row['reward']:.2f}"
                            ),
                        ),
                    )
                )

        rows.extend(match_rows)

    if not rows:
        raise ValueError("No decoupled-autoregressive experience was generated.")

    p_all = pd.DataFrame(rows)

    experience = TensorDict(
        {
            "state_board": torch.tensor(
                np.stack(p_all["board_state"].apply(Board.deserialize)),
                dtype=torch.float32,
            ),
            "state_aux": torch.tensor(
                np.stack(p_all["state_aux"].to_list()),
                dtype=torch.float32,
            ),
            "phase": torch.tensor(p_all["phase"].to_numpy(), dtype=torch.int64),
            "valid_mask": torch.tensor(
                np.stack(p_all["valid_mask"].to_list()),
                dtype=torch.float32,
            ),
            "action": torch.tensor(p_all["action"].to_numpy(), dtype=torch.int64),
            "reward": torch.tensor(p_all["reward"].to_numpy(), dtype=torch.float32),
            "done": torch.tensor(p_all["done"].to_numpy(), dtype=torch.bool),
            "next_state_board": torch.tensor(
                np.stack(p_all["board_next_state"].apply(Board.deserialize)),
                dtype=torch.float32,
            ),
            "next_state_aux": torch.tensor(
                np.stack(p_all["next_state_aux"].to_list()),
                dtype=torch.float32,
            ),
            "next_phase": torch.tensor(
                p_all["next_phase"].to_numpy(), dtype=torch.int64
            ),
            "next_valid_mask": torch.tensor(
                np.stack(p_all["next_valid_mask"].to_list()),
                dtype=torch.float32,
            ),
            "outcome": torch.tensor(p_all["outcome"].to_numpy(), dtype=torch.float32),
            "steps_to_terminal": torch.tensor(
                p_all["steps_to_terminal"].to_numpy(), dtype=torch.float32
            ),
        },
        batch_size=[p_all.shape[0]],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    if COLLECT_BOARDS:
        return experience, boards
    return experience


def gen_experience_unified_autoreg(
    *,
    p1_bot: BotAI,
    p2_bot: BotAI,
    n_last_states: int = 16,
    number_of_matches: int = 1000,
    verbose: bool = False,
    PROGRESS_MESSAGE: str = "Generating experience",
    mode_2x2: bool = False,
    REWARD_FUNCTION_TYPE: str = "propagate",
    COLLECT_BOARDS: bool = False,
    select_oracle=None,
):
    """Generate phase-decoupled transitions with phase-stable 32-d aux.

    Mirrors :func:`gen_experience_decoupled_autoreg` row-for-row. The only
    difference is that ``state_aux`` and ``next_state_aux`` are 32-d
    ``[offered_one_hot ; available_mask]`` vectors that carry the same input
    semantics across both phases.

    Aux semantics from the acting player's perspective:

    - place row (about to place ``pending_piece``):
        offered = pending_piece, available = pieces still in storage
    - select row (about to give a piece to opponent):
        offered = piece-just-placed-this-turn (zero-vec for the very first
        select of the game, before any placement), available = pieces still
        in storage
    """
    logger.debug("Generating unified-autoregressive experience...")

    matches_data, _ = play_games(
        matches=number_of_matches,
        player1=p1_bot,
        player2=p2_bot,
        delay=0,
        verbose=verbose,
        PROGRESS_MESSAGE=PROGRESS_MESSAGE,
        save_match=False,
        mode_2x2=mode_2x2,
    )

    rows: list[dict] = []
    boards: list[tuple[Board, Board]] = []

    for match_data in matches_data:
        match_rows: list[dict] = []
        move_history = match_data["move_history"]
        match_result = match_data["result"]

        current_board = "0"
        available_pieces = set(range(16))
        pending_piece = -1
        last_placed_piece = -1
        joint_state_index = 0

        for move_idx, move in enumerate(move_history):
            action_type = move["action"]
            player_pos = move["player_pos"]
            mov_description = f"{move_idx}|{action_type}|{player_pos}"
            outcome = _actor_outcome(player_pos, match_result)

            if action_type == "selected":
                state_board = current_board
                # Unified aux for select: offered=last-piece-I-placed (zero on
                # the very first select of the game), available=storage pool.
                state_aux = _unified_aux(last_placed_piece, available_pieces)
                # valid_mask for a select action is the available-piece mask
                # (independent of the aux layout).
                valid_mask = _available_pieces_mask(available_pieces)
                action = int(move["piece_index"])

                if action not in available_pieces:
                    raise ValueError(
                        f"Selected piece {action} not available in storage. {mov_description}"
                    )

                # Capture oracle target BEFORE mutating available_pieces — the
                # SELECT decision is over the pre-removal set.
                if select_oracle is not None:
                    target_sel_minimax, target_sel_minimax_mask = (
                        _minimax_select_target(
                            select_oracle,
                            state_board,
                            available_pieces,
                            mode_2x2=mode_2x2,
                        )
                    )
                else:
                    target_sel_minimax = np.zeros(16, dtype=np.float32)
                    target_sel_minimax_mask = np.zeros(16, dtype=np.float32)

                available_pieces.remove(action)
                pending_piece = action

                # Next state: opponent is about to place the piece I just gave.
                # Their aux: offered=pending_piece, available=current storage.
                next_state_board = current_board
                next_state_aux = _unified_aux(pending_piece, available_pieces)
                next_phase = PHASE_PLACE
                next_valid_mask = _valid_position_mask(current_board)

                match_rows.append(
                    {
                        "joint_state_index": joint_state_index,
                        "mov_description": mov_description,
                        "board_state": state_board,
                        "board_next_state": next_state_board,
                        "state_aux": state_aux,
                        "valid_mask": valid_mask,
                        "action": action,
                        "phase": PHASE_SELECT,
                        "done": False,
                        "next_state_aux": next_state_aux,
                        "next_phase": next_phase,
                        "next_valid_mask": next_valid_mask,
                        "outcome": outcome,
                        "target_sel_minimax": target_sel_minimax,
                        "target_sel_minimax_mask": target_sel_minimax_mask,
                    }
                )
                joint_state_index += 1

            elif action_type == "placed":
                if pending_piece == -1:
                    raise ValueError(
                        f"Encountered a place action without a pending selected piece. {mov_description}"
                    )

                state_board = current_board
                # Unified aux for place: offered=pending_piece (in hand),
                # available=storage pool (already excludes pending_piece).
                state_aux = _unified_aux(pending_piece, available_pieces)
                valid_mask = _valid_position_mask(current_board)
                action = int(move["position_index"])

                next_state_board = move["board_after"]
                done = move_idx == len(move_history) - 1

                if done:
                    # Terminal: aux is zeroed by convention. Targets mask out
                    # next-state contributions on terminal rows anyway.
                    next_state_aux = np.zeros(UNIFIED_AUX_DIM, dtype=np.float32)
                    next_phase = PHASE_SELECT
                    next_valid_mask = np.zeros(16, dtype=np.float32)
                else:
                    # Next state: I'm about to select what to give the opponent.
                    # My aux: offered=the_piece_I_just_placed (=pending_piece
                    # before reset), available=current storage.
                    next_state_aux = _unified_aux(pending_piece, available_pieces)
                    next_phase = PHASE_SELECT
                    next_valid_mask = _available_pieces_mask(available_pieces)

                match_rows.append(
                    {
                        "joint_state_index": joint_state_index,
                        "mov_description": mov_description,
                        "board_state": state_board,
                        "board_next_state": next_state_board,
                        "state_aux": state_aux,
                        "valid_mask": valid_mask,
                        "action": action,
                        "phase": PHASE_PLACE,
                        "done": done,
                        "next_state_aux": next_state_aux,
                        "next_phase": next_phase,
                        "next_valid_mask": next_valid_mask,
                        "outcome": outcome,
                        "target_sel_minimax": np.zeros(16, dtype=np.float32),
                        "target_sel_minimax_mask": np.zeros(16, dtype=np.float32),
                    }
                )

                # Advance the per-actor "what did I just place" tracker before
                # clearing pending_piece. This is the only state-tracking
                # difference vs the decoupled generator.
                current_board = next_state_board
                last_placed_piece = pending_piece
                pending_piece = -1
            else:
                raise ValueError(f"Unknown action {action_type}")

        if not match_rows:
            continue

        max_joint_state_index = max(row["joint_state_index"] for row in match_rows)
        if n_last_states <= max_joint_state_index + 1:
            joint_cutoff = max_joint_state_index - n_last_states + 1
            match_rows = [
                row for row in match_rows if row["joint_state_index"] >= joint_cutoff
            ]

        total_transitions = len(match_rows)
        for idx, row in enumerate(match_rows):
            steps_to_terminal = total_transitions - 1 - idx
            row["steps_to_terminal"] = steps_to_terminal
            row["reward"] = _phase_reward(
                REWARD_FUNCTION_TYPE=REWARD_FUNCTION_TYPE,
                phase=row["phase"],
                outcome=row["outcome"],
                done=row["done"],
                steps_to_terminal=steps_to_terminal,
            )

            if COLLECT_BOARDS:
                boards.append(
                    (
                        Board.serialized_2_board(
                            row["board_state"],
                            name=(
                                f"{row['mov_description']} | phase={row['phase']} | "
                                f"R={row['reward']:.2f}"
                            ),
                        ),
                        Board.serialized_2_board(
                            row["board_next_state"],
                            name=(
                                f"{row['mov_description']} | next_phase={row['next_phase']} | "
                                f"R={row['reward']:.2f}"
                            ),
                        ),
                    )
                )

        rows.extend(match_rows)

    if not rows:
        raise ValueError("No unified-autoregressive experience was generated.")

    p_all = pd.DataFrame(rows)

    experience = TensorDict(
        {
            "state_board": torch.tensor(
                np.stack(p_all["board_state"].apply(Board.deserialize)),
                dtype=torch.float32,
            ),
            "state_aux": torch.tensor(
                np.stack(p_all["state_aux"].to_list()),
                dtype=torch.float32,
            ),
            "phase": torch.tensor(p_all["phase"].to_numpy(), dtype=torch.int64),
            "valid_mask": torch.tensor(
                np.stack(p_all["valid_mask"].to_list()),
                dtype=torch.float32,
            ),
            "action": torch.tensor(p_all["action"].to_numpy(), dtype=torch.int64),
            "reward": torch.tensor(p_all["reward"].to_numpy(), dtype=torch.float32),
            "done": torch.tensor(p_all["done"].to_numpy(), dtype=torch.bool),
            "next_state_board": torch.tensor(
                np.stack(p_all["board_next_state"].apply(Board.deserialize)),
                dtype=torch.float32,
            ),
            "next_state_aux": torch.tensor(
                np.stack(p_all["next_state_aux"].to_list()),
                dtype=torch.float32,
            ),
            "next_phase": torch.tensor(
                p_all["next_phase"].to_numpy(), dtype=torch.int64
            ),
            "next_valid_mask": torch.tensor(
                np.stack(p_all["next_valid_mask"].to_list()),
                dtype=torch.float32,
            ),
            "outcome": torch.tensor(p_all["outcome"].to_numpy(), dtype=torch.float32),
            "steps_to_terminal": torch.tensor(
                p_all["steps_to_terminal"].to_numpy(), dtype=torch.float32
            ),
            "target_sel_minimax": torch.tensor(
                np.stack(p_all["target_sel_minimax"].to_list()),
                dtype=torch.float32,
            ),
            "target_sel_minimax_mask": torch.tensor(
                np.stack(p_all["target_sel_minimax_mask"].to_list()),
                dtype=torch.float32,
            ),
        },
        batch_size=[p_all.shape[0]],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    if COLLECT_BOARDS:
        return experience, boards
    return experience


def DQN_training_step(
    policy_net: NN_abstract,
    target_net: NN_abstract,
    GAMMA: float,
    exp_batch: TensorDict,
    LOSS_APPROACH: str = "combined_avg",
    TRANSITION_SCHEMA: str = TRANSITION_SCHEMA_JOINT,
    DECOUPLED_TARGET_STYLE: str = DECOUPLED_TARGET_TD_PLACE_MC_SELECT,
):
    """Perform one DQN training step using the given batch of experiences.

    Parameters
    ----------
    policy_net : NN_abstract
        The policy network being trained
    target_net : NN_abstract
        The target network for computing stable Q-value targets
    GAMMA : float
        Discount factor for future rewards
    exp_batch : TensorDict
        Batch of experiences with state, action, reward, next_state, done
    LOSS_APPROACH : str
        Approach for computing state-action values. Options are "combined_avg", "only_select", "only_place", "separate_bellman", "mc_select".
    TRANSITION_SCHEMA : str
        Transition layout. Options are "joint", "decoupled_autoreg", and
        "unified_autoreg" (decoupled targets with phase-stable 32-d aux).
    DECOUPLED_TARGET_STYLE : str
        Target rule for decoupled-autoregressive AND unified-autoregressive
        training (both share the same per-phase masked target machinery).
        Default is place TD + select Monte Carlo.
    Returns
    -------
    For "combined_avg", "only_select", "only_place":
        state_action_values : torch.Tensor
            Q-values for the actions taken in the batch
        expected_state_action_values : torch.Tensor
            Target Q-values computed using target network and Bellman equation
    For "separate_bellman", "mc_select":
        state_place_values : torch.Tensor
            Q_place values for placement actions taken
        expected_place : torch.Tensor
            Target for placement head (Bellman in both cases)
        state_select_values : torch.Tensor
            Q_select values for selection actions taken
        expected_select : torch.Tensor
            Target for selection head. For "separate_bellman" this is the Bellman
            target; for "mc_select" it is the Monte Carlo return
            gamma^steps_to_terminal * outcome, with no bootstrap through Q_select.
    """
    # Ensure networks are in correct mode
    policy_net.train()
    # target_net.eval()  # Target network is always in eval mode

    if TRANSITION_SCHEMA in (
        TRANSITION_SCHEMA_DECOUPLED_AUTOREG,
        TRANSITION_SCHEMA_UNIFIED_AUTOREG,
    ):
        # Unified-autoreg reuses the same per-phase masked target rule as the
        # decoupled schema; the only difference is state_aux/next_state_aux
        # dimensionality (32 vs 16), which is consumed inside the model's
        # ``q_values_phase``. No code change is required at the target layer.
        return DQN_training_step_decoupled_autoreg(
            policy_net=policy_net,
            target_net=target_net,
            GAMMA=GAMMA,
            exp_batch=exp_batch,
            TARGET_STYLE=DECOUPLED_TARGET_STYLE,
        )
    if TRANSITION_SCHEMA != TRANSITION_SCHEMA_JOINT:
        raise ValueError(f"Unknown TRANSITION_SCHEMA {TRANSITION_SCHEMA}")

    # Move experience batch to the same device as the model
    device = next(policy_net.parameters()).device
    exp_batch = exp_batch.to(device)

    pred_board_place, pred_piece = policy_net(
        exp_batch["state_board"], exp_batch["state_piece"]
    )

    # --- HANDLE SPECIAL CASES
    # First move (turn 0): action_place=-1 (no placement on empty board)
    # Terminal states: action_sel=-1 (no piece selection after winning)
    # Both first_move and terminal cannot occur simultaneously

    # Extract action indices
    action_pos = exp_batch["action_place"]
    action_sel = exp_batch["action_sel"]

    # Initialize Q-value tensors (will be 0 for invalid actions)
    state_place_action_values = torch.zeros_like(action_pos, dtype=torch.float32)
    state_sel_action_values = torch.zeros_like(action_sel, dtype=torch.float32)

    # --- Create masks for different experience types
    first_move_mask = action_pos == -1  # First move has no placement (empty board)
    final_move_mask = (
        action_sel == -1
    )  # Terminal states have no piece selection (game ended)
    non_terminal_mask = ~final_move_mask
    terminal_mask = exp_batch["done"]

    # Sanity checks
    assert (
        ~(first_move_mask & final_move_mask)
    ).all(), "Invalid experience with both first and final move."

    assert terminal_mask[
        final_move_mask
    ].all(), "All experiences with action_sel=-1 must be terminal states."

    # Extract valid action indices (excluding -1 values)
    action_pos_valid = action_pos[~first_move_mask]
    action_sel_valid = action_sel[~final_move_mask]

    # Gather Q-values for valid placement actions
    state_place_action_values[~first_move_mask] = pred_board_place.gather(
        1, action_pos_valid.unsqueeze(1).type(torch.int64)
    ).squeeze(1)

    # Gather Q-values for valid selection actions
    state_sel_action_values[non_terminal_mask] = pred_piece.gather(
        1, action_sel_valid.unsqueeze(1).type(torch.int64)
    ).squeeze(1)

    # Combine Q-values as a joint action
    # Each turn consists of TWO decisions: place AND select
    # The value of the state-action should reflect BOTH decisions
    #
    # For special cases:
    # - First moves (place=-1): placement gets Q=0, so only selection matters
    # - Terminal states (select=-1): selection gets Q=0, so only placement matters
    #
    if LOSS_APPROACH == "combined_avg":
        # Use AVERAGE to represent the joint action value
        # the expected return depends on BOTH actions
        # Takes the (place + select) / 2 when both are valid
        # For first moves only selection matters
        # For terminal states only placement matters
        # _factor divides by 2 when both actions are valid, otherwise by 1
        # Scale by the number of valid action components in this step:
        # - placement is valid when not first_move
        # - selection is valid when not final_move (non-terminal)
        place_valid = (~first_move_mask).type_as(state_place_action_values)
        select_valid = (non_terminal_mask).type_as(state_place_action_values)
        # 1 if one is valid, 2 if both are valid
        valid_count = place_valid + select_valid

        _factor = 1.0 / valid_count

        state_action_values = (
            state_place_action_values + state_sel_action_values
        ) * _factor

    elif LOSS_APPROACH == "only_select":
        # Use ONLY the selection action value for training
        state_action_values = state_sel_action_values
    elif LOSS_APPROACH == "only_place":
        state_action_values = state_place_action_values
    elif LOSS_APPROACH in ("separate_bellman", "mc_select"):
        # Handled below — returns per-head values instead of a combined scalar
        pass
    else:
        raise ValueError(f"Unknown LOSS_APPROACH {LOSS_APPROACH}")

    # Compute V(s_{t+1}) for all next states using target network
    # Initialize with zeros (terminal states have V=0 by definition)
    with torch.no_grad():
        _next_state_pos, _next_state_piece = target_net(
            exp_batch["next_state_board"][non_terminal_mask],
            exp_batch["next_state_piece"][non_terminal_mask],
        )

    if LOSS_APPROACH == "separate_bellman":
        # Separate Bellman targets per head: each head gets its own gradient
        # independently, preventing the lazy head problem.
        #   loss = (loss_place + loss_select) / 2
        # where:
        #   loss_place = SmoothL1(Q_place[a], R + γ * max_a' Q_place(s'))
        #   loss_select = SmoothL1(Q_select[a], R + γ * max_a' Q_select(s'))
        next_place_values = torch.zeros(
            exp_batch.shape, device=exp_batch["reward"].device
        )
        next_select_values = torch.zeros(
            exp_batch.shape, device=exp_batch["reward"].device
        )

        with torch.no_grad():
            next_place_values[non_terminal_mask] = _next_state_pos.max(dim=1).values
            next_select_values[non_terminal_mask] = _next_state_piece.max(dim=1).values

        reward = exp_batch["reward"]
        expected_place = reward + (next_place_values * GAMMA)
        expected_select = reward + (next_select_values * GAMMA)

        # Mask invalid actions: set target = prediction so loss contribution is zero.
        # First moves have no placement (action_place=-1) → Q_place_pred is 0 → set target to 0.
        # Terminal states have no selection (action_sel=-1) → Q_select_pred is 0 → set target to 0.
        # Without this, terminal states produce SmoothL1(0, ±1) every epoch,
        # pushing Q_select into tanh saturation.
        expected_place[first_move_mask] = 0.0
        expected_select[final_move_mask] = 0.0

        return (
            state_place_action_values,
            expected_place,
            state_sel_action_values,
            expected_select,
        )

    if LOSS_APPROACH == "mc_select":
        # Q_place uses the standard Bellman target; Q_select is supervised with the
        # Monte Carlo return computed from the actual game outcome, avoiding a
        # noisy self-bootstrap.
        #   loss_place  = SmoothL1(Q_place[a],  R + γ * max_a' Q_place(s'))
        #   loss_select = SmoothL1(Q_select[a], γ^steps_to_terminal * outcome)
        next_place_values = torch.zeros(
            exp_batch.shape, device=exp_batch["reward"].device
        )
        with torch.no_grad():
            next_place_values[non_terminal_mask] = _next_state_pos.max(dim=1).values

        reward = exp_batch["reward"]
        expected_place = reward + (next_place_values * GAMMA)

        outcome = exp_batch["outcome"]
        steps = exp_batch["steps_to_terminal"]
        expected_select = (GAMMA**steps) * outcome

        # Same masking as separate_bellman: zero-out invalid-action losses.
        expected_place[first_move_mask] = 0.0
        expected_select[final_move_mask] = 0.0

        return (
            state_place_action_values,
            expected_place,
            state_sel_action_values,
            expected_select,
        )

    # --- Shared path for combined_avg, only_select, only_place ---
    next_state_values = torch.zeros(exp_batch.shape, device=exp_batch["reward"].device)

    with torch.no_grad():
        if LOSS_APPROACH == "combined_avg":
            # Place and select are INDEPENDENT action spaces (position vs piece).
            # Take max over each independently, then average.
            _next_val = (
                _next_state_pos.max(dim=1).values + _next_state_piece.max(dim=1).values
            ) / 2
        elif LOSS_APPROACH == "only_select":
            _next_val = _next_state_piece.max(dim=1).values
        elif LOSS_APPROACH == "only_place":
            _next_val = _next_state_pos.max(dim=1).values

        next_state_values[non_terminal_mask] = _next_val

    # Bellman equation: Q(s,a) = R + γ * max_a' Q(s', a')
    # NOTE: The adversarial sign is already encoded in the "propagate" rewards
    # (P1 gets +R, P2 gets -R). No additional sign flip needed here.
    expected_state_action_values = exp_batch["reward"] + (next_state_values * GAMMA)

    return state_action_values, expected_state_action_values


def DQN_training_step_decoupled_autoreg(
    *,
    policy_net: NN_abstract,
    target_net: NN_abstract,
    GAMMA: float,
    exp_batch: TensorDict,
    TARGET_STYLE: str = DECOUPLED_TARGET_TD_PLACE_MC_SELECT,
):
    """Compute masked per-phase targets for decoupled-autoregressive batches.

    Target styles:
      - "td_place_mc_select": place uses TD/Bellman bootstrap into the next
        select phase; select uses Monte Carlo outcome supervision.
      - "mc_both": both heads use Monte Carlo outcome supervision
        (gamma^steps_to_terminal * outcome). target_net is not read.
      - "td_place_td_select": place uses TD/Bellman; select uses a 1-step
        Q_place_target bootstrap — target_select = γ * max_a Q_place_target(s').
        No Q_select self-bootstrap, no MC variance, no outcome-label imbalance.
        (Ng_auxSelect diagnostic.)
    """
    if TARGET_STYLE not in (
        DECOUPLED_TARGET_TD_PLACE_MC_SELECT,
        DECOUPLED_TARGET_MC_BOTH,
        DECOUPLED_TARGET_TD_PLACE_TD_SELECT,
        DECOUPLED_TARGET_TD_PLACE_MINIMAX_SELECT,
        DECOUPLED_TARGET_TD_PLACE_MINIMAX_SELECT_SCALAR,
    ):
        raise ValueError(f"Unknown decoupled target style {TARGET_STYLE}")

    q_values_phase = getattr(policy_net, "q_values_phase", None)
    target_q_values_phase = getattr(target_net, "q_values_phase", None)
    if q_values_phase is None or not callable(q_values_phase):
        raise TypeError(
            "Decoupled-autoreg training requires policy_net.q_values_phase(x_board, x_aux, phase)."
        )
    if target_q_values_phase is None or not callable(target_q_values_phase):
        raise TypeError(
            "Decoupled-autoreg training requires target_net.q_values_phase(x_board, x_aux, phase)."
        )

    device = next(policy_net.parameters()).device
    exp_batch = exp_batch.to(device)

    phase = exp_batch["phase"].to(torch.int64)
    action = exp_batch["action"].to(torch.int64)
    valid_mask = exp_batch["valid_mask"]
    place_mask = phase == PHASE_PLACE
    select_mask = phase == PHASE_SELECT

    action_validity = valid_mask.gather(1, action.unsqueeze(1)).squeeze(1)
    if not (action_validity > 0).all():
        raise ValueError(
            "Encountered a decoupled transition with an invalid chosen action."
        )

    q_place_all, q_select_all = policy_net(
        exp_batch["state_board"],
        exp_batch["state_aux"],
        phase=phase,
    )

    state_place_values = (
        q_place_all[place_mask].gather(1, action[place_mask].unsqueeze(1)).squeeze(1)
    )
    state_select_values = (
        q_select_all[select_mask].gather(1, action[select_mask].unsqueeze(1)).squeeze(1)
    )

    if TARGET_STYLE == DECOUPLED_TARGET_MC_BOTH:
        expected_place = (
            GAMMA ** exp_batch["steps_to_terminal"][place_mask]
        ) * exp_batch["outcome"][place_mask]
    else:
        expected_place = exp_batch["reward"][place_mask].clone()
        if place_mask.any():
            non_terminal_place_mask = place_mask & ~exp_batch["done"]
            if non_terminal_place_mask.any():
                with torch.no_grad():
                    next_q_values = target_q_values_phase(
                        exp_batch["next_state_board"][non_terminal_place_mask],
                        exp_batch["next_state_aux"][non_terminal_place_mask],
                        exp_batch["next_phase"][non_terminal_place_mask],
                    )
                    next_valid = exp_batch["next_valid_mask"][non_terminal_place_mask]
                    next_place_values = _masked_max(next_q_values, next_valid)
                expected_place[~exp_batch["done"][place_mask]] += (
                    GAMMA * next_place_values
                )

    expected_select = (
        GAMMA ** exp_batch["steps_to_terminal"][select_mask]
    ) * exp_batch["outcome"][select_mask]

    if TARGET_STYLE == DECOUPLED_TARGET_TD_PLACE_MINIMAX_SELECT_SCALAR:
        # Tc diagnostic: keep the scalar 4-tuple shape (Q_select at chosen
        # piece vs minimax target at chosen piece). Isolates "oracle replaces
        # MC noise" from "oracle gives 16× more signal per state".
        expected_select = (
            exp_batch["target_sel_minimax"][select_mask]
            .gather(1, action[select_mask].unsqueeze(1))
            .squeeze(1)
        )
        return (
            state_place_values,
            expected_place,
            state_select_values,
            expected_select,
        )

    if TARGET_STYLE == DECOUPLED_TARGET_TD_PLACE_MINIMAX_SELECT:
        # T-series: full per-piece minimax-oracle supervision for Q_select.
        # Override the scalar (state_select_values, expected_select) pair with
        # the FULL 16-d Q_select vector + per-piece minimax target + legality
        # mask. Caller distinguishes scalar/vector mode by len(return_tuple).
        state_select_values = q_select_all[select_mask]
        expected_select = exp_batch["target_sel_minimax"][select_mask]
        select_loss_mask = exp_batch["target_sel_minimax_mask"][select_mask]
        return (
            state_place_values,
            expected_place,
            state_select_values,
            expected_select,
            select_loss_mask,
        )

    if TARGET_STYLE == DECOUPLED_TARGET_TD_PLACE_TD_SELECT:
        # Ng_auxSelect: replace MC target for Q_select with a 1-step Q_place
        # bootstrap.  For each select transition, the next transition is a place
        # transition for the *same* player (in decoupled-autoreg the select
        # immediately precedes the opponent's place, so next_phase==PLACE_PLACE).
        # target_select = γ * max_a Q_place_target(next_state)
        # This uses the already-well-trained place head as a supervised oracle —
        # no Q_select self-bootstrap, no MC variance, no outcome-label imbalance.
        if select_mask.any():
            non_terminal_select_mask = select_mask & ~exp_batch["done"]
            if non_terminal_select_mask.any():
                with torch.no_grad():
                    # Use Q_place (not q_values_phase) so we always read the
                    # place head regardless of next_phase label.
                    next_q_place, _ = target_net(
                        exp_batch["next_state_board"][non_terminal_select_mask],
                        exp_batch["next_state_aux"][non_terminal_select_mask],
                        phase=exp_batch["next_phase"][non_terminal_select_mask],
                    )
                    next_valid = exp_batch["next_valid_mask"][non_terminal_select_mask]
                    next_place_max = _masked_max(next_q_place, next_valid)
                # Rebuild expected_select tensor for the full select slice
                expected_select_full = torch.zeros(
                    select_mask.sum(), device=exp_batch["reward"].device
                )
                # Index into non-terminal rows within the select slice
                non_terminal_within_select = (~exp_batch["done"])[select_mask]
                expected_select_full[non_terminal_within_select] = (
                    GAMMA * next_place_max
                )
                # terminal select rows stay 0 (done → no next state)
                expected_select = expected_select_full
            else:
                expected_select = torch.zeros(
                    select_mask.sum(), device=exp_batch["reward"].device
                )

    return (
        state_place_values,
        expected_place,
        state_select_values,
        expected_select,
    )
