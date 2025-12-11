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
    COLLECT_BOARDS: bool = False,
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

    for match_data in matches_data:
        exp = convert_2_state_action_reward(
            match_data, REWARD_FUNCTION_TYPE=REWARD_FUNCTION_TYPE
        )

        if n_last_states < exp.shape[0]:
            exp = exp.iloc[-n_last_states:]
        elif n_last_states >= exp.shape[0]:
            logger.warning(
                f"n_last_states ({n_last_states}) is greater than the number of states in the match ({exp.shape[0]}). Using all states."
            )
        exp_all.append(exp)

        if COLLECT_BOARDS:
            # Collect final boards
            for _, b in exp.iterrows():
                boards.append(
                    (
                        Board.serialized_2_board(
                            b["board_state"],
                            name=f"{b['mov_description']} | R={b['reward']}",
                        ),
                        Board.serialized_2_board(
                            b["board_next_state"],
                            name=f"{b['mov_description']} | R={b['reward']}",
                        ),
                    )
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
        },
        batch_size=[p_all.shape[0]],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    if COLLECT_BOARDS:
        return experience, boards
    else:
        return experience


def DQN_training_step(
    policy_net: NN_abstract,
    target_net: NN_abstract,
    GAMMA: float,
    exp_batch: TensorDict,
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

    Returns
    -------
    state_action_values : torch.Tensor
        Q-values for the actions taken in the batch
    expected_state_action_values : torch.Tensor
        Target Q-values computed using target network and Bellman equation
    """
    # Ensure networks are in correct mode
    policy_net.train()
    # target_net.eval()  # Target network is always in eval mode

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
    # Use AVERAGE to represent the joint action value
    # This is mathematically sound: the expected return depends on BOTH actions
    state_action_values = (state_place_action_values + state_sel_action_values) / 2

    # Compute V(s_{t+1}) for all next states using target network
    # Initialize with zeros (terminal states have V=0 by definition)
    next_state_values = torch.zeros(exp_batch.shape, device=exp_batch["reward"].device)

    # Compute V(s') = max_a Q(s', a) for non-terminal next states
    with torch.no_grad():
        _next_state_pos, _next_state_piece = target_net(
            exp_batch["next_state_board"][non_terminal_mask],
            exp_batch["next_state_piece"][non_terminal_mask],
        )
        # Combine using average (joint action value)
        _next_val = (_next_state_pos + _next_state_piece) / 2
        # Take maximum Q-value across all possible actions
        next_state_values[non_terminal_mask] = _next_val.max(dim=1).values

    # Compute target Q-values using Bellman equation: Q(s,a) = R + γ*max_a'Q(s',a')
    expected_state_action_values = (next_state_values * GAMMA) + exp_batch["reward"]

    return state_action_values, expected_state_action_values
