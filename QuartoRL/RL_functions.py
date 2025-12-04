# -*- coding: utf-8 -*-

"""
Python 3
01 / 06 / 2025
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""

from utils.logger import logger

from tensordict import set_list_to_stack, TensorDict

from quartopy import play_games, BotAI, Board

import torch
from torch.nn import SmoothL1Loss
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
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
    # Rewards are from perspective of player 1, and are inverted for the turns of the second player
    # NOTE: ``board_next_state`` includes the final board after placing the piece and thus have the same size as ``board_state`` in winning or drawing.
    # NOTE: ``piece_next_state`` not exist when winning, because there is no selected piece for next turn.

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
            reward = [0] * (_num_states)
            reward.append(R)
        case "propagate":
            reward = [R if i % 2 == 0 else R_2 for i in range(_num_states)]
        case "discount":
            gamma = 0.8
            reward = [
                R * (gamma**i) * (-1) ** (i % 2 == 1)
                for i in reversed(range(_num_states))
            ]
        case _:
            raise ValueError(f"Unknown REWARD_FUNCTION_TYPE {REWARD_FUNCTION_TYPE}")

    # ---- DONE
    done = [False] * _num_non_terminal_states
    done.append(True)  # onnly last state is the ending state

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
) -> TensorDict | tuple[TensorDict, list[Board]]:
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
    ```
    TensorDict(
        fields={
            action_place: Tensor(shape=torch.Size([400]),
            action_sel: Tensor(shape=torch.Size([400]),
            done: Tensor(shape=torch.Size([400]),
            next_state_board: Tensor(shape=torch.Size([400, 16, 4, 4]),
            next_state_piece: Tensor(shape=torch.Size([400, 16]),
            reward: Tensor(shape=torch.Size([400]),
            state_board: Tensor(shape=torch.Size([400, 16, 4, 4]),
            state_piece: Tensor(shape=torch.Size([400, 16])},
        batch_size=torch.Size([400]),
        device=cpu,
        is_shared=False)
    list[Board Board next state] [OPTIONAL] if COLLECT_BOARDS is True

    ```
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
        # elif n_last_states > exp.shape[0]:
        #     logger.warning(
        #         f"n_last_states ({n_last_states}) is greater than the number of states in the match ({exp.shape[0]}). Using all states."
        #     )
        exp_all.append(exp)

        if COLLECT_BOARDS:
            # Collect final boards
            for _, b in exp.iterrows():
                boards.append(
                    (
                        Board.serialized_2_board(
                            b["board_state"],
                            name=f"{b['mov_description']}R={b['reward']}",
                        ),
                        Board.serialized_2_board(b["board_next_state"], name="sad"),
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
    loss_fcn=SmoothL1Loss,
):
    pred_board_place, pred_piece = policy_net(
        exp_batch["state_board"], exp_batch["state_piece"]
    )
    # --- FILTER ACTIONS
    # filter -1 actions, because they are not valid actions.
    # Which one to choose? 0? but if the action is 0, then it is a valid action...
    action_sel = torch.where(
        exp_batch["action_sel"] == -1,
        # torch.zeros_like(exp_batch["action_sel"]),
        0,
        exp_batch["action_sel"],
    )
    action_pos = torch.where(
        exp_batch["action_place"] == -1,
        # torch.zeros_like(exp_batch["action_place"]),
        0,
        exp_batch["action_place"],
    )

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the  columns of actions taken. These are the actions which would've been taken for each batch state according to policy_net

    # se necesita unsqueeze(0) para que gather funcione correctamente (tener misma cantidad de dimensiones)
    # pred_piece is [batch_size, 16]
    # action_sel is [batch_size]
    # after gather, state_sel_action_values is [1, batch_size] so we need to squeeze it to [batch_size]
    state_sel_action_values = pred_piece.gather(
        1, action_sel.unsqueeze(0).type(torch.int64)
    ).squeeze()
    state_place_action_values = pred_board_place.gather(
        1, action_pos.unsqueeze(0).type(torch.int64)
    ).squeeze()
    state_action_values = state_sel_action_values + state_place_action_values

    # ------- Compute V(s_{t+1}) for all next states.
    # Prealloc with 0 because final states have 0 value
    next_state_values = torch.zeros_like(exp_batch["reward"])
    non_final_mask = torch.where(~exp_batch["done"])
    with torch.no_grad():
        _next_state_pos, _next_state_piece = target_net(
            exp_batch["next_state_board"][non_final_mask],
            exp_batch["next_state_piece"][non_final_mask],
        )
        _next_val = _next_state_pos + _next_state_piece
        next_state_values[non_final_mask] = _next_val.max(dim=1).values

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + exp_batch["reward"]

    # loss = loss_fcn(state_action_values, expected_state_action_values.unsqueeze(1))
    return state_action_values, expected_state_action_values
