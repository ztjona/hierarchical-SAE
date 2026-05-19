from utils.logger import logger

logger.info("Starting Importing...")

import torch
import torch.nn as nn
import torch.optim as optim
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from bot.CNN_bot import Quarto_bot
from bot.CNN_autoreg_bot import Quarto_bot as Quarto_autoreg_bot
from bot.CNN_unified_bot import Quarto_bot as Quarto_unified_bot
from models.CNN1 import QuartoCNN
from models.CNN_uncoupled import QuartoCNN as QuartoCNN_uncoupled
from models.CNN_unbound import QuartoCNN as QuartoCNN_unbound
from models.CNN_autoreg import (
    QuartoCNNAutoreg,
    QuartoCNNAutoregUnbound,
    QuartoCNNAutoregUnified,
    QuartoCNNAutoregUnifiedUnbound,
)
from models.CNN_autoreg_sa import (
    QuartoCNNAutoregUnifiedS1,
    QuartoCNNAutoregUnifiedS2,
    QuartoCNNAutoregUnifiedS4,
)
from QuartoRL import (
    gen_experience,
    run_contest,
    contest_2_win_rate,
    DQN_training_step,
    plot_win_rate,
    plot_loss,
    plot_grad_norm,
    plot_boards_comp,
    plot_Qv_progress,
    plot_Qv_horizon,
    SUMMARY_SUFFIX,
    append_record,
    build_checkpoint_record,
    build_final_record,
)
from tqdm.auto import tqdm
from pprint import pformat
import pickle
from colorama import init, Fore, Style
import socket
from os import path
import matplotlib.pyplot as plt

# ---- PARAMS ----
logger.info("Imports done.")

STARTING_NET = None  # Set to None to start with random weights
EXPERIMENT_NAME = "Ta_minimaxSelect(2)0515_DEPTH_1"
CHECKPOINT_FOLDER = f"./CHECKPOINTS/{EXPERIMENT_NAME}/"
SERIES_ROOT = EXPERIMENT_NAME.split("(", 1)[0] or EXPERIMENT_NAME
RESULTS_FOLDER = f"./results/{SERIES_ROOT}/"
TRANSITION_SCHEMA = "unified_autoreg"
ARCHITECTURE = QuartoCNNAutoregUnifiedS4
PLAYER_BOT_CLASS = Quarto_unified_bot

DECOUPLED_TARGET_STYLE = "td_place_mc_select"

# ---- T-series flag: minimax-oracle distillation on the SELECT head ----
# Ta(2) variant: depth=1 (cheap teacher floor). If WR / Q_select Δ match
# Ta(1) at depth=2, the cache (Td) gets much more attractive and we learn
# the signal lives only in immediate-force positions.
USE_MINIMAX_SELECT_TARGET = True
MINIMAX_SELECT_DEPTH = 1
MINIMAX_SELECT_SCALAR_TARGET = False  # full 16-d vector (matches Ta(1))

if USE_MINIMAX_SELECT_TARGET:
    if MINIMAX_SELECT_SCALAR_TARGET:
        DECOUPLED_TARGET_STYLE = "td_place_minimax_select_scalar"
    else:
        DECOUPLED_TARGET_STYLE = "td_place_minimax_select"
    from bot.minimax_bot import MinimaxBot

    SELECT_ORACLE = MinimaxBot(depth=MINIMAX_SELECT_DEPTH)
    logger.info(
        f"T-series: minimax-oracle select target ENABLED "
        f"(depth={MINIMAX_SELECT_DEPTH}, scalar={MINIMAX_SELECT_SCALAR_TARGET})"
    )
else:
    SELECT_ORACLE = None


def estimate_steps_per_match(n_last_states: int, transition_schema: str) -> int:
    if transition_schema in ("decoupled_autoreg", "unified_autoreg"):
        return max(1, 2 * n_last_states - 1)
    return n_last_states


LOSS_APPROACH = "mc_select"
REWARD_FUNCTION = "final"

GEN_EXPERIENCE_BY_EPOCH = True

N_MATCHES_EVAL = 30

BATCH_SIZE = 32
mode_2x2 = True

EPOCHS = 5_000

N_LAST_STATES_INIT = 4
N_LAST_STATES_FINAL = N_LAST_STATES_INIT

MATCHES_PER_EPOCH = 32
NUM_EPOCHs_BUFFER = 8

ENDGAME_FRACTION = 0.5
N_LAST_STATES_ENDGAME = 2

STEPS_PER_EPOCH = (
    estimate_steps_per_match(N_LAST_STATES_FINAL, TRANSITION_SCHEMA) * MATCHES_PER_EPOCH
)

if GEN_EXPERIENCE_BY_EPOCH:
    REPLAY_SIZE = NUM_EPOCHs_BUFFER * STEPS_PER_EPOCH
else:
    REPLAY_SIZE = STEPS_PER_EPOCH

TARGET_UPDATE_FREQ = 3

TEMPERATURE_EXPLORE = 2
TEMPERATURE_EXPLOIT = 0.1

FREQ_EPOCH_SAVING = 1000
CHECKPOINT_FREQ = 50

FREQ_EPOCH_PLOT_SHOW = 100_000

SMOOTHING_WINDOW = 10

Q_PLOT_TYPE = "hist"

# ###########################
MAX_GRAD_NORM = 1.0
LR = 7e-4
LR_F = LR
TAU = 0.01
GAMMA = 0.99

LOSS_ALPHA_PLACE = 1.0
LOSS_ALPHA_SELECT = 1.0

# ###########################
BASELINES = [
    {
        "path": "CHECKPOINTS//LOSS_APPROACHs_1212-2_only_select//20251212_2206-LOSS_APPROACHs_1212-2_only_select_E_1034.pt",
        "name": "bot_loss-BT",
        "bot": Quarto_bot,
        "params": {
            "deterministic": False,
            "temperature": 0.1,
            "model_class": QuartoCNN_uncoupled,
        },
    },
    {
        "path": "CHECKPOINTS//EXP_id03//20250922_1247-EXP_id03_epoch_0000.pt",
        "name": "bot_random",
        "bot": Quarto_bot,
        "params": {
            "deterministic": False,
            "temperature": 0.1,
            "model_class": QuartoCNN,
        },
    },
]

# ###########################
logger.info(f"PC name: {socket.gethostname()}")
logger.info(f"Experiment name:\t{EXPERIMENT_NAME}")
logger.info(
    f"Train conf.:\t{EPOCHS=}, {BATCH_SIZE=}, {LR=}, {LR_F=}, {GAMMA=}, {TAU=}, {MAX_GRAD_NORM=}"
)
logger.info(f"Exp. gen.:\t{MATCHES_PER_EPOCH=}, {STEPS_PER_EPOCH=}, {REPLAY_SIZE=}")
logger.info(f"Network updates:\tFull buffer sweep each epoch, {TARGET_UPDATE_FREQ=}")
logger.info(f"Exploration:\t{TEMPERATURE_EXPLORE=}, {TEMPERATURE_EXPLOIT=}")
logger.info(f"N_LAST_STATES:\tINIT={N_LAST_STATES_INIT}, FINAL={N_LAST_STATES_FINAL}")
logger.info(f"Checkpointing:\t{CHECKPOINT_FREQ=}, {FREQ_EPOCH_SAVING=}")
logger.info(
    f"ENDGAME_FRACTION={ENDGAME_FRACTION}, N_LAST_STATES_ENDGAME={N_LAST_STATES_ENDGAME}"
)
logger.info(f"LOSS_APPROACH={LOSS_APPROACH}")
logger.info(
    f"LOSS_ALPHA_PLACE={LOSS_ALPHA_PLACE}, LOSS_ALPHA_SELECT={LOSS_ALPHA_SELECT}"
)
logger.info(f"REWARD_FUNCTION={REWARD_FUNCTION}")
logger.info(f"TRANSITION_SCHEMA={TRANSITION_SCHEMA}")
logger.info(f"DECOUPLED_TARGET_STYLE={DECOUPLED_TARGET_STYLE}")
logger.info(
    f"USE_MINIMAX_SELECT_TARGET={USE_MINIMAX_SELECT_TARGET}, "
    f"MINIMAX_SELECT_DEPTH={MINIMAX_SELECT_DEPTH}, "
    f"MINIMAX_SELECT_SCALAR_TARGET={MINIMAX_SELECT_SCALAR_TARGET}"
)

# ###########################
RIVALS_IN_TOURNAMENT = -1
RIVALS_NAMEs = [b["name"] for b in BASELINES]
RIVALS_PATHs = [b["path"] for b in BASELINES]
RIVALS_CLASS = [b["bot"] for b in BASELINES]
RIVALS_PARAMs = [b["params"] for b in BASELINES]

win_rate: dict[str | int, list[float]] = {}
q_values_history: dict[str, list] = {
    "q_place": [],
    "q_select": [],
    "rewards": [],
    "outcome": [],
    "steps_to_terminal": [],
}

# ###########################
torch.manual_seed(5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

policy_net = ARCHITECTURE()
target_net = ARCHITECTURE()
logger.info(f"Architecture: {policy_net.name}")

policy_net.to(device)
target_net.to(device)
logger.info(f"Models moved to {device}")

if STARTING_NET is not None:
    logger.info(f"Loading starting checkpoint from: {STARTING_NET}")
    try:
        policy_net.load_state_dict(
            torch.load(STARTING_NET, map_location=device, weights_only=True)
        )
        logger.info("Successfully loaded starting checkpoint")
    except FileNotFoundError:
        logger.error(f"Checkpoint file not found: {STARTING_NET}")
        raise
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        raise
else:
    logger.info("Starting with random weights (no checkpoint provided)")

target_net.load_state_dict(policy_net.state_dict())

CKPT_NAME_GEN = lambda epoch: f"{EXPERIMENT_NAME}_E_{epoch:04d}"
policy_net.export_model(CKPT_NAME_GEN(0), CHECKPOINT_FOLDER)

# ###########################
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=REPLAY_SIZE),
    sampler=SamplerWithoutReplacement(),
)

ENDGAME_REPLAY_SIZE = (
    NUM_EPOCHs_BUFFER
    * estimate_steps_per_match(N_LAST_STATES_ENDGAME, TRANSITION_SCHEMA)
    * MATCHES_PER_EPOCH
)
endgame_replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=ENDGAME_REPLAY_SIZE),
    sampler=SamplerWithoutReplacement(),
)

# ###########################
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=LR_F)

loss_fcn = nn.SmoothL1Loss()

epochs_results = []
loss_data: dict[str, list[float | int]] = {
    "loss_values": [],
    "loss_place_values": [],
    "loss_select_values": [],
    "epoch_values": [],
}
grad_norm_data: dict[str, list[float | int]] = {
    "grad_norm_values": [],
    "grad_norm_fc2_place": [],
    "grad_norm_fc2_select": [],
    "epoch_values": [],
}
# ###########################
init(autoreset=True)

logger.info("Hyperparameters loaded.")
logger.info("Starting training...")

# -------------------------- TRAINING LOOP ---------------------------
step_i = -1
for e in tqdm(
    range(EPOCHS), desc=f"{Fore.GREEN}Epochs{Style.RESET_ALL}", position=0, leave=True
):
    p1 = PLAYER_BOT_CLASS(
        model=policy_net, deterministic=False, temperature=TEMPERATURE_EXPLORE
    )
    p2 = PLAYER_BOT_CLASS(
        model=policy_net, deterministic=False, temperature=TEMPERATURE_EXPLORE
    )

    logger.debug(f"Using temperatures: p1={p1.TEMPERATURE}, p2={p2.TEMPERATURE}")

    n_last_states = round(
        N_LAST_STATES_INIT
        + (N_LAST_STATES_FINAL - N_LAST_STATES_INIT) * (e / (EPOCHS - 1))
    )
    logger.info(f"Using n_last_states={n_last_states} for epoch {e + 1}")

    if GEN_EXPERIENCE_BY_EPOCH or e == 0:
        logger.info("Generating experience for epoch %d", e + 1)

        exp, boards = gen_experience(
            p1_bot=p1,
            p2_bot=p2,
            n_last_states=n_last_states,
            number_of_matches=MATCHES_PER_EPOCH,
            mode_2x2=mode_2x2,
            REWARD_FUNCTION_TYPE=REWARD_FUNCTION,
            TRANSITION_SCHEMA=TRANSITION_SCHEMA,
            PROGRESS_MESSAGE=f"{Fore.YELLOW}Generating experience for epoch {e + 1}{Style.RESET_ALL}",
            COLLECT_BOARDS=True,
            select_oracle=SELECT_ORACLE,
        )

        if ENDGAME_FRACTION > 0:
            endgame_exp = gen_experience(
                p1_bot=p1,
                p2_bot=p2,
                n_last_states=N_LAST_STATES_ENDGAME,
                number_of_matches=MATCHES_PER_EPOCH,
                mode_2x2=mode_2x2,
                REWARD_FUNCTION_TYPE=REWARD_FUNCTION,
                TRANSITION_SCHEMA=TRANSITION_SCHEMA,
                PROGRESS_MESSAGE=f"{Fore.CYAN}Generating endgame experience for epoch {e + 1}{Style.RESET_ALL}",
                COLLECT_BOARDS=False,
                select_oracle=SELECT_ORACLE,
            )
            endgame_replay_buffer.extend(endgame_exp)  # type: ignore
    else:
        replay_buffer.empty()
        logger.info(f"Reusing same previous experience for epoch {e + 1}")

    replay_buffer.extend(exp)  # type: ignore

    iter_per_epoch = max(len(replay_buffer) // BATCH_SIZE, 1)
    logger.info(
        f"Training during epoch {e} with {len(replay_buffer)} experiences, {iter_per_epoch} iterations."
    )
    for i in range(iter_per_epoch):
        if ENDGAME_FRACTION > 0 and len(endgame_replay_buffer) > 0:
            endgame_size = max(1, round(BATCH_SIZE * ENDGAME_FRACTION))
            curriculum_size = BATCH_SIZE - endgame_size
            endgame_batch = endgame_replay_buffer.sample(
                min(endgame_size, len(endgame_replay_buffer))
            )
            curriculum_batch = replay_buffer.sample(
                min(curriculum_size, len(replay_buffer))
            )
            exp_batch = torch.cat([curriculum_batch, endgame_batch], dim=0)
        else:
            exp_batch = replay_buffer.sample(BATCH_SIZE)

        if exp_batch.shape[0] < BATCH_SIZE:
            logger.warning(
                f"Not enough data to sample a full batch. Expected {BATCH_SIZE}, got {exp_batch.shape[0]}"
            )
            break
        step_i += 1
        dqn_result = DQN_training_step(
            policy_net=policy_net,
            target_net=target_net,
            exp_batch=exp_batch,  # type: ignore
            GAMMA=GAMMA,
            LOSS_APPROACH=LOSS_APPROACH,
            TRANSITION_SCHEMA=TRANSITION_SCHEMA,
            DECOUPLED_TARGET_STYLE=DECOUPLED_TARGET_STYLE,
        )
        if TRANSITION_SCHEMA in ("decoupled_autoreg", "unified_autoreg"):
            if isinstance(dqn_result, tuple) and len(dqn_result) == 5:
                q_place, target_place, q_select, target_select, sel_mask = dqn_result  # type: ignore
            else:
                q_place, target_place, q_select, target_select = dqn_result  # type: ignore
                sel_mask = None
            L_place = (
                loss_fcn(q_place, target_place)
                if q_place.numel() > 0
                else None
            )
            if sel_mask is not None:
                if q_select.numel() > 0 and sel_mask.sum() > 0:
                    diff = q_select - target_select
                    abs_diff = diff.abs()
                    l1_elt = torch.where(
                        abs_diff < 1.0, 0.5 * diff.pow(2), abs_diff - 0.5
                    )
                    L_select = (l1_elt * sel_mask).sum() / sel_mask.sum().clamp_min(1.0)
                else:
                    L_select = None
            else:
                L_select = (
                    loss_fcn(q_select, target_select)
                    if q_select.numel() > 0
                    else None
                )
            if L_place is None and L_select is None:
                raise ValueError(
                    "Per-phase batch produced no active place/select samples."
                )
            w_p = LOSS_ALPHA_PLACE if L_place is not None else 0.0
            w_s = LOSS_ALPHA_SELECT if L_select is not None else 0.0
            denom = w_p + w_s
            term_p = w_p * L_place if L_place is not None else 0.0
            term_s = w_s * L_select if L_select is not None else 0.0
            loss = (term_p + term_s) / denom
        elif LOSS_APPROACH in ("separate_bellman", "mc_select"):
            q_place, target_place, q_select, target_select = dqn_result  # type: ignore
            L_place = loss_fcn(q_place, target_place)
            L_select = loss_fcn(q_select, target_select)
            denom = LOSS_ALPHA_PLACE + LOSS_ALPHA_SELECT
            loss = (
                LOSS_ALPHA_PLACE * L_place + LOSS_ALPHA_SELECT * L_select
            ) / denom
        else:
            state_action_values, expected_state_action_values = dqn_result  # type: ignore
            loss = loss_fcn(state_action_values, expected_state_action_values)
            L_place = None
            L_select = None
        loss_data["loss_values"].append(loss.item())
        loss_data["loss_place_values"].append(
            float(L_place.item()) if L_place is not None else float("nan")
        )
        loss_data["loss_select_values"].append(
            float(L_select.item()) if L_select is not None else float("nan")
        )

        optimizer.zero_grad()
        loss.backward()

        def _head_grad_norm(prefix: str) -> float:
            sq = 0.0
            any_grad = False
            for name, p in policy_net.named_parameters():
                if not name.startswith(prefix):
                    continue
                if p.grad is None:
                    continue
                any_grad = True
                sq += float(p.grad.detach().pow(2).sum().item())
            return float(sq**0.5) if any_grad else float("nan")

        grad_norm_data["grad_norm_fc2_place"].append(_head_grad_norm("fc2_place"))
        grad_norm_data["grad_norm_fc2_select"].append(_head_grad_norm("fc2_select"))

        total_norm = torch.nn.utils.clip_grad_norm_(
            policy_net.parameters(), MAX_GRAD_NORM
        )
        grad_norm_data["grad_norm_values"].append(float(total_norm))
        if total_norm > MAX_GRAD_NORM:
            logger.warning(
                f"Gradient clipping activated! Total norm before clipping: {total_norm:.4f}"
            )
        optimizer.step()

        if (i + 1) % TARGET_UPDATE_FREQ == 0:
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)
            target_net.eval()

    q_place, q_select = p1.evaluate(exp)

    q_values_history["q_place"].append(
        q_place.detach().cpu().tolist() if hasattr(q_place, "detach") else q_place
    )
    q_values_history["q_select"].append(
        q_select.detach().cpu().tolist() if hasattr(q_select, "detach") else q_select
    )
    if len(q_values_history["rewards"]) == 0:
        reward = exp["reward"]
        q_values_history["rewards"].append(
            reward.detach().cpu().tolist() if hasattr(reward, "detach") else reward
        )
    outcome = exp["outcome"]
    q_values_history["outcome"].append(
        outcome.detach().cpu().tolist() if hasattr(outcome, "detach") else outcome
    )
    steps_to_terminal = exp["steps_to_terminal"]
    q_values_history["steps_to_terminal"].append(
        steps_to_terminal.detach().cpu().tolist()
        if hasattr(steps_to_terminal, "detach")
        else steps_to_terminal
    )

    loss_data["epoch_values"].append(step_i)
    grad_norm_data["epoch_values"].append(step_i)

    scheduler.step()
    logger.info(f"Current learning rate: {scheduler.get_last_lr()[0]}")

    p1.TEMPERATURE = TEMPERATURE_EXPLOIT

    contest_results = run_contest(
        player=p1,
        rivals=RIVALS_PATHs,
        rival_class=RIVALS_CLASS,
        rival_options=RIVALS_PARAMs,
        rivals_clip=RIVALS_IN_TOURNAMENT,
        rival_names=RIVALS_NAMEs,
        matches=N_MATCHES_EVAL,
        verbose=False,
        mode_2x2=mode_2x2,
        PROGRESS_MESSAGE=f"{Fore.MAGENTA}Running contest for epoch {e + 1}{Style.RESET_ALL}",
    )
    logger.info(f"Contest results after epoch {e + 1}")
    logger.info(contest_results)
    logger.info(pformat(contest_results))

    for rival_name, wr in contest_2_win_rate(contest_results).items():
        if rival_name not in win_rate:
            win_rate[rival_name] = []
        win_rate[rival_name].append(wr)

    if (CHECKPOINT_FREQ > 0 and (e + 1) % CHECKPOINT_FREQ == 0) or (e + 1) == EPOCHS:
        _fname = CKPT_NAME_GEN(e + 1)
        policy_net.export_model(_fname, checkpoint_folder=CHECKPOINT_FOLDER)

    epochs_results.append(dict(contest_results))

    if (FREQ_EPOCH_SAVING > 0 and (e + 1) % FREQ_EPOCH_SAVING == 0) or (
        e + 1
    ) == EPOCHS:
        logger.info("Saving results to disk...")
        pkl_path = path.join(CHECKPOINT_FOLDER, f"{EXPERIMENT_NAME}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(
                {
                    "epochs_results": epochs_results,
                    "loss_values": loss_data,
                    "grad_norm_data": grad_norm_data,
                    "win_rate": win_rate,
                    "q_values_history": q_values_history,
                },
                f,
            )

        jsonl_path = path.join(
            RESULTS_FOLDER, f"{EXPERIMENT_NAME}{SUMMARY_SUFFIX}"
        )
        ckpt_rec = build_checkpoint_record(
            epoch=e + 1,
            loss_data=loss_data,
            grad_norm_data=grad_norm_data,
            win_rate=win_rate,
            q_values_history=q_values_history,
        )
        append_record(jsonl_path, ckpt_rec)

        if (e + 1) == EPOCHS:
            config_snapshot = {
                "EXPERIMENT_NAME": EXPERIMENT_NAME,
                "STARTING_NET": STARTING_NET,
                "ARCHITECTURE": policy_net.name,
                "TRANSITION_SCHEMA": TRANSITION_SCHEMA,
                "DECOUPLED_TARGET_STYLE": DECOUPLED_TARGET_STYLE,
                "USE_MINIMAX_SELECT_TARGET": USE_MINIMAX_SELECT_TARGET,
                "MINIMAX_SELECT_DEPTH": MINIMAX_SELECT_DEPTH,
                "MINIMAX_SELECT_SCALAR_TARGET": MINIMAX_SELECT_SCALAR_TARGET,
                "LOSS_APPROACH": LOSS_APPROACH,
                "REWARD_FUNCTION": REWARD_FUNCTION,
                "LOSS_ALPHA_PLACE": LOSS_ALPHA_PLACE,
                "LOSS_ALPHA_SELECT": LOSS_ALPHA_SELECT,
                "BATCH_SIZE": BATCH_SIZE,
                "EPOCHS": EPOCHS,
                "LR": LR,
                "LR_F": LR_F,
                "TAU": TAU,
                "GAMMA": GAMMA,
                "MAX_GRAD_NORM": MAX_GRAD_NORM,
                "MATCHES_PER_EPOCH": MATCHES_PER_EPOCH,
                "NUM_EPOCHs_BUFFER": NUM_EPOCHs_BUFFER,
                "N_LAST_STATES_INIT": N_LAST_STATES_INIT,
                "N_LAST_STATES_FINAL": N_LAST_STATES_FINAL,
                "N_LAST_STATES_ENDGAME": N_LAST_STATES_ENDGAME,
                "ENDGAME_FRACTION": ENDGAME_FRACTION,
                "TEMPERATURE_EXPLORE": TEMPERATURE_EXPLORE,
                "TEMPERATURE_EXPLOIT": TEMPERATURE_EXPLOIT,
                "TARGET_UPDATE_FREQ": TARGET_UPDATE_FREQ,
                "GEN_EXPERIENCE_BY_EPOCH": GEN_EXPERIENCE_BY_EPOCH,
                "mode_2x2": mode_2x2,
            }
            final_rec = build_final_record(
                exp_name=EXPERIMENT_NAME,
                epochs=EPOCHS,
                config=config_snapshot,
                loss_data=loss_data,
                grad_norm_data=grad_norm_data,
                win_rate=win_rate,
                q_values_history=q_values_history,
            )
            append_record(jsonl_path, final_rec)

    if (e + 1) % FREQ_EPOCH_PLOT_SHOW == 0 or (e + 1) == EPOCHS:
        logger.debug("Plotting results...")
        plot_boards_comp(
            *boards,
            q_place=q_place,
            q_select=q_select,
            experiment_name=EXPERIMENT_NAME,
            FREQ_EPOCH_SAVING=FREQ_EPOCH_SAVING,
            FOLDER_SAVE=CHECKPOINT_FOLDER,
            current_epoch=e + 1,
        )

        plot_Qv_progress(
            q_values_history,
            exp["outcome"],
            fig_num=4,
            DISPLAY_PLOT=True,
            done_v=exp["done"],
            PLOT_TYPE=Q_PLOT_TYPE,
            group_label="Outcome",
            experiment_name=EXPERIMENT_NAME,
            FREQ_EPOCH_SAVING=FREQ_EPOCH_SAVING,
            FOLDER_SAVE=CHECKPOINT_FOLDER,
            current_epoch=e + 1,
        )

        plot_Qv_horizon(
            q_place,
            q_select,
            exp["outcome"],
            exp["steps_to_terminal"],
            phase=exp.get("phase", None),
            fig_num=5,
            DISPLAY_PLOT=True,
            experiment_name=EXPERIMENT_NAME,
            FREQ_EPOCH_SAVING=FREQ_EPOCH_SAVING,
            FOLDER_SAVE=CHECKPOINT_FOLDER,
            current_epoch=e + 1,
        )

        plot_win_rate(
            *win_rate.items(),
            FREQ_EPOCH_SAVING=FREQ_EPOCH_SAVING,
            FOLDER_SAVE=CHECKPOINT_FOLDER,
            SMOOTHING_WINDOW=SMOOTHING_WINDOW,
            DISPLAY_PLOT=True,
            experiment_name=EXPERIMENT_NAME,
        )

        plot_loss(
            loss_data,
            FREQ_EPOCH_SAVING=FREQ_EPOCH_SAVING,
            FOLDER_SAVE=CHECKPOINT_FOLDER,
            DISPLAY_PLOT=True,
            experiment_name=EXPERIMENT_NAME,
        )

        plot_grad_norm(
            grad_norm_data,
            MAX_GRAD_NORM=MAX_GRAD_NORM,
            FREQ_EPOCH_SAVING=FREQ_EPOCH_SAVING,
            FOLDER_SAVE=CHECKPOINT_FOLDER,
            DISPLAY_PLOT=True,
            experiment_name=EXPERIMENT_NAME,
        )
        logger.debug("Plots updated.")


logger.info("Training completed.")

plt.show(block=True)
