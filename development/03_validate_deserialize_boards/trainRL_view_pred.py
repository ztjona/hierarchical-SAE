from utils.logger import logger

logger.info("Starting Importing...")

import torch
import torch.nn as nn
import torch.optim as optim
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from bot.CNN_bot import Quarto_bot
from models.CNN1 import QuartoCNN
from models.CNN_uncoupled import QuartoCNN as QuartoCNN_uncoupled
from QuartoRL import (
    gen_experience,
    run_contest,
    contest_2_win_rate,
    DQN_training_step,
    plot_win_rate,
    plot_loss,
    plot_boards_comp,
    plot_Qv_progress,
)
from tqdm.auto import tqdm
from pprint import pformat
import pickle
from colorama import init, Fore, Style
import socket
import matplotlib.pyplot as plt

# ---- PARAMS ----
logger.info("Imports done.")

# STARTING_NET = "CHECKPOINTS//REF//20251023_1649-_E02_win_rate_epoch_0022.pt"
STARTING_NET = None  # Set to None to start with random weights
EXPERIMENT_NAME = "04_LOSS"
CHECKPOINT_FOLDER = f"./CHECKPOINTS/{EXPERIMENT_NAME}/"
# ARCHITECTURE = QuartoCNN
ARCHITECTURE = QuartoCNN_uncoupled
LOSS_APPROACH = "combined_avg"  # Options: "combined_avg", "only_select", "only_place"
REWARD_FUNCTION = "propagate"  # "final", "propagate", "discount"

# if True, experience is generated at the beginning of each epoch
# if False, experience is generated only at the first epoch and reused for the rest of epochs
# GEN_EXPERIENCE_BY_EPOCH = True
GEN_EXPERIENCE_BY_EPOCH = False

# The bot at the end of each epoch will be evaluated against a limited number of rivals known as BASELINES.
BASELINES = [
    {
        # "path": "CHECKPOINTS//EXP_id03//20250922_1247-EXP_id03_epoch_0009.pt",
        "path": "CHECKPOINTS//REF//20251023_1649-_E02_win_rate_epoch_0022.pt",
        "name": "bot_good_WR_B",
        "bot": Quarto_bot,
        "params": {
            "deterministic": False,
            "temperature": 0.1,
            "model_class": QuartoCNN,
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
N_MATCHES_EVAL = 30  # number of matches to evaluate the bot at the end of each epoch for the selected BASELINES


BATCH_SIZE = 30
# When True, checks for winning in 2x2 squares. False, only in rows, columns and diagonals.
mode_2x2 = True

# every epoch experience is generated with a new bot instance, models are saved at the end of each epoch
EPOCHS = 1000

# number of last states to consider in the experience generation at the beginning of training
N_LAST_STATES_INIT: int = 2
# number of last states to consider in the experience generation at the end of training. -1 means all states
N_LAST_STATES_FINAL = 2  # 16 is all states in 4x4 board

MATCHES_PER_EPOCH = 5000  # number self-play matches per epoch
# movs per match * #_matches per epoch (max 16, but avg less)
STEPS_PER_EPOCH = N_LAST_STATES_FINAL * MATCHES_PER_EPOCH
# number of times the network is updated per epoch
ITER_PER_EPOCH = STEPS_PER_EPOCH // BATCH_SIZE

if GEN_EXPERIENCE_BY_EPOCH:
    # EPOCHs x STEPS_PER_EPOCH, DATA from the last _#_ epochs
    REPLAY_SIZE = 50 * STEPS_PER_EPOCH
else:
    # only STEPS_PER_EPOCH, DATA from the first epoch
    REPLAY_SIZE = STEPS_PER_EPOCH

# update target network every n batches processed, ~x3/epoch
N_BATCHS_2_UPDATE_TARGET = ITER_PER_EPOCH // 3


# temperature for exploration, higher values lead to more exploration
TEMPERATURE_EXPLORE = 2  # view test of temperature

# temperature for exploitation, lower values lead to more exploitation
TEMPERATURE_EXPLOIT = 0.1

FREQ_EPOCH_SAVING = 1000  # save model, figures every n epochs


# Plots are shown every epoch until this number of epochs. After that, only every
# FREQ_EPOCH_PLOT_SHOW epochs. At the end, all plots are shown again.
FREQ_EPOCH_PLOT_SHOW = 300

# in iters if >= N_ITERS show epoch lines in loss plot
SMOOTHING_WINDOW = 10

# Q-value plotting configuration
Q_PLOT_TYPE = "hist"  # Options: "time_series" or "hist"

# ###########################
MAX_GRAD_NORM = 1.0
LR = 5e-5  # initial
LR_F = 5e-5
TAU = 0.01  # recommended value by CHATGPT
# TAU = 0.005
GAMMA = 0.99

logger.info(f"PC name: {socket.gethostname()}")
logger.info(f"Experiment name:\t{EXPERIMENT_NAME}")
logger.info(
    f"Train conf.:\t{EPOCHS=}, {BATCH_SIZE=}, {LR=}, {LR_F=}, {GAMMA=}, {TAU=}, {MAX_GRAD_NORM=}"
)
logger.info(f"Exp. gen.:\t{MATCHES_PER_EPOCH=}, {STEPS_PER_EPOCH=}, {REPLAY_SIZE=}")
logger.info(f"Network updates:\t{ITER_PER_EPOCH=}, {N_BATCHS_2_UPDATE_TARGET=}")
logger.info(f"Exploration:\t{TEMPERATURE_EXPLORE=}, {TEMPERATURE_EXPLOIT=}")
logger.info(f"N_LAST_STATES:\tINIT={N_LAST_STATES_INIT}, FINAL={N_LAST_STATES_FINAL}")
logger.info(f"LOSS_APPROACH={LOSS_APPROACH}")
logger.info(f"REWARD_FUNCTION={REWARD_FUNCTION}")

# ###########################
# Unpack baselines into rivals for evaluation
# limit the number of rivals for evaluation, -1 means no limit
RIVALS_IN_TOURNAMENT = -1
RIVALS_NAMEs = [b["name"] for b in BASELINES]
RIVALS_PATHs = [b["path"] for b in BASELINES]
RIVALS_CLASS = BASELINES[0]["bot"]  # assumes all baselines have the same class TODO
# assumes all baselines have the same params TODO
RIVALS_PARAMs = BASELINES[0]["params"]

win_rate: dict[str | int, list[float]] = {}  # list of win rates of epochs by rival
q_values_history: dict[str, list] = {
    "q_place": [],
    "q_select": [],
}  # Track Q-values over epochs

# ###########################
torch.manual_seed(5)

# Setup device - use CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

policy_net = ARCHITECTURE()
target_net = ARCHITECTURE()
logger.info(f"Architecture: {policy_net.name}")

# Move models to device
policy_net.to(device)
target_net.to(device)
logger.info(f"Models moved to {device}")

# Load starting checkpoint if provided
if STARTING_NET is not None:
    logger.info(f"Loading starting checkpoint from: {STARTING_NET}")
    try:
        policy_net.load_state_dict(torch.load(STARTING_NET, map_location=device))
        logger.info("Successfully loaded starting checkpoint")
    except FileNotFoundError:
        logger.error(f"Checkpoint file not found: {STARTING_NET}")
        raise
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        raise
else:
    logger.info("Starting with random weights (no checkpoint provided)")

# Set target net weights to policy net weights
target_net.load_state_dict(policy_net.state_dict())

CKPT_NAME_GEN = lambda epoch: f"{EXPERIMENT_NAME}_E_{epoch:04d}"
policy_net.export_model(CKPT_NAME_GEN(0), CHECKPOINT_FOLDER)

# ###########################
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=REPLAY_SIZE),
    sampler=SamplerWithoutReplacement(),
)

# ###########################
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=LR_F)

# The Huber loss acts like the mean squared error when the error is small, but like the mean absolute error when the error is large - this makes it more robust to outliers when the estimates of Q are very noisy.
loss_fcn = nn.SmoothL1Loss()

epochs_results = []  # to store the results of each epoch
loss_data: dict[str, list[float | int]] = {
    "loss_values": [],
    "epoch_values": [],  # iter value at the end of each epoch
}  # to track loss values during training
# ###########################
init(autoreset=True)  # COLORAMA

pbar = tqdm(
    total=EPOCHS * ITER_PER_EPOCH,
    desc=f"{Fore.CYAN}\n Update network{Style.RESET_ALL}",
    leave=True,
    position=1,
    unit="Iter.",
)

logger.info("Hyperparameters loaded.")
logger.info("Starting training...")

# -------------------------- TRAINING LOOP ---------------------------
step_i = -1  # counter of training steps
# Outer loop over epochs
for e in tqdm(
    range(EPOCHS), desc=f"{Fore.GREEN}Epochs{Style.RESET_ALL}", position=0, leave=True
):
    # load models
    p1 = Quarto_bot(
        model=policy_net, deterministic=False, temperature=TEMPERATURE_EXPLORE
    )
    p2 = Quarto_bot(
        model=policy_net, deterministic=False, temperature=TEMPERATURE_EXPLORE
    )  # self play

    logger.debug(f"Using temperatures: p1={p1.TEMPERATURE}, p2={p2.TEMPERATURE}")

    # Linearly interpolate n_last_states from N_LAST_STATES_INIT to N_LAST_STATES_FINAL over EPOCHS
    n_last_states = round(
        N_LAST_STATES_INIT
        + (N_LAST_STATES_FINAL - N_LAST_STATES_INIT) * (e / (EPOCHS - 1))
    )
    logger.info(f"Using n_last_states={n_last_states} for epoch {e + 1}")

    if GEN_EXPERIENCE_BY_EPOCH or e == 0:
        logger.info("Generating experience for epoch %d", e + 1)

        # ---- GENERATE EXPERIENCE by SELF-PLAY----
        exp, boards = gen_experience(
            p1_bot=p1,
            p2_bot=p2,
            n_last_states=n_last_states,
            number_of_matches=MATCHES_PER_EPOCH,
            mode_2x2=mode_2x2,
            REWARD_FUNCTION_TYPE=REWARD_FUNCTION,
            PROGRESS_MESSAGE=f"{Fore.YELLOW}Generating experience for epoch {e + 1}{Style.RESET_ALL}",
            COLLECT_BOARDS=True,
        )
        logger.info("Initial experience generated.")
    else:
        replay_buffer.empty()
        logger.info(f"Reusing same previous experience for epoch {e + 1}")

    replay_buffer.extend(exp)  # type: ignore
    logger.info(f"Training during epoch with {len(replay_buffer)} experiences.")
    for i in range(ITER_PER_EPOCH):
        pbar.update(1)
        # ---- SAMPLE BATCH FROM REPLAY BUFFER ----
        exp_batch = replay_buffer.sample(BATCH_SIZE)

        if exp_batch.shape[0] < BATCH_SIZE:
            logger.warning(
                f"Not enough data to sample a full batch. Expected {BATCH_SIZE}, got {exp_batch.shape[0]}"
            )
            continue
        step_i += 1
        # ---- TRAINING STEP ----
        state_action_values, expected_state_action_values = DQN_training_step(
            policy_net=policy_net,
            target_net=target_net,
            exp_batch=exp_batch,
            GAMMA=GAMMA,
            LOSS_APPROACH=LOSS_APPROACH,
        )
        loss = loss_fcn(state_action_values, expected_state_action_values)
        loss_data["loss_values"].append(loss.item())

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()

        # Optimization: grad clipping and optimization step
        # this is not strictly mandatory but it's good practice to keep
        # your gradient norm bounded
        # clip_grad_value_ NOT USED!
        total_norm = torch.nn.utils.clip_grad_norm_(
            policy_net.parameters(), MAX_GRAD_NORM
        )
        if total_norm > MAX_GRAD_NORM:
            logger.warning(
                f"Gradient clipping activated! Total norm before clipping: {total_norm:.4f}"
            )
        optimizer.step()
        # optimizer.zero_grad() # in PPO

        # ----------- Update target network
        if i % N_BATCHS_2_UPDATE_TARGET == 0:
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)
            target_net.eval()  # Ensure target network stays in eval mode

    # ------- END OF EPOCH -------
    # Evaluate Q-values for the experience batch
    q_place, q_select = p1.evaluate(exp)

    q_values_history["q_place"].append(q_place)
    q_values_history["q_select"].append(q_select)

    loss_data["epoch_values"].append(step_i)

    # We're also using a learning rate scheduler. Like the gradient clipping,
    # this is a nice-to-have but nothing necessary for PPO to work.
    scheduler.step()
    logger.info(f"Current learning rate: {scheduler.get_last_lr()[0]}")

    # ------- RUN CONTEST -----------
    # modify the bots to use different temperatures for exploration and exploitation
    # Ignore the last epoch, as it is the current model

    # Always False!
    # p1.DETERMINISTIC = False  # When True always repeat moves, is like only 1 game!
    # assert not p1.DETERMINISTIC, "p1 bot should be non-deterministic for evaluation"
    p1.TEMPERATURE = TEMPERATURE_EXPLOIT

    contest_results = run_contest(
        player=p1,
        rivals=RIVALS_PATHs,
        rival_class=RIVALS_CLASS,
        rival_options=RIVALS_PARAMs,
        rivals_clip=RIVALS_IN_TOURNAMENT,  # limit the number of rivals for evaluation, -1 means no limit
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

    # ------- SAVE RESULTS -----------
    # --- Save the model at the end of each epoch
    _fname = CKPT_NAME_GEN(e + 1)
    policy_net.export_model(_fname, checkpoint_folder=CHECKPOINT_FOLDER)

    # ------ Store results
    epochs_results.append(dict(contest_results))

    if (e + 1) % FREQ_EPOCH_SAVING == 0:
        with open(f"{EXPERIMENT_NAME}.pkl", "wb") as f:
            pickle.dump(
                {
                    "epochs_results": epochs_results,
                    "loss_values": loss_data,
                    "win_rate": win_rate,
                    "q_values_history": q_values_history,
                },
                f,
            )

    # ------- PLOT RESULTS -----------
    if (e + 1) % FREQ_EPOCH_PLOT_SHOW == 0 or (e + 1) == EPOCHS:
        logger.debug("Plotting results...")
        plot_boards_comp(
            *boards,
            q_place=q_place,
            q_select=q_select,
            experiment_name=EXPERIMENT_NAME,
        )

        plot_Qv_progress(
            q_values_history,
            exp["reward"],
            fig_num=4,
            DISPLAY_PLOT=True,
            done_v=exp["done"],
            PLOT_TYPE=Q_PLOT_TYPE,
            experiment_name=EXPERIMENT_NAME,
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
        logger.debug("Plots updated.")


logger.info("Training completed.")

plt.show(block=True)
