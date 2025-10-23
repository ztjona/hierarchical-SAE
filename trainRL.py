from utils.logger import logger

logger.info("Starting. Importing...")

import torch
import torch.nn as nn
import torch.optim as optim
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from bot.CNN_bot import Quarto_bot
from models.CNN1 import QuartoCNN
from QuartoRL import (
    gen_experience,
    run_contest,
    contest_2_win_rate,
    DQN_training_step,
    plot_win_rate,
    plot_loss,
)
from tqdm.auto import tqdm
from pprint import pprint
import pickle
from colorama import init, Fore, Style
import matplotlib.pyplot as plt
import numpy as np

# ---- PARAMS ----
logger.info("Imports done.")

EXPERIMENT_NAME = "E02_win_rate"
CHECKPOINT_FOLDER = f"./CHECKPOINTS/{EXPERIMENT_NAME}/"

# The bot at the end of each epoch will be evaluated against a limited number of rivals known as BASELINES.
BASELINES = [
    {
        "path": "CHECKPOINTS//EXP_id03//20250922_1247-EXP_id03_epoch_0009.pt",
        "name": "bot_good",
        "bot": Quarto_bot,
        "params": {"deterministic": False, "temperature": 0.1},
    },
    {
        "path": "CHECKPOINTS//EXP_id03//20250922_1247-EXP_id03_epoch_0000.pt",
        "name": "bot_random",
        "bot": Quarto_bot,
        "params": {"deterministic": False, "temperature": 0.1},
    },
    # {
    #     "path": "CHECKPOINTS//others//20250930_1010-EXP_id03_epoch_0017.pt",
    #     "name": "bot_Michael",
    #     "bot": Quarto_bot,
    #     "params": {"deterministic": False, "temperature": 0.1},
    # },
    {
        "path": "CHECKPOINTS//others//20251006_2218-EXP_id03_epoch_0010.pt",
        "name": "bot_Michael2",
        "bot": Quarto_bot,
        "params": {"deterministic": False, "temperature": 0.1},
    },
]
N_MATCHES_EVAL = 30  # number of matches to evaluate the bot at the end of each epoch for the selected BASELINES


BATCH_SIZE = 512
# When True, checks for winning in 2x2 squares. False, only in rows, columns and diagonals.
mode_2x2 = True

# every epoch experience is generated with a new bot instance, models are saved at the end of each epoch
EPOCHS = 5_000

MATCHES_PER_EPOCH = 310  # number self-play matches per epoch
# ~x10 of matches_per_epoch, used to generate experience
STEPS_PER_EPOCH = 10 * MATCHES_PER_EPOCH
# number of times the network is updated per epoch
ITER_PER_EPOCH = STEPS_PER_EPOCH // BATCH_SIZE

# ~EPOCHs x STEPS_PER_EPOCH, info from last epochs
REPLAY_SIZE = 50 * STEPS_PER_EPOCH

# update target network every n batches processed, ~x3/epoch
N_BATCHS_2_UPDATE_TARGET = ITER_PER_EPOCH // 2

# number of last states to consider in the experience generation at the beginning of training
N_LAST_STATES_INIT: int = 2
# number of last states to consider in the experience generation at the end of training. -1 means all states
N_LAST_STATES_FINAL: int = 16  # 16 is all states in 4x4 board

# temperature for exploration, higher values lead to more exploration
TEMPERATURE_EXPLORE = 2  # view test of temperature

# temperature for exploitation, lower values lead to more exploitation
TEMPERATURE_EXPLOIT = 0.1

FREQ_EPOCH_SAVING = 100  # save model, figures every n epochs
# in iters if >= N_ITERS show epoch lines in loss plot
SHOW_EPOCH_LINES = ITER_PER_EPOCH * 20
SMOOTHING_WINDOW = 10
# ###########################
MAX_GRAD_NORM = 1.0
LR = 1e-4
TAU = 0.01  # recommended value by CHATGPT
# TAU = 0.005
GAMMA = 0.99

# from configs_debug import * # Uncomment to use debug configs

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

# ###########################
torch.manual_seed(50)
policy_net = QuartoCNN()
target_net = QuartoCNN()
# Set target net weights to policy net weights
target_net.load_state_dict(policy_net.state_dict())

CKPT_NAME_GEN = lambda epoch: f"{EXPERIMENT_NAME}_epoch_{epoch:04d}"
policy_net.export_model(CKPT_NAME_GEN(0), CHECKPOINT_FOLDER)

# list of file names by epoch
# checkpoints_files: list[str] = [_fcheckpoint_name]

# ###########################
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=REPLAY_SIZE),
    sampler=SamplerWithoutReplacement(),
)

# ###########################
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=1e-6)

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
    range(EPOCHS), desc=f"{Fore.GREEN}Epochs{Style.RESET_ALL}", position=1, leave=True
):
    # load models
    p1 = Quarto_bot(
        model=policy_net, deterministic=False, temperature=TEMPERATURE_EXPLORE
    )
    p2 = Quarto_bot(
        model=policy_net, deterministic=False, temperature=TEMPERATURE_EXPLORE
    )  # self play

    logger.debug(f"Using temperatures: p1={p1.TEMPERATURE}, p2={p2.TEMPERATURE}")

    logger.debug("Generating experience for epoch %d", e + 1)

    # Linearly interpolate n_last_states from N_LAST_STATES_INIT to N_LAST_STATES_FINAL over EPOCHS
    n_last_states = round(
        N_LAST_STATES_INIT
        + (N_LAST_STATES_FINAL - N_LAST_STATES_INIT) * (e / (EPOCHS - 1))
    )
    logger.info(f"Using n_last_states={n_last_states} for epoch {e + 1}")

    # ---- GENERATE EXPERIENCE by SELF-PLAY----
    exp = gen_experience(
        p1_bot=p1,
        p2_bot=p2,
        n_last_states=n_last_states,
        number_of_matches=MATCHES_PER_EPOCH,
        mode_2x2=mode_2x2,
        PROGRESS_MESSAGE=f"{Fore.YELLOW}Generating experience for epoch {e + 1}{Style.RESET_ALL}",
    )

    replay_buffer.extend(exp)  # type: ignore

    for i in range(ITER_PER_EPOCH):
        pbar.update(1)
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
            loss_fcn=loss_fcn,  # type: ignore
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

    # ------- END OF EPOCH -------
    loss_data["epoch_values"].append(step_i)
    # Save the model at the end of each epoch
    _fname = CKPT_NAME_GEN(e + 1)
    policy_net.export_model(_fname, checkpoint_folder=CHECKPOINT_FOLDER)
    # checkpoints_files.append(_f_fname)

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
    logger.info(pprint(contest_results))

    for rival_name, wr in contest_2_win_rate(contest_results).items():
        if rival_name not in win_rate:
            win_rate[rival_name] = []
        win_rate[rival_name].append(wr)

    # store results
    epochs_results.append(dict(contest_results))
    # ------- SAVE RESULTS -----------
    if (e + 1) % FREQ_EPOCH_SAVING == 0:
        with open(f"{EXPERIMENT_NAME}.pkl", "wb") as f:
            pickle.dump(
                {
                    "epochs_results": epochs_results,
                    "loss_values": loss_data,
                    "win_rate": win_rate,
                },
                f,
            )

    # ------- PLOT RESULTS -----------
    plot_win_rate(
        *win_rate.items(),
        FREQ_EPOCH_SAVING=FREQ_EPOCH_SAVING,
        FOLDER_SAVE=CHECKPOINT_FOLDER,
        SMOOTHING_WINDOW=SMOOTHING_WINDOW,
    )
    # plot_contest_results(epochs_results)
    plot_loss(
        loss_data,
        FREQ_EPOCH_SAVING=FREQ_EPOCH_SAVING,
        FOLDER_SAVE=CHECKPOINT_FOLDER,
        SHOW_EPOCH_LINES=SHOW_EPOCH_LINES,
    )


logger.info("Training completed.")

plt.show(block=True)
