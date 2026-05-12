"""QC_unifiedNoMask(1)0511_ENDGAME_FRACTION_0.5 — first run of the QC series.

See ``Research-status.md`` → QC_unifiedNoMask and
``models/CNN_unified_nomask.py`` for the full architecture rationale.

What this experiment trains
---------------------------
- Architecture: ``QuartoCNNUnifiedNoMask`` (wider fc1=256, aux legality head,
  unified 32-d phase-stable aux, no phase embedding).
- Schema: ``unified_autoreg`` (decoupled targets with phase-stable aux).
- Loss: ``L = L_DQN + LAMBDA_LEGALITY · BCEWithLogits(legality_logits, is_empty_label)``.
- Data: ME(2) recipe — curriculum N=2→4, endgame fraction 0.5, buffer=8.
- Inference: ``Quarto_unified_nomask_bot`` (no legality filter).

Decision gate (WR-first per repository policy)
----------------------------------------------
- WR vs ``bot_random`` ≥ 95% (hard requirement)
- WR vs ``ME_endgame(2)_E_5000`` ≥ 50%
- ``invalid_argmax_rate`` < 0.05 averaged over last 100 epochs
  (legality has been learned, not masked)
"""

from utils.logger import logger

logger.info("Starting Importing...")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from bot.CNN_bot import Quarto_bot
from bot.CNN_autoreg_bot import Quarto_bot as Quarto_autoreg_bot
from bot.CNN_unified_nomask_bot import Quarto_bot as Quarto_unified_nomask_bot
from models.CNN1 import QuartoCNN
from models.CNN_autoreg import QuartoCNNAutoreg
from models.CNN_unified_nomask import (
    QuartoCNNUnifiedNoMask,
    legality_target_from_board,
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
)
from tqdm.auto import tqdm
from pprint import pformat
import pickle
from colorama import init, Fore, Style
import socket
from os import path

# ---- PARAMS ----
logger.info("Imports done.")

STARTING_NET = None  # train from scratch
EXPERIMENT_NAME = "QC_unifiedNoMask(1)0511"
CHECKPOINT_FOLDER = f"./CHECKPOINTS/{EXPERIMENT_NAME}/"

TRANSITION_SCHEMA = "unified_autoreg"
ARCHITECTURE = QuartoCNNUnifiedNoMask
PLAYER_BOT_CLASS = Quarto_unified_nomask_bot

DECOUPLED_TARGET_STYLE = "td_place_mc_select"


def estimate_steps_per_match(n_last_states: int, transition_schema: str) -> int:
    if transition_schema in ("decoupled_autoreg", "unified_autoreg"):
        return max(1, 2 * n_last_states - 1)
    return n_last_states


LOSS_APPROACH = "mc_select"  # ignored for unified_autoreg; kept for compat
REWARD_FUNCTION = "final"

GEN_EXPERIENCE_BY_EPOCH = True

N_MATCHES_EVAL = 30

BATCH_SIZE = 32
mode_2x2 = True

EPOCHS = 5_000

# ME(2)-style curriculum: warm up on N=2, expand to N=4
N_LAST_STATES_INIT = 2
N_LAST_STATES_FINAL = 4

MATCHES_PER_EPOCH = 32
NUM_EPOCHs_BUFFER = 8

ENDGAME_FRACTION = 0.5  # sweep variable for the QC series
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

FREQ_EPOCH_SAVING = 1_000
CHECKPOINT_FREQ = 250
FREQ_EPOCH_PLOT_SHOW = 1_000_000

SMOOTHING_WINDOW = 10
Q_PLOT_TYPE = "hist"

MAX_GRAD_NORM = 1.0
LR = 7e-4
LR_F = LR
TAU = 0.01
GAMMA = 0.99

# Auxiliary legality-head loss weight.
# See games-interp/training-recommendations.md T4 (λ ≈ 0.05 starting point).
LAMBDA_LEGALITY = 0.05

BASELINES = [
    {
        "path": (
            "CHECKPOINTS/ME_endgame(2)0429_ENDGAME_FRACTION_0.5/"
            "20260507_0829-ME_endgame(2)0429_ENDGAME_FRACTION_0.5_E_5000.pt"
        ),
        "name": "ME_endgame(2)_E_5000",
        "bot": Quarto_autoreg_bot,
        "params": {
            "deterministic": False,
            "temperature": 0.1,
            "model_class": QuartoCNNAutoreg,
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

logger.info(f"PC name: {socket.gethostname()}")
logger.info(f"Experiment name:\t{EXPERIMENT_NAME}")
logger.info(
    f"Train conf.:\t{EPOCHS=}, {BATCH_SIZE=}, {LR=}, {LR_F=}, {GAMMA=}, {TAU=}, "
    f"{MAX_GRAD_NORM=}, {LAMBDA_LEGALITY=}"
)
logger.info(f"Exp. gen.:\t{MATCHES_PER_EPOCH=}, {STEPS_PER_EPOCH=}, {REPLAY_SIZE=}")
logger.info(f"Network updates:\tFull buffer sweep each epoch, {TARGET_UPDATE_FREQ=}")
logger.info(f"Exploration:\t{TEMPERATURE_EXPLORE=}, {TEMPERATURE_EXPLOIT=}")
logger.info(f"N_LAST_STATES:\tINIT={N_LAST_STATES_INIT}, FINAL={N_LAST_STATES_FINAL}")
logger.info(f"Checkpointing:\t{CHECKPOINT_FREQ=}, {FREQ_EPOCH_SAVING=}")
logger.info(
    f"ENDGAME_FRACTION={ENDGAME_FRACTION}, N_LAST_STATES_ENDGAME={N_LAST_STATES_ENDGAME}"
)
logger.info(f"REWARD_FUNCTION={REWARD_FUNCTION}")
logger.info(f"TRANSITION_SCHEMA={TRANSITION_SCHEMA}")
logger.info(f"DECOUPLED_TARGET_STYLE={DECOUPLED_TARGET_STYLE}")

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
# QC-specific metrics
invalid_argmax_history: list[float] = []
legality_loss_history: list[float] = []

torch.manual_seed(5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

policy_net = ARCHITECTURE()
target_net = ARCHITECTURE()
logger.info(f"Architecture: {policy_net.name}")

policy_net.to(device)
target_net.to(device)
target_net.load_state_dict(policy_net.state_dict())

CKPT_NAME_GEN = lambda epoch: f"{EXPERIMENT_NAME}_E_{epoch:04d}"
policy_net.export_model(CKPT_NAME_GEN(0), CHECKPOINT_FOLDER)

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

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=LR_F)

loss_fcn = nn.SmoothL1Loss()
bce_fcn = nn.BCEWithLogitsLoss()

epochs_results = []
loss_data: dict[str, list[float | int]] = {"loss_values": [], "epoch_values": []}
grad_norm_data: dict[str, list[float | int]] = {
    "grad_norm_values": [],
    "epoch_values": [],
}

init(autoreset=True)

logger.info("Hyperparameters loaded.")
logger.info("Starting training...")

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

    n_last_states = round(
        N_LAST_STATES_INIT
        + (N_LAST_STATES_FINAL - N_LAST_STATES_INIT) * (e / (EPOCHS - 1))
    )
    logger.info(f"Using n_last_states={n_last_states} for epoch {e + 1}")

    # Reset the legality counters on p1 — this epoch's experience generation
    # is the diagnostic window for ``invalid_argmax_rate``.
    p1.reset_legality_counters()

    if GEN_EXPERIENCE_BY_EPOCH or e == 0:
        exp, boards = gen_experience(
            p1_bot=p1,
            p2_bot=p2,
            n_last_states=n_last_states,
            number_of_matches=MATCHES_PER_EPOCH,
            mode_2x2=mode_2x2,
            REWARD_FUNCTION_TYPE=REWARD_FUNCTION,
            TRANSITION_SCHEMA=TRANSITION_SCHEMA,
            PROGRESS_MESSAGE=f"{Fore.YELLOW}Gen experience epoch {e + 1}{Style.RESET_ALL}",
            COLLECT_BOARDS=True,
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
                PROGRESS_MESSAGE=f"{Fore.CYAN}Endgame exp epoch {e + 1}{Style.RESET_ALL}",
                COLLECT_BOARDS=False,
            )
            endgame_replay_buffer.extend(endgame_exp)  # type: ignore
    else:
        replay_buffer.empty()

    # Record the epoch's invalid-argmax rate from p1's self-play
    invalid_argmax_history.append(p1.invalid_argmax_rate())
    invalid_argmax_history.append(p2.invalid_argmax_rate())

    replay_buffer.extend(exp)  # type: ignore

    iter_per_epoch = max(len(replay_buffer) // BATCH_SIZE, 1)
    epoch_legality_losses: list[float] = []
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
                f"Not enough data to sample a full batch. Got {exp_batch.shape[0]}."
            )
            break
        step_i += 1

        # ── DQN loss (Q_place + Q_select)
        dqn_result = DQN_training_step(
            policy_net=policy_net,
            target_net=target_net,
            exp_batch=exp_batch,  # type: ignore
            GAMMA=GAMMA,
            LOSS_APPROACH=LOSS_APPROACH,
            TRANSITION_SCHEMA=TRANSITION_SCHEMA,
            DECOUPLED_TARGET_STYLE=DECOUPLED_TARGET_STYLE,
        )
        q_place, target_place, q_select, target_select = dqn_result  # type: ignore
        active_losses: list[torch.Tensor] = []
        if q_place.numel() > 0:
            active_losses.append(loss_fcn(q_place, target_place))
        if q_select.numel() > 0:
            active_losses.append(loss_fcn(q_select, target_select))
        if not active_losses:
            raise ValueError("Unified-autoreg batch produced no active samples.")
        dqn_loss = torch.stack(active_losses).mean()

        # ── Legality auxiliary loss (QC-specific)
        # Compute legality logits on the batch's state_board, supervised by
        # the deterministic ``is_empty`` label derived from the same tensor.
        state_board = exp_batch["state_board"].to(device).float()  # type: ignore
        state_aux = exp_batch["state_aux"].to(device).float()  # type: ignore
        legality_logits = policy_net.legality_logits(state_board, state_aux)
        legality_target = legality_target_from_board(state_board)
        legality_loss = bce_fcn(legality_logits, legality_target)
        epoch_legality_losses.append(legality_loss.item())

        loss = dqn_loss + LAMBDA_LEGALITY * legality_loss
        loss_data["loss_values"].append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(
            policy_net.parameters(), MAX_GRAD_NORM
        )
        grad_norm_data["grad_norm_values"].append(float(total_norm))
        optimizer.step()

        if (i + 1) % TARGET_UPDATE_FREQ == 0:
            tnet_sd = target_net.state_dict()
            pnet_sd = policy_net.state_dict()
            for key in pnet_sd:
                tnet_sd[key] = pnet_sd[key] * TAU + tnet_sd[key] * (1 - TAU)
            target_net.load_state_dict(tnet_sd)
            target_net.eval()

    scheduler.step()
    if epoch_legality_losses:
        legality_loss_history.append(
            sum(epoch_legality_losses) / len(epoch_legality_losses)
        )
    logger.info(
        f"Epoch {e + 1}: invalid_argmax_rate={invalid_argmax_history[-2]:.3f}/"
        f"{invalid_argmax_history[-1]:.3f}, "
        f"legality_loss={legality_loss_history[-1] if legality_loss_history else float('nan'):.4f}"
    )

    q_place, q_select = p1.evaluate(exp)
    q_values_history["q_place"].append(
        q_place.detach().cpu().tolist() if hasattr(q_place, "detach") else q_place
    )
    q_values_history["q_select"].append(
        q_select.detach().cpu().tolist() if hasattr(q_select, "detach") else q_select
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
    if len(q_values_history["rewards"]) == 0:
        reward = exp["reward"]
        q_values_history["rewards"].append(
            reward.detach().cpu().tolist() if hasattr(reward, "detach") else reward
        )

    loss_data["epoch_values"].append(step_i)
    grad_norm_data["epoch_values"].append(step_i)

    p1.TEMPERATURE = TEMPERATURE_EXPLOIT
    p1.reset_legality_counters()
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
        PROGRESS_MESSAGE=f"{Fore.MAGENTA}Contest epoch {e + 1}{Style.RESET_ALL}",
    )
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
        pkl_path = path.join(CHECKPOINT_FOLDER, f"{EXPERIMENT_NAME}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(
                {
                    "epochs_results": epochs_results,
                    "loss_values": loss_data,
                    "grad_norm_data": grad_norm_data,
                    "win_rate": win_rate,
                    "q_values_history": q_values_history,
                    # QC-only metrics
                    "invalid_argmax_history": invalid_argmax_history,
                    "legality_loss_history": legality_loss_history,
                    "LAMBDA_LEGALITY": LAMBDA_LEGALITY,
                },
                f,
            )


logger.info("Training completed.")
