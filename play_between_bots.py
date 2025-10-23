# -*- coding: utf-8 -*-

"""
Python 3
03 / 10 / 2025
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

"Either mathematics is too big for the human mind or the human mind is more than a machine."
-Kurt Godël
"""
from quartopy import play_games
from bot.CNN_bot import Quarto_bot
from bot.CNN_F_bot import Quarto_bot as F_bot

## CNN1
# medio malo
# random bot
_f = "CHECKPOINTS//EXP_id03//20250922_1920-EXP_id03_epoch_0377.pt"
_f2 = "CHECKPOINTS//EXP_id03//20250922_1247-EXP_id03_epoch_0000.pt"
_fgood = "CHECKPOINTS//EXP_id03//20250922_1247-EXP_id03_epoch_0009.pt"

bot_malo = Quarto_bot(model_path=_f, deterministic=False, temperature=0.1)
bot_rand = Quarto_bot(model_path=_f2, deterministic=False, temperature=0.1)
bot_good = Quarto_bot(model_path=_fgood, deterministic=False, temperature=0.1)

## CNNF
_f_Francis = (
    "CHECKPOINTS//others//20251013_1851-ba_increasing_n_last_states_epoch_1000.pt"
)

bot_francis = F_bot(model_path=_f_Francis)

## Michael
_f_Michael = "CHECKPOINTS//others//20250930_1010-EXP_id03_epoch_0017.pt"
bot_Michael = Quarto_bot(model_path=_f_Michael, deterministic=False, temperature=0.1)
_f_M2 = "CHECKPOINTS//others//20251006_2218-EXP_id03_epoch_0010.pt"
bot_Michael2 = Quarto_bot(model_path=_f_M2, deterministic=False, temperature=0.1)
# bot_Michael = Quarto_bot(model_path=_f_Michael, deterministic=True)

## Select bots to play
bot_A = bot_good
bot_A_m = "bot_good"
# bot_A = bot_Michael2
# bot_A_m = "bot_M2"

# bot_B = bot_francis
# bot_B_m = "bot_francis"
bot_B = bot_Michael
bot_B_m = "bot_Michael"
# bot_B = bot_Michael2
# bot_B_m = "bot_Michael2"

N_MATCHES = 500
VERBOSE = False

## Games
res, win_rate_p1 = play_games(
    matches=N_MATCHES,
    player1=bot_A,
    player2=bot_B,
    verbose=VERBOSE,
    save_match=False,
    mode_2x2=True,
)

res, win_rate_p2 = play_games(
    matches=N_MATCHES,
    player1=bot_B,
    player2=bot_A,
    verbose=VERBOSE,
    save_match=False,
    mode_2x2=True,
)

# print(res)
print(f"Player 1 {bot_A_m} vs Player 2 {bot_B_m} over {N_MATCHES} matches")
print(win_rate_p1)
print(f"Player 1 {bot_B_m} vs Player 2 {bot_A_m} over {N_MATCHES} matches")
print(win_rate_p2)
