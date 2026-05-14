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
from bot.CNN_unified_bot import Quarto_bot as Quarto_unified_bot
from bot.CNN_autoreg_bot import Quarto_bot as Quarto_autoreg_bot

from models.CNN_fdec import QuartoCNNExtended
from models.CNN_uncoupled import QuartoCNN as QuartoCNN_uncoupled
from models.CNN_autoreg import QuartoCNNAutoreg
from models.CNN_autoreg_sa import QuartoCNNAutoregUnifiedS4

from pprint import pprint

## CNN1
# medio malo
# random bot
_f = "CHECKPOINTS//EXP_id03//20250922_1920-EXP_id03_epoch_0377.pt"
_f2 = "CHECKPOINTS//EXP_id03//20250922_1247-EXP_id03_epoch_0000.pt"
_fgood = "CHECKPOINTS//EXP_id03//20250922_1247-EXP_id03_epoch_0009.pt"
# _fgood2 = "CHECKPOINTS//E02_win_rate//20251023_1326-E02_win_rate_epoch_0031.pt"

_fgood2B = "CHECKPOINTS//REF//20251023_1649-_E02_win_rate_epoch_0022.pt"
_f_B02rep = "CHECKPOINTS//B02replicate(5)0121_LR_0.001//20260123_0255-B02replicate(5)0121_LR_0.001_E_9997.pt"

_fFrancis_dec = (
    "CHECKPOINTS//Francis//20251204_0932-ba_increasing_n_last_states_epoch_0505.pt"
)

# previous champions
_f_loss_BT = "CHECKPOINTS//LOSS_APPROACHs_1212-2_only_select//20251212_2206-LOSS_APPROACHs_1212-2_only_select_E_1034.pt"
_f_Aa_replay = "CHECKPOINTS//Aa_replay(2)0226_NUM_EPOCHs_BUFFER_8//20260227_1103-Aa_replay(2)0226_NUM_EPOCHs_BUFFER_8_E_5000.pt"

_f_ME_endgame = ".//CHECKPOINTS//ME_endgame(2)0429_ENDGAME_FRACTION_0.5//20260507_0829-ME_endgame(2)0429_ENDGAME_FRACTION_0.5_E_5000.pt"

# Champion
_f_Sa_archScan = "CHECKPOINTS//Sa_archScan(3)0512_ARCH_S4_uniform512//20260514_0815-Sa_archScan(3)0512_ARCH_S4_uniform512_E_5000.pt"


bot_Sa_archScan = Quarto_unified_bot(
    model_path=_f_Sa_archScan,
    model_class=QuartoCNNAutoregUnifiedS4,
    deterministic=False,
    temperature=0.1,
)

bot_ME_endgame = Quarto_autoreg_bot(
    model_path=_f_ME_endgame,
    model_class=QuartoCNNAutoreg,
    deterministic=False,
    temperature=0.1,
)
bot_Aa_replay = Quarto_bot(
    model_path=_f_Aa_replay,
    model_class=QuartoCNN_uncoupled,
    deterministic=False,
    temperature=0.1,
)
bot_B02rep = Quarto_bot(
    model_path=_f_B02rep,
    model_class=QuartoCNN_uncoupled,
    deterministic=False,
    temperature=0.1,
)
bot_loss_BT = Quarto_bot(
    model_path=_f_loss_BT,
    model_class=QuartoCNN_uncoupled,
    deterministic=False,
    temperature=0.1,
)

bot_malo = Quarto_bot(model_path=_f, deterministic=False, temperature=0.1)
bot_rand = Quarto_bot(model_path=_f2, deterministic=False, temperature=0.1)
bot_good = Quarto_bot(model_path=_fgood, deterministic=False, temperature=0.1)
bot_good2B = Quarto_bot(model_path=_fgood2B, deterministic=False, temperature=0.1)
bot_Francis_dec = Quarto_bot(
    model_path=_fFrancis_dec,
    model_class=QuartoCNNExtended,
    deterministic=False,
    temperature=0.1,
)
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
# bot_A = bot_good
# bot_A_m = "bot_good"
# bot_A = bot_Michael2
# bot_A_m = "bot_M2"
# bot_A = bot_loss
# bot_A_m = "bot_loss"
# bot_A = bot_B02rep
# bot_A_m = "bot_B02rep(5)9800"


bot_A = bot_ME_endgame
bot_A_m = "bot_ME_endgame"

bot_A = bot_loss_BT
bot_A_m = "bot_loss BT"

bot_A = bot_rand
bot_A_m = "bot_random"

bot_A = bot_Aa_replay
bot_A_m = "bot_Aa_replay"

bot_B = bot_Sa_archScan
bot_B_m = "bot_Sa_archScan"

# bot_B = bot_Aa_replay
# bot_B_m = "bot_Aa_replay"

# bot_B = bot_rand
# bot_B_m = "bot_random"
# bot_B = bot_francis
# bot_B_m = "bot_francis"
# bot_B = bot_Michael
# bot_B_m = "bot_Michael"
# bot_B = bot_good2B
# bot_B_m = "bot_GoodE02_WR_base"
# bot_B = bot_Francis_dec
# bot_B_m = "bot_Francis_dec"
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

# pprint(res)
print(f"\n{'='*50}")
print(f"P1: {bot_A_m}  vs  P2: {bot_B_m}  ({N_MATCHES} matches)")
print(
    f"  P1 ({bot_A_m}) wins: {win_rate_p1['Player 1']} ({win_rate_p1['Player 1']/N_MATCHES*100:.1f}%)"
)
print(
    f"  P2 ({bot_B_m}) wins: {win_rate_p1['Player 2']} ({win_rate_p1['Player 2']/N_MATCHES*100:.1f}%)"
)

print(f"\nP1: {bot_B_m}  vs  P2: {bot_A_m}  ({N_MATCHES} matches)")
print(
    f"  P1 ({bot_B_m}) wins: {win_rate_p2['Player 1']} ({win_rate_p2['Player 1']/N_MATCHES*100:.1f}%)"
)
print(
    f"  P2 ({bot_A_m}) wins: {win_rate_p2['Player 2']} ({win_rate_p2['Player 2']/N_MATCHES*100:.1f}%)"
)

total_A = win_rate_p1["Player 1"] + win_rate_p2["Player 2"]
total_B = win_rate_p1["Player 2"] + win_rate_p2["Player 1"]
print(f"\n{'='*50}")
print(
    f"Overall — {bot_A_m}: {total_A}/{N_MATCHES*2} ({total_A/(N_MATCHES*2)*100:.1f}%)  |  {bot_B_m}: {total_B}/{N_MATCHES*2} ({total_B/(N_MATCHES*2)*100:.1f}%)"
)
print(f"{'='*50}")
