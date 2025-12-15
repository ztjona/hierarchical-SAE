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
from bot.human import Quarto_bot as Human_bot
from quartopy import play_games
from bot.CNN_bot import Quarto_bot

human = Human_bot()

# better
f = "CHECKPOINTS//EXP_id03//20250922_1247-EXP_id03_epoch_0009.pt"
# medio malo
# f = "CHECKPOINTS//EXP_id03//20250922_1920-EXP_id03_epoch_0377.pt"


from models.CNN_uncoupled import QuartoCNN as QuartoCNN_uncoupled

_f_loss_BT = "CHECKPOINTS\\LOSS_APPROACHs_1212-2_only_select\\20251212_2206-LOSS_APPROACHs_1212-2_only_select_E_1034.pt"

bot_loss_BT = Quarto_bot(
    model_path=_f_loss_BT,
    model_class=QuartoCNN_uncoupled,
    deterministic=False,
    temperature=0.1,
)
_, win_rate_p1 = play_games(
    matches=1,
    player1=bot_loss_BT,
    player2=human,
    verbose=True,
    save_match=True,
    mode_2x2=True,
)
