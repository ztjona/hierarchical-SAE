# Decoupled Autoregressive Design

Status: schema approved and first implementation pass completed on branch `exp/decoupled-autoreg`.

This note defines the proposed transition schema, flag names, and model API for
the decoupled-autoregressive Quarto agent before the full training path is
implemented.

## Why a separate design note

- The existing pipeline stores one joint turn tuple with both `action_place` and
  `action_sel`.
- The proposed approach splits those into separate action-phase transitions while
  keeping one shared model family.
- Freezing names now avoids drifting field names and method contracts across the
  bot, training loop, replay buffer, and plotting code.

## Proposed Flags

### `TRANSITION_SCHEMA`

Options:

- `joint`
- `decoupled_autoreg`

Meaning:

- `joint` keeps the current combined place+select turn tuple.
- `decoupled_autoreg` switches the replay buffer and target computation to one
  action phase per transition.

Default for now: `joint`

### `DECOUPLED_TARGET_STYLE`

Options initially planned:

- `td_place_mc_select`

Meaning:

- Place transitions use a TD/Bellman target.
- Select transitions use Monte Carlo outcome supervision.

Reason:

- It is the lowest-risk first decoupled target because place has the cleanest
  immediate consequences, while select has been the unstable branch.

## Proposed TensorDict Schemas

### Existing joint schema

Keys:

- `state_board`
- `state_piece`
- `action_place`
- `action_sel`
- `reward`
- `done`
- `next_state_board`
- `next_state_piece`
- `outcome`
- `steps_to_terminal`

### Proposed decoupled autoregressive schema

Keys:

- `state_board`
- `state_aux`
- `phase`
- `valid_mask`
- `action`
- `reward`
- `done`
- `next_state_board`
- `next_state_aux`
- `next_phase`
- `next_valid_mask`
- `outcome`
- `steps_to_terminal`

Field semantics:

- `state_board`: board tensor before the action phase.
- `state_aux`: 16-d auxiliary vector.
  - For `phase=place`: one-hot incoming piece.
  - For `phase=select`: available-piece mask.
- `phase`: integer code.
  - `0 = place`
  - `1 = select`
- `valid_mask`: 16-d binary mask for legal actions in the current phase.
- `action`: chosen action index in `[0, 15]`.
- `reward`: reward assigned to this phase transition under the chosen reward
  function.
- `done`: whether the episode ends after this phase.
- `next_state_board`: board tensor after the phase transition.
- `next_state_aux`: auxiliary tensor for the next phase.
- `next_phase`: next phase code when `done=False`.
- `next_valid_mask`: legal-action mask for the next phase.
- `outcome`: final winner/loser label from the acting player's perspective.
- `steps_to_terminal`: number of action phases remaining until terminal.

## Phase Semantics

### Place transition

- Input board: board before placing the current selected piece.
- Input aux: one-hot selected piece.
- Action: placement index.
- Next phase:
  - `select` if non-terminal.
  - terminal if the placement ends the game.

### Select transition

- Input board: board after the current placement.
- Input aux: available-piece mask.
- Action: selected piece index.
- Next phase: opponent place state.

## Proposed Model API

New autoregressive models should implement:

- `predict_phase(x_board, x_aux, phase, TEMPERATURE, DETERMINISTIC)`
- `q_values_phase(x_board, x_aux, phase)`

The model still has two heads internally (`place`, `select`), but the bot and
future training code call it phase-by-phase.

## Proposed Target Rule

Initial target style:

- Place transition:

  `y_place = r + gamma * max_a Q_select(next_state)`

- Select transition:

  `y_select = gamma^k * outcome`

This keeps the first decoupled experiment close to the most stable parts of the
current work: TD for place, Monte Carlo supervision for select.

### Final reward note

In the decoupled schema, `final` reward is attached only to terminal place
transitions. Select transitions always receive immediate reward `0` and learn
through Monte Carlo supervision or future bootstrap targets.

## Immediate Implementation Scope

Implemented in code:

- `bot/CNN_autoreg_bot.py`
- `models/CNN_autoreg.py`
- `TRANSITION_SCHEMA` dispatch in `QuartoRL/RL_functions.py`
- direct decoupled conversion from `move_history` in `QuartoRL/RL_functions.py`
- masked decoupled TD/MC training step in `QuartoRL/RL_functions.py`
- train-time flags in `trainRL.py`

Still intentionally deferred:

- phase-aware end-of-epoch Q-value evaluation for plotting
- decoupled Q-value plots during training (currently skipped)

The first training run can proceed without those plotting pieces.