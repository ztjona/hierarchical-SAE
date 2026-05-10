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

## Unified-Aux variant

Status: implemented on 2026-05-08 alongside the original decoupled-autoreg
schema. Adds a third `TRANSITION_SCHEMA` option (`unified_autoreg`) without
modifying any existing class, schema, or code path. The motivation, when to
prefer it over the original decoupled schema, and the cross-game implications
are summarized below.

### Why a unified variant

The original decoupled schema makes `state_aux` mean structurally different
things in the two phases (one-hot offered piece during place; available-piece
mask during select). That is sufficient for training, but it pumps two
distinct sub-distributions through the shared trunk. For interpretability work
(see sibling `games-interp/`), this forces phase-conditional analysis of every
hooked layer: a single SAE trained on `fc1` would pool a mixture of two
operating regimes. It also blocks easy cross-game pipelines (Othello and
4×4 tic-tac-toe have no select phase at all), because the per-game adapter
must special-case the two-mode trunk semantics.

The unified variant **decouples the targets** (TD on place, MC on select —
unchanged) from the **input semantics** (now phase-stable). The trunk receives
the same kind of vector at every step, so a single SAE pools cleanly and the
games-interp pipeline contracts to a `(board, aux) → activations → SAE → BSPs`
shape that generalizes across games.

### Aux semantics

The auxiliary input is a 32-d vector concatenating two 16-d blocks:

- offset `[0:16]` — `offered`: one-hot of the piece this player was handed
  for the current turn.
- offset `[16:32]` — `available`: mask of pieces still in the storage pool
  (i.e., neither on the board nor in this player's hand).

Definition of `offered` per phase, from the acting player's perspective:

- **place phase** (about to place the offered piece): `offered =
  pending_piece` (the piece in the player's hand). `available` = storage,
  which already excludes `pending_piece`.
- **select phase** (about to give a piece to opponent): `offered =
  last_placed_piece` — the piece this player placed earlier this turn. For
  the very first select of the game (Player A's opening, no prior placement),
  `offered = 0_16` (zero block). `available` = storage.

Both fields are well-defined at every transition. The `offered` field never
leaks the chosen action: in place phase it identifies the piece being placed
(input to the action, not the action), and in select phase it points to a
piece already on the board (not the candidate set).

### TensorDict schema

`unified_autoreg` reuses the decoupled key set
(`UNIFIED_AUTOREG_TENSORDICT_KEYS == DECOUPLED_AUTOREG_TENSORDICT_KEYS`).
Only `state_aux` and `next_state_aux` change shape: 16-d → 32-d. All other
fields (`phase`, `valid_mask`, `action`, `reward`, `done`, `next_phase`,
`next_valid_mask`, `outcome`, `steps_to_terminal`) carry identical semantics.

### Model API

`models.CNN_autoreg.QuartoCNNAutoregUnified` and its `*Unbound` sibling
implement the same `forward(x_board, x_aux, phase=...)` /
`q_values_phase(...)` / `predict_phase(...)` surface as
`QuartoCNNAutoreg`. The differences are deliberate:

- `aux` input dimension is 32 (asserted in `_shared_trunk`).
- The trunk has **no phase embedding**. `phase` is accepted by the public
  API but only routes which output head is read in `q_values_phase`; the
  trunk forward is phase-agnostic by construction.
- `trunk_in_channels = 18` (16 board + 2 aux channels, after projecting the
  32-d aux to 32 features and reshaping to (B, 2, 4, 4)). Same channel count
  as the original autoreg trunk (16 + 1 aux + 1 phase), so per-layer
  parameter counts at conv1 and downstream are identical.

### Training-step reuse

`DQN_training_step_decoupled_autoreg` is reused unchanged for the unified
schema. The per-phase masked TD/MC target rule is independent of `state_aux`
dimensionality — the model's `q_values_phase` consumes the aux internally,
so target computation does not need to know how wide it is.

### Bot

`bot.CNN_unified_bot.Quarto_bot` mirrors `bot.CNN_autoreg_bot.Quarto_bot` but
constructs the 32-d aux at every forward pass and tracks
`last_placed_piece` between this player's `place_piece(...)` and the
following `select(...)` so the select-phase `offered` block is correct.

### Backwards compatibility

All pre-2026-05-08 code paths are bit-identical:

- `QuartoCNNAutoreg`, `QuartoCNNAutoregUnbound`, `QuartoCNNAutoregLowDropout`
  are unchanged. The shared `_normalize_phase` was extracted to a free
  helper `_normalize_phase_tensor`, and the existing methods now delegate to
  it — a non-behavioral refactor.
- `gen_experience_decoupled_autoreg` and the `decoupled_autoreg` branch of
  `gen_experience` / `DQN_training_step` are unchanged.
- `Quarto_autoreg_bot` is unchanged.
- The `joint` schema and all `joint`-compatible models (`QuartoCNN`,
  `QuartoCNN_uncoupled`, `QuartoCNN_unbound`) are unchanged.
- Existing checkpoints (MA, MB, MC, MD, ME, MF) load and play exactly as
  before — they are saved/loaded by `QuartoCNNAutoreg.from_file(...)` which
  was untouched.