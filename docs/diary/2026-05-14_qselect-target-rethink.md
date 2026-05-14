# Q_select target rethink — minimax-oracle labels for the select head

Design note for the next experimental thread. Context:
`Research-status.md` → Current Open Problem; `series-R.md` and `series-S.md`
post-mortems. Status: **proposed**, no code yet.

## The hypothesis

Q_select saturation is **not** a head-level pathology (`series-P.md`), **not**
an FC-bottleneck capacity issue (`series-Q.md`), **not** a gradient-starvation
artefact (`series-R.md`), and only marginally responsive to trunk structure
(`series-S.md` — Sa(1) deepConv produced the largest Q_select Δ on record at
+0.170 but no visible heatmap separation, and `loss_select` floors near 0.24
regardless of how the trunk is shaped).

The next candidate, which every previous experiment has held fixed: **the
select-head targets themselves are noisy.**

Under `DECOUPLED_TARGET_STYLE="td_place_mc_select"` with
`REWARD_FUNCTION="final"`:

- **Q_select target = Monte-Carlo return from the select action**, i.e.
  player-perspective ±1 propagated back through γ=0.99 from the terminal of
  the same match.
- A `select` action happens *every time the player offers a piece*. In a
  full match (≤ 16 plies) that is up to 8 select actions per player, each
  carrying the same ±1 outcome label discounted by horizon to terminal.

### Why Q_select=−1 is correctly labelled

If you gave a piece on the move just before the opponent's winning placement,
that select *caused* the loss in a strong, local sense. The MC return label
of −1 attached to that select is the right label — it matches both the
ground-truth value and the human intuition. This is the well-grounded half
of the target distribution and is reflected in the data: the Q_select
distribution over `Outcome=−1` is concentrated at −1 across every series.

### Why Q_select=+1 is, on average, mislabelled

A select that occurred 6 moves before your eventual win is being labelled
**+1** — but the causal chain between "I gave this piece" and "I won six
moves later" runs through:

1. The opponent's *placement* of that piece (decision 1).
2. My subsequent placement (decision 2).
3. My next select (decision 3).
4. Their next placement + select (4, 5).
5. My placement + select (6, 7).
6. Terminal.

The select action's contribution to the +1 outcome is buried under 6
subsequent decisions that *also* contributed. Many of those select actions
in the +1 population were not decisive piece-gives — they were
non-catastrophic offers whose redemption was carried by later play.

The Q_select target therefore mixes:

- a small population of select actions that genuinely *caused* the win
  (e.g. forcing a piece the opponent had to misplace);
- a large population of select actions that were neutral and got credit
  by accident;
- and, in 2026-era replay buffers, a non-trivial fraction of select
  actions made under exploration noise that are essentially random.

All three are labelled identically. **`loss_select`'s floor near 0.24 is
plausibly the Bayes-optimal SmoothL1 error against this label distribution.**
No reweighting (Ra) and no architecture (Sa) can take it lower because the
information is not in the data.

## The proposal — minimax-oracle distillation on select only

There is a working minimax bot in the project (referenced by the user;
canonical path TBD on implementation — likely `bot/minimax_bot.py` or
similar; locate before coding). For Quarto with `mode_2x2=True`, the
branching factor is small enough that a depth-limited minimax with
alpha-beta is feasible at training time, particularly for the **select
sub-decision** alone: given a (board, available_pieces, offered_piece_set)
state, score every legal offer by its minimax value.

Use those scores as **supervised labels** for `Q_select` while the place
head continues to learn against the RL target:

- **`Q_place` target:** unchanged. TD/Bellman bootstrap as in
  `td_place_mc_select`.
- **`Q_select` target:** `v_minimax(board, offered_piece)` ∈ [−1, +1],
  optionally with a depth-dependent confidence weight.

The combined loss becomes:

```
L = α_place · SmoothL1(Q_place,  TD_target_place)
  + α_select · SmoothL1(Q_select, v_minimax_select)
```

This is supervised distillation of the select head from a clean oracle,
overlaid on RL-trained place. It is the *direct* test of the hypothesis: if
`loss_select`'s floor is a target-noise artefact and the trunk has the
capacity to represent the piece × board interaction (we have weak evidence
for this from Sa(1)), the floor should drop substantially and Q_select Δ
should finally exceed the +0.40 gate.

## What this is NOT

- **Not a full re-targeting of the policy.** Q_place still learns from
  bootstrapped return — we are not building a fully-supervised bot or
  replacing the RL loop.
- **Not the same as Q3 from `series-Q.md`** (handcrafted shaping bonus for
  "offering a piece that cannot complete a line"). Shaping is a hand-coded
  heuristic; minimax is the actual game value at the offered depth.
- **Not contingent on the minimax bot being a *strong* player.** Even a
  depth-2 or depth-3 oracle gives strictly better local credit assignment
  than MC return propagated through 8 moves of exploration noise.

## Design risks / open questions

1. **Cost.** Every `select` step in the replay buffer needs a minimax
   evaluation. With `MATCHES_PER_EPOCH=32` × ≤ 8 selects × 5000 epochs ≈
   1.3M evaluations. At mode_2x2 the per-call cost matters. Mitigation:
   cache by (board_canonical_form, offered_piece) — Quarto board states
   under symmetry are bounded; the same (s, a) recurs frequently.
2. **Depth choice.** Too shallow → the oracle is wrong; too deep → it
   evaluates the policy of an ideal opponent, which is not what the RL
   policy will face during self-play. Suggested start: **depth 4** (two
   plies each side past the offered piece). Document the choice and run a
   depth ablation (2 / 4 / 6) as `Ta_minimax_depth`.
3. **Distribution shift.** The minimax oracle scores states; during early
   training the replay buffer is dominated by *bad* states (random
   placements). The oracle is well-defined there, but the resulting select
   labels will be saturated at ±1 (forced wins / losses several moves
   ahead). Decide whether to filter buffer entries by "non-trivial" state
   or to accept the saturation and let the trunk learn the
   "this-state-is-already-decided" feature explicitly.
4. **Scale match.** Minimax values are in [−1, +1] by construction — same
   range as Q_select tanh output. No rescaling required.
5. **What about Q_place?** Place actions could in principle also be
   minimax-labelled. We deliberately do not propose that: place is the head
   that works. Replacing its target is a regression risk for no diagnostic
   gain. Keep the diagnostic clean: change one head's target, observe
   `loss_select` floor and Q_select Δ.

## Suggested experiment — T-series start

A new code-version letter is mandated by the new target source (rule from
`Research-status.md` → naming convention: "first letter bumps on
code/algorithm changes"). Proposed:

| Letter | Series | Topic |
|---|---|---|
| T | minimax-oracle select target | distill `Q_select` from minimax; `Q_place` unchanged |

- **Ta_minimaxSelect(1)** — depth 4, OA-family substrate (or Sa(3) if the
  confirmation run lands first), N=4, otherwise ME(2) recipe, 5000 epochs.
- Pre-registered gate:
  - `loss_select` final ≤ 0.10 (vs ~0.24 floor across R/S).
  - Q_select winners-minus-losers Δ ≥ +0.40 (the long-standing open-problem
    gate).
  - WR vs `bot_loss-BT` ≥ 70% (non-regression on place-driven WR).
- If Ta(1) hits all three, **the open problem is target noise** and the
  rest of the recipe (architecture, schema, schedule) was a red herring.
  Follow-ups: depth ablation, minimax on the curriculum vs the endgame
  buffer only, cache study.
- If Ta(1) fits `loss_select` cleanly but Q_select Δ stays small, the trunk
  *cannot* represent the piece × board interaction and the failure is
  representational after all. Sb / Sc become the next experiment in that
  branch.
- If WR collapses (place head destabilises), the supervised + RL co-training
  has a balance problem; the fix is α-scheduling on the select term, not
  rejecting the idea.

## Priority

This proposal **outranks** Rb (time-varying loss weights) and Sb / Sc in the
forward queue. Rationale: Ra's `loss_select` floor and Sa's numeric-only
Q_select Δ both point at target quality, and no previous experiment has
touched the target source for the select head.
