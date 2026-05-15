# Research diary

Architecture and design notes for the hierarchical-SAE Quarto project.
This folder is the long-form companion to `Research-status.md` (the
experiment ledger) and `readme.md` (the project overview).

## When to add an entry

- A new architecture or model class is introduced.
- A non-obvious training-loop / data-pipeline contract is added (e.g.
  a new `TRANSITION_SCHEMA`, a new auxiliary loss, a new bot inference
  rule).
- A pivot in research direction whose rationale would clutter
  `Research-status.md` if inlined.

Do **not** add entries here for routine experiment results — those
belong in `Research-status.md` as a `## <ExpName> — ...` section.

## File naming

`YYYY-MM-DD_short-slug.md` — date is the day the design landed in
code, slug is 2–4 words describing what the entry is *about*, not
which experiment uses it.

## Index

### Per-series ledgers

| Series | File | Topic |
|---|---|---|
| A | [series-A.md](series-A.md) | Replay / data / fine-tuning (combined_avg baseline) |
| F & G | [series-FG.md](series-FG.md) | Adversarial sign flip, separate_bellman |
| H–K | [series-HIJK.md](series-HIJK.md) | Terminal mask, unbound, final-only, coupled (joint pipeline) |
| L | [series-L.md](series-L.md) | Monte Carlo Q_select target (joint pipeline) |
| M | [series-M.md](series-M.md) | **Decoupled-autoregressive schema. Houses the current champion (ME_endgame(2)).** |
| N | [series-N.md](series-N.md) | Shared-trunk diagnostics: dropout, asymmetric LR, freeze-place, balanced-select |
| O | [series-O.md](series-O.md) | Unified 32-d aux variant of the M trunk (SAE substrate) |
| P | [series-P.md](series-P.md) | Frozen-trunk select-head test (negative result, 2026-05) |
| Q | [series-Q.md](series-Q.md) | Auxiliary legality + no inference mask (failed gate, 2026-05) |
| R | [series-R.md](series-R.md) | Per-head loss reweighting — `loss_select` floor, gradient-starvation rejected (2026-05) |
| S | [series-S.md](series-S.md) | Structural trunk variants — Sa(3) new interpretability champion candidate (2026-05) |
| T | [series-T.md](series-T.md) | Minimax-oracle distillation on the SELECT head (2026-05, in flight) |

### Design notes

| Date | Entry | Topic |
|---|---|---|
| 2026-05-08 | [unified-aux-trunk](2026-05-08_unified-aux-trunk.md) | 32-d phase-stable aux + `unified_autoreg` schema. Substrate of OA series. |
| 2026-05-11 | [qc-no-mask](2026-05-11_qc-no-mask.md) | QC architecture: wider fc1, auxiliary legality head, no inference mask. Substrate of Q-series. |
| 2026-05-14 | [qselect-target-rethink](2026-05-14_qselect-target-rethink.md) | Q_select target may be noisy; proposes minimax-oracle distillation. Substrate of T-series (proposed). |

## How agents should use this folder

Agents (Copilot, Claude Code, subagents) should consult the index
above as part of context-loading, then call `read_file` on the
specific entry whose topic is relevant. Do **not** quote diary content
into other docs — link to the entry by filename instead.
