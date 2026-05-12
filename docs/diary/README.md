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

| Date | Entry | Topic |
|---|---|---|
| 2026-05-08 | [unified-aux-trunk](2026-05-08_unified-aux-trunk.md) | 32-d phase-stable aux + `unified_autoreg` schema. Substrate of OA series. |
| 2026-05-11 | [qc-no-mask](2026-05-11_qc-no-mask.md) | QC architecture: wider fc1, auxiliary legality head, no inference mask. Substrate of Q-series. |

## How agents should use this folder

Agents (Copilot, Claude Code, subagents) should consult the index
above as part of context-loading, then call `read_file` on the
specific entry whose topic is relevant. Do **not** quote diary content
into other docs — link to the entry by filename instead.
