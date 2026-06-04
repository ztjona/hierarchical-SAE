# Competence-audit REPORT

Diagnostic findings for the `competence_audit` suite. Companion to
[`PLAN.md`](PLAN.md). Each section is a self-contained study; numbers are
DIRECT (from the emitted JSONL under `results/<exp>/`), interpretation is
flagged.

---

## Loss autopsy — why Ve(4) loses to a random opponent (2026-06-04)

**Tool.** [`loss_autopsy.py`](loss_autopsy.py). Game-trajectory counterpart to
the static Tests A/B in [`audit.py`](audit.py): instead of "over sampled
positions, does the argmax avoid losing pieces / take wins?", it plays full
games vs a random opponent and classifies each *actual loss*.

**Why every loss-vs-random is a SELECT-side event.** In Quarto you never place
the opponent's pieces; the opponent can only win by *placing a piece you handed
it* into a completing line. So every loss decomposes at the agent's final give:

- **avoidable** — the given piece had an immediate winning placement *while a
  safe piece was still available in storage*. A pure `Q_select` blunder (the
  residual `1 − safe_piece_recall`).
- **forced** — *every* available piece completes some line; the agent was
  already lost. No third class exists against a random opponent (it executes no
  multi-move forcing lines), so the fatal give is always a hot give — confirmed
  empirically (`anomalous = 0` in every run).

Classification reuses `audit.py`'s rule helpers (`_placing_wins`,
`_piece_is_losing`) verbatim — no rule drift vs Tests A/B.

### Setup

`Ve_oracleAblation(4)0522_DISABLE_NEVER_10k` @ epoch 10000
(`QuartoCNNAutoregUnifiedS4`), **deterministic argmax** agent, `mode_2x2=True`,
seed 1234, 2500 games/direction (5000/opponent). Two opponents: `uniform`
(`bot/random_bot.py`) and `benchmark` (the epoch-0 CNN used as "Random Baseline"
in `tools/benchmark_champion.py` → `champion-results.jsonl`).

### Results [DIRECT]

| Metric | Uniform random | Benchmark (epoch-0 CNN) |
|---|---|---|
| Loss rate (argmax agent) | **4.36%** (218/5000) | **4.12%** (206/5000) |
| — of losses **avoidable** | 32.6% (71) | 35.9% (74) |
| — of losses **forced** | 67.4% (147) | 64.1% (132) |
| — anomalous | 0 | 0 |
| **Avoidable rate** (of games) | **1.42%** | 1.48% |
| **Forced rate** (of games) | **2.94%** | 2.64% |
| Place-side missed-win rate | 6.73% (345/5126) | 6.64% (341/5132) |
| Mean pieces on board at fatal give | 9.2 | 9.2 |

**First-player check** (pooled over both opponents, 5000 games/side; two-prop
z-test):

| Side | Loss | Avoidable | Forced |
|---|---|---|---|
| Agent as P1 | 4.16% | 1.32% | 2.84% |
| Agent as P2 | 4.32% | 1.58% | 2.74% |
| z (P1−P2) | −0.40 | −1.09 | +0.30 |

All |z| < 1.96 → **no significant first-player effect.** The opening give (P1,
empty board) is structurally incapable of being fatal — no piece is hot — and
the dangerous gives all land mid-game (~9 pieces) where the two sides' board
distributions have converged. Loss mechanism is not a turn-order artefact.

### Interpretation [INFERENTIAL — numbers above are DIRECT]

1. **The headline 5.7% was partly the agent's own temperature.** That number
   (`champion-results.jsonl`) used a *stochastic* agent (temp=0.1). The
   deterministic argmax policy loses only ~4.2%. Shipping argmax at play time is
   a free ~1.5 pp.

2. **~⅔ of losses are FORCED, not avoidable.** Only ~1/3 of losses had a safe
   piece at the fatal give. Avoidable-blunder rate is ~1.4% of games; the forced
   rate is ~2.7%. The dominant loss mode is the agent *walking itself into
   all-hot positions*, not fumbling a visible safe-vs-hot choice.

3. **A perfect select head caps the gain at ~1.4 pp.** `safe_piece_recall → 1.0`
   (e.g. a `Q_select` margin/ranking loss) removes only the avoidable third:
   ~4.2% → ~2.8%. The select head is *not* where most of the WR-vs-random loss
   lives.

4. **"Forced" is locally — not globally — irreducible.** It means no safe piece
   *at that give*. The fatal give lands mid-game (~9 pieces), so the agent could
   often have avoided *reaching* the all-hot position 2–4 moves earlier. The
   ~2.7% forced floor is attackable, but only by **lookahead/planning** (deeper
   oracle so the agent steers away from forced positions) and **place-side play**
   — never by anything done at the give.

5. **The place head leaks 6.7% of immediate wins** (game-level Test A). In ~1 of
   15 positions where the agent was handed a winning piece, argmax `Q_place`
   didn't take it. This doesn't lose directly, but it prolongs games and
   manufactures more gives, feeding the forced-position problem. Cheap to fix.

### Implications for the forward queue [INFERENTIAL]

Leverage on WR-vs-random, re-ranked by the data (was: select-margin first):

1. **Place-side win-taking supervision** (cf. PLAN.md → Vf) — fixes the 6.7%
   miss, cheap (place head already has the win-check machinery), and trims
   forced exposure indirectly. Best WR-per-effort now.
2. **Deeper oracle / planning (depth-3)** — the only lever that attacks the
   dominant ~2.7% forced floor; expensive but now *justified* by the data.
3. **`Q_select` margin/ranking loss** — for the avoidable ~1.4%. Still worth
   doing (and improves the interpretability substrate), but no longer the
   headline.

Constraint of record (user, 2026-06-04): **the champion must be a pure learned
policy — no inference-time tactical search.** All fixes land in the weights.

### Reproduce

```
python analysis/competence_audit/loss_autopsy.py \
  --exp 'Ve_oracleAblation(4)0522_DISABLE_NEVER_10k' \
  --n-games 2500 --opponent uniform   --seed 1234
python analysis/competence_audit/loss_autopsy.py \
  --exp 'Ve_oracleAblation(4)0522_DISABLE_NEVER_10k' \
  --n-games 2500 --opponent benchmark --seed 1234
```
Records appended to
`analysis/competence_audit/results/Ve_oracleAblation(4)0522_DISABLE_NEVER_10k/loss_autopsy.jsonl`.
