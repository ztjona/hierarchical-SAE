"""measure.py - Cosine similarity between grad(L_place) and grad(L_select).

Diagnostic motivated by Nexus (arXiv:2604.09258): if the place- and select-head
losses pull the *shared trunk* in incompatible directions (cosine <= 0), the
"distant minima" geometry the paper warns about is present and Q_select head
saturation is plausibly a multi-task-optimization pathology. If cosine is
already high, the collapse is something else (most likely a Bellman / target
issue) and Nexus would not help.

Usage:
    measure.py <checkpoint> [options]
    measure.py --dir <folder> [options]
    measure.py -h

Arguments:
    <checkpoint>         Path to a .pt file (a single policy-net state_dict).

Options:
    --dir <folder>       Sweep all .pt files in this folder, sorted by name.
    --arch <name>        Architecture class. One of:
                         QuartoCNN | QuartoCNN_uncoupled | QuartoCNN_unbound |
                         QuartoCNNAutoreg | QuartoCNNAutoregUnbound.
                         [default: QuartoCNN_uncoupled]
    --schema <s>         joint | decoupled_autoreg. [default: joint]
    --loss-approach <s>  separate_bellman | mc_select.  Only used when
                         schema=joint; decoupled_autoreg always returns
                         per-head tensors. [default: mc_select]
    --matches <int>      Self-play matches to generate the experience pool
                         from. [default: 64]
    --n-last-states <n>  n_last_states for gen_experience. [default: 6]
    --n-batches <int>    Independent batches sampled from the pool, used to
                         estimate cosine variance. [default: 16]
    --batch-size <int>   Rows per batch. [default: 64]
    --gamma <float>      Bellman discount. [default: 0.8]
    --no-mode-2x2        Disable 2x2 winning mode (default is enabled, to
                         match the standard training setup).
    --reward-fn <s>      propagate | final | discount. [default: propagate]
    --device <s>         cpu | cuda | auto. [default: auto]
    --output <path>      Output JSON path. If omitted, derived from the
                         checkpoint name and written under
                         projects/gradient-similarity/results/.
    --plot               Also write a PNG plot next to the JSON. Only useful
                         in --dir mode.
    --seed <int>         Torch manual_seed. [default: 0]

Output (JSON):
    A list of records, one per checkpoint, each with mean / std cosine
    similarity for parameter groups: all / trunk / place_head / select_head,
    plus per-head loss values and gradient norms.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# ── resolve project root (same idiom as projects/supervised-cloning/train.py) ─
_here = Path(__file__).resolve().parent
_root = _here
while not (_root / "bot").is_dir():
    _root = _root.parent
    if _root == _root.parent:
        raise RuntimeError("Could not find project root containing 'bot/'.")
os.chdir(_root)
sys.path.insert(0, str(_root))

import torch
import torch.nn as nn
from docopt import docopt

from bot.CNN_bot import Quarto_bot
from bot.CNN_autoreg_bot import Quarto_bot as Quarto_autoreg_bot
from models.CNN1 import QuartoCNN
from models.CNN_uncoupled import QuartoCNN as QuartoCNN_uncoupled
from models.CNN_unbound import QuartoCNN as QuartoCNN_unbound
from models.CNN_autoreg import QuartoCNNAutoreg, QuartoCNNAutoregUnbound
from QuartoRL import gen_experience, DQN_training_step

ARCH_TABLE = {
    "QuartoCNN": (QuartoCNN, Quarto_bot),
    "QuartoCNN_uncoupled": (QuartoCNN_uncoupled, Quarto_bot),
    "QuartoCNN_unbound": (QuartoCNN_unbound, Quarto_bot),
    "QuartoCNNAutoreg": (QuartoCNNAutoreg, Quarto_autoreg_bot),
    "QuartoCNNAutoregUnbound": (QuartoCNNAutoregUnbound, Quarto_autoreg_bot),
}

# Parameter prefixes for the heads. Anything else is "trunk".
# CNN1 / CNN_uncoupled / CNN_unbound use fc2_board + fc2_piece;
# CNN_autoreg uses fc2_place + fc2_select.
# In CNN1 (coupled), fc2_piece consumes fc2_board's output, so the per-head split
# under-counts cross-head coupling — the trunk cosine is still meaningful.
PLACE_HEAD_PREFIXES = ("fc2_board", "fc2_place")
SELECT_HEAD_PREFIXES = ("fc2_piece", "fc2_select")


def _classify_params(model: nn.Module) -> dict[str, list[int]]:
    """Return parameter-index lists for: all / trunk / place_head / select_head."""
    groups = {"all": [], "trunk": [], "place_head": [], "select_head": []}
    for i, (name, _) in enumerate(model.named_parameters()):
        groups["all"].append(i)
        if name.startswith(PLACE_HEAD_PREFIXES):
            groups["place_head"].append(i)
        elif name.startswith(SELECT_HEAD_PREFIXES):
            groups["select_head"].append(i)
        else:
            groups["trunk"].append(i)
    return groups


def _flat(grads: list[torch.Tensor], idx: list[int]) -> torch.Tensor:
    if not idx:
        return torch.zeros(0)
    return torch.cat([grads[i].flatten() for i in idx])


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    nm = a.norm() * b.norm()
    if nm.item() < 1e-12:
        return float("nan")
    return (a @ b / nm).item()


def _per_head_losses(
    *,
    policy_net,
    target_net,
    exp_batch,
    schema: str,
    loss_approach: str,
    gamma: float,
    loss_fcn,
):
    """Return (L_place, L_select) as scalar tensors with grad attached.

    Skips empty heads (e.g. a batch with no place-phase rows under
    decoupled_autoreg) by returning ``None`` for that side.
    """
    if schema == "joint":
        out = DQN_training_step(
            policy_net=policy_net,
            target_net=target_net,
            GAMMA=gamma,
            exp_batch=exp_batch,
            LOSS_APPROACH=loss_approach,
            TRANSITION_SCHEMA="joint",
        )
        if not (isinstance(out, tuple) and len(out) == 4):
            raise ValueError(
                f"loss_approach={loss_approach!r} did not return per-head tensors. "
                "Use separate_bellman or mc_select."
            )
        q_place, target_place, q_select, target_select = out
    elif schema == "decoupled_autoreg":
        q_place, target_place, q_select, target_select = DQN_training_step(
            policy_net=policy_net,
            target_net=target_net,
            GAMMA=gamma,
            exp_batch=exp_batch,
            TRANSITION_SCHEMA="decoupled_autoreg",
        )
    else:
        raise ValueError(f"Unknown schema {schema!r}")

    L_place = loss_fcn(q_place, target_place) if q_place.numel() > 0 else None
    L_select = loss_fcn(q_select, target_select) if q_select.numel() > 0 else None
    return L_place, L_select


def _compute_grads(loss: torch.Tensor, params, retain_graph: bool):
    grads = torch.autograd.grad(
        loss, params, retain_graph=retain_graph, allow_unused=True
    )
    return [
        g.detach() if g is not None else torch.zeros_like(p)
        for g, p in zip(grads, params)
    ]


def measure_one(
    *,
    checkpoint_path: Path,
    arch_name: str,
    schema: str,
    loss_approach: str,
    matches: int,
    n_last_states: int,
    n_batches: int,
    batch_size: int,
    gamma: float,
    mode_2x2: bool,
    reward_fn: str,
    device: torch.device,
) -> dict:
    if arch_name not in ARCH_TABLE:
        raise ValueError(f"Unknown --arch {arch_name!r}.")
    arch_cls, bot_cls = ARCH_TABLE[arch_name]

    policy_net = arch_cls().to(device)
    target_net = arch_cls().to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    policy_net.load_state_dict(state)
    target_net.load_state_dict(state)
    target_net.eval()

    bot = bot_cls(model=policy_net, deterministic=False, temperature=1.0)
    exp = gen_experience(
        p1_bot=bot,
        p2_bot=bot,
        n_last_states=n_last_states,
        number_of_matches=matches,
        verbose=False,
        PROGRESS_MESSAGE=f"[grad-sim] {checkpoint_path.name}",
        mode_2x2=mode_2x2,
        REWARD_FUNCTION_TYPE=reward_fn,
        TRANSITION_SCHEMA=schema,
    )

    pool_size = exp.shape[0]
    if pool_size < batch_size:
        raise RuntimeError(
            f"Experience pool too small ({pool_size} < batch_size={batch_size}). "
            "Increase --matches or --n-last-states."
        )

    params = [p for p in policy_net.parameters() if p.requires_grad]
    groups = _classify_params(policy_net)
    loss_fcn = nn.SmoothL1Loss()

    cos_records = {g: [] for g in groups}
    norms_place = {g: [] for g in groups}
    norms_select = {g: [] for g in groups}
    L_place_vals: list[float] = []
    L_select_vals: list[float] = []
    skipped = 0

    for _ in range(n_batches):
        idx = torch.randint(0, pool_size, (batch_size,))
        batch = exp[idx]

        policy_net.zero_grad(set_to_none=True)
        L_place, L_select = _per_head_losses(
            policy_net=policy_net,
            target_net=target_net,
            exp_batch=batch,
            schema=schema,
            loss_approach=loss_approach,
            gamma=gamma,
            loss_fcn=loss_fcn,
        )
        if L_place is None or L_select is None:
            skipped += 1
            continue

        L_place_vals.append(L_place.item())
        L_select_vals.append(L_select.item())

        g_place = _compute_grads(L_place, params, retain_graph=True)
        g_select = _compute_grads(L_select, params, retain_graph=False)

        for name, idxs in groups.items():
            gp = _flat(g_place, idxs)
            gs = _flat(g_select, idxs)
            cos_records[name].append(_cosine(gp, gs))
            norms_place[name].append(gp.norm().item() if gp.numel() else 0.0)
            norms_select[name].append(gs.norm().item() if gs.numel() else 0.0)

    def _mean_std(xs: list[float]) -> tuple[float, float]:
        ts = torch.tensor([x for x in xs if not (isinstance(x, float) and x != x)])
        if ts.numel() == 0:
            return float("nan"), float("nan")
        if ts.numel() == 1:
            return ts.item(), 0.0
        return ts.mean().item(), ts.std().item()

    record = {
        "checkpoint": str(checkpoint_path),
        "arch": arch_name,
        "schema": schema,
        "loss_approach": loss_approach if schema == "joint" else None,
        "n_batches": n_batches,
        "n_batches_used": n_batches - skipped,
        "batch_size": batch_size,
        "experience_pool_size": pool_size,
        "L_place_mean": _mean_std(L_place_vals)[0],
        "L_select_mean": _mean_std(L_select_vals)[0],
        "cosine": {},
        "grad_norm_place": {},
        "grad_norm_select": {},
    }
    for name in groups:
        m, s = _mean_std(cos_records[name])
        record["cosine"][name] = {"mean": m, "std": s}
        m_p, s_p = _mean_std(norms_place[name])
        m_s, s_s = _mean_std(norms_select[name])
        record["grad_norm_place"][name] = {"mean": m_p, "std": s_p}
        record["grad_norm_select"][name] = {"mean": m_s, "std": s_s}
    return record


def _json_sanitize(obj):
    """Replace NaN/Inf with None for strict-JSON portability (browser, jq)."""
    import math

    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    return obj


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _checkpoints_from_args(args) -> list[Path]:
    if args["--dir"]:
        folder = Path(args["--dir"])
        if not folder.is_dir():
            raise FileNotFoundError(f"--dir {folder} is not a directory.")
        ckpts = sorted(folder.glob("*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No .pt files in {folder}.")
        return ckpts
    if args["<checkpoint>"]:
        p = Path(args["<checkpoint>"])
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint {p} not found.")
        return [p]
    raise SystemExit("Provide either <checkpoint> or --dir.")


def _resolve_output(args, ckpts: list[Path]) -> Path:
    if args["--output"]:
        return Path(args["--output"])
    out_dir = Path("projects/gradient-similarity/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args["--dir"]:
        stem = Path(args["--dir"]).name or "sweep"
    else:
        stem = ckpts[0].stem
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    return out_dir / f"{stem}_{ts}.json"


def _maybe_plot(records: list[dict], json_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[grad-sim] matplotlib not installed, skipping plot.")
        return

    xs = list(range(len(records)))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for group, color in [
        ("trunk", "C0"),
        ("all", "C1"),
        ("place_head", "C2"),
        ("select_head", "C3"),
    ]:
        means = [r["cosine"][group]["mean"] for r in records]
        stds = [r["cosine"][group]["std"] for r in records]
        axes[0].errorbar(xs, means, yerr=stds, label=group, color=color, capsize=2)
    axes[0].axhline(0, color="k", linewidth=0.5)
    axes[0].set_ylabel("CosSim(∇L_place, ∇L_select)")
    axes[0].set_title(f"Gradient similarity — {json_path.stem}")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(alpha=0.3)

    Lp = [r["L_place_mean"] for r in records]
    Ls = [r["L_select_mean"] for r in records]
    axes[1].plot(xs, Lp, label="L_place", color="C0")
    axes[1].plot(xs, Ls, label="L_select", color="C3")
    axes[1].set_ylabel("loss")
    axes[1].set_xlabel("checkpoint index")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    png_path = json_path.with_suffix(".png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[grad-sim] Plot written to {png_path}")


def main() -> None:
    args = docopt(__doc__)
    torch.manual_seed(int(args["--seed"]))

    ckpts = _checkpoints_from_args(args)
    device = _resolve_device(args["--device"])
    mode_2x2 = not args["--no-mode-2x2"]

    records: list[dict] = []
    for ckpt in ckpts:
        rec = measure_one(
            checkpoint_path=ckpt,
            arch_name=args["--arch"],
            schema=args["--schema"],
            loss_approach=args["--loss-approach"],
            matches=int(args["--matches"]),
            n_last_states=int(args["--n-last-states"]),
            n_batches=int(args["--n-batches"]),
            batch_size=int(args["--batch-size"]),
            gamma=float(args["--gamma"]),
            mode_2x2=mode_2x2,
            reward_fn=args["--reward-fn"],
            device=device,
        )
        records.append(rec)
        c = rec["cosine"]
        print(
            f"[grad-sim] {ckpt.name}  "
            f"trunk={c['trunk']['mean']:+.4f}+/-{c['trunk']['std']:.4f}  "
            f"all={c['all']['mean']:+.4f}+/-{c['all']['std']:.4f}  "
            f"L_place={rec['L_place_mean']:.4f}  "
            f"L_select={rec['L_select_mean']:.4f}  "
            f"used={rec['n_batches_used']}/{rec['n_batches']}"
        )

    json_path = _resolve_output(args, ckpts)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(_json_sanitize(records), f, indent=2)
    print(f"[grad-sim] JSON written to {json_path}")

    if args["--plot"] and len(records) >= 2:
        _maybe_plot(records, json_path)


if __name__ == "__main__":
    main()
