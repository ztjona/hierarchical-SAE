# -*- coding: utf-8 -*-
"""JSONL results I/O for training runs and offline comparison tools.

A run produces ``<EXPERIMENT_NAME>.summary.jsonl`` next to its pickle. The
file is append-friendly and one-record-per-line so it is robust to crashes
and trivial to scan from `jq`, Python, or an LLM.

Record types
------------

- ``{"type": "checkpoint", "epoch": int, ...}`` written every checkpoint
  cadence with smoothed loss / WR / Q-value diagnostics computed from the
  in-memory training state.
- ``{"type": "final", "exp_name": str, ...}`` written once at the end of
  training (or by the backfill helper) with the headline metrics and the
  full hyperparameter snapshot.

Schema is forward-compatible: readers MUST ignore unknown keys and tolerate
missing keys (older runs).
"""
from __future__ import annotations

import io
import json
import math
import pickle
from os import path
import os
from typing import Any

import numpy as np


SUMMARY_SUFFIX = ".summary.jsonl"


def _to_python(x: Any) -> Any:
    """Recursively convert numpy / torch scalars to plain Python."""
    if x is None:
        return None
    if isinstance(x, (bool, int, str)):
        return x
    if isinstance(x, float):
        return None if math.isnan(x) else x
    if isinstance(x, dict):
        return {str(k): _to_python(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_python(v) for v in x]
    if isinstance(x, np.ndarray):
        return _to_python(x.tolist())
    if isinstance(x, (np.floating, np.integer)):
        val = x.item()
        if isinstance(val, float) and math.isnan(val):
            return None
        return val
    # Fallback: stringify so the line stays valid JSON.
    return str(x)


def _nanmean(arr: Any) -> float | None:
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return None
    a = a[~np.isnan(a)]
    if a.size == 0:
        return None
    return float(a.mean())


def _smoothed_tail(values: list[float] | np.ndarray, window: int) -> float | None:
    if values is None:
        return None
    a = np.asarray(list(values), dtype=float)
    if a.size == 0:
        return None
    return _nanmean(a[-window:])


def compute_wr_trend(
    values: list[float] | np.ndarray,
    smooth_window: int = 100,
    back_half: bool = True,
) -> dict[str, float | bool | None]:
    """Linear-regression slope of a per-epoch WR series on its back half.

    Reports slope in **percentage points per 1000 epochs** plus a 95% CI and
    a two-sided p-value (normal approximation, accurate for the multi-1000
    sample sizes we run at). ``still_rising`` is True iff the lower CI bound
    is strictly positive — i.e. the WR was reliably climbing at the end of
    training.

    Returns None-valued fields when the series is too short to fit (need at
    least ``2 * smooth_window`` epochs in the analysis window).
    """
    none = {
        "slope_pp_per_1000ep": None,
        "ci_low_pp_per_1000ep": None,
        "ci_high_pp_per_1000ep": None,
        "p_value": None,
        "still_rising": None,
        "n_points": 0,
        "window_start_epoch": None,
        "window_end_epoch": None,
    }
    if values is None:
        return none
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return none

    # Rolling-mean smoothing matches the elsewhere-used 100-epoch window.
    if arr.size >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(arr, kernel, mode="valid")
        offset = smooth_window - 1  # smoothed[i] corresponds to epoch i+offset
    else:
        smoothed = arr
        offset = 0

    if back_half:
        start = smoothed.size // 2
        y_full = smoothed[start:]
        start_epoch = start + offset
    else:
        y_full = smoothed
        start_epoch = offset

    if y_full.size < max(20, 2 * smooth_window):
        # Not enough points for a meaningful regression.
        return none

    # Decimate by the smoothing window so adjacent regression points are
    # approximately independent — otherwise OLS underestimates SE on
    # autocorrelated rolling-mean output and `still_rising` over-fires.
    stride = max(1, smooth_window)
    idx = np.arange(0, y_full.size, stride)
    y = y_full[idx]
    x_epochs = idx.astype(float)  # epoch index relative to start of window

    finite = np.isfinite(y)
    if finite.sum() < 10:
        return none
    x = x_epochs[finite]
    y = y[finite]

    # OLS slope.
    x_mean = x.mean()
    y_mean = y.mean()
    sxx = float(np.sum((x - x_mean) ** 2))
    if sxx == 0.0:
        return none
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / sxx)
    intercept = y_mean - slope * x_mean
    y_pred = slope * x + intercept
    residuals = y - y_pred
    n = y.size
    if n <= 2:
        return none
    mse = float(np.sum(residuals ** 2) / (n - 2))
    se_slope = float(np.sqrt(mse / sxx)) if mse > 0 else 0.0

    # Convert decimal-WR-per-epoch → pp per 1000 epochs (×100 for pp, ×1000
    # for per-1000ep).
    SCALE = 100.0 * 1000.0
    slope_pp = slope * SCALE
    se_pp = se_slope * SCALE
    z = 1.96
    ci_low = slope_pp - z * se_pp
    ci_high = slope_pp + z * se_pp

    # Two-sided p-value, normal approximation (t→z for n >> 30).
    if se_pp > 0:
        from math import erf, sqrt

        z_stat = slope_pp / se_pp
        p_value = float(2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z_stat) / sqrt(2.0)))))
    else:
        p_value = 0.0 if slope_pp != 0 else 1.0

    return {
        "slope_pp_per_1000ep": float(slope_pp),
        "ci_low_pp_per_1000ep": float(ci_low),
        "ci_high_pp_per_1000ep": float(ci_high),
        "p_value": p_value,
        "still_rising": bool(ci_low > 0.0),
        "n_points": int(n),
        "window_start_epoch": int(start_epoch),
        "window_end_epoch": int(start_epoch + x[-1]),
    }


def summarize_q_outcome(
    q_arr: Any, outcome_arr: Any, window: int = 100
) -> dict[str, float | None]:
    """Mean Q values conditioned on outcome over the last ``window`` epochs."""
    try:
        q = np.asarray(q_arr, dtype=float)
        oc = np.asarray(outcome_arr, dtype=float)
    except Exception:
        return {"all": None, "win": None, "loss": None}
    if q.ndim != 2 or oc.shape != q.shape or q.shape[0] == 0:
        return {"all": None, "win": None, "loss": None}
    n = q.shape[0]
    sl = slice(max(0, n - window), n)
    q_w = q[sl]
    o_w = oc[sl]
    finite = ~np.isnan(q_w)
    win_mask = (o_w > 0) & finite
    loss_mask = (o_w < 0) & finite
    return {
        "all": float(np.nanmean(q_w)) if finite.any() else None,
        "win": float(q_w[win_mask].mean()) if win_mask.any() else None,
        "loss": float(q_w[loss_mask].mean()) if loss_mask.any() else None,
    }


def build_checkpoint_record(
    epoch: int,
    loss_data: dict,
    grad_norm_data: dict,
    win_rate: dict,
    q_values_history: dict,
    extra: dict | None = None,
    smooth_window: int = 100,
) -> dict:
    """Build a ``type=checkpoint`` record from the in-memory training state."""
    wr_smoothed = {
        str(rival): _smoothed_tail(values, smooth_window)
        for rival, values in (win_rate or {}).items()
    }
    qsel = summarize_q_outcome(
        q_values_history.get("q_select", []),
        q_values_history.get("outcome", []),
        window=smooth_window,
    )
    qplace = summarize_q_outcome(
        q_values_history.get("q_place", []),
        q_values_history.get("outcome", []),
        window=smooth_window,
    )
    rec: dict[str, Any] = {
        "type": "checkpoint",
        "epoch": int(epoch),
        "loss_smoothed": _smoothed_tail(loss_data.get("loss_values", []), smooth_window),
        "loss_place_smoothed": _smoothed_tail(
            loss_data.get("loss_place_values", []), smooth_window
        ),
        "loss_select_smoothed": _smoothed_tail(
            loss_data.get("loss_select_values", []), smooth_window
        ),
        "grad_norm_smoothed": _smoothed_tail(
            grad_norm_data.get("grad_norm_values", []), smooth_window
        ),
        "grad_norm_fc2_place_smoothed": _smoothed_tail(
            grad_norm_data.get("grad_norm_fc2_place", []), smooth_window
        ),
        "grad_norm_fc2_select_smoothed": _smoothed_tail(
            grad_norm_data.get("grad_norm_fc2_select", []), smooth_window
        ),
        "wr_smoothed": wr_smoothed,
        "q_select_winners": qsel["win"],
        "q_select_losers": qsel["loss"],
        "q_place_winners": qplace["win"],
        "q_place_losers": qplace["loss"],
    }
    if extra:
        rec.update(extra)
    return _to_python(rec)


def build_final_record(
    exp_name: str,
    epochs: int,
    config: dict,
    loss_data: dict,
    grad_norm_data: dict,
    win_rate: dict,
    q_values_history: dict,
    gates: dict | None = None,
    extra: dict | None = None,
    final_window_frac: float = 0.10,
    smooth_window_final: int = 100,
) -> dict:
    """Build a ``type=final`` record. ``final_window_frac`` controls the
    fraction of trailing epochs used for ``wr_final`` (default 10%)."""
    wr_final: dict[str, float | None] = {}
    wr_peak: dict[str, float | None] = {}
    wr_trend: dict[str, dict[str, float | bool | None]] = {}
    for rival, values in (win_rate or {}).items():
        arr = np.asarray(list(values), dtype=float)
        if arr.size == 0:
            wr_final[str(rival)] = None
            wr_peak[str(rival)] = None
            wr_trend[str(rival)] = compute_wr_trend([], smooth_window=smooth_window_final)
            continue
        tail = max(1, int(arr.size * final_window_frac))
        wr_final[str(rival)] = float(arr[-tail:].mean())
        # Smoothed peak: rolling-mean(window=smooth_window_final).max
        if arr.size >= smooth_window_final:
            kernel = np.ones(smooth_window_final) / smooth_window_final
            smoothed = np.convolve(arr, kernel, mode="valid")
        else:
            smoothed = arr
        wr_peak[str(rival)] = float(smoothed.max())
        wr_trend[str(rival)] = compute_wr_trend(
            arr, smooth_window=smooth_window_final, back_half=True
        )

    qsel = summarize_q_outcome(
        q_values_history.get("q_select", []),
        q_values_history.get("outcome", []),
        window=smooth_window_final,
    )
    qplace = summarize_q_outcome(
        q_values_history.get("q_place", []),
        q_values_history.get("outcome", []),
        window=smooth_window_final,
    )

    rec: dict[str, Any] = {
        "type": "final",
        "exp_name": exp_name,
        "epochs": int(epochs),
        "loss_final": _smoothed_tail(
            loss_data.get("loss_values", []), smooth_window_final
        ),
        "loss_place_final": _smoothed_tail(
            loss_data.get("loss_place_values", []), smooth_window_final
        ),
        "loss_select_final": _smoothed_tail(
            loss_data.get("loss_select_values", []), smooth_window_final
        ),
        "grad_norm_fc2_place_final": _smoothed_tail(
            grad_norm_data.get("grad_norm_fc2_place", []), smooth_window_final
        ),
        "grad_norm_fc2_select_final": _smoothed_tail(
            grad_norm_data.get("grad_norm_fc2_select", []), smooth_window_final
        ),
        "wr_final": wr_final,
        "wr_peak": wr_peak,
        "wr_trend": wr_trend,
        "q_select_winners": qsel["win"],
        "q_select_losers": qsel["loss"],
        "q_place_winners": qplace["win"],
        "q_place_losers": qplace["loss"],
        "gates": gates or {},
        "config": config or {},
    }
    if extra:
        rec.update(extra)
    return _to_python(rec)


def append_record(jsonl_path: str, record: dict) -> None:
    """Append a single JSON record to the JSONL file (one record per line)."""
    parent = path.dirname(jsonl_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


def write_records(jsonl_path: str, records: list[dict]) -> None:
    """Overwrite (or create) the JSONL with the given records."""
    parent = path.dirname(jsonl_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


def read_records(jsonl_path: str) -> list[dict]:
    """Read all records from a JSONL file. Returns [] if the file is absent."""
    if not path.exists(jsonl_path):
        return []
    out: list[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{jsonl_path}:{i} invalid JSON: {e}"
                ) from e
    return out


def final_record(jsonl_path: str) -> dict | None:
    """Return the most recent ``type=final`` record from the JSONL, if any."""
    for rec in reversed(read_records(jsonl_path)):
        if rec.get("type") == "final":
            return rec
    return None


# ---------- Pickle compatibility loader (for backfill) ----------


class _CPUUnpickler(pickle.Unpickler):
    """Maps CUDA tensors to CPU when unpickling legacy result files."""

    def find_class(self, module: str, name: str):  # noqa: D401
        if module == "torch.storage" and name == "_load_from_bytes":
            import torch

            return lambda b: torch.load(
                io.BytesIO(b), map_location="cpu", weights_only=False
            )
        return super().find_class(module, name)


def load_pickle_results(pkl_path: str) -> dict:
    """Load the legacy `<exp>.pkl` results file as a dict."""
    with open(pkl_path, "rb") as f:
        return _CPUUnpickler(f).load()
