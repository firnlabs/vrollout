"""Exports a W&B training history to CSV and plots it as a grid of small multiples.

To generate the plots:
    uv run --group notebook scripts/plot_training_history.py plot \
        docs/reports/2026-11-06-e0-e1-grpo-gsm8k/e1-history.csv \
        -o docs/reports/2026-11-06-e0-e1-grpo-gsm8k/e1-training.png
"""

import csv
import json
import math
import pathlib
from typing import Annotated

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import typer

COLUMNS = (
    "step",
    "reward_mean",
    "format_compliance_rate",
    "zero_advantage_group_frac",
    "entropy_proxy",
    "kl",
    "truncation_rate",
    "response_length_mean",
    "grad_norm",
    "clip_fraction",
    "mismatch_kl",
    "lr",
    "rollout_time_s",
    "train_time_s",
    "weight_sync_time_s",
    "step_time_s",
    "eval_accuracy_strict",
    "eval_accuracy_lenient",
)
STEP_TIME_PHASES = (
    ("rollout", "rollout_time_s"),
    ("train", "train_time_s"),
    ("weight sync", "weight_sync_time_s"),
)

# Reference palette, light mode; validated as a set (CVD ΔE ≥ 9.2). Slot 3 is
# below 3:1 contrast, so every series is also labelled directly.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
RAW = "#b9b8b3"

NUM_STEPS = 200
SMOOTH_WINDOW = 10

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _format_cell(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def _read_columns(path: pathlib.Path) -> dict[str, np.ndarray]:
    """Reads the history CSV into one float array per column, NaN for blanks."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        name: np.array([float(r[name]) if r[name] else math.nan for r in rows])
        for name in rows[0]
    }


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over the last `window` values, ignoring NaNs."""
    out = np.full_like(values, math.nan)
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1) : i + 1]
        chunk = chunk[~np.isnan(chunk)]
        if chunk.size:
            out[i] = chunk.mean()
    return out


def _style(ax: Axes, title: str, unit: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=6)
    ax.set_ylabel(unit, fontsize=9, color=INK_MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xlim(0, NUM_STEPS)


def _plot_smoothed(
    ax: Axes, steps: np.ndarray, values: np.ndarray, title: str, unit: str
) -> None:
    """Plots raw per-step values in grey under their rolling mean."""
    _style(ax, title, unit)
    ax.plot(steps, values, color=RAW, linewidth=1.0)
    ax.plot(steps, _rolling_mean(values, SMOOTH_WINDOW), color=SERIES[0], linewidth=2.0)


def _plot_eval(ax: Axes, steps: np.ndarray, accuracy: np.ndarray) -> None:
    """Plots eval accuracy at the steps where it was measured.

    Labels the first point (the E0 baseline) and the last.
    """
    _style(ax, "Greedy eval accuracy (strict, n = 500)", "accuracy (SE ≈ 0.022)")
    ax.margins(y=0.12)
    measured = ~np.isnan(accuracy)
    steps, accuracy = steps[measured], accuracy[measured]
    ax.plot(
        steps,
        accuracy,
        color=SERIES[0],
        linewidth=2.0,
        marker="o",
        markersize=5,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
    )
    label_style = {"textcoords": "offset points", "fontsize": 8, "color": INK_MUTED}
    ax.annotate(
        f"E0 {accuracy[0]:.3f}",
        (steps[0], accuracy[0]),
        xytext=(8, 2),
        ha="left",
        va="bottom",
        **label_style,
    )
    ax.annotate(
        f"{accuracy[-1]:.3f}",
        (steps[-1], accuracy[-1]),
        xytext=(-2, -8),
        ha="right",
        va="top",
        **label_style,
    )


def _plot_step_time(
    ax: Axes, steps: np.ndarray, phase_times: dict[str, np.ndarray]
) -> None:
    """Stacks the smoothed time of each step phase, labelled with its median."""
    _style(ax, "Step time (10-step mean)", "seconds")
    layers = [_rolling_mean(t, SMOOTH_WINDOW) for t in phase_times.values()]
    ax.stackplot(steps, layers, colors=SERIES, edgecolor=SURFACE, linewidth=1.0)

    medians = np.array([np.nanmedian(layer) for layer in layers])
    centers = np.cumsum(medians) - medians / 2
    for label, median, center in zip(phase_times, medians, centers, strict=True):
        ax.text(
            NUM_STEPS - 4,
            center,
            f"{label} {median:.1f} s",
            ha="right",
            va="center",
            fontsize=8,
            color=INK,
        )


@app.command()
def export(
    history: Annotated[
        pathlib.Path,
        typer.Argument(exists=True, dir_okay=False, help="W&B history JSON export"),
    ],
    output: Annotated[pathlib.Path, typer.Option("--output", "-o")],
) -> None:
    """Writes the per-step metrics used in the report to CSV, one row per step."""
    records = json.loads(history.read_text(encoding="utf-8"))["history"]
    records.sort(key=lambda r: r["_step"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for r in records:
            values = [r["_step"]] + [r.get(c) for c in COLUMNS[1:]]
            writer.writerow(_format_cell(v) for v in values)
    typer.echo(f"wrote {len(records)} rows to {output}")


@app.command()
def plot(
    history_csv: Annotated[pathlib.Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[pathlib.Path, typer.Option("--output", "-o")],
) -> None:
    """Plots reward, eval accuracy, NLL, KL, truncation and step time."""
    history = _read_columns(history_csv)
    steps = history["step"]
    # Step 0 holds only the pre-training eval; training metrics start at step 1.
    is_train = steps >= 1
    train = {name: values[is_train] for name, values in history.items()}

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    (ax_reward, ax_eval, ax_nll), (ax_kl, ax_trunc, ax_time) = axes

    _plot_smoothed(
        ax_reward, train["step"], train["reward_mean"], "Training reward", "mean reward"
    )
    _plot_eval(ax_eval, steps, history["eval_accuracy_strict"])
    _plot_smoothed(
        ax_nll,
        train["step"],
        train["entropy_proxy"],
        "Sampling NLL (entropy proxy)",
        "nats / token",
    )
    _plot_smoothed(ax_kl, train["step"], train["kl"], "KL to reference", "KL")
    _plot_smoothed(
        ax_trunc,
        train["step"],
        train["truncation_rate"] * 100,
        "Truncated rollouts",
        "% of rollouts",
    )
    _plot_step_time(
        ax_time,
        train["step"],
        {label: train[column] for label, column in STEP_TIME_PHASES},
    )

    for ax in axes[1]:
        ax.set_xlabel("training step", fontsize=9, color=INK_MUTED)
    fig.suptitle(
        f"E1: sync GRPO on Qwen2.5-0.5B-Instruct, GSM8K, {NUM_STEPS} steps "
        f"(grey: per step, blue: {SMOOTH_WINDOW}-step mean)",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor=SURFACE)
    typer.echo(f"wrote {output}")


if __name__ == "__main__":
    app()
