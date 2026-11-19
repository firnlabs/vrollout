"""Plots a sync and an async training run against each other for the report.

Writes two figures from the history CSVs that `plot_training_history.py export`
produces:

- Step time: per-step time of both runs, and the median step of each laid out per
  GPU, which shows what async overlaps.
- Wall clock: training reward and eval accuracy against hours of step time, the
  time a run spends in rollout, training and weight sync (eval and checkpoint
  saves excluded, as in the step time).

To generate the plots:
    E2=docs/reports/2026-11-18-e2-async-grpo-gsm8k
    uv run --group notebook scripts/plot_sync_async.py \
        docs/reports/2026-11-06-e0-e1-grpo-gsm8k/e1-history.csv \
        $E2/e2-history.csv \
        --step-time-output $E2/e1-e2-step-time.png \
        --wall-clock-output $E2/e1-e2-wall-clock.png
"""

import math
import pathlib
from typing import Annotated

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from plot_training_history import _read_columns
from plot_training_history import _rolling_mean
from plot_training_history import _save
from plot_training_history import _style
from plot_training_history import INK
from plot_training_history import INK_MUTED
from plot_training_history import SERIES
from plot_training_history import SMOOTH_WINDOW
import typer

REWARD_WINDOW = 25
GREY = "#8f8e89"

app = typer.Typer(add_completion=False)


def _train_rows(cols: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    keep = ~np.isnan(cols["step_time_s"])
    return {name: values[keep] for name, values in cols.items()}


def _plot_lanes(ax: Axes, runs: list[tuple[str, dict[str, np.ndarray]]]) -> None:
    """Draws each run's median step as one bar per GPU.

    Sync runs the phases one after another. Async generates the next batch on the
    rollout GPU while the trainer GPU trains and syncs, so the step is the longer of
    the two lanes.
    """
    rows: list[tuple[str, list[tuple[float, float, str, str]]]] = []
    step_ends: list[tuple[int, float, float]] = []
    for label, train in runs:
        rollout = float(np.median(train["rollout_time_s"]))
        trainer = float(np.median(train["train_time_s"]))
        sync = float(np.median(train["weight_sync_time_s"]))
        step = float(np.median(train["step_time_s"]))
        overlapped = step < rollout + trainer
        start = 0.0 if overlapped else rollout
        phases = np.maximum if overlapped else np.add
        other = float(
            np.median(
                train["step_time_s"]
                - phases(train["rollout_time_s"], train["train_time_s"])
                - train["weight_sync_time_s"]
            )
        )
        step_ends.append((len(rows), step, other))
        rows.append(
            (
                f"{label}: trainer GPU",
                [
                    (start, trainer, SERIES[1], "train"),
                    (start + trainer, sync, SERIES[2], "sync"),
                ],
            )
        )
        rows.append((f"{label}: rollout GPU", [(0.0, rollout, SERIES[0], "rollout")]))
    ax.set_facecolor("none")
    for i, step, other in step_ends:
        y = len(rows) - 1 - i
        ax.annotate(
            f"step {step:.1f} s\n(+{other:.1f} s other)",
            (step, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=INK,
        )
    for i, (_, bars) in enumerate(rows):
        y = len(rows) - 1 - i
        for start, width, color, name in bars:
            ax.barh(y, width, left=start, color=color, height=0.7)
            ax.text(
                start + width / 2,
                y,
                f"{name} {width:.1f} s",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if color != SERIES[2] else INK,
            )
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=9)
    ax.set_xlim(0, max(step for _, step, _ in step_ends) * 1.22)
    ax.set_title("Median step, per GPU", loc="left", fontsize=12, color=INK, pad=6)
    ax.set_xlabel("seconds", fontsize=10, color=INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def _plot_step_time(
    sync: dict[str, np.ndarray],
    async_: dict[str, np.ndarray],
    labels: tuple[str, str],
    output: pathlib.Path,
) -> None:
    fig, (ax_steps, ax_lanes) = plt.subplots(
        2, 1, figsize=(9, 6.2), layout="constrained", height_ratios=(1, 0.9)
    )
    _style(ax_steps, f"Step time ({SMOOTH_WINDOW}-step mean)", "seconds")
    for train, label, color in (
        (sync, labels[0], GREY),
        (async_, labels[1], SERIES[0]),
    ):
        smoothed = _rolling_mean(train["step_time_s"], SMOOTH_WINDOW)
        ax_steps.plot(train["step"], smoothed, color=color, linewidth=2.0)
        ax_steps.annotate(
            f"{label}  {np.median(train['step_time_s']):.1f} s median",
            (train["step"][-1], smoothed[-1]),
            xytext=(-4, 8),
            textcoords="offset points",
            ha="right",
            fontsize=9,
            color=color,
        )
    ax_steps.set_ylim(0, 60)
    _plot_lanes(ax_lanes, [(labels[0], sync), (labels[1], async_)])
    _save(fig, output)


def _plot_wall_clock(
    sync: dict[str, np.ndarray],
    async_: dict[str, np.ndarray],
    labels: tuple[str, str],
    output: pathlib.Path,
) -> None:
    fig, (ax_reward, ax_eval) = plt.subplots(
        1, 2, figsize=(9, 3.6), layout="constrained"
    )
    for cols, label, color in ((sync, labels[0], GREY), (async_, labels[1], SERIES[0])):
        train = _train_rows(cols)
        hours = np.cumsum(train["step_time_s"]) / 3600
        # skip the partial windows at the start, which swing with few steps
        full = slice(REWARD_WINDOW - 1, None)
        ax_reward.plot(
            hours[full],
            _rolling_mean(train["reward_mean"], REWARD_WINDOW)[full],
            color=color,
            linewidth=2.0,
            label=label,
        )
        # step 0's eval runs before any step time accrues
        elapsed = np.concatenate(([0.0], hours))
        evals = ~np.isnan(cols["eval_accuracy_strict"])
        steps = cols["step"][evals].astype(int)
        ax_eval.plot(
            elapsed[steps],
            cols["eval_accuracy_strict"][evals],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=4,
            label=label,
        )
    for ax, title, unit in (
        (ax_reward, f"Training reward ({REWARD_WINDOW}-step mean)", "mean reward"),
        (ax_eval, "Eval accuracy, 500 problems", "strict accuracy"),
    ):
        _style(ax, title, unit)
        ax.set_xlim(0, math.ceil(max(np.nansum(sync["step_time_s"]) / 3600, 1)))
        ax.set_xlabel("hours of step time", fontsize=10, color=INK_MUTED)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED)
    _save(fig, output)


@app.command()
def main(
    sync_csv: Annotated[pathlib.Path, typer.Argument(exists=True)],
    async_csv: Annotated[pathlib.Path, typer.Argument(exists=True)],
    step_time_output: Annotated[pathlib.Path, typer.Option()],
    wall_clock_output: Annotated[pathlib.Path, typer.Option()],
    sync_label: str = "E1 (sync)",
    async_label: str = "E2 (async)",
) -> None:
    """Writes the step-time and wall-clock comparison figures."""
    sync = _read_columns(sync_csv)
    async_ = _read_columns(async_csv)
    labels = (sync_label, async_label)
    _plot_step_time(_train_rows(sync), _train_rows(async_), labels, step_time_output)
    _plot_wall_clock(sync, async_, labels, wall_clock_output)


if __name__ == "__main__":
    app()
