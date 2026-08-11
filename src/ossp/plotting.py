"""Gantt chart and convergence plots. Imports matplotlib lazily."""

from __future__ import annotations

from pathlib import Path

from ossp.instance import Instance, Operation


def plot_gantt(instance: Instance, schedule: list[Operation], path: str | Path | None = None):
    """One row per machine, bars coloured by job."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 1 + 0.5 * instance.n_machines))
    colours = plt.cm.tab20(range(20))

    for op in schedule:
        if op.duration == 0:
            continue
        ax.barh(
            op.machine,
            op.duration,
            left=op.start,
            height=0.6,
            color=colours[op.job % 20],
            edgecolor="white",
            linewidth=0.5,
        )
        if op.duration > instance.lower_bound * 0.02:
            ax.text(
                op.start + op.duration / 2,
                op.machine,
                str(op.job),
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

    makespan = max((op.finish for op in schedule), default=0)
    ax.axvline(makespan, color="crimson", linestyle="--", linewidth=1)
    ax.text(makespan, -0.9, f" makespan {makespan}", color="crimson", va="bottom", fontsize=9)

    ax.set_yticks(range(instance.n_machines))
    ax.set_yticklabels([f"M{m}" for m in range(instance.n_machines)])
    ax.set_xlabel("Time")
    ax.set_title(f"{instance.name} — {instance.n_jobs}x{instance.n_machines}, makespan {makespan}")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig


def plot_convergence(history, lower_bound: int | None = None, path: str | Path | None = None):
    """Min / mean / max ant cost per generation."""
    import matplotlib.pyplot as plt

    generations = range(1, len(history) + 1)
    minimum = [row[0] for row in history]
    mean = [row[1] for row in history]
    maximum = [row[2] for row in history]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.fill_between(generations, minimum, maximum, alpha=0.15, color="gray", label="ant spread")
    ax.plot(generations, maximum, marker="^", markersize=4, linewidth=1, label="worst ant")
    ax.plot(generations, mean, marker="s", markersize=4, linewidth=1, label="mean")
    ax.plot(generations, minimum, marker="o", markersize=4, linewidth=2, label="best ant")
    if lower_bound:
        ax.axhline(lower_bound, color="crimson", linestyle="--", linewidth=1, label="lower bound")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Makespan")
    ax.set_title("ACO convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig
