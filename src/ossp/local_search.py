"""Local search: best-improvement descent over swaps, wrapped in ILS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ossp.instance import Instance


def swapped(sequence: np.ndarray, i: int, j: int) -> np.ndarray:
    """Return a *copy* with positions i and j exchanged.

    Returning a copy rather than mutating in place is load-bearing: the descent
    below evaluates many candidate neighbours and keeps only improving ones. An
    in-place swap would leave every rejected neighbour applied to the incumbent,
    so the search would wander away from its own best solution while continuing
    to report the best cost it had seen. See tests/test_local_search.py.
    """
    out = sequence.copy()
    if i != j:
        out[i], out[j] = out[j], out[i]
    return out


def descend(instance: Instance, sequence: np.ndarray) -> tuple[np.ndarray, int]:
    """Best-improvement descent: repeatedly apply the best improving swap.

    Each pass evaluates every pair (i, j) and applies the single best improving
    swap found, stopping when no swap improves the makespan — a local optimum
    with respect to the pairwise-swap neighbourhood.
    """
    best = np.asarray(sequence).copy()
    best_cost = instance.makespan(best)
    n = len(best)

    improved = True
    while improved:
        improved = False
        candidate_best, candidate_cost = None, best_cost
        for i in range(n - 1):
            for j in range(i + 1, n):
                neighbour = swapped(best, i, j)
                cost = instance.makespan(neighbour)
                if cost < candidate_cost:
                    candidate_best, candidate_cost = neighbour, cost
        if candidate_best is not None:
            best, best_cost = candidate_best, candidate_cost
            improved = True

    return best, best_cost


def perturb(sequence: np.ndarray, n_swaps: int, rng: np.random.Generator) -> np.ndarray:
    """Kick the solution out of its basin with `n_swaps` random swaps."""
    out = np.asarray(sequence).copy()
    n = len(out)
    for _ in range(max(1, n_swaps)):
        i, j = rng.choice(n, size=2, replace=False)
        out[i], out[j] = out[j], out[i]
    return out


@dataclass
class ILSResult:
    sequence: np.ndarray
    makespan: int
    restarts: int
    improvements: int


def iterated_local_search(
    instance: Instance,
    sequence,
    max_stalls: int = 30,
    n_swaps: int | None = None,
    seed: int = 0,
    verbose: bool = False,
) -> ILSResult:
    """Descend, perturb, descend again; stop after `max_stalls` failures."""
    rng = np.random.default_rng(seed)
    sequence = np.asarray(sequence).copy()
    if n_swaps is None:
        n_swaps = max(len(sequence) // 5, 1)

    best, best_cost = descend(instance, sequence)
    stalls = improvements = 0

    while stalls < max_stalls:
        candidate = perturb(best, n_swaps, rng)
        candidate, cost = descend(instance, candidate)
        if cost < best_cost:
            best, best_cost = candidate, cost
            improvements += 1
            stalls = 0
            if verbose:
                print(f"  ILS improved to {best_cost}")
        else:
            stalls += 1

    return ILSResult(best, best_cost, stalls, improvements)
