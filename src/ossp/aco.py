"""Ant Colony Optimization for the open-shop scheduling problem."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ossp.instance import Instance

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ACOParams:
    """Every knob in one validated place.

    `elite_reward` is deposited by the all-time-best solution and
    `generation_reward` by the current generation's best. Raising the elite
    share converges faster but risks stagnating on an early local optimum.
    """

    alpha: float = 1.0            # pheromone exponent
    beta: float = 1.0             # heuristic exponent
    rho: float = 0.1              # evaporation rate
    tau0: float = 1.0             # initial pheromone
    n_ants: int = 25
    n_generations: int = 20
    elite_reward: float = 2 / 3
    generation_reward: float = 1 / 3

    def __post_init__(self) -> None:
        if not 0 < self.rho <= 1:
            raise ValueError("rho must be in (0, 1]")
        if self.n_ants < 1 or self.n_generations < 1:
            raise ValueError("n_ants and n_generations must be positive")
        if self.tau0 <= 0:
            raise ValueError("tau0 must be positive")


@dataclass
class ACOResult:
    sequence: np.ndarray
    makespan: int
    history: list[tuple[int, float, int]] = field(default_factory=list)  # (min, mean, max)

    @property
    def best_per_generation(self) -> list[int]:
        return [row[0] for row in self.history]


class ACO:
    """Pheromone-guided construction of operation sequences."""

    def __init__(self, instance: Instance, params: ACOParams | None = None) -> None:
        self.instance = instance
        self.params = params or ACOParams()
        self.n = instance.n_operations
        # Row `n` is the virtual start node every ant departs from.
        self.pheromone = np.full((self.n + 1, self.n), self.params.tau0, dtype=float)
        # Heuristic: prefer short operations. Precomputed once, not per ant.
        durations = np.array([instance.duration_of(o) for o in range(self.n)], dtype=float)
        self.eta = 1.0 / np.maximum(durations, 1e-9)

    # --- construction -----------------------------------------------------
    def _build(self, rng: np.random.Generator) -> np.ndarray:
        remaining = np.ones(self.n, dtype=bool)
        sequence = np.empty(self.n, dtype=int)
        current = self.n  # virtual start

        for position in range(self.n):
            candidates = np.flatnonzero(remaining)
            weights = (
                self.pheromone[current, candidates] ** self.params.alpha
                * self.eta[candidates] ** self.params.beta
            )
            total = weights.sum()
            # Degenerate weights (underflow / overflow) fall back to uniform choice.
            probabilities = weights / total if np.isfinite(total) and total > 0 else None
            choice = rng.choice(candidates, p=probabilities)
            sequence[position] = choice
            remaining[choice] = False
            current = choice

        return sequence

    # --- pheromone --------------------------------------------------------
    def _deposit(self, sequence: np.ndarray, amount: float) -> None:
        current = self.n
        for operation in sequence:
            self.pheromone[current, operation] += amount
            current = operation

    def _update_pheromone(self, best: np.ndarray, generation_best: np.ndarray) -> None:
        self.pheromone *= 1 - self.params.rho
        self._deposit(best, self.params.rho * self.params.elite_reward)
        self._deposit(generation_best, self.params.rho * self.params.generation_reward)

    # --- main loop --------------------------------------------------------
    def run(self, seed: int = 0, progress: bool = False) -> ACOResult:
        rng = np.random.default_rng(seed)
        best_sequence: np.ndarray | None = None
        best_cost = float("inf")
        history: list[tuple[int, float, int]] = []

        for generation in range(1, self.params.n_generations + 1):
            sequences = [self._build(rng) for _ in range(self.params.n_ants)]
            costs = np.array([self.instance.makespan(s) for s in sequences])

            index = int(costs.argmin())
            generation_best = sequences[index]
            if costs[index] < best_cost:
                best_cost = int(costs[index])
                best_sequence = generation_best.copy()
                logger.info("generation %d: new best %d", generation, best_cost)

            self._update_pheromone(best_sequence, generation_best)
            history.append((int(costs.min()), float(costs.mean()), int(costs.max())))

            if progress:
                print(f"  gen {generation:>3}/{self.params.n_generations}  best {best_cost}")

        assert best_sequence is not None
        return ACOResult(best_sequence, int(best_cost), history)
