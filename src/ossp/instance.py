"""Problem instance and schedule decoding.

An OSSP instance is an n_jobs x n_machines matrix of processing times. A
candidate solution is a permutation of the n_jobs * n_machines operations,
where operation `o` means "job o // n_machines on machine o % n_machines".

The permutation is decoded greedily into a semi-active schedule: each operation
starts at the earliest time both its machine and its job are free. This is the
standard decoder for permutation-encoded shop problems — it is O(n) in the
number of operations and independent of the magnitude of the processing times.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Operation:
    """One scheduled operation, with the times assigned by the decoder."""

    job: int
    machine: int
    start: int
    duration: int

    @property
    def finish(self) -> int:
        return self.start + self.duration


class Instance:
    """An open-shop instance: processing times plus makespan evaluation."""

    def __init__(self, times: np.ndarray, name: str = "instance") -> None:
        times = np.asarray(times, dtype=int)
        if times.ndim != 2:
            raise ValueError("Processing times must be a 2-D matrix")
        if (times < 0).any():
            raise ValueError("Processing times must be non-negative")
        self.times = times
        self.name = name
        self.n_jobs, self.n_machines = times.shape
        # Flat Python list of durations indexed by operation id. `makespan` is
        # called millions of times inside the descent, and at these sizes plain
        # Python ints beat NumPy scalar indexing by roughly 5x.
        self._durations: list[int] = [int(v) for v in times.reshape(-1)]

    # --- construction -----------------------------------------------------
    @classmethod
    def from_file(cls, path: str | Path) -> Instance:
        """Read whitespace-separated processing times, one row per job."""
        path = Path(path)
        rows = [
            [int(value) for value in line.split()]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError(f"{path} contains no data")
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError(f"{path} has ragged rows")
        return cls(np.array(rows, dtype=int), name=path.stem)

    # --- basic properties -------------------------------------------------
    @property
    def n_operations(self) -> int:
        return self.n_jobs * self.n_machines

    @property
    def lower_bound(self) -> int:
        """Trivial lower bound: the busiest machine or the longest job.

        No schedule can finish before the most loaded machine has run every
        operation assigned to it, nor before the longest job has run on every
        machine. Useful as a sanity floor and for reporting optimality gaps.
        """
        return int(max(self.times.sum(axis=0).max(), self.times.sum(axis=1).max()))

    def job_of(self, operation: int) -> int:
        return operation // self.n_machines

    def machine_of(self, operation: int) -> int:
        return operation % self.n_machines

    def duration_of(self, operation: int) -> int:
        return int(self.times[self.job_of(operation), self.machine_of(operation)])

    # --- decoding ---------------------------------------------------------
    def decode(self, sequence) -> tuple[int, list[Operation]]:
        """Decode an operation permutation into a schedule.

        Each operation is placed at the earliest time its machine and its job
        are both idle. Returns the makespan and the scheduled operations.
        """
        machine_free = np.zeros(self.n_machines, dtype=int)
        job_free = np.zeros(self.n_jobs, dtype=int)
        scheduled: list[Operation] = []

        for operation in sequence:
            job = operation // self.n_machines
            machine = operation % self.n_machines
            duration = int(self.times[job, machine])
            start = int(max(machine_free[machine], job_free[job]))
            finish = start + duration
            machine_free[machine] = finish
            job_free[job] = finish
            scheduled.append(Operation(job, machine, start, duration))

        makespan = int(max(machine_free.max(initial=0), 0))
        return makespan, scheduled

    def makespan(self, sequence) -> int:
        """Makespan only — the hot path during search.

        Deliberately written in plain Python over flat lists: the descent calls
        this on the order of a million times, and NumPy's per-element indexing
        overhead dominates at these problem sizes.
        """
        n_machines = self.n_machines
        durations = self._durations
        machine_free = [0] * n_machines
        job_free = [0] * self.n_jobs

        for operation in sequence:
            job, machine = divmod(int(operation), n_machines)
            start = machine_free[machine]
            if job_free[job] > start:
                start = job_free[job]
            finish = start + durations[operation]
            machine_free[machine] = finish
            job_free[job] = finish

        return max(machine_free)

    def is_feasible(self, sequence) -> bool:
        """A sequence is valid iff it is a permutation of all operations."""
        return sorted(int(o) for o in sequence) == list(range(self.n_operations))

    def __repr__(self) -> str:
        return f"Instance({self.name!r}, {self.n_jobs}x{self.n_machines}, lb={self.lower_bound})"
