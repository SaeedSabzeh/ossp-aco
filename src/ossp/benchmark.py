"""Run the solver across instances and seeds, and report the results."""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ossp.aco import ACO, ACOParams
from ossp.instance import Instance
from ossp.local_search import iterated_local_search


@dataclass
class RunRecord:
    instance: str
    size: str
    seed: int
    lower_bound: int
    aco: int
    aco_ls: int
    aco_seconds: float
    ls_seconds: float

    @property
    def gap(self) -> float:
        """Percent above the lower bound — the true optimum sits at or above it."""
        return 100.0 * (self.aco_ls - self.lower_bound) / self.lower_bound


def solve_once(
    instance: Instance,
    params: ACOParams,
    seed: int = 0,
    local_search: bool = True,
    max_stalls: int = 30,
) -> RunRecord:
    started = time.perf_counter()
    aco_result = ACO(instance, params).run(seed=seed)
    aco_seconds = time.perf_counter() - started

    best = aco_result.makespan
    ls_seconds = 0.0
    if local_search:
        started = time.perf_counter()
        ils = iterated_local_search(instance, aco_result.sequence, max_stalls=max_stalls, seed=seed)
        ls_seconds = time.perf_counter() - started
        best = ils.makespan

    return RunRecord(
        instance=instance.name,
        size=f"{instance.n_jobs}x{instance.n_machines}",
        seed=seed,
        lower_bound=instance.lower_bound,
        aco=aco_result.makespan,
        aco_ls=best,
        aco_seconds=round(aco_seconds, 3),
        ls_seconds=round(ls_seconds, 3),
    )


def run_benchmark(
    instance_dir: str | Path,
    params: ACOParams | None = None,
    seeds: tuple[int, ...] = (0,),
    pattern: str = "*.txt",
    max_stalls: int = 30,
    verbose: bool = True,
) -> list[RunRecord]:
    params = params or ACOParams()
    paths = sorted(
        Path(instance_dir).glob(pattern),
        key=lambda p: (len(p.stem.split("_")[0]), p.stem),
    )
    records: list[RunRecord] = []

    for path in paths:
        instance = Instance.from_file(path)
        for seed in seeds:
            record = solve_once(instance, params, seed=seed, max_stalls=max_stalls)
            records.append(record)
            if verbose:
                print(
                    f"{record.instance:>8} seed {seed}  "
                    f"ACO {record.aco:>5}  +LS {record.aco_ls:>5}  "
                    f"lb {record.lower_bound:>5}  gap {record.gap:5.1f}%  "
                    f"{record.aco_seconds + record.ls_seconds:6.2f}s"
                )
    return records


def write_csv(records: list[RunRecord], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(records[0])) + ["gap"])
        writer.writeheader()
        for record in records:
            writer.writerow({**asdict(record), "gap": round(record.gap, 2)})
    return path


def markdown_table(records: list[RunRecord]) -> str:
    """Aggregate over seeds, one row per instance — ready to paste into a README."""
    by_instance: dict[str, list[RunRecord]] = {}
    for record in records:
        by_instance.setdefault(record.instance, []).append(record)

    lines = [
        "| Instance | Size | LB | ACO | ACO + LS | Improvement | Gap to LB | Time |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, runs in by_instance.items():
        best = min(runs, key=lambda r: r.aco_ls)
        aco = min(r.aco for r in runs)
        seconds = sum(r.aco_seconds + r.ls_seconds for r in runs) / len(runs)
        gain = 100.0 * (aco - best.aco_ls) / aco if aco else 0.0
        lines.append(
            f"| {name} | {best.size} | {best.lower_bound} | {aco} | **{best.aco_ls}** | "
            f"{gain:.1f}% | {best.gap:.1f}% | {seconds:.1f}s |"
        )
    return "\n".join(lines)


def summarise(records: list[RunRecord]) -> str:
    gaps = [r.gap for r in records]
    gains = [100.0 * (r.aco - r.aco_ls) / r.aco for r in records if r.aco]
    return (
        f"{len(records)} runs over {len({r.instance for r in records})} instances\n"
        f"mean gap to lower bound : {sum(gaps) / len(gaps):.1f}%\n"
        f"mean gain from local search: {sum(gains) / len(gains):.1f}%\n"
        f"total time: {sum(r.aco_seconds + r.ls_seconds for r in records):.1f}s"
    )
