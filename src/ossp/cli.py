"""Command line interface.

    ossp solve instances/44_1.txt --plot
    ossp bench instances/ --seeds 0 1 2 --out results/benchmark.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ossp.aco import ACOParams
from ossp.benchmark import markdown_table, run_benchmark, solve_once, summarise, write_csv
from ossp.instance import Instance


def add_params(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("ACO parameters")
    group.add_argument("--alpha", type=float, default=1.0, help="pheromone exponent")
    group.add_argument("--beta", type=float, default=1.0, help="heuristic exponent")
    group.add_argument("--rho", type=float, default=0.1, help="evaporation rate")
    group.add_argument("--tau0", type=float, default=1.0, help="initial pheromone")
    group.add_argument("--ants", type=int, default=25, help="ants per generation")
    group.add_argument("--generations", type=int, default=20, help="generations")
    group.add_argument("--elite-reward", type=float, default=2 / 3)
    group.add_argument("--generation-reward", type=float, default=1 / 3)
    group.add_argument("--max-stalls", type=int, default=30, help="ILS non-improving restarts")


def params_from(args: argparse.Namespace) -> ACOParams:
    return ACOParams(
        alpha=args.alpha,
        beta=args.beta,
        rho=args.rho,
        tau0=args.tau0,
        n_ants=args.ants,
        n_generations=args.generations,
        elite_reward=args.elite_reward,
        generation_reward=args.generation_reward,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ossp", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    solve = sub.add_parser("solve", help="solve a single instance")
    solve.add_argument("instance", type=Path)
    solve.add_argument("--seed", type=int, default=0)
    solve.add_argument("--no-local-search", action="store_true")
    solve.add_argument("--plot", action="store_true", help="write Gantt and convergence charts")
    solve.add_argument("--out", type=Path, default=Path("results"))
    add_params(solve)

    bench = sub.add_parser("bench", help="run across a directory of instances")
    bench.add_argument("directory", type=Path)
    bench.add_argument("--seeds", type=int, nargs="+", default=[0])
    bench.add_argument("--pattern", default="*.txt")
    bench.add_argument("--out", type=Path, help="write results CSV here")
    bench.add_argument("--markdown", action="store_true", help="print a README-ready table")
    add_params(bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.command == "solve":
        if not args.instance.exists():
            print(f"error: {args.instance} not found", file=sys.stderr)
            return 2
        instance = Instance.from_file(args.instance)
        print(instance)

        record = solve_once(
            instance,
            params_from(args),
            seed=args.seed,
            local_search=not args.no_local_search,
            max_stalls=args.max_stalls,
        )
        print(f"\nACO          {record.aco}")
        print(f"ACO + LS     {record.aco_ls}")
        print(f"lower bound  {record.lower_bound}   (gap {record.gap:.1f}%)")
        print(f"time         {record.aco_seconds + record.ls_seconds:.2f}s")

        if args.plot:
            from ossp.aco import ACO
            from ossp.local_search import iterated_local_search
            from ossp.plotting import plot_convergence, plot_gantt

            result = ACO(instance, params_from(args)).run(seed=args.seed)
            sequence = result.sequence
            if not args.no_local_search:
                sequence = iterated_local_search(
                    instance, sequence, max_stalls=args.max_stalls, seed=args.seed
                ).sequence
            _, schedule = instance.decode(sequence)
            gantt = args.out / f"{instance.name}_gantt.png"
            curve = args.out / f"{instance.name}_convergence.png"
            plot_gantt(instance, schedule, gantt)
            plot_convergence(result.history, instance.lower_bound, curve)
            print(f"\nwrote {gantt}\nwrote {curve}")
        return 0

    records = run_benchmark(
        args.directory,
        params_from(args),
        seeds=tuple(args.seeds),
        pattern=args.pattern,
        max_stalls=args.max_stalls,
    )
    if not records:
        print("error: no instances matched", file=sys.stderr)
        return 2

    print("\n" + summarise(records))
    if args.markdown:
        print("\n" + markdown_table(records))
    if args.out:
        print(f"\nwrote {write_csv(records, args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
