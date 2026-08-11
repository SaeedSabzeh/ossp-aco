from ossp.aco import ACOParams
from ossp.benchmark import markdown_table, run_benchmark, solve_once, summarise, write_csv

FAST = ACOParams(n_ants=4, n_generations=2)


def test_solve_once_reports_consistent_numbers(real):
    record = solve_once(real, FAST, seed=0, max_stalls=2)
    assert record.aco_ls <= record.aco
    assert record.aco_ls >= record.lower_bound
    assert record.gap >= 0


def test_local_search_can_be_disabled(real):
    record = solve_once(real, FAST, seed=0, local_search=False)
    assert record.aco_ls == record.aco
    assert record.ls_seconds == 0.0


def test_benchmark_runs_over_a_directory(tmp_path):
    from conftest import INSTANCE_DIR

    records = run_benchmark(
        INSTANCE_DIR, FAST, seeds=(0,), pattern="44_[12].txt", max_stalls=2, verbose=False
    )
    assert len(records) == 2
    assert {r.instance for r in records} == {"44_1", "44_2"}


def test_csv_export_round_trip(tmp_path, real):
    import csv

    records = [solve_once(real, FAST, seed=s, max_stalls=2) for s in (0, 1)]
    path = write_csv(records, tmp_path / "out.csv")
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert "gap" in rows[0]


def test_markdown_table_has_one_row_per_instance(real):
    records = [solve_once(real, FAST, seed=s, max_stalls=2) for s in (0, 1)]
    table = markdown_table(records)
    assert table.count("\n") == 2  # header, separator, one data row
    assert "44_1" in table


def test_summary_mentions_run_count(real):
    records = [solve_once(real, FAST, seed=0, max_stalls=2)]
    assert "1 runs" in summarise(records)
