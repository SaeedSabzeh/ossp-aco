import numpy as np
import pytest

from ossp.instance import Instance


def test_dimensions_and_operation_mapping(tiny):
    assert (tiny.n_jobs, tiny.n_machines, tiny.n_operations) == (2, 2, 4)
    # operation o -> job o // m, machine o % m
    assert [tiny.job_of(o) for o in range(4)] == [0, 0, 1, 1]
    assert [tiny.machine_of(o) for o in range(4)] == [0, 1, 0, 1]
    assert [tiny.duration_of(o) for o in range(4)] == [3, 2, 4, 1]


def test_makespan_by_hand(tiny):
    # job 0 on M0 (3), then job 0 on M1 (2): job 0 is busy until 3, so M1 starts at 3.
    # job 1 on M0 starts at 3 (M0 busy), runs to 7. job 1 on M1 starts at 7.
    assert tiny.makespan([0, 1, 2, 3]) == 8


def test_decode_respects_both_resource_constraints(real):
    sequence = list(range(real.n_operations))
    makespan, schedule = real.decode(sequence)
    assert makespan == real.makespan(sequence)

    # no machine runs two operations at once
    for machine in range(real.n_machines):
        ops = sorted((o for o in schedule if o.machine == machine), key=lambda o: o.start)
        assert all(a.finish <= b.start for a, b in zip(ops, ops[1:], strict=False))

    # no job is on two machines at once
    for job in range(real.n_jobs):
        ops = sorted((o for o in schedule if o.job == job), key=lambda o: o.start)
        assert all(a.finish <= b.start for a, b in zip(ops, ops[1:], strict=False))


def test_every_operation_scheduled_exactly_once(real):
    _, schedule = real.decode(range(real.n_operations))
    assert len(schedule) == real.n_operations
    assert len({(o.job, o.machine) for o in schedule}) == real.n_operations


def test_lower_bound_is_never_exceeded_by_optimum(real):
    rng = np.random.default_rng(0)
    for _ in range(50):
        sequence = rng.permutation(real.n_operations)
        assert real.makespan(sequence) >= real.lower_bound


def test_makespan_is_permutation_dependent(real):
    rng = np.random.default_rng(1)
    costs = {real.makespan(rng.permutation(real.n_operations)) for _ in range(30)}
    assert len(costs) > 1


def test_feasibility_check(real):
    assert real.is_feasible(range(real.n_operations))
    assert not real.is_feasible([0, 0, 1, 2])


def test_all_bundled_instances_load():
    from conftest import INSTANCE_DIR

    paths = sorted(INSTANCE_DIR.glob("*.txt"))
    assert len(paths) == 22
    for path in paths:
        instance = Instance.from_file(path)
        assert instance.n_jobs > 0 and instance.n_machines > 0
        assert instance.lower_bound > 0


def test_ragged_file_is_rejected(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("1 2 3\n4 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ragged"):
        Instance.from_file(path)


def test_negative_times_rejected():
    with pytest.raises(ValueError):
        Instance(np.array([[1, -2], [3, 4]]))
