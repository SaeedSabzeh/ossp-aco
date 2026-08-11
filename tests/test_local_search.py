import numpy as np

from ossp.instance import Instance
from ossp.local_search import descend, iterated_local_search, perturb, swapped


def test_swapped_does_not_mutate_its_input():
    """Regression: an in-place swap corrupts the descent's incumbent.

    If `swapped` mutated and returned the same array, every rejected neighbour
    would remain applied to the incumbent, so the search would drift away from
    its own best solution while still reporting the old cost.
    """
    original = np.array([1, 2, 3, 4])
    result = swapped(original, 0, 3)
    assert result is not original
    assert original.tolist() == [1, 2, 3, 4]
    assert result.tolist() == [4, 2, 3, 1]


def test_swapped_with_equal_indices_is_identity():
    assert swapped(np.array([1, 2, 3]), 1, 1).tolist() == [1, 2, 3]


def test_descent_reported_cost_matches_returned_solution(real):
    """The headline invariant: the number you report is the schedule you return."""
    rng = np.random.default_rng(0)
    for _ in range(10):
        start = rng.permutation(real.n_operations)
        sequence, cost = descend(real, start)
        assert real.makespan(sequence) == cost


def test_descent_never_worsens_the_starting_solution(real):
    rng = np.random.default_rng(1)
    for _ in range(10):
        start = rng.permutation(real.n_operations)
        _, cost = descend(real, start)
        assert cost <= real.makespan(start)


def test_descent_reaches_a_swap_local_optimum(real):
    rng = np.random.default_rng(2)
    sequence, cost = descend(real, rng.permutation(real.n_operations))
    n = len(sequence)
    for i in range(n - 1):
        for j in range(i + 1, n):
            assert real.makespan(swapped(sequence, i, j)) >= cost


def test_descent_preserves_feasibility(real):
    sequence, _ = descend(real, np.random.default_rng(3).permutation(real.n_operations))
    assert real.is_feasible(sequence)


def test_ils_reported_cost_matches_returned_solution(real):
    result = iterated_local_search(real, range(real.n_operations), max_stalls=5, seed=0)
    assert real.makespan(result.sequence) == result.makespan
    assert real.is_feasible(result.sequence)


def test_ils_is_at_least_as_good_as_plain_descent(real):
    start = np.random.default_rng(4).permutation(real.n_operations)
    _, descent_cost = descend(real, start)
    ils = iterated_local_search(real, start, max_stalls=10, seed=4)
    assert ils.makespan <= descent_cost


def test_ils_is_reproducible_given_a_seed(real):
    a = iterated_local_search(real, range(real.n_operations), max_stalls=5, seed=7)
    b = iterated_local_search(real, range(real.n_operations), max_stalls=5, seed=7)
    assert a.makespan == b.makespan
    assert a.sequence.tolist() == b.sequence.tolist()


def test_perturb_preserves_the_permutation(real):
    rng = np.random.default_rng(5)
    original = np.arange(real.n_operations)
    kicked = perturb(original, 4, rng)
    assert sorted(kicked.tolist()) == list(range(real.n_operations))
    assert original.tolist() == list(range(real.n_operations))  # input untouched


def test_descent_finds_the_optimum_on_a_trivial_instance():
    instance = Instance(np.array([[1, 100], [100, 1]]), name="trivial")
    _, cost = descend(instance, [0, 1, 2, 3])
    assert cost == instance.lower_bound == 101
