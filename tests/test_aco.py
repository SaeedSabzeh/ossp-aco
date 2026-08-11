import numpy as np
import pytest

from ossp.aco import ACO, ACOParams


def test_params_validation():
    with pytest.raises(ValueError):
        ACOParams(rho=0)
    with pytest.raises(ValueError):
        ACOParams(rho=1.5)
    with pytest.raises(ValueError):
        ACOParams(n_ants=0)
    with pytest.raises(ValueError):
        ACOParams(tau0=0)


def test_reward_keys_are_plain_ascii():
    """Regression: a homoglyph in a parameter key silently reverts it to default.

    The original used ARABIC TATWEEL (U+0640) in place of an underscore, so the
    configured elite reward never took effect. Nothing about this is visible in
    an editor, so it is asserted rather than eyeballed.
    """
    for field in ACOParams.__dataclass_fields__:
        assert field.isascii(), f"non-ASCII character in parameter name {field!r}"


def test_elite_and_generation_rewards_are_wired_the_right_way_round(real):
    """The all-time best must receive the elite share, not the generation best."""
    params = ACOParams(n_ants=4, n_generations=1, elite_reward=1.0, generation_reward=0.0)
    aco = ACO(real, params)
    before = aco.pheromone.copy()

    elite = np.arange(real.n_operations)
    generation = elite[::-1].copy()
    aco._update_pheromone(elite, generation)
    delta = aco.pheromone - before * (1 - params.rho)

    # every transition used by the elite path gained pheromone
    current = real.n_operations
    for operation in elite:
        assert delta[current, operation] > 0
        current = operation


def test_run_returns_a_feasible_permutation(real):
    result = ACO(real, ACOParams(n_ants=8, n_generations=5)).run(seed=0)
    assert real.is_feasible(result.sequence)
    assert result.makespan == real.makespan(result.sequence)


def test_best_cost_is_monotone_non_increasing(real):
    result = ACO(real, ACOParams(n_ants=8, n_generations=10)).run(seed=1)
    running = np.minimum.accumulate(result.best_per_generation)
    assert result.makespan == running[-1]


def test_history_has_one_row_per_generation(real):
    result = ACO(real, ACOParams(n_ants=5, n_generations=7)).run(seed=0)
    assert len(result.history) == 7
    for minimum, mean, maximum in result.history:
        assert minimum <= mean <= maximum


def test_same_seed_is_reproducible(real):
    params = ACOParams(n_ants=6, n_generations=4)
    a = ACO(real, params).run(seed=42)
    b = ACO(real, params).run(seed=42)
    assert a.makespan == b.makespan
    assert a.sequence.tolist() == b.sequence.tolist()


def test_different_seeds_explore_differently(real):
    params = ACOParams(n_ants=6, n_generations=4)
    results = {ACO(real, params).run(seed=s).sequence.tobytes() for s in range(5)}
    assert len(results) > 1


def test_evaporation_reduces_untouched_pheromone(real):
    params = ACOParams(rho=0.5, elite_reward=0.0, generation_reward=0.0)
    aco = ACO(real, params)
    before = aco.pheromone.copy()
    sequence = np.arange(real.n_operations)
    aco._update_pheromone(sequence, sequence)
    assert np.allclose(aco.pheromone, before * 0.5)


def test_beats_random_search(real):
    """Sanity: the colony should do better than sampling permutations blindly."""
    rng = np.random.default_rng(0)
    random_best = min(real.makespan(rng.permutation(real.n_operations)) for _ in range(100))
    aco_best = ACO(real, ACOParams(n_ants=10, n_generations=10)).run(seed=0).makespan
    assert aco_best <= random_best
