from pathlib import Path

import numpy as np
import pytest

from ossp.instance import Instance

INSTANCE_DIR = Path(__file__).resolve().parents[1] / "instances"


@pytest.fixture
def tiny() -> Instance:
    """2 jobs x 2 machines, small enough to reason about by hand."""
    return Instance(np.array([[3, 2], [4, 1]]), name="tiny")


@pytest.fixture
def real() -> Instance:
    return Instance.from_file(INSTANCE_DIR / "44_1.txt")
