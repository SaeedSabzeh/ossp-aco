.PHONY: install test lint fmt solve bench clean

install:
	pip install -e ".[dev]"

test:
	pytest --cov=ossp --cov-report=term-missing

lint:
	ruff check .

fmt:
	ruff check --fix . && ruff format .

solve:
	ossp solve instances/44_1.txt --plot

bench:
	ossp bench instances/ --seeds 0 1 2 --out results/benchmark.csv --markdown

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage dist build
	find . -name __pycache__ -type d -exec rm -rf {} +
