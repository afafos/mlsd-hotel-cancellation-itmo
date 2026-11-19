.PHONY: install lint format check test data audit eda baselines

install:  ## Create .venv with all dependencies and install git hooks
	uv sync
	uv run pre-commit install

lint:  ## Check code without changing files: ruff (lint + format) and mypy
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format:  ## Fix lint issues and format code
	uv run ruff check --fix .
	uv run ruff format .

check:  ## Run all pre-commit hooks on all files
	uv run pre-commit run --all-files

test:  ## Check snapshot boundaries, split isolation, ranking metrics and baselines
	uv run python -m unittest discover -s tests -v

data:  ## Download the Hotel Booking Demand dataset to data/raw/
	uv run python -m hotel_cancellation.data

audit:  ## Print the data facts quoted in the design doc
	uv run python -m hotel_cancellation.audit

eda:  ## Rebuild EDA tables and figures in reports/eda/, including feature-level analysis
	uv run python -m hotel_cancellation.eda

baselines:  ## Score B0/B1 on validation windows and 2016 backtests into reports/baselines/
	uv run python -m hotel_cancellation.baselines
