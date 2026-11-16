.PHONY: install lint format check data audit

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

data:  ## Download the Hotel Booking Demand dataset to data/raw/
	uv run python -m hotel_cancellation.data

audit:  ## Print the data facts quoted in the design doc
	uv run python -m hotel_cancellation.audit
