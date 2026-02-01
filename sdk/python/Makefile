.PHONY: help install install-dev test lint format typecheck check clean build publish

# Default target
help:
	@echo "Aragora Python SDK Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install package in production mode"
	@echo "  make install-dev  Install package with dev dependencies"
	@echo ""
	@echo "Quality:"
	@echo "  make test         Run test suite"
	@echo "  make lint         Run ruff linter"
	@echo "  make format       Format code with ruff"
	@echo "  make typecheck    Run mypy type checker"
	@echo "  make check        Run all checks (lint + typecheck + test)"
	@echo ""
	@echo "Build:"
	@echo "  make build        Build distribution packages"
	@echo "  make clean        Remove build artifacts"
	@echo "  make publish      Publish to PyPI (requires credentials)"

# Installation
install:
	pip install .

install-dev:
	pip install -e ".[dev]"

# Testing
test:
	python -m pytest tests/ -v --tb=short

test-quick:
	python -m pytest tests/ -q --tb=line

test-cov:
	python -m pytest tests/ --cov=aragora --cov-report=term-missing --cov-report=html

# Linting and formatting
lint:
	python -m ruff check aragora/ tests/

format:
	python -m ruff format aragora/ tests/
	python -m ruff check --fix aragora/ tests/

# Type checking
typecheck:
	python -m mypy aragora/ --strict

# Combined checks
check: lint typecheck test

# Build
build: clean
	python -m build

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Publishing (use with caution)
publish: build
	python -m twine upload dist/*

publish-test: build
	python -m twine upload --repository testpypi dist/*
