# Aragora Makefile
# Common development tasks for the Aragora multi-agent debate platform

.PHONY: help install dev test lint format typecheck clean docs serve docker

# Default target
help:
	@echo "Aragora Development Commands"
	@echo "============================"
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install production dependencies"
	@echo "  make dev          Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test         Run all tests"
	@echo "  make test-fast    Run fast tests only (no slow/e2e)"
	@echo "  make test-cov     Run tests with coverage"
	@echo "  make test-watch   Run tests in watch mode"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint         Run linter (ruff)"
	@echo "  make format       Format code (ruff format)"
	@echo "  make typecheck    Run type checker (mypy)"
	@echo "  make check        Run all checks (lint + typecheck)"
	@echo ""
	@echo "Development:"
	@echo "  make serve        Start development server"
	@echo "  make repl         Start interactive debate REPL"
	@echo "  make doctor       Run system health checks"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs         Generate documentation"
	@echo "  make docs-serve   Serve documentation locally"
	@echo ""
	@echo "Docker:"
	@echo "  make docker       Build Docker image"
	@echo "  make docker-run   Run Docker container"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        Remove build artifacts"
	@echo "  make clean-all    Remove all generated files"

# Setup
install:
	pip install -e .

dev:
	pip install -e ".[dev,research,mcp]"
	pre-commit install

# Testing
test:
	pytest tests/ -v --timeout=120

test-fast:
	pytest tests/ -v --timeout=60 -m "not slow and not e2e and not load"

test-cov:
	pytest tests/ -v --timeout=120 --cov=aragora --cov-report=html --cov-report=term

test-watch:
	pytest tests/ -v --timeout=60 -f

# Code Quality
lint:
	ruff check aragora/ tests/

format:
	ruff format aragora/ tests/
	ruff check --fix aragora/ tests/

typecheck:
	mypy aragora/ --ignore-missing-imports

check: lint typecheck

# Development
serve:
	python -m aragora.server.unified_server --port 8080

repl:
	python -m aragora.cli.main repl

doctor:
	python -m aragora.cli.doctor

# Documentation
docs:
	cd docs && mkdocs build

docs-serve:
	cd docs && mkdocs serve

# Docker
docker:
	docker build -t aragora:latest .

docker-run:
	docker run -p 8080:8080 --env-file .env aragora:latest

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-all: clean
	rm -rf .venv/
	rm -rf node_modules/
	rm -rf coverage.xml
	rm -rf .coverage

# Benchmarks
bench:
	pytest tests/ -v --benchmark-only --benchmark-group-by=func

bench-save:
	pytest tests/ -v --benchmark-only --benchmark-save=baseline

# Database
db-migrate:
	python -m aragora.migrations.run

db-reset:
	rm -f ~/.aragora/*.db
	python -m aragora.migrations.run

# Marketplace
marketplace-list:
	python -m aragora.cli.main marketplace list

marketplace-search:
	@read -p "Search query: " query; \
	python -m aragora.cli.main marketplace search "$$query"
