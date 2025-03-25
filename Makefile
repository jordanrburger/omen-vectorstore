.PHONY: setup clean install test dev format lint check help

help:
	@echo "OMEN Platform - Ontology-powered Metadata Engine"
	@echo ""
	@echo "Usage:"
	@echo "  make setup         Setup development environment"
	@echo "  make clean         Clean build artifacts"
	@echo "  make install       Install packages"
	@echo "  make dev           Install development dependencies"
	@echo "  make test          Run all tests"
	@echo "  make format        Format code with black and isort"
	@echo "  make lint          Run linters"
	@echo "  make type-check    Run mypy type checking"
	@echo "  make api           Run API server"
	@echo "  make docs          Build documentation"
	@echo "  make help          Show this help message"

setup: clean
	pip install -e ".[dev,keboola]"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf **/*.egg-info
	rm -rf **/*.pyc
	rm -rf **/__pycache__
	rm -rf **/**/__pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest

format:
	isort .
	black .

lint:
	flake8 .

type-check:
	mypy .

api:
	python -m omen.api.main

docs:
	mkdir -p docs
	echo "Documentation will be built here" > docs/index.md 