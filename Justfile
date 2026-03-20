@_:
  just --list

_require-uv:
  @uv --version > /dev/null || (echo "Please install uv: https://docs.astral.sh/uv/" && exit 1)

# install all project extras into the local environment
install: _require-uv
  uv sync --inexact --all-extras

# install the dev and lsp extras used for local testing and CI
install-dev: _require-uv
  uv sync --inexact --extra dev --extra lsp

# install only the docs extras used for documentation builds
install-docs: _require-uv
  uv sync --inexact --extra docs

# create or refresh the local virtual environment
venv: install
  @echo 'run `source .venv/bin/activate` to activate virtualenv'

# check code style and potential issues
lint: _require-uv
  uv run --extra dev ruff check src/fixit scripts examples docs/conf.py
  uv run --extra dev ruff format --check src/fixit scripts examples docs/conf.py
  uv run --extra dev python -m fixit lint src/fixit

# format code
format: _require-uv
  uv run --extra dev ruff format src/fixit scripts examples docs/conf.py

# fix automatically fixable linting issues
fix: _require-uv
  uv run --extra dev ruff check --fix src/fixit scripts examples docs/conf.py

# run the pytest suite
test: _require-uv
  uv run --extra dev --extra lsp pytest

# build the package
build: _require-uv
  uv build

# setup or update local dev environment and install pre-commit hooks
sync: install
  uv run --all-extras pre-commit install

# run tests with coverage and show a coverage report
coverage: _require-uv
  uv run --extra dev --extra lsp coverage run -m pytest
  uv run --extra dev coverage report

# build the docs and regenerate the builtins page
docs: _require-uv
  uv run --extra docs python scripts/document_rules.py
  uv run --extra docs sphinx-build -ab html docs html

# check source headers
headers: _require-uv
  uv run --extra dev python scripts/check_copyright.py

# clean build artifacts and caches
clean:
  rm -rf .venv .pytest_cache .ruff_cache build dist html htmlcov .coverage
  find . -type d -name "__pycache__" -exec rm -r {} +

# static type check with pyright
typecheck: _require-uv
  uv run --extra dev pyright

# static type check with pyrefly
pyrefly: _require-uv
  uv run --extra dev pyrefly check -c pyproject.toml

# check code for common misspellings
spell: _require-uv
  uv run --extra dev codespell

# run all quality checks
check: lint coverage typecheck pyrefly spell headers

# run the main local workflow
all: install test lint docs

# list available recipes
help:
  just --list

alias fmt := format
alias cov := coverage
alias distclean := clean
alias html := docs
alias pyright := typecheck
