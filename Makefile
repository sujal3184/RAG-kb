.PHONY: install run test lint format docker-up docker-down docker-logs docker-build \
        migrate migration downgrade

install:
	uv sync

run:
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	uv run pytest -v -m "not docker"

lint:
	uv run ruff check app tests
	uv run mypy app

format:
	uv run ruff check --fix app tests

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f app

# Create a new migration file by comparing models against the current DB.
# Usage: make migration name="add users table"
migration:
	uv run alembic revision --autogenerate -m "$(name)"

# Apply all pending migrations to the database.
migrate:
	uv run alembic upgrade head

# Undo the most recent migration.
downgrade:
	uv run alembic downgrade -1