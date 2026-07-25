.PHONY: install run test lint format docker-up docker-down docker-logs docker-build

install:
	uv sync

run:
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	uv run pytest -v

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