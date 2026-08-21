FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /code

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY app ./app
RUN uv sync --frozen --no-dev

FROM python:3.11-slim AS runtime

RUN groupadd --system app && useradd --system --create-home --home-dir /home/app --gid app app

WORKDIR /code

COPY --from=builder /code/.venv /code/.venv
COPY --from=builder /code/app /code/app

ENV PATH="/code/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV HOME=/home/app

RUN mkdir -p /home/app/.cache && chown -R app:app /home/app /code

USER app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]