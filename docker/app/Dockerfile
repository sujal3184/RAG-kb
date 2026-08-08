# =============================================================================
# Stage 1: "builder" — installs dependencies using uv
# =============================================================================
FROM python:3.11-slim AS builder

# Install uv (a fast Python package manager) by copying its binary directly
# from its official Docker image — this avoids needing curl/pip inside builder.
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /build

# Copy only dependency files first. Docker caches layers — if these files
# don't change, Docker reuses the cached "installed dependencies" layer
# instead of reinstalling everything, which makes rebuilds much faster.
COPY pyproject.toml uv.lock ./

# Install dependencies into a local virtual environment (.venv), without
# installing the project itself yet (--no-install-project), and without
# dev dependencies (pytest, ruff etc. don't belong in production).
RUN uv sync --frozen --no-install-project --no-dev

# Now copy the actual application code and install it too.
COPY app ./app
RUN uv sync --frozen --no-dev

# =============================================================================
# Stage 2: "runtime" — the actual image we ship/run
# =============================================================================
FROM python:3.11-slim AS runtime

# Create a non-root user. Running containers as "root" is a common security
# mistake — if the app is ever compromised, a non-root user limits the damage.
RUN groupadd --system app && useradd --system --gid app app

WORKDIR /code

# Copy only the built virtual environment and app code from the builder
# stage — nothing else (no uv binary, no build cache, no pip cache).
COPY --from=builder /build/.venv /code/.venv
COPY --from=builder /build/app /code/app

# Make sure the venv's Python/packages are used instead of the system Python.
ENV PATH="/code/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

USER app

EXPOSE 8000

# In production, no --reload (that's a dev-only feature, added via override file).
CMD ["sh", "-c", "ls -la /code/.venv/bin/ | head -30 && sleep 60"]