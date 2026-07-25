"""Smoke test: verifies all Docker services are reachable.

This is NOT part of the normal fast unit test suite (it needs Docker
running). Run it manually after `make docker-up` with:

    uv run pytest tests/test_docker_health.py -m docker -v
"""

import socket

import pytest

pytestmark = pytest.mark.docker


def _port_is_open(host: str, port: int, timeout: float = 3.0) -> bool:
    """Return True if a TCP connection to host:port succeeds."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def test_postgres_port_reachable() -> None:
    assert _port_is_open("localhost", 5432)


def test_redis_port_reachable() -> None:
    assert _port_is_open("localhost", 6379)


def test_qdrant_port_reachable() -> None:
    assert _port_is_open("localhost", 6333)


def test_app_health_endpoint_reachable() -> None:
    assert _port_is_open("localhost", 8000)