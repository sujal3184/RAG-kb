"""Tests for observability wiring — metrics endpoint and counters."""

import pytest
from httpx import AsyncClient

from app.observability.metrics import cache_operations_total, rag_queries_total


@pytest.mark.asyncio
async def test_metrics_endpoint_is_exposed(client: AsyncClient) -> None:
    """The /metrics endpoint should return Prometheus-format metrics."""
    response = await client.get("/metrics")

    assert response.status_code == 200
    assert "http_requests_total" in response.text


@pytest.mark.asyncio
async def test_request_id_header_is_returned(client: AsyncClient) -> None:
    """Every response should carry an X-Request-ID header for correlation."""
    response = await client.get("/api/v1/health")

    assert "x-request-id" in {k.lower() for k in response.headers}


@pytest.mark.asyncio
async def test_provided_request_id_is_echoed_back(client: AsyncClient) -> None:
    """If a client supplies its own X-Request-ID, it should be preserved."""
    response = await client.get(
        "/api/v1/health", headers={"X-Request-ID": "my-custom-id-123"}
    )

    assert response.headers["x-request-id"] == "my-custom-id-123"


@pytest.mark.asyncio
async def test_http_metrics_are_recorded(client: AsyncClient) -> None:
    """Making a request should increment the HTTP request counter."""
    await client.get("/api/v1/health")

    metrics_response = await client.get("/metrics")

    assert "http_requests_total" in metrics_response.text
    assert "/api/v1/health" in metrics_response.text


def test_cache_metric_labels_work() -> None:
    """Cache counters should accept the expected label combinations."""
    cache_operations_total.labels(namespace="embedding", result="hit").inc()
    cache_operations_total.labels(namespace="rag_response", result="miss").inc()


def test_rag_query_metric_labels_work() -> None:
    rag_queries_total.labels(cache_status="hit").inc()
    rag_queries_total.labels(cache_status="miss").inc()