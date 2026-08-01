"""OpenTelemetry tracing setup.

Provides distributed tracing so a single request's journey through the
app (HTTP -> retrieval -> rerank -> LLM -> DB) shows up as a timeline of
nested spans, making it obvious which stage is slow.

Fail-open: if the OTLP collector is unreachable, tracing silently
degrades (spans are dropped) rather than breaking the application.
"""

import logging

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.config.settings import Settings

logger = logging.getLogger(__name__)


def configure_tracing(settings: Settings, app=None) -> None:
    """Set up OpenTelemetry tracing and auto-instrumentation.

    Args:
        settings: app settings controlling whether tracing is enabled and
            where to export spans.
        app: the FastAPI app instance, so HTTP requests are auto-traced.
    """
    if not settings.OTEL_ENABLED:
        logger.info("OpenTelemetry tracing is disabled")
        return

    try:
        resource = Resource.create({"service.name": settings.OTEL_SERVICE_NAME})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT, insecure=True)
            )
        )
        trace.set_tracer_provider(provider)

        # Auto-instrument common libraries — gives us HTTP, DB, and outbound
        # HTTP-call spans for free, without touching any application code.
        if app is not None:
            FastAPIInstrumentor.instrument_app(app)
        SQLAlchemyInstrumentor().instrument()
        HTTPXClientInstrumentor().instrument()

        logger.info(
            "OpenTelemetry tracing configured",
            extra={"endpoint": settings.OTEL_EXPORTER_OTLP_ENDPOINT},
        )
    except Exception as exc:
        logger.warning(
            "Failed to configure OpenTelemetry tracing, continuing without it",
            extra={"error": str(exc)},
        )


def get_tracer(name: str) -> trace.Tracer:
    """Get a tracer for creating manual spans in application code.

    Args:
        name: usually __name__ of the calling module.
    """
    return trace.get_tracer(name)