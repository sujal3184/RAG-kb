"""Prometheus metrics definitions.

Metrics are defined ONCE here as module-level singletons (Prometheus
client requires this — defining the same metric twice raises an error)
and imported wherever they need to be incremented.
"""

from prometheus_client import Counter, Histogram

# --- HTTP-level metrics (populated by middleware) -------------------------

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests processed",
    labelnames=["method", "endpoint", "status_code"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    labelnames=["method", "endpoint"],
)

# --- RAG pipeline metrics -------------------------------------------------

rag_pipeline_stage_duration_seconds = Histogram(
    "rag_pipeline_stage_duration_seconds",
    "Duration of each RAG pipeline stage in seconds",
    labelnames=["stage"],  # retrieval | rerank | compress | llm
)

rag_queries_total = Counter(
    "rag_queries_total",
    "Total RAG queries processed",
    labelnames=["cache_status"],  # hit | miss
)

# --- Cache metrics ---------------------------------------------------------

cache_operations_total = Counter(
    "cache_operations_total",
    "Total cache operations",
    labelnames=["namespace", "result"],  # result: hit | miss | error
)

# --- Document processing metrics (Celery worker) ---------------------------

documents_processed_total = Counter(
    "documents_processed_total",
    "Total documents processed by background workers",
    labelnames=["status"],  # ready | failed
)

# --- LLM metrics -----------------------------------------------------------

llm_requests_total = Counter(
    "llm_requests_total",
    "Total LLM API requests",
    labelnames=["model", "outcome"],  # outcome: success | fallback | failure
)

llm_tokens_total = Counter(
    "llm_tokens_total",
    "Total tokens consumed by LLM requests",
    labelnames=["model", "token_type"],  # token_type: input | output
)