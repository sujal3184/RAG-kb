# Deployment Checklist

Work through this before exposing the application to real users.

## Secrets

- [ ] Generate a strong `SECRET_KEY`:
      `python -c "import secrets; print(secrets.token_urlsafe(64))"`
- [ ] Set a strong `POSTGRES_PASSWORD` (not the development default)
- [ ] Set `GRAFANA_ADMIN_PASSWORD`
- [ ] Set `METRICS_AUTH_TOKEN` to a random value
- [ ] Verify no secrets are committed: `git log -p | grep -i "sk-\|password\|secret"`
- [ ] Rotate any key that has ever been pasted into a chat, issue, or log

## Configuration

- [ ] `ENVIRONMENT=production` (disables `/docs`, enables strict CSP)
- [ ] `DEBUG=false`
- [ ] `LOG_JSON=true` (machine-parseable logs for aggregation)
- [ ] `CORS_ALLOWED_ORIGINS` set to your frontend's exact origin, not `*`
- [ ] Review `RATE_LIMIT_*` values against expected traffic

## Network

- [ ] Postgres, Redis, and Qdrant have **no** exposed host ports
      (verify: `docker compose -f docker-compose.yml -f docker-compose.prod.yml config`)
- [ ] `/metrics` restricted at the network layer — the token is a second
      line of defence, not the only one
- [ ] TLS terminated at a reverse proxy in front of the app
- [ ] Reverse proxy overwrites `X-Forwarded-For` (otherwise clients can
      spoof it and bypass rate limiting)

## Known Limitations

These are real and deliberate — understand them before scaling:

- **Rate limiting is per-instance.** Running multiple app replicas
  multiplies the effective limit. Move to a Redis-backed limiter before
  horizontal scaling.
- **Guardrails are pattern-based.** They catch obvious prompt injection
  but are evadable, and produce false positives on legitimate queries
  about security topics. Consider a classifier-based guardrail for
  adversarial traffic.
- **File storage is local disk.** Does not work across multiple app
  instances. Move to S3-compatible object storage before scaling out.
- **BM25 index is rebuilt per query.** Fine at current scale; move to a
  dedicated search engine if knowledge bases grow very large.
- **No OCR.** Scanned PDFs with no text layer extract as empty.

## Operations

- [ ] Automated Postgres backups configured and **restore tested**
- [ ] Qdrant snapshot schedule configured
- [ ] Log aggregation receiving the JSON logs
- [ ] Grafana dashboards built for: request rate, error rate, p95 latency,
      document processing failures, guardrail violations
- [ ] Alerts configured for: error rate spike, document processing failure
      rate, service down

## Verification

- [ ] `/api/v1/health` returns 200
- [ ] `/api/v1/health/ready` returns 200 (database reachable)
- [ ] `/docs` returns 404 (disabled in production)
- [ ] `/metrics` returns 401 without the token
- [ ] Security headers present: `curl -I https://your-domain/api/v1/health`
- [ ] Rate limiting triggers: send 70 rapid requests, expect 429
- [ ] Upload a document end-to-end and confirm it reaches `ready`
- [ ] Ask a question and confirm a cited answer comes back