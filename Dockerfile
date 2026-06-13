# =============================================================================
# Stage 1: builder — install dependencies into a virtual environment
# =============================================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps needed to compile psycopg2-binary (already binary, but kept for
# any future source builds) and general pip hygiene.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only the files pip needs to resolve dependencies first (layer cache).
COPY pyproject.toml ./

# Create an isolated venv and install runtime deps (no dev extras).
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install --no-cache-dir .


# =============================================================================
# Stage 2: runtime — lean image with only what's needed to run the service
# =============================================================================
FROM python:3.12-slim AS runtime

# Security: run as non-root.
RUN groupadd --gid 1001 appgroup && \
    useradd  --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Runtime system deps (libpq for psycopg2).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the venv from the builder stage.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code.
COPY app/       ./app/
COPY alembic/   ./alembic/
COPY alembic.ini ./alembic.ini

# Copy the entrypoint script.
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Directory for the fitted calibrator (mounted as a volume in production).
RUN mkdir -p /app/data && chown appuser:appgroup /app/data

USER appuser

# Expose the uvicorn port.
EXPOSE 8000

# Liveness probe target (used by docker-compose and orchestrators).
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
