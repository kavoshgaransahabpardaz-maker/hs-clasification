# =============================================================================
# Stage 1: builder — install dependencies into a virtual environment
# =============================================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# All deps use pre-compiled binary wheels (psycopg2-binary, numpy,
# scikit-learn) so no C compiler or libpq-dev is required here.

# Copy only the manifest so pip's dependency resolver runs in a cached layer.
COPY pyproject.toml ./

# Create an isolated venv and install runtime deps (no dev extras).
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir .


# =============================================================================
# Stage 2: runtime — lean image with only what's needed to run the service
# =============================================================================
FROM python:3.12-slim AS runtime

# Security: run as non-root.
RUN groupadd --gid 1001 appgroup && \
    useradd  --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# No apt-get needed:
#   psycopg2-binary bundles libpq inside the wheel.
#   Health check uses Python's built-in urllib (no curl required).

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
EXPOSE 7000

# Liveness probe — uses Python's built-in urllib; no curl needed.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7000/health')" || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7000", "--workers", "2"]
