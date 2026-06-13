#!/bin/sh
# docker-entrypoint.sh
#
# Runs before the main CMD in the app container:
#   1. Wait for Postgres to be ready (belt-and-suspenders on top of healthcheck).
#   2. Create the pgvector extension and all tables (idempotent).
#   3. Exec the CMD (uvicorn).
set -e

# ---------------------------------------------------------------------------
# 1. Wait for the database to accept connections
# ---------------------------------------------------------------------------
MAX_RETRIES=30
RETRY_DELAY=2

echo "[entrypoint] Waiting for database..."
i=0
until python - <<'EOF'
import sys
try:
    from app.db import engine
    with engine.connect() as conn:
        conn.execute(__import__("sqlalchemy").text("SELECT 1"))
    sys.exit(0)
except Exception as e:
    print(f"  DB not ready: {e}", file=sys.stderr)
    sys.exit(1)
EOF
do
    i=$((i + 1))
    if [ "$i" -ge "$MAX_RETRIES" ]; then
        echo "[entrypoint] ERROR: database did not become ready after $MAX_RETRIES attempts." >&2
        exit 1
    fi
    echo "[entrypoint] Attempt $i/$MAX_RETRIES — retrying in ${RETRY_DELAY}s..."
    sleep "$RETRY_DELAY"
done

echo "[entrypoint] Database is ready."

# ---------------------------------------------------------------------------
# 2. Initialise schema (creates extension + all tables if they don't exist)
# ---------------------------------------------------------------------------
echo "[entrypoint] Initialising schema..."
python - <<'EOF'
from app.db import init_db
init_db()
print("[entrypoint] Schema initialised.")
EOF

# ---------------------------------------------------------------------------
# 3. Hand off to CMD (uvicorn or whatever was passed)
# ---------------------------------------------------------------------------
echo "[entrypoint] Starting: $*"
exec "$@"
