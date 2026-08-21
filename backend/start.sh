#!/bin/sh
# ══════════════════════════════════════════════════════════════
# RadAssist AI — backend entrypoint
# ══════════════════════════════════════════════════════════════
#
# Waits for PostgreSQL, applies migrations, then starts the API.
#
# ⚠️  WHY THIS EXISTS WHEN docker-compose ALREADY HAS depends_on.
#
#     depends_on:
#       postgres:
#         condition: service_healthy
#
# That only orders `docker compose up`. It does NOT apply when the Docker
# daemon restarts containers itself — which is what happens on a machine
# reboot, because every service is `restart: unless-stopped`. The daemon
# brings them back in whatever order it likes, so the backend can run
# `alembic upgrade head` while Postgres is still in crash recovery:
#
#     asyncpg.exceptions.CannotConnectNowError:
#         the database system is starting up
#
# `alembic upgrade head && uvicorn ...` then fails the whole chain, the
# container exits, and the port mapping disappears — so the symptom the user
# actually sees is "connection closed unexpectedly", several layers away from
# the cause.
#
# Ordering guarantees that depend on the orchestrator are guarantees you lose
# exactly when the machine reboots unattended. The container has to be able to
# start correctly on its own.
# ══════════════════════════════════════════════════════════════

set -e

MAX_WAIT="${DB_WAIT_SECONDS:-90}"
WAITED=0

echo "⏳ Waiting for PostgreSQL to accept connections..."

# Ask the database itself rather than probing the TCP port. Postgres listens
# well before it is ready to serve: during recovery it accepts the connection
# and then rejects the session with "the database system is starting up".
# A port check would report success and we would be back where we started.
until python - <<'PY' 2>/dev/null
import asyncio, os, sys, urllib.parse as up

url = os.environ.get("DATABASE_URL", "")
# postgresql+asyncpg://user:pass@host:port/db  →  asyncpg's own arguments
parsed = up.urlparse(url.replace("postgresql+asyncpg://", "postgresql://"))

async def check() -> None:
    import asyncpg
    conn = await asyncpg.connect(
        user=up.unquote(parsed.username or ""),
        password=up.unquote(parsed.password or ""),
        database=(parsed.path or "/").lstrip("/"),
        host=parsed.hostname or "postgres",
        port=parsed.port or 5432,
        timeout=3,
    )
    await conn.execute("SELECT 1")
    await conn.close()

try:
    asyncio.run(check())
except Exception:
    sys.exit(1)
PY
do
    WAITED=$((WAITED + 2))
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "❌ PostgreSQL was not ready after ${MAX_WAIT}s."
        echo ""
        # ⚠️  RUN IT ONCE MORE WITH STDERR VISIBLE.
        # The loop above sends stderr to /dev/null so a database still in
        # recovery does not print a wall of identical tracebacks. The cost is
        # that EVERY failure looks like "still starting" — including a wrong
        # password, which will never resolve no matter how long you wait.
        # Twenty minutes of watching a counter, and the answer was one line
        # that had been suppressed the whole time.
        echo "   The last connection attempt reported:"
        python - <<'DIAG' 2>&1 | sed 's/^/   /'
import asyncio, os, urllib.parse as up
url = os.environ.get("DATABASE_URL", "")
parsed = up.urlparse(url.replace("postgresql+asyncpg://", "postgresql://"))

async def check():
    import asyncpg
    conn = await asyncpg.connect(
        user=up.unquote(parsed.username or ""),
        password=up.unquote(parsed.password or ""),
        database=(parsed.path or "/").lstrip("/"),
        host=parsed.hostname,
        port=parsed.port or 5432,
    )
    await conn.close()

try:
    asyncio.run(check())
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}")
    # The two that actually happen, and what they mean:
    if "password authentication failed" in str(exc):
        print("")
        print("POSTGRES_PASSWORD does not match what this volume was created")
        print("with. Postgres applies that variable ONLY when initialising an")
        print("empty data directory — changing it later has no effect on an")
        print("existing volume. Either set it to the original value, or")
        print("destroy the volume (docker compose down -v) and start fresh.")
    elif "does not exist" in str(exc):
        print("")
        print("The database named in DATABASE_URL is not the one this volume")
        print("holds. Check POSTGRES_DB.")
DIAG
        echo ""
        echo "   Full server log:  docker compose logs postgres"
        exit 1
    fi
    printf '   still starting (%ss)\n' "$WAITED"
    sleep 2
done

echo "✅ PostgreSQL ready after ${WAITED}s"

# Migrations run BEFORE uvicorn, and a failure still stops the container.
# Starting an app whose schema is wrong produces errors at query time that
# point nowhere near the cause.
echo "🔧 Applying migrations..."
alembic upgrade head

# exec so uvicorn becomes PID 1 and receives SIGTERM directly — without it,
# `docker-compose stop` waits the full 10s grace period on every restart.
echo "🚀 Starting API..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
