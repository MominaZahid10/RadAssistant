"""
Alembic environment — async-aware.

WHY ASYNC:
The application uses `postgresql+asyncpg://`. Most Alembic setups sidestep
this by adding psycopg2 as a second, synchronous driver purely for
migrations — but that's another dependency to install, and this project has
lost hours to package downloads failing on an unstable connection. Alembic
supports async engines natively via `connection.run_sync()`, so we use the
driver that's already here.

WHY THE URL COMES FROM app.config:
Putting a connection string in alembic.ini creates a second source of truth
that silently drifts from the application's. Migrations must target the same
database the app uses, always.
"""

import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Make `app` importable when alembic runs from the backend/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings          # noqa: E402
from app.core.database import Base           # noqa: E402
from app.models import document               # noqa: E402,F401 — registers the model

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Single source of truth for the connection string.
config.set_main_option("sqlalchemy.url", get_settings().DATABASE_URL)

# Autogenerate compares this against the live database.
# Every model must be imported above or autogenerate will propose DROPPING
# its table — a genuinely destructive failure mode.
target_metadata = Base.metadata


def _include_object(obj, name, type_, reflected, compare_to) -> bool:
    """
    Keep Alembic's attention on our own tables.

    Without this, autogenerate can propose dropping tables created by other
    tooling that happen to share the database.
    """
    if type_ == "table" and name == "alembic_version":
        return False
    return True


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of executing it (`alembic upgrade --sql`)."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=_include_object,
        compare_type=True,           # detect column type changes
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_object=_include_object,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,     # one-shot: no pooling needed
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
