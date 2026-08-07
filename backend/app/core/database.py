"""
RadAssist AI — Database Connection

WHY ASYNC?
When the backend calls the database, it has to WAIT for the response.
With synchronous code, the entire server freezes while waiting.
With async code, the server can handle OTHER requests while waiting.
This matters a LOT when multiple users are using the system, or when
we're doing slow operations like generating AI reports.

HOW IT WORKS:
1. create_async_engine() — Creates a connection POOL to PostgreSQL.
   A pool keeps several connections open and reuses them, instead of
   opening a new connection for every single database query.
   
2. async_sessionmaker() — Creates "sessions." A session is like a
   conversation with the database: you can make multiple queries
   within one session, and if something fails, you can roll back
   all changes together (a "transaction").

3. get_db() — An async generator that FastAPI uses to give each
   API request its own database session, and properly close it
   when the request is done (even if there's an error).
"""

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
)
from sqlalchemy.orm import DeclarativeBase
from app.config import get_settings

settings = get_settings()

# ── Engine: The connection pool ──────────────────────────────
engine = create_async_engine(
    settings.DATABASE_URL,
    # Log every SQL query to console (helpful for learning/debugging)
    echo=settings.DEBUG,
    # Keep up to 5 connections open, allow bursting to 10
    pool_size=5,
    max_overflow=10,
)

# ── Session Factory ──────────────────────────────────────────
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    # Don't auto-commit — we control when data is saved
    expire_on_commit=False,
)


# ── Base Model Class ─────────────────────────────────────────
class Base(DeclarativeBase):
    """
    Every database table model will inherit from this class.
    
    For example, in Phase 2+ we'll create:
        class Report(Base):
            __tablename__ = "reports"
            id = Column(Integer, primary_key=True)
            ...
    
    SQLAlchemy uses this base class to know which tables to create.
    """
    pass


# ── Dependency: Get a DB session per request ─────────────────
async def get_db():
    """
    FastAPI dependency that provides a database session.
    
    USAGE in an endpoint:
        @router.get("/something")
        async def get_something(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(SomeModel))
            ...
    
    The 'yield' keyword makes this a generator:
    - Code BEFORE yield runs at the START of the request
    - Code AFTER yield runs at the END of the request (cleanup)
    - If an exception occurs, the 'finally' block still runs
    """
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()
