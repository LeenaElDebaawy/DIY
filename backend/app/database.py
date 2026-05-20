import os
from collections.abc import AsyncGenerator

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
)
from sqlalchemy.orm import DeclarativeBase

# Load environment variables from backend/.env
load_dotenv()

# Read the database URL from the .env file
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://fixly_user:devpassword@localhost:5433/fixly",
)

# Create the async database engine.
# echo=True prints every SQL statement — useful during initial development
# but very noisy once the AI pipeline starts making dozens of queries per run.
# Set DATABASE_ECHO=true in .env to re-enable for debugging.
engine = create_async_engine(
    DATABASE_URL,
    echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
)

# Create async database sessions
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


# Base class for all database models
class Base(DeclarativeBase):
    pass


# This dependency gives route functions a database session
async def get_db() -> AsyncGenerator[AsyncSession, None]:

    # Open a database session
    async with AsyncSessionLocal() as session:

        # Hand the session to the route while it is running
        yield session