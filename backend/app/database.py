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

# Create the async database engine
engine = create_async_engine(DATABASE_URL, echo=True)

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