from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.router import router


# This lifespan function runs when the app starts and stops
@asynccontextmanager
async def lifespan(app: FastAPI):

    # Alembic now manages all database schema changes
    yield


# Create the FastAPI application
app = FastAPI(title="Fixly", lifespan=lifespan)

# Attach all route groups to the main app
app.include_router(router)