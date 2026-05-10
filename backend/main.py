from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.database import engine, Base
from app.router import router
import app.models

# This lifespan function runs when the app starts and stops
@asynccontextmanager
async def lifespan(app: FastAPI):

    # Creates all database tables from the models if they do not already exist
    Base.metadata.create_all(bind=engine)

    # Hand control back to FastAPI while app is running
    yield

# Create the FastAPI application
app= FastAPI(title='Fixly', lifespan=lifespan)

# Attach all route groups to the main app
app.include_router(router)