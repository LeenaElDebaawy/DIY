import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.router import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Knitra", lifespan=lifespan)

# Allow the NiceGUI frontend (port 3000) to call the API directly from the browser.
# This is needed for the Whisper transcription endpoint where JS sends audio as FormData.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Serve uploaded files at /static/uploads/<filename>
uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(uploads_dir, exist_ok=True)
app.mount("/static/uploads", StaticFiles(directory=uploads_dir), name="uploads")