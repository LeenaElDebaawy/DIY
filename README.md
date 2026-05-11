# Fixly

Fixly is a Python based repair guidance web app.

## Project Status

The project has currently completed the following phases:

### Phase 1 — Foundation
- FastAPI backend created
- NiceGUI frontend created
- backend and frontend can run locally

### Phase 2 — Router Structure
- backend routes moved into a cleaner `app/api` structure
- central router added through `app/router.py`

### Phase 3 — Initial Database Scaffold
- SQLAlchemy added
- initial database model scaffold created
- `/health` and `/health/db` endpoints added
- backend and frontend connection confirmed

### Phase 4 — PostgreSQL and Alembic Setup
- PostgreSQL moved into Docker
- backend database connection upgraded from SQLite to async PostgreSQL
- Alembic initialized for migrations
- full schema models created
- first migration generated and applied successfully

## Current Stack

### Backend
- FastAPI
- SQLAlchemy 2.0
- Async PostgreSQL with `asyncpg`
- Alembic

### Frontend
- NiceGUI

### Database
- PostgreSQL running in Docker

## Project Structure

```text
FixIT/
├── .gitignore
├── README.md
├── docker-compose.yml
├── backend/
│   ├── .env
│   ├── alembic.ini
│   ├── main.py
│   ├── requirements.txt
│   ├── alembic/
│   │   ├── env.py
│   │   ├── README
│   │   ├── script.py.mako
│   │   └── versions/
│   │       └── dd9275c4f79f_full_schema.py
│   └── app/
│       ├── __init__.py
│       ├── database.py
│       ├── models.py
│       ├── router.py
│       └── api/
│           ├── __init__.py
│           └── health.py
└── frontend/
    ├── main.py
    └── requirements.txt
```

## Current Database Tables

The current PostgreSQL schema includes:

- `users`
- `user_tokens`
- `sessions`
- `uploads`
- `analyses`
- `messages`

## Run Locally

### 1. Start PostgreSQL in Docker

From the project root:

```bash
docker compose up -d db
```

### 2. Start the backend

```bash
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn main:app --reload --host 127.0.0.1 --port 8080
```

### 3. Start the frontend

Open a second terminal:

```bash
cd frontend
.\.venv\Scripts\Activate.ps1
python main.py
```

## URLs

### Backend
- Health check: `http://127.0.0.1:8080/health`
- Database health check: `http://127.0.0.1:8080/health/db`

### Frontend
- `http://localhost:3000`

## Environment Notes

- PostgreSQL runs in Docker on port `5433`
- backend runs locally on port `8080`
- frontend runs locally on port `3000`

## What Has Been Achieved So Far

So far, the project has moved from a simple local scaffold to a more realistic application foundation with:

- Dockerized PostgreSQL
- async database access
- Alembic migration support
- full schema models for the main application entities
- working backend health and database health checks

## Next Phase

### Phase 5 — Authentication
The next planned phase is authentication, including:

- register
- login
- logout
- `/auth/me`
- password hashing
- bearer token based authentication
- a small matching frontend authentication UI
