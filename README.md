
# Fixly

Fixly is a repair guidance web app built in Python.

## Current Status

The project currently includes:

- a FastAPI backend
- a NiceGUI frontend
- a health check endpoint
- a simple frontend page that checks whether the backend is running

## Project Structure

```text
Fixly/
├── .gitignore
├── README.md
├── backend/
│   ├── main.py
│   └── requirements.txt
└── frontend/
    ├── main.py
    └── requirements.txt
```

## Run Locally

### Backend

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

## URLs

- Backend health check: http://127.0.0.1:8000/health
- Frontend: http://localhost:3000

## Notes

- The backend and frontend run as separate Python processes.
- The frontend sends a request to the backend health endpoint to confirm that the backend is running.
- This is the current foundation of the project. More features will be added in later phases.
=======
# Fixly

Fixly is a repair guidance web app built in Python.

## Current Status

The project currently includes:

- a FastAPI backend
- a NiceGUI frontend
- a health check endpoint
- a database health check endpoint
- a SQLite database foundation
- a simple frontend page that checks whether the backend is running

## Project Structure

```text
Fixly/
├── .gitignore
├── README.md
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── fixly.db
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

## Current Features

The project currently supports:

- backend and frontend running as separate Python processes
- a backend health check route
- a database connection health check route
- automatic SQLite database file creation on backend startup
- a simple `Case` model using SQLAlchemy
- a frontend page that checks backend availability

## Run Locally

### Backend

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8080
```

### Frontend

```bash
cd frontend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

## URLs

- Backend health check: http://127.0.0.1:8080/health
- Database health check: http://127.0.0.1:8080/health/db
- Frontend: http://localhost:3000

## Notes

- The backend and frontend run as separate Python processes.
- The frontend sends a request to the backend health endpoint to confirm that the backend is running.
- The backend currently uses SQLite for the database layer.
- The `fixly.db` file is created automatically inside the `backend` folder when the app starts.
- This is the current foundation of the project. More features will be added in the next phases.

