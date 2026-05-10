from fastapi import FastAPI

app= FastAPI(title='Fixly')

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Fixly backend is running",
    }