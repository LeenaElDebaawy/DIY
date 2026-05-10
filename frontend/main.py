import httpx 
from nicegui import ui

BACKEND_URL = "http://127.0.0.1:8000"

@ui.page("/")
async def homepage():
    ui.label("Fixly: Repair Guidance").style("font-size: 2rem; font-weight: bold")
    
    status =ui.label("Checking backend status...")

    async def check_backend():
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{BACKEND_URL}/health", timeout=3.0)
                data = response.json()
                status.set_text(f"{data['message']}")
        except Exception:
            status.set_text("Backend not reachable, is it running?")
    await check_backend()

ui.run(port=3000, title=" Fixly", reload= False)          
