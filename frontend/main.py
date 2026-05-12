import httpx
from nicegui import app, ui

# ── Connection details ────────────────────────────────────────────────────────

BACKEND = "http://127.0.0.1:8080"
API     = f"{BACKEND}/api/v1"


# ── Shared header ─────────────────────────────────────────────────────────────
# Call header() at the top of every page function.
# It reads app.storage.user to decide what to show.

def header():
    with ui.header().classes("flex items-center justify-between px-6 py-3 bg-slate-800 text-white"):

        # Logo — clicking it goes home
        ui.label("🔧 Fixly").classes("text-xl font-bold cursor-pointer") \
            .on("click", lambda: ui.navigate.to("/"))

        token = app.storage.user.get("token")

        if token:
            # Logged in — show name and logout button
            name = app.storage.user.get("name", "User")
            with ui.row().classes("items-center gap-4"):
                ui.label(f"Hi, {name}!").classes("text-sm")
                ui.button("Log out", on_click=do_logout).props("flat color=white size=sm")
        else:
            # Not logged in — show navigation links
            with ui.row().classes("gap-2"):
                # flat = transparent background, white text — readable on dark header
                ui.button("Log in",
                          on_click=lambda: ui.navigate.to("/login")) \
                    .props("flat color=white size=sm")
                # outline = transparent background with white border and text
                # This is what makes Register clearly visible on the dark header
                ui.button("Register",
                          on_click=lambda: ui.navigate.to("/register")) \
                    .props("outline color=white size=sm")


# ── Logout (shared action) ────────────────────────────────────────────────────
# Defined here so both the header and any future page can call it.

async def do_logout():
    token = app.storage.user.get("token")
    if token:
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{API}/auth/logout",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=5.0,
                )
        except Exception:
            pass  # Even if the request fails, clear local storage anyway
    app.storage.user.clear()
    ui.navigate.to("/login")


# ── Home page ─────────────────────────────────────────────────────────────────

@ui.page("/")
async def homepage():
    header()

    with ui.column().classes("items-center justify-center w-full mt-24 gap-6"):
        ui.label("Fixly").classes("text-5xl font-bold text-slate-800")
        ui.label("AI-powered repair guidance.").classes("text-gray-400 text-lg")

        if app.storage.user.get("token"):
            # Logged in — welcome them back
            name = app.storage.user.get("name", "")
            ui.label(f"Welcome back, {name}.").classes("text-gray-600")
            ui.button("Go to my cases →",
                      on_click=lambda: ui.navigate.to("/")) \
                .props("size=lg color=primary")
            # /cases does not exist yet — this button will be wired up in Phase 6
        else:
            # Not logged in — prompt them to get started
            with ui.row().classes("gap-3"):
                ui.button("Get started — Register",
                          on_click=lambda: ui.navigate.to("/register")) \
                    .props("size=lg color=primary")
                ui.button("Log in",
                          on_click=lambda: ui.navigate.to("/login")) \
                    .props("size=lg flat")


# ── Register page ─────────────────────────────────────────────────────────────

@ui.page("/register")
async def register_page():
    header()

    with ui.column().classes("items-center justify-center w-full mt-16"):
        with ui.card().classes("w-full max-w-sm p-6"):
            ui.label("Create an account").classes("text-2xl font-bold mb-2")

            email_input    = ui.input("Email address").classes("w-full")
            name_input     = ui.input("Your name").classes("w-full")
            password_input = ui.input(
                "Password",
                password=True,
                password_toggle_button=True,
            ).classes("w-full")

            # Error label — hidden until something goes wrong
            error_label = ui.label("").classes("text-red-500 text-sm min-h-5")

            async def do_register():
                error_label.set_text("")

                email    = email_input.value.strip()
                name     = name_input.value.strip()
                password = password_input.value

                # Basic client-side check before hitting the network
                if not email or not name or not password:
                    error_label.set_text("All three fields are required.")
                    return

                try:
                    async with httpx.AsyncClient() as client:
                        r = await client.post(
                            f"{API}/auth/register",
                            json={"email": email, "name": name, "password": password},
                            timeout=10.0,
                        )

                    if r.status_code == 200:
                        data = r.json()
                        # Store token and name — this is what header() reads
                        app.storage.user["token"] = data["token"]
                        app.storage.user["name"]  = data["user"]["name"]
                        app.storage.user["email"] = data["user"]["email"]
                        ui.navigate.to("/")

                    elif r.status_code == 409:
                        error_label.set_text("An account with this email already exists.")

                    elif r.status_code == 422:
                        # Pydantic validation error — show the first message
                        detail = r.json().get("detail", [])
                        msg = detail[0]["msg"] if detail else "Invalid input."
                        error_label.set_text(msg)

                    else:
                        error_label.set_text(f"Unexpected error ({r.status_code}). Please try again.")

                except Exception:
                    error_label.set_text("Cannot reach the backend. Is it running on port 8080?")

            ui.button("Create account", on_click=do_register) \
                .classes("w-full mt-2").props("color=primary size=lg")

            ui.separator().classes("my-3")

            with ui.row().classes("justify-center gap-1 text-sm text-gray-500"):
                ui.label("Already have an account?")
                ui.link("Log in here", "/login")


# ── Login page ────────────────────────────────────────────────────────────────

@ui.page("/login")
async def login_page():
    header()

    with ui.column().classes("items-center justify-center w-full mt-16"):
        with ui.card().classes("w-full max-w-sm p-6"):
            ui.label("Log in").classes("text-2xl font-bold mb-2")

            email_input    = ui.input("Email address").classes("w-full")
            password_input = ui.input(
                "Password",
                password=True,
                password_toggle_button=True,
            ).classes("w-full")

            error_label = ui.label("").classes("text-red-500 text-sm min-h-5")

            async def do_login():
                error_label.set_text("")

                email    = email_input.value.strip()
                password = password_input.value

                if not email or not password:
                    error_label.set_text("Email and password are required.")
                    return

                try:
                    async with httpx.AsyncClient() as client:
                        r = await client.post(
                            f"{API}/auth/login",
                            json={"email": email, "password": password},
                            timeout=10.0,
                        )

                    if r.status_code == 200:
                        data = r.json()
                        app.storage.user["token"] = data["token"]
                        app.storage.user["name"]  = data["user"]["name"]
                        app.storage.user["email"] = data["user"]["email"]
                        ui.navigate.to("/")

                    elif r.status_code == 401:
                        error_label.set_text("No account exists with this email, or the password is incorrect.")

                    else:
                        error_label.set_text(f"Unexpected error ({r.status_code}). Please try again.")

                except Exception:
                    error_label.set_text("Cannot reach the backend. Is it running on port 8080?")

            ui.button("Log in", on_click=do_login) \
                .classes("w-full mt-2").props("color=primary size=lg")

            ui.separator().classes("my-3")

            with ui.row().classes("justify-center gap-1 text-sm text-gray-500"):
                ui.label("Don't have an account?")
                ui.link("Register here", "/register")


# ── Entry point ───────────────────────────────────────────────────────────────

ui.run(
    port=3000,
    title="Fixly",
    reload=False,
    # storage_secret is required to use app.storage.user.
    # NiceGUI uses it to sign the browser cookie.
    # Use a long random string in any real deployment.
    storage_secret="fixly-dev-secret",
)