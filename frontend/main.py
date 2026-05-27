import asyncio
import json

import httpx
from nicegui import app, ui
from nicegui.events import UploadEventArguments

BACKEND = "http://127.0.0.1:8000"
API     = f"{BACKEND}/api/v1"

STATUS = {
    "open":               {"stripe": "#6366f1", "bg": "#ede9fe", "fg": "#4c1d95", "label": "Open"},
    "in_progress":        {"stripe": "#f59e0b", "bg": "#fef3c7", "fg": "#78350f", "label": "In Progress"},
    "resolved":           {"stripe": "#22c55e", "bg": "#dcfce7", "fg": "#14532d", "label": "Resolved"},
    "closed":             {"stripe": "#94a3b8", "bg": "#f1f5f9", "fg": "#475569", "label": "Closed"},
    "complete":           {"stripe": "#22c55e", "bg": "#dcfce7", "fg": "#14532d", "label": "Complete"},
    "failed":             {"stripe": "#ef4444", "bg": "#fee2e2", "fg": "#991b1b", "label": "Failed"},
    "needs_clarification":{"stripe": "#f59e0b", "bg": "#fef3c7", "fg": "#78350f", "label": "Needs info"},
    "running":            {"stripe": "#6366f1", "bg": "#ede9fe", "fg": "#4c1d95", "label": "Analysing"},
}

# Display labels for trade categories.
# Maps the stored trade_category value (from Stage1Output) to a human-readable string.
# Old legacy values are included so cases stored before the 10-category update still display correctly.
_CATEGORY_LABELS: dict[str, str] = {
    "plumbing":               "Plumbing",
    "electrical":             "Electrical",
    "appliance":              "Appliance",
    "carpentry/joinery":      "Carpentry / Joinery",
    "glazing":                "Glazing",
    "sealing/weatherproofing":"Sealing / Weatherproofing",
    "plastering/decorating":  "Plastering / Decorating",
    "structural":             "Structural",
    "hvac":                   "HVAC",
    "general/other":          "General",
    # Legacy labels stored before the 10-category update
    "carpentry":              "Carpentry",
    "painting":               "Decorating",
    "roofing":                "Structural",
    "other":                  "General",
}

def _cat_label(cat: str) -> str:
    """Return a display-friendly label for a trade_category value."""
    return _CATEGORY_LABELS.get((cat or "").lower(), (cat or "").replace("/", " / ").capitalize())


# ─────────────────────────────────────────────────────────────────────────────
# STYLES + JS
# ─────────────────────────────────────────────────────────────────────────────

def inject_styles():
    ui.add_head_html("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap"
          rel="stylesheet">

    <!--
      Material Icons must be loaded explicitly.
      Quasar uses it for the password-visibility eye icon, the upload
      add-file button, and other built-in controls.
      Without this, those icons render as raw text ("visibility_off", "add_box").
    -->
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons"
          rel="stylesheet">

    <style>
      /* ── Force Inter everywhere ── */
      *, *::before, *::after {
        font-family: 'Inter', sans-serif !important;
        box-sizing: border-box;
      }

      /*
        CRITICAL FIX: the * rule above breaks Quasar's icon rendering
        because Material Icons is a font-face — each character IS an icon.
        We restore it here so .q-icon / .material-icons work correctly.
      */
      .material-icons,
      .q-icon,
      i.q-icon {
        font-family: 'Material Icons' !important;
        font-style:   normal !important;
        font-weight:  normal !important;
        line-height:  1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space:  nowrap !important;
        direction:    ltr !important;
        -webkit-font-smoothing: antialiased !important;
        font-feature-settings: 'liga' !important;
      }

      :root { --q-primary: #6366f1; }

      html, body {
        margin: 0 !important;
        padding: 0 !important;
        background: #f8fafc !important;
      }
      .nicegui-content {
        padding: 0 !important;
        max-width: 100% !important;
        width: 100% !important;
      }

      /* ── Sidebar ── */
      .sidebar {
        position: fixed; top: 0; left: 0;
        width: 240px; height: 100vh;
        background: #0f172a;
        border-right: 1px solid rgba(255,255,255,0.07);
        display: flex; flex-direction: column;
        z-index: 500; overflow-y: auto;
        transition: transform 0.25s cubic-bezier(.4,0,.2,1);
      }
      .main { margin-left: 240px; min-height: 100vh; background: #f8fafc; }

      /* ── Mobile bar ── */
      .mobile-bar {
        display: none;
        position: fixed; top: 0; left: 0; right: 0;
        height: 56px; z-index: 499;
        background: #0f172a;
        border-bottom: 1px solid rgba(255,255,255,0.07);
        align-items: center;
        padding: 0 16px; gap: 12px;
      }
      .sidebar-overlay {
        display: none;
        position: fixed; inset: 0;
        background: rgba(0,0,0,0.45); z-index: 498;
      }
      .sidebar-overlay.open { display: block; }

      @media (max-width: 768px) {
        .sidebar { transform: translateX(-240px); z-index: 501; }
        .sidebar.open { transform: translateX(0); }
        .mobile-bar { display: flex; }
        .main { margin-left: 0 !important; padding-top: 56px; }
      }

      /* ── Nav items ── */
      .nav-item {
        display: flex; align-items: center; gap: 10px;
        padding: 9px 12px; margin: 1px 8px; border-radius: 7px;
        cursor: pointer; color: #64748b;
        font-size: 0.875rem; font-weight: 500;
        transition: background 0.12s, color 0.12s;
        user-select: none; text-decoration: none;
      }
      .nav-item:hover             { background: rgba(255,255,255,0.06); color: #cbd5e1; }
      .nav-item.is-active         { background: rgba(99,102,241,0.2);  color: #a5b4fc; }
      .nav-item.is-soon           { opacity: 0.45; cursor: default; }
      .nav-item.is-soon:hover     { background: transparent !important; color: #64748b !important; }
      .nav-item.is-danger         { color: #f87171; }
      .nav-item.is-danger:hover   { background: rgba(248,113,113,0.08); color: #f87171; }

      /* ── Case cards ── */
      .case-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
      }
      .case-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(0,0,0,0.09) !important;
      }

      /* ── Photo thumbnails (case detail grid) ── */
      .photo-thumb {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        cursor: pointer;
        background-size: cover;
        background-position: center;
        flex-shrink: 0;
        transition: border-color 0.15s, transform 0.15s, box-shadow 0.15s;
      }
      .photo-thumb:hover {
        border-color: #6366f1 !important;
        transform: scale(1.05);
        box-shadow: 0 4px 18px rgba(99,102,241,0.3);
      }

      /* ── Hero (homepage) — must fill the full viewport width ── */
      .hero {
        width: 100%;
        background: linear-gradient(140deg, #0f172a 0%, #1e293b 55%, #1b3158 100%);
      }

      /* ────────────────────────────────────────────────────────
         UPLOAD WIDGET OVERRIDES (Quasar q-uploader)

         Problems we fix:
           1. "0.0B / 0.00%" subtitle — hidden
           2. "add_box" icon now renders correctly (Material Icons fix above)
           3. Header gets our brand indigo gradient instead of Quasar blue
           4. Internal file list hidden (we render our own photo grid)
         ──────────────────────────────────────────────────────── */
      .q-uploader {
        box-shadow: none !important;
        border-radius: 10px !important;
        border: 2px dashed #e2e8f0 !important;
        background: #f8fafc !important;
        width: 100% !important;
      }
      .q-uploader__header {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 11px 16px !important;
        min-height: 0 !important;
      }
      .q-uploader__title {
        font-size: 0.875rem !important;
        font-weight: 600 !important;
      }
      /* ← this is the "0.0B / 0.00%" line */
      .q-uploader__subtitle { display: none !important; }
      /* Hide the internal file list — we manage our own photo grid */
      .q-uploader__list { display: none !important; }

      /* ── Analysis progress dot pulse ── */
      @keyframes fixly-pulse {
        0%, 100% { opacity: 1;   transform: scale(1); }
        50%       { opacity: 0.4; transform: scale(0.75); }
      }
      .fixly-pulse { animation: fixly-pulse 1.3s ease-in-out infinite; }

      /* ── Analysis result sections ── */
      .analysis-section-label {
        font-size: 0.68rem; font-weight: 700; letter-spacing: 0.09em;
        text-transform: uppercase; color: #94a3b8; margin-bottom: 6px;
      }
      .analysis-card {
        background: #f8fafc; border: 1px solid #f1f5f9;
        border-radius: 10px; padding: 14px 16px;
      }
      .step-card {
        background: #f8fafc; border: 1px solid #f1f5f9;
        border-radius: 10px; padding: 14px 16px;
        border-left: 3px solid #6366f1;
        margin-bottom: 8px;
      }

      /* ── Pipeline step status labels ── */
      .step-pending {
        font-size:0.83rem; color:#94a3b8;
        padding:3px 0; display:block;
      }
      .step-active {
        font-size:0.83rem; color:#6366f1; font-weight:600;
        padding:3px 0; display:block;
      }
      .step-done {
        font-size:0.83rem; color:#16a34a; font-weight:600;
        padding:3px 0; display:block;
      }

      /* ── Suggestion chips (empty chat state) ── */
      .suggestion-chip {
        display:inline-flex; align-items:center;
        background:#f1f5f9; border:1px solid #e2e8f0;
        border-radius:100px; padding:6px 14px;
        font-size:0.78rem; font-weight:500; color:#475569;
        cursor:pointer; white-space:nowrap;
        transition:background 0.12s,color 0.12s,border-color 0.12s;
        user-select:none;
      }
      .suggestion-chip:hover {
        background:#ede9fe; border-color:#c4b5fd; color:#4c1d95;
      }

      /* ── Composer icon buttons ── */
      .fixly-icon-btn {
        display:inline-flex; align-items:center; justify-content:center;
        width:36px; height:36px; border-radius:10px;
        border:1px solid #e2e8f0; background:#f8fafc;
        color:#64748b; cursor:pointer;
        transition:background 0.15s, color 0.15s, border-color 0.15s;
        text-decoration:none; flex-shrink:0;
        -webkit-appearance:none; appearance:none; outline:none;
        padding:0;
      }
      .fixly-icon-btn:hover  { background:#ede9fe; color:#6366f1; border-color:#c4b5fd; }
      .fixly-icon-btn.active { background:#fee2e2; color:#dc2626; border-color:#fca5a5; }
      .fixly-icon-btn-soon {
        display:inline-flex; align-items:center; justify-content:center;
        width:36px; height:36px; border-radius:10px;
        border:1px solid #f1f5f9; background:#f8fafc;
        color:#cbd5e1; cursor:default; flex-shrink:0; padding:0;
      }
      /* keep old names so nothing else breaks */
      .composer-action { display:none; }
      .composer-action-soon { display:none; }

      /* ── Progress step cards (in _draw_running) ── */
      .prog-step-done    { background:#dcfce7; border-radius:8px; padding:8px 12px; margin-bottom:6px; }
      .prog-step-active  { background:#ede9fe; border-radius:8px; padding:8px 12px; margin-bottom:6px; }
      .prog-step-pending { background:#f8fafc; border-radius:8px; padding:8px 12px; margin-bottom:6px; }

      /* ── Chat bubbles ── */
      .chat-wrap {
        display: flex; flex-direction: column; gap: 10px;
        max-height: 420px; overflow-y: auto;
        padding: 16px 20px;
        scroll-behavior: smooth;
      }
      .chat-bubble-user {
        background: #6366f1; color: white;
        border-radius: 16px 16px 4px 16px;
        padding: 10px 14px; max-width: 75%;
        font-size: 0.875rem; line-height: 1.55;
        align-self: flex-end; word-break: break-word;
      }
      .chat-bubble-assistant {
        background: #f1f5f9; color: #1e293b;
        border-radius: 16px 16px 16px 4px;
        padding: 10px 14px; max-width: 75%;
        font-size: 0.875rem; line-height: 1.55;
        align-self: flex-start; word-break: break-word;
      }
      .chat-bubble-typing {
        background: #f1f5f9; color: #94a3b8;
        border-radius: 16px 16px 16px 4px;
        padding: 10px 14px;
        font-size: 0.875rem; font-style: italic;
        align-self: flex-start;
      }

      /* ── Misc ── */
      .q-expansion-item__content { padding: 0 !important; }
      a.ilink { color: #6366f1; font-weight: 600; text-decoration: none; }
      a.ilink:hover { text-decoration: underline; }
    </style>

    <script>
      function toggleSidebar() {
        const sb = document.getElementById('fixly-sidebar');
        const ov = document.getElementById('fixly-overlay');
        if (sb) sb.classList.toggle('open');
        if (ov) ov.classList.toggle('open');
      }
      function closeSidebar() {
        const sb = document.getElementById('fixly-sidebar');
        const ov = document.getElementById('fixly-overlay');
        if (sb) sb.classList.remove('open');
        if (ov) ov.classList.remove('open');
      }
    </script>
    """)


# ─────────────────────────────────────────────────────────────────────────────
# MOBILE BAR
# ─────────────────────────────────────────────────────────────────────────────

def mobile_bar():
    ui.html("""
      <div id="fixly-overlay" class="sidebar-overlay" onclick="closeSidebar()"></div>
      <div class="mobile-bar">
        <button onclick="toggleSidebar()"
          style="background:none;border:none;cursor:pointer;
                 color:white;font-size:1.5rem;padding:4px 8px;line-height:1;">
          &#9776;
        </button>
        <div style="display:flex;align-items:center;gap:8px;">
          <div style="width:26px;height:26px;
            background:linear-gradient(135deg,#6366f1,#8b5cf6);
            border-radius:6px;display:flex;align-items:center;
            justify-content:center;font-size:13px;">&#128295;</div>
          <span style="color:white;font-size:1rem;font-weight:700;
            letter-spacing:-0.02em;">Knitra</span>
        </div>
      </div>
    """)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def sidebar(active_path: str = "/cases"):
    token = app.storage.user.get("token")
    name  = app.storage.user.get("name",  "User")      if token else "Guest"
    tier  = app.storage.user.get("tier",  "Free Plan") if token else ""

    with ui.element("div").classes("sidebar").props("id=fixly-sidebar"):

        # Logo
        logo = ui.element("div").style(
            "padding:22px 16px 18px;"
            "border-bottom:1px solid rgba(255,255,255,0.06);"
            "cursor:pointer; flex-shrink:0;"
        )
        with logo:
            ui.html("""
              <div style="display:flex;align-items:center;gap:10px;">
                <div style="width:32px;height:32px;
                  background:linear-gradient(135deg,#6366f1,#8b5cf6);
                  border-radius:7px;display:flex;align-items:center;
                  justify-content:center;font-size:16px;flex-shrink:0;">&#128295;</div>
                <span style="color:white;font-size:1.05rem;font-weight:700;
                  letter-spacing:-0.02em;">Knitra</span>
              </div>
            """)
        logo.on("click", lambda: ui.navigate.to("/"))

        # Nav
        with ui.element("div").style("flex:1;padding:12px 0;"):

            def nav(icon, label, path, soon=False, danger=False, cb=None):
                css = "nav-item"
                if path == active_path and not soon and not danger:
                    css += " is-active"
                if soon:   css += " is-soon"
                if danger: css += " is-danger"

                el = ui.element("div").classes(css)
                with el:
                    ui.html(f'<span style="font-size:1rem;flex-shrink:0;">{icon}</span>')
                    ui.html(f'<span style="flex:1;">{label}</span>')
                    if soon:
                        ui.html(
                            '<span style="background:rgba(99,102,241,0.22);color:#818cf8;'
                            'font-size:10px;padding:2px 6px;border-radius:4px;'
                            'font-weight:600;">Soon</span>'
                        )
                if danger:
                    pass
                elif cb:
                    el.on("click", cb)
                elif soon:
                    el.on("click", lambda l=label: ui.notify(
                        f"{l} is coming soon!", position="top-right", type="info"
                    ))
                else:
                    el.on("click", lambda p=path: ui.navigate.to(p))
                return el

            nav("🏠", "Home",     "/")
            nav("📋", "My Cases", "/cases")
            nav("➕", "New Case", "/new")

            ui.html(
                '<div style="height:1px;background:rgba(255,255,255,0.05);'
                'margin:10px 16px;"></div>'
            )
            ui.html(
                '<div style="padding:2px 20px 6px;font-size:10px;font-weight:600;'
                'color:#334155;letter-spacing:0.07em;text-transform:uppercase;">'
                'Coming soon</div>'
            )
            nav("📍", "Nearby Help", "#", soon=True)

        # Bottom
        with ui.element("div").style(
            "border-top:1px solid rgba(255,255,255,0.06);padding:12px 8px;"
        ):
            nav("⚙️", "Account & Plan", "/account")
            ui.html(
                '<div style="height:1px;background:rgba(255,255,255,0.05);'
                'margin:8px 8px;"></div>'
            )
            if token:
                with ui.element("div").style(
                    "display:flex;align-items:center;gap:10px;padding:8px 12px;"
                ):
                    ui.html(
                        f'<div style="width:32px;height:32px;background:#6366f1;'
                        f'border-radius:50%;display:flex;align-items:center;'
                        f'justify-content:center;color:white;font-weight:700;'
                        f'font-size:13px;flex-shrink:0;">{name[0].upper()}</div>'
                    )
                    with ui.element("div").style("flex:1;min-width:0;"):
                        ui.html(
                            f'<div style="color:#e2e8f0;font-size:0.825rem;font-weight:600;'
                            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
                            f'{name}</div>'
                        )
                        ui.html(
                            f'<div style="color:#6366f1;font-size:11px;font-weight:600;'
                            f'margin-top:1px;">{tier}</div>'
                        )
                logout_el = nav("↩", "Log out", "", danger=True)
                logout_el.on("click", do_logout)
            else:
                nav("🔑", "Log in",   "/login")
                nav("✨", "Register", "/register")


# ─────────────────────────────────────────────────────────────────────────────
# TOP NAV  (landing / auth pages — no sidebar)
# ─────────────────────────────────────────────────────────────────────────────

def top_nav():
    with ui.element("div").style(
        "width:100%;"
        "background:#0f172a;"
        "border-bottom:1px solid rgba(255,255,255,0.06);"
        "height:60px;padding:0 2rem;"
        "display:flex;align-items:center;justify-content:space-between;"
    ):
        logo = ui.element("div").style(
            "display:flex;align-items:center;gap:10px;cursor:pointer;"
        )
        with logo:
            ui.html(
                '<div style="width:30px;height:30px;'
                'background:linear-gradient(135deg,#6366f1,#8b5cf6);'
                'border-radius:7px;display:flex;align-items:center;'
                'justify-content:center;font-size:15px;">&#128295;</div>'
            )
            ui.html(
                '<span style="color:white;font-size:1.05rem;font-weight:700;'
                'letter-spacing:-0.02em;">Knitra</span>'
            )
        logo.on("click", lambda: ui.navigate.to("/"))

        token = app.storage.user.get("token")
        with ui.element("div").style("display:flex;align-items:center;gap:8px;"):
            if token:
                ui.button("My Cases", on_click=lambda: ui.navigate.to("/cases")) \
                    .props("flat color=white").style("font-size:0.875rem;font-weight:500;")
                ui.button("Log out", on_click=do_logout) \
                    .props("flat").style("color:#64748b;font-size:0.875rem;")
            else:
                ui.button("Log in", on_click=lambda: ui.navigate.to("/login")) \
                    .props("flat color=white").style("font-weight:500;")
                ui.button("Get started", on_click=lambda: ui.navigate.to("/register")) \
                    .props("unelevated color=primary") \
                    .style("font-weight:600;padding:0 16px;")


# ─────────────────────────────────────────────────────────────────────────────
# LOGOUT
# ─────────────────────────────────────────────────────────────────────────────

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
            pass
    app.storage.user.clear()
    ui.navigate.to("/login")


# ─────────────────────────────────────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────────────────────────────────────

@ui.page("/")
async def homepage():
    inject_styles()
    top_nav()

    # width:100% is required — without it NiceGUI's flex container
    # does not stretch the hero div to fill the viewport.
    with ui.element("div").classes("hero").style(
        "width:100%;min-height:520px;"
        "display:flex;align-items:center;justify-content:center;"
    ):
        with ui.element("div").style(
            "max-width:640px;margin:0 auto;padding:72px 24px;"
            "display:flex;flex-direction:column;align-items:center;"
            "text-align:center;gap:20px;"
        ):
            ui.html(
                '<h1 style="font-size:clamp(2.4rem,5.5vw,3.75rem);font-weight:800;'
                'color:white;margin:0;letter-spacing:-0.03em;line-height:1.1;">'
                'Fix anything.<br>'
                '<span style="color:#818cf8;">Step by step.</span></h1>'
            )
            ui.html(
                '<p style="color:#94a3b8;font-size:1.1rem;max-width:440px;'
                'margin:0;line-height:1.65;">'
                'Describe the problem, upload a photo, and get clear AI-guided '
                'instructions &#8212; from plumbing to electronics.</p>'
            )
            if app.storage.user.get("token"):
                name = app.storage.user.get("name", "")
                ui.html(
                    f'<p style="color:#94a3b8;font-size:0.9rem;margin:4px 0 0;">'
                    f'Welcome back, <strong style="color:white;">{name}</strong></p>'
                )
                ui.button("Go to my cases →",
                          on_click=lambda: ui.navigate.to("/cases")) \
                    .props("unelevated color=primary size=lg") \
                    .style("font-weight:600;padding:0 28px;margin-top:6px;")
            else:
                with ui.element("div").style(
                    "display:flex;gap:12px;margin-top:8px;flex-wrap:wrap;"
                    "justify-content:center;"
                ):
                    ui.button("Get started free",
                              on_click=lambda: ui.navigate.to("/register")) \
                        .props("unelevated color=primary size=lg") \
                        .style("font-weight:600;padding:0 28px;")
                    ui.button("Log in",
                              on_click=lambda: ui.navigate.to("/login")) \
                        .props("flat color=white size=lg") \
                        .style("font-weight:500;padding:0 20px;opacity:0.85;")

    with ui.element("div").style(
        "width:100%;background:white;"
        "border-top:1px solid #f1f5f9;border-bottom:1px solid #f1f5f9;"
    ):
        with ui.element("div").style(
            "max-width:860px;margin:0 auto;padding:48px 32px;"
            "display:flex;justify-content:space-between;gap:24px;flex-wrap:wrap;"
        ):
            for icon, title, desc in [
                ("📸", "Upload a photo",   "Show us what needs fixing"),
                ("🤖", "AI analyses it",   "Three-stage intelligent analysis"),
                ("🔧", "Get instructions", "Clear, step-by-step repair guide"),
            ]:
                ui.html(
                    f'<div style="display:flex;flex-direction:column;align-items:center;'
                    f'text-align:center;gap:10px;flex:1;min-width:160px;">'
                    f'<div style="font-size:2rem;">{icon}</div>'
                    f'<div style="font-weight:600;color:#1e293b;font-size:0.95rem;">{title}</div>'
                    f'<div style="color:#94a3b8;font-size:0.875rem;line-height:1.5;">{desc}</div>'
                    f'</div>'
                )


# ─────────────────────────────────────────────────────────────────────────────
# REGISTER
# ─────────────────────────────────────────────────────────────────────────────

@ui.page("/register")
async def register_page():
    inject_styles()
    top_nav()

    with ui.element("div").style(
        "width:100%;min-height:calc(100vh - 60px);"
        "display:flex;align-items:center;justify-content:center;padding:48px 16px;"
    ):
        with ui.element("div").style(
            "width:100%;max-width:420px;"
            "display:flex;flex-direction:column;gap:20px;"
        ):
            ui.html(
                '<div style="text-align:center;">'
                '<h2 style="font-size:1.75rem;font-weight:800;color:#0f172a;'
                'margin:0 0 6px;letter-spacing:-0.02em;">Create your account</h2>'
                '<p style="color:#64748b;margin:0;font-size:0.95rem;">'
                'Start diagnosing and fixing faster.</p></div>'
            )

            with ui.card().style(
                "border-radius:14px;box-shadow:0 2px 16px rgba(0,0,0,0.07);"
                "border:1px solid #e2e8f0;padding:32px;width:100%;"
            ):
                email_in    = ui.input("Email address").classes("w-full")
                name_in     = ui.input("Your name").classes("w-full")
                # password_toggle_button=True renders a Quasar eye icon.
                # It shows as the text "visibility_off" without Material Icons loaded.
                # Loading Material Icons (in inject_styles) fixes this automatically.
                password_in = ui.input(
                    "Password", password=True, password_toggle_button=True
                ).classes("w-full")
                err = ui.label("").classes("text-red-500 text-sm min-h-5")

                async def do_register():
                    err.set_text("")
                    email    = email_in.value.strip()
                    name     = name_in.value.strip()
                    password = password_in.value
                    if not email or not name or not password:
                        err.set_text("All three fields are required.")
                        return
                    try:
                        async with httpx.AsyncClient() as client:
                            r = await client.post(
                                f"{API}/auth/register",
                                json={"email": email, "name": name, "password": password},
                                timeout=10.0,
                            )
                        if r.status_code == 200:
                            d = r.json()
                            app.storage.user["token"] = d["token"]
                            app.storage.user["name"]  = d["user"]["name"]
                            app.storage.user["email"] = d["user"]["email"]
                            ui.navigate.to("/cases")
                        elif r.status_code == 409:
                            err.set_text("An account with this email already exists.")
                        elif r.status_code == 422:
                            detail = r.json().get("detail", [])
                            err.set_text(detail[0]["msg"] if detail else "Invalid input.")
                        else:
                            err.set_text(f"Unexpected error ({r.status_code}).")
                    except Exception:
                        err.set_text("Cannot reach the backend. Is it running on port 8080?")

                ui.button("Create account", on_click=do_register) \
                    .classes("w-full").style("margin-top:8px;font-weight:600;") \
                    .props("unelevated color=primary size=lg")

                ui.separator().classes("my-4")

                ui.html(
                    '<div style="text-align:center;font-size:0.875rem;color:#64748b;">'
                    'Already have an account? '
                    '<a href="/login" class="ilink">Log in here</a></div>'
                )


# ─────────────────────────────────────────────────────────────────────────────
# LOGIN
# ─────────────────────────────────────────────────────────────────────────────

@ui.page("/login")
async def login_page():
    inject_styles()
    top_nav()

    with ui.element("div").style(
        "width:100%;min-height:calc(100vh - 60px);"
        "display:flex;align-items:center;justify-content:center;padding:48px 16px;"
    ):
        with ui.element("div").style(
            "width:100%;max-width:420px;"
            "display:flex;flex-direction:column;gap:20px;"
        ):
            ui.html(
                '<div style="text-align:center;">'
                '<h2 style="font-size:1.75rem;font-weight:800;color:#0f172a;'
                'margin:0 0 6px;letter-spacing:-0.02em;">Welcome back</h2>'
                '<p style="color:#64748b;margin:0;font-size:0.95rem;">'
                'Log in to your Knitra account.</p></div>'
            )

            with ui.card().style(
                "border-radius:14px;box-shadow:0 2px 16px rgba(0,0,0,0.07);"
                "border:1px solid #e2e8f0;padding:32px;width:100%;"
            ):
                email_in    = ui.input("Email address").classes("w-full")
                password_in = ui.input(
                    "Password", password=True, password_toggle_button=True
                ).classes("w-full")
                err = ui.label("").classes("text-red-500 text-sm min-h-5")

                async def do_login():
                    err.set_text("")
                    email    = email_in.value.strip()
                    password = password_in.value
                    if not email or not password:
                        err.set_text("Email and password are required.")
                        return
                    try:
                        async with httpx.AsyncClient() as client:
                            r = await client.post(
                                f"{API}/auth/login",
                                json={"email": email, "password": password},
                                timeout=10.0,
                            )
                        if r.status_code == 200:
                            d = r.json()
                            app.storage.user["token"] = d["token"]
                            app.storage.user["name"]  = d["user"]["name"]
                            app.storage.user["email"] = d["user"]["email"]
                            ui.navigate.to("/cases")
                        elif r.status_code == 401:
                            err.set_text("Incorrect email or password.")
                        else:
                            err.set_text(f"Unexpected error ({r.status_code}).")
                    except Exception:
                        err.set_text("Cannot reach the backend. Is it running on port 8080?")

                ui.button("Log in", on_click=do_login) \
                    .classes("w-full").style("margin-top:8px;font-weight:600;") \
                    .props("unelevated color=primary size=lg")

                ui.separator().classes("my-4")

                ui.html(
                    '<div style="text-align:center;font-size:0.875rem;color:#64748b;">'
                    "Don&#39;t have an account? "
                    '<a href="/register" class="ilink">Register here</a></div>'
                )


# ─────────────────────────────────────────────────────────────────────────────
# CASES LIST  (/cases)
# ─────────────────────────────────────────────────────────────────────────────

@ui.page("/cases")
async def cases_page():
    inject_styles()

    token = app.storage.user.get("token")
    if not token:
        ui.navigate.to("/login")
        return

    # ── Delete dialog ─────────────────────────────────────────────────────────
    delete_ref = {"case_id": None}

    with ui.dialog() as del_dlg, ui.card().style(
        "border-radius:14px;padding:28px;width:100%;max-width:440px;"
    ):
        ui.html(
            '<h3 style="font-size:1.1rem;font-weight:700;color:#0f172a;margin:0 0 8px;">'
            'Delete this case?</h3>'
            '<p style="color:#64748b;font-size:0.875rem;margin:0 0 20px;line-height:1.6;">'
            'This will permanently remove the case and its related data. '
            'This action cannot be undone.</p>'
        )

        async def confirm_delete():
            cid = delete_ref["case_id"]
            if cid is None:
                return
            try:
                async with httpx.AsyncClient() as client:
                    r = await client.delete(
                        f"{API}/cases/{cid}",
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=10.0,
                    )
                if r.status_code == 204:
                    del_dlg.close()
                    ui.notify("Case deleted.", type="positive", position="top-right")
                    await load_cases()
                elif r.status_code == 401:
                    app.storage.user.clear()
                    ui.navigate.to("/login")
                else:
                    ui.notify(f"Delete failed ({r.status_code}).", type="negative")
            except Exception:
                ui.notify("Cannot reach the backend.", type="negative")

        with ui.element("div").style(
            "display:flex;justify-content:flex-end;gap:8px;"
        ):
            ui.button("Cancel", on_click=del_dlg.close) \
                .props("flat color=grey").style("font-weight:500;")
            ui.button("Delete case", on_click=confirm_delete) \
                .props("unelevated color=negative").style("font-weight:600;")

    # ── Mobile bar + Sidebar ──────────────────────────────────────────────────
    mobile_bar()
    sidebar("/cases")

    # ── Main content ──────────────────────────────────────────────────────────
    with ui.element("div").classes("main"):
        with ui.element("div").style("padding:36px 32px;max-width:900px;"):

            with ui.element("div").style(
                "display:flex;align-items:center;"
                "justify-content:space-between;margin-bottom:8px;flex-wrap:wrap;gap:12px;"
            ):
                ui.html(
                    '<div>'
                    '<h1 style="font-size:1.6rem;font-weight:800;color:#0f172a;'
                    'margin:0 0 2px;letter-spacing:-0.02em;">My Cases</h1>'
                    '<p style="color:#94a3b8;font-size:0.875rem;margin:0;">'
                    'Track and manage your repair projects.</p>'
                    '</div>'
                )
                ui.button("+ New case", on_click=lambda: ui.navigate.to("/new")) \
                    .props("unelevated color=primary") \
                    .style("font-weight:600;padding:0 20px;")

            ui.html(
                '<div style="width:100%;height:1px;background:#e2e8f0;'
                'margin-bottom:20px;"></div>'
            )

            # ── Filter bar ───────────────────────────────────────────────────
            with ui.element("div").style(
                "display:flex;flex-wrap:wrap;gap:10px;margin-bottom:18px;"
            ):
                _all_cat_opts = {
                    "": "All categories",
                    "plumbing":               "Plumbing",
                    "electrical":             "Electrical",
                    "appliance":              "Appliance",
                    "carpentry/joinery":      "Carpentry / Joinery",
                    "glazing":                "Glazing",
                    "sealing/weatherproofing":"Sealing / Weatherproofing",
                    "plastering/decorating":  "Plastering / Decorating",
                    "structural":             "Structural",
                    "hvac":                   "HVAC",
                    "general/other":          "General",
                }
                _sort_opts    = {"desc": "Newest first", "asc": "Oldest first"}
                _status_opts  = {
                    "":                    "All statuses",
                    "open":                "Open",
                    "running":             "Analysing",
                    "complete":            "Complete",
                    "failed":              "Failed",
                    "needs_clarification": "Needs info",
                }
                _conf_opts    = {
                    "":     "All confidence",
                    "high": "High (90+)",
                    "good": "Good (76-89)",
                    "low":  "Low (75 or below)",
                }
                _select_style = (
                    "min-width:170px;font-size:0.82rem;"
                    "background:white;"
                )
                flt_cat   = ui.select(_all_cat_opts,  value="", label="Category") \
                              .style(_select_style)
                flt_sort  = ui.select(_sort_opts,     value="desc", label="Sort") \
                              .style(_select_style)
                flt_stat  = ui.select(_status_opts,   value="", label="Status") \
                              .style(_select_style)
                flt_conf  = ui.select(_conf_opts,     value="", label="Confidence") \
                              .style(_select_style)

            status_lbl = ui.label("").style(
                "color:#94a3b8;font-size:0.875rem;display:block;margin-bottom:12px;"
            )
            container = ui.element("div").style(
                "display:flex;flex-direction:column;gap:14px;"
            )

            # All fetched cases — stored so filters can re-render without a new fetch
            _all_cases: list[dict] = []

            async def render_card(case: dict):
                cid  = case["id"]
                s    = STATUS.get(case["status"], STATUS["open"])
                conf = case.get("confidence")

                def open_delete(cid=cid):
                    delete_ref["case_id"] = cid
                    del_dlg.open()

                # Build confidence badge HTML (empty string if no confidence yet)
                if conf is not None:
                    if conf >= 90:
                        _cc, _cb = "#15803d", "#dcfce7"
                    elif conf >= 76:
                        _cc, _cb = "#92400e", "#fef3c7"
                    else:
                        _cc, _cb = "#dc2626", "#fee2e2"
                    _conf_badge = (
                        f'<span style="background:{_cb};color:{_cc};'
                        f'padding:3px 10px;border-radius:100px;font-size:11px;'
                        f'font-weight:700;white-space:nowrap;flex-shrink:0;">'
                        f'{conf}% AI confidence</span>'
                    )
                else:
                    _conf_badge = ""

                with container:
                    with ui.element("div").classes("case-card"):
                        with ui.element("div").style("display:flex;align-items:stretch;"):
                            ui.element("div").style(
                                f"width:4px;flex-shrink:0;background:{s['stripe']};"
                            )
                            with ui.element("div").style(
                                "flex:1;padding:18px 20px;"
                                "display:flex;flex-direction:column;gap:8px;"
                            ):
                                with ui.element("div").style(
                                    "display:flex;align-items:flex-start;"
                                    "justify-content:space-between;gap:12px;"
                                ):
                                    with ui.element("div").style(
                                        "flex:1;display:flex;flex-direction:column;gap:4px;"
                                    ):
                                        _title = case.get("title") or "Diagnosing…"
                                        ui.html(
                                            f'<span style="font-weight:700;color:#0f172a;'
                                            f'font-size:0.975rem;line-height:1.4;">'
                                            f'{_title}</span>'
                                        )
                                        # Category + confidence on same row
                                        _cat_raw = case.get("category") or ""
                                        _cat_html = (
                                            f'<span style="font-size:0.78rem;color:#6366f1;'
                                            f'font-weight:600;background:#ede9fe;'
                                            f'padding:1px 8px;border-radius:100px;">'
                                            f'{_cat_label(_cat_raw)}</span>'
                                        ) if _cat_raw else ""
                                        if _cat_html or _conf_badge:
                                            ui.html(
                                                f'<div style="display:flex;flex-wrap:wrap;'
                                                f'align-items:center;gap:6px;">'
                                                f'{_cat_html}{_conf_badge}</div>'
                                            )
                                    ui.html(
                                        f'<span style="background:{s["bg"]};color:{s["fg"]};'
                                        f'padding:3px 11px;border-radius:100px;font-size:11.5px;'
                                        f'font-weight:600;white-space:nowrap;flex-shrink:0;">'
                                        f'{s["label"]}</span>'
                                    )

                                ui.html(
                                    f'<span style="font-size:0.775rem;color:#cbd5e1;">'
                                    f'Opened {case["created_at"][:10]}</span>'
                                )

                                ui.html(
                                    '<div style="height:1px;background:#f1f5f9;margin:4px 0;"></div>'
                                )
                                with ui.element("div").style(
                                    "display:flex;align-items:center;"
                                    "justify-content:space-between;gap:8px;"
                                ):
                                    del_btn = ui.element("div").style(
                                        "display:flex;align-items:center;gap:6px;"
                                        "color:#f87171;font-size:0.8rem;font-weight:600;"
                                        "cursor:pointer;padding:4px 0;"
                                        "transition:color 0.12s;"
                                    )
                                    with del_btn:
                                        ui.html('<span style="font-size:0.9rem;">🗑</span>')
                                        ui.html('<span>Delete</span>')
                                    del_btn.on("click", open_delete)

                                    view_btn = ui.element("div").style(
                                        "display:flex;align-items:center;gap:6px;"
                                        "color:#6366f1;font-size:0.8rem;font-weight:700;"
                                        "cursor:pointer;padding:4px 0;"
                                    )
                                    with view_btn:
                                        ui.html('<span>Open case</span>')
                                        ui.html('<span>&#8594;</span>')
                                    view_btn.on(
                                        "click",
                                        lambda cid=cid: ui.navigate.to(f"/cases/{cid}")
                                    )

            async def apply_filters():
                """Re-render the container from _all_cases using current filter values."""
                container.clear()
                filtered = list(_all_cases)

                # Category filter
                cat_val = flt_cat.value or ""
                if cat_val:
                    filtered = [c for c in filtered if (c.get("category") or "") == cat_val]

                # Status filter
                stat_val = flt_stat.value or ""
                if stat_val:
                    filtered = [c for c in filtered if c.get("status") == stat_val]

                # Confidence filter
                conf_val = flt_conf.value or ""
                if conf_val:
                    def _conf_match(c: dict) -> bool:
                        cv = c.get("confidence")
                        if cv is None:
                            return False   # old cases without confidence are excluded from specific tiers
                        if conf_val == "high": return cv >= 90
                        if conf_val == "good": return 76 <= cv <= 89
                        if conf_val == "low":  return cv <= 75
                        return True
                    filtered = [c for c in filtered if _conf_match(c)]

                # Sort
                sort_val = flt_sort.value or "desc"
                if sort_val == "asc":
                    filtered = sorted(filtered, key=lambda c: c.get("created_at", ""))
                # "desc" is already the default order from the API

                if not filtered:
                    with container:
                        _active = any([cat_val, stat_val, conf_val])
                        _msg = (
                            "No cases match the current filters."
                            if _active else
                            "No cases yet"
                        )
                        _sub = (
                            "Try clearing a filter to see more results."
                            if _active else
                            "Open your first case to get AI-guided repair instructions."
                        )
                        ui.html(
                            f'<div style="background:white;border:1.5px dashed #e2e8f0;'
                            f'border-radius:14px;padding:60px 32px;'
                            f'display:flex;flex-direction:column;'
                            f'align-items:center;text-align:center;gap:12px;">'
                            f'<div style="font-size:2.8rem;opacity:0.3;">&#128295;</div>'
                            f'<div style="font-weight:700;color:#334155;font-size:1.05rem;">'
                            f'{_msg}</div>'
                            f'<div style="color:#94a3b8;font-size:0.875rem;'
                            f'max-width:280px;line-height:1.55;">'
                            f'{_sub}</div></div>'
                        )
                        if not _active:
                            ui.button("+ Open your first case",
                                      on_click=lambda: ui.navigate.to("/new")) \
                                .props("unelevated color=primary") \
                                .style("font-weight:600;margin-top:4px;")
                    return

                for case in filtered:
                    await render_card(case)

            async def _on_filter_change(_=None):
                await apply_filters()

            flt_cat.on_value_change(_on_filter_change)
            flt_sort.on_value_change(_on_filter_change)
            flt_stat.on_value_change(_on_filter_change)
            flt_conf.on_value_change(_on_filter_change)

            async def load_cases():
                nonlocal _all_cases
                container.clear()
                status_lbl.set_text("Loading your cases…")
                try:
                    async with httpx.AsyncClient() as client:
                        r = await client.get(
                            f"{API}/cases",
                            headers={"Authorization": f"Bearer {token}"},
                            timeout=10.0,
                        )
                    if r.status_code == 200:
                        _all_cases = r.json()
                        status_lbl.set_text("")
                        await apply_filters()
                    elif r.status_code == 401:
                        app.storage.user.clear()
                        ui.navigate.to("/login")
                    else:
                        status_lbl.set_text(f"Failed to load cases ({r.status_code}).")
                except Exception:
                    status_lbl.set_text(
                        "Cannot reach the backend. Is it running on port 8080?"
                    )

            await load_cases()


# ─────────────────────────────────────────────────────────────────────────────
# CASE DETAIL  (/cases/{case_id})
# ─────────────────────────────────────────────────────────────────────────────

@ui.page("/cases/{case_id}")
async def case_detail_page(case_id: int):
    inject_styles()

    token = app.storage.user.get("token")
    if not token:
        ui.navigate.to("/login")
        return

    # ── Fetch case + initial uploads ─────────────────────────────────────────
    case    = None
    uploads = []
    try:
        async with httpx.AsyncClient() as client:
            rc = await client.get(
                f"{API}/cases/{case_id}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10.0,
            )
            if rc.status_code == 200:
                case = rc.json()
            elif rc.status_code == 401:
                app.storage.user.clear()
                ui.navigate.to("/login")
                return
            elif rc.status_code == 404:
                ui.navigate.to("/cases")
                return

            ru = await client.get(
                f"{API}/cases/{case_id}/uploads",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10.0,
            )
            if ru.status_code == 200:
                uploads = ru.json()
    except Exception:
        pass

    if case is None:
        ui.navigate.to("/cases")
        return

    s = STATUS.get(case.get("status", "open"), STATUS["open"])

    # ── Lightbox dialog ───────────────────────────────────────────────────────
    with ui.dialog() as lightbox_dlg:
        with ui.card().style(
            "background:#050505;border-radius:14px;padding:0;"
            "max-width:95vw;overflow:hidden;"
            "display:flex;flex-direction:column;align-items:center;"
        ):
            lb_container = ui.element("div").style(
                "display:flex;align-items:center;justify-content:center;"
                "min-width:300px;min-height:200px;"
            )
            # Close bar
            with ui.element("div").style(
                "width:100%;display:flex;justify-content:flex-end;"
                "padding:8px 12px;background:rgba(0,0,0,0.6);"
            ):
                close_btn = ui.element("div").style(
                    "color:#94a3b8;font-size:0.8rem;font-weight:600;"
                    "cursor:pointer;padding:4px 10px;border-radius:6px;"
                    "background:rgba(255,255,255,0.07);"
                    "transition:background 0.12s,color 0.12s;"
                )
                with close_btn:
                    ui.html("&#10005; Close")
                close_btn.on("click", lightbox_dlg.close)

    def open_lightbox(url: str):
        lb_container.clear()
        with lb_container:
            ui.image(url).style(
                "max-width:85vw;max-height:80vh;"
                "object-fit:contain;display:block;"
            )
        lightbox_dlg.open()

    # ── Delete dialog ─────────────────────────────────────────────────────────
    with ui.dialog() as del_dlg, ui.card().style(
        "border-radius:14px;padding:28px;width:100%;max-width:440px;"
    ):
        ui.html(
            '<h3 style="font-size:1.1rem;font-weight:700;color:#0f172a;margin:0 0 8px;">'
            'Delete this case?</h3>'
            '<p style="color:#64748b;font-size:0.875rem;margin:0 0 20px;line-height:1.6;">'
            'This will permanently remove the case and its related data. '
            'This action cannot be undone.</p>'
        )

        async def confirm_delete_detail():
            try:
                async with httpx.AsyncClient() as client:
                    r = await client.delete(
                        f"{API}/cases/{case_id}",
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=10.0,
                    )
                if r.status_code == 204:
                    del_dlg.close()
                    ui.navigate.to("/cases")
                elif r.status_code == 401:
                    app.storage.user.clear()
                    ui.navigate.to("/login")
                else:
                    ui.notify(f"Delete failed ({r.status_code}).", type="negative")
            except Exception:
                ui.notify("Cannot reach the backend.", type="negative")

        with ui.element("div").style("display:flex;justify-content:flex-end;gap:8px;"):
            ui.button("Cancel", on_click=del_dlg.close) \
                .props("flat color=grey").style("font-weight:500;")
            ui.button("Delete case", on_click=confirm_delete_detail) \
                .props("unelevated color=negative").style("font-weight:600;")

    # ── Layout ────────────────────────────────────────────────────────────────
    mobile_bar()
    sidebar("/cases")

    with ui.element("div").classes("main"):
        with ui.element("div").style(
            "padding:32px;max-width:900px;"
            "display:flex;flex-direction:column;gap:20px;"
        ):
            # Back link
            back = ui.element("div").style(
                "display:inline-flex;align-items:center;gap:6px;"
                "color:#6366f1;font-size:0.875rem;font-weight:600;"
                "cursor:pointer;width:fit-content;"
            )
            with back:
                ui.html("&#8592; Back to My Cases")
            back.on("click", lambda: ui.navigate.to("/cases"))

            # ── Case header card ──────────────────────────────────────────────
            with ui.element("div").style(
                "background:white;border:1px solid #e2e8f0;"
                "border-radius:14px;padding:24px;overflow:hidden;position:relative;"
            ):
                ui.element("div").style(
                    f"position:absolute;top:0;left:0;right:0;"
                    f"height:4px;background:{s['stripe']};"
                )
                with ui.element("div").style("margin-top:8px;"):
                    with ui.element("div").style(
                        "display:flex;align-items:flex-start;"
                        "justify-content:space-between;gap:16px;flex-wrap:wrap;"
                    ):
                        with ui.element("div"):
                            _case_title = case.get("title") or "Diagnosing…"
                            # Keep a reference so _do_poll can update it in-place
                            # when Stage 1 completes and the backend writes the real title.
                            _title_el = ui.html(
                                f'<h1 style="font-size:1.45rem;font-weight:800;color:#0f172a;'
                                f'margin:0 0 10px;letter-spacing:-0.02em;">'
                                f'{_case_title}</h1>'
                            )
                            with ui.element("div").style(
                                "display:flex;align-items:center;gap:8px;flex-wrap:wrap;"
                            ):
                                if case.get("category"):
                                    ui.html(
                                        f'<span style="background:#ede9fe;color:#4c1d95;'
                                        f'padding:4px 12px;border-radius:100px;'
                                        f'font-size:12px;font-weight:600;">'
                                        f'{_cat_label(case["category"])}</span>'
                                    )
                                ui.html(
                                    f'<span style="background:{s["bg"]};color:{s["fg"]};'
                                    f'padding:4px 12px;border-radius:100px;'
                                    f'font-size:12px;font-weight:600;">{s["label"]}</span>'
                                )
                                ui.html(
                                    f'<span style="color:#cbd5e1;font-size:12px;">'
                                    f'Opened {case["created_at"][:10]}</span>'
                                )
                        ui.button("&#128465; Delete case", on_click=del_dlg.open) \
                            .props("flat color=negative") \
                            .style("font-size:0.8rem;font-weight:600;flex-shrink:0;")

            # ── Photos card ───────────────────────────────────────────────────
            with ui.element("div").style(
                "background:white;border:1px solid #e2e8f0;"
                "border-radius:14px;padding:24px;"
            ):
                # Header row
                with ui.element("div").style(
                    "display:flex;align-items:center;justify-content:space-between;"
                    "margin-bottom:16px;"
                ):
                    with ui.element("div"):
                        ui.html(
                            '<h2 style="font-size:1rem;font-weight:700;color:#0f172a;'
                            'margin:0 0 2px;">Photos</h2>'
                        )
                        ui.html(
                            '<p style="color:#94a3b8;font-size:0.78rem;margin:0;">'
                            'Click any image to view it full size.</p>'
                        )

                # Photo grid — 120 px thumbnails, CSS background-image for cover crop
                photo_grid = ui.element("div").style(
                    "display:flex;flex-wrap:wrap;gap:12px;margin-bottom:20px;"
                )

                def _render_grid(ups: list):
                    """Clear and repopulate the photo grid."""
                    photo_grid.clear()
                    with photo_grid:
                        if not ups:
                            ui.html(
                                '<div style="width:100%;padding:24px 0;'
                                'display:flex;flex-direction:column;align-items:center;'
                                'gap:8px;color:#cbd5e1;">'
                                '<div style="font-size:2rem;opacity:0.4;">&#128247;</div>'
                                '<div style="font-size:0.8rem;font-style:italic;">'
                                'No photos yet &#8212; upload the first one below.</div>'
                                '</div>'
                            )
                        else:
                            for up in ups:
                                url = f"{BACKEND}/static/uploads/{up['storage_path']}"
                                # Outer wrapper: fixed 120×120 with hover interactions
                                thumb = ui.element("div").classes("photo-thumb").style(
                                    f"width:120px;height:120px;"
                                    f"background-image:url('{url}');"
                                )
                                thumb.on("click", lambda u=url: open_lightbox(u))

                _render_grid(uploads)   # initial paint from server-fetched data

                async def refresh_photos():
                    try:
                        async with httpx.AsyncClient() as client:
                            r = await client.get(
                                f"{API}/cases/{case_id}/uploads",
                                headers={"Authorization": f"Bearer {token}"},
                                timeout=10.0,
                            )
                        if r.status_code == 200:
                            _render_grid(r.json())
                    except Exception:
                        pass

                # Upload status labels
                upload_ok  = ui.label("").style(
                    "color:#16a34a;font-size:12px;font-weight:500;min-height:16px;"
                )
                upload_err = ui.label("").style(
                    "color:#dc2626;font-size:12px;font-weight:500;min-height:16px;"
                )

                async def handle_upload(e: UploadEventArguments):
                    upload_ok.set_text("")
                    upload_err.set_text("")
                    content = e.content.read()
                    try:
                        async with httpx.AsyncClient() as client:
                            r = await client.post(
                                f"{API}/cases/{case_id}/uploads",
                                files={"file": (e.name, content, e.type)},
                                headers={"Authorization": f"Bearer {token}"},
                                timeout=30.0,
                            )
                        if r.status_code == 201:
                            upload_ok.set_text("&#10003; Photo uploaded successfully")
                            await refresh_photos()
                        elif r.status_code == 422:
                            d = r.json().get("detail", "Invalid file.")
                            upload_err.set_text(d if isinstance(d, str) else "Invalid file.")
                        elif r.status_code == 401:
                            app.storage.user.clear()
                            ui.navigate.to("/login")
                        else:
                            upload_err.set_text(f"Upload failed ({r.status_code}).")
                    except Exception:
                        upload_err.set_text("Cannot reach the backend.")

                # The upload widget — CSS in inject_styles() removes the "0.0B / 0.00%"
                # subtitle and gives the header our brand indigo gradient.
                ui.upload(
                    label="Add a photo",
                    on_upload=handle_upload,
                    max_file_size=10_000_000,
                    auto_upload=True,
                ).props("accept='image/*' flat").classes("w-full")

            # ── AI Analysis ───────────────────────────────────────────────────
            #
            # This section is a small state machine with four views:
            #   _draw_no_analysis()  — no analysis exists yet (404)
            #   _draw_running(data)  — pipeline is pending or running
            #   _draw_complete(data) — all three stages finished successfully
            #   _draw_failed(data)   — pipeline raised an exception
            #
            # A NiceGUI timer polls the backend every 3.5 s while the pipeline
            # is in flight, updating pipeline_stage after each stage so the UI
            # shows granular progress rather than a generic spinner.
            #
            # Rendering is centralised in _redraw(data):
            #   1. clears and refills btn_area  (the action button in the header)
            #   2. clears and refills analysis_box (the content area below)
            #
            # The timer deactivates itself as soon as status leaves
            # {pending, running}. It can be reactivated on "Analyse / Re-analyse".

            # ── Stage progress maps ───────────────────────────────────────────
            _STAGE_LABEL = {
                None:                     "Preparing&#8230;",
                "stage_1_running":        "Reviewing your uploaded images&#8230;",
                "stage_1_complete":       "Problem identified",
                "stage_2_running":        "Gathering tools and materials&#8230;",
                "stage_2_complete":       "Resources listed",
                "stage_3_running":        "Preparing your repair steps&#8230;",
                "stage_3_complete":       "Steps ready",
                "complete":               "Analysis complete",
                "needs_clarification":    "Clarification needed",
            }
            # pipeline_stage → approximate progress-bar percentage
            _STAGE_PCT = {
                None:                6,
                "stage_1_running":   18,
                "stage_1_complete":  36,
                "stage_2_running":   52,
                "stage_2_complete":  68,
                "stage_3_running":   82,
                "stage_3_complete":  95,
                "complete":          100,
            }

            # ── Backend helpers ───────────────────────────────────────────────

            async def _get_analysis():
                """GET /analysis — returns dict or None (404 / error)."""
                try:
                    async with httpx.AsyncClient() as c:
                        r = await c.get(
                            f"{API}/cases/{case_id}/analysis",
                            headers={"Authorization": f"Bearer {token}"},
                            timeout=10.0,
                        )
                    return r.json() if r.status_code == 200 else None
                except Exception:
                    return None

            async def _get_case():
                """GET /cases/{case_id} — returns dict or None."""
                try:
                    async with httpx.AsyncClient() as c:
                        r = await c.get(
                            f"{API}/cases/{case_id}",
                            headers={"Authorization": f"Bearer {token}"},
                            timeout=10.0,
                        )
                    return r.json() if r.status_code == 200 else None
                except Exception:
                    return None

            async def _post_analyse(text_context: str | None = None):
                """POST /analyse — returns (ok: bool, err_msg: str | None).

                text_context is forwarded to the Stage 1 vision model so the user
                can describe the issue when the photo alone was not clear enough.
                """
                try:
                    body: dict = {}
                    if text_context:
                        body["text_context"] = text_context
                    async with httpx.AsyncClient() as c:
                        r = await c.post(
                            f"{API}/cases/{case_id}/analyse",
                            json=body,
                            headers={"Authorization": f"Bearer {token}"},
                            timeout=10.0,
                        )
                    if r.status_code == 202:
                        return True, None
                    detail = r.json().get("detail", f"Error {r.status_code}")
                    return False, str(detail)
                except Exception:
                    return False, "Cannot reach the backend."

            # ── Outer card (permanent wrapper) ────────────────────────────────
            with ui.element("div").style(
                "background:white;border:1px solid #e2e8f0;"
                "border-radius:14px;overflow:hidden;"
            ):
                # ── Permanent header bar ──────────────────────────────────────
                with ui.element("div").style(
                    "padding:20px 24px 16px;"
                    "display:flex;align-items:center;justify-content:space-between;"
                    "border-bottom:1px solid #f1f5f9;"
                ):
                    with ui.element("div").style(
                        "display:flex;align-items:center;gap:10px;"
                    ):
                        ui.html(
                            '<div style="width:34px;height:34px;'
                            'background:linear-gradient(135deg,#6366f1,#8b5cf6);'
                            'border-radius:9px;display:flex;align-items:center;'
                            'justify-content:center;font-size:17px;'
                            'flex-shrink:0;">&#129302;</div>'
                        )
                        ui.html(
                            '<span style="font-size:1rem;font-weight:700;'
                            'color:#0f172a;">AI Analysis</span>'
                        )

                    # The action button (Analyse / Re-analyse / Retry) lives here.
                    # _redraw() clears and refills this area on each state change.
                    btn_area = ui.element("div")

                # ── Content area — cleared + redrawn by _redraw() ─────────────
                analysis_box = ui.element("div")

            # ── Rendering helpers ─────────────────────────────────────────────

            def _draw_no_analysis():
                """Empty state — no analysis has been triggered yet."""
                with ui.element("div").style(
                    "padding:44px 24px;display:flex;flex-direction:column;"
                    "align-items:center;text-align:center;gap:14px;"
                ):
                    ui.html(
                        '<div style="width:56px;height:56px;'
                        'background:linear-gradient(135deg,'
                        'rgba(99,102,241,0.09),rgba(139,92,246,0.14));'
                        'border-radius:14px;display:flex;align-items:center;'
                        'justify-content:center;font-size:28px;'
                        'opacity:0.75;">&#129302;</div>'
                    )
                    ui.html(
                        '<div style="font-weight:700;color:#334155;font-size:1rem;">'
                        'No analysis yet</div>'
                    )
                    ui.html(
                        '<div style="color:#94a3b8;font-size:0.855rem;'
                        'max-width:320px;line-height:1.65;">'
                        'Upload at least one photo above, then click '
                        '&#8220;Analyse this case&#8221;. Knitra will identify '
                        'the problem, list what you need, and write a '
                        'step-by-step repair plan.'
                        '</div>'
                    )

            def _draw_running(data: dict):
                """Progress view — customer-friendly 5-step breakdown."""
                stage = data.get("pipeline_stage")
                pct   = _STAGE_PCT.get(stage, 8)

                # Customer-friendly headline mapped to pipeline stage
                _TOP = {
                    None:                "Warming up&#8230;",
                    "stage_1_running":   "Reviewing your photos&#8230;",
                    "stage_1_complete":  "Issue identified &#8212; checking what you&#8217;ll need&#8230;",
                    "stage_2_running":   "Checking tools &amp; materials&#8230;",
                    "stage_2_complete":  "Resources ready &#8212; writing your guide&#8230;",
                    "stage_3_running":   "Writing your step-by-step repair guide&#8230;",
                    "stage_3_complete":  "Almost done&#8230;",
                }
                headline = _TOP.get(stage, "Analysing&#8230;")

                # Stage index for step state calculation
                _IDX = {
                    None: 0,
                    "stage_1_running": 1, "stage_1_complete": 2,
                    "stage_2_running": 3, "stage_2_complete": 4,
                    "stage_3_running": 5, "stage_3_complete": 6, "complete": 7,
                }
                cur = _IDX.get(stage, 0)

                # 5 steps: (label, emoji, active_from, done_from)
                _STEPS5 = [
                    ("Reviewing your photos",         "🔍", 1, 2),
                    ("Identifying the problem",       "🧠", 2, 3),
                    ("Checking tools &amp; materials","🔧", 3, 5),
                    ("Writing your repair guide",     "📋", 5, 7),
                    ("Finalising your guide",         "✅", 7, 8),
                ]

                with ui.element("div").style("padding:24px 24px 20px;"):
                    # Headline + pulse dot
                    ui.html(
                        f'<div style="display:flex;align-items:center;gap:10px;'
                        f'margin-bottom:4px;">'
                        f'<div class="fixly-pulse" style="width:9px;height:9px;'
                        f'background:#6366f1;border-radius:50%;flex-shrink:0;"></div>'
                        f'<div style="font-size:0.95rem;font-weight:700;'
                        f'color:#0f172a;">{headline}</div>'
                        f'</div>'
                    )
                    ui.html(
                        '<div style="color:#94a3b8;font-size:0.775rem;'
                        'margin-bottom:18px;padding-left:19px;">'
                        'This takes 1&#8211;2 minutes the first time while AI models load'
                        '</div>'
                    )

                    # Progress bar
                    ui.html(
                        f'<div style="background:#f1f5f9;border-radius:100px;'
                        f'height:6px;margin-bottom:20px;overflow:hidden;">'
                        f'<div style="width:{pct}%;height:100%;border-radius:100px;'
                        f'background:linear-gradient(90deg,#6366f1,#8b5cf6);'
                        f'transition:width 0.7s ease;"></div>'
                        f'</div>'
                    )

                    # Step cards
                    for label, emoji, active_from, done_from in _STEPS5:
                        if cur >= done_from:
                            bg, fg, wt, icon = "#dcfce7", "#15803d", "600", "✓"
                        elif cur >= active_from:
                            bg, fg, wt, icon = "#ede9fe", "#4c1d95", "600", "↻"
                        else:
                            bg, fg, wt, icon = "#f8fafc", "#94a3b8", "400", "○"

                        ui.html(
                            f'<div style="display:flex;align-items:center;gap:10px;'
                            f'background:{bg};border-radius:8px;padding:8px 12px;'
                            f'margin-bottom:6px;">'
                            f'<span style="font-size:0.9rem;flex-shrink:0;">{emoji}</span>'
                            f'<span style="color:{fg};font-size:0.845rem;'
                            f'font-weight:{wt};flex:1;">{label}</span>'
                            f'<span style="color:{fg};font-size:0.85rem;'
                            f'font-weight:700;flex-shrink:0;">{icon}</span>'
                            f'</div>'
                        )

            def _draw_complete(data: dict):
                """Full result view — renders all three stage outputs."""
                sr = data.get("structured_result") or {}
                s1 = sr.get("stage_1") or {}
                s2 = sr.get("stage_2") or {}
                s3 = sr.get("stage_3") or {}

                sev = s1.get("severity", "")
                _sev = {
                    "low":    ("background:#dcfce7;color:#15803d;", "Low severity"),
                    "medium": ("background:#fef3c7;color:#92400e;", "Medium severity"),
                    "high":   ("background:#fee2e2;color:#991b1b;", "High severity"),
                }
                sev_s, sev_l = _sev.get(sev, ("background:#f1f5f9;color:#475569;",
                                               sev.capitalize() + " severity"))

                # Risk assessment values — default to DIY safe for old analyses
                # that pre-date Block 3 and do not have these fields stored
                diy_ok   = s3.get("diy_recommended", True)
                pro_type = s3.get("professional_type") or ""

                diff = s3.get("difficulty_level", "")
                _diff = {
                    "beginner":     ("background:#dcfce7;color:#15803d;", "Beginner"),
                    "intermediate": ("background:#fef3c7;color:#92400e;", "Intermediate"),
                    "advanced":     ("background:#fee2e2;color:#991b1b;", "Advanced"),
                }
                diff_s, diff_l = _diff.get(diff, ("background:#f1f5f9;color:#475569;",
                                                   diff.capitalize()))

                with ui.element("div").style(
                    "padding:22px 24px;"
                    "display:flex;flex-direction:column;gap:18px;"
                ):
                    # ── 1. Diagnosis ──────────────────────────────────────────
                    ui.html('<div class="analysis-section-label">Diagnosis</div>')
                    with ui.element("div").classes("analysis-card"):
                        # Problem + badges
                        ui.html(
                            f'<div style="display:flex;align-items:center;gap:8px;'
                            f'flex-wrap:wrap;margin-bottom:10px;">'
                            f'<span style="font-weight:700;color:#0f172a;font-size:1rem;">'
                            f'{s1.get("problem_type","Unknown")}</span>'
                            f'<span style="background:#ede9fe;color:#4c1d95;padding:2px 10px;'
                            f'border-radius:100px;font-size:11px;font-weight:600;">'
                            f'{_cat_label(s1.get("trade_category",""))}</span>'
                            f'<span style="{sev_s}padding:2px 10px;'
                            f'border-radius:100px;font-size:11px;font-weight:600;">'
                            f'{sev_l}</span>'
                            f'</div>'
                        )
                        # Visual observations
                        for obs in s1.get("observations", []):
                            ui.html(
                                f'<div style="display:flex;align-items:flex-start;gap:8px;'
                                f'margin-bottom:5px;font-size:0.845rem;color:#334155;">'
                                f'<span style="color:#6366f1;flex-shrink:0;'
                                f'margin-top:2px;">&#8226;</span>'
                                f'<span>{obs}</span></div>'
                            )

                    # ── 1b. Confidence / evidence card ───────────────────────
                    conf         = s1.get("confidence", 0)
                    conf_reason  = (s1.get("confidence_reason") or "").strip()
                    vis_evidence = s1.get("visual_evidence") or []
                    unc_notes    = (s1.get("uncertainty_notes") or "").strip()

                    if conf >= 90:
                        conf_color, conf_bg, conf_label = "#15803d", "#dcfce7", "High"
                    elif conf >= 76:
                        conf_color, conf_bg, conf_label = "#92400e", "#fef3c7", "Good"
                    else:
                        conf_color, conf_bg, conf_label = "#dc2626", "#fee2e2", "Low"

                    with ui.element("div").style(
                        "background:#f8fafc;border:1px solid #e2e8f0;"
                        "border-radius:10px;padding:14px 16px;"
                        "display:flex;flex-direction:column;gap:8px;"
                    ):
                        # Header row: confidence % badge + label
                        _reason_html = (
                            f'<div style="font-size:0.78rem;color:#475569;'
                            f'line-height:1.5;margin-top:1px;">{conf_reason}</div>'
                        ) if conf_reason else ""
                        ui.html(
                            f'<div style="display:flex;align-items:center;gap:10px;'
                            f'margin-bottom:2px;">'
                            f'<div style="width:38px;height:38px;border-radius:50%;'
                            f'background:{conf_bg};display:flex;align-items:center;'
                            f'justify-content:center;flex-shrink:0;">'
                            f'<span style="font-size:0.72rem;font-weight:800;'
                            f'color:{conf_color};line-height:1;">{conf}%</span>'
                            f'</div>'
                            f'<div>'
                            f'<div style="font-size:0.8rem;font-weight:700;color:#0f172a;">'
                            f'AI Confidence: {conf_label}</div>'
                            f'{_reason_html}'
                            f'</div>'
                            f'</div>'
                        )
                        # Visual evidence bullets
                        if vis_evidence:
                            ui.html(
                                '<div style="font-size:0.72rem;font-weight:700;'
                                'color:#94a3b8;letter-spacing:0.05em;'
                                'text-transform:uppercase;margin-top:4px;">'
                                'What the AI saw</div>'
                            )
                            with ui.element("div").style(
                                "display:flex;flex-wrap:wrap;gap:5px;margin-top:2px;"
                            ):
                                for ev in vis_evidence:
                                    ui.html(
                                        f'<span style="background:white;border:1px solid #e2e8f0;'
                                        f'border-radius:6px;padding:2px 8px;'
                                        f'font-size:0.76rem;color:#475569;">'
                                        f'{ev}</span>'
                                    )
                        # Uncertainty notes
                        if unc_notes:
                            ui.html(
                                f'<div style="display:flex;align-items:flex-start;gap:6px;'
                                f'background:#fffbeb;border:1px solid #fde68a;'
                                f'border-radius:7px;padding:8px 10px;margin-top:2px;">'
                                f'<span style="font-size:0.85rem;flex-shrink:0;">&#9888;&#65039;</span>'
                                f'<div>'
                                f'<div style="font-size:0.73rem;font-weight:700;'
                                f'color:#92400e;margin-bottom:2px;">Unconfirmed</div>'
                                f'<div style="font-size:0.78rem;color:#78350f;'
                                f'line-height:1.5;">{unc_notes}</div>'
                                f'</div></div>'
                            )

                    # ── 1c. DIY safety banner ─────────────────────────────────
                    if diy_ok:
                        ui.html(
                            '<div style="display:flex;align-items:center;gap:10px;'
                            'background:#dcfce7;border:1px solid #86efac;'
                            'border-radius:10px;padding:12px 16px;">'
                            '<span style="font-size:1.1rem;flex-shrink:0;">&#9989;</span>'
                            '<div>'
                            '<div style="font-weight:700;color:#14532d;font-size:0.88rem;">'
                            'Suitable for DIY</div>'
                            '<div style="color:#15803d;font-size:0.8rem;margin-top:1px;">'
                            'A competent DIYer can safely carry out this repair.</div>'
                            '</div></div>'
                        )
                    else:   # not diy_ok
                        pro_label = pro_type.capitalize() if pro_type else "a qualified professional"
                        ui.html(
                            f'<div style="display:flex;align-items:center;gap:10px;'
                            f'background:#fef2f2;border:1px solid #fca5a5;'
                            f'border-radius:10px;padding:12px 16px;">'
                            f'<span style="font-size:1.1rem;flex-shrink:0;">&#128721;</span>'
                            f'<div>'
                            f'<div style="font-weight:700;color:#991b1b;font-size:0.88rem;">'
                            f'Professional required</div>'
                            f'<div style="color:#b91c1c;font-size:0.8rem;margin-top:1px;">'
                            f'This repair should be carried out by a {pro_label}.</div>'
                            f'</div></div>'
                        )

                    # ── 2. What you'll need ───────────────────────────────────
                    ui.html(
                        '<div class="analysis-section-label">'
                        'What you&#8217;ll need</div>'
                    )
                    # Estimated time chip
                    est = s2.get("estimated_time_minutes", 0)
                    est_str = (
                        f"{est} min" if est < 60
                        else f"{est // 60}h {est % 60}min" if est % 60
                        else f"{est // 60}h"
                    )
                    ui.html(
                        f'<div style="display:inline-flex;align-items:center;gap:6px;'
                        f'background:#ede9fe;border-radius:100px;'
                        f'padding:4px 14px;font-size:0.82rem;'
                        f'font-weight:600;color:#4c1d95;margin-bottom:10px;">'
                        f'&#9200; Estimated time: {est_str}</div>'
                    )
                    # Tools / Materials / Gear grid
                    with ui.element("div").style(
                        "display:grid;"
                        "grid-template-columns:repeat(auto-fit,minmax(190px,1fr));"
                        "gap:10px;"
                    ):
                        for section_title, items, opt_items, icon in [
                            ("Tools",           s2.get("tools", []),
                             s2.get("optional_tools", []),           "&#128295;"),
                            ("Materials",       s2.get("materials", []),
                             s2.get("optional_materials", []),       "&#128230;"),
                            ("Protective gear", s2.get("protective_gear", []),
                             [],                                     "&#129506;"),
                        ]:
                            with ui.element("div").classes("analysis-card"):
                                ui.html(
                                    f'<div style="font-weight:700;color:#0f172a;'
                                    f'font-size:0.84rem;margin-bottom:8px;">'
                                    f'{icon}&nbsp;{section_title}</div>'
                                )
                                for item in items:
                                    ui.html(
                                        f'<div style="display:flex;align-items:flex-start;'
                                        f'gap:6px;margin-bottom:4px;'
                                        f'font-size:0.82rem;color:#334155;">'
                                        f'<span style="color:#6366f1;flex-shrink:0;'
                                        f'margin-top:2px;">&#8226;</span>'
                                        f'<span>{item}</span></div>'
                                    )
                                if opt_items:
                                    ui.html(
                                        '<div style="font-size:0.72rem;font-weight:600;'
                                        'color:#94a3b8;letter-spacing:0.05em;'
                                        'text-transform:uppercase;margin:8px 0 5px;">'
                                        'Nice to have</div>'
                                    )
                                    for item in opt_items:
                                        ui.html(
                                            f'<div style="display:flex;'
                                            f'align-items:flex-start;gap:6px;'
                                            f'margin-bottom:4px;font-size:0.82rem;'
                                            f'color:#94a3b8;">'
                                            f'<span style="color:#c7d2fe;flex-shrink:0;'
                                            f'margin-top:2px;">&#8226;</span>'
                                            f'<span>{item}</span></div>'
                                        )

                    # ── 3. Repair steps ───────────────────────────────────────
                    ui.html(
                        '<div class="analysis-section-label">'
                        'Step-by-step repair plan</div>'
                    )
                    # Difficulty badge
                    ui.html(
                        f'<div style="{diff_s}border-radius:100px;'
                        f'padding:3px 12px;font-size:11px;font-weight:700;'
                        f'display:inline-block;margin-bottom:10px;">'
                        f'{diff_l} difficulty</div>'
                    )
                    for step in s3.get("steps", []):
                        warning = step.get("warning")
                        with ui.element("div").classes("step-card"):
                            # Step number circle + title
                            ui.html(
                                f'<div style="display:flex;align-items:center;'
                                f'gap:10px;margin-bottom:7px;">'
                                f'<div style="width:24px;height:24px;background:#6366f1;'
                                f'border-radius:50%;display:flex;align-items:center;'
                                f'justify-content:center;color:white;font-size:11px;'
                                f'font-weight:700;flex-shrink:0;">'
                                f'{step.get("step_number","")}</div>'
                                f'<span style="font-weight:700;color:#0f172a;'
                                f'font-size:0.9rem;">{step.get("title","")}</span>'
                                f'</div>'
                            )
                            # Step description
                            ui.html(
                                f'<div style="font-size:0.845rem;color:#334155;'
                                f'line-height:1.65;margin-left:34px;">'
                                f'{step.get("description","")}</div>'
                            )
                            # Per-step warning (only when not null)
                            if warning:
                                ui.html(
                                    f'<div style="display:flex;align-items:flex-start;'
                                    f'gap:8px;margin-top:9px;margin-left:34px;'
                                    f'background:#fff7ed;border:1px solid #fed7aa;'
                                    f'border-radius:7px;padding:8px 12px;">'
                                    f'<span style="font-size:0.9rem;'
                                    f'flex-shrink:0;">&#9888;&#65039;</span>'
                                    f'<span style="font-size:0.8rem;color:#92400e;'
                                    f'line-height:1.55;">{warning}</span>'
                                    f'</div>'
                                )

                    # ── 4. Safety notes ───────────────────────────────────────
                    safety = s3.get("safety_notes", [])
                    if safety:
                        ui.html(
                            '<div class="analysis-section-label">Safety notes</div>'
                        )
                        with ui.element("div").style(
                            "background:#fff7ed;border:1px solid #fed7aa;"
                            "border-radius:10px;padding:14px 16px;"
                        ):
                            for note in safety:
                                ui.html(
                                    f'<div style="display:flex;align-items:flex-start;'
                                    f'gap:8px;margin-bottom:7px;'
                                    f'font-size:0.845rem;color:#78350f;">'
                                    f'<span style="flex-shrink:0;'
                                    f'margin-top:1px;">&#9888;&#65039;</span>'
                                    f'<span>{note}</span></div>'
                                )

                    # ── 5. Professional guidance ──────────────────────────────
                    wcp = s3.get("when_to_call_professional", "")
                    if wcp or not diy_ok:
                        ui.html(
                            '<div class="analysis-section-label">'
                            'Professional guidance</div>'
                        )
                        panel_bg = "#fef2f2" if not diy_ok else "#fff7ed"
                        panel_border = "#fecaca" if not diy_ok else "#fed7aa"
                        with ui.element("div").style(
                            f"background:{panel_bg};border:1px solid {panel_border};"
                            "border-radius:10px;padding:14px 16px;"
                            "display:flex;flex-direction:column;gap:8px;"
                        ):
                            if not diy_ok and pro_type:
                                ui.html(
                                    f'<div style="display:flex;align-items:center;gap:8px;">'
                                    f'<span style="font-size:1rem;flex-shrink:0;">&#128272;</span>'
                                    f'<span style="font-weight:700;color:#991b1b;'
                                    f'font-size:0.845rem;">Contact a {pro_type}</span>'
                                    f'</div>'
                                )
                            if wcp:
                                ui.html(
                                    f'<div style="display:flex;align-items:flex-start;'
                                    f'gap:8px;">'
                                    f'<span style="font-size:1rem;'
                                    f'flex-shrink:0;">&#128680;</span>'
                                    f'<span style="font-size:0.845rem;color:#991b1b;'
                                    f'line-height:1.65;">{wcp}</span>'
                                    f'</div>'
                                )

                    # ── Metadata footer ───────────────────────────────────────
                    ms      = data.get("processing_time_ms") or 0
                    t_str   = f"{ms // 1000}s" if ms >= 1000 else f"{ms}ms"
                    model   = data.get("model", "")
                    prov    = data.get("provider", "")
                    ui.html(
                        f'<div style="color:#cbd5e1;font-size:11px;'
                        f'display:flex;gap:14px;flex-wrap:wrap;'
                        f'margin-top:4px;padding-top:8px;'
                        f'border-top:1px solid #f1f5f9;">'
                        f'<span>Provider: {prov}</span>'
                        f'<span>Model: {model}</span>'
                        f'<span>Processing time: {t_str}</span>'
                        f'</div>'
                    )

            def _draw_needs_clarification(data: dict):
                """
                Clarification gate — Stage 1 was not confident enough to diagnose.

                Shows a detailed breakdown of WHY confidence was low, WHAT the model
                saw vs. could not confirm, and exactly what photo or information is
                needed.  Also provides an inline description field so the user can
                add text context without leaving the page.
                """
                # ── Pull all Stage 1 data for explainability ──────────────────
                s1 = (data.get("stage_outputs") or {}).get("stage_1") or {}

                conf         = s1.get("confidence", 0)
                conf_reason  = (s1.get("confidence_reason") or "").strip()
                vis_evidence = s1.get("visual_evidence") or []
                unc_notes    = (s1.get("uncertainty_notes") or "").strip()
                observations = s1.get("observations") or []
                clarif_req   = (s1.get("clarification_request") or "").strip()

                # Fallback message if Stage 1 output is not available
                msg = (
                    clarif_req
                    or data.get("validation_issues")
                    or "Please upload a clearer, closer photo of the damaged area "
                       "or describe the problem in more detail, then click Re-analyse."
                )

                # Confidence badge colours
                if conf >= 76:
                    _cc, _cb, _cl = "#92400e", "#fef3c7", "Medium"
                else:
                    _cc, _cb, _cl = "#dc2626", "#fee2e2", "Low"

                # ── Detect what type of input the model needs ─────────────────
                _combined_hints = (clarif_req + " " + conf_reason).lower()
                _photo_tips: list[tuple[str, str]] = []
                if any(w in _combined_hints for w in ["close", "closer", "close-up", "detail", "clearer"]):
                    _photo_tips.append(("🔍", "Upload a closer, more detailed photo of the damaged area"))
                if any(w in _combined_hints for w in ["wider", "wide", "context", "surround", "full"]):
                    _photo_tips.append(("📐", "Include more surroundings — a wider shot shows the full extent"))
                if any(w in _combined_hints for w in ["light", "bright", "dark", "lit", "illumin", "blurr"]):
                    _photo_tips.append(("💡", "Take the photo in better light or move to a brighter area"))
                if any(w in _combined_hints for w in ["describe", "written", "description", "explain", "unclear", "not sure", "cannot determine"]):
                    _photo_tips.append(("✏️", "Add a written description of what you see and what the problem is"))
                if not _photo_tips:
                    _photo_tips = [
                        ("📷", "Upload a new photo — closer, well-lit, and in focus"),
                        ("✏️", "Add a written description to give the AI a second evidence stream"),
                    ]

                with ui.element("div").style("padding:22px 24px;"
                                             "display:flex;flex-direction:column;gap:16px;"):

                    # ── 1. Confidence explainability card ─────────────────────
                    ui.html('<div class="analysis-section-label">Why more information is needed</div>')
                    with ui.element("div").style(
                        "background:#fafafa;border:1px solid #e2e8f0;"
                        "border-radius:10px;padding:16px 18px;"
                        "display:flex;flex-direction:column;gap:10px;"
                    ):
                        # Confidence score row
                        _nc_reason_html = (
                            f'<div style="font-size:0.78rem;color:#475569;'
                            f'margin-top:2px;line-height:1.5;">{conf_reason}</div>'
                        ) if conf_reason else ""
                        ui.html(
                            f'<div style="display:flex;align-items:center;gap:10px;">'
                            f'<div style="width:42px;height:42px;border-radius:50%;'
                            f'background:{_cb};display:flex;align-items:center;'
                            f'justify-content:center;flex-shrink:0;">'
                            f'<span style="font-size:0.7rem;font-weight:800;'
                            f'color:{_cc};line-height:1;">{conf}%</span>'
                            f'</div>'
                            f'<div>'
                            f'<div style="font-size:0.82rem;font-weight:700;color:#0f172a;">'
                            f'AI Confidence: {_cl}</div>'
                            f'{_nc_reason_html}'
                            f'</div></div>'
                        )

                        # What the model could see
                        if observations or vis_evidence:
                            ui.html(
                                '<div style="font-size:0.72rem;font-weight:700;'
                                'color:#94a3b8;letter-spacing:0.05em;'
                                'text-transform:uppercase;margin-top:2px;">'
                                'What the AI could see</div>'
                            )
                            _evidence = vis_evidence if vis_evidence else observations
                            with ui.element("div").style(
                                "display:flex;flex-wrap:wrap;gap:5px;"
                            ):
                                for ev in _evidence:
                                    ui.html(
                                        f'<span style="background:white;border:1px solid #e2e8f0;'
                                        f'border-radius:6px;padding:2px 9px;'
                                        f'font-size:0.75rem;color:#475569;">{ev}</span>'
                                    )

                        # What it could NOT confirm
                        if unc_notes:
                            ui.html(
                                f'<div style="display:flex;align-items:flex-start;gap:7px;'
                                f'background:#fff7ed;border:1px solid #fed7aa;'
                                f'border-radius:7px;padding:9px 11px;">'
                                f'<span style="font-size:0.9rem;flex-shrink:0;">&#10067;</span>'
                                f'<div>'
                                f'<div style="font-size:0.73rem;font-weight:700;'
                                f'color:#9a3412;margin-bottom:2px;">Could not confirm</div>'
                                f'<div style="font-size:0.78rem;color:#7c2d12;'
                                f'line-height:1.5;">{unc_notes}</div>'
                                f'</div></div>'
                            )

                    # ── 2. What is needed card ────────────────────────────────
                    ui.html('<div class="analysis-section-label">What would help</div>')
                    with ui.element("div").style(
                        "background:#fffbeb;border:1px solid #fcd34d;"
                        "border-radius:10px;padding:16px 18px;"
                        "display:flex;flex-direction:column;gap:10px;"
                    ):
                        # The model's specific request
                        if msg:
                            ui.html(
                                f'<div style="font-size:0.845rem;color:#78350f;'
                                f'line-height:1.6;font-weight:500;">{msg}</div>'
                            )
                            ui.html('<div style="border-top:1px solid #fde68a;"></div>')

                        # Actionable photo / input tips
                        for _icon, _tip in _photo_tips:
                            ui.html(
                                f'<div style="display:flex;align-items:flex-start;gap:8px;">'
                                f'<span style="font-size:1rem;flex-shrink:0;">{_icon}</span>'
                                f'<span style="font-size:0.82rem;color:#78350f;'
                                f'line-height:1.5;">{_tip}</span>'
                                f'</div>'
                            )

                        ui.html('<div style="border-top:1px solid #fde68a;"></div>')

                        # ── Inline description input ──────────────────────────
                        ui.html(
                            '<div style="display:flex;align-items:center;gap:6px;'
                            'font-size:0.82rem;font-weight:700;color:#92400e;">'
                            '<span>&#128172;</span>'
                            '<span>Describe what you see and what the problem is</span>'
                            '</div>'
                        )
                        clarif_input = ui.textarea(
                            placeholder=(
                                "e.g. 'This is a hairline crack running diagonally across "
                                "my kitchen wall plaster near the window frame. It appeared "
                                "after some heavy rain.'"
                            )
                        ).props("outlined dense autogrow rows=2") \
                         .style(
                            "width:100%;font-size:0.86rem;"
                            "background:white;border-radius:8px;"
                        )
                        ui.html(
                            '<div style="font-size:0.77rem;color:#92400e;opacity:0.75;">'
                            'Your description gives the AI a second evidence stream alongside the photo. '
                            'Leave blank to re-analyse the photo only.'
                            '</div>'
                        )

                        # ── Re-analyse button ─────────────────────────────────
                        async def _on_clarification_analyse():
                            text = clarif_input.value.strip() or None
                            ok, err = await _post_analyse(text)
                            if ok:
                                fresh = await _get_analysis()
                                _redraw(fresh)
                                poll_timer.active = True
                            else:
                                ui.notify(
                                    err or "Failed to start analysis.",
                                    type="negative",
                                    position="top-right",
                                )

                        ui.button(
                            "Re-analyse →",
                            on_click=_on_clarification_analyse,
                        ).props("unelevated color=warning") \
                         .style(
                            "font-weight:700;font-size:0.875rem;"
                            "align-self:flex-start;"
                        )

            def _draw_failed(data: dict):
                """Error state — shows validation_issues and a retry button."""
                issue = data.get("validation_issues") or "Unknown error."
                with ui.element("div").style("padding:24px;"):
                    with ui.element("div").style(
                        "background:#fef2f2;border:1px solid #fecaca;"
                        "border-radius:10px;padding:16px 18px;"
                    ):
                        ui.html(
                            '<div style="font-weight:700;color:#991b1b;'
                            'margin-bottom:7px;">&#9888; Analysis failed</div>'
                        )
                        ui.html(
                            f'<div style="font-size:0.82rem;color:#7f1d1d;'
                            f'line-height:1.55;font-family:monospace;">'
                            f'{issue}</div>'
                        )

            # ── Master redraw — called on every state transition ──────────────
            def _redraw(data):
                """
                Redraw btn_area + analysis_box from data.
                data is the full analysis dict, or None if 404.
                """
                status = (data or {}).get("status")

                # Button area
                btn_area.clear()
                with btn_area:
                    if status in ("running", "needs_clarification"):
                        # running     — no button, pipeline can't be cancelled mid-flight
                        # needs_clarification — button lives inside the card so the user
                        #   can type a description and submit in one place
                        pass
                    else:
                        # Choose label and style based on current state
                        if status == "complete":
                            lbl, style = "Re-analyse", "flat"
                        elif status == "failed":
                            lbl, style = "Retry analysis", "unelevated"
                        else:
                            lbl, style = "Analyse this case", "unelevated"

                        async def _on_analyse():
                            ok, err = await _post_analyse()
                            if ok:
                                fresh = await _get_analysis()
                                _redraw(fresh)
                                poll_timer.active = True
                            else:
                                ui.notify(
                                    err or "Failed to start analysis.",
                                    type="negative",
                                    position="top-right",
                                )

                        ui.button(lbl, on_click=_on_analyse) \
                            .props(f"{style} color=primary size=sm") \
                            .style("font-weight:600;")

                # Content area
                analysis_box.clear()
                with analysis_box:
                    if data is None:
                        _draw_no_analysis()
                    elif status in ("pending", "running"):
                        _draw_running(data)
                    elif status == "complete":
                        _draw_complete(data)
                    elif status == "needs_clarification":
                        _draw_needs_clarification(data)
                    elif status == "failed":
                        _draw_failed(data)

            # ── Poll timer ────────────────────────────────────────────────────
            # Fires every 3.5 s while the pipeline is in flight.
            # Deactivates itself the moment status leaves {pending, running}.
            # Can be reactivated by the Analyse / Re-analyse / Retry button.
            _was_running   = False   # tracks whether we started in a running state
            _title_updated = False   # ensures we only re-fetch the case title once

            async def _do_poll():
                nonlocal _was_running, _title_updated
                data  = await _get_analysis()
                st    = (data or {}).get("status")
                stage = (data or {}).get("pipeline_stage")
                _redraw(data)

                # Update the case header title as soon as Stage 1 writes it to the DB.
                # Stage 1 completes before Stage 2 starts, so the title is available
                # at stage_1_complete. Re-fetching the case object is the only way to
                # get the updated title without a page reload.
                if not _title_updated and stage in ("stage_1_complete", "stage_2_running",
                                                     "stage_2_complete", "stage_3_running",
                                                     "stage_3_complete", "complete"):
                    fresh_case = await _get_case()
                    if fresh_case:
                        new_title = fresh_case.get("title") or ""
                        if new_title and new_title != "Diagnosing…":
                            _title_el.set_content(
                                f'<h1 style="font-size:1.45rem;font-weight:800;color:#0f172a;'
                                f'margin:0 0 10px;letter-spacing:-0.02em;">'
                                f'{new_title}</h1>'
                            )
                            _title_updated = True

                if st not in ("pending", "running"):
                    poll_timer.active = False
                    # If the pipeline just finished while the user was watching,
                    # rebuild the chat section in-place so it unlocks immediately.
                    # (_render_chat_section is defined below; closures resolve at call time)
                    if _was_running and st == "complete":
                        _was_running = False
                        await _render_chat_section(True)
                    elif _was_running and st in ("failed", "needs_clarification"):
                        _was_running = False

            poll_timer = ui.timer(3.5, _do_poll, active=False)

            # ── Initial page load ─────────────────────────────────────────────
            # Fetch whatever the current analysis state is (may be None, running,
            # complete, or failed) and render it immediately without waiting for
            # the first timer tick.
            _initial = await _get_analysis()
            _redraw(_initial)
            if (_initial or {}).get("status") in ("pending", "running"):
                _was_running = True
                poll_timer.active = True

            # ── Chat section — dynamic container ──────────────────────────────
            # Uses a plain div that can be cleared and rebuilt when analysis
            # completes while the user is watching (poll timer calls
            # _render_chat_section(True) instead of navigating away).
            _analysis_complete = (_initial or {}).get("status") == "complete"
            chat_section = ui.element("div")

            async def _render_chat_section(is_complete: bool):
                """
                Rebuild the chat section in-place.
                Safe to call multiple times — clears and recreates all child elements.
                Called at page load and by _do_poll when analysis transitions to complete.
                """
                chat_section.clear()
                with chat_section:
                    if not is_complete:
                        # ── Locked placeholder ─────────────────────────────────
                        with ui.element("div").style(
                            "background:white;border:1px solid #e2e8f0;"
                            "border-radius:14px;padding:28px 24px;"
                            "text-align:center;"
                        ):
                            ui.html(
                                '<div style="font-size:2rem;margin-bottom:10px;'
                                'opacity:0.4;">&#128172;</div>'
                                '<div style="font-weight:700;color:#334155;'
                                'font-size:0.95rem;margin-bottom:6px;">Ask Knitra</div>'
                                '<div style="color:#94a3b8;font-size:0.83rem;'
                                'line-height:1.5;">'
                                'The repair chat will open once the AI has finished '
                                'analysing your case. Analyse your photos above to '
                                'get started.</div>'
                            )
                        return

                    # ── Full chat UI ───────────────────────────────────────────
                    with ui.element("div").style(
                        "background:white;border:1px solid #e2e8f0;"
                        "border-radius:14px;overflow:hidden;"
                    ):
                        # Header
                        with ui.element("div").style(
                            "padding:18px 24px 14px;"
                            "display:flex;align-items:center;gap:10px;"
                            "border-bottom:1px solid #f1f5f9;"
                        ):
                            ui.html(
                                '<div style="width:34px;height:34px;'
                                'background:linear-gradient(135deg,#0f172a,#1e293b);'
                                'border-radius:9px;display:flex;align-items:center;'
                                'justify-content:center;font-size:17px;'
                                'flex-shrink:0;">&#128172;</div>'
                            )
                            with ui.element("div"):
                                ui.html(
                                    '<div style="font-size:1rem;font-weight:700;'
                                    'color:#0f172a;">Ask Knitra</div>'
                                )
                                ui.html(
                                    '<div style="font-size:0.75rem;color:#94a3b8;'
                                    'margin-top:1px;">AI assistant with full knowledge '
                                    'of your repair case</div>'
                                )

                        # Message list
                        chat_messages = ui.element("div").classes("chat-wrap")

                        _SUGGESTIONS = [
                            "What tools do I need?",
                            "Is this safe to DIY?",
                            "How long will this take?",
                            "What could go wrong?",
                        ]

                        def _render_messages(msgs: list):
                            chat_messages.clear()
                            with chat_messages:
                                if not msgs:
                                    ui.html(
                                        '<div style="text-align:center;color:#94a3b8;'
                                        'font-size:0.845rem;padding:16px 0 8px;">'
                                        'Ask anything about your repair&#8230;</div>'
                                    )
                                    with ui.element("div").style(
                                        "display:flex;flex-wrap:wrap;gap:8px;"
                                        "justify-content:center;padding:0 8px 8px;"
                                    ):
                                        for suggestion in _SUGGESTIONS:
                                            chip = ui.element("div").classes("suggestion-chip")
                                            with chip:
                                                ui.html(f'<span>{suggestion}</span>')
                                            chip.on(
                                                "click",
                                                lambda s=suggestion: _quick_send(s),
                                            )
                                else:
                                    for msg in msgs:
                                        cls = ("chat-bubble-user" if msg["role"] == "user"
                                               else "chat-bubble-assistant")
                                        ui.html(f'<div class="{cls}">{msg["content"]}</div>')

                        async def _load_chat():
                            try:
                                async with httpx.AsyncClient() as c:
                                    r = await c.get(
                                        f"{API}/cases/{case_id}/chat",
                                        headers={"Authorization": f"Bearer {token}"},
                                        timeout=10.0,
                                    )
                                if r.status_code == 200:
                                    _render_messages(r.json())
                            except Exception:
                                pass

                        # ── Input area ────────────────────────────────────────
                        # Token stored in JS so the chat mic/upload buttons can
                        # call the backend directly without going through Python.
                        ui.run_javascript(
                            f"window._knitraToken = {repr(token)};"
                            f"window._knitraCaseId = {case_id};"
                            f"window._knitraApi = {repr(API)};"
                        )

                        with ui.element("div").style(
                            "padding:14px 20px;"
                            "border-top:1px solid #f1f5f9;"
                        ).classes("chat-input-row"):

                            # Toolbar: mic + attachment (above the textarea)
                            with ui.element("div").style(
                                "display:flex;align-items:center;gap:6px;"
                                "margin-bottom:8px;"
                            ):
                                # Mic button — SVG microphone icon, class gives hover + active styles
                                ui.html(
                                    '<button id="knitra-chat-mic" '
                                    'onclick="knitraChatToggleVoice()" '
                                    'title="Voice input — tap and speak" '
                                    'class="fixly-icon-btn">'
                                    '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18"'
                                    ' viewBox="0 0 24 24" fill="none" stroke="currentColor"'
                                    ' stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                                    '<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>'
                                    '<path d="M19 10v2a7 7 0 0 1-14 0v-2"/>'
                                    '<line x1="12" x2="12" y1="19" y2="22"/>'
                                    '</svg>'
                                    '</button>'
                                )
                                # Attach button — paperclip SVG icon, triggers hidden file input
                                ui.html(
                                    '<button onclick="document.getElementById(\'knitra-chat-file\').click()" '
                                    'title="Attach a photo to this case" '
                                    'class="fixly-icon-btn">'
                                    '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18"'
                                    ' viewBox="0 0 24 24" fill="none" stroke="currentColor"'
                                    ' stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                                    '<path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57'
                                    ' A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48"/>'
                                    '</svg>'
                                    '</button>'
                                    '<input id="knitra-chat-file" type="file" '
                                    'accept="image/*,video/*" multiple style="display:none" '
                                    'onchange="knitraChatUpload(this)">'
                                )
                                # Status line (voice recording / upload feedback)
                                ui.html(
                                    '<span id="knitra-chat-status" '
                                    'style="font-size:0.78rem;color:#94a3b8;'
                                    'margin-left:4px;"></span>'
                                )

                            # Textarea + Send row
                            with ui.element("div").style(
                                "display:flex;gap:10px;align-items:flex-end;"
                            ):
                                chat_input = ui.textarea(
                                    placeholder="Ask about your repair… (Enter to send, Shift+Enter for newline)"
                                ).props("outlined dense autogrow rows=1") \
                                 .style("flex:1;font-size:0.875rem;") \
                                 .classes("knitra-chat-textarea")

                                send_btn = ui.button("Send", on_click=lambda: asyncio.create_task(_send_chat())) \
                                    .props("unelevated color=primary") \
                                    .style("font-weight:600;flex-shrink:0;") \
                                    .classes("knitra-send-btn")

                        # Enter-to-send: JS intercepts Enter (not Shift+Enter)
                        # and clicks the Send button so Python handles the send.
                        ui.run_javascript("""
                            (function setupEnter() {
                                const row = document.querySelector('.knitra-chat-textarea textarea');
                                if (!row) { setTimeout(setupEnter, 150); return; }
                                row.addEventListener('keydown', function(e) {
                                    if (e.key === 'Enter' && !e.shiftKey) {
                                        e.preventDefault();
                                        const btn = document.querySelector('.knitra-send-btn');
                                        if (btn) btn.click();
                                    }
                                });
                            })();
                        """)

                        async def _send_chat():
                            # Primary: read from NiceGUI-synced value
                            content = chat_input.value.strip()

                            # Fallback: if Vue/Quasar WebSocket sync hasn't fired yet
                            # (common after mic transcription injects text via JS),
                            # read directly from the DOM textarea value.
                            if not content:
                                try:
                                    dom_val = await ui.run_javascript(
                                        "(document.querySelector('.knitra-chat-textarea textarea')"
                                        "?.value || '').trim()",
                                        timeout=2.0,
                                    )
                                    if dom_val:
                                        content = str(dom_val).strip()
                                except Exception:
                                    pass

                            # Second fallback: text stored by the JS transcription handler
                            if not content:
                                try:
                                    stored = await ui.run_javascript(
                                        "window._knitraTranscribedText || ''",
                                        timeout=1.0,
                                    )
                                    if stored:
                                        content = str(stored).strip()
                                except Exception:
                                    pass

                            if not content:
                                return

                            # Collect any pending image IDs and clear them atomically
                            image_ids: list[int] = []
                            try:
                                raw_ids = await ui.run_javascript(
                                    "(window._knitraPendingImageIds || []).slice()",
                                    timeout=2.0,
                                )
                                if isinstance(raw_ids, list):
                                    image_ids = [int(x) for x in raw_ids if str(x).isdigit()]
                            except Exception:
                                pass

                            chat_input.set_value("")
                            # Clear JS-side state + toolbar status/thumbs strip
                            await ui.run_javascript(
                                "window._knitraTranscribedText = '';"
                                "window._knitraPendingImageIds = [];"
                                "var _st=document.getElementById('knitra-chat-status');"
                                "if(_st)_st.textContent='';"
                                "var _th=document.getElementById('knitra-chat-thumbs');"
                                "if(_th)_th.innerHTML='';"
                            )

                            with chat_messages:
                                # Show attachment notice if photos were attached
                                if image_ids:
                                    n = len(image_ids)
                                    noun = "photo" if n == 1 else "photos"
                                    ui.html(
                                        f'<div class="chat-bubble-user" '
                                        f'style="font-style:italic;opacity:0.75;">'
                                        f'&#128247; You attached {n} {noun}</div>'
                                    )
                                ui.html(
                                    f'<div class="chat-bubble-user">{content}</div>'
                                )
                                typing = ui.html(
                                    '<div class="chat-bubble-typing">'
                                    'Knitra is thinking&#8230;</div>'
                                )

                            try:
                                async with httpx.AsyncClient() as c:
                                    r = await c.post(
                                        f"{API}/cases/{case_id}/chat",
                                        json={"content": content, "image_ids": image_ids},
                                        headers={"Authorization": f"Bearer {token}"},
                                        timeout=120.0,
                                    )
                                typing.delete()
                                if r.status_code == 200:
                                    reply = r.json()
                                    with chat_messages:
                                        ui.html(
                                            f'<div class="chat-bubble-assistant">'
                                            f'{reply["content"]}</div>'
                                        )
                                elif r.status_code == 503:
                                    with chat_messages:
                                        ui.html(
                                            '<div class="chat-bubble-assistant" '
                                            'style="color:#dc2626;">'
                                            'AI model unavailable &#8212; '
                                            'is Ollama running?</div>'
                                        )
                                else:
                                    with chat_messages:
                                        ui.html(
                                            f'<div class="chat-bubble-assistant" '
                                            f'style="color:#dc2626;">'
                                            f'Error {r.status_code}</div>'
                                        )
                            except Exception:
                                typing.delete()
                                with chat_messages:
                                    ui.html(
                                        '<div class="chat-bubble-assistant" '
                                        'style="color:#dc2626;">'
                                        'Cannot reach the backend.</div>'
                                    )

                        # Chat mic + photo upload JS (scoped to chat, not the new-case form)
                        ui.add_body_html("""<script>
(function() {
  // Guard: only inject once even if chat section is rebuilt
  if (window._knitraChatJsReady) return;
  window._knitraChatJsReady = true;

  window._knitraChatRecorder    = null;
  window._knitraChatAudio       = [];
  window._knitraChatTimerInt    = null;   // interval ID for the recording clock
  window._knitraTranscribedText = '';     // last mic transcription — read by Python _send_chat
  window._knitraPendingImageIds = [];     // upload IDs queued for next chat send

  function _chatStatus(msg, color) {
    const el = document.getElementById('knitra-chat-status');
    if (el) { el.textContent = msg; el.style.color = color || '#94a3b8'; }
  }

  function _chatTimerStart() {
    let secs = 0;
    window._knitraChatTimerInt = setInterval(function() {
      secs++;
      const m = Math.floor(secs / 60).toString().padStart(2, '0');
      const s = (secs % 60).toString().padStart(2, '0');
      _chatStatus('Recording ' + m + ':' + s + ' — tap mic to stop', '#6366f1');
    }, 1000);
  }

  function _chatTimerStop() {
    if (window._knitraChatTimerInt) {
      clearInterval(window._knitraChatTimerInt);
      window._knitraChatTimerInt = null;
    }
  }

  // Show small thumbnails below the toolbar when photos are attached to the chat
  function _chatShowThumbs(files) {
    let strip = document.getElementById('knitra-chat-thumbs');
    if (!strip) {
      strip = document.createElement('div');
      strip.id = 'knitra-chat-thumbs';
      strip.style.cssText = 'display:flex;gap:6px;flex-wrap:wrap;margin-top:6px;';
      const statusEl = document.getElementById('knitra-chat-status');
      if (statusEl && statusEl.parentNode) {
        statusEl.parentNode.insertBefore(strip, statusEl.nextSibling);
      }
    }
    strip.innerHTML = '';
    files.forEach(function(file) {
      const url = URL.createObjectURL(file);
      const img = document.createElement('img');
      img.src = url;
      img.style.cssText = 'width:48px;height:48px;object-fit:cover;border-radius:6px;' +
                           'border:1.5px solid #bbf7d0;flex-shrink:0;';
      img.onload = function() { URL.revokeObjectURL(url); };
      strip.appendChild(img);
    });
  }

  function _chatClearThumbs() {
    const strip = document.getElementById('knitra-chat-thumbs');
    if (strip) strip.innerHTML = '';
  }

  window.knitraChatToggleVoice = async function() {
    const btn = document.getElementById('knitra-chat-mic');

    // Already recording → stop
    if (window._knitraChatRecorder && window._knitraChatRecorder.state === 'recording') {
      window._knitraChatRecorder.stop();
      return;
    }

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      _chatStatus('Mic blocked — allow access in browser settings.', '#dc2626');
      return;
    }

    const mime = ['audio/webm;codecs=opus','audio/webm','audio/ogg;codecs=opus','audio/mp4']
      .find(t => MediaRecorder.isTypeSupported(t)) || '';
    window._knitraChatAudio = [];
    let rec;
    try { rec = new MediaRecorder(stream, mime ? { mimeType: mime } : {}); }
    catch(e) { rec = new MediaRecorder(stream); }
    window._knitraChatRecorder = rec;

    rec.ondataavailable = function(e) {
      if (e.data && e.data.size > 0) window._knitraChatAudio.push(e.data);
    };

    rec.onstop = async function() {
      _chatTimerStop();
      stream.getTracks().forEach(t => t.stop());
      if (btn) { btn.classList.remove('active'); }
      const blob = new Blob(window._knitraChatAudio, { type: mime || 'audio/webm' });
      if (blob.size < 1000) { _chatStatus('Nothing recorded — try again.', '#94a3b8'); return; }
      _chatStatus('Transcribing…', '#0369a1');
      const fd = new FormData();
      fd.append('audio', blob, 'recording.webm');
      try {
        const resp = await fetch('http://127.0.0.1:8000/api/v1/transcribe', {
          method: 'POST',
          headers: { 'Authorization': 'Bearer ' + (window._knitraToken || '') },
          body: fd,
        });
        if (!resp.ok) throw new Error('Server error ' + resp.status);
        const { text } = await resp.json();
        if (text) {
          const trimmed = text.trim();
          // 1. Store in window variable so Python _send_chat() can read it as fallback
          window._knitraTranscribedText = (window._knitraTranscribedText
            ? window._knitraTranscribedText.trimEnd() + ' ' : '') + trimmed;
          // 2. Also inject into the DOM textarea so user can see / edit it
          const ta = document.querySelector('.knitra-chat-textarea textarea');
          if (ta) {
            ta.value = (ta.value ? ta.value.trimEnd() + ' ' : '') + trimmed;
            // Dispatch both 'input' and 'change' — Quasar listens to 'input'
            ta.dispatchEvent(new Event('input', { bubbles: true }));
            ta.dispatchEvent(new Event('change', { bubbles: true }));
            ta.focus();
          }
          _chatStatus('✔ Transcribed — press Send', '#15803d');
          setTimeout(function() { _chatStatus('', ''); }, 4000);
        } else {
          _chatStatus('Nothing heard — try again.', '#94a3b8');
        }
      } catch(err) {
        _chatStatus('Transcription failed: ' + err.message, '#dc2626');
      }
    };

    rec.start(200);
    if (btn) { btn.classList.add('active'); }
    _chatStatus('Recording 00:00 — tap mic to stop', '#6366f1');
    _chatTimerStart();
  };

  window.knitraChatUpload = async function(input) {
    const files = Array.from(input.files);
    if (!files.length) return;
    const caseId = window._knitraCaseId;
    const token  = window._knitraToken;
    if (!caseId || !token) { _chatStatus('Upload error: not authenticated.', '#dc2626'); return; }

    // Show thumbnails immediately for instant confirmation
    _chatShowThumbs(files);
    _chatStatus('Uploading ' + files.length + ' file' + (files.length > 1 ? 's' : '') + '…', '#0369a1');

    // Ensure pending ID list exists
    if (!Array.isArray(window._knitraPendingImageIds)) {
      window._knitraPendingImageIds = [];
    }

    let ok = 0, failMsg = '';
    for (const file of files) {
      const fd = new FormData();
      fd.append('file', file);
      try {
        const resp = await fetch('http://127.0.0.1:8000/api/v1/cases/' + caseId + '/uploads', {
          method: 'POST',
          headers: { 'Authorization': 'Bearer ' + token },
          body: fd,
        });
        if (resp.ok) {
          const data = await resp.json();
          // Store the upload ID so _send_chat can forward it to Knitra
          if (data && data.id) {
            window._knitraPendingImageIds.push(data.id);
          }
          ok++;
        } else if (resp.status === 413) {
          failMsg = 'File too large (max 10 MB).';
        } else if (resp.status === 415) {
          failMsg = 'Unsupported file type.';
        } else {
          failMsg = 'Server error ' + resp.status + '.';
        }
      } catch(e) {
        failMsg = 'Network error — check connection.';
      }
    }
    input.value = '';  // reset so same file can be re-selected
    if (ok > 0) {
      const noun = ok > 1 ? 'files' : 'file';
      _chatStatus(ok + ' ' + noun + ' ready — press Send to ask Knitra ✔', '#15803d');
      // Don't auto-clear status/thumbs — let user see what's queued until they send
    } else {
      _chatClearThumbs();
      _chatStatus(failMsg || 'Upload failed — check connection.', '#dc2626');
    }
  };
})();
</script>""")

                        async def _quick_send(text: str):
                            chat_input.set_value(text)
                            await _send_chat()

                        await _load_chat()

            # Initial render of chat section based on current analysis state
            await _render_chat_section(_analysis_complete)

            # ── Nearby Services placeholder ───────────────────────────────────
            with ui.element("div").style(
                "background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);"
                "border-radius:14px;padding:28px;"
                "border:1px solid rgba(255,255,255,0.06);"
            ):
                with ui.element("div").style(
                    "display:flex;align-items:flex-start;gap:16px;flex-wrap:wrap;"
                ):
                    ui.html(
                        '<div style="width:46px;height:46px;'
                        'background:rgba(99,102,241,0.2);border-radius:11px;'
                        'display:flex;align-items:center;justify-content:center;'
                        'font-size:22px;flex-shrink:0;">&#128205;</div>'
                    )
                    with ui.element("div").style("flex:1;min-width:200px;"):
                        with ui.element("div").style(
                            "display:flex;align-items:center;gap:8px;margin-bottom:8px;"
                        ):
                            ui.html(
                                '<span style="color:white;font-size:1rem;'
                                'font-weight:700;">Nearby Services</span>'
                            )
                            ui.html(
                                '<span style="background:rgba(99,102,241,0.3);'
                                'color:#a5b4fc;font-size:10px;padding:2px 9px;'
                                'border-radius:4px;font-weight:700;">Coming Soon</span>'
                            )
                        ui.html(
                            '<p style="color:#94a3b8;font-size:0.875rem;'
                            'line-height:1.65;margin:0 0 16px;">'
                            'Once AI identifies your repair type, we&#8217;ll surface '
                            'local plumbers, electricians, and tradespeople in your area '
                            '&#8212; so you can get expert help when you need it.</p>'
                        )
                        ui.html(
                            '<div style="background:rgba(255,255,255,0.05);'
                            'border:1px solid rgba(255,255,255,0.09);'
                            'border-radius:8px;padding:11px 14px;'
                            'display:flex;align-items:center;gap:10px;'
                            'color:#475569;font-size:0.875rem;cursor:default;">'
                            '<span>&#128205;</span>'
                            '<span>Enter your location to find nearby help&#8230;</span>'
                            '</div>'
                        )


# ─────────────────────────────────────────────────────────────────────────────
# NEW CASE  (/new)  — composer-style quick-start flow
#
# UX: single card with drop zone + thumbnail strip + text note + action bar.
# Images are compressed client-side (JS Canvas, max 1024px, JPEG 82%) before
# bytes are transferred to Python, significantly reducing Stage 1 payload.
#
# Flow:
#   1. User drops / selects up to 10 photos (multi-select native dialog).
#   2. JS compresses each image and stores as base64 in window._fixlyFiles.
#   3. User optionally types a context note.
#   4. Clicks "Analyse" → Python reads files via ui.run_javascript().
#   5. Create case → upload each file → trigger analysis → navigate.
# ─────────────────────────────────────────────────────────────────────────────

@ui.page("/new")
async def new_case_page():
    inject_styles()

    token = app.storage.user.get("token")
    if not token:
        ui.navigate.to("/login")
        return

    # Inject JS for client-side file compress + global state.
    # ui.add_head_html deduplicates identical head tags across page visits.
    ui.add_head_html("""<script>
    // ── Global state ──────────────────────────────────────────────────────────
    window._fixlyFiles    = [];   // compressed image objects {name,type,data}
    window._fixlyToken    = '';   // bearer token, set by deferred timer
    window._fixlyRecorder = null; // MediaRecorder instance while recording
    window._fixlyAudio    = [];   // audio chunks collected during recording

    // ── Image compression (Canvas, max 1024px, JPEG 82%) ─────────────────────
    function _fixlyCompress(file) {
      return new Promise(function(resolve) {
        const reader = new FileReader();
        reader.onload = function(e) {
          const img = new Image();
          img.onload = function() {
            const MAX = 1024;
            let w = img.naturalWidth, h = img.naturalHeight;
            if (Math.max(w, h) > MAX) {
              if (w >= h) { h = Math.round(h * MAX / w); w = MAX; }
              else        { w = Math.round(w * MAX / h); h = MAX; }
            }
            const c = document.createElement('canvas');
            c.width = w; c.height = h;
            c.getContext('2d').drawImage(img, 0, 0, w, h);
            resolve({
              name: file.name.replace(/\\.[^.]+$/, '.jpg'),
              type: 'image/jpeg',
              data: c.toDataURL('image/jpeg', 0.82)
            });
          };
          img.src = e.target.result;
        };
        reader.readAsDataURL(file);
      });
    }

    async function fixlyHandleFiles(fileList) {
      const toAdd = Array.from(fileList)
        .filter(function(f) { return f.type.startsWith('image/'); })
        .slice(0, 10 - window._fixlyFiles.length);
      if (!toAdd.length) return;
      const results = await Promise.all(toAdd.map(_fixlyCompress));
      window._fixlyFiles = window._fixlyFiles.concat(results);
      _fixlyRender();
    }

    // ── Photo area renderer — single container, completely replaced each call ─
    // This avoids any show/hide race between separate DOM elements.
    function _fixlyRender() {
      const area  = document.getElementById('fixly-media-area');
      const badge = document.getElementById('fixly-attach-badge');
      if (!area) return;
      const n = window._fixlyFiles.length;

      // Update badge on the attachment icon
      if (badge) {
        badge.style.display = n > 0 ? 'flex' : 'none';
        badge.textContent = String(n);
      }

      if (n === 0) {
        // ── Empty state: drop zone ────────────────────────────────────────────
        area.innerHTML = `
          <label for="fixly-file-input"
                 ondragover="event.preventDefault();this.querySelector('.fldz').style.borderColor='#6366f1';this.querySelector('.fldz').style.background='#ede9fe';"
                 ondragleave="this.querySelector('.fldz').style.borderColor='#e2e8f0';this.querySelector('.fldz').style.background='#f8fafc';"
                 ondrop="event.preventDefault();this.querySelector('.fldz').style.borderColor='#e2e8f0';this.querySelector('.fldz').style.background='#f8fafc';fixlyHandleFiles(event.dataTransfer.files);"
                 style="cursor:pointer;display:block;padding:0 20px;">
            <div class="fldz"
                 style="display:flex;flex-direction:column;align-items:center;gap:10px;
                        border:2px dashed #e2e8f0;border-radius:12px;
                        padding:36px 20px;margin:20px 0 0;background:#f8fafc;
                        transition:border-color 0.15s,background 0.15s;">
              <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40"
                   viewBox="0 0 24 24" fill="none" stroke="#94a3b8"
                   stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <rect width="18" height="18" x="3" y="3" rx="2"/>
                <circle cx="9" cy="9" r="2"/>
                <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>
              </svg>
              <div style="font-weight:700;color:#334155;font-size:0.9rem;">Tap to add photos</div>
              <div style="color:#94a3b8;font-size:0.78rem;text-align:center;line-height:1.5;">
                Drag &amp; drop &middot; Select multiple &middot; Up to 10 photos</div>
            </div>
          </label>`;
        return;
      }

      // ── Photos attached: green confirmation banner + thumbnails ───────────
      const thumbs = window._fixlyFiles.map((f, i) => `
        <div style="position:relative;flex-shrink:0;">
          <img src="${f.data}"
               style="width:90px;height:90px;object-fit:cover;
                      border-radius:10px;display:block;
                      border:2px solid #bbf7d0;" />
          <button onclick="fixlyRemove(${i})"
                  style="position:absolute;top:-7px;right:-7px;
                         width:22px;height:22px;background:#ef4444;
                         color:white;border:2.5px solid white;border-radius:50%;
                         cursor:pointer;font-size:15px;font-weight:700;
                         display:flex;align-items:center;justify-content:center;
                         line-height:1;padding:0;z-index:10;"
                  title="Remove this photo">&times;</button>
        </div>`).join('');

      area.innerHTML = `
        <div style="background:#f0fdf4;border-bottom:1px solid #bbf7d0;padding:14px 20px;">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;flex-wrap:wrap;">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
                 viewBox="0 0 24 24" fill="none" stroke="#16a34a"
                 stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
              <path d="M20 6 9 17l-5-5"/>
            </svg>
            <span style="font-weight:700;color:#15803d;font-size:0.88rem;">
              ${n} photo${n > 1 ? 's' : ''} attached
            </span>
            <span style="flex:1;"></span>
            <label for="fixly-file-input"
                   style="font-size:0.78rem;color:#6366f1;font-weight:600;
                          cursor:pointer;text-decoration:none;
                          padding:3px 10px;border-radius:6px;
                          border:1px solid #c7d2fe;background:#eef2ff;">
              + Add more
            </label>
            <button onclick="window._fixlyFiles=[];_fixlyRender()"
                    style="font-size:0.78rem;color:#dc2626;font-weight:600;
                           cursor:pointer;padding:3px 10px;border-radius:6px;
                           border:1px solid #fecaca;background:#fef2f2;
                           -webkit-appearance:none;">
              Clear all
            </button>
          </div>
          <div style="display:flex;flex-wrap:wrap;gap:8px;">${thumbs}</div>
        </div>`;
    }

    function fixlyRemove(i) {
      window._fixlyFiles.splice(i, 1);
      _fixlyRender();
    }

    function fixlyGetFiles() {
      return JSON.stringify(window._fixlyFiles);
    }

    function fixlyGetNote() {
      const ta = document.getElementById('fixly-note');
      return ta ? ta.value.trim() : '';
    }

    // ── Voice input via MediaRecorder → backend Whisper ───────────────────────
    // Reliable in ALL browsers (Chrome, Edge, Firefox, Safari 14.1+).
    // Audio is recorded as webm/opus, sent to /api/v1/transcribe,
    // transcribed by faster-whisper on the server, returned as text.

    async function fixlyToggleVoice() {
      const btn = document.getElementById('fixly-mic-btn');

      // If already recording → stop
      if (window._fixlyRecorder && window._fixlyRecorder.state === 'recording') {
        window._fixlyRecorder.stop();
        return;
      }

      // Request microphone access
      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      } catch (err) {
        _fixlyVoiceMsg(
          err.name === 'NotAllowedError'
            ? 'Microphone blocked — allow access in your browser and try again.'
            : 'Could not access microphone: ' + err.message,
          'error'
        );
        return;
      }

      // Pick the best supported MIME type
      const mime = ['audio/webm;codecs=opus','audio/webm','audio/ogg;codecs=opus','audio/mp4']
        .find(t => MediaRecorder.isTypeSupported(t)) || '';

      window._fixlyAudio = [];
      let recorder;
      try {
        recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : {});
      } catch(e) {
        recorder = new MediaRecorder(stream);
      }
      window._fixlyRecorder = recorder;

      recorder.ondataavailable = function(e) {
        if (e.data && e.data.size > 0) window._fixlyAudio.push(e.data);
      };

      recorder.onstop = async function() {
        // Stop all mic tracks immediately
        stream.getTracks().forEach(t => t.stop());
        _fixlySetMicState(false);

        const blob = new Blob(window._fixlyAudio, { type: mime || 'audio/webm' });
        if (blob.size < 2000) {
          _fixlyVoiceMsg('Nothing recorded — tap mic and speak, then tap again to stop.', 'info');
          return;
        }

        _fixlyVoiceMsg('Transcribing…', 'working');

        const fd = new FormData();
        fd.append('audio', blob, 'recording.webm');

        try {
          const resp = await fetch('http://127.0.0.1:8000/api/v1/transcribe', {
            method: 'POST',
            headers: { 'Authorization': 'Bearer ' + window._fixlyToken },
            body: fd,
          });
          if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || 'Server error ' + resp.status);
          }
          const { text } = await resp.json();
          if (text) {
            const ta = document.getElementById('fixly-note');
            if (ta) {
              ta.value = (ta.value ? ta.value.trimEnd() + ' ' : '') + text.trim();
              ta.style.height = 'auto';
              ta.style.height = ta.scrollHeight + 'px';
            }
            _fixlyVoiceMsg('', '');          // clear on success
          } else {
            _fixlyVoiceMsg('Nothing heard — try speaking more clearly or type instead.', 'info');
          }
        } catch (err) {
          _fixlyVoiceMsg('Transcription failed: ' + err.message, 'error');
        }
      };

      recorder.start(200);   // emit data every 200 ms
      _fixlySetMicState(true);
      _fixlyVoiceMsg('Recording… tap the mic again to stop', 'recording');
    }

    function _fixlySetMicState(active) {
      const btn = document.getElementById('fixly-mic-btn');
      if (!btn) return;
      if (active) {
        btn.classList.add('active');
        btn.title = 'Tap to stop recording';
      } else {
        btn.classList.remove('active');
        btn.title = 'Voice input — tap and speak';
      }
    }

    function _fixlyVoiceMsg(msg, type) {
      const el = document.getElementById('fixly-voice-preview');
      if (!el) return;
      el.textContent = msg;
      el.style.color =
        type === 'error'     ? '#dc2626' :
        type === 'recording' ? '#6366f1' :
        type === 'working'   ? '#0369a1' :
        '#94a3b8';
    }
    </script>""")

    mobile_bar()
    sidebar("/new")

    with ui.element("div").classes("main"):
        with ui.element("div").style(
            "padding:32px;max-width:660px;"
            "display:flex;flex-direction:column;gap:20px;"
        ):
            # Back
            back = ui.element("div").style(
                "display:inline-flex;align-items:center;gap:6px;"
                "color:#6366f1;font-size:0.875rem;font-weight:600;"
                "cursor:pointer;width:fit-content;"
            )
            with back:
                ui.html("&#8592; Back to My Cases")
            back.on("click", lambda: ui.navigate.to("/cases"))

            # Header
            ui.html(
                '<h1 style="font-size:1.6rem;font-weight:800;color:#0f172a;'
                'margin:0 0 2px;letter-spacing:-0.02em;">Diagnose a repair</h1>'
                '<p style="color:#94a3b8;font-size:0.875rem;margin:0;">'
                'Snap a photo of the problem &#8212; Knitra identifies it and '
                'writes your repair guide automatically.</p>'
            )

            # ── Single composer card ───────────────────────────────────────────
            with ui.element("div").style(
                "background:white;border:1px solid #e2e8f0;"
                "border-radius:16px;overflow:hidden;"
                "box-shadow:0 2px 12px rgba(0,0,0,0.04);"
            ):
                # Hidden file input — label elements trigger it natively (no JS .click() needed)
                ui.html(
                    '<input type="file" id="fixly-file-input" multiple accept="image/*"'
                    ' style="position:absolute;width:1px;height:1px;opacity:0;pointer-events:none;"'
                    ' onchange="fixlyHandleFiles(this.files);this.value=\'\'">'
                )

                # Single media area — _fixlyRender() rewrites its innerHTML entirely.
                # Either shows the drop zone (empty) or the green attachment panel (photos selected).
                ui.html('<div id="fixly-media-area"></div>')

                # ── Divider ────────────────────────────────────────────────────
                ui.html(
                    '<div style="height:1px;background:#f1f5f9;margin:16px 0 0;"></div>'
                )

                # ── Note textarea — plain HTML so voice can write into it ──────
                ui.html("""
                <div style="padding:8px 16px 4px;">
                  <textarea
                    id="fixly-note"
                    rows="2"
                    placeholder="Describe the issue (optional) — e.g. dripping pipe under the kitchen sink…"
                    style="width:100%;box-sizing:border-box;border:none;outline:none;
                           resize:none;font-size:0.9rem;color:#334155;
                           font-family:inherit;background:transparent;
                           min-height:52px;line-height:1.6;padding:0;"
                    oninput="this.style.height='auto';this.style.height=this.scrollHeight+'px';"
                  ></textarea>
                  <div id="fixly-voice-preview"
                       style="color:#94a3b8;font-size:0.78rem;font-style:italic;
                              min-height:14px;"></div>
                </div>""")

                # ── Bottom action bar — all pure HTML for reliable gesture handling
                ui.html("""
                <div style="padding:10px 16px 14px;display:flex;
                            align-items:center;gap:8px;">

                  <!-- Attachment icon with photo count badge -->
                  <label for="fixly-file-input" class="fixly-icon-btn"
                         title="Add photos" style="position:relative;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18"
                         viewBox="0 0 24 24" fill="none" stroke="currentColor"
                         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                      <path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57
                               A4 4 0 1 1 18 8.84l-8.59 8.57a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
                    </svg>
                    <!-- Badge: shown when photos are attached -->
                    <span id="fixly-attach-badge"
                          style="display:none;position:absolute;top:-6px;right:-6px;
                                 width:17px;height:17px;background:#6366f1;color:white;
                                 border-radius:50%;font-size:10px;font-weight:700;
                                 align-items:center;justify-content:center;
                                 border:2px solid white;line-height:1;">0</span>
                  </label>

                  <!-- Mic: native <button> → always a proper user gesture for getUserMedia -->
                  <button id="fixly-mic-btn" class="fixly-icon-btn"
                          onclick="fixlyToggleVoice()"
                          title="Voice input — tap and speak">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18"
                         viewBox="0 0 24 24" fill="none" stroke="currentColor"
                         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                      <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
                      <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                      <line x1="12" x2="12" y1="19" y2="22"/>
                    </svg>
                  </button>

                  <!-- Video — coming soon -->
                  <div class="fixly-icon-btn-soon" title="Video support coming soon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18"
                         viewBox="0 0 24 24" fill="none" stroke="currentColor"
                         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                      <path d="m22 8-6 4 6 4V8Z"/>
                      <rect width="14" height="12" x="2" y="6" rx="2" ry="2"/>
                    </svg>
                  </div>

                  <div style="flex:1;"></div>
                </div>
                """)

                # Submit button — stays NiceGUI (no file/mic picker needed here)
                submit_btn = ui.button("Analyse →") \
                    .props("unelevated color=primary") \
                    .style(
                        "font-weight:700;font-size:0.9rem;padding:0 22px;"
                        "border-radius:10px;letter-spacing:0.01em;"
                    )

            # ── Error label ───────────────────────────────────────────────────
            err_label = ui.label("").style(
                "color:#dc2626;font-size:0.875rem;min-height:20px;"
            )

            # ── Inline progress area (populated after clicking Analyse) ───────
            prog_area = ui.element("div")

            # ── Trust badges ──────────────────────────────────────────────────
            ui.html(
                '<div style="display:flex;gap:20px;flex-wrap:wrap;'
                'color:#94a3b8;font-size:0.775rem;">'
                '<span>&#128274; Photos stay on your device</span>'
                '<span>&#129302; 3 AI models, one plan</span>'
                '<span>&#128295; Tools, steps &amp; safety notes</span>'
                '</div>'
            )

    # ── Handler — defined after UI so all element refs are in scope ───────────
    async def _start_diagnosis():
        import base64 as _b64

        err_label.set_text("")

        # Pull compressed files from JS global
        raw_json = await ui.run_javascript("return fixlyGetFiles()")
        try:
            files = json.loads(raw_json or "[]")
        except Exception:
            files = []

        if not files:
            err_label.set_text(
                "Please add at least one photo before analysing."
            )
            return

        submit_btn.props("loading disable")

        # Build inline progress
        prog_area.clear()
        n = len(files)
        _texts = {
            "case":    "Creating your case",
            "upload":  f"Uploading {n} photo{'s' if n > 1 else ''}",
            "analyse": "Starting AI analysis",
        }
        _lbls = {}
        with prog_area:
            ui.element("div").style("height:6px;")
            with ui.element("div").style(
                "background:white;border:1px solid #e2e8f0;"
                "border-radius:12px;padding:18px 20px;"
            ):
                ui.html(
                    '<div style="font-weight:700;color:#334155;'
                    'font-size:0.855rem;margin-bottom:10px;">'
                    'Starting your diagnosis&#8230;</div>'
                )
                for k, t in _texts.items():
                    _lbls[k] = ui.label(f"○  {t}").classes("step-pending")

        def _active(k):
            _lbls[k].classes(remove="step-pending step-done", add="step-active")
            _lbls[k].set_text(f"↻  {_texts[k]}")

        def _done(k):
            _lbls[k].classes(remove="step-pending step-active", add="step-done")
            _lbls[k].set_text(f"✓  {_texts[k]}")

        # Step 1 — create case
        _active("case")
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    f"{API}/cases",
                    json={"title": "Diagnosing…"},
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=10.0,
                )
            if r.status_code == 401:
                app.storage.user.clear()
                ui.navigate.to("/login")
                return
            if r.status_code != 201:
                err_label.set_text(f"Could not create case ({r.status_code}).")
                submit_btn.props(remove="loading disable")
                return
            cid = r.json()["id"]
            _done("case")
        except Exception as exc:
            err_label.set_text(f"Backend error: {exc}")
            submit_btn.props(remove="loading disable")
            return

        # Step 2 — upload files (already compressed by JS)
        _active("upload")
        uploaded = 0
        for i, f in enumerate(files):
            _lbls["upload"].set_text(f"↻  Uploading photo {i + 1} of {n}…")
            try:
                raw_b64 = f["data"].split(",", 1)[1]
                img_bytes = _b64.b64decode(raw_b64)
                async with httpx.AsyncClient() as client:
                    r_up = await client.post(
                        f"{API}/cases/{cid}/uploads",
                        files={"file": (f["name"], img_bytes, f["type"])},
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=30.0,
                    )
                if r_up.status_code in (200, 201):
                    uploaded += 1
            except Exception:
                pass
        if uploaded == 0:
            err_label.set_text("Could not upload any photos — check your connection and try again.")
            submit_btn.props(remove="loading disable")
            return
        _done("upload")

        # Step 3 — trigger analysis
        _active("analyse")
        text_ctx = (await ui.run_javascript("return fixlyGetNote()") or "").strip() or None
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{API}/cases/{cid}/analyse",
                    json={"text_context": text_ctx} if text_ctx else {},
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=15.0,
                )
        except Exception:
            pass
        _done("analyse")

        # Brief pause so user sees all-green, then navigate
        await asyncio.sleep(0.7)
        await ui.run_javascript("window._fixlyFiles = []")
        ui.navigate.to(f"/cases/{cid}")

    submit_btn.on("click", _start_diagnosis)

    # Defer the JS reset until AFTER the WebSocket is ready.
    # (calling ui.run_javascript at page-init time races against the WS handshake
    #  and causes a TimeoutError; a short timer fires once the connection exists.)
    async def _reset_js_store():
        """
        Fires 400ms after page build — by then the WebSocket is open.
        Resets the file list, exposes the auth token to JS (needed for the
        Whisper transcription fetch call), and renders the empty drop zone.
        """
        tok = app.storage.user.get("token", "")
        try:
            await ui.run_javascript(
                f"window._fixlyFiles=[];"
                f"window._fixlyToken={json.dumps(tok)};"
                "if(typeof _fixlyRender==='function') _fixlyRender();",
                timeout=3.0,
            )
        except Exception:
            pass

    ui.timer(0.4, _reset_js_store, once=True)


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

ui.run(
    port=3000,
    title="Knitra",
    reload=False,
    storage_secret="fixly-dev-secret",
)
