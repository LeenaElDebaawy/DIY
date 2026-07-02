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
    "needs_clarification":{"stripe": "#f59e0b", "bg": "#fef3c7", "fg": "#78350f", "label": "Needs more info"},
    "needs_info":         {"stripe": "#f59e0b", "bg": "#fff7ed", "fg": "#92400e", "label": "Needs answers"},
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
        because Material Icons is a font-face, each character IS an icon.
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

      /* ── Video card in the case detail media grid ── */
      .video-thumb {
        position: relative;
        flex-shrink: 0;
        border-radius: 10px;
        border: 2px solid #6366f1;
        overflow: hidden;
        background: #0f172a;
      }
      .video-thumb video {
        display: block;
        width: 100%;
        height: 100%;
        object-fit: contain;
      }
      .video-thumb .video-badge {
        position: absolute;
        top: 4px;
        left: 4px;
        background: rgba(99,102,241,0.92);
        color: #fff;
        font-size: 0.58rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        padding: 2px 6px;
        border-radius: 4px;
        pointer-events: none;
        text-transform: uppercase;
      }

      /* ── Hero (homepage), must fill the full viewport width ── */
      .hero {
        width: 100%;
        background: linear-gradient(140deg, #0f172a 0%, #1e293b 55%, #1b3158 100%);
      }

      /* ────────────────────────────────────────────────────────
         UPLOAD WIDGET OVERRIDES (Quasar q-uploader)

         Problems we fix:
           1. "0.0B / 0.00%" subtitle, hidden
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
      /* Hide the internal file list, we manage our own photo grid */
      .q-uploader__list { display: none !important; }

      /* ── Analysis progress dot pulse ── */
      @keyframes fixly-pulse {
        0%, 100% { opacity: 1;   transform: scale(1); }
        50%       { opacity: 0.4; transform: scale(0.75); }
      }
      .fixly-pulse { animation: fixly-pulse 1.3s ease-in-out infinite; }

      /* ── Analysis result sections ── */
      .analysis-section-label {
        font-size: 0.8rem; font-weight: 800; letter-spacing: 0.06em;
        text-transform: uppercase; color: #334155;
        border-left: 3px solid #6366f1; padding-left: 9px;
        margin: 4px 0 10px;
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

      /* ── Mic recording state: solid red + pulsing halo so it is impossible
            to miss that the mic is live and tapping again will stop it ── */
      @keyframes knitra-rec-pulse {
        0%   { box-shadow:0 0 0 0 rgba(220,38,38,0.55); }
        70%  { box-shadow:0 0 0 8px rgba(220,38,38,0); }
        100% { box-shadow:0 0 0 0 rgba(220,38,38,0); }
      }
      .fixly-icon-btn.recording {
        background:#dc2626 !important; color:#ffffff !important;
        border-color:#dc2626 !important;
        animation:knitra-rec-pulse 1.25s ease-out infinite;
      }

      /* ── Chat composer status pill (recording / transcribing / done) ── */
      .knitra-status-pill {
        display:inline-flex; align-items:center; gap:6px;
        font-size:0.78rem; font-weight:600; line-height:1;
        padding:5px 11px; border-radius:100px; white-space:nowrap;
      }

      /* ── Pending-attachment chips above the textarea ── */
      .knitra-attach-wrap {
        display:flex; flex-direction:column; gap:6px; margin-bottom:8px;
      }
      .knitra-attach-head {
        font-size:0.72rem; font-weight:700; color:#1e40af;
        display:flex; align-items:center; gap:5px;
      }
      .knitra-attach-strip { display:flex; flex-wrap:wrap; gap:8px; }
      .knitra-attach-chip {
        position:relative; display:flex; align-items:center; gap:8px;
        background:#eff6ff; border:1px solid #bfdbfe; border-radius:10px;
        padding:6px 30px 6px 6px; max-width:230px;
      }
      .knitra-attach-chip img {
        width:40px; height:40px; object-fit:cover; border-radius:7px;
        flex-shrink:0; background:#dbeafe;
      }
      .knitra-attach-meta { display:flex; flex-direction:column; gap:1px; min-width:0; }
      .knitra-attach-name {
        font-size:0.76rem; font-weight:600; color:#1e3a8a;
        white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:130px;
      }
      .knitra-attach-state { font-size:0.68rem; font-weight:600; }
      .knitra-attach-x {
        position:absolute; top:4px; right:5px; width:18px; height:18px;
        border:none; border-radius:50%; background:#dbeafe; color:#1e40af;
        cursor:pointer; font-size:12px; line-height:1; padding:0;
        display:flex; align-items:center; justify-content:center;
      }
      .knitra-attach-x:hover { background:#bfdbfe; }

      /* ── Brief highlight when a voice transcript is inserted ── */
      @keyframes knitra-flash {
        0%   { background:#dcfce7; }
        100% { background:#ffffff; }
      }
      .knitra-text-flash textarea { animation:knitra-flash 1.4s ease-out; }

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
        transition: max-height 0.2s ease;
      }
      .chat-wrap.expanded { max-height: 760px; }
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

            # Admin-only: feedback / judge / quality dashboards (UX gate; server enforces via require_admin).
            if app.storage.user.get("is_admin", False):
                nav("💬", "Feedback", "/admin/feedback")
                nav("🏛", "Judge", "/admin/judge")
                nav("📊", "Quality", "/admin/quality")

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
# TOP NAV  (landing / auth pages, no sidebar)
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

    # width:100% is required, without it NiceGUI's flex container
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
                'instructions, from plumbing to electronics.</p>'
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
                            app.storage.user["is_admin"] = d["user"].get("is_admin", False)
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
                            app.storage.user["is_admin"] = d["user"].get("is_admin", False)
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
                    "needs_clarification": "Needs more info",
                    "needs_info":          "Needs answers",
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

            # All fetched cases, stored so filters can re-render without a new fetch
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
                                    with ui.element("div").style(
                                        "display:flex;flex-direction:column;"
                                        "align-items:flex-end;gap:4px;flex-shrink:0;"
                                    ):
                                        ui.html(
                                            f'<span style="background:{s["bg"]};color:{s["fg"]};'
                                            f'padding:3px 11px;border-radius:100px;font-size:11.5px;'
                                            f'font-weight:600;white-space:nowrap;">'
                                            f'{s["label"]}</span>'
                                        )
                                        # Awaiting Acknowledgement badge
                                        if (
                                            case.get("output_mode") == "assessment_first"
                                            and not case.get("hazard_ack", False)
                                        ):
                                            ui.html(
                                                '<span style="background:#fef3c7;color:#92400e;'
                                                'border:1px solid #f59e0b;'
                                                'padding:3px 11px;border-radius:100px;'
                                                'font-size:11px;font-weight:600;white-space:nowrap;">'
                                                '&#9888; Awaiting Acknowledgement</span>'
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
# Stage 2 recommendation tiers
# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 prefixes each tool/material with a tier tag, e.g. "[Confirmed] screwdriver".
# _tier_split() turns that leading tag into a coloured chip. Strings without a
# recognised tag (older analyses) render unchanged, so this is fully backward-compatible.
# tier: (label, background, text colour, icon, tooltip). Icons + tooltips make the
# meaning clear without a legend; "Don't buy yet" uses a neutral slate (not alarming
# red) so it reads as "hold off", not "error".
_TIER_CHIPS = {
    "confirmed":      ("Confirmed",     "#dcfce7", "#15803d", "✓",
                       "The photo clearly shows this is needed."),
    "likely":         ("Likely",        "#dbeafe", "#1d4ed8", "≈",
                       "Standard for this fault, but not confirmed from the photo."),
    "verify first":   ("Verify first",  "#fef3c7", "#b45309", "?",
                       "Check the size, type or material before buying this."),
    "do not buy yet": ("Don't buy yet", "#f1f5f9", "#475569", "⏳",
                       "Only buy this if a later check confirms it is needed."),
}


def _tier_split(item: str):
    """Split '[Confirmed] screwdriver' -> (chip_html, 'screwdriver').

    Returns ('', original) when there is no recognised leading [tier] tag.
    """
    s = (item or "").strip()
    if s.startswith("[") and "]" in s:
        tag = s[1:s.index("]")].strip().lower()
        meta = _TIER_CHIPS.get(tag)
        if meta:
            label, bg, fg, icon, tip = meta
            rest = s[s.index("]") + 1:].strip()
            chip = (
                f'<span title="{tip}" style="display:inline-block;flex-shrink:0;background:{bg};'
                f'color:{fg};font-size:0.64rem;font-weight:700;letter-spacing:0.02em;'
                f'text-transform:uppercase;border-radius:5px;padding:1px 7px;'
                f'margin-right:6px;white-space:nowrap;cursor:help;">{icon}&nbsp;{label}</span>'
            )
            return chip, rest
    return "", s


# ─────────────────────────────────────────────────────────────────────────────
# CASE DETAIL  (/cases/{case_id})
# ─────────────────────────────────────────────────────────────────────────────

def _clean_chat(s: str) -> str:
    """Convert streamed LLM reply to safe display HTML.

    Order matters:
      1. Normalise em/en dashes (plain text still, no HTML yet)
      2. HTML-escape & < > (makes the string safe)
      3. Convert markdown inline syntax to HTML tags
      4. Convert line breaks to <br>
    """
    import re as _re
    if not s:
        return s
    # 1. Normalise dashes
    for d in (chr(0x2014), chr(0x2013), chr(0x2015)):
        s = s.replace(" " + d + " ", ", ").replace(d, "-")
    # 2. HTML-escape (must happen before any HTML is injected)
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # 3. Markdown inline: bold then italic (bold first avoids ** / * overlap)
    s = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s, flags=_re.DOTALL)
    s = _re.sub(r'\*([^*\n]+?)\*', r'<em>\1</em>', s)
    # 4. Line-by-line: bullets and headings
    lines = s.split('\n')
    out: list[str] = []
    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith('- ') or stripped.startswith('* '):
            out.append('&bull;&nbsp;' + stripped[2:])
        elif stripped.startswith('# '):
            out.append('<strong>' + stripped[2:] + '</strong>')
        elif stripped.startswith('## ') or stripped.startswith('### '):
            out.append('<strong>' + stripped.lstrip('#').strip() + '</strong>')
        else:
            out.append(ln)
    s = '<br>'.join(out)
    # Collapse 3+ consecutive <br> to a double break
    s = _re.sub(r'(<br>){3,}', '<br><br>', s)
    return s


def _chat_user_html(text: str, img_srcs: list | None = None) -> str:
    """User chat bubble HTML, with optional image thumbnails above the text.

    img_srcs may be data: URLs (live send) or /static/uploads URLs (reload).
    """
    thumbs = ""
    if img_srcs:
        thumbs = (
            '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:6px;">'
            + "".join(
                f'<img src="{s}" alt="attached photo" '
                'style="width:88px;height:88px;object-fit:cover;border-radius:8px;'
                'display:block;border:1px solid rgba(255,255,255,0.35);" />'
                for s in img_srcs
            )
            + "</div>"
        )
    return f'<div class="chat-bubble-user">{thumbs}{text}</div>'


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
                                # Awaiting Acknowledgement badge — assessment_first, unacknowledged
                                if (
                                    case.get("output_mode") == "assessment_first"
                                    and not case.get("hazard_ack", False)
                                ):
                                    ui.html(
                                        '<span style="background:#fef3c7;color:#92400e;'
                                        'border:1px solid #f59e0b;'
                                        'padding:4px 12px;border-radius:100px;'
                                        'font-size:12px;font-weight:600;">'
                                        '&#9888; Awaiting Acknowledgement</span>'
                                    )
                                ui.html(
                                    f'<span style="color:#cbd5e1;font-size:12px;">'
                                    f'Opened {case["created_at"][:10]}</span>'
                                )
                        ui.button("Delete case", on_click=del_dlg.open) \
                            .props("flat color=negative icon=delete") \
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

                # Photo grid, 120 px thumbnails, CSS background-image for cover crop
                photo_grid = ui.element("div").style(
                    "display:flex;flex-wrap:wrap;gap:12px;margin-bottom:20px;"
                )

                def _render_grid(ups: list):
                    """Clear and repopulate the media grid (images + videos)."""
                    photo_grid.clear()
                    with photo_grid:
                        if not ups:
                            ui.html(
                                '<div style="width:100%;padding:24px 0;'
                                'display:flex;flex-direction:column;align-items:center;'
                                'gap:8px;color:#cbd5e1;">'
                                '<div style="font-size:2rem;opacity:0.4;">&#128247;</div>'
                                '<div style="font-size:0.8rem;font-style:italic;">'
                                'No photos yet, upload the first one below.</div>'
                                '</div>'
                            )
                        else:
                            for up in ups:
                                source_type = up.get("source_type", "image")
                                url = f"{BACKEND}/static/uploads/{up['storage_path']}"

                                if source_type == "video":
                                    # Use the first generated thumbnail as the poster
                                    # frame so the video card shows a preview image
                                    # even before the user presses play.
                                    thumbs = up.get("thumbnail_paths") or []
                                    poster_attr = ""
                                    if thumbs:
                                        poster_url = f"{BACKEND}/static/uploads/{thumbs[0]}"
                                        poster_attr = f' poster="{poster_url}"'
                                    ui.html(
                                        f'<div class="video-thumb" style="width:200px;height:120px;">'
                                        f'<video src="{url}"{poster_attr} controls preload="metadata">'
                                        f'</video>'
                                        f'<div class="video-badge">&#9654; Video</div>'
                                        f'</div>'
                                    )
                                else:
                                    # Image: existing behaviour, unchanged.
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

                # The upload widget, CSS in inject_styles() removes the "0.0B / 0.00%"
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
            #   _draw_no_analysis() , no analysis exists yet (404)
            #   _draw_running(data) , pipeline is pending or running
            #   _draw_complete(data), all three stages finished successfully
            #   _draw_failed(data)  , pipeline raised an exception
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
                "stage_a_running":        "Preparing safety assessment&#8230;",
                "stage_a_complete":       "Assessment ready",
                "stage_c_running":        "Preparing care advice&#8230;",
                "stage_c_complete":       "Care advice ready",
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
                "stage_a_running":   60,
                "stage_a_complete":  95,
                "stage_c_running":   60,
                "stage_c_complete":  95,
                "complete":          100,
            }

            # ── Backend helpers ───────────────────────────────────────────────

            async def _get_analysis():
                """GET /analysis, returns dict or None (404 / error)."""
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
                """GET /cases/{case_id}, returns dict or None."""
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
                """POST /analyse, returns (ok: bool, err_msg: str | None).

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

                # ── Content area, cleared + redrawn by _redraw() ─────────────
                analysis_box = ui.element("div")

            # ── Rendering helpers ─────────────────────────────────────────────

            def _draw_no_analysis():
                """Empty state, no analysis has been triggered yet."""
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
                """Progress view — adapts step labels to the pipeline branch being taken.

                output_mode is written to stage_outputs after Stage 1 completes and sets
                the mode. Before that it is absent, so we fall back to the DIY step set
                (reasonable assumption; it flips immediately after Stage 1 if the mode
                differs). All four modes produce a visually consistent card.
                """
                stage = data.get("pipeline_stage")

                # output_mode is available in stage_outputs once Stage 1 completes.
                output_mode = (data.get("stage_outputs") or {}).get("output_mode", "")

                # ── Mode-specific step definitions ────────────────────────────────
                # Each step tuple: (label, emoji, active_from_idx, done_from_idx)
                # _IDX maps pipeline_stage → current progress index for this mode.
                # _PCT maps pipeline_stage → progress bar percentage for this mode.

                if output_mode == "assessment_first":
                    _TOP = {
                        None:               "Warming up&#8230;",
                        "stage_1_running":  "Reviewing your photos&#8230;",
                        "stage_1_complete": "Safety concerns identified&#8230;",
                        "stage_a_running":  "Preparing safety assessment&#8230;",
                        "stage_a_complete": "Assessment ready, finalising&#8230;",
                    }
                    _STEPS5 = [
                        ("Reviewing your photos",          "🔍", 1, 2),
                        ("Identifying safety concerns",    "⚠️", 2, 3),
                        ("Preparing safety assessment",    "🛑", 3, 5),
                        ("Finalising assessment",          "✅", 5, 6),
                    ]
                    _IDX = {
                        None: 0,
                        "stage_1_running": 1, "stage_1_complete": 2,
                        "stage_a_running": 3, "stage_a_complete": 4,
                        "complete": 5,
                    }
                    _PCT = {
                        None: 6, "stage_1_running": 20, "stage_1_complete": 42,
                        "stage_a_running": 64, "stage_a_complete": 92, "complete": 100,
                    }

                elif output_mode == "care":
                    _TOP = {
                        None:               "Warming up&#8230;",
                        "stage_1_running":  "Reviewing your photos&#8230;",
                        "stage_1_complete": "Condition identified&#8230;",
                        "stage_c_running":  "Preparing care advice&#8230;",
                        "stage_c_complete": "Care advice ready, finalising&#8230;",
                    }
                    _STEPS5 = [
                        ("Reviewing your photos",        "🔍", 1, 2),
                        ("Identifying the condition",    "🧐", 2, 3),
                        ("Preparing care advice",        "🌿", 3, 5),
                        ("Finalising recommendations",   "✅", 5, 6),
                    ]
                    _IDX = {
                        None: 0,
                        "stage_1_running": 1, "stage_1_complete": 2,
                        "stage_c_running": 3, "stage_c_complete": 4,
                        "complete": 5,
                    }
                    _PCT = {
                        None: 6, "stage_1_running": 20, "stage_1_complete": 42,
                        "stage_c_running": 64, "stage_c_complete": 92, "complete": 100,
                    }

                elif output_mode == "clarification":
                    _TOP = {
                        None:               "Warming up&#8230;",
                        "stage_1_running":  "Reviewing your photos&#8230;",
                        "stage_1_complete": "Identifying what&#8217;s unclear&#8230;",
                    }
                    _STEPS5 = [
                        ("Reviewing your photos",              "🔍", 1, 2),
                        ("Identifying what&#8217;s unclear",   "🔎", 2, 3),
                        ("Preparing clarification questions",  "❓", 3, 4),
                    ]
                    _IDX = {
                        None: 0,
                        "stage_1_running": 1, "stage_1_complete": 2,
                        "complete": 3,
                    }
                    _PCT = {
                        None: 6, "stage_1_running": 28, "stage_1_complete": 82,
                        "complete": 100,
                    }

                elif output_mode in ("diy", "verify_first"):
                    # Confirmed DIY / verify_first path: show all 5 repair-oriented steps.
                    _TOP = {
                        None:               "Warming up&#8230;",
                        "stage_1_running":  "Reviewing your photos&#8230;",
                        "stage_1_complete": "Issue identified, checking what you&#8217;ll need&#8230;",
                        "stage_2_running":  "Checking tools &amp; materials&#8230;",
                        "stage_2_complete": "Resources ready, writing your guide&#8230;",
                        "stage_3_running":  "Writing your step-by-step repair guide&#8230;",
                        "stage_3_complete": "Almost done&#8230;",
                    }
                    _STEPS5 = [
                        ("Reviewing your photos",         "🔍", 1, 2),
                        ("Identifying the problem",       "🧠", 2, 3),
                        ("Checking tools &amp; materials","🔧", 3, 5),
                        ("Writing your repair guide",     "📋", 5, 7),
                        ("Finalising your guide",         "✅", 7, 8),
                    ]
                    _IDX = {
                        None: 0,
                        "stage_1_running": 1, "stage_1_complete": 2,
                        "stage_2_running": 3, "stage_2_complete": 4,
                        "stage_3_running": 5, "stage_3_complete": 6,
                        "complete": 7,
                    }
                    _PCT = {
                        None: 6, "stage_1_running": 18, "stage_1_complete": 36,
                        "stage_2_running": 52, "stage_2_complete": 68,
                        "stage_3_running": 82, "stage_3_complete": 95,
                        "complete": 100,
                    }

                else:
                    # output_mode not yet known (Stage 1 still running).
                    # Four neutral steps that reflect actual AI reasoning without
                    # committing to any particular output path (DIY / assessment /
                    # care / clarification). All four are active-then-done in the
                    # single stage_1_running → stage_1_complete window so the user
                    # sees meaningful progression even before mode is decided.
                    _TOP = {
                        None:               "Warming up&#8230;",
                        "stage_1_running":  "Reviewing your photos&#8230;",
                        "stage_1_complete": "Assessing the situation&#8230;",
                    }
                    _STEPS5 = [
                        ("Reviewing your photos",            "🔍", 1, 2),
                        ("Understanding the issue",          "🧠", 1, 2),
                        ("Assessing risk and uncertainty",   "⚖️", 1, 2),
                        ("Determining the safest next step", "🧭", 2, 3),
                    ]
                    _IDX = {
                        None: 0,
                        "stage_1_running": 1, "stage_1_complete": 2,
                        "complete": 3,
                    }
                    _PCT = {
                        None: 6, "stage_1_running": 22, "stage_1_complete": 55,
                        "complete": 100,
                    }

                headline  = _TOP.get(stage, "Analysing&#8230;")
                cur       = _IDX.get(stage, 0)
                pct       = _PCT.get(stage, 8)
                run_id    = data.get("id", 0)  # analysis row id — resets timer on new run

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
                        'margin-bottom:4px;padding-left:19px;">'
                        'This takes 1-2 minutes the first time while AI models load'
                        '</div>'
                    )
                    # Live elapsed timer — resets when analysis id changes
                    ui.html(
                        f'<div id="knitra-elapsed" '
                        f'data-runid="{run_id}" '
                        f'style="color:#a78bfa;font-size:0.72rem;font-weight:500;'
                        f'padding-left:19px;margin-bottom:14px;">'
                        f'Analysing for 0s&#8230;'
                        f'</div>'
                    )
                    ui.run_javascript(
                        f"(function(){{"
                        f"  var el=document.getElementById('knitra-elapsed');"
                        f"  if(!el)return;"
                        f"  var rid={run_id};"
                        f"  if(window._knitraRunId!==rid){{"
                        f"    window._knitraRunId=rid;"
                        f"    window._knitraRunStart=Date.now();"
                        f"  }}"
                        f"  if(window._knitraElapsedInt)clearInterval(window._knitraElapsedInt);"
                        f"  function tick(){{"
                        f"    var e=document.getElementById('knitra-elapsed');"
                        f"    if(!e){{clearInterval(window._knitraElapsedInt);return;}}"
                        f"    var sec=Math.floor((Date.now()-window._knitraRunStart)/1000);"
                        f"    e.textContent='Analysing for '+sec+'s…';"
                        f"  }}"
                        f"  tick();"
                        f"  window._knitraElapsedInt=setInterval(tick,1000);"
                        f"}})()"
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

            def _draw_assessment_first(sr: dict, hazard_ack: bool = False):
                """Render the assessment-first output mode.

                Only shows: diagnosis header + professional assessment banner,
                immediate actions, what not to do, why assessment is needed,
                information to gather, professional guidance.

                Deliberately omits: tools, materials, repair steps, difficulty,
                estimated time. The model's role in this mode is stabilisation
                and escalation, not repair execution.
                """
                s1 = sr.get("stage_1") or {}
                sa = sr.get("assessment") or {}

                sev = s1.get("severity", "")
                _sev = {
                    "low":    ("background:#dcfce7;color:#15803d;", "Low severity"),
                    "medium": ("background:#fef3c7;color:#92400e;", "Medium severity"),
                    "high":   ("background:#fee2e2;color:#991b1b;", "High severity"),
                    "critical": ("background:#fee2e2;color:#7f1d1d;", "Critical severity"),
                }
                sev_s, sev_l = _sev.get(sev, ("background:#f1f5f9;color:#475569;",
                                               sev.capitalize() + " severity"))

                def _section_list(items: list, icon: str = "&#8226;", icon_col: str = "#ef4444"):
                    for item in items:
                        ui.html(
                            f'<div style="display:flex;align-items:flex-start;gap:8px;'
                            f'margin-bottom:6px;font-size:0.845rem;color:#334155;">'
                            f'<span style="color:{icon_col};flex-shrink:0;margin-top:2px;">'
                            f'{icon}</span>'
                            f'<span>{item}</span></div>'
                        )

                with ui.element("div").style(
                    "padding:22px 24px;display:flex;flex-direction:column;gap:18px;"
                ):
                    # ── Diagnosis header ──────────────────────────────────────
                    ui.html('<div class="analysis-section-label">Diagnosis</div>')
                    with ui.element("div").classes("analysis-card"):
                        ui.html(
                            f'<div style="display:flex;align-items:center;gap:8px;'
                            f'flex-wrap:wrap;margin-bottom:10px;">'
                            f'<span style="font-weight:800;color:#0f172a;font-size:1.28rem;'
                            f'letter-spacing:-0.01em;line-height:1.25;">'
                            f'{s1.get("problem_type", "Unknown issue")}</span>'
                            f'<span style="background:#ede9fe;color:#4c1d95;padding:2px 10px;'
                            f'border-radius:100px;font-size:11px;font-weight:600;">'
                            f'{_cat_label(s1.get("trade_category",""))}</span>'
                            f'<span style="{sev_s}padding:2px 10px;'
                            f'border-radius:100px;font-size:11px;font-weight:600;">'
                            f'{sev_l}</span>'
                            f'</div>'
                        )
                        for obs in s1.get("observations", []):
                            ui.html(
                                f'<div style="display:flex;align-items:flex-start;gap:8px;'
                                f'margin-bottom:5px;font-size:0.845rem;color:#334155;">'
                                f'<span style="color:#6366f1;flex-shrink:0;margin-top:2px;">'
                                f'&#8226;</span><span>{obs}</span></div>'
                            )

                    # ── Professional assessment required banner ────────────────
                    pro_type    = sa.get("professional_type") or s1.get("safety_concerns", [""])[0] or "a qualified professional"
                    urgency     = sa.get("professional_urgency") or "as soon as possible"
                    urgency_col = "#7f1d1d" if urgency == "immediately" else "#92400e"
                    ui.html(
                        f'<div style="display:flex;align-items:center;gap:12px;'
                        f'background:#fef2f2;border:1.5px solid #fecaca;'
                        f'border-radius:12px;padding:14px 16px;">'
                        f'<span style="font-size:1.4rem;flex-shrink:0;">&#128680;</span>'
                        f'<div>'
                        f'<div style="font-weight:700;color:#991b1b;font-size:0.92rem;'
                        f'margin-bottom:3px;">Professional Assessment Required</div>'
                        f'<div style="font-size:0.815rem;color:#64748b;">'
                        f'Contact a <strong>{pro_type}</strong> '
                        f'<span style="color:{urgency_col};font-weight:600;">{urgency}</span>.'
                        f'</div></div></div>'
                    )

                    # ── Safety responsibility notice ───────────────────────────
                    ui.html(
                        '<div style="display:flex;align-items:flex-start;gap:9px;'
                        'background:#fff7ed;border:1px solid #fed7aa;'
                        'border-radius:9px;padding:11px 13px;">'
                        '<span style="font-size:0.95rem;flex-shrink:0;margin-top:1px;">&#9888;&#65039;</span>'
                        '<p style="margin:0;font-size:0.78rem;color:#7c2d12;line-height:1.6;">'
                        '<strong>Do not attempt this repair yourself.</strong> '
                        'The AI has identified safety indicators that require a professional to assess '
                        'the situation before any work is carried out. The guidance on this page is '
                        'for information only. Proceeding without professional assessment may be '
                        'dangerous. You are responsible for any actions you choose to take.'
                        '</p></div>'
                    )

                    # ── Hazard acknowledgement gate ───────────────────────────
                    # Shown once per case (stored permanently in the database).
                    # If the user has already acknowledged, render all detail
                    # sections immediately.  If not, show the acknowledgement
                    # card and return early — detail sections are hidden until
                    # the user clicks "I Understand".
                    # On confirm: PATCH /cases/{case_id}/hazard-ack is called,
                    # the page reloads, _draw_assessment_first re-runs with
                    # hazard_ack=True and the detail sections render directly.
                    if not hazard_ack:
                        with ui.element("div").style(
                            "background:white;border:1.5px solid #fecaca;"
                            "border-radius:14px;padding:24px 26px;"
                            "display:flex;flex-direction:column;gap:16px;"
                            "box-shadow:0 1px 6px rgba(0,0,0,0.04);"
                        ):
                            ui.html(
                                '<div style="font-size:1.05rem;font-weight:800;'
                                'color:#0f172a;margin-bottom:2px;">'
                                '&#9888;&#65039; This case has been flagged as hazardous</div>'
                                '<div style="font-size:0.845rem;color:#334155;line-height:1.65;">'
                                'The AI has identified indicators that make this repair unsuitable '
                                'for DIY work. A professional assessment is recommended before '
                                'anything is attempted.'
                                '</div>'
                                '<div style="font-size:0.845rem;color:#334155;line-height:1.65;">'
                                'Knitra will not provide repair instructions for this case. '
                                'Asking Knitra to help with this repair will not override this '
                                'recommendation.'
                                '</div>'
                                '<div style="font-size:0.845rem;color:#334155;line-height:1.65;">'
                                'Safety-related interactions within this case may be recorded for '
                                'audit, safety, and system-improvement purposes. You remain '
                                'responsible for any actions you choose to take.'
                                '</div>'
                            )

                            async def _on_hazard_ack():
                                try:
                                    async with httpx.AsyncClient() as c:
                                        await c.patch(
                                            f"{API}/cases/{case_id}/hazard-ack",
                                            headers={"Authorization": f"Bearer {token}"},
                                            timeout=10.0,
                                        )
                                except Exception:
                                    pass
                                ui.navigate.to(f"/cases/{case_id}")

                            ui.button(
                                "I understand and accept responsibility",
                                on_click=_on_hazard_ack,
                            ).props("unelevated color=negative").style(
                                "font-weight:700;width:100%;"
                            )

                        return

                    # ── Immediate Actions ─────────────────────────────────────
                    immediate = sa.get("immediate_actions") or []
                    if immediate:
                        ui.html('<div class="analysis-section-label">Immediate Actions</div>')
                        with ui.element("div").classes("analysis-card"):
                            ui.html(
                                '<div style="font-size:0.78rem;color:#64748b;margin-bottom:10px;">'
                                'Do these now, before any professional arrives.</div>'
                            )
                            _section_list(immediate, icon="&#10003;", icon_col="#15803d")

                    # ── What NOT to do ────────────────────────────────────────
                    do_not = sa.get("do_not_do") or []
                    if do_not:
                        ui.html('<div class="analysis-section-label">What NOT to Do</div>')
                        with ui.element("div").classes("analysis-card").style(
                            "border-left:3px solid #ef4444;"
                        ):
                            _section_list(do_not, icon="&#10007;", icon_col="#ef4444")

                    # ── Why assessment is needed ──────────────────────────────
                    why = sa.get("why_assessment_needed") or ""
                    if why:
                        ui.html('<div class="analysis-section-label">Why Assessment Is Needed</div>')
                        with ui.element("div").classes("analysis-card"):
                            ui.html(
                                f'<div style="font-size:0.845rem;color:#334155;line-height:1.65;">'
                                f'{why}</div>'
                            )

                    # ── Information to gather ─────────────────────────────────
                    info = sa.get("information_needed") or []
                    if info:
                        ui.html('<div class="analysis-section-label">Information to Prepare</div>')
                        with ui.element("div").classes("analysis-card"):
                            ui.html(
                                '<div style="font-size:0.78rem;color:#64748b;margin-bottom:10px;">'
                                'Have this ready for the professional.</div>'
                            )
                            _section_list(info, icon="&#8594;", icon_col="#6366f1")

                    # ── Uncertainty note (if present) ─────────────────────────
                    unc = (s1.get("uncertainty_notes") or "").strip()
                    if unc:
                        ui.html(
                            f'<div style="background:#f8fafc;border:1px solid #e2e8f0;'
                            f'border-radius:8px;padding:10px 14px;font-size:0.78rem;'
                            f'color:#64748b;line-height:1.55;">'
                            f'<strong>What the photo cannot confirm:</strong> {unc}</div>'
                        )

            def _draw_clarification(sr: dict):
                """Render the clarification-mode output.

                Shows: compact diagnosis header, what is known, the clarifying
                questions as interactive cards, and a note on why the answers
                matter. No tools, materials, steps, or professional guidance.
                """
                s1 = sr.get("stage_1") or {}
                cl = sr.get("clarification") or {}
                # Bounded-clarification fallback: pipeline hit MAX rounds and is no
                # longer asking questions.  Render a distinct terminal banner.
                _inconclusive = bool(cl.get("inconclusive"))

                sev = s1.get("severity", "")
                sev_labels = {
                    "low":    ("background:#dcfce7;color:#15803d;", "Low severity"),
                    "medium": ("background:#fef3c7;color:#92400e;", "Medium severity"),
                    "high":   ("background:#fee2e2;color:#991b1b;", "High severity"),
                }
                sev_s, sev_l = sev_labels.get(sev, ("background:#f1f5f9;color:#475569;",
                                                      sev.capitalize()))

                with ui.element("div").style(
                    "padding:22px 24px;display:flex;flex-direction:column;gap:18px;"
                ):
                    # ── Inconclusive fallback banner ───────────────────────────
                    # Shown ONLY when the bounded-clarification limit was reached.
                    # Visually distinct (amber) so it does not read as another
                    # needs-answers round.
                    if _inconclusive:
                        ui.html(
                            '<div style="background:#fffbeb;border:1px solid #fcd34d;'
                            'border-radius:12px;padding:16px 18px;'
                            'display:flex;align-items:flex-start;gap:12px;">'
                            '<span style="font-size:1.4rem;line-height:1;flex-shrink:0;">'
                            '&#9888;&#65039;</span>'
                            '<div>'
                            '<div style="font-weight:800;color:#92400e;font-size:0.95rem;'
                            'margin-bottom:4px;">Could not reach a confident assessment</div>'
                            '<div style="font-size:0.84rem;color:#78350f;line-height:1.6;">'
                            'Knitra could not reach a confident assessment from the '
                            'available photos and answers, so it has stopped asking '
                            'questions. Please provide clearer, well-lit photos from '
                            'another angle, or have the issue checked in person before '
                            'starting work.'
                            '</div>'
                            '</div>'
                            '</div>'
                        )

                    # ── Diagnosis header (compact) ─────────────────────────────
                    ui.html('<div class="analysis-section-label">Diagnosis</div>')
                    with ui.element("div").classes("analysis-card"):
                        ui.html(
                            f'<div style="display:flex;align-items:center;gap:8px;'
                            f'flex-wrap:wrap;margin-bottom:8px;">'
                            f'<span style="font-weight:800;color:#0f172a;font-size:1.18rem;">'
                            f'{s1.get("problem_type","Unknown issue")}</span>'
                            f'<span style="background:#ede9fe;color:#4c1d95;padding:2px 10px;'
                            f'border-radius:100px;font-size:11px;font-weight:600;">'
                            f'{_cat_label(s1.get("trade_category",""))}</span>'
                            f'<span style="{sev_s}padding:2px 10px;'
                            f'border-radius:100px;font-size:11px;font-weight:600;">'
                            f'{sev_l}</span>'
                            f'</div>'
                        )
                        what_known = cl.get("what_is_known") or ""
                        if what_known:
                            ui.html(
                                f'<div style="font-size:0.845rem;color:#334155;'
                                f'line-height:1.6;">{what_known}</div>'
                            )

                    # ── Clarifying questions ───────────────────────────────────
                    questions = (
                        cl.get("clarifying_questions")
                        or s1.get("clarifying_questions")
                        or []
                    )
                    if questions:
                        ui.html('<div class="analysis-section-label">Before I can advise</div>')
                        with ui.element("div").style(
                            "background:#eef2ff;border:1px solid #c7d2fe;"
                            "border-radius:12px;padding:16px;"
                            "display:flex;flex-direction:column;gap:10px;"
                        ):
                            ui.html(
                                '<div style="font-size:0.78rem;color:#3730a3;font-weight:600;'
                                'margin-bottom:4px;">I need a little more information to give '
                                'you the right advice:</div>'
                            )
                            for i, q in enumerate(questions, 1):
                                ui.html(
                                    f'<div style="display:flex;align-items:flex-start;gap:10px;">'
                                    f'<span style="background:#4f46e5;color:#fff;font-weight:700;'
                                    f'font-size:0.75rem;border-radius:100px;padding:2px 7px;'
                                    f'flex-shrink:0;margin-top:1px;">{i}</span>'
                                    f'<span style="font-size:0.845rem;color:#1e1b4b;'
                                    f'font-weight:500;line-height:1.55;">{q}</span>'
                                    f'</div>'
                                )
                        ui.html(
                            '<div style="font-size:0.78rem;color:#64748b;line-height:1.55;">'
                            'Answer in <strong>Ask Knitra</strong> below, or upload a '
                            'closer photo, then press <strong>Re-analyse</strong> for an '
                            'updated diagnosis.</div>'
                        )

                    # ── What is unclear ────────────────────────────────────────
                    # Suppressed in the inconclusive fallback: the amber banner
                    # above already carries this guidance, so rendering it here too
                    # would be redundant.  Kept as a safety net for the edge case
                    # where the inconclusive flag is absent but questions are empty.
                    unclear = cl.get("what_is_unclear") or []
                    if unclear and not _inconclusive:
                        ui.html('<div class="analysis-section-label">What the photo cannot confirm</div>')
                        with ui.element("div").classes("analysis-card"):
                            for item in unclear:
                                ui.html(
                                    f'<div style="display:flex;align-items:flex-start;gap:8px;'
                                    f'margin-bottom:6px;font-size:0.845rem;color:#475569;">'
                                    f'<span style="color:#94a3b8;flex-shrink:0;margin-top:2px;">'
                                    f'&#8226;</span><span>{item}</span></div>'
                                )

            def _draw_care(sr: dict):
                """Render the care-mode output.

                Shows: compact diagnosis header, condition summary, ordered care
                recommendations, what to avoid, and expected outcome.

                Deliberately omits: tools, materials, protective gear, repair
                steps, safety warnings, professional guidance, difficulty level.
                The tone is friendly and practical, not trade-formal.
                """
                s1 = sr.get("stage_1") or {}
                sc = sr.get("care") or {}

                sev = s1.get("severity", "")
                sev_labels = {
                    "low":    ("background:#dcfce7;color:#15803d;", "Low severity"),
                    "medium": ("background:#fef3c7;color:#92400e;", "Medium severity"),
                    "high":   ("background:#fee2e2;color:#991b1b;", "High severity"),
                }
                sev_s, sev_l = sev_labels.get(sev, ("background:#f1f5f9;color:#475569;",
                                                      sev.capitalize()))

                with ui.element("div").style(
                    "padding:22px 24px;display:flex;flex-direction:column;gap:18px;"
                ):
                    # ── Compact diagnosis header ──────────────────────────────
                    ui.html('<div class="analysis-section-label">What I Can See</div>')
                    with ui.element("div").classes("analysis-card"):
                        ui.html(
                            f'<div style="display:flex;align-items:center;gap:8px;'
                            f'flex-wrap:wrap;margin-bottom:8px;">'
                            f'<span style="font-weight:800;color:#0f172a;font-size:1.18rem;">'
                            f'{s1.get("problem_type","Unknown")}</span>'
                            f'<span style="background:#ede9fe;color:#4c1d95;padding:2px 10px;'
                            f'border-radius:100px;font-size:11px;font-weight:600;">'
                            f'{_cat_label(s1.get("trade_category",""))}</span>'
                            f'</div>'
                        )
                        summary = sc.get("condition_summary") or ""
                        if summary:
                            ui.html(
                                f'<div style="font-size:0.845rem;color:#475569;'
                                f'line-height:1.6;">{summary}</div>'
                            )

                    # ── Care Recommendations ──────────────────────────────────
                    recs = sc.get("recommendations") or []
                    if recs:
                        ui.html('<div class="analysis-section-label">Care Recommendations</div>')
                        with ui.element("div").classes("analysis-card"):
                            for i, rec in enumerate(recs, 1):
                                ui.html(
                                    f'<div style="display:flex;align-items:flex-start;gap:10px;'
                                    f'margin-bottom:10px;">'
                                    f'<span style="background:#ede9fe;color:#4c1d95;font-weight:700;'
                                    f'font-size:0.75rem;border-radius:100px;padding:2px 7px;'
                                    f'flex-shrink:0;margin-top:1px;">{i}</span>'
                                    f'<span style="font-size:0.845rem;color:#334155;'
                                    f'line-height:1.6;">{rec}</span>'
                                    f'</div>'
                                )

                    # ── What to Avoid ─────────────────────────────────────────
                    avoid = sc.get("what_to_avoid") or []
                    if avoid:
                        ui.html('<div class="analysis-section-label">What to Avoid</div>')
                        with ui.element("div").classes("analysis-card"):
                            for item in avoid:
                                ui.html(
                                    f'<div style="display:flex;align-items:flex-start;gap:8px;'
                                    f'margin-bottom:7px;font-size:0.845rem;color:#334155;">'
                                    f'<span style="color:#f59e0b;flex-shrink:0;margin-top:2px;">'
                                    f'&#9888;</span><span>{item}</span></div>'
                                )

                    # ── Expected Outcome ──────────────────────────────────────
                    outcome = sc.get("expected_outcome") or ""
                    if outcome:
                        ui.html(
                            f'<div style="background:#f0fdf4;border:1px solid #bbf7d0;'
                            f'border-radius:8px;padding:12px 16px;font-size:0.845rem;'
                            f'color:#15803d;line-height:1.6;">'
                            f'<strong>Expected result:</strong> {outcome}</div>'
                        )

            def _draw_time_footer(ms: int):
                """Subtle 'Completed in Xs' line shown below every completed result.
                Hidden when processing_time_ms is missing or zero."""
                if not ms:
                    return
                secs = ms // 1000
                t_str = f"{secs}s" if secs > 0 else f"{ms}ms"
                ui.html(
                    f'<div style="color:#cbd5e1;font-size:0.7rem;'
                    f'padding:6px 24px 14px;text-align:right;">'
                    f'Completed in {t_str}'
                    f'</div>'
                )

            def _draw_out_of_scope(sr: dict):
                """Render the out_of_scope mode: active emergency OR non-repair image.

                Shows ONLY guidance — never tools, materials, PPE, repair steps, or a
                trade-professional referral. Emergency uses a red banner and emergency
                guidance; non-repair / unclear uses a neutral amber 'send a clearer
                repair image' card.
                """
                oos = sr.get("out_of_scope") or {}
                is_emergency = bool(oos.get("is_emergency"))
                headline = oos.get("headline") or (
                    "This looks like an active emergency" if is_emergency
                    else "No repair issue identified"
                )
                guidance = oos.get("guidance") or (
                    "This appears to be an active emergency. Move away from danger, "
                    "evacuate if necessary, and contact emergency services. Do not "
                    "attempt a repair while the hazard is active."
                    if is_emergency else
                    "Knitra could not identify an in-scope household repair issue from this "
                    "image. Please upload a clear image of the damaged area or repair concern."
                )
                with ui.element("div").style(
                    "padding:22px 24px;display:flex;flex-direction:column;gap:14px;"
                ):
                    if is_emergency:
                        ui.html(
                            f'<div style="background:#fef2f2;border:1px solid #fca5a5;'
                            f'border-radius:12px;padding:18px 20px;'
                            f'display:flex;align-items:flex-start;gap:12px;">'
                            f'<span style="font-size:1.5rem;line-height:1;flex-shrink:0;">'
                            f'&#128721;</span>'
                            f'<div>'
                            f'<div style="font-weight:800;color:#991b1b;font-size:1rem;'
                            f'margin-bottom:6px;">{headline}</div>'
                            f'<div style="font-size:0.88rem;color:#7f1d1d;line-height:1.6;">'
                            f'{guidance}</div>'
                            f'<div style="font-size:0.82rem;color:#991b1b;font-weight:700;'
                            f'margin-top:10px;">Call emergency services now if anyone is in '
                            f'danger.</div>'
                            f'</div></div>'
                        )
                    else:
                        ui.html(
                            f'<div style="background:#fffbeb;border:1px solid #fcd34d;'
                            f'border-radius:12px;padding:18px 20px;'
                            f'display:flex;align-items:flex-start;gap:12px;">'
                            f'<span style="font-size:1.4rem;line-height:1;flex-shrink:0;">'
                            f'&#128247;</span>'
                            f'<div>'
                            f'<div style="font-weight:700;color:#92400e;font-size:0.95rem;'
                            f'margin-bottom:4px;">{headline}</div>'
                            f'<div style="font-size:0.86rem;color:#78350f;line-height:1.6;">'
                            f'{guidance}</div>'
                            f'</div></div>'
                        )

            def _draw_complete(data: dict):
                """Full result view. Routes to the correct render function based
                on output_mode stored in structured_result. Old analyses without
                output_mode fall through to the legacy DIY renderer."""
                sr = data.get("structured_result") or {}
                mode = sr.get("output_mode", "diy")
                s1   = sr.get("stage_1") or {}
                ms   = data.get("processing_time_ms") or 0

                # Route to mode-specific renderer
                if mode == "assessment_first":
                    _draw_assessment_first(sr, hazard_ack=data.get("hazard_ack", False))
                    _draw_time_footer(ms)
                    return

                if mode == "care":
                    _draw_care(sr)
                    _draw_time_footer(ms)
                    return

                if mode == "clarification":
                    _draw_clarification(sr)
                    _draw_time_footer(ms)
                    return

                if mode == "out_of_scope":
                    _draw_out_of_scope(sr)
                    _draw_time_footer(ms)
                    return

                # Legacy / DIY / verify_first path — full Stage 2 + Stage 3 render
                s2 = sr.get("stage_2") or {}
                s3 = sr.get("stage_3") or {}

                # ── Clarification history badge ───────────────────────────────
                # Shown when this analysis was produced by a re-analysis run that
                # included clarification answers (text_context has Q/A structure).
                # First-pass analyses have text_context=None and never show this.
                _tc = (data.get("text_context") or "").strip()
                _has_qa = bool(_tc) and ("Q1:" in _tc or _tc.startswith("User answers"))
                if _has_qa:
                    import re as _re
                    _qs = _re.findall(r"Q\d+:\s*(.+?)(?=\nA\d+:|\Z)", _tc, _re.DOTALL)
                    _as = _re.findall(r"A\d+:\s*(.+?)(?=\nQ\d+:|\Z)", _tc, _re.DOTALL)
                    _pairs = list(zip(
                        [q.strip() for q in _qs],
                        [a.strip() for a in _as],
                    ))
                    with ui.element("div").style(
                        "padding:14px 24px 0;"
                    ):
                        with ui.element("div").style(
                            "background:#f0fdf4;border:1px solid #86efac;"
                            "border-radius:10px;padding:12px 16px;"
                        ):
                            ui.html(
                                '<div style="display:flex;align-items:center;'
                                'justify-content:space-between;flex-wrap:wrap;gap:8px;">'
                                '<span style="font-weight:700;color:#166534;font-size:0.82rem;">'
                                '&#10003; Updated after clarification</span>'
                                '</div>'
                            )
                            if _pairs:
                                _hist_html = (
                                    '<details style="margin-top:8px;">'
                                    '<summary style="cursor:pointer;font-size:0.78rem;'
                                    'color:#166534;font-weight:600;list-style:none;'
                                    'display:flex;align-items:center;gap:4px;">'
                                    '&#9656; Previous clarification</summary>'
                                    '<div style="margin-top:8px;display:flex;'
                                    'flex-direction:column;gap:6px;">'
                                )
                                for _q, _a in _pairs:
                                    _hist_html += (
                                        f'<div style="font-size:0.78rem;line-height:1.5;">'
                                        f'<span style="color:#374151;font-weight:600;">Q:</span> '
                                        f'<span style="color:#374151;">{_q}</span><br>'
                                        f'<span style="color:#374151;font-weight:600;">A:</span> '
                                        f'<span style="color:#166534;">{_a}</span>'
                                        f'</div>'
                                    )
                                _hist_html += '</div></details>'
                                ui.html(_hist_html)

                sev = s1.get("severity", "")
                _sev = {
                    "low":    ("background:#dcfce7;color:#15803d;", "Low severity"),
                    "medium": ("background:#fef3c7;color:#92400e;", "Medium severity"),
                    "high":   ("background:#fee2e2;color:#991b1b;", "High severity"),
                }
                sev_s, sev_l = _sev.get(sev, ("background:#f1f5f9;color:#475569;",
                                               sev.capitalize() + " severity"))

                # Risk assessment values, default to DIY safe for old analyses
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
                            f'<span style="font-weight:800;color:#0f172a;font-size:1.28rem;'
                            f'letter-spacing:-0.01em;line-height:1.25;">'
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
                    clarif_req   = (s1.get("clarification_request") or "").strip()

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
                        # Visual evidence bullets, what INCREASED confidence
                        if vis_evidence:
                            ui.html(
                                '<div style="font-size:0.72rem;font-weight:700;'
                                'color:#15803d;letter-spacing:0.05em;'
                                'text-transform:uppercase;margin-top:4px;">'
                                'What increased confidence</div>'
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
                        # Uncertainty notes, what REDUCED confidence / still uncertain
                        if unc_notes:
                            ui.html(
                                f'<div style="display:flex;align-items:flex-start;gap:6px;'
                                f'background:#fffbeb;border:1px solid #fde68a;'
                                f'border-radius:7px;padding:8px 10px;margin-top:2px;">'
                                f'<span style="font-size:0.85rem;flex-shrink:0;">&#9888;&#65039;</span>'
                                f'<div>'
                                f'<div style="font-size:0.73rem;font-weight:700;'
                                f'color:#92400e;margin-bottom:2px;">What reduced confidence / still uncertain</div>'
                                f'<div style="font-size:0.78rem;color:#78350f;'
                                f'line-height:1.5;">{unc_notes}</div>'
                                f'</div></div>'
                            )

                        # How to improve confidence, shown for sub-90% complete cases.
                        # Uses the AI's own clarification request when present, else a
                        # sensible generic photo-guidance fallback. Medium-confidence
                        # complete cases get this too, not only clarification cases.
                        if conf < 90:
                            improve_msg = clarif_req or (
                                "A sharper, well-lit close-up of the affected area, plus one "
                                "wider shot for context, would let the AI confirm the details "
                                "marked uncertain above. Include a tape measure or a coin for scale "
                                "where size matters."
                            )
                            ui.html(
                                f'<div style="display:flex;align-items:flex-start;gap:6px;'
                                f'background:#eff6ff;border:1px solid #bfdbfe;'
                                f'border-radius:7px;padding:8px 10px;margin-top:2px;">'
                                f'<span style="font-size:0.85rem;flex-shrink:0;">&#128247;</span>'
                                f'<div>'
                                f'<div style="font-size:0.73rem;font-weight:700;'
                                f'color:#1e40af;margin-bottom:2px;">'
                                f'What would improve confidence</div>'
                                f'<div style="font-size:0.78rem;color:#1e3a8a;'
                                f'line-height:1.5;">{improve_msg}</div>'
                                f'<div style="font-size:0.72rem;color:#64748b;'
                                f'margin-top:4px;">Add a clearer photo above, then press '
                                f'<b>Re-analyse</b> for an updated confidence score.</div>'
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

                    # ── AI responsibility notice ──────────────────────────────
                    ui.html(
                        '<div style="display:flex;align-items:flex-start;gap:8px;'
                        'background:#f8fafc;border:1px solid #e2e8f0;'
                        'border-radius:8px;padding:10px 12px;">'
                        '<span style="font-size:0.85rem;flex-shrink:0;margin-top:1px;">&#8505;&#65039;</span>'
                        '<p style="margin:0;font-size:0.76rem;color:#64748b;line-height:1.55;">'
                        'This repair plan is AI-generated and may not reflect your exact situation. '
                        'Verify materials and steps before starting. Stop and consult a professional '
                        'if you encounter anything unexpected. '
                        '<strong style="color:#475569;">You are responsible for any actions you take.</strong>'
                        '</p></div>'
                    )

                    # ── Professional-required guard (rendering safety net) ─────
                    # If Stage 3 judged this NOT DIY-suitable, withhold the entire
                    # repair plan (What you'll need / tools / materials / PPE / steps).
                    # Showing "Professional required" alongside DIY steps is an unsafe
                    # contradiction. Backend routing now sends most such cases to
                    # assessment_first; this guard also fixes ALREADY-STORED analyses
                    # (e.g. case 167) without re-analysis. Render guidance only, then
                    # exit _draw_complete before any plan section.
                    if not diy_ok:
                        _who = f"a {pro_type}" if pro_type else "a qualified professional"
                        _reason = (s3.get("when_to_call_professional") or "").strip()
                        if not _reason:
                            _concerns = s1.get("safety_concerns") or []
                            _reason = (
                                "; ".join(_concerns) if _concerns
                                else "This repair carries risks that should be assessed in person."
                            )
                        with ui.element("div").style(
                            "padding:22px 24px;display:flex;flex-direction:column;gap:14px;"
                        ):
                            ui.html(
                                f'<div style="background:#fef2f2;border:1px solid #fecaca;'
                                f'border-radius:10px;padding:14px 16px;">'
                                f'<div style="font-weight:700;color:#991b1b;font-size:0.85rem;'
                                f'margin-bottom:4px;">Why a professional is needed</div>'
                                f'<div style="color:#7f1d1d;font-size:0.83rem;line-height:1.6;">'
                                f'{_reason}</div></div>'
                            )
                            ui.html(
                                f'<div style="background:#f8fafc;border:1px solid #e2e8f0;'
                                f'border-radius:10px;padding:14px 16px;">'
                                f'<div style="font-weight:700;color:#334155;font-size:0.85rem;'
                                f'margin-bottom:4px;">What to do next</div>'
                                f'<div style="color:#475569;font-size:0.83rem;line-height:1.6;">'
                                f'Contact {_who} to assess and carry out this repair. '
                                f'Do not attempt it yourself.</div></div>'
                            )
                            ui.html(
                                '<div style="display:flex;align-items:flex-start;gap:8px;'
                                'background:#fffbeb;border:1px solid #fcd34d;'
                                'border-radius:10px;padding:12px 14px;">'
                                '<span style="font-size:0.95rem;flex-shrink:0;">&#9888;&#65039;</span>'
                                '<div style="font-size:0.82rem;color:#78350f;line-height:1.55;">'
                                'Step-by-step repair instructions are intentionally withheld for this '
                                'case because it should be handled by a professional.</div></div>'
                            )
                        _draw_time_footer(data.get("processing_time_ms") or 0)
                        return

                    # ── Safety acknowledgement gate ──────────────────────────
                    # Shown once per browser session before sections 2-5 are
                    # revealed.  The ack state is stored in app.storage.user
                    # (keyed by case_id so each case is gated independently).
                    # On confirm: ack is stored and the page reloads, which
                    # re-enters _draw_complete with the ack already set so
                    # the tools/materials/steps sections render immediately.
                    # The `return` below exits _draw_complete; Python's `with`
                    # context manager unwinds the outer div cleanly.
                    _ack_key = f"ack_{case_id}"
                    if not app.storage.user.get(_ack_key, False):
                        with ui.element("div").style(
                            "background:white;border:1.5px solid #e2e8f0;"
                            "border-radius:14px;padding:24px 26px;"
                            "display:flex;flex-direction:column;gap:16px;"
                            "box-shadow:0 1px 6px rgba(0,0,0,0.04);"
                        ):
                            ui.html(
                                '<div style="font-size:1.05rem;font-weight:800;'
                                'color:#0f172a;margin-bottom:2px;">Before you begin</div>'
                                '<div style="font-size:0.82rem;color:#64748b;line-height:1.6;">'
                                'This repair plan is AI-generated. Please read and confirm '
                                'each point before viewing the tools and repair steps.</div>'
                            )

                            _checks: list = []

                            def _update_confirm_btn(_=None):
                                if all(c.value for c in _checks):
                                    _confirm_btn.enable()
                                else:
                                    _confirm_btn.disable()

                            for _lbl in [
                                "I understand this advice is AI-generated and may be "
                                "incomplete or incorrect.",
                                "I will check local safety regulations before starting "
                                "any work.",
                                "I will stop immediately and contact a qualified "
                                "professional if I am unsure or encounter anything "
                                "unexpected.",
                                "I accept responsibility for the actions I choose to "
                                "take based on this advice.",
                            ]:
                                _checks.append(
                                    ui.checkbox(_lbl, on_change=_update_confirm_btn)
                                    .style("font-size:0.83rem;align-items:flex-start;")
                                )

                            async def _on_confirm():
                                app.storage.user[_ack_key] = True
                                ui.navigate.to(f"/cases/{case_id}")

                            _confirm_btn = (
                                ui.button(
                                    "I understand — show me the steps",
                                    on_click=_on_confirm,
                                )
                                .props("unelevated color=primary")
                                .style(
                                    "font-weight:700;width:100%;margin-top:4px;"
                                )
                            )
                            _confirm_btn.disable()

                            ui.html(
                                '<p style="font-size:0.72rem;color:#94a3b8;'
                                'text-align:center;margin:0;line-height:1.5;">'
                                'This acknowledgement does not constitute a safety '
                                'assessment. If in doubt, always consult a qualified '
                                'professional.</p>'
                            )

                        _draw_time_footer(data.get("processing_time_ms") or 0)
                        return

                    # ── 2. What you'll need ── (shown after acknowledgement) ───
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
                                    chip, text = _tier_split(item)
                                    ui.html(
                                        f'<div style="display:flex;align-items:flex-start;'
                                        f'gap:6px;margin-bottom:6px;'
                                        f'font-size:0.82rem;color:#334155;">'
                                        f'<span style="color:#6366f1;flex-shrink:0;'
                                        f'margin-top:2px;">&#8226;</span>'
                                        f'<span style="display:flex;flex-wrap:wrap;'
                                        f'align-items:center;">{chip}'
                                        f'<span>{text}</span></span></div>'
                                    )
                                if opt_items:
                                    ui.html(
                                        '<div style="font-size:0.72rem;font-weight:600;'
                                        'color:#94a3b8;letter-spacing:0.05em;'
                                        'text-transform:uppercase;margin:8px 0 5px;">'
                                        'Nice to have</div>'
                                    )
                                    for item in opt_items:
                                        chip, text = _tier_split(item)
                                        ui.html(
                                            f'<div style="display:flex;'
                                            f'align-items:flex-start;gap:6px;'
                                            f'margin-bottom:6px;font-size:0.82rem;'
                                            f'color:#94a3b8;">'
                                            f'<span style="color:#c7d2fe;flex-shrink:0;'
                                            f'margin-top:2px;">&#8226;</span>'
                                            f'<span style="display:flex;flex-wrap:wrap;'
                                            f'align-items:center;">{chip}'
                                            f'<span>{text}</span></span></div>'
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

                    # ── Completed time footer (shared helper, hidden if ms=0) ──
                    _draw_time_footer(data.get("processing_time_ms") or 0)

            def _draw_needs_clarification(data: dict):
                """
                Clarification gate, Stage 1 was not confident enough to diagnose.

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

                _is_video = s1.get("input_modality") == "video"
                _media    = "video" if _is_video else "photo"

                # Fallback message if Stage 1 output is not available
                msg = (
                    clarif_req
                    or data.get("validation_issues")
                    or (
                        f"Please upload a clearer, closer {_media} of the damaged area "
                        "or describe the problem in more detail, then click Re-analyse."
                    )
                )

                # Confidence badge colours
                if conf >= 76:
                    _cc, _cb, _cl = "#92400e", "#fef3c7", "Medium"
                else:
                    _cc, _cb, _cl = "#dc2626", "#fee2e2", "Low"

                # ── Detect what type of input the model needs ─────────────────
                _combined_hints = (clarif_req + " " + conf_reason).lower()
                _media_tips: list[tuple[str, str]] = []
                if any(w in _combined_hints for w in ["close", "closer", "close-up", "detail", "clearer"]):
                    _media_tips.append(("🔍", f"Upload a closer, more detailed {_media} of the damaged area"))
                if any(w in _combined_hints for w in ["wider", "wide", "context", "surround", "full"]):
                    _media_tips.append(("📐", "Include more surroundings to show the full extent"))
                if any(w in _combined_hints for w in ["light", "bright", "dark", "lit", "illumin", "blurr"]):
                    if _is_video:
                        _media_tips.append(("💡", "Re-record the video in better lighting or move to a brighter area"))
                    else:
                        _media_tips.append(("💡", "Take the photo in better light or move to a brighter area"))
                if any(w in _combined_hints for w in ["describe", "written", "description", "explain", "unclear", "not sure", "cannot determine"]):
                    _media_tips.append(("✏️", "Add a written description of what you see and what the problem is"))
                if not _media_tips:
                    if _is_video:
                        _media_tips = [
                            ("🎥", "Re-record the video, closer and in good lighting"),
                            ("✏️", "Add a written description to give the AI a second evidence stream"),
                        ]
                    else:
                        _media_tips = [
                            ("📷", "Upload a new photo, closer, well-lit, and in focus"),
                            ("✏️", "Add a written description to give the AI a second evidence stream"),
                        ]
                _photo_tips = _media_tips

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
                """Error state, shows validation_issues and a retry button."""
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

            # ── Feedback widget (Phase 1) ────────────────────────────────────
            # Shown under any completed result. Mode-aware second question. Posts
            # to /cases/{id}/feedback. Safety options are sent as safety_category
            # (the backend also auto-flags hazardous/emergency modes).
            def _draw_feedback_widget(data: dict):
                analysis_id = data.get("id")
                sr = data.get("structured_result") or {}
                mode = sr.get("output_mode", "diy")
                if not analysis_id:
                    return
                _done_key = f"fb_done_{analysis_id}"

                # Mode-aware issue options: (label, kind, value). kind: "issue" | "safety".
                _common = [
                    ("Wrong severity / risk", "issue", "wrong_severity"),
                    ("Wrong trade / category", "issue", "wrong_trade"),
                    ("Something else", "issue", "other"),
                ]
                _by_mode = {
                    "diy": [
                        ("Steps were unclear or incomplete", "issue", "steps_unclear"),
                        ("Wrong DIY recommendation", "issue", "wrong_diy_recommendation"),
                        ("Should have recommended a professional", "issue", "wrong_professional_recommendation"),
                    ],
                    "verify_first": [
                        ("Steps were unclear or incomplete", "issue", "steps_unclear"),
                        ("Wrong DIY recommendation", "issue", "wrong_diy_recommendation"),
                    ],
                    "assessment_first": [
                        ("Professional escalation was not appropriate", "issue", "wrong_professional_recommendation"),
                        ("Recommendation felt unsafe", "safety", "unsafe_recommendation"),
                        ("A real hazard was missed", "safety", "missed_hazard"),
                    ],
                    "clarification": [
                        ("The questions were not useful", "issue", "clarification_not_useful"),
                        ("Scope decision was wrong", "issue", "scope_incorrect"),
                    ],
                    "care": [
                        ("The care advice was not useful", "issue", "care_not_useful"),
                    ],
                    "out_of_scope": [
                        ("Scope decision was incorrect", "issue", "scope_incorrect"),
                        ("This was wrongly called an emergency", "safety", "incorrect_emergency"),
                        ("A real emergency was missed", "safety", "missed_emergency"),
                    ],
                }
                _opts = _by_mode.get(mode, _by_mode["diy"]) + _common
                _label_map = {lbl: (kind, val) for (lbl, kind, val) in _opts}

                _heading = {
                    "assessment_first": "Was the professional recommendation right?",
                    "clarification":    "Were these questions helpful?",
                    "care":             "Was this care advice useful?",
                    "out_of_scope":     "Was this the right call?",
                }.get(mode, "Was this assessment helpful?")

                with ui.element("div").style(
                    "margin-top:10px;background:white;border:1px solid #e2e8f0;"
                    "border-radius:14px;padding:18px 22px;"
                ):
                    if app.storage.user.get(_done_key, False):
                        ui.html(
                            '<div style="display:flex;align-items:center;gap:8px;'
                            'color:#15803d;font-size:0.86rem;font-weight:600;">'
                            '<span>&#10003;</span><span>Thanks — your feedback was recorded.</span></div>'
                        )
                        return

                    ui.html(
                        f'<div style="font-weight:700;color:#0f172a;font-size:0.9rem;'
                        f'margin-bottom:10px;">{_heading}</div>'
                    )
                    rating = ui.radio(
                        {"helpful": "Helpful", "partial": "Partially", "not": "No"},
                    ).props("inline").style("font-size:0.85rem;")

                    # ── Reason chips (shown for Partially / No only) ──────────
                    _CHIP_OPTIONS = [
                        ("wrong_diagnosis",                   "Wrong diagnosis"),
                        ("too_cautious",                      "Too cautious"),
                        ("not_cautious_enough",               "Not cautious enough"),
                        ("not_detailed_enough",               "Not detailed enough"),
                        ("wrong_tools_materials",             "Wrong tools/materials"),
                        ("steps_unclear",                     "Steps unclear"),
                        ("wrong_professional_recommendation", "Wrong professional recommendation"),
                        ("advice_didnt_match_upload",         "Advice didn't match upload"),
                        ("repeated_question",                 "Repeated a question I answered"),
                        ("advice_felt_unsafe",                "Advice felt unsafe"),
                        ("other",                            "Other"),
                    ]
                    _CHIP_OFF = (
                        "padding:3px 11px;border-radius:100px;font-size:0.78rem;"
                        "border:1.5px solid #e2e8f0;background:white;color:#64748b;"
                        "font-weight:500;min-height:0;"
                    )
                    _CHIP_ON = (
                        "padding:3px 11px;border-radius:100px;font-size:0.78rem;"
                        "border:1.5px solid #6366f1;background:#e0e7ff;color:#3730a3;"
                        "font-weight:600;min-height:0;"
                    )
                    _chip_sel  = {v: False for v, _ in _CHIP_OPTIONS}
                    _chip_btns: dict = {}

                    with ui.element("div").style("display:none;margin-top:10px;") as _chips_container:
                        ui.html(
                            '<div style="font-size:0.78rem;color:#64748b;margin-bottom:6px;">'
                            'What was the issue? <span style="font-weight:400;">'
                            '(optional — choose all that apply)</span></div>'
                        )
                        with ui.element("div").style("display:flex;flex-wrap:wrap;gap:6px;"):
                            for _cv, _cl in _CHIP_OPTIONS:
                                _btn = ui.button(_cl).props("flat no-caps").style(_CHIP_OFF)
                                _chip_btns[_cv] = _btn

                                def _make_toggle(v, b):
                                    def _toggle():
                                        _chip_sel[v] = not _chip_sel[v]
                                        b.style(_CHIP_ON if _chip_sel[v] else _CHIP_OFF)
                                    return _toggle

                                _btn.on_click(_make_toggle(_cv, _btn))

                    comment = ui.textarea(
                        placeholder="Optional: what actually happened? (helps us improve)",
                    ).props("outlined dense autogrow rows=2").style(
                        "width:100%;margin-top:8px;font-size:0.85rem;"
                    )

                    def _on_rating(_=None):
                        show = rating.value in ("partial", "not")
                        _chips_container.style(
                            "display:block;margin-top:10px;"
                            if show else "display:none;margin-top:10px;"
                        )
                        if not show:
                            for v in _chip_sel:
                                _chip_sel[v] = False
                            for b in _chip_btns.values():
                                b.style(_CHIP_OFF)
                    rating.on_value_change(_on_rating)

                    _msg = ui.label("").style("font-size:0.8rem;color:#dc2626;min-height:16px;margin-top:6px;")

                    async def _submit():
                        if not rating.value:
                            _msg.set_text("Please choose Helpful, Partially, or No.")
                            return
                        body = {"analysis_id": analysis_id, "rating": rating.value}
                        if rating.value in ("partial", "not"):
                            selected_chips = [v for v, sel in _chip_sel.items() if sel]
                            if selected_chips:
                                body["reason_chips"] = selected_chips
                        if (comment.value or "").strip():
                            body["comment"] = comment.value.strip()
                        try:
                            async with httpx.AsyncClient() as c:
                                r = await c.post(
                                    f"{API}/cases/{case_id}/feedback",
                                    json=body,
                                    headers={"Authorization": f"Bearer {token}"},
                                    timeout=10.0,
                                )
                            if r.status_code in (200, 201):
                                app.storage.user[_done_key] = True
                                _redraw(data)   # re-render → thank-you state
                            else:
                                _msg.set_text(f"Could not submit ({r.status_code}). Please try again.")
                        except Exception:
                            _msg.set_text("Could not reach the server. Please try again.")

                    ui.button("Submit feedback", on_click=_submit) \
                        .props("unelevated color=primary size=sm") \
                        .style("font-weight:600;margin-top:8px;")

            # ── Admin judge panel ─────────────────────────────────────────────
            # Visible only to admins (UX gate; server enforces require_admin).
            # Lets an admin click "Run Judge" to evaluate output quality on-demand.
            def _draw_judge_panel():
                _VERDICT_STYLE = {
                    "pass":         ("Pass",         "#15803d", "#dcfce7"),
                    "needs_review": ("Needs Review", "#92400e", "#fef3c7"),
                    "unsafe":       ("Unsafe",       "#991b1b", "#fee2e2"),
                }
                _TAG_LABELS = {
                    "over_cautious_routing":                          "Over-cautious routing",
                    "under_cautious_routing":                         "Under-cautious routing",
                    "missing_safety_warning":                         "Missing safety warning",
                    "hallucinated_hazard":                            "Hallucinated hazard",
                    "vague_repair_steps":                             "Vague repair steps",
                    "vague_tools_or_materials":                       "Vague tools/materials",
                    "poor_grounding":                                 "Poor grounding",
                    "unnecessary_clarification":                      "Unnecessary clarification",
                    "missing_clarification":                          "Missing clarification",
                    "unclear_user_language":                          "Unclear language",
                    "wrong_trade_category":                           "Wrong trade category",
                    "weak_step_ordering":                             "Weak step ordering",
                    "emergency_wording_could_be_stronger":            "Emergency wording too weak",
                    "emergency_repair_steps_inappropriately_expected":"Repair steps wrongly expected",
                    "unsupported_professional_escalation":            "Unsupported escalation",
                    "unsupported_hazard_claim":                       "Unsupported hazard claim",
                    "wrong_professional_type":                        "Wrong professional type",
                    "over_specific_urgency":                          "Urgency over-specified",
                    "useful_but_overcautious":                        "Useful but overcautious",
                    "evidence_uncertainty_not_explained":             "Uncertainty not explained",
                    "correct_but_unclear_reasoning":                  "Correct but unclear",
                }

                def _score_color(s: int) -> str:
                    if s >= 80: return "#15803d"
                    if s >= 65: return "#92400e"
                    return "#991b1b"

                def _render_result(j: dict, container):
                    container.clear()
                    with container:
                        verdict   = j.get("final_verdict", "needs_review")
                        vstyle    = _VERDICT_STYLE.get(verdict, _VERDICT_STYLE["needs_review"])
                        overall   = j.get("overall_score", 0)
                        dims      = j.get("dimension_scores") or {}
                        tags      = j.get("issue_tags") or []
                        judged_at = j.get("judged_at", "")[:10]
                        rubric    = j.get("rubric_mode", j.get("output_mode", ""))

                        # New structured fields (Phase 2), with fallback to legacy
                        what_went_well  = j.get("what_went_well") or j.get("strengths") or []
                        what_needs_work = j.get("what_needs_work") or j.get("missing_details") or []
                        claims          = j.get("claims") or []
                        findings        = j.get("findings") or []
                        mode_review     = j.get("mode_review") or ""
                        prof_review     = j.get("professional_type_review") or ""
                        # qa_summary with fallback to old judge_reasoning_summary
                        summary = (
                            j.get("qa_summary")
                            or j.get("judge_reasoning_summary")
                            or ""
                        )
                        # Legacy fallback: if no findings, use suggested_improvements
                        legacy_imps = j.get("suggested_improvements") or []

                        _CLAIM_STYLE = {
                            "confirmed":           ("#dcfce7", "#15803d", "✓ Confirmed"),
                            "reasonable_inference":("#dbeafe", "#1e40af", "~ Reasonable"),
                            "unsupported":         ("#ffedd5", "#9a3412", "✗ Unsupported"),
                            "overclaimed":         ("#fee2e2", "#991b1b", "⚠ Overclaimed"),
                        }

                        def _section_header(title: str):
                            ui.html(
                                f'<div style="font-size:0.72rem;font-weight:700;color:#475569;'
                                f'text-transform:uppercase;letter-spacing:0.07em;'
                                f'margin:14px 0 6px;">{title}</div>'
                            )

                        # ── Score + verdict header ──────────────────────────
                        _rubric_badge = (
                            f'&nbsp;<span style="background:#e0e7ff;color:#3730a3;'
                            f'font-size:0.7rem;font-weight:600;padding:2px 8px;'
                            f'border-radius:100px;">{rubric}</span>'
                            if rubric else ""
                        )
                        ui.html(
                            f'<div style="display:flex;align-items:center;gap:14px;'
                            f'flex-wrap:wrap;margin-bottom:10px;">'
                            f'<div style="font-size:2.4rem;font-weight:800;'
                            f'color:{_score_color(overall)};">{overall}</div>'
                            f'<div>'
                            f'<span style="background:{vstyle[2]};color:{vstyle[1]};'
                            f'font-size:0.75rem;font-weight:700;padding:3px 10px;'
                            f'border-radius:100px;">{vstyle[0]}</span>'
                            f'{_rubric_badge}'
                            f'<div style="color:#64748b;font-size:0.75rem;margin-top:4px;">'
                            f'Judged {judged_at} · {j.get("judge_model","")}</div>'
                            f'</div></div>'
                        )

                        # ── QA summary ──────────────────────────────────────
                        if summary:
                            bg = "#fee2e2" if verdict == "unsafe" else "#f1f5f9"
                            tc = "#7f1d1d" if verdict == "unsafe" else "#334155"
                            ui.html(
                                f'<div style="background:{bg};border-radius:8px;padding:10px 14px;'
                                f'font-size:0.85rem;color:{tc};line-height:1.55;margin-bottom:10px;">'
                                f'{summary}</div>'
                            )

                        # ── Issue tag chips ─────────────────────────────────
                        if tags:
                            chips_html = "".join(
                                f'<span style="background:#fef3c7;color:#92400e;font-size:0.72rem;'
                                f'font-weight:600;padding:3px 9px;border-radius:100px;margin:2px;">'
                                f'{_TAG_LABELS.get(t, t)}</span>'
                                for t in tags
                            )
                            ui.html(f'<div style="display:flex;flex-wrap:wrap;gap:3px;margin-bottom:8px;">{chips_html}</div>')

                        # ── What went well ──────────────────────────────────
                        if what_went_well:
                            _section_header("What went well")
                            bullets = "".join(
                                f'<div style="display:flex;gap:8px;margin-bottom:4px;">'
                                f'<span style="color:#15803d;font-weight:700;flex-shrink:0;">✓</span>'
                                f'<span style="font-size:0.83rem;color:#1e293b;line-height:1.5;">{item}</span>'
                                f'</div>'
                                for item in what_went_well
                            )
                            ui.html(f'<div style="background:#f0fdf4;border-radius:8px;padding:10px 12px;margin-bottom:8px;">{bullets}</div>')

                        # ── What needs work ─────────────────────────────────
                        if what_needs_work:
                            _section_header("What needs work")
                            bullets = "".join(
                                f'<div style="display:flex;gap:8px;margin-bottom:4px;">'
                                f'<span style="color:#b45309;font-weight:700;flex-shrink:0;">⚠</span>'
                                f'<span style="font-size:0.83rem;color:#1e293b;line-height:1.5;">{item}</span>'
                                f'</div>'
                                for item in what_needs_work
                            )
                            ui.html(f'<div style="background:#fffbeb;border-radius:8px;padding:10px 12px;margin-bottom:8px;">{bullets}</div>')

                        # ── Evidence audit ──────────────────────────────────
                        if claims:
                            _section_header("Evidence Audit")
                            rows = []
                            for c in claims:
                                v = c.get("verdict", "unsupported")
                                cs = _CLAIM_STYLE.get(v, _CLAIM_STYLE["unsupported"])
                                rows.append(
                                    f'<div style="border:1px solid #e2e8f0;border-radius:7px;'
                                    f'padding:8px 10px;margin-bottom:5px;background:#fafafa;">'
                                    f'<div style="display:flex;align-items:flex-start;gap:8px;flex-wrap:wrap;">'
                                    f'<span style="background:{cs[0]};color:{cs[1]};font-size:0.68rem;'
                                    f'font-weight:700;padding:2px 7px;border-radius:100px;white-space:nowrap;'
                                    f'flex-shrink:0;margin-top:1px;">{cs[2]}</span>'
                                    f'<span style="font-size:0.82rem;font-weight:600;color:#1e293b;'
                                    f'flex:1;">{c.get("claim","")}</span>'
                                    f'</div>'
                                    f'<div style="font-size:0.79rem;color:#64748b;margin-top:4px;'
                                    f'line-height:1.4;">{c.get("note","")}</div>'
                                    f'</div>'
                                )
                            ui.html(f'<div style="margin-bottom:8px;">{"".join(rows)}</div>')

                        # ── Specific findings (new) or legacy improvements ──
                        if findings:
                            _section_header("Specific Findings")
                            for f in findings:
                                tag_label = _TAG_LABELS.get(f.get("tag",""), f.get("tag",""))
                                excerpt   = f.get("excerpt","")
                                problem   = f.get("problem","")
                                better    = f.get("better","")
                                pfix      = f.get("pipeline_fix","")
                                ui.html(
                                    f'<div style="border:1px solid #e0e7ff;border-radius:8px;'
                                    f'padding:10px 12px;margin-bottom:8px;background:#fafbff;">'
                                    f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:6px;">'
                                    f'<span style="background:#fef3c7;color:#92400e;font-size:0.68rem;'
                                    f'font-weight:700;padding:2px 8px;border-radius:100px;">{tag_label}</span>'
                                    f'</div>'
                                    + (
                                        f'<div style="background:#f1f5f9;border-left:3px solid #94a3b8;'
                                        f'border-radius:0 5px 5px 0;padding:5px 10px;margin-bottom:7px;'
                                        f'font-size:0.8rem;color:#475569;font-style:italic;">"{excerpt}"</div>'
                                        if excerpt else ""
                                    )
                                    + (
                                        f'<div style="font-size:0.8rem;color:#7c2d12;'
                                        f'margin-bottom:6px;line-height:1.45;">'
                                        f'<strong>Problem:</strong> {problem}</div>'
                                        if problem else ""
                                    )
                                    + (
                                        f'<div style="background:#f0fdf4;border-radius:5px;'
                                        f'padding:6px 10px;margin-bottom:6px;font-size:0.8rem;'
                                        f'color:#14532d;line-height:1.45;">'
                                        f'<strong>Better:</strong> {better}</div>'
                                        if better else ""
                                    )
                                    + (
                                        f'<div style="font-size:0.75rem;color:#64748b;'
                                        f'line-height:1.4;">Pipeline fix: {pfix}</div>'
                                        if pfix else ""
                                    )
                                    + f'</div>'
                                )
                        elif legacy_imps:
                            # Fallback: show legacy suggested_improvements for old results
                            _section_header("Suggested Improvements")
                            for imp in legacy_imps:
                                ui.html(
                                    f'<div style="border-left:3px solid #6366f1;padding:8px 12px;'
                                    f'margin-bottom:6px;background:#fafafa;border-radius:0 6px 6px 0;">'
                                    f'<div style="font-size:0.78rem;font-weight:600;color:#1e293b;">'
                                    f'{_TAG_LABELS.get(imp.get("issue",""), imp.get("issue",""))}</div>'
                                    f'<div style="font-size:0.78rem;color:#475569;margin-top:2px;">'
                                    f'{imp.get("proposed_fix","")}</div>'
                                    f'</div>'
                                )

                        # ── Mode review ─────────────────────────────────────
                        if mode_review:
                            _section_header("Mode Evaluation")
                            ui.html(
                                f'<div style="background:#f8fafc;border-radius:8px;padding:10px 12px;'
                                f'font-size:0.83rem;color:#334155;line-height:1.55;margin-bottom:8px;">'
                                f'{mode_review}</div>'
                            )

                        # ── Professional / verify-first review ───────────────
                        # assessment_first → "Professional Assessment Review"
                        # verify_first     → "Verify-first Conditions Review"
                        # all other modes  → hidden (field is null for those modes)
                        if prof_review and rubric == "assessment_first":
                            _section_header("Professional Assessment Review")
                            ui.html(
                                f'<div style="background:#eff6ff;border-left:3px solid #3b82f6;'
                                f'border-radius:0 8px 8px 0;padding:10px 12px;'
                                f'font-size:0.83rem;color:#1e3a5f;line-height:1.55;margin-bottom:8px;">'
                                f'{prof_review}</div>'
                            )
                        elif prof_review and rubric == "verify_first":
                            _section_header("Verify-first Conditions Review")
                            ui.html(
                                f'<div style="background:#eff6ff;border-left:3px solid #3b82f6;'
                                f'border-radius:0 8px 8px 0;padding:10px 12px;'
                                f'font-size:0.83rem;color:#1e3a5f;line-height:1.55;margin-bottom:8px;">'
                                f'{prof_review}</div>'
                            )

                        # ── Dimension scores (lower priority, always present) ─
                        _section_header("Dimension Scores")
                        _DIM_LABELS = {
                            "safety_correctness":            "Safety",
                            "routing_appropriateness":       "Routing",
                            "repair_step_detail":            "Step detail",
                            "tools_materials_specificity":   "Tools",
                            "grounding":                     "Grounding",
                            "hallucination_absence":         "No hallucinations",
                            "safety_warning_completeness":   "Warnings",
                            "clarification_appropriateness": "Clarification",
                            "user_clarity":                  "User clarity",
                        }
                        with ui.element("div").style(
                            "display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:5px;margin-bottom:10px;"
                        ):
                            for key, label in _DIM_LABELS.items():
                                val = dims.get(key)
                                if val is None:
                                    col, display = "#94a3b8", "N/A"
                                else:
                                    col, display = _score_color(int(val)), str(val)
                                ui.html(
                                    f'<div style="background:#f8fafc;border:1px solid #e2e8f0;'
                                    f'border-radius:7px;padding:6px 9px;">'
                                    f'<div style="font-size:0.7rem;color:#64748b;">{label}</div>'
                                    f'<div style="font-size:1rem;font-weight:700;color:{col};">{display}</div>'
                                    f'</div>'
                                )

                with ui.element("div").style("margin-top:18px;"):
                    with ui.expansion(
                        "Quality Evaluation (Admin)",
                        icon="verified",
                    ).style(
                        "border:1px solid #e0e7ff;border-radius:12px;"
                        "background:#fafbff;font-weight:600;color:#3730a3;"
                    ):
                        result_area = ui.element("div").style("padding:4px 0 8px;")

                        async def _load_existing():
                            try:
                                async with httpx.AsyncClient() as c:
                                    r = await c.get(
                                        f"{API}/cases/{case_id}/judge",
                                        headers={"Authorization": f"Bearer {token}"},
                                        timeout=15.0,
                                    )
                                if r.status_code == 200:
                                    _render_result(r.json(), result_area)
                                else:
                                    result_area.clear()
                                    with result_area:
                                        ui.html(
                                            '<div style="color:#94a3b8;font-size:0.84rem;'
                                            'padding:8px 0;">No evaluation yet. Click Run Judge below.</div>'
                                        )
                            except Exception:
                                pass

                        async def _run_judge():
                            result_area.clear()
                            with result_area:
                                ui.html(
                                    '<div style="color:#6366f1;font-size:0.84rem;padding:8px 0;">'
                                    'Running judge (CPU inference — may take 1-3 min)…</div>'
                                )
                            try:
                                async with httpx.AsyncClient() as c:
                                    r = await c.post(
                                        f"{API}/cases/{case_id}/judge",
                                        headers={"Authorization": f"Bearer {token}"},
                                        timeout=240.0,
                                    )
                                if r.status_code == 200:
                                    _render_result(r.json(), result_area)
                                elif r.status_code == 503:
                                    result_area.clear()
                                    with result_area:
                                        ui.html(
                                            '<div style="color:#92400e;font-size:0.84rem;padding:8px 0;">'
                                            'Judge is disabled. Set JUDGE_ENABLED=true in backend/.env '
                                            'and restart, then pull qwen2.5:7b via Ollama.</div>'
                                        )
                                else:
                                    detail = r.json().get("detail", r.text) if r.headers.get("content-type","").startswith("application/json") else r.text
                                    ui.notify(f"Judge failed: {detail}", type="negative", position="top-right")
                            except Exception as exc:
                                ui.notify(f"Judge error: {exc}", type="negative", position="top-right")

                        ui.timer(0.1, _load_existing, once=True)

                        with ui.element("div").style("margin-top:10px;"):
                            ui.button("Run Judge", icon="play_arrow", on_click=_run_judge) \
                                .props("unelevated color=indigo size=sm") \
                                .style("font-weight:600;")

            # ── Master redraw, called on every state transition ──────────────
            def _redraw(data):
                """
                Redraw btn_area + analysis_box from data.
                data is the full analysis dict, or None if 404.
                """
                status = (data or {}).get("status")
                # Track whether the current analysis is in clarification output mode
                _sr_now = (data or {}).get("structured_result") or {}
                _clarif_mode_ref["v"] = (
                    _sr_now.get("output_mode") == "clarification"
                )
                # Track the bounded-clarification fallback (no questions, inconclusive)
                _inconclusive_ref["v"] = bool(
                    (_sr_now.get("clarification") or {}).get("inconclusive")
                )
                # Track out-of-scope (emergency / not_repair) — chat gives no repair help
                _out_of_scope_ref["v"] = (
                    (_sr_now.get("out_of_scope") or {}).get("scope", "")
                    if _sr_now.get("output_mode") == "out_of_scope" else ""
                )
                # Track diy/verify_first — the only modes with the "Before you begin" gate
                _diy_mode_ref["v"] = _sr_now.get("output_mode") in ("diy", "verify_first")
                # Lock Ask Knitra when the case is assessment_first and unacknowledged
                _chat_locked_ref["v"] = (
                    ((data or {}).get("structured_result") or {}).get("output_mode") == "assessment_first"
                    and not (data or {}).get("hazard_ack", False)
                )

                # Button area
                btn_area.clear()
                with btn_area:
                    if status in ("running", "needs_clarification"):
                        # running    , no button, pipeline can't be cancelled mid-flight
                        # needs_clarification, button lives inside the card so the user
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
                        _draw_feedback_widget(data)
                        if app.storage.user.get("is_admin", False):
                            _draw_judge_panel()
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

            # Mutable cell: True when the latest complete analysis has output_mode=clarification.
            # Updated by _redraw; read by _render_chat_section.
            _clarif_mode_ref = {"v": False}

            # Mutable cell: True when the clarification output is the bounded-rounds
            # fallback (inconclusive, no questions).  In that state the chat must NOT
            # tell the user to "answer the questions below" — there are none.
            # Updated by _redraw; read by _render_chat_section.
            _inconclusive_ref = {"v": False}

            # Mutable cell: "emergency" / "not_repair" / "" — set when output_mode is
            # out_of_scope. Ask Knitra must not give repair guidance in these states.
            # Updated by _redraw; read by _render_chat_section.
            _out_of_scope_ref = {"v": ""}

            # Mutable cell: True when output_mode is diy/verify_first — the ONLY modes
            # that hide repair steps behind the "Before you begin" acknowledgement. The
            # DIY chat gate must apply only to these; otherwise it wrongly blocks Ask
            # Knitra for assessment_first (after hazard ack) and care cases, which have
            # no such steps gate. Updated by _redraw; read by _render_chat_section.
            _diy_mode_ref = {"v": False}

            # Mutable cell: True when the case is assessment_first AND the user has NOT yet
            # acknowledged the hazard notice.  Locks Ask Knitra until "I Understand" is clicked.
            # Updated by _redraw; read by _render_chat_section.
            _chat_locked_ref = {"v": False}

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
                if not _title_updated and stage in (
                    "stage_1_complete",
                    "stage_2_running", "stage_2_complete",
                    "stage_3_running", "stage_3_complete",
                    "stage_a_running", "stage_a_complete",
                    "stage_c_running", "stage_c_complete",
                    "complete",
                ):
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
                    if _was_running and st in ("complete", "needs_clarification"):
                        _was_running = False
                        await _render_chat_section(st)
                    elif _was_running and st == "failed":
                        _was_running = False

            poll_timer = ui.timer(3.5, _do_poll, active=False)

            # ── Initial page load ─────────────────────────────────────────────
            # Fetch whatever the current analysis state is (may be None, running,
            # complete, or failed) and render it immediately without waiting for
            # the first timer tick.
            _initial = await _get_analysis()
            # Seed clarif mode ref BEFORE _redraw so _render_chat_section reads it correctly.
            _clarif_mode_ref["v"] = (
                ((_initial or {}).get("structured_result") or {}).get("output_mode") == "clarification"
            )
            # Seed inconclusive ref (bounded-clarification fallback)
            _inconclusive_ref["v"] = bool(
                (((_initial or {}).get("structured_result") or {}).get("clarification") or {})
                .get("inconclusive")
            )
            # Seed out-of-scope ref (emergency / not_repair)
            _initial_sr = ((_initial or {}).get("structured_result") or {})
            _out_of_scope_ref["v"] = (
                (_initial_sr.get("out_of_scope") or {}).get("scope", "")
                if _initial_sr.get("output_mode") == "out_of_scope" else ""
            )
            # Seed diy-mode ref (diy / verify_first only)
            _diy_mode_ref["v"] = _initial_sr.get("output_mode") in ("diy", "verify_first")
            # Seed chat-locked ref: assessment_first + not yet acknowledged
            _chat_locked_ref["v"] = (
                ((_initial or {}).get("structured_result") or {}).get("output_mode") == "assessment_first"
                and not (_initial or {}).get("hazard_ack", False)
            )
            _redraw(_initial)
            if (_initial or {}).get("status") in ("pending", "running"):
                _was_running = True
                poll_timer.active = True

            # ── Chat section, dynamic container ──────────────────────────────
            # Uses a plain div that can be cleared and rebuilt when analysis
            # completes while the user is watching (poll timer calls
            # _render_chat_section(True) instead of navigating away).
            _initial_status    = (_initial or {}).get("status", "")
            _analysis_complete = _initial_status in ("complete", "needs_clarification")
            chat_section = ui.element("div")

            async def _render_chat_section(analysis_status):
                """
                Rebuild the chat section in-place.
                Safe to call multiple times, clears and recreates all child elements.
                Called at page load and by _do_poll when analysis transitions to
                'complete' or 'needs_clarification'.

                analysis_status: the current analysis status string, or a bool
                (True = complete) for backwards compatibility.
                """
                # Accept bare True/False for legacy call sites
                if analysis_status is True:
                    analysis_status = "complete"
                elif analysis_status is False:
                    analysis_status = "pending"
                is_complete      = analysis_status in ("complete", "needs_clarification")
                is_clarification = analysis_status == "needs_clarification"
                # Also treat output_mode=clarification (status=complete) as clarification
                is_clarif_mode   = _clarif_mode_ref["v"] or is_clarification
                # Bounded-clarification fallback: clarif mode but NO questions to answer.
                is_inconclusive  = _inconclusive_ref["v"]

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

                    # ── Hazard-locked placeholder ──────────────────────────────
                    # Shown for assessment_first cases when hazard_ack is False.
                    # Cleared automatically on page reload after "I Understand" is
                    # clicked (at that point hazard_ack=True, so this branch is skipped).
                    if _chat_locked_ref["v"]:
                        with ui.element("div").style(
                            "background:white;border:1px solid #fecaca;"
                            "border-radius:14px;padding:28px 24px;"
                            "text-align:center;"
                        ):
                            ui.html(
                                '<div style="font-size:2rem;margin-bottom:10px;'
                                'opacity:0.5;">&#128274;</div>'
                                '<div style="font-weight:700;color:#991b1b;'
                                'font-size:0.95rem;margin-bottom:8px;">Ask Knitra</div>'
                                '<div style="color:#64748b;font-size:0.83rem;'
                                'line-height:1.6;max-width:380px;margin:0 auto;">'
                                'Ask Knitra is available after you acknowledge the hazardous '
                                'case notice above. This does not override the recommendation '
                                'for professional assessment.'
                                '</div>'
                            )
                        return

                    # ── Out-of-scope placeholder ───────────────────────────────
                    # For emergency / not_repair cases Ask Knitra must not give repair
                    # guidance. Emergency points to emergency services; non-repair asks
                    # for a proper repair image.
                    if _out_of_scope_ref["v"]:
                        _oos_scope = _out_of_scope_ref["v"]
                        if _oos_scope == "emergency":
                            _oos_icon, _oos_color, _oos_msg = (
                                "&#128721;", "#991b1b",
                                "Ask Knitra is unavailable for active emergencies. Move to "
                                "safety and contact emergency services. Knitra does not provide "
                                "guidance while a hazard is active.",
                            )
                        else:
                            _oos_icon, _oos_color, _oos_msg = (
                                "&#128247;", "#92400e",
                                "Ask Knitra opens once a household repair has been identified. "
                                "Please upload a clear image of the repair problem above.",
                            )
                        with ui.element("div").style(
                            "background:white;border:1px solid #e2e8f0;"
                            "border-radius:14px;padding:28px 24px;text-align:center;"
                        ):
                            ui.html(
                                f'<div style="font-size:2rem;margin-bottom:10px;'
                                f'opacity:0.5;">{_oos_icon}</div>'
                                f'<div style="font-weight:700;color:{_oos_color};'
                                f'font-size:0.95rem;margin-bottom:8px;">Ask Knitra</div>'
                                f'<div style="color:#64748b;font-size:0.83rem;'
                                f'line-height:1.6;max-width:380px;margin:0 auto;">'
                                f'{_oos_msg}</div>'
                            )
                        return

                    # ── DIY safety-gate placeholder ────────────────────────────
                    # In DIY mode, the repair steps are hidden behind the "Before
                    # you begin" acknowledgement. Keep the chat locked to the same
                    # gate so there is no gap between seeing the steps and being
                    # able to ask questions about them.
                    # Applies ONLY to diy/verify_first (the modes that gate repair steps
                    # behind "Before you begin"). For assessment_first the hazard
                    # acknowledgement is the relevant gate — once given, Ask Knitra opens.
                    _ack_key     = f"ack_{case_id}"
                    _diy_unacked = (
                        _diy_mode_ref["v"]
                        and not is_clarif_mode
                        and not _chat_locked_ref["v"]
                        and not app.storage.user.get(_ack_key, False)
                    )
                    if _diy_unacked:
                        with ui.element("div").style(
                            "background:white;border:1px solid #e2e8f0;"
                            "border-radius:14px;padding:28px 24px;"
                            "text-align:center;"
                        ):
                            ui.html(
                                '<div style="font-size:2rem;margin-bottom:10px;'
                                'opacity:0.35;">&#128172;</div>'
                                '<div style="font-weight:700;color:#334155;'
                                'font-size:0.95rem;margin-bottom:8px;">Ask Knitra</div>'
                                '<div style="color:#64748b;font-size:0.83rem;'
                                'line-height:1.6;max-width:380px;margin:0 auto;">'
                                'Read and confirm the safety notice above to open Ask Knitra.'
                                '</div>'
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
                                if is_inconclusive:
                                    _chat_subtitle = (
                                        "No more questions right now. Add a clearer photo or "
                                        "description above, then press Re-analyse — or check "
                                        "the issue in person."
                                    )
                                elif is_clarif_mode:
                                    _chat_subtitle = (
                                        "Answer the questions below, then press Re-analyse "
                                        "for an updated plan"
                                    )
                                else:
                                    _chat_subtitle = (
                                        "Knows your full case. Chat answers are conversational, "
                                        "press Re-analyse above for an official updated diagnosis."
                                    )
                                ui.html(
                                    f'<div style="font-size:0.75rem;color:#94a3b8;'
                                    f'margin-top:1px;">{_chat_subtitle}</div>'
                                )
                                ui.html(
                                    '<div style="font-size:0.7rem;color:#94a3b8;'
                                    'margin-top:3px;line-height:1.45;">'
                                    'AI advice only &mdash; not a substitute for a qualified professional.'
                                    '</div>'
                                )
                            # Expand / collapse the message area for more room.
                            ui.html(
                                '<button title="Give the chat more space" '
                                'onclick="var w=document.querySelector(&#39;.chat-wrap&#39;);'
                                'if(w){var ex=w.classList.toggle(&#39;expanded&#39;);'
                                'this.textContent=ex?&#39;Collapse&#39;:&#39;Expand&#39;;}" '
                                'style="margin-left:auto;flex-shrink:0;background:#f1f5f9;'
                                'border:1px solid #e2e8f0;border-radius:8px;padding:5px 11px;'
                                'font-size:0.72rem;font-weight:600;color:#475569;cursor:pointer;">'
                                'Expand</button>'
                            )

                        # Message list
                        chat_messages = ui.element("div").classes("chat-wrap")

                        if is_inconclusive:
                            _SUGGESTIONS = [
                                "What kind of photo would help most?",
                                "Why couldn't you assess this?",
                                "Should I call a professional?",
                                "What should I check in person?",
                            ]
                        elif is_clarif_mode:
                            _SUGGESTIONS = [
                                "It's a plastic push-fit fitting",
                                "The wall is plasterboard",
                                "There's no sign of damp",
                                "The pipe is copper",
                                "Can I describe this instead?",
                            ]
                        else:
                            _SUGGESTIONS = [
                                "What tools do I need?",
                                "Is this safe to DIY?",
                                "How long will this take?",
                                "What could go wrong?",
                                "How is confidence calculated?",
                            ]

                        async def _reanalyse_from_chat(text_context: str = ""):
                            """Trigger the official Re-analyse then show pipeline progress in-place.

                            text_context is passed to the analysis endpoint so Stage 1 can
                            incorporate clarification answers.  The manual button calls this
                            with no args (empty string); auto-reanalysis passes the joined answers.
                            Instead of reloading the page, we activate the poll timer so the
                            analysis panel updates with live pipeline stages while chat stays visible.
                            """
                            nonlocal _was_running, _title_updated
                            try:
                                body = {"text_context": text_context} if text_context else {}
                                async with httpx.AsyncClient() as c:
                                    r = await c.post(
                                        f"{API}/cases/{case_id}/analyse",
                                        json=body,
                                        headers={"Authorization": f"Bearer {token}"},
                                        timeout=30.0,
                                    )
                                if r.status_code in (200, 202):
                                    _was_running = True
                                    _title_updated = False
                                    fresh = await _get_analysis()
                                    _redraw(fresh)
                                    poll_timer.active = True
                                else:
                                    ui.notify(f"Could not start re-analysis ({r.status_code}).",
                                              type="negative", position="top-right")
                            except Exception:
                                ui.notify("Could not reach the backend.",
                                          type="negative", position="top-right")

                        async def _new_case_from_photo(image_ids, trade):
                            """Spawn a new case from a chat photo (different repair) and go to it."""
                            try:
                                async with httpx.AsyncClient() as c:
                                    r = await c.post(
                                        f"{API}/cases/{case_id}/chat/spawn-case",
                                        json={"image_ids": image_ids, "apparent_trade": trade},
                                        headers={"Authorization": f"Bearer {token}"},
                                        timeout=30.0,
                                    )
                                if r.status_code == 200:
                                    ui.navigate.to(f"/cases/{r.json().get('case_id')}")
                                else:
                                    ui.notify(f"Could not create a new case ({r.status_code}).",
                                              type="negative", position="top-right")
                            except Exception:
                                ui.notify("Could not reach the backend.",
                                          type="negative", position="top-right")

                        def _render_followup_action(meta):
                            """Contextual action under an assistant reply about an attached photo."""
                            if not isinstance(meta, dict):
                                return
                            verdict = meta.get("followup_verdict")
                            if verdict == "same":
                                ui.button("Re-analyse with this photo",
                                          on_click=lambda: _reanalyse_from_chat()) \
                                    .props("outline color=primary size=sm icon=autorenew") \
                                    .style("margin:0 0 6px;font-size:0.74rem;font-weight:600;")
                            elif verdict == "different":
                                _ids = meta.get("image_ids") or []
                                _trade = meta.get("apparent_trade") or ""
                                ui.button("Start a new case with this photo",
                                          on_click=lambda i=_ids, t=_trade: _new_case_from_photo(i, t)) \
                                    .props("outline color=primary size=sm icon=add_circle") \
                                    .style("margin:0 0 6px;font-size:0.74rem;font-weight:600;")
                            elif verdict == "not_repair":
                                # The attached photo is not a DIY / repair subject (e.g. a document,
                                # screenshot, person, food, landscape). The backend already detected
                                # this and the assistant's reply explains it; the UI adds an explicit
                                # amber notice + a one-click way to choose a different photo, so the
                                # user always has a clear next step instead of just prose.
                                ui.html(
                                    '<div style="display:flex;align-items:center;gap:8px;'
                                    'background:#fef3c7;border:1px solid #fcd34d;color:#92400e;'
                                    'border-radius:8px;padding:8px 12px;margin:2px 0 8px;'
                                    'font-size:0.78rem;font-weight:600;line-height:1.4;">'
                                    '<span style="font-size:0.95rem;flex-shrink:0;">&#9888;</span>'
                                    '<span>This does not look like a DIY repair photo. '
                                    'Knitra can only help with repair and maintenance issues.</span>'
                                    '</div>'
                                )
                                ui.button("Upload a different photo",
                                          on_click=lambda: ui.run_javascript(
                                              "(function(){"
                                              "var el=document.getElementById('knitra-chat-file');"
                                              "if(el){el.click();}"
                                              "})()"
                                          )) \
                                    .props("outline color=primary size=sm icon=image") \
                                    .style("margin:0 0 6px;font-size:0.74rem;font-weight:600;")

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
                                        if msg["role"] == "user":
                                            md = msg.get("msg_metadata") or {}
                                            paths = md.get("image_paths") or []
                                            srcs = [f"{BACKEND}/static/uploads/{p}" for p in paths]
                                            ui.html(_chat_user_html(msg["content"], srcs))
                                        else:
                                            ui.html(
                                                f'<div class="chat-bubble-assistant">'
                                                f'{msg["content"]}</div>'
                                            )
                                            _render_followup_action(msg.get("msg_metadata"))

                                    # Refresh recovery: if the last message is a user bubble
                                    # with no following assistant reply, the previous send was
                                    # cut short (page reload, navigation, network hiccup).
                                    # The user message + photo are persisted server-side and
                                    # already visible above; surface a clear notice + retry
                                    # button so the user has an actionable next step instead
                                    # of silent uncertainty.
                                    last = msgs[-1]
                                    if last.get("role") == "user":
                                        _had_photo = bool(
                                            (last.get("msg_metadata") or {}).get("image_paths")
                                        )
                                        _label = ("Your last message and photo were saved, but "
                                                  "Knitra did not finish replying (the page "
                                                  "refreshed mid-thought). Tap to ask again."
                                                  if _had_photo else
                                                  "Your last message was saved, but Knitra did "
                                                  "not finish replying (the page refreshed mid"
                                                  "-thought). Tap to ask again.")
                                        ui.html(
                                            '<div style="display:flex;align-items:center;gap:8px;'
                                            'background:#eef2ff;border:1px solid #c7d2fe;'
                                            'color:#3730a3;border-radius:8px;padding:8px 12px;'
                                            'margin:2px 0 8px;font-size:0.78rem;font-weight:500;'
                                            'line-height:1.45;">'
                                            '<span style="font-size:0.95rem;flex-shrink:0;">'
                                            '&#128172;</span>'
                                            f'<span>{_label}</span></div>'
                                        )
                                        _last_content = last.get("content") or ""
                                        _last_img_ids = (last.get("msg_metadata") or {}).get(
                                            "image_ids"
                                        ) or []
                                        ui.button(
                                            "Ask Knitra again",
                                            on_click=lambda t=_last_content,
                                                          i=list(_last_img_ids):
                                                _resume_after_refresh(t, i),
                                        ).props("outline color=primary size=sm icon=replay") \
                                          .style("margin:0 0 6px;font-size:0.74rem;"
                                                 "font-weight:600;")

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
                                # Mic button, SVG microphone icon, class gives hover + active styles
                                ui.html(
                                    '<button id="knitra-chat-mic" '
                                    'onclick="knitraChatToggleVoice()" '
                                    'title="Voice input, tap and speak" '
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
                                # Attach button, paperclip SVG icon, triggers hidden file input
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
                                # Status line (voice recording / upload feedback) -
                                # JS turns this into a coloured pill per state.
                                ui.html(
                                    '<span id="knitra-chat-status" '
                                    'class="knitra-status-pill" '
                                    'style="display:none;margin-left:4px;"></span>'
                                )

                            # Persistent attachment area, lives ABOVE the textarea and
                            # survives chat re-renders. JS (_chatRenderAttachments) repaints
                            # it from window._knitraPendingAttachments after every render,
                            # so queued photos never silently vanish.
                            ui.html(
                                '<div id="knitra-chat-attachments" '
                                'class="knitra-attach-wrap" style="display:none;"></div>'
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

                                send_btn = ui.button("Send", on_click=lambda: _send_chat()) \
                                    .props("unelevated color=primary") \
                                    .style("font-weight:600;flex-shrink:0;") \
                                    .classes("knitra-send-btn")

                        # ── Clarification Re-analyse CTA ──────────────────────
                        # Shown only in clarification output-mode so the user
                        # has a clear "I've answered, now what?" action.
                        # In the inconclusive fallback there are no questions, so the
                        # copy and button label avoid implying answers are pending.
                        if is_clarif_mode:
                            with ui.element("div").style(
                                "padding:0 20px 14px;"
                                "display:flex;align-items:center;gap:10px;"
                            ):
                                if is_inconclusive:
                                    _cta_text = (
                                        'No questions are pending. Add a clearer photo or a '
                                        'short description above, then re-analyse — or have '
                                        'the issue checked in person.'
                                    )
                                    _cta_label = "Re-analyse"
                                else:
                                    _cta_text = (
                                        'Knitra will automatically re-analyse once the required '
                                        'questions are answered. Use the button if you prefer '
                                        'to trigger it manually.'
                                    )
                                    _cta_label = "Re-analyse with answers"
                                ui.html(
                                    f'<div style="font-size:0.78rem;color:#64748b;flex:1;">'
                                    f'{_cta_text}'
                                    f'</div>'
                                )
                                ui.button(
                                    _cta_label,
                                    on_click=_reanalyse_from_chat,
                                ).props("outline color=primary size=sm icon=autorenew") \
                                 .style("font-size:0.74rem;font-weight:700;flex-shrink:0;")

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

                        _busy = {"v": False}   # in-flight guard (mutable cell, no nonlocal needed)

                        async def _send_chat():
                            # Re-entrancy guard: ignore double-clicks / Enter while a reply is
                            # still in flight, and show the Send button as loading + disabled.
                            if _busy["v"]:
                                return
                            _busy["v"] = True
                            send_btn.props("loading")
                            send_btn.disable()
                            try:
                                await _do_send()
                            finally:
                                _busy["v"] = False
                                send_btn.props(remove="loading")
                                send_btn.enable()

                        async def _do_send():
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

                            # Collect any pending image IDs first, so an image-only send
                            # (no typed text, just an attached photo) is also valid.
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

                            # Bail out ONLY if there is nothing to send at all.
                            # Either a typed message OR at least one attached photo is enough.
                            if not content and not image_ids:
                                return

                            # If the user only attached a photo (no text), use a short implicit
                            # prompt so the assistant has something to react to. Shown in the
                            # user bubble too, so the message thread reads naturally.
                            if not content and image_ids:
                                content = "Please look at this photo."

                            # Capture attachment thumbnails (data URLs) BEFORE clearing the
                            # chips, so the sent message can show the actual photo immediately.
                            image_dataurls: list[str] = []
                            if image_ids:
                                try:
                                    raw_urls = await ui.run_javascript(
                                        "(window._knitraPendingAttachments||[])"
                                        ".map(function(a){return a.dataUrl||'';})",
                                        timeout=2.0,
                                    )
                                    if isinstance(raw_urls, list):
                                        image_dataurls = [str(u) for u in raw_urls if u]
                                except Exception:
                                    pass

                            chat_input.set_value("")
                            # Clear JS-side state: transcript, queued image IDs, the
                            # persistent attachment chips, and the status pill.
                            await ui.run_javascript(
                                "window._knitraTranscribedText = '';"
                                "window._knitraChatStatusState = null;"
                                "if(window._chatClearAttachments)window._chatClearAttachments();"
                                "var _st=document.getElementById('knitra-chat-status');"
                                "if(_st){_st.style.display='none';_st.textContent='';}"
                            )

                            with chat_messages:
                                # User bubble with any attached photo thumbnails inline.
                                ui.html(_chat_user_html(content, image_dataurls))
                                # Image-aware loading state. For an image send we cycle through
                                # step-by-step labels (vision pass -> case comparison -> reply
                                # composition) so the user sees Knitra is actively progressing,
                                # not idling. The interval is cleared when the first reply token
                                # arrives (see "typing.delete()" further down).
                                _think = ("&#128247; Looking at your photo&#8230;" if image_ids
                                          else "Knitra is thinking&#8230;")
                                typing = ui.html(
                                    f'<div id="knitra-chat-typing" '
                                    f'class="chat-bubble-typing">{_think}</div>'
                                )
                            if image_ids:
                                # Rotate labels every 3.5s so the wait feels purposeful.
                                # innerText is read each tick to avoid stale closures.
                                await ui.run_javascript(
                                    "(function(){"
                                    "  var labels = ["
                                    "    '\\uD83D\\uDCF7 Looking at your photo\\u2026',"
                                    "    '\\uD83D\\uDD0D Comparing with your case\\u2026',"
                                    "    '\\uD83D\\uDCAD Thinking about a reply\\u2026'"
                                    "  ];"
                                    "  var i = 0;"
                                    "  if (window._knitraChatProgressInt) {"
                                    "    clearInterval(window._knitraChatProgressInt);"
                                    "  }"
                                    "  window._knitraChatProgressInt = setInterval(function(){"
                                    "    var el = document.getElementById('knitra-chat-typing');"
                                    "    if (!el) {"
                                    "      clearInterval(window._knitraChatProgressInt);"
                                    "      window._knitraChatProgressInt = null;"
                                    "      return;"
                                    "    }"
                                    "    i = (i + 1) % labels.length;"
                                    "    el.innerHTML = labels[i];"
                                    "  }, 3500);"
                                    "})()"
                                )

                            # Stream the reply token-by-token for a live "typing" effect.
                            # The stream ends with a '\x1e<json>' sentinel carrying the
                            # follow-up verdict metadata (for the action button).
                            try:
                                raw = ""
                                meta = {}
                                assistant_bubble = None
                                _i = 0
                                async with httpx.AsyncClient(timeout=120.0) as c:
                                    async with c.stream(
                                        "POST",
                                        f"{API}/cases/{case_id}/chat/stream",
                                        json={"content": content, "image_ids": image_ids},
                                        headers={"Authorization": f"Bearer {token}"},
                                    ) as resp:
                                        if resp.status_code != 200:
                                            await resp.aread()
                                            raise RuntimeError(f"stream status {resp.status_code}")
                                        async for chunk in resp.aiter_text():
                                            if not chunk:
                                                continue
                                            raw += chunk
                                            _vis = _clean_chat(raw.split("\x1e", 1)[0])
                                            if not _vis:
                                                continue   # nothing visible yet (sentinel only)
                                            if assistant_bubble is None:
                                                # First visible token: swap the thinking dots
                                                # for the reply bubble (keeps "thinking" during
                                                # the model-load latency).
                                                try:
                                                    typing.delete()
                                                except Exception:
                                                    pass
                                                # Stop the per-step progress label rotation.
                                                await ui.run_javascript(
                                                    "if(window._knitraChatProgressInt){"
                                                    "clearInterval(window._knitraChatProgressInt);"
                                                    "window._knitraChatProgressInt=null;}"
                                                )
                                                with chat_messages:
                                                    assistant_bubble = ui.html(
                                                        '<div class="chat-bubble-assistant"></div>'
                                                    )
                                            _i += 1
                                            if _i % 2 == 0:   # throttle UI repaints
                                                assistant_bubble.set_content(
                                                    f'<div class="chat-bubble-assistant">{_vis}</div>'
                                                )
                                # Final paint + follow-up action from the metadata sentinel.
                                _text, _, _metajson = raw.partition("\x1e")
                                _final = _clean_chat(_text) or "(no response)"
                                if assistant_bubble is None:
                                    try:
                                        typing.delete()
                                    except Exception:
                                        pass
                                    await ui.run_javascript(
                                        "if(window._knitraChatProgressInt){"
                                        "clearInterval(window._knitraChatProgressInt);"
                                        "window._knitraChatProgressInt=null;}"
                                    )
                                    with chat_messages:
                                        assistant_bubble = ui.html(
                                            '<div class="chat-bubble-assistant"></div>'
                                        )
                                assistant_bubble.set_content(
                                    f'<div class="chat-bubble-assistant">{_final}</div>'
                                )
                                try:
                                    meta = json.loads(_metajson) if _metajson.strip() else {}
                                except Exception:
                                    meta = {}
                                with chat_messages:
                                    _render_followup_action(meta)

                                # Auto-reanalysis: backend confirmed all clarification
                                # questions are answered — show acknowledgement, pause
                                # so the user can read it, then fire.
                                if meta.get("auto_reanalyse"):
                                    with chat_messages:
                                        ui.html(
                                            '<div class="chat-bubble-assistant" '
                                            'style="background:#f0fdf4;border:1px solid #86efac;'
                                            'font-style:italic;color:#166534;">'
                                            'Thanks &#8212; I now have enough information to '
                                            'continue. Re-analysing your case using your '
                                            'answers&#8230;'
                                            '</div>'
                                        )
                                    await asyncio.sleep(2.0)
                                    await _reanalyse_from_chat(
                                        text_context=meta.get("auto_reanalysis_context", "")
                                    )
                            except Exception:
                                try:
                                    typing.delete()
                                except Exception:
                                    pass
                                await ui.run_javascript(
                                    "if(window._knitraChatProgressInt){"
                                    "clearInterval(window._knitraChatProgressInt);"
                                    "window._knitraChatProgressInt=null;}"
                                )
                                with chat_messages:
                                    ui.html(
                                        '<div class="chat-bubble-assistant" '
                                        'style="color:#dc2626;">'
                                        'Cannot reach the backend. Your message is back in the box, '
                                        'press Send to retry.</div>'
                                    )
                                chat_input.set_value(content)

                        # Chat mic + photo upload JS (scoped to chat, not the new-case form).
                        #
                        # MUST be ui.run_javascript, NOT ui.add_body_html. A <script> added via
                        # ui.add_body_html() is injected into the DOM but the browser never
                        # executes it, so knitraChatToggleVoice / knitraChatUpload stayed
                        # undefined and the mic + paperclip silently did nothing. run_javascript
                        # executes on the client (the same proven mechanism that sets
                        # _knitraCaseId / _knitraToken above); the guard below keeps setup once.
                        ui.run_javascript("""
(function() {
  // Guard: only inject once even if chat section is rebuilt
  if (window._knitraChatJsReady) return;
  window._knitraChatJsReady = true;

  window._knitraChatRecorder    = null;
  window._knitraChatAudio       = [];
  window._knitraChatTimerInt    = null;   // interval ID for the recording clock
  window._knitraTranscribedText = '';     // last mic transcription, read by Python _send_chat
  window._knitraPendingImageIds = [];     // upload IDs queued for next chat send
  // Canonical attachment state. Survives chat re-renders so queued photos never
  // silently vanish. Each item: {id, name, dataUrl, status}  status ∈ uploading|ready|failed
  window._knitraPendingAttachments = [];
  // Sticky status so a chat re-render can restore the last message/state.
  window._knitraChatStatusState = null;   // {msg, kind}  kind ∈ recording|busy|ok|warn|error

  // ── Status pill ───────────────────────────────────────────────────────────
  // One prominent, colour-coded pill instead of a faint grey text node, so the
  // user cannot miss recording / transcribing / success / failure feedback.
  const _STATUS_STYLE = {
    recording: 'background:#fee2e2;color:#b91c1c;',
    busy:      'background:#e0f2fe;color:#0369a1;',
    ok:        'background:#dcfce7;color:#15803d;',
    warn:      'background:#fef9c3;color:#854d0e;',
    error:     'background:#fee2e2;color:#b91c1c;',
  };
  function _chatStatus(msg, kind, sticky) {
    window._knitraChatStatusState = msg ? { msg: msg, kind: kind || 'busy' } : null;
    _chatPaintStatus();
    if (!sticky && msg && (kind === 'ok' || kind === 'warn')) {
      setTimeout(function() {
        // Only clear if this exact message is still showing
        const st = window._knitraChatStatusState;
        if (st && st.msg === msg) { _chatStatus('', null); }
      }, 4500);
    }
  }
  function _chatPaintStatus() {
    const el = document.getElementById('knitra-chat-status');
    if (!el) return;
    const st = window._knitraChatStatusState;
    if (!st || !st.msg) { el.style.display = 'none'; el.textContent = ''; return; }
    el.textContent = st.msg;
    el.style.cssText = 'display:inline-flex;margin-left:4px;' +
      (_STATUS_STYLE[st.kind] || _STATUS_STYLE.busy);
    el.className = 'knitra-status-pill';
  }

  // ── Recording timer ───────────────────────────────────────────────────────
  function _chatTimerStart() {
    let secs = 0;
    window._knitraChatTimerInt = setInterval(function() {
      secs++;
      const m = Math.floor(secs / 60).toString().padStart(2, '0');
      const s = (secs % 60).toString().padStart(2, '0');
      _chatStatus('\\u25CF Recording ' + m + ':' + s + ', tap mic to stop', 'recording', true);
    }, 1000);
  }
  function _chatTimerStop() {
    if (window._knitraChatTimerInt) {
      clearInterval(window._knitraChatTimerInt);
      window._knitraChatTimerInt = null;
    }
  }

  // ── Attachment chips (persistent, repainted after every chat render) ───────
  window._chatRenderAttachments = function() {
    const wrap = document.getElementById('knitra-chat-attachments');
    if (!wrap) return;
    const atts = window._knitraPendingAttachments || [];
    if (!atts.length) { wrap.style.display = 'none'; wrap.innerHTML = ''; return; }
    wrap.style.display = 'flex';
    const n = atts.length;
    let html = '<div class="knitra-attach-head">' +
      '\\uD83D\\uDCCE ' + n + ' ' + (n === 1 ? 'photo' : 'photos') +
      ' will be sent with your next message</div><div class="knitra-attach-strip">';
    atts.forEach(function(a, i) {
      const stateTxt = a.status === 'uploading' ? '\\u23F3 Uploading…'
        : a.status === 'failed' ? '\\u26A0\\uFE0F Failed'
        : '\\u2714 Ready to send';
      const stateColor = a.status === 'uploading' ? '#0369a1'
        : a.status === 'failed' ? '#b91c1c' : '#15803d';
      html += '<div class="knitra-attach-chip">' +
        (a.dataUrl ? '<img src="' + a.dataUrl + '" alt="">' : '') +
        '<div class="knitra-attach-meta">' +
          '<span class="knitra-attach-name" title="' + (a.name || 'photo') + '">' +
            (a.name || 'photo') + '</span>' +
          '<span class="knitra-attach-state" style="color:' + stateColor + ';">' +
            stateTxt + '</span>' +
        '</div>' +
        '<button class="knitra-attach-x" title="Remove" ' +
          'onclick="window._chatRemoveAttachment(' + i + ')">\\u00D7</button>' +
      '</div>';
    });
    html += '</div>';
    wrap.innerHTML = html;
  };

  window._chatRemoveAttachment = function(idx) {
    const atts = window._knitraPendingAttachments || [];
    const a = atts[idx];
    if (!a) return;
    atts.splice(idx, 1);
    // Keep the id list in sync so the removed photo is not sent
    if (a.id != null) {
      window._knitraPendingImageIds = (window._knitraPendingImageIds || [])
        .filter(function(x) { return x !== a.id; });
    }
    window._chatRenderAttachments();
    if (!atts.length) { _chatStatus('', null); }
  };

  // Clear everything queued (called by Python after a successful send)
  window._chatClearAttachments = function() {
    window._knitraPendingAttachments = [];
    window._knitraPendingImageIds = [];
    window._chatRenderAttachments();
  };

  // Repaint persistent UI after a chat re-render (called from Python).
  window._knitraRepaintChat = function() {
    _chatPaintStatus();
    window._chatRenderAttachments();
    // Restore the live recording highlight on the (newly recreated) mic button
    const btn = document.getElementById('knitra-chat-mic');
    if (btn && window._knitraChatRecorder &&
        window._knitraChatRecorder.state === 'recording') {
      btn.classList.add('recording');
    }
  };

  // ── Voice input ────────────────────────────────────────────────────────────
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
      _chatStatus('\\uD83C\\uDFA4 Mic blocked, allow access in browser settings.', 'error', true);
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
      const mb = document.getElementById('knitra-chat-mic');
      if (mb) { mb.classList.remove('recording'); }
      const blob = new Blob(window._knitraChatAudio, { type: mime || 'audio/webm' });
      if (blob.size < 1000) {
        _chatStatus('\\uD83C\\uDFA4 Nothing recorded, try again.', 'warn');
        return;
      }
      _chatStatus('\\u23F3 Transcribing your voice…', 'busy', true);
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
        if (text && text.trim()) {
          const trimmed = text.trim();
          // 1. Store in window variable so Python _send_chat() can read it as fallback
          window._knitraTranscribedText = (window._knitraTranscribedText
            ? window._knitraTranscribedText.trimEnd() + ' ' : '') + trimmed;
          // 2. Also inject into the DOM textarea so user can see / edit it
          const ta = document.querySelector('.knitra-chat-textarea textarea');
          if (ta) {
            ta.value = (ta.value ? ta.value.trimEnd() + ' ' : '') + trimmed;
            // Dispatch both 'input' and 'change', Quasar listens to 'input'
            ta.dispatchEvent(new Event('input', { bubbles: true }));
            ta.dispatchEvent(new Event('change', { bubbles: true }));
            ta.focus();
            // Brief green flash so the user SEES the words land in the box
            const field = ta.closest('.knitra-chat-textarea');
            if (field) {
              field.classList.remove('knitra-text-flash');
              void field.offsetWidth;              // restart the animation
              field.classList.add('knitra-text-flash');
            }
          }
          _chatStatus('\\u2714 Added to your message, review and press Send', 'ok');
        } else {
          _chatStatus('\\uD83C\\uDFA4 Nothing heard, try speaking again.', 'warn');
        }
      } catch(err) {
        _chatStatus('\\u26A0\\uFE0F Transcription failed: ' + err.message, 'error', true);
      }
    };

    rec.start(200);
    if (btn) { btn.classList.add('recording'); btn.title = 'Recording, tap to stop'; }
    _chatStatus('\\u25CF Recording 00:00, tap mic to stop', 'recording', true);
    _chatTimerStart();
  };

  // ── Photo attachment ────────────────────────────────────────────────────────
  window.knitraChatUpload = async function(input) {
    const files = Array.from(input.files);
    if (!files.length) return;
    const caseId = window._knitraCaseId;
    const token  = window._knitraToken;
    if (!caseId || !token) {
      _chatStatus('\\u26A0\\uFE0F Upload error: not signed in.', 'error', true);
      return;
    }
    if (!Array.isArray(window._knitraPendingImageIds)) window._knitraPendingImageIds = [];
    if (!Array.isArray(window._knitraPendingAttachments)) window._knitraPendingAttachments = [];

    _chatStatus('\\u23F3 Uploading ' + files.length +
      ' file' + (files.length > 1 ? 's' : '') + '…', 'busy', true);

    // Read each file to a data URL (persists across re-render) and show an
    // "uploading" chip immediately, before the network round-trip completes.
    const pending = [];
    for (const file of files) {
      const dataUrl = await new Promise(function(res) {
        const fr = new FileReader();
        fr.onload = function() { res(fr.result); };
        fr.onerror = function() { res(''); };
        fr.readAsDataURL(file);
      });
      const att = { id: null, name: file.name, dataUrl: dataUrl, status: 'uploading' };
      window._knitraPendingAttachments.push(att);
      pending.push({ file: file, att: att });
    }
    window._chatRenderAttachments();

    let ok = 0, failMsg = '';
    for (const p of pending) {
      const fd = new FormData();
      fd.append('file', p.file);
      try {
        const resp = await fetch('http://127.0.0.1:8000/api/v1/cases/' + caseId + '/uploads', {
          method: 'POST',
          headers: { 'Authorization': 'Bearer ' + token },
          body: fd,
        });
        if (resp.ok) {
          const data = await resp.json();
          if (data && data.id) {
            window._knitraPendingImageIds.push(data.id);
            p.att.id = data.id;
          }
          p.att.status = 'ready';
          ok++;
        } else if (resp.status === 413) {
          p.att.status = 'failed'; failMsg = 'File too large (max 10 MB).';
        } else if (resp.status === 415) {
          p.att.status = 'failed'; failMsg = 'Unsupported file type.';
        } else {
          p.att.status = 'failed'; failMsg = 'Server error ' + resp.status + '.';
        }
      } catch(e) {
        p.att.status = 'failed'; failMsg = 'Network error, check connection.';
      }
      window._chatRenderAttachments();
    }
    input.value = '';  // reset so same file can be re-selected
    if (ok > 0) {
      const noun = ok > 1 ? 'photos' : 'photo';
      _chatStatus('\\u2714 ' + ok + ' ' + noun +
        ' attached, press Send to ask Knitra', 'ok', true);
    } else {
      _chatStatus('\\u26A0\\uFE0F ' + (failMsg || 'Upload failed, check connection.'), 'error', true);
    }
  };
})();
""")

                        async def _quick_send(text: str):
                            chat_input.set_value(text)
                            await _send_chat()

                        async def _resume_after_refresh(text: str, image_ids: list):
                            """Re-issue the last unanswered user message after a page reload.

                            The user message + photo are already persisted server-side. Rather
                            than re-uploading or re-persisting them, we push the same image_ids
                            into the chat send pipeline so the assistant gets called with the
                            identical inputs. Note: this creates a second user-message row with
                            the same image_ids, the model sees both copies of the question. That
                            is acceptable for a clear retry, and the original message remains
                            visible above so the user is not confused about what they asked.
                            """
                            try:
                                ids_json = json.dumps([int(x) for x in (image_ids or [])
                                                       if str(x).isdigit()])
                            except Exception:
                                ids_json = "[]"
                            await ui.run_javascript(
                                f"window._knitraPendingImageIds = {ids_json};"
                            )
                            chat_input.set_value(text or "")
                            await _send_chat()

                        await _load_chat()

                        # Repaint persistent attachment chips + status pill so that
                        # anything queued before this re-render reappears (a chat
                        # refresh must never make attachments/feedback silently vanish).
                        ui.run_javascript(
                            "if(window._knitraRepaintChat)window._knitraRepaintChat();"
                        )

            # Initial render of chat section based on current analysis state
            await _render_chat_section(_initial_status)

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
                            '- so you can get expert help when you need it.</p>'
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
# NEW CASE  (/new) , composer-style quick-start flow
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
    window._fixlyFiles      = [];   // compressed image objects {name,type,data}
    window._fixlyVideoFile  = null; // raw File object for a video upload (or null)
    window._fixlyVideoObjUrl = null; // object URL for video preview (revoked on remove)
    window._fixlyToken      = '';   // bearer token, set by deferred timer
    window._fixlyRecorder   = null; // MediaRecorder instance while recording
    window._fixlyAudio      = [];   // audio chunks collected during recording

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
      // Image-only path. Selecting images clears any existing video selection.
      if (window._fixlyVideoFile) {
        if (window._fixlyVideoObjUrl) { URL.revokeObjectURL(window._fixlyVideoObjUrl); }
        window._fixlyVideoFile = null;
        window._fixlyVideoObjUrl = null;
      }
      const toAdd = Array.from(fileList)
        .filter(function(f) { return f.type.startsWith('image/'); })
        .slice(0, 10 - window._fixlyFiles.length);
      if (!toAdd.length) return;
      const results = await Promise.all(toAdd.map(_fixlyCompress));
      window._fixlyFiles = window._fixlyFiles.concat(results);
      _fixlyRender();
    }

    // Called by the video file input (fixly-video-input). Selecting a video
    // clears any existing image selection and shows the blue preview card.
    function fixlyHandleVideoFile(fileList) {
      const vf = Array.from(fileList).find(function(f) { return f.type.startsWith('video/'); });
      if (!vf) return;
      if (vf.size > 100 * 1024 * 1024) {
        alert('Video is too large. Maximum size is 100 MB.');
        return;
      }
      if (window._fixlyVideoObjUrl) { URL.revokeObjectURL(window._fixlyVideoObjUrl); }
      window._fixlyFiles = [];   // clear any images
      window._fixlyVideoFile = vf;
      window._fixlyVideoObjUrl = URL.createObjectURL(vf);
      _fixlyRender();
    }

    // Upload a video file directly from the browser to the backend.
    // Called from Python via ui.run_javascript() after the case is created.
    // Returns the upload response JSON on success, or null on failure.
    async function fixlyUploadVideo(token, caseId) {
      if (!window._fixlyVideoFile) return null;
      const fd = new FormData();
      fd.append('file', window._fixlyVideoFile);
      try {
        const resp = await fetch(
          'http://127.0.0.1:8000/api/v1/cases/' + caseId + '/uploads',
          { method: 'POST', headers: { 'Authorization': 'Bearer ' + token }, body: fd }
        );
        if (resp.ok) return await resp.json();
        return { error: resp.status };
      } catch(e) {
        return null;
      }
    }

    // ── Photo/video area renderer, single container, completely replaced each call ─
    // This avoids any show/hide race between separate DOM elements.
    function _fixlyRender() {
      const area  = document.getElementById('fixly-media-area');
      const badge = document.getElementById('fixly-attach-badge');
      if (!area) return;

      // ── Video attached: show preview instead of image grid ────────────────
      if (window._fixlyVideoFile && window._fixlyVideoObjUrl) {
        const vf = window._fixlyVideoFile;
        const sizeMB = (vf.size / 1024 / 1024).toFixed(1);
        if (badge) { badge.style.display = 'flex'; badge.textContent = '1'; }
        area.innerHTML = `
          <div style="background:#eff6ff;border-bottom:1px solid #bfdbfe;padding:14px 20px;">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;flex-wrap:wrap;">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
                   viewBox="0 0 24 24" fill="none" stroke="#1d4ed8"
                   stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="23 7 16 12 23 17 23 7"/>
                <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
              </svg>
              <span style="font-weight:700;color:#1e40af;font-size:0.88rem;">1 video attached</span>
              <span style="flex:1;"></span>
              <button onclick="if(window._fixlyVideoObjUrl){URL.revokeObjectURL(window._fixlyVideoObjUrl);}window._fixlyVideoFile=null;window._fixlyVideoObjUrl=null;_fixlyRender();"
                      style="font-size:0.78rem;color:#dc2626;font-weight:600;
                             cursor:pointer;padding:3px 10px;border-radius:6px;
                             border:1px solid #fecaca;background:#fef2f2;
                             -webkit-appearance:none;">
                Remove
              </button>
            </div>
            <video src="${window._fixlyVideoObjUrl}" controls
                   style="width:100%;max-width:360px;border-radius:10px;
                          display:block;background:#000;max-height:220px;">
            </video>
            <p style="font-size:0.78rem;color:#64748b;margin:6px 0 0;">
              ${vf.name} &middot; ${sizeMB} MB &middot;
              <em>Video analysis — allow extra time</em>
            </p>
          </div>`;
        return;
      }

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
            ? 'Microphone blocked, allow access in your browser and try again.'
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
          _fixlyVoiceMsg('Nothing recorded, tap mic and speak, then tap again to stop.', 'info');
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
            _fixlyVoiceMsg('Nothing heard, try speaking more clearly or type instead.', 'info');
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
        btn.title = 'Voice input, tap and speak';
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

    # ── Camera capture (Stage 1B: in-app webcam via getUserMedia) ──────────────
    # Isolated head block. Reuses fixlyHandleFiles/_fixlyFiles from the block above
    # so a captured frame flows through the EXACT same compress/preview/submit path
    # as an uploaded image. No backend, model, or schema involvement.
    ui.add_head_html("""<script>
    window._fixlyCamStream = null;
    window._fixlyCamFacing = 'environment';   // rear camera by default

    function _fixlyCamMsg(msg, isError) {
      const el = document.getElementById('fixly-cam-msg');
      if (!el) return;
      el.textContent = msg || '';
      el.style.color = isError ? '#fca5a5' : '#e2e8f0';
      el.style.display = msg ? 'block' : 'none';
    }

    function _fixlyCamStop() {
      if (window._fixlyCamStream) {
        window._fixlyCamStream.getTracks().forEach(function(t){ t.stop(); });
        window._fixlyCamStream = null;
      }
      const v = document.getElementById('fixly-cam-video');
      if (v) v.srcObject = null;
    }

    function fixlyCloseCamera() {
      _fixlyCamStop();
      const ov = document.getElementById('fixly-cam-overlay');
      if (ov) ov.style.display = 'none';
    }

    async function _fixlyCamStart() {
      _fixlyCamStop();
      const v  = document.getElementById('fixly-cam-video');
      const fb = document.getElementById('fixly-cam-fallback');
      if (fb) fb.style.display = 'none';
      try {
        window._fixlyCamStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: window._fixlyCamFacing }, audio: false
        });
        if (v) v.srcObject = window._fixlyCamStream;
        _fixlyCamMsg('', false);
      } catch (err) {
        const name = (err && err.name) ? err.name : '';
        _fixlyCamMsg(
          name === 'NotAllowedError'
            ? 'Camera access was blocked. Allow it in your browser, or use your device camera / add photos from your gallery instead.'
            : 'No in-app camera is available here. Use your device camera, or add photos from your gallery instead.',
          true
        );
        if (fb) fb.style.display = 'inline-flex';  // offer the native-camera fallback
      }
    }

    async function fixlyOpenCamera() {
      // No getUserMedia at all (old browser / insecure context) -> native input.
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        const inp = document.getElementById('fixly-camera-input');
        if (inp) inp.click();
        return;
      }
      const ov = document.getElementById('fixly-cam-overlay');
      if (ov) ov.style.display = 'flex';
      await _fixlyCamStart();
    }

    async function fixlySwitchCamera() {
      window._fixlyCamFacing = (window._fixlyCamFacing === 'environment') ? 'user' : 'environment';
      await _fixlyCamStart();
    }

    function fixlyCapture() {
      const v = document.getElementById('fixly-cam-video');
      if (!v || !v.videoWidth) { _fixlyCamMsg('Camera is still starting, try again in a moment.', true); return; }
      const c = document.getElementById('fixly-cam-canvas');
      c.width = v.videoWidth; c.height = v.videoHeight;
      c.getContext('2d').drawImage(v, 0, 0, c.width, c.height);
      c.toBlob(function(blob) {
        if (!blob) { _fixlyCamMsg('Could not capture the frame, try again.', true); return; }
        // Reuse the existing path exactly: File -> fixlyHandleFiles -> _fixlyCompress
        // (1024px / JPEG 82%) -> window._fixlyFiles -> preview grid -> submit/upload.
        const file = new File([blob], 'camera-' + Date.now() + '.jpg', { type: 'image/jpeg' });
        fixlyHandleFiles([file]);
        fixlyCloseCamera();
      }, 'image/jpeg', 0.92);
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
                'Snap a photo of the problem, Knitra identifies it and '
                'writes your repair guide automatically.</p>'
            )

            # ── AI disclaimer ─────────────────────────────────────────────────
            ui.html(
                '<div style="display:flex;align-items:flex-start;gap:10px;'
                'background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;'
                'padding:12px 14px;">'
                '<span style="font-size:1rem;flex-shrink:0;margin-top:1px;">&#8505;&#65039;</span>'
                '<p style="margin:0;font-size:0.78rem;color:#475569;line-height:1.6;">'
                '<strong style="color:#334155;">AI-assisted advice &mdash; always use your judgement.</strong> '
                'Knitra uses a local AI model and may make mistakes. It is not a substitute for a '
                'qualified professional. Never attempt repairs involving electrical mains, gas, or '
                'structural load without professional guidance. '
                'You are responsible for any actions you choose to take.'
                '</p></div>'
            )

            # ── Single composer card ───────────────────────────────────────────
            with ui.element("div").style(
                "background:white;border:1px solid #e2e8f0;"
                "border-radius:16px;overflow:hidden;"
                "box-shadow:0 2px 12px rgba(0,0,0,0.04);"
            ):
                # Hidden file inputs — triggered by their corresponding label/button elements.
                # fixly-file-input: multi-image from gallery (drop zone + paperclip icon).
                # fixly-video-input: single video from the 🎥 icon button.
                ui.html(
                    '<input type="file" id="fixly-file-input" multiple accept="image/*"'
                    ' style="position:absolute;width:1px;height:1px;opacity:0;pointer-events:none;"'
                    ' onchange="fixlyHandleFiles(this.files);this.value=\'\'"> '
                    '<input type="file" id="fixly-video-input" accept="video/*"'
                    ' style="position:absolute;width:1px;height:1px;opacity:0;pointer-events:none;"'
                    ' onchange="fixlyHandleVideoFile(this.files);this.value=\'\'">'
                )

                # Stage 1A: native camera input (separate from the gallery input above,
                # so the gallery keeps multi-select). On mobile, capture="environment"
                # opens the rear camera directly. Feeds the same fixlyHandleFiles path.
                ui.html(
                    '<input type="file" id="fixly-camera-input" accept="image/*" capture="environment"'
                    ' style="position:absolute;width:1px;height:1px;opacity:0;pointer-events:none;"'
                    ' onchange="fixlyHandleFiles(this.files);this.value=\'\';fixlyCloseCamera()">'
                )

                # Stage 1B: in-app webcam overlay (hidden until "Take Photo" is pressed).
                ui.html("""
                <div id="fixly-cam-overlay" style="display:none;position:fixed;inset:0;z-index:9999;
                     background:rgba(15,23,42,0.92);align-items:center;justify-content:center;
                     flex-direction:column;gap:14px;padding:20px;">
                  <video id="fixly-cam-video" autoplay playsinline muted
                         style="max-width:100%;max-height:68vh;border-radius:12px;background:#000;"></video>
                  <div id="fixly-cam-msg" style="display:none;color:#e2e8f0;font-size:0.85rem;
                       max-width:440px;text-align:center;line-height:1.55;"></div>
                  <div style="display:flex;gap:10px;flex-wrap:wrap;justify-content:center;align-items:center;">
                    <button onclick="fixlyCapture()"
                            style="background:#6366f1;color:#fff;border:none;border-radius:10px;
                                   padding:11px 22px;font-weight:700;font-size:0.9rem;cursor:pointer;">
                      Capture
                    </button>
                    <button onclick="fixlySwitchCamera()"
                            style="background:transparent;color:#e2e8f0;border:1px solid #475569;
                                   border-radius:10px;padding:11px 16px;font-weight:600;font-size:0.85rem;cursor:pointer;">
                      Switch camera
                    </button>
                    <button onclick="fixlyCloseCamera()"
                            style="background:transparent;color:#e2e8f0;border:1px solid #475569;
                                   border-radius:10px;padding:11px 16px;font-weight:600;font-size:0.85rem;cursor:pointer;">
                      Cancel
                    </button>
                    <label for="fixly-camera-input" id="fixly-cam-fallback"
                           style="display:none;background:#0f172a;color:#e2e8f0;border:1px solid #6366f1;
                                  border-radius:10px;padding:11px 16px;font-weight:600;font-size:0.85rem;cursor:pointer;">
                      Use device camera
                    </label>
                  </div>
                  <canvas id="fixly-cam-canvas" style="display:none;"></canvas>
                </div>
                """)

                # Single media area, _fixlyRender() rewrites its innerHTML entirely.
                # Either shows the drop zone (empty) or the green attachment panel (photos selected).
                ui.html('<div id="fixly-media-area"></div>')

                # ── Divider ────────────────────────────────────────────────────
                ui.html(
                    '<div style="height:1px;background:#f1f5f9;margin:16px 0 0;"></div>'
                )

                # ── Note textarea, plain HTML so voice can write into it ──────
                ui.html("""
                <div style="padding:8px 16px 4px;">
                  <textarea
                    id="fixly-note"
                    rows="2"
                    placeholder="Describe the issue (optional), e.g. dripping pipe under the kitchen sink…"
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

                # ── Bottom action bar, all pure HTML for reliable gesture handling
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
                          title="Voice input, tap and speak">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18"
                         viewBox="0 0 24 24" fill="none" stroke="currentColor"
                         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                      <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
                      <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                      <line x1="12" x2="12" y1="19" y2="22"/>
                    </svg>
                  </button>

                  <!-- Take Photo: in-app webcam (laptop) / device camera (mobile) -->
                  <button id="fixly-cam-btn" class="fixly-icon-btn"
                          onclick="fixlyOpenCamera()"
                          title="Take a photo with your camera">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18"
                         viewBox="0 0 24 24" fill="none" stroke="currentColor"
                         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                      <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/>
                      <circle cx="12" cy="13" r="3"/>
                    </svg>
                  </button>

                  <!-- Video upload button — triggers the hidden fixly-video-input -->
                  <label for="fixly-video-input" class="fixly-icon-btn" title="Add a video">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18"
                         viewBox="0 0 24 24" fill="none" stroke="currentColor"
                         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                      <path d="m22 8-6 4 6 4V8Z"/>
                      <rect width="14" height="12" x="2" y="6" rx="2" ry="2"/>
                    </svg>
                  </label>

                  <div style="flex:1;"></div>
                </div>
                """)

                # Submit button, stays NiceGUI (no file/mic picker needed here)
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

    # ── Handler, defined after UI so all element refs are in scope ───────────
    async def _start_diagnosis():
        import base64 as _b64

        err_label.set_text("")

        # Pull compressed image files and check for a video from JS globals
        raw_json = await ui.run_javascript("return fixlyGetFiles()")
        try:
            files = json.loads(raw_json or "[]")
        except Exception:
            files = []

        has_video = await ui.run_javascript("return !!window._fixlyVideoFile")

        if not files and not has_video:
            err_label.set_text(
                "Please add at least one photo or video before analysing."
            )
            return

        submit_btn.props("loading disable")

        # Build inline progress
        prog_area.clear()
        n = len(files)
        if has_video:
            upload_label = "Uploading video"
        else:
            upload_label = f"Uploading {n} photo{'s' if n > 1 else ''}"
        _texts = {
            "case":    "Creating your case",
            "upload":  upload_label,
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

        # Step 1, create case
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

        # Step 2, upload files
        _active("upload")
        uploaded = 0

        if has_video:
            # ── Video path: JS uploads directly via fetch (avoids base64/WS transfer) ──
            _lbls["upload"].set_text("↻  Uploading video…")
            try:
                video_result = await ui.run_javascript(
                    f"return await fixlyUploadVideo('{token}', {cid})"
                )
                if video_result and not video_result.get("error"):
                    uploaded = 1
            except Exception:
                pass
            if uploaded == 0:
                err_label.set_text("Could not upload the video. Check your connection and try again.")
                submit_btn.props(remove="loading disable")
                return
        else:
            # ── Image path (original, unchanged) ──────────────────────────────
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
                err_label.set_text("Could not upload any photos, check your connection and try again.")
                submit_btn.props(remove="loading disable")
                return

        _done("upload")

        # Step 3, trigger analysis
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
        Fires 400ms after page build, by then the WebSocket is open.
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
# ADMIN — Feedback dashboard (Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

@ui.page("/admin/feedback")
async def admin_feedback_page():
    token = app.storage.user.get("token")
    if not token:
        ui.navigate.to("/login")
        return
    # UX gate: non-admins are silently sent back to their cases. (Access is also
    # enforced server-side — the admin API returns 403, handled below.)
    if not app.storage.user.get("is_admin", False):
        ui.navigate.to("/cases")
        return

    H = {"Authorization": f"Bearer {token}"}

    async def _get(path):
        try:
            async with httpx.AsyncClient() as c:
                r = await c.get(f"{API}/admin/feedback/{path}", headers=H, timeout=10.0)
            if r.status_code == 403:
                return "forbidden"
            if r.status_code == 200:
                return r.json()
        except Exception:
            return None
        return None

    overview = await _get("overview")
    if overview == "forbidden":
        ui.navigate.to("/cases")
        return

    inject_styles()   # global CSS (.sidebar / .main / .nav-item) — same as every page
    priority = await _get("priority") or []
    modes    = await _get("modes") or []
    safety   = await _get("safety") or []
    comments = await _get("comments?limit=8") or []
    overview = overview or {}

    _ISSUE_LABELS = {
        "wrong_trade": "Wrong trade/category", "wrong_severity": "Wrong severity/risk",
        "wrong_diy_recommendation": "Wrong DIY recommendation",
        "wrong_professional_recommendation": "Wrong professional recommendation",
        "clarification_not_useful": "Clarification not useful",
        "scope_incorrect": "Scope decision incorrect", "steps_unclear": "Steps unclear",
        "care_not_useful": "Care advice not useful", "other": "Other",
    }
    _CHIP_LABELS = {
        "wrong_diagnosis":                    "Wrong diagnosis",
        "too_cautious":                       "Too cautious",
        "not_cautious_enough":                "Not cautious enough",
        "not_detailed_enough":                "Not detailed enough",
        "wrong_tools_materials":              "Wrong tools/materials",
        "steps_unclear":                      "Steps unclear",
        "wrong_professional_recommendation":  "Wrong professional recommendation",
        "advice_didnt_match_upload":          "Advice didn't match upload",
        "repeated_question":                  "Repeated a question I answered",
        "advice_felt_unsafe":                 "Advice felt unsafe",
        "other":                              "Other",
    }

    def _chip_badges_html(chips: list) -> str:
        if not chips:
            return ""
        badges = "".join(
            f'<span style="display:inline-block;background:#e0e7ff;color:#3730a3;'
            f'border-radius:100px;font-size:0.7rem;padding:2px 9px;font-weight:600;'
            f'margin:2px 3px 2px 0;">'
            f'{_CHIP_LABELS.get(c, c.replace("_", " "))}</span>'
            for c in chips
        )
        return (
            f'<div style="margin-top:5px;">'
            f'<span style="font-size:0.72rem;color:#64748b;">Reason chips: </span>'
            f'{badges}</div>'
        )

    mobile_bar()
    sidebar("/admin/feedback")

    with ui.element("div").classes("main"):
        with ui.element("div").style(
            "padding:32px;max-width:920px;display:flex;flex-direction:column;gap:18px;"
        ):
            tf = overview.get("total_feedback", 0)
            maturity = ("Established", "#15803d") if tf >= 50 else \
                       ("Emerging", "#92400e") if tf >= 20 else ("Early data", "#b45309")

            # Header + data-maturity pill
            ui.html(
                f'<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">'
                f'<h1 style="font-size:1.5rem;font-weight:800;color:#0f172a;margin:0;">Feedback</h1>'
                f'<span style="background:#f1f5f9;color:{maturity[1]};font-size:0.72rem;'
                f'font-weight:700;padding:3px 10px;border-radius:100px;">{maturity[0]} · '
                f'{tf} response{"s" if tf != 1 else ""}</span></div>'
            )

            # Safety bar (preempts everything)
            s_open = overview.get("safety_open", 0)
            if s_open > 0:
                ui.html(
                    f'<div style="background:#fef2f2;border:1px solid #fca5a5;border-radius:12px;'
                    f'padding:14px 18px;display:flex;align-items:center;gap:10px;">'
                    f'<span style="font-size:1.2rem;">&#128721;</span>'
                    f'<span style="font-weight:700;color:#991b1b;font-size:0.9rem;">'
                    f'{s_open} safety review{"s" if s_open != 1 else ""} pending</span></div>'
                )
            else:
                ui.html(
                    '<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:12px;'
                    'padding:12px 18px;color:#15803d;font-weight:600;font-size:0.86rem;">'
                    '&#10003; No safety reviews pending</div>'
                )

            # Headline takeaway sentence (counts-first)
            helpful = overview.get("helpful", 0)
            top_issue = _ISSUE_LABELS.get(priority[0]["issue_category"], priority[0]["issue_category"]) if priority else None
            _take = (
                f"Based on {tf} response{'s' if tf != 1 else ''}: "
                f"helpful {helpful}/{tf}"
                + (f"; top issue: {top_issue} ({priority[0]['count']})" if top_issue else "")
                + f"; {overview.get('safety_total', 0)} safety review(s)."
            )
            _context_note = (
                "The analysis count reflects iterative testing across output modes and routing scenarios "
                "during development. The feedback system was introduced recently, so only a subset of "
                "those analyses have recorded feedback."
            )
            ui.html(
                f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;'
                f'padding:12px 16px;font-size:0.86rem;color:#334155;line-height:1.5;">'
                f'{_take}'
                f'<div style="margin-top:6px;font-size:0.78rem;color:#94a3b8;font-style:italic;">'
                f'{_context_note}</div></div>'
            )

            # Health cards — count-first, N-tagged
            def _card(label, value, sub=""):
                return (
                    f'<div style="background:#f8fafc;border-radius:10px;padding:14px 16px;flex:1;min-width:140px;">'
                    f'<div style="font-size:0.72rem;color:#94a3b8;font-weight:600;text-transform:uppercase;'
                    f'letter-spacing:0.04em;">{label}</div>'
                    f'<div style="font-size:1.5rem;font-weight:800;color:#0f172a;margin-top:2px;">{value}</div>'
                    f'<div style="font-size:0.72rem;color:#94a3b8;">{sub}</div></div>'
                )
            rr = round(overview.get("response_rate", 0) * 100)
            ui.html(
                '<div style="display:flex;gap:12px;flex-wrap:wrap;">'
                + _card("Analyses", overview.get("total_analyses", 0))
                + _card("Feedback", tf, f"{rr}% response rate")
                + _card("Helpful", f"{helpful}/{tf}", "of responses")
                + _card("Safety reviews", overview.get("safety_total", 0), f"{s_open} pending")
                + '</div>'
            )

            # Priority Board — weighted-count ranking (no chart; honest bars)
            ui.html('<div style="font-weight:700;color:#0f172a;font-size:0.95rem;margin-top:4px;">Fix first</div>')
            if not priority:
                ui.html('<div style="color:#94a3b8;font-size:0.84rem;">No issues reported yet.</div>')
            else:
                _maxs = max(p["priority_score"] for p in priority) or 1
                rows = ""
                for p in priority:
                    label = _ISSUE_LABELS.get(p["issue_category"], p["issue_category"])
                    pct = int(100 * p["priority_score"] / _maxs)
                    rows += (
                        f'<div style="margin-bottom:8px;">'
                        f'<div style="display:flex;justify-content:space-between;font-size:0.82rem;'
                        f'color:#334155;margin-bottom:3px;"><span>{label}</span>'
                        f'<span style="color:#64748b;">{p["count"]} report(s) · score {p["priority_score"]}</span></div>'
                        f'<div style="background:#eef2ff;border-radius:6px;height:8px;">'
                        f'<div style="background:#6366f1;height:8px;border-radius:6px;width:{pct}%;"></div></div></div>'
                    )
                ui.html(f'<div style="background:white;border:1px solid #e2e8f0;border-radius:12px;padding:16px 18px;">{rows}</div>')

            # Performance by mode — scorecards (each shows own N)
            ui.html('<div style="font-weight:700;color:#0f172a;font-size:0.95rem;margin-top:4px;">By output mode</div>')
            if not modes:
                ui.html('<div style="color:#94a3b8;font-size:0.84rem;">No feedback yet.</div>')
            else:
                cards = ""
                for m in modes:
                    cards += (
                        f'<div style="background:#f8fafc;border-radius:10px;padding:12px 14px;min-width:150px;flex:1;">'
                        f'<div style="font-weight:700;color:#334155;font-size:0.82rem;">{m["output_mode"]}</div>'
                        f'<div style="font-size:0.78rem;color:#64748b;margin-top:3px;">'
                        f'{m["helpful"]} helpful · {m["partial"]} partial · {m["not"]} no '
                        f'<span style="color:#94a3b8;">(N={m["total"]})</span></div></div>'
                    )
                ui.html(f'<div style="display:flex;gap:12px;flex-wrap:wrap;">{cards}</div>')

            # Safety Review Queue — shows auto-flagged hazard cases and user-reported safety concerns
            ui.html('<div style="font-weight:700;color:#0f172a;font-size:0.95rem;margin-top:4px;">Safety Review Queue</div>')
            ui.html(
                '<div style="font-size:0.78rem;color:#64748b;margin-bottom:4px;">'
                'Includes cases automatically flagged because they involved a hazardous scenario, '
                'and cases where the user explicitly reported a safety concern.</div>'
            )
            if not safety:
                ui.html('<div style="color:#94a3b8;font-size:0.84rem;">No safety reviews.</div>')
            else:
                for inc in safety:
                    # Determine whether this was auto-flagged by the system or user-reported.
                    _has_user_report = bool(inc.get("safety_category"))
                    _mode = inc.get("output_mode") or ""
                    if _has_user_report:
                        _flag_label = "USER-REPORTED SAFETY CONCERN"
                        _flag_bg    = "#fef2f2"
                        _flag_fg    = "#991b1b"
                        _flag_border = "#fca5a5"
                        _reason = (
                            f"The user selected a safety category: "
                            f"<strong>{inc['safety_category'].replace('_', ' ')}</strong>."
                        )
                    elif _mode == "assessment_first":
                        _flag_label  = "AUTO-FLAGGED · Assessment First Case"
                        _flag_bg     = "#fffbeb"
                        _flag_fg     = "#92400e"
                        _flag_border = "#fde68a"
                        _reason = (
                            "This case was automatically added to the review queue because it involved "
                            "a hazardous scenario requiring professional assessment. "
                            "The user did not report a safety concern."
                        )
                    elif _mode == "out_of_scope":
                        _flag_label  = "AUTO-FLAGGED · Emergency Case"
                        _flag_bg     = "#fffbeb"
                        _flag_fg     = "#92400e"
                        _flag_border = "#fde68a"
                        _reason = (
                            "This case was automatically added because it was classified as an active emergency. "
                            "The user did not report a safety concern."
                        )
                    else:
                        _flag_label  = "AUTO-FLAGGED"
                        _flag_bg     = "#fffbeb"
                        _flag_fg     = "#92400e"
                        _flag_border = "#fde68a"
                        _reason = "This case was automatically added to the safety review queue."

                    # Feedback detail lines
                    _rating_label   = {"helpful": "Helpful", "partial": "Partially helpful", "not": "Not helpful"}.get(
                        inc.get("rating", ""), inc.get("rating", "—")
                    )
                    _issue_label  = _ISSUE_LABELS.get(inc.get("issue_category"), None)
                    _comment_text = inc.get("comment") or None
                    _inc_chips    = inc.get("reason_chips") or []

                    _feedback_rows = (
                        f'<div style="font-size:0.8rem;color:#334155;line-height:1.7;">'
                        f'<span style="color:#64748b;">Rating:</span> {_rating_label}&nbsp;&nbsp;'
                        f'<span style="color:#64748b;">Issue:</span> {_issue_label or "None"}&nbsp;&nbsp;'
                        f'<span style="color:#64748b;">Safety category:</span> '
                        f'{inc.get("safety_category", "None") or "None"}'
                        + (f'<br><span style="color:#64748b;">Comment:</span> {_comment_text}' if _comment_text else
                           '&nbsp;&nbsp;<span style="color:#64748b;">Comment:</span> None')
                        + _chip_badges_html(_inc_chips)
                        + '</div>'
                    )

                    with ui.element("div").style(
                        f"background:white;border:1px solid {_flag_border};border-radius:12px;"
                        "padding:14px 16px;display:flex;flex-direction:column;gap:10px;"
                    ):
                        # Flag banner
                        ui.html(
                            f'<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px;">'
                            f'<span style="background:{_flag_bg};color:{_flag_fg};font-weight:700;'
                            f'font-size:0.78rem;padding:3px 10px;border-radius:6px;letter-spacing:0.03em;">'
                            f'{_flag_label}</span>'
                            f'<span style="font-size:0.74rem;color:#64748b;">'
                            f'case {inc.get("session_id")} &middot; analysis {inc.get("analysis_id")} '
                            f'&middot; mode: {_mode or "—"} &middot; status: {inc.get("status")}'
                            + (f' ({inc.get("resolution")})' if inc.get("resolution") else "")
                            + '</span></div>'
                        )
                        # Original feedback
                        ui.html(_feedback_rows)
                        # Reason explanation
                        ui.html(
                            f'<div style="background:#f8fafc;border-radius:8px;padding:8px 12px;'
                            f'font-size:0.78rem;color:#64748b;line-height:1.5;">'
                            f'<strong style="color:#475569;">Reason:</strong> {_reason}</div>'
                        )
                        # Review controls
                        with ui.element("div").style("display:flex;gap:8px;flex-wrap:wrap;align-items:center;"):
                            st = ui.select(["new", "reviewing", "resolved"], value=inc.get("status", "new")) \
                                .props("outlined dense").style("min-width:120px;")
                            res = ui.select(
                                {"": "—", "fixed": "fixed", "accepted_limitation": "accepted limitation", "dismissed": "dismissed"},
                                value=inc.get("resolution") or "",
                            ).props("outlined dense").style("min-width:150px;")
                            note = ui.input(
                                value=inc.get("reviewer_note") or "",
                                placeholder="reviewer note",
                            ).props("outlined dense").style("flex:1;min-width:160px;")

                            async def _update(_=None, fid=inc["id"], st=st, res=res, note=note):
                                body = {"status": st.value}
                                if res.value:
                                    body["resolution"] = res.value
                                if (note.value or "").strip():
                                    body["reviewer_note"] = note.value.strip()
                                try:
                                    async with httpx.AsyncClient() as c:
                                        await c.patch(f"{API}/admin/feedback/{fid}", json=body, headers=H, timeout=10.0)
                                    ui.navigate.to("/admin/feedback")
                                except Exception:
                                    ui.notify("Could not update incident.", type="negative")

                            ui.button("Update", on_click=_update).props("unelevated color=primary size=sm")

            # User Voice — recent comments (qualitative evidence)
            ui.html('<div style="font-weight:700;color:#0f172a;font-size:0.95rem;margin-top:4px;">Recent comments</div>')
            if not comments:
                ui.html('<div style="color:#94a3b8;font-size:0.84rem;">No comments yet.</div>')
            else:
                rows = ""
                for c in comments:
                    tag = _ISSUE_LABELS.get(c.get("issue_category"), c.get("issue_category") or c.get("rating"))
                    _c_chips = c.get("reason_chips") or []
                    rows += (
                        f'<div style="border-bottom:1px solid #f1f5f9;padding:10px 0;">'
                        f'<div style="font-size:0.83rem;color:#334155;">{c.get("comment")}</div>'
                        f'<div style="font-size:0.72rem;color:#94a3b8;margin-top:2px;">'
                        f'{c.get("rating")} · {c.get("output_mode")} · {tag}</div>'
                        + _chip_badges_html(_c_chips)
                        + '</div>'
                    )
                ui.html(f'<div style="background:white;border:1px solid #e2e8f0;border-radius:12px;padding:6px 16px;">{rows}</div>')


# ─────────────────────────────────────────────────────────────────────────────
# ADMIN — Judge summary dashboard
# ─────────────────────────────────────────────────────────────────────────────

@ui.page("/admin/judge")
async def admin_judge_page():
    token = app.storage.user.get("token")
    if not token:
        ui.navigate.to("/login")
        return
    if not app.storage.user.get("is_admin", False):
        ui.navigate.to("/cases")
        return

    H = {"Authorization": f"Bearer {token}"}

    summary = None
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{API}/admin/judge/summary", headers=H, timeout=15.0)
        if r.status_code == 403:
            ui.navigate.to("/cases")
            return
        if r.status_code == 200:
            summary = r.json()
    except Exception:
        pass

    summary = summary or {}

    _VERDICT_STYLE = {
        "pass":         ("#dcfce7", "#15803d"),
        "needs_review": ("#fef3c7", "#92400e"),
        "unsafe":       ("#fee2e2", "#991b1b"),
        "unknown":      ("#f1f5f9", "#475569"),
    }
    _TAG_LABELS = {
        "over_cautious_routing":                          "Over-cautious routing",
        "under_cautious_routing":                         "Under-cautious routing",
        "missing_safety_warning":                         "Missing safety warning",
        "hallucinated_hazard":                            "Hallucinated hazard",
        "vague_repair_steps":                             "Vague repair steps",
        "vague_tools_or_materials":                       "Vague tools/materials",
        "poor_grounding":                                 "Poor grounding",
        "unnecessary_clarification":                      "Unnecessary clarification",
        "missing_clarification":                          "Missing clarification",
        "unclear_user_language":                          "Unclear language",
        "wrong_trade_category":                           "Wrong trade category",
        "weak_step_ordering":                             "Weak step ordering",
        "emergency_wording_could_be_stronger":            "Emergency wording too weak",
        "emergency_repair_steps_inappropriately_expected":"Repair steps wrongly expected",
        "unsupported_professional_escalation":            "Unsupported escalation",
        "unsupported_hazard_claim":                       "Unsupported hazard claim",
        "wrong_professional_type":                        "Wrong professional type",
        "over_specific_urgency":                          "Urgency over-specified",
        "useful_but_overcautious":                        "Useful but overcautious",
        "evidence_uncertainty_not_explained":             "Uncertainty not explained",
        "correct_but_unclear_reasoning":                  "Correct but unclear",
    }

    inject_styles()
    mobile_bar()
    sidebar("/admin/judge")

    with ui.element("div").classes("main"):
        with ui.element("div").style(
            "padding:32px;max-width:860px;display:flex;flex-direction:column;gap:18px;"
        ):
            total_judged    = summary.get("total_judged", 0)
            verdict_counts  = summary.get("verdict_counts") or {}
            top_tags        = summary.get("top_issue_tags") or []
            score_dist      = summary.get("score_distribution") or {}
            suggested_fixes = summary.get("suggested_fixes") or []
            judge_model     = summary.get("judge_model", "—")

            ui.html(
                f'<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">'
                f'<h1 style="font-size:1.5rem;font-weight:800;color:#0f172a;margin:0;">Judge Summary</h1>'
                f'<span style="background:#f1f5f9;color:#475569;font-size:0.72rem;'
                f'font-weight:700;padding:3px 10px;border-radius:100px;">'
                f'{total_judged} evaluation{"s" if total_judged != 1 else ""} · {judge_model}</span></div>'
            )

            if total_judged == 0:
                ui.html(
                    '<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;'
                    'padding:24px;color:#94a3b8;font-size:0.9rem;text-align:center;">'
                    'No judge evaluations yet. Open a completed case and click Run Judge '
                    '(requires JUDGE_ENABLED=true in backend/.env).</div>'
                )
            else:
                # Verdict counts
                with ui.element("div").style(
                    "display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;"
                ):
                    for verdict, (bg, fg) in _VERDICT_STYLE.items():
                        count = verdict_counts.get(verdict, 0)
                        if count == 0:
                            continue
                        ui.html(
                            f'<div style="background:{bg};border-radius:12px;padding:16px 18px;">'
                            f'<div style="font-size:1.8rem;font-weight:800;color:{fg};">{count}</div>'
                            f'<div style="font-size:0.78rem;font-weight:600;color:{fg};">'
                            f'{verdict.replace("_", " ").title()}</div></div>'
                        )

                # Score distribution
                if score_dist:
                    ui.html(
                        '<div style="font-weight:700;color:#0f172a;font-size:0.95rem;">Score distribution</div>'
                    )
                    dist_html = "".join(
                        f'<div style="display:flex;justify-content:space-between;padding:6px 0;'
                        f'border-bottom:1px solid #f1f5f9;font-size:0.84rem;">'
                        f'<span style="color:#334155;">{bucket}</span>'
                        f'<span style="font-weight:700;color:#0f172a;">{cnt}</span></div>'
                        for bucket, cnt in score_dist.items()
                    )
                    ui.html(
                        f'<div style="background:white;border:1px solid #e2e8f0;border-radius:12px;'
                        f'padding:4px 16px;">{dist_html}</div>'
                    )

                # Top issue tags
                if top_tags:
                    ui.html(
                        '<div style="font-weight:700;color:#0f172a;font-size:0.95rem;margin-top:4px;">Top issue tags</div>'
                    )
                    tags_html = "".join(
                        f'<div style="display:flex;justify-content:space-between;align-items:center;'
                        f'padding:7px 0;border-bottom:1px solid #f1f5f9;">'
                        f'<span style="background:#fef3c7;color:#92400e;font-size:0.72rem;font-weight:600;'
                        f'padding:2px 8px;border-radius:100px;">'
                        f'{_TAG_LABELS.get(entry["tag"], entry["tag"])}</span>'
                        f'<span style="font-weight:700;color:#0f172a;font-size:0.84rem;">{entry["count"]}</span>'
                        f'</div>'
                        for entry in top_tags
                    )
                    ui.html(
                        f'<div style="background:white;border:1px solid #e2e8f0;border-radius:12px;'
                        f'padding:4px 16px;">{tags_html}</div>'
                    )

                # Suggested fixes
                if suggested_fixes:
                    ui.html(
                        '<div style="font-weight:700;color:#0f172a;font-size:0.95rem;margin-top:4px;">Suggested prompt fixes</div>'
                    )
                    for fix in suggested_fixes:
                        tag_label = _TAG_LABELS.get(fix["tag"], fix["tag"])
                        ui.html(
                            f'<div style="border-left:3px solid #6366f1;padding:10px 14px;'
                            f'background:#fafbff;border-radius:0 8px 8px 0;margin-bottom:6px;">'
                            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
                            f'<span style="background:#e0e7ff;color:#3730a3;font-size:0.7rem;font-weight:700;'
                            f'padding:2px 7px;border-radius:100px;">{fix["count"]}×</span>'
                            f'<span style="font-size:0.84rem;font-weight:600;color:#1e293b;">{tag_label}</span></div>'
                            f'<div style="font-size:0.8rem;color:#475569;">{fix["proposed_fix"]}</div>'
                            f'</div>'
                        )


# ─────────────────────────────────────────────────────────────────────────────
# Quality Dashboard  /admin/quality
# ─────────────────────────────────────────────────────────────────────────────

# ── Module-level label maps (used only by the quality page) ──────────────────
_QA_TAG_LABELS: dict[str, str] = {
    "over_cautious_routing":                           "Over-cautious routing",
    "under_cautious_routing":                          "Under-cautious routing",
    "missing_safety_warning":                          "Missing safety warning",
    "hallucinated_hazard":                             "Hallucinated hazard",
    "vague_repair_steps":                              "Vague repair steps",
    "vague_tools_or_materials":                        "Vague tools/materials",
    "poor_grounding":                                  "Poor grounding",
    "unnecessary_clarification":                       "Unnecessary clarification",
    "missing_clarification":                           "Missing clarification",
    "unclear_user_language":                           "Unclear language",
    "wrong_trade_category":                            "Wrong trade category",
    "weak_step_ordering":                              "Weak step ordering",
    "emergency_wording_could_be_stronger":             "Emergency wording too weak",
    "emergency_repair_steps_inappropriately_expected": "Repair steps wrongly expected",
    "unsupported_professional_escalation":             "Unsupported escalation",
    "unsupported_hazard_claim":                        "Unsupported hazard claim",
    "wrong_professional_type":                         "Wrong professional type",
    "over_specific_urgency":                           "Urgency over-specified",
    "useful_but_overcautious":                         "Useful but overcautious",
    "evidence_uncertainty_not_explained":              "Uncertainty not explained",
    "correct_but_unclear_reasoning":                   "Correct but unclear",
    "hallucinate_hazard":                              "Hallucinated hazard",
}
_QA_CHIP_LABELS: dict[str, str] = {
    "wrong_diagnosis":                   "Wrong diagnosis",
    "too_cautious":                      "Too cautious",
    "not_cautious_enough":               "Not cautious enough",
    "not_detailed_enough":               "Not detailed enough",
    "wrong_tools_materials":             "Wrong tools/materials",
    "steps_unclear":                     "Steps unclear",
    "wrong_professional_recommendation": "Wrong professional recommendation",
    "advice_didnt_match_upload":         "Advice didn't match upload",
    "repeated_question":                 "Repeated a question",
    "advice_felt_unsafe":                "Advice felt unsafe",
    "other":                             "Other",
}
_QA_ALIGN_LABEL = {
    "likely_good":             "Likely Good",
    "confirmed_issue":         "Confirmed Issue",
    "judge_calibration_issue": "Calibration Gap",
    "safety_override":         "Safety Override",
    "needs_judge":             "Needs Judge",
    "monitor":                 "Monitor",
    "no_signal":               "No Signal",
}
_QA_ALIGN_BG = {
    "likely_good":             "#dcfce7",
    "confirmed_issue":         "#fee2e2",
    "judge_calibration_issue": "#fef3c7",
    "safety_override":         "#fef2f2",
    "needs_judge":             "#e0e7ff",
    "monitor":                 "#f1f5f9",
    "no_signal":               "#f8fafc",
}
_QA_ALIGN_FG = {
    "likely_good":             "#15803d",
    "confirmed_issue":         "#991b1b",
    "judge_calibration_issue": "#92400e",
    "safety_override":         "#991b1b",
    "needs_judge":             "#3730a3",
    "monitor":                 "#475569",
    "no_signal":               "#94a3b8",
}
_QA_VERDICT_BG = {"pass": "#dcfce7", "needs_review": "#fef3c7", "unsafe": "#fee2e2"}
_QA_VERDICT_FG = {"pass": "#15803d", "needs_review": "#92400e", "unsafe": "#991b1b"}
_QA_RATING_BG  = {"helpful": "#dcfce7", "partial": "#fef3c7", "not": "#fee2e2"}
_QA_RATING_FG  = {"helpful": "#15803d", "partial": "#92400e", "not": "#991b1b"}
_QA_RATING_LBL = {"helpful": "Helpful",  "partial": "Partially", "not": "No"}
_QA_PRIO_COL   = {"high": "#dc2626",    "medium": "#f59e0b",    "low": "#94a3b8"}
_QA_SIG_COL    = {"both": "#3730a3",    "judge": "#92400e",     "user": "#1e40af"}


@ui.page("/admin/quality")
async def admin_quality_page():
    token = app.storage.user.get("token")
    if not token:
        ui.navigate.to("/login")
        return
    if not app.storage.user.get("is_admin", False):
        ui.navigate.to("/cases")
        return

    H = {"Authorization": f"Bearer {token}"}
    summary: dict = {}
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{API}/admin/quality/summary", headers=H, timeout=30.0)
        if r.status_code == 403:
            ui.navigate.to("/cases")
            return
        if r.status_code == 200:
            summary = r.json()
    except Exception:
        pass

    # Unpack ─────────────────────────────────────────────────────────────────
    totals           = summary.get("totals") or {}
    cases            = summary.get("cases") or []
    needs_attention  = summary.get("needs_attention") or []
    verdict_counts   = summary.get("verdict_counts") or {}
    rating_counts    = summary.get("rating_counts") or {}
    align_counts     = summary.get("alignment_counts") or {}
    top_judge_tags   = summary.get("top_judge_tags") or []
    top_user_chips   = summary.get("top_user_chips") or []
    recurring_issues = summary.get("recurring_issues") or []
    suggestions      = summary.get("improvement_suggestions") or []

    # ── Render helpers ────────────────────────────────────────────────────────
    def _mcard(value, label, sub="", bg="#f8fafc", fg="#0f172a", val_fg: str | None = None) -> str:
        vfg = val_fg or fg
        sub_html = f'<div style="font-size:0.7rem;color:{fg};opacity:0.7;margin-top:1px;">{sub}</div>' if sub else ""
        return (
            f'<div style="background:{bg};border-radius:10px;padding:14px 16px;'
            f'flex:1;min-width:115px;">'
            f'<div style="font-size:0.68rem;color:{fg};opacity:0.65;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.05em;">{label}</div>'
            f'<div style="font-size:1.55rem;font-weight:800;color:{vfg};margin-top:3px;">{value}</div>'
            f'{sub_html}</div>'
        )

    def _sh(title: str) -> None:
        ui.html(
            f'<div style="font-weight:700;color:#0f172a;font-size:0.92rem;'
            f'margin:16px 0 7px 0;">{title}</div>'
        )

    def _bar_row(label: str, count: int, max_count: int, color: str = "#6366f1") -> str:
        pct = int(100 * count / max_count) if max_count else 0
        return (
            f'<div style="display:flex;align-items:center;gap:10px;padding:5px 0;'
            f'border-bottom:1px solid #f8fafc;">'
            f'<div style="flex:1;font-size:0.82rem;color:#334155;min-width:0;'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{label}</div>'
            f'<div style="width:130px;flex-shrink:0;">'
            f'<div style="background:#e2e8f0;border-radius:4px;height:8px;">'
            f'<div style="background:{color};height:8px;border-radius:4px;width:{pct}%;"></div>'
            f'</div></div>'
            f'<div style="font-weight:700;color:#0f172a;font-size:0.82rem;'
            f'width:22px;text-align:right;">{count}</div>'
            f'</div>'
        )

    def _tag_badge(tag: str) -> str:
        lbl = _QA_TAG_LABELS.get(tag, tag.replace("_", " "))
        return (
            f'<span style="background:#fef3c7;color:#92400e;font-size:0.68rem;'
            f'font-weight:600;padding:1px 7px;border-radius:100px;">{lbl}</span> '
        )

    def _chip_badge(chip: str) -> str:
        lbl = _QA_CHIP_LABELS.get(chip, chip.replace("_", " "))
        return (
            f'<span style="background:#e0e7ff;color:#3730a3;font-size:0.68rem;'
            f'font-weight:600;padding:1px 7px;border-radius:100px;">{lbl}</span> '
        )

    def _verdict_chip(verdict: str | None) -> str:
        if not verdict:
            return '<span style="color:#cbd5e1;font-size:0.75rem;">—</span>'
        bg  = _QA_VERDICT_BG.get(verdict, "#f8fafc")
        fg  = _QA_VERDICT_FG.get(verdict, "#475569")
        lbl = verdict.replace("_", " ").title()
        return (
            f'<span style="background:{bg};color:{fg};font-size:0.7rem;'
            f'font-weight:600;padding:2px 8px;border-radius:100px;">{lbl}</span>'
        )

    def _rating_chip(rating: str | None) -> str:
        if not rating:
            return '<span style="color:#cbd5e1;font-size:0.75rem;">—</span>'
        bg  = _QA_RATING_BG.get(rating, "#f8fafc")
        fg  = _QA_RATING_FG.get(rating, "#475569")
        lbl = _QA_RATING_LBL.get(rating, rating)
        return (
            f'<span style="background:{bg};color:{fg};font-size:0.7rem;'
            f'font-weight:600;padding:2px 8px;border-radius:100px;">{lbl}</span>'
        )

    def _align_chip(alignment: str) -> str:
        bg  = _QA_ALIGN_BG.get(alignment, "#f8fafc")
        fg  = _QA_ALIGN_FG.get(alignment, "#475569")
        lbl = _QA_ALIGN_LABEL.get(alignment, alignment)
        return (
            f'<span style="background:{bg};color:{fg};font-size:0.7rem;'
            f'font-weight:600;padding:2px 8px;border-radius:100px;">{lbl}</span>'
        )

    def _prio_badge(prio: str) -> str:
        _bg  = {"high": "#fee2e2", "medium": "#fef3c7", "low": "#f1f5f9"}
        _col = {"high": "#991b1b", "medium": "#92400e", "low": "#475569"}
        bg  = _bg.get(prio, "#f1f5f9")
        col = _col.get(prio, "#475569")
        return (
            f'<span style="background:{bg};color:{col};font-weight:700;font-size:0.68rem;'
            f'text-transform:uppercase;padding:2px 8px;border-radius:100px;">{prio}</span>'
        )

    _TH = (
        'style="padding:7px 8px;text-align:left;font-size:0.7rem;'
        'color:#94a3b8;font-weight:600;white-space:nowrap;"'
    )
    _TD = 'style="padding:7px 8px;font-size:0.78rem;color:#334155;vertical-align:middle;"'

    def _table_wrap(thead_html: str, tbody_html: str) -> str:
        return (
            f'<div style="overflow-x:auto;">'
            f'<table style="width:100%;border-collapse:collapse;background:white;'
            f'border:1px solid #e2e8f0;border-radius:12px;overflow:hidden;">'
            f'<thead><tr style="background:#f8fafc;">{thead_html}</tr></thead>'
            f'<tbody>{tbody_html}</tbody></table></div>'
        )

    def _bar_card(rows_html: str) -> str:
        return (
            f'<div style="background:white;border:1px solid #e2e8f0;'
            f'border-radius:12px;padding:10px 16px;">{rows_html}</div>'
        )

    def _case_link(case_id: int, url: str | None = None) -> str:
        href = url or f"/cases/{case_id}"
        return (
            f'<a href="{href}" target="_blank" '
            f'style="color:#6366f1;font-weight:700;text-decoration:none;">'
            f'#{case_id}</a>'
        )

    def _fmt_date(iso: str | None) -> str:
        """ISO 8601 string → '28 Jun 15:45'. Returns '—' for missing/invalid."""
        if not iso:
            return "—"
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            return f"{dt.day} {dt.strftime('%b %H:%M')}"
        except Exception:
            return "—"

    def _prop_bar(title: str, total: int, segments: list) -> str:
        """Stacked horizontal proportion bar.
        segments: [(label, count, bar_color, text_color), ...]
        """
        bar_segs = ""
        legend   = ""
        for label, count, bar_col, text_col in segments:
            pct = round(100 * count / total) if total else 0
            min_w = "2px" if count > 0 else "0"
            bar_segs += (
                f'<div style="background:{bar_col};width:{pct}%;min-width:{min_w};'
                f'transition:width 0.3s;"></div>'
            )
            legend += (
                f'<div style="display:flex;align-items:center;gap:5px;">'
                f'<span style="display:inline-block;width:9px;height:9px;border-radius:2px;'
                f'background:{bar_col};flex-shrink:0;"></span>'
                f'<span style="font-size:0.76rem;color:#64748b;">{label}</span>'
                f'<span style="font-size:0.84rem;font-weight:700;color:{text_col};">{count}</span>'
                f'<span style="font-size:0.7rem;color:#94a3b8;">{pct}%</span>'
                f'</div>'
            )
        total_label = f"{total} total" if total else "no data yet"
        return (
            f'<div style="background:white;border:1px solid #e2e8f0;border-radius:12px;'
            f'padding:14px 16px;margin-bottom:10px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
            f'margin-bottom:9px;">'
            f'<span style="font-size:0.85rem;font-weight:700;color:#0f172a;">{title}</span>'
            f'<span style="font-size:0.72rem;color:#94a3b8;">{total_label}</span>'
            f'</div>'
            f'<div style="display:flex;height:13px;border-radius:7px;overflow:hidden;'
            f'background:#f1f5f9;margin-bottom:10px;">{bar_segs}</div>'
            f'<div style="display:flex;gap:16px;flex-wrap:wrap;">{legend}</div>'
            f'</div>'
        )

    # ── Page ─────────────────────────────────────────────────────────────────
    inject_styles()
    mobile_bar()
    sidebar("/admin/quality")

    with ui.element("div").classes("main"):
        with ui.element("div").style(
            "padding:28px 32px;max-width:980px;display:flex;flex-direction:column;gap:0;"
        ):
            tc = totals.get("total_cases", 0)
            ui.html(
                f'<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:18px;">'
                f'<h1 style="font-size:1.5rem;font-weight:800;color:#0f172a;margin:0;">Quality Dashboard</h1>'
                f'<span style="background:#f1f5f9;color:#475569;font-size:0.72rem;font-weight:700;'
                f'padding:3px 10px;border-radius:100px;">'
                f'{tc} case{"s" if tc != 1 else ""} · last 200</span></div>'
            )

            if not summary:
                ui.html(
                    '<div style="background:#fef2f2;border:1px solid #fca5a5;border-radius:12px;'
                    'padding:16px;color:#991b1b;font-size:0.86rem;">'
                    'Could not load quality summary. Make sure the backend is running.</div>'
                )
                return

            # ── Headline metrics (computed once, used in banner + tabs) ───────
            _total_fb  = sum(rating_counts.values())
            _hlth_h    = rating_counts.get("helpful", 0) / max(_total_fb, 1)
            _avg_s_raw = totals.get("avg_judge_score")
            _hlth_j    = (_avg_s_raw or 70) / 100
            _hlth_ci   = align_counts.get("confirmed_issue", 0) / max(tc, 1)
            _health    = round(_hlth_h * 40 + _hlth_j * 40 + (1 - _hlth_ci) * 20)
            _hlth_col  = "#15803d" if _health >= 75 else "#92400e" if _health >= 55 else "#991b1b"
            _hlth_bg   = "#dcfce7" if _health >= 75 else "#fef3c7" if _health >= 55 else "#fee2e2"
            _hlth_lbl  = "Healthy" if _health >= 75 else "Monitor" if _health >= 55 else "Critical"

            _both_cases = [c for c in cases if c.get("judge_verdict") and c.get("fb_rating")]
            _agree_n    = sum(1 for c in _both_cases
                              if c["alignment"] in ("likely_good", "confirmed_issue"))
            _agree_pct  = round(100 * _agree_n / max(len(_both_cases), 1)) if _both_cases else None
            _agree_col  = ("#15803d" if (_agree_pct or 0) >= 70
                           else "#92400e" if (_agree_pct or 0) >= 50 else "#991b1b")

            from datetime import datetime as _dt, timezone as _tz, timedelta as _td
            from collections import Counter as _Ctr
            _cutoff  = _dt.now(_tz.utc) - _td(days=7)
            _recent  = [c for c in cases if c.get("created_at") and
                        _dt.fromisoformat(c["created_at"].replace("Z", "+00:00")) >= _cutoff]
            _wk_tags = _Ctr()
            for _wc in _recent:
                for _wt in (_wc.get("judge_tags") or []):
                    _wk_tags[_wt] += 1
                for _wch in (_wc.get("fb_chips") or []):
                    _wk_tags[_wch] += 1
            _top3_wk = _wk_tags.most_common(3)

            # ── QA Health banner ──────────────────────────────────────────────
            ui.html(
                f'<div style="display:flex;gap:16px;align-items:center;flex-wrap:wrap;'
                f'background:white;border:1px solid #e2e8f0;border-radius:12px;'
                f'padding:12px 18px;margin-bottom:16px;">'
                f'<div style="display:flex;align-items:baseline;gap:6px;" '
                f'title="Composite: helpful rate ×40 + avg judge score ×40 + (1 − confirmed-issue rate) ×20">'
                f'<span style="font-size:0.78rem;font-weight:700;color:#64748b;">QA Health</span>'
                f'<span style="font-size:1.7rem;font-weight:800;color:{_hlth_col};">{_health}</span>'
                f'<span style="font-size:0.72rem;color:#94a3b8;">/100</span>'
                f'<span style="background:{_hlth_bg};color:{_hlth_col};font-size:0.68rem;font-weight:700;'
                f'padding:2px 8px;border-radius:100px;margin-left:4px;">{_hlth_lbl}</span>'
                f'</div>'
                f'<div style="width:1px;height:30px;background:#e2e8f0;flex-shrink:0;"></div>'
                + (
                    f'<div style="display:flex;align-items:baseline;gap:6px;" '
                    f'title="Cases where judge and user both agreed (Likely Good or Confirmed Issue), '
                    f'divided by all cases that have both a judge verdict and user feedback">'
                    f'<span style="font-size:0.78rem;font-weight:700;color:#64748b;">Judge–User Agreement</span>'
                    f'<span style="font-size:1.7rem;font-weight:800;color:{_agree_col};">{_agree_pct}%</span>'
                    f'<span style="font-size:0.72rem;color:#94a3b8;">({len(_both_cases)} paired)</span>'
                    f'</div>'
                    if _agree_pct is not None else
                    f'<span style="font-size:0.8rem;color:#94a3b8;font-style:italic;">'
                    f'Judge–User Agreement: no paired cases yet</span>'
                )
                + f'<div style="margin-left:auto;font-size:0.68rem;color:#cbd5e1;white-space:nowrap;">'
                f'helpful×40 · judge×40 · conflict×20</div>'
                f'</div>'
            )

            # ── Tabs ─────────────────────────────────────────────────────────
            with ui.tabs().props("align=left dense").style(
                "font-weight:600;font-size:0.8rem;"
                "border-bottom:2px solid #e2e8f0;margin-bottom:20px;"
            ) as tab_bar:
                t_ov  = ui.tab("Overview")
                t_fb  = ui.tab("User Feedback")
                t_jdg = ui.tab("Judge")
                t_aln = ui.tab("Alignment")
                t_rec = ui.tab("Recurring Issues")
                t_sug = ui.tab("Suggestions")

            with ui.tab_panels(tab_bar, value=t_ov).style("width:100%;"):

                # ════════════════════════════════════════════════════════════
                # TAB 1: Overview
                # ════════════════════════════════════════════════════════════
                with ui.tab_panel(t_ov):
                    tf = totals.get("total_feedback", 0)
                    tj = totals.get("total_judged", 0)
                    ta = totals.get("avg_judge_score")
                    tn = totals.get("needs_attention", 0)

                    ui.html(
                        '<div style="display:flex;gap:10px;flex-wrap:wrap;">'
                        + _mcard(tc, "Total Cases")
                        + _mcard(tf, "With Feedback",
                                 f"{round(100*tf/tc)}% rate" if tc else "",
                                 "#f0fdf4" if tf else "#f8fafc", "#15803d" if tf else "#94a3b8")
                        + _mcard(tj, "Judged",
                                 f"{round(100*tj/tc)}% coverage" if tc else "",
                                 "#f0f9ff" if tj else "#f8fafc", "#0369a1" if tj else "#94a3b8")
                        + (
                            _mcard(
                                str(ta), "Avg Judge Score", "",
                                "#dcfce7" if ta >= 85 else "#fef3c7" if ta >= 70 else "#fee2e2",
                                "#0f172a",
                                "#15803d" if ta >= 85 else "#92400e" if ta >= 70 else "#991b1b",
                            ) if ta is not None else
                            _mcard("—", "Avg Judge Score")
                        )
                        + _mcard(tn, "Needs Attention",
                                 "high + medium priority" if tn else "",
                                 "#fef2f2" if tn else "#f0fdf4",
                                 "#991b1b" if tn else "#15803d",
                                 "#991b1b" if tn else "#15803d")
                        + '</div>'
                    )

                    # Feedback rating stacked bar
                    ui.html(_prop_bar(
                        "Feedback Ratings",
                        sum(rating_counts.values()),
                        [
                            ("Helpful",   rating_counts.get("helpful", 0), "#4ade80", "#15803d"),
                            ("Partially", rating_counts.get("partial", 0), "#fbbf24", "#92400e"),
                            ("No",        rating_counts.get("not", 0),     "#f87171", "#991b1b"),
                        ],
                    ))

                    # Judge verdict stacked bar
                    ui.html(_prop_bar(
                        "Judge Verdicts",
                        sum(verdict_counts.values()),
                        [
                            ("Pass",         verdict_counts.get("pass", 0),         "#4ade80", "#15803d"),
                            ("Needs Review", verdict_counts.get("needs_review", 0), "#fbbf24", "#92400e"),
                            ("Unsafe",       verdict_counts.get("unsafe", 0),       "#f87171", "#991b1b"),
                        ],
                    ))

                    # Top recurring issue callout
                    if recurring_issues:
                        top_ri = recurring_issues[0]
                        sig_col = _QA_SIG_COL.get(top_ri["signal"], "#475569")
                        _sh("Top Recurring Issue")
                        ui.html(
                            f'<div style="background:#fffbeb;border:1px solid #fde68a;'
                            f'border-radius:10px;padding:12px 16px;font-size:0.84rem;color:#92400e;">'
                            f'<strong>{top_ri["label"]}</strong> — '
                            f'{top_ri["total"]} occurrence(s) '
                            f'(judge: {top_ri["judge_count"]}, user: {top_ri["user_count"]}) · '
                            f'<span style="color:{sig_col};font-weight:700;">'
                            f'{top_ri["signal"]} signal</span></div>'
                        )

                    # Top 3 this week
                    _sh(f"This Week · {len(_recent)} case{'s' if len(_recent) != 1 else ''}")
                    if _top3_wk:
                        _wk_rows = ""
                        for _rank, (_lbl, _cnt) in enumerate(_top3_wk, 1):
                            _readable = (
                                _QA_TAG_LABELS.get(_lbl)
                                or _QA_CHIP_LABELS.get(_lbl)
                                or _lbl.replace("_", " ").title()
                            )
                            _rank_col = ["#6366f1", "#f59e0b", "#94a3b8"][_rank - 1]
                            _wk_rows += (
                                f'<div style="display:flex;align-items:center;gap:10px;'
                                f'padding:8px 0;border-bottom:1px solid #f1f5f9;">'
                                f'<span style="font-size:0.8rem;font-weight:800;color:{_rank_col};'
                                f'width:18px;text-align:center;">#{_rank}</span>'
                                f'<span style="font-size:0.84rem;color:#0f172a;flex:1;">{_readable}</span>'
                                f'<span style="background:#f1f5f9;color:#475569;font-size:0.72rem;'
                                f'font-weight:700;padding:2px 10px;border-radius:100px;">{_cnt}×</span>'
                                f'</div>'
                            )
                        ui.html(
                            f'<div style="background:white;border:1px solid #e2e8f0;border-radius:10px;'
                            f'padding:4px 14px;">{_wk_rows}</div>'
                        )
                    elif _recent:
                        ui.html('<div style="font-size:0.84rem;color:#94a3b8;">No tagged issues in the last 7 days.</div>')
                    else:
                        ui.html('<div style="font-size:0.84rem;color:#94a3b8;">No cases in the last 7 days.</div>')

                # ════════════════════════════════════════════════════════════
                # TAB 2: User Feedback
                # ════════════════════════════════════════════════════════════
                with ui.tab_panel(t_fb):
                    _sh("Ratings")
                    ui.html(
                        '<div style="display:flex;gap:10px;flex-wrap:wrap;">'
                        + "".join(
                            _mcard(
                                rating_counts.get(r, 0), _QA_RATING_LBL[r],
                                "", _QA_RATING_BG[r], _QA_RATING_FG[r], _QA_RATING_FG[r],
                            )
                            for r in ("helpful", "partial", "not")
                        )
                        + '</div>'
                    )

                    if top_user_chips:
                        _sh("Top Reason Chips")
                        max_c = top_user_chips[0]["count"] or 1
                        ui.html(_bar_card(
                            "".join(_bar_row(e["label"], e["count"], max_c, "#6366f1")
                                    for e in top_user_chips)
                        ))
                    else:
                        ui.html('<div style="color:#94a3b8;font-size:0.84rem;margin-top:8px;">No reason chips recorded yet.</div>')

                    _fb_f    = {'rating': 'All', 'chips': 'All', 'mode': 'All'}
                    _fb_base = sorted(
                        [c for c in cases if c["fb_rating"]],
                        key=lambda c: {"not": 0, "partial": 1, "helpful": 2}.get(c["fb_rating"], 3),
                    )
                    _OM_OPTS = ['All', 'diy', 'verify_first', 'assessment_first',
                                'active_emergency', 'clarification', 'care', 'out_of_scope']

                    @ui.refreshable
                    def _fb_table():
                        f = _fb_base
                        if _fb_f['rating'] != 'All':
                            _rmap = {'Helpful': 'helpful', 'Partially': 'partial', 'No': 'not'}
                            f = [c for c in f if c['fb_rating'] == _rmap.get(_fb_f['rating'])]
                        if _fb_f['chips'] == 'Yes':
                            f = [c for c in f if c.get('fb_chips')]
                        elif _fb_f['chips'] == 'No':
                            f = [c for c in f if not c.get('fb_chips')]
                        if _fb_f['mode'] != 'All':
                            f = [c for c in f if (c.get('output_mode') or '') == _fb_f['mode']]
                        _af = sum(1 for v in _fb_f.values() if v != 'All')
                        _af_note = f' · {_af} filter{"s" if _af != 1 else ""} active' if _af else ''
                        ui.html(
                            f'<div style="font-size:0.76rem;color:#94a3b8;margin-bottom:8px;">'
                            f'Showing {len(f)} of {len(_fb_base)} cases{_af_note}</div>'
                        )
                        if not f:
                            ui.html(
                                '<div style="background:#f8fafc;border:1px solid #e2e8f0;'
                                'border-radius:8px;padding:12px 14px;font-size:0.84rem;'
                                'color:#94a3b8;">No cases match these filters.</div>'
                            )
                            return
                        rows_html = ""
                        for c in f[:30]:
                            chips_html = "".join(_chip_badge(ch) for ch in (c["fb_chips"] or [])[:4])
                            rows_html += (
                                f'<tr style="border-bottom:1px solid #f1f5f9;">'
                                f'<td {_TD}>{_case_link(c["case_id"], c.get("case_url"))}</td>'
                                f'<td {_TD}>{c["output_mode"] or "—"}</td>'
                                f'<td {_TD} style="max-width:130px;color:#475569;font-size:0.75rem;">'
                                f'{(c["problem_type"] or "")[:35] or "—"}</td>'
                                f'<td {_TD} style="color:#94a3b8;font-size:0.75rem;white-space:nowrap;">'
                                f'{_fmt_date(c.get("fb_created_at"))}</td>'
                                f'<td {_TD}>{_rating_chip(c["fb_rating"])}</td>'
                                f'<td {_TD}>{chips_html or "—"}</td>'
                                f'<td {_TD} style="max-width:180px;color:#64748b;">'
                                f'{c["fb_comment"] or "—"}</td>'
                                f'</tr>'
                            )
                        th_html = "".join(
                            f'<th {_TH}>{h}</th>'
                            for h in ("Case", "Mode", "Problem", "Feedback date", "Rating", "Chips", "Comment")
                        )
                        ui.html(_table_wrap(th_html, rows_html))

                    if _fb_base:
                        _sh("Feedback Cases")
                        with ui.row().style(
                            'gap:8px;flex-wrap:wrap;align-items:flex-end;'
                            'background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;'
                            'padding:10px 12px;margin-bottom:12px;'
                        ):
                            _fb_r_sel = ui.select(['All', 'Helpful', 'Partially', 'No'], value='All', label='Rating').style('width:120px;')
                            _fb_r_sel.on_value_change(lambda e: (_fb_f.update({'rating': e.value}), _fb_table.refresh()))
                            _fb_c_sel = ui.select(['All', 'Yes', 'No'], value='All', label='Has chips').style('width:110px;')
                            _fb_c_sel.on_value_change(lambda e: (_fb_f.update({'chips': e.value}), _fb_table.refresh()))
                            _fb_m_sel = ui.select(_OM_OPTS, value='All', label='Mode').style('width:160px;')
                            _fb_m_sel.on_value_change(lambda e: (_fb_f.update({'mode': e.value}), _fb_table.refresh()))
                            def _reset_fb():
                                _fb_f.update({'rating': 'All', 'chips': 'All', 'mode': 'All'})
                                _fb_r_sel.value = 'All'
                                _fb_c_sel.value = 'All'
                                _fb_m_sel.value = 'All'
                                _fb_table.refresh()
                            ui.button('↺ Reset', on_click=_reset_fb).props('flat dense').style(
                                'color:#6366f1;font-size:0.76rem;align-self:flex-end;margin-left:4px;'
                            )
                        _fb_table()
                    else:
                        ui.html('<div style="color:#94a3b8;font-size:0.84rem;margin-top:8px;">No feedback yet.</div>')

                # ════════════════════════════════════════════════════════════
                # TAB 3: Judge Evaluation
                # ════════════════════════════════════════════════════════════
                with ui.tab_panel(t_jdg):
                    avg_s = totals.get("avg_judge_score")
                    _sh("Verdicts")
                    ui.html(
                        '<div style="display:flex;gap:10px;flex-wrap:wrap;">'
                        + "".join(
                            _mcard(
                                verdict_counts.get(v, 0),
                                v.replace("_", " ").title(),
                                "", _QA_VERDICT_BG.get(v, "#f8fafc"),
                                "#0f172a", _QA_VERDICT_FG.get(v, "#475569"),
                            )
                            for v in ("pass", "needs_review", "unsafe")
                        )
                        + _mcard(str(avg_s) if avg_s is not None else "—", "Avg Score")
                        + '</div>'
                    )

                    if top_judge_tags:
                        _sh("Top Issue Tags")
                        max_t = top_judge_tags[0]["count"] or 1
                        ui.html(_bar_card(
                            "".join(_bar_row(e["label"], e["count"], max_t, "#f59e0b")
                                    for e in top_judge_tags[:12])
                        ))
                    else:
                        ui.html('<div style="color:#94a3b8;font-size:0.84rem;margin-top:8px;">No judge evaluations yet.</div>')

                    judged = [c for c in cases if c["judge_verdict"]]
                    judged.sort(key=lambda c: (
                        {"unsafe": 0, "needs_review": 1, "pass": 2}.get(c["judge_verdict"], 3),
                        (c["judge_score"] or 100),
                    ))

                    if judged:
                        _jdg_f = {'verdict': 'All', 'mode': 'All', 'score': 'All'}
                        _SCORE_BUCKETS = [
                            ("90–100", 90, 101, "#15803d"),
                            ("80–89",  80,  90, "#4ade80"),
                            ("70–79",  70,  80, "#f59e0b"),
                            ("60–69",  60,  70, "#f97316"),
                            ("<60",     0,  60, "#dc2626"),
                        ]
                        _SCORE_BUCKET_MAP = {b[0]: (b[1], b[2]) for b in _SCORE_BUCKETS}
                        _JOM = ['All', 'diy', 'verify_first', 'assessment_first',
                                'active_emergency', 'clarification', 'care', 'out_of_scope']

                        @ui.refreshable
                        def _jdg_table():
                            f = judged
                            if _jdg_f['verdict'] != 'All':
                                _vmap = {'Pass': 'pass', 'Needs Review': 'needs_review', 'Unsafe': 'unsafe'}
                                f = [c for c in f if c['judge_verdict'] == _vmap.get(_jdg_f['verdict'])]
                            if _jdg_f['mode'] != 'All':
                                f = [c for c in f if (c.get('output_mode') or '') == _jdg_f['mode']]
                            if _jdg_f['score'] != 'All':
                                lo, hi = _SCORE_BUCKET_MAP[_jdg_f['score']]
                                f = [c for c in f if c['judge_score'] is not None and lo <= c['judge_score'] < hi]
                            # Score distribution — updates with filter
                            _dist = {
                                b[0]: sum(1 for c in f if c["judge_score"] is not None and b[1] <= c["judge_score"] < b[2])
                                for b in _SCORE_BUCKETS
                            }
                            _max_d = max(_dist.values()) or 1
                            ui.html(_bar_card(
                                "".join(
                                    _bar_row(bucket, _dist[bucket], _max_d, col)
                                    for bucket, _, __, col in _SCORE_BUCKETS
                                    if _dist[bucket] > 0
                                ) or '<div style="color:#94a3b8;font-size:0.82rem;padding:4px 0;">No scored cases in filter.</div>'
                            ))
                            _af = sum(1 for v in _jdg_f.values() if v != 'All')
                            _af_note = f' · {_af} filter{"s" if _af != 1 else ""} active' if _af else ''
                            ui.html(
                                f'<div style="font-size:0.76rem;color:#94a3b8;margin:8px 0;">'
                                f'Showing {len(f)} of {len(judged)} judged cases{_af_note}</div>'
                            )
                            if not f:
                                ui.html(
                                    '<div style="background:#f8fafc;border:1px solid #e2e8f0;'
                                    'border-radius:8px;padding:12px 14px;font-size:0.84rem;'
                                    'color:#94a3b8;">No cases match these filters.</div>'
                                )
                                return
                            rows_html = ""
                            for c in f[:30]:
                                tags_html = "".join(_tag_badge(t) for t in (c["judge_tags"] or [])[:2])
                                score_col = (
                                    "#991b1b" if (c["judge_score"] or 100) < 60
                                    else "#92400e" if (c["judge_score"] or 100) < 80
                                    else "#15803d"
                                )
                                prob = (c["problem_type"] or "")[:35] or "—"
                                rows_html += (
                                    f'<tr style="border-bottom:1px solid #f1f5f9;">'
                                    f'<td {_TD}>{_case_link(c["case_id"], c.get("case_url"))}</td>'
                                    f'<td {_TD}>{c["output_mode"] or "—"}</td>'
                                    f'<td {_TD} style="max-width:140px;color:#475569;font-size:0.75rem;">{prob}</td>'
                                    f'<td {_TD} style="color:#94a3b8;font-size:0.75rem;white-space:nowrap;">'
                                    f'{_fmt_date(c.get("judged_at"))}</td>'
                                    f'<td {_TD}>'
                                    f'<span style="font-size:1rem;font-weight:800;color:{score_col};">'
                                    f'{c["judge_score"] if c["judge_score"] is not None else "—"}</span></td>'
                                    f'<td {_TD}>{_verdict_chip(c["judge_verdict"])}</td>'
                                    f'<td {_TD}>{tags_html or "<span style=\"color:#cbd5e1;\">none</span>"}</td>'
                                    f'</tr>'
                                )
                            th_html = "".join(
                                f'<th {_TH}>{h}</th>'
                                for h in ("Case", "Mode", "Problem", "Judged", "Score", "Verdict", "Top Tags")
                            )
                            ui.html(_table_wrap(th_html, rows_html))

                        _sh("Score Distribution & Cases")
                        with ui.row().style(
                            'gap:8px;flex-wrap:wrap;align-items:flex-end;'
                            'background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;'
                            'padding:10px 12px;margin-bottom:12px;'
                        ):
                            _jdg_v_sel = ui.select(['All', 'Pass', 'Needs Review', 'Unsafe'], value='All', label='Verdict').style('width:130px;')
                            _jdg_v_sel.on_value_change(lambda e: (_jdg_f.update({'verdict': e.value}), _jdg_table.refresh()))
                            _jdg_m_sel = ui.select(_JOM, value='All', label='Mode').style('width:160px;')
                            _jdg_m_sel.on_value_change(lambda e: (_jdg_f.update({'mode': e.value}), _jdg_table.refresh()))
                            _jdg_s_sel = ui.select(['All'] + [b[0] for b in _SCORE_BUCKETS], value='All', label='Score range').style('width:130px;')
                            _jdg_s_sel.on_value_change(lambda e: (_jdg_f.update({'score': e.value}), _jdg_table.refresh()))
                            def _reset_jdg():
                                _jdg_f.update({'verdict': 'All', 'mode': 'All', 'score': 'All'})
                                _jdg_v_sel.value = 'All'
                                _jdg_m_sel.value = 'All'
                                _jdg_s_sel.value = 'All'
                                _jdg_table.refresh()
                            ui.button('↺ Reset', on_click=_reset_jdg).props('flat dense').style(
                                'color:#6366f1;font-size:0.76rem;align-self:flex-end;margin-left:4px;'
                            )
                        _jdg_table()

                # ════════════════════════════════════════════════════════════
                # TAB 4: Alignment
                # ════════════════════════════════════════════════════════════
                with ui.tab_panel(t_aln):
                    # Critical cases pinned at top
                    critical = sorted(
                        [c for c in cases if c["alignment"] in ("safety_override", "confirmed_issue")],
                        key=lambda c: {"safety_override": 0, "confirmed_issue": 1}.get(c["alignment"], 2),
                    )
                    if critical:
                        ui.html(
                            f'<div style="background:#fef2f2;border:1px solid #fca5a5;border-radius:10px;'
                            f'padding:10px 14px;margin-bottom:12px;display:flex;align-items:center;gap:10px;'
                            f'flex-wrap:wrap;">'
                            f'<span style="font-size:0.9rem;font-weight:700;color:#991b1b;">⚠ Critical</span>'
                            f'<span style="font-size:0.78rem;color:#b91c1c;">'
                            f'{len(critical)} case{"s" if len(critical) != 1 else ""} '
                            f'with conflicting judge/user signals</span>'
                            f'<span style="font-size:0.7rem;color:#fca5a5;margin-left:auto;">'
                            f'Always shown — not affected by filters below</span>'
                            f'</div>'
                        )
                        crit_rows = ""
                        for c in critical[:8]:
                            prob      = (c["problem_type"] or "")[:30] or "—"
                            reason    = c.get("priority_reason") or "—"
                            case_href = c.get("case_url") or f'/cases/{c["case_id"]}'
                            crit_rows += (
                                f'<tr style="border-bottom:1px solid #fef2f2;">'
                                f'<td {_TD}>{_case_link(c["case_id"], c.get("case_url"))}</td>'
                                f'<td {_TD} style="max-width:120px;color:#475569;font-size:0.75rem;">{prob}</td>'
                                f'<td {_TD}>{_align_chip(c["alignment"])}</td>'
                                f'<td {_TD}>{_verdict_chip(c["judge_verdict"])}</td>'
                                f'<td {_TD}>{_rating_chip(c["fb_rating"])}</td>'
                                f'<td {_TD} style="max-width:160px;color:#64748b;font-size:0.75rem;">{reason}</td>'
                                f'<td {_TD}>'
                                f'<a href="{case_href}" target="_blank" '
                                f'style="color:#6366f1;font-weight:600;font-size:0.78rem;text-decoration:none;">'
                                f'Open →</a></td>'
                                f'</tr>'
                            )
                        crit_th = "".join(
                            f'<th {_TH}>{h}</th>'
                            for h in ("Case", "Problem", "Alignment", "Judge", "User", "Reason", "")
                        )
                        ui.html(_table_wrap(crit_th, crit_rows))

                    _sh("Judge vs User Signal")

                    align_order = [
                        "likely_good", "confirmed_issue", "judge_calibration_issue",
                        "safety_override", "needs_judge", "monitor", "no_signal",
                    ]
                    ui.html(
                        '<div style="display:flex;gap:10px;flex-wrap:wrap;">'
                        + "".join(
                            _mcard(
                                align_counts.get(k, 0),
                                _QA_ALIGN_LABEL.get(k, k),
                                "",
                                _QA_ALIGN_BG.get(k, "#f8fafc"),
                                _QA_ALIGN_FG.get(k, "#475569"),
                                _QA_ALIGN_FG.get(k, "#475569"),
                            )
                            for k in align_order
                            if align_counts.get(k, 0) > 0
                        )
                        + '</div>'
                    )

                    # Alignment explainer
                    ui.html(
                        '<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;'
                        'padding:10px 14px;font-size:0.75rem;color:#64748b;margin:12px 0;">'
                        '<strong>Key:</strong> '
                        '<span style="color:#15803d;">Likely Good</span> = judge pass + user helpful &nbsp;·&nbsp; '
                        '<span style="color:#991b1b;">Confirmed Issue</span> = judge flag + user complaint &nbsp;·&nbsp; '
                        '<span style="color:#92400e;">Calibration Gap</span> = judge pass but user said No &nbsp;·&nbsp; '
                        '<span style="color:#991b1b;">Safety Override</span> = judge unsafe but user said Helpful'
                        '</div>'
                    )

                    # ── 4 alignment signal cards ──────────────────────────────
                    _sh("What the Data Shows")
                    _paired_all   = [c for c in cases if c.get("judge_verdict") and c.get("fb_rating")]
                    _n_paired     = len(_paired_all)
                    _both_good    = [c for c in _paired_all if c["judge_verdict"] == "pass"   and c["fb_rating"] == "helpful"]
                    _both_flagged = [c for c in _paired_all if c["judge_verdict"] in ("needs_review", "unsafe") and c["fb_rating"] in ("partial", "not")]
                    _safety_conf  = [c for c in _paired_all if c["judge_verdict"] == "unsafe" and c["fb_rating"] == "helpful"]
                    _calib_gap    = [c for c in _paired_all if c["judge_verdict"] == "pass"   and c["fb_rating"] in ("not", "partial")]

                    def _ac_pct(lst):
                        return round(100 * len(lst) / _n_paired) if _n_paired else 0

                    def _ac_ex(lst):
                        return " ".join(
                            f'<a href="/cases/{c["case_id"]}" target="_blank" '
                            f'style="color:inherit;font-weight:700;text-decoration:none;opacity:0.8;">'
                            f'#{c["case_id"]}</a>'
                            for c in lst[:3]
                        )

                    def _ac(icon, title, lst, bg, border, fg, fg_sub, desc):
                        n   = len(lst)
                        pct = _ac_pct(lst)
                        ex  = _ac_ex(lst)
                        return (
                            f'<div style="background:{bg};border:1.5px solid {border};border-radius:14px;'
                            f'padding:16px;flex:1;min-width:185px;">'
                            f'<div style="font-size:0.95rem;margin-bottom:4px;">{icon}</div>'
                            f'<div style="font-size:0.8rem;font-weight:700;color:{fg};margin-bottom:8px;">{title}</div>'
                            f'<div style="font-size:2.4rem;font-weight:800;color:{fg};line-height:1;">{n}</div>'
                            f'<div style="font-size:0.72rem;color:{fg_sub};margin-top:3px;">'
                            f'{pct}% of {_n_paired} paired</div>'
                            f'<div style="font-size:0.75rem;color:{fg};opacity:0.85;margin-top:10px;">{desc}</div>'
                            + (f'<div style="font-size:0.7rem;color:{fg_sub};margin-top:8px;">e.g. {ex}</div>' if ex else "")
                            + f'</div>'
                        )

                    ui.html(
                        f'<div style="display:flex;gap:12px;flex-wrap:wrap;margin:14px 0;">'
                        + _ac("✅", "Both Good",
                              _both_good, "#f0fdf4", "#86efac", "#15803d", "#4ade80",
                              "Judge passed and user said Helpful.")
                        + _ac("⚠️", "Both Flagged",
                              _both_flagged, "#fffbeb", "#fde68a", "#92400e", "#d97706",
                              "Judge and user both identified an issue.")
                        + _ac("🔴", "Safety Conflict",
                              _safety_conf, "#fef2f2", "#fca5a5", "#991b1b", "#f87171",
                              "User said Helpful — judge flagged Unsafe.")
                        + _ac("🟡", "Calibration Gap",
                              _calib_gap, "#fefce8", "#fef08a", "#b45309", "#eab308",
                              "User was unhappy despite judge passing.")
                        + f'</div>'
                        + (f'<div style="font-size:0.72rem;color:#94a3b8;margin-bottom:12px;">'
                           f'{_n_paired} cases have both signals. Cases appear in only one card.</div>'
                           if _n_paired else
                           f'<div style="font-size:0.72rem;color:#94a3b8;margin-bottom:12px;">'
                           f'No cases have both a judge verdict and user feedback yet.</div>')
                    )

                    if needs_attention:
                        _aln_f   = {'alignment': 'All', 'priority': 'All'}
                        _ALN_OPTS = ['All', 'likely_good', 'confirmed_issue',
                                     'judge_calibration_issue', 'safety_override',
                                     'needs_judge', 'monitor', 'no_signal']
                        _PRIO_OPTS = ['All', 'high', 'medium', 'low']

                        @ui.refreshable
                        def _aln_table():
                            f = needs_attention
                            if _aln_f['alignment'] != 'All':
                                f = [c for c in f if c['alignment'] == _aln_f['alignment']]
                            if _aln_f['priority'] != 'All':
                                f = [c for c in f if c['priority'] == _aln_f['priority']]
                            _af = sum(1 for v in _aln_f.values() if v != 'All')
                            _af_note = f' · {_af} filter{"s" if _af != 1 else ""} active' if _af else ''
                            ui.html(
                                f'<div style="font-size:0.76rem;color:#94a3b8;margin-bottom:8px;">'
                                f'Showing {len(f)} of {len(needs_attention)} cases needing attention{_af_note}</div>'
                            )
                            if not f:
                                ui.html(
                                    '<div style="background:#f8fafc;border:1px solid #e2e8f0;'
                                    'border-radius:8px;padding:12px 14px;font-size:0.84rem;'
                                    'color:#94a3b8;">No cases match these filters.</div>'
                                )
                                return
                            rows_html = ""
                            for c in f[:25]:
                                action    = c.get("suggested_action") or ""
                                prob      = (c["problem_type"] or "")[:30] or "—"
                                reason    = c.get("priority_reason") or "—"
                                case_href = c.get("case_url") or f'/cases/{c["case_id"]}'
                                row_hl    = 'border-left:3px solid #f87171;' if c['priority'] == 'high' else 'border-left:3px solid transparent;'
                                action_sub = (
                                    f'<div style="color:#94a3b8;font-size:0.68rem;margin-top:2px;">{action}</div>'
                                    if action else ''
                                )
                                rows_html += (
                                    f'<tr style="border-bottom:1px solid #f1f5f9;{row_hl}">'
                                    f'<td {_TD}>{_case_link(c["case_id"], c.get("case_url"))}</td>'
                                    f'<td {_TD}>{c["output_mode"] or "—"}</td>'
                                    f'<td {_TD} style="max-width:120px;color:#475569;font-size:0.75rem;">{prob}</td>'
                                    f'<td {_TD}>{_verdict_chip(c["judge_verdict"])}</td>'
                                    f'<td {_TD}>{_rating_chip(c["fb_rating"])}</td>'
                                    f'<td {_TD}>{_align_chip(c["alignment"])}</td>'
                                    f'<td {_TD}>{_prio_badge(c["priority"])}</td>'
                                    f'<td {_TD} style="max-width:170px;">'
                                    f'<div style="color:#64748b;font-size:0.75rem;">{reason}</div>'
                                    f'{action_sub}</td>'
                                    f'<td {_TD} style="white-space:nowrap;">'
                                    f'<a href="{case_href}" target="_blank" '
                                    f'style="color:#6366f1;font-weight:600;font-size:0.78rem;text-decoration:none;">'
                                    f'Open →</a></td>'
                                    f'</tr>'
                                )
                            th_html = "".join(
                                f'<th {_TH}>{h}</th>'
                                for h in ("Case", "Mode", "Problem", "Judge", "User", "Alignment", "Priority", "Reason", "Open")
                            )
                            ui.html(_table_wrap(th_html, rows_html))

                        _sh("Cases Needing Attention")
                        with ui.row().style(
                            'gap:8px;flex-wrap:wrap;align-items:flex-end;'
                            'background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;'
                            'padding:10px 12px;margin-bottom:12px;'
                        ):
                            _aln_a_sel = ui.select(_ALN_OPTS, value='All', label='Alignment').style('width:190px;')
                            _aln_a_sel.on_value_change(lambda e: (_aln_f.update({'alignment': e.value}), _aln_table.refresh()))
                            _aln_p_sel = ui.select(_PRIO_OPTS, value='All', label='Priority').style('width:110px;')
                            _aln_p_sel.on_value_change(lambda e: (_aln_f.update({'priority': e.value}), _aln_table.refresh()))
                            def _reset_aln():
                                _aln_f.update({'alignment': 'All', 'priority': 'All'})
                                _aln_a_sel.value = 'All'
                                _aln_p_sel.value = 'All'
                                _aln_table.refresh()
                            ui.button('↺ Reset', on_click=_reset_aln).props('flat dense').style(
                                'color:#6366f1;font-size:0.76rem;align-self:flex-end;margin-left:4px;'
                            )
                        _aln_table()
                    else:
                        ui.html(
                            '<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;'
                            'padding:14px 16px;font-size:0.84rem;color:#15803d;margin-top:12px;">'
                            '&#10003; No cases currently need attention.</div>'
                        )

                # ════════════════════════════════════════════════════════════
                # TAB 5: Recurring Issues
                # ════════════════════════════════════════════════════════════
                with ui.tab_panel(t_rec):

                    if top_judge_tags:
                        _sh("By Judge Issue Tag")
                        max_t = top_judge_tags[0]["count"] or 1
                        ui.html(_bar_card(
                            "".join(_bar_row(e["label"], e["count"], max_t, "#f59e0b")
                                    for e in top_judge_tags[:12])
                        ))
                    else:
                        ui.html('<div style="color:#94a3b8;font-size:0.84rem;margin-top:8px;">No judge evaluations yet.</div>')

                    if top_user_chips:
                        _sh("By User Reason Chip")
                        max_c = top_user_chips[0]["count"] or 1
                        ui.html(_bar_card(
                            "".join(_bar_row(e["label"], e["count"], max_c, "#6366f1")
                                    for e in top_user_chips)
                        ))
                    else:
                        ui.html('<div style="color:#94a3b8;font-size:0.84rem;margin-top:8px;">No reason chips recorded yet.</div>')

                    if recurring_issues:
                        _sh("Combined Signal")
                        rows_html = ""
                        for ri in recurring_issues[:20]:
                            sig_col = _QA_SIG_COL.get(ri["signal"], "#475569")
                            ex_str  = " ".join(
                                f'<a href="/cases/{i}" target="_blank" '
                                f'style="color:#6366f1;font-weight:600;text-decoration:none;">#{i}</a>'
                                for i in ri["example_case_ids"][:3]
                            )
                            rows_html += (
                                f'<tr style="border-bottom:1px solid #f1f5f9;">'
                                f'<td {_TD}>{ri["label"]}</td>'
                                f'<td {_TD} style="text-align:center;font-weight:700;">{ri["total"]}</td>'
                                f'<td {_TD} style="text-align:center;color:#92400e;">{ri["judge_count"] or "—"}</td>'
                                f'<td {_TD} style="text-align:center;color:#3730a3;">{ri["user_count"] or "—"}</td>'
                                f'<td {_TD}><span style="color:{sig_col};font-weight:700;'
                                f'font-size:0.7rem;text-transform:uppercase;">{ri["signal"]}</span></td>'
                                f'<td {_TD}>{ex_str or "—"}</td>'
                                f'</tr>'
                            )
                        th_html = "".join(
                            f'<th {_TH}>{h}</th>'
                            for h in ("Issue", "Total", "Judge", "User", "Signal", "Examples")
                        )
                        ui.html(_table_wrap(th_html, rows_html))

                # ════════════════════════════════════════════════════════════
                # TAB 6: Suggestions
                # ════════════════════════════════════════════════════════════
                with ui.tab_panel(t_sug):
                    ui.html(
                        '<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;'
                        'padding:10px 14px;font-size:0.8rem;color:#166534;margin-bottom:14px;">'
                        '&#9432; Suggestions are generated from recurring judge issue tags and user feedback reason chips. '
                        'They are advisory only and do not automatically change the repair pipeline. '
                        'Cards labelled <strong style="background:#6366f1;color:white;padding:1px 6px;'
                        'border-radius:100px;font-size:0.7rem;">JUDGE + USER</strong> mean both signals agree — '
                        'treat these as higher confidence.</div>'
                    )

                    if not suggestions:
                        ui.html(
                            '<div style="color:#94a3b8;font-size:0.84rem;">'
                            'No recurring issues found yet — run the judge on more cases '
                            'and collect more feedback to generate suggestions.</div>'
                        )
                    else:
                        # ── Proposal dialog helpers (defined once, used per card) ──

                        def _pfld(_l: str, _v: str, _b: bool = False) -> str:
                            _vw = (f'<strong style="color:#0f172a;">{_v}</strong>' if _b
                                   else f'<span style="color:#0f172a;">{_v}</span>')
                            return (
                                f'<div style="font-size:0.82rem;">'
                                f'<span style="color:#94a3b8;font-size:0.7rem;font-weight:600;'
                                f'text-transform:uppercase;letter-spacing:0.04em;">{_l}</span><br>'
                                f'{_vw}</div>'
                            )

                        def _prop_loading() -> str:
                            return (
                                '<div style="background:#6366f1;padding:16px 20px;">'
                                '<div style="font-size:0.95rem;font-weight:800;color:white;">'
                                'Advisory Fix Proposal</div>'
                                '<div style="font-size:0.72rem;color:#c7d2fe;margin-top:2px;">'
                                'AI-generated · draft only · not applied automatically</div></div>'
                                '<div style="padding:24px 20px;display:flex;align-items:center;'
                                'gap:10px;color:#64748b;">'
                                '<span style="font-size:1.2rem;">&#9203;</span>'
                                '<span style="font-size:0.84rem;">Generating proposal via Ollama…</span></div>'
                            )

                        def _prop_ai(_data: dict, _sg: dict) -> str:
                            _ex = ", ".join(
                                f"#{i}" for i in (_sg.get("example_case_ids") or [])[:5]
                            ) or "—"
                            return (
                                '<div style="background:#6366f1;padding:16px 20px;">'
                                '<div style="font-size:0.95rem;font-weight:800;color:white;">'
                                'Advisory Fix Proposal</div>'
                                '<div style="font-size:0.72rem;color:#c7d2fe;margin-top:2px;">'
                                'AI-generated · draft only · not applied automatically</div></div>'
                                '<div style="padding:18px 20px;display:flex;flex-direction:column;gap:12px;">'
                                + _pfld("Proposal title",   _data.get("title",            "—"), True)
                                + _pfld("Problem observed", _data.get("problem_observed",  "—"))
                                + _pfld("Evidence",         _data.get("evidence",          "—"))
                                + _pfld("Suggested change", _data.get("suggested_change",  "—"))
                                + _pfld("Pipeline area",    _data.get("pipeline_area",     _sg.get("pipeline_area", "—")))
                                + _pfld("Expected impact",  _data.get("expected_impact",   "—"))
                                + _pfld("Risks / cautions", _data.get("risks_or_cautions", "—"))
                                + _pfld("Example cases",    _ex)
                                + '<div style="background:#fef9c3;border-radius:8px;padding:10px 12px;'
                                  'font-size:0.76rem;color:#92400e;">&#9888; '
                                + _data.get("safety_note",
                                            "This is an advisory QA proposal only. "
                                            "It does not automatically change the repair pipeline.")
                                + '</div></div>'
                            )

                        def _prop_static(_sg: dict, _err: dict | None = None) -> str:
                            _tr = " · ".join(
                                _QA_TAG_LABELS.get(k) or _QA_CHIP_LABELS.get(k)
                                or k.replace("_", " ").title()
                                for k in (_sg.get("trigger_issues") or [])
                            )
                            _ex = ", ".join(
                                f"#{i}" for i in (_sg.get("example_case_ids") or [])[:5]
                            ) or "—"
                            _eb = (
                                f'<div style="background:#fef2f2;border-radius:8px;padding:8px 12px;'
                                f'font-size:0.74rem;color:#991b1b;">'
                                f'&#9888; {_err.get("message", "AI generation unavailable")} '
                                f'— static summary shown.</div>'
                                if _err else ""
                            )
                            return (
                                '<div style="background:#64748b;padding:16px 20px;">'
                                '<div style="font-size:0.95rem;font-weight:800;color:white;">'
                                'Fix Proposal (static)</div>'
                                '<div style="font-size:0.72rem;color:#e2e8f0;margin-top:2px;">'
                                'Draft only · not applied automatically</div></div>'
                                '<div style="padding:18px 20px;display:flex;flex-direction:column;gap:10px;">'
                                + _eb
                                + _pfld("Suggestion",    _sg.get("suggestion",   "—"), True)
                                + _pfld("Pipeline area", _sg.get("pipeline_area", "—"))
                                + _pfld("Triggered by",  _tr)
                                + _pfld("Signal",        (_sg.get("signal") or "").upper())
                                + _pfld("Occurrences",   str(_sg.get("total_occurrences", 0)))
                                + _pfld("Example cases", _ex)
                                + '<div style="background:#fef9c3;border-radius:8px;padding:10px 12px;'
                                  'font-size:0.76rem;color:#92400e;">'
                                  '&#9888; This is an advisory QA proposal only. '
                                  'It does not automatically change the repair pipeline.</div>'
                                  '</div>'
                            )

                        def _make_proposal(_sg: dict) -> None:
                            with ui.dialog() as _dlg:
                                with ui.card().style(
                                    "max-width:600px;width:100%;border-radius:14px;padding:0;"
                                    "overflow:hidden;max-height:85vh;"
                                    "display:flex;flex-direction:column;"
                                ):
                                    _ph = ui.html("").style("overflow-y:auto;flex:1;")
                                    with ui.row().style(
                                        "justify-content:flex-end;padding:0 16px 12px;flex-shrink:0;"
                                    ):
                                        ui.button("Close", on_click=_dlg.close).props(
                                            "flat dense"
                                        ).style("color:#6366f1;")

                            async def _open():
                                _ph.content = _prop_loading()
                                _dlg.open()
                                try:
                                    async with httpx.AsyncClient() as _hc:
                                        _resp = await _hc.post(
                                            f"{API}/admin/quality/fix-proposal",
                                            headers=H,
                                            json={
                                                "suggestion":        _sg["suggestion"],
                                                "signal":            _sg.get("signal", ""),
                                                "trigger_issues":    _sg.get("trigger_issues") or [],
                                                "total_occurrences": _sg.get("total_occurrences", 0),
                                                "pipeline_area":     _sg.get("pipeline_area", ""),
                                                "example_case_ids":  (_sg.get("example_case_ids") or [])[:5],
                                            },
                                            timeout=90.0,
                                        )
                                    _d = _resp.json() if _resp.status_code == 200 else None
                                except Exception as _exc:
                                    _d = {"error": "connection_failed", "message": str(_exc)[:120]}
                                if _d and _d.get("ai_generated"):
                                    _ph.content = _prop_ai(_d, _sg)
                                else:
                                    _ph.content = _prop_static(_sg, _d)

                            with ui.row().style("justify-content:flex-end;padding-top:6px;"):
                                ui.button(
                                    "Generate fix proposal →", on_click=_open
                                ).props("flat dense").style(
                                    "color:#6366f1;font-size:0.76rem;font-weight:600;"
                                )

                        for sg in suggestions:
                            _sig = sg["signal"]
                            sig_border = {"both": "#6366f1", "judge": "#f59e0b", "user": "#3b82f6"}.get(_sig, "#e2e8f0")
                            sig_bg     = "#f8f7ff" if _sig == "both" else "white"
                            sig_badge_text = {"both": "JUDGE + USER", "judge": "JUDGE", "user": "USER FEEDBACK"}.get(_sig, _sig.upper())
                            sig_badge_bg   = {"both": "#6366f1", "judge": "#d97706", "user": "#3b82f6"}.get(_sig, "#94a3b8")
                            ex_str = "  ".join(
                                f'<a href="/cases/{i}" target="_blank" '
                                f'style="color:#6366f1;font-weight:600;text-decoration:none;">#{i}</a>'
                                for i in sg["example_case_ids"][:5]
                            )
                            trigger_str = " · ".join(
                                _QA_TAG_LABELS.get(k) or _QA_CHIP_LABELS.get(k)
                                or k.replace("_", " ").title()
                                for k in sg["trigger_issues"]
                            )
                            with ui.element("div").style(
                                f"background:{sig_bg};border:1px solid #e2e8f0;"
                                f"border-left:4px solid {sig_border};"
                                "border-radius:12px;padding:14px 16px;margin-bottom:10px;"
                            ):
                                ui.html(
                                    f'<div style="display:flex;justify-content:space-between;'
                                    f'align-items:flex-start;gap:12px;">'
                                    f'<div style="flex:1;min-width:0;">'
                                    f'<div style="margin-bottom:6px;">'
                                    f'<span style="background:{sig_badge_bg};color:white;font-size:0.65rem;'
                                    f'font-weight:800;padding:2px 8px;border-radius:100px;'
                                    f'letter-spacing:0.04em;white-space:nowrap;">{sig_badge_text}</span>'
                                    f'</div>'
                                    f'<div style="font-weight:700;color:#0f172a;font-size:0.88rem;'
                                    f'margin-bottom:5px;">{sg["suggestion"]}</div>'
                                    f'<div style="font-size:0.78rem;color:#475569;margin-bottom:4px;">'
                                    f'<strong>Pipeline area:</strong> {sg["pipeline_area"]}</div>'
                                    f'<div style="font-size:0.73rem;color:#64748b;">'
                                    f'Triggered by: {trigger_str}</div>'
                                    + (
                                        f'<div style="font-size:0.73rem;color:#64748b;margin-top:2px;">'
                                        f'Example cases: {ex_str}</div>'
                                        if ex_str else ""
                                    )
                                    + f'</div>'
                                    f'<div style="text-align:right;flex-shrink:0;min-width:56px;">'
                                    f'<div style="font-size:1.4rem;font-weight:800;color:#0f172a;">'
                                    f'{sg["total_occurrences"]}</div>'
                                    f'<div style="font-size:0.68rem;color:#94a3b8;">occurrences</div>'
                                    f'</div></div>'
                                )
                                _make_proposal(sg)


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

ui.run(
    port=3000,
    title="Knitra",
    reload=False,
    storage_secret="fixly-dev-secret",
)
