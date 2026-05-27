# Knitra : Multimodal AI Repair Assessment Platform

Knitra is a locally-hosted, privacy-first AI repair assistant that helps homeowners and tradespeople diagnose household problems and plan repairs from a photo. Upload an image of the fault, optionally describe the problem by typing or speaking, and receive a fully structured repair assessment: problem classification, confidence score, visual evidence reasoning, essential and optional tools and materials, safety-graded step-by-step repair instructions, and a contextual AI chat assistant that already knows your case.

All inference runs locally via [Ollama](https://ollama.com). No images, audio, or repair data leave your machine.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Features](#2-key-features)
3. [Architecture Overview](#3-architecture-overview)
4. [AI Pipeline](#4-ai-pipeline)
5. [Prompt Engineering Strategy](#5-prompt-engineering-strategy)
6. [Explainability and Confidence System](#6-explainability-and-confidence-system)
7. [Ask Knitra — Contextual Chat Assistant](#7-ask-knitra--contextual-chat-assistant)
8. [Tech Stack](#8-tech-stack)
9. [Project Structure](#9-project-structure)
10. [Installation and Setup](#10-installation-and-setup)
11. [API Reference](#11-api-reference)
12. [Runtime Verification](#12-runtime-verification)
13. [Current Limitations](#13-current-limitations)
14. [Future Roadmap](#14-future-roadmap)

---

## 1. Project Overview

### What Problem Knitra Solves

Most people who encounter a household fault — a cracked tile, a leaking pipe fitting, a broken hinge, a damp patch — have three poor options: pay a tradesperson for a problem they could have fixed themselves, attempt a repair without proper guidance and make it worse, or spend hours searching fragmented forum posts for advice that may not apply to their situation.

Knitra provides a fourth option: upload a photo, get a structured AI-generated repair plan within minutes, and then ask follow-up questions in a case-aware conversation that already understands exactly what the problem is and what tools are involved.

### What Knitra Is

Knitra is a **multimodal AI repair assessment platform** with three capabilities:

**Structured Repair Analysis** — a three-stage local AI pipeline (vision → planning → generation) that examines uploaded photos, classifies the fault by trade category and severity, and produces a complete repair plan with tools, materials, ordered steps, safety notes, and a difficulty rating.

**Confidence-Aware Analysis with Explainability** — every assessment includes a confidence score (0–100), a list of what the model visually confirmed, a list of what it could not confirm, and — when confidence is too low to diagnose safely — an explicit clarification gate that halts the pipeline and explains to the user precisely why and what additional information is needed.

**Ask Knitra** — a case-anchored conversational AI assistant that opens after analysis is complete. The assistant has full context of the case: the problem type, trade category, severity, confidence score, visual evidence, tools, materials, and every repair step. Users can ask follow-up questions, attach additional photos for re-analysis, and use voice input via browser microphone.

---

## 2. Key Features

### Analysis and Diagnosis

| Feature | Description |
|---|---|
| **Image-based repair analysis** | Upload one or more photos; the vision model identifies the fault, trade category, and severity |
| **Multi-stage AI pipeline** | Three locally-running models handle vision, planning, and step generation independently |
| **Confidence scoring** | Every analysis returns a 0–100 confidence score based on image clarity and diagnostic certainty |
| **Clarification gate** | If confidence falls below the safety threshold, the pipeline halts and asks for better photos or a description instead of producing a low-quality plan |
| **Explainability** | The UI shows what the model visually confirmed, what it could not confirm, and why confidence was at its reported level |
| **Uncertainty reasoning** | Unconfirmed details (e.g. crack depth, pipe material, water source) are explicitly surfaced rather than silently assumed |
| **Needs-clarification UI** | A dedicated card explains the confidence shortfall, shows auto-detected photo tips based on the specific uncertainty, and offers a re-analyse path |
| **Safety and risk guidance** | High-severity cases trigger explicit professional-referral recommendations. Gas, structural, electrical-mains, and significant water-damage repairs always escalate to a qualified-tradesperson recommendation |
| **Trade category normalisation** | 10-category controlled vocabulary (`plumbing`, `electrical`, `appliance`, `carpentry/joinery`, `glazing`, `sealing/weatherproofing`, `plastering/decorating`, `structural`, `hvac`, `general/other`) normalised from 20+ raw model output variants via Pydantic validator |

### Case Management and UI

| Feature | Description |
|---|---|
| **My Cases dashboard** | Filterable list of all repair cases with confidence badge, category, status, and date |
| **Filter bar** | Filter by trade category, status, confidence tier, and date sort |
| **Case status mirroring** | Case status (`open → running → complete / needs_clarification / failed`) is mirrored from the pipeline to the `sessions` table so filters work accurately |
| **Drag-and-drop upload** | Multi-photo selection with instant thumbnails, client-side Canvas compression before upload |
| **Photo grid and lightbox** | Full-screen image viewer with keyboard navigation |
| **Case auto-titling** | Case title is updated automatically from Stage 1 `problem_type` output once diagnosis is complete |

### Ask Knitra Chat

| Feature | Description |
|---|---|
| **Contextual conversation** | Every chat turn has the full repair case context in the system prompt — tools, steps, severity, visual evidence, uncertainty notes |
| **Follow-up reasoning** | The assistant can answer questions like "what if I don't have a pipe wrench?" or "is this safe to do at night?" with specific reference to the repair plan |
| **Microphone transcription** | Browser `MediaRecorder` records audio, `faster-whisper` transcribes server-side; text is injected into the chat textarea with Python-side DOM fallback for reliable sync |
| **Follow-up image uploads** | Attach additional photos mid-conversation; the vision model re-analyses them and their description is appended to Knitra's system prompt for that turn |
| **Needs-clarification chat** | When analysis halted due to low confidence, Knitra uses a specialised system prompt that explains the confidence shortfall and guides the user toward a better submission |
| **Repair-scope enforcement** | Off-topic messages are politely redirected back to the repair case |

---

## 3. Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                     Browser / NiceGUI SPA                    │
│  (port 3000 — Python-rendered, WebSocket-synced UI)          │
└────────────────────────┬─────────────────────────────────────┘
                         │  HTTP / REST
┌────────────────────────▼─────────────────────────────────────┐
│                  FastAPI Backend  (port 8000)                 │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ Auth / Cases│  │ Uploads /    │  │  Analysis Service  │  │
│  │ REST routes │  │ Transcription│  │  (background task) │  │
│  └─────────────┘  └──────────────┘  └────────┬───────────┘  │
│                                              │               │
│  ┌───────────────────────────────────────────▼───────────┐  │
│  │                  Three-Stage AI Pipeline               │  │
│  │                                                        │  │
│  │  Stage 1 ──► Stage 2 ──► Stage 3                      │  │
│  │  Vision      Planning    Generation                    │  │
│  │  Qwen2.5-VL  Mistral 7b  Llama 3.1:8b                 │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Chat Service — per-case contextual assistant        │   │
│  │  Llama 3.1:8b + Qwen2.5-VL (follow-up images)       │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└──────────────────────────┬───────────────────────────────────┘
                           │  asyncpg
┌──────────────────────────▼───────────────────────────────────┐
│            PostgreSQL 16  (Docker, port 5433)                │
│  users · user_tokens · sessions · uploads · analyses ·       │
│  messages                                                    │
└──────────────────────────────────────────────────────────────┘
                           │  HTTP
┌──────────────────────────▼───────────────────────────────────┐
│               Ollama  (port 11434, local inference)          │
│  qwen2.5vl:3b  ·  mistral:7b-instruct  ·  llama3.1:8b       │
└──────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

**FastAPI backend** (`backend/main.py`) — async REST API serving auth, cases, uploads, analysis triggering, chat, and audio transcription. CORS-enabled for the NiceGUI frontend origin. Uploads served as static files.

**NiceGUI frontend** (`frontend/main.py`) — single-file Python SPA. Renders all pages (landing, register, login, cases list, new case composer, case detail). Communicates with the backend via `httpx` async HTTP. JavaScript is injected selectively for DOM interactions the Python event loop cannot handle (file picker, `MediaRecorder`, canvas compression).

**Analysis service** — launched as `asyncio.create_task` so the HTTP response returns immediately while the pipeline runs in the background with its own DB session. Writes `pipeline_stage` after each step so the frontend can show granular progress via polling. Mirrors status to both `analyses.status` and `sessions.status`.

**Chat service** — stateless per-call service. Builds a case-aware system prompt from the latest complete analysis result, loads the full message history, calls `CHAT_MODEL`, and persists both the user and assistant turns.

**Ollama** — all model inference is local. No cloud API keys. Models communicate via the Ollama HTTP API on `localhost:11434`.

**faster-whisper** — CPU-based speech-to-text running inside the FastAPI process. `tiny` model (~75 MB). Transcription is performed server-side to avoid exposing audio to third-party APIs.

**Stage 4 — Product Recommendation Module (planned)** — a future fourth pipeline stage will match required tools and materials to a product database, generating recommendation cards with brand, size, UK retailer links, and estimated cost. The `products` table is reserved in the roadmap schema.

---

## 4. AI Pipeline

### Overview

The analysis pipeline runs three sequential local model calls, each with a specific responsibility. The output of each stage informs the next. All stages are retried up to `PIPELINE_RETRY_COUNT` times on validation failure, with JSON healing attempted before each retry.

```
Upload photos + text context
        │
        ▼
┌───────────────────────────┐
│  Stage 1 — Vision         │  Qwen2.5-VL:3b
│  Classifies fault from    │  num_ctx: 4096
│  image(s) + description   │  Images: resized to 512px max
└──────────────┬────────────┘  Output: Stage1Output (JSON)
               │
               ▼  Confidence gate
          ┌────┴────────────────────┐
          │ confidence < threshold? │
          └────┬────────────────────┘
               │ yes → needs_clarification
               │ no  ↓
┌──────────────▼────────────┐
│  Stage 2 — Planning       │  Mistral:7b-instruct
│  Lists tools, materials,  │  num_ctx: 2048
│  gear, estimated time     │
└──────────────┬────────────┘  Output: Stage2Output (JSON)
               │
┌──────────────▼────────────┐
│  Stage 3 — Generation     │  Llama3.1:8b
│  Step-by-step repair plan │  num_ctx: 2048
│  safety notes, difficulty │
│  when to call a pro       │
└──────────────┬────────────┘  Output: Stage3Output (JSON)
               │
               ▼
       Analysis complete
   structured_result written
   session.status → "complete"
```

### Stage 1 : Vision and Issue Detection

Qwen2.5-VL examines the uploaded images alongside any text description provided by the user. It classifies the problem type, assigns a trade category from the controlled 10-category vocabulary, assesses severity (`low` / `medium` / `high`), records visual evidence it can confirm, records what it cannot confirm, produces a confidence score (0–100), and — if confidence is below threshold — generates a specific clarification request explaining what additional information would resolve the uncertainty.

**Graceful fallback**: if image analysis fails all retries (e.g. corrupted upload, unsupported format), Stage 1 falls back to text-only analysis using the user's description. The fallback is recorded in `stage_outputs.stage_1_fallback` and surfaced in the UI.

**Image preprocessing**: images are resized to a configurable max dimension (default 512 px) with aspect ratio preserved and re-encoded as JPEG before base64 encoding. This reduces the token count sent to the vision model by 60–80% versus full-resolution phone photos, with no meaningful loss in diagnostic accuracy for macro-scale home repair faults.

### Stage 2 : Repair Planning

Mistral:7b-instruct receives the Stage 1 output and generates a structured list of:
- Essential tools (must-have)
- Optional tools ("nice to have")
- Essential materials
- Optional materials
- Protective gear
- Estimated time in minutes

Essential and optional items are stored and displayed separately in the UI.

### Stage 3 : Step-by-Step Generation

Llama3.1:8b receives Stage 1 and Stage 2 outputs and generates:
- Numbered repair steps (each with an optional warning note)
- Difficulty level
- Safety notes
- Explicit guidance on when to call a professional

### Confidence Gate

After Stage 1, the pipeline checks the returned confidence score. If it falls below the configured threshold, the analysis status is set to `needs_clarification` and the pipeline halts. The `stage_outputs.stage_1` field is preserved so the frontend and chat service can explain precisely what the model observed, what it could not determine, and what would help.

### Safety Escalation

Regardless of confidence, cases classified as `severity: high` trigger an explicit professional recommendation in both the analysis result display and the Knitra system prompt. Cases involving electrical mains, gas, structural load-bearing elements, or significant water damage always include a UK-qualified tradesperson referral (Gas Safe, Part P, Building Regulations as applicable).

---

## 5. Prompt Engineering Strategy

Knitra's pipeline uses distinct prompt strategies at each stage, tuned for the specific cognitive task that stage performs.

### Staged Prompting

Each stage receives only the context it needs:
- Stage 1 receives the raw images and user description — no prior AI output
- Stage 2 receives the Stage 1 JSON output (problem type, observations, severity) — no images
- Stage 3 receives Stage 1 and Stage 2 outputs — no images, no raw description

This prevents context bleed between stages and keeps each model focused on its specific task.

### Structured JSON Generation

Stages 1, 2, and 3 all use `format="json"` in the Ollama call and receive explicit Pydantic-shaped JSON schema examples in the system prompt. This constrains output format and dramatically reduces the rate of malformed or hallucinated response structures. A JSON healing function (`_heal_truncated_json`) attempts bracket/brace repair on truncated output before triggering a retry, reducing retry frequency on low-context machines.

### Hallucination Reduction

- Stage 1 is instructed to report only what is visually confirmed in the image, and to explicitly populate `visual_evidence`, `uncertainty_notes`, and `confidence_reason` separately — preventing the model from blending confirmed observations with inferences
- Stage 2 is instructed to list only tools directly applicable to the Stage 1 problem type
- Stage 3 is instructed to reference the specific materials and tools from Stage 2 by name rather than generating new ones
- Confidence scores below threshold halt generation entirely, preventing a low-quality plan from being presented as authoritative

### Confidence-Aware Prompting

The Stage 1 prompt explicitly instructs the model to report a conservative confidence score and to populate `clarification_request` with a specific, actionable message whenever the image is ambiguous, blurry, too distant, poorly lit, or shows multiple simultaneous issues. This separates diagnosis confidence from response confidence, giving the clarification gate a reliable signal.

### Contextual Prompting — Knitra Chat

Knitra's system prompt is dynamically built per-case at chat time from the stored analysis result. It includes:
- The full `Stage1Output` (problem, severity, trade, confidence, visual evidence, uncertainty notes)
- The full `Stage2Output` (essential and optional tools, materials, time estimate)
- The full `Stage3Output` (all steps, safety notes, when-to-call-a-pro)
- The user's original question or description (if provided at case creation)
- A repair-scope constraint that politely redirects off-topic questions

When follow-up images are attached, a separate lightweight Qwen2.5-VL pass is run on those images. The vision output is appended to the system prompt as contextual observations before the chat model responds.

### Clarification Prompting

When a case has `status: needs_clarification`, Knitra's system prompt is built from the `stage_outputs.stage_1` data rather than a `structured_result`. It instructs Knitra to:
1. Explain clearly why confidence was low, referencing the specific `confidence_reason` field
2. Tell the user exactly what photo or information would resolve the uncertainty
3. Offer provisional guidance if the user describes the fault in text
4. Suggest whether a closer, wider, brighter, or differently-angled photo would help most

### Repair Workflow Prompting

The Stage 3 prompt instructs the model to generate steps that are:
- Sequenced in the correct dependency order (e.g. turn off water before disconnecting a joint)
- Granular enough that a non-expert can follow them without external reference
- Each annotated with a `warning` field for any step that carries risk
- Accompanied by safety notes that address the specific trade category of the fault

---

## 6. Explainability and Confidence System

### Confidence Tiers

| Score | Tier | Behaviour |
|---|---|---|
| 80–100 | High | Pipeline proceeds to Stage 2; confidence badge shown in green |
| 60–79 | Medium | Pipeline proceeds; amber badge; uncertainty notes surfaced in UI |
| 0–59 | Low | Confidence gate triggers; pipeline halts; `needs_clarification` state |

The threshold between proceeding and halting is configurable. The default is set to require a minimum of 60% confidence for Stage 2 to run.

### Visual Evidence

Stage 1 populates `visual_evidence` with a list of specific things the model could confirm from the image (e.g. "visible crack in grout line running diagonally across two tiles", "water staining on wall surface below window frame"). These are displayed as evidence chips in the case detail UI.

### Uncertainty Reasoning

Stage 1 separately populates `uncertainty_notes` with things it could not confirm (e.g. "cannot determine depth of crack or whether substrate is affected", "unable to confirm pipe material from image angle"). These are displayed in an amber panel labelled "Could not confirm" in both the analysis result and the needs-clarification card.

### Clarification Requests

When the confidence gate triggers, `clarification_request` contains a specific, actionable message generated by Stage 1. This is displayed prominently in the needs-clarification UI card. The frontend also applies automatic photo-tip heuristics based on the `uncertainty_notes` content — detecting phrases like "blurry", "too far", "dark" and surfacing the relevant tip (move closer, improve lighting, etc.).

### Explainable AI Behaviour

The needs-clarification card shows:
- A large confidence percentage circle (colour-coded by tier)
- The confidence reason in plain English
- A "What the AI could see" chip list from `visual_evidence`
- A "Could not confirm" amber panel from `uncertainty_notes`
- The specific clarification request
- Auto-detected photo improvement tips
- A re-analyse textarea and button for the user's follow-up submission

The confidence score is also shown on case cards in the My Cases list, allowing users to see at a glance which of their cases have high-confidence versus uncertain diagnoses.

---

## 7. Ask Knitra : Contextual Chat Assistant

### How It Works

Ask Knitra is locked until at least one `complete` or `needs_clarification` analysis exists for the case. This is enforced at both the API level (the `send_message` service checks for an analysis before accepting a chat message) and the UI level (the chat section renders a locked placeholder until the pipeline finishes).

When the pipeline completes, the chat section unlocks in-place — no page reload. The full message history is loaded from the `messages` table so the conversation context is preserved across sessions.

### Contextual Awareness

Every chat call dynamically builds a system prompt from the database. Knitra is never a generic assistant — it always enters the conversation knowing:

- The exact problem type and trade category of this case
- The confidence level and what the model could and could not see
- All tools and materials the repair plan calls for
- Every step in the repair plan, in order
- All safety notes and professional-referral conditions

When the user asks "what if I don't have a pipe wrench?", Knitra can answer with reference to the specific tools in this case's Stage 2 output. When the user asks "is step 3 dangerous?", Knitra can look up that step's warning field.

### Microphone Transcription

The chat toolbar includes a microphone button that:
1. Requests `getUserMedia({ audio: true })` and starts a `MediaRecorder` session
2. Displays a live recording timer
3. On stop, sends the audio blob to `POST /api/v1/transcribe` as multipart form data
4. faster-whisper transcribes server-side and returns the text
5. The text is stored in `window._knitraTranscribedText` and injected into the chat textarea
6. Python's `_send_chat()` reads the DOM value directly (with a `ui.run_javascript` fallback) to ensure transcribed text is captured even if Vue/Quasar's WebSocket value sync has not yet fired

### Follow-Up Image Uploads

The paperclip button in the chat toolbar opens a file picker (`accept="image/*,video/*"`). On selection:
1. Each file is uploaded to `POST /api/v1/cases/{id}/uploads` and the returned upload `id` is stored in `window._knitraPendingImageIds`
2. Thumbnail previews appear immediately below the toolbar
3. The toolbar status shows "N files ready — press Send to ask Knitra ✔"
4. When Send is pressed, `_send_chat()` reads the pending ID list, includes it as `image_ids` in the POST body, and clears the queue
5. The backend loads the uploads by ID (case-scoped to prevent cross-case access), runs a lightweight Qwen2.5-VL vision pass on each image, and appends the natural-language description to Knitra's system prompt for that turn
6. The chat thread shows an italic attachment-notice bubble ("📷 You attached N photos") before the user's message

### Message Persistence

All chat messages — both user turns and assistant replies — are stored in the `messages` table with `role`, `content`, `message_type`, and `created_at`. The full history is loaded on every call so the model has complete multi-turn context.

---

## 8. Tech Stack

| Component | Technology | Version |
|---|---|---|
| Backend framework | FastAPI | Latest stable |
| Backend runtime | Python | 3.12 |
| ORM | SQLAlchemy (async) | 2.0 |
| DB driver | asyncpg | Latest |
| Database | PostgreSQL | 16 (Docker) |
| Migrations | Alembic | Latest |
| Frontend | NiceGUI | 2.7 |
| HTTP client | httpx | Latest |
| Schema validation | Pydantic | v2 |
| Vision model | Qwen2.5-VL:3b | via Ollama |
| Planning model | Mistral:7b-instruct | via Ollama |
| Generation + chat model | Llama3.1:8b | via Ollama |
| Inference runtime | Ollama | Latest |
| Speech-to-text | faster-whisper (tiny, CPU int8) | Latest |
| Image processing | Pillow | Latest |
| Auth | Bearer token + bcrypt | DB-stored tokens |
| Containerisation | Docker Compose | v2 |

---

## 9. Project Structure

```
FixIT/
├── docker-compose.yml              # PostgreSQL service definition
├── pyrightconfig.json              # Pyright venv path override
├── backend/
│   ├── main.py                     # FastAPI app entry point, CORS, static mount
│   ├── requirements.txt
│   ├── pyrightconfig.json          # Backend-scoped type checker config
│   ├── .env                        # Environment variables (gitignored)
│   ├── uploads/                    # Uploaded images on disk (gitignored)
│   ├── alembic/
│   │   └── versions/               # DB migration scripts
│   └── app/
│       ├── models.py               # SQLAlchemy ORM (all tables + Session.confidence property)
│       ├── database.py             # Async engine, session factory, expire_on_commit=False
│       ├── router.py               # Central route assembly
│       ├── api/
│       │   ├── auth.py             # POST /register /login /logout, GET /me
│       │   ├── cases.py            # CRUD repair cases + PATCH (title/category/status)
│       │   ├── uploads.py          # POST/GET case uploads
│       │   ├── analysis.py         # POST trigger + GET poll result
│       │   ├── chat.py             # GET history + POST message (with image_ids)
│       │   ├── transcribe.py       # POST audio → faster-whisper → text
│       │   └── health.py           # GET /health liveness
│       ├── core/
│       │   └── deps.py             # get_current_user FastAPI dependency
│       ├── schemas/
│       │   ├── auth.py             # Register/Login request + response schemas
│       │   ├── cases.py            # CaseResponse (includes confidence: Optional[int])
│       │   ├── uploads.py          # UploadResponse
│       │   ├── analysis.py         # Stage1/2/3Output + AnalysisResult + trade normaliser
│       │   └── chat.py             # MessageRequest (content + image_ids) + MessageResponse
│       └── services/
│           ├── auth_service.py     # Registration, login, token management
│           ├── cases_service.py    # CRUD + selectinload for confidence property
│           ├── uploads_service.py  # File validation, UUID storage, Pillow dimensions
│           ├── analysis_service.py # 3-stage Ollama pipeline + confidence gate
│           └── chat_service.py     # System prompt builder + follow-up image vision
└── frontend/
    └── main.py                     # NiceGUI SPA (all pages, CSS, JS in one file)
```

---

## 10. Installation and Setup

### Prerequisites

- Python 3.12+
- Docker Desktop (for PostgreSQL)
- [Ollama](https://ollama.com) installed and running locally
- Node.js is **not** required — NiceGUI ships its own static assets

---

### Step 1 : Start PostgreSQL

```bash
docker compose up -d db
```

PostgreSQL will start on port `5433` (non-default to avoid conflicts with any local PostgreSQL installation).

---

### Step 2 : Pull AI Models

One-time download, approximately 10 GB total. All models run fully locally after this step.

```bash
ollama pull qwen2.5vl:3b          # Stage 1 — vision (approx. 2.3 GB)
ollama pull mistral:7b-instruct   # Stage 2 — planning (approx. 4.1 GB)
ollama pull llama3.1:8b           # Stage 3 + chat (approx. 4.7 GB)
```

Verify Ollama is running:

```bash
curl http://localhost:11434/api/tags
```

---

### Step 3 : Backend Setup

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
alembic upgrade head
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

### Step 4 : Frontend Setup

Open a second terminal:

```bash
cd frontend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python main.py
```

---

### Environment Variables

Create `backend/.env` with the following values. All variables have sensible defaults in code; only `DATABASE_URL` is required if you change the Docker Compose settings.

```env
# Database
DATABASE_URL=postgresql+asyncpg://fixly_user:devpassword@localhost:5433/fixly

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_KEEP_ALIVE=5m

# Model assignments
STAGE_1_MODEL=qwen2.5vl:3b
STAGE_2_MODEL=mistral:7b-instruct
STAGE_3_MODEL=llama3.1:8b
CHAT_MODEL=llama3.1:8b

# Pipeline tuning
PIPELINE_RETRY_COUNT=1

# Context windows (critical — do not reduce Stage 1 below 4096)
STAGE_1_NUM_CTX=4096
STAGE_1_NUM_PREDICT=400
STAGE_1_IMAGE_MAX_DIM=512

STAGE_2_NUM_CTX=2048
STAGE_2_NUM_PREDICT=512

STAGE_3_NUM_CTX=2048
STAGE_3_NUM_PREDICT=1024

# Chat
CHAT_NUM_CTX=2048
CHAT_NUM_PREDICT=768
```

> **Why `STAGE_1_NUM_CTX=4096`?** Qwen2.5-VL encodes each 512 px image as approximately 300–700 internal tokens. With a system prompt (~200 tokens) and JSON response (~300 tokens), the default Ollama context window of 512 causes mid-JSON truncation on every Stage 1 call. 4096 gives comfortable headroom for up to three images with retries.

---

### Service URLs

| Service | URL |
|---|---|
| Knitra frontend | http://localhost:3000 |
| FastAPI backend | http://localhost:8000 |
| FastAPI Swagger UI | http://localhost:8000/docs |
| Ollama API | http://localhost:11434 |
| PostgreSQL | localhost:5433 |

---

## 11. API Reference

All routes under `/api/v1/` require `Authorization: Bearer <token>` except where noted.

### Authentication

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/api/v1/auth/register` | — | Create account, return token |
| `POST` | `/api/v1/auth/login` | — | Log in, return token |
| `GET` | `/api/v1/auth/me` | ✓ | Return current user |
| `POST` | `/api/v1/auth/logout` | ✓ | Invalidate token server-side |

### Cases

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/api/v1/cases` | ✓ | Open a new repair case |
| `GET` | `/api/v1/cases` | ✓ | List all cases (newest first, with confidence) |
| `GET` | `/api/v1/cases/{id}` | ✓ | Get single case |
| `PATCH` | `/api/v1/cases/{id}` | ✓ | Update title, category, or status |
| `DELETE` | `/api/v1/cases/{id}` | ✓ | Delete case + all child rows + disk files |

### Uploads

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/api/v1/cases/{id}/uploads` | ✓ | Upload image/video to a case |
| `GET` | `/api/v1/cases/{id}/uploads` | ✓ | List uploads for a case |

### Analysis

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/api/v1/cases/{id}/analyse` | ✓ | Trigger pipeline (returns 202 instantly) |
| `GET` | `/api/v1/cases/{id}/analysis` | ✓ | Poll for status, stage, and result |

### Chat

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/api/v1/cases/{id}/chat` | ✓ | Fetch full message history |
| `POST` | `/api/v1/cases/{id}/chat` | ✓ | Send message (with optional `image_ids`) |

**Chat request body:**
```json
{
  "content": "What tools do I need for step 3?",
  "image_ids": [42, 43]
}
```
`image_ids` is optional (`[]` by default). When provided, the backend loads those uploads, runs a lightweight Qwen2.5-VL description pass, and prepends the vision context to Knitra's system prompt for that turn.

### Transcription

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/api/v1/transcribe` | ✓ | Upload audio blob, return `{ "text": "..." }` |

### Health

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | — | Liveness probe |

---

## 12. Runtime Verification

### Backend

```bash
curl http://localhost:8000/docs
# Expected: 200 OK — Swagger UI HTML

curl http://localhost:8000/health
# Expected: 200 OK
```

### Authentication Flow

```bash
curl -s -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","name":"Test","password":"Pass1234!"}' | jq .

# Expected: { "token": "...", "user": { ... } }
```

### Analysis Pipeline

1. Register and log in to obtain a token
2. `POST /api/v1/cases` — create a case
3. `POST /api/v1/cases/{id}/uploads` — attach one or more photos
4. `POST /api/v1/cases/{id}/analyse` — trigger the pipeline (returns 202 immediately)
5. Poll `GET /api/v1/cases/{id}/analysis` every few seconds
6. Watch `pipeline_stage` cycle through `stage_1_running → stage_1_complete → stage_2_running → stage_2_complete → stage_3_running → complete`
7. Verify `status == "complete"` and `structured_result` is populated

### Confidence Gate

To test the clarification path, upload a blurry or low-information photo (e.g. a plain white wall). The pipeline should return `status: needs_clarification` with `stage_outputs.stage_1.confidence` below the threshold, a populated `confidence_reason`, and a `clarification_request`.

### Chat with Image Attachments

1. Complete a case analysis so `status == "complete"`
2. Open the case in the UI and navigate to Ask Knitra
3. Click the paperclip icon, select an image
4. Verify the upload ID appears in `window._knitraPendingImageIds` (browser console)
5. Type a question and press Send
6. Verify the POST body includes `image_ids: [<id>]`
7. Verify Knitra's response references the attached image content

### Microphone Transcription

1. Click the microphone icon in the Ask Knitra toolbar
2. Speak for a few seconds, then click again to stop
3. The status line should show "Transcribing…" then "✔ Transcribed — press Send"
4. Verify the text appears in the textarea
5. Press Send and verify `_send_chat()` reads the content correctly

---

## 13. Current Limitations

**Video analysis** — video files can be uploaded to a case (the file picker accepts `video/*`) but the analysis pipeline processes still images only. Video frames are not currently extracted. Frame extraction and multi-frame analysis are in the development roadmap.

**Model accuracy** — Qwen2.5-VL:3b is a 3-billion parameter model running on CPU. It performs well on clear, well-lit photos of common domestic repair faults but may misclassify unusual fault types, unusual building materials, or faults that require trade-specific knowledge to identify visually. The confidence gate and uncertainty notes system is designed specifically to surface and handle these cases gracefully.

**Context window constraints** — on machines with limited RAM, increasing `num_ctx` values may cause Ollama to fail or produce degraded output. The defaults are tuned for 16 GB RAM. On lower-spec machines, reducing `STAGE_1_IMAGE_MAX_DIM` to 384 and `STAGE_1_NUM_PREDICT` to 300 can help.

**Pipeline speed** — on CPU-only inference, the full three-stage pipeline takes approximately 97–120 seconds. GPU acceleration via Ollama's CUDA or Metal backends reduces this substantially.

**Multimodal follow-up** — follow-up image analysis in chat uses a single lightweight vision pass (200-token output) rather than the full Stage 1 schema. This is intentional for latency reasons but means follow-up image analysis provides a natural-language description rather than a structured reclassification.

**Trade category coverage** — the 10-category controlled vocabulary covers the most common domestic repair types. Faults that span multiple trades (e.g. water damage that has affected both plumbing and plastering) are classified to the primary trade and may not capture the full scope in Stage 1.

---

## 14. Future Roadmap

| # | Feature | Status | Notes |
|---|---|---|---|
| 1 | Three-stage AI pipeline | ✅ Complete | Qwen2.5-VL / Mistral / Llama3.1 via Ollama |
| 2 | Confidence gate + clarification | ✅ Complete | Halts on low confidence, explains reasoning |
| 3 | Explainability UI | ✅ Complete | Visual evidence chips, uncertainty panel |
| 4 | Case-aware chat (Ask Knitra) | ✅ Complete | Full context anchoring, multi-turn persistence |
| 5 | Voice input (mic transcription) | ✅ Complete | faster-whisper, DOM fallback sync |
| 6 | Follow-up image uploads in chat | ✅ Complete | Vision pass per attachment, image_ids in POST |
| 7 | My Cases filter bar | ✅ Complete | Category, status, confidence, date sort |
| 8 | Trade category normalisation | ✅ Complete | 10-category vocabulary, Pydantic validator |
| 9 | Status pipeline mirroring | ✅ Complete | sessions.status mirrors analyses.status |
| 10 | Video upload + frame extraction | 🔄 In progress | File picker accepts video; frame extraction pending |
| 11 | Stage 4 — Product recommendations | 📋 Planned | Match tools/materials to product DB; UK retailer cards |
| 12 | Camera capture | 📋 Planned | `getUserMedia({video:true})` live capture |
| 13 | Richer risk analysis | 📋 Planned | `risk_factors` list in Stage3Output; pre-pipeline rules |
| 14 | Hybrid safety overrides | 📋 Planned | Keyword-based pre-pipeline rules for gas/electrical |
| 15 | Nearby tradespeople | 📋 Planned | Trade label → radius geo-query → tradesperson list |
| 16 | PDF case export | 📋 Planned | Printable diagnosis + repair plan |
| 17 | WebSocket/SSE push | 📋 Planned | Replace 3.5 s polling with server-push progress |
| 18 | Backend task queue | 📋 Planned | Concurrent analysis requests without blocking |
| 19 | Usage tiers | 📋 Planned | Free (N cases/month) vs Pro (unlimited + priority) |

---

## Key Design Decisions

**Why three separate models rather than one?**
Each pipeline stage is a fundamentally different cognitive task. Stage 1 requires a vision-language model capable of examining images. Stage 2 is a structured planning retrieval task where Mistral's JSON output reliability outperforms general-purpose models. Stage 3 is long-form procedural generation where Llama 3.1:8b produces more coherent multi-step instructions. Separating stages means each model can be independently upgraded, swapped, or fine-tuned without affecting the others.

**Why `asyncio.create_task` instead of FastAPI `BackgroundTasks`?**
FastAPI's `BackgroundTasks` closes the request DB session when the request lifecycle ends. The analysis pipeline runs for up to two minutes and requires multiple DB writes (one per stage, plus status updates). `asyncio.create_task` keeps the coroutine alive beyond the request with its own `AsyncSessionLocal` session.

**Why `expire_on_commit=False` in the database session factory?**
SQLAlchemy's default `expire_on_commit=True` marks all ORM attributes as expired after each commit, requiring a fresh SELECT on next access. The analysis pipeline makes several commits within a single long-running async session and needs to continue reading and mutating the `Analysis` object between them. `expire_on_commit=False` prevents unexpected lazy-load errors in this pattern.

**Why a Python property for `Session.confidence` rather than a DB column?**
The confidence value lives in `Analysis.structured_result.stage_1.confidence` — a JSONB field on the `analyses` table. Rather than duplicating this into a `sessions` column and creating a migration, `Session.confidence` is a Python property that reads from the pre-loaded `analyses` relationship at serialisation time. `selectinload(Session.analyses)` is applied in `list_cases` so the property resolves without triggering lazy-load errors in the async context.

**Why `<label for="file-input">` instead of `input.click()` for the file picker?**
Browser security requires file picker invocations to originate from a direct synchronous user gesture. A NiceGUI button click travels through a WebSocket round-trip before the Python event loop can call `ui.run_javascript("input.click()")` — by then the gesture context has expired and the browser silently blocks it. A `<label for="file-input">` element activates the picker natively with no JavaScript and no gesture timing constraints.

**Why a Pydantic `field_validator` for trade category normalisation?**
The vision model returns raw strings that vary across runs: "carpentry", "Carpentry/Joinery", "joinery", "woodwork". Normalising these to a controlled 10-category vocabulary at the Pydantic validation layer means the normalisation applies on every analysis write path — whether from a fresh pipeline run or a retry — without requiring any additional service layer logic or migration to fix legacy data.
