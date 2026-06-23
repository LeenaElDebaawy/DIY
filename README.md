# Knitra — Local DIY Repair Analysis System

Knitra is a locally hosted repair assessment application. A user uploads a photo or
video of a household fault, optionally describes it in writing, and receives a
structured assessment: a classification of the problem, a routing decision based on
risk and evidence quality, and — where appropriate — a step-by-step repair plan with
tools, materials, and safety guidance.

All inference runs locally. No images, audio, or repair data leave the machine.

The system is built around a single principle: **a repair assistant is only useful if
it is honest about what it does not know.** Generating a confident-sounding plan for
a situation it cannot safely assess is treated as a failure mode, not a feature.
The pipeline therefore spends as much effort deciding *whether* to advise as it does
producing advice.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Features](#2-features)
3. [Architecture](#3-architecture)
4. [Analysis Pipeline](#4-analysis-pipeline)
5. [Routing Behaviour](#5-routing-behaviour)
6. [Safety Mechanisms](#6-safety-mechanisms)
7. [Video Analysis](#7-video-analysis)
8. [Ask Knitra — Contextual Chat](#8-ask-knitra--contextual-chat)
9. [Technology Stack](#9-technology-stack)
10. [Installation and Setup](#10-installation-and-setup)
11. [Environment Variables](#11-environment-variables)
12. [API Reference](#12-api-reference)
13. [Testing and Verification](#13-testing-and-verification)
14. [Known Limitations](#14-known-limitations)
15. [Design Principles](#15-design-principles)

---

## 1. Project Overview

### The problem

Someone facing a household fault usually has three unsatisfying options: pay a
professional for something they might have handled themselves, attempt the repair
without proper guidance, or sift through fragmented online advice that may not apply
to their situation.

A naïve AI assistant makes a fourth option that is arguably worse than all three: it
produces a confident-sounding plan regardless of whether the evidence actually
supports one. For a domain where mistakes can mean flooding, electrocution, or
structural damage, confident guesswork is a liability.

### What Knitra does

- Accepts an image or video upload of a household repair problem
- Runs a structured visual diagnosis using a local vision model
- Routes the case to one of six output modes based on risk level, evidence quality,
  and problem type — not a single confidence threshold
- For low-risk, clearly visible problems: generates a specific, trade-appropriate
  repair plan with tools, materials, and ordered steps
- For high-risk or unclear situations: escalates or asks targeted questions rather
  than producing advice the evidence does not support

---

## 2. Features

### Analysis and diagnosis

| Feature | Description |
|---|---|
| **Image upload and analysis** | One or more photographs are passed to a local vision model for structured classification. |
| **Video upload and analysis** | Short video clips are processed by extracting frames, which are then analysed the same way as images. Video analysis requires `VIDEO_ANALYSIS_ENABLED=true`. |
| **Stage 1 classification** | Problem type, trade category, severity, confidence, risk level, DIY suitability, and explicit lists of what is and is not visible in the media. |
| **Stage 2 root-cause analysis** | Essential and optional tools, materials, protective gear, estimated time, ranked likely causes, and a repair confidence assessment. |
| **Stage 3 repair plan** | Numbered steps in WHAT/WHY/RESULT format, difficulty rating, relevant safety notes, and guidance on when to stop and call a professional. |
| **Six output modes** | Each case routes deterministically to one of six outcomes (see §5). The same diagnosis always produces the same route. |
| **Trade profile routing** | Stage 2 and Stage 3 prompts are trade-specific (carpentry, plumbing, electrical, plastering, etc.), not generic. |
| **Structured risk signals** | Routing is driven by explicit model-produced signals — risk level, professional assessment recommendation, DIY suitability — not a single number. |

### Safety and trust

| Feature | Description |
|---|---|
| **One-way escalation** | Post-processing rules can raise (never lower) the model's own risk signals before routing. |
| **Hazard acknowledgement** | Hazardous cases require the user to read and acknowledge a safety notice before the conversational assistant opens. |
| **Clarification with memory** | When more information is needed, the system asks targeted questions and carries prior answers forward across rounds. |
| **Bounded clarification** | Clarification cannot loop indefinitely; an explicit inconclusive outcome surfaces if confidence cannot be reached. |
| **Auto re-analysis from chat** | Answering a clarification question in chat triggers a fresh official analysis automatically. |

### Case management

| Feature | Description |
|---|---|
| **Case dashboard** | Filterable list of all cases with confidence, trade category, status, and date. |
| **Case history and detail** | Full diagnosis, routing decision, Stage 2/3 output, and clarification history per case. |
| **Automatic titling** | Cases are titled from the diagnosis once analysis completes. |
| **Drag-and-drop upload** | Photo and video selection with instant thumbnails; client-side image compression before upload. |

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Browser / NiceGUI interface               │
│        (Python-rendered, WebSocket-synced single-page UI)     │
└────────────────────────────┬─────────────────────────────────┘
                             │  HTTP / REST
┌────────────────────────────▼─────────────────────────────────┐
│                       FastAPI backend                         │
│                                                               │
│   Auth · Cases · Uploads · Transcription · Analysis · Chat    │
│                                                               │
│   ┌─────────────────────────────────────────────────────┐    │
│   │              Analysis pipeline (background task)     │    │
│   │                                                      │    │
│   │   Image ──► Stage 1 (Ollama / Qwen2.5-VL)           │    │
│   │   Video ──► frame extract (PyAV)                     │    │
│   │             ──► Stage 1 (HF Transformers / Qwen2.5-VL│    │
│   │                                                      │    │
│   │   Stage 1 output ──► routing decision                │    │
│   │                                                      │    │
│   │   diy / verify_first ──► Stage 2 (Mistral 7B)       │    │
│   │                          ──► Stage 3 (Llama 3.1 8B) │    │
│   │                                                      │    │
│   │   assessment_first / needs_clarification /           │    │
│   │   out_of_scope / care ──► stop, return focused output│    │
│   └─────────────────────────────────────────────────────┘    │
│                                                               │
│   ┌─────────────────────────────────────────────────────┐    │
│   │   Chat service — per-case contextual assistant        │    │
│   └─────────────────────────────────────────────────────┘    │
└────────────────────────────┬─────────────────────────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        ▼                                          ▼
┌────────────────────┐                  ┌───────────────────────┐
│   PostgreSQL        │                  │   Ollama (local)      │
│   users · sessions  │                  │   Stage 1 (image) ·  │
│   uploads · analyses│                  │   Stage 2 · Stage 3 · │
│   messages          │                  │   chat                │
└────────────────────┘                  └───────────────────────┘
```

### Components

**`frontend/main.py` (NiceGUI).** Python-rendered single-page interface covering
authentication, the case list, case composer, and case detail views. JavaScript is
used only where the browser requires it directly — file picker, microphone capture,
and client-side image compression.

**`backend/main.py` (FastAPI entry point).** The ASGI entry point; start as
`uvicorn main:app` from the `backend/` directory.

**`backend/app/services/analysis_service.py`.** The main analysis pipeline. Handles
the image path (Stages 1, 2, 3), routing logic, safety post-processing, clarification
management, and all output mode generation. Stages run as a background task so the
HTTP response returns immediately.

**`backend/app/services/video_analysis_service.py`.** The video path. Extracts frames
using PyAV, runs Qwen2.5-VL via HuggingFace Transformers, applies post-processing
rules, and returns a `VideoStage1Output` that feeds into the existing routing and
Stage 2/3 pipeline. The model is loaded once per process.

**`backend/app/services/uploads_service.py`.** Handles file ingestion (images and
video), storage, and MIME type detection.

**`backend/app/schemas/analysis.py`.** Pydantic schemas for all pipeline stages and
output modes: `Stage1Output` (image path), `VideoStage1Output` (video path),
`Stage2Output`, `Stage3Output`, and the six output-mode schemas.

**`backend/app/schemas/uploads.py`.** Upload schemas including video-specific fields
added in the Alembic migration.

**PostgreSQL (Docker).** Stores users, sessions, uploads, analyses, and chat history.
Analyses include a `stage_outputs` JSONB column (intermediate stage data) and a
`structured_result` JSONB column (final output for the frontend).

**Alembic migrations.** `backend/alembic/versions/` — including
`e5f6a7b8c9d0_add_video_fields_to_uploads.py` which adds video-specific columns to
the uploads table.

---

## 4. Analysis Pipeline

### Image path

1. One or more images are passed to the Stage 1 model (Qwen2.5-VL via Ollama)
   alongside any user-provided text description.
2. Stage 1 returns a `Stage1Output`: problem type, trade category, severity,
   confidence, risk level, DIY suitability, what was and was not visible, and
   clarifying questions if needed.
3. Post-processing rules may raise (never lower) risk signals.
4. The routing decision runs on the adjusted Stage 1 output.
5. If the case routes to `diy` or `verify_first`: Stage 2 then Stage 3 run.
6. All other routes stop after Stage 1 and return a focused output.

### Video path

1. The video file is read and frames are extracted using PyAV (`av` package).
   By default 12 frames are sampled evenly across the clip.
2. Extracted frames are passed to the Stage 1 model (Qwen2.5-VL loaded via
   HuggingFace Transformers). The model is loaded once per process (~30 s on first
   use); subsequent calls reuse the in-memory model.
3. Stage 1 returns a `VideoStage1Output`: a superset of `Stage1Output` with
   additional video-specific fields including `key_moments`, `what_can_be_confirmed`,
   `what_cannot_be_confirmed`, `hidden_damage_risk`, `temporal_confidence`, and
   `motion_detected`.
4. `_apply_video_safety_adjustments()` runs post-processing rules:
   - **Rule -1**: upgrades `scope` from `not_repair` to `repair` when problem type
     and trade category are both present and clearly indicate a repair task.
   - **Rule -0**: downgrades `scope` to `not_repair` for decorative or aesthetic
     tasks (mural painting, wall art) when no damage signals are present. Does not
     apply to structural, electrical, or plumbing trades.
   - **Rules 0–5**: escalate risk level, set hazard flags, adjust DIY suitability,
     and enforce confidence floors where the evidence warrants it.
5. From this point, the `VideoStage1Output` is passed into the same routing and
   Stage 2/3 pipeline used by the image path. No separate video-specific Stage 2
   or Stage 3 exists.

### Stage 2 (planning)

Runs only for `diy` and `verify_first` routes. Takes the Stage 1 output as context
and produces:

- Essential and optional tools (with specific sizing notes for the trade)
- Essential and optional materials
- Protective gear
- Estimated time in minutes
- Ranked likely causes with confidence percentages (`likely_causes`)
- A repair confidence assessment citing the Stage 1 evidence
  (`repair_confidence_reasoning`)

Trade-specific profiles are injected into the Stage 2 prompt. For example, the
carpentry/joinery profile explicitly prohibits Allen keys and lubricant for concealed
hinge work and requires Pozi/Phillips screwdrivers. If `trade_category` is
`general/other` but `problem_type` mentions "hinge", "cabinet door", "cupboard door",
or "wardrobe door", the profile is automatically upgraded to carpentry/joinery.

### Stage 3 (repair steps)

Takes Stage 1 and Stage 2 outputs as context and produces:

- Numbered steps in WHAT / WHY / RESULT format
- Difficulty level and DIY recommendation
- Safety notes (only those relevant to the specific trade and problem)
- Guidance on when to stop and call a professional

### When Stage 2 and Stage 3 do not run

| Output mode | Stage 2 | Stage 3 |
|---|---|---|
| `diy` | ✅ | ✅ |
| `verify_first` | ✅ | ✅ |
| `assessment_first` | ✗ | ✗ |
| `needs_clarification` | ✗ | ✗ |
| `out_of_scope` | ✗ | ✗ |
| `care` | ✗ | ✗ |

---

## 5. Routing Behaviour

The routing decision runs on the adjusted Stage 1 output. The same Stage 1 output
always produces the same route. Gates are checked in priority order:

| Priority | Gate | Output mode |
|---|---|---|
| 1 (highest) | `scope != 'repair'` | `out_of_scope` |
| 2 | `professional_assessment_recommended=True` OR `risk_level` is `high`/`critical` | `assessment_first` |
| 3 | `needs_clarification=True` OR confidence below floor | `needs_clarification` |
| 4 | High hidden damage risk, observed hazard, or conditional DIY suitability | `verify_first` |
| 5 | Low-risk maintenance task | `care` |
| 6 (default) | Everything else | `diy` |

**Gate 2 takes priority over Gate 3.** A high-risk case (`risk_level=high` or
`professional_assessment_recommended=True`) bypasses the clarification gate entirely
and routes to `assessment_first` even if `needs_clarification=True`. A clarification
question ("Is the boiler still on?") does not change a safety recommendation ("call a
professional"), so routing clarification first would be incorrect.

### Output mode descriptions

| Mode | When chosen | User receives |
|---|---|---|
| **`diy`** | Repair is clear, low-risk, and suitable for a competent non-professional. | Full repair plan: tools, materials, steps, difficulty, safety notes. |
| **`verify_first`** | Repair is plausibly DIY but hidden damage may be present or risk is elevated. | Full repair plan plus a specific check to perform before starting. |
| **`assessment_first`** | Professional inspection is needed; a concrete hazard can be named. | What to do immediately, what to avoid, which professional to contact. No repair steps. |
| **`needs_clarification`** | Diagnosis depends on information not present in the media. | Summary of what is known and unknown, plus targeted questions. |
| **`out_of_scope`** | The task is not a repair (e.g. decorative painting, aesthetic redecoration with no damage). | Explanation of why the system cannot assist. |
| **`care`** | Low-risk maintenance or care task. | Practical, material-specific care guidance. |

### Safety escalation for specific categories

Post-processing rules raise (never lower) risk signals for:

- Electrical faults (exposed wiring, consumer unit issues)
- Gas appliances (boilers, gas supply, carbon monoxide risk)
- Water heaters and immersion heaters
- Structural concerns (load-bearing walls, roof structure, foundation cracks)
- Mould and damp where hidden damage behind surfaces is likely

For these categories, `risk_level` may be raised to `high` and
`professional_assessment_recommended` set to `True`, routing the case to
`assessment_first` regardless of any clarification flag.

---

## 6. Safety Mechanisms

### Hazard acknowledgement

When a case routes to `assessment_first` and a named hazard is present, it enters an
"Awaiting Acknowledgement" sub-status. The interface shows a safety notice — what to
do immediately, what to avoid, and which professional to contact. Conversational
assistance is withheld until the user acknowledges the notice. Acknowledgement is
timestamped and does **not** unlock a repair plan or override the recommendation to
call a professional.

### Clarification with memory

When a case routes to `needs_clarification`, the assistant presents its questions in
chat. Once answered, a fresh official analysis is launched automatically incorporating
those answers. Answers from prior rounds are carried forward so the same question is
never asked twice.

### Bounded clarification

If repeated clarification rounds do not produce a confident diagnosis, the system
surfaces an explicit inconclusive outcome and stops asking. This is a legitimate
result, not an error.

### Updated-after-clarification history

When clarification changes the outcome, the interface marks the result as updated and
shows the questions and answers that informed the revision.

---

## 7. Video Analysis

### Overview

Video analysis is available but **disabled by default**. To enable it, set
`VIDEO_ANALYSIS_ENABLED=true` in `backend/.env` and restart the backend.

When a video is uploaded and analysis is triggered:

1. PyAV (`av` package) extracts evenly spaced frames from the clip. The number of
   frames is configurable via `VIDEO_NFRAMES` (default: 12).
2. The frames are passed to Qwen2.5-VL loaded via HuggingFace Transformers. On the
   first call the model loads from disk into GPU or CPU memory (~30 s). Subsequent
   calls within the same process reuse the loaded model.
3. The resulting `VideoStage1Output` feeds into the standard routing and Stage 2/3
   pipeline without modification.

### Usage notes

- **Do not run the backend with `--reload` during video analysis.** The file-watcher
  will restart the process mid-analysis, killing the in-progress inference. Start
  the backend as:
  ```
  uvicorn main:app --host 0.0.0.0 --port 8000
  ```
- **Process one video at a time.** Local GPU inference is memory-heavy. Concurrent
  video analyses will contend for VRAM and may fail or produce degraded output.
- **The local video model is for development and testing.** `VIDEO_MODEL_LOCAL`
  defaults to `Qwen/Qwen2.5-VL-7B-Instruct` loaded via HuggingFace Transformers.
  A cloud inference path (`VIDEO_INFERENCE_MODE=cloud`) is implemented for future use
  with a remote vLLM endpoint but is not required for local development.

### Video-specific Stage 1 fields

`VideoStage1Output` includes all `Stage1Output` fields plus:

| Field | Description |
|---|---|
| `key_moments` | Frame descriptions of diagnostically important moments in the clip. |
| `what_can_be_confirmed` | What the video frames allow the model to verify. |
| `what_cannot_be_confirmed` | What cannot be determined from the available frames. |
| `hidden_damage_risk` | Whether damage not visible in the frames is likely (`low`, `medium`, `high`). |
| `temporal_confidence` | How consistently the problem appears across frames (0–100). |
| `motion_detected` | Whether movement relevant to the fault is visible. |

---

## 8. Ask Knitra — Contextual Chat

A per-case conversational assistant opens once analysis is complete (subject to the
safety gating in §6). Every conversation is anchored to the specific case — problem
type, severity, what was and was not visible, and the full Stage 2/3 output where
available.

Capabilities:

- **Voice input.** Dictated messages are transcribed locally using faster-whisper
  and placed in the message box.
- **Follow-up images.** Additional photos can be attached mid-conversation; the
  vision model describes them and the assistant uses that description in its reply.
- **Scope awareness.** Off-topic requests are redirected to the repair case. The
  assistant will not help circumvent an active safety warning.

The full conversation is stored per case and persists across sessions.

---

## 9. Technology Stack

| Component | Technology |
|---|---|
| Backend framework | FastAPI (async) |
| Language | Python 3.12 |
| ORM | SQLAlchemy 2.0 (async) |
| Database | PostgreSQL (via Docker) |
| Migrations | Alembic |
| Frontend | NiceGUI |
| Schema validation | Pydantic v2 |
| **Image Stage 1 model** | Qwen2.5-VL 7B via Ollama |
| **Video Stage 1 model** | Qwen2.5-VL 7B via HuggingFace Transformers (local) |
| **Stage 2 model** | Mistral 7B Instruct via Ollama |
| **Stage 3 and chat model** | Llama 3.1 8B via Ollama |
| Video frame extraction | PyAV (`av` package) |
| Speech-to-text | faster-whisper (CPU, local) |
| Image processing | Pillow |
| Authentication | Bearer token with hashed credentials |
| Containerisation | Docker Compose |

All inference runs locally. No cloud API keys are required for the standard setup.

---

## 10. Installation and Setup

### Prerequisites

- Python 3.12 or newer
- Docker Desktop (for PostgreSQL)
- [Ollama](https://ollama.com), installed and running
- A CUDA-capable GPU is recommended for video analysis; CPU-only is supported but
  slow for video inference
- Node.js is **not** required

### 1. Start the database

```bash
docker compose up -d db
```

### 2. Pull the Ollama models

One-time download. After this, all Ollama inference is local.

```bash
ollama pull qwen2.5vl:7b          # Stage 1 — image classification
ollama pull mistral:7b-instruct   # Stage 2 — tools, materials, root causes
ollama pull llama3.1:8b           # Stage 3 — repair steps, and chat
```

Confirm Ollama is reachable:

```bash
curl http://localhost:11434/api/tags
```

### 3. Configure the backend

```bash
cd backend
cp .env.example .env
# Edit .env as needed — see §11 for all variables
```

### 4. Install backend dependencies and run migrations

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
alembic upgrade head
```

### 5. Start the backend

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

> **Do not use `--reload` if you plan to run video analysis.** The file-watcher
> restarts the process mid-analysis. `--reload` is safe for image-only development.

### 6. Start the frontend

In a second terminal:

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

### Convenience scripts (Windows)

PowerShell scripts at the project root (`start_knitra.ps1`, `stop_knitra.ps1`,
`knitra.ps1`) start and stop the full stack for development on Windows.

### Service URLs

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| Ollama | http://localhost:11434 |
| PostgreSQL | localhost:5433 |

---

## 11. Environment Variables

All variables are read from `backend/.env`. Defaults are applied in code when a
variable is absent.

### Database

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://fixly:fixly@localhost:5433/fixly` | PostgreSQL connection string |
| `DATABASE_ECHO` | `false` | Log every SQL statement to stdout |

### Ollama / image pipeline

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `STAGE_1_MODEL` | `qwen2.5vl:7b` | Image Stage 1 vision model |
| `STAGE_2_MODEL` | `mistral:7b-instruct` | Stage 2 planning model |
| `STAGE_3_MODEL` | `llama3.1:8b` | Stage 3 generation and chat model |
| `PIPELINE_RETRY_COUNT` | `2` | Retries per stage on invalid model output |
| `OLLAMA_KEEP_ALIVE` | `5m` | How long Ollama keeps a model in RAM after last use |
| `STAGE_1_NUM_CTX` | `4096` | Stage 1 context window (must fit image tokens + prompt) |

### Video pipeline

| Variable | Default | Description |
|---|---|---|
| `VIDEO_ANALYSIS_ENABLED` | `false` | Set to `true` to enable video analysis |
| `VIDEO_INFERENCE_MODE` | `local` | `local` (HF Transformers) or `cloud` (remote vLLM) |
| `VIDEO_MODEL_LOCAL` | `Qwen/Qwen2.5-VL-7B-Instruct` | HuggingFace model ID for local inference |
| `VIDEO_MODEL_CLOUD` | `Qwen/Qwen2.5-VL-72B-Instruct` | HuggingFace model ID for cloud inference |
| `VIDEO_INFERENCE_ENDPOINT` | *(empty)* | Remote vLLM endpoint URL (cloud mode only) |
| `VIDEO_INFERENCE_API_KEY` | *(empty)* | Bearer token for remote endpoint (cloud mode only) |
| `VIDEO_NFRAMES` | `12` | Frames sampled per video |
| `VIDEO_MAX_NEW_TOKENS` | `2048` | Token budget for the video model response |
| `VIDEO_USE_4BIT` | `true` | Enable 4-bit NF4 quantisation (reduces VRAM usage) |

---

## 12. API Reference

All routes under `/api/v1/` require a bearer token except registration, login, and
the health probe.

### Authentication

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/auth/register` | Create an account and return a token |
| `POST` | `/api/v1/auth/login` | Log in and return a token |
| `GET` | `/api/v1/auth/me` | Return the current user |
| `POST` | `/api/v1/auth/logout` | Invalidate the current token |

### Cases

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/cases` | Open a new case |
| `GET` | `/api/v1/cases` | List cases (newest first) |
| `GET` | `/api/v1/cases/{id}` | Retrieve a single case |
| `PATCH` | `/api/v1/cases/{id}` | Update title, category, or status |
| `DELETE` | `/api/v1/cases/{id}` | Delete a case and its data |

### Uploads

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/cases/{id}/uploads` | Upload an image or video file to a case |
| `GET` | `/api/v1/cases/{id}/uploads` | List a case's uploads |

### Analysis

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/cases/{id}/analyse` | Trigger analysis (returns immediately) |
| `GET` | `/api/v1/cases/{id}/analysis` | Poll for status, progress, and result |

### Chat

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/cases/{id}/chat` | Retrieve the conversation |
| `POST` | `/api/v1/cases/{id}/chat` | Send a message and receive a reply |
| `POST` | `/api/v1/cases/{id}/chat/stream` | Send a message and stream the reply; may signal an automatic re-analysis |
| `POST` | `/api/v1/cases/{id}/chat/spawn-case` | Create a new case from a photo attached in chat |

### Transcription and health

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/transcribe` | Transcribe an uploaded audio clip |
| `GET` | `/health` | Liveness probe |

---

## 13. Testing and Verification

There are no required test steps to run the application. The regression cases below
are the recommended manual verification path when making changes to the routing logic,
safety rules, or Stage 2/3 prompts.

> Developer note: `backend/spikes/` contains local development scripts for unit
> testing routing logic, replaying stored Stage 1 outputs, and running end-to-end
> video tests. These are not part of the application and are not required to run or
> deploy it.

### Recommended regression cases

When making changes to the pipeline, verify these cases:

1. **Cabinet hinge** — expected: `verify_first` or `diy`; Stage 2 must list a
   cross-head screwdriver (not an Allen key), no lubricant; Stage 3 steps must
   reference depth, side, and height adjustment screws.
2. **Decorative mural / wall art** — expected: `out_of_scope`; Stage 2/3 must not
   run.
3. **Peeling paint or damp-damaged wall** — expected: `repair` scope, routed
   normally; must not be caught by the decorative task filter.
4. **Water heater / high-risk plumbing** — expected: `assessment_first` even if
   `needs_clarification=True`; the clarification gate must be bypassed.
5. **Image analysis regression** — confirm that image-path cases are unaffected by
   video routing rules; `Stage1Output` has no `what_cannot_be_confirmed` attribute.

---

## 14. Known Limitations

- **Local video inference is slow.** The first video analysis in a process takes
  ~30 s to load the model plus inference time. On CPU-only machines, video analysis
  is not practical for routine use.

- **Small local models can be inconsistent.** Mistral 7B (Stage 2) may return fewer
  than three likely causes; Llama 3.1 8B (Stage 3) may omit some trade-specific
  steps across runs. Output quality is bounded by the model size.

- **Diagnosis depends on media quality.** Blurry, distant, or poorly lit images and
  videos reduce diagnostic accuracy. The clarification and uncertainty mechanisms
  surface this rather than hiding it, but cannot recover information the media does
  not contain.

- **Frontend display of new Stage 2 fields.** The `likely_causes` and
  `repair_confidence_reasoning` fields are produced by the pipeline but may not yet
  be displayed in all frontend views.

- **`difficulty_level` for hinge adjustments** is sometimes reported as
  `intermediate` when `beginner` would be more accurate. This is model output
  variability and is not currently post-processed.

- **Clarification answer matching is sequential.** Answers are paired with questions
  in order. Out-of-sequence answers may display incorrectly in the history view,
  though all answers are used in the re-analysis.

- **Safety mechanisms are prompt-level and interface-level.** The hazard gate and
  the assistant's refusal to bypass warnings are enforced through the interface and
  model instruction. They are not equivalent to a hard server-side block.

- **Not a replacement for professional assessment in high-risk situations.** The
  system escalates to `assessment_first` for high-risk categories, but the quality of
  advice for borderline cases depends on local model accuracy, which varies.

---

## 15. Design Principles

**Decide before advising.** The system treats "should I produce a repair plan?" as a
distinct decision from "what is the repair plan?". This separation is what allows it
to ask, escalate, or decline rather than always answering.

**Route on structured signals, not a single number.** Outcomes are chosen from
explicit risk and suitability signals rather than a confidence threshold alone. This
keeps the system from proceeding just because a model reported a high number, or
halting just because it reported a low one.

**Specialise each stage.** Diagnosis, planning, and step generation are different
tasks. A separate, appropriate model handles each, receiving only the context it
needs. Any one stage can be upgraded or replaced independently.

**Keep the official result authoritative.** Conversation is for understanding and
clarification. It never silently rewrites the assessment a user acts on. Only a full,
validated re-analysis updates the official result.

**Run locally, by default.** All inference — vision, language, and speech — runs on
the user's machine. Repair photographs of someone's home are sensitive, and local
inference is a design requirement, not an afterthought.

**Be honest about uncertainty.** Surfacing what the system cannot determine, bounding
clarification rounds, and presenting an inconclusive outcome when warranted are
treated as core features, not edge cases.
