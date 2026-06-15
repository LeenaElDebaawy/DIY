# Knitra : A Trust-Oriented Multimodal Repair Assessment System

Knitra is a locally hosted, privacy-preserving research prototype that helps people
assess household repair problems from a photograph. A user uploads an image of a
fault, optionally describes it by typing or speaking, and receives a structured
assessment: a classification of the problem, an explanation of what the system can
and cannot determine from the evidence, and — where appropriate — a step-by-step
repair plan with tools, materials, and safety guidance.

All inference runs locally through [Ollama](https://ollama.com) using open-source
models. No images, audio, or repair data leave the machine.

Knitra began as a straightforward image-to-repair-plan assistant. Since its initial
demonstration it has been redesigned around a single principle: **a repair assistant
is only useful if it is honest about what it does not know.** The current system
therefore spends as much effort deciding *whether* it can safely advise as it does
generating advice. This README describes the system as it stands today, not the
original demo.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Features](#2-key-features)
3. [Architecture Overview](#3-architecture-overview)
4. [The Analysis Pipeline and Output Modes](#4-the-analysis-pipeline-and-output-modes)
5. [Safety and Trustworthiness Mechanisms](#5-safety-and-trustworthiness-mechanisms)
6. [Explainability and Confidence](#6-explainability-and-confidence)
7. [Ask Knitra — Contextual Chat Assistant](#7-ask-knitra--contextual-chat-assistant)
8. [Technology Stack](#8-technology-stack)
9. [Installation and Setup](#9-installation-and-setup)
10. [API Reference](#10-api-reference)
11. [Current Limitations](#11-current-limitations)
12. [Future Work](#12-future-work)
13. [Design Principles](#13-design-principles)

---

## 1. Project Overview

### The problem

Someone facing a household fault — a leaking pipe joint, a cracked tile, a damp
patch, exposed wiring — usually has three unsatisfying options: pay a professional
for something they might have handled themselves, attempt the repair without proper
guidance and risk making it worse, or sift through fragmented online advice that may
not apply to their exact situation.

A naïve AI assistant makes a fourth option that is arguably worse than all three: it
produces a confident-sounding plan regardless of whether the photograph actually
supports one. For a domain where mistakes can mean flooding, electrocution, or
structural damage, confident guesswork is a liability rather than a feature.

### What Knitra is

Knitra is a **multimodal repair assessment system** built around three ideas:

- **Structured visual diagnosis.** A local vision–language model examines the
  uploaded photograph (and any text the user provides) and produces a structured
  assessment rather than free-form prose: a problem classification, a trade
  category, severity and risk signals, and an explicit account of what is and is not
  visible in the image.

- **Decision-making before advice.** The system does not assume every case warrants
  a repair plan. Based on the structured diagnosis, it routes each case into one of
  five outcomes — proceed with a plan, ask for clarification, recommend professional
  assessment, advise verification first, or give care/maintenance guidance.

- **Trust as a first-class concern.** Where the evidence is insufficient, Knitra
  says so. It asks targeted questions, remembers the answers across rounds, refuses
  to loop indefinitely, and — for hazardous situations — withholds conversational
  assistance until the user has acknowledged the safety warning.

The result is a system whose primary output is not always a repair plan. Sometimes
the most useful and honest result is "I cannot safely advise on this from the
photograph you have given me, and here is exactly why."

---

## 2. Key Features

### Diagnosis and decision-making

| Feature | Description |
|---|---|
| **Image-based diagnosis** | A vision–language model identifies the fault, trade category, and severity from one or more photographs. |
| **Five output modes** | Each case is routed deterministically to one of five outcomes (see §4), so the response always matches what the evidence supports. |
| **Structured risk signals** | Routing is driven by explicit, model-produced risk signals — not a single confidence number — making behaviour predictable and auditable. |
| **Confidence and uncertainty reporting** | Every diagnosis records a confidence score, what was visually confirmed, and what could not be confirmed. |
| **One-way safety guardrail** | For inherently dangerous categories, the system may raise (never lower) the model's own risk signals before deciding how to respond. |
| **Trade category normalisation** | An 11-category controlled vocabulary, normalised from the many raw variants a model may emit, so downstream behaviour stays consistent. |

### Safety and trustworthiness

| Feature | Description |
|---|---|
| **Hazard acknowledgement workflow** | Hazardous cases require the user to read and acknowledge a safety notice before conversational help opens. Acknowledgement is timestamped for auditability and never overrides a recommendation to seek professional assessment. |
| **Clarification auto re-analysis** | When the system needs more information, the user can answer in chat and a fresh official assessment is launched automatically using those answers. |
| **Clarification memory** | Answers from earlier rounds are retained and reused, so the user is not asked the same thing twice. |
| **Updated-after-clarification history** | When a result changes because of clarification, the interface marks it as updated and lets the user review the questions and answers that informed the revised diagnosis. |
| **Bounded clarification** | Clarification cannot loop forever; if confidence cannot be reached after repeated rounds, the system surfaces an explicit inconclusive outcome. |

### Case management and interface

| Feature | Description |
|---|---|
| **Case dashboard** | A filterable list of all cases showing confidence, category, status, and date. |
| **Filtering** | Filter by trade category, status, and confidence tier, with date sorting. |
| **Drag-and-drop upload** | Multi-photo selection with instant thumbnails and client-side compression before upload. |
| **Automatic case titling** | A case is titled from its diagnosis once analysis completes. |
| **Ask Knitra** | A case-aware conversational assistant that opens after analysis and already knows the full context of the case. |
| **Voice and follow-up images in chat** | Microphone input is transcribed locally; additional photos can be attached mid-conversation. |

---

## 3. Architecture Overview

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
│   │              Analysis service (background)            │    │
│   │                                                       │    │
│   │   Vision diagnosis ──► route by structured signals    │    │
│   │       │                                               │    │
│   │       ├─ DIY / Verify First ─► planning ─► generation │    │
│   │       ├─ Assessment First   ─► assessment output      │    │
│   │       ├─ Care               ─► care guidance          │    │
│   │       └─ Clarification      ─► targeted questions      │    │
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
│   users · sessions  │                  │   vision · planning · │
│   uploads · analyses│                  │   generation · chat   │
│   messages          │                  │   models              │
└────────────────────┘                  └───────────────────────┘
```

### Components

**Backend (FastAPI).** An asynchronous REST API serving authentication, case
management, uploads, analysis, chat, and local audio transcription. The analysis
pipeline runs as a background task so the interface stays responsive and can report
progress as each stage completes.

**Frontend (NiceGUI).** A Python-rendered single-page interface covering the landing,
authentication, case list, case composer, and case detail views. JavaScript is used
only where the browser requires it directly — the file picker, microphone capture,
and client-side image compression.

**Database (PostgreSQL).** Stores users, cases, uploads, analyses, and the full chat
history. Case status — including the acknowledgement state of hazardous cases — is
mirrored here so the dashboard and filters remain accurate.

**Inference runtime (Ollama).** All models run locally. There are no cloud API keys
and no external inference calls. Models are addressed over Ollama's local HTTP
interface.

**Speech-to-text.** Transcription runs locally on the CPU inside the backend, so
audio is never sent to a third party.

---

## 4. The Analysis Pipeline and Output Modes

### Diagnosis first, advice second

Every case begins with a single visual diagnosis step. A vision–language model
examines the photograph alongside any user-provided description and returns a
structured result: the problem type, a normalised trade category, a severity
rating, a confidence score, a list of what it could visually confirm, a list of
what it could not, and a set of structured risk signals describing how dangerous the
situation may be, how much damage might be hidden from view, and whether the job is
suitable for a competent non-professional.

These structured signals — not a single confidence threshold — determine what
happens next. The routing decision is deterministic: the same diagnosis always
produces the same outcome, which makes the system's behaviour predictable and
reviewable.

### The five output modes

| Mode | Chosen when | What the user receives |
|---|---|---|
| **DIY** | The diagnosis is clear and the repair is visible, well-understood, and suitable for a competent non-professional. | A full repair plan: tools, materials, ordered steps, difficulty, and safety notes. |
| **Verify First** | The repair is plausibly do-it-yourself, but important damage may be hidden or the risk is elevated. | A full repair plan framed around checking the true extent of the problem before committing. |
| **Assessment First** | A professional should inspect the situation in person, and a concrete hazard can be named. | A focused assessment: what to do immediately, what to avoid, why professional assessment is needed, what evidence to gather, and which professional to contact. No repair steps. |
| **Clarification** | The diagnosis depends on information the photograph cannot provide. | A short summary of what is known and unknown, and targeted questions whose answers would change the recommendation. |
| **Care** | The subject is a low-risk maintenance or care task rather than a repair. | Practical, material-specific care guidance, without the safety and professional-referral apparatus that would imply danger. |

Only **DIY** and **Verify First** run the full planning and step-generation pipeline.
The other three modes deliberately stop early and return a focused, appropriate
response instead of a repair plan that the evidence does not justify.

### The planning pipeline (DIY and Verify First)

When a case warrants a full plan, two further model stages run in sequence:

1. **Planning.** A language model takes the structured diagnosis and produces the
   essential and optional tools, essential and optional materials, protective
   equipment, and an estimated time.

2. **Generation.** A second language model takes the diagnosis and the plan and
   produces ordered, numbered repair steps — each with an optional warning — along
   with a difficulty rating, safety notes, and explicit guidance on when to call a
   professional.

Each stage receives only the context it needs, which keeps each model focused and
reduces the risk of one stage contaminating another. Stages are validated against a
fixed output structure and retried on malformed output.

---

## 5. Safety and Trustworthiness Mechanisms

These mechanisms were the main focus of development after the initial demonstration.
Together they shift Knitra from "an assistant that produces repair plans" to "an
assistant that decides, transparently, whether a repair plan is appropriate at all."

### Hazard acknowledgement workflow

When the diagnosis indicates that a professional should assess the situation in
person and a concrete hazard can be named, the case is routed to **Assessment First**
and enters an **"Awaiting Acknowledgement"** sub-status.

In this state the case shows a safety notice describing the hazard, what to do
immediately, what to avoid, and which professional to contact. Conversational
assistance is withheld until the user explicitly acknowledges the notice. The
acknowledgement is recorded with a timestamp, so there is a durable record that the
warning was shown and seen.

Acknowledgement only confirms that the user has read the warning. It does **not**
unlock a repair plan and does **not** override the recommendation to seek
professional assessment — the system continues to recommend professional help
regardless.

### Locked assistance until acknowledgement

Conversational assistance is gated behind safety in two situations. For hazardous
cases, the chat remains locked until the hazard notice is acknowledged. For ordinary
do-it-yourself cases, the chat is locked behind the same "before you begin" safety
acknowledgement that gates the repair steps, so there is no window in which a user
can ask for help before seeing the relevant safety conditions.

As an additional layer, the assistant is instructed never to help a user bypass an
active hazard; if asked to do so, it restates the need for professional assessment
rather than complying.

### Clarification auto re-analysis

When a case is routed to **Clarification**, the assistant presents its questions in
the chat. The user can simply answer them in conversation. Once the round's
questions are answered, Knitra automatically launches a fresh official analysis that
incorporates those answers — the user does not need to restart the process or press
a separate button. A manual re-analysis option remains available for users who
prefer to trigger it explicitly or add a free-text note.

Crucially, conversational answers never silently become the official result. Only a
re-analysis updates the assessment a user acts on, so the safety-graded output
always comes from the full, validated pipeline rather than from a chat reply.

### Clarification memory across rounds

Clarification can take more than one round. Knitra preserves the answers given in
earlier rounds and carries them forward into each subsequent analysis, so the model
sees the full history of what the user has told it. This prevents the common and
frustrating failure mode of an assistant re-asking a question that has already been
answered.

### Updated-after-clarification history

When clarification changes the outcome, the interface makes this explicit. The
revised result is clearly marked as having been updated following clarification, and
the user can expand a history view showing the questions asked and the answers given
that contributed to the revised diagnosis. The reasoning trail behind a changed
result is therefore visible rather than hidden.

### Bounded clarification and inconclusive outcomes

Clarification is bounded. If repeated rounds still do not produce a confident
diagnosis, Knitra stops asking and surfaces an explicit **inconclusive** outcome
rather than trapping the user in an endless question loop. The inconclusive result is
presented distinctly — it makes clear that no further questions are being asked, that
the system could not reach a confident assessment from the available evidence, and
that the user should provide clearer photographs or have the issue checked in person.
This is treated as a legitimate and honest outcome, not an error.

---

## 6. Explainability and Confidence

Knitra is designed so a user can always see *why* it reached a conclusion, not only
*what* the conclusion was.

- **Confidence score.** Every diagnosis carries a confidence value, shown both on
  the case and in the dashboard, so users can tell at a glance which diagnoses are
  firm and which are tentative.

- **What was confirmed.** The diagnosis lists the specific things the model could
  see and verify in the image, shown as evidence in the interface.

- **What could not be confirmed.** Separately, the diagnosis records what the image
  could not establish — for example the depth of a crack, the material of a pipe, or
  whether damage extends behind a surface. These uncertainties are surfaced
  explicitly rather than silently assumed away.

- **Why clarification is needed.** When the system asks questions, it explains what
  is unclear and why the answers would change its recommendation, and it offers
  practical guidance on the kind of photograph that would help most.

Confidence informs the explanation the user sees, but it is the structured risk
signals — not the confidence number alone — that decide how the system responds.
This separation is deliberate: it keeps the system from proceeding simply because a
model reported a high number, and from halting simply because it reported a low one.

---

## 7. Ask Knitra — Contextual Chat Assistant

Ask Knitra is a per-case conversational assistant that opens once an analysis exists
for the case (subject to the safety gating described in §5). It is never a generic
chatbot: every conversation is anchored to the specific case, so the assistant
already knows the problem, its severity, what the model could and could not see, and
— for cases with a plan — the tools, materials, and every repair step.

This means a user can ask grounded follow-up questions such as "what if I don't have
the right tool for step three?" or "is this step dangerous?" and receive answers that
refer to the actual plan rather than generic advice.

The assistant supports:

- **Voice input.** The user can dictate a message; audio is transcribed locally and
  placed into the message box.

- **Follow-up images.** Additional photographs can be attached mid-conversation; the
  vision model describes them and that description informs the assistant's reply.

- **Scope awareness.** Off-topic requests are redirected back to the repair case, and
  the assistant will not help circumvent an active safety warning.

The full conversation is stored per case, so context is preserved across sessions.

---

## 8. Technology Stack

| Component | Technology |
|---|---|
| Backend framework | FastAPI (async) |
| Language | Python 3.12 |
| ORM | SQLAlchemy 2.0 (async) |
| Database | PostgreSQL (via Docker) |
| Migrations | Alembic |
| Frontend | NiceGUI |
| Schema validation | Pydantic v2 |
| Vision model | Qwen2.5-VL (7B default; 3B available as a lighter alternative) |
| Planning model | Mistral 7B Instruct |
| Generation and chat model | Llama 3.1 8B |
| Inference runtime | Ollama (local) |
| Speech-to-text | faster-whisper (CPU) |
| Image processing | Pillow |
| Authentication | Bearer token with hashed credentials |
| Containerisation | Docker Compose |

All models are open source and run locally through Ollama.

---

## 9. Installation and Setup

### Prerequisites

- Python 3.12 or newer
- Docker Desktop (for PostgreSQL)
- [Ollama](https://ollama.com), installed and running locally
- Node.js is **not** required — the interface ships its own assets

### 1. Start the database

```bash
docker compose up -d db
```

### 2. Pull the models

This is a one-time download of several gigabytes. After this, all inference is local.

```bash
ollama pull qwen2.5vl:7b          # vision diagnosis (default)
ollama pull qwen2.5vl:3b          # optional lighter vision model for low-memory machines
ollama pull mistral:7b-instruct   # planning
ollama pull llama3.1:8b           # generation and chat
```

Confirm Ollama is reachable:

```bash
curl http://localhost:11434/api/tags
```

### 3. Start the backend

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

### 4. Start the frontend

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

### Configuration

Local configuration — the database connection, model assignments, and inference
tuning — is read from an environment file in the backend, and sensible defaults are
applied in code. A sample is provided; on a typical 16 GB machine the defaults work
without changes. On memory-constrained machines, switching the vision model to the
lighter 3B variant reduces resource use at some cost to diagnostic quality.

> **A note on the vision model.** The 7B vision model is the default because it
> produces noticeably better diagnoses and category reasoning than the 3B model, at
> the cost of higher memory use and slower runs. The 3B model remains available as a
> drop-in alternative for lower-specification hardware.

### Convenience scripts (Windows)

PowerShell scripts are provided at the project root to start the full stack, stop it,
and manage it interactively, for development on Windows.

### Service URLs

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend | http://localhost:8000 |
| API documentation | http://localhost:8000/docs |
| Ollama | http://localhost:11434 |
| PostgreSQL | localhost:5433 |

---

## 10. API Reference

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
| `GET` | `/api/v1/cases` | List cases (newest first, with confidence) |
| `GET` | `/api/v1/cases/{id}` | Retrieve a single case |
| `PATCH` | `/api/v1/cases/{id}` | Update title, category, or status |
| `DELETE` | `/api/v1/cases/{id}` | Delete a case and its associated data |

### Uploads

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/cases/{id}/uploads` | Upload an image to a case |
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

## 11. Current Limitations

Knitra is a research prototype. The following limitations are stated plainly so that
the system's scope is not overstated.

- **Diagnosis depends on image quality.** Performance is bounded by what is visible
  in the photograph. Blurry, distant, poorly lit, or partial images reduce
  diagnostic quality. The clarification and uncertainty mechanisms are designed to
  surface this rather than hide it, but they cannot recover information the image
  does not contain.

- **Video is not yet analysed.** Video files can be uploaded, but the pipeline
  currently processes still images only. Frame extraction and multi-frame reasoning
  are planned but not implemented.

- **Clarification answers are matched sequentially.** Answers are currently paired
  with questions in order. If a user answers out of sequence, the clarification
  history can occasionally display the pairing incorrectly. The full set of answers
  is still used for the diagnosis, so the assessment itself is not compromised, but
  the displayed history may not reflect the intended pairing.

- **Confidence explanations do not yet incorporate clarification answers.** After a
  clarification round, the diagnosis is updated correctly, but the accompanying
  confidence and uncertainty wording is not yet rephrased to explicitly reference
  what the user told the system.

- **Hazard safeguards are primarily prompt-level and interface-level.** The hazard
  acknowledgement gate and the assistant's refusal to bypass hazards are enforced
  through the interface and through model instruction. These are effective in normal
  use but are not equivalent to a hard, independently enforced server-side block.

- **Local model accuracy.** The models are open-source models running locally. They
  perform well on common, clearly photographed domestic faults but may misclassify
  unusual faults, uncommon materials, or cases requiring specialist visual judgement.
  Because vision inference is non-deterministic, borderline cases can occasionally be
  routed differently across runs; the system errs toward caution when uncertain.

---

## 12. Future Work

The following directions extend the system's usefulness while staying within its
trust-first design philosophy.

- **Sustainability preferences.** A user-controlled option to bias guidance toward
  repair over replacement and toward lower-impact materials and methods, so
  environmental cost becomes an explicit factor in the advice.

- **Nearby professional services.** Where the system recommends professional
  assessment, suggest appropriately qualified local tradespeople based on the trade
  category, turning a recommendation into an actionable next step.

- **Tool and material recommendations.** Connect the required tools and materials to
  concrete product suggestions — for example brand, size, and supplier — so a user
  can move from plan to purchase.

- **Video understanding.** Explore short-clip analysis using open-source pretrained
  video or multi-frame models, allowing the system to reason about faults that are
  hard to capture in a single still (intermittent drips, movement, sound cues).

- **Stronger explainable AI.** Investigate visual explanation techniques such as
  saliency maps or local interpretable model-agnostic explanations (LIME) to ground
  *why* a fault was classified as it was, complementing the existing textual evidence
  and uncertainty reporting.

- **Improved clarification reasoning and answer matching.** Move from sequential
  answer matching to content-aware matching so out-of-order answers are handled
  correctly, and rephrase confidence and uncertainty explanations to explicitly
  reflect the user's clarification answers.

---

## 13. Design Principles

A few decisions shape the system and are worth stating explicitly for anyone
reviewing or extending it.

**Decide before advising.** The system treats "should I produce a repair plan?" as a
distinct decision from "what is the repair plan?". Separating these is what allows
Knitra to ask, escalate, or decline rather than always answering.

**Route on structured signals, not a single number.** Outcomes are chosen from a set
of explicit risk and suitability signals rather than a lone confidence threshold.
This keeps the system from proceeding just because a model sounded confident, or
stopping just because it sounded unsure, and makes its behaviour reproducible and
reviewable.

**Specialise each stage.** Diagnosis, planning, and step generation are
fundamentally different tasks. Using a separate, appropriate model for each — and
giving each only the context it needs — keeps every stage focused and allows any one
model to be upgraded or replaced independently.

**Keep the official result authoritative.** Conversation is for understanding and
clarification; it never silently rewrites the assessment a user acts on. Only a full,
validated re-analysis updates the official result.

**Run locally, by default.** All inference — vision, language, and speech — runs on
the user's own machine. Repair photographs of someone's home are sensitive, and
keeping them local is a design requirement rather than an afterthought.

**Be honest about uncertainty.** Surfacing what the system cannot determine, bounding
how long it will keep asking, and presenting an inconclusive outcome when warranted
are treated as core features, not edge cases.
```

