<h1 align="center">🩻 RadAssistant</h1>

<p align="center">
  <strong>A radiology assistant that cites every claim, refuses what it can't support,<br>and runs entirely inside your own network.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/tests-596%20passing-2C6380?style=flat-square" alt="596 tests passing">
  <img src="https://img.shields.io/badge/keyword%405-94.4%25-2C6380?style=flat-square" alt="keyword@5 94.4%">
  <img src="https://img.shields.io/badge/corpus-20%2C811%20chunks-2C6380?style=flat-square" alt="20,811 chunks">
  <img src="https://img.shields.io/badge/deploy-docker%20compose-2C6380?style=flat-square" alt="Docker Compose">
</p>

<p align="center">
  <sub>Research and educational prototype. Not a diagnostic device — it assists, it does not replace clinical judgement.</sub>
</p>

---

## Demo

https://github.com/user-attachments/assets/34c50e5f-efaa-4a9e-b69a-4dbb374919f6

<sub>Ask a question · read an uploaded report · draft and sign off a structured report · compare against a prior study · watch it decline something outside its corpus.</sub>

---

## Not another RAG chatbot

**Every claim is traceable.**
Answers carry `[Source N]` markers. Click one and it scrolls to the exact passage it came from, with its relevance score. Nothing is asserted that isn't in a retrieved chunk.

**It is built to refuse.**
Two stages terminate *before* any LLM call — an out-of-scope filter and a relevance threshold. For a clinical tool, a confident wrong answer is worse than a refusal.

**It runs where the data lives.**
One Docker Compose stack — API, Postgres, Qdrant, reverse proxy — with no dependency on any external service beyond the LLM provider. Hospitals legally cannot put patient data on a hosted database. This is built for that constraint.

---

## Quick start

```bash
git clone https://github.com/MominaZahid10/RadAssistant.git
cd RadAssistant
cp backend/.env.example backend/.env    # add GROQ_API_KEY (free) + JWT_SECRET
docker compose up --build
```

Open **http://localhost:3000**, create an account, then seed the knowledge base:

```bash
curl -X POST localhost:8000/api/v1/knowledge/seed -H "Authorization: Bearer $TOKEN"
```

Production deployment — TLS, hardening, self-hosted or split — in **[DEPLOYMENT.md](DEPLOYMENT.md)**.

---

## How it works

```
question
  ├─▶ out-of-scope filter ──────▶ redirect        (no embedding, no LLM call)
  │
  ├─▶ RECALL ─┬─▶ Qdrant vector search   (passages ABOUT it)
  │           └─▶ BM25 lexical search    (passages CONTAINING it)
  │              └─▶ union → 48 candidates
  │
  ├─▶ RERANK ──▶ cross-encoder scores each (query, chunk) pair → top 12
  ├─▶ cap 3 chunks per document, merge adjacent
  ├─▶ best score < 0.35? ───────▶ "not in my knowledge base"  (no LLM call)
  └─▶ stream tokens with inline citations
```

A cheap bi-encoder recalls broadly; an expensive cross-encoder reorders precisely. Running the cross-encoder over the whole corpus would be accurate and unusably slow. Running only the bi-encoder puts the right passage at rank 7 instead of rank 1.

> **Retrieval was measured, not assumed.**
> Vector-only scored 55.6% keyword@5. The reranker took it to 72.2%, hybrid BM25 to 77.8%, and contextual chunk headers to **94.4%**. Each stage was kept only because the evaluation said it earned its latency.

---

## Three modes, deliberately not interchangeable

| Mode | Purpose | Hard constraint |
|:--|:--|:--|
| **Ask** | Answer from the corpus | Every claim carries `[Source N]` |
| **Draft** | Dictated findings → structured report | Never add a finding; never alter a number, level or laterality |
| **Compare** | Prior study vs current | Absence of a finding is not resolution of it |

Ask *mandates* citations. Draft *forbids* them — a `[Source 3]` in a report a radiologist signs would end up in the patient record. In Compare, measurements are paired and differenced in Python before the model sees them, so it narrates settled arithmetic rather than computing it.

Answers also adapt to the reader: **Attending** gets concise standard terminology, **Resident** gets step-by-step reasoning with terms defined.

---

## Stack

| Layer | Technology |
|:--|:--|
| Frontend | Next.js 16 · React 19 · Tailwind 4 |
| Backend | FastAPI · SQLAlchemy 2.0 async · Alembic |
| Databases | PostgreSQL 16 · Qdrant 1.10 (384-dim, cosine) |
| Retrieval | all-MiniLM-L6-v2 · ms-marco-MiniLM-L-6-v2 reranker · BM25 |
| LLM | Groq → Mistral → OpenAI, automatic failover |
| Vision | `qwen3.6-27b` for report photos, Tesseract fallback |
| Auth | JWT (HS256, pinned) · bcrypt · per-user ownership |
| Edge | Caddy 2 — TLS, routing, SSE pass-through |

Two databases because they answer different questions. Postgres knows *which documents exist and what state they're in*; Qdrant knows *what text means the same as this question*. A UUID links them.

---

## Testing

```bash
docker compose exec backend python -m pytest -q    # 571 backend tests, ~35s
cd frontend && npm run test:e2e                    # 25 Playwright tests, real stack
```

The backend suite needs no database, network or model download — safe to run anywhere, including CI on a cold machine. The E2E suite drives a real browser against real Postgres, real Qdrant and a real model, including switching conversations mid-answer, which is the race that broke the first implementation.

---

## Known limitations

| | |
|:--|:--|
| **Rate-limit counters are in-process** | They reset on restart. A provider-side spend cap is the real bound. |
| **`vision_service.py` is Groq-only** | Unlike the LLM service, it has no fallback chain. |
| **Single-turn retrieval** | Conversations persist, but each question retrieves independently, so follow-ups lose prior context. |
| **General-purpose embeddings** | MiniLM isn't medical-domain; PubMedBERT is the intended upgrade. |
| **PMC figures aren't stored** | Ingestion recorded PMIDs rather than PMCIDs, so figure URLs can't be reconstructed without re-running the corpus. |
| **No monitoring** | No uptime checks or metrics export. |

---

## License

© 2026 Momina Zahid. All rights reserved.
