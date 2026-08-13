# 🩻 RadAssist AI

**An Explainable, Retrieval-Augmented Radiology Reporting & Clinical Decision Support System**

RadAssist AI answers radiology questions from a curated medical knowledge base, with inline citations linking every claim back to its source. The differentiator isn't the chat interface — it's **traceability**. The model may only answer from retrieved evidence, must cite each factual claim, and is explicitly forbidden from asserting a diagnosis.

> **Not a diagnostic device.** Research and educational prototype. It assists; it does not replace clinical judgement.

---

## 📋 Status

| Phase | Name | Status |
|-------|------|--------|
| 1 | Architecture & Environment Setup | ✅ Complete |
| 2 | Knowledge Base & Ingestion Pipeline | ✅ Complete |
| 3 | Core RAG + LLM Chat (MVP) | ✅ Complete |
| 4 | Multimodal Ingestion | ✅ Complete |
| 5 | Decision Support Features | ✅ Complete |
| 6 | Explainability, Auth & Hardening | ✅ Complete |
| 7 | Deployment & Pilot | 🔮 Planned |

**Current corpus:** 296 PMC articles · 20,811 chunks
**Tests:** 571 passing in ~35s (no database, network, or model download required)
**Retrieval:** keyword@5 94.4% · MRR 0.866 · ~4.6s end to end

Retrieval was measured, not assumed. Vector-only scored 55.6% keyword@5; adding
a cross-encoder reranker took it to 72.2%, hybrid BM25 to 77.8%, and contextual
chunk headers to the current figure. Each step was kept only because the
evaluation said it earned its latency.

---

## 🏗️ Architecture

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Next.js 16 + React 19 + Tailwind 4 | Chat UI, streaming, evidence panel, report editor |
| **Backend** | FastAPI + SQLAlchemy 2.0 (async) | API, RAG orchestration, LLM integration |
| **Database** | PostgreSQL 16 + Alembic | Document metadata, reports, users, audit |
| **Vector DB** | Qdrant 1.10 | Semantic search (384-dim, cosine) |
| **Embeddings** | all-MiniLM-L6-v2 | Query and chunk vectors |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 | Second-stage precision over recalled candidates |
| **Lexical** | BM25 | Exact-term recall, unioned with vector search |
| **Vision** | `qwen/qwen3.6-27b` via Groq | Reading photographed reports, Tesseract as fallback |
| **LLM** | Groq → Mistral → OpenAI | Answer generation, automatic failover |
| **Auth** | JWT (HS256, pinned) + bcrypt | Per-user ownership of reports and images |

**Why two databases?** Postgres answers *"which documents exist and what's their status"* — pagination, filtering, audit. Qdrant answers *"what text means the same as this question"*. A document UUID links them.

### Query pipeline

Two-stage retrieval: a cheap bi-encoder recalls broadly, then an expensive
cross-encoder reorders precisely. Running the cross-encoder over the whole
corpus would be accurate and unusably slow; running only the bi-encoder is fast
and puts the right passage at rank 7 instead of rank 1.

```
question
  ├─▶ out-of-scope filter ──────▶ redirect (no embedding, no LLM call)
  │
  ├─▶ RECALL ─┬─▶ embed + Qdrant search      (semantic: passages ABOUT it)
  │           └─▶ BM25 lexical search        (exact: passages CONTAINING it)
  │              └─▶ union → 48 candidates
  │
  ├─▶ RERANK ──▶ cross-encoder scores each (query, chunk) pair jointly
  │              └─▶ keep top 12
  │
  ├─▶ cap 3 chunks per document   ← stops one paper monopolising context
  ├─▶ merge adjacent chunks       ← removes overlap duplication
  ├─▶ best score < 0.35? ────────▶ "not in my knowledge base" (no LLM call)
  ├─▶ mode-specific scaffold + [Source N] blocks
  └─▶ stream tokens, normalise citations, emit SSE
```

Chunks carry a **contextual header** — their document title and section — so a
passage saying "this finding is typically unilateral" still retrieves for the
condition it belongs to. That change alone moved keyword@5 from 77.8% to 94.4%.

Two stages exist specifically to **avoid answering**: the out-of-scope filter
and the relevance threshold both terminate before any LLM call. For a clinical
tool, a confident wrong answer is worse than a refusal.

### Modes

The same pipeline serves three prompts, and they are deliberately not
interchangeable — the general scaffold *mandates* inline citations, while report
mode *forbids* them, because a citation marker in a draft a radiologist signs
would end up in the patient record.

| Mode | Purpose | Hard constraint |
|---|---|---|
| `qa` | Answer a question from the corpus | Every claim carries `[Source N]` |
| `report` | Turn dictated findings into a structured draft | Never add a finding; never alter a number, level or laterality |
| `comparison` | Compare a prior study against a current one | Absence of a finding is not resolution of it |

---

## 🚀 Quick Start

```bash
cp backend/.env.example backend/.env
```

Then fill in two things — the app **will not start** without them:

- `JWT_SECRET` — `openssl rand -hex 32`. There is no default, deliberately: a
  signing key with a fallback is one that reaches production unchanged.
- One LLM key (`GROQ_API_KEY` is free and the default provider).

```bash
docker-compose up --build
```

Open http://localhost:3000 and **create an account**. Every clinical route
requires a token, so the knowledge-base calls below need one too:

```bash
TOKEN=$(curl -s -X POST localhost:8000/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"you@example.com","password":"..."}' | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4)

curl -X POST localhost:8000/api/v1/knowledge/seed      -H "Authorization: Bearer $TOKEN"
curl -X POST localhost:8000/api/v1/knowledge/fetch-pmc -H "Authorization: Bearer $TOKEN"
```

| Service | URL |
|---|---|
| Chat UI | http://localhost:3000 |
| API docs | http://localhost:8000/docs |
| Health | http://localhost:8000/api/v1/health (open, no token) |
| Qdrant dashboard | http://localhost:6333/dashboard |

Run the tests: `docker-compose exec backend python -m pytest -q`.

### Stop

```bash
docker-compose down       # containers only, data preserved
docker-compose down -v    # also deletes Postgres + Qdrant data
```

The embedding model cache lives in `backend/.hf_cache` on the host, so `down -v` does **not** delete it.

---

## 📚 Knowledge Base

| Source | `source_type` | Verifiable? |
|---|---|---|
| PubMed Central Open Access | `pmc_open_access` | ✅ PMID, DOI, clickable URL |
| StatPearls via NCBI | `statpearls` | ✅ PMID |
| Curated radiology summaries | `curated_summary` | ❌ no PMID/DOI — see below |

The 14 curated entries are **summaries written to reflect** their named sources, not verbatim extracts. They carry no PMID or DOI, so they're labelled `curated_summary` rather than `textbook` — calling them primary sources in the evidence panel would overstate what they are. A test enforces this.

RadioPaedia is the reference standard clinicians use, but it prohibits bulk scraping. PMC Open Access is used instead: explicitly licensed for reuse, and every article verifiable.

---

## ⚙️ Configuration

`backend/.env.example` documents every setting with its default. The ones that
matter most are below; the rest are annotated in the file itself.

| Provider | Env var | Free tier | Default model |
|---|---|---|---|
| Groq | `GROQ_API_KEY` | ✅ | `openai/gpt-oss-120b` |
| Mistral | `MISTRAL_API_KEY` | ✅ limited | `mistral-large-latest` |
| OpenAI | `OPENAI_API_KEY` | ❌ | `gpt-4o-mini` |

> ⚠️ **Groq retires models aggressively.** `llama-3.3-70b-versatile` shut down 2026-08-16. If generation breaks, check [Groq deprecations](https://console.groq.com/docs/deprecations). A test guards against defaulting to a known-retired ID.

`NCBI_EMAIL` must be a real address for PMC/StatPearls ingestion — NCBI's terms require a contactable email, and placeholder values are detected and rejected.

### Authentication

| Env var | Purpose |
|---|---|
| `JWT_SECRET` | Token signing key. **No default** — the app refuses to start without one, and rejects anything under 32 characters or containing an obvious placeholder. Generate with `openssl rand -hex 32`. |
| `JWT_EXPIRE_MINUTES` | Defaults to `720` — twelve hours, one clinical shift. There are no refresh tokens, so expiry means signing in again. |
| `ALLOW_REGISTRATION` | `true` opens self-service signup. Set **`false`** for any clinical deployment; accounts then come from `python scripts/create_user.py`. |
| `MIN_PASSWORD_LENGTH` | Defaults to `12`. bcrypt silently truncates past 72 **bytes**, so over-long passwords are rejected rather than quietly shortened. |

Every route except `/health` and `/auth/*` requires a bearer token. Reports and
images carry an owner, and requesting someone else's returns **404, not 403** —
403 would confirm the row exists, which turns a list of UUIDs into a directory.

> ⚠️ Open registration is only safe **because** of that ownership scoping.
> `backend/tests/test_ownership.py` is what enforces it. If those tests are ever
> weakened, `ALLOW_REGISTRATION` has to go `false` in the same commit.

---

## 🐛 Known Limitations

- **PMC figures are not stored.** Ingestion recorded PMIDs rather than PMCIDs, so figure URLs can't be constructed. The fetch pipeline is built and tested; the fix is capturing the PMCID at ingest and re-running the corpus.
- **Rate limit state is in-process.** It resets on restart and doesn't hold across replicas. Redis is the correct fix, deferred to Phase 7.
- **No password reset.** A reset flow is another credential path needing the same protection as the password itself. For a pilot, recovery is an operator recreating the account.
- **Single-turn.** No conversation history, so follow-ups lose context.
- **No frontend tests.** The 571 tests are all backend.
- **General-purpose embeddings.** MiniLM isn't medical-domain; PubMedBERT (768-dim) is the intended upgrade.

---

## 📁 Project Structure

```
RAG/
├── backend/
│   ├── app/
│   │   ├── main.py            # App entry, startup checks
│   │   ├── config.py          # Environment settings
│   │   ├── api/v1/endpoints/  # health, auth, knowledge, chat,
│   │   │                      # images, reports
│   │   ├── core/              # DB, security, deps, limits, errors
│   │   ├── models/            # SQLAlchemy tables
│   │   ├── schemas/           # Pydantic request/response
│   │   ├── data/              # curated seed knowledge
│   │   └── services/          # embedding, qdrant, ingestion, llm,
│   │                          # rag, pmc_fetcher, lexical, reranker,
│   │                          # vision, dicom, comparison, quality
│   ├── alembic/versions/      # 0001–0006, idempotent
│   ├── eval/                  # retrieval harness + question set
│   ├── scripts/create_user.py # operator account creation
│   ├── tests/                 # 571 tests
│   └── Dockerfile
├── frontend/src/
│   ├── app/                   # chat page, login, markdown renderer
│   ├── components/            # ReportEditor, AuthedImage
│   └── lib/                   # api.ts (SSE client), auth.ts
├── docker-compose.yml
└── refetch_corpus.ps1         # rebuild the corpus from scratch
```

---

## 📝 License

Developed as an internship deliverable.
