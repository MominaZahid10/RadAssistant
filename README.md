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
| 4 | Multimodal Ingestion | 🔮 Planned |
| 5 | Decision Support Features | 🔮 Planned |
| 6 | Explainability, Auth & Hardening | 🔮 Planned |
| 7 | Deployment & Pilot | 🔮 Planned |

**Current corpus:** 232 documents · 10,494 chunks · 0 failed
**Tests:** 167 passing in ~10s (no database, network, or model download required)

---

## 🏗️ Architecture

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Next.js 16 + React 19 + Tailwind 4 | Chat UI, streaming, evidence panel |
| **Backend** | FastAPI + SQLAlchemy 2.0 (async) | API, RAG orchestration, LLM integration |
| **Database** | PostgreSQL 16 | Document metadata, status, audit |
| **Vector DB** | Qdrant 1.10 | Semantic search (384-dim, cosine) |
| **Embeddings** | all-MiniLM-L6-v2 | Query and chunk vectors |
| **LLM** | Groq → Mistral → OpenAI | Answer generation, automatic failover |

**Why two databases?** Postgres answers *"which documents exist and what's their status"* — pagination, filtering, audit. Qdrant answers *"what text means the same as this question"*. A document UUID links them.

### Query pipeline

```
question
  ├─▶ out-of-scope filter ──────▶ redirect (no embedding, no LLM call)
  ├─▶ embed + Qdrant search (top 48)
  ├─▶ cap 3 chunks per document   ← stops one paper monopolising context
  ├─▶ merge adjacent chunks       ← removes overlap duplication
  ├─▶ best score < 0.35? ────────▶ "not in my knowledge base" (no LLM call)
  ├─▶ grounding scaffold + register + [Source N] blocks
  └─▶ stream tokens, normalise citations, emit SSE
```

Two stages exist specifically to **avoid answering**. For a clinical tool, a confident wrong answer is worse than a refusal.

---

## 🚀 Quick Start

```bash
docker-compose up --build

curl -X POST localhost:8000/api/v1/knowledge/seed        # 14 curated articles
curl -X POST localhost:8000/api/v1/knowledge/fetch-pmc   # peer-reviewed full text
```

| Service | URL |
|---|---|
| Chat UI | http://localhost:3000 |
| API docs | http://localhost:8000/docs |
| Health | http://localhost:8000/api/v1/health |
| Qdrant dashboard | http://localhost:6333/dashboard |

Verify everything: `bash verify_phase3.sh` (27 end-to-end checks) and `cd backend && pytest tests/ -v`.

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

Copy `backend/.env.example` → `backend/.env`.

| Provider | Env var | Free tier | Default model |
|---|---|---|---|
| Groq | `GROQ_API_KEY` | ✅ | `openai/gpt-oss-120b` |
| Mistral | `MISTRAL_API_KEY` | ✅ limited | `mistral-large-latest` |
| OpenAI | `OPENAI_API_KEY` | ❌ | `gpt-4o-mini` |

> ⚠️ **Groq retires models aggressively.** `llama-3.3-70b-versatile` shut down 2026-08-16. If generation breaks, check [Groq deprecations](https://console.groq.com/docs/deprecations). A test guards against defaulting to a known-retired ID.

`NCBI_EMAIL` must be a real address for PMC/StatPearls ingestion — NCBI's terms require a contactable email, and placeholder values are detected and rejected.

---

## 🐛 Known Limitations

- **Retrieval precision at scale.** Scaling to 232 documents surfaced a real weakness: term-dense research papers outrank explanatory passages on definitional queries. Per-document capping helps; the proper fix is two-stage retrieval with a cross-encoder reranker. **Next priority.**
- **No retrieval evaluation yet.** Chunk size, overlap, and the 0.35 threshold are reasoned defaults, not measured. An evaluation harness (~15 question/expected-source pairs, recall@5) is the prerequisite for proving any retrieval change actually helps.
- **No authentication.** Upload and delete are open.
- **No Alembic migrations.** `create_all` creates tables but never alters them.
- **Single-turn.** No conversation history, so follow-ups lose context.
- **General-purpose embeddings.** MiniLM isn't medical-domain; PubMedBERT (768-dim) is the intended upgrade.

---

## 📁 Project Structure

```
RAG/
├── backend/
│   ├── app/
│   │   ├── main.py            # App entry, startup checks
│   │   ├── config.py          # Environment settings
│   │   ├── api/v1/endpoints/  # health, knowledge, chat
│   │   ├── core/              # async DB connection
│   │   ├── models/            # SQLAlchemy tables
│   │   ├── schemas/           # Pydantic request/response
│   │   ├── data/              # curated seed knowledge
│   │   └── services/          # embedding, qdrant, ingestion,
│   │                          # llm, rag, pmc_fetcher
│   ├── tests/                 # 167 tests
│   └── Dockerfile
├── frontend/src/
│   ├── app/                   # chat page, markdown renderer
│   ├── components/
│   └── lib/api.ts             # SSE streaming client
├── docker-compose.yml
├── verify_phase3.sh           # 27 end-to-end checks
└── PROJECT_UPDATE.md          # status report
```

---

## 📝 License

Developed as an internship deliverable.
