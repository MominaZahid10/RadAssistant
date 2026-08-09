# RadAssist AI — Phase 1 & 2 Code Review

> **STATUS: all backend blockers and should-fix items in this document have been
> fixed and verified (50 passing tests in `backend/tests/`).** Two items remain
> open by choice: the Phase 2 frontend UI, and Alembic migrations. See
> `FIXES_APPLIED.md` for what changed and why.

**Reviewed:** 9 Aug 2026 · backend/, frontend/, docker-compose.yml against the Phase 1 & Phase 2 plans.

**Verdict:** Architecture is solid and above typical internship level. Layering (config → core → models → schemas → services → api) is correct, async SQLAlchemy is used properly, and the Postgres-for-metadata / Qdrant-for-vectors split is the right call. But there are **4 bugs that will break at runtime or silently corrupt the knowledge base**, and Phase 2 Step 10 (seeding) was written but never wired up — so the knowledge base is currently empty and unreachable.

---

## 🔴 Blockers — fix before demoing

### 1. Tesseract is not installed → OCR silently poisons the knowledge base
`backend/Dockerfile` installs `build-essential` only. `pytesseract` is a *wrapper*; the `tesseract` binary is a system package.

In the container, `parse_image()` raises `TesseractNotFoundError`, which is caught by the broad `except Exception` and returns the string `"[OCR failed: ...]"`. That string is ~30 chars, so it passes `ingestion.py`'s `len(raw_text.strip()) < 10` check, gets embedded, stored in Qdrant, and the document is marked **`completed`**. You now have junk vectors that will surface in RAG retrieval.

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*
```

And make the failure loud — in `parse_image`, `raise` instead of returning a placeholder string, or have `ingest_document` reject text starting with `[OCR`.

### 2. The knowledge seeder is dead code
`services/knowledge_seeder.py` (Phase 2 Step 10, ~330 lines) is imported by **nothing**. There is no `/knowledge/seed` endpoint and no startup call. Your 14 curated radiology articles never reach Qdrant.

Add to `endpoints/knowledge.py`:

```python
@router.post("/seed", summary="Seed the knowledge base")
async def seed(db: AsyncSession = Depends(get_db)):
    return await seed_knowledge_base(db)
```

Keep it manual (not on startup) — startup seeding makes container restarts slow and unpredictable.

### 3. Ingestion blocks the entire event loop
`upload_document` is `async def`, but `embedding_service.encode()` is synchronous, CPU-bound PyTorch. Calling it directly inside an async handler blocks the whole server — a 50 MB PDF freezes every other request, including `/health`, for minutes. Same problem in `parse_pdf`.

The `status="processing"` field also does nothing as written, since you only return after all the work is finished.

Minimum fix:
```python
from fastapi.concurrency import run_in_threadpool
vectors = await run_in_threadpool(embedding_service.encode, chunks, ...)
```
Better (and much stronger for the report): return `201 {status: "processing"}` immediately via `BackgroundTasks`, and let the frontend poll `GET /documents/{id}`. That's what the status column was designed for.

### 4. Chunk pagination is wrong
`qdrant_service.get_chunks_by_document()` passes an integer to `client.scroll(offset=...)`. Qdrant's scroll `offset` is a **point ID to resume from**, not a row count. Page 2 will return wrong or duplicate results. Also, `chunks.sort()` only sorts within a page, so global order isn't guaranteed.

Simplest correct fix: scroll everything for that document (`limit=10_000`), sort by `chunk_index`, then slice in Python.

---

## 🟠 Should fix

| # | Issue | Where | Why it matters |
|---|---|---|---|
| 5 | **`.env` committed with placeholder secrets, no `.env.example`** | `backend/.env` | The file header literally says "This .env.example is safe to commit" but it's named `.env`. Rename to `.env.example` and create a real gitignored `.env`. |
| 6 | **`NCBI_EMAIL=your_email@example.com` is truthy** | `backend/.env` | `seed_knowledge_base` gates NCBI fetching on `if settings.NCBI_EMAIL:` — a placeholder passes the check and you'll hammer NCBI with a fake email. Leave it empty. |
| 7 | **No payload indexes in Qdrant** | `qdrant_service.py` | Every `delete_by_document` and `source_type` filter does a full collection scan. Add `create_payload_index` on `document_id` and `source_type` in `ensure_collection()`. Cheap, and it's exactly the kind of detail that reads as "knows vector DBs." |
| 8 | **`qdrant/qdrant:latest` unpinned vs `qdrant-client==1.10.0`** | `docker-compose.yml` | Server/client drift will break you silently. Pin to a matching minor, e.g. `qdrant/qdrant:v1.10.1`. Related: `info.vectors_count` returns `None` on recent servers, so `/stats` will report null. Use `points_count`. |
| 9 | **`client.search()` is deprecated** | `qdrant_service.py:233` | Migrate to `client.query_points(...)` — it's the supported path going forward. |
| 10 | **Alembic installed but unused** | `requirements.txt` / `main.py` | You use `Base.metadata.create_all`, which creates tables but **never alters existing ones**. When Phase 3 adds columns to `documents`, nothing happens and you'll debug it for an hour. Either run `alembic init` now or drop alembic from requirements and note the tradeoff. |
| 11 | **`package-lock.json` is gitignored** | `.gitignore:28` | Your Dockerfile does `COPY package.json package-lock.json* ./` expecting it. Ignoring lockfiles means non-reproducible builds. Remove that line. |
| 12 | **HuggingFace cache not persisted** | `docker-compose.yml` | The 80 MB model re-downloads on every fresh container. Add `- hf_cache:/root/.cache/huggingface` to the backend volumes. |
| 13 | **`UPLOAD_DIR=/app/uploads` writes into your source tree** | compose mounts `./backend:/app` | Uploads land in `backend/uploads/` on your host and aren't gitignored. Either use a named volume or add it to `.gitignore`. (You also never actually write files there — `upload_document` works purely from bytes. Decide which you want.) |
| 14 | **`--reload` baked into the Dockerfile CMD** | `backend/Dockerfile:45` | Dev-only flag hardcoded into the image. Move it to a `command:` override in compose so the image stays deployable. |
| 15 | **`list[str]` settings will crash if ever set via env** | `config.py` (`CORS_ORIGINS`, `ALLOWED_EXTENSIONS`) | pydantic-settings v2 runs `json.loads` on complex types. `CORS_ORIGINS=http://x` in `.env` → startup crash. Works today only because they're absent from `.env`. Add a `field_validator` that splits on commas. |
| 16 | **New `QdrantClient` per health check** | `health.py:72` | Reuse `qdrant_service.client` instead of constructing a connection on every request. |
| 17 | **No auth on upload/delete** | all endpoints | For a clinical tool this is worth naming explicitly as a known gap in your report, even if you defer it. |

---

## 🟡 Phase 2 delivered no frontend

`frontend/src/lib/api.ts` still only has `getHealth()` and `getRoot()`. `page.tsx` is a mock chat with a hardcoded `setTimeout` reply. You have seven working knowledge endpoints and **zero UI touching them**.

The Phase 2 plan technically didn't ask for frontend work — but for an internship deliverable, a page that uploads a PDF and runs a real semantic search against `/knowledge/search` is dramatically more convincing than a fake chatbot. It's maybe 150 lines and it makes Phase 2 *visible*.

---

## 🟢 What's genuinely good

- **Correct separation of concerns.** Services don't import FastAPI; endpoints don't touch Qdrant internals directly. Many student projects put everything in `main.py`.
- **Async SQLAlchemy 2.0 done right** — `async_sessionmaker`, `expire_on_commit=False`, dependency-injected sessions with cleanup.
- **Postgres + Qdrant dual store** with a UUID linking them. The reasoning in `models/document.py` is correct and well-argued.
- **`ensure_collection()` is idempotent and dimension-checks** on startup — that's a real production instinct.
- **Re-ingestion is safe** (`delete_by_document` before upsert). Easy thing to miss.
- **Batched upserts** at 100 points, `normalize_embeddings=True` matched to COSINE distance. Both correct.
- **Recursive character splitting written from scratch** instead of importing LangChain. For an internship this is a *plus* — you can explain every line of it.
- **Health check tests dependencies, not just liveness**, and returns 503 correctly.
- **Seed content quality is high** — the ABCDEFGHI and RIPE mnemonics, Fleischner criteria, and CTPA findings are accurate and properly attributed.
- **Configurable embedding model** with a documented swap path to PubMedBERT.

---

## Is this "technically good" for an internship?

Yes — the design is right, and the design is the part that's hard to fix later. What's currently missing is the **operational rigor** that separates a good student project from a professional one:

1. **Zero tests.** This is the single biggest gap. Even 6 pytest cases on `chunk_text` (empty input, text shorter than chunk_size, overlap correctness, `chunk_overlap >= chunk_size` guard) plus one `httpx.AsyncClient` test on `/health` would meaningfully change how this reads.
2. **Broad `except Exception` swallowing failures.** Bug #1 exists *because* errors are converted into success paths. Catch specific exceptions; let unexpected ones bubble.
3. **No `README.md`** with setup steps, architecture diagram, and known limitations. Reviewers read this first.
4. **No evaluation of the RAG itself.** Before Phase 3, write ~15 question/expected-source pairs and measure recall@5 against your seeded corpus. That gives you a number to defend chunk size, overlap, and the eventual MiniLM → PubMedBERT swap — and "I measured it" is the strongest thing you can say in an internship review.

### Suggested order
1. Tesseract in Dockerfile (#1)
2. `/knowledge/seed` endpoint (#2), then verify Qdrant actually fills up
3. `.env.example` + empty `NCBI_EMAIL` (#5, #6)
4. Non-blocking ingestion (#3)
5. Scroll pagination (#4)
6. Payload indexes + pin Qdrant image (#7, #8)
7. A minimal upload + search page
8. Tests + README

Items 1–3 are ~30 minutes total and remove the risk of demoing a knowledge base that is silently empty or full of `[OCR failed]` strings.
