# Fixes Applied — Phase 1 & 2

All items from `PHASE_1_2_REVIEW.md` except the frontend UI and Alembic.
Verified by 50 passing tests in `backend/tests/`.

```
cd backend
pip install -r requirements.txt
pytest tests/ -v          # 50 passed
```

---

## Blockers fixed

### 1. Tesseract missing → knowledge base silently poisoned
`backend/Dockerfile` now installs `tesseract-ocr` and `tesseract-ocr-eng`.

More importantly, the *failure mode* is fixed. Parsers previously returned
`"[OCR failed: ...]"` as if it were content; that string cleared the length
check, got embedded, and the document was marked **completed**. Now every
parser either returns real text or raises `ParseError`, and the document is
marked **failed** with the actual reason.

Also added along the way:
- Scanned PDFs (no extractable text) now produce a clear message telling you to
  upload as an image for OCR, instead of a mystery empty result.
- DOCX **tables** are now extracted. They live outside `doc.paragraphs`, so
  every protocol table and dose chart was being silently dropped.
- Minimum usable text raised from 10 to 50 characters.

### 2. Seeder was dead code → knowledge base was empty
`knowledge_seeder.py` was imported by nothing. Added:

```
POST /api/v1/knowledge/seed
```

Runs in the background, idempotent, returns immediately. Kept manual rather
than on-startup so container restarts stay fast and predictable.

### 3. Ingestion blocked the entire event loop
`embedding_service.encode()` is synchronous CPU-bound PyTorch and was being
called directly inside `async def` — freezing the whole server, health checks
included, for the duration of every upload.

- Parse, chunk, embed and store now run via `run_in_threadpool`.
- `POST /upload` returns `201 {status: "processing"}` immediately via
  `BackgroundTasks`; the frontend polls `GET /documents/{id}`.
- The background worker opens its **own** DB session — the request-scoped one
  is closed once the response is sent.
- The worker cannot raise: Starlette swallows background-task exceptions, which
  would strand documents at `processing` forever. Everything is caught and
  written back to the row.

### 4. Qdrant scroll pagination was wrong
`scroll(offset=...)` takes a **point ID to resume from**, not a row count.
Passing an integer returned wrong pages, and results came back in random UUID
order. Now scrolls the document's full chunk set using the real cursor, sorts
by `chunk_index`, then slices — with a safety cap.

---

## Also fixed

| Issue | Change |
|---|---|
| `.env` committed with placeholders, no `.env.example` | Created `.env.example`; cleaned `.env`; blanked all placeholder keys |
| `NCBI_EMAIL=your_email@example.com` was truthy | Added `ncbi_is_configured()` rejecting placeholder/malformed addresses; removed the `radassist@example.com` fallback that faked an identity to a public API |
| No Qdrant payload indexes | `create_payload_index` on `document_id` and `source_type` — unindexed filters full-scan the collection |
| `qdrant/qdrant:latest` vs pinned client | Pinned to `v1.10.1` |
| `client.search()` deprecated | Migrated to `query_points()` |
| `vectors_count` returns `None` on modern servers | Falls back to `points_count` |
| `--reload` baked into the image | Moved to a `command:` override in compose |
| Model re-downloaded every rebuild | Added `hf_cache` named volume |
| Uploads written into your source tree | Added `uploads_data` named volume |
| New `QdrantClient` per health check | Reuses the shared client, via threadpool |
| Health check didn't check the embedding model | Added — ingestion could be dead while `/health` said "healthy" |
| `package-lock.json` gitignored | Un-ignored; `npm ci` needs it and the Dockerfile copies it |

### The config bug worth knowing about

`CORS_ORIGINS: list[str]` looked fine but was a landmine. **pydantic-settings
JSON-decodes complex field types inside its env source, which runs before any
`field_validator`.** So the natural thing —

```
CORS_ORIGINS=http://localhost:3000
```

— crashes the app at startup with `JSONDecodeError`, and no validator can
intercept it. My first fix (a `mode="before"` validator) was inert for exactly
this reason; the test caught it.

Correct fix: declare the field as `str`, expose the parsed list via a
`@property`. Both comma-separated and JSON-array forms now work, and every call
site is unchanged.

### A real bug the tests found

```python
chunk_size = chunk_size or settings.CHUNK_SIZE     # ← wrong
```

`0` is falsy but meaningful. `chunk_overlap=0` (a legitimate "no overlap"
request) was silently rewritten to `50`, and `chunk_size=0` fell through to
`512`, bypassing validation entirely. Changed to `is None` checks, plus explicit
`ValueError` when `chunk_overlap >= chunk_size` — that config makes the
character-level fallback step by zero and loop forever.

---

## Test suite

50 tests, ~1 second, no database / network / model download. Heavy libraries are
stubbed in `conftest.py`, so they run on every save.

| File | Covers |
|---|---|
| `test_chunking.py` | Size limits, content preservation, config guards, zero-overlap regression, paragraph integrity |
| `test_config.py` | The CSV/JSON parsing bug above, extension normalisation, model/dimension coherence |
| `test_seeder.py` | NCBI placeholder gate, seed data integrity, unique titles, source attribution |

---

## Still open — deliberate

**1. Phase 2 has no frontend.** Seven working endpoints, and `page.tsx` is still
a mock chat with a hardcoded `setTimeout`. An upload + real semantic search page
is ~150 lines and is what makes Phase 2 demoable. Recommend doing this before
Phase 3.

**2. No Alembic migrations.** Still using `Base.metadata.create_all`, which
creates tables but never *alters* them. The moment Phase 3 adds a column to
`documents`, nothing will happen and it will be confusing. Either run
`alembic init` before Phase 3 or drop alembic from requirements.

**3. No auth on upload/delete.** Fine for now — but name it explicitly as a
known limitation in your report rather than leaving it unmentioned.

**4. No RAG evaluation yet.** Before Phase 3, write ~15 question/expected-source
pairs and measure recall@5 against the seeded corpus. That turns "I chose 512
characters" into "I measured 512 against 256 and 1024," which is the single
strongest thing you can say in an internship review.

---

## Verify it works

```bash
docker-compose up --build

curl localhost:8000/api/v1/health           # all four components healthy
curl -X POST localhost:8000/api/v1/knowledge/seed
curl localhost:8000/api/v1/knowledge/stats  # 14 documents, chunks > 0

curl -X POST localhost:8000/api/v1/knowledge/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"radiographic findings of pneumothorax","limit":3}'
```

The last call is the real test — it should return chunks about absent lung
markings and the visible pleural line, with similarity scores and traceable
source documents.
