# Phase 3.5 + Phase 4 — Implementation Plan

**Phase 3.5:** Retrieval Quality (evaluation + reranking)
**Phase 4:** Multimodal Ingestion — DICOM & Image Foundation

---

## Why Phase 3.5 comes first

Phase 3 ended with a real, diagnosed defect: on a 232-document corpus, the query *"radiographic findings of pneumothorax"* returns term-dense research papers (liposuction complications, an ML detection study) and the model correctly refuses to answer because none of them describe the finding.

Two reasons to fix this before adding images:

1. **You cannot evaluate what you cannot measure.** Phase 4 adds image retrieval on top of text retrieval. Building that on retrieval you can't score means any future regression is invisible.
2. **Four bugs this project hit produced plausible-looking output rather than errors** — OCR placeholder text, mid-word chunks, `【N】` citations, and silently-ignored API parameters. A scored evaluation set is the only mechanism that catches that class systematically.

Estimated: **half a day.**

---

# PHASE 3.5 — Retrieval Quality

## Step 1 — Evaluation Harness

**What:** A fixed set of questions with expected source documents, scored automatically.

**Why first:** Gives a *baseline number* before any change. Without it, "reranking helped" is a vibe.

### Files

| Action | File | Purpose |
|---|---|---|
| NEW | `backend/eval/questions.yaml` | 18–20 question / expected-source pairs |
| NEW | `backend/eval/run_eval.py` | Scores recall@k, MRR, refusal rate |
| NEW | `backend/eval/README.md` | How to add questions, how to read results |

### Question set design

Deliberately spans four categories, because they fail differently:

| Category | Example | Tests |
|---|---|---|
| **Definitional** | "What are the radiographic findings of pneumothorax?" | The failure we found — explanatory content vs term-dense papers |
| **Specific criteria** | "What are the Fleischner criteria for a 7mm solid nodule?" | Precise numeric recall |
| **Comparative** | "How does cardiogenic differ from non-cardiogenic pulmonary oedema?" | Multi-source synthesis |
| **Should refuse** | "What is the ideal soil pH for tomatoes?" | Guardrail — must NOT retrieve confidently |

### Metrics

- **recall@5 / recall@12** — did the expected source appear in the top N?
- **MRR** — how high did it rank? (rank 1 is much better than rank 11)
- **Refusal rate** — must be 100% on the "should refuse" set, near 0% on answerable ones
- **Mean top score** — watch for score inflation without relevance (exactly what we saw: 0.78 → 0.83 while quality dropped)

**Deliverable:** `python eval/run_eval.py` prints a scorecard and writes `eval/results/baseline.json`.

> [!IMPORTANT]
> The baseline will probably look bad on definitional queries. **That's the point.** A number you can improve beats a suspicion you can't.

---

## Step 2 — Cross-Encoder Reranking

**What:** Two-stage retrieval. Vector search for *recall* (top 50), cross-encoder for *precision* (top 12).

**Why it fixes the problem:** A bi-encoder embeds question and passage **separately**, so similarity reflects topical overlap — a paper repeating "pneumothorax" 40 times scores highly. A cross-encoder reads question and passage **together** and scores *does this passage answer this question*. That distinction is exactly the failure mode we hit.

### Files

| Action | File | Purpose |
|---|---|---|
| NEW | `backend/app/services/reranker.py` | Cross-encoder service, lazy-loaded, threadpool |
| MODIFY | `backend/app/services/rag_service.py` | Insert rerank between retrieve and cap |
| MODIFY | `backend/app/config.py` | `RERANKER_MODEL`, `RERANK_ENABLED`, `RERANK_CANDIDATES` |
| NEW | `backend/tests/test_reranker.py` | Ordering, disabled-mode passthrough, model-missing behaviour |

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80 MB, CPU-friendly, ~50ms for 50 passages).

> [!IMPORTANT]
> **Same cache mechanism as the embedding model.** It downloads to `backend/.hf_cache`, which survives `docker-compose down -v`. Populate it once with `HF_HUB_OFFLINE=0`.
> `RERANK_ENABLED=false` must degrade gracefully to current behaviour — so a missing model never breaks chat.

**Deliverable:** `run_eval.py` re-run shows the delta. Expected: recall@5 improves materially on definitional queries; the pneumothorax question answers correctly again.

---

# PHASE 4 — Multimodal Ingestion (Foundation)

**Goal:** Ingest, store, de-identify and serve medical images — DICOM files and figures from the PMC corpus — with metadata in Postgres and files on disk, linked to the existing documents.

**Explicitly NOT in this phase:** AI interpretation of images, image-similarity search, diagnosis. This is the plumbing every later multimodal feature needs.

---

## Step 0 — Alembic Migrations

**What:** Set up Alembic and capture the current schema as the initial migration.

**Why now, not later:** Phase 4 adds a `medical_images` table. The app currently uses `Base.metadata.create_all`, which **creates tables but never alters them**. The moment we add a column to an existing table, nothing happens and it fails silently — the exact class of bug this project keeps hitting.

### Files

| Action | File |
|---|---|
| NEW | `backend/alembic.ini`, `backend/alembic/env.py` |
| NEW | `backend/alembic/versions/0001_initial.py` (captures `documents` as-is) |
| MODIFY | `backend/app/main.py` (replace `create_all` with a migration check) |

**Deliverable:** `alembic upgrade head` builds the schema from scratch on an empty database, and `alembic current` reports the version.

> [!IMPORTANT]
> This is the one place I'd push back on deferring. Every phase from here changes the schema, and `create_all` will keep looking like it worked.

---

## Step 1 — Image Model & Storage Layout

**What:** The `medical_images` table and on-disk storage convention.

### Schema

```python
class MedicalImage(Base):
    id              UUID   primary key
    document_id     UUID   FK → documents.id, nullable   # figures link to their article
    filename        str
    storage_path    str                                   # relative, under IMAGE_DIR
    thumbnail_path  str | None
    mime_type       str
    file_size       int
    width, height   int

    # Clinical metadata (DICOM or inferred)
    modality        str | None    # CR, DX, CT, MR, US, ...
    body_part       str | None
    study_date      date | None
    view_position   str | None    # PA, AP, LATERAL

    # Provenance
    source_type     str           # 'dicom_upload' | 'pmc_figure' | 'image_upload'
    source_url      str | None
    caption         str | None    # PMC figure caption — the text half of the pair

    dicom_metadata  JSONB | None  # de-identified tags only
    is_deidentified bool
    created_at      datetime
```

> [!IMPORTANT]
> **Files on disk, metadata in Postgres — never image blobs in the database.**
> A chest CT series is 100–500 MB. Storing that in Postgres bloats backups, breaks replication, and makes every query slower. Standard practice is a filesystem (later: S3) with the path in the DB.

**Deliverable:** Table created via Alembic migration `0002_medical_images`.

---

## Step 2 — DICOM Parsing & De-identification

**What:** Read DICOM, extract metadata, strip PHI, convert pixel data to viewable PNG.

### Files

| Action | File |
|---|---|
| NEW | `backend/app/services/dicom_service.py` |
| MODIFY | `backend/requirements.txt` (`pydicom`, `pylibjpeg`, `numpy` already present) |
| NEW | `backend/tests/test_dicom_service.py` |

### Three things that must be right

**1. De-identification is non-negotiable.**
DICOM tags carry PatientName, PatientID, PatientBirthDate, InstitutionName, ReferringPhysician, and often free-text StudyDescription. We use an **allowlist**, not a blocklist — only explicitly permitted tags are retained. A blocklist misses private vendor tags, and "we forgot a tag" is a data-protection incident, not a bug.

Retained: `Modality`, `BodyPartExamined`, `ViewPosition`, `StudyDate` (year only), `Rows`, `Columns`, `PhotometricInterpretation`, `WindowCenter`, `WindowWidth`.

**2. Pixel data needs windowing.**
DICOM stores raw Hounsfield/intensity values, often 12–16 bit. Naively scaling to 8-bit produces a grey, unreadable image. We apply the VOI LUT (WindowCenter/WindowWidth from the file, falling back to min/max) — this is why DICOM viewers have brightness/contrast presets.

**3. MONOCHROME1 must be inverted.**
`PhotometricInterpretation` can be MONOCHROME1 (0 = white) or MONOCHROME2 (0 = black). Ignoring it renders X-rays as photographic negatives — bones black, air white. Easy to miss because the image still *looks* like an image.

**Deliverable:** `parse_dicom(bytes)` returns de-identified metadata + a correctly-windowed PNG. Test asserts zero PHI tags survive.

---

## Step 3 — PMC Figure Extraction

**What:** Extract figures + captions from the PMC articles already ingested.

**Why this is the good corpus:** ~200 articles already ingested, already CC-licensed, already radiology. Each figure has a caption — genuinely paired image-text data, which is what makes later multimodal retrieval possible.

### Files

| Action | File |
|---|---|
| MODIFY | `backend/app/services/pmc_fetcher.py` — capture `<fig>` instead of stripping it |
| NEW | `backend/app/services/figure_fetcher.py` — download images from PMC |
| MODIFY | `backend/app/api/v1/endpoints/knowledge.py` — `POST /knowledge/fetch-figures` |

> [!IMPORTANT]
> **This reverses a Phase 3 decision.** `_strip_noise()` currently removes `<fig>` elements because captions without images were noise polluting the text chunks. Now we want both — so figures get **extracted before stripping**, and the caption still stays out of the text chunk (it belongs to the image record, not the article body).
> Figures inherit the article's licence, and we only ingest from the Open Access Subset, so redistribution is permitted — but the licence is recorded per image regardless.

**Deliverable:** `POST /knowledge/fetch-figures` populates `medical_images` with figures linked to their parent documents.

---

## Step 4 — Upload & Serving API

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/images/upload` | DICOM or PNG/JPG. Async, same pattern as documents |
| `GET` | `/api/v1/images` | List with filters (modality, body_part, source_type) |
| `GET` | `/api/v1/images/{id}` | Metadata |
| `GET` | `/api/v1/images/{id}/file` | The image itself |
| `GET` | `/api/v1/images/{id}/thumbnail` | 256px thumbnail |
| `DELETE` | `/api/v1/images/{id}` | Remove record + files |
| `GET` | `/api/v1/knowledge/documents/{id}/images` | Figures for an article |

> [!IMPORTANT]
> **Reuses the Phase 3 patterns that already work:** `_require_embedding_model()`-style guards, background processing with its own DB session, `202` + poll rather than blocking, and failures recorded with a real reason rather than a generic traceback.

**Deliverable:** Upload a DICOM via Swagger, get back de-identified metadata, view the rendered PNG in the browser.

---

## Step 5 — Frontend Image Viewer

### Files

| Action | File |
|---|---|
| MODIFY | `frontend/src/lib/api.ts` — image types + methods |
| NEW | `frontend/src/components/ImageViewer.tsx` — lightbox, zoom, metadata panel |
| NEW | `frontend/src/components/ImageUpload.tsx` — drag-drop, DICOM-aware |
| MODIFY | `frontend/src/app/page.tsx` — thumbnails in the evidence panel |
| NEW | `frontend/src/app/images/page.tsx` — browse/filter gallery |

**Key integration:** when a retrieved chunk comes from an article that has figures, the evidence panel shows thumbnails alongside the text. That's the first genuinely multimodal moment in the product — a citation you can *look at*.

**Deliverable:** Ask a question, expand sources, see figures from the cited paper, click to enlarge with metadata.

---

## Step 6 — Tests & Verification

| Action | File | Covers |
|---|---|---|
| NEW | `backend/tests/test_dicom_service.py` | PHI stripping, windowing, MONOCHROME1 inversion, corrupt files |
| NEW | `backend/tests/test_image_api.py` | Upload contract, filters, 404s, deletion cleanup |
| NEW | `backend/tests/test_figure_fetcher.py` | JATS `<fig>` parsing, caption extraction, licence propagation |
| MODIFY | `verify_phase4.sh` | End-to-end checks, same style as `verify_phase3.sh` |

**The test I care most about:** upload a DICOM containing PatientName, PatientID and InstitutionName, then assert none of them appear anywhere in the stored metadata, filename, or JSONB. De-identification failing silently is the worst outcome in this phase.

---

## Summary

| Phase | Steps | Estimate |
|---|---|---|
| **3.5** | Evaluation harness → Cross-encoder reranking | ~half a day |
| **4** | Alembic → Model → DICOM → PMC figures → API → Frontend → Tests | ~2–3 days |

### Risks

| Risk | Mitigation |
|---|---|
| **Network instability** (the main cost of Phase 3) | Reranker model uses the host cache; PMC figures download incrementally and resume |
| **De-identification gaps** | Allowlist not blocklist; explicit test |
| **Storage growth** | Size cap per upload, thumbnails, `IMAGE_DIR` on a named volume |
| **Alembic on an existing DB** | Initial migration stamps current state; test on a fresh DB first |

---

## Approval gates

I'll stop after each step, explain what was built and why, and wait for your go-ahead before continuing:

```
Phase 3.5  Step 1  Evaluation harness ....... ⏸ awaiting approval
           Step 2  Cross-encoder reranking .. ⏸
Phase 4    Step 0  Alembic .................. ⏸
           Step 1  Image model .............. ⏸
           Step 2  DICOM + de-identification  ⏸
           Step 3  PMC figure extraction .... ⏸
           Step 4  Upload & serving API ..... ⏸
           Step 5  Frontend viewer .......... ⏸
           Step 6  Tests & verification ..... ⏸
```

**Ready to start with Phase 3.5 Step 1 (evaluation harness)?**
