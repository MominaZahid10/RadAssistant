# Phase 5 — Decision Support Features

**Plan document:** *Similar-case retrieval, differential diagnosis suggestions, report quality checker, patient timeline comparison.* (Weeks 9–10)

---

## Where the project actually stands

Assessed against the phase table in `RadAssist_AI_Project_Document.docx`, not against
what was intended.

| Phase | Name | Status | Evidence |
|---|---|---|---|
| 0 | Discovery & Requirements | **Done** | Project document, success metrics, scope |
| 1 | Architecture & Environment | **Done** | Docker Compose, FastAPI + Next.js, Postgres + Qdrant, Alembic |
| 2 | Knowledge Base & Ingestion | **Done** | PDF/DOCX/OCR loaders, chunking, embeddings, 296 articles / 20,811 chunks |
| 3 | Core RAG + Report Generation | **Partial** | RAG, evidence panel, citations all working. **Report generation is unreachable** |
| 3.5 | *(added)* Retrieval quality | **Done** | Cross-encoder rerank, hybrid BM25, contextual headers. kw@5 55.6% → 100% |
| 4 | Multimodal Ingestion | **Mostly done** | Vision reading, DICOM de-ident, image API. **Prior-report comparison missing** |
| 5 | Decision Support | **Not started** | This document |
| 6 | Explainability, Auth & Hardening | **Partial** | Evidence panel done. **No auth, no logging, no rate limiting** |
| 7 | Deployment & Pilot | Not started | Runs locally only |
| 8 | Iteration & Portfolio | Not started | — |

### Two gaps that predate Phase 5

**1. Report generation is written but not wired up.** `REPORT_SYSTEM_PROMPT` and
`mode="report"` exist in `rag_service.py`, but `chat.py` hardcodes `mode="qa"` at
both call sites. No endpoint, no UI, no way to reach it.

This is Core Feature 5.1 and the headline Phase 3 deliverable — *"text-findings-in,
retrieval-grounded report-out"*. It is also load-bearing for Phase 5: a report
quality checker needs a report to check.

**2. Prior-report comparison is missing.** Phase 4 specifies *"prior-report PDF/DOCX
upload and comparison against current findings."* Upload works; comparison does not
exist. It is the same capability Phase 5 calls "patient timeline comparison", so it
is folded into Step 4 below rather than counted twice.

> **Recommendation:** do Step 0 first. It is small, it closes the Phase 3 gap, and
> every later step in this phase depends on it.

---

## Step 0 — Expose report generation *(prerequisite)*

**What:** make `mode="report"` reachable, and add a findings-entry UI.

| Action | File |
|---|---|
| MODIFY | `backend/app/schemas/rag.py` — `mode` field on `ChatRequest` |
| MODIFY | `backend/app/api/v1/endpoints/chat.py` — pass `request.mode` instead of `"qa"` |
| MODIFY | `frontend/src/app/page.tsx` — mode toggle: **Ask** / **Draft report** |

**Why an enum and not a free string:** an unknown mode must fail loudly at
validation. Silently falling back to `"qa"` would mean a clinician requesting a
report gets a chat answer and no error — the exact class of silent substitution
that inverted a finding in Phase 4.

**Deliverable:** type findings, get a structured Findings/Impression draft with
citations.

**Estimate:** ~1 hour.

---

## Step 1 — Similar-case retrieval

**What:** given the current findings, retrieve semantically similar prior reports
and show what was said about them.

| Action | File |
|---|---|
| NEW | `backend/app/services/similar_case_service.py` |
| MODIFY | `backend/app/api/v1/endpoints/chat.py` — `POST /chat/similar-cases` |
| MODIFY | `frontend/src/app/page.tsx` — similar-cases panel |

**The retrieval is already built.** This is a filtered query against the existing
Qdrant collection restricted to `source_type in (report_upload, curated_summary)`,
reusing `retrieve_context()` — bi-encoder recall, cross-encoder rerank, per-document
capping. No new infrastructure.

> **The honest constraint, stated up front.** There is no corpus of prior reports.
> The knowledge base is 296 open-access papers and 14 curated summaries. Similar-case
> retrieval over papers is *literature retrieval*, which the chat already does.
>
> This step is only meaningful once report uploads accumulate. Two options:
> **(a)** build it against uploaded reports and demo with 5–10 uploads, honest that
> the corpus is small; **(b)** generate a synthetic set of anonymised reports.
> **(a)** is more defensible — synthetic cases retrieved by a model that generated
> them is a closed loop that proves nothing.

**Deliverable:** "Cases like this" panel, with a visible count so a thin corpus is
apparent rather than hidden.

**Estimate:** ~3 hours.

---

## Step 2 — Differential diagnosis suggestions

**What:** from the findings, suggest differentials — each with its supporting
evidence and an explicit discriminator.

| Action | File |
|---|---|
| MODIFY | `backend/app/services/rag_service.py` — `DIFFERENTIAL_PROMPT` |
| MODIFY | `backend/app/api/v1/endpoints/chat.py` — `mode="differential"` |

**This is the most dangerous feature in the project.** Everything so far reports what
a document says. This generates clinical hypotheses that no source states. The
grounding scaffold does not cover it, because there is nothing to ground *to*.

Constraints, non-negotiable:

- Every differential names **which finding** supports it, quoted from the input
- Every differential names **what would discriminate** it — the next test, the
  distinguishing feature
- Ordered by consistency with the findings, **never** presented as a probability the
  system cannot compute
- Framed as *"findings compatible with"*, never *"the diagnosis is"*
- Refuses when findings are too sparse, rather than listing textbook differentials
  for the body part

The project document's own risk table is explicit: *"Always label AI image findings
as suggestions; never auto-populate the report."* The same applies here, harder.

**Deliverable:** differentials with per-item evidence and discriminators.

**Estimate:** ~3 hours.

---

## Step 3 — Report quality checker

**What:** check a draft for missing sections, inconsistent terminology, and ambiguous
wording.

| Action | File |
|---|---|
| NEW | `backend/app/services/quality_service.py` |
| MODIFY | `backend/app/api/v1/endpoints/chat.py` — `POST /chat/quality-check` |
| MODIFY | `frontend/src/app/page.tsx` — inline warnings on the draft |

**Deterministic checks first, model second.** Missing sections, laterality stated
without a side, a measurement with no unit, hedging stacked on hedging
("possibly may represent") — these are rules, and a rule that fires the same way
every time is worth more here than a model that usually notices. The model handles
only what rules cannot: terminology drift within a report, internal contradiction
between findings and impression.

This is also the one Phase 5 feature that is **directly measurable** against the
project's own success metric — *"Reduction in missing-section / inconsistent-terminology
flags over time"* — so it should report counts, not prose.

**Deliverable:** a structured issue list with severity, each pointing at the offending
line.

**Estimate:** ~4 hours.

---

## Step 4 — Prior-report comparison / patient timeline

**What:** upload a prior report alongside the current one; get new, resolved, stable
and changed findings.

| Action | File |
|---|---|
| NEW | `backend/app/services/comparison_service.py` |
| MODIFY | `backend/app/schemas/rag.py` — `prior_text` on `ChatRequest` |
| MODIFY | `frontend/src/app/page.tsx` — two-document upload, diff view |

**Closes the outstanding Phase 4 deliverable as well as Phase 5's timeline item.**
The plumbing exists: `attached_text` already carries an uploaded document as the
primary source, and the vision reader handles report photos. This adds a *second*
document and an explicit comparison prompt.

**The failure mode to design against:** a measurement changing from 8mm to 9mm is
either interval growth or inter-reader variation, and the report cannot tell you
which. The output must say *"reported as 8mm previously, 9mm now"* and stop —
never *"interval growth"*, which is a clinical judgement the documents do not support.

**Deliverable:** four-column comparison — New / Resolved / Stable / Changed — each
row quoting both reports.

**Estimate:** ~4 hours.

---

## Step 5 — Tests & verification

| Action | File | Covers |
|---|---|---|
| NEW | `backend/tests/test_similar_cases.py` | filtering, empty-corpus honesty |
| NEW | `backend/tests/test_differential.py` | refusal on sparse findings, evidence required per item |
| NEW | `backend/tests/test_quality_service.py` | each deterministic rule, no false positives on a good report |
| NEW | `backend/tests/test_comparison.py` | new/resolved/stable classification, no invented interval change |
| NEW | `verify_phase5.sh` | end-to-end, same style as `verify_phase4.sh` |

**The test I care most about:** a differential request with one vague finding must
produce a refusal, not a plausible list. Generating confident differentials from
nothing is this phase's equivalent of the inverted lordosis — fluent, useful-looking,
and wrong.

**Estimate:** ~3 hours.

---

## Summary

| Step | What | Est. |
|---|---|---|
| 0 | Expose report generation *(prerequisite)* | 1h |
| 1 | Similar-case retrieval | 3h |
| 2 | Differential diagnosis | 3h |
| 3 | Report quality checker | 4h |
| 4 | Prior-report comparison / timeline | 4h |
| 5 | Tests & verification | 3h |
| | **Total** | **~18h** |

### Risks

| Risk | Mitigation |
|---|---|
| No prior-report corpus for similar-case retrieval | Show the corpus size in the UI; do not disguise a thin index as a rich one |
| Differentials read as diagnosis | Evidence + discriminator required per item; refuse on sparse input; wording fixed in the prompt and asserted in tests |
| Comparison invents interval change | Quote both reports verbatim; never characterise the delta |
| Quality checker fires on valid reports | Deterministic rules tested against a known-good report for false positives |
| Scope creep into Phase 6 | Auth, logging and rate limiting are explicitly Phase 6 — not here |

---

## Approval gates

```
Phase 5  Step 0  Expose report generation ..... ⏸ awaiting approval
         Step 1  Similar-case retrieval ....... ⏸
         Step 2  Differential diagnosis ....... ⏸
         Step 3  Report quality checker ....... ⏸
         Step 4  Prior-report comparison ...... ⏸
         Step 5  Tests & verification ......... ⏸
```

**Start with Step 0?** It is an hour, it closes the Phase 3 gap, and Steps 3 and 4
both need a report to operate on.
