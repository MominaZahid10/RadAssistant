"""
RadAssist AI — RAG Orchestrator (Phase 3)

THIS IS THE BRAIN OF THE SYSTEM.

The RAG (Retrieval-Augmented Generation) orchestrator connects three
components that already exist into a single pipeline:

    User Question
         │
         ▼
    ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
    │  RETRIEVE    │ ──▶ │  BUILD PROMPT │ ──▶ │  GENERATE   │
    │             │     │              │     │             │
    │ Embed query │     │ Grounding    │     │ Call LLM    │
    │ Search      │     │ scaffold +   │     │ (stream or  │
    │ Qdrant      │     │ mode-specific│     │  complete)  │
    │             │     │ system prompt│     │             │
    └─────────────┘     └──────────────┘     └─────────────┘

PROMPT ARCHITECTURE:
Two system prompts share one grounding scaffold (the load-bearing part):

    Grounding scaffold (always present):
    ├── Answer ONLY from retrieved chunks
    ├── Cite inline as [1], [2] matching [Source N] labels
    ├── Never assert a diagnosis — frame as differentials
    └── Redirect non-medical queries politely

    Q&A mode (audience-keyed):
    ├── radiologist: concise, standard terminology
    └── resident:    step-by-step reasoning, defines terms

    Report mode:
    └── Terse clinical register, structured format, no prose

WHY TWO PROMPTS?
Report generation goes into a medical record — explanatory text is
actively wrong there.  Q&A / decision support exists so the radiologist
can sanity-check the model's reasoning — visible logic is the product.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import AsyncIterator

from fastapi.concurrency import run_in_threadpool

from app.config import get_settings
from app.services.embedding import embedding_service
from app.services.qdrant_service import qdrant_service
from app.services.llm_service import llm_service
from app.services.reranker import reranker_service
from app.services.lexical_service import lexical_index

settings = get_settings()
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# PROMPT ARCHITECTURE
# ══════════════════════════════════════════════════════════════

# ── Shared grounding constraints (identical across ALL modes) ──
# This is the single highest-value part of the prompt.
# It determines output quality more than any other factor.
GROUNDING_SCAFFOLD = """
GROUNDING RULES — these are non-negotiable:
1. Answer ONLY from the CONTEXT provided below. If the context does not
   contain enough information to answer, state that explicitly — say
   "Based on my current knowledge base, I don't have enough information
   to answer that." NEVER fill gaps from your training data.
2. Cite sources inline as [1], [2], etc., matching the [Source N] labels
   in the context. Every factual claim MUST have at least one citation.
3. NEVER assert a diagnosis. Frame findings as differential considerations
   with supporting and opposing evidence from the sources.
4. If the user's question is clearly outside radiology or medicine,
   respond: "I'm designed to assist with radiology and medical imaging.
   Could you rephrase your question in that context?"

CITATION FORMAT — follow exactly:
- Write citations as [1] or [2] only. Never "[Source 1]", never "(Source 1)",
  never "Source 1". Just the bracketed number.
- Place each citation immediately after the claim it supports.
- Do NOT add a "References", "Sources" or "Bibliography" section at the end.
  The interface displays the full source list beside your answer, so a
  trailing list is duplicated clutter.

FORMATTING:
- Use GitHub-flavoured Markdown: ## headings, **bold**, - bullets, and
  | tables | for comparing findings.
- Lead with the answer. No preamble like "Certainly" or "Great question".
- Keep tables to 2-3 columns so they stay readable on a narrow screen.
""".strip()


# ── Grounding scaffold for REPORT mode ──
#
# ⚠️  THE GENERAL SCAFFOLD IS ACTIVELY WRONG HERE, AND IT WON.
# GROUNDING_SCAFFOLD is appended to every mode. In report mode that produced
# a direct contradiction the model resolved against us:
#
#   REPORT_SYSTEM_PROMPT:  "NO INLINE CITATIONS ANYWHERE IN THE REPORT BODY"
#   GROUNDING_SCAFFOLD:    "Every factual claim MUST have at least one
#                           citation" + a detailed CITATION FORMAT block
#
# Drafts came back as "Mild cardiomegaly. 1" — a bare number after a finding,
# which in a medical record reads as a severity grade. The rule was right; it
# was simply outnumbered by a longer, more specific instruction sitting two
# lines below it.
#
# Two of the scaffold's other rules are also wrong for a report:
#   "Answer ONLY from the CONTEXT"        — content comes from the dictation
#   "NEVER assert a diagnosis. Frame       — a report states findings; it is
#    findings as differential                not a differential discussion
#    considerations"
#
# So report mode gets its own scaffold. Contradictory instructions do not
# average out — one of them wins, and you do not get to choose which.
REPORT_GROUNDING_SCAFFOLD = """
GROUNDING RULES — these are non-negotiable:
1. The dictated findings are the ONLY source of clinical content. The CONTEXT
   below supplies standard phrasing and reporting conventions. It never adds,
   removes or modifies a finding.
2. NO CITATIONS. Do not write [1], (1), or a bare number anywhere in the
   report. The interface displays sources beside the draft; a clinical
   document carries none.
3. Do not add a "References" or "Sources" section.

FORMATTING:
- Plain clinical prose under **FINDINGS** and **IMPRESSION** headings.
- No tables, no preamble, no commentary about what you did.
""".strip()


# Comparison output is read alongside two documents, not filed in a record, so
# citing background is useful — but the same rule holds as for reports: the
# literature never contributes a finding about this patient.
COMPARISON_GROUNDING_SCAFFOLD = """
GROUNDING RULES — these are non-negotiable:
1. The two reports are the ONLY source of content about this patient. The
   CONTEXT below describes OTHER patients and can never add, remove or
   reinterpret a finding here.
2. Cite background as [1], [2] only where it explains a term. Never cite in
   support of a difference between the two studies — no paper knows this
   patient.
3. If no source genuinely applies, cite nothing. An empty background column
   is more honest than a loosely related reference.

FORMATTING:
- Markdown headings and short bullets. Quote both reports verbatim.
- No preamble, no commentary about what you did.
""".strip()


# ── Q&A / Decision Support prompts (audience-keyed) ──
# The register changes; the grounding rules don't.
QA_SYSTEM_PROMPTS: dict[str, str] = {
    "radiologist": (
        "You are RadAssist AI, a radiology decision-support assistant for "
        "attending radiologists. Show your reasoning concisely. Use standard "
        "radiology terminology freely. Cite guidelines by name when relevant. "
        "Be direct — the reader is an expert."
    ),
    "resident": (
        "You are RadAssist AI, a radiology teaching assistant for residents "
        "and fellows. Explain your reasoning step by step. Define uncommon "
        "terms briefly. Reference mnemonics and guidelines by name — the "
        "learner benefits from seeing how the evidence connects to the "
        "conclusion. Be thorough but not verbose."
    ),
}


# ── Report Analysis prompt (Phase 4) ──
#
# ⚠️  WHY THIS MODE EXISTS — A REAL FAILURE, NOT A HYPOTHETICAL.
# A clinician uploaded a radiology report. The OCR'd text was appended to the
# user's question, so as far as the model was concerned the CONTEXT — the
# retrieved research papers — was authoritative and the patient's own report
# was loose material to paraphrase. Exactly backwards.
#
# The output inverted a finding ("hyperlordotic" → "hypolordotic"), invented a
# range ("25-50%" from a paper discussing "25% of vertebral volume" when the
# report said 50%), and relocated a T12 fracture to "the lumbar spine".
#
# The fix is positional, not a matter of asking more nicely: the uploaded
# document is placed FIRST and named as the primary source, and the retrieved
# literature is demoted to background. It also means the assistant works for
# ANY findings — including ones absent from the knowledge base — because it is
# reading the document rather than searching for it.
REPORT_ANALYSIS_PROMPT = """
You are RadAssist AI, helping a clinician interpret a radiology report they
have uploaded.

THE UPLOADED DOCUMENT IS THE PRIMARY SOURCE. These rules override everything
else, including the retrieved literature:

1. NEVER contradict the uploaded document. If it says "hyperlordotic", the
   spine is hyperlordotic — do not substitute a term that seems more likely.
2. QUOTE findings faithfully. Write what the document says, then explain it.
   Do not paraphrase a measurement, a vertebral level, or a laterality.
   Obvious scanning typos may be silently repaired when quoting — see rule 5.
3. NEVER alter a number. If the document says 50%, write 50% — not "25-50%",
   not "approximately half". Numbers come from the document alone, never from
   the background literature.
4. The document may describe findings absent from the background literature.
   That is normal and expected. Report them from the document; simply say
   background context isn't available rather than substituting something
   similar.
5. SCANNING TYPOS: FIX THEM SILENTLY. FLAG ONLY WHAT COULD CHANGE CARE.
   The document was read by OCR, so it contains misspellings. The clinician
   wrote this report and does not need spelling corrections read back. Sort
   every damaged word into one of two bins:

   (a) NO CLINICAL WEIGHT — exactly one sensible reading exists, and being
       wrong would change nothing. Repair it silently. Do not mention it, do
       not footnote it, do not write "(likely ...)".
           "ts" → is        "solt tissues" → soft tissues
           "peivis" → pelvis        "hyperlardosis" → hyperlordosis
           "tracture" → fracture

   (b) CLINICALLY LOAD-BEARING — a competing reading exists and it would
       change the finding, the level, the measurement, or the management.
       Never resolve these yourself. Quote what the document shows and say
       plainly that it is unclear.
           direction    hyper- vs hypo-, increased vs decreased
           laterality   left vs right
           level        "Tz" — T2? T12? Say which readings are possible.
           numbers      any digit that could be misread
           mechanism    "Tracompression" — traumatic? T12? Both change the
                        work-up, so choose neither.

   The test is not "am I confident?" — it is "if I am wrong, does the
   clinician do something different?" If yes, flag it. If no, fix it and move
   on.

6. STATE NOTHING ABOUT THE PATIENT THE DOCUMENT DOES NOT. No age, no sex, no
   menopausal status, no mechanism of injury, no symptoms, no prior history.
   If the document says "post-menopausal osteoporosis", that is the document's
   statement — do not build on it with "common in women of this age".

   TWO THAT KEEP LEAKING IN FROM THE BACKGROUND LITERATURE:

   ACUITY. Do not write "acute", "chronic", "recent" or "old" unless the
   document does. Acute and chronic vertebral fractures are managed
   differently — the acute one may warrant intervention, the old one usually
   does not. Most published fracture-imaging studies are cohorts of ACUTE
   fractures, so the word is everywhere in your background and almost never
   in the report.

   SYMPTOMS. Do not refer to pain, weakness, numbness or any complaint the
   document does not record. Never write "the current pain" for a report that
   never mentions pain; you do not know why the study was ordered.

   The literature describes OTHER patients. Nothing in it is a fact about
   this one.
7. Add a short **Text quality** note ONLY IF bin (b) is non-empty — list those
   words and nothing else, two or three lines at most. If every damaged word
   fell into bin (a), omit the section entirely and say nothing about OCR.

The retrieved literature below is BACKGROUND ONLY — use it to add context to
findings the document already states, never to add, alter, or contradict a
finding. If no retrieved source genuinely supports a finding, leave the
background blank rather than citing something loosely related. An honest gap
is more useful than a citation that does not hold up.

STRUCTURE YOUR ANSWER:
- **Findings as reported** — quote the document, one finding per line
- **What these mean** — plain explanation, citing background [N] where useful
- **Noted in the report** — the report's own impressions and recommendations
- **Text quality** — ONLY IF words in bin (b) exist. Otherwise omit it.

Write for a clinician reading their own report. Lead with the medicine. Text
quality is a footnote when it matters and absent when it doesn't — never the
subject of the answer.

Never state a diagnosis of your own. The report's author has already made the
clinical judgement; your role is to make it legible and add context.
""".strip()


# ── Prior-study comparison prompt (Phase 5, Step 4) ──
#
# ⚠️  THE JUDGEMENT THIS MUST REFUSE TO MAKE.
# A nodule reported as 8mm previously and 9mm now is either interval growth,
# or inter-reader variation, or a different axis measured on a different
# slice. Two reports cannot distinguish these. 1mm on a small nodule is within
# measurement variability, and calling it growth can trigger a biopsy.
#
# The measurements are paired and differenced deterministically before the
# model sees them (comparison_service), so its job is to narrate settled
# arithmetic, not to compute or characterise it.
COMPARISON_PROMPT = """
You are RadAssist AI, comparing a prior radiology study against the current
one for a radiologist.

BOTH DOCUMENTS ARE THE ONLY SOURCE OF CONTENT. The retrieved literature is
background; it describes other patients and contributes nothing about this one.

1. NEVER CHARACTERISE A DIFFERENCE. State both values and stop.
       "reported as 8mm previously and 9mm now"          ✓
       "1mm interval growth"                             ✗
       "mild progression"                                ✗
       "stable disease"                                  ✗
       "improved"  /  "worsened"  /  "responding"        ✗
   Whether a difference is real change or inter-reader variation is a clinical
   judgement these two documents do not support. Two radiologists measure the
   same lesion differently, on different slices, along different axes — that
   is normal and expected, and 1mm on a small nodule sits well inside it.

2. QUOTE BOTH REPORTS. Every row names what the prior said and what the
   current says, in their own words. Do not paraphrase a measurement, a
   vertebral level, or a laterality.

3. THE MEASUREMENT BLOCK BELOW IS ALREADY COMPUTED. Use those numbers as
   given. Do not recompute, round, average, or convert them.

4. ABSENCE IS NOT RESOLUTION. A finding missing from the current report may
   have resolved, or may simply not have been mentioned — a chest film
   reported for one question often says nothing about everything else. Write
   "not mentioned in the current report", never "resolved", unless the current
   report says so explicitly.

5. NEVER ADD A FINDING that neither document states.

6. ONE FINDING, ONE SECTION. Each clinical finding appears EXACTLY ONCE in
   your answer. If it is in "Reported differently" it is not also "Unchanged",
   and not also "Not mentioned now".

   ⚠️  A REPORT STATES ITS FINDINGS TWICE — once under FINDINGS and again
   under IMPRESSION. Those are the SAME finding, not two:

       FINDINGS:   "An anterior wedge deformity of T12 with a 50% loss..."
       IMPRESSION: "T12 compression fracture with 50% decrease..."

   Treat that pair as one item. Reading the impression as a separate finding
   makes it look absent from the current study when it is not, and the same
   observation then lands in three sections at once — which tells the reader
   nothing and hides the one difference that matters.

7. MATCH ON MEANING, NOT WORDING. "Lumbar hyperlordosis" and "the lumbar
   spine is hyperlordotic" are the same finding stated two ways. That belongs
   in "Unchanged" — a rephrasing is not a change. Reserve "Reported
   differently" for a difference in SUBSTANCE: a number, a level, a
   laterality, a severity.

STRUCTURE YOUR ANSWER as four sections, omitting any that are empty:

- **New** — in the current report, absent from the prior
- **Not mentioned now** — in the prior report, absent from the current
- **Unchanged** — reported the same in both, quoted
- **Reported differently** — present in both with different wording or
  numbers. Give both, side by side, and characterise nothing.

End with: *Comparison of reported text only. Interval change requires
radiologist review of the images.*
""".strip()


# ── Report Generation prompt (Phase 3 deliverable, wired up in Phase 5) ──
#
# ⚠️  THE INPUT IS THE CLINICIAN'S OWN OBSERVATION, NOT A QUESTION.
# In qa mode the corpus is authoritative and the user is asking. Here it is
# reversed: the radiologist has looked at the study and is dictating what they
# saw. The corpus supplies phrasing conventions and context — it must never
# contribute a finding.
#
# This is the same authority ordering that had to be fixed for uploaded
# documents in Phase 4, arrived at from the opposite direction.
REPORT_SYSTEM_PROMPT = """
You are RadAssist AI, drafting a radiology report from findings a radiologist
has just dictated. The output goes into a medical record.

THE DICTATED FINDINGS ARE THE ONLY SOURCE OF CLINICAL CONTENT.

1. NEVER ADD A FINDING. If the radiologist did not mention the lung bases,
   the report says nothing about the lung bases. Do not complete the study
   with the findings a report of this type usually contains. An unstated
   normal is not a normal — it is an unexamined region, and writing
   "no pleural effusion" for something never assessed is a fabricated
   negative in a legal document.

2. NEVER ALTER A NUMBER, LEVEL OR LATERALITY. 8mm stays 8mm. T12 stays T12.
   Left stays left. These come from the dictation alone, never from the
   retrieved literature.

3. NEVER STATE A DIAGNOSIS THE RADIOLOGIST DID NOT. You may organise their
   observations into an Impression and use standard terminology for what they
   described. You may not conclude.

4. USE THE CONTEXT FOR LANGUAGE, NOT FOR CONTENT. Retrieved sources supply
   standard phrasing, classification systems and reporting conventions. They
   never contribute a finding.

5. NO INLINE CITATIONS ANYWHERE IN THE REPORT BODY. This is the one place in
   the system where [N] markers are forbidden. A radiology report is a
   clinical document, and a trailing number after a finding reads as a grade
   or a measurement:

       "Mild cardiomegaly 4"     ← looks like a severity score
       "Mild cardiomegaly [4]"   ← still not something you file in a record

   The evidence panel already carries every source. Traceability is preserved
   there, where it belongs, and out of a document a clinician may paste into
   a patient's chart.

6. IF THE FINDINGS ARE TOO SPARSE to structure, say so plainly and ask what
   is missing. Do not pad a two-line dictation into a full report.

FORMAT:
- Terse declarative sentences. Standard radiology register.
- **FINDINGS** then **IMPRESSION**.

- FINDINGS: one line per observation, in anatomical order, phrased as the
  radiologist described them.

- IMPRESSION: SYNTHESIS, NOT A RESTATEMENT. This is the part a referring
  clinician reads, and repeating the findings list back adds nothing.
    * Combine related observations into a single clinical statement.
    * Include a normal ONLY where its absence is clinically meaningful —
      "no pleural effusion" earns a place beside cardiomegaly because it
      argues against decompensation. "Clear lung fields" alone does not.
    * Order by significance, not by the order dictated.
    * If everything dictated is a single finding, the impression is one line.
      Do not manufacture items to fill a list.

  Dictated:  Mild cardiomegaly. No pleural effusion. Clear lung fields.
             Degenerative changes of the thoracic spine.
  Impression: 1. Mild cardiomegaly without pulmonary oedema or pleural
                 effusion.
              2. Degenerative change of the thoracic spine.
  NOT:        four numbered lines repeating all four findings.

- No preamble, no explanation, no conversational hedging.
- End with: *Draft for radiologist review - not a final report.*

That closing line is not decoration. This system produces drafts for a human
to approve, and an output that omits the label can be pasted into a record as
though it were signed.
""".strip()


# Minimum Qdrant similarity score to consider a chunk relevant.
# Below this, the context is probably noise and the LLM should say
# "I don't have enough information" rather than hallucinating from
# weak signal.
RELEVANCE_THRESHOLD = 0.35

# How many chunks to retrieve per question.
#
# ⚠️  WAS 5, RAISED TO 12 — here's why, because it's counter-intuitive:
# When the corpus was 14 documents, 5 was plenty. At 230+ documents, narrow
# research papers that repeat a term densely (a study on "pneumothorax after
# liposuction", an ML detection paper) outrank the passage that calmly
# *explains* the finding. Cosine similarity rewards term density, not
# answer-ness — so the genuinely useful chunk got pushed to rank 8-12 and the
# model correctly reported it had nothing to work with.
#
# Retrieving more is the cheap fix: gpt-oss-120b has a 131K context window, so
# 12 chunks (~6K chars) costs nothing meaningful, and adjacent-chunk merging
# collapses duplicates before the model sees them.
#
# The principled fix is two-stage retrieval — vector search for recall, then a
# cross-encoder rerank for precision. That's worth doing once there's an
# evaluation set to prove it helps.
DEFAULT_RETRIEVAL_LIMIT = settings.RETRIEVAL_LIMIT

# ── Source diversity ─────────────────────────────────────────
# ⚠️  THE ACTUAL FIX FOR THE PROBLEM ABOVE.
# Observed on a 232-document corpus, query "radiographic findings of
# pneumothorax". Of the top 20 chunks:
#
#     6  Value of focused lung ultrasound ... after CT-guided lung biopsy
#     4  Pneumothorax as a Complication of Liposuction
#     2  Pneumothorax — Types, Imaging, and Management   ← the one that answers
#     2  Enhanced pneumothorax visualization in ICU patients
#     ...
#
# Two papers took half the slots. Both mention "pneumothorax" constantly in
# narrow contexts, so every one of their chunks scores highly — while the
# passage that actually describes the visceral pleural line sits at rank ~13.
# The model then correctly reported it had nothing to answer with.
#
# Raising the retrieval limit doesn't fix this: the dominant paper simply
# takes more slots too. Capping per-document representation does — it forces
# the top-N to span multiple sources, which is also what you want from an
# evidence panel for a clinical tool. Corroboration across papers beats four
# excerpts from one.
#
# We over-fetch, cap, then trim back to `limit`.
MAX_CHUNKS_PER_DOCUMENT = settings.MAX_CHUNKS_PER_DOCUMENT
_OVERFETCH_FACTOR = 4

# How many document-derived terms to search with when a file is attached.
# Enough to characterise a report's subject matter (region, modality, the
# handful of named findings), few enough to stay inside the embedding model's
# 256-wordpiece window with the user's own question alongside it.
_DOC_QUERY_TERMS = 24


# ══════════════════════════════════════════════════════════════
# CITATION NORMALISATION
# ══════════════════════════════════════════════════════════════
# ⚠️  OBSERVED IN PRODUCTION with openai/gpt-oss-120b on Groq:
# the model emits citations using CJK "lenticular" brackets —
#
#     U+3010 【   LEFT BLACK LENTICULAR BRACKET
#     U+3011 】   RIGHT BLACK LENTICULAR BRACKET
#
# so the stream contained 【3】 where the prompt asked for [3]. The answer
# LOOKS correct to a human reader, which is what makes this nasty: nothing
# errors, nothing logs, and the text reads fine — but the frontend's citation
# parser finds zero matches, so no citation is clickable and the Evidence
# Panel silently stops working. The single feature Phase 3 exists to deliver
# fails invisibly.
#
# Prompting alone can't fix this reliably — it's a tokeniser-level habit of
# the model family. We normalise on the way out instead, so the API contract
# ("citations are [N]") holds no matter which provider answers.
#
# Note these arrive as SEPARATE tokens (【, 3, 】), so a per-token character
# replacement is sufficient and no cross-token buffering is needed.
_CITATION_BRACKETS = {
    "【": "[",   # 【
    "】": "]",   # 】
    "［": "[",   # ［ fullwidth
    "］": "]",   # ］ fullwidth
}


def normalise_citations(text: str) -> str:
    """
    Rewrite non-ASCII citation brackets to plain [ and ].

    Safe to apply per-token during streaming: each bracket is a single
    character, so replacement never depends on surrounding tokens.
    """
    if not text:
        return text
    for weird, plain in _CITATION_BRACKETS.items():
        if weird in text:
            text = text.replace(weird, plain)
    return text


# ══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════


@dataclass
class RetrievedChunk:
    """A single chunk retrieved from Qdrant, enriched with a 1-based ID."""
    chunk_id: int           # 1-based, matches [Source N] in prompt
    text: str
    score: float
    document_id: str | None = None
    document_title: str | None = None
    source_type: str | None = None
    chunk_index: int | None = None
    # Cross-encoder relevance (Phase 3.5). Unbounded logit, ordering only —
    # never render this as a percentage. `score` stays the cosine value.
    rerank_score: float | None = None
    # Qdrant point id — used to dedupe vector and lexical candidates.
    point_id: str | None = None
    # BM25 score when this chunk came from lexical retrieval.
    bm25_score: float | None = None


# ══════════════════════════════════════════════════════════════
# OUT-OF-SCOPE QUERY DETECTION
# ══════════════════════════════════════════════════════════════
# Cheap pre-filter for queries that obviously have nothing to do with
# medicine, so we don't pay for an embedding pass, a Qdrant search AND an
# LLM call just to answer "what's the weather".
#
# ⚠️  DELIBERATELY CONSERVATIVE. A false positive here refuses a legitimate
# clinical question, which is far worse than wasting one API call on a joke.
# So we only reject when BOTH are true:
#   1. the query matches an obvious non-medical pattern, and
#   2. it contains no medical vocabulary at all
#
# Anything ambiguous falls through to the normal pipeline, where the
# relevance threshold and the grounding scaffold are the real safety net.

_OUT_OF_SCOPE_PATTERNS = [
    r"\bweather\b",
    r"\btell me a joke\b",
    r"\bwho (won|is winning)\b.*\b(game|match|cup|election)\b",
    r"\bstock (price|market)\b",
    r"\b(recipe|cook|bake)\b",
    r"\bwrite (me )?(a )?(poem|song|story)\b",
    r"\btranslate\b.*\bto (spanish|french|german|urdu|chinese)\b",
    r"\bwhat('s| is) the capital of\b",
    r"\bsports? scores?\b",
]

# If any of these appear, treat the query as in-scope no matter what.
# Kept broad on purpose — recall matters more than precision here.
_MEDICAL_VOCABULARY = [
    "radiolog", "radiograph", "x-ray", "xray", "ct ", "mri", "ultrasound",
    "imaging", "scan", "contrast", "chest", "lung", "pulmonar", "cardiac",
    "abdom", "pleural", "pneumo", "fracture", "nodule", "lesion", "mass",
    "effusion", "embol", "stroke", "haemorrhage", "hemorrhage", "infarct",
    "diagnos", "differential", "findings", "patient", "clinical", "report",
    "anatomy", "patholog", "tumour", "tumor", "cancer", "artery", "vein",
    "bone", "brain", "liver", "kidney", "spine", "trauma", "sepsis",
    "oedema", "edema", "consolidation", "opacity", "modality", "sign",
]


def is_out_of_scope(query: str) -> bool:
    """
    True only if the query is clearly non-medical.

    Biased heavily toward answering: when in doubt, return False and let the
    full pipeline handle it.
    """
    lowered = query.lower().strip()

    # Any medical vocabulary at all → always in scope.
    if any(term in lowered for term in _MEDICAL_VOCABULARY):
        return False

    return any(re.search(p, lowered) for p in _OUT_OF_SCOPE_PATTERNS)


OUT_OF_SCOPE_REPLY = (
    "I'm designed to assist with radiology and medical imaging. "
    "Could you rephrase your question in that context?"
)


# ══════════════════════════════════════════════════════════════
# CONTEXT DEDUPLICATION
# ══════════════════════════════════════════════════════════════


def cap_per_document(
    chunks: list[RetrievedChunk],
    max_per_doc: int = MAX_CHUNKS_PER_DOCUMENT,
) -> list[RetrievedChunk]:
    """
    Keep at most `max_per_doc` chunks from any single document.

    Input is assumed to be sorted best-first (Qdrant returns it that way), and
    relative order is preserved — so this only ever removes lower-ranked chunks
    from documents that are already well represented.

    Prevents one term-dense paper from monopolising the context window and
    pushing genuinely explanatory passages out of range.
    """
    if max_per_doc <= 0:
        return chunks

    seen: dict[str, int] = {}
    kept: list[RetrievedChunk] = []

    for chunk in chunks:
        # Chunks with no document_id can't be grouped — keep them all rather
        # than silently collapsing unrelated content together.
        key = chunk.document_id or f"__unkeyed__{id(chunk)}"
        count = seen.get(key, 0)
        if count < max_per_doc:
            seen[key] = count + 1
            kept.append(chunk)

    return kept


def merge_adjacent_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    Merge chunks that come from the same document and sit next to each other.

    WHY:
    Chunks overlap by design (50 chars), so retrieving chunk 4 and chunk 5 of
    the same article hands the LLM the boundary text twice. That wastes context
    window, and repetition biases the model toward over-weighting whatever
    happens to be duplicated.

    Merging also produces a cleaner Evidence Panel: one coherent passage
    instead of two fragments that visibly overlap.

    Chunk IDs are renumbered 1..N afterwards so the [N] citations the model is
    told to use still line up with what the frontend receives.
    """
    if len(chunks) < 2:
        return chunks

    # Group by document, preserving the best score seen per document.
    ordered = sorted(
        chunks,
        key=lambda c: (c.document_id or "", c.chunk_index if c.chunk_index is not None else 0),
    )

    merged: list[RetrievedChunk] = []
    for chunk in ordered:
        prev = merged[-1] if merged else None

        is_adjacent = (
            prev is not None
            and prev.document_id == chunk.document_id
            and prev.chunk_index is not None
            and chunk.chunk_index is not None
            and chunk.chunk_index == prev.chunk_index + 1
        )

        if is_adjacent:
            # Stitch, removing the duplicated overlap region if we can find it.
            joined = _stitch(prev.text, chunk.text)
            prev.text = joined
            prev.chunk_index = chunk.chunk_index          # extend the range
            prev.score = max(prev.score, chunk.score)     # keep the best score
        else:
            merged.append(chunk)

    # Re-sort by relevance and renumber so [1] is the strongest source.
    merged.sort(key=lambda c: c.score, reverse=True)
    for i, chunk in enumerate(merged, start=1):
        chunk.chunk_id = i

    return merged


def _stitch(first: str, second: str, max_overlap: int = 200) -> str:
    """
    Join two consecutive chunks, dropping the repeated overlap if present.

    Looks for the longest suffix of `first` that is also a prefix of `second`
    and splices there. Falls back to a plain join when no overlap is found
    (which happens when the chunker split on a paragraph boundary).
    """
    window = min(len(first), len(second), max_overlap)
    for size in range(window, 20, -1):
        if first[-size:] == second[:size]:
            return first + second[size:]
    return f"{first}\n{second}"


@dataclass
class RAGResult:
    """The complete output of the RAG pipeline (non-streaming)."""
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    model: str = ""


# ══════════════════════════════════════════════════════════════
# RAG SERVICE
# ══════════════════════════════════════════════════════════════


class RAGService:
    _last_query_vector: list[float] | None = None

    """
    Orchestrates the full Retrieval-Augmented Generation pipeline.

    USAGE:
        from app.services.rag_service import rag_service

        # Non-streaming
        result = await rag_service.answer("findings of pneumothorax?")
        print(result.answer)     # The grounded answer
        print(result.sources)    # The chunks that were used

        # Streaming
        sources, stream = await rag_service.answer_stream("pneumothorax?")
        async for token in stream:
            print(token, end="")
        # sources is available immediately (retrieved before streaming starts)
    """

    # ── RETRIEVE ─────────────────────────────────────────────

    def _retrieve_sync(
        self,
        query: str,
        limit: int,
        source_type: str | None,
    ) -> list[RetrievedChunk]:
        """
        The blocking half of retrieval. Never call this directly from an
        async context — use `retrieve_context()`.
        """
        # Embed the query using the same model that embedded the documents.
        query_vector = embedding_service.encode_single(query)
        # Retained so lexical-only hits can be given a real cosine score
        # instead of a meaningless BM25 logit in the evidence panel.
        self._last_query_vector = query_vector

        # Search Qdrant.
        raw_results = qdrant_service.search(
            query_vector=query_vector,
            limit=limit,
            source_type=source_type,
            score_threshold=0.0,  # We apply our own threshold below
        )

        # Convert to RetrievedChunk with 1-based IDs.
        chunks: list[RetrievedChunk] = []
        for i, result in enumerate(raw_results, start=1):
            chunks.append(RetrievedChunk(
                chunk_id=i,
                text=result["text"],
                score=result["score"],
                document_id=result.get("document_id"),
                # Prefer the human-readable title; fall back to filename.
                # Seeded articles set both, uploads only set filename.
                document_title=result.get("title") or result.get("filename"),
                source_type=result.get("source_type"),
                chunk_index=result.get("chunk_index"),
                point_id=result.get("point_id"),
            ))

        return chunks

    async def retrieve_context(
        self,
        query: str,
        limit: int = DEFAULT_RETRIEVAL_LIMIT,
        source_type: str | None = None,
        attached_text: str | None = None,
        attached_warnings: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Embed the query and search Qdrant for relevant chunks.

        Returns chunks sorted by relevance (highest score first),
        each tagged with a 1-based chunk_id for citation.

        ⚠️  WHY run_in_threadpool — THE SAME BUG AS PHASE 2 INGESTION:
        `embedding_service.encode_single()` is synchronous, CPU-bound PyTorch,
        and `qdrant_service.search()` is a blocking HTTP call. Running either
        directly on the event loop stalls the ENTIRE server — every concurrent
        chat request, every health check — for the duration.

        It's less visible here than in ingestion (~50ms, not minutes), but it
        happens on every single chat message rather than on occasional uploads,
        so under any concurrency it serialises the whole application.
        """
        # ── Stage 0: what are we actually searching FOR? ──
        # With a document attached, the user's words are an instruction
        # ("review this"), not a description of the subject matter. Searching
        # the instruction retrieves papers about reviewing reports. Search the
        # DOCUMENT instead. See _build_document_query().
        search_query = query
        if attached_text:
            search_query = await run_in_threadpool(
                self._build_document_query, query, attached_text
            )

        # ── Stage 1: RECALL — vector + lexical, unioned ──
        # Two retrievers with near-orthogonal failure modes. Embeddings find
        # passages that are ABOUT the topic; BM25 finds passages containing
        # the exact terms. Measured need: the chunk answering "radiographic
        # findings of pneumothorax" was never in the vector candidate pool,
        # so reranking could not reach it.
        n_candidates = max(
            settings.RERANK_CANDIDATES if settings.RERANK_ENABLED else limit * _OVERFETCH_FACTOR,
            limit,
        )
        candidates = await run_in_threadpool(
            self._retrieve_sync, search_query, n_candidates, source_type
        )
        candidates = await self._add_lexical_candidates(
            search_query, candidates, source_type
        )
        if not candidates:
            return []

        # ── Stage 2: cross-encoder rerank for PRECISION ──
        candidates = await self._rerank(search_query, candidates)

        # ── Stage 3: diversity cap, THEN trim ──
        # ⚠️  ORDER MATTERS, AND IT CHANGED IN PHASE 3.5.
        # The cap used to run on the raw vector order. But the evaluation
        # baseline showed the correct document ranking #1 while the passage
        # answering the question went unretrieved — so capping before scoring
        # could discard the very chunk we needed, purely because two weaker
        # chunks from the same document happened to be embedded closer.
        # Capping AFTER reranking means we keep each document's genuinely most
        # relevant passages.
        diverse = cap_per_document(candidates, MAX_CHUNKS_PER_DOCUMENT)[:limit]

        # Collapse overlapping neighbours before the LLM ever sees them.
        return merge_adjacent_chunks(diverse)

    @staticmethod
    def _ensure_lexical_index() -> None:
        """Build (or rebuild after ingestion) the BM25 index. Blocking."""
        if not lexical_index.is_built:
            records = qdrant_service.iter_all_chunks()
            if records:
                lexical_index.build(records)

    def _build_document_query(self, query: str, attached_text: str) -> str:
        """
        Derive the retrieval query from an uploaded document. Blocking.

        ⚠️  THE BUG THIS FIXES.
        A clinician uploaded a spine report and typed a generic instruction.
        Retrieval embedded the instruction, so all twelve retrieved sources
        were papers about *the practice of radiology reporting* — reporting
        errors, structured-reporting templates, NLP reporting quality. Not one
        concerned osteopenia, vertebral compression or spinal curvature. The
        model cited [1] for every row of its table because no source genuinely
        supported any of them.

        The document is what the answer is about, so the document is what we
        search with. `salient_terms` scores it by TF-IDF against the live
        corpus, which drops boilerplate ("report", "findings") and OCR garbage
        ("hyperiordotic", "peivis") in one pass.

        The user's own words are kept: when they ask something specific rather
        than "review this", that intent should still steer retrieval. Generic
        instructions wash out on their own — their terms carry near-zero IDF.

        Falls back to the original query on any failure. Retrieval quality is
        an enhancement here; the document itself is already the primary source
        in the prompt, so a bad query degrades the background section rather
        than the answer.
        """
        try:
            self._ensure_lexical_index()
            terms = lexical_index.salient_terms(
                attached_text, top_k=_DOC_QUERY_TERMS
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not derive query from document: %s", e)
            terms = []

        if not terms:
            # No corpus statistics available (hybrid disabled, empty index, or
            # a document sharing no vocabulary with the corpus). The opening
            # of a report carries the modality, region and headline finding,
            # which still beats searching the instruction alone.
            return f"{query} {attached_text[:400]}".strip()

        logger.info("Document-derived retrieval terms: %s", " ".join(terms))
        return f"{query} {' '.join(terms)}".strip()

    async def _add_lexical_candidates(
        self,
        query: str,
        vector_hits: list[RetrievedChunk],
        source_type: str | None,
    ) -> list[RetrievedChunk]:
        """
        Union BM25 hits into the candidate pool, deduplicated by point ID.

        Order is irrelevant here — the cross-encoder rescores everything
        afterwards. The only job of this stage is to make sure the answering
        chunk is *present* to be scored.

        Returns `vector_hits` unchanged if hybrid retrieval is disabled or the
        index is unavailable. Never raises: lexical retrieval is an
        enhancement, and losing it must not break search.
        """
        if not settings.HYBRID_ENABLED:
            return vector_hits

        def _lexical_work() -> list[RetrievedChunk]:
            # Build (or rebuild after ingestion) on first use.
            self._ensure_lexical_index()
            if not lexical_index.is_built:
                return []

            hits = lexical_index.search(query, limit=settings.LEXICAL_CANDIDATES)
            if source_type:
                hits = [h for h in hits if h.get("source_type") == source_type]
            if not hits:
                return []

            seen = {c.point_id for c in vector_hits if c.point_id}
            new = [h for h in hits if h.get("point_id") not in seen]
            if not new:
                return []

            # Lexical-only hits have no cosine score, but the evidence panel
            # displays one. Fetch the stored vectors and compute the real
            # value rather than showing a BM25 logit as a "% match".
            cosines = qdrant_service.cosine_scores_for_points(
                [h["point_id"] for h in new], self._last_query_vector or []
            )

            return [
                RetrievedChunk(
                    chunk_id=0,                     # renumbered after reranking
                    text=h["text"],
                    score=cosines.get(h["point_id"], 0.0),
                    document_id=h.get("document_id"),
                    document_title=h.get("title") or h.get("filename"),
                    source_type=h.get("source_type"),
                    chunk_index=h.get("chunk_index"),
                    point_id=h.get("point_id"),
                    bm25_score=h.get("bm25_score"),
                )
                for h in new
            ]

        try:
            extra = await run_in_threadpool(_lexical_work)
        except Exception as e:  # noqa: BLE001
            logger.warning("Lexical retrieval failed, using vector only: %s", e)
            return vector_hits

        if extra:
            logger.debug("Lexical added %d candidates beyond vector search", len(extra))
        return vector_hits + extra

    async def _rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        Reorder chunks by cross-encoder relevance.

        Returns the input unchanged when reranking is unavailable — that is a
        supported state, not an error. See reranker.py.

        NOTE: `chunk.score` keeps the ORIGINAL cosine similarity. Cross-encoder
        outputs are unbounded logits (~-11..+11) and would be meaningless shown
        as "74% match" in the evidence panel. The rerank score is recorded
        separately for debugging; only the ordering changes.
        """
        scores = await run_in_threadpool(
            reranker_service.score, query, [c.text for c in chunks]
        )
        if scores is None:
            return chunks

        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = score

        ordered = sorted(chunks, key=lambda c: c.rerank_score or 0.0, reverse=True)

        if logger.isEnabledFor(logging.DEBUG):
            moved = sum(1 for i, c in enumerate(ordered) if chunks[i] is not c)
            logger.debug("Rerank reordered %d/%d chunks", moved, len(chunks))

        return ordered

    # ── BUILD PROMPT ─────────────────────────────────────────

    def build_messages(
        self,
        query: str,
        context: list[RetrievedChunk],
        *,
        mode: str = "qa",
        audience: str = "radiologist",
        attached_text: str | None = None,
        attached_warnings: list[str] | None = None,
        prior_text: str | None = None,
    ) -> list[dict]:
        """
        Assemble the full message list for the LLM.

        Structure:
            [0] system: role + grounding scaffold + formatted context
            [1] user:   the actual query

        The context is formatted so each chunk has a labeled [Source N]
        header that the LLM can cite inline.
        """
        # ── Select the mode-specific system prompt ──
        # Comparison is checked FIRST. It also arrives with attached_text (the
        # prior study), so any later branch would swallow it and the model
        # would analyse one document instead of comparing two.
        if mode == "comparison" or prior_text:
            role_prompt = COMPARISON_PROMPT
        elif mode == "report":
            role_prompt = REPORT_SYSTEM_PROMPT
        elif mode == "report_analysis" or attached_text:
            # An uploaded document changes the task fundamentally: the model
            # is reading a specific patient's report, not answering from a
            # corpus. See REPORT_ANALYSIS_PROMPT for what went wrong when
            # this distinction wasn't made.
            role_prompt = REPORT_ANALYSIS_PROMPT
        else:
            role_prompt = QA_SYSTEM_PROMPTS.get(
                audience,
                QA_SYSTEM_PROMPTS["radiologist"],
            )

        # ── Format the retrieved context ──
        if context:
            context_lines = ["", "CONTEXT (use ONLY this to answer):", ""]
            for chunk in context:
                # Header: [Source N] (title | type | score)
                title = chunk.document_title or "Unknown"
                stype = chunk.source_type or "general"
                header = (
                    f"[Source {chunk.chunk_id}] "
                    f"({title} | {stype} | score: {chunk.score:.2f})"
                )
                context_lines.append(header)
                context_lines.append(chunk.text)
                context_lines.append("")  # blank line separator

            context_block = "\n".join(context_lines)
        else:
            context_block = (
                "\n\nCONTEXT: No relevant information was found in the "
                "knowledge base for this query."
            )

        # ── Uploaded document goes FIRST, above the literature ──
        # Position is the fix. Appending it to the user's question made the
        # model treat retrieved papers as authoritative and the patient's
        # own report as loose material — which inverted a finding.
        document_block = ""
        if attached_text:
            caveat = ""
            if attached_warnings:
                caveat = (
                    "\n\n⚠️  TEXT QUALITY WARNINGS — the document was read by OCR "
                    "and may contain character errors:\n"
                    + "\n".join(f"  - {w}" for w in attached_warnings)
                    + "\nApply rule 5: repair harmless misspellings silently, "
                    "and flag only the words where a competing reading would "
                    "change the finding, level, number or management."
                )
            document_block = (
                "\n\n" + "=" * 60 + "\n"
                "UPLOADED DOCUMENT — THE PRIMARY SOURCE\n"
                + "=" * 60 + "\n"
                + attached_text
                + "\n" + "=" * 60
                + "\nEND OF UPLOADED DOCUMENT" + caveat + "\n"
            )

        # ── Prior study, and the arithmetic already done on it ──
        prior_block = ""
        if prior_text:
            from app.services.comparison_service import (
                compare_measurements,
                format_facts,
            )

            # ⚠️  NUMBERS ARE PAIRED AND DIFFERENCED BEFORE THE MODEL SEES THEM.
            # Measurements are where a language model is least reliable and
            # where an error costs most — Phase 4 opened with a stated 50%
            # coming back as "25-50%". The comparison is computed here and
            # handed over as settled fact, so the model narrates rather than
            # calculates.
            facts = format_facts(compare_measurements(prior_text, attached_text or query))

            prior_block = (
                "\n\n" + "=" * 60 + "\n"
                "PRIOR STUDY — for comparison\n"
                + "=" * 60 + "\n"
                + prior_text
                + "\n" + "=" * 60
                + "\nEND OF PRIOR STUDY\n"
                + (f"\n{facts}\n" if facts else "")
            )

        # ⚠️  THE SCAFFOLD MUST MATCH THE MODE.
        # GROUNDING_SCAFFOLD mandates a citation on every claim. Appending it
        # to REPORT_SYSTEM_PROMPT — which forbids citations in the report body
        # — gave the model two contradictory instructions, and the longer,
        # more detailed one won. Drafts came back reading "Mild cardiomegaly. 1".
        if mode == "comparison" or prior_text:
            scaffold = COMPARISON_GROUNDING_SCAFFOLD
        elif mode == "report":
            scaffold = REPORT_GROUNDING_SCAFFOLD
        else:
            scaffold = GROUNDING_SCAFFOLD

        system_content = (
            f"{role_prompt}\n\n{scaffold}"
            f"{prior_block}{document_block}\n{context_block}"
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]

    # ── CHECK RELEVANCE ──────────────────────────────────────

    @staticmethod
    def _has_relevant_context(context: list[RetrievedChunk]) -> bool:
        """
        Return True if at least one chunk meets the relevance threshold.

        If all chunks score below RELEVANCE_THRESHOLD, the LLM would be
        generating from noise — better to return a clear "I don't know."
        """
        if not context:
            return False
        return any(c.score >= RELEVANCE_THRESHOLD for c in context)

    # ── ANSWER (non-streaming) ───────────────────────────────

    async def answer(
        self,
        query: str,
        *,
        mode: str = "qa",
        audience: str = "radiologist",
        limit: int = DEFAULT_RETRIEVAL_LIMIT,
        source_type: str | None = None,
        attached_text: str | None = None,
        attached_warnings: list[str] | None = None,
        prior_text: str | None = None,
    ) -> RAGResult:
        """
        Full RAG pipeline: retrieve → build prompt → generate.

        Returns a RAGResult with the answer text, source chunks, and
        the model name that produced the answer.
        """
        # 0. Cheap out-of-scope check — costs nothing, skips embedding +
        #    Qdrant + LLM entirely for obviously non-medical queries.
        #    Never applied when a document is attached: the instruction may be
        #    as bare as "have a look", and refusing to read an uploaded report
        #    on the strength of the covering sentence would be absurd.
        # Report mode is exempt too: dictated findings are terse clinical
        # fragments, not questions, and "Mild cardiomegaly. No effusion."
        # should never be tested against a conversational scope filter.
        if mode == "qa" and not attached_text and is_out_of_scope(query):
            logger.info("Out-of-scope query rejected without LLM call: %s", query)
            return RAGResult(
                answer=OUT_OF_SCOPE_REPLY,
                sources=[],
                model=llm_service.active_model,
            )

        # 1. Retrieve context from Qdrant.
        context = await self.retrieve_context(
            query, limit=limit, source_type=source_type,
            attached_text=attached_text,
        )

        # 2. Check relevance — if nothing relevant, short-circuit.
        # UNLESS a document was uploaded: the model is reading THAT, and
        # weak corpus coverage is irrelevant to whether it can do so.
        # Report mode is exempt for the same reason an attachment is: the
        # clinical content comes from the dictation, not the corpus. Refusing
        # to draft a report because the knowledge base has nothing similar
        # would make the feature fail on exactly the unusual findings where a
        # structured draft is most useful.
        if (
            mode not in ("report", "comparison")
            and not attached_text
            and not prior_text
            and not self._has_relevant_context(context)
        ):
            logger.info("No relevant context for query: %s (best score: %.3f)",
                        query, context[0].score if context else 0.0)
            return RAGResult(
                answer=(
                    "Based on my current knowledge base, I don't have enough "
                    "information to answer that question. Try rephrasing, or "
                    "check if relevant documents have been uploaded to the "
                    "knowledge base."
                ),
                sources=context,  # Return what we found (even if low-scoring)
                model=llm_service.active_model,
            )

        # 3. Build the prompt.
        messages = self.build_messages(
            query, context, mode=mode, audience=audience,
            attached_text=attached_text, attached_warnings=attached_warnings,
            prior_text=prior_text,
        )

        # 4. Generate the answer.
        answer_text = await llm_service.generate(messages)

        return RAGResult(
            # Normalise 【N】 → [N] so the citation contract holds regardless
            # of which model answered. See normalise_citations().
            answer=normalise_citations(answer_text),
            sources=context,
            model=llm_service.active_model,
        )

    # ── ANSWER (streaming) ───────────────────────────────────

    async def answer_stream(
        self,
        query: str,
        *,
        mode: str = "qa",
        audience: str = "radiologist",
        limit: int = DEFAULT_RETRIEVAL_LIMIT,
        source_type: str | None = None,
        attached_text: str | None = None,
        attached_warnings: list[str] | None = None,
        prior_text: str | None = None,
    ) -> tuple[list[RetrievedChunk], AsyncIterator[str]]:
        """
        Streaming RAG pipeline: retrieve → build prompt → stream tokens.

        Returns a tuple of (sources, token_stream):
        - sources: the retrieved chunks (available IMMEDIATELY, before
          streaming starts — the frontend needs them for the evidence panel)
        - token_stream: an async iterator yielding tokens one at a time

        If no relevant context is found, the stream yields the "I don't
        have enough information" message and stops.
        """
        # 0. Out-of-scope check (see answer() for rationale, including why an
        #    attached document bypasses it).
        # Report mode is exempt too: dictated findings are terse clinical
        # fragments, not questions, and "Mild cardiomegaly. No effusion."
        # should never be tested against a conversational scope filter.
        if mode == "qa" and not attached_text and is_out_of_scope(query):
            logger.info("Out-of-scope query rejected without LLM call: %s", query)

            async def _out_of_scope_stream() -> AsyncIterator[str]:
                yield OUT_OF_SCOPE_REPLY

            return [], _out_of_scope_stream()

        # 1. Retrieve context.
        context = await self.retrieve_context(
            query, limit=limit, source_type=source_type,
            attached_text=attached_text,
        )

        # 2. Check relevance (skipped when a document was uploaded — see answer()).
        # Report mode is exempt for the same reason an attachment is: the
        # clinical content comes from the dictation, not the corpus. Refusing
        # to draft a report because the knowledge base has nothing similar
        # would make the feature fail on exactly the unusual findings where a
        # structured draft is most useful.
        if (
            mode not in ("report", "comparison")
            and not attached_text
            and not prior_text
            and not self._has_relevant_context(context)
        ):
            logger.info("No relevant context (streaming) for: %s", query)

            async def _no_context_stream() -> AsyncIterator[str]:
                yield (
                    "Based on my current knowledge base, I don't have enough "
                    "information to answer that question. Try rephrasing, or "
                    "check if relevant documents have been uploaded to the "
                    "knowledge base."
                )

            return context, _no_context_stream()

        # 3. Build prompt.
        messages = self.build_messages(
            query, context, mode=mode, audience=audience,
            attached_text=attached_text, attached_warnings=attached_warnings,
            prior_text=prior_text,
        )

        # 4. Stream the answer, normalising citation brackets per token.
        async def _normalised_stream() -> AsyncIterator[str]:
            async for token in llm_service.generate_stream(messages):
                yield normalise_citations(token)

        return context, _normalised_stream()


# ══════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ══════════════════════════════════════════════════════════════
# Import this anywhere with:
#   from app.services.rag_service import rag_service
# ══════════════════════════════════════════════════════════════

rag_service = RAGService()
