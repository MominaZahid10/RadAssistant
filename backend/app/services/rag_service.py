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


# ── Report Generation prompt ──
# Strictly clinical register — output goes into a medical record.
REPORT_SYSTEM_PROMPT = (
    "You are RadAssist AI generating a radiology report section. Output "
    "ONLY the report text — no preamble, no explanation, no hedging prose. "
    "Use standard radiology reporting conventions: terse declarative "
    "sentences, standard terminology, structured format (Findings / "
    "Impression). Cite evidence from context with inline [N] references."
)


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
            ))

        return chunks

    async def retrieve_context(
        self,
        query: str,
        limit: int = DEFAULT_RETRIEVAL_LIMIT,
        source_type: str | None = None,
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
        # ── Stage 1: vector search for RECALL ──
        # Deliberately over-fetch. Cheap, and the precision stage below is
        # what decides what actually reaches the model.
        n_candidates = max(
            settings.RERANK_CANDIDATES if settings.RERANK_ENABLED else limit * _OVERFETCH_FACTOR,
            limit,
        )
        candidates = await run_in_threadpool(
            self._retrieve_sync, query, n_candidates, source_type
        )
        if not candidates:
            return []

        # ── Stage 2: cross-encoder rerank for PRECISION ──
        candidates = await self._rerank(query, candidates)

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
        if mode == "report":
            role_prompt = REPORT_SYSTEM_PROMPT
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

        # ── Assemble the system message ──
        system_content = f"{role_prompt}\n\n{GROUNDING_SCAFFOLD}\n{context_block}"

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
    ) -> RAGResult:
        """
        Full RAG pipeline: retrieve → build prompt → generate.

        Returns a RAGResult with the answer text, source chunks, and
        the model name that produced the answer.
        """
        # 0. Cheap out-of-scope check — costs nothing, skips embedding +
        #    Qdrant + LLM entirely for obviously non-medical queries.
        if is_out_of_scope(query):
            logger.info("Out-of-scope query rejected without LLM call: %s", query)
            return RAGResult(
                answer=OUT_OF_SCOPE_REPLY,
                sources=[],
                model=llm_service.active_model,
            )

        # 1. Retrieve context from Qdrant.
        context = await self.retrieve_context(query, limit=limit, source_type=source_type)

        # 2. Check relevance — if nothing relevant, short-circuit.
        if not self._has_relevant_context(context):
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
        # 0. Out-of-scope check (see answer() for rationale).
        if is_out_of_scope(query):
            logger.info("Out-of-scope query rejected without LLM call: %s", query)

            async def _out_of_scope_stream() -> AsyncIterator[str]:
                yield OUT_OF_SCOPE_REPLY

            return [], _out_of_scope_stream()

        # 1. Retrieve context.
        context = await self.retrieve_context(query, limit=limit, source_type=source_type)

        # 2. Check relevance.
        if not self._has_relevant_context(context):
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
