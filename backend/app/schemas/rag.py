"""
RadAssist AI — RAG Schemas (Phase 3)

Request/response shapes for the chat endpoint.
These define the JSON contract between the frontend and the RAG pipeline.
"""

from pydantic import BaseModel, Field
from enum import Enum


# ══════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════


class ChatMode(str, Enum):
    """
    What the model is being asked to DO, as opposed to who for (Audience).

    qa         — answer a question from the knowledge base
    report     — draft a structured report from findings the user typed
    comparison — compare a prior study against the current findings

    ⚠️  WHY THIS IS AN ENUM AND NOT A FREE STRING.
    An unrecognised mode must fail at validation with a 422. The alternative —
    falling back to "qa" — means a clinician who asked for a report gets a
    chat answer and no error anywhere. That is the same class of silent
    substitution that inverted a clinical finding in Phase 4: the system did
    something reasonable-looking instead of the thing that was asked for, and
    said nothing.

    ⚠️  AND WHY THIS FIELD HAD TO EXIST AT ALL.
    REPORT_SYSTEM_PROMPT and `mode="report"` were written in Phase 3 and
    supported by rag_service throughout — but chat.py hardcoded mode="qa" at
    both call sites, so no request could ever reach it. Report generation,
    the headline deliverable, was unreachable code for two phases.
    """
    QA = "qa"
    REPORT = "report"
    COMPARISON = "comparison"


class Audience(str, Enum):
    """
    Who the Q&A response is written for.  Controls the register
    (tone/detail level) of the system prompt — one-line swap.

    radiologist — Concise, standard terminology, minimal explanation.
    resident    — Step-by-step reasoning, defines terms, references mnemonics.
    """
    RADIOLOGIST = "radiologist"
    RESIDENT = "resident"


# ══════════════════════════════════════════════════════════════
# REQUEST
# ══════════════════════════════════════════════════════════════


class ChatRequest(BaseModel):
    """
    A question sent to the RAG chat endpoint.

    Example:
    {
        "query": "What are the radiographic findings of pneumothorax?",
        "stream": true,
        "audience": "radiologist",
        "include_sources": false
    }
    """
    query: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="The user's question or instruction.",
    )
    mode: ChatMode = Field(
        default=ChatMode.QA,
        description=(
            "What to produce.\n\n"
            "`qa` (default) — answer the question from the knowledge base.\n\n"
            "`report` — treat the input as radiology FINDINGS and draft a "
            "structured report (Findings / Impression) in clinical register. "
            "Output goes into a medical record, so it carries no "
            "conversational preamble or hedging prose.\n\n"
            "`comparison` — compare `prior_text` against the current findings "
            "and report what is new, no longer mentioned, unchanged, or "
            "reported differently."
        ),
    )
    prior_text: str | None = Field(
        default=None,
        max_length=50_000,
        description=(
            "A previous report to compare against. Measurements in both are "
            "paired and differenced deterministically before the model sees "
            "them, so the comparison narrates settled arithmetic rather than "
            "computing it.\n\n"
            "The output states both values and never characterises the "
            "difference: 8mm previously and 9mm now is either interval growth "
            "or inter-reader variation, and two reports cannot distinguish "
            "them."
        ),
    )
    stream: bool = Field(
        default=True,
        description="If true, response streams token-by-token via SSE.",
    )
    audience: Audience = Field(
        default=Audience.RADIOLOGIST,
        description=(
            "Who the response is written for. "
            "'radiologist' = concise, standard terminology. "
            "'resident' = explains reasoning, defines terms."
        ),
    )
    attached_text: str | None = Field(
        default=None,
        max_length=50_000,
        description=(
            "Text of a document the user uploaded (e.g. OCR of a report "
            "photo). Sent SEPARATELY from the question, not concatenated "
            "into it: appending it to the query made the model treat the "
            "retrieved literature as authoritative and the patient's own "
            "report as loose material — which inverted a clinical finding. "
            "Supplied here, it is placed above the literature and named as "
            "the primary source."
        ),
    )
    attached_warnings: list[str] | None = Field(
        default=None,
        description=(
            "Quality caveats about attached_text, e.g. low OCR confidence. "
            "Passed to the model so it can flag unreliable passages rather "
            "than stating misread text as fact."
        ),
    )
    include_sources: bool = Field(
        default=True,
        description=(
            "If true (default), the response includes the retrieved source "
            "chunks used to generate the answer.\n\n"
            "Defaults to TRUE because the grounding prompt instructs the model "
            "to cite every claim inline as [1], [2] — so with sources omitted, "
            "the answer contains citations pointing at nothing the client ever "
            "received. Traceability is the product; set this to false only for "
            "programmatic callers that genuinely don't render evidence."
        ),
    )


# ══════════════════════════════════════════════════════════════
# SOURCE REFERENCE
# ══════════════════════════════════════════════════════════════


class SourceReference(BaseModel):
    """
    A single retrieved chunk that was used as evidence for the answer.

    The chunk_id (1-based) matches the inline [1], [2] citations in
    the LLM's answer text, so the frontend can highlight/link them.
    """
    chunk_id: int = Field(
        description="1-based ID matching inline citations [1], [2], etc.",
    )
    text: str = Field(
        description="The actual text content of the chunk.",
    )
    score: float = Field(
        description="Cosine similarity score (0.0–1.0). Higher = more relevant.",
    )
    document_title: str | None = Field(
        default=None,
        description="Title of the source document.",
    )
    source_type: str | None = Field(
        default=None,
        description="Type of source: textbook, guideline, etc.",
    )
    chunk_index: int | None = Field(
        default=None,
        description="Position of this chunk within its parent document.",
    )
    document_id: str | None = Field(
        default=None,
        description="UUID of the source document in PostgreSQL.",
    )


# ══════════════════════════════════════════════════════════════
# RESPONSE (non-streaming)
# ══════════════════════════════════════════════════════════════


class ChatResponse(BaseModel):
    """
    Full chat response (used when stream=false).

    Example:
    {
        "answer": "A pneumothorax presents with... [1] ... [2]",
        "sources": [...],
        "query": "What are the findings of pneumothorax?",
        "model": "llama-3.3-70b-versatile"
    }
    """
    answer: str
    sources: list[SourceReference] | None = None
    query: str
    model: str


# ══════════════════════════════════════════════════════════════
# MODEL INFO (for /chat/models endpoint)
# ══════════════════════════════════════════════════════════════


class PMCFetchRequest(BaseModel):
    """
    Options for PMC Open Access ingestion.

    Sent as a JSON body rather than query parameters — `topics` is a list of
    free-text medical phrases containing spaces and commas, which is painful
    to express in a URL and easy to get silently wrong. An earlier version
    used query params, and a request sending topics in the body had them
    quietly ignored in favour of the defaults, with a 202 response that looked
    successful.
    """
    topics: list[str] | None = Field(
        default=None,
        description=(
            "Search phrases to ingest. Omit to use the built-in radiology "
            "topic list spanning chest, neuro, MSK, abdominal, breast and "
            "paediatric imaging."
        ),
        examples=[["pneumothorax chest imaging", "pleural effusion imaging"]],
    )
    max_per_topic: int = Field(
        default=10, ge=1, le=50,
        description="Articles to retrieve per topic (1-50).",
    )


class FigureFetchRequest(BaseModel):
    """
    Options for PMC figure extraction.

    A JSON body rather than query parameters, for the same reason
    PMCFetchRequest is: an earlier endpoint declared its options as query
    params while the client sent them in the body, so they were silently
    ignored and the 202 looked successful.
    """
    limit_documents: int | None = Field(
        default=None, ge=1,
        description=(
            "Process only the N most recently ingested PMC articles. Useful "
            "for a quick trial run before committing to the whole corpus. "
            "Omit to process everything."
        ),
        examples=[10],
    )
    max_figures_per_document: int = Field(
        default=8, ge=1, le=30,
        description=(
            "Cap per article. Review papers can carry 40+ figures, most of "
            "them multi-panel plates that add little beyond the first few."
        ),
    )


class ProviderInfo(BaseModel):
    """Info about a single LLM provider."""
    configured: bool
    default_model: str


class ModelInfoResponse(BaseModel):
    """Response from GET /chat/models."""
    active_provider: str
    active_model: str
    providers: dict[str, ProviderInfo]
