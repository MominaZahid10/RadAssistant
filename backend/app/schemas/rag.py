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


class ProviderInfo(BaseModel):
    """Info about a single LLM provider."""
    configured: bool
    default_model: str


class ModelInfoResponse(BaseModel):
    """Response from GET /chat/models."""
    active_provider: str
    active_model: str
    providers: dict[str, ProviderInfo]
