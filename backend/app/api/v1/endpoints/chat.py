"""
RadAssist AI — Chat Endpoint (Phase 3)

THIS IS WHERE THE RAG PIPELINE MEETS THE OUTSIDE WORLD.

Two routes:
    POST /api/v1/chat       → Ask the AI a question (streaming or non-streaming)
    GET  /api/v1/chat/models → See which LLM providers are configured

STREAMING (SSE):
When the frontend sends { "stream": true }, this endpoint returns a
Server-Sent Events stream.  The event protocol is:

    event: sources
    data: {"sources": [...]}        ← sent FIRST so the UI can show evidence immediately

    event: token
    data: {"token": "The"}          ← one per token, arrives as the LLM generates

    event: token
    data: {"token": " findings"}

    event: done
    data: {"model": "llama-3.3-70b-versatile"}  ← signals end of stream

    event: error
    data: {"error": "All LLM providers failed..."}  ← only on failure

WHY SSE INSTEAD OF WEBSOCKETS?
SSE is simpler (unidirectional, auto-reconnects, works through proxies),
and chat is inherently request-response — the client sends one question,
the server streams one answer.  WebSockets add complexity we don't need.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.rag import (
    ChatRequest,
    ChatResponse,
    SourceReference,
    ModelInfoResponse,
)
from app.core.limits import (
    CHAT,
    CHAT_DAILY,
    INSTANCE_DAILY,
    per_instance,
    per_user,
)
from app.services.rag_service import rag_service
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


# ══════════════════════════════════════════════════════════════
# POST /api/v1/chat — Ask the AI
# ══════════════════════════════════════════════════════════════


@router.post(
    "",
    summary="Ask RadAssist AI a question",
    description=(
        "Send a medical/radiology question. The RAG pipeline retrieves "
        "relevant evidence from the knowledge base and generates a grounded "
        "answer. Set `stream=true` for token-by-token SSE streaming."
    ),
    # Every message is a paid model call. 20/min sits far above human typing
    # speed and far below what a script could spend.
    # ⚠️  THREE CEILINGS, ANSWERING THREE DIFFERENT QUESTIONS.
    #   CHAT           — burst control. Stops a script hammering the endpoint.
    #   CHAT_DAILY     — what one account may spend in a day. On a public
    #                    demo with one shared login, this is the public
    #                    allowance.
    #   INSTANCE_DAILY — what the whole deployment may spend in a day.
    # A per-minute cap alone permits 28,800 paid calls a day, which is a
    # burst limit wearing a budget's clothes.
    dependencies=[
        Depends(per_user(CHAT, "chat")),
        Depends(per_user(CHAT_DAILY, "chat-daily")),
        Depends(per_instance(INSTANCE_DAILY, "chat-daily")),
    ],
)
async def chat(request: ChatRequest):
    """
    Main chat endpoint — the single entry point for all AI interactions.

    Two modes:
    - stream=false → returns ChatResponse JSON (simple, good for testing)
    - stream=true  → returns SSE stream (production, typewriter UX)
    """
    if request.stream:
        return _stream_response(request)
    else:
        return await _complete_response(request)


# ── Non-streaming path ───────────────────────────────────────


async def _complete_response(request: ChatRequest) -> ChatResponse:
    """
    Full response at once — retrieve, generate, return.
    Used when stream=false or for programmatic API consumers.
    """
    try:
        result = await rag_service.answer(
            query=request.query,
            # ⚠️  WAS HARDCODED "qa". REPORT_SYSTEM_PROMPT existed and
            # rag_service supported mode="report" from Phase 3 onward, but
            # these two lines meant nothing could ever reach it — report
            # generation, the project's headline feature, was dead code.
            mode=request.mode.value,
            audience=request.audience.value,
            attached_text=request.attached_text,
            attached_warnings=request.attached_warnings,
            prior_text=request.prior_text,
        )
    except RuntimeError as e:
        # No LLM provider configured, or all providers failed
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        # ⚠️  THE EXCEPTION IS NOT PUT IN THE RESPONSE.
        # It used to be interpolated into `detail`, handing the caller file
        # paths and driver messages. logger.exception records the traceback
        # against this request's id; the caller gets that id and nothing else.
        logger.exception("Unexpected error in chat")
        raise HTTPException(status_code=500, detail="Internal error.")

    # Build source references (only if requested)
    sources = None
    if request.include_sources:
        sources = [
            SourceReference(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=chunk.score,
                document_title=chunk.document_title,
                source_type=chunk.source_type,
                chunk_index=chunk.chunk_index,
                document_id=chunk.document_id,
            )
            for chunk in result.sources
        ]

    return ChatResponse(
        answer=result.answer,
        sources=sources,
        query=request.query,
        model=result.model,
    )


# ── Streaming path (SSE) ─────────────────────────────────────


def _stream_response(request: ChatRequest) -> StreamingResponse:
    """
    Return a StreamingResponse that yields SSE events.

    The generator is wrapped in StreamingResponse — FastAPI sends each
    yielded string to the client as it's produced, without buffering
    the entire response.
    """
    return StreamingResponse(
        _sse_generator(request),
        media_type="text/event-stream",
        headers={
            # Prevent proxy buffering (nginx, cloudflare, etc.)
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _sse_generator(request: ChatRequest) -> AsyncIterator[str]:
    """
    The actual SSE event generator.

    Event sequence:
    1. sources  — retrieved chunks (always sent, even if empty)
    2. token    — one per LLM token
    3. done     — signals end of generation
    4. error    — only on failure (replaces done)
    """
    try:
        # Get the stream + sources from the RAG service.
        sources, token_stream = await rag_service.answer_stream(
            query=request.query,
            # ⚠️  WAS HARDCODED "qa". REPORT_SYSTEM_PROMPT existed and
            # rag_service supported mode="report" from Phase 3 onward, but
            # these two lines meant nothing could ever reach it — report
            # generation, the project's headline feature, was dead code.
            mode=request.mode.value,
            audience=request.audience.value,
            attached_text=request.attached_text,
            attached_warnings=request.attached_warnings,
            prior_text=request.prior_text,
        )

        # ── Event 1: Sources ──
        # Always sent first so the frontend can render the evidence panel
        # before the first token arrives.
        sources_data = []
        if request.include_sources:
            sources_data = [
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "score": chunk.score,
                    "document_title": chunk.document_title,
                    "source_type": chunk.source_type,
                    "chunk_index": chunk.chunk_index,
                    "document_id": chunk.document_id,
                }
                for chunk in sources
            ]

        yield _sse_event("sources", {"sources": sources_data})

        # ── Event 2+: Tokens ──
        # Each token from the LLM is sent as a separate SSE event.
        async for token in token_stream:
            yield _sse_event("token", {"token": token})

        # ── Final Event: Done ──
        yield _sse_event("done", {"model": llm_service.active_model})

    except RuntimeError as e:
        # No LLM configured or all providers failed
        logger.error("Chat stream error: %s", e)
        yield _sse_event("error", {"error": str(e)})

    except Exception:
        logger.exception("Unexpected error in chat stream")
        # Same reasoning as above: an SSE error event reaches the browser and
        # ends up in client-side logs, so it carries no exception text either.
        yield _sse_event("error", {"error": "Internal error."})


def _sse_event(event: str, data: dict) -> str:
    """
    Format a single SSE event.

    SSE spec: each event is "event: <name>\ndata: <json>\n\n"
    The double newline is the event terminator.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ══════════════════════════════════════════════════════════════
# GET /api/v1/chat/models — Provider info
# ══════════════════════════════════════════════════════════════


@router.get(
    "/models",
    response_model=ModelInfoResponse,
    summary="Get LLM provider info",
    description=(
        "Returns which LLM providers are configured and which model "
        "is currently active. Useful for the frontend status display."
    ),
)
async def get_models() -> ModelInfoResponse:
    """Return info about all configured LLM providers."""
    info = llm_service.get_provider_info()
    return ModelInfoResponse(**info)
