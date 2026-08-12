"""
Tests for the Phase 3 chat endpoint.

These tests mock the RAG and LLM services so they run without API keys,
Qdrant, or network access — same philosophy as the existing test suite.

Run:
    cd backend && pytest tests/test_chat.py -v
"""

import json
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Stub LLM provider SDKs before any app imports ──────────────────
# The existing conftest stubs embeddings/qdrant/docparsers.
# We additionally need stubs for groq, mistralai, openai so that
# llm_service.py can import without the packages being installed
# (or without them trying to open network connections).

for mod_name in ("groq", "mistralai", "openai"):
    sys.modules.setdefault(mod_name, types.ModuleType(mod_name))

# Provide the lazy-imported classes that the providers call
groq_mod = sys.modules["groq"]
if not hasattr(groq_mod, "AsyncGroq"):
    groq_mod.AsyncGroq = MagicMock  # type: ignore

mistral_mod = sys.modules["mistralai"]
if not hasattr(mistral_mod, "Mistral"):
    mistral_mod.Mistral = MagicMock  # type: ignore

openai_mod = sys.modules["openai"]
if not hasattr(openai_mod, "AsyncOpenAI"):
    openai_mod.AsyncOpenAI = MagicMock  # type: ignore

# NOW we can safely import the app
from app.services.rag_service import RAGService, RetrievedChunk, RAGResult
from app.services.llm_service import LLMService


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════


def _make_chunks(n: int = 3) -> list[RetrievedChunk]:
    """Create N fake retrieved chunks for testing."""
    return [
        RetrievedChunk(
            chunk_id=i,
            text=f"Chunk {i} text about radiology findings.",
            score=0.9 - (i * 0.1),
            document_id=f"doc-{i}",
            document_title=f"Document {i}",
            source_type="textbook",
            chunk_index=i,
        )
        for i in range(1, n + 1)
    ]


def _make_rag_result(answer: str = "Test answer [1].", n_chunks: int = 3) -> RAGResult:
    """Create a fake RAGResult for non-streaming tests."""
    return RAGResult(
        answer=answer,
        sources=_make_chunks(n_chunks),
        model="test-model-v1",
    )


async def _fake_token_stream():
    """Async generator that yields tokens one at a time."""
    for token in ["The ", "findings ", "include ", "opacity."]:
        yield token


# ══════════════════════════════════════════════════════════════
# TEST FIXTURES
# ══════════════════════════════════════════════════════════════


@pytest.fixture
def mock_rag_service():
    """Mock rag_service with controllable return values."""
    with patch("app.api.v1.endpoints.chat.rag_service") as mock:
        mock.answer = AsyncMock(return_value=_make_rag_result())
        mock.answer_stream = AsyncMock(
            return_value=(_make_chunks(), _fake_token_stream())
        )
        yield mock


@pytest.fixture
def mock_llm_service():
    """Mock llm_service to provide model name."""
    with patch("app.api.v1.endpoints.chat.llm_service") as mock:
        mock.active_model = "test-model-v1"
        mock.get_provider_info.return_value = {
            "active_provider": "groq",
            "active_model": "test-model-v1",
            "providers": {
                "groq": {"configured": True, "default_model": "llama-3.3-70b-versatile"},
                "mistral": {"configured": False, "default_model": "mistral-large-latest"},
                "openai": {"configured": False, "default_model": "gpt-4o-mini"},
            },
        }
        yield mock


@pytest.fixture
def client(mock_rag_service, mock_llm_service):
    """Create a TestClient with mocked services."""
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


# ══════════════════════════════════════════════════════════════
# TESTS: POST /api/v1/chat (non-streaming)
# ══════════════════════════════════════════════════════════════


class TestChatNonStreaming:
    """Tests for POST /api/v1/chat with stream=false."""

    def test_basic_chat(self, client, mock_rag_service):
        """A basic question returns a valid ChatResponse."""
        response = client.post(
            "/api/v1/chat",
            json={
                "query": "What is pneumothorax?",
                "stream": False,
                "audience": "radiologist",
                "include_sources": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Test answer [1]."
        assert data["query"] == "What is pneumothorax?"
        assert data["model"] == "test-model-v1"
        assert len(data["sources"]) == 3

    def test_chat_without_sources(self, client, mock_rag_service):
        """When include_sources=false, sources should be null."""
        response = client.post(
            "/api/v1/chat",
            json={
                "query": "What is pneumothorax?",
                "stream": False,
                "include_sources": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["sources"] is None

    def test_source_fields(self, client, mock_rag_service):
        """Each source has the expected fields."""
        response = client.post(
            "/api/v1/chat",
            json={"query": "test", "stream": False, "include_sources": True},
        )
        source = response.json()["sources"][0]
        assert source["chunk_id"] == 1
        assert "text" in source
        assert "score" in source
        assert "document_title" in source
        assert "source_type" in source

    def test_audience_default(self, client, mock_rag_service):
        """Default audience is 'radiologist' when not specified."""
        client.post(
            "/api/v1/chat",
            json={"query": "test", "stream": False},
        )
        # Check the audience passed to rag_service.answer
        call_kwargs = mock_rag_service.answer.call_args
        assert call_kwargs.kwargs.get("audience") == "radiologist"

    def test_audience_resident(self, client, mock_rag_service):
        """Audience 'resident' is passed through correctly."""
        client.post(
            "/api/v1/chat",
            json={"query": "test", "stream": False, "audience": "resident"},
        )
        call_kwargs = mock_rag_service.answer.call_args
        assert call_kwargs.kwargs.get("audience") == "resident"


# ══════════════════════════════════════════════════════════════
# TESTS: Validation
# ══════════════════════════════════════════════════════════════


class TestChatValidation:
    """Tests for request validation."""

    def test_empty_query_rejected(self, client):
        """An empty query should return 422."""
        response = client.post(
            "/api/v1/chat",
            json={"query": "", "stream": False},
        )
        assert response.status_code == 422

    def test_single_char_query_rejected(self, client):
        """A single-character query (below min_length=2) should return 422."""
        response = client.post(
            "/api/v1/chat",
            json={"query": "x", "stream": False},
        )
        assert response.status_code == 422

    def test_too_long_query_rejected(self, client):
        """A query exceeding max_length=2000 should return 422."""
        response = client.post(
            "/api/v1/chat",
            json={"query": "x" * 2001, "stream": False},
        )
        assert response.status_code == 422

    def test_invalid_audience_rejected(self, client):
        """An invalid audience value should return 422."""
        response = client.post(
            "/api/v1/chat",
            json={"query": "test query", "stream": False, "audience": "student"},
        )
        assert response.status_code == 422

    def test_missing_query_rejected(self, client):
        """A request without a query field should return 422."""
        response = client.post(
            "/api/v1/chat",
            json={"stream": False},
        )
        assert response.status_code == 422


# ══════════════════════════════════════════════════════════════
# TESTS: Error handling
# ══════════════════════════════════════════════════════════════


class TestChatErrors:
    """Tests for error conditions."""

    def test_no_llm_configured_returns_503(self, client, mock_rag_service):
        """RuntimeError (no LLM configured) → 503."""
        mock_rag_service.answer = AsyncMock(
            side_effect=RuntimeError("No LLM provider configured")
        )
        response = client.post(
            "/api/v1/chat",
            json={"query": "test query", "stream": False},
        )
        assert response.status_code == 503
        assert "No LLM provider configured" in response.json()["detail"]

    def test_unexpected_error_returns_500(self, client, mock_rag_service):
        """Unexpected exceptions → 500."""
        mock_rag_service.answer = AsyncMock(
            side_effect=ValueError("Something unexpected")
        )
        response = client.post(
            "/api/v1/chat",
            json={"query": "test query", "stream": False},
        )
        assert response.status_code == 500


# ══════════════════════════════════════════════════════════════
# TESTS: POST /api/v1/chat (streaming)
# ══════════════════════════════════════════════════════════════


class TestChatStreaming:
    """Tests for POST /api/v1/chat with stream=true (SSE)."""

    def test_stream_returns_sse(self, client, mock_rag_service, mock_llm_service):
        """Streaming response has text/event-stream content type."""
        response = client.post(
            "/api/v1/chat",
            json={"query": "test query", "stream": True, "include_sources": True},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

    def test_stream_event_sequence(self, client, mock_rag_service, mock_llm_service):
        """SSE stream follows: sources → token(s) → done."""
        response = client.post(
            "/api/v1/chat",
            json={"query": "test query", "stream": True, "include_sources": True},
        )
        body = response.text
        events = _parse_sse_events(body)

        # First event should be sources
        assert events[0]["event"] == "sources"
        assert "sources" in events[0]["data"]

        # Middle events should be tokens
        token_events = [e for e in events if e["event"] == "token"]
        assert len(token_events) > 0

        # Tokens should reconstruct the original text
        full_text = "".join(e["data"]["token"] for e in token_events)
        assert full_text == "The findings include opacity."

        # Last event should be done
        assert events[-1]["event"] == "done"
        assert events[-1]["data"]["model"] == "test-model-v1"

    def test_stream_without_sources(self, client, mock_rag_service, mock_llm_service):
        """When include_sources=false, sources event has empty list."""
        response = client.post(
            "/api/v1/chat",
            json={"query": "test query", "stream": True, "include_sources": False},
        )
        events = _parse_sse_events(response.text)
        sources_event = next(e for e in events if e["event"] == "sources")
        assert sources_event["data"]["sources"] == []


# ══════════════════════════════════════════════════════════════
# TESTS: GET /api/v1/chat/models
# ══════════════════════════════════════════════════════════════


class TestChatModels:
    """Tests for GET /api/v1/chat/models."""

    def test_get_models(self, client, mock_llm_service):
        """Returns model info with provider details."""
        response = client.get("/api/v1/chat/models")
        assert response.status_code == 200
        data = response.json()
        assert data["active_provider"] == "groq"
        assert data["active_model"] == "test-model-v1"
        assert "groq" in data["providers"]
        assert data["providers"]["groq"]["configured"] is True
        assert data["providers"]["mistral"]["configured"] is False


# ══════════════════════════════════════════════════════════════
# SSE PARSER HELPER
# ══════════════════════════════════════════════════════════════


def _parse_sse_events(body: str) -> list[dict]:
    """Parse a raw SSE response body into a list of {event, data} dicts."""
    events = []
    current_event = ""
    current_data = ""

    for line in body.strip().split("\n"):
        if line.startswith("event: "):
            current_event = line[7:].strip()
        elif line.startswith("data: "):
            current_data = line[6:]
        elif line == "":
            if current_event and current_data:
                events.append({
                    "event": current_event,
                    "data": json.loads(current_data),
                })
                current_event = ""
                current_data = ""

    # Handle last event if no trailing blank line
    if current_event and current_data:
        events.append({
            "event": current_event,
            "data": json.loads(current_data),
        })

    return events


# ══════════════════════════════════════════════════════════════
# TESTS: Report mode (Phase 5, Step 0)
# ══════════════════════════════════════════════════════════════
#
# ⚠️  WHY THESE EXIST.
# REPORT_SYSTEM_PROMPT and rag_service's mode="report" branch were written in
# Phase 3 and supported throughout — but chat.py hardcoded mode="qa" at both
# call sites, so no request could ever reach them. Report generation, the
# project's headline feature and the stated Phase 3 deliverable, was
# unreachable code for two phases and nothing failed.
#
# Every test here asserts the WIRING, not the prompt. A prompt nothing can
# invoke is worth exactly nothing.


class TestChatMode:
    def test_mode_defaults_to_qa(self, client, mock_rag_service):
        client.post("/api/v1/chat", json={"query": "test", "stream": False})
        assert mock_rag_service.answer.call_args.kwargs.get("mode") == "qa"

    def test_report_mode_reaches_the_service(self, client, mock_rag_service):
        """The regression this whole step exists to prevent."""
        client.post(
            "/api/v1/chat",
            json={"query": "Mild cardiomegaly.", "stream": False, "mode": "report"},
        )
        assert mock_rag_service.answer.call_args.kwargs.get("mode") == "report"

    def test_report_mode_reaches_the_service_when_streaming(
        self, client, mock_rag_service, mock_llm_service
    ):
        """Both call sites were hardcoded. Both must be covered."""
        with client.stream(
            "POST",
            "/api/v1/chat",
            json={"query": "Mild cardiomegaly.", "stream": True, "mode": "report"},
        ) as response:
            list(response.iter_lines())
        assert mock_rag_service.answer_stream.call_args.kwargs.get("mode") == "report"

    def test_unknown_mode_is_rejected_not_silently_downgraded(self, client):
        """
        ⚠️  A 422 IS THE POINT.
        Falling back to "qa" would mean a clinician who asked for a report
        gets a chat answer, with no error anywhere in the system. That is the
        same silent substitution that inverted a clinical finding in Phase 4:
        something reasonable-looking happened instead of what was asked for.
        """
        response = client.post(
            "/api/v1/chat",
            json={"query": "test", "stream": False, "mode": "summary"},
        )
        assert response.status_code == 422


class TestReportPrompt:
    """The prompt's clinical safety rules, asserted so they cannot be edited away."""

    def test_report_prompt_forbids_adding_unstated_findings(self):
        from app.services.rag_service import REPORT_SYSTEM_PROMPT

        p = REPORT_SYSTEM_PROMPT
        assert "NEVER ADD A FINDING" in p
        # The specific danger: completing a study with its usual normals.
        assert "fabricated" in p
        assert "no pleural effusion" in p

    def test_report_prompt_protects_numbers_levels_and_laterality(self):
        from app.services.rag_service import REPORT_SYSTEM_PROMPT

        p = REPORT_SYSTEM_PROMPT
        assert "NEVER ALTER A NUMBER, LEVEL OR LATERALITY" in p
        assert "8mm stays 8mm" in p

    def test_report_prompt_separates_language_from_content(self):
        """Context supplies phrasing; it must never contribute a finding."""
        from app.services.rag_service import REPORT_SYSTEM_PROMPT

        assert "USE THE CONTEXT FOR LANGUAGE, NOT FOR CONTENT" in REPORT_SYSTEM_PROMPT

    def test_report_prompt_requires_the_draft_label(self):
        """
        The project document states this as a design constraint: the system
        produces drafts only and must make the human-in-the-loop explicit.
        An unlabelled output can be pasted into a record as though signed.
        """
        from app.services.rag_service import REPORT_SYSTEM_PROMPT

        assert "Draft for radiologist review" in REPORT_SYSTEM_PROMPT

    def test_report_mode_selects_the_report_prompt(self):
        from app.services.rag_service import (
            REPORT_SYSTEM_PROMPT,
            RAGService,
            RetrievedChunk,
        )

        messages = RAGService().build_messages(
            "Mild cardiomegaly. No pleural effusion.",
            [RetrievedChunk(chunk_id=1, text="Cardiomegaly is defined as...",
                            score=0.8, document_title="Chest imaging")],
            mode="report",
        )
        assert REPORT_SYSTEM_PROMPT in messages[0]["content"]


class TestReportPromptOutputRules:
    """
    ⚠️  DEFECTS OBSERVED IN THE FIRST REAL DRAFT.
    The wiring worked; the output had three problems, two of them clinical:

      "* Mild cardiomegaly 4"   a citation rendering as a bare number, which
                                in a medical record reads as a severity grade
      IMPRESSION 1-4            a 1:1 restatement of the findings list, with
                                normals ("Clear lung fields") given their own
                                numbered impression items
    """

    def test_inline_citations_are_forbidden_in_the_report_body(self):
        from app.services.rag_service import REPORT_SYSTEM_PROMPT

        p = REPORT_SYSTEM_PROMPT
        assert "NO INLINE CITATIONS ANYWHERE IN THE REPORT BODY" in p
        # The observed rendering, kept as the worked example.
        assert "Mild cardiomegaly 4" in p
        assert "severity score" in p
        # Traceability is not dropped, only relocated.
        assert "evidence panel" in p.lower()

    def test_impression_must_synthesise_not_restate(self):
        from app.services.rag_service import REPORT_SYSTEM_PROMPT

        p = REPORT_SYSTEM_PROMPT
        assert "SYNTHESIS, NOT A RESTATEMENT" in p
        assert "Do not manufacture items to fill a list" in p

    def test_normals_need_to_earn_their_place_in_the_impression(self):
        from app.services.rag_service import REPORT_SYSTEM_PROMPT

        p = REPORT_SYSTEM_PROMPT
        assert "clinically meaningful" in p
        # "no pleural effusion" belongs beside cardiomegaly; "clear lung
        # fields" on its own does not.
        assert "Clear lung fields" in p

    def test_prompt_carries_a_worked_example(self):
        """An abstract rule about synthesis is easy to comply with nominally."""
        from app.services.rag_service import REPORT_SYSTEM_PROMPT

        p = REPORT_SYSTEM_PROMPT
        assert "Dictated:" in p and "Impression:" in p and "NOT:" in p


class TestScaffoldMatchesMode:
    """
    ⚠️  CONTRADICTORY INSTRUCTIONS DO NOT AVERAGE OUT.
    GROUNDING_SCAFFOLD was appended to every mode. In report mode that put
    "NO INLINE CITATIONS ANYWHERE IN THE REPORT BODY" two lines above "Every
    factual claim MUST have at least one citation" plus a CITATION FORMAT
    block. The longer, more specific instruction won, and drafts came back
    reading "Mild cardiomegaly. 1" — a bare number that in a medical record
    reads as a severity grade.

    You do not get to choose which of two conflicting rules the model obeys.
    The only fix is not to send both.
    """

    def _system(self, mode: str) -> str:
        from app.services.rag_service import RAGService, RetrievedChunk

        return RAGService().build_messages(
            "Mild cardiomegaly.",
            [RetrievedChunk(chunk_id=1, text="Cardiomegaly is CTR > 0.5.",
                            score=0.8, document_title="Chest imaging")],
            mode=mode,
        )[0]["content"]

    def test_report_mode_does_not_receive_the_citation_mandate(self):
        system = self._system("report")
        assert "Every factual claim MUST have at least one citation" not in system
        assert "CITATION FORMAT" not in system

    def test_report_mode_receives_its_own_scaffold(self):
        from app.services.rag_service import REPORT_GROUNDING_SCAFFOLD

        assert REPORT_GROUNDING_SCAFFOLD in self._system("report")

    def test_report_scaffold_forbids_bare_numbers_too(self):
        """[1] was banned; the model then emitted '1'. Both are excluded."""
        from app.services.rag_service import REPORT_GROUNDING_SCAFFOLD

        assert "bare number" in REPORT_GROUNDING_SCAFFOLD

    def test_report_mode_drops_the_answer_only_from_context_rule(self):
        """
        Wrong for a report: the clinical content comes from the dictation,
        not the corpus.
        """
        system = self._system("report")
        assert "Answer ONLY from the CONTEXT" not in system

    def test_report_mode_drops_the_differential_framing_rule(self):
        """A report states findings; it is not a differential discussion."""
        system = self._system("report")
        assert "differential considerations" not in system

    def test_qa_mode_still_gets_the_full_grounding_scaffold(self):
        """Regression guard: the fix must not weaken the Q&A path."""
        from app.services.rag_service import GROUNDING_SCAFFOLD

        system = self._system("qa")
        assert GROUNDING_SCAFFOLD in system
        assert "Every factual claim MUST have at least one citation" in system
