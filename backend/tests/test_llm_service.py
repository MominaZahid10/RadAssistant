"""
Tests for the multi-provider LLM service (Phase 3).

All API calls are mocked — these run without any API key and never spend
tokens, so they're safe in CI and fast enough to run on every save.

The most important test here is `test_no_fallback_after_streaming_starts`.
See the comment on that test for why.
"""

import pytest

from app.services import llm_service as llm_mod
from app.services.llm_service import LLMService, _DEFAULT_MODELS, _FALLBACK_ORDER


# ══════════════════════════════════════════════════════════════
# FAKE PROVIDERS
# ══════════════════════════════════════════════════════════════


class FakeProvider:
    """A scriptable stand-in for a real LLM provider."""

    def __init__(
        self,
        name: str,
        *,
        configured: bool = True,
        complete_result: str = "ok",
        complete_error: Exception | None = None,
        stream_tokens: list[str] | None = None,
        stream_error: Exception | None = None,
        error_after: int | None = None,
    ):
        self.name = name
        self._configured = configured
        self._complete_result = complete_result
        self._complete_error = complete_error
        self._stream_tokens = stream_tokens or ["a", "b", "c"]
        self._stream_error = stream_error
        self._error_after = error_after
        self.calls = 0

    def is_configured(self) -> bool:
        return self._configured

    async def complete(self, messages, *, model, temperature, max_tokens) -> str:
        self.calls += 1
        if self._complete_error:
            raise self._complete_error
        return self._complete_result

    async def stream(self, messages, *, model, temperature, max_tokens):
        self.calls += 1
        if self._stream_error:
            raise self._stream_error
        for i, token in enumerate(self._stream_tokens):
            if self._error_after is not None and i == self._error_after:
                raise RuntimeError(f"{self.name} died mid-stream")
            yield token


@pytest.fixture
def service(monkeypatch):
    """An LLMService with all three providers replaced by fakes."""
    monkeypatch.setattr(llm_mod.settings, "LLM_PROVIDER", "groq", raising=False)
    monkeypatch.setattr(llm_mod.settings, "LLM_MODEL", "", raising=False)
    monkeypatch.setattr(llm_mod.settings, "LLM_TEMPERATURE", 0.2, raising=False)
    monkeypatch.setattr(llm_mod.settings, "LLM_MAX_TOKENS", 100, raising=False)
    return LLMService()


def _install(service, **providers):
    service._providers = providers
    return service


# ══════════════════════════════════════════════════════════════
# MODEL DEFAULTS
# ══════════════════════════════════════════════════════════════


def test_groq_default_is_not_a_retired_model():
    """
    Groq shut down llama-3.3-70b-versatile and llama-3.1-8b-instant on
    2026-08-16. Requests to retired IDs return errors, so defaulting to one
    breaks every chat request with a confusing 404.

    https://console.groq.com/docs/deprecations
    """
    retired = {
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "qwen/qwen3-32b",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    }
    assert _DEFAULT_MODELS["groq"] not in retired


def test_every_provider_has_a_default_model():
    for provider in _FALLBACK_ORDER:
        assert _DEFAULT_MODELS.get(provider), f"{provider} has no default model"


# ══════════════════════════════════════════════════════════════
# FALLBACK CHAIN
# ══════════════════════════════════════════════════════════════


def test_chain_puts_primary_first(service):
    _install(
        service,
        groq=FakeProvider("groq"),
        mistral=FakeProvider("mistral"),
        openai=FakeProvider("openai"),
    )
    assert [p.name for p in service._get_fallback_chain()] == ["groq", "mistral", "openai"]


def test_chain_skips_unconfigured_providers(service):
    _install(
        service,
        groq=FakeProvider("groq", configured=False),
        mistral=FakeProvider("mistral"),
        openai=FakeProvider("openai", configured=False),
    )
    assert [p.name for p in service._get_fallback_chain()] == ["mistral"]


def test_chain_honours_non_default_primary(service, monkeypatch):
    monkeypatch.setattr(llm_mod.settings, "LLM_PROVIDER", "openai", raising=False)
    svc = LLMService()
    _install(
        svc,
        groq=FakeProvider("groq"),
        mistral=FakeProvider("mistral"),
        openai=FakeProvider("openai"),
    )
    assert svc._get_fallback_chain()[0].name == "openai"


@pytest.mark.asyncio
async def test_generate_falls_back_when_primary_fails(service):
    primary = FakeProvider("groq", complete_error=RuntimeError("rate limited"))
    backup = FakeProvider("mistral", complete_result="from mistral")
    _install(service, groq=primary, mistral=backup, openai=FakeProvider("openai", configured=False))

    assert await service.generate([{"role": "user", "content": "hi"}]) == "from mistral"
    assert primary.calls == 1 and backup.calls == 1


@pytest.mark.asyncio
async def test_generate_raises_when_nothing_configured(service):
    _install(
        service,
        groq=FakeProvider("groq", configured=False),
        mistral=FakeProvider("mistral", configured=False),
        openai=FakeProvider("openai", configured=False),
    )
    with pytest.raises(RuntimeError, match="No LLM provider configured"):
        await service.generate([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_generate_raises_when_all_providers_fail(service):
    _install(
        service,
        groq=FakeProvider("groq", complete_error=RuntimeError("boom")),
        mistral=FakeProvider("mistral", complete_error=RuntimeError("bang")),
        openai=FakeProvider("openai", configured=False),
    )
    with pytest.raises(RuntimeError, match="All LLM providers failed"):
        await service.generate([{"role": "user", "content": "hi"}])


# ══════════════════════════════════════════════════════════════
# STREAMING
# ══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_stream_yields_tokens_in_order(service):
    _install(service, groq=FakeProvider("groq", stream_tokens=["The", " chest", " film"]))
    out = [t async for t in service.generate_stream([{"role": "user", "content": "hi"}])]
    assert out == ["The", " chest", " film"]


@pytest.mark.asyncio
async def test_stream_falls_back_before_first_token(service):
    """Failing to even open the stream is invisible to the client — fall back."""
    primary = FakeProvider("groq", stream_error=RuntimeError("401 unauthorized"))
    backup = FakeProvider("mistral", stream_tokens=["x", "y"])
    _install(service, groq=primary, mistral=backup, openai=FakeProvider("openai", configured=False))

    out = [t async for t in service.generate_stream([{"role": "user", "content": "hi"}])]
    assert out == ["x", "y"]


@pytest.mark.asyncio
async def test_no_fallback_after_streaming_starts(service):
    """
    THE IMPORTANT ONE.

    If a provider dies after emitting tokens, we must NOT restart on the next
    provider — the client would receive provider A's partial answer spliced
    onto provider B's complete answer as a single message.

    For a clinical tool that's worse than a visible failure: the output reads
    as coherent prose but is two models' reasoning stitched together, and the
    [1]/[2] citations in the first half may not correspond to the sources
    backing the second half.
    """
    dying = FakeProvider("groq", stream_tokens=["Pneumo", "thorax", " is"], error_after=2)
    backup = FakeProvider("mistral", stream_tokens=["COMPLETELY", " DIFFERENT"])
    _install(service, groq=dying, mistral=backup, openai=FakeProvider("openai", configured=False))

    received = []
    with pytest.raises(RuntimeError, match="died mid-stream"):
        async for token in service.generate_stream([{"role": "user", "content": "hi"}]):
            received.append(token)

    assert received == ["Pneumo", "thorax"]
    assert backup.calls == 0, "fallback provider must not run after streaming began"


# ══════════════════════════════════════════════════════════════
# CONNECTIVITY / INTROSPECTION
# ══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_connectivity_reports_the_provider_that_actually_answered(service):
    """
    A health check that reports the primary provider while a fallback is
    really serving traffic is worse than no health check.
    """
    _install(
        service,
        groq=FakeProvider("groq", complete_error=RuntimeError("down")),
        mistral=FakeProvider("mistral", complete_result="OK"),
        openai=FakeProvider("openai", configured=False),
    )
    result = await service.check_connectivity()

    assert result["status"] == "ok"
    assert result["provider"] == "mistral"
    assert result["is_primary"] is False


@pytest.mark.asyncio
async def test_connectivity_reports_not_configured(service):
    _install(
        service,
        groq=FakeProvider("groq", configured=False),
        mistral=FakeProvider("mistral", configured=False),
        openai=FakeProvider("openai", configured=False),
    )
    assert (await service.check_connectivity())["status"] == "not_configured"


def test_provider_info_shape(service):
    _install(
        service,
        groq=FakeProvider("groq"),
        mistral=FakeProvider("mistral", configured=False),
        openai=FakeProvider("openai", configured=False),
    )
    info = service.get_provider_info()

    assert info["active_provider"] == "groq"
    assert info["active_model"] == _DEFAULT_MODELS["groq"]
    assert info["providers"]["groq"]["configured"] is True
    assert info["providers"]["mistral"]["configured"] is False
