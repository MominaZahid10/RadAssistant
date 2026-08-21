"""
RadAssist AI — LLM Service (Phase 3)

WHAT THIS FILE DOES:
Provides a single, provider-agnostic interface for calling Large Language
Models.  The rest of the codebase never imports `groq`, `mistralai`, or
`openai` directly — it calls `llm_service.generate()` or
`llm_service.generate_stream()` and this module routes to whichever
provider is configured.

PROVIDER ARCHITECTURE:
    ┌─────────────────────────────────────────────────┐
    │               LLMService (facade)               │
    │  generate(messages) → str                       │
    │  generate_stream(messages) → AsyncIterator[str] │
    └────────────────┬────────────────────────────────┘
                     │ delegates to active provider
         ┌───────────┼───────────────┐
         ▼           ▼               ▼
    GroqProvider  MistralProvider  OpenAIProvider
    (AsyncGroq)  (Mistral)        (AsyncOpenAI)

WHY MULTIPLE PROVIDERS?
- Groq:    Free tier, fast inference — great for development
- Mistral: Free tier (limited) — fallback when Groq is rate-limited
- OpenAI:  Paid, highest quality — production use

SWITCHING:
Set LLM_PROVIDER in .env to "groq", "mistral", or "openai".
The service auto-selects a default model per provider when LLM_MODEL is blank.

FALLBACK:
If the primary provider fails (rate limit, network error, missing key),
the service tries the next configured provider automatically.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Default models per provider — used when LLM_MODEL is blank.
#
# ⚠️  GROQ MODEL LIFECYCLE — CHECK THIS BEFORE DEBUGGING 404s:
# Groq retires models aggressively. `llama-3.3-70b-versatile` and
# `llama-3.1-8b-instant` both shut down on 2026-08-16; requests to them now
# return errors. Groq's own recommended replacement for the 70B model is
# `openai/gpt-oss-120b`, which is a Production-tier model (131K context,
# 65K max completion, ~500 tok/sec).
#
# If generation suddenly starts failing, check:
#   https://console.groq.com/docs/deprecations
# and list what your key can actually reach:
#   curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"
_DEFAULT_MODELS: dict[str, str] = {
    "groq": "openai/gpt-oss-120b",
    "mistral": "mistral-large-latest",
    # ⚠️  THIS CONSTANT IS WHAT THE OPENAI PATH ACTUALLY USES.
    # `_resolve_model` honours settings.LLM_MODEL only on the PRIMARY
    # provider. When OpenAI is configured as a FALLBACK — which is the
    # recommended shape for this deployment, Groq first for speed and cost,
    # OpenAI to catch a rate limit or a retired Groq model — LLM_MODEL is
    # ignored and this default is what runs. Leaving a stale name here means
    # the fallback 404s at exactly the moment it was supposed to save you.
    "openai": "gpt-5.6-luna",
}

# Provider priority for fallback — tried in this order.
_FALLBACK_ORDER: list[str] = ["groq", "mistral", "openai"]


# ══════════════════════════════════════════════════════════════
# PROVIDER INTERFACE
# ══════════════════════════════════════════════════════════════


class LLMProvider(ABC):
    """
    Base class that every provider implements.

    The contract is small on purpose — two methods — so adding a new
    provider (Anthropic, Gemini, local Ollama, …) is a single file
    with two async functions.
    """

    name: str

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Return the full completion as a single string."""

    @abstractmethod
    async def stream(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Yield tokens one at a time as they arrive from the API."""

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if the provider has a valid API key."""


# ══════════════════════════════════════════════════════════════
# GROQ PROVIDER
# ══════════════════════════════════════════════════════════════


class GroqProvider(LLMProvider):
    """
    Groq — free tier, extremely fast inference on open models.
    Uses the official `groq` Python SDK (AsyncGroq for non-blocking calls).
    """

    name = "groq"

    def __init__(self) -> None:
        self._api_key = settings.GROQ_API_KEY
        self._client = None

    def _get_client(self):
        """Lazy-init the async client so import-time errors don't crash startup."""
        if self._client is None:
            from groq import AsyncGroq
            self._client = AsyncGroq(api_key=self._api_key)
        return self._client

    def is_configured(self) -> bool:
        return bool(self._api_key and self._api_key.strip())

    async def complete(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        return response.choices[0].message.content or ""

    async def stream(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        client = self._get_client()
        response = await client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content is not None:
                yield delta.content


# ══════════════════════════════════════════════════════════════
# MISTRAL PROVIDER
# ══════════════════════════════════════════════════════════════


class MistralProvider(LLMProvider):
    """
    Mistral AI — quality models with a limited free tier.
    Uses the official `mistralai` SDK v2 (Mistral client).
    """

    name = "mistral"

    def __init__(self) -> None:
        self._api_key = settings.MISTRAL_API_KEY
        self._client = None

    def _get_client(self):
        if self._client is None:
            from mistralai import Mistral
            self._client = Mistral(api_key=self._api_key)
        return self._client

    def is_configured(self) -> bool:
        return bool(self._api_key and self._api_key.strip())

    async def complete(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        client = self._get_client()
        response = await client.chat.complete_async(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    async def stream(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        client = self._get_client()
        response = await client.chat.stream_async(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        async for chunk in response:
            content = chunk.data.choices[0].delta.content
            if content is not None:
                yield content


# ══════════════════════════════════════════════════════════════
# OPENAI PROVIDER
# ══════════════════════════════════════════════════════════════


class OpenAIProvider(LLMProvider):
    """
    OpenAI — premium quality, paid API.  Production provider.
    Uses the official `openai` SDK (AsyncOpenAI).
    """

    name = "openai"

    def __init__(self) -> None:
        self._api_key = settings.OPENAI_API_KEY
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    def is_configured(self) -> bool:
        return bool(self._api_key and self._api_key.strip())

    async def complete(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        return response.choices[0].message.content or ""

    async def stream(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        client = self._get_client()
        response = await client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content is not None:
                yield delta.content


# ══════════════════════════════════════════════════════════════
# LLM SERVICE (FACADE)
# ══════════════════════════════════════════════════════════════


class LLMService:
    """
    Unified LLM interface used by the rest of the application.

    USAGE:
        from app.services.llm_service import llm_service

        # Non-streaming (full response at once)
        answer = await llm_service.generate(messages)

        # Streaming (token-by-token)
        async for token in llm_service.generate_stream(messages):
            print(token, end="")

    The active provider is set by LLM_PROVIDER in .env.
    If it fails, the service falls back through the other configured providers.
    """

    def __init__(self) -> None:
        # Build the provider registry — all providers are instantiated
        # but only the configured ones (with API keys) are usable.
        self._providers: dict[str, LLMProvider] = {
            "groq": GroqProvider(),
            "mistral": MistralProvider(),
            "openai": OpenAIProvider(),
        }
        self._primary_name = settings.LLM_PROVIDER.lower()

    @property
    def active_provider_name(self) -> str:
        """The name of the currently configured primary provider."""
        return self._primary_name

    @property
    def active_model(self) -> str:
        """The model that will be used for generation."""
        if settings.LLM_MODEL:
            return settings.LLM_MODEL
        return _DEFAULT_MODELS.get(self._primary_name, "unknown")

    def get_provider_info(self) -> dict:
        """
        Return info about all providers — used by /health and /chat/models.
        """
        return {
            "active_provider": self._primary_name,
            "active_model": self.active_model,
            "providers": {
                name: {
                    "configured": provider.is_configured(),
                    "default_model": _DEFAULT_MODELS.get(name, "unknown"),
                }
                for name, provider in self._providers.items()
            },
        }

    def _get_fallback_chain(self) -> list[LLMProvider]:
        """
        Build the ordered list of providers to try.

        Primary first, then the rest in _FALLBACK_ORDER, skipping
        any that don't have an API key configured.
        """
        chain: list[LLMProvider] = []

        # Primary provider first (if configured)
        primary = self._providers.get(self._primary_name)
        if primary and primary.is_configured():
            chain.append(primary)

        # Then the rest, in fallback order
        for name in _FALLBACK_ORDER:
            if name == self._primary_name:
                continue
            provider = self._providers.get(name)
            if provider and provider.is_configured():
                chain.append(provider)

        return chain

    def _resolve_model(self, provider: LLMProvider) -> str:
        """
        Decide which model to use for a given provider.

        If the user set LLM_MODEL explicitly AND we're on the primary
        provider, use it.  Otherwise use the provider's default.
        """
        if settings.LLM_MODEL and provider.name == self._primary_name:
            return settings.LLM_MODEL
        return _DEFAULT_MODELS.get(provider.name, "unknown")

    async def generate(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Generate a complete response (non-streaming).

        Tries the primary provider, then falls back through configured
        alternatives if the primary fails.

        Args:
            messages: OpenAI-style message list
                      [{"role": "system", "content": "..."}, ...]
            temperature: Override the global LLM_TEMPERATURE setting.
            max_tokens: Override the global LLM_MAX_TOKENS setting.

        Returns:
            The full response text.

        Raises:
            RuntimeError: If no provider is configured or all providers fail.
        """
        temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
        tokens = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS

        chain = self._get_fallback_chain()
        if not chain:
            raise RuntimeError(
                "No LLM provider configured. Set at least one of "
                "GROQ_API_KEY, MISTRAL_API_KEY, or OPENAI_API_KEY in .env"
            )

        last_error: Exception | None = None
        for provider in chain:
            model = self._resolve_model(provider)
            try:
                logger.info("LLM generate: trying %s / %s", provider.name, model)
                result = await provider.complete(
                    messages,
                    model=model,
                    temperature=temp,
                    max_tokens=tokens,
                )
                logger.info("LLM generate: %s succeeded (%d chars)", provider.name, len(result))
                return result
            except Exception as e:
                last_error = e
                logger.warning(
                    "LLM provider '%s' failed: %s — trying next",
                    provider.name,
                    e,
                )

        raise RuntimeError(
            f"All LLM providers failed. Last error: {last_error}"
        )

    async def generate_stream(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream tokens one at a time from the LLM.

        Same fallback logic as generate(), but yields tokens as they arrive.

        NOTE: Fallback only happens BEFORE streaming starts.  If a provider
        begins streaming and then errors mid-stream, that error propagates
        rather than silently switching providers (which would produce a
        Frankenstein response from two different models).

        Yields:
            Individual text tokens as strings.

        Raises:
            RuntimeError: If no provider is configured or all providers fail
                          to *start* streaming.
        """
        temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
        tokens = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS

        chain = self._get_fallback_chain()
        if not chain:
            raise RuntimeError(
                "No LLM provider configured. Set at least one of "
                "GROQ_API_KEY, MISTRAL_API_KEY, or OPENAI_API_KEY in .env"
            )

        last_error: Exception | None = None
        for provider in chain:
            model = self._resolve_model(provider)

            # ⚠️  THE LOAD-BEARING FLAG.
            # Without this, an error AFTER tokens have already been yielded
            # would be caught below and the loop would restart on the next
            # provider — the client would receive provider A's partial answer
            # immediately followed by provider B's complete answer, spliced
            # into one message. For a clinical tool that is worse than an
            # outright failure: the output looks coherent but is two different
            # models' reasoning stitched together, and the citations from the
            # first half may not match the sources of the second.
            #
            # So: fall back only while we can still do so invisibly, i.e.
            # before a single token has reached the client.
            started = False

            try:
                logger.info("LLM stream: trying %s / %s", provider.name, model)
                stream = provider.stream(
                    messages,
                    model=model,
                    temperature=temp,
                    max_tokens=tokens,
                )
                async for token in stream:
                    started = True
                    yield token

                logger.info("LLM stream: %s completed", provider.name)
                return

            except Exception as e:
                if started:
                    # Committed. Surface the real error rather than silently
                    # producing a two-model Frankenstein response.
                    logger.error(
                        "LLM provider '%s' failed MID-STREAM: %s — "
                        "not falling back (would splice two models' output)",
                        provider.name,
                        e,
                    )
                    raise

                last_error = e
                logger.warning(
                    "LLM provider '%s' failed before first token: %s — trying next",
                    provider.name,
                    e,
                )

        raise RuntimeError(
            f"All LLM providers failed to stream. Last error: {last_error}"
        )

    async def check_connectivity(self) -> dict:
        """
        Quick connectivity check — sends a tiny prompt to verify the API key works.
        Used by the health check and startup verification.

        Returns:
            {"status": "ok", "provider": "groq", "model": "..."} on success,
            {"status": "error", "error": "..."} on failure.
        """
        chain = self._get_fallback_chain()
        if not chain:
            return {
                "status": "not_configured",
                "provider": self._primary_name,
                "error": (
                    "No LLM provider has an API key. Set GROQ_API_KEY, "
                    "MISTRAL_API_KEY, or OPENAI_API_KEY in .env"
                ),
            }

        # Call the provider DIRECTLY rather than via generate(), so the result
        # reports which provider actually answered. Going through generate()
        # would silently fall back and then report the primary provider's name
        # — a health check that lies about which model is serving traffic is
        # worse than no health check.
        test_messages = [{"role": "user", "content": "Say OK"}]
        last_error: Exception | None = None

        for provider in chain:
            model = self._resolve_model(provider)
            try:
                result = await provider.complete(
                    test_messages,
                    model=model,
                    temperature=0.0,
                    max_tokens=5,
                )
                return {
                    "status": "ok",
                    "provider": provider.name,
                    "model": model,
                    "is_primary": provider.name == self._primary_name,
                    "test_response": (result or "").strip()[:50],
                }
            except Exception as e:
                last_error = e
                logger.warning("Connectivity check failed for %s: %s", provider.name, e)

        return {
            "status": "error",
            "provider": self._primary_name,
            "model": self.active_model,
            "error": str(last_error),
        }


# ══════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ══════════════════════════════════════════════════════════════
# Import this anywhere with:
#   from app.services.llm_service import llm_service
#
# The service is ready to use immediately — providers are lazy-initialized
# on first call, so no heavy startup cost.
# ══════════════════════════════════════════════════════════════

llm_service = LLMService()
