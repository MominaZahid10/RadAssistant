"""
Tests for vision-model report reading (Phase 4.5).

THE CASE THIS EXISTS TO FIX, measured on a real upload:

    image:      424×471 photo of a chiropractic lumbar spine report
    Tesseract:  "The lumbar spine ts hypoiordotic."
                "\\wodge deformity of Ts wh 250% os of antic venibral body"
                "Tracompression tracture"
    downstream: the assistant reported DECREASED lumbar curvature — the
                opposite of the finding — and a fracture at the wrong level

Every one of those errors is recoverable from context, which is what a
vision-language model can do and a glyph classifier cannot.

⚠️  BUT THE VLM INTRODUCES THE OPPOSITE FAILURE MODE.
Tesseract fails visibly, by producing nonsense. A VLM fails invisibly, by
producing fluent text that was never on the page. Most of what follows tests
the defences against that: self-reported uncertainty, and cross-checking
against Tesseract.
"""

import json

import pytest

from app.services import vision_service as vision_module
from app.services.vision_service import (
    TRANSCRIPTION_PROMPT,
    VisionError,
    VisionResult,
    VisionService,
    agreement,
)


# ══════════════════════════════════════════════════════════════
# STUB CLIENT
# ══════════════════════════════════════════════════════════════


class _FakeResponse:
    def __init__(self, content: str):
        message = type("M", (), {"content": content})()
        choice = type("C", (), {"message": message})()
        self.choices = [choice]


class _FakeCompletions:
    def __init__(self, content: str | None = None, error: Exception | None = None):
        self._content = content
        self._error = error
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return _FakeResponse(self._content)


class _FakeClient:
    def __init__(self, content: str | None = None, error: Exception | None = None):
        self.completions = _FakeCompletions(content, error)
        self.chat = type("Chat", (), {"completions": self.completions})()


def _service(content=None, error=None, monkeypatch=None):
    svc = VisionService()
    client = _FakeClient(content, error)
    svc._get_client = lambda: client            # type: ignore[method-assign]
    svc.is_available = lambda: True             # type: ignore[method-assign]
    return svc, client


GOOD_REPLY = json.dumps({
    "transcription": (
        "The lumbar spine is hyperlordotic.\n"
        "Marked osteopenia is noted throughout the lumbar spine and pelvis.\n"
        "An anterior wedge deformity of T12 with a 50% loss of anterior "
        "vertebral body height is noted."
    ),
    "unclear": [
        {
            "reads_as": "T12",
            "possible": ["T12", "T2"],
            "matters_because": "level",
        }
    ],
})


# ══════════════════════════════════════════════════════════════
# TRANSCRIPTION
# ══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_transcribes_and_returns_text():
    svc, _ = _service(GOOD_REPLY)
    result = await svc.transcribe(b"\x89PNG fake", "image/png")

    assert "hyperlordotic" in result.text
    assert "50%" in result.text
    assert isinstance(result, VisionResult)


@pytest.mark.asyncio
async def test_sends_the_image_as_a_data_uri():
    svc, client = _service(GOOD_REPLY)
    await svc.transcribe(b"abc", "image/jpeg")

    content = client.completions.calls[0]["messages"][0]["content"]
    image_part = next(p for p in content if p["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.asyncio
async def test_transcription_is_deterministic():
    """
    There is exactly one right answer on the page. Sampling can only move us
    away from it.
    """
    svc, client = _service(GOOD_REPLY)
    await svc.transcribe(b"abc")
    assert client.completions.calls[0]["temperature"] == 0.0


@pytest.mark.asyncio
async def test_requests_json_mode():
    svc, client = _service(GOOD_REPLY)
    await svc.transcribe(b"abc")
    assert client.completions.calls[0]["response_format"] == {"type": "json_object"}


# ══════════════════════════════════════════════════════════════
# SELF-REPORTED UNCERTAINTY
# ══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_unclear_tokens_become_warnings():
    """
    A level the model could not read must reach the clinician, not be quietly
    resolved to whichever reading scanned best.
    """
    svc, _ = _service(GOOD_REPLY)
    result = await svc.transcribe(b"abc")

    assert result.unclear
    joined = " ".join(result.warnings)
    assert "T12" in joined
    assert "level" in joined


def test_prompt_forbids_guessing_the_dangerous_categories():
    p = TRANSCRIPTION_PROMPT
    assert "NEVER GUESS THESE" in p
    for axis in ("vertebral level", "direction", "laterality", "digit", "mechanism"):
        assert axis in p, f"{axis} must be named as un-guessable"
    assert "hyper- vs hypo-" in p


def test_prompt_permits_context_resolution():
    """The whole point of using a VLM over Tesseract."""
    assert "hypoiordotic" in TRANSCRIPTION_PROMPT
    assert "hyperlordotic" in TRANSCRIPTION_PROMPT


def test_prompt_forbids_inventing_findings():
    assert "Do not invent a finding" in TRANSCRIPTION_PROMPT
    assert "Do not summarise" in TRANSCRIPTION_PROMPT


# ══════════════════════════════════════════════════════════════
# CROSS-CHECK AGAINST TESSERACT
# ══════════════════════════════════════════════════════════════


def test_agreement_is_high_when_the_readers_concur():
    ocr = "The lumbar spine is hyperlordotic marked osteopenia"
    vlm = "The lumbar spine is hyperlordotic. Marked osteopenia is noted."
    assert agreement(vlm, ocr) == 1.0


def test_agreement_is_low_when_the_vlm_invents():
    """The failure this check exists to catch: fluent, plausible, not on the page."""
    ocr = "lumbar spine hyperlordotic osteopenia pelvis wedge deformity"
    vlm = "The chest radiograph demonstrates no acute cardiopulmonary process."
    score = agreement(vlm, ocr)
    assert score is not None and score < 0.2


def test_agreement_ignores_text_only_the_vlm_could_read():
    """
    Scored as recall of OCR tokens, not precision. Punishing the VLM for
    reading MORE than Tesseract would penalise exactly the improvement we
    switched to it for.
    """
    ocr = "lumbar spine"
    vlm = "lumbar spine hyperlordotic osteopenia pelvis wedge deformity T12 50%"
    assert agreement(vlm, ocr) == 1.0


def test_agreement_is_none_without_an_ocr_baseline():
    assert agreement("anything at all", "") is None


@pytest.mark.asyncio
async def test_low_agreement_produces_a_warning():
    svc, _ = _service(GOOD_REPLY)
    result = await svc.transcribe(
        b"abc", ocr_text="chest radiograph pneumothorax pleural effusion apex"
    )
    assert result.low_agreement
    assert any("disagree" in w for w in result.warnings)


@pytest.mark.asyncio
async def test_agreement_absent_when_ocr_did_not_run():
    svc, _ = _service(GOOD_REPLY)
    result = await svc.transcribe(b"abc")
    assert result.agreement is None
    assert result.low_agreement is False


# ══════════════════════════════════════════════════════════════
# PARSING — models do not always honour JSON mode
# ══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_handles_a_fenced_json_block():
    svc, _ = _service("```json\n" + GOOD_REPLY + "\n```")
    result = await svc.transcribe(b"abc")
    assert "hyperlordotic" in result.text


@pytest.mark.asyncio
async def test_handles_prose_wrapped_around_the_object():
    svc, _ = _service("Here is the transcription:\n" + GOOD_REPLY + "\nHope that helps!")
    result = await svc.transcribe(b"abc")
    assert "hyperlordotic" in result.text


@pytest.mark.asyncio
async def test_falls_back_to_raw_text_when_the_reply_is_not_json():
    """
    An unparseable reply that still contains the report beats an exception —
    losing a good transcription to a formatting slip would be absurd.
    """
    svc, _ = _service("The lumbar spine is hyperlordotic.")
    result = await svc.transcribe(b"abc")
    assert "hyperlordotic" in result.text


@pytest.mark.asyncio
async def test_malformed_unclear_field_is_survivable():
    svc, _ = _service(json.dumps({"transcription": "Findings.", "unclear": "oops"}))
    result = await svc.transcribe(b"abc")
    assert result.text == "Findings."
    assert result.unclear == []


# ══════════════════════════════════════════════════════════════
# FAILURE — must be loud, never a placeholder
# ══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_empty_transcription_raises():
    """
    ⚠️  THE PHASE 2 LESSON. An early OCR path returned "[OCR failed: ...]" as
    content; it cleared the length check, got embedded, and the document was
    marked completed. A failed read must fail.
    """
    svc, _ = _service(json.dumps({"transcription": "   ", "unclear": []}))
    with pytest.raises(VisionError):
        await svc.transcribe(b"abc")


@pytest.mark.asyncio
async def test_api_failure_raises_vision_error():
    svc, _ = _service(error=RuntimeError("429 rate limited"))
    with pytest.raises(VisionError, match="could not read"):
        await svc.transcribe(b"abc")


@pytest.mark.asyncio
async def test_oversized_image_is_rejected_before_the_api_call():
    svc, client = _service(GOOD_REPLY)
    with pytest.raises(VisionError, match="MB"):
        await svc.transcribe(b"x" * (19 * 1024 * 1024))
    assert client.completions.calls == [], "should not have hit the API"


@pytest.mark.asyncio
async def test_unavailable_service_raises_rather_than_returning_empty():
    svc = VisionService()
    svc.is_available = lambda: False            # type: ignore[method-assign]
    with pytest.raises(VisionError):
        await svc.transcribe(b"abc")


# ══════════════════════════════════════════════════════════════
# AVAILABILITY — degrade like the reranker, never fail the upload
# ══════════════════════════════════════════════════════════════


def test_unavailable_without_an_api_key(monkeypatch):
    monkeypatch.setattr(vision_module.settings, "GROQ_API_KEY", "")
    monkeypatch.setattr(vision_module.settings, "VISION_ENABLED", True)
    svc = VisionService()
    assert svc.is_available() is False
    assert "GROQ_API_KEY" in (svc.unavailable_reason() or "")


def test_unavailable_when_disabled(monkeypatch):
    monkeypatch.setattr(vision_module.settings, "GROQ_API_KEY", "sk-test")
    monkeypatch.setattr(vision_module.settings, "VISION_ENABLED", False)
    svc = VisionService()
    assert svc.is_available() is False
    assert "Tesseract" in (svc.unavailable_reason() or "")


def test_available_with_key_and_flag(monkeypatch):
    monkeypatch.setattr(vision_module.settings, "GROQ_API_KEY", "sk-test")
    monkeypatch.setattr(vision_module.settings, "VISION_ENABLED", True)
    svc = VisionService()
    assert svc.is_available() is True
    assert svc.unavailable_reason() is None


def test_default_model_is_one_groq_still_serves():
    """
    ⚠️  REGRESSION GUARD. Groq retired every other vision model it hosted:
    Llama 4 Scout (07/17/26), Llama 4 Maverick (03/09/26), the Llama 3.2
    vision previews (04/14/25) and LLaVA (10/28/24). This project has already
    been broken once by a Groq retirement.
    """
    from app.config import get_settings

    model = get_settings().VISION_MODEL
    retired = {
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "llama-3.2-90b-vision-preview",
        "llama-3.2-11b-vision-preview",
        "llava-v1.5-7b-4096-preview",
    }
    assert model not in retired, f"{model} has been shut down by Groq"
    assert model == "qwen/qwen3.6-27b"
