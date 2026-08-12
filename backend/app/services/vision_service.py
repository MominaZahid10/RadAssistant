"""
RadAssist AI — Vision Service (Phase 4.5)

Reads a photographed radiology report with a vision-language model instead of
a character-level OCR engine.

════════════════════════════════════════════════════════════════════
WHY THIS REPLACES TESSERACT AS THE PRIMARY READER
════════════════════════════════════════════════════════════════════
Tesseract classifies glyphs. It has no idea what a radiology report is, so on
a 424×471 photo it produced:

    "The lumbar spine ts hypoiordotic."          → hyperlordotic
    "\\wodge deformity of Ts wh 250% os of..."    → 50%
    "Tracompression tracture"                    → T12 compression fracture
    "Marke osteopenas"                           → Marked osteopenia

Every one of those is recoverable from context. "hypoiordotic" is not a word;
"hyperlordotic" is, it fits a lumbar spine report, and it is consistent with
the impression line further down the same page. A VLM reads pixels and
language jointly, so it resolves what a glyph classifier cannot.

Two rounds of prompt engineering went into teaching the answering model to
cope with garbled input. That was treating the symptom. This treats the cause.

════════════════════════════════════════════════════════════════════
WHY qwen/qwen3.6-27b, AND NOT SOMETHING ELSE
════════════════════════════════════════════════════════════════════
It is the only vision model Groq still serves. Checked against
https://console.groq.com/docs/deprecations — the alternatives are gone:

    meta-llama/llama-4-scout-17b-16e-instruct     shut down 2026-07-17
    meta-llama/llama-4-maverick-17b-128e-instruct shut down 2026-03-09
    llama-3.2-90b-vision-preview                  shut down 2025-04-14
    llama-3.2-11b-vision-preview                  shut down 2025-04-14
    llava-v1.5-7b-4096-preview                    shut down 2024-10-28

That is not a reason to distrust it — Groq names qwen3.6-27b as the
recommended migration target for llama-3.3-70b-versatile, alongside the
gpt-oss-120b this project already answers with. It is on the supported path.

Practical reasons it is also the right choice here:
  - same provider, same GROQ_API_KEY, same OpenAI-compatible surface — no new
    account, no new secret, no new outage to reason about
  - JSON mode, so the model returns a parseable object rather than prose we
    would have to regex
  - 131K context; 20MB image limit, far above a phone photo of a report

⚠️  THE NEW FAILURE MODE, STATED PLAINLY.
Tesseract fails by producing nonsense — obvious, and safe because it is
obvious. A VLM fails by producing *fluent, plausible, wrong* text, which is
far more dangerous in a clinical tool. It could read "hypolordotic" and be
confidently wrong where Tesseract was visibly wrong.

Two defences, neither optional:
  1. The model must report its own uncertain tokens (`unclear` below), and is
     told explicitly that guessing a level or a direction is worse than
     admitting it cannot read one.
  2. Tesseract still runs, and `agreement()` scores the two transcriptions
     against each other. Independent extractors with different failure modes
     — the same argument that motivated hybrid BM25 + vector retrieval. When
     they disagree, the clinician is told.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class VisionError(Exception):
    """The image could not be read by the vision model."""


# Below this token-level agreement with Tesseract, the two readers disagree
# enough that one of them is likely fabricating. Not a hard failure — a photo
# Tesseract could barely read will legitimately score low — but it is
# surfaced to the clinician.
_MIN_AGREEMENT = 0.55

# Groq rejects requests over 20MB. Report photos are far smaller, but a
# 200MB scan would otherwise fail with an opaque 400.
_MAX_IMAGE_BYTES = 18 * 1024 * 1024


TRANSCRIPTION_PROMPT = """
You are transcribing a photographed or scanned radiology report for a
clinical system. Accuracy matters more than completeness.

Return ONLY a JSON object:

{
  "transcription": "the full text of the report, preserving line structure",
  "unclear": [
    {
      "reads_as": "what you see on the page",
      "possible": ["candidate 1", "candidate 2"],
      "matters_because": "level | direction | laterality | number | mechanism"
    }
  ]
}

RULES:

1. Transcribe what is on the page. Do not summarise, reorder, interpret, or
   add clinical commentary. Keep headings, section labels and line breaks.

2. Where the image is degraded, use the surrounding language to resolve the
   text — that is why you are being used instead of a character-level OCR
   engine. "hypoiordotic" in a lumbar spine report is "hyperlordotic".

3. BUT NEVER GUESS THESE. If any of the following is not clearly legible, put
   it in "unclear" with every reading you consider possible, and use your best
   reading in the transcription only if one is clearly better:
       - a vertebral level or other anatomical numbering (T12 vs T2 vs L2)
       - a direction (hyper- vs hypo-, increased vs decreased)
       - a laterality (left vs right)
       - any digit or percentage
       - a mechanism (traumatic vs pathological vs osteoporotic)
   Being wrong about one of these changes what happens to the patient.
   Reporting that you cannot read it costs nothing.

4. Do not invent a finding that is not on the page. If a section is cut off or
   obscured, transcribe what is visible and stop.

5. If the image is not a medical report at all, return an empty transcription
   and say so in "unclear".
""".strip()


@dataclass
class VisionResult:
    """A transcription plus everything needed to judge whether to trust it."""

    text: str
    model: str
    unclear: list[dict] = field(default_factory=list)
    # Token overlap with the independent Tesseract read, when available.
    # None means OCR did not run or produced nothing to compare against.
    agreement: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def low_agreement(self) -> bool:
        return self.agreement is not None and self.agreement < _MIN_AGREEMENT


# ══════════════════════════════════════════════════════════════
# CROSS-CHECK
# ══════════════════════════════════════════════════════════════


_WORD = re.compile(r"[a-z0-9%]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def agreement(vlm_text: str, ocr_text: str) -> float | None:
    """
    Token-level overlap between the two independent readers, 0.0-1.0.

    ⚠️  WHY BOTH READERS STAY IN THE PIPELINE.
    A VLM's failure mode is fluent invention: it can return a well-formed
    report that is not the one in the image, and nothing about the output
    looks wrong. Tesseract cannot invent — it can only mangle what is there.
    So a low overlap means one of two things, and both are worth a warning:
    the photo is bad enough that Tesseract produced noise, or the VLM wrote
    something that is not on the page.

    This is the same reasoning that put BM25 next to vector search in Phase
    3.6: two extractors whose mistakes are uncorrelated catch each other's.

    Scored asymmetrically, as a recall of OCR tokens found in the VLM output.
    Precision would punish the VLM for reading text Tesseract missed entirely,
    which is exactly the improvement we are hoping for.
    """
    ocr_tokens = set(_tokens(ocr_text))
    if not ocr_tokens:
        return None

    vlm_tokens = set(_tokens(vlm_text))
    if not vlm_tokens:
        return 0.0

    return round(len(ocr_tokens & vlm_tokens) / len(ocr_tokens), 3)


# ══════════════════════════════════════════════════════════════
# SERVICE
# ══════════════════════════════════════════════════════════════


class VisionService:
    """
    Transcribes report images via Groq's vision model.

    Degrades exactly like reranker_service and dicom_service: `is_available()`
    is checked first, failures raise VisionError, and the caller falls back to
    Tesseract. A missing vision model must never fail an upload.
    """

    def __init__(self) -> None:
        self._client = None

    # ── Availability ─────────────────────────────────────────

    def is_available(self) -> bool:
        return bool(settings.VISION_ENABLED and settings.GROQ_API_KEY.strip())

    def unavailable_reason(self) -> str | None:
        """Human-readable explanation for /health, or None when available."""
        if not settings.VISION_ENABLED:
            return "VISION_ENABLED is false — reports are read with Tesseract OCR."
        if not settings.GROQ_API_KEY.strip():
            return (
                "GROQ_API_KEY is not set. The vision model runs on Groq; "
                "without a key, reports fall back to Tesseract OCR."
            )
        return None

    @property
    def model(self) -> str:
        return settings.VISION_MODEL

    def _get_client(self):
        if self._client is None:
            from groq import AsyncGroq
            self._client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        return self._client

    # ── Transcription ────────────────────────────────────────

    async def transcribe(
        self,
        image_bytes: bytes,
        mime_type: str = "image/png",
        *,
        ocr_text: str | None = None,
    ) -> VisionResult:
        """
        Read a report image. Raises VisionError; never returns a placeholder.

        ⚠️  THE PHASE 2 LESSON, AGAIN.
        An early version of the OCR path returned "[OCR failed: ...]" as if it
        were content. It cleared the length check, got embedded, and the
        document was marked *completed*. A failed read must fail loudly so the
        caller can fall back — never quietly produce text-shaped garbage.
        """
        if not self.is_available():
            raise VisionError(self.unavailable_reason() or "Vision model unavailable.")

        if len(image_bytes) > _MAX_IMAGE_BYTES:
            raise VisionError(
                f"Image is {len(image_bytes) / 1e6:.1f}MB; the vision model "
                f"accepts up to {_MAX_IMAGE_BYTES / 1e6:.0f}MB."
            )

        data_uri = (
            f"data:{mime_type};base64,"
            + base64.b64encode(image_bytes).decode("ascii")
        )

        try:
            response = await self._get_client().chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": TRANSCRIPTION_PROMPT},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    }
                ],
                # Deterministic. This is transcription, not composition — there
                # is exactly one right answer on the page, and sampling can
                # only move us away from it.
                temperature=0.0,
                max_tokens=4096,
                response_format={"type": "json_object"},
                stream=False,
            )
            raw = response.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            raise VisionError(f"Vision model could not read this image: {e}") from e

        text, unclear = self._parse(raw)

        if not text.strip():
            raise VisionError(
                "The vision model found no readable report text in this image."
            )

        warnings: list[str] = []
        score = agreement(text, ocr_text) if ocr_text else None

        if score is not None and score < _MIN_AGREEMENT:
            warnings.append(
                f"The vision model and OCR disagree on this image "
                f"(agreement {score:.0%}). Verify the text against the "
                f"original before relying on it."
            )

        for item in unclear:
            reads_as = str(item.get("reads_as", "")).strip()
            possible = item.get("possible") or []
            if not reads_as:
                continue
            options = ", ".join(str(p) for p in possible if p)
            matters = str(item.get("matters_because", "")).strip()
            warnings.append(
                f"Could not read {reads_as!r} with confidence"
                + (f" — possibly {options}" if options else "")
                + (f" ({matters})" if matters else "")
                + "."
            )

        return VisionResult(
            text=text.strip(),
            model=self.model,
            unclear=unclear,
            agreement=score,
            warnings=warnings,
        )

    # ── Parsing ──────────────────────────────────────────────

    @staticmethod
    def _parse(raw: str) -> tuple[str, list[dict]]:
        """
        Pull the transcription out of the model's reply.

        JSON mode is requested, but a model that emits a fenced block or a
        stray sentence before the object must not cost us the transcription.
        Falling back to the raw text is correct: an unparseable reply that
        still contains the report is more useful than an exception.
        """
        candidate = raw.strip()

        # Strip a ```json fence if one slipped through.
        if candidate.startswith("```"):
            candidate = re.sub(r"^```[a-z]*\s*", "", candidate)
            candidate = re.sub(r"\s*```$", "", candidate)

        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            # Last resort: find the outermost object in the reply.
            match = re.search(r"\{.*\}", candidate, re.DOTALL)
            if not match:
                logger.warning("Vision reply was not JSON; using it verbatim.")
                return raw, []
            try:
                payload = json.loads(match.group(0))
            except (json.JSONDecodeError, ValueError):
                logger.warning("Vision reply was not JSON; using it verbatim.")
                return raw, []

        if not isinstance(payload, dict):
            return raw, []

        text = payload.get("transcription") or ""
        if not isinstance(text, str):
            text = str(text)

        unclear = payload.get("unclear") or []
        if not isinstance(unclear, list):
            unclear = []
        unclear = [u for u in unclear if isinstance(u, dict)]

        return text, unclear


# ══════════════════════════════════════════════════════════════
# SINGLETON
# ══════════════════════════════════════════════════════════════

vision_service = VisionService()
