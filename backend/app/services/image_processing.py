"""
RadAssist AI — Image Processing (Phase 4)

Format normalisation, thumbnails, and OCR for photographed reports.

This is the path that works TODAY — it needs only Pillow and Tesseract, both
already installed. DICOM (dicom_service.py) needs pydicom and activates once
the image is rebuilt.

WHAT A "REPORT PHOTO" IS:
A clinician photographs or scans a paper radiology report and uploads it. We
OCR it to text, then feed that text through the existing Phase 2 ingestion
pipeline — so the report becomes searchable alongside everything else, and the
image is kept as the visual source.

    photo → OCR → text → chunk → embed → Qdrant
      └─────────── kept as MedicalImage, linked to the document
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class ImageProcessingError(Exception):
    """An image could not be read or converted."""


# Formats a browser can display directly. Anything else is converted to PNG:
# TIFF is common for scanned medical documents and no browser renders it.
WEB_DISPLAYABLE = {"PNG", "JPEG", "WEBP", "GIF"}

# Images larger than this are downscaled before storage. A 40-megapixel phone
# photo of an A4 report carries no more readable text than a 4-megapixel one,
# but costs ten times the disk and decode time.
MAX_STORED_DIMENSION = 3000


def _open(file_bytes: bytes):
    from PIL import Image
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.load()          # force decode now, so truncation errors surface here
        return img
    except Exception as e:
        raise ImageProcessingError(f"Could not read this image: {e}") from e


def normalise(file_bytes: bytes) -> tuple[bytes, str, int, int]:
    """
    Convert an uploaded image to a web-displayable, sensibly-sized form.

    Returns (bytes, mime_type, width, height).
    """
    from PIL import Image

    img = _open(file_bytes)
    original_format = (img.format or "").upper()

    # Strip alpha and exotic modes — medical images are greyscale or RGB, and
    # a palette or CMYK image will not save as JPEG.
    if img.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        rgba = img.convert("RGBA")
        background.paste(rgba, mask=rgba.split()[-1])
        img = background
    elif img.mode not in ("L", "RGB"):
        img = img.convert("RGB")

    # Downscale oversized uploads.
    if max(img.size) > MAX_STORED_DIMENSION:
        img.thumbnail((MAX_STORED_DIMENSION, MAX_STORED_DIMENSION), Image.LANCZOS)

    buf = io.BytesIO()
    if original_format in WEB_DISPLAYABLE and original_format != "GIF":
        fmt = original_format
        mime = f"image/{fmt.lower()}"
        img.save(buf, format=fmt, quality=90 if fmt == "JPEG" else None, optimize=True)
    else:
        # TIFF, BMP, GIF and DICOM-derived output all become PNG — lossless,
        # universally displayable, and safe for greyscale medical content
        # where JPEG artefacts could be mistaken for findings.
        fmt, mime = "PNG", "image/png"
        img.save(buf, format="PNG", optimize=True)

    return buf.getvalue(), mime, img.width, img.height


def make_thumbnail(file_bytes: bytes, max_px: int | None = None) -> bytes:
    """
    Produce a small JPEG preview.

    JPEG rather than PNG: thumbnails are photographic-ish and viewed at a size
    where compression artefacts are invisible, so JPEG is several times
    smaller. The full-resolution image stays lossless — no one diagnoses from
    a 256px preview, but nobody should be shown a lossy one at full size
    either.
    """
    from PIL import Image

    max_px = max_px or settings.THUMBNAIL_MAX_PX
    img = _open(file_bytes)

    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")

    img.thumbnail((max_px, max_px), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82, optimize=True)
    return buf.getvalue()


# Tesseract needs roughly 300 DPI to resolve letterforms reliably. A page of
# A4 text at 300 DPI is ~2480px on the long edge; anything much below ~1600px
# starts producing character-level errors.
#
# ⚠️  WHY THIS THRESHOLD IS A SAFETY CONTROL, NOT A QUALITY TWEAK:
# On a 424×471 report, Tesseract read "hyperlordotic" as "hypoiordotic" —
# inverting the clinical finding — and "T12 with a 50%" as "Ts wh 250%".
# The model downstream repeated both faithfully. In a clinical tool a single
# character error can reverse a diagnosis, so low-resolution input is
# upscaled before OCR and flagged afterwards.
_MIN_OCR_LONG_EDGE = 1600

# Below this mean per-word confidence, the text is too unreliable to hand to
# a model that will state it as fact.
_MIN_OCR_CONFIDENCE = 60.0


@dataclass
class OcrResult:
    """Extracted text plus the signals needed to judge whether to trust it."""
    text: str
    confidence: float          # mean per-word confidence, 0-100
    upscaled: bool             # was the image enlarged before OCR?
    low_confidence: bool       # below the threshold — warn the user
    warnings: list[str] = field(default_factory=list)


def _prepare_for_ocr(img):
    """
    Improve a photograph's legibility before OCR.

    Each step targets a specific, observed failure:
      greyscale   — colour fringing from phone cameras corrupts edges
      upscale     — sub-300-DPI text is where character errors come from
      autocontrast— phone photos of paper are typically low-contrast
      sharpen     — recovers edges softened by the upscale
    """
    from PIL import Image, ImageFilter, ImageOps

    if img.mode != "L":
        img = img.convert("L")

    upscaled = False
    long_edge = max(img.size)
    if long_edge < _MIN_OCR_LONG_EDGE:
        scale = min(_MIN_OCR_LONG_EDGE / long_edge, 4.0)   # cap: 4x is already interpolation
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.LANCZOS,
        )
        upscaled = True

    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.filter(ImageFilter.SHARPEN)

    return img, upscaled


def ocr_report_image(file_bytes: bytes) -> OcrResult:
    """
    Extract text from a photographed or scanned report, with a confidence score.

    Raises ImageProcessingError rather than returning an error string.

    ⚠️  THE PHASE 2 LESSON, RESTATED.
    An earlier version returned "[OCR failed: ...]" as if it were content —
    it cleared the length check, got embedded, and the document was marked
    *completed*. A failed parse must fail.

    ⚠️  AND THE PHASE 4 LESSON.
    A parse that half-worked is more dangerous than one that failed, because
    the output looks like a report. Confidence is measured and returned so the
    caller can warn rather than silently present misread text as clinical fact.
    """
    try:
        import pytesseract
    except ImportError as e:
        raise ImageProcessingError(
            "pytesseract is not installed — cannot read text from images."
        ) from e

    img = _open(file_bytes)
    original_long_edge = max(img.size)
    img, upscaled = _prepare_for_ocr(img)

    try:
        # PSM 6 — "a single uniform block of text". Radiology reports are
        # single-column documents; the default (PSM 3, full auto page
        # segmentation) tries to detect columns and misreads headed forms.
        config = "--psm 6"
        data = pytesseract.image_to_data(
            img, config=config, output_type=pytesseract.Output.DICT
        )
        text = pytesseract.image_to_string(img, config=config)
    except pytesseract.TesseractNotFoundError as e:
        raise ImageProcessingError(
            "The Tesseract OCR engine is not installed in this container. "
            "Install it with: apt-get install -y tesseract-ocr tesseract-ocr-eng"
        ) from e
    except Exception as e:
        raise ImageProcessingError(f"OCR failed on this image: {e}") from e

    text = text.strip()
    if not text:
        raise ImageProcessingError(
            "OCR found no readable text. The image may be blurred, poorly lit, "
            "or at too steep an angle. A flat, well-lit, straight-on scan works best."
        )

    # Mean confidence over words Tesseract actually scored (-1 = no score).
    scores = [
        float(c) for c in data.get("conf", [])
        if str(c).lstrip("-").replace(".", "").isdigit() and float(c) >= 0
    ]
    confidence = round(sum(scores) / len(scores), 1) if scores else 0.0

    warnings: list[str] = []
    if original_long_edge < _MIN_OCR_LONG_EDGE:
        warnings.append(
            f"Low resolution ({original_long_edge}px). Text was upscaled before "
            f"reading, but character errors are likely — a single wrong letter "
            f"can reverse a finding (e.g. hyper- vs hypo-). "
            f"A scan of at least {_MIN_OCR_LONG_EDGE}px is strongly preferred."
        )
    if confidence and confidence < _MIN_OCR_CONFIDENCE:
        warnings.append(
            f"Low OCR confidence ({confidence:.0f}%). Verify the extracted text "
            f"against the original before relying on it."
        )

    return OcrResult(
        text=text,
        confidence=confidence,
        upscaled=upscaled,
        low_confidence=bool(confidence and confidence < _MIN_OCR_CONFIDENCE),
        warnings=warnings,
    )


def looks_like_dicom(file_bytes: bytes, filename: str = "") -> bool:
    """
    Detect DICOM without needing pydicom.

    Every conformant file has the ASCII magic "DICM" at byte offset 128, after
    a 128-byte preamble. Checking the bytes rather than the extension matters
    because DICOM files frequently have no extension at all — PACS exports are
    often named things like `IM000001`.
    """
    if len(file_bytes) > 132 and file_bytes[128:132] == b"DICM":
        return True
    return filename.lower().endswith((".dcm", ".dicom"))
