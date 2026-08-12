"""
RadAssist AI — DICOM Parsing & De-identification (Phase 4)

Turns a DICOM file into (a) allowlisted, PHI-free metadata and (b) a
correctly-windowed PNG a browser can display.

⚠️  THREE THINGS THAT FAIL SILENTLY IF YOU GET THEM WRONG.
Each produces *an image* and *some metadata* — just the wrong ones. None of
them raise. That's why each has an explicit section below and a test.

    1. PHI leakage        — an allowlist, never a blocklist
    2. VOI windowing      — raw values render as flat grey mush
    3. MONOCHROME1        — X-rays render as photographic negatives

DEPENDENCY NOTE:
pydicom is imported lazily. If it isn't installed, `is_available()` returns
False and DICOM uploads are rejected with a clear message — the rest of the
image pipeline (report photos, PMC figures) keeps working. Given how much
time this project has lost to package installs failing on an unstable
connection, an optional dependency that degrades cleanly beats a hard import
that takes the service down.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from datetime import date

logger = logging.getLogger(__name__)


class DicomError(Exception):
    """A DICOM file could not be read, or could not be safely de-identified."""


# ══════════════════════════════════════════════════════════════
# DE-IDENTIFICATION — ALLOWLIST
# ══════════════════════════════════════════════════════════════
# ⚠️  ALLOWLIST, NOT BLOCKLIST. THIS IS THE MOST IMPORTANT DECISION HERE.
#
# A blocklist means "remove PatientName, PatientID, PatientBirthDate..." and
# anything you didn't think of survives. DICOM has ~4,000 standard tags plus
# private vendor tags that vary by scanner manufacturer, and free-text fields
# routinely contain names ("CHEST PA - SMITH, JOHN" in StudyDescription).
#
# "We forgot a tag" is a data-protection incident, not a bug. So: only tags
# named here survive. Everything else — standard, private, or unknown — is
# discarded without inspection.
#
# Nothing in this list can identify a person:
#   modality/body part/view — clinical, describes the image not the patient
#   study YEAR only         — full dates are quasi-identifiers under HIPAA
#                             Safe Harbor; year alone is not
#   image geometry          — pixel dimensions and windowing
DICOM_TAG_ALLOWLIST: dict[str, str] = {
    "Modality": "modality",
    "BodyPartExamined": "body_part",
    "ViewPosition": "view_position",
    "PatientPosition": "patient_position",       # HFS/FFS — orientation, not identity
    "Rows": "rows",
    "Columns": "columns",
    "PhotometricInterpretation": "photometric_interpretation",
    "BitsStored": "bits_stored",
    "WindowCenter": "window_center",
    "WindowWidth": "window_width",
    "RescaleIntercept": "rescale_intercept",
    "RescaleSlope": "rescale_slope",
    "SliceThickness": "slice_thickness",
    "KVP": "kvp",                                 # acquisition parameter
    "Manufacturer": "manufacturer",               # scanner make, not a person
}

# Explicitly enumerated so the test can assert none of them survive. This is
# documentation and a regression guard — the allowlist above is what actually
# does the work.
KNOWN_PHI_TAGS = frozenset({
    "PatientName", "PatientID", "PatientBirthDate", "PatientAge", "PatientSex",
    "PatientAddress", "PatientTelephoneNumbers", "OtherPatientIDs",
    "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName",
    "ReferringPhysicianName", "PerformingPhysicianName", "OperatorsName",
    "PhysiciansOfRecord", "RequestingPhysician",
    "AccessionNumber", "StudyID", "StudyDescription", "SeriesDescription",
    "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID",
    "StudyDate", "StudyTime", "SeriesDate", "AcquisitionDate", "ContentDate",
    "DeviceSerialNumber", "StationName",
})


@dataclass
class DicomResult:
    """Output of a successful parse."""
    png_bytes: bytes
    metadata: dict = field(default_factory=dict)
    modality: str | None = None
    body_part: str | None = None
    view_position: str | None = None
    study_date: date | None = None          # year only — see _safe_study_year
    width: int = 0
    height: int = 0


def is_available() -> bool:
    """True if pydicom is installed and DICOM parsing can run."""
    try:
        import pydicom  # noqa: F401
        return True
    except ImportError:
        return False


def unavailable_reason() -> str:
    return (
        "DICOM support requires pydicom, which is not installed in this image. "
        "Add `pydicom` to requirements.txt and rebuild:\n"
        "    docker-compose build backend && docker-compose up -d backend\n"
        "Report photos and standard images work without it."
    )


# ══════════════════════════════════════════════════════════════
# PIXEL RENDERING
# ══════════════════════════════════════════════════════════════


def _apply_windowing(pixels, ds):
    """
    Map raw stored values to 0-255 for display.

    ⚠️  WITHOUT THIS THE IMAGE IS FLAT GREY MUSH.
    DICOM stores 12-16 bit values — CT in Hounsfield units, roughly -1000
    (air) to +3000 (bone). Naively scaling that whole range into 8 bits
    compresses all soft tissue, which occupies maybe 80 HU, into a couple of
    indistinguishable grey levels. Every finding disappears.

    WindowCenter/WindowWidth define which slice of that range to show — this
    is exactly what the brightness/contrast presets in a DICOM viewer set
    ("lung window", "bone window"). We use the values stored in the file, and
    fall back to the actual min/max when absent.
    """
    import numpy as np

    arr = pixels.astype(np.float64)

    # Modality LUT: stored value → real-world units (e.g. Hounsfield).
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    arr = arr * slope + intercept

    center = getattr(ds, "WindowCenter", None)
    width = getattr(ds, "WindowWidth", None)

    # These can be multi-valued (one window per preset) — take the first.
    if isinstance(center, (list, tuple)) or type(center).__name__ == "MultiValue":
        center = center[0] if len(center) else None
    if isinstance(width, (list, tuple)) or type(width).__name__ == "MultiValue":
        width = width[0] if len(width) else None

    try:
        center = float(center) if center is not None else None
        width = float(width) if width is not None else None
    except (TypeError, ValueError):
        center = width = None

    if center is None or width is None or width <= 0:
        # No window in the file — use the data's own range.
        lo, hi = float(arr.min()), float(arr.max())
    else:
        lo, hi = center - width / 2.0, center + width / 2.0

    if hi <= lo:
        hi = lo + 1.0

    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def _handle_photometric(arr, ds):
    """
    Invert MONOCHROME1.

    ⚠️  IGNORING THIS RENDERS X-RAYS AS PHOTOGRAPHIC NEGATIVES —
    bones black, air white. Easy to miss in review because the output still
    *looks* like a medical image, just an odd one.

    MONOCHROME1: minimum stored value displays as WHITE
    MONOCHROME2: minimum stored value displays as BLACK  (the usual case)
    """
    interpretation = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
    if interpretation == "MONOCHROME1":
        return 255 - arr
    return arr


# ══════════════════════════════════════════════════════════════
# METADATA
# ══════════════════════════════════════════════════════════════


def _safe_study_year(ds) -> date | None:
    """
    Study date reduced to 1 January of its year.

    WHY NOT THE FULL DATE:
    Under HIPAA Safe Harbor, dates more precise than a year are quasi-
    identifiers — combined with a couple of other attributes they can
    re-identify an individual. The year is enough to say "this is a 2019
    study" without that risk.
    """
    raw = getattr(ds, "StudyDate", None)
    if not raw:
        return None
    try:
        year = int(str(raw)[:4])
        if 1900 <= year <= 2100:
            return date(year, 1, 1)
    except (ValueError, TypeError):
        pass
    return None


def _extract_allowlisted(ds) -> dict:
    """Copy across only the tags named in DICOM_TAG_ALLOWLIST."""
    out: dict = {}
    for tag, key in DICOM_TAG_ALLOWLIST.items():
        value = getattr(ds, tag, None)
        if value is None:
            continue
        # DICOM value types (DSfloat, IS, PersonName, MultiValue) aren't
        # JSON-serialisable — coerce to primitives.
        if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
            value = [str(v) for v in value]
        elif isinstance(value, (int, float, str)):
            pass
        else:
            value = str(value)
        out[key] = value
    return out


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════


def parse_dicom(file_bytes: bytes) -> DicomResult:
    """
    Parse a DICOM file into de-identified metadata plus a viewable PNG.

    Raises DicomError on anything unreadable — never returns a partial or
    placeholder result. A parse that half-worked must fail, because the
    alternative is storing an image whose PHI status is unknown.
    """
    if not is_available():
        raise DicomError(unavailable_reason())

    import numpy as np
    import pydicom
    from PIL import Image

    # ── Read ──
    try:
        ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
    except Exception as e:
        raise DicomError(f"Not a readable DICOM file: {e}") from e

    if not hasattr(ds, "PixelData"):
        raise DicomError(
            "This DICOM file contains no pixel data — it may be a structured "
            "report, a presentation state, or a metadata-only object."
        )

    # ── Pixels ──
    try:
        pixels = ds.pixel_array
    except Exception as e:
        raise DicomError(
            f"Could not decode pixel data: {e}. The file may use a compressed "
            f"transfer syntax needing an extra handler (pylibjpeg, gdcm)."
        ) from e

    # Multi-frame (a CT series in one file): take the middle slice, which is
    # far more likely to contain anatomy than the first.
    if pixels.ndim == 3:
        pixels = pixels[len(pixels) // 2]
    if pixels.ndim != 2:
        raise DicomError(f"Unsupported pixel array shape: {pixels.shape}")

    arr = _apply_windowing(pixels, ds)
    arr = _handle_photometric(arr, ds)

    height, width = arr.shape

    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG", optimize=True)

    # ── Metadata (allowlisted only) ──
    metadata = _extract_allowlisted(ds)

    return DicomResult(
        png_bytes=buf.getvalue(),
        metadata=metadata,
        modality=metadata.get("modality"),
        body_part=metadata.get("body_part"),
        view_position=metadata.get("view_position"),
        study_date=_safe_study_year(ds),
        width=int(width),
        height=int(height),
    )
