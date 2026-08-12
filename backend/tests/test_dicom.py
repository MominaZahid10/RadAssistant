"""
Tests for DICOM parsing and de-identification (Phase 4 Step 2).

THE TEST THAT MATTERS IS `test_no_known_phi_tag_survives`.

De-identification failing silently is the worst outcome available in this
phase: the image looks fine, the metadata looks fine, and a patient name sits
in the database. Nothing raises. Nobody notices until someone reads a JSONB
column.

The rest verify the two rendering details that also fail silently — VOI
windowing and MONOCHROME1 — which produce *an* image, just the wrong one.
"""

import pytest

from app.services import dicom_service
from app.services.dicom_service import (
    DICOM_TAG_ALLOWLIST,
    KNOWN_PHI_TAGS,
    DicomError,
    _extract_allowlisted,
    _handle_photometric,
    _safe_study_year,
)


class FakeDataset:
    """
    Stand-in for a pydicom Dataset — attribute access over a plain dict.

    Lets the de-identification logic be tested without pydicom installed,
    which matters because pydicom isn't in the image yet.
    """

    def __init__(self, **tags):
        self._tags = tags
        for k, v in tags.items():
            setattr(self, k, v)


# ══════════════════════════════════════════════════════════════
# PHI — THE CRITICAL BOUNDARY
# ══════════════════════════════════════════════════════════════


def test_no_known_phi_tag_survives():
    """
    A realistic dataset carrying every PHI tag we know of, plus legitimate
    clinical tags. Nothing identifying may appear in the output.
    """
    ds = FakeDataset(
        # ── PHI, all of which must be discarded ──
        PatientName="SMITH^JOHN^A",
        PatientID="MRN-8842119",
        PatientBirthDate="19571103",
        PatientAge="068Y",
        PatientSex="M",
        PatientAddress="14 Elm Street, Springfield",
        PatientTelephoneNumbers="+1-555-0142",
        InstitutionName="St Mary's General Hospital",
        InstitutionAddress="88 Hospital Road",
        ReferringPhysicianName="JONES^SARAH",
        PerformingPhysicianName="PATEL^R",
        OperatorsName="TECH12",
        AccessionNumber="ACC-99120",
        StudyID="ST-4471",
        StudyDescription="CHEST PA - SMITH, JOHN",   # names hide in free text
        SeriesDescription="AP ERECT - MRN 8842119",
        StudyInstanceUID="1.2.840.113619.2.55.3.1234",
        SeriesInstanceUID="1.2.840.113619.2.55.3.5678",
        SOPInstanceUID="1.2.840.113619.2.55.3.9012",
        StudyDate="20190314",
        StudyTime="143022",
        DeviceSerialNumber="SN-77120",
        StationName="CR-ROOM-4",
        # ── Legitimate clinical tags ──
        Modality="CR",
        BodyPartExamined="CHEST",
        ViewPosition="PA",
        Rows=2048,
        Columns=2500,
    )

    result = _extract_allowlisted(ds)
    blob = repr(result).lower()

    # No PHI tag name may appear as a key.
    for tag in KNOWN_PHI_TAGS:
        assert tag.lower() not in [k.lower() for k in result], f"PHI key survived: {tag}"

    # No PHI *value* may appear anywhere in the output, at any nesting depth.
    for leaked in (
        "smith", "john", "mrn-8842119", "19571103", "elm street", "555-0142",
        "st mary", "jones", "sarah", "patel", "tech12", "acc-99120",
        "1.2.840.113619", "sn-77120", "cr-room-4", "143022",
    ):
        assert leaked not in blob, f"PHI value leaked into metadata: {leaked!r}"


def test_clinical_tags_are_kept():
    """De-identification must not strip the data that makes the image useful."""
    ds = FakeDataset(Modality="CT", BodyPartExamined="ABDOMEN", ViewPosition="AP")
    result = _extract_allowlisted(ds)

    assert result["modality"] == "CT"
    assert result["body_part"] == "ABDOMEN"
    assert result["view_position"] == "AP"


def test_unknown_tags_are_discarded_by_default():
    """
    An allowlist means tags nobody anticipated — private vendor tags, future
    standard additions — are dropped without inspection. This is the whole
    reason for choosing an allowlist over a blocklist.
    """
    ds = FakeDataset(
        Modality="CR",
        SomeVendorPrivateTag="patient: john smith, dob 1957",
        UnknownFutureTag="anything at all",
    )
    result = _extract_allowlisted(ds)

    assert set(result) == {"modality"}
    assert "john smith" not in repr(result).lower()


def test_allowlist_contains_no_phi_tags():
    """Guard against someone adding an identifying tag to the allowlist."""
    overlap = set(DICOM_TAG_ALLOWLIST) & KNOWN_PHI_TAGS
    assert not overlap, f"PHI tags present in the allowlist: {overlap}"


def test_study_date_is_reduced_to_year():
    """
    Under HIPAA Safe Harbor, dates finer than a year are quasi-identifiers.
    The year alone is enough to say "a 2019 study" without that risk.
    """
    result = _safe_study_year(FakeDataset(StudyDate="20190314"))

    assert result.year == 2019
    assert (result.month, result.day) == (1, 1), "day/month must not survive"


def test_missing_or_malformed_study_date_returns_none():
    assert _safe_study_year(FakeDataset()) is None
    assert _safe_study_year(FakeDataset(StudyDate="")) is None
    assert _safe_study_year(FakeDataset(StudyDate="notadate")) is None
    assert _safe_study_year(FakeDataset(StudyDate="18000101")) is None   # implausible


def test_metadata_values_are_json_serialisable():
    """
    pydicom returns DSfloat, IS, PersonName and MultiValue types. Storing
    those in a JSONB column fails at commit time, long after the parse looked
    successful.
    """
    import json

    class MultiValue(list):
        pass

    ds = FakeDataset(
        Modality="CT",
        WindowCenter=MultiValue([40, 400]),
        WindowWidth=MultiValue([400, 1500]),
        Rows=512,
    )
    json.dumps(_extract_allowlisted(ds))   # must not raise


# ══════════════════════════════════════════════════════════════
# RENDERING — silent failures that still produce an image
# ══════════════════════════════════════════════════════════════


def test_monochrome1_is_inverted():
    """
    ⚠️  IGNORING PhotometricInterpretation RENDERS X-RAYS AS NEGATIVES —
    bones black, air white. The output still looks like a medical image, just
    a wrong one, so it survives visual review easily.
    """
    np = pytest.importorskip("numpy")

    arr = np.array([[0, 128, 255]], dtype=np.uint8)

    inverted = _handle_photometric(arr, FakeDataset(PhotometricInterpretation="MONOCHROME1"))
    assert list(inverted[0]) == [255, 127, 0]

    unchanged = _handle_photometric(arr, FakeDataset(PhotometricInterpretation="MONOCHROME2"))
    assert list(unchanged[0]) == [0, 128, 255]


def test_missing_photometric_defaults_to_monochrome2():
    """MONOCHROME2 is the common case; defaulting to inversion would be worse."""
    np = pytest.importorskip("numpy")
    arr = np.array([[0, 255]], dtype=np.uint8)
    assert list(_handle_photometric(arr, FakeDataset())[0]) == [0, 255]


def test_windowing_uses_stored_window_values():
    """
    ⚠️  WITHOUT WINDOWING THE IMAGE IS FLAT GREY.
    CT covers roughly -1000 to +3000 HU. Soft tissue occupies ~80 HU of that,
    so scaling the whole range into 8 bits collapses every soft-tissue finding
    into one or two indistinguishable grey levels.

    A lung window (centre -600, width 1500) must map air near black and soft
    tissue near white.
    """
    np = pytest.importorskip("numpy")
    from app.services.dicom_service import _apply_windowing

    pixels = np.array([[-1000, -600, 150]], dtype=np.int16)
    ds = FakeDataset(WindowCenter=-600, WindowWidth=1500, RescaleSlope=1, RescaleIntercept=0)

    out = _apply_windowing(pixels, ds)

    assert out[0][0] < 60, "air should be near black"
    assert out[0][2] > 200, "soft tissue should be near white"
    assert out.dtype == np.uint8


def test_windowing_falls_back_to_data_range():
    """No window in the file — use the actual min/max rather than failing."""
    np = pytest.importorskip("numpy")
    from app.services.dicom_service import _apply_windowing

    pixels = np.array([[100, 500, 900]], dtype=np.int16)
    out = _apply_windowing(pixels, FakeDataset())

    assert out[0][0] == 0
    assert out[0][2] == 255


def test_windowing_handles_multivalue_window():
    """Files often carry several presets; take the first rather than crashing."""
    np = pytest.importorskip("numpy")
    from app.services.dicom_service import _apply_windowing

    class MultiValue(list):
        pass

    pixels = np.array([[0, 40, 80]], dtype=np.int16)
    ds = FakeDataset(WindowCenter=MultiValue([40, 400]), WindowWidth=MultiValue([80, 1500]))

    out = _apply_windowing(pixels, ds)
    assert out.dtype == np.uint8


def test_windowing_survives_zero_width():
    """A zero or negative width would divide by zero."""
    np = pytest.importorskip("numpy")
    from app.services.dicom_service import _apply_windowing

    pixels = np.array([[10, 20]], dtype=np.int16)
    out = _apply_windowing(pixels, FakeDataset(WindowCenter=15, WindowWidth=0))
    assert out.dtype == np.uint8


# ══════════════════════════════════════════════════════════════
# GRACEFUL DEGRADATION
# ══════════════════════════════════════════════════════════════


def test_parse_raises_clearly_when_pydicom_missing(monkeypatch):
    """
    pydicom isn't in the image yet. The failure must name the cause and the
    fix, not surface an ImportError from deep in a call stack.
    """
    monkeypatch.setattr(dicom_service, "is_available", lambda: False)

    with pytest.raises(DicomError, match="pydicom"):
        dicom_service.parse_dicom(b"\x00" * 200)


def test_unavailable_reason_names_the_fix():
    reason = dicom_service.unavailable_reason()
    assert "requirements.txt" in reason
    assert "docker-compose build" in reason


# ══════════════════════════════════════════════════════════════
# FORMAT DETECTION
# ══════════════════════════════════════════════════════════════


def test_dicom_detected_by_magic_bytes():
    """
    PACS exports are frequently named `IM000001` with no extension, so
    sniffing the "DICM" marker at offset 128 matters more than the filename.
    """
    from app.services.image_processing import looks_like_dicom

    data = b"\x00" * 128 + b"DICM" + b"\x00" * 64
    assert looks_like_dicom(data, "IM000001") is True


def test_dicom_detected_by_extension_fallback():
    from app.services.image_processing import looks_like_dicom
    assert looks_like_dicom(b"short", "study.dcm") is True


def test_png_is_not_mistaken_for_dicom():
    from app.services.image_processing import looks_like_dicom
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200
    assert looks_like_dicom(png, "chest.png") is False


# ══════════════════════════════════════════════════════════════
# AVAILABILITY REPORTING
# ══════════════════════════════════════════════════════════════


def test_unavailable_reason_is_always_a_string():
    """
    ⚠️  THE TRAP THIS DOCUMENTS.
    vision_service.unavailable_reason() returns None when healthy;
    dicom_service.unavailable_reason() returns the same explanation
    unconditionally. Code that tests this one for None reported DICOM as
    broken on a container where pydicom was installed and working — which
    is exactly what /health did until this was caught.

    Callers must branch on is_available(), never on the reason being None.
    """
    from app.services import dicom_service

    reason = dicom_service.unavailable_reason()
    assert isinstance(reason, str) and reason
    assert reason is not None, "this function never signals health via None"


def test_health_endpoint_branches_on_is_available():
    """Regression guard on the /health wiring, not on dicom_service itself."""
    import inspect
    from app.api.v1.endpoints import health as health_module

    source = inspect.getsource(health_module)
    assert "dicom_service.is_available()" in source, (
        "/health must ask is_available(); unavailable_reason() is never None"
    )
