"""
Tests for the image API (Phase 4 Step 4).

Focus on the contract and the security-relevant behaviour: what the API
exposes, what it refuses, and what it never leaks.
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.image import ImageResponse, ImageSourceType


client = TestClient(app)


# ══════════════════════════════════════════════════════════════
# WHAT THE API MUST NOT EXPOSE
# ══════════════════════════════════════════════════════════════


def test_storage_paths_are_not_in_the_response_schema():
    """
    ⚠️  storage_path and thumbnail_path are internal.

    Returning them would leak the server's filesystem layout and invite
    clients to construct their own file URLs — which is exactly how a
    path-traversal endpoint gets built by accident. Clients get an opaque
    /images/{id}/file URL and the server resolves it.
    """
    fields = set(ImageResponse.model_fields)

    assert "storage_path" not in fields
    assert "thumbnail_path" not in fields
    assert "file_url" in fields
    assert "thumbnail_url" in fields


def test_response_exposes_deidentification_status():
    """
    A consumer must be able to tell whether an image has been de-identified.
    Leaving it implicit invites the assumption that everything is safe.
    """
    assert "is_deidentified" in ImageResponse.model_fields


# ══════════════════════════════════════════════════════════════
# ROUTE ORDERING
# ══════════════════════════════════════════════════════════════


def test_stats_route_is_not_shadowed_by_the_id_route():
    """
    FastAPI matches routes in declaration order. If /{image_id} were declared
    first, /images/stats would be captured by it and fail UUID validation with
    a confusing 422 rather than returning statistics.
    """
    paths = [r.path for r in app.routes if hasattr(r, "path")]
    assert paths.index("/api/v1/images/stats") < paths.index("/api/v1/images/{image_id}")


# ══════════════════════════════════════════════════════════════
# UPLOAD VALIDATION
# ══════════════════════════════════════════════════════════════


def test_empty_file_is_rejected():
    r = client.post(
        "/api/v1/images/upload",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


def test_oversized_file_is_rejected():
    from app.config import get_settings

    limit = get_settings().MAX_IMAGE_SIZE_MB
    r = client.post(
        "/api/v1/images/upload",
        files={"file": ("huge.png", b"x" * ((limit + 1) * 1024 * 1024), "image/png")},
    )
    assert r.status_code == 413


def test_dicom_upload_is_refused_when_pydicom_is_missing(monkeypatch):
    """
    Refuse up front rather than accepting the file and marking it failed a
    moment later. Same reasoning as the embedding-model guard: a request that
    cannot possibly succeed should have no side effects.
    """
    from app.services import dicom_service

    monkeypatch.setattr(dicom_service, "is_available", lambda: False)

    dicom_bytes = b"\x00" * 128 + b"DICM" + b"\x00" * 100
    r = client.post(
        "/api/v1/images/upload",
        files={"file": ("IM000001", dicom_bytes, "application/octet-stream")},
    )

    assert r.status_code == 503
    assert "pydicom" in r.json()["detail"]


def test_dicom_is_detected_without_a_file_extension(monkeypatch):
    """
    PACS exports are routinely named `IM000001`. Detection must read the
    magic bytes at offset 128, not the filename.
    """
    from app.services import dicom_service

    seen = {}
    monkeypatch.setattr(dicom_service, "is_available", lambda: False)

    dicom_bytes = b"\x00" * 128 + b"DICM" + b"\x00" * 100
    r = client.post(
        "/api/v1/images/upload",
        files={"file": ("IM000001", dicom_bytes, "application/octet-stream")},
    )
    # A 503 (not a 201) proves it was routed as DICOM despite the filename.
    assert r.status_code == 503


# ══════════════════════════════════════════════════════════════
# NOT FOUND
# ══════════════════════════════════════════════════════════════


# NOTE: 404-on-unknown-id is deliberately NOT tested here.
# It requires a live database, and this suite is DB-free by design — that's
# what keeps it at ~15s and runnable on every save. The 404 path is covered
# by verify_phase4.sh against a running stack, where it's a more honest test
# anyway. Validation that happens BEFORE the database is reached (below) is
# fair game.


def test_malformed_uuid_returns_422():
    assert client.get("/api/v1/images/not-a-uuid").status_code == 422


# ══════════════════════════════════════════════════════════════
# SOURCE TYPES
# ══════════════════════════════════════════════════════════════


def test_source_type_enum_covers_every_ingestion_path():
    values = {e.value for e in ImageSourceType}
    assert values == {
        "dicom_upload",     # a study
        "report_upload",    # photographed paper report, OCR'd
        "pmc_figure",       # figure from an open-access article
        "image_upload",     # anything else
    }


def test_list_endpoint_accepts_the_documented_filters():
    """The gallery filters on these; a rename would break it silently."""
    import inspect
    from app.api.v1.endpoints.images import list_images

    params = set(inspect.signature(list_images).parameters)
    for expected in ("source_type", "modality", "body_part", "status_filter", "document_id"):
        assert expected in params
