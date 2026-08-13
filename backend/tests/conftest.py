"""
Shared pytest configuration.

These tests are deliberately FAST and DEPENDENCY-FREE — no PostgreSQL, no
Qdrant, no embedding model download, no network. That means you can run them
on every save, which is the only way a test suite actually gets used.

To achieve that we stub the heavy third-party libraries before importing any
application code. We are testing OUR logic (chunking, config parsing, the
NCBI gate), not PyTorch or Qdrant.

Run:
    cd backend && pytest tests/ -v
"""

import sys
import types


def _install_stub(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    sys.modules.setdefault(name, module)
    return sys.modules[name]


class _AnyCallable:
    """Accepts any construction and returns a no-op for any attribute."""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _stub_heavy_dependencies() -> None:
    # ── Embeddings (would download ~80MB and load PyTorch) ──
    st = _install_stub("sentence_transformers")
    st.SentenceTransformer = _AnyCallable
    st.CrossEncoder = _AnyCallable          # Phase 3.5 reranker

    # ── Document parsing ────────────────────────────────────
    _install_stub("fitz")
    _install_stub("pytesseract")

    pil = _install_stub("PIL")
    pil.Image = types.SimpleNamespace(open=lambda *a, **k: None)
    _install_stub("PIL.Image")

    docx = _install_stub("docx")
    docx.Document = _AnyCallable

    # ── NCBI client ─────────────────────────────────────────
    _install_stub("Bio")

    # ── DICOM (Phase 4) ─────────────────────────────────────
    # Deliberately NOT stubbed. dicom_service imports it lazily and
    # degrades when absent, and the tests exercise that path directly
    # with a FakeDataset — which is closer to the real deployment,
    # where pydicom may genuinely not be installed.

    # ── Qdrant (would try to open a TCP connection at import) ──
    qc = _install_stub("qdrant_client")
    qc.QdrantClient = _AnyCallable

    http = _install_stub("qdrant_client.http")
    models = types.ModuleType("qdrant_client.http.models")
    for attr in (
        "VectorParams", "Filter", "FieldCondition",
        "MatchValue", "FilterSelector", "PointStruct",
    ):
        setattr(models, attr, _AnyCallable)
    models.Distance = types.SimpleNamespace(COSINE="Cosine")
    models.PayloadSchemaType = types.SimpleNamespace(KEYWORD="keyword")
    sys.modules["qdrant_client.http.models"] = models
    http.models = models

    exceptions = _install_stub("qdrant_client.http.exceptions")
    exceptions.UnexpectedResponse = type("UnexpectedResponse", (Exception,), {})


_stub_heavy_dependencies()


# ══════════════════════════════════════════════════════════════
# AUTHENTICATED BY DEFAULT (Phase 6, Step 2)
# ══════════════════════════════════════════════════════════════
#
# ⚠️  WHY THIS IS autouse, AND WHY THAT IS SAFE.
#
# Every clinical route now requires a signed-in user. Without this, 25 tests
# that have nothing to do with authentication would fail with 401 — and the
# obvious "fix" is to weaken the protection until the suite goes green, which
# is how a security control gets quietly reverted.
#
# So the dependency is OVERRIDDEN rather than the protection removed. Tests
# for chat, images and reports carry on testing chat, images and reports.
#
# This cannot hide a genuine authorisation hole, because the tests that check
# protection do not go through a client at all: tests/test_authz.py inspects
# the router's dependency graph directly, so an override at the app level is
# invisible to it. The one file that could be fooled is the one file that
# does not use this.

import uuid as _uuid

import pytest as _pytest


@_pytest.fixture(autouse=True)
def _authenticated_by_default():
    """Stand in for a signed-in user on every request the tests make."""
    # Imported lazily: app.main pulls in the whole application, and doing that
    # at conftest import time would run before the stubs above are installed.
    from app.core.deps import get_current_user
    from app.main import app
    from app.models.user import User

    stand_in = User(
        id=_uuid.uuid4(),
        email="test-suite@radassist.local",
        hashed_password="not-a-real-hash",
        full_name="Test Suite",
        is_active=True,
        is_admin=True,
    )

    app.dependency_overrides[get_current_user] = lambda: stand_in
    try:
        yield stand_in
    finally:
        # Removed rather than left in place, so a test that deliberately
        # exercises the unauthenticated path can clear it and get a real 401.
        app.dependency_overrides.pop(get_current_user, None)
