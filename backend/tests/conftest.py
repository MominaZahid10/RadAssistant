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
