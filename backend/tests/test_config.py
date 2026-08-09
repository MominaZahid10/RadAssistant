"""
Tests for application configuration parsing.

WHY THESE EXIST — a real bug this caught:
pydantic-settings v2 JSON-decodes any "complex" field type (list/dict/set)
inside its environment source, which runs BEFORE any field_validator. So
declaring `CORS_ORIGINS: list[str]` and then writing the natural thing in .env:

    CORS_ORIGINS=http://localhost:3000

crashes the entire app at startup with a JSONDecodeError, and no validator can
intercept it. The fix is to declare the field as `str` and expose a parsed
`list[str]` via a property.

That failure mode is invisible until someone edits .env — exactly the kind of
thing that breaks a deployment on demo day.
"""

import pytest

from app.config import Settings, _split_csv


# ══════════════════════════════════════════════════════════════
# THE CSV PARSER
# ══════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", []),
        ("   ", []),
        ("a", ["a"]),
        ("a,b,c", ["a", "b", "c"]),
        ("a, b ,  c ", ["a", "b", "c"]),          # tolerates padding
        ("a,,b", ["a", "b"]),                      # drops empties
        ('["a","b"]', ["a", "b"]),                 # JSON array form
        ('[ "a" , "b" ]', ["a", "b"]),
        ("[malformed", ["[malformed"]),            # falls back, doesn't crash
    ],
)
def test_split_csv(raw, expected):
    assert _split_csv(raw) == expected


# ══════════════════════════════════════════════════════════════
# CORS_ORIGINS
# ══════════════════════════════════════════════════════════════


def test_cors_accepts_plain_comma_separated(monkeypatch):
    """This is the exact input that used to crash the app at startup."""
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000,https://radassist.app")
    settings = Settings(_env_file=None)
    assert settings.CORS_ORIGINS == ["http://localhost:3000", "https://radassist.app"]


def test_cors_accepts_json_array(monkeypatch):
    """The old JSON syntax must keep working so nobody's config breaks."""
    monkeypatch.setenv("CORS_ORIGINS", '["http://a.com","http://b.com"]')
    settings = Settings(_env_file=None)
    assert settings.CORS_ORIGINS == ["http://a.com", "http://b.com"]


def test_cors_default_allows_local_frontend(monkeypatch):
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    settings = Settings(_env_file=None)
    assert "http://localhost:3000" in settings.CORS_ORIGINS


# ══════════════════════════════════════════════════════════════
# ALLOWED_EXTENSIONS
# ══════════════════════════════════════════════════════════════


def test_extensions_are_normalised(monkeypatch):
    """
    The upload endpoint compares against os.path.splitext() output, which is
    always lowercase-with-dot. Config must be normalised to match or valid
    uploads get rejected.
    """
    monkeypatch.setenv("ALLOWED_EXTENSIONS", "pdf,.DOCX, txt ,PNG")
    settings = Settings(_env_file=None)
    assert settings.ALLOWED_EXTENSIONS == [".pdf", ".docx", ".txt", ".png"]


def test_default_extensions_cover_all_supported_parsers(monkeypatch):
    """
    Every format the ingestion pipeline can parse must be uploadable, and
    nothing more — otherwise users hit "unsupported file type" from deep
    inside the pipeline instead of a clean 400 at the boundary.
    """
    monkeypatch.delenv("ALLOWED_EXTENSIONS", raising=False)
    settings = Settings(_env_file=None)

    parseable = {".pdf", ".docx", ".txt", ".md",
                 ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    assert set(settings.ALLOWED_EXTENSIONS) == parseable


# ══════════════════════════════════════════════════════════════
# CHUNKING CONFIG COHERENCE
# ══════════════════════════════════════════════════════════════


def test_default_chunk_overlap_is_smaller_than_chunk_size(monkeypatch):
    """If this ever inverts, every ingestion raises. Catch it here instead."""
    monkeypatch.delenv("CHUNK_SIZE", raising=False)
    monkeypatch.delenv("CHUNK_OVERLAP", raising=False)
    settings = Settings(_env_file=None)
    assert settings.CHUNK_OVERLAP < settings.CHUNK_SIZE


def test_embedding_dimension_matches_default_model(monkeypatch):
    """
    all-MiniLM-L6-v2 emits 384-dim vectors. If EMBEDDING_MODEL and
    EMBEDDING_DIMENSION drift apart, Qdrant rejects every upsert at runtime.
    """
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_DIMENSION", raising=False)
    settings = Settings(_env_file=None)
    if settings.EMBEDDING_MODEL == "all-MiniLM-L6-v2":
        assert settings.EMBEDDING_DIMENSION == 384
