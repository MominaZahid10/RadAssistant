"""
Tests for the text chunking logic (Phase 2, Stage 2).

WHY TEST THIS FIRST?
Chunking is the highest-leverage piece of a RAG pipeline. If chunks are
malformed, everything downstream degrades quietly: embeddings are computed
on garbage, retrieval returns irrelevant passages, and the LLM produces
confident nonsense grounded in nothing. Unlike a crash, you won't get a
stack trace — you'll just get bad answers.

These tests are also fast (no model loading, no network, no database), so
there's no excuse not to run them on every change.

Run with:
    pytest backend/tests/ -v
"""

import pytest

from app.services.ingestion import chunk_text


# ══════════════════════════════════════════════════════════════
# BASIC BEHAVIOUR
# ══════════════════════════════════════════════════════════════


def test_empty_string_returns_no_chunks():
    assert chunk_text("") == []


def test_whitespace_only_returns_no_chunks():
    assert chunk_text("   \n\n\t  ") == []


def test_short_text_stays_in_one_chunk():
    text = "Pneumothorax shows absent lung markings."
    assert chunk_text(text) == [text]


def test_text_exactly_at_chunk_size_is_not_split():
    text = "a" * 100
    assert chunk_text(text, chunk_size=100, chunk_overlap=10) == [text]


# ══════════════════════════════════════════════════════════════
# SIZE CONSTRAINTS
# ══════════════════════════════════════════════════════════════


def test_no_chunk_exceeds_chunk_size():
    """
    The embedding model truncates input beyond its token limit. A chunk that
    overflows chunk_size risks silently losing its tail during encoding —
    the text is stored in Qdrant but only part of it influenced the vector.
    """
    text = "Pneumonia inflames the air sacs in the lungs. " * 50
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=30)

    assert chunks, "expected at least one chunk"
    oversized = [c for c in chunks if len(c) > 200]
    assert not oversized, f"{len(oversized)} chunks exceeded chunk_size"


def test_long_text_produces_multiple_chunks():
    text = "The chest radiograph demonstrates consolidation. " * 50
    assert len(chunk_text(text, chunk_size=200, chunk_overlap=30)) > 1


def test_no_empty_or_whitespace_chunks():
    """Empty chunks would waste an embedding call and pollute search results."""
    text = "Section one.\n\n\n\nSection two.\n\n\n\n\n\nSection three.\n\n"
    chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)
    assert all(c.strip() for c in chunks)


# ══════════════════════════════════════════════════════════════
# CONTENT PRESERVATION
# ══════════════════════════════════════════════════════════════


def test_no_content_is_lost():
    """
    Every word in the source must appear somewhere in the output. Silent data
    loss here means a finding a radiologist relies on may simply not be in
    the knowledge base — with no error to indicate it.
    """
    text = "Pneumothorax pleural line absent markings tension mediastinal shift. " * 20
    chunks = chunk_text(text, chunk_size=150, chunk_overlap=25)

    source_words = set(text.split())
    chunked_words = set(" ".join(chunks).split())
    assert source_words <= chunked_words, f"lost: {source_words - chunked_words}"


def test_very_long_unbroken_token_is_split_by_characters():
    """
    Fallback path: text with no paragraph, line, sentence, or word boundaries
    (e.g. a base64 blob or a malformed OCR run) must still be chunked rather
    than returned oversized.
    """
    chunks = chunk_text("A" * 1000, chunk_size=100, chunk_overlap=20)
    assert all(len(c) <= 100 for c in chunks)
    assert len(chunks) > 1


# ══════════════════════════════════════════════════════════════
# CONFIGURATION GUARDS
# ══════════════════════════════════════════════════════════════
# The character-level fallback advances by (chunk_size - chunk_overlap).
# If overlap >= size that step is <= 0, so the splitter either yields nothing
# or loops forever. These must fail loudly at the boundary, not hang the
# worker thread in production.
# ══════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "chunk_size,chunk_overlap,reason",
    [
        (100, 100, "overlap equal to size cannot advance"),
        (100, 150, "overlap larger than size cannot advance"),
        (0, 0, "zero size is meaningless"),
        (-10, 5, "negative size is meaningless"),
        (100, -5, "negative overlap is meaningless"),
    ],
)
def test_invalid_config_raises_instead_of_hanging(chunk_size, chunk_overlap, reason):
    with pytest.raises(ValueError):
        chunk_text("x" * 500, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


# ══════════════════════════════════════════════════════════════
# SPLIT QUALITY
# ══════════════════════════════════════════════════════════════


def test_paragraphs_that_fit_are_not_fragmented():
    """
    Medical text carries meaning at the paragraph level — a paragraph about
    pneumothorax findings should survive intact rather than being cut in half,
    whenever it fits inside chunk_size.

    NOTE: chunks may still *begin* mid-sentence. That is intentional — the
    overlap window deliberately carries the tail of the previous chunk forward
    so a finding spanning a boundary appears complete in at least one chunk.
    """
    paragraphs = [f"Finding number {i}. It has clinical significance." for i in range(6)]
    text = "\n\n".join(paragraphs)

    chunks = chunk_text(text, chunk_size=120, chunk_overlap=10)

    assert all(len(c) <= 120 for c in chunks)
    # Each original paragraph fits within chunk_size, so each should appear
    # somewhere in full rather than being split across two chunks.
    for paragraph in paragraphs:
        assert any(paragraph in c for c in chunks), f"fragmented: {paragraph!r}"


def test_zero_overlap_is_respected_not_overridden():
    """
    Regression test: chunk_overlap=0 is a legitimate request for no overlap.
    An `or`-based default would silently replace it with the configured value.
    """
    text = "abcdefghij" * 30  # no separators at all -> character fallback
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=0)

    # With zero overlap the pieces should reconstruct the input exactly.
    assert "".join(chunks) == text
