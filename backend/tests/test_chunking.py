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

import re

import pytest

from app.services.ingestion import chunk_text, _tail_overlap


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


# ══════════════════════════════════════════════════════════════
# OVERLAP WORD BOUNDARIES
# ══════════════════════════════════════════════════════════════
# Regression tests for a defect found in live search output: the overlap
# window sliced by raw character count, so chunks could begin mid-word.
# ══════════════════════════════════════════════════════════════


@pytest.mark.parametrize("width", [12, 20, 25, 30, 40, 50])
def test_tail_overlap_never_starts_mid_word(width):
    """
    Whatever the overlap width, the returned text must begin at a real word —
    never at a fragment left behind by slicing on character count.
    """
    text = "It is a critical diagnosis that must not be missed on chest imaging."
    words = set(text.replace(".", "").split())

    overlap = _tail_overlap(text, width)
    first_word = overlap.split()[0].strip(".,;:")

    assert first_word in words, (
        f"width={width} produced fragment {first_word!r} from raw slice "
        f"{text[-width:]!r}"
    )


def test_tail_overlap_fixes_the_observed_production_case():
    """
    The exact string from the seeded Pneumothorax article whose chunk 1 began
    with 'e missed on chest imaging' in live search results.
    """
    text = "It is a critical diagnosis that must not be missed on chest imaging."

    # Reproduce the defect: a raw character slice starts mid-word.
    raw = text[-26:]
    assert raw.split()[0] not in text.split(), f"expected a fragment, got {raw!r}"

    assert _tail_overlap(text, 26).startswith("missed on chest imaging")


def test_tail_overlap_keeps_raw_window_when_no_whitespace():
    """
    A single very long token (chemical name, OCR artefact) has no boundary to
    snap to. Returning empty would silently drop the overlap entirely.
    """
    assert _tail_overlap("A" * 200, 40) == "A" * 40


def test_no_chunk_begins_mid_word():
    """
    The exact failure seen in production search results. Every chunk after the
    first begins with overlap text, and that text must start at a word.
    """
    text = (
        "Pneumothorax is the presence of air in the pleural space, causing "
        "partial or complete lung collapse. It is a critical diagnosis that "
        "must not be missed on chest imaging. The visceral pleural line is a "
        "thin white line separated from the chest wall with no lung markings "
        "beyond it, which is the hallmark radiographic finding."
    )
    chunks = chunk_text(text, chunk_size=120, chunk_overlap=40)

    words = set(re.findall(r"[A-Za-z]+", text))
    for chunk in chunks:
        first = re.match(r"[A-Za-z]+", chunk)
        if first:
            assert first.group() in words, (
                f"chunk begins with word fragment {first.group()!r}: {chunk[:60]!r}"
            )


def test_overlap_still_provides_context():
    """
    Snapping to a word boundary must not eliminate the overlap — the whole
    point is that a finding spanning a boundary appears complete somewhere.
    """
    text = "Alpha bravo charlie delta echo foxtrot golf hotel india juliet. " * 6
    chunks = chunk_text(text, chunk_size=150, chunk_overlap=40)

    assert len(chunks) > 1
    # Consecutive chunks should share at least one word.
    for earlier, later in zip(chunks, chunks[1:]):
        shared = set(earlier.split()) & set(later.split())
        assert shared, f"no overlap between {earlier[-40:]!r} and {later[:40]!r}"


def test_zero_overlap_is_respected_not_overridden():
    """
    Regression test: chunk_overlap=0 is a legitimate request for no overlap.
    An `or`-based default would silently replace it with the configured value.
    """
    text = "abcdefghij" * 30  # no separators at all -> character fallback
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=0)

    # With zero overlap the pieces should reconstruct the input exactly.
    assert "".join(chunks) == text
