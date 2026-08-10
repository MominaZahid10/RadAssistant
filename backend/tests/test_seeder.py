"""
Tests for knowledge base seed data and the NCBI configuration gate.

WHY THE NCBI GATE MATTERS:
The original gate was `if settings.NCBI_EMAIL:` — but the .env shipped with
`NCBI_EMAIL=your_email@example.com`, which is a truthy string. The "skip if
not configured" branch could therefore never fire, and the app would identify
itself to a public government API with a fake address. NCBI's terms of use
require a contactable email; violating that gets IPs throttled or blocked.
"""

import pytest

from app.data.seed_knowledge import SEED_KNOWLEDGE
from app.services import knowledge_seeder


# ══════════════════════════════════════════════════════════════
# NCBI CONFIGURATION GATE
# ══════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "email",
    [
        "",
        "   ",
        "your_email@example.com",      # the value we actually shipped
        "YOUR_EMAIL@EXAMPLE.COM",      # case variant
        "  your_email@example.com  ",  # padded
        "radassist@example.com",       # the old hardcoded fallback
        "changeme",
        "notanemail",                  # no @
        "missing@tld",                 # no dot in domain
    ],
)
def test_placeholder_emails_do_not_enable_ncbi(monkeypatch, email):
    monkeypatch.setattr(knowledge_seeder.settings, "NCBI_EMAIL", email, raising=False)
    assert knowledge_seeder.ncbi_is_configured() is False


@pytest.mark.parametrize(
    "email",
    ["haider.tanoli41@gmail.com", "researcher@university.edu", "a.b@sub.domain.org"],
)
def test_real_emails_enable_ncbi(monkeypatch, email):
    monkeypatch.setattr(knowledge_seeder.settings, "NCBI_EMAIL", email, raising=False)
    assert knowledge_seeder.ncbi_is_configured() is True


# ══════════════════════════════════════════════════════════════
# SEED DATA INTEGRITY
# ══════════════════════════════════════════════════════════════


def test_seed_knowledge_is_not_empty():
    assert len(SEED_KNOWLEDGE) >= 14


def test_every_entry_has_required_fields():
    for entry in SEED_KNOWLEDGE:
        assert entry.get("title"), f"missing title: {entry}"
        assert entry.get("content"), f"missing content: {entry['title']}"
        assert entry.get("source_type"), f"missing source_type: {entry['title']}"


def test_titles_are_unique():
    """
    seed_curated_knowledge() deduplicates by (title, source_type). Duplicate
    titles would make seeding non-idempotent — running it twice would skip
    legitimate articles.
    """
    titles = [e["title"] for e in SEED_KNOWLEDGE]
    duplicates = {t for t in titles if titles.count(t) > 1}
    assert not duplicates, f"duplicate titles: {duplicates}"


def test_content_is_substantial():
    """
    Entries shorter than a few hundred characters produce one thin chunk that
    rarely wins retrieval — not worth the storage.
    """
    for entry in SEED_KNOWLEDGE:
        assert len(entry["content"]) > 800, (
            f"'{entry['title']}' is only {len(entry['content'])} chars"
        )


def test_every_entry_has_source_attribution():
    """
    Provenance is a core requirement of this project — the whole premise is
    traceable, non-black-box answers. Unattributed content undermines that.
    """
    for entry in SEED_KNOWLEDGE:
        assert entry.get("source_attribution"), (
            f"'{entry['title']}' has no source_attribution"
        )


def test_source_types_are_from_known_set():
    """Unknown source_types silently break the /stats breakdown and filters."""
    known = {
        "curated_summary",   # written-to-reflect content, no PMID/DOI
        "pmc_open_access",   # full text from PMC, verifiable
        "statpearls",        # NCBI abstracts, verifiable
        "textbook", "guideline", "report", "research_paper",
        "clinical_note", "curated", "general",
    }
    for entry in SEED_KNOWLEDGE:
        assert entry["source_type"] in known, (
            f"'{entry['title']}' has unexpected source_type "
            f"'{entry['source_type']}'"
        )


def test_curated_content_is_not_labelled_as_a_primary_source():
    """
    These entries are summaries written to reflect their named sources — they
    contain no PMID, DOI or URL, so nothing in them can be traced to a page or
    paper. Labelling them "textbook" or "guideline" in the evidence panel
    would tell a radiologist they're reading a primary source when they are
    not. Verifiable content comes from the PMC/StatPearls ingestion instead.
    """
    overstated = {"textbook", "guideline", "research_paper", "statpearls"}
    wrong = [e["title"] for e in SEED_KNOWLEDGE if e["source_type"] in overstated]
    assert not wrong, f"curated summaries mislabelled as primary sources: {wrong}"
