"""
Tests for BM25 lexical retrieval (Phase 3.6).

THE CASE THIS EXISTS TO FIX, measured on a 248-document corpus:

    query:  "What are the radiographic findings of pneumothorax?"
    corpus: contains a chunk beginning "CXR Findings of Pneumothorax:
            - Visceral pleural line: A thin white line..."
    vector search: returns 12 papers about pneumothorax DETECTION and never
                   surfaces that chunk at all.

Embeddings blur exact terms; BM25 is built on them. `test_finds_the_chunk_
vector_search_missed` reproduces the scenario directly.
"""

import pytest

from app.services.lexical_service import LexicalIndex, tokenize


# Modelled on real corpus content — one explanatory passage among several
# term-dense research abstracts, which is the distribution that broke
# vector-only retrieval.
CORPUS = [
    {"point_id": "p1", "document_id": "d1", "title": "Pneumothorax — Types, Imaging, and Management",
     "source_type": "curated_summary", "chunk_index": 6,
     "text": "CXR Findings of Pneumothorax: Visceral pleural line: A thin white line "
             "separated from the chest wall with NO lung markings beyond it. This is the "
             "hallmark finding."},
    {"point_id": "p2", "document_id": "d2", "title": "Deep learning detection of pneumothorax",
     "source_type": "pmc_open_access", "chunk_index": 3,
     "text": "We trained a convolutional network on chest radiographs to detect "
             "pneumothorax. Pneumothorax detection accuracy reached 0.91 AUC across the "
             "pneumothorax validation set."},
    {"point_id": "p3", "document_id": "d3", "title": "Ultrasound after CT-guided biopsy",
     "source_type": "pmc_open_access", "chunk_index": 8,
     "text": "Pneumothorax was monitored by ultrasound after biopsy. Pneumothorax volume "
             "was estimated at intervals following the pneumothorax diagnosis."},
    {"point_id": "p4", "document_id": "d4", "title": "Fleischner Society nodule guidance",
     "source_type": "curated_summary", "chunk_index": 1,
     "text": "The Fleischner Society recommends follow-up CT at 6 to 12 months for a "
             "solid nodule larger than 6 mm in a low-risk patient."},
]


@pytest.fixture
def index():
    idx = LexicalIndex()
    idx.build(CORPUS)
    return idx


# ══════════════════════════════════════════════════════════════
# TOKENIZATION
# ══════════════════════════════════════════════════════════════


def test_tokenize_lowercases_and_splits():
    assert tokenize("Visceral Pleural Line!") == ["visceral", "pleural", "line"]


def test_tokenize_drops_stopwords():
    out = tokenize("What are the findings of pneumothorax")
    assert "the" not in out and "of" not in out and "are" not in out
    assert "findings" in out and "pneumothorax" in out


def test_tokenize_drops_single_characters():
    assert "a" not in tokenize("a thin white line")


def test_tokenize_does_not_stem():
    """
    Medical terms are precise. Stemming 'pleural'→'pleur' risks colliding
    distinct terms, and exactness is the entire reason for a lexical stage.
    """
    assert "pleural" in tokenize("pleural effusion")
    assert "pleur" not in tokenize("pleural effusion")


def test_tokenize_handles_empty():
    assert tokenize("") == []
    assert tokenize("the of and") == []


# ══════════════════════════════════════════════════════════════
# THE CASE THIS WAS BUILT FOR
# ══════════════════════════════════════════════════════════════


def test_finds_the_chunk_vector_search_missed(index):
    """
    THE REGRESSION CASE. Vector search returned twelve detection papers and
    never surfaced the passage containing the exact phrase. BM25 must rank it
    first, because it holds the rare terms while the others merely repeat
    'pneumothorax'.
    """
    hits = index.search("What are the radiographic findings of pneumothorax?", limit=4)
    assert hits, "expected lexical hits"
    assert hits[0]["point_id"] == "p1"


def test_exact_phrase_terms_outrank_term_density(index):
    """
    p2 and p3 repeat 'pneumothorax' far more often than p1. Term-frequency
    saturation (k1) plus IDF must still put the passage with the rare,
    discriminating terms on top.
    """
    hits = index.search("visceral pleural line lung markings", limit=4)
    assert hits[0]["point_id"] == "p1"


def test_rare_terms_score_higher_than_common_ones(index):
    """'pneumothorax' is in 3 of 4 docs, so its IDF must be low."""
    common = index.search("pneumothorax", limit=4)
    rare = index.search("visceral", limit=4)
    assert rare[0]["bm25_score"] > common[0]["bm25_score"]


# ══════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════


def test_results_are_sorted_by_score(index):
    hits = index.search("pneumothorax findings", limit=4)
    scores = [h["bm25_score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_scores_are_positive(index):
    """
    Without the +1 inside the IDF log, terms appearing in most documents get
    NEGATIVE weight and actively push relevant chunks down the ranking.
    """
    for h in index.search("pneumothorax", limit=4):
        assert h["bm25_score"] > 0


def test_limit_is_respected(index):
    assert len(index.search("pneumothorax", limit=2)) <= 2


def test_metadata_is_carried_through(index):
    """
    Results must be usable without a second round-trip to Qdrant.
    """
    hit = index.search("visceral pleural line", limit=1)[0]
    for key in ("point_id", "document_id", "title", "source_type", "chunk_index"):
        assert key in hit
    assert hit["title"] == "Pneumothorax — Types, Imaging, and Management"


def test_unrelated_query_matches_nothing(index):
    assert index.search("cardiac catheterisation angioplasty stent", limit=5) == []


# ══════════════════════════════════════════════════════════════
# LIFECYCLE — must never break search
# ══════════════════════════════════════════════════════════════


def test_unbuilt_index_returns_empty():
    """Callers treat this as 'no lexical contribution', never as an error."""
    assert LexicalIndex().search("pneumothorax", limit=5) == []


def test_stopword_only_query_returns_empty(index):
    assert index.search("what are the of and", limit=5) == []


def test_empty_query_returns_empty(index):
    assert index.search("", limit=5) == []


def test_build_skips_chunks_with_no_usable_tokens():
    idx = LexicalIndex()
    idx.build([
        {"point_id": "a", "text": "the of and"},      # stopwords only
        {"point_id": "b", "text": ""},                 # empty
        {"point_id": "c", "text": "pneumothorax pleural line"},
    ])
    assert idx.get_info()["chunks"] == 1


def test_stale_flag_forces_rebuild(index):
    assert index.is_built is True
    index.mark_stale()
    assert index.is_built is False, "newly ingested docs must trigger a rebuild"


def test_info_reports_state(index):
    info = index.get_info()
    assert info["built"] is True
    assert info["stale"] is False
    assert info["chunks"] == len(CORPUS)


# ══════════════════════════════════════════════════════════════
# SALIENT TERMS — building a query from an uploaded document
# ══════════════════════════════════════════════════════════════
#
# THE OBSERVED FAILURE, reproduced below:
# A clinician uploaded a lumbar spine report and typed a generic instruction.
# All twelve retrieved sources were papers about the PRACTICE of radiology
# reporting — reporting errors, structured-reporting templates, NLP reporting
# quality — because the instruction, not the report, was what got embedded.


# A corpus deliberately containing both traps: meta-papers about reporting
# (which the instruction pulls) and the clinical content actually wanted.
MIXED_CORPUS = [
    {"point_id": "r1", "document_id": "m1",
     "title": "Revealing the most common reporting errors",
     "text": "Radiology report proofreading. Residents write a preliminary report and "
             "the attending edits the report. Report findings and report impression "
             "sections were compared across reports."},
    {"point_id": "r2", "document_id": "m2",
     "title": "Structured reporting for fibrosing lung disease",
     "text": "A structured report template was developed by consensus. The report "
             "format and reporting standards were agreed among radiologists writing "
             "the findings section of each report."},
    {"point_id": "c1", "document_id": "m3",
     "title": "Vertebral compression fracture imaging",
     "text": "Anterior wedge deformity of the vertebral body indicates a compression "
             "fracture. Loss of anterior vertebral body height is graded by severity. "
             "Retropulsion into the spinal canal must be excluded."},
    {"point_id": "c2", "document_id": "m4",
     "title": "Osteopenia and postmenopausal osteoporosis",
     "text": "Osteopenia describes reduced bone mineral density. Severe postmenopausal "
             "osteoporosis of the lumbar spine and pelvis predisposes to vertebral "
             "fracture. Bone density is assessed by DXA."},
    {"point_id": "c3", "document_id": "m5",
     "title": "Lumbar lordosis and sagittal alignment",
     "text": "Lumbar lordosis is the normal curvature of the lumbar spine. Increased "
             "lordosis alters sagittal alignment and loading of the facet joints."},
]

# The OCR output from the report that motivated this, garbled words intact.
OCR_REPORT = """
    The lumbar spine ts hyperiordotic.
    Marked osteopenia is noted throughout the lumbar spine and peivis.
    An anterior wedge deformity of Tz with a 50% loss of anterior vertebral
    body height is noted. There is no evidence of posterior vertebral body
    collapse or bony projection into the neutral canal.
    All joint spaces appear well maintained. The solt tissues are unremarkable.
    IMPRESSION: Tracompression tracture. Severe post-menopausal osteoporosis.
"""


@pytest.fixture
def mixed_index():
    idx = LexicalIndex()
    idx.build(MIXED_CORPUS)
    return idx


def test_salient_terms_surface_the_clinical_subject(mixed_index):
    """The terms that describe what the report is ABOUT must be selected."""
    terms = set(mixed_index.salient_terms(OCR_REPORT, top_k=24))
    for expected in ("osteopenia", "vertebral", "wedge", "lumbar", "osteoporosis"):
        assert expected in terms, f"{expected!r} missing from {sorted(terms)}"


def test_salient_terms_drop_ocr_garbage(mixed_index):
    """
    Misread words are the HIGHEST-IDF tokens in the document — nothing is
    rarer than a word that doesn't exist. Requiring corpus membership is what
    keeps them out of the query.
    """
    terms = set(mixed_index.salient_terms(OCR_REPORT, top_k=50))
    for garbage in ("hyperiordotic", "peivis", "solt", "tracompression", "tracture"):
        assert garbage not in terms, f"OCR misread {garbage!r} leaked into the query"


def test_salient_terms_deprioritise_report_boilerplate(mixed_index):
    """
    'report'/'findings' appear across the corpus, so their IDF is low and they
    sink beneath the clinical terms. This is what stopped retrieval returning
    papers about the practice of writing reports.
    """
    ranked = mixed_index.salient_terms(
        "Please review the attached report findings.\n" + OCR_REPORT, top_k=8
    )
    assert "report" not in ranked
    assert "findings" not in ranked
    assert any(t in ranked for t in ("osteopenia", "vertebral", "osteoporosis"))


def test_salient_terms_respect_top_k(mixed_index):
    assert len(mixed_index.salient_terms(OCR_REPORT, top_k=5)) == 5


def test_salient_terms_on_unbuilt_index_returns_empty():
    """Never raises — the caller falls back to the raw query."""
    assert LexicalIndex().salient_terms(OCR_REPORT) == []


def test_salient_terms_on_unrelated_document_returns_nothing_invented(mixed_index):
    """A document sharing no vocabulary with the corpus yields no terms,
    rather than terms the corpus cannot support."""
    assert mixed_index.salient_terms("qwerty zxcvbn plugh xyzzy") == []
