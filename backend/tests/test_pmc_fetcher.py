"""
Tests for the PubMed Central Open Access fetcher.

No network — every test parses a JATS XML fixture inline.

The two behaviours that matter most here:
  1. Only confirmably open-access articles are ingested.
  2. The article's OWN citation markers are stripped, so they can't collide
     with our [N] citation scheme.
"""

import xml.etree.ElementTree as ET

import pytest

from app.services.pmc_fetcher import (
    parse_pmc_article,
    _is_open_access,
    _strip_noise,
    _text_of,
    DEFAULT_TOPICS,
    _MIN_ARTICLE_CHARS,
)


def article_xml(
    *,
    title="Imaging of Pneumothorax: A Review",
    licence='<license license-type="open-access"><license-p>CC BY</license-p></license>',
    body="<sec><title>Findings</title><p>The visceral pleural line is the hallmark sign.</p></sec>",
    abstract="<abstract><p>A review of pneumothorax imaging.</p></abstract>",
    pmcid="PMC1234567",
    pmid="31234567",
    doi="10.1000/example",
) -> ET.Element:
    filler = "Additional clinical detail describing the finding in context. " * 20
    return ET.fromstring(f"""
    <article>
      <front>
        <journal-meta><journal-title>Radiology Review</journal-title></journal-meta>
        <article-meta>
          <article-id pub-id-type="pmc">{pmcid}</article-id>
          <article-id pub-id-type="pmid">{pmid}</article-id>
          <article-id pub-id-type="doi">{doi}</article-id>
          <title-group><article-title>{title}</article-title></title-group>
          <pub-date><year>2025</year></pub-date>
          {abstract}
          <permissions>{licence}</permissions>
        </article-meta>
      </front>
      <body>{body}<sec><p>{filler}</p></sec></body>
    </article>
    """)


# ══════════════════════════════════════════════════════════════
# OPEN ACCESS GATING
# ══════════════════════════════════════════════════════════════


def test_open_access_licence_is_accepted():
    assert _is_open_access(article_xml()) is True


def test_creative_commons_url_is_accepted():
    lic = '<license xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="https://creativecommons.org/licenses/by/4.0/"><license-p>CC BY 4.0</license-p></license>'
    assert _is_open_access(article_xml(licence=lic)) is True


def test_article_with_no_licence_is_rejected():
    """
    Ambiguity must default to REJECT. Skipping a usable article costs nothing;
    ingesting a restricted one is a licensing problem.
    """
    assert _is_open_access(article_xml(licence="")) is False


def test_restricted_licence_is_rejected():
    lic = '<license license-type="publisher-specific"><license-p>All rights reserved.</license-p></license>'
    assert _is_open_access(article_xml(licence=lic)) is False


def test_parse_returns_none_for_non_open_access():
    assert parse_pmc_article(article_xml(licence="")) is None


# ══════════════════════════════════════════════════════════════
# CITATION MARKER STRIPPING
# ══════════════════════════════════════════════════════════════
# THE IMPORTANT ONE. JATS body text embeds the article's own reference
# markers as <xref ref-type="bibr">12</xref>, which flatten to bare numbers.
# Left in, they collide with our [N] citation scheme: the LLM sees them in
# context, may echo them, and the frontend then tries to resolve "[12]"
# against source 12 of 5.
# ══════════════════════════════════════════════════════════════


def test_xref_citation_markers_are_stripped():
    body = (
        "<sec><p>Pneumothorax is common "
        '<xref ref-type="bibr" rid="b1">[1]</xref> and may be tension '
        '<xref ref-type="bibr" rid="b2">[2]</xref>.</p></sec>'
    )
    record = parse_pmc_article(article_xml(body=body))

    assert record is not None
    assert "[1]" not in record["content"]
    assert "[2]" not in record["content"]
    assert "Pneumothorax is common" in record["content"]


def test_tables_and_figures_are_stripped():
    body = (
        "<sec><p>Real prose here.</p>"
        "<table-wrap><table><tr><td>cell</td></tr></table></table-wrap>"
        "<fig><caption><p>Figure caption text</p></caption></fig></sec>"
    )
    record = parse_pmc_article(article_xml(body=body))

    assert "Real prose here." in record["content"]
    assert "cell" not in record["content"]
    assert "Figure caption text" not in record["content"]


def test_reference_list_is_stripped():
    body = (
        "<sec><p>Body prose.</p></sec>"
        "<ref-list><ref><mixed-citation>Smith J. Journal. 2020;1:1-10.</mixed-citation></ref></ref-list>"
    )
    record = parse_pmc_article(article_xml(body=body))
    assert "Smith J." not in record["content"]


# ══════════════════════════════════════════════════════════════
# METADATA EXTRACTION — what makes the evidence verifiable
# ══════════════════════════════════════════════════════════════


def test_identifiers_are_captured():
    record = parse_pmc_article(article_xml())

    assert record["pmcid"] == "PMC1234567"
    assert record["pmid"] == "31234567"
    assert record["doi"] == "10.1000/example"
    assert record["journal"] == "Radiology Review"
    assert record["year"] == "2025"


def test_source_url_points_at_a_real_article():
    """
    This URL is the whole point — it's what a radiologist clicks to check a
    claim. The curated seed content has nothing equivalent.
    """
    record = parse_pmc_article(article_xml())
    assert record["url"] == "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1234567/"


def test_falls_back_to_pubmed_url_without_pmcid():
    record = parse_pmc_article(article_xml(pmcid=""))
    assert record["url"] == "https://pubmed.ncbi.nlm.nih.gov/31234567/"


def test_title_and_abstract_included_in_content():
    record = parse_pmc_article(article_xml())
    assert record["title"] in record["content"]
    assert "A review of pneumothorax imaging." in record["content"]


# ══════════════════════════════════════════════════════════════
# CONTENT QUALITY GATES
# ══════════════════════════════════════════════════════════════


def test_metadata_only_article_is_rejected():
    """Records with no retrievable body aren't worth embedding."""
    stub = ET.fromstring("""
    <article><front><article-meta>
      <title-group><article-title>Short</article-title></title-group>
      <permissions><license license-type="open-access"><license-p>CC BY</license-p></license></permissions>
    </article-meta></front><body><p>Too short.</p></body></article>
    """)
    assert parse_pmc_article(stub) is None


def test_article_without_title_is_rejected():
    stub = ET.fromstring("""
    <article><front><article-meta>
      <permissions><license license-type="open-access"><license-p>CC BY</license-p></license></permissions>
    </article-meta></front><body><p>Body without a title.</p></body></article>
    """)
    assert parse_pmc_article(stub) is None


def test_accepted_article_meets_minimum_length():
    record = parse_pmc_article(article_xml())
    assert len(record["content"]) >= _MIN_ARTICLE_CHARS


# ══════════════════════════════════════════════════════════════
# TEXT NORMALISATION
# ══════════════════════════════════════════════════════════════


def test_whitespace_is_normalised():
    el = ET.fromstring("<p>Line one.\n\n   Line   two.\t\tLine three.</p>")
    assert _text_of(el) == "Line one. Line two. Line three."


def test_text_of_handles_none():
    assert _text_of(None) == ""


def test_strip_noise_is_safe_on_clean_xml():
    el = ET.fromstring("<sec><p>Nothing to strip.</p></sec>")
    _strip_noise(el)
    assert _text_of(el) == "Nothing to strip."


# ══════════════════════════════════════════════════════════════
# TOPIC COVERAGE
# ══════════════════════════════════════════════════════════════


def test_topics_extend_beyond_chest_imaging():
    """
    The curated seed set is heavily chest-weighted. The whole point of PMC
    ingestion is broadening coverage, so the default topics must reach into
    other subspecialties.
    """
    blob = " ".join(DEFAULT_TOPICS).lower()
    for subspecialty in ("stroke", "musculoskeletal", "abdominal", "breast", "paediatric"):
        assert subspecialty in blob, f"no {subspecialty} coverage in default topics"


def test_topics_are_unique():
    assert len(set(DEFAULT_TOPICS)) == len(DEFAULT_TOPICS)
