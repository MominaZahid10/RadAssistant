"""
Tests for PMC figure extraction (Phase 4, Step 3).

Entirely offline — JATS fixtures, no network. That is deliberate: this
project's most common failure is ISP-level DNS filtering, and a test suite
that needs NCBI to be reachable is a test suite that fails for reasons having
nothing to do with the code.

THE BUG THIS SUITE EXISTS TO CATCH:
`_strip_noise()` deletes <fig>. If figures are extracted after it runs, every
article yields zero figures, no exception is raised, and the endpoint reports
a successful run over an empty corpus. Ordering is the whole feature, so it is
asserted directly.
"""

import xml.etree.ElementTree as ET

import pytest

from app.services import figure_fetcher
from app.services.pmc_fetcher import (
    _licence_of,
    _strip_noise,
    extract_figures,
    parse_pmc_article,
)


# ══════════════════════════════════════════════════════════════
# FIXTURES — shaped like real PMC output
# ══════════════════════════════════════════════════════════════

ARTICLE_XML = """
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <front>
    <journal-meta><journal-title>European Radiology</journal-title></journal-meta>
    <article-meta>
      <article-id pub-id-type="pmc">1234567</article-id>
      <article-title>Imaging of vertebral compression fractures</article-title>
      <pub-date><year>2024</year></pub-date>
      <permissions>
        <license license-type="open-access"
                 xlink:href="https://creativecommons.org/licenses/by/4.0/">
          <license-p>This article is licensed under CC BY 4.0.</license-p>
        </license>
      </permissions>
      <abstract><p>We reviewed imaging findings in vertebral compression
      fractures across 163 consecutive patients presenting to a single
      tertiary centre over a four-year period, comparing modalities.</p></abstract>
    </article-meta>
  </front>
  <body>
    <sec>
      <title>Findings</title>
      <p>Anterior wedge deformity was present in most cases
         <xref ref-type="bibr">12</xref>, and posterior wall involvement
         was rare. Height loss was graded on sagittal reformats in every
         patient, with two readers scoring independently and disagreements
         resolved by consensus with a third reader. Marked osteopenia of
         the lumbar spine and pelvis was recorded separately using a
         semi-quantitative scale, and bone mineral density was available
         from dual-energy x-ray absorptiometry in a subset. Retropulsion
         into the spinal canal was assessed on axial images and, where
         present, prompted cross-sectional follow-up. Facet joint spaces
         and intervertebral disc heights were reported as maintained or
         reduced, and paraspinal soft tissues were reviewed for
         haematoma or swelling in every case. Inter-reader agreement was
         substantial for anterior height loss and moderate for posterior
         wall involvement, consistent with previously published series.</p>
      <fig id="Fig1">
        <label>Fig. 1</label>
        <caption><p>Sagittal CT showing anterior wedge deformity of T12
        with 50% height loss.</p></caption>
        <graphic xlink:href="gr1.jpg"/>
      </fig>
      <fig id="Fig2">
        <label>Fig. 2</label>
        <caption><p>MRI STIR sequence demonstrating marrow oedema.</p></caption>
        <graphic xlink:href="12880_2023_Fig2_HTML"/>
      </fig>
    </sec>
  </body>
</article>
""".strip()


@pytest.fixture
def article() -> ET.Element:
    return ET.fromstring(ARTICLE_XML)


# ══════════════════════════════════════════════════════════════
# ORDERING — the load-bearing constraint
# ══════════════════════════════════════════════════════════════


def test_extract_figures_must_run_before_strip_noise(article):
    """
    ⚠️  THE SILENT FAILURE THIS PREVENTS.
    _strip_noise() removes <fig>. Extracting afterwards returns [] with no
    error, so the endpoint would report a successful run having stored
    nothing. Reversing two lines in parse_pmc_article() would do it.
    """
    body = article.find(".//body")

    before = extract_figures(article)
    assert len(before) == 2, "figures must be visible before stripping"

    _strip_noise(body)
    after = extract_figures(article)
    assert after == [], "strip_noise is expected to remove <fig> — it still does"


def test_captions_stay_out_of_the_article_text(article):
    """
    The Phase 3 reasoning has NOT been reversed, only the figure handling.
    A caption inside a text chunk is noise: "Fig. 1. Sagittal CT..." retrieves
    for "sagittal CT" and then tells the reader nothing, because the chunk has
    no image. The caption belongs to the image record instead.
    """
    parsed = parse_pmc_article(article)
    assert parsed is not None
    assert "Sagittal CT showing anterior wedge" not in parsed["content"]
    assert "marrow oedema" not in parsed["content"]
    # ...but the body prose survives.
    assert "Anterior wedge deformity was present" in parsed["content"]


def test_parse_returns_figures_alongside_the_text(article):
    parsed = parse_pmc_article(article)
    assert parsed is not None
    assert len(parsed["figures"]) == 2
    assert parsed["figures"][0]["href"] == "gr1.jpg"
    assert "Sagittal CT" in parsed["figures"][0]["caption"]


def test_article_citation_markers_are_still_stripped(article):
    """
    Regression guard on the Phase 3 fix. JATS embeds the article's OWN
    citation markers, which flatten to bare "12" and collide with our [N]
    citation scheme — the model can echo one and the frontend then resolves
    source 12 of 5. Adding figure extraction must not have reintroduced them.
    """
    parsed = parse_pmc_article(article)
    assert parsed is not None
    body = parsed["content"]
    assert "[12]" not in body
    # The xref sat between "cases" and ", and posterior" — the marker is gone
    # and the sentence closes up.
    assert "in most cases , and posterior wall" in body or \
           "in most cases, and posterior wall" in body


# ══════════════════════════════════════════════════════════════
# FIGURE EXTRACTION
# ══════════════════════════════════════════════════════════════


def test_label_and_caption_are_captured(article):
    figs = extract_figures(article)
    assert figs[0]["label"] == "Fig. 1"
    assert figs[0]["fig_id"] == "Fig1"
    assert figs[1]["label"] == "Fig. 2"


def test_figure_without_a_graphic_is_dropped():
    """A figure we cannot fetch is not worth a database row."""
    xml = """
    <article xmlns:xlink="http://www.w3.org/1999/xlink"><body>
      <fig id="F1"><label>Fig. 1</label><caption><p>No image.</p></caption></fig>
    </body></article>
    """
    assert extract_figures(ET.fromstring(xml)) == []


def test_namespaced_href_is_found():
    """PMC exports vary in how they prefix xlink. Match on the local name."""
    xml = """
    <article xmlns:xlink="http://www.w3.org/1999/xlink"><body>
      <fig><caption><p>C</p></caption><graphic xlink:href="img9.tif"/></fig>
    </body></article>
    """
    assert extract_figures(ET.fromstring(xml))[0]["href"] == "img9.tif"


def test_no_figures_is_not_an_error():
    xml = "<article><body><sec><p>Text only.</p></sec></body></article>"
    assert extract_figures(ET.fromstring(xml)) == []


# ══════════════════════════════════════════════════════════════
# LICENCE — recorded, never assumed
# ══════════════════════════════════════════════════════════════


def test_licence_parsed_from_creative_commons_url(article):
    assert _licence_of(article) == "CC-BY 4.0"


def test_non_commercial_licence_is_distinguished():
    """
    ⚠️  WHY THE EXACT TERMS MATTER. 'Open access' is not one licence. CC-BY
    permits derivatives with attribution; CC-BY-NC-ND permits neither
    commercial use nor derivatives. Collapsing them to 'open access' loses
    the only part anyone will later need.
    """
    xml = """
    <article xmlns:xlink="http://www.w3.org/1999/xlink"><front><article-meta>
      <license xlink:href="https://creativecommons.org/licenses/by-nc-nd/4.0/"/>
    </article-meta></front></article>
    """
    assert _licence_of(ET.fromstring(xml)) == "CC-BY-NC-ND 4.0"


def test_licence_falls_back_to_the_declared_type():
    xml = """<article><front><article-meta>
      <license license-type="open-access"><license-p>Free to read.</license-p></license>
    </article-meta></front></article>"""
    assert _licence_of(ET.fromstring(xml)) == "open-access"


def test_missing_licence_returns_empty_not_a_guess():
    """Never invent permission we did not confirm."""
    assert _licence_of(ET.fromstring("<article><body/></article>")) == ""


# ══════════════════════════════════════════════════════════════
# URL CONSTRUCTION
# ══════════════════════════════════════════════════════════════


def test_figure_url_uses_the_articles_pmcid():
    url = figure_fetcher._figure_url("PMC1234567", "gr1.jpg")
    assert url == "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1234567/bin/gr1.jpg"


def test_figure_url_tolerates_a_bare_numeric_id():
    assert "PMC1234567" in figure_fetcher._figure_url("1234567", "gr1.jpg")


def test_pmcid_recovered_from_the_stored_source_url():
    doc = type("D", (), {
        "source_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7654321/"
    })()
    assert figure_fetcher._pmcid_of(doc) == "PMC7654321"


def test_pmcid_absent_is_empty_not_an_exception():
    doc = type("D", (), {
        "source_url": "https://example.org/article", "filename": "x.txt",
    })()
    assert figure_fetcher._pmcid_of(doc) == ""

    doc_none = type("D", (), {"source_url": None, "filename": None})()
    assert figure_fetcher._pmcid_of(doc_none) == ""


# ══════════════════════════════════════════════════════════════
# NON-FIGURE IMAGERY
# ══════════════════════════════════════════════════════════════


@pytest.mark.parametrize("href", [
    "journal-logo.png", "ORCID-icon.svg", "cover-image.jpg",
    "site-banner.gif", "author-headshot.jpg",
])
def test_decorative_images_are_skipped(href):
    """Otherwise the corpus fills with dozens of identical journal marks."""
    assert figure_fetcher._should_skip(href) is True


@pytest.mark.parametrize("href", [
    "gr1.jpg", "12880_2023_1021_Fig2_HTML", "fx1.tif", "1471-2342-14-9-2.png",
])
def test_real_figure_names_are_not_skipped(href):
    assert figure_fetcher._should_skip(href) is False


# ══════════════════════════════════════════════════════════════
# CONFIGURATION GUARDS — fail with a reason, never a traceback
# ══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_missing_ncbi_email_reports_why(monkeypatch):
    """
    NCBI rejects placeholder addresses outright. Returning a 202 that quietly
    does nothing is how the earlier fetch-pmc bug went unnoticed.
    """
    monkeypatch.setattr(
        figure_fetcher, "ncbi_is_configured", lambda: False
    )
    result = await figure_fetcher.fetch_figures_for_corpus(db=None)
    assert result["figures_stored"] == 0
    assert any("NCBI_EMAIL" in e for e in result["errors"])


def test_error_list_is_bounded():
    """
    On a failing connection every figure errors. An unbounded list would grow
    to thousands of near-identical entries and bury the one that explains the
    cause.
    """
    import inspect
    source = inspect.getsource(figure_fetcher.fetch_figures_for_corpus)
    assert 'len(results["errors"]) < 25' in source


def test_each_figure_failure_is_isolated():
    """
    ⚠️  THE LOAD-BEARING TRY. One 404 must cost one figure, not the run.
    Several partial runs are expected to converge on a complete corpus, which
    only works if a failure does not abort everything after it.
    """
    import inspect
    source = inspect.getsource(figure_fetcher.fetch_figures_for_corpus)
    # The download+store pair sits inside its own try, inside the figure loop.
    assert "for figure in figures[:max_figures_per_document]:" in source
    figure_loop = source.split("for figure in figures[:max_figures_per_document]:")[1]
    assert "try:" in figure_loop
    assert "results[\"figures_failed\"] += 1" in figure_loop


def test_commits_per_article_so_interruptions_keep_progress():
    import inspect
    source = inspect.getsource(figure_fetcher.fetch_figures_for_corpus)
    assert "if stored_here:" in source
    assert "await db.commit()" in source


# ══════════════════════════════════════════════════════════════
# PMCID RECOVERY — the silent skip
# ══════════════════════════════════════════════════════════════
#
# ⚠️  OBSERVED: a run reported documents_scanned=10 and every other counter
# zero, with an empty error list. Two different `continue` statements produced
# that identical output, so it said nothing about which had fired.
#
# pmc_fetcher builds source_url PMC-first, PubMed-fallback. An article whose
# JATS omitted <article-id pub-id-type="pmc"> gets a pubmed.ncbi.nlm.nih.gov
# URL with no PMCID in it at all.


def _doc(source_url=None, filename="", title="T"):
    return type("D", (), {
        "source_url": source_url, "filename": filename, "title": title,
    })()


def test_pmcid_from_a_pmc_url():
    doc = _doc("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7654321/")
    assert figure_fetcher._pmcid_of(doc) == "PMC7654321"


def test_pmcid_falls_back_to_the_filename():
    """A PubMed source URL carries no PMCID; ingestion's filename does."""
    doc = _doc("https://pubmed.ncbi.nlm.nih.gov/38123456/", "pmc_PMC7654321.txt")
    assert figure_fetcher._pmcid_of(doc) == "PMC7654321"


def test_pmid_only_article_yields_nothing_rather_than_a_guess():
    """
    Figures live under the PMCID. An article identifiable only by PMID is
    genuinely unusable here — that must be reported, never approximated by
    treating the PMID as a PMCID.
    """
    doc = _doc("https://pubmed.ncbi.nlm.nih.gov/38123456/", "pmc_38123456.txt")
    assert figure_fetcher._pmcid_of(doc) == ""


def test_every_skip_reason_has_its_own_counter():
    """
    An all-zero summary with an empty error list is not a diagnostic. Each
    reason a document can be passed over must be separately countable.
    """
    import inspect
    source = inspect.getsource(figure_fetcher.fetch_figures_for_corpus)
    for counter in (
        "skipped_no_pmcid",
        "articles_fetched",
        "articles_without_figures",
        "figures_skipped_decorative",
    ):
        assert f'results["{counter}"]' in source, f"{counter} is never recorded"


def test_missing_pmcid_is_reported_with_the_evidence():
    """The error must show what was actually stored, or it cannot be acted on."""
    import inspect
    source = inspect.getsource(figure_fetcher.fetch_figures_for_corpus)
    assert "source_url=" in source and "filename=" in source
