"""
RadAssist AI — PMC Figure Fetcher (Phase 4, Step 3)

Downloads figures from the open-access articles already in the corpus and
stores them as `medical_images` linked to their parent document.

WHY THIS IS THE GOOD CORPUS:
~250 articles are already ingested, already CC-licensed, already radiology.
Every figure carries a caption written by the authors — genuinely paired
image-text data, which is what makes multimodal retrieval possible later. No
scraping, no licensing grey area, no labelling effort.

    article (already ingested)
        └── <fig> ──→ MedicalImage(source_type="pmc_figure")
                        caption   = the figure legend  ← the text half
                        licence   = the article's terms
                        document_id → back to the article

════════════════════════════════════════════════════════════════════
⚠️  PARTIAL SUCCESS IS THE DESIGN GOAL, NOT A CONCESSION
════════════════════════════════════════════════════════════════════
This project has lost entire afternoons to ISP-level DNS filtering: 25%
packet loss, `getaddrinfo` failures, pip and git both dying mid-operation.
A figure run touches hundreds of URLs, so on a bad connection SOME of them
WILL fail. That must not be an all-or-nothing outcome.

So every figure is isolated: a failure is counted, recorded with its reason,
and the loop continues. Re-running tops up what is missing rather than
starting over, because figures already stored are skipped by source_url.
Three flaky runs therefore converge on a complete corpus, which is the only
behaviour that is actually usable here.

The same reasoning as `fetch_and_ingest_pmc`, applied one level down: there,
one bad article must not kill a topic; here, one bad image must not kill an
article.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.document import Document
from app.models.image import MedicalImage
from app.services import image_processing, image_storage
from app.services.knowledge_seeder import ncbi_is_configured
from app.services.pmc_fetcher import extract_figures, _licence_of

settings = get_settings()
logger = logging.getLogger(__name__)


# Be a good citizen: NCBI asks for <= 3 requests/second without an API key.
# Figures come from the same infrastructure, so the same courtesy applies.
_REQUEST_DELAY_SECONDS = 0.4

# A figure larger than this is a supplementary poster or a multi-panel plate
# at print resolution — not worth the disk for a thumbnail in an evidence
# panel.
_MAX_FIGURE_BYTES = 15 * 1024 * 1024

# Images that are almost certainly not figures: journal logos, ORCID icons,
# "author photo" headshots. Cheap to exclude by filename, and they otherwise
# pollute the corpus with dozens of identical marks.
_SKIP_HREF_HINTS = ("logo", "icon", "orcid", "cover", "banner", "headshot")


def _figure_url(pmcid: str, href: str) -> str:
    """
    Build the download URL for a figure.

    JATS gives a bare graphic name — "gr1", "12880_2023_1021_Fig2_HTML" — not
    a URL. PMC serves these from a per-article path, so the article's PMCID is
    what turns a name into something fetchable.
    """
    pmcid = pmcid.strip()
    if not pmcid.upper().startswith("PMC"):
        pmcid = f"PMC{pmcid}"

    # Some exports already include an extension, most do not. PMC resolves the
    # extensionless form, so leave whatever we were given alone.
    return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/bin/{href}"


def _should_skip(href: str) -> bool:
    lowered = href.lower()
    return any(hint in lowered for hint in _SKIP_HREF_HINTS)


async def _download(url: str) -> bytes:
    """
    Fetch one figure. Raises on any failure — the caller isolates it.

    httpx rather than the Entrez client: this is a plain static file, not an
    E-utilities call, and going through Bio.Entrez would add nothing but a
    misleading stack trace when it fails.
    """
    import httpx

    async with httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        headers={
            # NCBI blocks unidentified bulk clients, and rightly so. The same
            # contact address the E-utilities calls already use.
            "User-Agent": (
                f"RadAssistAI/0.1 (+{settings.NCBI_EMAIL or 'unknown'})"
            ),
        },
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        content = response.content

    if not content:
        raise ValueError("empty response")
    if len(content) > _MAX_FIGURE_BYTES:
        raise ValueError(
            f"{len(content) / 1e6:.1f}MB exceeds the "
            f"{_MAX_FIGURE_BYTES / 1e6:.0f}MB figure limit"
        )
    return content


async def _fetch_article_xml(pmcid: str) -> ET.Element | None:
    """Re-fetch one article's JATS so its <fig> elements can be read."""
    from Bio import Entrez

    handle = await asyncio.to_thread(
        Entrez.efetch, db="pmc", id=pmcid.replace("PMC", ""), rettype="xml"
    )
    try:
        raw = await asyncio.to_thread(handle.read)
    finally:
        handle.close()

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")

    root = ET.fromstring(raw)
    for node in root.iter():
        if node.tag.split("}")[-1] == "article":
            return node
    return None


async def _store_figure(
    db: AsyncSession,
    document: Document,
    figure: dict,
    licence: str,
    url: str,
    data: bytes,
) -> None:
    """Normalise, thumbnail and record one figure."""
    stored, mime, width, height = image_processing.normalise(data)

    image_id = uuid.uuid4()
    suffix = f".{mime.split('/')[-1]}"
    rel = image_storage.new_relative_path(image_id, suffix)
    size = image_storage.write_bytes(rel, stored)

    thumb_rel = None
    try:
        thumb_rel = image_storage.thumbnail_path_for(rel)
        image_storage.write_bytes(thumb_rel, image_processing.make_thumbnail(stored))
    except Exception as e:  # noqa: BLE001
        thumb_rel = None
        logger.warning("Thumbnail failed for figure %s: %s", url, e)

    label = figure.get("label") or ""
    caption = figure.get("caption") or ""

    db.add(MedicalImage(
        id=image_id,
        document_id=document.id,
        filename=figure.get("href") or f"{image_id}{suffix}",
        storage_path=rel,
        thumbnail_path=thumb_rel,
        mime_type=mime,
        file_size=size,
        width=width,
        height=height,
        source_type="pmc_figure",
        source_url=url,
        # The caption is the text half of the pair. Prefixed with the label so
        # "Fig. 2" is still identifiable once the image is out of the article.
        caption=" ".join(p for p in (label, caption) if p).strip() or None,
        licence=licence or None,
        # ⚠️  NEVER True for anything but a parsed DICOM. A published figure
        # has no PHI to remove, but "no PHI present" and "de-identification
        # ran" are different claims and only one of them is ours to make.
        is_deidentified=False,
        status="completed",
    ))


async def fetch_figures_for_corpus(
    db: AsyncSession,
    limit_documents: int | None = None,
    max_figures_per_document: int = 8,
) -> dict:
    """
    Download figures for already-ingested PMC articles.

    Idempotent: figures already stored (matched by source_url) are skipped, so
    re-running after a flaky connection tops up rather than duplicating.
    """
    # ⚠️  EVERY SKIP IS COUNTED.
    # The first version had two silent `continue`s — no PMCID, and no <fig>
    # in the article — and both produced an identical all-zero summary with
    # an empty error list. "Nothing happened and I won't say why" is not a
    # diagnostic. Each reason now has its own counter.
    results: dict = {
        "documents_scanned": 0,
        "skipped_no_pmcid": 0,
        "articles_fetched": 0,
        "articles_without_figures": 0,
        "documents_with_figures": 0,
        "figures_found": 0,
        "figures_stored": 0,
        "figures_skipped_existing": 0,
        "figures_skipped_decorative": 0,
        "figures_failed": 0,
        "errors": [],
    }

    if not ncbi_is_configured():
        results["errors"].append(
            "NCBI_EMAIL is not set to a real address. NCBI requires a "
            "contactable email on every request. Set it in backend/.env"
        )
        return results

    try:
        from Bio import Entrez
    except ImportError:
        results["errors"].append("biopython not installed — pip install biopython")
        return results

    try:
        import httpx  # noqa: F401
    except ImportError:
        results["errors"].append("httpx not installed — pip install httpx")
        return results

    Entrez.email = settings.NCBI_EMAIL
    if settings.NCBI_API_KEY and not settings.NCBI_API_KEY.startswith("your_"):
        Entrez.api_key = settings.NCBI_API_KEY

    # Only articles that came from PMC and recorded their PMCID — the URL
    # carries it, and without it a figure name cannot be resolved to a file.
    stmt = (
        select(Document)
        .where(Document.source_type == "pmc_open_access")
        .order_by(Document.created_at.desc())
    )
    if limit_documents:
        stmt = stmt.limit(limit_documents)

    documents = list((await db.execute(stmt)).scalars().all())

    # Every figure URL already stored, fetched once. Checking per figure would
    # be a query per image — hundreds of round-trips to answer a question one
    # set membership test can.
    existing_urls = set(
        (await db.execute(
            select(MedicalImage.source_url)
            .where(MedicalImage.source_type == "pmc_figure")
        )).scalars().all()
    )

    for document in documents:
        results["documents_scanned"] += 1

        pmcid = _pmcid_of(document)
        if not pmcid:
            results["skipped_no_pmcid"] += 1
            if len(results["errors"]) < 25:
                results["errors"].append(
                    f"no PMCID for '{document.title[:60]}' "
                    f"(source_url={document.source_url!r}, "
                    f"filename={document.filename!r})"
                )
            continue

        # ── Per-ARTICLE isolation ──
        try:
            article = await _fetch_article_xml(pmcid)
            if article is None:
                results["errors"].append(f"{pmcid}: no <article> in the efetch reply")
                continue
            results["articles_fetched"] += 1
            figures = extract_figures(article)
            licence = _licence_of(article)
        except Exception as e:  # noqa: BLE001
            results["figures_failed"] += 1
            if len(results["errors"]) < 25:
                results["errors"].append(f"{pmcid}: could not re-fetch article — {e}")
            continue

        if not figures:
            results["articles_without_figures"] += 1
            continue

        results["documents_with_figures"] += 1
        stored_here = 0

        for figure in figures[:max_figures_per_document]:
            href = figure.get("href") or ""
            if not href or _should_skip(href):
                results["figures_skipped_decorative"] += 1
                continue

            results["figures_found"] += 1
            url = _figure_url(pmcid, href)

            if url in existing_urls:
                results["figures_skipped_existing"] += 1
                continue

            # ── Per-FIGURE isolation ──
            # THE LOAD-BEARING TRY. One 404, one timeout, one DNS failure must
            # cost exactly one figure. Without this, a single bad image ends
            # the run and everything after it is lost.
            try:
                data = await _download(url)
                await _store_figure(db, document, figure, licence, url, data)
                existing_urls.add(url)
                results["figures_stored"] += 1
                stored_here += 1
            except Exception as e:  # noqa: BLE001
                results["figures_failed"] += 1
                # Keep the error list bounded — on a failing connection this
                # would otherwise grow to thousands of near-identical entries
                # and bury the one that explains the cause.
                if len(results["errors"]) < 25:
                    results["errors"].append(f"{url}: {e}")
                logger.warning("Figure failed %s: %s", url, e)

            await asyncio.sleep(_REQUEST_DELAY_SECONDS)

        # Commit per article, not at the end. An interrupted run — which on
        # this connection is the expected case — keeps everything it managed
        # to fetch instead of rolling all of it back.
        if stored_here:
            try:
                await db.commit()
            except Exception as e:  # noqa: BLE001
                await db.rollback()
                results["errors"].append(f"{pmcid}: commit failed — {e}")

    return results


def _pmcid_of(document: Document) -> str:
    """
    Recover the PMCID from the stored article. Three places, in order.

    ⚠️  THE SOURCE URL IS NOT ALWAYS A PMC URL.
    pmc_fetcher builds it as PMC-first, PubMed-fallback:

        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1234567/   ← has it
        https://pubmed.ncbi.nlm.nih.gov/38123456/               ← does not

    An article whose JATS omitted <article-id pub-id-type="pmc"> gets the
    second form, and reading only the URL then yields nothing — silently, for
    every such article.

    The filename is the reliable second source, because ingestion writes
    `pmc_{pmcid or pmid}.txt`, and the description carries the PMID as a last
    resort. Figures live under the PMCID, so an article we can only identify
    by PMID is genuinely unusable here — but that should be counted, not
    guessed at.
    """
    # 1. The canonical location.
    url = (document.source_url or "").rstrip("/")
    if "/PMC" in url:
        tail = url.rsplit("/PMC", 1)[-1]
        digits = "".join(c for c in tail if c.isdigit())
        if digits:
            return f"PMC{digits}"

    # 2. The filename: ingestion writes pmc_PMC1234567.txt
    filename = document.filename or ""
    if "PMC" in filename:
        tail = filename.rsplit("PMC", 1)[-1]
        digits = "".join(c for c in tail if c.isdigit())
        if digits:
            return f"PMC{digits}"

    return ""


async def figure_stats(db: AsyncSession) -> dict:
    """Coverage summary — how much of the corpus actually has figures."""
    total_figures = (await db.execute(
        select(func.count(MedicalImage.id))
        .where(MedicalImage.source_type == "pmc_figure")
    )).scalar() or 0

    documents_with = (await db.execute(
        select(func.count(func.distinct(MedicalImage.document_id)))
        .where(MedicalImage.source_type == "pmc_figure")
    )).scalar() or 0

    pmc_documents = (await db.execute(
        select(func.count(Document.id))
        .where(Document.source_type == "pmc_open_access")
    )).scalar() or 0

    with_caption = (await db.execute(
        select(func.count(MedicalImage.id))
        .where(MedicalImage.source_type == "pmc_figure")
        .where(MedicalImage.caption.isnot(None))
    )).scalar() or 0

    return {
        "figures": total_figures,
        "documents_with_figures": documents_with,
        "pmc_documents": pmc_documents,
        "coverage": (
            round(documents_with / pmc_documents, 3) if pmc_documents else 0.0
        ),
        "figures_with_captions": with_caption,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
