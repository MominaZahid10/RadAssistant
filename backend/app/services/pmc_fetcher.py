"""
RadAssist AI — PubMed Central Open Access Fetcher

WHY THIS EXISTS:
The curated seed articles are accurate, but they are *summaries attributed to*
sources rather than *extracts from* them — they carry no PMID, DOI or URL, so a
radiologist reading the evidence panel cannot verify anything. For a system
whose whole premise is traceable answers, that's the weakest link.

This module fixes that by ingesting the **PMC Open Access Subset**: full-text,
peer-reviewed articles that are explicitly licensed for download and reuse.
Every ingested article carries a real PMCID, PMID and DOI, and a source_url a
clinician can click.

    Curated seed        →  accurate, unverifiable, 14 articles
    PMC Open Access     →  peer-reviewed, verifiable, hundreds of articles

WHAT "OPEN ACCESS SUBSET" MEANS:
PMC hosts millions of articles, but only a subset carries a licence permitting
bulk retrieval and redistribution. We restrict every query with
`"open access"[filter]` so we never pull anything outside it. Articles whose
licence we can't positively identify are skipped rather than assumed to be fine.

RATE LIMITS (NCBI policy):
    without API key — 3 requests/second
    with API key    — 10 requests/second
We stay well under both, and NCBI requires a contactable email on every call.
"""

from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.document import Document
from app.services.ingestion import ingest_text_content
from app.services.knowledge_seeder import ncbi_is_configured

settings = get_settings()
logger = logging.getLogger(__name__)


# Default radiology topics. Broad on purpose — the curated set is heavily
# chest-weighted, so these deliberately reach into neuro, MSK, abdominal,
# paediatric and breast imaging to widen coverage.
DEFAULT_TOPICS: list[str] = [
    "chest radiograph interpretation",
    "computed tomography pulmonary angiography embolism",
    "stroke imaging CT perfusion MRI",
    "musculoskeletal MRI fracture",
    "abdominal CT acute abdomen",
    "pulmonary nodule Fleischner follow-up",
    "breast imaging mammography BI-RADS",
    "paediatric radiology imaging",
    "neuroimaging brain tumour MRI",
    "contrast media adverse reaction radiology",
    "ultrasound abdominal diagnosis",
    "structured reporting radiology",
]

# Politeness delay between NCBI calls. 0.4s ≈ 2.5 req/s, under the 3/s
# unauthenticated ceiling with margin for jitter.
_REQUEST_DELAY = 0.4

# Skip anything shorter than this after parsing — usually a metadata-only
# record with no retrievable body.
_MIN_ARTICLE_CHARS = 600

# Guard against a single enormous review consuming the whole corpus.
_MAX_ARTICLE_CHARS = 120_000


# ══════════════════════════════════════════════════════════════
# JATS XML PARSING
# ══════════════════════════════════════════════════════════════
# PMC returns JATS XML. We want readable prose, not markup — and crucially we
# must strip the article's OWN reference markers (see _clean_element below).


def _strip_noise(elem: ET.Element) -> None:
    """
    Remove elements that would pollute the extracted text.

    ⚠️  `<xref ref-type="bibr">` MATTERS MORE THAN IT LOOKS.
    JATS body text embeds the article's own citation markers, which render as
    bare bracketed numbers — "as shown previously [12]". If ingested as-is,
    those collide directly with OUR citation scheme: the LLM sees [12] in the
    context and may echo it, and the frontend would then try to resolve it
    against source 12 of 5. Stripping them keeps the [N] namespace ours alone.

    Also dropped: tables (unreadable as linear text), figure captions without
    their images, and formula markup.
    """
    for tag in (
        "xref", "table-wrap", "table", "graphic", "inline-graphic",
        "disp-formula", "inline-formula", "media", "supplementary-material",
        "fig", "ref-list", "back",
    ):
        for parent in elem.iter():
            for child in list(parent):
                # Tags may carry a namespace prefix.
                if child.tag.split("}")[-1] == tag:
                    parent.remove(child)


def _text_of(elem: ET.Element | None) -> str:
    """Flatten an element to whitespace-normalised text."""
    if elem is None:
        return ""
    text = " ".join(t for t in elem.itertext())
    return re.sub(r"\s+", " ", text).strip()


def _find_text(root: ET.Element, path: str) -> str:
    return _text_of(root.find(path))


def _article_id(root: ET.Element, id_type: str) -> str:
    for node in root.iter():
        if node.tag.split("}")[-1] == "article-id" and node.get("pub-id-type") == id_type:
            return (node.text or "").strip()
    return ""


def _is_open_access(root: ET.Element) -> bool:
    """
    Only accept articles we can positively confirm are open access.

    We already restrict the search with `"open access"[filter]`, but this is a
    second check on the article itself. Defaulting to *reject* on ambiguity is
    deliberate: the cost of skipping a usable article is nothing, while the
    cost of ingesting a restricted one is a licensing problem.
    """
    for node in root.iter():
        tag = node.tag.split("}")[-1]
        if tag == "license":
            lic = (node.get("license-type") or "").lower()
            href = " ".join(v for k, v in node.attrib.items() if "href" in k).lower()
            blob = f"{lic} {href} {_text_of(node)}".lower()
            if "open" in blob or "creativecommons.org" in blob or "cc-by" in blob:
                return True
    return False


def parse_pmc_article(root: ET.Element) -> dict | None:
    """
    Turn one JATS <article> element into an ingestible record.

    Returns None if the article isn't confirmably open access, or has no
    usable body text.
    """
    if not _is_open_access(root):
        return None

    title = _find_text(root, ".//article-meta//article-title")
    if not title:
        return None

    pmcid = _article_id(root, "pmc")
    pmid = _article_id(root, "pmid")
    doi = _article_id(root, "doi")
    journal = _find_text(root, ".//journal-meta//journal-title")
    year = _find_text(root, ".//article-meta//pub-date/year")

    abstract_el = root.find(".//article-meta//abstract")
    body_el = root.find(".//body")

    # Strip noise before flattening — order matters.
    for el in (abstract_el, body_el):
        if el is not None:
            _strip_noise(el)

    abstract = _text_of(abstract_el)
    body = _text_of(body_el)

    content_parts = [title]
    if journal:
        content_parts.append(f"Journal: {journal}" + (f" ({year})" if year else ""))
    if abstract:
        content_parts.append(f"Abstract: {abstract}")
    if body:
        content_parts.append(body)

    content = "\n\n".join(p for p in content_parts if p)

    if len(content) < _MIN_ARTICLE_CHARS:
        return None
    if len(content) > _MAX_ARTICLE_CHARS:
        content = content[:_MAX_ARTICLE_CHARS] + "\n\n[truncated]"

    url = (
        f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid.lstrip('PMC')}/"
        if pmcid
        else (f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None)
    )

    return {
        "title": title,
        "content": content,
        "pmcid": pmcid,
        "pmid": pmid,
        "doi": doi,
        "journal": journal,
        "year": year,
        "url": url,
    }


# ══════════════════════════════════════════════════════════════
# FETCH + INGEST
# ══════════════════════════════════════════════════════════════


async def fetch_and_ingest_pmc(
    db: AsyncSession,
    topics: list[str] | None = None,
    max_per_topic: int = 10,
) -> dict:
    """
    Search PMC Open Access for each topic, then ingest the full text.

    Idempotent: articles already present (matched by title + source_type) are
    skipped, so re-running tops the corpus up rather than duplicating it.
    """
    results = {
        "topics_searched": 0,
        "found": 0,
        "ingested": 0,
        "skipped_existing": 0,
        "skipped_not_open_access": 0,
        "failed": 0,
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

    Entrez.email = settings.NCBI_EMAIL
    if settings.NCBI_API_KEY and not settings.NCBI_API_KEY.startswith("your_"):
        Entrez.api_key = settings.NCBI_API_KEY

    topics = topics or DEFAULT_TOPICS

    for topic in topics:
        results["topics_searched"] += 1
        try:
            # The `"open access"[filter]` clause is what keeps us inside the
            # subset that permits bulk download. Do not remove it.
            query = f'({topic}) AND "open access"[filter]'

            handle = await asyncio.to_thread(
                Entrez.esearch, db="pmc", term=query,
                retmax=max_per_topic, sort="relevance",
            )
            search = Entrez.read(handle)
            handle.close()
            await asyncio.sleep(_REQUEST_DELAY)

            ids = search.get("IdList", [])
            if not ids:
                continue

            handle = await asyncio.to_thread(
                Entrez.efetch, db="pmc", id=",".join(ids), retmode="xml",
            )
            raw = handle.read()
            handle.close()
            await asyncio.sleep(_REQUEST_DELAY)

            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")

            try:
                tree = ET.fromstring(raw)
            except ET.ParseError as e:
                results["errors"].append(f"XML parse failed for '{topic}': {e}")
                continue

            articles = [n for n in tree.iter() if n.tag.split("}")[-1] == "article"]

            for node in articles:
                record = parse_pmc_article(node)
                if record is None:
                    results["skipped_not_open_access"] += 1
                    continue

                results["found"] += 1

                existing = await db.execute(
                    select(Document).where(
                        Document.title == record["title"],
                        Document.source_type == "pmc_open_access",
                    )
                )
                if existing.scalar_one_or_none():
                    results["skipped_existing"] += 1
                    continue

                ident = record["pmcid"] or record["pmid"] or "unknown"
                citation_bits = [b for b in (
                    record["journal"], record["year"],
                    f"PMID: {record['pmid']}" if record["pmid"] else "",
                    f"DOI: {record['doi']}" if record["doi"] else "",
                ) if b]

                doc = Document(
                    filename=f"pmc_{ident}.txt",
                    file_type="txt",
                    file_size=len(record["content"].encode("utf-8")),
                    title=record["title"],
                    source_type="pmc_open_access",
                    source_url=record["url"],
                    description=" · ".join(citation_bits),
                    status="processing",
                )
                db.add(doc)
                await db.commit()
                await db.refresh(doc)

                outcome = await ingest_text_content(
                    text=record["content"],
                    title=record["title"],
                    document_id=str(doc.id),
                    source_type="pmc_open_access",
                    source_url=record["url"],
                )

                doc.status = outcome["status"]
                doc.chunk_count = outcome["chunk_count"]
                if outcome["status"] == "failed":
                    doc.error_message = outcome["message"]
                    results["failed"] += 1
                else:
                    results["ingested"] += 1
                await db.commit()

        except Exception as e:  # noqa: BLE001 — one bad topic shouldn't stop the rest
            logger.exception("PMC fetch failed for topic %r", topic)
            results["errors"].append(f"Topic '{topic}' failed: {e}")
            continue

    return results
