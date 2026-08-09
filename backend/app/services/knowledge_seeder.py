"""
RadAssist AI — Knowledge Base Seeder Service

WHAT THIS FILE DOES:
Populates the knowledge base with curated medical content and articles
fetched from trusted sources (NCBI/StatPearls). This runs automatically
on first startup or can be triggered manually via API.

TWO DATA SOURCES:

1. CURATED SEED DATA (app/data/seed_knowledge.py):
   Hand-written, radiologist-level content covering core topics.
   Based on StatPearls, RadioPaedia, ACR guidelines, and standard
   radiology textbooks. Loaded immediately — no internet required.

2. NCBI/StatPearls FETCHER (requires internet + optional API key):
   Fetches real peer-reviewed articles from PubMed/StatPearls via
   the NCBI E-utilities API. These are the same articles doctors
   and medical students use worldwide.

WHY BOTH?
- Curated data guarantees a working knowledge base even offline
- NCBI fetching adds breadth and keeps content current
- Together, they provide comprehensive coverage
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.document import Document
from app.services.ingestion import ingest_text_content
from app.data.seed_knowledge import SEED_KNOWLEDGE

settings = get_settings()

# Placeholder values that people leave in .env by accident. Sending these to
# NCBI would violate their terms of use and can get the IP throttled/blocked,
# so we treat them as "not configured" rather than trusting the raw string.
_PLACEHOLDER_EMAILS = {
    "your_email@example.com",
    "your@email.com",
    "radassist@example.com",
    "user@example.com",
    "email@example.com",
    "changeme",
}


def ncbi_is_configured() -> bool:
    """
    True only if NCBI_EMAIL looks like a real, usable address.

    WHY THE PLACEHOLDER CHECK?
    The original gate was `if settings.NCBI_EMAIL:` — but the shipped .env
    contained `your_email@example.com`, which is truthy. That meant the
    "skip if not configured" branch never fired and we'd hit a public API
    with a fake identity.
    """
    email = (settings.NCBI_EMAIL or "").strip().lower()
    if not email or email in _PLACEHOLDER_EMAILS:
        return False
    # Minimal sanity check — NCBI requires a contactable address.
    return "@" in email and "." in email.split("@")[-1]


async def seed_curated_knowledge(db: AsyncSession) -> dict:
    """
    Ingest all curated radiology knowledge into the knowledge base.
    
    This is the PRIMARY seeding mechanism. It loads the hand-written
    medical content from seed_knowledge.py through the full ingestion
    pipeline (chunk → embed → store in Qdrant).
    
    IDEMPOTENT: Checks if each article already exists by title.
    Running this multiple times will NOT create duplicates.
    
    Args:
        db: Database session for PostgreSQL operations
        
    Returns:
        Summary dict with counts of ingested, skipped, and failed articles
    """
    results = {
        "total": len(SEED_KNOWLEDGE),
        "ingested": 0,
        "skipped": 0,
        "failed": 0,
        "details": [],
    }
    
    for entry in SEED_KNOWLEDGE:
        title = entry["title"]
        
        # ── Check if already ingested (skip duplicates) ─────
        existing = await db.execute(
            select(Document).where(
                Document.title == title,
                Document.source_type == entry.get("source_type", "curated"),
            )
        )
        if existing.scalar_one_or_none():
            results["skipped"] += 1
            results["details"].append(f"⏭️  Skipped (exists): {title}")
            continue
        
        # ── Create document record in PostgreSQL ────────────
        doc = Document(
            filename=f"{title.lower().replace(' ', '_')[:80]}.txt",
            file_type="txt",
            file_size=len(entry["content"].encode("utf-8")),
            title=title,
            source_type=entry.get("source_type", "curated"),
            description=entry.get("source_attribution", "Curated radiology knowledge"),
            status="processing",
        )
        db.add(doc)
        await db.commit()
        await db.refresh(doc)
        
        # ── Run ingestion pipeline (chunk → embed → store) ──
        result = await ingest_text_content(
            text=entry["content"],
            title=title,
            document_id=str(doc.id),
            source_type=entry.get("source_type", "curated"),
        )
        
        # ── Update document status ──────────────────────────
        doc.status = result["status"]
        doc.chunk_count = result["chunk_count"]
        if result["status"] == "failed":
            doc.error_message = result["message"]
            results["failed"] += 1
            results["details"].append(f"❌ Failed: {title} — {result['message']}")
        else:
            results["ingested"] += 1
            results["details"].append(
                f"✅ Ingested: {title} ({result['chunk_count']} chunks)"
            )
        
        doc.updated_at = datetime.now(timezone.utc)
        await db.commit()
    
    return results


async def fetch_and_ingest_statpearls(
    db: AsyncSession,
    search_terms: list[str] | None = None,
    max_articles: int = 10,
) -> dict:
    """
    Fetch radiology articles from NCBI/StatPearls and ingest them.
    
    Uses the NCBI E-utilities API to search PubMed for StatPearls
    articles related to radiology, then ingests their abstracts
    into the knowledge base.
    
    NCBI E-utilities is a FREE, public API provided by the National
    Library of Medicine (NLM). StatPearls articles on PubMed are
    peer-reviewed, continuously updated, and used by clinicians worldwide.
    
    Args:
        db: Database session
        search_terms: Custom search terms. Defaults to radiology-focused queries.
        max_articles: Max number of articles to fetch per search term.
        
    Returns:
        Summary dict with counts
    """
    # Default search terms covering key radiology topics
    if search_terms is None:
        search_terms = [
            "radiology imaging diagnosis",
            "chest radiograph interpretation",
            "CT scan findings emergency",
            "MRI brain neuroimaging",
            "musculoskeletal radiology fracture",
            "abdominal imaging acute abdomen",
            "pulmonary embolism CT angiography",
            "stroke neuroimaging",
        ]
    
    results = {
        "total_fetched": 0,
        "ingested": 0,
        "skipped": 0,
        "failed": 0,
        "errors": [],
    }
    
    try:
        from Bio import Entrez, Medline

        # NCBI requires a real contact address — refuse rather than fake one.
        if not ncbi_is_configured():
            results["errors"].append(
                "NCBI_EMAIL is not set to a real address. NCBI's terms of use "
                "require a contactable email. Set NCBI_EMAIL in .env."
            )
            return results

        Entrez.email = settings.NCBI_EMAIL
        if settings.NCBI_API_KEY and not settings.NCBI_API_KEY.startswith("your_"):
            Entrez.api_key = settings.NCBI_API_KEY
        
        for term in search_terms:
            try:
                # Search for StatPearls articles matching the term
                search_query = f'StatPearls[Book] AND "{term}"[All Fields]'
                
                handle = Entrez.esearch(
                    db="pubmed",
                    term=search_query,
                    retmax=max_articles,
                    sort="relevance",
                )
                search_results = Entrez.read(handle)
                handle.close()
                
                id_list = search_results.get("IdList", [])
                if not id_list:
                    continue
                
                # Fetch article details
                handle = Entrez.efetch(
                    db="pubmed",
                    id=",".join(id_list),
                    rettype="medline",
                    retmode="text",
                )
                records = list(Medline.parse(handle))
                handle.close()
                
                for record in records:
                    title = record.get("TI", "Untitled")
                    abstract = record.get("AB", "")
                    pmid = record.get("PMID", "")
                    authors = ", ".join(record.get("AU", [])[:3])
                    
                    if not abstract or len(abstract) < 100:
                        continue
                    
                    results["total_fetched"] += 1
                    
                    # Check if already ingested
                    existing = await db.execute(
                        select(Document).where(
                            Document.title == title,
                            Document.source_type == "statpearls",
                        )
                    )
                    if existing.scalar_one_or_none():
                        results["skipped"] += 1
                        continue
                    
                    # Build full content with metadata
                    full_content = f"""
{title}

Authors: {authors}
Source: StatPearls / PubMed (PMID: {pmid})
Type: Peer-reviewed medical reference article

{abstract}
"""
                    
                    # Create document record
                    doc = Document(
                        filename=f"statpearls_{pmid}.txt",
                        file_type="txt",
                        file_size=len(full_content.encode("utf-8")),
                        title=title,
                        source_type="statpearls",
                        source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        description=f"StatPearls article by {authors}. PMID: {pmid}",
                        status="processing",
                    )
                    db.add(doc)
                    await db.commit()
                    await db.refresh(doc)
                    
                    # Ingest
                    result = await ingest_text_content(
                        text=full_content,
                        title=title,
                        document_id=str(doc.id),
                        source_type="statpearls",
                        source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    )
                    
                    doc.status = result["status"]
                    doc.chunk_count = result["chunk_count"]
                    if result["status"] == "failed":
                        doc.error_message = result["message"]
                        results["failed"] += 1
                    else:
                        results["ingested"] += 1
                    
                    doc.updated_at = datetime.now(timezone.utc)
                    await db.commit()
                    
                    # Rate limiting — respect NCBI's terms
                    import asyncio
                    await asyncio.sleep(0.5)
                    
            except Exception as e:
                results["errors"].append(f"Search '{term}' failed: {str(e)}")
                continue
                
    except ImportError:
        results["errors"].append(
            "Biopython not installed. Run: pip install biopython"
        )
    except Exception as e:
        results["errors"].append(f"NCBI fetch failed: {str(e)}")
    
    return results


async def seed_knowledge_base(db: AsyncSession) -> dict:
    """
    Full knowledge base seeding — curated data + NCBI (if configured).
    
    This is the main entry point called from the seed API endpoint
    or during initial setup.
    
    Returns:
        Combined results from both seeding sources
    """
    print("=" * 50)
    print("🌱 Seeding RadAssist Knowledge Base...")
    print("=" * 50)
    
    # ── Step 1: Curated Knowledge (always runs) ─────────────
    print("\n📚 Loading curated radiology knowledge...")
    curated_results = await seed_curated_knowledge(db)
    print(f"   Ingested: {curated_results['ingested']}")
    print(f"   Skipped:  {curated_results['skipped']}")
    print(f"   Failed:   {curated_results['failed']}")
    
    # ── Step 2: NCBI/StatPearls (if configured) ─────────────
    ncbi_results = {"total_fetched": 0, "ingested": 0, "skipped": 0, "failed": 0, "errors": []}
    
    if ncbi_is_configured():
        print("\n🔬 Fetching StatPearls articles from NCBI...")
        ncbi_results = await fetch_and_ingest_statpearls(db, max_articles=5)
        print(f"   Fetched:  {ncbi_results['total_fetched']}")
        print(f"   Ingested: {ncbi_results['ingested']}")
        print(f"   Skipped:  {ncbi_results['skipped']}")
        if ncbi_results["errors"]:
            for err in ncbi_results["errors"]:
                print(f"   ⚠️  {err}")
    else:
        print("\n⏭️  Skipping NCBI fetch (NCBI_EMAIL not set to a real address)")
        print("   Set a real NCBI_EMAIL in .env to enable StatPearls fetching")
    
    print("\n" + "=" * 50)
    print("✅ Knowledge base seeding complete!")
    print("=" * 50)
    
    return {
        "curated": curated_results,
        "ncbi": ncbi_results,
    }
