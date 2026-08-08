"""
RadAssist AI — Document Ingestion Service

THIS IS THE HEART OF PHASE 2.

When a doctor uploads a file or we fetch a medical article, this service
processes it through a 4-stage pipeline:

    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │  PARSE  │ →  │  CHUNK  │ →  │  EMBED  │ →  │  STORE  │
    │         │    │         │    │         │    │         │
    │ Extract │    │ Split   │    │ Convert │    │ Save to │
    │ text    │    │ into    │    │ to      │    │ Qdrant  │
    │ from    │    │ small   │    │ vectors │    │ + update│
    │ PDF/    │    │ pieces  │    │ (384    │    │ Postgres│
    │ DOCX/   │    │ with    │    │ numbers │    │ status  │
    │ IMG/    │    │ overlap │    │ each)   │    │         │
    │ TXT     │    │         │    │         │    │         │
    └─────────┘    └─────────┘    └─────────┘    └─────────┘

STAGE 1 — PARSE:
  Extracts raw text from any supported file format:
  - PDF: PyMuPDF (fitz) — C-based, very fast
  - DOCX: python-docx — reads Word documents
  - TXT/MD: Plain file read
  - Images: Pillow + pytesseract (OCR) — reads text from photos/scans

STAGE 2 — CHUNK:
  Splits the extracted text into small, overlapping pieces.
  WHY? Embedding models have a token limit (~256 tokens for MiniLM).
  Also, smaller chunks give more precise search results.

  Example (chunk_size=512, overlap=50):
  ┌────────────────────── Chunk 0 ──────────────────────┐
  │ "Pneumonia is an infection that inflames the air     │
  │  sacs in one or both lungs. The air sacs may fill   │
  │  with fluid or pus (purulent material), causing..." │
  └───────────────────────────────┬──────────────────────┘
                                  │ overlap (50 chars)
                    ┌─────────────┴────────────────────────┐
                    │ "...causing cough with phlegm or pus, │
                    │  fever, chills, and difficulty         │
                    │  breathing. On chest X-ray, pneumonia  │
                    │  appears as..."                        │
                    └──────────────────────────────────────┘
                                Chunk 1

STAGE 3 — EMBED:
  Converts each chunk into a 384-dimensional vector using the
  embedding model. Chunks about similar topics get similar vectors.

STAGE 4 — STORE:
  Saves vectors + text to Qdrant, updates PostgreSQL document status.
"""

import os
import io
import fitz  # PyMuPDF — imported as 'fitz' (historical name)
from docx import Document as DocxDocument
from PIL import Image

from app.config import get_settings
from app.services.embedding import embedding_service
from app.services.qdrant_service import qdrant_service

settings = get_settings()


# ══════════════════════════════════════════════════════════════
# STAGE 1 — DOCUMENT PARSING
# ══════════════════════════════════════════════════════════════
# Each parser extracts raw text from a specific file format.
# They all return a plain string — the rest of the pipeline
# doesn't care what format the original file was.
# ══════════════════════════════════════════════════════════════


def parse_pdf(file_bytes: bytes) -> str:
    """
    Extract text from a PDF file using PyMuPDF.
    
    WHY PyMuPDF?
    - Built on the MuPDF C engine → 10x faster than pure Python alternatives
    - Handles digital PDFs perfectly (most medical guidelines/textbooks)
    - Preserves reading order even in multi-column layouts
    
    HOW IT WORKS:
    1. Opens the PDF from bytes (no need to save to disk)
    2. Iterates through every page
    3. Extracts text from each page, preserving paragraph structure
    4. Joins all pages with newlines
    
    Args:
        file_bytes: Raw bytes of the PDF file
        
    Returns:
        Extracted text as a single string
    """
    text_parts = []
    
    # fitz.open() can read from bytes via a stream — no temp file needed
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc):
            # get_text("text") extracts plain text in reading order
            # "blocks" mode would give us more structure, but plain
            # text is sufficient for chunking and embedding
            page_text = page.get_text("text")
            if page_text.strip():
                text_parts.append(page_text)
    
    return "\n\n".join(text_parts)


def parse_docx(file_bytes: bytes) -> str:
    """
    Extract text from a DOCX (Word) file.
    
    HOW IT WORKS:
    1. Opens the DOCX from bytes (it's actually a ZIP file internally)
    2. Iterates through all paragraphs
    3. Joins paragraphs with newlines, skipping empty ones
    
    DOCX STRUCTURE:
    A Word document is a ZIP file containing XML. python-docx
    handles all the XML parsing for us. We just iterate paragraphs.
    
    Args:
        file_bytes: Raw bytes of the DOCX file
        
    Returns:
        Extracted text as a single string
    """
    doc = DocxDocument(io.BytesIO(file_bytes))
    
    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)
    
    return "\n\n".join(paragraphs)


def parse_text(file_bytes: bytes) -> str:
    """
    Extract text from a plain text or Markdown file.
    
    The simplest parser — just decode bytes to string.
    Handles both UTF-8 and Latin-1 encoding (covers 99% of text files).
    
    Args:
        file_bytes: Raw bytes of the text file
        
    Returns:
        The decoded text string
    """
    # Try UTF-8 first (most common), fall back to Latin-1 (never fails)
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1")


def parse_image(file_bytes: bytes) -> str:
    """
    Extract text from an image using OCR (Optical Character Recognition).
    
    WHEN IS THIS USED?
    - Scanned medical documents (old paper records digitized as images)
    - Photos of handwritten notes
    - Screenshots of reports from other systems
    - Reference images with text annotations
    
    HOW IT WORKS:
    1. Opens the image with Pillow (PIL)
    2. Passes it to Tesseract OCR engine
    3. Tesseract recognizes text patterns and returns a string
    
    REQUIREMENTS:
    - Tesseract must be installed in the Docker container
    - For best results, images should be clear and well-lit
    - Handwriting recognition is basic — works best with printed text
    
    Args:
        file_bytes: Raw bytes of the image file (PNG, JPG, TIFF, etc.)
        
    Returns:
        Extracted text from the image
    """
    try:
        import pytesseract
        
        image = Image.open(io.BytesIO(file_bytes))
        
        # Convert to RGB if necessary (some formats like PNG have alpha channel)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Run OCR — pytesseract sends the image to the Tesseract engine
        # and returns the recognized text
        text = pytesseract.image_to_string(image)
        return text.strip()
        
    except ImportError:
        print("⚠️  pytesseract not available — OCR disabled")
        return "[OCR not available — install tesseract to extract text from images]"
    except Exception as e:
        print(f"⚠️  OCR failed: {e}")
        return f"[OCR failed: {str(e)}]"


def parse_file(file_bytes: bytes, file_type: str) -> str:
    """
    Route a file to the correct parser based on its type.
    
    This is the ENTRY POINT for Stage 1. The rest of the pipeline
    calls this function and gets back plain text regardless of format.
    
    Args:
        file_bytes: Raw bytes of the uploaded file
        file_type: File extension without dot (e.g., "pdf", "docx", "png")
        
    Returns:
        Extracted text as a string
        
    Raises:
        ValueError: If the file type is not supported
    """
    parsers = {
        "pdf": parse_pdf,
        "docx": parse_docx,
        "txt": parse_text,
        "md": parse_text,
        # Image formats — all go through OCR
        "png": parse_image,
        "jpg": parse_image,
        "jpeg": parse_image,
        "tiff": parse_image,
        "bmp": parse_image,
    }
    
    parser = parsers.get(file_type.lower())
    if not parser:
        raise ValueError(
            f"Unsupported file type: '{file_type}'. "
            f"Supported: {', '.join(parsers.keys())}"
        )
    
    return parser(file_bytes)


# ══════════════════════════════════════════════════════════════
# STAGE 2 — TEXT CHUNKING
# ══════════════════════════════════════════════════════════════
# Splits extracted text into smaller pieces optimized for embedding
# and retrieval. This is implemented from scratch (no LangChain
# dependency) using a recursive splitting strategy.
# ══════════════════════════════════════════════════════════════


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """
    Split text into overlapping chunks using recursive character splitting.
    
    STRATEGY (Recursive Splitting):
    We try to split at the BEST boundary first, falling back to worse ones:
    1. Double newline (paragraph break) — BEST, preserves complete paragraphs
    2. Single newline (line break) — good, preserves sentences
    3. Period + space (sentence end) — acceptable, keeps sentence integrity
    4. Space (word boundary) — fallback, at least doesn't break words
    5. Any character — last resort, only for very long words (rare)
    
    WHY THIS ORDER?
    Medical text has meaning at the paragraph level. A paragraph about
    "pneumonia findings" should ideally stay in ONE chunk. If we split
    at character boundaries, we might cut "pneu|monia" in half.
    
    Args:
        text: The full extracted text to split
        chunk_size: Max characters per chunk (default from config: 512)
        chunk_overlap: Overlap between chunks (default from config: 50)
        
    Returns:
        List of text chunks, each at most chunk_size characters
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
    
    # Clean the text — normalize whitespace, remove excessive blank lines
    text = text.strip()
    if not text:
        return []
    
    # If the entire text fits in one chunk, just return it
    if len(text) <= chunk_size:
        return [text]
    
    # Separators in order of preference (best → worst)
    separators = ["\n\n", "\n", ". ", " ", ""]
    
    chunks = []
    _recursive_split(text, separators, chunk_size, chunk_overlap, chunks)
    
    # Filter out empty/whitespace-only chunks
    chunks = [c.strip() for c in chunks if c.strip()]
    
    return chunks


def _recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
    result: list[str],
) -> None:
    """
    Internal recursive function that does the actual splitting.
    
    Algorithm:
    1. Take the first (best) separator
    2. Split the text by that separator
    3. Merge pieces back together up to chunk_size
    4. If any piece is still too big, recurse with the next separator
    """
    if not text:
        return
    
    # Base case: text fits in a chunk
    if len(text) <= chunk_size:
        result.append(text)
        return
    
    # Try each separator (best first)
    separator = separators[0] if separators else ""
    remaining_separators = separators[1:] if len(separators) > 1 else [""]
    
    if separator:
        pieces = text.split(separator)
    else:
        # Last resort: split by character (for very long words)
        pieces = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]
        result.extend(pieces)
        return
    
    # Merge pieces back together, respecting chunk_size
    current_chunk = ""
    
    for piece in pieces:
        # If adding this piece would exceed chunk_size
        test_chunk = current_chunk + separator + piece if current_chunk else piece
        
        if len(test_chunk) <= chunk_size:
            current_chunk = test_chunk
        else:
            # Save the current chunk if it has content
            if current_chunk:
                result.append(current_chunk)
                
                # Start new chunk with overlap from previous chunk
                if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                    # Take the last 'overlap' characters as the start of next chunk
                    overlap_text = current_chunk[-chunk_overlap:]
                    current_chunk = overlap_text + separator + piece
                else:
                    current_chunk = piece
            else:
                current_chunk = piece
            
            # If the current chunk is STILL too big, recurse with next separator
            if len(current_chunk) > chunk_size:
                _recursive_split(current_chunk, remaining_separators, chunk_size, chunk_overlap, result)
                current_chunk = ""
    
    # Don't forget the last chunk
    if current_chunk:
        if len(current_chunk) > chunk_size:
            _recursive_split(current_chunk, remaining_separators, chunk_size, chunk_overlap, result)
        else:
            result.append(current_chunk)


# ══════════════════════════════════════════════════════════════
# STAGE 3+4 — THE FULL INGESTION PIPELINE
# ══════════════════════════════════════════════════════════════
# Combines parsing, chunking, embedding, and storage into one
# function that takes a file in and puts knowledge out.
# ══════════════════════════════════════════════════════════════


async def ingest_document(
    file_bytes: bytes,
    filename: str,
    file_type: str,
    document_id: str,
    source_type: str = "general",
    title: str | None = None,
) -> dict:
    """
    THE MAIN PIPELINE — Process a document from raw bytes to stored vectors.
    
    This is called by the upload API endpoint. It runs the full 4-stage
    pipeline and returns a summary of what was done.
    
    Args:
        file_bytes: Raw bytes of the uploaded file
        filename: Original filename (for metadata)
        file_type: File extension without dot ("pdf", "png", etc.)
        document_id: UUID from PostgreSQL (links vectors back to metadata)
        source_type: Category ("textbook", "guideline", "statpearls", etc.)
        title: Optional document title
        
    Returns:
        Dict with processing results:
        {
            "status": "completed",
            "chunk_count": 45,
            "char_count": 23456,
            "message": "Successfully processed 45 chunks"
        }
    """
    try:
        # ── STAGE 1: Parse ──────────────────────────────────
        print(f"📄 Parsing {filename} ({file_type})...")
        raw_text = parse_file(file_bytes, file_type)
        
        if not raw_text or len(raw_text.strip()) < 10:
            return {
                "status": "failed",
                "chunk_count": 0,
                "char_count": 0,
                "message": "No meaningful text could be extracted from the file."
            }
        
        print(f"   → Extracted {len(raw_text)} characters")
        
        # ── STAGE 2: Chunk ──────────────────────────────────
        print(f"✂️  Chunking text (size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP})...")
        chunks = chunk_text(raw_text)
        
        if not chunks:
            return {
                "status": "failed",
                "chunk_count": 0,
                "char_count": len(raw_text),
                "message": "Text was extracted but no valid chunks were produced."
            }
        
        print(f"   → Created {len(chunks)} chunks")
        
        # ── STAGE 3: Embed ──────────────────────────────────
        print(f"🧠 Embedding {len(chunks)} chunks...")
        vectors = embedding_service.encode(
            chunks,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            show_progress=len(chunks) > 50,  # Show progress bar for large docs
        )
        print(f"   → Generated {len(vectors)} vectors ({settings.EMBEDDING_DIMENSION}D each)")
        
        # ── STAGE 4: Store in Qdrant ────────────────────────
        print(f"💾 Storing in Qdrant collection '{settings.QDRANT_COLLECTION}'...")
        
        # Metadata attached to each vector — used for filtering and display
        metadata = {
            "filename": filename,
            "source_type": source_type,
        }
        if title:
            metadata["title"] = title
        
        # Delete any existing chunks for this document (re-ingestion safe)
        qdrant_service.delete_by_document(document_id)
        
        # Insert new chunks
        upserted = qdrant_service.upsert_chunks(
            document_id=document_id,
            chunks=chunks,
            vectors=vectors,
            metadata=metadata,
        )
        
        print(f"   → Stored {upserted} vectors in Qdrant")
        print(f"✅ Successfully ingested: {filename}")
        
        return {
            "status": "completed",
            "chunk_count": len(chunks),
            "char_count": len(raw_text),
            "message": f"Successfully processed {len(chunks)} chunks from {filename}"
        }
        
    except Exception as e:
        error_msg = f"Ingestion failed for {filename}: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "status": "failed",
            "chunk_count": 0,
            "char_count": 0,
            "message": error_msg,
        }


async def ingest_text_content(
    text: str,
    title: str,
    document_id: str,
    source_type: str = "general",
    source_url: str | None = None,
) -> dict:
    """
    Ingest plain text content directly (not from a file).
    
    WHEN IS THIS USED?
    - Ingesting StatPearls articles fetched from NCBI API
    - Ingesting curated medical knowledge (seed data)
    - Ingesting text pasted by a doctor (no file upload needed)
    
    Skips Stage 1 (parsing) since we already have raw text.
    
    Args:
        text: The raw text content to ingest
        title: Title for the content
        document_id: UUID from PostgreSQL
        source_type: Category ("statpearls", "curated", etc.)
        source_url: Original URL if fetched from the internet
        
    Returns:
        Dict with processing results (same format as ingest_document)
    """
    try:
        if not text or len(text.strip()) < 10:
            return {
                "status": "failed",
                "chunk_count": 0,
                "char_count": 0,
                "message": "Text content is too short or empty."
            }
        
        # ── Chunk ──
        chunks = chunk_text(text)
        if not chunks:
            return {
                "status": "failed",
                "chunk_count": 0,
                "char_count": len(text),
                "message": "No valid chunks produced from the text."
            }
        
        # ── Embed ──
        vectors = embedding_service.encode(
            chunks,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
        )
        
        # ── Store ──
        metadata = {
            "filename": title,
            "source_type": source_type,
            "title": title,
        }
        if source_url:
            metadata["source_url"] = source_url
        
        qdrant_service.delete_by_document(document_id)
        upserted = qdrant_service.upsert_chunks(
            document_id=document_id,
            chunks=chunks,
            vectors=vectors,
            metadata=metadata,
        )
        
        return {
            "status": "completed",
            "chunk_count": len(chunks),
            "char_count": len(text),
            "message": f"Successfully processed {len(chunks)} chunks from '{title}'"
        }
        
    except Exception as e:
        return {
            "status": "failed",
            "chunk_count": 0,
            "char_count": 0,
            "message": f"Ingestion failed: {str(e)}",
        }
