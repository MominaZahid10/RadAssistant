"""
RadAssist AI — Qdrant Vector Store Service

WHAT THIS FILE DOES:
Manages all interactions with Qdrant — our vector database.
Qdrant stores the actual text chunks and their embedding vectors,
enabling semantic search (finding content by MEANING, not keywords).

HOW QDRANT WORKS (Simple Analogy):
Think of a regular database as a filing cabinet — you find things
by label ("give me the folder labeled 'pneumonia'").

Qdrant is more like a librarian who UNDERSTANDS content:
"Find me everything RELATED to lung infections in chest X-rays"
Even if the stored text never uses the exact phrase "lung infections,"
Qdrant finds relevant content because it compares MEANINGS (vectors).

KEY CONCEPTS:
- Collection: Like a table in PostgreSQL — holds all our vectors
- Point: A single entry = vector + payload (the text + metadata)
- Payload: Extra data attached to each vector (source info, text, etc.)
- HNSW Index: The algorithm Qdrant uses for fast nearest-neighbor search
  (finds the closest vectors without comparing every single one)

COSINE SIMILARITY:
We use cosine distance to compare vectors. It measures the angle
between two vectors:
- 1.0 = identical direction (same meaning)
- 0.0 = perpendicular (unrelated)
- -1.0 = opposite (but rare in practice)
"""

import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import get_settings

settings = get_settings()

# How many points to pull per scroll round-trip when reading a whole document.
_SCROLL_BATCH_SIZE = 256

# Guard against pathological documents when previewing chunks.
_MAX_CHUNKS_PER_DOCUMENT = 20_000


class QdrantService:
    """
    Manages the Qdrant vector store for the RadAssist knowledge base.
    
    RESPONSIBILITIES:
    1. Create/verify the collection at startup
    2. Insert document chunks (after embedding)
    3. Search for relevant chunks (semantic search)
    4. Delete chunks when a document is removed
    5. Provide collection statistics
    """

    def __init__(self):
        """
        Create a connection to Qdrant.
        
        Qdrant runs as a separate Docker container (see docker-compose.yml).
        We connect to it over HTTP using hostname "qdrant" (Docker networking).
        """
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            timeout=30,  # 30 seconds — generous for large batch operations
        )
        self.collection_name = settings.QDRANT_COLLECTION
        self.dimension = settings.EMBEDDING_DIMENSION

    def ensure_collection(self) -> None:
        """
        Create the vector collection if it doesn't already exist.
        
        Called at app startup. This is IDEMPOTENT — calling it multiple
        times is safe. If the collection already exists, this does nothing.
        
        WHAT'S CONFIGURED:
        - Vector size: 384 (matches all-MiniLM-L6-v2 output)
        - Distance: COSINE (angle-based similarity, standard for text)
        - HNSW: Default settings (m=16, ef_construct=100)
          These control the speed/accuracy tradeoff of the search index.
          Defaults are fine for our scale (<1M vectors).
        """
        try:
            # Check if collection already exists
            collections = self.client.get_collections().collections
            existing_names = [c.name for c in collections]

            if self.collection_name in existing_names:
                # Verify the dimension matches (catches config mismatches)
                info = self.client.get_collection(self.collection_name)
                existing_dim = info.config.params.vectors.size
                if existing_dim != self.dimension:
                    print(
                        f"⚠️  Collection '{self.collection_name}' exists with "
                        f"dimension {existing_dim}, but config says {self.dimension}. "
                        f"If you changed models, delete the collection and restart."
                    )
                else:
                    print(f"✅ Qdrant collection '{self.collection_name}' verified ({existing_dim}D)")
                self._ensure_payload_indexes()
                return

            # Create new collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=self.dimension,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
            print(f"✅ Created Qdrant collection: '{self.collection_name}' ({self.dimension}D, COSINE)")
            self._ensure_payload_indexes()

        except Exception as e:
            print(f"❌ Failed to initialize Qdrant collection: {e}")
            raise

    def _ensure_payload_indexes(self) -> None:
        """
        Create payload indexes on the fields we filter by.

        WHY THIS MATTERS:
        Qdrant's HNSW index makes *vector* search fast, but filtering on a
        payload field without an index forces a full scan of the collection.
        We filter on `document_id` (every delete and every chunk preview) and
        `source_type` (filtered search), so both need indexes.

        At 1,000 vectors you won't notice. At 500,000 — a realistic size once
        real textbooks are ingested — an unindexed delete goes from
        milliseconds to seconds.

        Idempotent: re-creating an existing index is a no-op on the server,
        and any error here is non-fatal (search still works, just slower).
        """
        for field in ("document_id", "source_type"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
                )
                print(f"   ↳ payload index ensured on '{field}'")
            except Exception as e:
                # Already exists, or the server is an older version — not fatal.
                print(f"   ↳ payload index on '{field}' skipped ({type(e).__name__})")

    def upsert_chunks(
        self,
        document_id: str,
        chunks: list[str],
        vectors: list[list[float]],
        metadata: dict | None = None,
    ) -> int:
        """
        Store document chunks and their vectors in Qdrant.
        
        "Upsert" = Update if exists, Insert if new. This is safer than
        plain insert because re-processing a document won't create duplicates.
        
        Each chunk becomes a "point" in Qdrant with:
        - id: UUID (unique per chunk)
        - vector: The 384-float embedding
        - payload: The text content + metadata for retrieval
        
        Args:
            document_id: The PostgreSQL document UUID (links back to metadata)
            chunks: List of text strings (the actual content)
            vectors: List of embedding vectors (one per chunk)
            metadata: Extra info to attach (filename, source_type, etc.)
            
        Returns:
            Number of points successfully upserted.
        """
        if not chunks or not vectors:
            return 0

        if len(chunks) != len(vectors):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(vectors)} vectors. "
                "Each chunk must have exactly one vector."
            )

        # Build Qdrant points
        points = []
        for i, (chunk_text, vector) in enumerate(zip(chunks, vectors)):
            # Payload = the data stored alongside the vector.
            # When we search, Qdrant returns the matching payload,
            # so we can show the actual text and its source.
            payload = {
                "text": chunk_text,
                "document_id": str(document_id),
                "chunk_index": i,
            }
            # Merge any extra metadata (filename, source_type, etc.)
            if metadata:
                payload.update(metadata)

            points.append(
                qdrant_models.PointStruct(
                    id=str(uuid.uuid4()),  # Unique ID per chunk
                    vector=vector,
                    payload=payload,
                )
            )

        # Batch upsert — Qdrant handles this efficiently
        # For very large documents (1000+ chunks), we batch in groups of 100
        batch_size = 100
        total_upserted = 0

        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
            total_upserted += len(batch)

        return total_upserted

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        source_type: str | None = None,
        score_threshold: float = 0.3,
    ) -> list[dict]:
        """
        Semantic search — find the most relevant chunks for a query.
        
        THIS IS THE CORE OF RAG RETRIEVAL.
        In Phase 3, when a radiologist asks a question, we:
        1. Embed their question → query_vector
        2. Call this method → get relevant medical content
        3. Feed that content to the LLM → get an informed answer
        
        The LLM then generates a response GROUNDED in real medical
        literature, not just its training data (reducing hallucination).
        
        Args:
            query_vector: The embedded search query (384 floats)
            limit: Max number of results to return
            source_type: Optional filter (e.g., only "textbook" sources)
            score_threshold: Minimum similarity score (0.0-1.0).
                           Below this, results are probably noise.
                           
        Returns:
            List of dicts with: text, score, document_id, metadata
        """
        # Build optional filters
        query_filter = None
        if source_type:
            query_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="source_type",
                        match=qdrant_models.MatchValue(value=source_type),
                    )
                ]
            )

        # Execute the search.
        # NOTE: client.search() is deprecated in qdrant-client >= 1.10 in
        # favour of query_points(). Same semantics; results live under
        # `.points` instead of being returned directly.
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )

        # Convert Qdrant results to clean dicts
        return [
            {
                "text": hit.payload.get("text", ""),
                "score": round(hit.score, 4),
                "document_id": hit.payload.get("document_id"),
                "filename": hit.payload.get("filename"),
                "source_type": hit.payload.get("source_type"),
                "chunk_index": hit.payload.get("chunk_index"),
            }
            for hit in response.points
        ]

    def delete_by_document(self, document_id: str) -> bool:
        """
        Delete ALL chunks belonging to a specific document.
        
        Used when:
        - A doctor deletes a document from the knowledge base
        - Re-ingesting a document (delete old chunks, then insert new ones)
        
        Args:
            document_id: The UUID of the document to remove
            
        Returns:
            True if deletion was successful
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qdrant_models.FilterSelector(
                    filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="document_id",
                                match=qdrant_models.MatchValue(value=str(document_id)),
                            )
                        ]
                    )
                ),
            )
            return True
        except Exception as e:
            print(f"❌ Failed to delete chunks for document {document_id}: {e}")
            return False

    def get_collection_info(self) -> dict:
        """
        Get statistics about the Qdrant collection.
        
        Used by the /stats endpoint and health checks.
        Returns: point count, segment info, index status.
        """
        try:
            info = self.client.get_collection(self.collection_name)

            # NOTE: `vectors_count` is None on recent Qdrant servers — it was
            # deprecated because it's ambiguous for multi-vector collections.
            # `points_count` is the reliable field, and since we store exactly
            # one vector per point, they're equivalent for us.
            points_count = info.points_count or 0

            return {
                "collection_name": self.collection_name,
                "vectors_count": info.vectors_count if info.vectors_count is not None else points_count,
                "points_count": points_count,
                "dimension": self.dimension,
                "status": str(info.status),
            }
        except UnexpectedResponse:
            return {
                "collection_name": self.collection_name,
                "vectors_count": 0,
                "points_count": 0,
                "dimension": self.dimension,
                "status": "not_found",
            }
        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "error": str(e),
                "status": "error",
            }

    def get_chunks_by_document(
        self,
        document_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """
        Retrieve all chunks belonging to a specific document.
        
        Used by the chunk preview endpoint — lets developers/admins
        inspect how a document was split, to verify ingestion quality.
        
        ⚠️  WHY WE DON'T PASS `offset` TO scroll():
        Qdrant's scroll `offset` parameter is a POINT ID to resume from — it is
        NOT a "skip N rows" counter like SQL's OFFSET. Passing an integer there
        makes Qdrant look for a point whose ID is that integer, which silently
        returns the wrong page (or nothing).

        Scroll also returns points in ID order, which is random for us because
        chunk IDs are UUIDs. So page 2 wouldn't even contain the chunks that
        logically follow page 1.

        Correct approach: pull every chunk for this one document, sort by
        chunk_index, then slice in Python. A single document's chunk count is
        bounded (hundreds, not millions), so this is cheap and always correct.

        Args:
            document_id: The UUID of the document
            limit: Max chunks to return (pagination)
            offset: How many chunks to skip (pagination)

        Returns:
            List of dicts with: text, chunk_index, char_count
        """
        doc_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="document_id",
                    match=qdrant_models.MatchValue(value=str(document_id)),
                )
            ]
        )

        try:
            all_points = []
            next_page = None  # Qdrant's real cursor: a point ID, or None to start

            while True:
                points, next_page = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=doc_filter,
                    limit=_SCROLL_BATCH_SIZE,
                    offset=next_page,
                    with_payload=True,
                    with_vectors=False,  # Don't return vectors — just text/metadata
                )
                all_points.extend(points)

                # next_page is None once we've seen every matching point.
                if next_page is None:
                    break

                # Safety valve: a single document should never have this many
                # chunks. If it does, something upstream is wrong.
                if len(all_points) >= _MAX_CHUNKS_PER_DOCUMENT:
                    print(
                        f"⚠️  Document {document_id} has more than "
                        f"{_MAX_CHUNKS_PER_DOCUMENT} chunks — truncating preview."
                    )
                    break

            chunks = []
            for point in all_points:
                text = point.payload.get("text", "")
                chunks.append({
                    "chunk_index": point.payload.get("chunk_index", 0),
                    "text": text,
                    "char_count": len(text),
                })

            # Sort by chunk_index so they appear in document order, THEN
            # paginate. Sorting after slicing (the old bug) only orders
            # within a page.
            chunks.sort(key=lambda c: c["chunk_index"])
            return chunks[offset : offset + limit]

        except Exception as e:
            print(f"❌ Failed to get chunks for document {document_id}: {e}")
            return []


# ══════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ══════════════════════════════════════════════════════════════
# Import this anywhere with:
#   from app.services.qdrant_service import qdrant_service
#
# The collection is ensured to exist during app startup (main.py).
# ══════════════════════════════════════════════════════════════

qdrant_service = QdrantService()
