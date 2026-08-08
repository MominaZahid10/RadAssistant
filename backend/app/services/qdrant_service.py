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

        except Exception as e:
            print(f"❌ Failed to initialize Qdrant collection: {e}")
            raise

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

        # Execute the search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
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
            for hit in results
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
            return {
                "collection_name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
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
        
        Args:
            document_id: The UUID of the document
            limit: Max chunks to return (pagination)
            offset: Skip this many chunks (pagination)
            
        Returns:
            List of dicts with: text, chunk_index, char_count
        """
        try:
            # Use scroll to get points by filter (not by vector similarity)
            results, _next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="document_id",
                            match=qdrant_models.MatchValue(value=str(document_id)),
                        )
                    ]
                ),
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,  # Don't return vectors — just text/metadata
            )

            chunks = []
            for point in results:
                text = point.payload.get("text", "")
                chunks.append({
                    "chunk_index": point.payload.get("chunk_index", 0),
                    "text": text,
                    "char_count": len(text),
                })

            # Sort by chunk_index so they appear in document order
            chunks.sort(key=lambda c: c["chunk_index"])
            return chunks

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
