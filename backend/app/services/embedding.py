"""
RadAssist AI — Embedding Service

WHAT THIS FILE DOES:
Loads an AI model that converts text into "embeddings" — lists of numbers
(vectors) that capture the MEANING of the text. Similar medical concepts
produce similar vectors, which is how semantic search works.

HOW EMBEDDINGS WORK (Simple Analogy):
Imagine placing every medical concept on a giant map:
- "pneumonia" and "lung infection" would be CLOSE together
- "pneumonia" and "knee surgery" would be FAR apart

An embedding model creates this "map" mathematically. Each text gets
coordinates (a vector of 384 numbers for our model). We then search
by finding which stored texts have coordinates closest to the query.

WHY SINGLETON PATTERN?
Loading the model takes 2-5 seconds and ~80MB of RAM. If we loaded it
for every request, the system would be painfully slow. Instead, we load
it ONCE at startup and reuse the same instance for all requests.

WHAT IS sentence-transformers?
A Python library (by UBI/HuggingFace) that wraps transformer models
specifically for generating embeddings. It handles:
- Downloading the model from HuggingFace Hub (first run only)
- Tokenization (splitting text into model-readable pieces)
- Inference (running the neural network)
- Pooling (combining token outputs into one vector per text)

MODEL CHOICE:
Currently: all-MiniLM-L6-v2 (384 dim, ~80MB, fast on CPU)
Future:    PubMedBERT (768 dim, ~400MB, better for medical text)
The model name is in config.py — swap it there without touching this code.
"""

from sentence_transformers import SentenceTransformer
from app.config import get_settings

settings = get_settings()


class EmbeddingService:
    """
    Manages the embedding model lifecycle and provides encoding methods.
    
    USAGE:
        # Initialize once (at app startup)
        embedding_service = EmbeddingService()
        
        # Encode text into vectors (can be called many times)
        vectors = embedding_service.encode(["pneumonia findings", "chest x-ray"])
        # Returns: [[0.12, -0.45, 0.78, ...], [0.33, -0.12, 0.56, ...]]
        #           ↑ 384 numbers each          ↑ 384 numbers each
    """

    def __init__(self, model_name: str | None = None):
        """
        Load the embedding model.
        
        On FIRST run, the model is downloaded from HuggingFace (~80MB).
        After that, it's cached locally and loads from disk instantly.
        
        Args:
            model_name: Override the default model from config.
                        Useful for testing with different models.
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.model: SentenceTransformer | None = None
        self.dimension: int = settings.EMBEDDING_DIMENSION
        self._loaded = False

    def load(self) -> None:
        """
        Actually load the model into memory.
        
        WHY SEPARATE FROM __init__?
        We want to control WHEN the model loads. During testing,
        we might want to create the service object without loading
        the heavy model. In production, we call load() at startup.
        """
        if self._loaded:
            return

        import os

        offline = os.getenv("HF_HUB_OFFLINE", "0") in ("1", "true", "True")
        mode = "offline, from cache" if offline else "online"
        print(f"📦 Loading embedding model: {self.model_name} ({mode})...")

        try:
            self.model = SentenceTransformer(self.model_name)
            # Get the actual dimension from the model (verify config matches)
            actual_dim = self.model.get_sentence_embedding_dimension()
            if actual_dim != self.dimension:
                print(
                    f"⚠️  Config says {self.dimension} dimensions, "
                    f"but model produces {actual_dim}. Using model's value."
                )
                self.dimension = actual_dim
            self._loaded = True
            print(f"✅ Embedding model loaded: {self.model_name} ({self.dimension}D)")
        except Exception as e:
            print(f"❌ Failed to load embedding model '{self.model_name}': {e}")
            if offline:
                # The most likely cause, and the error text from huggingface_hub
                # doesn't make it obvious.
                print(
                    "   HF_HUB_OFFLINE=1 is set, so no download was attempted.\n"
                    "   If this is a model you haven't used before, start once with\n"
                    "   HF_HUB_OFFLINE=0 to populate the cache:\n"
                    "       HF_HUB_OFFLINE=0 docker-compose up -d backend"
                )
            else:
                print(
                    "   Could not reach huggingface.co. Check container DNS:\n"
                    "       docker-compose exec backend getent hosts huggingface.co"
                )
            raise

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Convert a list of text strings into embedding vectors.
        
        This is THE core function of the RAG system. Every piece of
        text that enters the knowledge base passes through here.
        
        Args:
            texts: List of text strings to embed.
                   Example: ["Pneumonia is an infection...", "The chest X-ray shows..."]
            batch_size: How many texts to process at once.
                        Larger = faster but uses more memory.
                        Default: 32 (from config)
            show_progress: Show a progress bar (useful for large ingestions).
        
        Returns:
            List of vectors, one per input text.
            Each vector is a list of floats (384 numbers for MiniLM).
            
        Example:
            >>> service.encode(["hello world"])
            [[0.123, -0.456, 0.789, ...]]  # 384 floats
        """
        if not self._loaded:
            self.load()

        if not texts:
            return []

        batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE

        # SentenceTransformer.encode() handles all the complexity:
        # 1. Tokenizes each text (splits into subword tokens)
        # 2. Pads/truncates to model's max length (256 tokens for MiniLM)
        # 3. Runs through the neural network
        # 4. Pools token embeddings into one vector per text
        # 5. Optionally normalizes vectors (for cosine similarity)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # Normalized for cosine similarity
        )

        # Convert numpy arrays to plain Python lists (JSON-serializable)
        return embeddings.tolist()

    def encode_single(self, text: str) -> list[float]:
        """
        Convenience method to embed a single text string.
        Used for search queries (where you have just one query, not a batch).
        
        Args:
            text: A single string to embed.
            
        Returns:
            A single vector (list of 384 floats).
        """
        results = self.encode([text])
        return results[0]

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._loaded

    def get_info(self) -> dict:
        """
        Return model information for health checks and debugging.
        
        Used by the /health and /stats endpoints to show what
        embedding model is active and its capabilities.
        """
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "loaded": self._loaded,
            "max_tokens": 256 if "MiniLM" in self.model_name else 512,
        }


# ══════════════════════════════════════════════════════════════
# SINGLETON INSTANCE
# ══════════════════════════════════════════════════════════════
# This is the ONE instance used across the entire application.
# Import it anywhere with:
#   from app.services.embedding import embedding_service
#
# It's created but NOT loaded yet — load() is called in main.py
# during startup, so the model is ready before the first request.
# ══════════════════════════════════════════════════════════════

embedding_service = EmbeddingService()
