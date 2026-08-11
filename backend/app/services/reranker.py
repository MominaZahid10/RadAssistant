"""
RadAssist AI — Cross-Encoder Reranker (Phase 3.5)

WHY THIS EXISTS — the measured problem:

    Source recall@12   88.9%   ← we find the right ARTICLE
    Keyword recall@12  72.2%   ← we retrieve the wrong CHUNKS from it

On four evaluation questions the correct document ranked #1 and the answer
still wasn't in the retrieved text. For "What are the radiographic findings of
pneumothorax?" we pulled the article's *Definition* and *Types* passages while
"CXR Findings: visceral pleural line..." went unretrieved.

WHY A BI-ENCODER CAN'T FIX THAT:
The embedding model encodes the question and each passage *separately*, then
compares vectors. Both "Pneumothorax is air in the pleural space" and "CXR
findings: a thin visceral pleural line" are equally *about* pneumothorax, so
they embed to similar places. Nothing in that representation captures
"...and this one answers the question that was asked."

WHAT A CROSS-ENCODER DOES DIFFERENTLY:
It feeds the query and passage through the model **together** as one sequence,
so attention runs across both. The output is a single relevance score for that
specific pair. It cannot be precomputed or indexed — which is exactly why it's
accurate, and why it's used as a second stage over a small candidate set
rather than as the primary index.

    Stage 1  vector search over 10,494 chunks  →  48 candidates   (recall)
    Stage 2  cross-encoder scores 48 pairs     →  top 12          (precision)

DESIGN CONSTRAINT — MUST DEGRADE SILENTLY:
If the model is missing, unloadable, or disabled, this returns None and the
caller keeps the vector ordering. Chat keeps working. Given how much time this
project has lost to model downloads failing on an unstable connection, a
reranker that can take the whole system down would be a bad trade.
"""

from __future__ import annotations

import logging
import os

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class RerankerService:
    """
    Scores (query, passage) pairs for relevance.

    USAGE:
        scores = reranker_service.score(query, [c.text for c in chunks])
        if scores is not None:
            chunks = [c for _, c in sorted(zip(scores, chunks), reverse=True)]

    `score()` returns None whenever reranking is unavailable, so callers
    always have a defined fallback: keep the original order.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.RERANKER_MODEL
        self._model = None
        self._loaded = False
        self._load_failed = False  # don't retry a known-broken load every call

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def is_enabled(self) -> bool:
        return settings.RERANK_ENABLED

    def load(self) -> bool:
        """
        Load the cross-encoder. Returns True on success.

        Never raises — a reranker that can't load must not prevent the
        application from starting. The caller degrades to vector ordering.
        """
        if self._loaded:
            return True
        if self._load_failed:
            return False
        if not settings.RERANK_ENABLED:
            logger.info("Reranking disabled (RERANK_ENABLED=false)")
            return False

        offline = os.getenv("HF_HUB_OFFLINE", "0") in ("1", "true", "True")
        mode = "offline, from cache" if offline else "online"
        print(f"📦 Loading reranker: {self.model_name} ({mode})...")

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, max_length=512)
            self._loaded = True
            print(f"✅ Reranker loaded: {self.model_name}")
            return True

        except Exception as e:  # noqa: BLE001 — deliberately broad
            self._load_failed = True
            print(f"⚠️  Reranker unavailable: {e}")
            if offline:
                print(
                    "   HF_HUB_OFFLINE=1, so no download was attempted.\n"
                    "   To enable reranking, populate the cache once:\n"
                    "       HF_HUB_OFFLINE=0 docker-compose up -d backend\n"
                    "   Retrieval still works — results keep vector ordering."
                )
            else:
                print("   Retrieval still works — results keep vector ordering.")
            return False

    def score(
        self,
        query: str,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[float] | None:
        """
        Relevance score for each (query, text) pair. Higher is more relevant.

        ⚠️  SCORES ARE UNBOUNDED LOGITS, NOT SIMILARITIES.
        ms-marco cross-encoders emit roughly -11..+11. They are meaningful for
        ORDERING only — never display them as a percentage, and never compare
        them against the 0-1 cosine scores shown in the evidence panel.

        Returns None if reranking is unavailable, meaning "keep your order".
        """
        if not texts:
            return None
        if not settings.RERANK_ENABLED:
            return None
        if not self._loaded and not self.load():
            return None

        try:
            pairs = [(query, t) for t in texts]
            raw = self._model.predict(pairs, batch_size=batch_size)
            return [float(s) for s in raw]
        except Exception as e:  # noqa: BLE001
            logger.warning("Reranking failed, falling back to vector order: %s", e)
            return None

    def get_info(self) -> dict:
        """For /health and debugging."""
        return {
            "enabled": settings.RERANK_ENABLED,
            "model": self.model_name,
            "loaded": self._loaded,
            "load_failed": self._load_failed,
            "candidates": settings.RERANK_CANDIDATES,
        }


# ══════════════════════════════════════════════════════════════
# SINGLETON
# ══════════════════════════════════════════════════════════════
# Loaded lazily on first use rather than at startup — it's optional, and
# startup already waits on the embedding model.

reranker_service = RerankerService()
