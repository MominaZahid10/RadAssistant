"""
RadAssist AI — BM25 Lexical Retrieval (Phase 3.6)

WHY THIS EXISTS — the measured failure:

    Query:  "What are the radiographic findings of pneumothorax?"
    Corpus: contains a chunk that literally begins
            "CXR Findings of Pneumothorax: - Visceral pleural line: A thin
             white line separated from the chest wall with NO lung markings"

    Vector search returns, instead, twelve papers about pneumothorax
    DETECTION — deep learning, ICU monitoring, post-biopsy ultrasound —
    scoring 0.71-0.78. The perfect answer never enters the candidate pool.

WHY THE CROSS-ENCODER COULDN'T SAVE IT:
Reranking is a precision stage. It reorders what stage 1 hands it. At
RERANK_CANDIDATES=48 that chunk scrapes in and gets promoted to rank 1; at 24
it doesn't make the pool and reranking is powerless. That's a RECALL failure,
and no amount of reranking fixes recall.

WHY BM25 IS THE RIGHT COMPLEMENT:
Embeddings capture meaning but blur exact terms — "visceral pleural line" and
"pleural effusion imaging" land near each other. BM25 does the opposite: it
scores rare, exact term matches highly and ignores semantics entirely. Their
failure modes are close to orthogonal, which is exactly what you want in a
hybrid.

    embeddings  →  "what is this passage ABOUT?"       (recall by meaning)
    BM25        →  "does it contain THESE WORDS?"      (recall by term)
    cross-enc   →  "does it ANSWER this question?"     (precision)

WHY A HAND-ROLLED BM25 RATHER THAN A LIBRARY:
It's ~50 lines of standard, stable maths (Robertson/Sparck-Jones), and this
project has repeatedly lost hours to package downloads failing on an unstable
connection. One fewer dependency is worth more here than the code it saves.
"""

from __future__ import annotations

import logging
import math
import re
import threading
from collections import Counter

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


# ── BM25 parameters ──────────────────────────────────────────
# k1 controls term-frequency saturation: how quickly repeated occurrences
#    stop adding score. 1.2-2.0 is the standard range.
# b  controls length normalisation: 1.0 fully penalises long documents,
#    0.0 ignores length. 0.75 is the near-universal default.
# These are the values used by Lucene/Elasticsearch and are a sane starting
# point; tune against eval/run_eval.py if needed.
_K1 = 1.5
_B = 0.75

# Words too common to discriminate between medical passages. Kept short on
# purpose — over-filtering removes real signal ("no lung markings" matters,
# and "no" is a stopword in most lists).
_STOPWORDS = frozenset("""
a an and are as at be been by for from has have how in into is it its of on
or that the their there these this to was were what when where which who
why will with would you your do does did can could should
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """
    Lowercase, split on non-alphanumerics, drop stopwords and single chars.

    Deliberately simple — no stemming. Medical terminology is precise, and
    stemming "pleural" → "pleur" risks colliding distinct terms. Exactness is
    the whole reason we're adding a lexical stage.
    """
    return [
        t for t in _TOKEN_RE.findall(text.lower())
        if len(t) > 1 and t not in _STOPWORDS
    ]


class LexicalIndex:
    """
    In-memory BM25 index over every chunk in the vector store.

    SIZE: at ~10,500 chunks averaging 500 characters, the postings and payload
    together sit in the tens of megabytes. Fine for this scale. If the corpus
    reaches millions of chunks, move to Qdrant sparse vectors or Elasticsearch
    — the interface here would stay the same.

    STALENESS: the index is a snapshot. Newly ingested documents aren't
    searchable lexically until it's rebuilt. `mark_stale()` is called after
    ingestion so the next search rebuilds automatically.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._built = False
        self._stale = True

        self._payloads: list[dict] = []          # parallel to doc ids
        self._doc_lens: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._doc_freq: Counter = Counter()      # term → number of chunks containing it
        self._avg_len: float = 0.0
        self._n_docs: int = 0

    # ── State ────────────────────────────────────────────────

    @property
    def is_built(self) -> bool:
        return self._built and not self._stale

    def mark_stale(self) -> None:
        """Call after ingestion so the next search rebuilds."""
        self._stale = True

    def get_info(self) -> dict:
        return {
            "enabled": settings.HYBRID_ENABLED,
            "built": self._built,
            "stale": self._stale,
            "chunks": self._n_docs,
        }

    # ── Build ────────────────────────────────────────────────

    def build(self, records: list[dict]) -> None:
        """
        Build the index from chunk payloads.

        Each record needs at least a "text" key; everything else is carried
        through so search results can be returned without a second round-trip
        to Qdrant.
        """
        with self._lock:
            payloads, doc_lens, term_freqs = [], [], []
            doc_freq: Counter = Counter()

            for rec in records:
                # Index the title-prefixed variant when present: a chunk without its
                # document context is much harder to find by term.
                tokens = tokenize(rec.get("search_text") or rec.get("text", ""))
                if not tokens:
                    continue
                tf = Counter(tokens)
                payloads.append(rec)
                doc_lens.append(len(tokens))
                term_freqs.append(tf)
                doc_freq.update(tf.keys())      # once per doc, not per occurrence

            # ── Inverted index: term → [(doc_index, term_frequency), ...] ──
            # ⚠️  WITHOUT THIS, SEARCH SCORES EVERY CHUNK.
            # The first implementation looped over all 11,315 chunks per query
            # and added 1.2 SECONDS to every retrieval — measured, not
            # theoretical. A query touches maybe 3-5 terms, which appear in a
            # small fraction of chunks; there's no reason to score the rest.
            #
            # This is why every real search engine is built on postings lists.
            postings: dict[str, list[tuple[int, int]]] = {}
            for i, tf in enumerate(term_freqs):
                for term, freq in tf.items():
                    postings.setdefault(term, []).append((i, freq))

            self._payloads = payloads
            self._doc_lens = doc_lens
            self._postings = postings
            self._doc_freq = doc_freq
            self._n_docs = len(payloads)
            self._avg_len = (sum(doc_lens) / len(doc_lens)) if doc_lens else 0.0
            self._built = True
            self._stale = False

        print(f"✅ Lexical (BM25) index built: {self._n_docs:,} chunks, "
              f"{len(doc_freq):,} unique terms")

    # ── Search ───────────────────────────────────────────────

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """
        Return the top-scoring chunks for `query`, each with a `bm25_score`.

        Empty list if the index isn't built or the query has no usable terms —
        callers treat that as "no lexical contribution", never as an error.
        """
        if not self._built or self._n_docs == 0:
            return []

        q_terms = tokenize(query)
        if not q_terms:
            return []

        # Precompute IDF per query term.
        #   idf = ln(1 + (N - df + 0.5) / (df + 0.5))
        # The +1 inside the log keeps IDF non-negative for terms appearing in
        # most documents — without it, very common terms get negative weight
        # and actively push relevant chunks DOWN.
        idf: dict[str, float] = {}
        for term in set(q_terms):
            df = self._doc_freq.get(term, 0)
            if df:
                idf[term] = math.log(1 + (self._n_docs - df + 0.5) / (df + 0.5))

        if not idf:
            return []

        # Walk only the postings lists for the query's terms — typically a few
        # hundred chunks rather than all 11,315.
        accum: dict[int, float] = {}
        for term, term_idf in idf.items():
            for doc_i, freq in self._postings.get(term, ()):
                dl_ratio = (
                    self._doc_lens[doc_i] / self._avg_len if self._avg_len else 1.0
                )
                # Saturating term frequency, length-normalised.
                accum[doc_i] = accum.get(doc_i, 0.0) + term_idf * (
                    (freq * (_K1 + 1)) / (freq + _K1 * (1 - _B + _B * dl_ratio))
                )

        scored = sorted(
            ((score, i) for i, score in accum.items() if score > 0),
            reverse=True,
        )

        results = []
        for score, i in scored[:limit]:
            rec = dict(self._payloads[i])
            rec["bm25_score"] = round(score, 4)
            results.append(rec)
        return results

    # ── Salient terms ────────────────────────────────────────

    def salient_terms(self, text: str, top_k: int = 24) -> list[str]:
        """
        The most distinctive corpus-known terms in `text`, strongest first.

        Used to build a retrieval query from an UPLOADED DOCUMENT rather than
        from the user's instruction.

        ⚠️  WHY THIS EXISTS.
        A clinician uploads a report and types "review this". Retrieval
        embedded *that* — and dutifully returned papers about the practice of
        radiology reporting: "Revealing the most common reporting errors",
        "The reporting quality of NLP studies". Every source was about
        reports-as-a-genre; not one was about the patient's osteopenia or
        compression fracture. The background section was decorative, and the
        model cited [1] for everything because nothing actually supported
        anything.

        Scoring is TF-IDF against the live index, which buys two things free:

        1. STOPWORD AND BOILERPLATE REMOVAL. "report", "findings", "patient"
           appear in nearly every chunk, so their IDF is near zero and they
           sink. No hand-maintained medical stoplist to keep current.

        2. OCR GARBAGE REMOVAL. Terms absent from the corpus are dropped
           outright. On the report that motivated this, that discards
           "hyperiordotic", "peivis", "solt" and "tracompression" — misreads
           that would otherwise be the highest-IDF terms in the document,
           since nothing is rarer than a word that does not exist.

        The cost is that a genuinely novel finding — real, but absent from the
        corpus — is also dropped. That is the right trade: an unknown term
        cannot retrieve anything anyway, and the document itself still reaches
        the model as the primary source. This only shapes what BACKGROUND gets
        pulled alongside it.
        """
        if not self._built or self._n_docs == 0:
            return []

        tokens = tokenize(text)
        if not tokens:
            return []

        scored: list[tuple[float, str]] = []
        for term, freq in Counter(tokens).items():
            df = self._doc_freq.get(term, 0)
            if not df:
                continue        # not in the corpus: OCR noise, or unretrievable
            idf = math.log(1 + (self._n_docs - df + 0.5) / (df + 0.5))
            # Sub-linear TF: a term repeated six times is more central than one
            # mentioned once, but not six times more.
            scored.append((idf * (1.0 + math.log(freq)), term))

        scored.sort(reverse=True)
        return [term for _, term in scored[:top_k]]


# ══════════════════════════════════════════════════════════════
# SINGLETON
# ══════════════════════════════════════════════════════════════

lexical_index = LexicalIndex()
