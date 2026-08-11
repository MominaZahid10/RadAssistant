#!/usr/bin/env python3
"""
Does a keyword actually exist in the corpus, and is it reachable?

    python eval/check_keyword.py "visceral pleural line"

An evaluation keyword that isn't in the corpus makes the harness measure a
gap that doesn't exist — you tune retrieval forever against something it can
never satisfy. eval/README.md says to verify keywords before adding them;
this is the tool for doing that.

Three outcomes:
  NOT IN CORPUS       the phrase was never ingested — fix the question, not retrieval
  IN CORPUS, MISSED   the chunk exists but retrieval doesn't surface it — real gap
  RETRIEVED           working; the eval question should be passing
"""

import sys
import httpx

API = "http://localhost:8000"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    phrase = sys.argv[1].lower()
    query = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]

    with httpx.Client(base_url=API, timeout=120.0) as c:
        # ── 1. Is it reachable through normal retrieval? ──
        hits = c.post("/api/v1/knowledge/search",
                      json={"query": query, "limit": 12}).json()["results"]

        found_at = next(
            (i for i, h in enumerate(hits, 1) if phrase in (h.get("text") or "").lower()),
            None,
        )

        print(f'\nquery:   "{query}"')
        print(f'phrase:  "{phrase}"\n')
        print("RETRIEVED CHUNKS")
        for i, h in enumerate(hits[:12], 1):
            mark = "→" if phrase in (h.get("text") or "").lower() else " "
            print(f'  {mark} {i:>2}. {h["score"]:.3f}  {(h.get("filename") or "?")[:52]}')

        if found_at:
            print(f"\n✅ RETRIEVED — phrase present at rank {found_at}")
            return 0

        # ── 2. Not retrieved. Does it exist in the corpus at all? ──
        #
        # ⚠️  THIS USED TO ONLY CHECK DOCUMENTS THAT WERE RETRIEVED, which is
        # circular: if retrieval missed the document, the document was never
        # inspected, and the tool confidently reported "NOT IN CORPUS" for a
        # phrase sitting in the seed data. Scan EVERY document.
        print("\n⚠️  Phrase not in any retrieved chunk. Scanning the whole corpus...")

        page, checked, matches = 1, 0, []
        while True:
            resp = c.get("/api/v1/knowledge/documents",
                         params={"page": page, "page_size": 50}).json()
            docs = resp["documents"]
            if not docs:
                break

            for doc in docs:
                checked += 1
                try:
                    chunks = c.get(f'/api/v1/knowledge/documents/{doc["id"]}/chunks',
                                   params={"page_size": 500}).json()["chunks"]
                except Exception:
                    continue
                for ch in chunks:
                    if phrase in ch["text"].lower():
                        matches.append((doc, ch))
                        break

            if page * 50 >= resp.get("total", 0):
                break
            page += 1

        print(f"   scanned {checked} documents")

        if matches:
            print(f"\n❗ IN CORPUS BUT NOT RETRIEVED — found in {len(matches)} document(s)")
            for doc, ch in matches[:3]:
                print(f'\n   document: {doc["title"][:64]}')
                print(f'   type:     {doc["source_type"]}')
                print(f'   chunk #{ch["chunk_index"]}: {ch["text"][:170]}...')
            print("\n   → A real retrieval gap, not a bad evaluation keyword.")
            print("     Note that reranking can only reorder what stage 1 returns:")
            print("     if this chunk isn't in the top RERANK_CANDIDATES by vector")
            print("     similarity, the cross-encoder never gets to see it.")
            print("     Try raising RERANK_CANDIDATES, or add lexical (BM25)")
            print("     retrieval alongside vector search.")
            return 1

        print("\n❌ NOT IN CORPUS — the phrase was never ingested.")
        print("   → Fix the evaluation question, not retrieval.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
