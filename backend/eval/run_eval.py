#!/usr/bin/env python3
"""
RadAssist AI — Retrieval Evaluation Harness

Scores retrieval against a fixed question set so changes can be measured
rather than eyeballed one query at a time.

    python eval/run_eval.py                      # retrieval only — free, ~20s
    python eval/run_eval.py --save baseline      # store as the comparison point
    python eval/run_eval.py --vs baseline        # show deltas against it
    python eval/run_eval.py --with-chat          # also test guardrails (uses LLM tokens)

WHY THIS EXISTS:
Four defects in this project produced *plausible-looking output* rather than
errors — OCR placeholder text stored as content, chunks starting mid-word,
citations in the wrong bracket character, and an API silently ignoring its
parameters. Every one survived code review and was caught by chance.

Then scaling the corpus 14 → 232 documents made one query worse while its
similarity scores went UP (0.78 → 0.83). Precision fell while the number that
looks like quality rose.

A human spot-checking one query cannot catch that. A scorecard can.

WHAT'S MEASURED:
    recall@k        did an expected source appear in the top k?
    keyword recall  did the retrieved TEXT contain the answer phrase?
    MRR             how high did the first correct hit rank?
    refusal         did the guardrails hold on questions with no answer?

Keyword recall is the metric to trust as the corpus grows: document titles
churn every time PMC is re-fetched, but "does the retrieved text contain
'visceral pleural line'" stays meaningful.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import httpx
    import yaml
except ImportError:
    sys.exit("Missing deps. Run:  pip install httpx pyyaml")


EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"

# Phrase the system emits when it declines. Must stay in sync with
# rag_service.GROUNDING_SCAFFOLD and OUT_OF_SCOPE_REPLY.
REFUSAL_MARKERS = (
    "don't have enough information",
    "designed to assist with radiology",
)

C_OK, C_BAD, C_WARN, C_DIM, C_BOLD, C_END = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)


# ══════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════


def source_rank(results: list[dict], expected: list[str]) -> int | None:
    """
    1-based rank of the first result whose document title matches any expected
    fragment. None if absent.

    Matching is case-insensitive substring — PMC titles are long and exact
    matching would be unusable.
    """
    if not expected:
        return None
    wanted = [e.lower() for e in expected]
    for i, r in enumerate(results, start=1):
        title = (r.get("filename") or "").lower()
        if any(w in title for w in wanted):
            return i
    return None


def keyword_rank(results: list[dict], keywords: list[str]) -> int | None:
    """
    1-based rank of the first result whose TEXT contains any expected keyword.

    More robust than title matching: it survives corpus changes and measures
    what actually matters — whether the answer reached the model's context.
    """
    if not keywords:
        return None
    wanted = [k.lower() for k in keywords]
    for i, r in enumerate(results, start=1):
        text = (r.get("text") or "").lower()
        if any(w in text for w in wanted):
            return i
    return None


def hit_at(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def looks_like_refusal(answer: str) -> bool:
    low = (answer or "").lower()
    return any(m in low for m in REFUSAL_MARKERS)


# ══════════════════════════════════════════════════════════════
# EXECUTION
# ══════════════════════════════════════════════════════════════


def evaluate(api: str, questions: list[dict], limit: int, with_chat: bool) -> dict:
    rows: list[dict] = []

    with httpx.Client(base_url=api, timeout=90.0) as client:
        for q in questions:
            qid, category = q["id"], q["category"]
            should_answer = q.get("should_answer", True)

            row: dict = {
                "id": qid,
                "category": category,
                "question": q["question"],
                "should_answer": should_answer,
            }

            # ── Retrieval ──
            t0 = time.time()
            try:
                resp = client.post(
                    "/api/v1/knowledge/search",
                    json={"query": q["question"], "limit": limit},
                )
                resp.raise_for_status()
                results = resp.json().get("results", [])
                row["error"] = None
            except httpx.HTTPStatusError as e:
                # Include the response body — FastAPI puts the actual reason
                # in `detail`, and without it "22 errored" tells you nothing.
                body = e.response.text[:300]
                row["error"] = f"HTTP {e.response.status_code}: {body}"
                results = []
            except Exception as e:  # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {e}"
                results = []

            row["latency_ms"] = round((time.time() - t0) * 1000)
            row["n_results"] = len(results)
            row["top_score"] = results[0]["score"] if results else 0.0
            row["distinct_documents"] = len({r.get("document_id") for r in results})

            s_rank = source_rank(results, q.get("expect_sources", []))
            k_rank = keyword_rank(results, q.get("expect_keywords", []))

            row.update(
                source_rank=s_rank,
                keyword_rank=k_rank,
                source_hit_5=hit_at(s_rank, 5),
                source_hit_12=hit_at(s_rank, 12),
                keyword_hit_5=hit_at(k_rank, 5),
                keyword_hit_12=hit_at(k_rank, 12),
                mrr=reciprocal_rank(k_rank if k_rank else s_rank),
            )

            # ── Generation (optional — costs tokens) ──
            if with_chat:
                try:
                    cr = client.post(
                        "/api/v1/chat",
                        json={"query": q["question"], "stream": False,
                              "include_sources": False},
                    )
                    cr.raise_for_status()
                    answer = cr.json().get("answer", "")
                    refused = looks_like_refusal(answer)
                    row["refused"] = refused
                    # Correct when: should_answer and it didn't refuse,
                    #            or not should_answer and it did.
                    row["guardrail_ok"] = (refused != should_answer)
                    row["answer_preview"] = answer[:120]
                except Exception as e:  # noqa: BLE001
                    row["refused"] = None
                    row["guardrail_ok"] = None
                    row["answer_preview"] = f"ERROR: {e}"

            rows.append(row)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api": api,
        "limit": limit,
        "with_chat": with_chat,
        "rows": rows,
        "summary": summarise(rows, with_chat),
    }


def summarise(rows: list[dict], with_chat: bool) -> dict:
    answerable = [r for r in rows if r["should_answer"]]
    refusable = [r for r in rows if not r["should_answer"]]

    def pct(subset: list[dict], key: str) -> float:
        return round(100 * sum(1 for r in subset if r.get(key)) / len(subset), 1) if subset else 0.0

    summary = {
        "n_questions": len(rows),
        "n_answerable": len(answerable),
        "source_recall_5": pct(answerable, "source_hit_5"),
        "source_recall_12": pct(answerable, "source_hit_12"),
        "keyword_recall_5": pct(answerable, "keyword_hit_5"),
        "keyword_recall_12": pct(answerable, "keyword_hit_12"),
        "mrr": round(statistics.mean([r["mrr"] for r in answerable]), 3) if answerable else 0.0,
        "mean_top_score": round(statistics.mean([r["top_score"] for r in answerable]), 3) if answerable else 0.0,
        "mean_distinct_docs": round(statistics.mean([r["distinct_documents"] for r in answerable]), 1) if answerable else 0.0,
        "median_latency_ms": round(statistics.median([r["latency_ms"] for r in rows])) if rows else 0,
        "errors": sum(1 for r in rows if r.get("error")),
    }

    # Per-category keyword recall — definitional is the one that regressed.
    for cat in sorted({r["category"] for r in answerable}):
        subset = [r for r in answerable if r["category"] == cat]
        summary[f"keyword_recall_12__{cat}"] = pct(subset, "keyword_hit_12")

    if with_chat:
        checked = [r for r in rows if r.get("guardrail_ok") is not None]
        summary["guardrail_pass_rate"] = pct(checked, "guardrail_ok")
        summary["refusal_correct_on_refusable"] = pct(
            [r for r in refusable if r.get("guardrail_ok") is not None], "guardrail_ok"
        )

    return summary


# ══════════════════════════════════════════════════════════════
# REPORTING
# ══════════════════════════════════════════════════════════════


def print_report(report: dict, baseline: dict | None) -> None:
    rows, s = report["rows"], report["summary"]

    print(f"\n{C_BOLD}PER-QUESTION{C_END}")
    print(f"{C_DIM}{'id':<24}{'cat':<14}{'src':>5}{'kw':>5}{'docs':>6}{'top':>7}{C_END}")
    print("─" * 62)

    for r in rows:
        if not r["should_answer"]:
            continue
        srank = r["source_rank"]
        krank = r["keyword_rank"]

        # Colour on keyword rank — the metric that survives corpus churn.
        if krank and krank <= 5:
            colour = C_OK
        elif krank and krank <= 12:
            colour = C_WARN
        else:
            colour = C_BAD

        print(
            f"{colour}{r['id']:<24}{C_END}{r['category']:<14}"
            f"{(srank or '—'):>5}{(krank or '—'):>5}"
            f"{r['distinct_documents']:>6}{r['top_score']:>7.3f}"
        )

    refusals = [r for r in rows if not r["should_answer"]]
    if refusals and report["with_chat"]:
        print(f"\n{C_BOLD}GUARDRAILS{C_END}")
        for r in refusals:
            ok = r.get("guardrail_ok")
            mark = f"{C_OK}refused{C_END}" if ok else f"{C_BAD}ANSWERED{C_END}"
            print(f"  {r['id']:<24}{mark}")

    print(f"\n{C_BOLD}SUMMARY{C_END}")
    keys = [
        ("keyword_recall_5", "Keyword recall@5", "%"),
        ("keyword_recall_12", "Keyword recall@12", "%"),
        ("source_recall_5", "Source recall@5", "%"),
        ("source_recall_12", "Source recall@12", "%"),
        ("mrr", "MRR", ""),
        ("mean_top_score", "Mean top score", ""),
        ("mean_distinct_docs", "Mean distinct docs", ""),
        ("median_latency_ms", "Median latency", "ms"),
    ]
    if report["with_chat"]:
        keys.append(("guardrail_pass_rate", "Guardrail pass rate", "%"))

    for key, label, unit in keys:
        val = s.get(key, 0)
        line = f"  {label:<22}{val}{unit}"
        if baseline:
            prev = baseline["summary"].get(key)
            if isinstance(prev, (int, float)) and isinstance(val, (int, float)):
                delta = round(val - prev, 3)
                if abs(delta) > 1e-9:
                    # Latency: lower is better. Everything else: higher.
                    good = delta < 0 if "latency" in key else delta > 0
                    col = C_OK if good else C_BAD
                    line += f"   {col}{'+' if delta > 0 else ''}{delta}{C_END}"
                    line += f" {C_DIM}(was {prev}){C_END}"
        print(line)

    print(f"\n{C_BOLD}BY CATEGORY{C_END} {C_DIM}(keyword recall@12){C_END}")
    for key in sorted(k for k in s if k.startswith("keyword_recall_12__")):
        cat = key.split("__", 1)[1]
        val = s[key]
        col = C_OK if val >= 80 else (C_WARN if val >= 50 else C_BAD)
        print(f"  {cat:<22}{col}{val}%{C_END}")

    if s["errors"]:
        print(f"\n{C_BOLD}{C_BAD}ERRORS ({s['errors']}/{len(rows)}){C_END}")

        # Group by message — 22 identical failures is one problem, not 22.
        by_msg: dict[str, list[str]] = {}
        for r in rows:
            if r.get("error"):
                by_msg.setdefault(r["error"], []).append(r["id"])

        for msg, ids in by_msg.items():
            print(f"\n  {C_BAD}{msg}{C_END}")
            print(f"  {C_DIM}affected: {len(ids)} question(s) — "
                  f"{', '.join(ids[:3])}{'...' if len(ids) > 3 else ''}{C_END}")

        print(f"\n  {C_WARN}Try:{C_END}")
        print(f"    curl {report['api']}/api/v1/health")
        print(f"    curl -X POST {report['api']}/api/v1/knowledge/search \\")
        print(f"      -H 'Content-Type: application/json' \\")
        print(f"      -d '{{\"query\":\"pneumothorax\",\"limit\":3}}'")
        print(f"\n  {C_DIM}A ~10ms failure means connection refused or an immediate")
        print(f"  HTTP error — not slow retrieval. Check the backend finished")
        print(f"  starting (the embedding model takes ~8s to load).{C_END}")


# ══════════════════════════════════════════════════════════════


def main() -> int:
    p = argparse.ArgumentParser(description="Score RadAssist retrieval quality.")
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--limit", type=int, default=12,
                   help="Chunks to retrieve per question (default: 12)")
    p.add_argument("--with-chat", action="store_true",
                   help="Also test generation/guardrails. Uses LLM tokens.")
    p.add_argument("--save", metavar="NAME",
                   help="Save results as eval/results/NAME.json")
    p.add_argument("--vs", metavar="NAME",
                   help="Compare against a previously saved run")
    args = p.parse_args()

    questions = yaml.safe_load((EVAL_DIR / "questions.yaml").read_text(encoding="utf-8"))["questions"]

    baseline = None
    if args.vs:
        path = RESULTS_DIR / f"{args.vs}.json"
        if not path.exists():
            print(f"{C_BAD}No baseline at {path}{C_END}")
            return 1
        baseline = json.loads(path.read_text(encoding="utf-8"))

    print(f"{C_BOLD}RadAssist retrieval evaluation{C_END}")
    print(f"{C_DIM}{len(questions)} questions · limit={args.limit} · "
          f"chat={'on' if args.with_chat else 'off'}{C_END}")

    # Fail fast with a useful message rather than 22 identical errors.
    try:
        with httpx.Client(base_url=args.api, timeout=15.0) as c:
            h = c.get("/api/v1/health").json()
        emb = h.get("components", {}).get("embedding_model", "?")
        if "loaded" not in str(emb):
            print(f"\n{C_BAD}Embedding model is not ready: {emb}{C_END}")
            print(f"{C_DIM}Retrieval cannot work until it loads. "
                  f"Check: docker-compose logs backend --tail=20{C_END}\n")
            return 1
        print(f"{C_DIM}backend ok · {emb}{C_END}")
    except Exception as e:  # noqa: BLE001
        print(f"\n{C_BAD}Cannot reach {args.api} — {type(e).__name__}: {e}{C_END}")
        print(f"{C_DIM}Start it with: docker-compose up -d{C_END}\n")
        return 1

    report = evaluate(args.api, questions, args.limit, args.with_chat)
    print_report(report, baseline)

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / f"{args.save}.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n{C_DIM}Saved → {out}{C_END}")

    print()
    return 1 if report["summary"]["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
