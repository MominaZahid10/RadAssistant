# Retrieval Evaluation

Scores retrieval against a fixed question set, so changes can be **measured** rather than eyeballed one query at a time.

```bash
cd backend
python eval/run_eval.py                  # retrieval only — free, ~20s
python eval/run_eval.py --save baseline  # store the comparison point
python eval/run_eval.py --vs baseline    # show deltas
python eval/run_eval.py --with-chat      # also test guardrails (uses tokens)
```

Requires the stack running (`docker-compose up -d`) — it hits the live API, so it measures what's actually deployed rather than an approximation.

---

## Why this exists

Four defects in this project produced **plausible-looking output rather than errors**:

| Defect | What it looked like |
|---|---|
| OCR failure stored as content | Document marked `completed` with `"[OCR failed: ...]"` as its text |
| Chunk overlap sliced mid-word | Passage beginning `"e missed on chest imaging"` |
| `【N】` instead of `[N]` citations | Prose read perfectly; no citation was clickable |
| API ignoring its parameters | `202 Accepted` while silently using defaults |

Every one survived code review.

Then scaling 14 → 232 documents made a query **worse while its similarity scores went up** (0.78 → 0.83). Precision fell while the number that looks like quality rose. A human spot-checking one query cannot catch that.

---

## Metrics

| Metric | Meaning | Trust it? |
|---|---|---|
| **Keyword recall@k** | Did the retrieved *text* contain the answer phrase? | **Primary.** Survives corpus churn |
| Source recall@k | Did the expected *document* appear? | Brittle — PMC titles change on every re-fetch |
| **MRR** | How high did the first correct hit rank? | Rank 1 ≫ rank 11, even though both are "hits" |
| Mean distinct docs | How many different sources in the results? | Low = one paper monopolising context |
| Mean top score | Average best similarity | **Watch for inflation without relevance** |
| Guardrail pass rate | Did refusals hold? (`--with-chat`) | Must be 100% on `refuse` questions |

**Keyword recall is the one to optimise.** Document titles come and go as the PMC corpus is re-fetched, but *"does the retrieved text contain `visceral pleural line`"* stays meaningful regardless.

`mean_top_score` is a trap and is reported deliberately: it went **up** during the Phase 3 regression while quality went down. If it rises while keyword recall falls, retrieval is getting more confident and less useful.

---

## Question categories

Scored separately because they fail differently.

| Category | Tests | Count |
|---|---|---|
| `definitional` | "What does X look like on imaging?" — **the Phase 3 failure mode**, where term-dense papers outrank explanatory content | 8 |
| `criteria` | Precise thresholds, named guidelines, numbers | 5 |
| `comparative` | Requires synthesising more than one source | 5 |
| `refuse` | Must **not** be answered — guardrail check | 4 |

Watch `keyword_recall_12__definitional` most closely. That's the category that broke.

---

## Adding questions

```yaml
- id: unique_snake_case
  category: definitional          # definitional | criteria | comparative | refuse
  question: "What are the CT findings of appendicitis?"
  expect_sources:                 # optional — title substrings
    - "Acute Abdomen"
  expect_keywords:                # required for answerable questions
    - "appendiceal"
    - "periappendiceal fat"
  should_answer: true
```

Two rules:

- **Keywords must be lowercase and distinctive.** `"imaging"` or `"patient"` appear everywhere and would pass trivially.
- **Verify the phrase actually exists in the corpus first**, or you're measuring a gap that isn't real:

```bash
curl -s -X POST localhost:8000/api/v1/knowledge/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"appendicitis CT","limit":10}' | grep -io "periappendiceal"
```

---

## Reading a run

```
keyword_recall_12__definitional   38%    ← the regression
keyword_recall_12__criteria       80%
mean_distinct_docs                2.4    ← low: one paper dominating
mean_top_score                    0.83   ← high scores, poor relevance
```

That shape — high scores, low distinct documents, poor definitional recall — is the signature of single-stage vector retrieval on a large corpus. It's what cross-encoder reranking (Phase 3.5 Step 2) is meant to fix.

---

## Workflow

```bash
python eval/run_eval.py --save baseline      # before changing anything
# ... make a retrieval change ...
python eval/run_eval.py --vs baseline        # did it actually help?
python eval/run_eval.py --save reranked      # keep the new point of comparison
```

Deltas are colour-coded: green is better, red is worse, and latency is inverted (lower is better).
