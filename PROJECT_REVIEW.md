# RadAssist AI — Where the project stands

Written after Phase 6 Step 2. An honest assessment, including the parts that
are weaker than they look.

---

## By the numbers

| | |
|---|---|
| Backend application code | 13,309 lines |
| Backend tests | 6,100 lines · **506 tests** |
| Frontend | 3,070 lines · **0 tests** |
| Services | 17 |
| Database migrations | 5 |
| Knowledge base | 296 open-access articles · 20,811 chunks |
| Git commits | 13 |

---

## Phase status against the plan

| Phase | Status |
|---|---|
| 0 Discovery | Done |
| 1 Architecture & environment | Done |
| 2 Knowledge base & ingestion | Done |
| 3 Core RAG + report generation | Done *(report generation was unreachable until Phase 5 Step 0)* |
| 3.5 Retrieval quality *(added)* | Done |
| 4 Multimodal ingestion | Done, minus PMC figures |
| 5 Decision support | 4 of 6 steps; 1 blocked, 1 delivered via Q&A |
| 6 Explainability, auth, hardening | Steps 0–2 and 6 done; 3, 4, 5, 7 remaining |
| 7 Deployment & pilot | **Not started** |
| 8 Iteration & portfolio | **Not started** |

Roughly week 11 of a 14-week plan, and that is about where the calendar says
you should be.

---

## What is genuinely strong

**Almost every design decision traces to an observed failure.** That is rare,
and it is the most defensible thing in the repository. Not "we used a
cross-encoder because reranking is good practice" but "vector-only retrieval
never surfaced the passage answering the pneumothorax question, here is the
measurement, here is what changed." A reviewer can follow the reasoning to a
specific broken output.

**The tests assert what must NOT happen.** Most test suites check that code
does what it claims. A large share of these check silence and refusal: the
quality checker staying quiet on a valid report, the comparison prompt refusing
to call 8mm→9mm growth, the report prompt refusing to add an unstated normal,
the login endpoint taking the same time for an unknown account as a real one.
Those are the properties that actually matter here and the ones nobody writes
tests for.

**Retrieval was improved by measurement, not by intuition.**

| Configuration | keyword recall@5 |
|---|---|
| Vector only | 55.6% |
| + cross-encoder rerank | 72.2% |
| + hybrid BM25 | 77.8% |
| + contextual chunk headers | 100% |

**The security reasoning is real rather than cargo-culted.** No default
`JWT_SECRET`; a dummy hash burned on unknown emails so response time cannot
enumerate accounts; the user reloaded from the database on every request so a
deactivated account cannot ride a valid token for twelve hours; a route
enumeration test that fails closed for endpoints nobody has written yet.

---

## What is weaker than it looks

### 1. The headline metric is stale — this is the biggest gap

`contextual.json` shows **keyword recall@5 = 100%**, and that number appears in
the Phase 3.5 write-up. It was measured **before** the `<xref>` tail-truncation
bug was found, on a corpus where nearly every article was cut off at its first
citation marker.

The corpus is now roughly ten times larger in text. Those numbers describe a
system that no longer exists, and they could move in either direction — more
text means more to retrieve from, and also more to retrieve *wrongly* from.

**Fix:** re-run `eval/run_eval.py --save`. One command, and it either confirms
the result on real data or produces a more interesting story about what
truncation was hiding.

### 2. Zero frontend tests

3,070 lines with no automated coverage. The bugs found there this week were all
found by hand: the FileList emptied by a value reset, `<img>` unable to send an
auth header, an `accept` filter silently rejecting valid files. Each one was
invisible until someone clicked.

Not necessarily worth fixing now — but worth naming in the write-up rather than
letting a reviewer notice it.

### 3. Chat history does not survive a reload

Refresh the page and the conversation is gone. For a tool whose premise is
"draft, review, sign off", losing the draft to an accidental refresh is the
kind of thing a pilot user reports on day one. Reports persist; the
conversation around them does not.

### 4. Retrieval latency, ~5.2 seconds

Raised during Phase 3.5 and never resolved. It is the one quality a user feels
directly, and every other number improved while this one did not.

### 5. Thirteen commits for 22,000 lines

For a portfolio project the commit history *is* documentation — it is how a
reviewer sees the reasoning unfold. Thirteen large commits compress that into
"a lot of code appeared." The work is more impressive than the log makes it
look.

### 6. Three features are partial

- **PMC figures** — pipeline built and tested; ingestion stored PMIDs rather
  than PMCIDs, so URLs cannot be constructed. Diagnosed, not fixed.
- **Similar-case retrieval** — blocked on having no corpus of prior reports.
- **Differential diagnosis** — reachable through Q&A, no dedicated mode.

All three are defensible as written-up scope decisions. None is defensible as
an unexplained gap.

---

## The honest steer

**The code is stronger than the presentation.**

There is more engineering depth here than an internship project needs — and
that is fine, because the depth is the interesting part. The risk is where the
remaining time goes.

What a reviewer actually encounters is: a URL that works, a README, a demo, and
a write-up. Right now the project has none of those. Phases 7 and 8 exist for
exactly this, and they are the two that have not started.

**Suggested order from here:**

1. **Re-run the eval** (~5 min) — the current numbers are not measurements of
   the current system
2. **Finish auth** — registration plus ownership, so anyone can try it
3. **Deploy** (Phase 7) — a working URL is worth more than another feature
4. **Write it up** (Phase 8) — the failure-driven story is the strongest thing
   here and it is currently only in code comments
5. Rate limiting and error handling **before** the URL is public, not after

**What to stop doing:** adding features. Every phase from here on adds surface
area to something already substantial. The marginal feature is worth less than
the first deployment.

---

## One thing worth saying plainly

The most valuable artefact in this repository is not the RAG pipeline. It is
the record of things that went wrong and what was done about them:

- OCR reading `hyperlordotic` as `hypoiordotic`, inverting a clinical finding
- Retrieved literature outranking the patient's own report because of prompt
  position
- A stated 50% becoming "25-50%" from a background paper
- `_strip_noise` deleting the rest of every paragraph after each citation
  marker, silently, for the entire corpus
- Two contradictory prompt instructions where the longer one won

Most projects at this level do not have that record because most do not look
hard enough to find any of it. That is the thing to lead with.
