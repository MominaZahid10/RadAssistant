"""
Tests for the RAG orchestrator (Phase 3).

Focus: prompt assembly and the grounding scaffold. These are the load-bearing
parts — a prompt that silently loses its grounding rules produces confident,
fluent, ungrounded medical text, which is the single worst failure mode this
system has. It won't crash, and it won't look wrong.
"""

from collections import Counter

import pytest

from app.services.rag_service import (
    RAGService,
    RetrievedChunk,
    GROUNDING_SCAFFOLD,
    QA_SYSTEM_PROMPTS,
    REPORT_SYSTEM_PROMPT,
    RELEVANCE_THRESHOLD,
    OUT_OF_SCOPE_REPLY,
    normalise_citations,
    is_out_of_scope,
    merge_adjacent_chunks,
    cap_per_document,
)


@pytest.fixture
def rag():
    return RAGService()


def chunk(cid=1, score=0.8, text="Pneumothorax shows a visceral pleural line.", **kw):
    return RetrievedChunk(
        chunk_id=cid,
        text=text,
        score=score,
        document_title=kw.get("title", "Pneumothorax — Imaging"),
        source_type=kw.get("source_type", "textbook"),
        document_id=kw.get("document_id", "doc-1"),
        chunk_index=kw.get("chunk_index", 0),
    )


# ══════════════════════════════════════════════════════════════
# GROUNDING SCAFFOLD — present in EVERY mode
# ══════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "qa", "audience": "radiologist"},
        {"mode": "qa", "audience": "resident"},
        {"mode": "report"},
        {"mode": "qa", "audience": "nonsense-value"},
    ],
)
def test_grounding_scaffold_always_present(rag, kwargs):
    messages = rag.build_messages("q?", [chunk()], **kwargs)
    assert GROUNDING_SCAFFOLD in messages[0]["content"]


def test_grounding_scaffold_covers_the_four_hard_rules():
    text = GROUNDING_SCAFFOLD.lower()
    assert "only" in text                       # answer only from context
    assert "[1]" in GROUNDING_SCAFFOLD          # inline citation format
    assert "never assert a diagnosis" in text   # no diagnostic claims
    assert "training data" in text              # no parametric fill-in


def test_unknown_audience_falls_back_to_radiologist(rag):
    messages = rag.build_messages("q?", [chunk()], mode="qa", audience="astronaut")
    assert QA_SYSTEM_PROMPTS["radiologist"] in messages[0]["content"]


# ══════════════════════════════════════════════════════════════
# MESSAGE STRUCTURE
# ══════════════════════════════════════════════════════════════


def test_message_shape(rag):
    messages = rag.build_messages("What is a pneumothorax?", [chunk()])
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "What is a pneumothorax?"}


def test_audience_swaps_only_the_register(rag):
    rad = rag.build_messages("q?", [chunk()], audience="radiologist")[0]["content"]
    res = rag.build_messages("q?", [chunk()], audience="resident")[0]["content"]

    assert QA_SYSTEM_PROMPTS["radiologist"] in rad
    assert QA_SYSTEM_PROMPTS["resident"] in res
    assert rad != res
    # ...but the grounding rules are identical in both.
    assert GROUNDING_SCAFFOLD in rad and GROUNDING_SCAFFOLD in res


def test_report_mode_uses_clinical_prompt(rag):
    content = rag.build_messages("Describe findings", [chunk()], mode="report")[0]["content"]
    assert REPORT_SYSTEM_PROMPT in content
    assert QA_SYSTEM_PROMPTS["radiologist"] not in content


# ══════════════════════════════════════════════════════════════
# CONTEXT FORMATTING — citation IDs must line up
# ══════════════════════════════════════════════════════════════


def test_each_chunk_gets_a_numbered_source_label(rag):
    chunks = [chunk(cid=i, text=f"Finding {i}") for i in (1, 2, 3)]
    content = rag.build_messages("q?", chunks)[0]["content"]

    for i in (1, 2, 3):
        assert f"[Source {i}]" in content
        assert f"Finding {i}" in content


def test_source_header_includes_title_type_and_score(rag):
    content = rag.build_messages("q?", [chunk(score=0.87)])[0]["content"]
    assert "Pneumothorax — Imaging" in content
    assert "textbook" in content
    assert "0.87" in content


def test_empty_context_is_stated_explicitly(rag):
    """
    With no context the model must be told so plainly — otherwise it answers
    from training data, which is exactly what the grounding rules forbid.
    """
    content = rag.build_messages("q?", [])[0]["content"]
    assert "No relevant information" in content


def test_chunk_ids_are_one_based_to_match_citations(rag):
    """
    The prompt tells the model to cite [1], [2]. If chunk_ids were 0-based the
    first citation would be [0] and every reference would be off by one.
    """
    content = rag.build_messages("q?", [chunk(cid=1), chunk(cid=2)])[0]["content"]
    assert "[Source 1]" in content
    assert "[Source 0]" not in content


# ══════════════════════════════════════════════════════════════
# RELEVANCE THRESHOLD
# ══════════════════════════════════════════════════════════════


def test_no_context_is_not_relevant(rag):
    assert rag._has_relevant_context([]) is False


def test_all_weak_scores_are_not_relevant(rag):
    weak = [chunk(cid=i, score=RELEVANCE_THRESHOLD - 0.01) for i in (1, 2)]
    assert rag._has_relevant_context(weak) is False


def test_one_strong_score_is_enough(rag):
    mixed = [
        chunk(cid=1, score=RELEVANCE_THRESHOLD - 0.1),
        chunk(cid=2, score=RELEVANCE_THRESHOLD + 0.1),
    ]
    assert rag._has_relevant_context(mixed) is True


def test_score_exactly_at_threshold_counts(rag):
    assert rag._has_relevant_context([chunk(score=RELEVANCE_THRESHOLD)]) is True


def test_threshold_is_in_a_sane_range():
    """
    Too low and noise gets treated as evidence; too high and the system
    refuses to answer questions it actually has material for.
    """
    assert 0.2 <= RELEVANCE_THRESHOLD <= 0.6


# ══════════════════════════════════════════════════════════════
# WEAK-CONTEXT SHORT CIRCUIT
# ══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_weak_context_answers_without_calling_the_llm(rag, monkeypatch):
    """
    Below threshold we must return a canned 'I don't know' rather than letting
    the model improvise from noise — and we shouldn't pay for a token either.
    """
    weak = [chunk(score=0.05)]

    async def fake_retrieve(*a, **k):
        return weak

    def explode(*a, **k):
        raise AssertionError("LLM must not be called for weak context")

    monkeypatch.setattr(rag, "retrieve_context", fake_retrieve)
    monkeypatch.setattr("app.services.rag_service.llm_service.generate", explode)

    result = await rag.answer("something unrelated")

    assert "don't have enough information" in result.answer
    assert result.sources == weak


# ══════════════════════════════════════════════════════════════
# CITATION NORMALISATION
# ══════════════════════════════════════════════════════════════
# Regression tests for a bug caught in a live SSE stream: gpt-oss-120b
# emitted 【3】 (CJK lenticular brackets) instead of [3]. Nothing errored,
# the prose read fine, but no citation was parseable — the Evidence Panel
# silently did nothing.
# ══════════════════════════════════════════════════════════════


def test_lenticular_brackets_become_ascii():
    assert normalise_citations("markings are absent【3】.") == "markings are absent[3]."


def test_fullwidth_brackets_become_ascii():
    assert normalise_citations("finding［2］") == "finding[2]"


@pytest.mark.parametrize("token", ["【", "3", "】"])
def test_normalisation_is_safe_per_token(token):
    """
    The brackets arrive as SEPARATE stream tokens, so normalisation must work
    on a single character with no surrounding context.
    """
    out = normalise_citations(token)
    assert "【" not in out and "】" not in out


def test_streamed_tokens_reassemble_into_ascii_citation():
    """Replays the exact token sequence observed in the live stream."""
    observed = ["no", " vessels", " or", " bron", "chi", ")", "【", "3", "】", " |\n"]
    assert "".join(normalise_citations(t) for t in observed).endswith(")[3] |\n")


def test_normalisation_leaves_normal_text_alone():
    text = "The visceral pleural line is visible [1], with no markings [2]."
    assert normalise_citations(text) == text


def test_normalisation_handles_empty():
    assert normalise_citations("") == ""


# ══════════════════════════════════════════════════════════════
# OUT-OF-SCOPE DETECTION
# ══════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "query",
    [
        "What's the weather today?",
        "tell me a joke",
        "write me a poem about the sea",
        "what is the capital of France",
        "give me a recipe for pasta",
    ],
)
def test_obvious_non_medical_queries_are_rejected(query):
    assert is_out_of_scope(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "What are the radiographic findings of pneumothorax?",
        "How is pulmonary embolism diagnosed on CTPA?",
        "Explain the Fleischner criteria",
        "chest x-ray interpretation approach",
        "differential for a solitary lung nodule",
        # Adversarial: non-medical trigger word inside a real clinical question.
        "Does cold weather affect chest x-ray interpretation?",
        "Write a report for this CT scan",
        # Vague but plausibly clinical — must fall through, not be refused.
        "what should I look for first?",
        "summarise the key findings",
    ],
)
def test_clinical_queries_are_never_rejected(query):
    """
    A false positive refuses a real clinical question — far worse than
    wasting one API call. This test guards the conservative bias.
    """
    assert is_out_of_scope(query) is False


@pytest.mark.asyncio
async def test_out_of_scope_skips_retrieval_and_llm(rag, monkeypatch):
    def explode(*a, **k):
        raise AssertionError("must not run for out-of-scope queries")

    monkeypatch.setattr(rag, "retrieve_context", explode)
    monkeypatch.setattr("app.services.rag_service.llm_service.generate", explode)

    result = await rag.answer("what's the weather today?")

    assert result.answer == OUT_OF_SCOPE_REPLY
    assert result.sources == []


# ══════════════════════════════════════════════════════════════
# CONTEXT DEDUPLICATION
# ══════════════════════════════════════════════════════════════


def test_adjacent_chunks_from_same_document_merge():
    a = chunk(cid=1, score=0.9, text="Air in the pleural space. ", document_id="d1", chunk_index=4)
    b = chunk(cid=2, score=0.8, text="The lung collapses.", document_id="d1", chunk_index=5)

    merged = merge_adjacent_chunks([a, b])

    assert len(merged) == 1
    assert "pleural space" in merged[0].text and "lung collapses" in merged[0].text


def test_overlap_is_not_duplicated_when_merging():
    """Chunks overlap by design; the shared region must appear once."""
    overlap = "missed on chest imaging. "
    a = chunk(cid=1, text=f"It must not be {overlap}", document_id="d1", chunk_index=0)
    b = chunk(cid=2, text=f"{overlap}Types of pneumothorax:", document_id="d1", chunk_index=1)

    merged = merge_adjacent_chunks([a, b])
    assert merged[0].text.count("missed on chest imaging") == 1


def test_non_adjacent_chunks_are_kept_separate():
    a = chunk(cid=1, score=0.9, document_id="d1", chunk_index=0)
    b = chunk(cid=2, score=0.8, document_id="d1", chunk_index=7)
    assert len(merge_adjacent_chunks([a, b])) == 2


def test_chunks_from_different_documents_never_merge():
    a = chunk(cid=1, score=0.9, document_id="d1", chunk_index=4)
    b = chunk(cid=2, score=0.8, document_id="d2", chunk_index=5)
    assert len(merge_adjacent_chunks([a, b])) == 2


def test_merged_chunks_are_renumbered_contiguously():
    """
    The prompt tells the model to cite [1]..[N]. After merging, IDs must be
    1..N with no gaps, or citations reference sources that don't exist.
    """
    chunks = [
        chunk(cid=1, score=0.5, document_id="d1", chunk_index=0),
        chunk(cid=2, score=0.9, document_id="d1", chunk_index=1),
        chunk(cid=3, score=0.7, document_id="d2", chunk_index=0),
    ]
    merged = merge_adjacent_chunks(chunks)
    assert [c.chunk_id for c in merged] == list(range(1, len(merged) + 1))


def test_merged_chunks_are_ordered_best_first():
    chunks = [
        chunk(cid=1, score=0.4, document_id="d1", chunk_index=0),
        chunk(cid=2, score=0.95, document_id="d2", chunk_index=0),
    ]
    merged = merge_adjacent_chunks(chunks)
    assert merged[0].score == 0.95
    assert merged[0].chunk_id == 1


def test_merge_keeps_the_best_score_of_the_group():
    a = chunk(cid=1, score=0.6, document_id="d1", chunk_index=0)
    b = chunk(cid=2, score=0.9, document_id="d1", chunk_index=1)
    assert merge_adjacent_chunks([a, b])[0].score == 0.9


def test_cap_per_document_limits_any_single_source():
    """
    Regression test for a real retrieval failure on a 232-document corpus:
    two term-dense papers took 10 of the top 20 slots for "radiographic
    findings of pneumothorax", pushing the passage that actually describes
    the finding out of the context window. The model then correctly reported
    it had nothing to answer with.
    """
    chunks = (
        [chunk(cid=i, score=0.83 - i * 0.001, document_id="ultrasound", chunk_index=i) for i in range(6)]
        + [chunk(cid=i, score=0.82 - i * 0.001, document_id="liposuction", chunk_index=i) for i in range(4)]
        + [chunk(cid=i, score=0.79 - i * 0.001, document_id="curated", chunk_index=i) for i in range(2)]
    )
    chunks.sort(key=lambda c: -c.score)

    capped = cap_per_document(chunks, 3)
    counts = Counter(c.document_id for c in capped)

    assert counts["ultrasound"] == 3
    assert counts["liposuction"] == 3
    assert counts["curated"] == 2, "under-represented source must not be trimmed"


def test_cap_preserves_relevance_order():
    chunks = [chunk(cid=i, score=1.0 - i * 0.1, document_id=f"d{i % 3}", chunk_index=i) for i in range(9)]
    capped = cap_per_document(chunks, 2)
    assert capped == sorted(capped, key=lambda c: -c.score)


def test_cap_keeps_the_highest_scoring_chunks_of_each_document():
    """When trimming, the chunks dropped must be the weakest ones."""
    chunks = [chunk(cid=i, score=0.9 - i * 0.1, document_id="d1", chunk_index=i) for i in range(5)]
    capped = cap_per_document(chunks, 2)
    assert [c.chunk_index for c in capped] == [0, 1]


def test_cap_does_not_group_chunks_without_a_document_id():
    """Unkeyed chunks must not be collapsed together as if one document."""
    chunks = [
        RetrievedChunk(chunk_id=i, text=f"t{i}", score=0.9, document_id=None)
        for i in range(5)
    ]
    assert len(cap_per_document(chunks, 2)) == 5


def test_cap_disabled_by_zero():
    chunks = [chunk(cid=i, document_id="d1", chunk_index=i) for i in range(5)]
    assert cap_per_document(chunks, 0) == chunks


def test_merge_handles_trivial_inputs():
    assert merge_adjacent_chunks([]) == []
    single = [chunk()]
    assert merge_adjacent_chunks(single) == single


@pytest.mark.asyncio
async def test_weak_context_short_circuits_streaming_too(rag, monkeypatch):
    async def fake_retrieve(*a, **k):
        return [chunk(score=0.01)]

    def explode(*a, **k):
        raise AssertionError("LLM must not be called for weak context")

    monkeypatch.setattr(rag, "retrieve_context", fake_retrieve)
    monkeypatch.setattr("app.services.rag_service.llm_service.generate_stream", explode)

    _sources, stream = await rag.answer_stream("something unrelated")
    text = "".join([t async for t in stream])

    assert "don't have enough information" in text
