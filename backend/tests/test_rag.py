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
# GROUNDING SCAFFOLD — every mode is grounded, but not identically
# ══════════════════════════════════════════════════════════════
#
# ⚠️  THIS TEST USED TO ASSERT GROUNDING_SCAFFOLD IN *EVERY* MODE,
# INCLUDING report, AND THAT WAS THE BUG IT WAS PROTECTING.
#
# The general scaffold mandates a citation on every factual claim. Report mode
# forbids citations in the report body — a bare number after a finding reads
# as a severity grade in a medical record. Sending both produced drafts saying
# "Mild cardiomegaly. 1", because the longer, more specific instruction won.
#
# Two of the scaffold's other rules are also wrong for a report: "answer ONLY
# from the CONTEXT" (the content comes from the dictation) and "frame findings
# as differential considerations" (a report states findings).
#
# So report mode has its own scaffold. The invariant being protected was never
# "this exact string appears everywhere" — it was "no mode is ever ungrounded",
# and that still holds.


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "qa", "audience": "radiologist"},
        {"mode": "qa", "audience": "resident"},
        {"mode": "qa", "audience": "nonsense-value"},
    ],
)
def test_qa_modes_get_the_general_grounding_scaffold(rag, kwargs):
    messages = rag.build_messages("q?", [chunk()], **kwargs)
    assert GROUNDING_SCAFFOLD in messages[0]["content"]


def test_report_mode_gets_the_report_scaffold_instead(rag):
    from app.services.rag_service import REPORT_GROUNDING_SCAFFOLD

    system = rag.build_messages("q?", [chunk()], mode="report")[0]["content"]
    assert REPORT_GROUNDING_SCAFFOLD in system
    assert GROUNDING_SCAFFOLD not in system


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "qa", "audience": "radiologist"},
        {"mode": "qa", "audience": "resident"},
        {"mode": "report"},
        {"mode": "qa", "audience": "nonsense-value"},
    ],
)
def test_no_mode_is_ever_ungrounded(rag, kwargs):
    """
    The real invariant. Whichever scaffold applies, every mode must be told
    that the retrieved context does not get to invent clinical content.
    """
    from app.services.rag_service import REPORT_GROUNDING_SCAFFOLD

    system = rag.build_messages("q?", [chunk()], **kwargs)[0]["content"]
    assert "GROUNDING RULES — these are non-negotiable:" in system
    assert (
        GROUNDING_SCAFFOLD in system or REPORT_GROUNDING_SCAFFOLD in system
    )


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


# ══════════════════════════════════════════════════════════════
# RETRIEVAL QUERY FROM AN UPLOADED DOCUMENT
# ══════════════════════════════════════════════════════════════
#
# ⚠️  THE FAILURE THIS PREVENTS.
# A clinician uploaded a lumbar spine report and typed a generic instruction.
# Retrieval embedded the INSTRUCTION, so all twelve sources returned were
# papers about the practice of radiology reporting — "Revealing the most
# common reporting errors", "The reporting quality of NLP studies". Not one
# concerned osteopenia, vertebral compression or spinal curvature, and the
# model cited [1] for every claim because no source supported any of them.
#
# The answer itself was safe: the document is the primary source in the prompt
# regardless. But an evidence panel of twelve irrelevant papers is worse than
# no evidence panel — it's a traceability claim the sources don't back.

from app.services import rag_service as rag_module
from app.services.lexical_service import LexicalIndex

_SPINE_CORPUS = [
    {"point_id": "r1", "document_id": "m1", "title": "Common reporting errors",
     "text": "Radiology report proofreading. The resident writes a preliminary report "
             "and the attending edits the report. Report findings and report impression "
             "sections were compared across reports."},
    {"point_id": "c1", "document_id": "m2", "title": "Vertebral compression fracture",
     "text": "Anterior wedge deformity of the vertebral body indicates a compression "
             "fracture. Loss of anterior vertebral body height is graded by severity."},
    {"point_id": "c2", "document_id": "m3", "title": "Osteopenia and osteoporosis",
     "text": "Osteopenia describes reduced bone mineral density. Severe postmenopausal "
             "osteoporosis of the lumbar spine and pelvis predisposes to fracture."},
]

_OCR_REPORT = (
    "The lumbar spine ts hyperiordotic. Marked osteopenia is noted throughout the "
    "lumbar spine and peivis. An anterior wedge deformity of Tz with a 50% loss of "
    "anterior vertebral body height is noted. IMPRESSION: Severe post-menopausal "
    "osteoporosis. Tracompression tracture."
)


@pytest.fixture
def spine_index(monkeypatch):
    """A built lexical index standing in for the live corpus."""
    idx = LexicalIndex()
    idx.build(_SPINE_CORPUS)
    monkeypatch.setattr(rag_module, "lexical_index", idx)
    monkeypatch.setattr(
        rag_module.RAGService, "_ensure_lexical_index", staticmethod(lambda: None)
    )
    return idx


def test_document_query_searches_the_document_not_the_instruction(rag, spine_index):
    derived = rag._build_document_query(
        "Please review the attached report.", _OCR_REPORT
    ).lower()

    # What the report is actually about.
    for term in ("osteopenia", "vertebral", "wedge", "osteoporosis"):
        assert term in derived, f"{term!r} missing from derived query: {derived}"


def test_document_query_excludes_ocr_misreads(rag, spine_index):
    """
    Misread words are the highest-IDF tokens in the document — nothing is
    rarer than a word that doesn't exist. They must not steer retrieval.
    """
    derived = rag._build_document_query("Review this.", _OCR_REPORT).lower()
    for garbage in ("hyperiordotic", "peivis", "tracompression", "tracture"):
        assert garbage not in derived


def test_document_query_keeps_the_users_own_question(rag, spine_index):
    """
    A specific question is intent, not boilerplate, and must still steer
    retrieval alongside the document.
    """
    derived = rag._build_document_query(
        "What does the anterior height loss imply for management?", _OCR_REPORT
    ).lower()
    assert "management" in derived


def test_document_query_falls_back_when_index_unavailable(rag, monkeypatch):
    """
    Hybrid disabled, empty corpus, or a build failure. Degrades to the
    document's opening lines — still the subject matter, never a crash.
    """
    monkeypatch.setattr(rag_module, "lexical_index", LexicalIndex())   # unbuilt
    monkeypatch.setattr(
        rag_module.RAGService, "_ensure_lexical_index", staticmethod(lambda: None)
    )
    derived = rag._build_document_query("Review this.", _OCR_REPORT)
    assert "Review this." in derived
    assert "lumbar spine" in derived


def test_document_query_survives_a_broken_index(rag, monkeypatch):
    """Retrieval quality is an enhancement; a failure here must not raise."""
    def _boom():
        raise RuntimeError("Qdrant unreachable")

    monkeypatch.setattr(
        rag_module.RAGService, "_ensure_lexical_index", staticmethod(_boom)
    )
    derived = rag._build_document_query("Review this.", _OCR_REPORT)
    assert derived.startswith("Review this.")


def test_plain_query_is_left_alone(rag, spine_index):
    """With no attachment, nothing about retrieval changes."""
    assert rag._build_document_query("", "") == ""


@pytest.mark.asyncio
async def test_retrieve_context_searches_with_the_derived_query(
    rag, spine_index, monkeypatch
):
    """End of the wire: the derived query is what actually reaches the embedder."""
    seen: list[str] = []

    def _fake_retrieve(query, limit, source_type):
        seen.append(query)
        return []

    monkeypatch.setattr(rag, "_retrieve_sync", _fake_retrieve)

    await rag.retrieve_context(
        "Please review the attached report.", attached_text=_OCR_REPORT
    )

    assert len(seen) == 1
    assert "osteopenia" in seen[0].lower()


@pytest.mark.asyncio
async def test_retrieve_context_without_attachment_is_unchanged(rag, monkeypatch):
    seen: list[str] = []

    def _fake_retrieve(query, limit, source_type):
        seen.append(query)
        return []

    monkeypatch.setattr(rag, "_retrieve_sync", _fake_retrieve)
    await rag.retrieve_context("What are the findings of pneumothorax?")

    assert seen == ["What are the findings of pneumothorax?"]


# ══════════════════════════════════════════════════════════════
# REPORT ANALYSIS — repair harmless typos, flag only what changes care
# ══════════════════════════════════════════════════════════════
#
# ⚠️  TWO FAILURES, IN OPPOSITE DIRECTIONS, ONE WEEK APART.
#
# First the model expanded the OCR fragment "Tracompression tracture" into
# "traumatic compression fracture" — inventing a MECHANISM on a study whose own
# impression was "severe post-menopausal osteoporosis". Traumatic and
# osteoporotic fractures are worked up differently.
#
# Tightening the rule then over-corrected: the model began footnoting every
# misspelling, including "ts" for "is" and "solt" for "soft". The answer became
# a list of OCR complaints with the medicine buried underneath — read back to
# the clinician who wrote the report and already knows what it says.
#
# The rule is therefore not "how confident am I?" but "if I am wrong, does the
# clinician do something different?" Direction, laterality, level, number and
# mechanism change care. Spelling does not.

from app.services.rag_service import REPORT_ANALYSIS_PROMPT


def test_harmless_typos_are_repaired_silently():
    p = REPORT_ANALYSIS_PROMPT
    assert "FIX THEM SILENTLY" in p
    # The specific words that were being read back to the clinician.
    for word in ('"ts"', '"solt tissues"', '"peivis"'):
        assert word in p, f"{word} must stay as a worked example of a silent fix"
    assert "Do not mention it" in p


def test_load_bearing_ambiguities_are_still_flagged():
    """The over-correction must not swing back past the original bug."""
    p = REPORT_ANALYSIS_PROMPT
    assert "CLINICALLY LOAD-BEARING" in p
    for axis in ("direction", "laterality", "level", "numbers", "mechanism"):
        assert axis in p
    assert "hyper- vs hypo-" in p
    assert "Tracompression" in p, "the observed failure must stay as an example"


def test_the_test_is_consequence_not_confidence():
    assert 'not "am I confident?"' in REPORT_ANALYSIS_PROMPT
    assert "does the\n   clinician do something different?" in REPORT_ANALYSIS_PROMPT


def test_text_quality_section_is_conditional():
    """It appeared unconditionally and dominated the answer."""
    p = REPORT_ANALYSIS_PROMPT
    assert "ONLY IF bin (b) is non-empty" in p
    assert "omit the section entirely" in p
    assert "Lead with the medicine" in p


def test_report_prompt_forbids_inventing_patient_details():
    p = REPORT_ANALYSIS_PROMPT.lower()
    assert "no age, no sex" in p
    assert "mechanism of injury" in p


def test_report_prompt_prefers_an_empty_background_to_a_loose_citation():
    assert "leave the\nbackground blank" in REPORT_ANALYSIS_PROMPT


def test_report_prompt_retains_the_primary_source_rules():
    """Regression guard: the Phase 4 fixes must not be edited away."""
    p = REPORT_ANALYSIS_PROMPT
    assert "THE UPLOADED DOCUMENT IS THE PRIMARY SOURCE" in p
    assert "NEVER contradict the uploaded document" in p
    assert "NEVER alter a number" in p
    assert "BACKGROUND ONLY" in p


def test_report_analysis_prompt_selected_when_a_document_is_attached(rag):
    messages = rag.build_messages(
        "what are the findings of this report?",
        [chunk()],
        attached_text="The lumbar spine is hyperlordotic.",
        attached_warnings=["Low resolution (424px)."],
    )
    system = messages[0]["content"]
    assert "FIX THEM SILENTLY" in system
    assert "Low resolution (424px)." in system


def test_ocr_caveat_does_not_contradict_rule_five(rag):
    """
    The caveat attached to the warnings block used to say 'quote it as-is and
    say it is unclear' for ANY garbled word — the exact instruction that
    produced the footnote spam. It must defer to rule 5 instead.
    """
    system = rag.build_messages(
        "review this", [chunk()],
        attached_text="text", attached_warnings=["Low OCR confidence (41%)."],
    )[0]["content"]
    assert "Apply rule 5" in system
    assert "quote it as-is" not in system


def test_report_prompt_blocks_acuity_and_symptom_leakage():
    """
    ⚠️  OBSERVED after the vision model fixed the transcription. With clean
    text the model still wrote "acute compression fracture" and "the current
    pain" — neither in the report. Both came from the background literature,
    which is overwhelmingly cohorts of ACUTE fractures in symptomatic
    patients. Acute vs chronic changes management; the report recorded no
    symptom at all.
    """
    p = REPORT_ANALYSIS_PROMPT
    assert "ACUITY" in p
    assert '"acute", "chronic", "recent" or "old"' in p
    assert "SYMPTOMS" in p
    assert "the current pain" in p
    assert "The literature describes OTHER patients" in p
