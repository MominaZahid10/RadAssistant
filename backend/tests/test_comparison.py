"""
Tests for prior-study comparison (Phase 5, Step 4).

Closes the outstanding Phase 4 deliverable — *"prior-report upload and
comparison against current findings"* — as well as Phase 5's timeline item.

⚠️  THE JUDGEMENT THIS FEATURE MUST REFUSE TO MAKE.
A nodule reported as 8mm previously and 9mm now is EITHER interval growth OR
inter-reader variation OR a different axis measured on a different slice. Two
reports cannot distinguish these. 1mm on a small nodule is within measurement
variability, and calling it growth can trigger a biopsy.

So the measurements are paired and differenced deterministically — a language
model is least reliable with numbers and an error costs most there — and the
prompt is forbidden from characterising the result.
"""

import pytest

from app.services.comparison_service import (
    compare_measurements,
    extract_measurements,
    format_facts,
)
from app.services.rag_service import (
    COMPARISON_GROUNDING_SCAFFOLD,
    COMPARISON_PROMPT,
    GROUNDING_SCAFFOLD,
    RAGService,
    RetrievedChunk,
)


PRIOR = "Nodule measuring 8 mm in the right upper lobe. No pleural effusion."
CURRENT = "Nodule now measures 9 mm in the right upper lobe. No pleural effusion."


# ══════════════════════════════════════════════════════════════
# EXTRACTION
# ══════════════════════════════════════════════════════════════


def test_measurements_are_extracted_with_their_context():
    found = extract_measurements(PRIOR)
    assert len(found) == 1
    assert found[0].value == 8.0
    assert found[0].unit == "mm"
    assert "right upper lobe" in found[0].context


@pytest.mark.parametrize("text,value,unit", [
    ("a 12mm lesion", 12.0, "mm"),
    ("a 1.5 cm lesion", 1.5, "cm"),
    ("50% height loss", 50.0, "%"),
    ("40 percent stenosis", 40.0, "%"),
])
def test_units_are_recognised(text, value, unit):
    m = extract_measurements(text)[0]
    assert (m.value, m.unit) == (value, unit)


def test_text_without_measurements_yields_nothing():
    assert extract_measurements("Clear lung fields. No effusion.") == []


# ══════════════════════════════════════════════════════════════
# PAIRING
# ══════════════════════════════════════════════════════════════


def test_the_same_lesion_is_paired_across_studies():
    facts = compare_measurements(PRIOR, CURRENT)
    assert len(facts.pairs) == 1
    pair = facts.pairs[0]
    assert pair.prior.value == 8.0
    assert pair.current.value == 9.0
    assert pair.delta_mm == 1.0


def test_unrelated_findings_are_not_paired():
    """
    ⚠️  A WRONG PAIR IS WORSE THAN NO PAIR.
    Matching a liver lesion to a lung nodule and printing a "difference"
    between them fabricates a comparison. Unmatched measurements are listed
    separately, which prompts the radiologist to look rather than misleading
    them.
    """
    facts = compare_measurements(
        "Hepatic lesion measuring 20 mm in segment VI.",
        "Pulmonary nodule measuring 4 mm in the left lower lobe.",
    )
    assert facts.pairs == []
    assert len(facts.prior_only) == 1
    assert len(facts.current_only) == 1


def test_identical_measurements_are_recognised_as_unchanged():
    facts = compare_measurements(PRIOR, PRIOR)
    assert facts.pairs[0].identical is True
    assert facts.pairs[0].delta_mm == 0.0


def test_cm_and_mm_are_converted_and_the_reader_is_told():
    """
    Real and common: 1.2cm previously, 13mm now. Converting is safe because
    both units are printed — but a silent conversion is where a tenfold error
    would hide, so it is announced.
    """
    facts = compare_measurements(
        "Nodule measuring 1.2 cm in the right upper lobe.",
        "Nodule measuring 13 mm in the right upper lobe.",
    )
    assert facts.pairs[0].delta_mm == 1.0
    assert any("Units differ" in w for w in facts.warnings)


def test_percentages_are_not_differenced_against_lengths():
    """A percentage and a millimetre are not the same kind of quantity."""
    facts = compare_measurements(
        "Vertebral body height loss of 50%.",
        "Vertebral body height loss measuring 8 mm.",
    )
    for pair in facts.pairs:
        assert pair.delta_mm is None
        assert pair.comparable is False


def test_a_finding_only_in_the_prior_is_reported_separately():
    facts = compare_measurements(
        "Nodule measuring 8 mm in the right upper lobe.",
        "Clear lung fields.",
    )
    assert len(facts.prior_only) == 1
    assert facts.pairs == []


# ══════════════════════════════════════════════════════════════
# THE FACTS BLOCK — wording matters, the model copies it
# ══════════════════════════════════════════════════════════════


def test_facts_state_both_values_without_characterising_them():
    text = format_facts(compare_measurements(PRIOR, CURRENT))
    assert "8 mm previously" in text and "9 mm now" in text
    # "1mm larger as printed" is a statement about the two numbers.
    assert "as printed" in text
    # None of these may appear — the model copies the prompt's register.
    for forbidden in ("growth", "progression", "worsen", "improve", "stable disease"):
        assert forbidden not in text.lower()


def test_facts_carry_the_uncertainty_note():
    text = format_facts(compare_measurements(PRIOR, CURRENT))
    assert "NOT established by these documents" in text


def test_facts_tell_the_model_not_to_recompute():
    """The arithmetic is done; the model's job is to narrate it."""
    text = format_facts(compare_measurements(PRIOR, CURRENT))
    assert "do not recompute" in text.lower()


def test_empty_comparison_produces_no_block():
    assert format_facts(compare_measurements("Clear lungs.", "Clear lungs.")) == ""


# ══════════════════════════════════════════════════════════════
# THE PROMPT
# ══════════════════════════════════════════════════════════════


def test_prompt_forbids_characterising_a_difference():
    """The single most important rule in this feature."""
    p = COMPARISON_PROMPT
    assert "NEVER CHARACTERISE A DIFFERENCE" in p
    assert "1mm interval growth" in p          # the worked counter-example
    assert "inter-reader variation" in p


def test_prompt_distinguishes_absence_from_resolution():
    """
    A finding missing from the current report may have resolved, or may
    simply not have been mentioned — a film reported for one question often
    says nothing about anything else.
    """
    p = COMPARISON_PROMPT
    assert "ABSENCE IS NOT RESOLUTION" in p
    assert "not mentioned in the current report" in p


def test_prompt_requires_both_reports_quoted():
    assert "QUOTE BOTH REPORTS" in COMPARISON_PROMPT


def test_prompt_defers_interval_change_to_the_images():
    assert "requires\nradiologist review of the images" in COMPARISON_PROMPT


# ══════════════════════════════════════════════════════════════
# WIRING
# ══════════════════════════════════════════════════════════════


def _system(**kwargs) -> str:
    return RAGService().build_messages(
        CURRENT,
        [RetrievedChunk(chunk_id=1, text="Nodule follow-up guidance.",
                        score=0.7, document_title="Fleischner")],
        **kwargs,
    )[0]["content"]


def test_comparison_mode_selects_the_comparison_prompt():
    assert COMPARISON_PROMPT in _system(mode="comparison", prior_text=PRIOR)


def test_prior_text_alone_is_enough_to_trigger_comparison():
    """A caller sending a prior study clearly wants a comparison."""
    assert COMPARISON_PROMPT in _system(prior_text=PRIOR)


def test_comparison_beats_the_attached_document_branch():
    """
    ⚠️  ORDER MATTERS IN build_messages.
    A comparison arrives WITH attached_text (the current study read from an
    upload). If the attachment branch ran first it would swallow the request
    and the model would analyse one document instead of comparing two.
    """
    system = _system(mode="comparison", prior_text=PRIOR, attached_text=CURRENT)
    assert COMPARISON_PROMPT in system


def test_the_computed_measurements_reach_the_prompt():
    system = _system(mode="comparison", prior_text=PRIOR)
    assert "MEASUREMENTS ESTABLISHED BY DIRECT COMPARISON" in system
    assert "8 mm previously" in system


def test_comparison_gets_its_own_scaffold():
    """
    The general scaffold says "answer ONLY from the CONTEXT" — wrong here,
    where the content comes from two patient documents and the context
    describes other patients entirely.
    """
    system = _system(mode="comparison", prior_text=PRIOR)
    assert COMPARISON_GROUNDING_SCAFFOLD in system
    assert GROUNDING_SCAFFOLD not in system


def test_comparison_scaffold_forbids_citing_a_difference():
    assert "no paper knows this\n   patient" in COMPARISON_GROUNDING_SCAFFOLD


def test_prior_study_is_labelled_and_delimited():
    system = _system(mode="comparison", prior_text=PRIOR)
    assert "PRIOR STUDY — for comparison" in system
    assert "END OF PRIOR STUDY" in system


def test_qa_mode_is_untouched_by_all_of_this():
    """Regression guard: the default path must not have changed."""
    assert GROUNDING_SCAFFOLD in _system(mode="qa")
    assert COMPARISON_PROMPT not in _system(mode="qa")


# ══════════════════════════════════════════════════════════════
# REGRESSION: the two bugs the tests above caught
# ══════════════════════════════════════════════════════════════


def test_a_decimal_point_does_not_truncate_context():
    """
    ⚠️  SILENT PAIRING FAILURE.
    Splitting sentences on a bare "." cut "Nodule measuring 1.2 cm in the
    right upper lobe" down to "Nodule measuring 1". The context words
    collapsed to {"nodule"}, similarity fell below the pairing threshold, and
    every decimal measurement stopped matching its follow-up — on reports
    where decimals are everywhere. Nothing raised.
    """
    m = extract_measurements("Nodule measuring 1.2 cm in the right upper lobe.")[0]
    assert "right upper lobe" in m.context


def test_sentences_still_split_on_real_full_stops():
    """The fix must not merge separate findings into one context."""
    text = "Nodule measuring 8 mm in the lung. Hepatic cyst measuring 20 mm."
    first, second = extract_measurements(text)
    assert "Hepatic" not in first.context
    assert "lung" not in second.context


def test_percentages_are_extracted_at_all():
    """
    `(?:mm|cm|%)\\b` never matched a percentage: a word boundary needs a word
    character, and the char after '%' is a space. The same trap caught the
    quality checker's 50% rule.
    """
    assert extract_measurements("50% height loss")[0].unit == "%"
    assert extract_measurements("Anterior height loss of 50%.")[0].value == 50.0


# ══════════════════════════════════════════════════════════════
# STUDY MISMATCH — the wrong prior selected
# ══════════════════════════════════════════════════════════════
#
# ⚠️  OBSERVED IN THE FIRST REAL RUN.
# A lumbar spine report was compared against dictated chest findings. Every
# item landed in "New" and "Not mentioned now" — which reads as a dramatic
# interval change rather than two unrelated documents. Studies do get
# mis-selected from a worklist, and this is what that failure looks like.

SPINE = (
    "The lumbar spine is hyperlordotic. Marked osteopenia is noted throughout "
    "the lumbar spine and pelvis. An anterior wedge deformity of T12 with a "
    "50% loss of anterior vertebral body height is noted."
)
CHEST = (
    "Nodule now measures 9 mm in the right upper lobe. Small left pleural "
    "effusion. Heart size within normal limits."
)


def test_unrelated_studies_are_flagged():
    facts = compare_measurements(SPINE, CHEST)
    assert facts.mismatch_warning
    assert "DIFFERENT studies" in facts.mismatch_warning


def test_the_mismatch_warning_leads_the_facts_block():
    """
    It has to be the first thing the model reads, or the comparison gets
    narrated confidently and the caveat arrives as a footnote nobody reaches.
    """
    text = format_facts(compare_measurements(SPINE, CHEST))
    assert text.startswith("⚠️  STUDY MISMATCH")
    assert "SAY THIS FIRST" in text


def test_related_studies_are_not_flagged():
    """
    ⚠️  A FALSE MISMATCH WARNING WOULD UNDERMINE EVERY LEGITIMATE COMPARISON.
    Two reports of the same region, written months apart by different readers,
    still share their anatomy.
    """
    facts = compare_measurements(
        "Nodule measuring 8 mm in the right upper lobe. No pleural effusion. "
        "Heart size normal.",
        CHEST,
    )
    assert facts.mismatch_warning == ""


def test_mismatch_survives_when_neither_report_has_measurements():
    """
    The case where the reader has no other signal at all that the wrong prior
    was picked — so the warning must not be dropped with the empty block.
    """
    text = format_facts(compare_measurements(
        "The lumbar spine is hyperlordotic with marked osteopenia throughout "
        "the vertebral bodies and pelvis.",
        "Clear lung fields bilaterally. Heart size within normal limits. No "
        "pleural effusion or pneumothorax.",
    ))
    assert "STUDY MISMATCH" in text


def test_short_documents_are_not_judged():
    """Too little text to tell. Silence beats a guess."""
    assert compare_measurements("Normal.", "Normal study.").mismatch_warning == ""


# ══════════════════════════════════════════════════════════════
# SECTION EXCLUSIVITY — observed in the first real comparison
# ══════════════════════════════════════════════════════════════
#
# ⚠️  ONE FINDING LANDED IN THREE SECTIONS AT ONCE.
# Comparing a chiropractic lumbar report against re-dictated findings, the
# hyperlordosis appeared under "Not mentioned now", "Unchanged" AND "Reported
# differently" simultaneously.
#
# Cause: a radiology report states each finding twice — once under FINDINGS,
# again under IMPRESSION. The model counted each occurrence as a separate
# item. The prior's impression line then looked absent from the current study
# when it plainly was not, and the one difference that mattered (50% -> 60%)
# was buried among duplicates.


def test_prompt_requires_one_finding_per_section():
    p = COMPARISON_PROMPT
    assert "ONE FINDING, ONE SECTION" in p
    assert "EXACTLY ONCE" in p


def test_prompt_explains_that_reports_state_findings_twice():
    """The specific structural feature that caused the duplication."""
    p = COMPARISON_PROMPT
    assert "once under FINDINGS and again" in p
    assert "Treat that pair as one item" in p


def test_prompt_distinguishes_rephrasing_from_a_real_difference():
    """
    "Lumbar hyperlordosis" vs "the lumbar spine is hyperlordotic" is one
    finding written two ways. Calling that "reported differently" dilutes the
    section that should carry only substantive changes.
    """
    p = COMPARISON_PROMPT
    assert "MATCH ON MEANING, NOT WORDING" in p
    assert "a rephrasing is not a change" in p
    # And what does count.
    assert "a number, a level, a\n   laterality, a severity" in p
