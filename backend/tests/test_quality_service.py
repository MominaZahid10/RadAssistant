"""
Tests for the report quality checker (Phase 5, Step 3).

⚠️  HALF OF THIS FILE TESTS SILENCE.
A checker that flags correct reports gets switched off within a week, and then
catches nothing at all. False positives are the failure mode that ends the
feature, so every rule is tested against a known-good report as well as a
broken one.

The project document's measurable claim is *"reduction in missing-section /
inconsistent-terminology flags over time"*. That is a count, and a count needs
a detector whose sensitivity does not drift — which is why these rules are
regexes rather than a model.
"""

import pytest

from app.services.quality_service import Severity, check_report


# A correct, unremarkable report. NOTHING may fire on this.
GOOD_REPORT = """
**FINDINGS**
- Mild cardiomegaly.
- No pleural effusion.
- Clear lung fields.
- Degenerative changes of the thoracic spine.

**IMPRESSION**
- Mild cardiomegaly without pleural effusion.
- Degenerative changes of the thoracic spine.

*Draft for radiologist review - not a final report.*
""".strip()


def codes(text: str) -> set[str]:
    return {i.code for i in check_report(text).issues}


# ══════════════════════════════════════════════════════════════
# SILENCE ON A GOOD REPORT
# ══════════════════════════════════════════════════════════════


def test_a_correct_report_produces_no_errors_or_warnings():
    """
    The single most important test here. INFO-level style notes are
    tolerable; anything louder on a valid report makes the tool noise.
    """
    result = check_report(GOOD_REPORT)
    loud = [i for i in result.issues if i.severity != Severity.INFO]
    assert loud == [], f"false positives: {[(i.code, i.message) for i in loud]}"


def test_standard_severity_wording_is_not_flagged_as_vague():
    """
    "Mild cardiomegaly" is accepted radiological phrasing. Demanding a
    measurement for it would fire on almost every real report.
    """
    assert "unquantified_size" not in codes(GOOD_REPORT)


def test_the_draft_disclaimer_does_not_count_as_content():
    """An impression containing only the disclaimer is still empty."""
    text = "**FINDINGS**\n- Mild cardiomegaly.\n\n**IMPRESSION**\n\n*Draft for radiologist review - not a final report.*"
    assert "empty_impression" in codes(text)


# ══════════════════════════════════════════════════════════════
# STRUCTURE
# ══════════════════════════════════════════════════════════════


def test_missing_findings_is_an_error():
    text = "**IMPRESSION**\n- Mild cardiomegaly."
    assert "missing_findings" in codes(text)


def test_missing_impression_is_an_error():
    text = "**FINDINGS**\n- Mild cardiomegaly."
    assert "missing_impression" in codes(text)


def test_empty_findings_is_caught_even_with_the_heading_present():
    text = "**FINDINGS**\n-\n\n**IMPRESSION**\n- Mild cardiomegaly."
    assert "empty_findings" in codes(text)


def test_conclusion_is_accepted_as_an_impression_heading():
    """Reporting conventions vary; rejecting 'CONCLUSION' would be pedantry."""
    text = "**FINDINGS**\n- Mild cardiomegaly.\n\n**CONCLUSION**\n- Mild cardiomegaly."
    assert "missing_impression" not in codes(text)


def test_empty_report():
    assert "empty_report" in codes("   ")


# ══════════════════════════════════════════════════════════════
# PLACEHOLDERS
# ══════════════════════════════════════════════════════════════


@pytest.mark.parametrize("junk", ["TODO", "TBD", "XXX", "PATIENT NAME", "____"])
def test_template_placeholders_are_errors(junk):
    text = f"**FINDINGS**\n- {junk} here.\n\n**IMPRESSION**\n- Mild cardiomegaly."
    assert "placeholder_text" in codes(text)


def test_placeholder_issue_points_at_a_line():
    text = "**FINDINGS**\n- Normal.\n- TODO finish this.\n\n**IMPRESSION**\n- Normal."
    issue = next(i for i in check_report(text).issues if i.code == "placeholder_text")
    assert issue.line == 3
    assert "TODO" in issue.excerpt


# ══════════════════════════════════════════════════════════════
# MEASUREMENTS
# ══════════════════════════════════════════════════════════════


def test_measurement_without_a_unit_is_an_error():
    """mm versus cm is a tenfold error, and it changes management."""
    text = "**FINDINGS**\n- Nodule measuring 8 in the right upper lobe.\n\n**IMPRESSION**\n- Pulmonary nodule."
    assert "measurement_without_unit" in codes(text)


@pytest.mark.parametrize("measurement", ["8 mm", "8mm", "1.2 cm", "50%"])
def test_properly_united_measurements_are_not_flagged(measurement):
    text = (
        f"**FINDINGS**\n- Nodule measuring {measurement} in the lung.\n\n"
        f"**IMPRESSION**\n- Pulmonary nodule measuring {measurement}."
    )
    assert "measurement_without_unit" not in codes(text)


def test_impression_may_not_introduce_a_measurement():
    """
    ⚠️  THE PHASE 4 FAILURE, GENERALISED.
    A report stating 50% was summarised as "25-50%" — a figure borrowed from
    background literature. An impression summarises; it never introduces a
    number that is not above it.
    """
    text = (
        "**FINDINGS**\n- Anterior wedge deformity with 50% height loss.\n\n"
        "**IMPRESSION**\n- Compression fracture with 25% height loss."
    )
    assert "impression_measurement_not_in_findings" in codes(text)


def test_impression_repeating_a_findings_measurement_is_fine():
    text = (
        "**FINDINGS**\n- Anterior wedge deformity with 50% height loss.\n\n"
        "**IMPRESSION**\n- Compression fracture with 50% height loss."
    )
    assert "impression_measurement_not_in_findings" not in codes(text)


# ══════════════════════════════════════════════════════════════
# LATERALITY
# ══════════════════════════════════════════════════════════════


def test_impression_contradicting_findings_laterality_is_an_error():
    text = (
        "**FINDINGS**\n- Consolidation in the right lower lobe.\n\n"
        "**IMPRESSION**\n- Left lower lobe pneumonia."
    )
    assert "laterality_mismatch" in codes(text)


def test_matching_laterality_is_not_flagged():
    text = (
        "**FINDINGS**\n- Consolidation in the right lower lobe.\n\n"
        "**IMPRESSION**\n- Right lower lobe pneumonia."
    )
    assert "laterality_mismatch" not in codes(text)


def test_a_report_with_no_laterality_at_all_is_not_flagged():
    """Plenty of findings have no side. Silence is correct here."""
    assert "laterality_mismatch" not in codes(GOOD_REPORT)


def test_bilateral_findings_are_not_flagged():
    text = (
        "**FINDINGS**\n- Opacities in the left and right lower lobes.\n\n"
        "**IMPRESSION**\n- Bilateral pneumonia, left greater than right."
    )
    assert "laterality_mismatch" not in codes(text)


# ══════════════════════════════════════════════════════════════
# LANGUAGE
# ══════════════════════════════════════════════════════════════


def test_stacked_hedging_is_a_warning():
    text = (
        "**FINDINGS**\n- Opacity possibly may represent infection, though "
        "malignancy cannot be excluded.\n\n**IMPRESSION**\n- Indeterminate opacity."
    )
    assert "stacked_hedging" in codes(text)


def test_a_single_hedge_is_not_flagged():
    """One qualifier is clinical caution, not a defect."""
    text = (
        "**FINDINGS**\n- Opacity possibly infective.\n\n"
        "**IMPRESSION**\n- Indeterminate opacity."
    )
    assert "stacked_hedging" not in codes(text)


def test_impression_introducing_an_unrelated_term_is_a_warning():
    text = (
        "**FINDINGS**\n- Mild cardiomegaly.\n\n"
        "**IMPRESSION**\n- Mild cardiomegaly. Pneumothorax noted."
    )
    assert "impression_introduces_term" in codes(text)


def test_inflected_forms_do_not_count_as_new_terms():
    """
    "Degenerative changes" summarised as "degeneration" is normal reporting,
    not a new conclusion. Flagging it would make the rule unusable.
    """
    text = (
        "**FINDINGS**\n- Degenerative changes of the thoracic spine.\n\n"
        "**IMPRESSION**\n- Thoracic spine degeneration."
    )
    assert "impression_introduces_term" not in codes(text)


# ══════════════════════════════════════════════════════════════
# ROBUSTNESS AND ORDERING
# ══════════════════════════════════════════════════════════════


def test_issues_are_sorted_worst_first():
    text = (
        "**FINDINGS**\n- Nodule measuring 8 in the lung. Possibly may "
        "represent infection.\n\n**IMPRESSION**\n- Indeterminate nodule."
    )
    severities = [i.severity for i in check_report(text).issues]
    assert severities == sorted(severities, key=lambda s: Severity.RANK[s])


def test_checker_never_raises_on_odd_input():
    """
    It fails at exactly the moment someone typed something unusual, which is
    when they most need it.
    """
    for text in ("***", "\n\n\n", "FINDINGS" * 500, "🩻 报告 findings", "[]{}()"):
        check_report(text)   # must not raise


def test_counts_are_exposed_for_the_success_metric():
    text = "**IMPRESSION**\n- TODO."
    result = check_report(text)
    assert result.errors >= 1
    assert result.is_clean is False
    assert check_report(GOOD_REPORT).errors == 0


def test_laterality_is_not_reported_twice():
    """
    ⚠️  OBSERVED IN THE FIRST REAL RUN.
    A left/right contradiction produced its own ERROR and then appeared again
    in the "introduced terms" warning. One problem, two flags — which inflates
    the count the reviewer sees and starts the slide from useful to noisy.
    """
    text = (
        "**FINDINGS**\n- Consolidation in the right lower lobe.\n\n"
        "**IMPRESSION**\n- Left lower lobe pneumonia."
    )
    result = check_report(text)

    # The dedicated rule still fires.
    assert any(i.code == "laterality_mismatch" for i in result.issues)

    # ...and the generic one no longer repeats it.
    introduced = [
        i for i in result.issues if i.code == "impression_introduces_term"
    ]
    for issue in introduced:
        assert "left" not in issue.message.lower().split(": ")[-1].split(", ")


def test_genuinely_new_terms_are_still_reported():
    """The de-duplication must not silence the rule it shares ground with."""
    text = (
        "**FINDINGS**\n- Consolidation in the right lower lobe.\n\n"
        "**IMPRESSION**\n- Left lower lobe pneumonia with pneumothorax."
    )
    codes_found = {i.code for i in check_report(text).issues}
    assert "impression_introduces_term" in codes_found
