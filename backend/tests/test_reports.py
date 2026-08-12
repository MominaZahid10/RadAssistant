"""
Tests for the report sign-off loop (Phase 5, Step 0.5).

The Review & Sign-off Layer the project document specifies:
*"the radiologist edits, approves, or rejects."*

Before this, a draft was a chat message — not storable, not editable, not
approvable. The human-in-the-loop constraint was a sentence printed under the
output rather than anything the system enforced.

These tests exercise the model's invariants directly. The endpoints are thin
CRUD over them; what actually needs protecting is that the model's output and
the human's corrections stay distinguishable, and that a signature cannot
float free of the text it was given for.
"""

from datetime import datetime, timezone

import pytest

from app.models.report import Report, ReportStatus


def _report(ai_draft="FINDINGS\nMild cardiomegaly.", edited=None, status="draft"):
    return Report(
        findings_input="Mild cardiomegaly.",
        ai_draft=ai_draft,
        edited_text=edited,
        status=status,
    )


# ══════════════════════════════════════════════════════════════
# THE CENTRAL INVARIANT
# ══════════════════════════════════════════════════════════════
#
# ⚠️  ai_draft AND edited_text MUST NEVER COLLAPSE INTO ONE COLUMN.
# A single mutable `text` field would be simpler and would destroy the only
# record of what the model wrote versus what a human corrected before signing.
#
# That delta is the evidence for two claims this project makes: the safety
# argument (how often does the draft need correcting?) and the efficiency
# metric in the project document (time saved means little if you don't know
# how much of the draft survived). Neither is reconstructable afterwards.


def test_ai_draft_and_edit_are_stored_separately():
    r = _report(ai_draft="AI wording.", edited="Human wording.")
    assert r.ai_draft == "AI wording."
    assert r.edited_text == "Human wording."


def test_final_text_prefers_the_human_edit():
    assert _report(ai_draft="AI.", edited="Human.").final_text == "Human."


def test_final_text_falls_back_to_the_draft():
    """NULL edited_text means accepted as written — a real state, not a gap."""
    assert _report(ai_draft="AI.", edited=None).final_text == "AI."


def test_untouched_draft_does_not_count_as_edited():
    assert _report(edited=None).was_edited is False


def test_identical_resubmission_does_not_count_as_edited():
    """
    ⚠️  THE EDITOR WRITES BACK ON EVERY SAVE.
    Testing `edited_text is not None` would mark a report edited merely
    because it was opened and approved unchanged — inflating the edit rate,
    which is the one measured signal of how much the model gets wrong.
    """
    r = _report(ai_draft="Same text.", edited="Same text.")
    assert r.was_edited is False


def test_whitespace_only_change_does_not_count_as_edited():
    r = _report(ai_draft="Same text.", edited="  Same text.\n")
    assert r.was_edited is False


def test_a_real_change_counts_as_edited():
    r = _report(ai_draft="Mild cardiomegaly.", edited="Moderate cardiomegaly.")
    assert r.was_edited is True


# ══════════════════════════════════════════════════════════════
# LIFECYCLE
# ══════════════════════════════════════════════════════════════


def test_new_reports_start_as_drafts():
    assert _report().status == "draft"


def test_status_values_are_the_documented_three():
    assert set(ReportStatus.ALL) == {"draft", "approved", "rejected"}


def test_rejected_reports_are_kept_not_deleted():
    """
    A rejected draft is evidence about the model. Deleting it would leave the
    edit and approval rates measuring only the drafts that went well.
    """
    r = _report(status=ReportStatus.REJECTED)
    assert r.status == "rejected"
    assert r.ai_draft


# ══════════════════════════════════════════════════════════════
# SIGN-OFF GUARDS — endpoint behaviour, asserted on the source
# ══════════════════════════════════════════════════════════════


def test_approved_reports_cannot_be_silently_edited():
    """
    ⚠️  WHAT MAKES A SIGNATURE MEAN ANYTHING.
    Approval is a claim about a specific wording. If an approved report could
    still be edited, the signature would float free of the text it was given
    for, and nothing in the record would show the words changed after the
    clinician signed. Reopening to draft is deliberate and leaves a trail.
    """
    import inspect
    from app.api.v1.endpoints import reports

    source = inspect.getsource(reports.update_report)
    assert "status_code=409" in source
    assert "ReportStatus.APPROVED" in source
    assert "reopen" in source.lower()


def test_reopening_clears_the_previous_signoff():
    """
    Otherwise a reopened report still shows as reviewed by someone who has
    not seen the current wording.
    """
    import inspect
    from app.api.v1.endpoints import reports

    source = inspect.getsource(reports.update_report)
    assert "report.reviewed_at = None" in source
    assert "report.reviewed_by = None" in source


def test_stats_route_is_declared_before_the_id_route():
    """
    FastAPI matches in declaration order. Reversed, 'stats' is parsed as a
    UUID path parameter and the endpoint 422s — the same trap already fixed
    once on the images router.
    """
    import inspect
    from app.api.v1.endpoints import reports

    source = inspect.getsource(reports)
    assert source.index('"/stats"') < source.index('"/{report_id}"')


def test_edit_rate_is_measured_against_reviewed_reports_only():
    """
    Dividing by all reports would make the rate fall whenever someone
    generated drafts and did not review them — movement that looks like the
    model improving.
    """
    import inspect
    from app.api.v1.endpoints import reports

    source = inspect.getsource(reports.get_report_stats)
    assert "reviewed = approved + rejected" in source
    assert "edited / reviewed" in source


def test_edit_counting_ignores_whitespace_in_sql_too():
    """The Python property trims; the aggregate query must agree with it."""
    import inspect
    from app.api.v1.endpoints import reports

    source = inspect.getsource(reports.get_report_stats)
    assert "btrim" in source


# ══════════════════════════════════════════════════════════════
# PROVENANCE
# ══════════════════════════════════════════════════════════════


def test_sources_and_model_are_snapshotted_on_the_report():
    """
    The corpus gets re-indexed and the model gets swapped. Without a snapshot
    taken at generation time, "which sources informed this report" becomes
    unanswerable — and that traceability is the product.
    """
    r = Report(
        findings_input="x",
        ai_draft="y",
        model="openai/gpt-oss-120b",
        sources=[{"chunk_id": 1, "document_title": "Chest imaging"}],
    )
    assert r.model == "openai/gpt-oss-120b"
    assert r.sources[0]["document_title"] == "Chest imaging"


def test_deleting_an_image_must_not_destroy_its_report():
    import inspect
    from app.models import report as report_model

    source = inspect.getsource(report_model)
    assert 'ondelete="SET NULL"' in source
    assert "CASCADE" not in source
