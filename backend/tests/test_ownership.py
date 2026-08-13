"""
Per-user isolation (Phase 6, Step 3).

⚠️  THIS IS WHAT MAKES OPEN REGISTRATION SAFE.

Self-service signup was rejected while every signed-in user could see every
report — anyone who found the URL could then read uploaded patient material.
Ownership is the change that made it defensible: a new account lands in an
empty workspace.

The two are coupled. If these tests are ever weakened, ALLOW_REGISTRATION has
to go false in the same commit, because the argument for one rests entirely on
the other.
"""

import ast
import inspect
import textwrap

import pytest

from app.api.v1.endpoints import images as images_module
from app.api.v1.endpoints import reports as reports_module


# ══════════════════════════════════════════════════════════════
# 404, NOT 403
# ══════════════════════════════════════════════════════════════


def _code_only(fn) -> str:
    """
    Source with comments and docstrings removed.

    ⚠️  NEEDED BECAUSE THE EXPLANATION CONTAINS THE FORBIDDEN STRING.
    The comment above _owned explains why it returns 404 "rather than 403" —
    so a naive substring search fails on the reasoning rather than the code.
    The same trap already caught the SQL_ECHO test, which asserted against a
    comment quoting the setting it had replaced.
    """
    source = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(source)

    # ast.unparse drops comments entirely; stripping docstrings takes the rest.
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Module)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]

    return ast.unparse(tree)


@pytest.mark.parametrize("module", [reports_module, images_module])
def test_someone_elses_row_returns_404_not_403(module):
    """
    ⚠️  403 CONFIRMS THE ROW EXISTS.
    "Forbidden" tells the caller they found something real and are merely not
    allowed to see it — so anyone holding a list of ids learns which are
    genuine. 404 says nothing at all: a report that never existed and one
    belonging to somebody else are indistinguishable.
    """
    code = _code_only(module._owned)
    assert "404" in code
    assert "403" not in code, "an ownership failure must not be distinguishable"
    # The ownership comparison itself.
    assert "user_id != user.id" in code


@pytest.mark.parametrize("module", [reports_module, images_module])
def test_unowned_rows_stay_readable(module):
    """
    Rows created before authentication existed have user_id NULL. Hiding them
    would strand the pilot's own history behind a rule about a period when
    there were no users; claiming them for the first caller would fabricate an
    attribution. They stay readable and stay unowned.
    """
    assert "user_id is not None" in _code_only(module._owned)


# ══════════════════════════════════════════════════════════════
# THE OWNER COMES FROM THE TOKEN
# ══════════════════════════════════════════════════════════════


def test_report_owner_is_never_taken_from_the_request_body():
    """
    ⚠️  AN OWNER A CLIENT CAN SET IS NOT AN OWNER.
    The first thing anyone tries is posting somebody else's id.
    """
    from app.schemas.report import ReportCreate

    assert "user_id" not in ReportCreate.model_fields
    assert "user_id=user.id" in inspect.getsource(reports_module.create_report)


def test_image_owner_is_never_taken_from_the_form():
    source = inspect.getsource(images_module.upload_image)
    assert "user_id=user.id" in source


def test_sign_off_records_the_authenticated_user():
    """
    ⚠️  THE LINE THAT TURNS SIGN-OFF FROM A CLAIM INTO A RECORD.
    `reviewed_by` used to be a name the client typed, which anyone could set
    to anyone. A signature you can address to someone else is not a signature.
    """
    source = inspect.getsource(reports_module.update_report)
    assert "report.reviewed_by_user_id = user.id" in source
    assert "report.reviewed_by = user.email" in source
    # And the old client-supplied field is no longer what decides.
    assert "payload.reviewed_by or report.reviewed_by" not in source


def test_reopening_clears_the_signoff_including_the_user_reference():
    source = inspect.getsource(reports_module.update_report)
    assert "report.reviewed_by_user_id = None" in source


# ══════════════════════════════════════════════════════════════
# EVERY READ PATH IS SCOPED
# ══════════════════════════════════════════════════════════════


@pytest.mark.parametrize("fn_name", [
    "list_reports", "get_report", "update_report", "delete_report",
    "get_report_stats",
])
def test_report_routes_are_scoped(fn_name):
    """
    Named individually so a failure says WHICH route leaks, not merely that
    one does.
    """
    source = inspect.getsource(getattr(reports_module, fn_name))
    assert "user: User = Depends(get_current_user)" in source
    assert ("_owned(" in source) or ("_visible_to(user)" in source)


@pytest.mark.parametrize("fn_name", [
    "list_images", "get_image", "get_image_file", "get_image_thumbnail",
    "delete_image", "image_stats",
])
def test_image_routes_are_scoped(fn_name):
    """
    get_image_file is the one that matters most — it returns the actual
    photograph of a patient's report.
    """
    source = inspect.getsource(getattr(images_module, fn_name))
    assert "user: User = Depends(get_current_user)" in source


def test_serving_a_file_checks_ownership_not_just_the_id():
    """
    ⚠️  KNOWING A UUID MUST NOT BE ENOUGH.
    Ids leak through logs, browser history, referrer headers and screenshots.
    Before this, any signed-in user holding one could fetch the image.
    """
    source = inspect.getsource(images_module._serve)
    assert "_owned(image_id, user, db)" in source
    assert "db.get(MedicalImage, image_id)" not in source


def test_stats_are_per_user():
    """
    An unscoped edit_rate mixes everyone's corrections together — so one
    careless user moves a number the others are judged by, and the count
    leaks how much work other people have done.
    """
    source = inspect.getsource(reports_module.get_report_stats)
    assert "_visible_to(user)" in source


# ══════════════════════════════════════════════════════════════
# DELETING A USER MUST NOT DELETE THEIR WORK
# ══════════════════════════════════════════════════════════════


@pytest.mark.parametrize("model_name,column", [
    ("Report", "user_id"),
    ("Report", "reviewed_by_user_id"),
    ("MedicalImage", "user_id"),
])
def test_user_references_set_null_rather_than_cascading(model_name, column):
    """
    Approvals must stay auditable after someone leaves — which is also why
    users are deactivated rather than deleted. A CASCADE here would erase the
    evidence that the audit trail exists to hold.
    """
    from app import models

    table = getattr(models, model_name).__table__
    fk = next(
        fk for fk in table.foreign_keys
        if fk.parent.name == column
    )
    assert fk.ondelete == "SET NULL"


@pytest.mark.parametrize("model_name", ["Report", "MedicalImage"])
def test_ownership_is_nullable(model_name):
    """
    Pre-auth rows have no known author. Backfilling them to whoever registered
    first would invent an attribution, which is precisely what an audit trail
    exists to prevent — and worse than a gap, because a gap is visibly a gap.
    """
    from app import models

    assert getattr(models, model_name).__table__.columns["user_id"].nullable is True


def test_the_migration_does_not_backfill():
    """Asserted on the migration itself, where the temptation lives."""
    from pathlib import Path

    source = Path("alembic/versions/0006_ownership.py").read_text(encoding="utf-8")
    assert "UPDATE" not in source.upper().replace("UPDATED_AT", "")
    assert "not backfilled" in source.lower()
