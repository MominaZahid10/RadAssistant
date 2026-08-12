"""
RadAssist AI — Reports API (Phase 5, Step 0.5)

    POST   /api/v1/reports              save a draft
    GET    /api/v1/reports              list, newest first
    GET    /api/v1/reports/stats        sign-off statistics
    GET    /api/v1/reports/{id}         one report
    PATCH  /api/v1/reports/{id}         edit and/or sign off
    DELETE /api/v1/reports/{id}         remove

The Review & Sign-off Layer from the project document. Until now a draft was a
chat message: not storable, not editable, not approvable. The human-in-the-loop
constraint was a line of text printed under the output rather than a mechanism
anything enforced.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.report import Report, ReportStatus
from app.schemas.report import (
    QualityCheckRequest,
    QualityCheckResponse,
    QualityIssue,
    ReportCreate,
    ReportListResponse,
    ReportResponse,
    ReportStats,
    ReportStatusEnum,
    ReportUpdate,
)
from app.services.quality_service import check_report

router = APIRouter(prefix="/reports", tags=["Reports"])


def _to_response(report: Report) -> ReportResponse:
    """
    Serialise, computing the derived fields.

    `final_text` and `was_edited` are Python properties, not columns, so
    from_attributes reads them but they must exist on the instance — which is
    why this goes through the model rather than a dict.
    """
    return ReportResponse(
        id=report.id,
        findings_input=report.findings_input,
        ai_draft=report.ai_draft,
        edited_text=report.edited_text,
        final_text=report.final_text,
        was_edited=report.was_edited,
        status=report.status,
        reviewed_by=report.reviewed_by,
        reviewed_at=report.reviewed_at,
        review_note=report.review_note,
        model=report.model,
        sources=report.sources,
        image_id=report.image_id,
        created_at=report.created_at,
        updated_at=report.updated_at,
    )


# ══════════════════════════════════════════════════════════════
# CREATE
# ══════════════════════════════════════════════════════════════


@router.post(
    "",
    response_model=ReportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Save a generated draft",
    description=(
        "Persists a draft so it can be edited and signed off.\n\n"
        "`ai_draft` is stored immutably. Edits go to `edited_text`, keeping "
        "what the model wrote distinguishable from what a human corrected — "
        "which is the only way to measure how often the draft needed changing."
    ),
)
async def create_report(
    payload: ReportCreate,
    db: AsyncSession = Depends(get_db),
):
    report = Report(
        findings_input=payload.findings_input,
        ai_draft=payload.ai_draft,
        model=payload.model,
        sources=payload.sources,
        image_id=payload.image_id,
        status=ReportStatus.DRAFT,
    )
    db.add(report)
    await db.commit()
    await db.refresh(report)
    return _to_response(report)


# ══════════════════════════════════════════════════════════════
# READ
# ══════════════════════════════════════════════════════════════
# ⚠️  /stats IS DECLARED BEFORE /{report_id}.
# FastAPI matches in declaration order. Reversed, "stats" is parsed as a UUID
# path parameter and the endpoint 422s — the same trap already documented on
# the images router.


@router.get(
    "/stats",
    response_model=ReportStats,
    summary="Sign-off statistics",
    description=(
        "How many drafts were approved, rejected, and — the number that "
        "matters — how often a human had to reword the model's output before "
        "signing it."
    ),
)
async def get_report_stats(db: AsyncSession = Depends(get_db)):
    total = (await db.execute(select(func.count(Report.id)))).scalar() or 0

    by_status: dict[str, int] = {}
    rows = await db.execute(
        select(Report.status, func.count(Report.id)).group_by(Report.status)
    )
    for name, count in rows.all():
        by_status[name] = count

    # Counted in SQL rather than by loading every row: this is a dashboard
    # figure and the table grows without bound.
    edited = (await db.execute(
        select(func.count(Report.id))
        .where(Report.edited_text.isnot(None))
        .where(func.btrim(Report.edited_text) != func.btrim(Report.ai_draft))
    )).scalar() or 0

    approved = by_status.get(ReportStatus.APPROVED, 0)
    rejected = by_status.get(ReportStatus.REJECTED, 0)
    reviewed = approved + rejected

    return ReportStats(
        total=total,
        draft=by_status.get(ReportStatus.DRAFT, 0),
        approved=approved,
        rejected=rejected,
        edited=edited,
        # Denominator is REVIEWED reports, not all of them. Unreviewed drafts
        # have not had the chance to need editing, and counting them would
        # make the rate fall simply because someone generated more drafts.
        edit_rate=round(edited / reviewed, 3) if reviewed else 0.0,
        approval_rate=round(approved / reviewed, 3) if reviewed else 0.0,
    )


@router.post(
    "/quality-check",
    response_model=QualityCheckResponse,
    summary="Check a report draft for defects",
    description=(
        "Runs deterministic checks for missing sections, template "
        "placeholders, measurements without units, contradictory laterality, "
        "stacked hedging, and figures appearing in the impression that are "
        "absent from the findings.\n\n"
        "**Rules, not a model.** The same draft always produces the same "
        "flags, each points at a line, and a clean report produces silence. "
        "The success metric is a reduction in these counts over time, which "
        "requires a detector whose sensitivity does not drift.\n\n"
        "Stateless — the text does not need to be saved first, so the editor "
        "can check as the reviewer types."
    ),
)
async def quality_check(payload: QualityCheckRequest):
    # No DB, no LLM, no network. Fast enough to run on every keystroke pause,
    # which is the only way a checker actually gets used.
    result = check_report(payload.text)
    return QualityCheckResponse(
        issues=[
            QualityIssue(
                code=i.code,
                severity=i.severity,
                message=i.message,
                line=i.line,
                excerpt=i.excerpt,
            )
            for i in result.issues
        ],
        errors=result.errors,
        warnings=result.warnings,
        is_clean=result.is_clean,
    )


@router.get(
    "",
    response_model=ReportListResponse,
    summary="List reports, newest first",
)
async def list_reports(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status_filter: ReportStatusEnum | None = Query(
        None, alias="status", description="Filter by lifecycle state."
    ),
    db: AsyncSession = Depends(get_db),
):
    query = select(Report)
    count_query = select(func.count(Report.id))

    if status_filter:
        query = query.where(Report.status == status_filter.value)
        count_query = count_query.where(Report.status == status_filter.value)

    total = (await db.execute(count_query)).scalar() or 0
    rows = (await db.execute(
        query.order_by(Report.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )).scalars().all()

    return ReportListResponse(
        reports=[_to_response(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{report_id}", response_model=ReportResponse, summary="One report")
async def get_report(report_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    report = await db.get(Report, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return _to_response(report)


# ══════════════════════════════════════════════════════════════
# UPDATE — edit and sign off
# ══════════════════════════════════════════════════════════════


@router.patch(
    "/{report_id}",
    response_model=ReportResponse,
    summary="Edit a draft, or approve/reject it",
    description=(
        "Send `edited_text` to save wording, `status` to sign off, or both "
        "together.\n\n"
        "**An approved report cannot be edited.** Reopen it to `draft` first. "
        "A signed report that can still change is not a signed report."
    ),
)
async def update_report(
    report_id: uuid.UUID,
    payload: ReportUpdate,
    db: AsyncSession = Depends(get_db),
):
    report = await db.get(Report, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")

    # ⚠️  THE GUARD THAT MAKES SIGN-OFF MEAN ANYTHING.
    # Approval is a claim about a specific wording. If an approved report
    # could still be edited, the signature would float free of the text it
    # was given for — and nothing in the record would show that the words
    # changed after the clinician signed. Reopening is deliberate, explicit,
    # and leaves the status trail visible.
    if (
        report.status == ReportStatus.APPROVED
        and payload.edited_text is not None
        and payload.status != ReportStatusEnum.DRAFT
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "This report is approved and cannot be edited. Set status to "
                "'draft' to reopen it, then edit."
            ),
        )

    if payload.edited_text is not None:
        report.edited_text = payload.edited_text

    if payload.status is not None:
        report.status = payload.status.value

        if payload.status in (ReportStatusEnum.APPROVED, ReportStatusEnum.REJECTED):
            report.reviewed_at = datetime.now(timezone.utc)
            report.reviewed_by = payload.reviewed_by or report.reviewed_by
        else:
            # Reopened to draft: the previous sign-off no longer applies and
            # must not linger, or the report will show as reviewed by someone
            # who has not seen the current wording.
            report.reviewed_at = None
            report.reviewed_by = None

    if payload.review_note is not None:
        report.review_note = payload.review_note

    report.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(report)
    return _to_response(report)


# ══════════════════════════════════════════════════════════════
# DELETE
# ══════════════════════════════════════════════════════════════


@router.delete("/{report_id}", summary="Delete a report")
async def delete_report(report_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    report = await db.get(Report, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")

    await db.delete(report)
    await db.commit()
    return {"message": "Report deleted", "id": str(report_id)}
