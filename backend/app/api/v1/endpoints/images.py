"""
RadAssist AI — Image API (Phase 4)

    POST   /api/v1/images/upload            DICOM, report photo, or plain image
    GET    /api/v1/images                   list, with filters
    GET    /api/v1/images/{id}              metadata
    GET    /api/v1/images/{id}/file         the image itself
    GET    /api/v1/images/{id}/thumbnail    256px preview
    DELETE /api/v1/images/{id}              remove record and files
    GET    /api/v1/images/stats             dashboard overview

ONE UPLOAD ENDPOINT, THREE FILE KINDS:
The route sniffs the content rather than trusting the extension — PACS exports
are routinely named `IM000001` with no extension at all. A caller shouldn't
have to know which of three endpoints to use for a file they were just handed.

    DICOM         → de-identify, window, render PNG
    report photo  → OCR to text, ingest as a searchable document
    other image   → normalise and store

REUSES THE PATTERNS THAT ALREADY WORK:
- 201 + background processing, poll for status (Phase 2 uploads)
- background task owns its own DB session (the request-scoped one is closed)
- failures recorded with a real reason, never a generic traceback
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.database import async_session, get_db
from app.models.image import MedicalImage
from app.schemas.image import (
    ImageListResponse,
    ImageResponse,
    ImageSourceType,
    ImageStats,
    ImageUploadResponse,
)
from app.services import dicom_service, image_processing, image_storage
from app.services.vision_service import VisionError, vision_service

settings = get_settings()

router = APIRouter(prefix="/images", tags=["Images"])


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════


def _with_urls(img: MedicalImage) -> ImageResponse:
    """
    Attach client-facing URLs.

    Storage paths never leave the server — they'd expose the filesystem layout
    and tempt clients into building their own paths, which is how a traversal
    endpoint gets created by accident.
    """
    resp = ImageResponse.model_validate(img)
    resp.file_url = f"/api/v1/images/{img.id}/file"
    resp.thumbnail_url = (
        f"/api/v1/images/{img.id}/thumbnail" if img.thumbnail_path else None
    )
    return resp


async def _read_report(file_bytes: bytes, mime: str) -> tuple[str, list[str]]:
    """
    Read a photographed report to text. Returns (text, warnings).

    ⚠️  TWO READERS, DELIBERATELY.
    Tesseract runs first and is cheap. Its output is not used directly when
    the vision model is available — it is used to CHECK the vision model.

    The two fail in opposite ways. Tesseract mangles glyphs it cannot resolve
    ("hyperlordotic" → "hypoiordotic"), which is dangerous but visible. A VLM
    resolves those correctly from context, but can fail by fluently inventing
    text that was never on the page — which is far more dangerous, because
    nothing about the output looks wrong.

    Scoring one against the other turns the second failure mode back into a
    visible one. Same reasoning as hybrid BM25 + vector retrieval in Phase
    3.6: two extractors whose errors are uncorrelated catch each other's.

    Falls back to Tesseract alone whenever vision is disabled, unconfigured,
    or errors. A vision outage must never fail an upload.
    """
    ocr = None
    try:
        ocr = image_processing.ocr_report_image(file_bytes)
    except image_processing.ImageProcessingError as e:
        # Not fatal yet — the vision model may still read it. Only if BOTH
        # fail does the upload fail.
        print(f"⚠️  OCR failed, relying on the vision model: {e}")

    if vision_service.is_available():
        try:
            result = await vision_service.transcribe(
                file_bytes, mime, ocr_text=ocr.text if ocr else None
            )
            warnings = list(result.warnings)
            if ocr is None:
                warnings.append(
                    "OCR could not read this image, so the vision model's "
                    "transcription could not be cross-checked."
                )
            return result.text, warnings
        except VisionError as e:
            print(f"⚠️  Vision model unavailable, falling back to OCR: {e}")

    if ocr is None:
        raise image_processing.ImageProcessingError(
            "Neither the vision model nor OCR could read this image. "
            "A flat, well-lit, straight-on scan works best."
        )

    return ocr.text, list(ocr.warnings)


async def _process_upload(
    image_id: uuid.UUID,
    file_bytes: bytes,
    filename: str,
    declared_type: str | None,
) -> None:
    """
    Parse, de-identify, render and thumbnail — after the response is sent.

    Must never raise: Starlette swallows background-task exceptions, which
    would strand the row at status="processing" with no explanation. Every
    outcome is written back to the database.
    """
    async with async_session() as db:
        fields: dict = {}
        try:
            is_dicom = image_processing.looks_like_dicom(file_bytes, filename)

            if is_dicom:
                # ── DICOM: de-identify and render ──
                result = dicom_service.parse_dicom(file_bytes)
                stored_bytes = result.png_bytes
                mime = "image/png"
                suffix = ".png"

                fields.update(
                    modality=result.modality,
                    body_part=result.body_part,
                    view_position=result.view_position,
                    study_date=result.study_date,
                    dicom_metadata=result.metadata,
                    width=result.width,
                    height=result.height,
                    source_type=ImageSourceType.DICOM_UPLOAD.value,
                    # Set ONLY here, only after parse_dicom returned — which
                    # it does not do unless allowlisting completed.
                    is_deidentified=True,
                )
            else:
                # ── Standard image ──
                stored_bytes, mime, width, height = image_processing.normalise(file_bytes)
                suffix = f".{mime.split('/')[-1]}"
                fields.update(width=width, height=height)

                if declared_type == ImageSourceType.REPORT_UPLOAD.value:
                    # A photographed report: read it to text so it becomes
                    # searchable alongside everything else.
                    text, warnings = await _read_report(file_bytes, mime)
                    fields["ocr_text"] = text
                    fields["source_type"] = ImageSourceType.REPORT_UPLOAD.value
                    # Surfaced to the client so it can warn the user AND pass
                    # the caveats to the model — misread text stated as fact
                    # is the dangerous failure here.
                    if warnings:
                        fields["description"] = " ".join(warnings)
                else:
                    fields["source_type"] = (
                        declared_type or ImageSourceType.IMAGE_UPLOAD.value
                    )

            # ── Store the image ──
            rel = image_storage.new_relative_path(image_id, suffix)
            size = image_storage.write_bytes(rel, stored_bytes)
            fields.update(storage_path=rel, mime_type=mime, file_size=size)

            # ── Thumbnail (non-fatal) ──
            # A missing preview is a cosmetic problem; failing the whole
            # upload over it would lose the image itself.
            try:
                thumb_rel = image_storage.thumbnail_path_for(rel)
                image_storage.write_bytes(
                    thumb_rel, image_processing.make_thumbnail(stored_bytes)
                )
                fields["thumbnail_path"] = thumb_rel
            except Exception as e:  # noqa: BLE001
                print(f"⚠️  Thumbnail failed for {image_id}: {e}")

            fields["status"] = "completed"
            fields["error_message"] = None

        except (dicom_service.DicomError, image_processing.ImageProcessingError) as e:
            # Expected, user-facing failures — report them verbatim.
            fields = {"status": "failed", "error_message": str(e)}
        except Exception as e:  # noqa: BLE001
            fields = {"status": "failed", "error_message": f"Internal error: {e}"}

        try:
            img = await db.get(MedicalImage, image_id)
            if img is not None:
                for key, value in fields.items():
                    setattr(img, key, value)
                img.updated_at = datetime.now(timezone.utc)
                await db.commit()
        except Exception as e:  # noqa: BLE001
            await db.rollback()
            print(f"❌ Could not record image result for {image_id}: {e}")


# ══════════════════════════════════════════════════════════════
# UPLOAD
# ══════════════════════════════════════════════════════════════


@router.post(
    "/upload",
    response_model=ImageUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a DICOM study, report photo, or image",
    description=(
        "Accepts DICOM files, photographs or scans of paper reports, and "
        "standard images (PNG/JPG/TIFF).\n\n"
        "The file type is detected from its **contents**, not its extension — "
        "PACS exports are frequently named `IM000001` with no extension.\n\n"
        "**DICOM files are de-identified on ingest.** Only an allowlist of "
        "clinical tags is retained; patient name, ID, dates, institution and "
        "all unrecognised tags are discarded.\n\n"
        "Set `source_type=report_upload` for a photographed report and its "
        "text will be extracted by OCR.\n\n"
        "Returns immediately with `status=processing`; poll "
        "`GET /images/{id}`."
    ),
)
async def upload_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="DICOM, PNG, JPG, or TIFF"),
    source_type: str = Form(
        default=ImageSourceType.IMAGE_UPLOAD.value,
        description="dicom_upload | report_upload | image_upload",
    ),
    description: str | None = Form(default=None),
    document_id: uuid.UUID | None = Form(
        default=None,
        description="Optional link to an existing text document.",
    ),
    db: AsyncSession = Depends(get_db),
):
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Filename is required.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File is empty.")

    max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File is {len(file_bytes) / 1024 / 1024:.1f}MB; the limit is "
            f"{settings.MAX_IMAGE_SIZE_MB}MB.",
        )

    is_dicom = image_processing.looks_like_dicom(file_bytes, file.filename)

    # Reject DICOM up front when pydicom is missing, rather than accepting the
    # file and marking it failed a moment later. Same reasoning as the
    # embedding-model guard: no side effects from a request that cannot work.
    if is_dicom and not dicom_service.is_available():
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            dicom_service.unavailable_reason(),
        )

    detected = "dicom" if is_dicom else (
        "report" if source_type == ImageSourceType.REPORT_UPLOAD.value else "image"
    )

    img = MedicalImage(
        filename=file.filename,
        # Placeholder until the background task stores the file and knows the
        # final extension. NOT NULL on the column, so it needs a value now.
        storage_path="",
        mime_type="application/octet-stream",
        source_type=(
            ImageSourceType.DICOM_UPLOAD.value if is_dicom else source_type
        ),
        description=description,
        document_id=document_id,
        status="processing",
        is_deidentified=False,      # never assumed — set only after it runs
    )
    db.add(img)
    await db.commit()
    await db.refresh(img)

    background_tasks.add_task(
        _process_upload,
        image_id=img.id,
        file_bytes=file_bytes,
        filename=file.filename,
        declared_type=source_type,
    )

    return ImageUploadResponse(
        id=img.id,
        filename=img.filename,
        status="processing",
        detected_type=detected,
        message=(
            f"'{file.filename}' accepted"
            + (" and will be de-identified" if is_dicom else "")
            + f". Poll GET /api/v1/images/{img.id} for status."
        ),
    )


# ══════════════════════════════════════════════════════════════
# READ
# ══════════════════════════════════════════════════════════════


@router.get("", response_model=ImageListResponse, summary="List images")
async def list_images(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    source_type: str | None = None,
    modality: str | None = None,
    body_part: str | None = None,
    status_filter: str | None = None,
    document_id: uuid.UUID | None = None,
    db: AsyncSession = Depends(get_db),
):
    query = select(MedicalImage)
    count_q = select(func.count(MedicalImage.id))

    for column, value in (
        (MedicalImage.source_type, source_type),
        (MedicalImage.modality, modality),
        (MedicalImage.body_part, body_part),
        (MedicalImage.status, status_filter),
        (MedicalImage.document_id, document_id),
    ):
        if value is not None:
            query = query.where(column == value)
            count_q = count_q.where(column == value)

    total = (await db.execute(count_q)).scalar() or 0
    query = (
        query.order_by(MedicalImage.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    rows = (await db.execute(query)).scalars().all()

    return ImageListResponse(
        images=[_with_urls(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


# ⚠️  /stats is declared BEFORE /{image_id}.
# FastAPI matches routes in declaration order, so a later /stats would be
# swallowed by /{image_id} and fail UUID validation with a confusing 422.
@router.get("/stats", response_model=ImageStats, summary="Image statistics")
async def image_stats(db: AsyncSession = Depends(get_db)):
    total = (await db.execute(select(func.count(MedicalImage.id)))).scalar() or 0

    async def count_where(condition) -> int:
        return (
            await db.execute(select(func.count(MedicalImage.id)).where(condition))
        ).scalar() or 0

    by_source = {
        row[0]: row[1]
        for row in (
            await db.execute(
                select(MedicalImage.source_type, func.count(MedicalImage.id))
                .group_by(MedicalImage.source_type)
            )
        ).all()
    }
    by_modality = {
        row[0]: row[1]
        for row in (
            await db.execute(
                select(MedicalImage.modality, func.count(MedicalImage.id))
                .where(MedicalImage.modality.isnot(None))
                .group_by(MedicalImage.modality)
            )
        ).all()
    }

    storage = image_storage.storage_stats()

    return ImageStats(
        total_images=total,
        completed=await count_where(MedicalImage.status == "completed"),
        failed=await count_where(MedicalImage.status == "failed"),
        processing=await count_where(MedicalImage.status == "processing"),
        by_source_type=by_source,
        by_modality=by_modality,
        deidentified_count=await count_where(MedicalImage.is_deidentified.is_(True)),
        storage_bytes=storage["bytes"],
        storage_files=storage["files"],
    )


@router.get("/{image_id}", response_model=ImageResponse, summary="Image metadata")
async def get_image(image_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    img = await db.get(MedicalImage, image_id)
    if img is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Image {image_id} not found.")
    return _with_urls(img)


@router.get("/{image_id}/file", summary="The image file")
async def get_image_file(image_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    return await _serve(image_id, db, thumbnail=False)


@router.get("/{image_id}/thumbnail", summary="256px preview")
async def get_image_thumbnail(image_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    return await _serve(image_id, db, thumbnail=True)


async def _serve(image_id: uuid.UUID, db: AsyncSession, thumbnail: bool):
    """
    Stream a stored file.

    The path comes from the database and is resolved through image_storage,
    which rejects anything outside the storage root. Clients never supply a
    path — only an id — so there's no user-controlled component in the
    filesystem lookup at all.
    """
    img = await db.get(MedicalImage, image_id)
    if img is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Image {image_id} not found.")

    rel = img.thumbnail_path if thumbnail else img.storage_path
    if not rel:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "Thumbnail not generated for this image."
            if thumbnail
            else f"Image has no stored file (status: {img.status}).",
        )

    try:
        path = image_storage.resolve(rel)
    except ValueError:
        # Should be unreachable — paths are server-generated. If it happens,
        # the database has been tampered with, so refuse rather than serve.
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Invalid storage path.")

    if not path.is_file():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "The file is recorded in the database but missing from disk.",
        )

    return FileResponse(
        path,
        media_type="image/jpeg" if thumbnail else img.mime_type,
        filename=img.filename,
    )


# ══════════════════════════════════════════════════════════════
# DELETE
# ══════════════════════════════════════════════════════════════


@router.delete("/{image_id}", summary="Delete an image")
async def delete_image(image_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """
    Remove the database row and both files.

    Files are deleted first. If that fails we still remove the row — an
    orphaned file wastes disk, but an orphaned row breaks every listing that
    tries to serve it.
    """
    img = await db.get(MedicalImage, image_id)
    if img is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Image {image_id} not found.")

    removed = []
    for rel in (img.storage_path, img.thumbnail_path):
        if rel and image_storage.delete(rel):
            removed.append(rel)

    filename = img.filename
    await db.delete(img)
    await db.commit()

    return {
        "message": f"Deleted '{filename}'.",
        "image_id": str(image_id),
        "files_removed": len(removed),
    }
