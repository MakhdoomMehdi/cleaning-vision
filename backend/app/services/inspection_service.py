"""Inspection orchestration: a synchronous text comparison.

    validate uploads -> store + persist inputs -> load the before pool and the after photo
      -> one OpenAI review -> store the text report

Each stage that can fail records *which* stage failed. Processing is synchronous, so an
inspection is never left "queued".
"""

import hashlib
import logging
from datetime import datetime, timezone

from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Inspection, InspectionImage, ReferenceImage
from app.providers.openai_provider import ReviewImage, VisionProviderError, VisionReviewer
from app.schemas.inspection import (
    ChangeBox,
    InspectionDecision,
    InspectionDetail,
    InspectionImageOut,
    InspectionResults,
    InspectionStatus,
    InspectionSummary,
    VisionStatus,
)
from app.schemas.reference import LocationType
from app.schemas.review import ChangeMark, ReviewMode, RoomReview
from app.services.image_service import ImageService, ImageValidationError, ValidatedImage
from app.services.reference_service import ReferenceService
from app.services.uploads import UploadedFile, prepare_uploads
from app.storage.base import ImageStorage, StorageError

logger = logging.getLogger(__name__)

INSPECTION_CATEGORY = "inspections"
_PHRASE_LIMIT = 12
_PHRASE_CHARS = 300


class InspectionError(Exception):
    pass


class NoReferencesError(InspectionError):
    pass


class TooManyReferencesError(InspectionError):
    pass


class InspectionNotFoundError(InspectionError):
    pass


class VisionUnavailable(Exception):
    """Comparison was requested without an OpenAI API key."""


def _phrases(items: list[str]) -> list[str]:
    cleaned: list[str] = []
    for item in items:
        text = " ".join(item.split())
        if text:
            cleaned.append(text[:_PHRASE_CHARS])
        if len(cleaned) == _PHRASE_LIMIT:
            break
    return cleaned


def _clean_mark(mark: ChangeMark, after_count: int) -> ChangeMark | None:
    """Keep a box only when it can be drawn on the after photo."""
    description = " ".join(mark.description.split())[:_PHRASE_CHARS]
    if not description or after_count < 1:
        return None
    photo_number = 1 if after_count == 1 or mark.side != "after" else mark.photo_number
    if photo_number < 1 or photo_number > after_count:
        return None
    x1, x2 = sorted((mark.x1, mark.x2))
    y1, y2 = sorted((mark.y1, mark.y2))
    if x2 - x1 < 0.01 or y2 - y1 < 0.01:
        return None
    origin_x = origin_y = None
    if mark.kind == "moved" and mark.origin_x is not None and mark.origin_y is not None:
        origin_x = min(1.0, max(0.0, mark.origin_x))
        origin_y = min(1.0, max(0.0, mark.origin_y))
    return ChangeMark(
        kind=mark.kind,
        description=description,
        side="after",
        photo_number=photo_number,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        origin_x=origin_x,
        origin_y=origin_y,
    )


def _matched_photo(report: RoomReview, before_count: int) -> int:
    if before_count < 1:
        return 1
    if 1 <= report.matched_photo <= before_count:
        return report.matched_photo
    return 1


def normalize_review(report: RoomReview, before_count: int, after_count: int) -> RoomReview:
    summary = " ".join(report.summary.split())[:1000]
    views = " ".join(report.views_compared.split())[:500]
    matched = _matched_photo(report, before_count)
    if not report.could_compare:
        return RoomReview(
            mode=ReviewMode.BOXES,
            matched_photo=matched,
            summary=summary,
            views_compared=views,
            could_compare=False,
        )
    marks: list[ChangeMark] = []
    for mark in report.marks:
        cleaned = _clean_mark(mark, after_count)
        if cleaned is not None:
            marks.append(cleaned)
        if len(marks) == _PHRASE_LIMIT:
            break
    return RoomReview(
        mode=ReviewMode.BOXES,
        damage=_phrases([mark.description for mark in marks if mark.kind == "damage"]),
        missing=_phrases([mark.description for mark in marks if mark.kind == "missing"]),
        new=_phrases([mark.description for mark in marks if mark.kind == "new"]),
        moved=_phrases([mark.description for mark in marks if mark.kind == "moved"]),
        marks=marks,
        matched_photo=matched,
        summary=summary,
        views_compared=views,
        could_compare=True,
    )


def _matched_image_id(
    report: RoomReview | None, images: list[InspectionImageOut]
) -> str | None:
    refs = [image for image in images if image.role == "reference"]
    if not refs:
        return None
    number = report.matched_photo if report is not None else 1
    wanted = f"ref_{number}"
    if any(image.image_id == wanted for image in refs):
        return wanted
    return refs[0].image_id


def _comparison_images(
    images: list[InspectionImageOut], matched_image_id: str | None
) -> list[InspectionImageOut]:
    """The matched before photo and the after photo. The rest of the pool stays stored."""
    return [
        image
        for image in images
        if image.role == "inspection" or image.image_id == matched_image_id
    ]


def _boxes(report: RoomReview) -> list[ChangeBox]:
    if report.readiness:
        return [
            ChangeBox(
                image_id=f"new_{issue.photo_number}",
                kind="issue",
                description=f"{issue.severity.capitalize()}: {issue.comment}",
                x1=issue.x1,
                y1=issue.y1,
                x2=issue.x2,
                y2=issue.y2,
            )
            for issue in report.issues
        ]
    return [
        ChangeBox(
            image_id=f"{'ref' if mark.side == 'before' else 'new'}_{mark.photo_number}",
            kind=mark.kind,
            description=mark.description,
            x1=mark.x1,
            y1=mark.y1,
            x2=mark.x2,
            y2=mark.y2,
            origin_x=mark.origin_x,
            origin_y=mark.origin_y,
        )
        for mark in report.marks
    ]


def decide(report: RoomReview) -> InspectionDecision:
    if report.readiness:
        if not report.could_compare:
            return InspectionDecision.INCONCLUSIVE
        if report.guest_ready is False or report.issues:
            return InspectionDecision.POTENTIAL_ISSUES
        return InspectionDecision.NO_ISSUE_DETECTED
    if not report.could_compare:
        return InspectionDecision.INCONCLUSIVE
    if report.damage or report.missing or report.new or report.moved:
        return InspectionDecision.POTENTIAL_ISSUES
    return InspectionDecision.NO_ISSUE_DETECTED


def load_report(payload: dict | None) -> RoomReview | None:
    if not payload:
        return None
    try:
        return RoomReview.model_validate(payload)
    except ValidationError:
        return None


def _failure_message(exc: Exception, stage: str) -> str:
    """User-safe description; the full exception is logged server-side only."""
    if isinstance(exc, VisionUnavailable):
        return "OPENAI_API_KEY is not set, so the photos could not be compared."
    if isinstance(exc, VisionProviderError):
        if stage == "readiness":
            return "The photo could not be checked."
        return "The photos could not be compared."
    if isinstance(exc, StorageError):
        return "A stored image could not be read."
    if isinstance(exc, ImageValidationError):
        return "A stored image could not be decoded."
    return f"Unexpected error during stage '{stage}'."


class InspectionService:
    def __init__(
        self,
        db: Session,
        storage: ImageStorage,
        images: ImageService,
        references: ReferenceService,
        max_reference_images: int,
        vision: VisionReviewer | None = None,
    ) -> None:
        self._db = db
        self._storage = storage
        self._images = images
        self._references = references
        self._max_reference_images = max_reference_images
        self._vision = vision

    @property
    def max_upload_bytes(self) -> int:
        return self._images.max_upload_bytes

    def create(
        self,
        property_id: str,
        location: LocationType,
        files: list[UploadedFile],
    ) -> Inspection:
        refs = self._references.list_references(property_id, location)
        if len(refs) > self._max_reference_images:
            raise TooManyReferencesError(
                f"This location has {len(refs)} reference images; at most "
                f"{self._max_reference_images} can be compared in one inspection."
            )
        prepared = prepare_uploads(self._images, files)
        if not refs:
            inspection = self._persist_inputs(property_id, location, refs, prepared, ReviewMode.BOXES)
            self._run_readiness(inspection.id)
            return self._db.get_one(Inspection, inspection.id)

        inspection = self._persist_inputs(property_id, location, refs, prepared, ReviewMode.BOXES)
        self._run_pipeline(inspection.id)
        return self._db.get_one(Inspection, inspection.id)

    def _run_readiness(self, inspection_id: str) -> None:
        stage = "load_images"
        loaded: dict[str, ValidatedImage] = {}
        try:
            rows = [row for row in self._image_rows(inspection_id) if row.role == "inspection"]
            for row in rows:
                loaded[row.image_ref] = self._images.validate(self._storage.read(self._key_for(row)))
            stage = "readiness"
            if self._vision is None:
                raise VisionUnavailable()
            photos = [
                ReviewImage(caption=f"Photo {index}", image=loaded[row.image_ref].image)
                for index, row in enumerate(rows, start=1)
            ]
            report = self._vision.check_ready(photos)
            decision = decide(report)
            inspection = self._db.get_one(Inspection, inspection_id)
            inspection.comparison = report.model_dump(mode="json")
            inspection.decision = decision.value
            inspection.vision_status = VisionStatus.COMPLETED.value
            inspection.vision_model = self._vision.model_name
            inspection.vision_error = None
            inspection.status = InspectionStatus.COMPLETED.value
            inspection.completed_at = datetime.now(timezone.utc)
            self._db.commit()
        except Exception as exc:
            logger.exception("Inspection %s failed during stage %s", inspection_id, stage)
            self._db.rollback()
            self._mark_failed(inspection_id, stage, _failure_message(exc, stage))
        finally:
            for item in loaded.values():
                item.image.close()

    def _persist_inputs(self, property_id, location, refs, prepared, mode: ReviewMode) -> Inspection:  # type: ignore[no-untyped-def]
        saved_keys: list[str] = []
        try:
            inspection = Inspection(
                property_id=property_id,
                location=location.value,
                status=InspectionStatus.PROCESSING.value,
                review_mode=mode.value,
            )
            self._db.add(inspection)
            self._db.flush()

            for position, ref in enumerate(refs, start=1):
                self._db.add(
                    InspectionImage(
                        inspection_id=inspection.id,
                        image_ref=f"ref_{position}",
                        role="reference",
                        position=position,
                        reference_image_id=ref.id,
                        original_filename=ref.original_filename,
                        mime_type=ref.mime_type,
                        width=ref.width,
                        height=ref.height,
                        size_bytes=ref.size_bytes,
                        sha256=ref.sha256,
                    )
                )
            for position, item in enumerate(prepared, start=1):
                key = self._storage.save(
                    item.file.data, category=INSPECTION_CATEGORY, extension=item.extension
                )
                saved_keys.append(key)
                self._db.add(
                    InspectionImage(
                        inspection_id=inspection.id,
                        image_ref=f"new_{position}",
                        role="inspection",
                        position=position,
                        storage_key=key,
                        original_filename=item.display_name,
                        mime_type=item.mime_type,
                        width=item.width,
                        height=item.height,
                        size_bytes=len(item.file.data),
                        sha256=hashlib.sha256(item.file.data).hexdigest(),
                    )
                )
            self._db.commit()
        except Exception:
            self._db.rollback()
            for key in saved_keys:
                self._storage.delete(key)
            raise
        return inspection

    def _run_pipeline(self, inspection_id: str) -> None:
        stage = "load_images"
        loaded: dict[str, ValidatedImage] = {}
        try:
            rows = self._image_rows(inspection_id)
            for row in rows:
                loaded[row.image_ref] = self._images.validate(self._storage.read(self._key_for(row)))

            stage = "review"
            if self._vision is None:
                raise VisionUnavailable()
            before = [
                ReviewImage(caption=f"Before photo {index}", image=loaded[row.image_ref].image)
                for index, row in enumerate((r for r in rows if r.role == "reference"), start=1)
            ]
            after = [
                ReviewImage(caption=f"After photo {index}", image=loaded[row.image_ref].image)
                for index, row in enumerate((r for r in rows if r.role == "inspection"), start=1)
            ]
            inspection = self._db.get_one(Inspection, inspection_id)
            report = normalize_review(
                self._vision.review(before, after),
                before_count=len(before),
                after_count=len(after),
            )
            decision = decide(report)
            inspection.comparison = report.model_dump(mode="json")
            inspection.decision = decision.value
            inspection.vision_status = VisionStatus.COMPLETED.value
            inspection.vision_model = self._vision.model_name
            inspection.vision_error = None
            inspection.status = InspectionStatus.COMPLETED.value
            inspection.completed_at = datetime.now(timezone.utc)
            self._db.commit()
            logger.info(
                "Inspection %s completed: decision=%s damage=%d missing=%d new=%d",
                inspection_id,
                decision.value,
                len(report.damage),
                len(report.missing),
                len(report.new),
            )
        except Exception as exc:
            logger.exception("Inspection %s failed during stage %s", inspection_id, stage)
            self._db.rollback()
            self._mark_failed(inspection_id, stage, _failure_message(exc, stage))
        finally:
            for item in loaded.values():
                item.image.close()

    def _mark_failed(self, inspection_id: str, stage: str, message: str) -> None:
        inspection = self._db.get_one(Inspection, inspection_id)
        inspection.status = InspectionStatus.FAILED.value
        inspection.failed_stage = stage
        inspection.error_message = message
        vision_stage = stage in {"review", "readiness"}
        inspection.vision_status = (
            VisionStatus.SKIPPED.value if vision_stage and self._vision is None else inspection.vision_status
        )
        if vision_stage and self._vision is not None:
            inspection.vision_status = VisionStatus.FAILED.value
            inspection.vision_model = self._vision.model_name
            inspection.vision_error = message
        inspection.completed_at = datetime.now(timezone.utc)
        self._db.commit()

    def _image_rows(self, inspection_id: str) -> list[InspectionImage]:
        stmt = (
            select(InspectionImage)
            .where(InspectionImage.inspection_id == inspection_id)
            .order_by(InspectionImage.role.desc(), InspectionImage.position)
        )
        return list(self._db.scalars(stmt))

    def _key_for(self, row: InspectionImage) -> str:
        if row.storage_key is not None:
            return row.storage_key
        assert row.reference_image_id is not None
        return self._db.get_one(ReferenceImage, row.reference_image_id).storage_key

    def get(self, inspection_id: str) -> Inspection:
        row = self._db.get(Inspection, inspection_id)
        if row is None:
            raise InspectionNotFoundError(inspection_id)
        return row

    def list_inspections(
        self,
        property_id: str | None = None,
        location: LocationType | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Inspection], int]:
        conditions = []
        if property_id is not None:
            conditions.append(Inspection.property_id == property_id)
        if location is not None:
            conditions.append(Inspection.location == location.value)
        total = self._db.scalar(select(func.count()).select_from(Inspection).where(*conditions)) or 0
        rows = self._db.scalars(
            select(Inspection)
            .where(*conditions)
            .order_by(Inspection.created_at.desc(), Inspection.id)
            .limit(limit)
            .offset(offset)
        )
        return list(rows), total

    def get_image(self, inspection_id: str, image_ref: str) -> tuple[InspectionImage, bytes]:
        self.get(inspection_id)
        row = self._db.scalar(
            select(InspectionImage).where(
                InspectionImage.inspection_id == inspection_id,
                InspectionImage.image_ref == image_ref,
            )
        )
        if row is None:
            raise InspectionNotFoundError(f"{inspection_id}/{image_ref}")
        try:
            return row, self._storage.read(self._key_for(row))
        except StorageError:
            logger.error("Image %s of inspection %s is not readable", image_ref, inspection_id)
            raise InspectionNotFoundError(f"{inspection_id}/{image_ref}") from None

    def summaries(self, rows: list[Inspection]) -> list[InspectionSummary]:
        counts: dict[tuple[str, str], int] = {}
        if rows:
            stmt = (
                select(InspectionImage.inspection_id, InspectionImage.role, func.count())
                .where(InspectionImage.inspection_id.in_([r.id for r in rows]))
                .group_by(InspectionImage.inspection_id, InspectionImage.role)
            )
            counts = {(i, role): n for i, role, n in self._db.execute(stmt)}
        return [self._summary(row, counts) for row in rows]

    @staticmethod
    def _summary(row: Inspection, counts: dict[tuple[str, str], int]) -> InspectionSummary:
        report = load_report(row.comparison)
        change_count = None
        if report is not None:
            change_count = (
                len(report.issues)
                if report.readiness
                else len(report.damage) + len(report.missing) + len(report.new) + len(report.moved)
            )
        return InspectionSummary(
            id=row.id,
            property_id=row.property_id,
            location=LocationType(row.location),
            status=InspectionStatus(row.status),
            created_at=row.created_at,
            completed_at=row.completed_at,
            reference_image_count=counts.get((row.id, "reference"), 0),
            inspection_image_count=counts.get((row.id, "inspection"), 0),
            change_count=change_count,
            decision=InspectionDecision(row.decision) if row.decision else None,
        )

    @staticmethod
    def _image_out(inspection_id: str, row: InspectionImage) -> dict[str, object]:
        return {
            "image_id": row.image_ref,
            "role": row.role,
            "original_filename": row.original_filename,
            "width": row.width,
            "height": row.height,
            "image_url": f"/api/v1/inspections/{inspection_id}/images/{row.image_ref}",
        }

    def detail(self, row: Inspection) -> InspectionDetail:
        summary = self.summaries([row])[0]
        images = [InspectionImageOut(**self._image_out(row.id, img)) for img in self._image_rows(row.id)]
        return InspectionDetail(
            **summary.model_dump(),
            failed_stage=row.failed_stage,
            error_message=row.error_message,
            vision_model=row.vision_model,
            images=images,
        )

    def results(self, row: Inspection) -> InspectionResults:
        report = load_report(row.comparison)
        summary = self.summaries([row])[0]
        note = None
        if row.decision == InspectionDecision.NO_ISSUE_DETECTED.value and not (report and report.summary):
            note = (
                "No damage, missing items, or new items were reported in the photos that could be "
                "compared. This is not a certification that the room is clean."
            )
        elif row.decision == InspectionDecision.INCONCLUSIVE.value and not (report and report.summary):
            note = "These photos do not show the same part of the room clearly enough to compare."
        mode = ReviewMode.BOXES
        images = [InspectionImageOut(**self._image_out(row.id, img)) for img in self._image_rows(row.id)]
        matched_image_id = _matched_image_id(report, images)
        return InspectionResults(
            inspection_id=row.id,
            property_id=row.property_id,
            location=LocationType(row.location),
            status=InspectionStatus(row.status),
            failed_stage=row.failed_stage,
            error_message=row.error_message,
            mode=mode.value,
            readiness=bool(report and report.readiness),
            guest_ready=report.guest_ready if report else None,
            overall_score=report.overall_score if report else None,
            images=_comparison_images(images, matched_image_id),
            damage=report.damage if report else [],
            missing=report.missing if report else [],
            new=report.new if report else [],
            boxes=_boxes(report) if report else [],
            matched_image_id=matched_image_id,
            summary=report.summary or None if report else None,
            views_compared=report.views_compared or None if report else None,
            decision=summary.decision,
            vision_status=VisionStatus(row.vision_status) if row.vision_status else None,
            vision_error=row.vision_error,
            note=note,
        )
