"""Inspection orchestration: a deterministic, synchronous pipeline.

    validate uploads -> store + persist inputs -> load images -> YOLO detection (all images)
      -> viewpoint assessment + conservative matching -> persist results

Each stage that can fail records *which* stage failed on the inspection (status "failed"); no
stage is silently skipped. Processing is synchronous, so an inspection is never left "queued".
"""

import hashlib
import logging
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import DetectionRecord, FindingRecord, Inspection, InspectionImage, ReferenceImage
from app.providers.openai_provider import OpenAIVisionProvider, PairImages, VisionProviderError
from app.providers.yolo_provider import DetectionProviderError
from app.schemas.comparison import ComparisonResult
from app.schemas.detection import Detection, ImageDetections
from app.schemas.finding import Finding, InspectionDecision, VisionStatus
from app.schemas.inspection import (
    DetectorRunInfo,
    InspectionDetail,
    InspectionImageOut,
    InspectionImageResult,
    InspectionModels,
    InspectionResults,
    InspectionStatus,
    InspectionSummary,
)
from app.schemas.reference import LocationType
from app.services.detection_service import DetectionService
from app.services.image_service import ImageService, ImageValidationError, ValidatedImage
from app.services.matching_service import ComparisonImage, MatchingService
from app.services.reference_service import ReferenceService
from app.services.uploads import UploadedFile, prepare_uploads
from app.services.validation_service import ValidationService
from app.storage.base import ImageStorage, StorageError

logger = logging.getLogger(__name__)

INSPECTION_CATEGORY = "inspections"


class InspectionError(Exception):
    pass


class NoReferencesError(InspectionError):
    pass


class TooManyReferencesError(InspectionError):
    pass


class InspectionNotFoundError(InspectionError):
    pass


def qualify_detections(result: ImageDetections, image_ref: str) -> ImageDetections:
    """Make detection ids unique across the inspection: det_001 -> ref_1_det_001."""
    return ImageDetections(
        image_id=image_ref,
        width=result.width,
        height=result.height,
        detections=[
            Detection(id=f"{image_ref}_{d.id}", label=d.label, confidence=d.confidence, bbox=d.bbox)
            for d in result.detections
        ],
    )


def _failure_message(exc: Exception, stage: str) -> str:
    """User-safe description; the full exception is logged server-side only."""
    if isinstance(exc, DetectionProviderError):
        return "Object detection failed or the detector model is unavailable."
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
        detector: DetectionService,
        matcher: MatchingService,
        max_reference_images: int,
        vision: OpenAIVisionProvider | None = None,
        validator: ValidationService | None = None,
    ) -> None:
        self._db = db
        self._storage = storage
        self._images = images
        self._references = references
        self._detector = detector
        self._matcher = matcher
        self._max_reference_images = max_reference_images
        self._vision = vision
        self._validator = validator or ValidationService()

    @property
    def max_upload_bytes(self) -> int:
        return self._images.max_upload_bytes

    # ------------------------------------------------------------------ create

    def create(
        self, property_id: str, location: LocationType, files: list[UploadedFile]
    ) -> Inspection:
        refs = self._references.list_references(property_id, location)
        if not refs:
            raise NoReferencesError(
                "No reference images are registered for this property and location. "
                "Upload reference images first."
            )
        if len(refs) > self._max_reference_images:
            raise TooManyReferencesError(
                f"This location has {len(refs)} reference images; at most "
                f"{self._max_reference_images} can be compared in one inspection."
            )

        prepared = prepare_uploads(self._images, files)  # all-or-nothing validation
        inspection = self._persist_inputs(property_id, location, refs, prepared)
        self._run_pipeline(inspection.id)
        return self._db.get_one(Inspection, inspection.id)

    def _persist_inputs(self, property_id, location, refs, prepared) -> Inspection:  # type: ignore[no-untyped-def]
        info = self._detector.model_info()
        saved_keys: list[str] = []
        try:
            inspection = Inspection(
                property_id=property_id,
                location=location.value,
                status=InspectionStatus.PROCESSING.value,
                yolo_model=info.name,
                yolo_conf_threshold=info.conf_threshold,
                yolo_iou_threshold=info.iou_threshold,
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

    # ------------------------------------------------------------------ pipeline

    def _run_pipeline(self, inspection_id: str) -> None:
        stage = "load_images"
        loaded: dict[str, ValidatedImage] = {}
        try:
            rows = self._image_rows(inspection_id)
            for row in rows:
                loaded[row.image_ref] = self._images.validate(self._storage.read(self._key_for(row)))

            stage = "detect_objects"
            detections: dict[str, ImageDetections] = {}
            for row in rows:
                result = self._detector.detect(loaded[row.image_ref], image_id=row.image_ref)
                qualified = qualify_detections(result, row.image_ref)
                detections[row.image_ref] = qualified
                for det in qualified.detections:
                    self._db.add(
                        DetectionRecord(
                            inspection_image_id=row.id,
                            detection_ref=det.id,
                            label=det.label,
                            confidence=det.confidence,
                            x1=det.bbox[0],
                            y1=det.bbox[1],
                            x2=det.bbox[2],
                            y2=det.bbox[3],
                        )
                    )
            self._db.commit()  # keep detections even if a later stage fails

            stage = "compare_images"
            comparison = self._matcher.compare(
                references=[
                    ComparisonImage(r.image_ref, loaded[r.image_ref].image, detections[r.image_ref])
                    for r in rows
                    if r.role == "reference"
                ],
                inspections=[
                    ComparisonImage(r.image_ref, loaded[r.image_ref].image, detections[r.image_ref])
                    for r in rows
                    if r.role == "inspection"
                ],
            )

            stage = "vision_compare"
            vision_status = VisionStatus.SKIPPED
            vision_error: str | None = None
            vision_model: str | None = None
            vision_output = None
            if self._vision is None:
                vision_error = "OPENAI_API_KEY is not set; stains, debris and damage were not assessed."
            else:
                vision_model = self._vision.model_name
                try:
                    vision_output = self._vision.compare(
                        pairs=self._vision_pairs(comparison, loaded),
                        summary=comparison.summary,
                        candidates=comparison.candidates,
                        known_detection_ids=[
                            d.id for result in detections.values() for d in result.detections
                        ],
                    )
                    vision_status = VisionStatus.COMPLETED
                except VisionProviderError:
                    logger.exception("Inspection %s vision stage failed", inspection_id)
                    vision_status = VisionStatus.FAILED
                    vision_error = "The vision model could not analyse the images."

            stage = "validate_findings"
            sizes = {ref: (item.width, item.height) for ref, item in loaded.items()}
            known_ids = {d.id for result in detections.values() for d in result.detections}
            findings, decision, _note = self._validator.validate(
                comparison, vision_output, vision_status, sizes, known_ids
            )

            stage = "save_results"
            inspection = self._db.get_one(Inspection, inspection_id)
            inspection.comparison = comparison.model_dump(mode="json")
            inspection.matching_version = comparison.matching_version
            inspection.decision = decision.value
            inspection.vision_status = vision_status.value
            inspection.vision_model = vision_model
            inspection.vision_error = vision_error
            inspection.status = InspectionStatus.COMPLETED.value
            inspection.completed_at = datetime.now(timezone.utc)
            self._store_findings(inspection_id, findings)
            self._db.commit()
            logger.info(
                "Inspection %s completed: coverage=%s candidates=%d findings=%d decision=%s vision=%s",
                inspection_id,
                comparison.summary.coverage,
                len(comparison.candidates),
                len(findings),
                decision.value,
                vision_status.value,
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
        inspection.completed_at = datetime.now(timezone.utc)
        self._db.commit()

    def _image_rows(self, inspection_id: str) -> list[InspectionImage]:
        stmt = (
            select(InspectionImage)
            .where(InspectionImage.inspection_id == inspection_id)
            .order_by(InspectionImage.role.desc(), InspectionImage.position)  # reference first
        )
        return list(self._db.scalars(stmt))

    def _vision_pairs(self, comparison: ComparisonResult, loaded: dict[str, ValidatedImage]) -> list[PairImages]:
        ranked = sorted(
            (pair for pair in comparison.summary.pairs if pair.reliable),
            key=lambda pair: pair.inliers,
            reverse=True,
        )
        selected = ranked[:3]
        if not selected:
            refs = [key for key in loaded if key.startswith("ref_")]
            news = [key for key in loaded if key.startswith("new_")]
            if refs and news:
                return [
                    PairImages(
                        reference_image_id=refs[0],
                        inspection_image_id=news[0],
                        reference=loaded[refs[0]].image,
                        inspection=loaded[news[0]].image,
                    )
                ]
            return []
        return [
            PairImages(
                reference_image_id=pair.reference_image_id,
                inspection_image_id=pair.inspection_image_id,
                reference=loaded[pair.reference_image_id].image,
                inspection=loaded[pair.inspection_image_id].image,
            )
            for pair in selected
        ]

    def _store_findings(self, inspection_id: str, findings: list[Finding]) -> None:
        for finding in findings:
            self._db.add(
                FindingRecord(
                    inspection_id=inspection_id,
                    finding_ref=finding.id,
                    kind=finding.kind.value,
                    label=finding.label,
                    source=finding.source,
                    before_image_id=finding.before_image_id,
                    after_image_id=finding.after_image_id,
                    before_detection_id=finding.before_detection_id,
                    after_detection_id=finding.after_detection_id,
                    region=finding.region.model_dump(mode="json") if finding.region else None,
                    evidence_strength=finding.evidence_strength,
                    requires_review=finding.requires_review,
                    explanation=finding.explanation[:1000],
                    evidence=finding.evidence,
                )
            )

    def _load_findings(self, inspection_id: str) -> list[Finding]:
        rows = self._db.scalars(
            select(FindingRecord)
            .where(FindingRecord.inspection_id == inspection_id)
            .order_by(FindingRecord.finding_ref)
        )
        findings: list[Finding] = []
        for row in rows:
            findings.append(
                Finding(
                    id=row.finding_ref,
                    kind=row.kind,  # type: ignore[arg-type]
                    label=row.label,
                    source=row.source,  # type: ignore[arg-type]
                    before_image_id=row.before_image_id,
                    after_image_id=row.after_image_id,
                    before_detection_id=row.before_detection_id,
                    after_detection_id=row.after_detection_id,
                    region=row.region,
                    evidence_strength=row.evidence_strength,  # type: ignore[arg-type]
                    requires_review=row.requires_review,
                    explanation=row.explanation,
                    evidence=row.evidence or [],
                )
            )
        return findings

    @staticmethod
    def _note_for(decision: InspectionDecision | None, vision_status: VisionStatus | None) -> str | None:
        if decision is None:
            return None
        return ValidationService._note(decision, vision_status or VisionStatus.SKIPPED)

    def _key_for(self, row: InspectionImage) -> str:
        if row.storage_key is not None:
            return row.storage_key
        assert row.reference_image_id is not None
        return self._db.get_one(ReferenceImage, row.reference_image_id).storage_key

    # ------------------------------------------------------------------ queries

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

    # ------------------------------------------------------------------ schema builders

    def summaries(self, rows: list[Inspection]) -> list[InspectionSummary]:
        counts: dict[tuple[str, str], int] = {}
        if rows:
            stmt = (
                select(InspectionImage.inspection_id, InspectionImage.role, func.count())
                .where(InspectionImage.inspection_id.in_([r.id for r in rows]))
                .group_by(InspectionImage.inspection_id, InspectionImage.role)
            )
            counts = {(i, role): n for i, role, n in self._db.execute(stmt)}
        finding_counts: dict[str, int] = {}
        if rows:
            finding_counts = {
                inspection_id: count
                for inspection_id, count in self._db.execute(
                    select(FindingRecord.inspection_id, func.count())
                    .where(FindingRecord.inspection_id.in_([r.id for r in rows]))
                    .group_by(FindingRecord.inspection_id)
                ).all()
            }
        return [self._summary(row, counts, finding_counts) for row in rows]

    @staticmethod
    def _summary(
        row: Inspection,
        counts: dict[tuple[str, str], int],
        finding_counts: dict[str, int],
    ) -> InspectionSummary:
        comparison = row.comparison or {}
        return InspectionSummary(
            id=row.id,
            property_id=row.property_id,
            location=LocationType(row.location),
            status=InspectionStatus(row.status),
            created_at=row.created_at,
            completed_at=row.completed_at,
            reference_image_count=counts.get((row.id, "reference"), 0),
            inspection_image_count=counts.get((row.id, "inspection"), 0),
            coverage=(comparison.get("summary") or {}).get("coverage"),
            candidate_count=len(comparison["candidates"]) if "candidates" in comparison else None,
            finding_count=finding_counts.get(row.id) if row.comparison is not None else None,
            decision=InspectionDecision(row.decision) if row.decision else None,
        )

    @staticmethod
    def _models(row: Inspection) -> InspectionModels:
        return InspectionModels(
            detector=DetectorRunInfo(
                name=row.yolo_model,
                conf_threshold=row.yolo_conf_threshold,
                iou_threshold=row.yolo_iou_threshold,
            ),
            matching_version=row.matching_version,
            vision_model=row.vision_model,
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
        images = [
            InspectionImageOut(**self._image_out(row.id, img)) for img in self._image_rows(row.id)
        ]
        return InspectionDetail(
            **summary.model_dump(),
            failed_stage=row.failed_stage,
            error_message=row.error_message,
            models=self._models(row),
            images=images,
        )

    def results(self, row: Inspection) -> InspectionResults:
        image_rows = self._image_rows(row.id)
        by_image: dict[str, list[Detection]] = {img.id: [] for img in image_rows}
        records = self._db.scalars(
            select(DetectionRecord)
            .where(DetectionRecord.inspection_image_id.in_(list(by_image)))
            .order_by(DetectionRecord.detection_ref)
        )
        for rec in records:
            by_image[rec.inspection_image_id].append(
                Detection(
                    id=rec.detection_ref,
                    label=rec.label,
                    confidence=rec.confidence,
                    bbox=(rec.x1, rec.y1, rec.x2, rec.y2),
                )
            )

        images = [
            InspectionImageResult(**self._image_out(row.id, img), detections=by_image[img.id])
            for img in image_rows
        ]
        comparison = ComparisonResult.model_validate(row.comparison) if row.comparison else None
        findings = self._load_findings(row.id)
        if not findings and comparison is not None:
            findings = [self._validator._from_candidate(c) for c in comparison.candidates]
        decision = InspectionDecision(row.decision) if row.decision else None
        vision_status = VisionStatus(row.vision_status) if row.vision_status else None
        return InspectionResults(
            inspection_id=row.id,
            property_id=row.property_id,
            location=LocationType(row.location),
            status=InspectionStatus(row.status),
            failed_stage=row.failed_stage,
            error_message=row.error_message,
            models=self._models(row),
            images=images,
            comparison=comparison.summary if comparison else None,
            matches=comparison.matches if comparison else [],
            candidates=comparison.candidates if comparison else [],
            findings=findings,
            decision=decision,
            vision_status=vision_status,
            vision_error=row.vision_error,
            note=self._note_for(decision, vision_status),
        )
