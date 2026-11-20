"""SQLAlchemy models. Images live on disk; the database stores only keys and metadata.

Tables: reference_images, inspections, inspection_images, detections, findings.
Matching candidates stay as JSON on `inspections.comparison`. Validated findings are rows.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class UTCDateTime(TypeDecorator[datetime]):
    """Stores UTC; always returns timezone-aware datetimes (SQLite drops tzinfo otherwise)."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:  # type: ignore[no-untyped-def]
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("Naive datetime not allowed; use timezone-aware UTC")
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:  # type: ignore[no-untyped-def]
        return value.replace(tzinfo=timezone.utc) if value is not None else None


def new_uuid() -> str:
    return str(uuid.uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ReferenceImage(Base):
    __tablename__ = "reference_images"
    __table_args__ = (Index("ix_reference_images_property_location", "property_id", "location"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    property_id: Mapped[str] = mapped_column(String(64), nullable=False)
    location: Mapped[str] = mapped_column(String(32), nullable=False)
    # Display-only; never used to build a filesystem path.
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    mime_type: Mapped[str] = mapped_column(String(32), nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utc_now)


class Inspection(Base):
    __tablename__ = "inspections"
    __table_args__ = (
        Index("ix_inspections_property_location", "property_id", "location"),
        Index("ix_inspections_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    property_id: Mapped[str] = mapped_column(String(64), nullable=False)
    location: Mapped[str] = mapped_column(String(32), nullable=False)
    # processing | completed | failed. Processing is synchronous; there is no "queued" state.
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    failed_stage: Mapped[str | None] = mapped_column(String(32), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)

    # Model names / versions / thresholds used, so results are reproducible and auditable.
    yolo_model: Mapped[str] = mapped_column(String(255), nullable=False)
    yolo_conf_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    yolo_iou_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    matching_version: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Matching-stage output (summary, object matches, candidate differences) as JSON.
    comparison: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    vision_status: Mapped[str | None] = mapped_column(String(16), nullable=True)
    vision_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    vision_error: Mapped[str | None] = mapped_column(String(500), nullable=True)


class InspectionImage(Base):
    """An image taking part in an inspection: either a snapshot link to a reference image
    (role 'reference', no file of its own) or a newly uploaded inspection image."""

    __tablename__ = "inspection_images"
    __table_args__ = (UniqueConstraint("inspection_id", "image_ref"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    inspection_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("inspections.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Stable id used in API responses and detection ids: ref_1, ref_2, ..., new_1, new_2, ...
    image_ref: Mapped[str] = mapped_column(String(16), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # reference | inspection
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    reference_image_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("reference_images.id"), nullable=True
    )
    storage_key: Mapped[str | None] = mapped_column(String(255), nullable=True, unique=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(32), nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)


class DetectionRecord(Base):
    __tablename__ = "detections"
    __table_args__ = (UniqueConstraint("inspection_image_id", "detection_ref"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    inspection_image_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("inspection_images.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Unique within the inspection, e.g. "ref_1_det_001" / "new_2_det_003".
    detection_ref: Mapped[str] = mapped_column(String(32), nullable=False)
    label: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    # [x1, y1, x2, y2] in original-image pixels.
    x1: Mapped[float] = mapped_column(Float, nullable=False)
    y1: Mapped[float] = mapped_column(Float, nullable=False)
    x2: Mapped[float] = mapped_column(Float, nullable=False)
    y2: Mapped[float] = mapped_column(Float, nullable=False)


class FindingRecord(Base):
    __tablename__ = "findings"
    __table_args__ = (UniqueConstraint("inspection_id", "finding_ref"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    inspection_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("inspections.id", ondelete="CASCADE"), nullable=False, index=True
    )
    finding_ref: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    label: Mapped[str] = mapped_column(String(80), nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    before_image_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    after_image_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    before_detection_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    after_detection_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    region: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    evidence_strength: Mapped[str] = mapped_column(String(8), nullable=False)
    requires_review: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    explanation: Mapped[str] = mapped_column(String(1000), nullable=False)
    evidence: Mapped[list[str]] = mapped_column(JSON, nullable=False)
