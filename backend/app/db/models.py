"""SQLAlchemy models. Images live on disk; the database stores only keys and metadata.

Tables: reference_images, inspections, inspection_images.
The text review (damage, missing, new) is JSON on `inspections.comparison`.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    DateTime,
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

    # semantic | boxes. Chosen when the inspection is created.
    review_mode: Mapped[str] = mapped_column(String(16), nullable=False, default="semantic")
    # OpenAI review: damage, missing, new, and optional boxes.
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
    # URL key only: ref_1, ref_2, ..., new_1, new_2, ... The model is not asked to use these.
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
