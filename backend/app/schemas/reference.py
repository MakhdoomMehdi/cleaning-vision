from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel

# Letters, digits, '_', '.', '-'; must start alphanumeric; max 64 chars.
PROPERTY_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"


class LocationType(StrEnum):
    GARAGE = "garage"
    BEDROOM = "bedroom"
    KITCHEN = "kitchen"
    BATHROOM = "bathroom"
    CEILING = "ceiling"
    ROOM_CORNER = "room_corner"


class LocationOption(BaseModel):
    value: LocationType
    label: str


class LocationListResponse(BaseModel):
    items: list[LocationOption]


class ReferenceImageOut(BaseModel):
    id: str
    property_id: str
    location: LocationType
    original_filename: str
    mime_type: str
    width: int
    height: int
    size_bytes: int
    created_at: datetime
    # Relative to the API host. Image bytes are served by the API (no public file paths).
    image_url: str


class ReferenceListResponse(BaseModel):
    items: list[ReferenceImageOut]
    total: int
