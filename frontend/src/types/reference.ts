export interface LocationOption {
  value: string;
  label: string;
}

export interface ReferenceImage {
  id: string;
  property_id: string;
  location: string;
  original_filename: string;
  mime_type: string;
  width: number;
  height: number;
  size_bytes: number;
  created_at: string;
  /** Path relative to the API host; resolve with `referenceImageUrl`. */
  image_url: string;
}

export interface ReferenceListResponse {
  items: ReferenceImage[];
  total: number;
}

export interface LocationListResponse {
  items: LocationOption[];
}
