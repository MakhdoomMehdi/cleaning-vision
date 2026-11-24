export type InspectionStatus = "processing" | "completed" | "failed";

export type InspectionDecision =
  | "no_issue_detected"
  | "potential_issues"
  | "inconclusive"
  | "manual_review_required";

export type VisionStatus = "skipped" | "completed" | "failed";

export interface ChangeBox {
  image_id: string;
  kind: "damage" | "missing" | "new" | "moved";
  description: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  origin_x: number | null;
  origin_y: number | null;
}

export interface InspectionImageResult {
  image_id: string;
  role: "reference" | "inspection";
  original_filename: string;
  width: number;
  height: number;
  image_url: string;
}

export interface InspectionResults {
  inspection_id: string;
  property_id: string;
  location: string;
  status: InspectionStatus;
  failed_stage: string | null;
  error_message: string | null;
  images: InspectionImageResult[];
  matched_image_id: string | null;
  damage: string[];
  missing: string[];
  new: string[];
  boxes: ChangeBox[];
  summary: string | null;
  views_compared: string | null;
  decision: InspectionDecision | null;
  vision_status: VisionStatus | null;
  vision_error: string | null;
  note: string | null;
}

export interface InspectionSummary {
  id: string;
  property_id: string;
  location: string;
  status: InspectionStatus;
  created_at: string;
  decision: InspectionDecision | null;
  change_count: number | null;
}

export interface InspectionListResponse {
  items: InspectionSummary[];
  total: number;
}
