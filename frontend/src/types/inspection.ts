import type { BBox, Detection } from "./detection";

export type InspectionStatus = "processing" | "completed" | "failed";

export type InspectionDecision =
  | "no_issue_detected"
  | "potential_issues"
  | "inconclusive"
  | "manual_review_required";

export type VisionStatus = "skipped" | "completed" | "failed";

export type FindingKind =
  | "possible_missing"
  | "possible_moved"
  | "possible_new"
  | "possible_damage"
  | "possible_debris"
  | "possible_stain"
  | "not_visible"
  | "possibly_occluded"
  | "inconclusive";

export interface ApproximateRegion {
  image_id: string;
  bbox: BBox;
  approximate: true;
}

export interface Finding {
  id: string;
  kind: FindingKind;
  label: string;
  source: "matcher" | "vision" | "both";
  before_image_id: string | null;
  after_image_id: string | null;
  before_detection_id: string | null;
  after_detection_id: string | null;
  region: ApproximateRegion | null;
  evidence_strength: "low" | "medium";
  requires_review: boolean;
  explanation: string;
  evidence: string[];
}

export interface InspectionImageResult {
  image_id: string;
  role: "reference" | "inspection";
  original_filename: string;
  width: number;
  height: number;
  image_url: string;
  detections: Detection[];
}

export interface ComparisonSummary {
  valid: boolean;
  coverage: "full" | "partial" | "insufficient";
  needs_additional_image: boolean;
  recommendation: string | null;
  quality_issues: string[];
}

export interface InspectionResults {
  inspection_id: string;
  property_id: string;
  location: string;
  status: InspectionStatus;
  failed_stage: string | null;
  error_message: string | null;
  images: InspectionImageResult[];
  comparison: ComparisonSummary | null;
  findings: Finding[];
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
  finding_count: number | null;
  coverage: "full" | "partial" | "insufficient" | null;
}

export interface InspectionListResponse {
  items: InspectionSummary[];
  total: number;
}

export type IssueTone = "missing" | "new" | "damage" | "moved" | "other";

export interface IssueMark {
  id: string;
  label: string;
  bbox: BBox;
  tone: IssueTone;
  approximate?: boolean;
}
