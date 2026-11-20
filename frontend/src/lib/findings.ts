import type { Detection } from "../types/detection";
import type {
  Finding,
  FindingKind,
  InspectionDecision,
  InspectionImageResult,
  IssueMark,
  IssueTone,
} from "../types/inspection";

const TONE: Record<FindingKind, IssueTone> = {
  possible_missing: "missing",
  possible_new: "new",
  possible_moved: "moved",
  possible_damage: "damage",
  possible_debris: "damage",
  possible_stain: "damage",
  not_visible: "other",
  possibly_occluded: "other",
  inconclusive: "other",
};

const KIND_LABEL: Record<FindingKind, string> = {
  possible_missing: "Missing",
  possible_new: "New",
  possible_moved: "Moved",
  possible_damage: "Damage",
  possible_debris: "Debris",
  possible_stain: "Stain",
  not_visible: "Not in this view",
  possibly_occluded: "Possibly hidden",
  inconclusive: "Inconclusive",
};

const DECISION_LABEL: Record<InspectionDecision, string> = {
  no_issue_detected: "No issues detected in these photos",
  potential_issues: "Potential issues — review required",
  inconclusive: "Inconclusive — another photo needed",
  manual_review_required: "Manual review required",
};

export function findingKindLabel(kind: FindingKind): string {
  return KIND_LABEL[kind];
}

export function decisionLabel(decision: InspectionDecision | null): string {
  return decision ? DECISION_LABEL[decision] : "Not finished";
}

function detectionMap(images: InspectionImageResult[]): Map<string, Detection> {
  const map = new Map<string, Detection>();
  for (const image of images) {
    for (const detection of image.detections) map.set(detection.id, detection);
  }
  return map;
}

/** Marks to draw on one image: detector boxes for missing/new, approximate regions for damage. */
export function marksForImage(
  imageId: string,
  images: InspectionImageResult[],
  findings: Finding[],
): IssueMark[] {
  const detections = detectionMap(images);
  const marks: IssueMark[] = [];

  for (const finding of findings) {
    if (!isIssueFinding(finding)) continue;
    const tone = TONE[finding.kind];
    const caption = `${KIND_LABEL[finding.kind]}: ${finding.label}`;

    if (finding.before_image_id === imageId && finding.before_detection_id) {
      const det = detections.get(finding.before_detection_id);
      if (det) {
        marks.push({ id: `${finding.id}-before`, label: caption, bbox: det.bbox, tone });
      }
    }
    if (finding.after_image_id === imageId && finding.after_detection_id) {
      const det = detections.get(finding.after_detection_id);
      if (det) {
        marks.push({ id: `${finding.id}-after`, label: caption, bbox: det.bbox, tone });
      }
    }
    if (finding.region && finding.region.image_id === imageId) {
      marks.push({
        id: `${finding.id}-region`,
        label: caption,
        bbox: finding.region.bbox,
        tone,
        approximate: true,
      });
    }
  }

  return marks;
}

export function isIssueFinding(finding: Finding): boolean {
  return (
    finding.kind === "possible_missing" ||
    finding.kind === "possible_new" ||
    finding.kind === "possible_moved" ||
    finding.kind === "possible_damage" ||
    finding.kind === "possible_debris" ||
    finding.kind === "possible_stain"
  );
}
