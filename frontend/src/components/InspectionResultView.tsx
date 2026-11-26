import { inspectionImageUrl } from "../services/api";
import type { ChangeBox, InspectionDecision, InspectionImageResult, InspectionResults } from "../types/inspection";

const DECISION_LABEL: Record<InspectionDecision, string> = {
  no_issue_detected: "No changes reported in the compared views",
  potential_issues: "Changes to review",
  inconclusive: "Could not compare these photos",
  manual_review_required: "Manual review required",
};

const KIND_LABEL = {
  damage: "Damaged",
  missing: "Missing",
  new: "Other",
  moved: "Moved",
} as const;

const FINDING_COLORS = [
  "#2563eb",
  "#ea580c",
  "#dc2626",
  "#16a34a",
  "#9333ea",
  "#0891b2",
  "#ca8a04",
  "#db2777",
  "#4f46e5",
  "#0d9488",
  "#c026d3",
  "#65a30d",
] as const;

function findingColor(index: number): string {
  return FINDING_COLORS[index % FINDING_COLORS.length];
}

function PlainPhoto({ image, title }: { image: InspectionImageResult; title: string }) {
  return (
    <figure>
      <figcaption className="mb-1 text-sm font-medium">{title}</figcaption>
      <img
        src={inspectionImageUrl(image.image_url)}
        alt={image.original_filename}
        className="w-full rounded border border-neutral-200 bg-neutral-100"
      />
    </figure>
  );
}

function FindingsPhoto({ image, boxes }: { image: InspectionImageResult; boxes: ChangeBox[] }) {
  return (
    <figure>
      <figcaption className="mb-1 text-sm font-medium">Findings</figcaption>
      <div className="overflow-hidden rounded border border-neutral-200 bg-neutral-100">
        <div className="relative">
          <img
            src={inspectionImageUrl(image.image_url)}
            alt={`Findings on ${image.original_filename}`}
            className="block h-auto w-full"
          />
          <div className="absolute inset-0">
            {boxes.map((box, index) => {
              const color = findingColor(index);
              return (
                <div key={`${box.kind}-${index}`}>
                  <div
                    className="absolute border-2"
                    style={{
                      left: `${box.x1 * 100}%`,
                      top: `${box.y1 * 100}%`,
                      width: `${(box.x2 - box.x1) * 100}%`,
                      height: `${(box.y2 - box.y1) * 100}%`,
                      borderColor: color,
                    }}
                  />
                  {box.kind === "moved" && box.origin_x != null && box.origin_y != null && (
                    <span
                      className="absolute h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white"
                      style={{
                        left: `${box.origin_x * 100}%`,
                        top: `${box.origin_y * 100}%`,
                        backgroundColor: color,
                      }}
                    />
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
      <FindingList boxes={boxes} />
    </figure>
  );
}

function findingText(box: ChangeBox): string {
  if (box.kind === "issue") {
    return box.description;
  }
  return `${KIND_LABEL[box.kind]}: ${box.description}`;
}

function FindingList({ boxes }: { boxes: ChangeBox[] }) {
  if (boxes.length === 0) {
    return null;
  }
  return (
    <ul className="mt-2 space-y-1 text-sm">
      {boxes.map((box, index) => (
        <li key={`${box.kind}-${index}`} className="font-medium" style={{ color: findingColor(index) }}>
          {findingText(box)}
        </li>
      ))}
    </ul>
  );
}

const READINESS_LABEL: Record<InspectionDecision, string> = {
  no_issue_detected: "Guest ready",
  potential_issues: "Not guest ready",
  inconclusive: "Could not judge this photo",
  manual_review_required: "Manual review required",
};

export default function InspectionResultView({ results }: { results: InspectionResults }) {
  const before =
    results.images.find((image) => image.image_id === results.matched_image_id) ??
    results.images.find((image) => image.role === "reference");
  const photos = results.images.filter((image) => image.role === "inspection");
  const after = photos[0];
  const headline =
    results.readiness && results.decision
      ? results.guest_ready === false
        ? "Not guest ready"
        : results.guest_ready === true
          ? "Guest ready"
          : READINESS_LABEL[results.decision]
      : results.decision
        ? DECISION_LABEL[results.decision]
        : "Not finished";

  return (
    <div className="space-y-4">
      <div className="rounded border border-neutral-200 bg-neutral-50 p-3 text-sm">
        {results.status === "failed" ? (
          <p role="alert" className="font-medium text-red-700">
            {results.error_message ?? "The photos could not be compared."}
          </p>
        ) : (
          <p className="font-medium">{headline}</p>
        )}
        <p className="mt-1 text-neutral-600">
          {results.readiness
            ? "Each color matches one issue listed under the photo."
            : "Each color matches one finding listed under the photo. A moved item is one box where it is now, and a pin where it used to be."}
        </p>
        {results.readiness && results.overall_score != null && (
          <p className="mt-1 text-neutral-700">Score {results.overall_score} out of 100.</p>
        )}
        {results.summary && <p className="mt-1 text-neutral-700">{results.summary}</p>}
        {results.note && <p className="mt-1 text-neutral-700">{results.note}</p>}
      </div>

      <div className={`grid gap-4 ${before ? "md:grid-cols-3" : "md:grid-cols-2"}`}>
        {before && <PlainPhoto image={before} title="Before" />}
        {before && after && <PlainPhoto image={after} title="After" />}
        {before && after && (
          <FindingsPhoto image={after} boxes={results.boxes.filter((box) => box.image_id === after.image_id)} />
        )}
        {!before &&
          photos.map((image) => (
            <FindingsPhoto
              key={image.image_id}
              image={image}
              boxes={results.boxes.filter((box) => box.image_id === image.image_id)}
            />
          ))}
      </div>
    </div>
  );
}
