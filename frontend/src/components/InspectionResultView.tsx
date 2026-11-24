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

const BOX_COLOR = {
  damage: "#2563eb",
  missing: "#f97316",
  new: "#dc2626",
  moved: "#dc2626",
} as const;

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
      <div className="pt-6">
        <div className="relative">
          <img
            src={inspectionImageUrl(image.image_url)}
            alt={`Findings on ${image.original_filename}`}
            className="block w-full rounded border border-neutral-200 bg-neutral-100"
          />
          {boxes.map((box, index) => (
            <div key={`${box.kind}-${index}`}>
              <div
                className="absolute border-2"
                style={{
                  left: `${box.x1 * 100}%`,
                  top: `${box.y1 * 100}%`,
                  width: `${(box.x2 - box.x1) * 100}%`,
                  height: `${(box.y2 - box.y1) * 100}%`,
                  borderColor: BOX_COLOR[box.kind],
                }}
              >
                <span
                  className="absolute bottom-full left-0 mb-0.5 max-w-[14rem] truncate px-1 py-0.5 text-[11px] font-semibold leading-tight text-white"
                  style={{ backgroundColor: BOX_COLOR[box.kind] }}
                >
                  {KIND_LABEL[box.kind]}: {box.description}
                </span>
              </div>
              {box.kind === "moved" && box.origin_x != null && box.origin_y != null && (
                <div
                  className="absolute flex -translate-x-1/2 -translate-y-full flex-col items-center"
                  style={{ left: `${box.origin_x * 100}%`, top: `${box.origin_y * 100}%` }}
                >
                  <span
                    className="mb-0.5 whitespace-nowrap px-1 py-0.5 text-[11px] font-semibold text-white"
                    style={{ backgroundColor: BOX_COLOR.moved }}
                  >
                    Was here
                  </span>
                  <span
                    className="h-3 w-3 rounded-full border-2 border-white"
                    style={{ backgroundColor: BOX_COLOR.moved }}
                  />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </figure>
  );
}

export default function InspectionResultView({ results }: { results: InspectionResults }) {
  const before =
    results.images.find((image) => image.image_id === results.matched_image_id) ??
    results.images.find((image) => image.role === "reference");
  const after = results.images.find((image) => image.role === "inspection");
  const boxes = after ? results.boxes.filter((box) => box.image_id === after.image_id) : [];

  return (
    <div className="space-y-4">
      <div className="rounded border border-neutral-200 bg-neutral-50 p-3 text-sm">
        {results.status === "failed" ? (
          <p role="alert" className="font-medium text-red-700">
            {results.error_message ?? "The photos could not be compared."}
          </p>
        ) : (
          <p className="font-medium">
            {results.decision ? DECISION_LABEL[results.decision] : "Not finished"}
          </p>
        )}
        <p className="mt-1 text-neutral-600">
          Blue is damaged, orange is missing, red is new or moved. A moved item is one box
          where it is now, and a pin where it used to be.
        </p>
        {results.summary && <p className="mt-1 text-neutral-700">{results.summary}</p>}
        {results.note && <p className="mt-1 text-neutral-700">{results.note}</p>}
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {before && <PlainPhoto image={before} title="Before" />}
        {after && <PlainPhoto image={after} title="After" />}
        {after && <FindingsPhoto image={after} boxes={boxes} />}
      </div>
    </div>
  );
}
