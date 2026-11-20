import { inspectionImageUrl } from "../services/api";
import type {
  ChangeBox,
  InspectionDecision,
  InspectionImageResult,
  InspectionResults,
} from "../types/inspection";

const DECISION_LABEL: Record<InspectionDecision, string> = {
  no_issue_detected: "No changes reported in the compared views",
  potential_issues: "Changes to review",
  inconclusive: "Could not compare these photos",
  manual_review_required: "Manual review required",
};

const KIND_LABEL = { damage: "Damage", missing: "Missing", new: "New" } as const;

const BOX_COLOR = {
  damage: "border-red-600 bg-red-600/10 text-red-800",
  missing: "border-amber-600 bg-amber-500/10 text-amber-900",
  new: "border-blue-600 bg-blue-600/10 text-blue-800",
} as const;

function Photo({
  image,
  title,
  boxes,
}: {
  image: InspectionImageResult;
  title: string;
  boxes: ChangeBox[];
}) {
  return (
    <figure>
      <figcaption className="mb-1 text-sm font-medium">{title}</figcaption>
      <div className="relative">
        <img
          src={inspectionImageUrl(image.image_url)}
          alt={image.original_filename}
          className="w-full rounded border border-neutral-200 bg-neutral-100"
        />
        {boxes.map((box, index) => (
          <div
            key={`${box.kind}-${index}`}
            className={`absolute border-2 ${BOX_COLOR[box.kind]}`}
            style={{
              left: `${box.x1 * 100}%`,
              top: `${box.y1 * 100}%`,
              width: `${(box.x2 - box.x1) * 100}%`,
              height: `${(box.y2 - box.y1) * 100}%`,
            }}
          >
            <span className="absolute left-0 top-0 max-w-[14rem] bg-white/95 px-1 text-[11px] font-medium leading-tight">
              {KIND_LABEL[box.kind]}: {box.description}
            </span>
          </div>
        ))}
      </div>
    </figure>
  );
}

function ChangeList({ title, items }: { title: string; items: string[] }) {
  return (
    <section>
      <h3 className="text-sm font-medium">{title}</h3>
      {items.length === 0 ? (
        <p className="mt-1 text-sm text-neutral-600">None reported.</p>
      ) : (
        <ul className="mt-1 list-disc space-y-1 pl-5 text-sm text-neutral-800">
          {items.map((item, index) => (
            <li key={`${title}-${index}`}>{item}</li>
          ))}
        </ul>
      )}
    </section>
  );
}

export default function InspectionResultView({ results }: { results: InspectionResults }) {
  const before = results.images.filter((image) => image.role === "reference");
  const after = results.images.filter((image) => image.role === "inspection");
  const boxes = results.mode === "boxes" ? results.boxes : [];

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
          {results.mode === "boxes"
            ? "Boxes mark each change. The label says what it is."
            : "Semantic check: a text list of damage, missing items, and new items."}
        </p>
        {results.summary && <p className="mt-1 text-neutral-700">{results.summary}</p>}
        {results.views_compared && (
          <p className="mt-1 text-neutral-600">Views used: {results.views_compared}</p>
        )}
        {results.note && <p className="mt-1 text-neutral-700">{results.note}</p>}
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-3">
          {before.map((image, index) => (
            <Photo
              key={image.image_id}
              image={image}
              title={before.length > 1 ? `Before photo ${index + 1}` : "Before"}
              boxes={boxes.filter((box) => box.image_id === image.image_id)}
            />
          ))}
        </div>
        <div className="space-y-3">
          {after.map((image, index) => (
            <Photo
              key={image.image_id}
              image={image}
              title={after.length > 1 ? `After photo ${index + 1}` : "After"}
              boxes={boxes.filter((box) => box.image_id === image.image_id)}
            />
          ))}
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <ChangeList title="Damage" items={results.damage} />
        <ChangeList title="Missing" items={results.missing} />
        <ChangeList title="New" items={results.new} />
      </div>
    </div>
  );
}
