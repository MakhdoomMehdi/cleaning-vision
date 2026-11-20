import { decisionLabel, findingKindLabel, isIssueFinding, marksForImage } from "../lib/findings";
import { inspectionImageUrl } from "../services/api";
import type { Finding, InspectionResults } from "../types/inspection";
import IssueOverlay from "./IssueOverlay";

function Photo({
  title,
  src,
  alt,
  width,
  height,
  marks,
}: {
  title: string;
  src: string;
  alt: string;
  width: number;
  height: number;
  marks: ReturnType<typeof marksForImage>;
}) {
  return (
    <figure>
      <figcaption className="mb-1 text-sm font-medium">{title}</figcaption>
      <div
        className="relative w-full overflow-hidden rounded border border-neutral-200 bg-neutral-100"
        style={{ aspectRatio: `${width} / ${height}` }}
      >
        <img src={src} alt={alt} className="absolute inset-0 h-full w-full object-fill" />
        <IssueOverlay width={width} height={height} marks={marks} />
      </div>
    </figure>
  );
}

function FindingCard({ finding }: { finding: Finding }) {
  return (
    <li className="rounded border border-neutral-200 bg-white p-3">
      <p className="text-sm font-medium">
        {findingKindLabel(finding.kind)} — {finding.label}
      </p>
      <p className="mt-1 text-sm text-neutral-700">{finding.explanation}</p>
      <p className="mt-1 text-xs text-neutral-500">
        {finding.source} · {finding.evidence_strength} evidence
        {finding.region?.approximate ? " · dashed box is approximate" : ""}
      </p>
    </li>
  );
}

export default function InspectionResultView({ results }: { results: InspectionResults }) {
  const before = results.images.find((image) => image.role === "reference");
  const after = results.images.find((image) => image.role === "inspection");
  const issues = results.findings.filter(isIssueFinding);
  const context = results.findings.filter((finding) => !isIssueFinding(finding));

  return (
    <div className="space-y-4">
      <div className="rounded border border-neutral-200 bg-neutral-50 p-3 text-sm">
        <p className="font-medium">{decisionLabel(results.decision)}</p>
        {results.note && <p className="mt-1 text-neutral-700">{results.note}</p>}
        {results.comparison?.needs_additional_image && results.comparison.recommendation && (
          <p className="mt-1 text-amber-800">{results.comparison.recommendation}</p>
        )}
        {results.vision_status === "skipped" && (
          <p className="mt-1 text-amber-800">
            Add OPENAI_API_KEY in backend/.env to detect stains, debris and damage. Object
            missing/new still runs locally.
          </p>
        )}
        {results.vision_error && (
          <p className="mt-1 text-red-700">{results.vision_error}</p>
        )}
        {results.status === "failed" && (
          <p role="alert" className="mt-1 text-red-700">
            {results.error_message ?? "Inspection failed."}
          </p>
        )}
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {before && (
          <Photo
            title="Before (reference)"
            src={inspectionImageUrl(before.image_url)}
            alt={before.original_filename}
            width={before.width}
            height={before.height}
            marks={marksForImage(before.image_id, results.images, results.findings)}
          />
        )}
        {after && (
          <Photo
            title="After (inspection)"
            src={inspectionImageUrl(after.image_url)}
            alt={after.original_filename}
            width={after.width}
            height={after.height}
            marks={marksForImage(after.image_id, results.images, results.findings)}
          />
        )}
      </div>

      <div>
        <h3 className="text-sm font-medium">Issues to review ({issues.length})</h3>
        {issues.length === 0 ? (
          <p className="mt-2 text-sm text-neutral-600">
            No missing, new, moved, damaged or debris items were flagged in these photos.
          </p>
        ) : (
          <ul className="mt-2 space-y-2">
            {issues.map((finding) => (
              <FindingCard key={finding.id} finding={finding} />
            ))}
          </ul>
        )}
      </div>

      {context.length > 0 && (
        <details className="text-sm">
          <summary className="cursor-pointer font-medium text-neutral-600">
            Viewpoint notes ({context.length})
          </summary>
          <ul className="mt-2 space-y-2">
            {context.map((finding) => (
              <FindingCard key={finding.id} finding={finding} />
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
