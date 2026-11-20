import { bboxToPercent } from "../lib/geometry";
import type { IssueMark, IssueTone } from "../types/inspection";

const STROKE: Record<IssueTone, string> = {
  missing: "#dc2626",
  new: "#d97706",
  damage: "#ea580c",
  moved: "#7c3aed",
  other: "#2563eb",
};

const LABEL: Record<IssueTone, string> = {
  missing: "bg-red-600",
  new: "bg-amber-600",
  damage: "bg-orange-600",
  moved: "bg-violet-600",
  other: "bg-blue-600",
};

interface IssueOverlayProps {
  width: number;
  height: number;
  marks: IssueMark[];
}

export default function IssueOverlay({ width, height, marks }: IssueOverlayProps) {
  if (marks.length === 0) return null;

  return (
    <div className="pointer-events-none absolute inset-0" data-testid="issue-overlay">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        className="absolute inset-0 h-full w-full"
        aria-hidden="true"
      >
        {marks.map((mark) => (
          <rect
            key={mark.id}
            x={mark.bbox[0]}
            y={mark.bbox[1]}
            width={mark.bbox[2] - mark.bbox[0]}
            height={mark.bbox[3] - mark.bbox[1]}
            fill="transparent"
            stroke={STROKE[mark.tone]}
            strokeWidth={3}
            strokeDasharray={mark.approximate ? "8 6" : undefined}
            vectorEffect="non-scaling-stroke"
          />
        ))}
      </svg>
      {marks.map((mark) => {
        const box = bboxToPercent(mark.bbox, width, height);
        return (
          <span
            key={`${mark.id}-label`}
            className={`absolute -translate-y-full max-w-[90%] truncate rounded-sm px-1 text-[11px] leading-4 text-white ${LABEL[mark.tone]}`}
            style={{ left: `${box.left}%`, top: `${box.top}%` }}
          >
            {mark.label}
            {mark.approximate ? " (approx.)" : ""}
          </span>
        );
      })}
    </div>
  );
}
