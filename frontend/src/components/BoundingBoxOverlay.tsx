import { bboxToPercent } from "../lib/geometry";
import type { Detection } from "../types/detection";

interface BoundingBoxOverlayProps {
  /** Original image size in pixels: the coordinate space the bboxes live in. */
  width: number;
  height: number;
  detections: Detection[];
  showBoxes?: boolean;
  showLabels?: boolean;
}

/**
 * Transparent layer to place over an image that is displayed at the same aspect ratio.
 * Rectangles are drawn in an SVG whose viewBox equals the original pixel size, so they scale
 * with the image automatically. Labels are HTML positioned in percentages so text keeps a
 * constant readable size at any rendered width.
 */
export default function BoundingBoxOverlay({
  width,
  height,
  detections,
  showBoxes = true,
  showLabels = true,
}: BoundingBoxOverlayProps) {
  if (!showBoxes) return null;

  return (
    <div className="pointer-events-none absolute inset-0" data-testid="bbox-overlay">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        className="absolute inset-0 h-full w-full"
        aria-hidden="true"
      >
        {detections.map((det) => (
          <rect
            key={det.id}
            data-testid={`bbox-${det.id}`}
            x={det.bbox[0]}
            y={det.bbox[1]}
            width={det.bbox[2] - det.bbox[0]}
            height={det.bbox[3] - det.bbox[1]}
            fill="rgba(37, 99, 235, 0.08)"
            stroke="#2563eb"
            strokeWidth={2}
            vectorEffect="non-scaling-stroke"
          />
        ))}
      </svg>

      {showLabels &&
        detections.map((det) => {
          const box = bboxToPercent(det.bbox, width, height);
          return (
            <span
              key={det.id}
              data-testid={`label-${det.id}`}
              className="absolute -translate-y-full whitespace-nowrap rounded-sm bg-blue-600 px-1 text-[11px] leading-4 text-white"
              style={{ left: `${box.left}%`, top: `${box.top}%` }}
            >
              {det.label} {Math.round(det.confidence * 100)}%
            </span>
          );
        })}
    </div>
  );
}
