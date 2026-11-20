import type { Detection } from "../types/detection";
import BoundingBoxOverlay from "./BoundingBoxOverlay";

interface AnnotatedImageProps {
  src: string;
  alt: string;
  width: number;
  height: number;
  detections: Detection[];
  showBoxes?: boolean;
  showLabels?: boolean;
}

/** Image plus bounding-box overlay. The wrapper takes the image's aspect ratio, so they align. */
export default function AnnotatedImage({
  src,
  alt,
  width,
  height,
  detections,
  showBoxes,
  showLabels,
}: AnnotatedImageProps) {
  return (
    <div
      className="relative w-full overflow-visible"
      style={{ aspectRatio: `${width} / ${height}` }}
    >
      <img src={src} alt={alt} className="absolute inset-0 h-full w-full object-fill" />
      <BoundingBoxOverlay
        width={width}
        height={height}
        detections={detections}
        showBoxes={showBoxes}
        showLabels={showLabels}
      />
    </div>
  );
}
