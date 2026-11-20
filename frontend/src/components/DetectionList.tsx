import type { Detection } from "../types/detection";

export default function DetectionList({ detections }: { detections: Detection[] }) {
  if (detections.length === 0) {
    return (
      <p className="text-sm text-neutral-600">
        No objects of a supported class were detected. This does not mean the scene is empty:
        the model only recognises a fixed set of classes.
      </p>
    );
  }

  return (
    <ul className="divide-y divide-neutral-200 text-sm">
      {detections.map((det) => (
        <li key={det.id} className="flex items-center justify-between gap-3 py-1.5">
          <span>
            <span className="font-medium">{det.label}</span>{" "}
            <span className="text-neutral-500">({det.id})</span>
          </span>
          <span className="text-neutral-700">{Math.round(det.confidence * 100)}%</span>
        </li>
      ))}
    </ul>
  );
}
