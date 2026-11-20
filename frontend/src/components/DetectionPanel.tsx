import { useState } from "react";
import { ApiError, detectImage } from "../services/api";
import { useObjectUrl } from "../lib/useObjectUrl";
import type { DetectResponse } from "../types/detection";
import AnnotatedImage from "./AnnotatedImage";
import DetectionList from "./DetectionList";
import ImageUploader from "./ImageUploader";

type DetectState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "success"; data: DetectResponse }
  | { kind: "error"; message: string };

/** Standalone YOLO check: upload one image, see boxes, labels and confidences. */
export default function DetectionPanel() {
  const [file, setFile] = useState<File | null>(null);
  const [state, setState] = useState<DetectState>({ kind: "idle" });
  const [showBoxes, setShowBoxes] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const previewUrl = useObjectUrl(file);

  function handleFilesChange(files: File[]) {
    setFile(files[0] ?? null);
    setState({ kind: "idle" });
  }

  async function runDetection() {
    if (!file) return;
    setState({ kind: "loading" });
    try {
      const data = await detectImage(file);
      setState({ kind: "success", data });
    } catch (err) {
      const message = err instanceof ApiError ? err.message : "Unexpected error during detection.";
      setState({ kind: "error", message });
    }
  }

  const loading = state.kind === "loading";

  return (
    <section className="rounded-lg border border-neutral-200 bg-white p-4">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-neutral-500">
        Object detection (YOLO)
      </h2>

      <div className="mt-3">
        <ImageUploader label="Image" disabled={loading} onFilesChange={handleFilesChange} />
      </div>

      <button
        type="button"
        onClick={runDetection}
        disabled={!file || loading}
        className="mt-3 rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {loading ? "Detecting…" : "Detect objects"}
      </button>

      {state.kind === "error" && (
        <p role="alert" className="mt-3 text-sm text-red-700">
          {state.message}
        </p>
      )}

      {state.kind === "success" && previewUrl && (
        <div className="mt-4 grid gap-4 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <div className="mb-2 flex gap-4 text-sm">
              <label className="flex items-center gap-1.5">
                <input
                  type="checkbox"
                  checked={showBoxes}
                  onChange={(e) => setShowBoxes(e.target.checked)}
                />
                Show bounding boxes
              </label>
              <label className="flex items-center gap-1.5">
                <input
                  type="checkbox"
                  checked={showLabels}
                  disabled={!showBoxes}
                  onChange={(e) => setShowLabels(e.target.checked)}
                />
                Show object labels
              </label>
            </div>
            <AnnotatedImage
              src={previewUrl}
              alt="Analysed image with detected objects"
              width={state.data.width}
              height={state.data.height}
              detections={state.data.detections}
              showBoxes={showBoxes}
              showLabels={showLabels}
            />
          </div>
          <div>
            <h3 className="mb-1 text-sm font-medium">
              Detected objects ({state.data.detections.length})
            </h3>
            <DetectionList detections={state.data.detections} />
            <p className="mt-3 text-xs text-neutral-500">
              Model {state.data.model.name} on {state.data.model.device}, confidence ≥{" "}
              {state.data.model.conf_threshold}. {state.data.note}
            </p>
          </div>
        </div>
      )}
    </section>
  );
}
