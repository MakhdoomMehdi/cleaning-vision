import { useState } from "react";
import { ApiError, createInspection, getInspectionResults } from "../services/api";
import type { InspectionResults } from "../types/inspection";
import ImageUploader from "./ImageUploader";

interface InspectionUploadProps {
  propertyId: string | null;
  location: string;
  locationLabel: string;
  onComplete: (results: InspectionResults) => void;
}

type State =
  | { kind: "idle" }
  | { kind: "running" }
  | { kind: "error"; message: string };

export default function InspectionUpload({
  propertyId,
  location,
  locationLabel,
  onComplete,
}: InspectionUploadProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [state, setState] = useState<State>({ kind: "idle" });
  const [uploaderKey, setUploaderKey] = useState(0);

  const running = state.kind === "running";
  const canRun = Boolean(propertyId) && Boolean(location) && files.length > 0 && !running;

  async function run() {
    if (!propertyId || files.length === 0) return;
    setState({ kind: "running" });
    try {
      const created = await createInspection(propertyId, location, files);
      const results = await getInspectionResults(created.id);
      setFiles([]);
      setUploaderKey((k) => k + 1);
      setState({ kind: "idle" });
      onComplete(results);
    } catch (err) {
      const message = err instanceof ApiError ? err.message : "Inspection failed unexpectedly.";
      setState({ kind: "error", message });
    }
  }

  return (
    <div>
      <p className="mb-3 text-sm text-neutral-600">
        Upload the <strong>after</strong> photo of the {locationLabel || "room"}. The saved before
        photos are sent with it. This can take up to a minute.
      </p>
      <ImageUploader
        key={uploaderKey}
        label="After photo"
        multiple={false}
        disabled={running}
        onFilesChange={(selected) => {
          setFiles(selected);
          setState({ kind: "idle" });
        }}
      />
      <button
        type="button"
        onClick={run}
        disabled={!canRun}
        className="mt-3 rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {running ? "Comparing photos…" : "Compare with reference"}
      </button>
      {!propertyId && (
        <p className="mt-2 text-xs text-neutral-500">Enter a valid property identifier first.</p>
      )}
      {state.kind === "error" && (
        <p role="alert" className="mt-3 text-sm text-red-700">
          {state.message}
        </p>
      )}
    </div>
  );
}
