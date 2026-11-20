import { useState } from "react";
import { ApiError, uploadReferences } from "../services/api";
import ImageUploader from "./ImageUploader";

interface ReferenceUploadProps {
  /** A valid property id, or null when not yet entered/valid. */
  propertyId: string | null;
  location: string;
  locationLabel: string;
  onUploaded: () => void;
}

type UploadState =
  | { kind: "idle" }
  | { kind: "uploading" }
  | { kind: "success"; count: number }
  | { kind: "error"; message: string };

export default function ReferenceUpload({
  propertyId,
  location,
  locationLabel,
  onUploaded,
}: ReferenceUploadProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [state, setState] = useState<UploadState>({ kind: "idle" });
  // Remounting the uploader clears its selection and previews after a successful save.
  const [uploaderKey, setUploaderKey] = useState(0);

  const uploading = state.kind === "uploading";
  const canSave = Boolean(propertyId) && Boolean(location) && files.length > 0 && !uploading;

  async function save() {
    if (!propertyId || files.length === 0) return;
    setState({ kind: "uploading" });
    try {
      const result = await uploadReferences(propertyId, location, files);
      setState({ kind: "success", count: result.items.length });
      setFiles([]);
      setUploaderKey((k) => k + 1);
      onUploaded();
    } catch (err) {
      const message = err instanceof ApiError ? err.message : "Upload failed unexpectedly.";
      setState({ kind: "error", message });
    }
  }

  return (
    <div>
      <ImageUploader
        key={uploaderKey}
        label="Reference images"
        multiple
        disabled={uploading}
        onFilesChange={(selected) => {
          setFiles(selected);
          setState({ kind: "idle" });
        }}
      />

      <button
        type="button"
        onClick={save}
        disabled={!canSave}
        className="mt-3 rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {uploading ? "Saving…" : "Save reference images"}
      </button>
      {!propertyId && (
        <p className="mt-2 text-xs text-neutral-500">Enter a valid property identifier first.</p>
      )}

      {state.kind === "error" && (
        <p role="alert" className="mt-3 text-sm text-red-700">
          {state.message}
        </p>
      )}
      {state.kind === "success" && (
        <p role="status" className="mt-3 text-sm text-green-700">
          Saved {state.count} reference image{state.count === 1 ? "" : "s"} for {locationLabel}.
        </p>
      )}
    </div>
  );
}
