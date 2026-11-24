import { useEffect, useState } from "react";
import { ApiError, deleteReference, listReferences, referenceImageUrl } from "../services/api";
import type { ReferenceImage } from "../types/reference";

interface ReferenceGalleryProps {
  /** A valid property id, or null when the user has not entered a valid one yet. */
  propertyId: string | null;
  location: string;
  /** Change this value to force a reload (e.g. after an upload). */
  refreshKey?: number;
}

type GalleryState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; items: ReferenceImage[] };

const DEBOUNCE_MS = 300;

export default function ReferenceGallery({
  propertyId,
  location,
  refreshKey = 0,
}: ReferenceGalleryProps) {
  const [state, setState] = useState<GalleryState>({ kind: "loading" });
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  useEffect(() => {
    if (!propertyId || !location) return;
    const controller = new AbortController();
    setState({ kind: "loading" });

    const timer = setTimeout(() => {
      listReferences(propertyId, location, controller.signal)
        .then((data) => setState({ kind: "ready", items: data.items }))
        .catch((err: unknown) => {
          if (controller.signal.aborted) return;
          const message = err instanceof ApiError ? err.message : "Could not load reference images.";
          setState({ kind: "error", message });
        });
    }, DEBOUNCE_MS);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [propertyId, location, refreshKey]);

  async function remove(item: ReferenceImage) {
    const confirmed = window.confirm(`Delete ${item.original_filename}? The file will be removed.`);
    if (!confirmed) return;
    setDeleteError(null);
    setDeletingId(item.id);
    try {
      await deleteReference(item.id);
      setState((current) =>
        current.kind === "ready"
          ? { kind: "ready", items: current.items.filter((image) => image.id !== item.id) }
          : current,
      );
    } catch (err) {
      const message = err instanceof ApiError ? err.message : "Could not delete that photo.";
      setDeleteError(message);
    } finally {
      setDeletingId(null);
    }
  }

  if (!propertyId) {
    return (
      <p className="text-sm text-neutral-600">
        Enter a property identifier to see its saved reference images.
      </p>
    );
  }
  if (state.kind === "loading") {
    return <p className="text-sm text-neutral-600">Loading reference images…</p>;
  }
  if (state.kind === "error") {
    return (
      <p role="alert" className="text-sm text-red-700">
        {state.message}
      </p>
    );
  }
  if (state.items.length === 0) {
    return (
      <p className="text-sm text-neutral-600">
        No reference images saved for this location yet. Upload some below.
      </p>
    );
  }

  return (
    <div>
      <p className="mb-2 text-sm text-neutral-600">
        {state.items.length} reference image{state.items.length === 1 ? "" : "s"} saved
      </p>
      {deleteError && (
        <p role="alert" className="mb-2 text-sm text-red-700">
          {deleteError}
        </p>
      )}
      <ul className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        {state.items.map((item) => (
          <li
            key={item.id}
            className="overflow-hidden rounded border border-neutral-200 bg-neutral-100"
          >
            <img
              src={referenceImageUrl(item.image_url)}
              alt={`Reference ${item.original_filename}`}
              loading="lazy"
              className="h-28 w-full object-cover"
            />
            <div className="flex items-center justify-between gap-2 px-2 py-1">
              <p className="truncate text-xs text-neutral-600">
                {item.original_filename} · {item.width}×{item.height}
              </p>
              <button
                type="button"
                onClick={() => remove(item)}
                disabled={deletingId === item.id}
                className="shrink-0 text-xs font-medium text-red-700 hover:underline disabled:opacity-50"
              >
                {deletingId === item.id ? "Deleting…" : "Delete"}
              </button>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
