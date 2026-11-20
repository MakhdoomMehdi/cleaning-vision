import { useEffect, useState } from "react";
import { ApiError, listReferences, referenceImageUrl } from "../services/api";
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
            <p className="truncate px-2 py-1 text-xs text-neutral-600">
              {item.original_filename} · {item.width}×{item.height}
            </p>
          </li>
        ))}
      </ul>
    </div>
  );
}
