import { useCallback, useEffect, useState } from "react";
import InspectionResultView from "../components/InspectionResultView";
import InspectionUpload from "../components/InspectionUpload";
import LocationSelector from "../components/LocationSelector";
import ReferenceGallery from "../components/ReferenceGallery";
import ReferenceUpload from "../components/ReferenceUpload";
import { isValidPropertyId } from "../lib/propertyId";
import { ApiError, getHealth, getInspectionResults, getLocations, listInspections } from "../services/api";
import type { HealthResponse } from "../types/health";
import type { InspectionResults, InspectionSummary } from "../types/inspection";
import type { LocationOption } from "../types/reference";

type HealthState =
  | { kind: "loading" }
  | { kind: "ok"; data: HealthResponse }
  | { kind: "error"; message: string };

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="rounded-lg border border-neutral-200 bg-white p-4">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-neutral-500">{title}</h2>
      <div className="mt-3">{children}</div>
    </section>
  );
}

export default function Dashboard() {
  const [health, setHealth] = useState<HealthState>({ kind: "loading" });
  const [locations, setLocations] = useState<LocationOption[]>([]);
  const [locationsError, setLocationsError] = useState<string | null>(null);
  const [propertyId, setPropertyId] = useState("");
  const [location, setLocation] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);
  const [results, setResults] = useState<InspectionResults | null>(null);
  const [history, setHistory] = useState<InspectionSummary[]>([]);
  const [referenceCount, setReferenceCount] = useState(0);
  const handleReferenceCount = useCallback((count: number) => setReferenceCount(count), []);

  useEffect(() => {
    const controller = new AbortController();
    getHealth(controller.signal)
      .then((data) => setHealth({ kind: "ok", data }))
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        const message = err instanceof ApiError ? err.message : "Unexpected error";
        setHealth({ kind: "error", message });
      });
    getLocations(controller.signal)
      .then((data) => {
        setLocations(data.items);
        setLocation((current) => current || data.items[0]?.value || "");
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setLocationsError(err instanceof ApiError ? err.message : "Could not load locations.");
      });
    return () => controller.abort();
  }, []);

  const validPropertyId = isValidPropertyId(propertyId) ? propertyId : null;
  const locationLabel = locations.find((l) => l.value === location)?.label ?? location;

  useEffect(() => {
    if (!validPropertyId || !location) {
      setHistory([]);
      return;
    }
    const controller = new AbortController();
    listInspections(validPropertyId, location, controller.signal)
      .then((data) => setHistory(data.items))
      .catch(() => setHistory([]));
    return () => controller.abort();
  }, [validPropertyId, location, refreshKey, results?.inspection_id]);

  return (
    <main className="mx-auto max-w-5xl space-y-6 px-4 py-6">
      <Card title="Backend status">
        {health.kind === "loading" && (
          <p className="text-sm text-neutral-600">Checking backend…</p>
        )}

        {health.kind === "error" && (
          <p role="alert" className="text-sm text-red-700">
            {health.message}
          </p>
        )}

        {health.kind === "ok" && (
          <dl className="grid grid-cols-1 gap-x-6 gap-y-1 text-sm sm:grid-cols-2">
            <dt className="text-neutral-500">Status</dt>
            <dd className="font-medium text-green-700">{health.data.status}</dd>
            <dt className="text-neutral-500">OpenAI key configured</dt>
            <dd>{health.data.openai_configured ? "yes" : "no — comparison disabled"}</dd>
            <dt className="text-neutral-500">OpenAI model</dt>
            <dd>{health.data.openai_model}</dd>
          </dl>
        )}
      </Card>

      <Card title="1. Property and location">
        {locationsError && (
          <p role="alert" className="mb-3 text-sm text-red-700">
            {locationsError}
          </p>
        )}
        <LocationSelector
          propertyId={propertyId}
          location={location}
          locations={locations}
          onPropertyIdChange={setPropertyId}
          onLocationChange={setLocation}
        />
        <h3 className="mb-2 mt-5 text-sm font-medium">Saved reference (before) images</h3>
        <ReferenceGallery
          propertyId={validPropertyId}
          location={location}
          refreshKey={refreshKey}
          onCount={handleReferenceCount}
        />
      </Card>

      <Card title="2. Upload the before / reference photo">
        <ReferenceUpload
          propertyId={validPropertyId}
          location={location}
          locationLabel={locationLabel}
          onUploaded={() => setRefreshKey((k) => k + 1)}
        />
      </Card>

      <Card title="3. Upload the after photo and compare">
        <InspectionUpload
          propertyId={validPropertyId}
          location={location}
          locationLabel={locationLabel}
          hasReferences={referenceCount > 0}
          onComplete={(next) => setResults(next)}
        />
      </Card>

      {results && (
        <Card title={results.readiness ? "Guest readiness" : "Comparison result"}>
          <InspectionResultView results={results} />
        </Card>
      )}

      {history.length > 0 && (
        <Card title="Recent inspections">
          <ul className="space-y-2 text-sm">
            {history.map((item) => (
              <li key={item.id}>
                <button
                  type="button"
                  className="text-blue-700 underline"
                  onClick={() => {
                    getInspectionResults(item.id).then(setResults).catch(() => undefined);
                  }}
                >
                  {new Date(item.created_at).toLocaleString()} — {item.status}
                  {item.decision ? ` — ${item.decision.split("_").join(" ")}` : ""}
                  {item.change_count != null ? ` (${item.change_count} changes)` : ""}
                </button>
              </li>
            ))}
          </ul>
        </Card>
      )}

    </main>
  );
}
