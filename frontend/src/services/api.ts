import type { HealthResponse } from "../types/health";
import type {
  InspectionListResponse,
  InspectionResults,
  InspectionSummary,
} from "../types/inspection";
import type { LocationListResponse, ReferenceListResponse } from "../types/reference";

export const API_BASE_URL: string = (
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000"
).replace(/\/+$/, "");

export class ApiError extends Error {
  readonly status: number | null;

  constructor(message: string, status: number | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch {
    throw new ApiError(`Cannot reach the backend at ${API_BASE_URL}. Is it running?`);
  }

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // Non-JSON error body; keep the generic message.
    }
    throw new ApiError(detail, response.status);
  }

  return (await response.json()) as T;
}

export function getLocations(signal?: AbortSignal): Promise<LocationListResponse> {
  return request<LocationListResponse>("/api/v1/locations", { signal });
}

export function listReferences(
  propertyId: string,
  location: string,
  signal?: AbortSignal,
): Promise<ReferenceListResponse> {
  const query = new URLSearchParams({ property_id: propertyId, location });
  return request<ReferenceListResponse>(`/api/v1/references?${query.toString()}`, { signal });
}

export function uploadReferences(
  propertyId: string,
  location: string,
  files: File[],
  signal?: AbortSignal,
): Promise<ReferenceListResponse> {
  const form = new FormData();
  form.append("property_id", propertyId);
  form.append("location", location);
  for (const file of files) form.append("files", file);
  return request<ReferenceListResponse>("/api/v1/references", {
    method: "POST",
    body: form,
    signal,
  });
}

export async function deleteReference(referenceId: string, signal?: AbortSignal): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/api/v1/references/${referenceId}`, {
      method: "DELETE",
      signal,
    });
  } catch {
    throw new ApiError(`Cannot reach the backend at ${API_BASE_URL}. Is it running?`);
  }
  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // Empty or non-JSON error body.
    }
    throw new ApiError(detail, response.status);
  }
}

export function referenceImageUrl(imagePath: string): string {
  return `${API_BASE_URL}${imagePath}`;
}

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return request<HealthResponse>("/health", { signal });
}

export function createInspection(
  propertyId: string,
  location: string,
  files: File[],
  signal?: AbortSignal,
): Promise<InspectionSummary> {
  const form = new FormData();
  form.append("property_id", propertyId);
  form.append("location", location);
  for (const file of files) form.append("files", file);
  return request<InspectionSummary>("/api/v1/inspections", {
    method: "POST",
    body: form,
    signal,
  });
}

export function getInspectionResults(
  inspectionId: string,
  signal?: AbortSignal,
): Promise<InspectionResults> {
  return request<InspectionResults>(`/api/v1/inspections/${inspectionId}/results`, { signal });
}

export function listInspections(
  propertyId: string,
  location: string,
  signal?: AbortSignal,
): Promise<InspectionListResponse> {
  const query = new URLSearchParams({ property_id: propertyId, location });
  return request<InspectionListResponse>(`/api/v1/inspections?${query.toString()}`, { signal });
}

export function inspectionImageUrl(imagePath: string): string {
  return `${API_BASE_URL}${imagePath}`;
}
