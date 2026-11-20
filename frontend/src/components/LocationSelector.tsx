import { useId } from "react";
import { isValidPropertyId } from "../lib/propertyId";
import type { LocationOption } from "../types/reference";

interface LocationSelectorProps {
  propertyId: string;
  location: string;
  locations: LocationOption[];
  onPropertyIdChange: (value: string) => void;
  onLocationChange: (value: string) => void;
}

export default function LocationSelector({
  propertyId,
  location,
  locations,
  onPropertyIdChange,
  onLocationChange,
}: LocationSelectorProps) {
  const propertyInputId = useId();
  const locationInputId = useId();
  const showError = propertyId !== "" && !isValidPropertyId(propertyId);

  return (
    <div className="grid gap-4 sm:grid-cols-2">
      <div>
        <label htmlFor={propertyInputId} className="block text-sm font-medium text-neutral-700">
          Property identifier
        </label>
        <input
          id={propertyInputId}
          type="text"
          value={propertyId}
          onChange={(e) => onPropertyIdChange(e.target.value.trim())}
          placeholder="e.g. house-12"
          maxLength={64}
          autoComplete="off"
          aria-invalid={showError}
          className="mt-1 block w-full rounded border border-neutral-300 px-3 py-1.5 text-sm"
        />
        {showError ? (
          <p role="alert" className="mt-1 text-xs text-red-700">
            Use letters, digits, "_", "." or "-" (start with a letter or digit, max 64 characters).
          </p>
        ) : (
          <p className="mt-1 text-xs text-neutral-500">
            A simple label for the property; no accounts are needed in this MVP.
          </p>
        )}
      </div>

      <div>
        <label htmlFor={locationInputId} className="block text-sm font-medium text-neutral-700">
          Location
        </label>
        <select
          id={locationInputId}
          value={location}
          onChange={(e) => onLocationChange(e.target.value)}
          disabled={locations.length === 0}
          className="mt-1 block w-full rounded border border-neutral-300 bg-white px-3 py-1.5 text-sm"
        >
          {locations.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
