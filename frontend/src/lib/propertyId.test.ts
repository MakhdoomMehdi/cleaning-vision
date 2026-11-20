import { describe, expect, it } from "vitest";
import { isValidPropertyId } from "./propertyId";

describe("isValidPropertyId", () => {
  it.each(["house-12", "A1", "unit_3.b", "x".repeat(64)])("accepts %s", (value) => {
    expect(isValidPropertyId(value)).toBe(true);
  });

  it.each(["", " ", "-lead", "a b", "../etc", "a/b", "x".repeat(65)])("rejects %j", (value) => {
    expect(isValidPropertyId(value)).toBe(false);
  });
});
