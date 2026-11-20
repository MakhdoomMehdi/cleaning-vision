import { describe, expect, it } from "vitest";
import { validateImageFile } from "./imageFile";

const file = (name: string, type: string, bytes: number) =>
  new File([new Uint8Array(bytes)], name, { type });

describe("validateImageFile", () => {
  it.each(["image/jpeg", "image/png", "image/webp"])("accepts %s", (type) => {
    expect(validateImageFile(file("a", type, 100), 1000)).toBeNull();
  });

  it("rejects unsupported types", () => {
    expect(validateImageFile(file("doc.pdf", "application/pdf", 100), 1000)).toMatch(
      /not a supported image type/,
    );
    expect(validateImageFile(file("a.gif", "image/gif", 100), 1000)).not.toBeNull();
  });

  it("rejects empty files", () => {
    expect(validateImageFile(file("a.png", "image/png", 0), 1000)).toMatch(/empty/);
  });

  it("rejects files over the size limit", () => {
    expect(validateImageFile(file("big.png", "image/png", 2 * 1024 * 1024), 1024 * 1024)).toMatch(
      /larger than 1 MB/,
    );
  });
});
