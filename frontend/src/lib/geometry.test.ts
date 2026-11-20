import { describe, expect, it } from "vitest";
import { bboxToPercent } from "./geometry";

describe("bboxToPercent", () => {
  it("converts original-pixel boxes to percentages of the image", () => {
    expect(bboxToPercent([128, 72, 640, 360], 1280, 720)).toEqual({
      left: 10,
      top: 10,
      width: 40,
      height: 40,
    });
  });

  it("gives identical percentages regardless of rendered size (scaling is size-independent)", () => {
    const a = bboxToPercent([100, 50, 300, 250], 1000, 500);
    const b = bboxToPercent([200, 100, 600, 500], 2000, 1000); // same scene at 2x resolution
    expect(a).toEqual(b);
  });

  it("clamps boxes to the image and normalises swapped corners", () => {
    const box = bboxToPercent([1300, 300, -10, 100], 1000, 500);
    expect(box).toEqual({ left: 0, top: 20, width: 100, height: 40 });
  });

  it("rejects non-positive image dimensions", () => {
    expect(() => bboxToPercent([0, 0, 1, 1], 0, 100)).toThrow();
  });
});
