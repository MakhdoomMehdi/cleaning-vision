import type { BBox } from "../types/detection";

export interface PercentBox {
  left: number;
  top: number;
  width: number;
  height: number;
}

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

/**
 * Convert a bbox in original-image pixels to percentages of the image size.
 * Percentages stay correct at any rendered size, so boxes scale with the image when it is resized.
 */
export function bboxToPercent(bbox: BBox, imageWidth: number, imageHeight: number): PercentBox {
  if (imageWidth <= 0 || imageHeight <= 0) {
    throw new Error("Image dimensions must be positive");
  }
  const x1 = clamp(Math.min(bbox[0], bbox[2]), 0, imageWidth);
  const x2 = clamp(Math.max(bbox[0], bbox[2]), 0, imageWidth);
  const y1 = clamp(Math.min(bbox[1], bbox[3]), 0, imageHeight);
  const y2 = clamp(Math.max(bbox[1], bbox[3]), 0, imageHeight);
  return {
    left: (x1 / imageWidth) * 100,
    top: (y1 / imageHeight) * 100,
    width: ((x2 - x1) / imageWidth) * 100,
    height: ((y2 - y1) / imageHeight) * 100,
  };
}
