import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Detection } from "../types/detection";
import BoundingBoxOverlay from "./BoundingBoxOverlay";

const detections: Detection[] = [
  { id: "det_001", label: "chair", confidence: 0.94, bbox: [128, 72, 640, 360] },
];

describe("BoundingBoxOverlay", () => {
  it("uses the original image size as the SVG coordinate space so boxes scale on resize", () => {
    render(<BoundingBoxOverlay width={1280} height={720} detections={detections} />);

    const svg = screen.getByTestId("bbox-overlay").querySelector("svg");
    expect(svg).toHaveAttribute("viewBox", "0 0 1280 720");

    const rect = screen.getByTestId("bbox-det_001");
    expect(rect).toHaveAttribute("x", "128");
    expect(rect).toHaveAttribute("y", "72");
    expect(rect).toHaveAttribute("width", "512");
    expect(rect).toHaveAttribute("height", "288");
  });

  it("positions labels in percentages of the image and shows confidence", () => {
    render(<BoundingBoxOverlay width={1280} height={720} detections={detections} />);

    const label = screen.getByTestId("label-det_001");
    expect(label).toHaveTextContent("chair 94%");
    expect(label).toHaveStyle({ left: "10%", top: "10%" });
  });

  it("hides labels but keeps boxes when labels are toggled off", () => {
    render(
      <BoundingBoxOverlay width={1280} height={720} detections={detections} showLabels={false} />,
    );
    expect(screen.getByTestId("bbox-det_001")).toBeInTheDocument();
    expect(screen.queryByTestId("label-det_001")).not.toBeInTheDocument();
  });

  it("renders nothing when boxes are toggled off", () => {
    render(
      <BoundingBoxOverlay width={1280} height={720} detections={detections} showBoxes={false} />,
    );
    expect(screen.queryByTestId("bbox-overlay")).not.toBeInTheDocument();
  });

  it("renders an empty overlay for zero detections", () => {
    render(<BoundingBoxOverlay width={1280} height={720} detections={[]} />);
    expect(screen.getByTestId("bbox-overlay").querySelectorAll("rect")).toHaveLength(0);
  });
});
