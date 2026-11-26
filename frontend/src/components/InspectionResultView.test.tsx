import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { InspectionResults } from "../types/inspection";
import InspectionResultView from "./InspectionResultView";

const results: InspectionResults = {
  inspection_id: "insp-1",
  property_id: "house-1",
  location: "bedroom",
  status: "completed",
  failed_stage: null,
  error_message: null,
  readiness: false,
  guest_ready: null,
  overall_score: null,
  retake: false,
  images: [
    {
      image_id: "ref_1",
      role: "reference",
      original_filename: "before.jpg",
      width: 800,
      height: 600,
      image_url: "/api/v1/inspections/insp-1/images/ref_1",
    },
    {
      image_id: "new_1",
      role: "inspection",
      original_filename: "after.jpg",
      width: 800,
      height: 600,
      image_url: "/api/v1/inspections/insp-1/images/new_1",
    },
  ],
  matched_image_id: "ref_1",
  damage: ["scratch on the door"],
  missing: ["remote on the nightstand"],
  new: [],
  boxes: [
    {
      image_id: "new_1",
      kind: "damage",
      description: "scratch on the door",
      x1: 0.1,
      y1: 0.2,
      x2: 0.3,
      y2: 0.4,
      origin_x: null,
      origin_y: null,
    },
    {
      image_id: "new_1",
      kind: "missing",
      description: "remote on the nightstand",
      x1: 0.5,
      y1: 0.5,
      x2: 0.7,
      y2: 0.8,
      origin_x: null,
      origin_y: null,
    },
  ],
  summary: "A scratch and a missing remote.",
  views_compared: "The bed wall.",
  decision: "potential_issues",
  vision_status: "completed",
  vision_error: null,
  note: null,
};

describe("InspectionResultView", () => {
  it("lists each finding under the photo in the same color as its box", () => {
    const { container } = render(<InspectionResultView results={results} />);

    const scratch = screen.getByText("Damaged: scratch on the door");
    const remote = screen.getByText("Missing: remote on the nightstand");
    const boxes = container.querySelectorAll(".border-2");

    expect(screen.queryByText("Was here")).not.toBeInTheDocument();
    expect(boxes).toHaveLength(2);
    expect(scratch).toHaveStyle({ color: "rgb(37, 99, 235)" });
    expect(boxes[0]).toHaveStyle({ borderColor: "rgb(37, 99, 235)" });
    expect(remote).toHaveStyle({ color: "rgb(234, 88, 12)" });
    expect(boxes[1]).toHaveStyle({ borderColor: "rgb(234, 88, 12)" });
  });
});
