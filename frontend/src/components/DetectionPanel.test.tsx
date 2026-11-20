import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import * as api from "../services/api";
import type { DetectResponse } from "../types/detection";
import DetectionPanel from "./DetectionPanel";

const png = () => new File([new Uint8Array(10)], "room.png", { type: "image/png" });

const response = (detections: DetectResponse["detections"]): DetectResponse => ({
  image_id: "img_001",
  width: 1280,
  height: 720,
  detections,
  model: { name: "yolo11n.pt", conf_threshold: 0.25, iou_threshold: 0.45, device: "cpu" },
  note: "Detections only cover the classes the model was trained on.",
});

async function selectFile(file: File) {
  const user = userEvent.setup();
  render(<DetectionPanel />);
  await user.upload(screen.getByLabelText("Image"), file);
  return user;
}

describe("DetectionPanel", () => {
  it("disables the button until a valid image is selected, then shows a preview", async () => {
    render(<DetectionPanel />);
    expect(screen.getByRole("button", { name: "Detect objects" })).toBeDisabled();

    await userEvent.upload(screen.getByLabelText("Image"), png());
    expect(screen.getByAltText("Preview of room.png")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Detect objects" })).toBeEnabled();
  });

  it("rejects an unsupported file type with a visible error", async () => {
    const user = userEvent.setup({ applyAccept: false });
    render(<DetectionPanel />);
    await user.upload(
      screen.getByLabelText("Image"),
      new File(["x"], "notes.pdf", { type: "application/pdf" }),
    );

    expect(screen.getByRole("alert")).toHaveTextContent("not a supported image type");
    expect(screen.getByRole("button", { name: "Detect objects" })).toBeDisabled();
  });

  it("shows a loading state while detecting, then renders boxes and the list", async () => {
    let resolve!: (value: DetectResponse) => void;
    vi.spyOn(api, "detectImage").mockReturnValue(new Promise((r) => (resolve = r)));

    const user = await selectFile(png());
    await user.click(screen.getByRole("button", { name: "Detect objects" }));
    expect(screen.getByRole("button", { name: "Detecting…" })).toBeDisabled();

    resolve(response([{ id: "det_001", label: "chair", confidence: 0.94, bbox: [10, 20, 300, 400] }]));

    expect(await screen.findByTestId("bbox-det_001")).toBeInTheDocument();
    expect(screen.getByText("Detected objects (1)")).toBeInTheDocument();
    expect(screen.getByTestId("label-det_001")).toHaveTextContent("chair 94%");
  });

  it("explains empty results without claiming the scene is empty", async () => {
    vi.spyOn(api, "detectImage").mockResolvedValue(response([]));

    const user = await selectFile(png());
    await user.click(screen.getByRole("button", { name: "Detect objects" }));

    expect(await screen.findByText(/does not mean the scene is empty/)).toBeInTheDocument();
  });

  it("shows API failures and lets the user retry", async () => {
    const spy = vi
      .spyOn(api, "detectImage")
      .mockRejectedValueOnce(new api.ApiError("Object detection unavailable", 503))
      .mockResolvedValueOnce(response([]));

    const user = await selectFile(png());
    await user.click(screen.getByRole("button", { name: "Detect objects" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Object detection unavailable");

    await user.click(screen.getByRole("button", { name: "Detect objects" }));
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
    expect(spy).toHaveBeenCalledTimes(2);
  });
});
