import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import * as api from "../services/api";
import ReferenceUpload from "./ReferenceUpload";

const png = (name = "room.png") => new File([new Uint8Array(10)], name, { type: "image/png" });

function setup(propertyId: string | null = "house-1") {
  const onUploaded = vi.fn();
  render(
    <ReferenceUpload
      propertyId={propertyId}
      location="garage"
      locationLabel="Garage"
      onUploaded={onUploaded}
    />,
  );
  return { onUploaded, user: userEvent.setup() };
}

const saveButton = () => screen.getByRole("button", { name: /Save reference images|Saving/ });

describe("ReferenceUpload", () => {
  it("keeps Save disabled until a property id and at least one valid image exist", async () => {
    const { user } = setup();
    expect(saveButton()).toBeDisabled();

    await user.upload(screen.getByLabelText("Reference images"), png());
    expect(saveButton()).toBeEnabled();
  });

  it("explains that a valid property identifier is needed first", async () => {
    const { user } = setup(null);
    await user.upload(screen.getByLabelText("Reference images"), png());

    expect(saveButton()).toBeDisabled();
    expect(screen.getByText(/Enter a valid property identifier first/)).toBeInTheDocument();
  });

  it("previews several selected images", async () => {
    const { user } = setup();
    await user.upload(screen.getByLabelText("Reference images"), [png("a.png"), png("b.png")]);

    expect(screen.getByAltText("Preview of a.png")).toBeInTheDocument();
    expect(screen.getByAltText("Preview of b.png")).toBeInTheDocument();
  });

  it("uploads all files, reports success, clears previews and notifies the parent", async () => {
    let resolve!: (value: { items: never[]; total: number }) => void;
    const spy = vi.spyOn(api, "uploadReferences").mockReturnValue(
      new Promise((r) => (resolve = r as typeof resolve)),
    );
    const { user, onUploaded } = setup();
    await user.upload(screen.getByLabelText("Reference images"), [png("a.png"), png("b.png")]);
    await user.click(saveButton());

    expect(screen.getByRole("button", { name: "Saving…" })).toBeDisabled();
    expect(spy).toHaveBeenCalledWith("house-1", "garage", [
      expect.objectContaining({ name: "a.png" }),
      expect.objectContaining({ name: "b.png" }),
    ]);

    resolve({ items: [{}, {}] as never[], total: 2 });
    expect(await screen.findByRole("status")).toHaveTextContent("Saved 2 reference images for Garage.");
    expect(onUploaded).toHaveBeenCalledTimes(1);
    expect(screen.queryByAltText("Preview of a.png")).not.toBeInTheDocument();
    expect(saveButton()).toBeDisabled();
  });

  it("shows the backend error and keeps the selection so the user can retry", async () => {
    vi.spyOn(api, "uploadReferences").mockRejectedValue(
      new api.ApiError("bad.jpg: File is not a valid image.", 415),
    );
    const { user, onUploaded } = setup();
    await user.upload(screen.getByLabelText("Reference images"), png("bad.png"));
    await user.click(saveButton());

    expect(await screen.findByRole("alert")).toHaveTextContent("File is not a valid image.");
    expect(onUploaded).not.toHaveBeenCalled();
    expect(screen.getByAltText("Preview of bad.png")).toBeInTheDocument();
    expect(saveButton()).toBeEnabled();
  });
});
