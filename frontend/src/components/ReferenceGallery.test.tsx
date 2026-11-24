import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import * as api from "../services/api";
import type { ReferenceImage, ReferenceListResponse } from "../types/reference";
import ReferenceGallery from "./ReferenceGallery";

const ref = (id: string, name: string): ReferenceImage => ({
  id,
  property_id: "house-1",
  location: "garage",
  original_filename: name,
  mime_type: "image/jpeg",
  width: 800,
  height: 600,
  size_bytes: 1234,
  created_at: "2026-11-19T00:00:00Z",
  image_url: `/api/v1/references/${id}/image`,
});

const list = (items: ReferenceImage[]): ReferenceListResponse => ({ items, total: items.length });

describe("ReferenceGallery", () => {
  it("asks for a property id and makes no request when none is valid", () => {
    const spy = vi.spyOn(api, "listReferences");
    render(<ReferenceGallery propertyId={null} location="garage" />);

    expect(screen.getByText(/Enter a property identifier/)).toBeInTheDocument();
    expect(spy).not.toHaveBeenCalled();
  });

  it("shows a loading state, then the saved reference images served by the API", async () => {
    vi.spyOn(api, "listReferences").mockResolvedValue(list([ref("a1", "wall.jpg"), ref("b2", "door.jpg")]));
    render(<ReferenceGallery propertyId="house-1" location="garage" />);

    expect(screen.getByText("Loading reference images…")).toBeInTheDocument();

    const img = await screen.findByAltText("Reference wall.jpg");
    expect(img).toHaveAttribute("src", expect.stringMatching(/\/api\/v1\/references\/a1\/image$/));
    expect(screen.getByText("2 reference images saved")).toBeInTheDocument();
    expect(api.listReferences).toHaveBeenCalledWith("house-1", "garage", expect.any(AbortSignal));
  });

  it("shows an empty state when the location has no references", async () => {
    vi.spyOn(api, "listReferences").mockResolvedValue(list([]));
    render(<ReferenceGallery propertyId="house-1" location="garage" />);

    expect(await screen.findByText(/No reference images saved for this location yet/)).toBeInTheDocument();
  });

  it("shows API failures", async () => {
    vi.spyOn(api, "listReferences").mockRejectedValue(new api.ApiError("Backend down", 500));
    render(<ReferenceGallery propertyId="house-1" location="garage" />);

    expect(await screen.findByRole("alert")).toHaveTextContent("Backend down");
  });

  it("deletes a reference image and removes it from the gallery", async () => {
    vi.spyOn(api, "listReferences").mockResolvedValue(list([ref("a1", "wall.jpg"), ref("b2", "door.jpg")]));
    const remove = vi.spyOn(api, "deleteReference").mockResolvedValue();
    vi.spyOn(window, "confirm").mockReturnValue(true);
    const user = userEvent.setup();
    render(<ReferenceGallery propertyId="house-1" location="garage" />);

    const [deleteWall] = await screen.findAllByRole("button", { name: "Delete" });
    await user.click(deleteWall);

    expect(remove).toHaveBeenCalledWith("a1");
    expect(await screen.findByText("1 reference image saved")).toBeInTheDocument();
    expect(screen.queryByAltText("Reference wall.jpg")).not.toBeInTheDocument();
    expect(screen.getByAltText("Reference door.jpg")).toBeInTheDocument();
  });

  it("reloads when refreshKey changes", async () => {
    const spy = vi.spyOn(api, "listReferences").mockResolvedValue(list([]));
    const { rerender } = render(<ReferenceGallery propertyId="house-1" location="garage" refreshKey={0} />);
    await screen.findByText(/No reference images/);

    rerender(<ReferenceGallery propertyId="house-1" location="garage" refreshKey={1} />);
    await screen.findByText(/No reference images/);
    expect(spy).toHaveBeenCalledTimes(2);
  });
});
