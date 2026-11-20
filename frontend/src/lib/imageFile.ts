export const ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"] as const;

const DEFAULT_MAX_UPLOAD_MB = 10;

export function getMaxUploadBytes(): number {
  const configured = Number(import.meta.env.VITE_MAX_UPLOAD_MB);
  const mb = Number.isFinite(configured) && configured > 0 ? configured : DEFAULT_MAX_UPLOAD_MB;
  return mb * 1024 * 1024;
}

/**
 * Quick client-side check for a friendlier UX. The backend remains the authority: it validates
 * the decoded content and the pixel dimensions. Returns an error message, or null if acceptable.
 */
export function validateImageFile(
  file: File,
  maxBytes: number = getMaxUploadBytes(),
): string | null {
  if (!(ALLOWED_IMAGE_TYPES as readonly string[]).includes(file.type)) {
    return `"${file.name}" is not a supported image type. Use JPEG, PNG or WebP.`;
  }
  if (file.size === 0) {
    return `"${file.name}" is empty.`;
  }
  if (file.size > maxBytes) {
    const mb = (maxBytes / (1024 * 1024)).toFixed(0);
    return `"${file.name}" is larger than ${mb} MB.`;
  }
  return null;
}
