/** Mirrors the backend rule: letters, digits, '_', '.', '-'; starts alphanumeric; max 64 chars. */
const PROPERTY_ID_RE = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/;

export function isValidPropertyId(value: string): boolean {
  return PROPERTY_ID_RE.test(value);
}
