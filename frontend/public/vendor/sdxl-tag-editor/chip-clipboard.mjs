// Structured clipboard shared by every Tag Editor node in this page.
// The OS clipboard remains plain text so chips can still be pasted elsewhere.

let entries = [];
export const CHIP_CLIPBOARD_MIME = "application/x-sdxl-tag-chips+json";

export function parseChipClipboard(value) {
  try {
    const data = JSON.parse(value);
    if (!Array.isArray(data) || !data.length || data.some((e) => !e || (e.type !== "divider" && typeof e.raw !== "string"))) return [];
    return data.map((e) => e.type === "divider" ? { type: "divider" } : {
      raw: e.raw, flags: Object.fromEntries(Object.entries(e.flags || {}).filter(([k, v]) =>
        k === "color" ? typeof v === "string" : ["bypassed", "locked", "ai_imported"].includes(k) && v === true)),
    });
  } catch { return []; }
}

export function writeChipClipboard(nextEntries) {
  entries = nextEntries.map((entry) => entry.type === "divider"
    ? { type: "divider" }
    : { raw: entry.raw, flags: { ...(entry.flags || {}) } });
}

export function readChipClipboard() {
  return entries.map((entry) => entry.type === "divider"
    ? { type: "divider" }
    : { raw: entry.raw, flags: { ...entry.flags } });
}
