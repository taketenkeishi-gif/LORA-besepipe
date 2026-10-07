// tag-editor/parse.js
//
// Canonical tag parsing / formatting helpers. Mirrors the syntax understood by
// the Python backend (core/canonical_tags.py): plain tags, weighted `(tag:1.1)`,
// bracket `[tag]`, double-paren `((tag))`, and `<lora:name:weight>`.
//
// A "tag" object is: { id, type, raw, text?, name?, weight?, ...flags }
// where flags may include: bypassed, locked, color, ai_imported.

let _uid = 0;
/** Monotonic unique id for chip identity across re-renders. */
export function uid() {
  return ++_uid;
}

/**
 * Parse one already-trimmed token into a typed tag descriptor (without id).
 * Returns null for empty input.
 */
export function parseOne(raw) {
  raw = (raw ?? "").trim();
  if (!raw) return null;

  let m = raw.match(/^<lora:([^:>]+):(\d*\.?\d*)>$/);
  if (m) return { type: "lora", name: m[1], weight: parseFloat(m[2]) || 1, raw };

  m = raw.match(/^\((.+):(\d*\.?\d+)\)$/s);
  if (m) return { type: "weighted", text: m[1], weight: parseFloat(m[2]), raw };

  m = raw.match(/^\(\((.+)\)\)$/s);
  if (m) return { type: "dparen", text: m[1], raw };

  m = raw.match(/^\[(.+)\]$/s);
  if (m) return { type: "bracket", text: m[1], raw };

  return { type: "plain", text: raw, raw };
}

/**
 * A visual section divider. Serialized as `{ type: "divider" }` (no `raw`), so
 * the Python backend's coerce_raws drops it (it has no usable `raw`) and it
 * never reaches CLIP. Not editable as text; rendered as a horizontal rule.
 */
export function makeDivider() {
  return { type: "divider", id: uid() };
}

/** Parse one token and attach a fresh id + optional flag overrides. */
export function makeTag(raw, flags = {}) {
  const t = parseOne(raw);
  if (!t) return null;
  return { ...t, id: uid(), ...flags };
}

/** Split a comma-separated prompt respecting nested ()[]<> so weights survive. */
export function splitTags(str) {
  if (!str || !str.trim()) return [];
  const out = [];
  let depth = 0;
  let cur = "";
  for (const ch of str) {
    if ("([<".includes(ch)) depth++;
    else if (")]>".includes(ch)) depth--;
    if (ch === "," && depth === 0) {
      const t = makeTag(cur.trim());
      if (t) out.push(t);
      cur = "";
    } else {
      cur += ch;
    }
  }
  const t = makeTag(cur.trim());
  if (t) out.push(t);
  return out;
}

/** Trigger tags affect CLIP; LoRA weights themselves are owned by the loader. */
export function loraTriggerTags(triggerText) {
  return splitTags(triggerText);
}

/** Remove legacy prompt-only LoRA tokens; actual LoRA application lives upstream. */
export function withoutInertLoraTags(tags) {
  return Array.isArray(tags) ? tags.filter((tag) => tag?.type !== "lora") : [];
}

/** Human label shown on a chip. */
export function chipLabel(tag) {
  switch (tag.type) {
    case "lora": return `⧉ ${tag.name}`;
    case "weighted":
    case "dparen":
    case "bracket": return tag.text;
    default: return tag.text || tag.raw;
  }
}

/** Weight badge text for a chip, or "" if none. */
export function chipWeight(tag) {
  if (tag.type === "weighted") return String(tag.weight);
  if (tag.type === "lora") return String(tag.weight);
  if (tag.type === "bracket") return "0.9";
  if (tag.type === "dparen") return "1.21";
  return "";
}

/** Normalized comparison key for dedup / blocklist matching. */
export function coreKey(tag) {
  switch (tag.type) {
    case "lora": return "lora:" + (tag.name || "").toLowerCase();
    case "weighted":
    case "dparen":
    case "bracket": return (tag.text || "").toLowerCase();
    default: return (tag.text || tag.raw || "").toLowerCase();
  }
}

const round1 = (v) => Math.round(v * 10) / 10;

/**
 * Return a NEW tag (preserving id + flags) with weight adjusted by delta.
 * Plain tags at weight 1.0 collapse back to their bare form.
 */
export function adjustWeight(tag, delta) {
  const flags = _flags(tag);
  let newRaw;
  if (tag.type === "weighted") {
    const w = Math.max(0.1, round1(tag.weight + delta));
    newRaw = w === 1.0 ? tag.text : `(${tag.text}:${w})`;
  } else if (tag.type === "lora") {
    const w = Math.max(0, round1(tag.weight + delta));
    newRaw = `<lora:${tag.name}:${w}>`;
  } else if (tag.type === "plain" || tag.type === "dparen" || tag.type === "bracket") {
    const base = tag.text || tag.raw;
    const baseW = tag.type === "bracket" ? 0.9 : tag.type === "dparen" ? 1.21 : 1.0;
    const w = Math.max(0.1, round1(baseW + delta));
    newRaw = w === 1.0 ? base : `(${base}:${w})`;
  } else {
    return tag;
  }
  const next = makeTag(newRaw, flags);
  if (next) next.id = tag.id;
  return next || tag;
}

/**
 * Return a NEW tag (preserving id + flags) with weight set to an EXACT value.
 * Implemented on top of adjustWeight so all type rules stay in one place.
 */
export function setWeight(tag, w) {
  let cur;
  if (tag.type === "weighted" || tag.type === "lora") cur = tag.weight;
  else if (tag.type === "bracket") cur = 0.9;
  else if (tag.type === "dparen") cur = 1.21;
  else cur = 1.0;
  return adjustWeight(tag, round1(w) - cur);
}

/** Current effective weight of a tag (for popover initial value). */
export function tagWeight(tag) {
  if (tag.type === "weighted" || tag.type === "lora") return tag.weight;
  if (tag.type === "bracket") return 0.9;
  if (tag.type === "dparen") return 1.21;
  return 1.0;
}

/** Toggle emphasis: bare <-> (tag:1.1). Preserves id + flags. */
export function toggleEmphasis(tag) {
  const flags = _flags(tag);
  const base = tag.type === "lora" ? tag.raw : (tag.text || tag.raw);
  let newRaw;
  if (tag.type === "weighted") newRaw = base;
  else newRaw = `(${base}:1.1)`;
  const next = makeTag(newRaw, flags);
  if (next) next.id = tag.id;
  return next || tag;
}

function _flags(tag) {
  const f = {};
  if (tag.bypassed) f.bypassed = true;
  if (tag.locked) f.locked = true;
  if (tag.color) f.color = tag.color;
  if (tag.ai_imported) f.ai_imported = true;
  return f;
}
