// tag-editor/lora-scan.js
//
// Scans the CURRENT ComfyUI workflow graph for LoRA loader nodes, ported
// from the original widget's `_scanLoraNodes`. Takes a plain array of
// LiteGraph nodes (injected by unified_tag_editor.js via `getGraphNodes()`
// so this module stays host-agnostic like the rest of tag-editor/*) and
// returns every LoRA it can find, deduplicated by filename.
//
// Four strategies, matching the original:
//   D — UnifiedModelStack-family nodes: parse their `lora_stack` JSON widget.
//   A — standard LoraLoader/LoraLoaderModelOnly-style nodes: a widget named
//       lora / lora_name / lora_file, paired with a strength/weight widget.
//   B — any node whose TYPE contains "lora" (third-party nodes like "Power
//       Lora Loader"): scan its widgets for a *.safetensors/*.ckpt/*.pt value.
//   C — legacy fallback: scan raw `widgets_values` for the same file pattern.

const FILE_RE = /\.(safetensors|ckpt|pt)$/i;
const LORA_WIDGET_RE = /^lora(?:_name|_file)?$/i;
const STRENGTH_WIDGET_RE = /strength|weight|lora_wt/i;

function widgetValue(node, re) {
  const w = node.widgets?.find((x) => re.test(x.name || ""));
  return w ? w.value : undefined;
}

function round2(v) {
  const n = parseFloat(v);
  return Number.isFinite(n) ? Math.round(n * 100) / 100 : 1;
}

function addResult(results, seen, name, strength) {
  if (!name) return;
  const key = String(name).toLowerCase();
  if (seen.has(key)) return;
  seen.add(key);
  results.push({ name: String(name), strength: round2(strength ?? 1) });
}

/** Strategy D: UnifiedModelStack's lora_stack JSON widget. */
function scanModelStackNode(node, results, seen) {
  const w = node.widgets?.find((x) => x.name === "lora_stack");
  if (!w || typeof w.value !== "string") return;
  let entries;
  try { entries = JSON.parse(w.value); } catch { return; }
  if (!Array.isArray(entries)) return;
  for (const e of entries) {
    if (!e || e.enabled === false) continue;
    addResult(results, seen, e.lora_name, e.strength_model ?? e.strength);
  }
}

/** Strategy A: a widget literally named lora/lora_name/lora_file. */
function scanNamedLoraWidget(node, results, seen) {
  const name = widgetValue(node, LORA_WIDGET_RE);
  if (!name) return false;
  const strength = widgetValue(node, STRENGTH_WIDGET_RE);
  addResult(results, seen, name, strength);
  return true;
}

/** Strategy B: node.type mentions "lora" — scan widgets for a filename. */
function scanLoraTypedNode(node, results, seen) {
  for (const w of node.widgets || []) {
    if (typeof w.value === "string" && FILE_RE.test(w.value)) {
      addResult(results, seen, w.value, 1);
      return;
    }
  }
}

/** Strategy C: legacy fallback over raw widgets_values. */
function scanWidgetsValuesFallback(node, results, seen) {
  const wv = node.widgets_values;
  if (!Array.isArray(wv)) return;
  for (let i = 0; i < wv.length; i++) {
    if (typeof wv[i] === "string" && FILE_RE.test(wv[i])) {
      const next = wv[i + 1];
      addResult(results, seen, wv[i], typeof next === "number" ? next : 1);
    }
  }
}

/** Scan `nodes` (a LiteGraph node array) for LoRAs. Returns [{name, strength}]. */
export function scanLoraNodes(nodes) {
  const results = [];
  const seen = new Set();
  for (const node of nodes || []) {
    const type = node?.type || node?.comfyClass || "";
    if (/UnifiedModelStack/i.test(type)) {
      scanModelStackNode(node, results, seen);
      continue;
    }
    if (scanNamedLoraWidget(node, results, seen)) continue;
    if (/lora/i.test(type)) {
      scanLoraTypedNode(node, results, seen);
      continue;
    }
    scanWidgetsValuesFallback(node, results, seen);
  }
  return results;
}

/** raw "path/to/Name.safetensors" -> "Name.safetensors". */
export function loraShortName(name) {
  return String(name).replace(/\\/g, "/").split("/").pop();
}

// ── Trigger-word DB (localStorage, per LoRA filename) ────────────────────────
const TRIGGER_LS_KEY = "ute_te_lora_triggers";

function loadTriggerDB() {
  try { return JSON.parse(localStorage.getItem(TRIGGER_LS_KEY) || "{}"); } catch { return {}; }
}
function saveTriggerDB(db) {
  try { localStorage.setItem(TRIGGER_LS_KEY, JSON.stringify(db)); } catch { /* quota / disabled */ }
}

export const LoraTriggerDB = {
  /** Stored trigger tags for `loraName`, or an underscore->space guess if unset. */
  get(loraName) {
    const key = loraShortName(loraName).toLowerCase();
    const db = loadTriggerDB();
    if (db[key] != null) return db[key];
    return loraShortName(loraName).replace(/\.(safetensors|ckpt|pt)$/i, "").replace(/_/g, " ");
  },
  set(loraName, triggerText) {
    const key = loraShortName(loraName).toLowerCase();
    const db = loadTriggerDB();
    db[key] = triggerText;
    saveTriggerDB(db);
  },
};
