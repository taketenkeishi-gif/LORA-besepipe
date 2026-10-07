// tag-editor/state.js
//
// The editor's in-memory state model plus (de)serialization and undo history.
//
// SERIALIZE SHAPE — verified against the Python backend:
//   nodes/tag_editor_node.py reads state.get("prefix"/"suffix"/"blocklist"/
//   "mainTags"/"negative"/"clipSkip"); core/canonical_tags.coerce_raws accepts
//   EITHER a comma-string OR a list of {raw, bypassed?} dicts and drops any item
//   with bypassed === true. So every section is emitted as [{raw, ...flags}] and
//   clipSkip as a number. Extra keys (characters, stash, taggerSettings, etc.)
//   are ignored by the backend but round-tripped for the UI.

import { makeTag, makeDivider, splitTags } from "./parse.js";

export const SECTIONS = ["prefix", "mainTags", "suffix", "blocklist", "negative"];

export function defaultState() {
  return {
    prefix: [],
    mainTags: [],
    suffix: [],
    blocklist: [],
    negative: [],
    stash: [],
    characters: [],
    // 1 = 最終層（skip なし・標準 CLIPTextEncode と同じ）。
    // 旧既定値は -2 だったが、負値は ComfyUI の clip_layer 規約であって
    // UI 表記（A1111 慣習の正値）とは別物。新規ノードが意図せず 1 層
    // スキップした状態で始まらないよう 1 を既定にする。
    clipSkip: 1,
    activeTab: "tags",
    editorView: { polarity: "positive", mode: "edit", prefixOpen: false, scroll: {}, outputScroll: {}, selected: {} },
    // 各ボックスの「最低の高さ」(px)。ユーザーがドラッグで決めた値を保持する。
    // height ではなく min-height として使うので、中身が増えればここを超えて
    // 自然に伸びる（縮みすぎて使いにくいのを防ぐための下限指定）。
    // キー例: "chips:mainTags" / "preview:pos" / "preview:neg"
    boxHeights: {},
    taggerSettings: {
      model: "wd-vit-large-tagger-v3",
      threshold: 0.35,
      charThreshold: 0.85,
      excludeTags: "",
      replaceUnderscore: true,
    },
  };
}

const serTag = (t) => (t.type === "divider" ? { type: "divider" } : {
  raw: t.raw,
  ...(t.locked ? { locked: true } : {}),
  ...(t.color ? { color: t.color } : {}),
  ...(t.ai_imported ? { ai_imported: true } : {}),
  ...(t.bypassed ? { bypassed: true } : {}),
});

/** state object -> JSON string stored in the hidden editor_state widget. */
export function serialize(state) {
  return JSON.stringify({
    prefix: state.prefix.map(serTag),
    mainTags: state.mainTags.map(serTag),
    suffix: state.suffix.map(serTag),
    blocklist: state.blocklist.map(serTag),
    negative: state.negative.map(serTag),
    stash: state.stash.map(serTag),
    characters: state.characters.map((ch) => ({
      ...ch,
      id: ch.id,
      name: ch.name,
      tags: ch.tags.map(serTag),
    })),
    clipSkip: state.clipSkip,
    activeTab: "tags",
    editorView: state.editorView,
    boxHeights: state.boxHeights,
    taggerSettings: state.taggerSettings,
  });
}

/** Accept an array of {raw,...} (current) OR a comma-string (legacy). */
function toArr(v) {
  if (Array.isArray(v)) {
    return v
      .map((e) =>
        e && e.type === "divider" ? makeDivider() : makeTag(e && e.raw ? e.raw : "", {
          ...(e && e.locked ? { locked: true } : {}),
          ...(e && e.color ? { color: e.color } : {}),
          ...(e && e.ai_imported ? { ai_imported: true } : {}),
          ...(e && e.bypassed ? { bypassed: true } : {}),
        })
      )
      .filter(Boolean);
  }
  if (typeof v === "string" && v.trim()) return splitTags(v);
  return [];
}

/** JSON string -> state object (tolerant of malformed / partial / legacy input). */
export function deserialize(json) {
  const s = defaultState();
  if (!json || json === "{}") return s;
  let d;
  try {
    d = typeof json === "string" ? JSON.parse(json) : json;
  } catch {
    return s;
  }
  if (!d || typeof d !== "object") return s;

  s.prefix = toArr(d.prefix);
  s.mainTags = toArr(d.mainTags);
  s.suffix = toArr(d.suffix);
  s.blocklist = toArr(d.blocklist);
  s.negative = toArr(d.negative);
  s.stash = toArr(d.stash);
  s.characters = Array.isArray(d.characters)
    ? d.characters.map((ch) => ({
        ...ch,
        id: ch.id || cryptoId(),
        name: String(ch.name ?? ""),
        tags: toArr(ch.tags),
      }))
    : [];
  if (typeof d.clipSkip === "number") s.clipSkip = d.clipSkip;
  s.activeTab = "tags";
  if (d.editorView && typeof d.editorView === "object") {
    const v = d.editorView;
    s.editorView = {
      polarity: d.activeTab === "characters" ? "positive" : v.polarity === "negative" ? "negative" : "positive",
      mode: v.mode === "output" ? "output" : "edit", prefixOpen: !!v.prefixOpen,
      scroll: {}, outputScroll: {}, selected: {},
    };
    for (const side of ["positive", "negative"]) {
      const offset = Number(v.scroll?.[side]);
      if (Number.isFinite(offset)) s.editorView.scroll[side] = Math.max(0, Math.min(100000, offset));
      const outputOffset = Number(v.outputScroll?.[side]);
      if (Number.isFinite(outputOffset)) s.editorView.outputScroll[side] = Math.max(0, Math.min(100000, outputOffset));
      s.editorView.selected[side] = (Array.isArray(v.selected?.[side]) ? v.selected[side] : []).filter((p) =>
        p && ["prefix", "mainTags", "negative"].includes(p.section) && Number.isInteger(p.index) && p.index >= 0);
    }
  }
  if (d.boxHeights && typeof d.boxHeights === "object") {
    // 数値かつ妥当な範囲のものだけ拾う（壊れた値でノードが巨大化しないよう上限を設ける）
    for (const [k, v] of Object.entries(d.boxHeights)) {
      const px = Number(v);
      if (Number.isFinite(px) && px >= 16 && px <= 1200) s.boxHeights[k] = Math.round(px);
    }
  }
  if (d.taggerSettings && typeof d.taggerSettings === "object") {
    Object.assign(s.taggerSettings, d.taggerSettings);
  }
  return s;
}

let _cid = 0;
export function cryptoId() {
  return "c" + Date.now().toString(36) + (++_cid).toString(36);
}

// ── Undo / redo history ──────────────────────────────────────────────────────
export class History {
  constructor(limit = 100) {
    this._limit = limit;
    this._undo = [];
    this._redo = [];
  }
  push(snapshotJson) {
    this._undo.push(snapshotJson);
    if (this._undo.length > this._limit) this._undo.shift();
    this._redo.length = 0;
  }
  canUndo() {
    return this._undo.length > 1;
  }
  canRedo() {
    return this._redo.length > 0;
  }
  /** Move current -> redo, return previous snapshot (or null). */
  undo(currentJson) {
    if (this._undo.length < 2) return null;
    const cur = this._undo.pop();
    this._redo.push(cur ?? currentJson);
    return this._undo[this._undo.length - 1];
  }
  redo() {
    if (!this._redo.length) return null;
    const next = this._redo.pop();
    this._undo.push(next);
    return next;
  }
}
