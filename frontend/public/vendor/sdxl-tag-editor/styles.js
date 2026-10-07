// tag-editor/styles.js
//
// Injects one scoped stylesheet (idempotent). Everything is namespaced under
// `.ute-root` so it cannot leak into ComfyUI's own UI.
//
// Palette / spacing ported faithfully from the original widget
// (sdxl_tag_editor_ORIGINAL_SESSION_START.js, `#sdxl-te-style` block): muted
// dark-blue base (#1a1a22), dark-green POSITIVE group (#0e1f0e), dark-red
// NEGATIVE group (#150e0e), square 3px chips (not pills), monospace inputs,
// compact 9–11px type. Class names stay `.ute-*`; only the values are ported.

const STYLE_ID = "unified-tag-editor-styles";

const CSS = `
.ute-root {
  font: 12px/1.3 sans-serif;
  color: #ccc;
  background: #1a1a22;
  border-radius: 4px;
  box-sizing: border-box;
  width: 100%;
  height: 100%;
  overflow: auto;
  padding: 4px 5px;
  display: flex;
  flex-direction: column;
  gap: 2px;
  user-select: none;
  outline: none;
}
.ute-root * { box-sizing: border-box; }

/* ── tabs ── */
.ute-tabs {
  display: flex; gap: 0;
  padding: 0 2px 0;
  border-bottom: 2px solid #2a2a3a;
  margin-bottom: 1px;
}
.ute-tab {
  flex: 1; text-align: center;
  padding: 4px 14px; font-size: 11px; cursor: pointer;
  border: 1px solid transparent; border-bottom: 2px solid transparent;
  border-radius: 3px 3px 0 0; color: #4a4a6a; background: transparent;
  margin-bottom: -2px; font-weight: 600;
}
.ute-tab:hover { color: #aaa; background: #1d1d2c; }
.ute-tab.active {
  background: #1a1a22; border-color: #3a3a55; border-bottom-color: #1a1a22;
  color: #eee;
}

/* ── toolbar (bigger icon+label buttons, tight gaps) ── */
.ute-toolbar {
  display: flex; flex-wrap: wrap; gap: 3px; align-items: center;
  padding: 3px 1px; margin-bottom: 1px; border-bottom: 1px solid #222228;
}
.ute-btn {
  display: inline-flex; align-items: center; gap: 5px;
  background: #252535; border: 1px solid #3a3a4a; color: #ccc;
  font-size: 11px; font-weight: 500; padding: 6px 11px; border-radius: 4px;
  cursor: pointer; white-space: nowrap; line-height: 1;
}
.ute-btn:hover { background: #303050; color: #fff; border-color: #55f; }
.ute-btn:disabled { opacity: .4; cursor: default; }
.ute-btn:disabled:hover { background: #252535; border-color: #3a3a4a; }
.ute-btn.warn { border-color: #5a1a1a; color: #d88; }
.ute-btn.warn:hover { background: #400; border-color: #c33; color: #f88; }
/* Icon-only buttons: no label span, so drop the horizontal padding that was
   sized for icon+text (avoids reading as an oversized empty button). */
.ute-btn.icon-only { padding: 6px 9px; }
.ute-icon { flex-shrink: 0; }
.ute-btn-label { flex-shrink: 0; }

/* ── Positive / Negative group wrappers ── */
.ute-group { border-radius: 4px; padding: 3px 5px 4px; margin: 1px 0; }
.ute-group.pos { border: 1px solid #1a3a1a; background: #0e1f0e; }
.ute-group.neg { border: 1px solid #2a1a1a; background: #150e0e; }
.ute-group-head {
  font-size: 9px; font-weight: 600; letter-spacing: .08em;
  text-transform: uppercase; margin-bottom: 1px; padding: 0 2px;
}
.ute-group.pos > .ute-group-head { color: #3a8a3a; }
.ute-group.neg > .ute-group-head { color: #8a3a3a; }

/* ── collapsible section headers ── */
.ute-section { }
.ute-sec-head {
  display: flex; align-items: center; gap: 4px;
  padding: 2px 5px; cursor: pointer; user-select: none;
  font-size: 10px; color: #666; text-transform: uppercase; letter-spacing: .06em;
  border-left: 2px solid #2a2a3a; margin: 3px 0 0;
}
.ute-sec-head:hover { color: #aaa; border-left-color: #44f; }
.ute-group.pos .ute-sec-head { border-left-color: #2a3a2a; color: #4a6a4a; }
.ute-group.pos .ute-sec-head:hover { color: #88cc88; border-left-color: #336633; }
.ute-group.neg .ute-sec-head { border-left-color: #4a2a2a; color: #7a4a4a; }
.ute-group.neg .ute-sec-head:hover { color: #cc8888; border-left-color: #883333; }
.ute-sec-caret { font-size: 8px; min-width: 10px; }
.ute-sec-title { flex: 1; }
.ute-sec-hint {
  font-size: 9px; color: #333; text-transform: none; letter-spacing: 0;
  margin-left: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.ute-sec-count { font-size: 9px; color: #444; }

/* ── tag areas ── */
.ute-chips {
  display: flex; flex-wrap: wrap; gap: 3px; min-height: 24px;
  background: #111317; border: 1px solid #2a2a35; border-radius: 3px;
  padding: 3px; margin-bottom: 1px; position: relative; align-content: flex-start;
}
.ute-chips.dropzone-active,
.ute-chips.drag-over { border-color: #55f; background: #111428; }

/* ── chips (square, ported from original 3px radius) ── */
.ute-chip {
  display: inline-flex; align-items: center; gap: 3px; max-width: 100%;
  border-radius: 3px; padding: 2px 5px 2px 6px; font-size: 11px;
  white-space: nowrap; cursor: grab; border: 1px solid #3a3a4a;
  background: #252535; color: #ccc; user-select: none; position: relative;
  transition: background .08s, border-color .08s;
}
.ute-chip:hover { background: #303050; border-color: #55f; }

/* ── テキストチップ（文章） ──────────────────────────────────────────────
   タグ用チップは white-space:nowrap で 1 行に収める前提だが、文章を入れると
   途中で切れて読めない。文章とみなした要素だけ全幅・折り返し表示にする。
   並べ替え・選択・削除といったチップの操作性はそのまま保つ。 */
.ute-chip.prose {
  display: flex; align-items: flex-start; width: 100%; max-width: 100%;
  white-space: pre-wrap; word-break: break-word; line-height: 1.45;
  padding: 4px 6px; text-align: left; cursor: grab;
  background: #1e2430; border-color: #33405a;
}
.ute-chip.prose:hover { background: #263043; border-color: #55f; }
.ute-chip.prose .ute-chip-label {
  flex: 1; min-width: 0; white-space: pre-wrap; overflow: visible;
  text-overflow: clip; font-size: 11px;
}
.ute-chip.prose .ute-chip-x { align-self: flex-start; margin-top: 1px; }

.ute-chip.selected { background: #1e3a6a; border-color: #3af; box-shadow: 0 0 0 1px #3af6; }
.ute-chip.bypassed { opacity: .42; text-decoration: line-through; }
.ute-chip.lora { background: #14142a; border-color: #5555bb; }
.ute-chip.lora:hover { background: #1e1e3c; border-color: #7777ee; }
.ute-chip.dragging { opacity: .4; }
.ute-chip.drop-before { box-shadow: -3px 0 0 -1px #3af; }
.ute-chip.drop-after { box-shadow: 3px 0 0 -1px #3af; }
.ute-chip-label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ute-chip-w {
  font-size: 9px; background: #3a3a50; padding: 0 3px; border-radius: 2px;
  color: #9df; margin-left: 1px;
}
.ute-chip-x { font-size: 10px; color: #666; cursor: pointer; padding: 0 1px; line-height: 1; }
.ute-chip-x:hover { color: #f55; }

/* ── chip badges (AI import / double-paren) + duplicate highlight ── */
.ute-chip-badge {
  font-size: 8px; line-height: 1; padding: 1px 3px; border-radius: 2px;
  font-weight: 700; letter-spacing: .02em;
}
.ute-chip-badge.ai { background: #2a1a3a; color: #c9f; border: 1px solid #6a3a8a; }
.ute-chip-badge.dparen { background: #3a2a10; color: #fc9; border: 1px solid #8a6a20; }
.ute-chip.dup { border-color: #cc8; box-shadow: 0 0 0 1px #cc84; }
.ute-chip.dup .ute-chip-label { color: #ee8; }

/* ── alt-swap flash animation (Alt+Click cycle feedback) ── */
@keyframes ute-alt-swap {
  0% { background: #3a5a1e; border-color: #8ec54a; box-shadow: 0 0 0 2px #8ec54a; }
  100% { }
}
.ute-chip.alt-swap { animation: ute-alt-swap .45s ease-out; }
.ute-chip-edit {
  background: #12122a; border: 1px solid #55f; color: #fff; padding: 1px 4px;
  border-radius: 2px; font: 10px monospace; outline: none;
  min-width: 50px; max-width: 240px;
}

/* ── section divider (full-width rule, ported from original .sdxl-te-divider) ── */
.ute-divider {
  flex-basis: 100%; width: 100%; height: 6px; margin: 2px 0; padding: 2px 0;
  cursor: grab; box-sizing: content-box;
}
.ute-divider::before {
  content: ""; display: block; height: 2px; border-radius: 1px;
  background: #3a3a55; transition: background .12s;
}
.ute-divider:hover::before { background: #5a5aaa; }
.ute-divider.selected::before { background: #3af; box-shadow: 0 0 0 1px #3af6; }
.ute-divider.dragging { opacity: .4; }
.ute-divider.drop-before { box-shadow: inset 0 3px 0 -1px #3af; }
.ute-divider.drop-after { box-shadow: inset 0 -3px 0 -1px #3af; }

/* ── add-row inputs (monospace, ported) ── */
.ute-addrow { display: flex; gap: 3px; margin-top: 1px; margin-bottom: 2px; }
.ute-input, .ute-textarea {
  flex: 1; background: #111; border: 1px solid #333; color: #ccc;
  font: 10px monospace; padding: 2px 5px; border-radius: 2px; outline: none;
}
.ute-input:focus, .ute-textarea:focus { border-color: #55f; }
.ute-input::placeholder, .ute-textarea::placeholder { color: #333; }
.ute-textarea { width: 100%; resize: vertical; min-height: 40px; }

/* ── previews (selectable text — overrides .ute-root's user-select:none so the
   final prompt can be drag-selected and copied like any normal text) ── */
.ute-preview {
  background: #0c131d; border: 1px solid #1e3555; border-radius: 3px;
  padding: 3px 6px; font: 10px monospace; color: #7ac;
  word-break: break-all; white-space: pre-wrap; overflow-y: auto;
  /* 高さは JS (_makeResizable) が min-height で与える。
     以前は max-height:48px で頭打ちにしていたが、長いプロンプトが
     常に極小の窓に押し込まれて読めなかった。上限は設けず、
     ユーザーがドラッグで決めた下限を基準に中身の分だけ伸ばす。 */
  min-height: 20px;
  user-select: text; -webkit-user-select: text; cursor: text;
}
.ute-group.neg .ute-preview { background: #1d0c0c; border-color: #3a1515; color: #c88; }

/* ── clip / rows ── */
.ute-row {
  display: flex; align-items: center; gap: 6px; margin-top: 2px;
  padding: 2px 5px; background: #111317; border: 1px solid #222230;
  border-radius: 3px; font-size: 10px; color: #666;
}
.ute-num, .ute-select {
  width: 54px; background: #111; border: 1px solid #333; color: #bbb;
  font-size: 10px; padding: 2px 5px; border-radius: 2px; outline: none;
}
.ute-num:focus, .ute-select:focus { border-color: #55f; }
.ute-select { width: auto; }

/* ── tagger ── */
.ute-dropzone {
  border: 2px dashed #333; border-radius: 4px; min-height: 70px; padding: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 11px; color: #555; cursor: pointer; background: #111317;
  text-align: center; transition: border-color .15s, background .15s;
}
.ute-dropzone:hover, .ute-dropzone.over { border-color: #55f; background: #13133a; color: #aaf; }
.ute-dropzone img { max-width: 100%; max-height: 160px; border-radius: 3px; }
.ute-tagger-tags { display: flex; flex-wrap: wrap; gap: 4px; padding: 4px 0; }
.ute-tt {
  display: inline-flex; gap: 4px; align-items: center; padding: 2px 6px;
  border-radius: 3px; background: #181a2a; border: 1px solid #2a2a4a;
  cursor: pointer; font-size: 10px;
}
.ute-tt:hover { background: #202040; border-color: #44f; }
.ute-tt .c { color: #66c; font-size: 9px; }
.ute-status { font-size: 10px; color: #555; padding: 2px 0; min-height: 16px; }

/* ── character cards ── */
.ute-char-card { border: 1px solid #2a2a3a; border-radius: 4px; margin: 5px 0; overflow: hidden; }
.ute-char-head { display: flex; gap: 4px; align-items: center; padding: 4px 6px; background: #1e1e2e; }
.ute-char-name {
  flex: 1; min-width: 0; background: #111; border: 1px solid #333; color: #ccc;
  font-size: 11px; padding: 2px 6px; border-radius: 2px; outline: none;
}
.ute-char-name:focus { border-color: #55f; }

/* ── chip lock + color labels (ported from original) ── */
.ute-chip-lock { font-size: 9px; opacity: .8; margin-right: 1px; }
.ute-chip.locked { cursor: default; }
.ute-chip.color-character { background:#14142a; border-color:#5555bb; }
.ute-chip.color-character:hover { background:#1e1e3c; border-color:#7777ee; }
.ute-chip.color-style     { background:#0a1a0e; border-color:#336644; }
.ute-chip.color-style:hover { background:#122218; border-color:#449955; }
.ute-chip.color-outfit    { background:#1a1000; border-color:#7a4400; }
.ute-chip.color-outfit:hover { background:#221500; border-color:#9a5500; }
.ute-chip.color-important { background:#191400; border-color:#887700; }
.ute-chip.color-important:hover { background:#221c00; border-color:#aa9900; }
.ute-chip.color-warning   { background:#1a0808; border-color:#882222; }
.ute-chip.color-warning:hover { background:#220a0a; border-color:#aa3333; }
`;

// ── Global (document.body) chrome for the floating context menu + weight
// popover. NOT namespaced under .ute-root because these mount on document.body,
// but the class names are still ute-* so they can't collide with ComfyUI.
const GLOBAL_CSS = `
.ute-ctxmenu {
  position: fixed; background: #1a1a2a; border: 1px solid #3a3a4a;
  border-radius: 4px; z-index: 10000; overflow: hidden; min-width: 148px;
  box-shadow: 0 3px 12px #0009; font: 11px/1.4 sans-serif; padding: 2px 0;
}
.ute-ctxmenu div { padding: 5px 12px; cursor: pointer; color: #bbb; }
.ute-ctxmenu div:hover { background: #2a2a4a; color: #fff; }
.ute-ctxmenu div.disabled { color: #555; cursor: default; }
.ute-ctxmenu div.disabled:hover { background: transparent; color: #555; }
.ute-ctxmenu div.warn { color: #c66; }
.ute-ctxmenu div.warn:hover { background: #400; color: #f88; }
.ute-ctxmenu hr { border: none; border-top: 1px solid #2a2a3a; margin: 2px 0; }
.ute-ctxmenu .ute-ctx-hdr {
  font-size: 9px; color: #555; padding: 3px 12px 1px; cursor: default;
  text-transform: uppercase; letter-spacing: .05em;
}
.ute-ctxmenu .ute-ctx-hdr:hover { background: transparent; color: #555; }
.ute-ctxmenu .ute-color-item { display: flex; align-items: center; gap: 7px; }
.ute-ctxmenu .ute-color-item.active { font-weight: 600; }
.ute-popover {
  position: fixed; background: #1a1a2a; border: 1px solid #444; border-radius: 4px;
  padding: 6px 10px; z-index: 10001; display: flex; gap: 5px; align-items: center;
  font: 11px/1.4 sans-serif; color: #ccc; box-shadow: 0 3px 12px #0009;
}
.ute-popover input {
  width: 62px; background: #111; border: 1px solid #444; color: #ccc;
  padding: 2px 5px; border-radius: 2px; font-size: 11px; outline: none;
}
.ute-popover .ute-btn {
  background: #252535; border: 1px solid #3a3a4a; color: #aaa;
  font-size: 10px; padding: 2px 8px; border-radius: 3px; cursor: pointer;
}
.ute-popover .ute-btn:hover { background: #303050; color: #fff; }

/* ── LoRA picker (mounts on document.body) ── */
.ute-lora-picker {
  position: fixed; z-index: 10001; min-width: 260px; max-width: 360px;
  background: #1a1a2a; border: 1px solid #3a3a4a; border-radius: 4px;
  padding: 6px; display: flex; flex-direction: column; gap: 4px;
  font: 11px/1.4 sans-serif; color: #ccc; box-shadow: 0 3px 12px #0009;
}
.ute-lora-picker-hdr {
  font-size: 9px; color: #888; text-transform: uppercase; letter-spacing: .05em;
  padding: 2px 2px 4px;
}
.ute-lora-picker-empty { font-size: 10px; color: #666; padding: 0 2px 4px; }
.ute-lora-row {
  display: flex; align-items: center; gap: 5px; padding: 3px 2px;
  border-radius: 3px;
}
.ute-lora-row:hover { background: #22223a; }
.ute-lora-row-name {
  flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis;
  white-space: nowrap; color: #ddd;
}
.ute-lora-row-strength {
  font-size: 9px; background: #3a3a50; padding: 0 3px; border-radius: 2px; color: #9df;
}
.ute-lora-row-trigger {
  width: 90px; background: #111; border: 1px solid #333; color: #ccc;
  font: 10px monospace; padding: 2px 4px; border-radius: 2px; outline: none;
}
.ute-lora-row-trigger:focus { border-color: #55f; }

/* ── autocomplete dropdown (mounts on document.body) ── */
.ute-autocomplete {
  position: fixed; z-index: 10001;
  background: #1a1a2e; border: 1px solid #3a3a5a; border-radius: 4px;
  overflow: hidden; overflow-y: auto; max-height: 160px;
  font-size: 11px; box-shadow: 0 3px 10px #0009;
}
.ute-ac-item {
  display: flex; justify-content: space-between; align-items: center;
  padding: 4px 10px; cursor: pointer; color: #aaa; white-space: nowrap;
}
.ute-ac-item.ac-sel, .ute-ac-item:hover { background: #2a2a4a; color: #fff; }
.ute-ac-usage { color: #555; font-size: 9px; margin-left: 8px; }
.ute-ac-hdr {
  padding: 2px 10px; font-size: 9px; color: #445; letter-spacing: .05em;
  text-transform: uppercase; background: #12121e; cursor: default;
}
`;

export function injectStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const el = document.createElement("style");
  el.id = STYLE_ID;
  el.textContent = CSS + GLOBAL_CSS + `
.ute-root {overflow:hidden;min-height:0}
.ute-tabs,.ute-toolbar {flex-shrink:0}
.ute-editor-body {display:flex;flex-direction:column;gap:5px;flex:1;min-height:0;overflow:hidden}
.ute-editor-surface {flex:1;min-height:70px;overflow:auto;overscroll-behavior:contain}
.ute-editor-surface:has(.ute-preview) .ute-section {height:100%;display:flex;flex-direction:column}
.ute-editor-surface .ute-preview {width:100%;flex:1;min-height:120px!important;resize:none!important;font-size:12px;line-height:1.6;color:#c3ddf1}
.ute-input {min-height:30px;font-size:12px}
.ute-input::placeholder {color:#9da8ba}
.ute-viewbar {display:flex;gap:4px;flex-wrap:wrap;flex-shrink:0}
.ute-viewbar .active {border-color:#83ceff;background:#25445d;color:#fff}
.ute-pan-handle {margin-left:auto;cursor:grab;touch-action:none}
.ute-panning,.ute-panning * {cursor:grabbing!important}
.ute-clip-settings {flex-shrink:0;font-size:11px}
.ute-clip-settings summary {padding:5px;cursor:pointer}
/* Reserve a small delete lane in every state: showing the control must never
   cover the label or change chip width. Selection/bypass add no badge. */
.ute-chip {min-height:0;min-width:0;gap:3px;line-height:1.3;flex-shrink:0;padding-right:20px}
.ute-chip.locked {padding-right:5px}
.ute-chip-label {min-width:0;white-space:nowrap}
.ute-chip-x {position:absolute;right:0;top:0;bottom:0;display:inline-flex;align-items:center;justify-content:center;width:18px;min-width:0;height:auto;background:inherit;border:0;border-radius:2px;color:#aaa;font-size:13px;line-height:1;padding:0;opacity:0;pointer-events:none}
.ute-chip:hover .ute-chip-x,.ute-chip.selected .ute-chip-x,.ute-chip:focus-within .ute-chip-x {opacity:1;pointer-events:auto}
.ute-chip-x:hover {color:#f55}
.ute-tab {min-height:32px;color:#b4bcc9}
.ute-root :focus-visible {outline:2px solid #92d5ff;outline-offset:1px}
.ute-tree-menu {position:fixed;z-index:10020;min-width:210px;max-width:min(340px,calc(100vw - 16px));max-height:calc(100vh - 16px);overflow-y:auto;padding:5px;background:#202630;border:1px solid #78879e;border-radius:6px;box-shadow:0 5px 22px #0009;color:#e4eaf5;font:13px/1.4 sans-serif;box-sizing:border-box}
.ute-tree-menu button {display:block;width:100%;text-align:left;min-height:32px;padding:6px 10px;background:transparent;border:0;border-radius:3px;color:inherit;font:inherit;cursor:pointer}
.ute-tree-menu button:hover,.ute-tree-menu button:focus {background:#35475e;outline:2px solid #8ed3ff;outline-offset:-2px}
.ute-tree-menu button:disabled {color:#9ca8ba;opacity:1;cursor:default}
.ute-tree-menu button.warn {color:#ffb4b4}
.ute-tree-menu strong {display:block;overflow-wrap:anywhere;padding:5px 8px;max-width:310px}
.ute-tree-menu hr {border:0;border-top:1px solid #465265;margin:4px 0}
.ute-clip-panel {position:fixed;z-index:10003;width:430px;max-width:calc(100vw - 16px);max-height:calc(100vh - 16px);overflow:auto;box-sizing:border-box;padding:12px;background:#1b2029;border:1px solid #71839a;border-radius:6px;color:#e1e6ef;font:12px sans-serif;box-shadow:0 8px 24px #0008}
.ute-clip-panel p {margin:8px 0;line-height:1.45}
.ute-clip-panel-header {display:flex;align-items:center;gap:6px}
.ute-clip-panel-header strong {flex:1}
.ute-clip-setting {display:flex;align-items:center;gap:8px;border-top:1px solid #3a4351;padding:9px 0;flex-wrap:wrap}
.ute-clip-setting-label {display:flex;align-items:center;gap:8px;flex:1;min-width:190px}
.ute-clip-setting-label span {flex:1;overflow-wrap:anywhere}
.ute-clip-setting input {width:76px;min-height:30px;box-sizing:border-box;color:#eee;background:#12171f;border:1px solid #71839a;border-radius:4px;padding:4px}
.ute-clip-panel button {min-height:30px}
.ute-clip-panel :focus-visible {outline:2px solid #8ed3ff;outline-offset:2px}
.ute-clip-status {color:#a7d9fa}
`;
  document.head.appendChild(el);
}
