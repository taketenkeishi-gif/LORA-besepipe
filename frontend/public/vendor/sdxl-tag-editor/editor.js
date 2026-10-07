// tag-editor/editor.js
//
// The editor orchestrator: owns state + history + selection, renders the two
// tabs (Tags / Characters), the shared toolbar, sections, chips and previews,
// and wires drag-and-drop + the tagger panel. Public surface used by the entry
// module: getJSON(), setJSON(json), setTaggerTags(text), _widgetHeight,
// _onDocPaste (a document paste listener the entry tears down on node removal).

import { injectStyles } from "./styles.js";
import { h, stop } from "./dom.js";
import { defaultState, serialize, deserialize, History, SECTIONS } from "./state.js";
import {
  makeTag, makeDivider, splitTags, loraTriggerTags, withoutInertLoraTags, chipLabel, chipWeight, coreKey, adjustWeight, toggleEmphasis,
  setWeight, tagWeight,
} from "./parse.js";
import { FloatingMenu, showWeightPopover, closeWeightPopover, COLOR_MAP } from "./context_menu.js";
import { removeSelectedBypassed } from "./remove-selected-bypassed.mjs";
import { nextSwapText, SwapDB } from "./swap.js";
import { attachAutocomplete, bumpHistory } from "./autocomplete.js";
import { DragController } from "./drag.js";
import { getIcon } from "./icons.js";
import { copyPlainTextToOsClipboard } from "./os-clipboard.js";
import { readChipClipboard, writeChipClipboard, CHIP_CLIPBOARD_MIME, parseChipClipboard } from "./chip-clipboard.mjs?v=20260915-native-clipboard";
import { pasteIndex } from "./paste-position.mjs";
import { scanLoraNodes } from "./lora-scan.js";
import { showLoraPicker, closeLoraPicker } from "./lora-picker.js";
import { showLoraClipControl } from "./lora-clip-control.mjs";
import { TaggerPanel } from "./tagger.js";
import * as chars from "./characters.js";

// WD14 TAGGER panel is hidden and frozen: not constructed (no network calls,
// no event wiring), not rendered, not wired to paste. Flip to re-enable.
const TAGGER_ENABLED = false;

// Labels + descriptive hint texts ported verbatim from the original widget
// (sdxl_tag_editor_ORIGINAL_SESSION_START.js `_mkSecHdr` calls).
const SECTION_META = {
  prefix: { label: "固定タグ", multi: true, hint: "先頭に加える画風・品質など" },
  mainTags: { label: "タグ", multi: true, hint: "クリックで選択・ダブルクリックで編集・ドラッグで順序変更" },
  negative: { label: "除外したい内容", multi: true, hint: "Negative conditioningへ" },
};

// Route-to-section menu labels (order matches the original context menu).
// SUFFIX and BLOCKLIST are intentionally omitted: both sections are hidden
// from the UI, so neither is offered as a move-to target (any existing data
// in them, from older saved graphs, still serializes/applies — just isn't
// editable here anymore).
const SECTION_LABELS = [
  ["prefix", "固定タグ"],
  ["mainTags", "描きたい内容のタグ"],
  ["negative", "除外したい内容のタグ"],
];


/**
 * その要素が「タグ」ではなく「文章」かを判定する。
 *
 * 文章を処理できるようになった結果、1 個のチップに長文が入って途中で
 * 切れるようになった。見た目だけ切り替えるための判定なので、
 * 誤判定しても機能は壊れない（表示が全幅になるだけ）。
 *
 * 判定: LoRA/区切りは対象外。改行を含む、文末ピリオドで区切られた文が
 * ある、または十分に長い（語数・文字数）ものを文章とみなす。
 */
function isProseTag(tag) {
  if (!tag || tag.type === "divider" || tag.type === "lora") return false;
  const raw = String(tag.raw ?? "");
  if (!raw) return false;
  if (raw.includes("\n")) return true;
  if (/[.!?]\s+\S/.test(raw)) return true;       // 「. 」で続く = 複数文
  if (raw.length >= 60) return true;              // 長すぎるものは事実上文章
  const words = raw.trim().split(/\s+/).length;
  return words >= 10;                             // 語数が多いものも文章扱い
}

export class TagEditor {
  /**
   * @param {HTMLElement} root
   * @param {(json:string)=>void} onChange
   * @param {() => any[]} [getGraphNodes]  returns the current ComfyUI workflow's
   *   LiteGraph node array, for the LoRA-picker's "scan the graph" feature.
   *   Injected by unified_tag_editor.js so this module stays host-agnostic
   *   (no direct `app` import here); omit/return [] where graph access isn't
   *   available and the picker just offers manual entry.
   */
  constructor(root, onChange, getGraphNodes, getHostNode) {
    injectStyles();
    this._root = root;
    this._onChange = onChange || (() => {});
    this._getGraphNodes = getGraphNodes || (() => []);
    this._getHostNode = getHostNode || (() => null);
    this._state = defaultState();
    this._history = new History();
    this._selection = new Set();
    this._anchor = null; // { section, index }
    this._collapsed = { prefix: true };
    this._taggerReadout = "";
    this._widgetHeight = 640;
    this._swapAnimId = null; // chip id to flash after an Alt+Click swap

    this._menu = new FloatingMenu();

    root.classList.add("ute-root");
    root.tabIndex = 0;
    this._buildShell();
    this._drag = new DragController((payload, dst, idx) => this._moveTags(payload, dst, idx));
    // Frozen: not constructed at all when disabled, so no fetch()/status poll
    // or event listeners ever run for it.
    this._tagger = TAGGER_ENABLED
      ? new TaggerPanel({
          getSettings: () => this._state.taggerSettings,
          setSettings: (s) => { this._state.taggerSettings = s; this.commit(); },
          onImport: (names) => this._importToMain(names),
        })
      : null;

    this._history.push(serialize(this._state));
    this._installKeyboard();
    this._installPaste();
    this.render();
  }

  // ── public API ──────────────────────────────────────────────────────────────
  getJSON() {
    return serialize(this._state);
  }

  setJSON(json) {
    this._state = deserialize(json);
    this._collapsed.prefix = !this._state.editorView.prefixOpen;
    let removedLegacyLoraTag = false;
    for (const section of ["prefix", "mainTags", "suffix"]) {
      const filtered = withoutInertLoraTags(this._state[section]);
      removedLegacyLoraTag ||= filtered.length !== this._state[section].length;
      this._state[section] = filtered;
    }
    if (removedLegacyLoraTag) this._onChange(serialize(this._state));
    this._selection.clear();
    this._restoreViewSelection();
    this._history = new History();
    this._history.push(serialize(this._state));
    this.render();
  }

  setTaggerTags(text) {
    this._taggerReadout = typeof text === "string" ? text : "";
    if (this._state.activeTab === "tags") this.render();
  }

  /**
   * 生成後にバックエンドが返した「実際に適用された CLIP skip」を UI へ反映する。
   * 入力欄の値と実処理が食い違って見えないようにするため。
   * _clipRow() が未描画のタイミングでも落ちないよう保持しておき、
   * 次に描画されたときに反映する。
   */
  setClipSkipReport(rep) {
    if (!rep || typeof rep !== "object") return;
    this._clipSkipReport = rep;
    if (typeof this._applyClipSkipReport === "function") {
      this._applyClipSkipReport(rep);
    }
  }

  // ── state mutation plumbing ──────────────────────────────────────────────────
  /** Persist current state: push history, notify host (updates hidden widget). */
  commit() {
    this._saveView?.(false);
    const snap = serialize(this._state);
    this._history.push(snap);
    this._onChange(snap);
  }

  _sectionArray(section) {
    return this._state[section];
  }

  _findTag(id) {
    for (const sec of SECTIONS) {
      const arr = this._state[sec];
      const idx = arr.findIndex((t) => t.id === id);
      if (idx >= 0) return { sec, idx, tag: arr[idx] };
    }
    for (const ch of this._state.characters) {
      const idx = ch.tags.findIndex((t) => t.id === id);
      if (idx >= 0) return { char: ch, idx, tag: ch.tags[idx] };
    }
    return null;
  }

  /**
   * Move one or more tags (by {section,id}, in doc order) into dstSection at
   * dstIndex, preserving their relative order. Used for both single-chip drag
   * (payload.length === 1) and multi-selection drag (the whole selection moves
   * together instead of collapsing to the one chip the gesture started on).
   */
  _moveTags(payload, dstSection, dstIndex) {
    const dstArr = this._state[dstSection];
    if (!dstArr || !payload || !payload.length) return;

    const ids = new Set(payload.map((p) => p.id));
    const dstArrBefore = dstArr.slice();

    // How many of the moving tags were already in dst, before dstIndex — the
    // dstIndex the drop handler computed is against the pre-removal DOM, so we
    // must shift it back by however many of those will vanish before it.
    let adjustment = 0;
    dstArrBefore.forEach((t, i) => { if (ids.has(t.id) && i < dstIndex) adjustment++; });

    // Collect the actual tag objects, in doc order, before mutating anything.
    const tags = [];
    for (const { section: sec, id } of payload) {
      const arr = this._state[sec];
      const t = arr && arr.find((x) => x.id === id);
      if (t) tags.push(t);
    }
    if (!tags.length) return;

    for (const sec of SECTIONS) this._state[sec] = this._state[sec].filter((t) => !ids.has(t.id));
    for (const ch of this._state.characters) ch.tags = ch.tags.filter((t) => !ids.has(t.id));

    let idx = Math.max(0, Math.min(dstIndex - adjustment, this._state[dstSection].length));
    this._state[dstSection].splice(idx, 0, ...tags);
    this.commit();
    this.render();
  }

  _addTags(section, text) {
    const tags = splitTags(text);
    if (!tags.length) return;
    this._state[section].push(...tags);
    const area = section === "negative" ? "negative" : "positive";
    for (const t of tags) bumpHistory(t.raw, area);
    this.commit();
    this.render();
  }

  _importToMain(names) {
    for (const n of names) {
      const t = makeTag(n, { ai_imported: true });
      if (t) this._state.mainTags.push(t);
    }
    this.commit();
    if (this._state.activeTab === "tags") this.render();
  }

  _deleteSelection() {
    if (!this._selection.size) return;
    for (const sec of SECTIONS) {
      this._state[sec] = this._state[sec].filter((t) => !this._selection.has(t.id));
    }
    for (const ch of this._state.characters) {
      ch.tags = ch.tags.filter((t) => !this._selection.has(t.id));
    }
    this._selection.clear();
    this.commit();
    this.render();
  }

  _removeSelectedBypassed() {
    if (!this._selection.size) return;
    const filter = (tags) => {
      const result = removeSelectedBypassed(tags, this._selection);
      for (const id of result.removedIds) this._selection.delete(id);
      return result;
    };
    let removed = 0;
    for (const sec of SECTIONS) {
      const result = filter(this._state[sec]);
      this._state[sec] = result.remaining;
      removed += result.removedIds.length;
    }
    for (const ch of this._state.characters) {
      const result = filter(ch.tags);
      ch.tags = result.remaining;
      removed += result.removedIds.length;
    }
    if (!removed) return;
    this.commit();
    this.render();
  }

  _weightSelection(delta) {
    if (!this._selection.size) return;
    this._transformSelected((t) => adjustWeight(t, delta));
  }

  _emphasisSelection() {
    if (!this._selection.size) return;
    this._transformSelected((t) => toggleEmphasis(t));
  }

  _transformSelected(fn) {
    const apply = (arr) => arr.map((t) => (this._selection.has(t.id) ? fn(t) : t));
    for (const sec of SECTIONS) this._state[sec] = apply(this._state[sec]);
    for (const ch of this._state.characters) ch.tags = apply(ch.tags);
    this.commit();
    this.render();
  }

  _dedupe() {
    for (const sec of SECTIONS) {
      const seen = new Set();
      this._state[sec] = this._state[sec].filter((t) => {
        if (t.type === "divider") return true; // dividers are never dedup targets
        const k = coreKey(t);
        if (seen.has(k)) return false;
        seen.add(k);
        return true;
      });
    }
    this.commit();
    this.render();
  }

  /** Set of coreKeys that appear more than once in an array (for dup highlight). */
  _dupeKeys(arr) {
    const seen = new Set();
    const dupes = new Set();
    for (const t of arr) {
      if (t.type === "divider") continue;
      const k = coreKey(t);
      if (seen.has(k)) dupes.add(k);
      else seen.add(k);
    }
    return dupes;
  }

  /** Toggle bypass on the whole selection (all-on if any active, else all-off). */
  _bypassSelection() {
    if (!this._selection.size) return;
    const sel = this._selectedTagsInOrder();
    const anyActive = sel.some((t) => !t.bypassed);
    this._transformSelected((t) => ({ ...t, bypassed: anyActive }));
  }

  /** Convert underscores to spaces in the selected tags (danbooru → natural). */
  _convertUnderscoreSelected() {
    if (!this._selection.size) return;
    this._transformSelected((t) => {
      if (t.type === "lora" || t.type === "divider" || !t.raw.includes("_")) return t;
      const next = makeTag(t.raw.replace(/_/g, " "), tagFlags(t));
      if (next) next.id = t.id;
      return next || t;
    });
  }

  _undo() {
    const snap = this._history.undo(serialize(this._state));
    if (snap == null) return;
    this._state = deserialize(snap);
    this._selection.clear();
    this._onChange(snap);
    this.render();
  }

  _redo() {
    const snap = this._history.redo();
    if (snap == null) return;
    this._state = deserialize(snap);
    this._selection.clear();
    this._onChange(snap);
    this.render();
  }

  // ── selection ────────────────────────────────────────────────────────────────
  _clickSelect(e, section, index, id) {
    if (e.metaKey || e.ctrlKey) {
      if (this._selection.has(id)) this._selection.delete(id);
      else this._selection.add(id);
      this._anchor = { section, index };
    } else if (e.shiftKey && this._anchor && this._anchor.section === section) {
      const [a, b] = [this._anchor.index, index].sort((x, y) => x - y);
      const arr = this._state[section];
      for (let i = a; i <= b; i++) if (arr[i]) this._selection.add(arr[i].id);
    } else {
      this._selection.clear();
      this._selection.add(id);
      this._anchor = { section, index };
    }
    this._focusSection = section;
    // Keep focus inside the widget so Delete/weight/copy work right after a
    // click (root has tabIndex 0 and survives re-renders, unlike chips).
    try { this._root.focus({ preventScroll: true }); } catch { this._root.focus?.(); }
    this._saveView();
    for (const chip of this._root.querySelectorAll("[data-ute-id]")) {
      const selected = this._selection.has(Number(chip.dataset.uteId));
      chip.classList.toggle("selected", selected);
    }
    this._renderToolbar();
  }

  _selectAllInFocus() {
    const sec = this._focusSection;
    if (!sec || !this._state[sec]) return;
    for (const t of this._state[sec]) this._selection.add(t.id);
    this._saveView();
    this.render();
  }

  // ── rendering ────────────────────────────────────────────────────────────────
  _buildShell() {
    this._tabsEl = h("div", { class: "ute-tabs" });
    this._toolbarEl = h("div", { class: "ute-toolbar" });
    this._bodyEl = h("div", { class: "ute-editor-body" });
    this._root.append(this._tabsEl, this._toolbarEl, this._bodyEl);
  }

  render() {
    // If a chip is mid-edit, ANY other action that reaches render() — Undo/
    // Redo, a context-menu action on a different chip, Ctrl+A, etc. — would
    // otherwise silently discard the in-progress edit (render() unconditionally
    // rebuilds every chip from `_state`, and the live <input>'s uncommitted
    // value is never read back). Route through the edit's own finish() first
    // so the interruption commits the live input (Escape explicitly cancels)
    // instead of a silent data loss with
    // no defined outcome. finish() calls render() itself once _activeChipEdit
    // is cleared, so just hand off and return — this call doesn't rebuild.
    if (this._activeChipEdit) {
      const finishActive = this._activeChipEdit;
      this._activeChipEdit = null;
      finishActive(true);
      return;
    }
    this._menu.close();
    closeWeightPopover();
    closeLoraPicker();
    this._renderTabs();
    this._renderToolbar();
    this._bodyEl.textContent = "";
    this._renderTagsTab();
  }

  _saveView(notify = true) {
    const view = this._state.editorView;
    if (!view) return;
    view.selected[view.polarity] = [...this._selection].map((id) => {
      const loc = this._findTag(id);
      return loc?.sec ? { section: loc.sec, index: loc.idx } : null;
    }).filter(Boolean);
    if (this._editorSurface) {
      if (view.mode === "output") {
        const output = this._editorSurface.querySelector("textarea.ute-preview");
        if (output) (view.outputScroll ??= {})[view.polarity] = output.scrollTop;
      } else view.scroll[view.polarity] = this._editorSurface.scrollTop;
    }
    if (notify) this._onChange(serialize(this._state));
  }

  _restoreViewSelection() {
    const view = this._state.editorView;
    this._selection.clear();
    for (const item of view.selected[view.polarity] || []) {
      if ((view.polarity === "negative") !== (item.section === "negative")) continue;
      const tag = this._state[item.section]?.[item.index];
      if (tag) this._selection.add(tag.id);
    }
    this._focusSection = view.polarity === "negative" ? "negative" : "mainTags";
  }

  _renderTabs() {
    this._tabsEl.textContent = "";
    const view = this._state.editorView;
    for (const [side, label] of [["positive", "描きたい内容"], ["negative", "除外したい内容"]]) {
      const count = side === "positive" ? this._state.prefix.length + this._state.mainTags.length : this._state.negative.length;
      const button = h("button", { type: "button", class: "ute-tab" + (view.polarity === side ? " active" : ""), "aria-pressed": view.polarity === side }, `${label} ${count}`);
      stop(button, false);
      button.addEventListener("click", () => {
        this._finishViewEdit(); this._saveView(); view.polarity = side;
        this._restoreViewSelection(); this._onChange(serialize(this._state)); this.render();
      });
      this._tabsEl.append(button);
    }
  }

  _finishViewEdit() {
    if (!this._activeChipEdit) return;
    const finish = this._activeChipEdit; this._activeChipEdit = null; finish(true);
  }

  /**
   * Icon + label toolbar buttons, ordered by real-world usage frequency
   * (weight/undo/delete during everyday tag tuning first; one-off setup
   * actions like LoRA/AI-import last).
   */
  _renderToolbar() {
    this._toolbarEl.textContent = "";
    // Icon-only for anything the icon alone already says (tooltip carries the
    // full name) — showing a text label next to it was pure duplication (e.g.
    // the "( )" emphasis icon followed by an "( )" label). The weight buttons
    // are the one exception: the icon can show "more/less", not the specific
    // ±0.1 amount, which is real information worth the extra width.
    const btn = (iconName, label, fn, cls, showLabel) => {
      const b = h("button", { class: "ute-btn" + (showLabel ? "" : " icon-only") + (cls ? " " + cls : ""), title: label });
      const ic = getIcon(iconName);
      if (ic) b.appendChild(ic);
      if (showLabel) b.appendChild(h("span", { class: "ute-btn-label" }, label));
      stop(b, false);
      b.addEventListener("click", fn);
      return b;
    };
    const undo = btn("undo", "元に戻す", () => this._undo());
    undo.disabled = !this._history.canUndo();
    const redo = btn("redo", "やり直す", () => this._redo());
    redo.disabled = !this._history.canRedo();
    this._toolbarEl.append(
      btn("weightMinus", "−0.1", () => this._weightSelection(-0.1), null, true),
      btn("weightPlus", "+0.1", () => this._weightSelection(0.1), null, true),
      undo, redo,
      btn("trash", "選択を削除", () => this._deleteSelection(), "warn"),
      btn("deselect", "選択を解除", () => { this._selection.clear(); this.render(); }),
      btn("emphasis", "強調括弧を切替", () => this._emphasisSelection()),
      btn("dedupe", "重複タグを整理", () => this._dedupe()),
      btn("lora", "CLIP強度", (e) => showLoraClipControl({ x: e.clientX, y: e.clientY }, this._getHostNode), null, true),
      btn("lora", "トリガー追加", (e) => this._addLora(e), null, true),
    );
  }

  /**
   * "Add LoRA": scans the current workflow graph for LoRA loader nodes and
   * opens a picker to insert one/all/none of them (falls back to typing a
   * name manually) — restores the original "auto-detect from workflow"
   * behavior that a prior refactor accidentally reduced to a bare prompt().
   */
  _addLora(e) {
    const found = scanLoraNodes(this._getGraphNodes());
    const anchor = e ? { x: e.clientX, y: e.clientY } : { x: 100, y: 100 };
    showLoraPicker(anchor, found, (lora, triggerText) => this._insertLora(lora, triggerText));
  }

  /** Insert only the LoRA trigger tags; the loader node owns LoRA application. */
  _insertLora(_lora, triggerText) {
    const tags = loraTriggerTags(triggerText);
    if (!tags.length) return;
    this._state.prefix.push(...tags);
    this.commit();
    this.render();
  }

  _importAi() {
    const text = window.prompt("Positiveへ読み込むタグ（カンマ区切り）:");
    if (text == null) return;
    const tags = splitTags(text).map((t) => ({ ...t, ai_imported: true }));
    if (tags.length) { this._state.mainTags.push(...tags); this.commit(); this.render(); }
  }

  _renderTagsTab() {
    const view = this._state.editorView;
    const bar = h("div", { class: "ute-viewbar" });
    for (const [mode, label] of [["edit", "タグを編集"], ["output", "出力文字列"]]) {
      const button = h("button", { type: "button", class: "ute-btn" + (view.mode === mode ? " active" : ""), "aria-pressed": view.mode === mode }, label);
      stop(button, false);
      button.addEventListener("click", () => { this._finishViewEdit(); this._saveView(); view.mode = mode; this._onChange(serialize(this._state)); this.render(); });
      bar.append(button);
    }
    const pan = h("button", { type: "button", class: "ute-btn ute-pan-handle", title: "左ドラッグで画面を移動。ノード上は中ボタン／Space＋ドラッグも使用できます。" }, "✥ 画面移動");
    bar.append(pan);
    this._bodyEl.append(bar);
    const surface = h("div", { class: "ute-editor-surface" });
    this._editorSurface = surface;
    if (view.mode === "output") {
      surface.append(this._previewSection(view.polarity === "negative" ? "除外したい内容の出力" : "描きたい内容の出力", view.polarity === "negative" ? this._negativePreview() : this._positivePreview(), view.polarity === "negative" ? "neg" : "pos"));
    } else if (view.polarity === "negative") surface.append(this._section("negative"));
    else surface.append(this._section("prefix"), this._section("mainTags"));
    this._bodyEl.append(surface);
    const restoreEditScroll = view.scroll[view.polarity] || 0;
    const restoreOutputScroll = view.outputScroll?.[view.polarity] || 0;
    surface.addEventListener("scroll", () => {
      if (view.mode !== "edit" || this._editorSurface !== surface) return;
      view.scroll[view.polarity] = surface.scrollTop;
      clearTimeout(this._viewScrollTimer);
      this._viewScrollTimer = setTimeout(() => this._onChange(serialize(this._state)), 150);
    });
    const output = surface.querySelector("textarea.ute-preview");
    if (output) {
      output.addEventListener("scroll", () => {
        if (view.mode !== "output" || this._editorSurface !== surface) return;
        (view.outputScroll ??= {})[view.polarity] = output.scrollTop;
        clearTimeout(this._viewScrollTimer);
        this._viewScrollTimer = setTimeout(() => this._onChange(serialize(this._state)), 150);
      });
    }
    const settings = h("details", { class: "ute-clip-settings" });
    settings.append(h("summary", {}, "CLIPの読み取り層（LoRA強度とは別）"), this._clipRow());
    this._bodyEl.append(settings);
    // Restore after the footer has taken its space. Restoring earlier clamps
    // to the temporarily larger surface's smaller scroll range and loses
    // roughly one footer-height on every edit/output round trip.
    requestAnimationFrame(() => {
      if (this._editorSurface !== surface || !surface.isConnected) return;
      surface.scrollTop = view.mode === "edit" ? restoreEditScroll : 0;
      if (output) output.scrollTop = restoreOutputScroll;
    });
  }

  _section(section) {
    const meta = SECTION_META[section];
    const arr = this._state[section];
    const collapsed = !!this._collapsed[section];

    const headKids = [
      h("span", { class: "ute-sec-caret" }, collapsed ? "▸" : "▾"),
      h("span", { class: "ute-sec-title" }, meta.label),
    ];
    if (meta.hint) headKids.push(h("span", { class: "ute-sec-hint" }, meta.hint));
    headKids.push(h("span", { class: "ute-sec-count" }, String(arr.length)));
    const head = h("div", { class: "ute-sec-head" }, headKids);
    stop(head, false);
    head.addEventListener("click", () => {
      this._collapsed[section] = !collapsed;
      if (section === "prefix") { this._state.editorView.prefixOpen = collapsed; this._saveView(); }
      this.render();
    });

    const wrap = h("div", { class: "ute-section" }, head);
    if (collapsed) return wrap;

    const zone = h("div", { class: "ute-chips" });
    zone.dataset.uteSection = section;
    // ドラッグで下限の高さを決められるようにする（中身が増えれば超えて伸びる）
    this._makeResizable(zone, "chips:" + section, 24);
    this._drag.bindZone(zone, section);
    const dupeKeys = this._dupeKeys(arr);
    arr.forEach((tag, i) =>
      zone.appendChild(tag.type === "divider" ? this._dividerEl(tag, section, i) : this._chip(tag, section, i, dupeKeys))
    );
    // Right-click on empty area → "Add divider" (ported from original _showAreaCtx).
    zone.addEventListener("contextmenu", (e) => {
      if (e.target !== zone) return;
      e.preventDefault();
      e.stopPropagation();
      this._showAreaMenu(e, section);
    });
    wrap.appendChild(zone);
    wrap.appendChild(this._addRow(section));
    return wrap;
  }

  /**
   * A full-width section divider. Not text-editable (nothing to type), but
   * otherwise behaves like a chip: click to select, drag to reorder/move
   * (alone or as part of a multi-selection — e.g. a divider plus the tags
   * around it), right-click to remove. Previously had neither, which made it
   * both hard to select and impossible to move without deleting + recreating.
   */
  _dividerEl(tag, section, index) {
    const cls = ["ute-divider"];
    if (this._selection.has(tag.id)) cls.push("selected");
    const el = h("div", { class: cls.join(" "), title: "区切り線：クリックで選択・ドラッグで移動・右クリックで操作" });
    el.dataset.uteId = String(tag.id); // keep in [data-ute-id] order so drop-index math stays aligned with the array

    el.addEventListener("click", (e) => {
      if (e.detail >= 2) return; // no edit-mode for dividers; don't fight a stray dblclick
      this._clickSelect(e, section, index, tag.id);
    });
    el.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      e.stopPropagation();
      this._menu.open(e.clientX, e.clientY, [
        { label: "これだけ選択", fn: () => this._selectOnly(tag.id) },
        { sep: true },
        { label: "区切り線を削除", cls: "warn", fn: () => this._removeTag(tag.id) },
      ]);
    });
    this._drag.bindChip(el, section, tag.id, () => this._dragGroupFor(section, tag.id));
    return el;
  }

  /** Empty-area context menu: insert a divider at the cursor position. */
  _showAreaMenu(e, section) {
    this._menu.open(e.clientX, e.clientY, [
      { label: "─ 区切り線を追加", fn: () => this._addDivider(section, e.clientX, e.clientY) },
    ]);
  }

  /**
   * Insert a divider into `section`. With clientX/clientY it lands at the cursor
   * position (matching the original); without them it appends (harness/tests).
   */
  _addDivider(section, clientX, clientY) {
    const arr = this._state[section];
    if (!arr) return;
    let insertIdx = arr.length;
    if (clientX != null && clientY != null) {
      const zone = this._root.querySelector(`[data-ute-section="${section}"]`);
      if (zone) {
        const elems = [...zone.querySelectorAll("[data-ute-id]")];
        for (let i = 0; i < elems.length; i++) {
          const r = elems[i].getBoundingClientRect();
          if (clientY < r.bottom && clientX < r.left + r.width / 2) { insertIdx = i; break; }
          if (clientY < r.top) { insertIdx = i; break; }
        }
      }
    }
    arr.splice(insertIdx, 0, makeDivider());
    this.commit();
    this.render();
  }

  _chip(tag, section, index, dupeKeys) {
    const cls = ["ute-chip"];
    if (this._selection.has(tag.id)) cls.push("selected");
    if (tag.bypassed) cls.push("bypassed");
    if (tag.locked) cls.push("locked");
    if (tag.color) cls.push("color-" + tag.color);
    if (tag.type === "lora") cls.push("lora");
    if (dupeKeys && tag.type !== "divider" && dupeKeys.has(coreKey(tag))) cls.push("dup");
    // Keep every chip on one line; full text remains available in title/edit.

    const chip = h("div", { class: cls.join(" "), title: tag.raw });
    chip.dataset.uteId = String(tag.id);
    // One-shot flash after an Alt+Click swap (consumed on the next render).
    if (this._swapAnimId === tag.id) { chip.classList.add("alt-swap"); this._swapAnimId = null; }

    if (tag.locked) chip.appendChild(h("span", { class: "ute-chip-lock", title: "保護中" }, "🔒"));
    if (tag.type === "dparen") chip.appendChild(h("span", { class: "ute-chip-badge dparen", title: "二重強調 (( ))" }, "(( ))"));
    if (tag.ai_imported) chip.appendChild(h("span", { class: "ute-chip-badge ai", title: "AIから追加" }, "AI"));
    chip.appendChild(h("span", { class: "ute-chip-label" }, chipLabel(tag)));
    const w = chipWeight(tag);
    if (w && w !== "1") chip.appendChild(h("span", { class: "ute-chip-w" }, w));

    // Locked tags cannot be removed inline (matches original — use the menu to unlock).
    if (!tag.locked) {
      const x = h("button", { type: "button", class: "ute-chip-x", title: "タグを削除", "aria-label": `${chipLabel(tag)} を削除` }, "×");
      x.addEventListener("pointerdown", (e) => e.stopPropagation());
      x.addEventListener("click", (e) => {
        e.stopPropagation();
        this._removeTag(tag.id);
      });
      chip.appendChild(x);
    }

    chip.addEventListener("click", (e) => {
      // e.detail >= 2 means this "click" is the second half of a dblclick
      // gesture (regardless of modifiers) — always skip both the alt-cycle
      // and plain-select branches below so neither fights the upcoming
      // 'dblclick'/'click' resolution (an Alt+double-click would otherwise
      // cycle the tag twice AND still open edit-mode on the result).
      if (e.detail >= 2) return;
      // Alt+Click cycles the tag through its swap ring (custom or built-in).
      if (e.altKey && !e.ctrlKey && !e.shiftKey && !e.metaKey) {
        e.preventDefault();
        e.stopPropagation();
        this._altCycleTag(tag.id);
        return;
      }
      this._clickSelect(e, section, index, tag.id);
    });
    chip.addEventListener("dblclick", (e) => {
      e.stopPropagation();
      if (e.altKey) return; // Alt+dblclick is swap-cycle territory, not edit-mode.
      if (tag.locked) return;
      // The captured `chip` reference may have been detached by a render()
      // triggered from the first click of this double-click gesture (e.g. a
      // selection change fired before the 'detail>=2' guard above kicked in
      // for the second click). Re-resolve the live element for this tag id
      // so the edit input is appended to something actually on screen.
      const live = (this._root && this._root.querySelector) ? this._root.querySelector(`[data-ute-id="${tag.id}"]`) : null;
      this._editChip(live && live.isConnected ? live : chip, tag, section, index);
    });
    chip.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (section) this._showChipMenu(e, tag, section);
      else this._showCharChipMenu(e, tag);
    });

    if (section && !tag.locked) {
      this._drag.bindChip(chip, section, tag.id, () => this._dragGroupFor(section, tag.id));
    } else {
      chip.addEventListener("pointerdown", (e) => e.stopPropagation());
    }
    return chip;
  }

  /**
   * Tags to drag together when a gesture starts on `id`: the whole selection
   * (in doc order) if `id` is part of a multi-selection, else just itself.
   */
  _dragGroupFor(section, id) {
    if (this._selection.has(id) && this._selection.size > 1) {
      const out = [];
      for (const sec of SECTIONS) {
        for (const t of this._state[sec]) if (this._selection.has(t.id)) out.push({ section: sec, id: t.id });
      }
      return out;
    }
    return [{ section, id }];
  }

  _editChip(chip, tag, section, index) {
    const input = h("input", { class: "ute-chip-edit", value: tag.raw });
    stop(input);
    // Enter a dedicated editing state: the chip must stop being a drag source
    // and stop reacting to clicks while the input is live, otherwise a
    // click-drag meant to select/position text inside the input is instead
    // read as "drag this chip" (native HTML5 DnD checks the closest
    // draggable ancestor, not just the click target) and a plain click
    // bubbles up to the chip's own 'click' handler and re-renders, blowing
    // the input away mid-edit. Both are undone by `render()` once editing
    // ends (Enter / Escape / blur), since that always rebuilds a fresh chip.
    chip.draggable = false;
    input.addEventListener("mousedown", (e) => e.stopPropagation());
    input.addEventListener("click", (e) => e.stopPropagation());
    chip.textContent = "";
    chip.appendChild(input);
    input.focus();
    input.select();
    let done = false;
    const finish = (save, redraw = true) => {
      if (done) return;
      done = true;
      if (this._activeChipEdit === finish) this._activeChipEdit = null;
      if (save) {
        const next = makeTag(input.value.trim(), tagFlags(tag));
        const arr = section ? this._state[section] : this._charTagsById(tag.id);
        if (next && arr) { next.id = tag.id; const i = arr.findIndex((t) => t.id === tag.id); if (i >= 0) arr[i] = next; }
        this.commit();
      }
      if (redraw) this.render();
      else {
        // Blur can precede the click on a view button or another input. Replace
        // only this chip so that the intended click/focus target stays alive.
        const loc = this._findTag(tag.id);
        if (chip.isConnected && loc) {
          const arr = loc.sec ? this._state[loc.sec] : loc.char.tags;
          chip.replaceWith(this._chip(loc.tag, loc.sec, loc.idx, this._dupeKeys(arr)));
        }
        const undo = this._toolbarEl?.querySelector('[title="元に戻す"]');
        const redo = this._toolbarEl?.querySelector('[title="やり直す"]');
        if (undo) undo.disabled = !this._history.canUndo();
        if (redo) redo.disabled = !this._history.canRedo();
      }
    };
    this._activeChipEdit = finish;
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); finish(true); }
      else if (e.key === "Escape") { e.preventDefault(); finish(false); }
    });
    // Leaving an edit commits the text; explicit Escape remains cancellation.
    input.addEventListener("blur", () => {
      setTimeout(() => { if (document.activeElement !== input) finish(true, false); }, 0);
    });
  }

  _charTagsById(id) {
    for (const ch of this._state.characters) {
      if (ch.tags.some((t) => t.id === id)) return ch.tags;
    }
    return null;
  }

  _removeTag(id) {
    const loc = this._findTag(id);
    if (!loc) return;
    if (loc.sec) this._state[loc.sec].splice(loc.idx, 1);
    else if (loc.char) loc.char.tags.splice(loc.idx, 1);
    this._selection.delete(id);
    this.commit();
    this.render();
  }

  // ── context menus (ported from original _showContextMenu family) ─────────────
  _showChipMenu(e, tag, section) {
    this._root.focus({ preventScroll: true });
    // If this chip is part of a multi-selection, every action below (except
    // "Select only") applies to the WHOLE selection, not just this one chip.
    const multi = this._selection.has(tag.id) && this._selection.size > 1;
    const selected = () => this._ensureSelected(tag.id);
    const prospectiveSelection = this._selection.has(tag.id) ? this._selectedTagsInOrder() : [tag];
    const items = [
      { header: `${multi ? `${this._selection.size}個を選択 · ` : ""}${chipLabel(tag)}` },
      { label: "コピー", fn: () => { selected(); this._copySelection(); } },
      { label: "切り取り", disabled: !!tag.locked, fn: () => { selected(); this._cutSelection(); } },
      { label: "文字列としてコピー", fn: () => { selected(); this._copyAsText(); } },
      { label: "複製", fn: () => { selected(); this._duplicateSelection(); } },
      { label: "重み・表記", children: [
        { label: "重みを指定…", disabled: !!tag.locked || multi, fn: () => this._showWeightPopoverFor(tag) },
        ...(tag.type === "dparen" ? [{ label: "二重括弧を解除", fn: () => this._unwrapDparen(tag.id) }] : []),
        ...(tag.raw.includes("_") && tag.type !== "lora" ? [{ label: "下線を空白に変換", fn: () => { selected(); this._convertUnderscoreSelected(); } }] : []),
      ] },
      { label: "移動先", children: SECTION_LABELS.filter(([key]) => key !== section).map(([key,label]) => ({ label, fn: () => { selected(); this._moveSelectedToSection(key); } })) },
      { label: "選択・状態", children: [
        { label: "これだけ選択", fn: () => this._selectOnly(tag.id) },
        { label: tag.locked ? "保護を解除" : "タグを保護", fn: () => { selected(); this._toggleLockSelection(tag.id); } },
        { label: tag.bypassed ? "タグを有効にする" : "タグを無効にする", fn: () => { selected(); this._bypassSelection(); } },
      ] },
    ];
    this._appendSwapItems(items, tag);
    this._appendColorItems(items, tag);
    items.push({ sep: true },
      { label: "選択から無効タグを削除", cls: "warn", disabled: !prospectiveSelection.some((t) => t.type !== "divider" && t.bypassed), fn: () => { selected(); this._removeSelectedBypassed(); } },
      { label: "タグを削除", cls: "warn", disabled: !multi && !!tag.locked, fn: () => { selected(); this._deleteSelection(); } });
    this._menu.open(e.clientX, e.clientY, items);
  }

  _showCharChipMenu(e, tag) {
    const loc = this._findTag(tag.id);
    const items = [];
    items.push({ label: "↩ Move to Main Tags", fn: () => { this._ensureSelected(tag.id); this._moveSelectedToSection("mainTags"); } });
    for (const other of this._state.characters) {
      if (loc?.char && other.id === loc.char.id) continue;
      items.push({ label: `→ ${other.name || "Character"}`, fn: () => { this._ensureSelected(tag.id); this._moveSelectedToChar(other.id); } });
    }
    items.push({ sep: true });
    items.push({ label: "✕ Remove", cls: "warn", fn: () => { this._ensureSelected(tag.id); this._deleteSelection(); } });
    this._appendColorItems(items, tag);
    this._menu.open(e.clientX, e.clientY, items);
  }

  _appendColorItems(items, tag) {
    const cur = tag.color ?? null;
    items.push({ label: "色ラベル", children: COLOR_MAP.map(([val,label,clr]) => ({
      label, color: clr, active: cur === val, fn: () => this._applyColor(tag,val),
    })) });
  }

  // ── single-tag / bulk mutations backing the menus ────────────────────────────
  /** Locate the mutable array + index for a tag id (section or character). */
  _tagSlot(id) {
    const loc = this._findTag(id);
    if (!loc) return null;
    const arr = loc.sec ? this._state[loc.sec] : loc.char.tags;
    const i = arr.findIndex((t) => t.id === id);
    return i >= 0 ? { arr, i } : null;
  }

  _transformTag(id, fn) {
    const slot = this._tagSlot(id);
    if (!slot) return;
    slot.arr[slot.i] = fn(slot.arr[slot.i]);
    this.commit();
    this.render();
  }

  /**
   * Toggle lock for the WHOLE current selection (all-lock if the anchor tag is
   * currently unlocked, all-unlock otherwise) — mirrors `_bypassSelection`.
   * Locked tags are dropped from the selection (matches the single-tag original:
   * a locked chip can't stay selected for further bulk edits).
   */
  _toggleLockSelection(anchorId) {
    const slot = this._tagSlot(anchorId);
    if (!slot) return;
    const wantLocked = !slot.arr[slot.i].locked;
    this._transformSelected((t) => ({ ...t, locked: wantLocked }));
    if (wantLocked) this._selection.clear();
  }

  /** Move every selected tag into section `dst`, preserving relative order. */
  _moveSelectedToSection(dst) {
    if (!this._selection.size || !this._state[dst]) return;
    this._saveView();
    const tags = [];
    for (const sec of SECTIONS) {
      for (const t of this._state[sec]) if (this._selection.has(t.id)) tags.push(t);
      this._state[sec] = this._state[sec].filter((t) => !this._selection.has(t.id));
    }
    for (const ch of this._state.characters) {
      for (const t of ch.tags) if (this._selection.has(t.id)) tags.push(t);
      ch.tags = ch.tags.filter((t) => !this._selection.has(t.id));
    }
    if (!tags.length) return;
    this._state[dst].push(...tags);
    this._selection = new Set(tags.map((tag) => tag.id));
    const view = this._state.editorView;
    view.polarity = dst === "negative" ? "negative" : "positive";
    view.mode = "edit";
    view.selected = { positive: [], negative: [] };
    if (dst === "prefix") { view.prefixOpen = true; this._collapsed.prefix = false; }
    this._focusSection = dst;
    this._anchor = { section: dst, index: this._state[dst].length - tags.length };
    this._editorSurface = null; // don't copy the previous side's scroll into this side
    this.commit();
    this.render();
    this._root.querySelector(`[data-ute-id="${tags[0].id}"]`)?.scrollIntoView({ block: "nearest" });
    this._root.focus({ preventScroll: true });
    this._saveView();
  }

  /** Move every selected tag onto character `charId`, preserving relative order. */
  _moveSelectedToChar(charId) {
    const ch = this._state.characters.find((c) => c.id === charId);
    if (!ch || !this._selection.size) return;
    const tags = [];
    for (const sec of SECTIONS) {
      for (const t of this._state[sec]) if (this._selection.has(t.id)) tags.push(t);
      this._state[sec] = this._state[sec].filter((t) => !this._selection.has(t.id));
    }
    for (const other of this._state.characters) {
      if (other.id === charId) continue;
      for (const t of other.tags) if (this._selection.has(t.id)) tags.push(t);
      other.tags = other.tags.filter((t) => !this._selection.has(t.id));
    }
    if (!tags.length) return;
    ch.tags.push(...tags);
    this.commit();
    this.render();
  }

  _selectOnly(id) {
    this._selection.clear();
    this._selection.add(id);
    this._saveView();
    this.render();
  }

  /** Ensure `id` is part of the current selection; if not, select only it. */
  _ensureSelected(id) {
    if (!this._selection.has(id)) {
      this._selection.clear();
      this._selection.add(id);
    }
  }

  /** Turn a ((tag)) chip back into a plain tag. */
  _unwrapDparen(id) {
    this._transformTag(id, (t) => {
      if (t.type !== "dparen") return t;
      const next = makeTag(t.text, tagFlags(t));
      if (next) next.id = t.id;
      return next || t;
    });
  }

  // ── Alt+Click swap / cycle ────────────────────────────────────────────────────
  /** Replace a tag with `newRaw`, preserving id + flags, and flash it. */
  _swapTagRaw(id, newRaw) {
    const slot = this._tagSlot(id);
    if (!slot) return;
    const tag = slot.arr[slot.i];
    const next = makeTag(newRaw, tagFlags(tag));
    if (!next) return;
    next.id = tag.id;
    this._swapAnimId = tag.id;
    slot.arr[slot.i] = next;
    this.commit();
    this.render();
  }

  /** Cycle a tag to the next member of its swap ring (custom → built-in). */
  _altCycleTag(id) {
    const slot = this._tagSlot(id);
    if (!slot) return;
    const next = nextSwapText(slot.arr[slot.i].raw);
    if (next) this._swapTagRaw(id, next);
  }

  _applySwapTo(id, target) {
    this._swapTagRaw(id, target);
  }

  /** Append the "Swap" section (ring members + add/clear) to a chip menu. */
  _appendSwapItems(items, tag) {
    const self = SwapDB._norm(tag.raw);
    const group = (SwapDB.getGroup(tag.raw) ?? []).filter((t) => t !== self);
    items.push({ label: "置換（Alt＋クリックで巡回）", children: [
      ...group.map((target) => ({ label: `⇄ ${target}`, fn: () => this._applySwapTo(tag.id, target) })),
      { label: "置換候補を追加…", fn: () => this._addSwapTarget(tag) },
      ...(group.length ? [{ label: "置換候補を消去", cls: "warn", fn: () => SwapDB.clearSwaps(tag.raw) }] : []),
    ] });
  }

  _addSwapTarget(tag) {
    const target = window.prompt(`「${tag.raw}」の置換候補（Alt＋クリックで巡回）:`);
    if (target && target.trim()) SwapDB.addSwap(tag.raw, target.trim());
  }

  /** Apply a color to a tag; if it is part of a multi-selection, colour all. */
  _applyColor(tag, color) {
    const targets = this._selection.has(tag.id) && this._selection.size > 1
      ? new Set(this._selection)
      : new Set([tag.id]);
    const paint = (t) => (targets.has(t.id) ? { ...t, color: color || undefined } : t);
    for (const sec of SECTIONS) this._state[sec] = this._state[sec].map(paint);
    for (const ch of this._state.characters) ch.tags = ch.tags.map(paint);
    this.commit();
    this.render();
  }

  _showWeightPopoverFor(tag) {
    const el = this._root.querySelector(`[data-ute-id="${tag.id}"]`);
    if (!el) return;
    const rect = el.getBoundingClientRect();
    showWeightPopover(rect, tagWeight(tag), (w) => this._transformTag(tag.id, (t) => {
      const next = setWeight(t, w);
      return next || t;
    }));
  }

  _addRow(section) {
    const input = h("input", { class: "ute-input", placeholder: SECTION_META[section].multi ? "タグを追加（カンマで複数）…" : "タグを追加…" });
    stop(input);
    attachAutocomplete(input, section);
    const add = () => {
      const v = input.value.trim();
      if (v) { this._addTags(section, v); }
    };
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); add(); }
    });
    const btn = h("button", { class: "ute-btn" }, "＋ 追加");
    stop(btn, false);
    btn.addEventListener("click", add);
    return h("div", { class: "ute-addrow" }, [input, btn]);
  }

  /**
   * ボックスをドラッグでリサイズ可能にし、決めた高さを「下限」として保存する。
   *
   * ネイティブの `resize: vertical` は inline の height を書き込むが、それを
   * そのまま使うと中身が増えたときにスクロールへ押し込まれてしまう。
   * ユーザーの要望は「初期サイズは自分で決めたい。足りなければ伸びてよい」
   * なので、確定時に height を min-height へ転写して height は空に戻す。
   * こうすると指定値が下限になり、中身が増えれば自然に伸びる。
   *
   * @param {HTMLElement} el
   * @param {string} key      boxHeights の保存キー（例 "chips:mainTags"）
   * @param {number} fallback 未保存時の下限 px
   */
  _makeResizable(el, key, fallback) {
    const saved = Number(this._state.boxHeights?.[key]);
    const base = Number.isFinite(saved) && saved >= 16 ? saved : fallback;
    el.style.minHeight = base + "px";
    el.style.resize = "vertical";
    if (!el.style.overflowY) el.style.overflowY = "auto";

    const commit = () => {
      const h = parseInt(el.style.height, 10);
      if (!Number.isFinite(h)) return;
      el.style.height = "";                       // 下限指定へ変換
      const px = Math.max(16, Math.min(1200, Math.round(h)));
      el.style.minHeight = px + "px";
      if (!this._state.boxHeights) this._state.boxHeights = {};
      if (this._state.boxHeights[key] !== px) {
        this._state.boxHeights[key] = px;
        this.commit();
      }
    };
    // ネイティブ resize ハンドルの操作は pointerup で終わる。
    // 取りこぼし対策で mouseup / pointerleave でも確定する。
    el.addEventListener("pointerup", commit);
    el.addEventListener("mouseup", commit);
    el.addEventListener("pointerleave", commit);
    return el;
  }

  _previewSection(title, text, key) {
    const box = h("textarea", { class: "ute-preview", readOnly: true, value: text || "", "aria-label": title, placeholder: "出力は空です" });

    // プレビューは「読んでコピーする」ための領域。ComfyUI のキャンバスは
    // Ctrl+A / Ctrl+C / Ctrl+X などを自前のショートカットとして拾ってしまい、
    // そのままだと全ノード選択やノードのコピーが走って本文をコピーできない。
    // この要素の上では非破壊のテキスト操作をブラウザ既定の動作へ通す。
    box.tabIndex = 0;               // フォーカスを受けられるようにする
    box.addEventListener("pointerdown", (e) => e.stopPropagation());
    box.addEventListener("keydown", (e) => {
      // 選択・コピー系はブラウザに任せる（既定動作は止めない）
      e.stopPropagation();
    });

    if (key) this._makeResizable(box, "preview:" + key, 48);
    return h("div", { class: "ute-section" }, [
      h("div", { class: "ute-sec-head" }, [
        h("span", { class: "ute-sec-caret" }, "▾"),
        h("span", { class: "ute-sec-title" }, title),
      ]),
      box,
    ]);
  }

  // CLIP skip の値が実際にどう解釈されるかを返す。
  // バックエンド (tag_editor_node.py) と同じ規則:
  //   |n| <= 1  -> clip_layer を呼ばない（最終層 = 標準 CLIPTextEncode と同じ）
  //   |n| >  1  -> stop_at_clip_layer = -|n| を適用
  // ComfyUI の clip_layer は「呼んだ時点で」隠れ状態を返す経路に入るため、
  // skip しないときは呼ばないことが標準との一致条件になる。
  static describeClipSkip(n) {
    const v = Number.isFinite(n) ? Math.trunc(n) : 1;
    const a = Math.abs(v);
    if (a <= 1) {
      return { applied: false, stopAt: null, text: "最終層を使用（skip なし・標準と同じ）" };
    }
    const stopAt = Math.max(-24, -a);
    return {
      applied: true,
      stopAt,
      text: `stop_at_clip_layer=${stopAt}（${a - 1} 層スキップ）`,
    };
  }

  _clipRow() {
    // 表示は A1111 慣習の正の値に統一する（1 = 最終層）。
    // 保存済みの負値（旧デフォルト -2 等）は絶対値へ正規化して見せる。
    const initial = Math.abs(Math.trunc(this._state.clipSkip ?? 1)) || 1;
    const inp = h("input", {
      class: "ute-num", type: "number", step: "1", min: "1", max: "24",
      value: String(initial),
      title: "1 = 最終層（skip なし）。2 以上で層をさかのぼる。",
    });
    stop(inp);

    // 実際の解釈をその場に出す。生成後はバックエンドが返した実適用値で上書きされる。
    const note = h("span", { class: "ute-hint" }, "");
    note.style.cssText = "font-size:11px;opacity:0.8;margin-left:6px";

    const render = (info, fromBackend) => {
      note.textContent = (fromBackend ? "適用済: " : "→ ") + info.text;
      note.style.opacity = fromBackend ? "1" : "0.8";
    };
    render(TagEditor.describeClipSkip(initial), false);
    // 直近の生成で報告された実適用値があれば、再描画時にも復元する
    const pending = this._clipSkipReport;

    // 生成後にバックエンドから届いた実適用値を反映するためのフック
    this._applyClipSkipReport = (rep) => {
      if (!rep) return;
      const a = Math.abs(Math.trunc(rep.raw ?? initial)) || 1;
      inp.value = String(a);
      render(
        rep.applied
          ? { text: `stop_at_clip_layer=${rep.stop_at_layer}（${a - 1} 層スキップ）` }
          : { text: "最終層を使用（skip なし・標準と同じ）" },
        true,
      );
    };

    inp.addEventListener("change", () => {
      let v = parseInt(inp.value, 10);
      if (Number.isNaN(v) || v === 0) v = 1;
      v = Math.min(24, Math.abs(v));
      this._state.clipSkip = v;
      inp.value = String(v);
      render(TagEditor.describeClipSkip(v), false);
      this.commit();
    });

    if (pending) this._applyClipSkipReport(pending);

    // 保存値が負のまま残っていたら、この時点で正規化して保存し直す
    if (this._state.clipSkip !== initial) {
      this._state.clipSkip = initial;
      this.commit();
    }

    return h("div", { class: "ute-row" }, [h("label", {}, "CLIP skip"), inp, note]);
  }

  // ── previews (client mirror of core/canonical_tags.build_output) ────────────
  _activeRaws(section) {
    return this._state[section].filter((t) => !t.bypassed && t.type !== "divider");
  }

  _positivePreview() {
    const combined = [
      ...this._activeRaws("prefix"),
      ...this._activeRaws("mainTags"),
      ...this._activeRaws("suffix"),
    ];
    const block = new Set(this._state.blocklist.map((t) => coreKey(t)));
    const seen = new Set();
    const out = [];
    for (const t of combined) {
      const k = coreKey(t);
      if (block.has(k) || seen.has(k)) continue;
      seen.add(k);
      out.push(t.raw);
    }
    return out.join(", ");
  }

  _negativePreview() {
    return this._activeRaws("negative").map((t) => t.raw).join(", ");
  }

  // ── Characters tab ───────────────────────────────────────────────────────────
  _renderCharactersTab() {
    const add = h("button", { class: "ute-btn" }, "＋ Add character");
    stop(add, false);
    add.addEventListener("click", () => { chars.addCharacter(this._state); this.commit(); this.render(); });
    this._bodyEl.appendChild(h("div", { class: "ute-toolbar" }, add));

    for (const ch of this._state.characters) {
      this._bodyEl.appendChild(this._characterCard(ch));
    }
    if (!this._state.characters.length) {
      this._bodyEl.appendChild(h("div", { class: "ute-preview" }, "No characters yet."));
    }
  }

  _characterCard(ch) {
    const name = h("input", { class: "ute-char-name", value: ch.name });
    stop(name);
    name.addEventListener("change", () => { chars.renameCharacter(this._state, ch.id, name.value); this.commit(); });

    const del = h("button", { class: "ute-btn warn" }, "Remove → Main");
    stop(del, false);
    del.addEventListener("click", () => { chars.removeCharacter(this._state, ch.id); this.commit(); this.render(); });

    const card = h("div", { class: "ute-char-card" }, h("div", { class: "ute-char-head" }, [name, del]));

    const zone = h("div", { class: "ute-chips" });
    ch.tags.forEach((tag, i) => {
      const chip = this._chip(tag, null, i);
      // Removing a character-chip returns it to Main (overrides the X handler).
      const x = chip.querySelector(".ute-chip-x");
      if (x) {
        const clone = x.cloneNode(true);
        x.replaceWith(clone);
        clone.addEventListener("pointerdown", (e) => e.stopPropagation());
        clone.addEventListener("click", (e) => {
          e.stopPropagation();
          chars.returnTagToMain(this._state, ch.id, tag.id);
          this.commit();
          this.render();
        });
      }
      zone.appendChild(chip);
    });
    card.appendChild(zone);

    const input = h("input", { class: "ute-input", placeholder: "add tags (comma)…" });
    stop(input);
    attachAutocomplete(input, "char");
    const addTags = () => {
      const v = input.value.trim();
      if (!v) return;
      for (const t of splitTags(v)) bumpHistory(t.raw, "positive");
      chars.addTagsToCharacter(this._state, ch.id, splitTags(v).map((t) => t.raw));
      this.commit();
      this.render();
    };
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); addTags(); } });
    const addBtn = h("button", { class: "ute-btn" }, "＋");
    stop(addBtn, false);
    addBtn.addEventListener("click", addTags);
    card.appendChild(h("div", { class: "ute-addrow" }, [input, addBtn]));
    return card;
  }

  // ── keyboard + paste ─────────────────────────────────────────────────────────
  _installKeyboard() {
    // The structured clipboard is shared by all Tag Editor nodes. The OS
    // clipboard is still written/read best-effort but may be blocked by the browser.
    this._hover = false;
    this._root.addEventListener("pointerenter", () => { this._hover = true; });
    this._root.addEventListener("pointerleave", () => { this._hover = false; });

    // Document-level CAPTURE listener so we intercept before ComfyUI's global
    // shortcuts (which live on document) can fire. Gated by _isActiveEditor so
    // multiple editors / the rest of the app are unaffected. Stored on the
    // instance so the entry module can tear it down in onRemoved.
    this._docKeyCapture = (e) => {
      if (!this._isActiveEditor()) return;

      const tagName = (e.target && e.target.tagName) || "";
      const typing = tagName === "INPUT" || tagName === "TEXTAREA" || e.target?.isContentEditable;
      const mod = e.ctrlKey || e.metaKey;
      const k = e.key && e.key.length === 1 ? e.key.toLowerCase() : e.key;

      // Claim a key: stop ComfyUI's global handler AND the native default.
      const claim = () => {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
      };

      // Inside a text field (chip-edit / add-tag input), Ctrl+Z/Y must stay
      // native text-undo — hijacking them globally here would silently
      // discard whatever the user is typing (the global undo forces a
      // render() that tears down the live edit) while also undoing the
      // wrong thing (tag-list history instead of the keystroke). Only
      // outside any text field does Ctrl+Z/Y drive the tag-list history.
      if (!typing && mod && k === "z") { claim(); if (e.shiftKey) this._redo(); else this._undo(); return; }
      if (!typing && mod && k === "y") { claim(); this._redo(); return; }

      // Inside a text field, leave native editing shortcuts (undo/redo/copy/
      // cut/paste/delete/+/-) to the browser.
      if (typing) return;

      if (mod && k === "a") { claim(); this._selectAllInFocus(); return; }
      if (mod && k === "c") { claim(); this._copySelection(); return; }
      if (mod && k === "x") { claim(); this._cutSelection(); return; }
      if (mod && k === "v") {
        // Give native paste an editable target even when a chip/canvas has focus.
        this._prepareNativePaste();
        // Keep the native paste event: it supplies data without requesting
        // permission to read the clipboard through the asynchronous API.
        e.stopPropagation();
        e.stopImmediatePropagation();
        return;
      }
      if (mod && k === "d") { claim(); this._duplicateSelection(); return; }
      if (mod && k === "b") { claim(); this._bypassSelection(); return; }
      if (mod) return; // don't swallow other ctrl/meta combos

      if (e.key === "Escape") {
        if (!this._selection.size) return; // let Escape bubble when nothing selected
        claim();
        this._selection.clear();
        this.render();
        return;
      }
      if (k === "Delete" || k === "Backspace") { claim(); this._deleteSelection(); return; }
      if (k === "+" || k === "=") { claim(); this._weightSelection(0.1); return; }
      if (k === "-" || k === "_") { claim(); this._weightSelection(-0.1); return; }
    };
    document.addEventListener("keydown", this._docKeyCapture, true);
  }

  /** True when this editor should own keyboard input (hover or internal focus). */
  _prepareNativePaste() {
    this._cancelNativePaste?.();
    const previous = document.activeElement;
    const target = document.createElement("textarea");
    target.setAttribute("aria-label", "Paste tags");
    target.style.cssText = "position:fixed;left:-9999px;top:0;width:1px;height:1px;opacity:0";
    this._pasteTarget = target;
    this._root.appendChild(target);
    this._cancelNativePaste = () => {
      clearTimeout(this._pasteTimer);
      this._pasteTarget = null;
      target.remove();
      if (previous?.isConnected) previous.focus?.({ preventScroll: true });
    };
    target.focus({ preventScroll: true });
    this._pasteTimer = setTimeout(this._cancelNativePaste, 1500);
  }

  _isActiveEditor() {
    if (!this._root || !this._root.isConnected) return false;
    const ae = document.activeElement;
    const focusedEditor = ae?.closest?.(".ute-root");
    if (focusedEditor) return focusedEditor === this._root;
    // Workflow switches can leave the remembered pointer-enter flag stale.
    // Only the currently hovered, visible editor may claim unfocused keys.
    return this._root.getClientRects().length > 0 && this._root.matches(":hover");
  }

  /** Selected tags across all sections + characters, in document order. */
  _selectedTagsInOrder() {
    const out = [];
    for (const sec of SECTIONS) {
      for (const t of this._state[sec]) if (this._selection.has(t.id)) out.push(t);
    }
    for (const ch of this._state.characters) {
      for (const t of ch.tags) if (this._selection.has(t.id)) out.push(t);
    }
    return out;
  }

  _copySelection() {
    const tags = this._selectedTagsInOrder();
    if (!tags.length) return;
    // The page-wide buffer preserves per-chip metadata, including bypass, for
    // paste between Tag Editor nodes. The OS clipboard only gets canonical text
    // (flags can't survive it) — that's what pasting into a plain text field
    // outside the widget should produce.
    writeChipClipboard(tags.map((t) => t.type === "divider"
      ? { type: "divider" }
      : { raw: t.raw, flags: tagFlags(t) }));
    copyPlainTextToOsClipboard(tags.filter((t) => t.type !== "divider").map((t) => t.raw).join(", "), readChipClipboard());
  }

  /** Copy the selection to the OS clipboard as plain comma-separated text. */
  _copyAsText() {
    const tags = this._selectedTagsInOrder();
    if (!tags.length) return;
    writeChipClipboard([]);
    copyPlainTextToOsClipboard(tags.filter((t) => t.type !== "divider").map((t) => t.raw).join(", "));
  }

  _cutSelection() {
    if (!this._selection.size) return;
    this._copySelection();
    this._deleteSelection(); // pushes history + re-renders + writes editor_state
  }

  /**
   * Insert tags into the active section, select them, snapshot history.
   * Accepts raw strings, {raw, flags} entries, or structured dividers.
   * Pasted chips keep their metadata and dividers remain dividers.
   */
  _pasteRaws(raws) {
    const tags = raws
      .map((r) => r?.type === "divider"
        ? makeDivider()
        : (typeof r === "string" ? makeTag(r) : makeTag(r.raw, r.flags || {})))
      .filter(Boolean);
    if (!tags.length) return;
    const sec = this._focusSection && this._state[this._focusSection] ? this._focusSection : "mainTags";
    const insertAt = pasteIndex(this._state[sec], this._selection);
    this._state[sec].splice(insertAt, 0, ...tags);
    this._selection.clear();
    for (const t of tags) this._selection.add(t.id);
    this._focusSection = sec;
    this.commit();
    this.render();
  }

  _pasteClipboard(text) {
    const copiedChips = readChipClipboard();
    const copiedText = copiedChips.filter((t) => t.type !== "divider").map((t) => t.raw).join(", ");
    if (copiedChips.length && text === copiedText) {
      this._pasteRaws(copiedChips);
      return;
    }
    const raws = splitTags(text).map((t) => t.raw);
    if (raws.length) this._pasteRaws(raws);
  }

  /** Duplicate each selected tag in place (right after its original). */
  _duplicateSelection() {
    if (!this._selection.size) return;
    const newSel = new Set();
    const dup = (arr) => {
      const result = [];
      for (const t of arr) {
        result.push(t);
        if (this._selection.has(t.id)) {
          const c = t.type === "divider" ? makeDivider() : makeTag(t.raw, tagFlags(t));
          if (c) { result.push(c); newSel.add(c.id); }
        }
      }
      return result;
    };
    for (const sec of SECTIONS) this._state[sec] = dup(this._state[sec]);
    for (const ch of this._state.characters) ch.tags = dup(ch.tags);
    this._selection = newSel;
    this.commit();
    this.render();
  }

  _installPaste() {
    // Browser Edit/Copy and accessibility copy commands can deliver a native
    // clipboard event without a preceding keydown. Preserve chip metadata on
    // that path too, while leaving text-field editing native.
    this._onDocCopy = (e) => {
      if (!this._isActiveEditor() || !e.clipboardData) return;
      const typing = e.target?.tagName === "INPUT" || e.target?.tagName === "TEXTAREA" || e.target?.isContentEditable;
      if (typing) return;
      const tags = this._selectedTagsInOrder();
      if (!tags.length) return;
      const chips = tags.map((t) => t.type === "divider" ? { type: "divider" } : { raw: t.raw, flags: tagFlags(t) });
      writeChipClipboard(chips);
      e.clipboardData.setData("text/plain", tags.filter((t) => t.type !== "divider").map((t) => t.raw).join(", "));
      e.clipboardData.setData(CHIP_CLIPBOARD_MIME, JSON.stringify(chips));
      e.preventDefault();
      e.stopImmediatePropagation();
      if (e.type === "cut") this._deleteSelection();
    };
    document.addEventListener("copy", this._onDocCopy, true);
    document.addEventListener("cut", this._onDocCopy, true);
    // Legit document-level listener (spec-approved for image paste). The entry
    // module removes it in onRemoved via this reference.
    this._onDocPaste = (e) => {
      if (this._isActiveEditor()) {
        const typing = e.target !== this._pasteTarget && (e.target?.tagName === "INPUT" || e.target?.tagName === "TEXTAREA" || e.target?.isContentEditable);
        const text = e.clipboardData?.getData("text/plain");
        const chips = parseChipClipboard(e.clipboardData?.getData(CHIP_CLIPBOARD_MIME));
        const dividerCopy = text === "" && Array.from(e.clipboardData?.types || []).includes("text/plain") && readChipClipboard().some((t) => t.type === "divider");
        if (!typing && (text || chips.length || dividerCopy)) {
          e.preventDefault();
          e.stopPropagation();
          e.stopImmediatePropagation();
          this._cancelNativePaste?.();
          if (chips.length) this._pasteRaws(chips);
          else this._pasteClipboard(text);
          return;
        }
      }
      if (!TAGGER_ENABLED || !this._tagger) return;
      if (this._state.activeTab !== "tags") return;
      if (!this._root.isConnected) return;
      const items = e.clipboardData?.items || [];
      for (const it of items) {
        if (it.type && it.type.startsWith("image/")) {
          this._tagger.handlePaste(it);
          break;
        }
      }
    };
    document.addEventListener("paste", this._onDocPaste, true);
  }
}

function tagFlags(t) {
  const f = {};
  if (t.locked) f.locked = true;
  if (t.color) f.color = t.color;
  if (t.ai_imported) f.ai_imported = true;
  if (t.bypassed) f.bypassed = true;
  return f;
}
