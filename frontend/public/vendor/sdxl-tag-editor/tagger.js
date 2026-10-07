// tag-editor/tagger.js
//
// WD14-style image tagger panel. Talks to the CURRENT backend routes:
//   GET  /unified_tag_editor/status
//   POST /unified_tag_editor/download   (JSON { model })
//   POST /unified_tag_editor/analyze    (multipart: image + settings)
//
// UI: dropzone (drag-drop / click-to-browse / Ctrl+V paste), model select +
// Download, threshold / char-threshold / exclude / replace-underscore controls,
// Analyze / Re-analyze / Clear, and an extracted-tags panel whose chips import
// into Main on click (or all via "Import to Main").

import { h, stop } from "./dom.js";

const BASE = "/api/unified_tag_editor"; // LoRA Studio: proxied to ComfyUI (only local change to the vendored copy)

export class TaggerPanel {
  /**
   * @param {object} opts
   * @param {()=>object} opts.getSettings  returns { model, threshold, charThreshold, excludeTags, replaceUnderscore }
   * @param {(settings:object)=>void} opts.setSettings  persist changed settings
   * @param {(names:string[])=>void} opts.onImport  import tag names into Main
   */
  constructor(opts) {
    this._opts = opts;
    this._imageBlob = null;
    this._results = [];
    this._status = null;
    this.el = this._build();
    this._refreshStatus();
  }

  destroy() {
    if (this._objUrl) URL.revokeObjectURL(this._objUrl);
  }

  _build() {
    const s = this._opts.getSettings();

    this._preview = h("div", { class: "ute-dropzone" }, "Drop / click / paste an image");
    this._preview.addEventListener("click", () => this._file.click());
    this._preview.addEventListener("dragover", (e) => {
      e.preventDefault();
      this._preview.classList.add("over");
    });
    this._preview.addEventListener("dragleave", () => this._preview.classList.remove("over"));
    this._preview.addEventListener("drop", (e) => {
      e.preventDefault();
      this._preview.classList.remove("over");
      const f = e.dataTransfer.files?.[0];
      if (f) this._setImage(f);
    });

    this._file = h("input", { type: "file", accept: "image/*", style: "display:none" });
    this._file.addEventListener("change", () => {
      if (this._file.files?.[0]) this._setImage(this._file.files[0]);
    });

    this._model = h("select", { class: "ute-select" });
    this._model.value = s.model;
    stop(this._model);
    this._model.addEventListener("change", () => this._patch({ model: this._model.value }));

    const dl = h("button", { class: "ute-btn" }, "Download");
    stop(dl);
    dl.addEventListener("click", () => this._download());

    const thr = this._numRow("Threshold", s.threshold, (v) => this._patch({ threshold: v }));
    const cthr = this._numRow("Char thr", s.charThreshold, (v) => this._patch({ charThreshold: v }));

    this._exclude = h("input", { class: "ute-input", placeholder: "exclude (comma)", value: s.excludeTags });
    stop(this._exclude);
    this._exclude.addEventListener("change", () => this._patch({ excludeTags: this._exclude.value }));

    this._repl = h("input", { type: "checkbox" });
    this._repl.checked = !!s.replaceUnderscore;
    this._repl.addEventListener("change", () => this._patch({ replaceUnderscore: this._repl.checked }));

    const analyze = h("button", { class: "ute-btn" }, "Analyze");
    stop(analyze);
    analyze.addEventListener("click", () => this._analyze());
    this._analyzeBtn = analyze;

    const clr = h("button", { class: "ute-btn warn" }, "Clear");
    stop(clr);
    clr.addEventListener("click", () => this._clear());

    const importAll = h("button", { class: "ute-btn" }, "Import to Main");
    stop(importAll);
    importAll.addEventListener("click", () => {
      if (this._results.length) this._opts.onImport(this._results.map((r) => r.name));
    });

    this._statusEl = h("div", { class: "ute-status" }, "…");
    this._tagsEl = h("div", { class: "ute-tagger-tags" });

    return h("div", { class: "ute-group" }, [
      h("div", { class: "ute-group-head" }, "WD14 TAGGER"),
      h("div", { style: "padding:6px 8px; display:flex; flex-direction:column; gap:6px" }, [
        this._preview,
        this._file,
        h("div", { class: "ute-toolbar" }, [this._model, dl]),
        h("div", { class: "ute-toolbar" }, [thr, cthr]),
        this._exclude,
        h("label", { class: "ute-row", style: "padding:0" }, [this._repl, " replace underscores"]),
        h("div", { class: "ute-toolbar" }, [analyze, clr, importAll]),
        this._statusEl,
        this._tagsEl,
      ]),
    ]);
  }

  _numRow(label, value, onChange) {
    const inp = h("input", { class: "ute-num", type: "number", step: "0.05", min: "0", max: "1", value: String(value) });
    stop(inp);
    inp.addEventListener("change", () => {
      const v = parseFloat(inp.value);
      if (!Number.isNaN(v)) onChange(v);
    });
    return h("label", { class: "ute-row", style: "padding:0" }, [label + " ", inp]);
  }

  _patch(delta) {
    const next = { ...this._opts.getSettings(), ...delta };
    this._opts.setSettings(next);
  }

  _setImage(fileOrBlob) {
    this._imageBlob = fileOrBlob;
    if (this._objUrl) URL.revokeObjectURL(this._objUrl);
    this._objUrl = URL.createObjectURL(fileOrBlob);
    this._preview.textContent = "";
    this._preview.appendChild(h("img", { src: this._objUrl }));
  }

  /** Called by the editor when a paste event carries an image while this tab is active. */
  handlePaste(item) {
    const blob = item.getAsFile();
    if (blob) this._setImage(blob);
  }

  hasImage() {
    return !!this._imageBlob;
  }

  async _refreshStatus() {
    try {
      const r = await fetch(`${BASE}/status`);
      const st = await r.json();
      this._status = st;
      this._model.textContent = "";
      for (const name of Object.keys(st.models || {})) {
        const dn = st.models[name].downloaded ? " ✓" : " ⬇";
        this._model.appendChild(h("option", { value: name }, name + dn));
      }
      const cur = this._opts.getSettings().model;
      if (st.models && st.models[cur]) this._model.value = cur;
      const deps = [];
      if (!st.pil_ok) deps.push("Pillow");
      if (!st.np_ok) deps.push("numpy");
      if (!st.ort_ok) deps.push("onnxruntime");
      this._statusEl.textContent = deps.length ? `Missing: ${deps.join(", ")}` : "Backend ready.";
    } catch (e) {
      this._statusEl.textContent = "Tagger backend unreachable.";
    }
  }

  async _download() {
    const model = this._model.value;
    this._statusEl.textContent = `Downloading ${model}…`;
    try {
      const r = await fetch(`${BASE}/download`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model }),
      });
      const d = await r.json();
      this._statusEl.textContent = d.error ? `Error: ${d.error}` : `Downloaded ${model}.`;
      this._refreshStatus();
    } catch (e) {
      this._statusEl.textContent = "Download failed.";
    }
  }

  async _analyze() {
    if (!this._imageBlob) {
      this._statusEl.textContent = "No image selected.";
      return;
    }
    const s = this._opts.getSettings();
    const fd = new FormData();
    fd.append("image", this._imageBlob, "image.png");
    fd.append("model", s.model);
    fd.append("threshold", String(s.threshold));
    fd.append("char_threshold", String(s.charThreshold));
    fd.append("exclude_tags", s.excludeTags || "");
    fd.append("replace_underscore", s.replaceUnderscore ? "true" : "false");

    this._analyzeBtn.disabled = true;
    this._statusEl.textContent = "Analyzing…";
    try {
      const r = await fetch(`${BASE}/analyze`, { method: "POST", body: fd });
      const d = await r.json();
      if (d.error) {
        this._statusEl.textContent = d.need_download ? `Model not downloaded — click Download.` : `Error: ${d.error}`;
        return;
      }
      this._results = d.tags || [];
      this._statusEl.textContent = `${this._results.length} tags.`;
      this._analyzeBtn.textContent = "Re-analyze";
      this._renderResults();
    } catch (e) {
      this._statusEl.textContent = "Analyze failed.";
    } finally {
      this._analyzeBtn.disabled = false;
    }
  }

  _renderResults() {
    this._tagsEl.textContent = "";
    for (const t of this._results) {
      const chip = h("span", { class: "ute-tt", title: `${t.confidence}` }, [
        t.name,
        h("span", { class: "c" }, String(t.confidence)),
      ]);
      chip.addEventListener("click", () => this._opts.onImport([t.name]));
      this._tagsEl.appendChild(chip);
    }
  }

  _clear() {
    this._imageBlob = null;
    this._results = [];
    if (this._objUrl) URL.revokeObjectURL(this._objUrl);
    this._objUrl = null;
    this._preview.textContent = "Drop / click / paste an image";
    this._tagsEl.textContent = "";
    this._analyzeBtn.textContent = "Analyze";
    this._statusEl.textContent = "Cleared.";
  }
}
