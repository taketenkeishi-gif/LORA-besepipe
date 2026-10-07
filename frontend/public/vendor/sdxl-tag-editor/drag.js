// tag-editor/drag.js
//
// Native HTML5 drag-and-drop for reordering chips within a section AND moving
// chips across sections (prefix/main/suffix/blocklist/negative).
//
// Why this feels native and cannot fight ComfyUI's canvas:
//   * Uses the browser's built-in draggable / dragstart / dragover / drop —
//     the OS/browser renders the drag image and tracks the pointer, so there is
//     ZERO document-level pointermove/pointerup tracking and ZERO canvas
//     offset/scale reads. The ComfyUI DomWidget overlay keeps owning position.
//   * We stopPropagation on the chip's pointerdown ONLY, so LiteGraph never
//     starts a node-drag when the gesture begins on a chip — but the widget
//     root and the node title bar remain freely draggable/pannable.
//   * A live drop indicator (before/after the nearest chip, or end-of-zone) is
//     drawn by toggling CSS classes during dragover.
//
// The controller is UI-agnostic: it only reports "move these {section,id} tags
// to dstSection at dstIndex" via the onMove callback supplied by the editor,
// which performs the state mutation + re-render. Dragging a chip that is part
// of a multi-selection moves the WHOLE selection together (not just that chip).

export class DragController {
  /**
   * @param {(payload:Array<{section:string,id:number}>, dstSection:string, dstIndex:number)=>void} onMove
   */
  constructor(onMove) {
    this._onMove = onMove;
    this._payload = null; // Array<{ section, id }>
  }

  /**
   * Make a chip element draggable and the source of a move.
   * `getGroup()` returns the full list of {section,id} to drag together — the
   * whole multi-selection if this chip is part of one, else just this chip.
   */
  bindChip(chipEl, section, id, getGroup) {
    chipEl.draggable = true;

    chipEl.addEventListener("dragstart", (e) => {
      this._payload = getGroup ? getGroup() : [{ section, id }];
      if (!this._payload || !this._payload.length) this._payload = [{ section, id }];
      for (const p of this._payload) {
        document.querySelector(`[data-ute-id="${p.id}"]`)?.classList.add("dragging");
      }
      e.dataTransfer.effectAllowed = "move";
      // Some browsers require data to be set for the drag to start.
      try { e.dataTransfer.setData("text/plain", String(id)); } catch { /* noop */ }
      e.stopPropagation();
    });

    chipEl.addEventListener("dragend", () => {
      for (const el of document.querySelectorAll(".ute-chip.dragging,.ute-divider.dragging")) el.classList.remove("dragging");
      this._payload = null;
      for (const zone of document.querySelectorAll(".ute-chips")) { zone.classList.remove("dropzone-active"); this._clearIndicators(zone); }
    });

    // Prevent LiteGraph from starting a node-drag when the press begins on a chip.
    chipEl.addEventListener("pointerdown", (e) => e.stopPropagation());
  }

  /**
   * Make a chip-container a drop zone for `section`. Chips inside must carry a
   * `data-ute-id` attribute so the insert index can be computed.
   */
  bindZone(zoneEl, section) {
    zoneEl.addEventListener("dragover", (e) => {
      if (!this._payload || !this._payload.length) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = "move";
      zoneEl.classList.add("dropzone-active");
      this._showIndicator(zoneEl, e.clientX, e.clientY);
    });

    zoneEl.addEventListener("dragleave", (e) => {
      // Only clear when actually leaving the zone (not entering a child).
      if (e.relatedTarget && zoneEl.contains(e.relatedTarget)) return;
      zoneEl.classList.remove("dropzone-active");
      this._clearIndicators(zoneEl);
    });

    zoneEl.addEventListener("drop", (e) => {
      if (!this._payload || !this._payload.length) return;
      e.preventDefault();
      e.stopPropagation();
      const idx = this._computeIndex(zoneEl, e.clientX, e.clientY);
      const payload = this._payload;
      zoneEl.classList.remove("dropzone-active");
      this._clearIndicators(zoneEl);
      this._payload = null;
      this._onMove(payload, section, idx);
    });
  }

  // ── indicator + index math ─────────────────────────────────────────────────
  _chips(zoneEl) {
    return [...zoneEl.querySelectorAll("[data-ute-id]")];
  }

  /**
   * Group chips into visual rows (flex-wrap gives every chip in the same row
   * a shared `top`; cluster with a tolerance so sub-pixel jitter doesn't
   * split one row into two).
   */
  _rows(chips) {
    const withRect = chips
      .map((c) => ({ el: c, r: c.getBoundingClientRect() }))
      .sort((a, b) => a.r.top - b.r.top || a.r.left - b.r.left);
    const rows = [];
    for (const item of withRect) {
      const row = rows[rows.length - 1];
      if (row && Math.abs(item.r.top - row.top) < row.height / 2) {
        row.items.push(item);
        row.bottom = Math.max(row.bottom, item.r.bottom);
      } else {
        rows.push({ top: item.r.top, bottom: item.r.bottom, height: item.r.height, items: [item] });
      }
    }
    return rows;
  }

  /**
   * Nearest chip to (x,y), row-aware. Plain Euclidean distance-to-every-chip
   * picks the wrong target when the pointer is in the GAP between two
   * wrapped rows (a chip on the row below can be numerically closer than the
   * intended one at the end of the row above) — instead, first pick the row
   * whose vertical span contains (or is closest to) the pointer, then find
   * the nearest chip by X within just that row.
   */
  _nearest(zoneEl, x, y) {
    const chips = this._chips(zoneEl);
    if (!chips.length) return null;
    const rows = this._rows(chips);

    let bestRow = rows[0];
    let bestRowDist = Infinity;
    for (const row of rows) {
      if (y >= row.top && y <= row.bottom) { bestRow = row; bestRowDist = -1; break; }
      const center = (row.top + row.bottom) / 2;
      const d = Math.abs(y - center);
      if (d < bestRowDist) { bestRowDist = d; bestRow = row; }
    }

    let best = null;
    let bestDist = Infinity;
    for (const item of bestRow.items) {
      const cx = item.r.left + item.r.width / 2;
      const d = Math.abs(x - cx);
      if (d < bestDist) { bestDist = d; best = { el: item.el, after: x > cx }; }
    }
    return best;
  }

  _showIndicator(zoneEl, x, y) {
    this._clearIndicators(zoneEl);
    const near = this._nearest(zoneEl, x, y);
    if (near) near.el.classList.add(near.after ? "drop-after" : "drop-before");
  }

  _clearIndicators(zoneEl) {
    if (!zoneEl) return;
    for (const c of zoneEl.querySelectorAll(".drop-before,.drop-after")) {
      c.classList.remove("drop-before", "drop-after");
    }
  }

  _computeIndex(zoneEl, x, y) {
    const chips = this._chips(zoneEl);
    if (!chips.length) return 0;
    const near = this._nearest(zoneEl, x, y);
    if (!near) return chips.length;
    const base = chips.indexOf(near.el);
    return near.after ? base + 1 : base;
  }
}
