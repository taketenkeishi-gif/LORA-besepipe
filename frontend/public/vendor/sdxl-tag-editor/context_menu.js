import { TreeMenu } from "./tree-menu.mjs?v=20260915-ui-refinement-2";
// tag-editor/context_menu.js
//
// Floating right-click context menu + weight popover, ported from the original
// widget (sdxl_tag_editor `_showContextMenu` / `_showWeightPopover` /
// `_appendColorMenu`). These are the "fine adjustments" the rebuild had dropped.
//
// UI-agnostic: the editor passes an item list; this module only renders the
// floating element, positions it, and tears it down on outside click. Nothing
// here touches ComfyUI's canvas — the menu is a fixed-position child of
// document.body (z-index above the graph) dismissed by a one-shot mousedown.
//
// Item shape (array, in order):
//   { sep: true }                              → horizontal rule
//   { header: "Color label" }                  → non-clickable section label
//   { label, fn, disabled?, cls?, color?, active? }  → clickable row
//
// COLOR_MAP is exported so the editor and the swatch rows stay in sync.

const MENU_ID = "ute-active-ctx";
const POP_ID = "ute-active-popover";

export const COLOR_MAP = [
  [null, "○ 標準", "#555"],
  ["character", "● キャラクター", "#6677ee"],
  ["style", "● 画風", "#44aa66"],
  ["outfit", "● 衣装", "#cc7733"],
  ["important", "● 重要", "#ccaa00"],
  ["warning", "● 注意", "#cc4444"],
];

export class FloatingMenu extends TreeMenu {}

/**
 * Weight popover anchored under a chip rect. onApply(number) is called on Enter
 * or "設定"; the popover self-dismisses. `rect` must be captured BEFORE any
 * re-render (a detached element reports {0,0}).
 */
export function showWeightPopover(rect, initial, onApply) {
  closeWeightPopover();
  const p = document.createElement("div");
  p.className = "ute-popover";
  p.id = POP_ID;

  const PW = 220, PH = 36;
  let left = rect.left;
  let top = rect.bottom + 5;
  if (left + PW > window.innerWidth - 8) left = window.innerWidth - PW - 8;
  if (top + PH > window.innerHeight - 8) top = rect.top - PH - 5;
  if (left < 4) left = 4;
  p.style.left = `${left}px`;
  p.style.top = `${top}px`;

  const lbl = document.createElement("span");
  lbl.textContent = "重み:";

  const inp = document.createElement("input");
  inp.type = "number";
  inp.step = "0.1";
  inp.min = "0.1";
  inp.value = String(initial ?? 1.0);

  const apply = () => {
    const w = parseFloat(inp.value);
    closeWeightPopover();
    if (!Number.isNaN(w)) onApply(w);
  };
  inp.addEventListener("pointerdown", (e) => e.stopPropagation());
  inp.addEventListener("keydown", (e) => {
    e.stopPropagation();
    if (e.key === "Enter") { e.preventDefault(); apply(); }
    if (e.key === "Escape") { e.preventDefault(); closeWeightPopover(); }
  });

  const ok = document.createElement("button");
  ok.className = "ute-btn";
  ok.textContent = "設定";
  ok.addEventListener("click", (e) => { e.stopPropagation(); apply(); });

  const cl = document.createElement("button");
  cl.className = "ute-btn";
  cl.textContent = "✕";
  cl.addEventListener("click", (e) => { e.stopPropagation(); closeWeightPopover(); });

  p.append(lbl, inp, ok, cl);
  p.addEventListener("pointerdown", (e) => e.stopPropagation());
  document.body.appendChild(p);
  requestAnimationFrame(() => { inp.focus(); inp.select(); });

  setTimeout(() => {
    const onDown = (ev) => {
      if (!p.contains(ev.target)) closeWeightPopover();
    };
    document.addEventListener("mousedown", onDown, { once: true, capture: true });
  }, 60);
}

export function closeWeightPopover() {
  document.getElementById(POP_ID)?.remove();
}
