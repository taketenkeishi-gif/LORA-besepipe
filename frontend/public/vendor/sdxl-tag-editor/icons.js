// tag-editor/icons.js
//
// Small inline SVG icon set for the toolbar. Built purely with DOM APIs (no
// innerHTML/HTML strings) — every shape is its own SVG element node, so there
// is no markup-injection surface at all, even though these are static,
// developer-authored icons with no user-controlled input.

const SVG_NS = "http://www.w3.org/2000/svg";

function shape(tag, attrs) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

/** Build a 16x16 icon from a list of {tag, ...attrs} shape descriptors. */
function icon(shapes, viewBox = "0 0 20 20") {
  const svg = shape("svg", { viewBox, width: "15", height: "15", class: "ute-icon" });
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "1.7");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  for (const s of shapes) svg.appendChild(shape(s.tag, s));
  return svg;
}

const ICONS = {
  undo: () => icon([
    { tag: "path", d: "M6 5 3 8l3 3" },
    { tag: "path", d: "M3 8h9a4.5 4.5 0 1 1-3.5 7.3" },
  ]),
  redo: () => icon([
    { tag: "path", d: "M14 5l3 3-3 3" },
    { tag: "path", d: "M17 8H8a4.5 4.5 0 1 0 3.5 7.3" },
  ]),
  weightMinus: () => icon([
    { tag: "circle", cx: "10", cy: "10", r: "7" },
    { tag: "path", d: "M6.5 10h7" },
  ]),
  weightPlus: () => icon([
    { tag: "circle", cx: "10", cy: "10", r: "7" },
    { tag: "path", d: "M6.5 10h7" },
    { tag: "path", d: "M10 6.5v7" },
  ]),
  emphasis: () => icon([
    { tag: "path", d: "M8 3.5C6 6 6 14 8 16.5" },
    { tag: "path", d: "M12 3.5c2 2.5 2 10.5 0 13" },
  ]),
  trash: () => icon([
    { tag: "path", d: "M4 6h12" },
    { tag: "path", d: "M8 6V4.5A1.5 1.5 0 0 1 9.5 3h1A1.5 1.5 0 0 1 12 4.5V6" },
    { tag: "path", d: "M5.5 6l.7 9.5A1.5 1.5 0 0 0 7.7 17h4.6a1.5 1.5 0 0 0 1.5-1.5L14.5 6" },
    { tag: "path", d: "M8.5 9v5" },
    { tag: "path", d: "M11.5 9v5" },
  ]),
  deselect: () => icon([
    { tag: "rect", x: "3.5", y: "3.5", width: "13", height: "13", rx: "2", "stroke-dasharray": "3 2.2" },
    { tag: "path", d: "M7.5 7.5l5 5" },
    { tag: "path", d: "M12.5 7.5l-5 5" },
  ]),
  dedupe: () => icon([
    { tag: "rect", x: "4", y: "4", width: "9", height: "9", rx: "1.5" },
    { tag: "path", d: "M9 13v1.5A1.5 1.5 0 0 0 10.5 16H15a1.5 1.5 0 0 0 1.5-1.5V9A1.5 1.5 0 0 0 15 7.5h-1.5" },
    { tag: "path", d: "M6 8.2l1.4 1.4L10 7" },
  ]),
  lora: () => icon([
    { tag: "path", d: "M7.2 4.5h2a1 1 0 0 1 2 0h1.6a1 1 0 0 1 1 1v1.8a1 1 0 0 0 0 2v1.6a1 1 0 0 1-1 1h-1.6a1 1 0 0 0-2 0H7.2a1 1 0 0 1-1-1v-1.6a1 1 0 0 0 0-2V5.5a1 1 0 0 1 1-1z" },
  ]),
  importAi: () => icon([
    { tag: "path", d: "M10 3v9" },
    { tag: "path", d: "M6.5 8.5 10 12l3.5-3.5" },
    { tag: "path", d: "M4 14v1.5A1.5 1.5 0 0 0 5.5 17h9a1.5 1.5 0 0 0 1.5-1.5V14" },
  ]),
};

/** Return an icon element by name, or null if unknown. */
export function getIcon(name) {
  return ICONS[name] ? ICONS[name]() : null;
}
