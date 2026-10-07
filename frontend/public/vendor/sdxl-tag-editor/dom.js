// tag-editor/dom.js
//
// Minimal DOM construction helper + a stopPropagation binder.

/**
 * Create an element. `attrs` keys: class, style, plus any attribute/property.
 * `children` may be a string, Node, or array of those.
 */
export function h(tag, attrs = {}, children = null) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style") el.style.cssText = v;
    else if (k === "value") el.value = v;
    else if (k === "checked") el.checked = v;
    else if (k in el) {
      try { el[k] = v; } catch { el.setAttribute(k, v); }
    } else el.setAttribute(k, v);
  }
  append(el, children);
  return el;
}

export function append(el, children) {
  if (children == null) return;
  if (Array.isArray(children)) {
    for (const c of children) append(el, c);
  } else if (children instanceof Node) {
    el.appendChild(children);
  } else {
    el.appendChild(document.createTextNode(String(children)));
  }
}

/**
 * Stop LiteGraph from hijacking typing/clicks on an interactive control by
 * swallowing pointerdown (and keydown for text inputs) at the element level.
 * Applied to CONTROLS only — never to the widget root — so node-drag via the
 * title bar and canvas panning keep working.
 */
export function stop(el, keys = true) {
  el.addEventListener("pointerdown", (e) => e.stopPropagation());
  if (keys) el.addEventListener("keydown", (e) => e.stopPropagation());
}
