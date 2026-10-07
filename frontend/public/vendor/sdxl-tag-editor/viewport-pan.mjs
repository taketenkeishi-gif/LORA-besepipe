// Forward a bounded pan gesture through LiteGraph's native pointer pipeline.
// Native _processMiddleButton / processMouseMove own threshold, ds.scale and offset.
export function bindViewportPan(root, getCanvas) {
  let space = false, active = null;
  const isText = (el) => el?.closest?.('input,textarea,[contenteditable="true"]');
  const send = (kind, e) => {
    const target = active?.canvas;
    if (!target) return;
    target.dispatchEvent(new PointerEvent(kind, { bubbles: true, cancelable: true,
      pointerId: e.pointerId, pointerType: e.pointerType || "mouse", isPrimary: true,
      clientX: e.clientX, clientY: e.clientY, button: kind === "pointermove" ? -1 : 1,
      buttons: kind === "pointerup" || kind === "pointercancel" ? 0 : 4 }));
  };
  const finish = (e, cancel = false) => {
    if (!active) return;
    e?.stopPropagation?.();
    const previous = active;
    send(cancel ? "pointercancel" : "pointerup", e || previous.last);
    active = null;
    try { root.releasePointerCapture(previous.id); } catch {}
    root.classList.remove("ute-panning");
  };
  const down = (e) => {
    if (active || !(e.button === 1 || (e.button === 0 && (e.target.closest?.(".ute-pan-handle") || space && !isText(e.target))))) return;
    const canvas = getCanvas()?.canvas;
    if (!canvas) return;
    e.preventDefault(); e.stopImmediatePropagation();
    active = { canvas, id: e.pointerId, last: e };
    send("pointerdown", e);
    root.setPointerCapture(e.pointerId);
    root.classList.add("ute-panning");
  };
  const move = (e) => { if (active && e.pointerId === active.id) { e.preventDefault(); e.stopImmediatePropagation(); active.last = e; send("pointermove", e); } };
  const up = (e) => { if (active && e.pointerId === active.id) finish(e); };
  const cancel = (e) => finish(e, true);
  const keydown = (e) => {
    if (e.code === "Space" && root.contains(document.activeElement) && !isText(e.target)) { space = true; e.preventDefault(); }
    if (e.key === "Escape" && active) { e.preventDefault(); e.stopImmediatePropagation(); finish(active.last, true); }
  };
  const keyup = (e) => { if (e.code === "Space") space = false; };
  const blur = () => { space = false; if (active) finish(active.last, true); };
  root.addEventListener("pointerdown", down, true);
  root.addEventListener("pointermove", move, true);
  root.addEventListener("pointerup", up, true);
  root.addEventListener("pointercancel", cancel, true);
  root.addEventListener("lostpointercapture", cancel);
  window.addEventListener("keydown", keydown, true);
  document.addEventListener("keyup", keyup, true);
  window.addEventListener("blur", blur);
  return () => {
    blur();
    root.removeEventListener("pointerdown", down, true); root.removeEventListener("pointermove", move, true);
    root.removeEventListener("pointerup", up, true); root.removeEventListener("pointercancel", cancel, true);
    root.removeEventListener("lostpointercapture", cancel);
    window.removeEventListener("keydown", keydown, true); document.removeEventListener("keyup", keyup, true);
    window.removeEventListener("blur", blur);
  };
}
