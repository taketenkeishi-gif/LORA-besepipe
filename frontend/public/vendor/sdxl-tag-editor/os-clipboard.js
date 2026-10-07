// tag-editor/os-clipboard.js
//
// Best-effort plain-text write to the REAL OS clipboard, so copying tag chips
// (Ctrl+C / right-click Copy) and pasting into an ordinary text field outside
// the widget (another app, a ComfyUI text node, etc.) yields the expected
// comma-separated tag text.
//
// `navigator.clipboard.writeText()` alone is not reliable here: inside a
// DOM widget deep in LiteGraph's canvas UI, the async Clipboard API can
// silently reject (missing focus/permission in this embedding context) and
// the failure was previously swallowed, so nothing ever reached the OS
// clipboard. We now fall back to the classic hidden-textarea + execCommand
// trick, which works synchronously off any user gesture regardless of the
// async API's availability.

import { CHIP_CLIPBOARD_MIME } from "./chip-clipboard.mjs?v=20260915-native-clipboard";

function execCommandFallback(text, chips) {
  const previousFocus = document.activeElement;
  let ta;
  const onCopy = (event) => {
    if (!event.clipboardData) return;
    event.clipboardData.setData("text/plain", text);
    if (chips) event.clipboardData.setData(CHIP_CLIPBOARD_MIME, JSON.stringify(chips));
    event.preventDefault();
    event.stopImmediatePropagation();
  };
  try {
    ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.top = "-9999px";
    ta.style.left = "-9999px";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    document.addEventListener("copy", onCopy, true);
    return document.execCommand("copy");
  } catch { return false; }
  finally {
    document.removeEventListener("copy", onCopy, true);
    ta?.remove();
    previousFocus?.focus?.({ preventScroll: true });
  }
}

/** Write `text` to the OS clipboard, trying the async API then falling back. */
export function copyPlainTextToOsClipboard(text, chips) {
  // User-initiated copy events can carry metadata without clipboard-read permission.
  if (chips && execCommandFallback(text, chips)) return;
  try {
    const p = navigator.clipboard?.writeText?.(text);
    if (p && typeof p.then === "function") {
      p.catch(() => execCommandFallback(text));
      return;
    }
  } catch { /* fall through to the sync fallback below */ }
  execCommandFallback(text);
}
