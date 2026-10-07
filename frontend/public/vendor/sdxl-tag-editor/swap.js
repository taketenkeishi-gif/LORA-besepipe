// tag-editor/swap.js
//
// Alt+Click tag "swap / cycle" support, ported from the original widget
// (sdxl_tag_editor `_SLOT_GROUPS` / `_cycleSlot` / `_SwapDB`).
//
// Two priority tiers decide the "next" replacement for a tag:
//   1. A user-defined swap ring persisted in localStorage ("ute_te_swaps"),
//      built via the right-click "Swap ›" submenu.
//   2. A hardcoded set of built-in semantic rings (hair colours, expressions…).
//
// Keys are normalised (lowercase, underscores→spaces, trimmed) so "long_hair"
// and "long hair" collapse to the same ring member.

const _SLOT_GROUPS = [
  ["open_mouth", "closed_mouth", "parted_lips", "slight_smile"],
  ["smile", "grin", "expressionless", "serious", "frown", "pout", "wink"],
  ["1girl", "1boy", "2girls", "2boys", "multiple_girls"],
  ["solo", "duo"],
  ["long_hair", "short_hair", "medium_hair", "very_long_hair"],
  ["ponytail", "twin_tails", "braid", "side_ponytail", "low_ponytail"],
  ["blonde_hair", "brown_hair", "black_hair", "white_hair", "pink_hair",
   "blue_hair", "red_hair", "silver_hair", "purple_hair", "orange_hair"],
  ["blue_eyes", "brown_eyes", "green_eyes", "red_eyes", "purple_eyes", "golden_eyes"],
  ["large_breasts", "medium_breasts", "small_breasts", "flat_chest", "huge_breasts"],
  ["standing", "sitting", "lying", "kneeling", "leaning_forward"],
  ["looking_at_viewer", "looking_away", "looking_to_the_side", "looking_up", "looking_down"],
  ["outdoors", "indoors", "simple_background", "white_background", "black_background"],
  ["day", "night", "sunset", "sunrise"],
  ["school_uniform", "sailor_uniform", "swimsuit", "bikini", "dress", "casual"],
  ["cowboy_shot", "upper_body", "full_body", "close-up", "from_above", "from_below"],
];

/** Next member of a built-in slot ring for `raw`, or null. Spaces in output. */
function cycleSlot(raw) {
  try {
    const key = (raw || "").trim().replace(/ /g, "_").toLowerCase();
    for (const g of _SLOT_GROUPS) {
      const i = g.findIndex((s) => s === key || s.replace(/_/g, " ") === key.replace(/_/g, " "));
      if (i >= 0) return g[(i + 1) % g.length].replace(/_/g, " ");
    }
  } catch { /* ignore */ }
  return null;
}

/** User-defined swap rings, persisted in localStorage. */
export const SwapDB = (() => {
  const LS_KEY = "ute_te_swaps";
  let _d = null;
  const _load = () => {
    if (_d) return _d;
    try { _d = JSON.parse(localStorage.getItem(LS_KEY) || "{}"); } catch { _d = {}; }
    return _d;
  };
  const _flush = () => { try { localStorage.setItem(LS_KEY, JSON.stringify(_d)); } catch { /* quota / disabled */ } };
  const _norm = (s) => (s || "").toLowerCase().replace(/_/g, " ").trim();

  const getGroup = (text) => _load()[_norm(text)]?.group ?? null;

  const addSwap = (text, target) => {
    const d = _load();
    const ka = _norm(text), kb = _norm(target);
    if (!ka || !kb || ka === kb) return;
    const group = d[ka]?.group ? [...d[ka].group] : [ka];
    if (!group.includes(kb)) group.push(kb);
    for (const k of group) d[k] = { group };
    _flush();
  };

  const clearSwaps = (text) => {
    const d = _load();
    const k = _norm(text);
    const group = d[k]?.group ?? [k];
    for (const m of group) delete d[m];
    _flush();
  };

  const nextSwap = (text) => {
    const group = getGroup(text);
    if (!group || group.length < 2) return null;
    const k = _norm(text);
    const i = group.findIndex((s) => s === k);
    return group[(i + 1) % group.length] ?? null;
  };

  return { getGroup, addSwap, clearSwaps, nextSwap, _norm };
})();

/**
 * Resolve the next replacement text for a tag: custom ring first, then built-in
 * slot groups. Returns null if the tag belongs to no ring.
 */
export function nextSwapText(raw) {
  return SwapDB.nextSwap(raw) ?? cycleSlot(raw);
}
