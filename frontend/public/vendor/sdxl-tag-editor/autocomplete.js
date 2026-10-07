// tag-editor/autocomplete.js
//
// Tag autocomplete for the add-tag inputs, ported from the original widget
// (sdxl_tag_editor `_SEED_TAGS` / `_makeTagHistory` / `_attachAutocomplete`).
//
// Suggestions come from two merged, offline sources — no network fetch:
//   1. A bundled seed list of common booru tags (_SEED_TAGS).
//   2. Per-user usage counts persisted in localStorage, so tags you actually
//      use float to the top. Positive and negative areas keep separate history.
//
// `@name` searches saved presets (localStorage) and expands them into the input.
// Keyboard: ↑/↓ move, Tab cycles, Enter commits the highlighted item, Esc closes.
// The keydown listener is capture-phase so it beats the input's own Enter=add
// handler (via stopImmediatePropagation) and never double-adds a tag.

const _SEED_TAGS = [
  // quality
  "masterpiece", "best_quality", "high_quality", "ultra_detailed", "highly_detailed",
  "absurdres", "highres", "8k", "official_art", "detailed_background",
  // count / subject
  "1girl", "2girls", "1boy", "multiple_girls", "solo", "duo", "no_humans",
  // hair length / style
  "long_hair", "short_hair", "medium_hair", "very_long_hair", "ponytail",
  "twin_tails", "braid", "side_ponytail", "bangs", "hair_between_eyes",
  // hair colour
  "blonde_hair", "brown_hair", "black_hair", "white_hair", "pink_hair",
  "blue_hair", "red_hair", "silver_hair", "purple_hair", "orange_hair", "green_hair",
  // eyes
  "blue_eyes", "brown_eyes", "green_eyes", "red_eyes", "purple_eyes", "golden_eyes",
  "yellow_eyes", "heterochromia", "closed_eyes",
  // expression
  "smile", "grin", "expressionless", "serious", "frown", "pout", "wink",
  "open_mouth", "closed_mouth", "parted_lips", "blush", "tears",
  // body
  "large_breasts", "medium_breasts", "small_breasts", "flat_chest", "huge_breasts",
  // clothing
  "school_uniform", "sailor_uniform", "swimsuit", "bikini", "dress", "casual",
  "hoodie", "jacket", "skirt", "thighhighs", "gloves", "necktie",
  // pose
  "standing", "sitting", "lying", "kneeling", "leaning_forward", "arms_up",
  "hand_on_hip", "crossed_arms",
  // looking
  "looking_at_viewer", "looking_away", "looking_to_the_side", "looking_up", "looking_down",
  // framing
  "cowboy_shot", "upper_body", "full_body", "close-up", "portrait", "from_above", "from_below",
  // background / setting
  "outdoors", "indoors", "simple_background", "white_background", "black_background",
  "gradient_background", "scenery", "cityscape",
  // lighting / time
  "day", "night", "sunset", "sunrise", "backlighting", "dramatic_lighting", "rim_lighting",
  // art style
  "anime_style", "realistic", "sketch", "watercolor", "flat_color", "lineart",
  // misc
  "solo_focus", "cat_ears", "cat_tail", "fox_ears", "fox_tail", "elf_ears",
  "horns", "wings", "tail", "hair_ornament", "hair_ribbon", "hair_bow",
  "glasses", "sunglasses", "hat", "detailed_face", "detailed_eyes", "perfect_face",
  "bokeh", "depth_of_field", "film_grain", "lens_flare", "motion_blur",
];

/** Factory for a localStorage-backed usage-history + candidate matcher. */
export function makeTagHistory(lsKey, seedTags = []) {
  let _d = null;
  const _load = () => {
    if (_d) return _d;
    try { _d = JSON.parse(localStorage.getItem(lsKey) || "{}"); } catch { _d = {}; }
    return _d;
  };
  const bump = (name) => {
    if (!name || !name.trim()) return;
    const k = name.trim().replace(/ /g, "_");
    const d = _load();
    d[k] = { n: ((d[k]?.n) | 0) + 1, t: Date.now() };
    try { localStorage.setItem(lsKey, JSON.stringify(d)); } catch { /* quota / disabled */ }
  };
  const getCandidates = (prefix, limit = 8) => {
    if (!prefix) return [];
    const p = prefix.toLowerCase().replace(/ /g, "_");
    if (p.length < 1) return [];
    const d = _load();
    const seen = new Set();
    const matches = [];
    const pn = p.replace(/_/g, " ");
    const tryAdd = (k) => {
      if (seen.has(k)) return;
      seen.add(k);
      const kn = k.replace(/_/g, " ");
      if (!k.startsWith(p) && !kn.startsWith(pn)) return;
      const hh = d[k] || null;
      matches.push({ name: kn, n: hh?.n || 0, t: hh?.t || 0 });
    };
    for (const s of seedTags) tryAdd(s.replace(/ /g, "_"));
    for (const k of Object.keys(d)) tryAdd(k);
    matches.sort((a, b) => b.n - a.n || b.t - a.t || a.name.localeCompare(b.name));
    return matches.slice(0, limit);
  };
  const getRecent = (limit = 10) => {
    const d = _load();
    return Object.entries(d)
      .map(([k, v]) => ({ name: k.replace(/_/g, " "), n: v.n || 0, t: v.t || 0 }))
      .filter((e) => e.t > 0)
      .sort((a, b) => b.t - a.t)
      .slice(0, limit);
  };
  return { bump, getCandidates, getRecent };
}

const _TagHistory = makeTagHistory("ute_te_hist", _SEED_TAGS);
const _TagHistoryNeg = makeTagHistory("ute_te_hist_neg", []);

/** Read-only preset accessor for @-command expansion. */
const TagPresets = (() => {
  const LS_KEY = "ute_te_presets";
  const _load = () => {
    try { return JSON.parse(localStorage.getItem(LS_KEY) || "{}"); } catch { return {}; }
  };
  const list = () => {
    const d = _load();
    return Object.keys(d).sort((a, b) => (d[b]?.usage || 0) - (d[a]?.usage || 0));
  };
  const info = (name) => _load()[name] || null;
  const get = (name) => _load()[name]?.tags || null;
  return { list, info, get };
})();

/** Bump usage history for a raw tag string (call when a tag is added). */
export function bumpHistory(raw, area) {
  const hist = area === "negative" ? _TagHistoryNeg : _TagHistory;
  hist.bump(raw);
}

/**
 * Attach autocomplete to an add-tag input. `areaKey === "negative"` uses the
 * separate negative history. Non-destructive: only rewrites the input value;
 * the input's own Enter handler still performs the actual add.
 */
export function attachAutocomplete(inp, areaKey) {
  let _el = null;
  let _sel = -1;
  let _cands = [];
  let _hdr = "";

  const close = () => {
    _el?.remove(); _el = null; _sel = -1; _cands = []; _hdr = "";
    inp._acOpen = false;
  };

  const render = (hdrLabel) => {
    if (hdrLabel !== undefined) _hdr = hdrLabel;
    _el?.remove();
    if (!_cands.length) { inp._acOpen = false; return; }
    inp._acOpen = true;
    const drop = document.createElement("div");
    drop.className = "ute-autocomplete";
    try {
      const r = inp.getBoundingClientRect();
      drop.style.left = `${r.left}px`;
      drop.style.top = `${r.bottom + 2}px`;
      drop.style.minWidth = `${Math.max(r.width, 160)}px`;
    } catch { /* detached */ }
    if (_hdr) {
      const hd = document.createElement("div");
      hd.className = "ute-ac-hdr";
      hd.textContent = _hdr;
      drop.appendChild(hd);
    }
    _cands.forEach((c, i) => {
      const item = document.createElement("div");
      item.className = "ute-ac-item" + (i === _sel ? " ac-sel" : "");
      const nm = document.createElement("span");
      nm.textContent = c.name;
      item.appendChild(nm);
      if (c.n > 0) {
        const us = document.createElement("span");
        us.className = "ute-ac-usage";
        us.textContent = c.isPreset ? `${c.n} tags` : `×${c.n}`;
        item.appendChild(us);
      }
      item.addEventListener("mousedown", (ev) => { ev.preventDefault(); commit(c); });
      drop.appendChild(item);
    });
    document.body.appendChild(drop);
    _el = drop;
  };

  const commit = (cand) => {
    try {
      const c = typeof cand === "string" ? { name: cand, isPreset: false } : cand;
      const v = inp.value;
      const lc = v.lastIndexOf(",");
      const base = lc >= 0 ? v.slice(0, lc + 1) + " " : "";
      if (c.isPreset) {
        const raws = TagPresets.get(c.presetName);
        inp.value = raws && raws.length ? base + raws.join(", ") + ", " : base;
      } else {
        inp.value = base + c.name + ", ";
      }
      close();
      inp.focus();
      inp.setSelectionRange(inp.value.length, inp.value.length);
    } catch { close(); }
  };

  const hist = areaKey === "negative" ? _TagHistoryNeg : _TagHistory;

  const showRecent = () => {
    _cands = hist.getRecent(8).map((c) => ({ ...c, isPreset: false }));
    _sel = -1;
    render("recently used");
  };

  inp.addEventListener("focus", () => {
    if (!inp.value.trim()) showRecent();
  });

  inp.addEventListener("input", () => {
    try {
      const v = inp.value;
      const lc = v.lastIndexOf(",");
      const rawToken = (lc >= 0 ? v.slice(lc + 1) : v).trim();
      if (!rawToken) { showRecent(); return; }

      if (rawToken.startsWith("@")) {
        const q = rawToken.slice(1).toLowerCase();
        const names = TagPresets.list().filter((n) => !q || n.toLowerCase().includes(q));
        _cands = names.slice(0, 8).map((n) => {
          const inf = TagPresets.info(n);
          return { name: "@" + n, isPreset: true, presetName: n, n: inf?.tags?.length || 0, t: inf?.lastUsed || 0 };
        });
        _sel = -1; render("presets"); return;
      }

      const token = rawToken.replace(/ /g, "_");
      _cands = hist.getCandidates(token.toLowerCase()).map((c) => ({ ...c, isPreset: false }));
      _sel = -1; render("suggestions");
    } catch { close(); }
  });

  inp.addEventListener("keydown", (ev) => {
    if (!inp._acOpen) return;
    if (ev.key === "ArrowDown") {
      ev.preventDefault(); _sel = Math.min(_sel + 1, _cands.length - 1); render();
    } else if (ev.key === "ArrowUp") {
      ev.preventDefault(); _sel = Math.max(_sel - 1, -1); render();
    } else if (ev.key === "Tab") {
      ev.preventDefault(); ev.stopImmediatePropagation();
      _sel = _cands.length ? (_sel + 1) % _cands.length : -1; render();
    } else if (ev.key === "Enter" && _sel >= 0) {
      ev.preventDefault(); ev.stopImmediatePropagation(); commit(_cands[_sel]);
    } else if (ev.key === "Escape") {
      close();
    }
  }, true);

  inp.addEventListener("blur", () => setTimeout(close, 150));
}
