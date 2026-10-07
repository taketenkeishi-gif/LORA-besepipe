// tag-editor/characters.js
//
// Pure state operations for the Characters tab. A "character" is a named bucket
// of tags: { id, name, tags:[tag] }. These helpers mutate the passed-in state
// object; the editor calls commit()+render() afterwards.
//
// NOTE (simplified vs legacy): the legacy file contained elaborate
// multi-character phrase-synthesis / anti-mix heuristics (position words, verb
// inference, per-person category regexes). Those brittle heuristics are
// intentionally NOT reproduced. This clean version keeps the reliable core:
// characters as named tag buckets, with "return tags to Main" on removal. The
// synthesis-to-prompt step is left to Main composition, which is deterministic.

import { cryptoId } from "./state.js";
import { makeTag } from "./parse.js";

export function addCharacter(state, name = "Character") {
  const ch = { id: cryptoId(), name, tags: [] };
  state.characters.push(ch);
  return ch;
}

/** Remove a character; return its tags to Main (deduped by the caller). */
export function removeCharacter(state, id) {
  const idx = state.characters.findIndex((c) => c.id === id);
  if (idx < 0) return;
  const [ch] = state.characters.splice(idx, 1);
  for (const t of ch.tags) {
    state.mainTags.push(makeTag(t.raw, tagFlags(t)));
  }
}

export function renameCharacter(state, id, name) {
  const ch = state.characters.find((c) => c.id === id);
  if (ch) ch.name = name;
}

export function addTagsToCharacter(state, id, rawList) {
  const ch = state.characters.find((c) => c.id === id);
  if (!ch) return;
  for (const raw of rawList) {
    const t = makeTag(raw);
    if (t) ch.tags.push(t);
  }
}

/** Remove one tag from a character and push it back to Main. */
export function returnTagToMain(state, charId, tagId) {
  const ch = state.characters.find((c) => c.id === charId);
  if (!ch) return;
  const idx = ch.tags.findIndex((t) => t.id === tagId);
  if (idx < 0) return;
  const [t] = ch.tags.splice(idx, 1);
  state.mainTags.push(makeTag(t.raw, tagFlags(t)));
}

function tagFlags(t) {
  const f = {};
  if (t.locked) f.locked = true;
  if (t.color) f.color = t.color;
  if (t.ai_imported) f.ai_imported = true;
  if (t.bypassed) f.bypassed = true;
  return f;
}
