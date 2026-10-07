/** Return the insertion index immediately after the last selected chip. */
export function pasteIndex(tags, selectedIds) {
  let index = tags.length;
  for (let i = 0; i < tags.length; i++) {
    if (selectedIds.has(tags[i].id)) index = i + 1;
  }
  return index;
}
