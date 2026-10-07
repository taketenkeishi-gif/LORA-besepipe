/** Remove only selected, bypassed tag chips while preserving all other entries. */
export function removeSelectedBypassed(tags, selectedIds) {
  const removedIds = [];
  const remaining = tags.filter((tag) => {
    const remove = tag.type !== "divider" && tag.bypassed && selectedIds.has(tag.id);
    if (remove) removedIds.push(tag.id);
    return !remove;
  });
  return { remaining: removedIds.length ? remaining : tags, removedIds };
}
