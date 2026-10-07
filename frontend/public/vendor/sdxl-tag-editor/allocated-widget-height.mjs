/**
 * LiteGraph's node.computeSize measures an intrinsic minimum, whereas the DOM
 * overlay asks widget.computeSize for the allocated display height. Returning
 * node.size during the former feeds the existing allocation back into the minimum.
 * Keep those contexts separate; never change the user's node.size here.
 */
export function installAllocatedWidgetHeight(node, widget, { minH = 40, chrome = 12, onHeight } = {}) {
  const measureNode = node.computeSize;
  let minimumDepth = 0;
  node.computeSize = function (...args) {
    minimumDepth++;
    try { return measureNode.apply(this, args); }
    finally { minimumDepth--; }
  };
  widget.computeSize = function (width) {
    if (minimumDepth || !Number.isFinite(this.last_y)) return [width, minH];
    const nodeHeight = Number(node.size?.[1]);
    const height = Number.isFinite(nodeHeight) ? Math.max(minH, nodeHeight - this.last_y - chrome) : minH;
    onHeight?.(height);
    return [width, height];
  };
}
