let current = null;
export class TreeMenu {
  close(restore = false) {
    this._cleanup?.(); this._cleanup = null;
    this._panels?.forEach((panel) => panel.remove()); this._panels = [];
    if (restore && this._anchor?.isConnected) this._anchor.focus({ preventScroll: true });
    if (current === this) current = null;
  }
  open(x, y, items) {
    current?.close(); this.close(); current = this;
    this._anchor = document.activeElement;
    this._panels = [];
    const trim = (level) => {
      this._panels.splice(level).forEach((panel) => {
        panel._parent?.setAttribute("aria-expanded", "false"); panel.remove();
      });
    };
    const build = (entries, left, top, level = 0, parent = null) => {
      trim(level);
      const panel = document.createElement("div");
      panel.className = "ute-tree-menu"; panel.setAttribute("role", "menu");
      panel.setAttribute("aria-label", parent?.textContent || "タグ操作"); panel._parent = parent;
      panel.style.left = `${left}px`; panel.style.top = `${top}px`;
      this._panels.push(panel);
      const controls = [];
      entries.forEach((item) => {
        if (item.sep) { const line = document.createElement("hr"); line.setAttribute("role", "separator"); panel.append(line); return; }
        if (item.header) { const label = document.createElement("strong"); label.textContent = item.header; panel.append(label); return; }
        const btn = document.createElement("button"); btn.type = "button";
        btn.setAttribute("role", "menuitem"); btn.textContent = `${item.active ? "✓ " : ""}${item.label}${item.children ? "　›" : ""}`;
        btn.disabled = !!item.disabled; btn.className = item.cls || "";
        if (item.color) btn.style.borderLeft = `4px solid ${item.color}`;
        if (item.disabled) btn.title = "現在の選択では使用できません";
        if (item.children) { btn.setAttribute("aria-haspopup", "menu"); btn.setAttribute("aria-expanded", "false"); }
        const openChild = (focus = false) => {
          if (!item.children || btn.disabled) return;
          const rect = btn.getBoundingClientRect();
          const child = build(item.children, panel.getBoundingClientRect().right - 1, rect.top, level + 1, btn);
          btn.setAttribute("aria-expanded", "true");
          if (focus) child.querySelector("button:not(:disabled)")?.focus();
        };
        btn.addEventListener("pointerenter", () => { if (item.children) openChild(); else trim(level + 1); });
        btn.addEventListener("click", (event) => {
          event.stopPropagation();
          if (btn.disabled) return;
          if (item.children) openChild(true);
          else { this.close(true); item.fn?.(); }
        });
        btn.addEventListener("keydown", (event) => {
          const enabled = controls.filter((c) => !c.disabled), index = enabled.indexOf(btn);
          const key = event.key;
          if (["ArrowDown", "ArrowUp", "Home", "End", "ArrowRight", "ArrowLeft", "Escape", "Tab"].includes(key)) { event.preventDefault(); event.stopPropagation(); }
          if (key === "ArrowDown" || key === "ArrowUp") enabled[(index + (key === "ArrowDown" ? 1 : -1) + enabled.length) % enabled.length]?.focus();
          if (key === "Home") enabled[0]?.focus();
          if (key === "End") enabled.at(-1)?.focus();
          if (key === "ArrowRight") openChild(true);
          if (key === "ArrowLeft" && parent) { trim(level); parent.focus(); }
          if (key === "Escape" || key === "Tab") this.close(true);
        });
        controls.push(btn); panel.append(btn);
      });
      panel.addEventListener("pointerdown", (event) => event.stopPropagation());
      document.body.append(panel);
      const rect = panel.getBoundingClientRect();
      if (parent && rect.right > innerWidth - 8) panel.style.left = `${Math.max(8, this._panels[level - 1].getBoundingClientRect().left - rect.width + 1)}px`;
      else panel.style.left = `${Math.max(8, Math.min(left, innerWidth - rect.width - 8))}px`;
      panel.style.top = `${Math.max(8, Math.min(top, innerHeight - rect.height - 8))}px`;
      return panel;
    };
    const panel = build(items, x, y);
    panel.querySelector("button:not(:disabled)")?.focus();
    const outside = (event) => { if (!this._panels.some((p) => p.contains(event.target))) this.close(); };
    document.addEventListener("pointerdown", outside, true);
    this._cleanup = () => document.removeEventListener("pointerdown", outside, true);
  }
}
