// A view/controller for an existing loader setting. Never patches CLIP itself.
export function resolveClipLoader(host) {
  if (!host?.graph) return { error: "CLIPの接続元を確認できません。" };
  const input = host.inputs?.find((item) => item.name === "clip" && item.type === "CLIP");
  if (input?.link == null) return { error: "CLIP入力が未接続です。モデルスタックのCLIP出力を接続してください。" };
  const graph = host.graph;
  const link = graph.links?.get?.(input.link) ?? graph.links?.[input.link];
  const loader = link && graph.getNodeById?.(link.origin_id);
  if (!loader || loader.outputs?.[link.origin_slot]?.type !== "CLIP") {
    return { error: "CLIP接続を解決できません。接続を確認してください。" };
  }
  if (loader.type !== "UnifiedModelStackLoader") {
    return { error: `接続元「${loader.title || loader.type}」のCLIP強度操作には未対応です。上流ローダーで設定してください。` };
  }
  if (loader.mode != null && loader.mode !== 0) {
    return { error: "接続先モデルスタックが無効／バイパス状態です。ローダー側の状態を確認してください。" };
  }
  const control = loader._umsClipControl;
  if (!control?.getEntries || !control?.updateClipStrength) {
    return { error: "モデルスタックのCLIP強度操作を読み込めません。ワークフローを再表示してください。" };
  }
  return { loader, control, entries: control.getEntries() };
}

export function changeConnectedClipStrength(getHost, snapshot, index, name, value) {
  const live = resolveClipLoader(getHost());
  if (live.error) return { ok: false, error: live.error };
  if (live.loader !== snapshot.loader || JSON.stringify(live.entries) !== JSON.stringify(snapshot.entries)) {
    return { ok: false, error: "接続またはスタックが変更されました。更新して対象を確認してください。" };
  }
  const entry = live.entries.find((item) => item.index === index && item.lora_name === name);
  if (!entry || !entry.enabled || entry.bypassed) return { ok: false, error: "このLoRAは無効／バイパス状態です。" };
  if (!Number.isFinite(value) || value < -100 || value > 100) return { ok: false, error: "CLIP強度は−100〜100の数値で指定してください。" };
  return live.control.updateClipStrength(index, name, value);
}

const PANEL_ID = "ute-lora-clip-panel";
export function showLoraClipControl(anchor, getHost) {
  document.getElementById(PANEL_ID)?.remove();
  const panel = document.createElement("section");
  panel.id = PANEL_ID;
  panel.className = "ute-clip-panel";
  panel.setAttribute("aria-label", "LoRAのCLIP強度");
  panel.style.left = `${anchor.x}px`;
  panel.style.top = `${anchor.y}px`;
  panel.addEventListener("pointerdown", (event) => event.stopPropagation());
  panel.addEventListener("keydown", (event) => {
    event.stopPropagation();
    if (event.key === "Escape") panel.remove();
  });
  function button(text, action) {
    const el = document.createElement("button");
    el.type = "button"; el.className = "ute-btn"; el.textContent = text;
    el.addEventListener("click", action); return el;
  }
  function text(tag, value, cls) {
    const el = document.createElement(tag); el.textContent = value;
    if (cls) el.className = cls; return el;
  }
  function refresh(message = "") {
    panel.replaceChildren();
    const header = text("div", "", "ute-clip-panel-header");
    header.append(text("strong", "LoRAのCLIP強度"), button("更新", () => refresh()), button("閉じる", () => panel.remove()));
    panel.append(header, text("p", "接続先ローダーの設定値を変更します。次回生成で使用。通常タグ・Model強度は変更しません。"));
    const status = text("p", message, "ute-clip-status");
    status.setAttribute("role", "status");
    panel.append(status);
    const snapshot = resolveClipLoader(getHost());
    if (snapshot.error) { status.textContent = snapshot.error; return; }
    panel.append(text("p", `${snapshot.loader.title || snapshot.loader.type} #${snapshot.loader.id} · CLIP直結`));
    if (!snapshot.entries.length) { status.textContent = "接続先にLoRAが登録されていません。"; return; }
    for (const entry of snapshot.entries) {
      const row = text("div", "", "ute-clip-setting");
      const label = text("label", "", "ute-clip-setting-label");
      label.append(text("span", `CLIP · ${entry.index + 1}. ${entry.lora_name}`));
      const input = document.createElement("input");
      input.type = "number"; input.min = "-100"; input.max = "100"; input.step = "0.01";
      input.value = String(entry.strength_clip);
      input.setAttribute("aria-label", `${entry.index + 1}. ${entry.lora_name} CLIP強度`);
      input.disabled = !entry.enabled || entry.bypassed;
      label.append(input);
      const apply = button("設定", () => {
        const result = changeConnectedClipStrength(getHost, snapshot, entry.index, entry.lora_name, input.valueAsNumber);
        if (result.ok) refresh(`CLIP設定値を ${result.strength_clip} に変更しました。`);
        else status.textContent = result.error;
      });
      apply.disabled = input.disabled;
      row.append(label, apply);
      if (input.disabled) row.append(text("span", entry.bypassed ? "バイパス" : "無効"));
      panel.append(row);
    }
  }
  refresh();
  document.body.append(panel);
  const rect = panel.getBoundingClientRect();
  panel.style.left = `${Math.max(4, Math.min(anchor.x, window.innerWidth - rect.width - 8))}px`;
  panel.style.top = `${Math.max(4, Math.min(anchor.y, window.innerHeight - rect.height - 8))}px`;
  panel.querySelector("button")?.focus();
}
