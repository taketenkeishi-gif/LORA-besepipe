// tag-editor/lora-picker.js
//
// Floating "LoRA nodes found in graph" picker, opened by the toolbar's LoRA
// button. Mirrors context_menu.js's floating-panel conventions (fixed
// position, dismissed by the next outside mousedown) but needs its own
// renderer since rows have an inline editable trigger-word field, which the
// generic FloatingMenu item model doesn't support.

import { loraShortName, LoraTriggerDB } from "./lora-scan.js";

const PANEL_ID = "ute-lora-picker";

export function closeLoraPicker() {
  document.getElementById(PANEL_ID)?.remove();
}

/**
 * @param {{x:number,y:number}} anchor  viewport coords to open at
 * @param {Array<{name:string,strength:number}>} loras
 * @param {(lora, triggerText:string) => void} onInsert  insert this one LoRA
 *   (+ its current trigger-word text, may be empty)
 */
export function showLoraPicker(anchor, loras, onInsert) {
  closeLoraPicker();

  const p = document.createElement("div");
  p.className = "ute-lora-picker";
  p.id = PANEL_ID;
  p.style.left = `${anchor.x}px`;
  p.style.top = `${anchor.y}px`;

  const header = document.createElement("div");
  header.className = "ute-lora-picker-hdr";
  header.textContent = loras.length
    ? `LoRAトリガー候補 · ワークフロー内 ${loras.length} 件`
    : "LoRAトリガー候補がありません";
  p.appendChild(header);
  const explanation = document.createElement("p");
  explanation.textContent = "プロンプト文字列に追加します。LoRAのCLIP強度は「CLIP強度」で設定。未登録の候補はファイル名から推測しています。";
  explanation.style.cssText = "padding:4px 10px;line-height:1.4";
  p.appendChild(explanation);

  if (!loras.length) {
    const hint = document.createElement("div");
    hint.className = "ute-lora-picker-empty";
    hint.textContent = "ワークフローにLoRAローダーを追加するか、下の手動入力を使用してください。";
    p.appendChild(hint);
  } else {
    if (loras.length > 1) {
      const allBtn = document.createElement("button");
      allBtn.className = "ute-btn";
      allBtn.textContent = "全候補をトリガーとして追加";
      allBtn.addEventListener("pointerdown", (e) => e.stopPropagation());
      allBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        for (const lora of loras) onInsert(lora, LoraTriggerDB.get(lora.name));
        closeLoraPicker();
      });
      p.appendChild(allBtn);
    }

    for (const lora of loras) {
      p.appendChild(buildRow(lora, onInsert));
    }
  }

  const manual = document.createElement("button");
  manual.className = "ute-btn";
  manual.textContent = "トリガーを手動入力…";
  manual.addEventListener("pointerdown", (e) => e.stopPropagation());
  manual.addEventListener("click", (e) => {
    e.stopPropagation();
    closeLoraPicker();
    const trigger = window.prompt("追加するトリガー文字列（カンマ区切り）:");
    if (trigger && trigger.trim()) onInsert({ name: "", strength: 1 }, trigger.trim());
  });
  p.appendChild(manual);

  p.addEventListener("pointerdown", (e) => e.stopPropagation());
  document.body.appendChild(p);

  const r = p.getBoundingClientRect();
  if (r.right > window.innerWidth - 6) p.style.left = `${Math.max(4, window.innerWidth - r.width - 6)}px`;
  if (r.bottom > window.innerHeight - 6) p.style.top = `${Math.max(4, window.innerHeight - r.height - 6)}px`;

  setTimeout(() => {
    const onDown = (ev) => { if (!p.contains(ev.target)) closeLoraPicker(); };
    document.addEventListener("mousedown", onDown, { once: true, capture: true });
  }, 60);
}

function buildRow(lora, onInsert) {
  const row = document.createElement("div");
  row.className = "ute-lora-row";

  const nameEl = document.createElement("span");
  nameEl.className = "ute-lora-row-name";
  nameEl.textContent = loraShortName(lora.name);
  nameEl.title = lora.name;

  const triggerInput = document.createElement("input");
  triggerInput.className = "ute-lora-row-trigger";
  triggerInput.value = LoraTriggerDB.get(lora.name);
  triggerInput.title = "Trigger tag(s) inserted into the positive prompt";
  triggerInput.addEventListener("pointerdown", (e) => e.stopPropagation());
  triggerInput.addEventListener("keydown", (e) => e.stopPropagation());
  triggerInput.addEventListener("change", () => LoraTriggerDB.set(lora.name, triggerInput.value));

  const insertBtn = document.createElement("button");
  insertBtn.className = "ute-btn";
  insertBtn.textContent = "追加";
  insertBtn.title = "候補文字列をプロンプトへ追加（LoRA強度は変更しません）";
  insertBtn.addEventListener("pointerdown", (e) => e.stopPropagation());
  insertBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    LoraTriggerDB.set(lora.name, triggerInput.value);
    onInsert(lora, triggerInput.value);
    closeLoraPicker();
  });

  row.append(nameEl, triggerInput, insertBtn);
  return row;
}
