// Operate the "動画からLoRA" workspace like a user, in a REAL desktop window, with real mouse/keyboard input sent through the
// browser's input pipeline (CDP Input.*), and measure each step.  Use the isolated test instance so the user's window is untouched:
//   LORA_STUDIO_CDP_PORT=9234 node tools/desktop_flow_eval.mjs <shots-dir>
// Every step prints {step, ok, ms, ...facts}; a screenshot per step goes to <shots-dir>.  Edits made here are undone at the end.
import fs from 'fs';
const port = process.env.LORA_STUDIO_CDP_PORT || '9234';
const shots = process.argv[2] || '.';
fs.mkdirSync(shots, {recursive: true});
const targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
const page = targets.find(t => t.type === 'page' && t.url.startsWith('http://127.0.0.1:5175'));
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((ok, bad) => { ws.onopen = ok; ws.onerror = bad; });
let id = 0; const waiting = new Map();
ws.onmessage = m => { const d = JSON.parse(m.data); if (d.id && waiting.has(d.id)) { waiting.get(d.id)(d); waiting.delete(d.id); } };
const call = (method, params = {}) => new Promise(r => { const i = ++id; waiting.set(i, r); ws.send(JSON.stringify({id: i, method, params})); });
const js = async expr => { const r = await call('Runtime.evaluate', {expression: expr, awaitPromise: true, returnByValue: true}); if (r.result?.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails).slice(0, 300)); return r.result?.result?.value; };
const sleep = ms => new Promise(r => setTimeout(r, ms));
const now = () => performance.now();
async function waitFor(expr, ms = 15000) { const t = now(); while (now() - t < ms) { if (await js(`!!(${expr})`)) return Math.round(now() - t); await sleep(25); } return null; }
async function rect(sel, nth = 0) { return js(`(()=>{const e=[...document.querySelectorAll(${JSON.stringify(sel)})][${nth}];if(!e)return null;e.scrollIntoView({block:'nearest'});const r=e.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2,l:r.left,t:r.top,w:r.width,h:r.height}})()`); }
async function mouse(type, x, y, mods = 0, buttons = 1) { await call('Input.dispatchMouseEvent', {type, x, y, button: 'left', buttons: type === 'mouseReleased' ? 0 : buttons, clickCount: 1, modifiers: mods}); }
async function clickAt(x, y, mods = 0) { await mouse('mouseMoved', x, y, mods, 0); await mouse('mousePressed', x, y, mods); await mouse('mouseReleased', x, y, mods); }
async function click(sel, nth = 0, mods = 0) { const r = await rect(sel, nth); if (!r) throw new Error('not found: ' + sel); await clickAt(r.x, r.y, mods); return r; }
async function clickText(sel, text) { const n = await js(`[...document.querySelectorAll(${JSON.stringify(sel)})].findIndex(e=>e.textContent.includes(${JSON.stringify(text)}))`); if (n < 0) throw new Error('no ' + text); return click(sel, n); }
async function key(k, code, mods = 0) { await call('Input.dispatchKeyEvent', {type: 'keyDown', key: k, code, modifiers: mods, windowsVirtualKeyCode: {Delete: 46, Escape: 27, z: 90, a: 65}[k] || 0}); await call('Input.dispatchKeyEvent', {type: 'keyUp', key: k, code, modifiers: mods}); }
async function shot(name) { const r = await call('Page.captureScreenshot', {format: 'jpeg', quality: 70}); fs.writeFileSync(`${shots}/${name}.jpg`, Buffer.from(r.result.data, 'base64')); }
const visibleLoaded = `(()=>{const vh=innerHeight;const im=[...document.querySelectorAll('.ce-pane img[data-rel]')].filter(i=>{const r=i.getBoundingClientRect();return r.bottom>0&&r.top<vh&&r.width>0});return im.length>0&&im.every(i=>i.complete&&i.naturalWidth>0)})()`;
const out = [];
const log = (step, ok, extra = {}) => { const row = {step, ok, ...extra}; out.push(row); console.log(JSON.stringify(row)); };
const CTRL = 2, SHIFT = 8;

await js(`document.visibilityState`).then(v => log('window visible', v === 'visible', {visibility: v}));
// 1. open the workspace from the header
let t = now();
await clickText('header button', '動画から');
let ms = await waitFor(`document.querySelector('.vl-root')&&document.querySelector('.vl-step')`);
log('open 動画からLoRA', ms !== null, {ms, steps: await js(`[...document.querySelectorAll('.vl-step')].map(b=>b.textContent)`)});
await shot('01-open');
// 2. ① videos
await click('.vl-step', 0);
ms = await waitFor(`document.querySelectorAll('.vl-video').length>0`);
log('① 動画の一覧', ms !== null, {ms, videos: await js(`document.querySelectorAll('.vl-video').length`), done: await js(`document.querySelectorAll('.vl-video.done').length`), stage: await js(`document.querySelector('.vl-stage')?.textContent`)});
await shot('02-videos');
// 3. ② a character with outfits
await click('.vl-step', 1);
await waitFor(`document.querySelector('.ce-root .cs-char')`);
t = now();
await click('.ce-root .cs-char:not(.held)', 1);
ms = await waitFor(`document.querySelectorAll('.ce-outfit').length>0`);
const vis = await waitFor(visibleLoaded, 20000);
log('② キャラを開く（衣装ごと）', ms !== null, {outfits_ms: ms, visible_images_ms: vis === null ? 'timeout' : Math.round(now() - t), outfits: await js(`[...document.querySelectorAll('.ce-outfit-name')].map(e=>e.textContent)`)});
await shot('03-character');
// 4. drag a box from empty space over the first images
const g = await rect('.ce-outfit .ce-grid', 0);
const pane = await rect('.ce-pane');
await mouse('mouseMoved', g.l - 4, g.t + 4, 0, 0);
await mouse('mousePressed', g.l - 4, g.t + 4);
for (let k = 1; k <= 8; k++) await mouse('mouseMoved', g.l - 4 + k * 40, g.t + 4 + k * 20);
await mouse('mouseReleased', g.l + 316, g.t + 164);
await sleep(150);
log('範囲選択（ドラッグ）', (await js(`document.querySelectorAll('.ce-pane img.sel').length`)) > 0, {selected: await js(`document.querySelectorAll('.ce-pane img.sel').length`), head: await js(`document.querySelector('.ce-head').textContent`), dimmed_others: await js(`getComputedStyle(document.querySelector('.ce-pane img:not(.sel)')).opacity`)});
await shot('04-box-select');
await key('Escape', 'Escape');
// 5. click + shift-click range
await click('.ce-outfit .ce-grid img', 0);
await click('.ce-outfit .ce-grid img', 9, SHIFT);
log('Shiftで範囲', (await js(`document.querySelectorAll('.ce-pane img.sel').length`)) === 10, {selected: await js(`document.querySelectorAll('.ce-pane img.sel').length`)});
await key('Escape', 'Escape');
// 6. a pending pile as compare rows
const pileName = await js(`[...document.querySelectorAll('.ce-root .cs-char.held .cs-name')].map(e=>e.textContent).find(n=>n.endsWith('の候補'))`);
t = now();
await clickText('.ce-root .cs-char.held', pileName);
ms = await waitFor(`document.querySelectorAll('.ce-row').length>0`);
const vis2 = await waitFor(visibleLoaded, 20000);
const row0 = await js(`(()=>{const r=document.querySelector('.ce-row');return {dest:r.querySelector('.rt-SelectTrigger')?.textContent,refs:r.querySelectorAll('.ce-refs img').length,images:r.querySelectorAll('.ce-grid img').length,buttons:[...r.querySelectorAll('button')].map(b=>b.textContent),meta:r.querySelector('.ce-row-dest .rt-Text:last-child')?.textContent}})()`);
log('保留を見比べ一覧で開く', ms !== null, {pile: pileName, rows_ms: ms, visible_images_ms: vis2 === null ? 'timeout' : Math.round(now() - t), rows: await js(`document.querySelectorAll('.ce-row').length`), first_row: row0});
await shot('05-pile-rows');
// 7. the first row into its character, then undo
const before = await js(`[...document.querySelectorAll('.ce-root .cs-char')].map(c=>c.querySelector('.cs-name').textContent+'='+c.querySelector('.cs-count').textContent)`);
t = now();
await click('.ce-row .ce-row-actions button', 0);
ms = await waitFor(`document.querySelector('.ce-head')?.textContent.includes('移しました')`);
const after = await js(`[...document.querySelectorAll('.ce-root .cs-char')].map(c=>c.querySelector('.cs-name').textContent+'='+c.querySelector('.cs-count').textContent)`);
log('行を帰属先へ', ms !== null, {ms, notice: await js(`document.querySelector('.ce-head').textContent`), changed: after.filter(x => !before.includes(x))});
await shot('06-row-moved');
t = now();
await key('z', 'KeyZ', CTRL);
ms = await waitFor(`document.querySelector('.ce-head')?.textContent.includes('元に戻しました')`);
const restored = await js(`[...document.querySelectorAll('.ce-root .cs-char')].map(c=>c.querySelector('.cs-name').textContent+'='+c.querySelector('.cs-count').textContent)`);
log('Ctrl+Zで元に戻す', ms !== null && JSON.stringify(restored) === JSON.stringify(before), {ms});
// 8. select two images in the pile and Delete, then undo
await waitFor(`document.querySelectorAll('.ce-row .ce-grid img').length>1`);
await click('.ce-row .ce-grid img', 0);
await click('.ce-row .ce-grid img', 1, CTRL);
log('Ctrl+クリックで2枚選択', (await js(`document.querySelectorAll('.ce-pane img.sel').length`)) === 2, {selected: await js(`document.querySelectorAll('.ce-pane img.sel').length`), first_row_images: await js(`document.querySelector('.ce-row').querySelectorAll('.ce-grid img').length`)});
const binBefore = await js(`document.querySelector('.cs-drop[title^="除外"] span').textContent`);
await key('Delete', 'Delete');
ms = await waitFor(`document.querySelector('.ce-head')?.textContent.includes('除外しました')`);
log('Deleteで除外', ms !== null, {ms, bin: `${binBefore} -> ${await js(`document.querySelector('.cs-drop[title^="除外"] span').textContent`)}`});
await key('z', 'KeyZ', CTRL);
await waitFor(`document.querySelector('.ce-head')?.textContent.includes('元に戻しました')`);
// 9. tick two characters, ③ LoRA
const ticks = await js(`document.querySelectorAll('.ce-root .cs-char:not(.held) button[role=checkbox]').length`);
await click('.ce-root .cs-char:not(.held) button[role=checkbox]', 0);
await click('.ce-root .cs-char:not(.held) button[role=checkbox]', 1);
await click('.vl-step', 2);
ms = await waitFor(`document.querySelectorAll('.vl-lora-char').length===2&&document.querySelectorAll('.vl-outfit').length>0`);
log('③ LoRA（2人を選択）', ms !== null, {ms, ticks, step: await js(`document.querySelectorAll('.vl-step')[2].textContent`), chars: await js(`[...document.querySelectorAll('.vl-lora-char')].map(c=>c.querySelector('input').value+': '+[...c.querySelectorAll('.vl-outfit')].map(o=>o.textContent).join(' | '))`), button: await js(`document.querySelector('.vl-lora-go button')?.textContent`)});
await shot('07-lora');
// untick again (the picks are a per-viewer convenience; do not leave them)
await click('.vl-lora-head button[role=checkbox]', 0); await sleep(150);
await click('.vl-lora-head button[role=checkbox]', 0); await sleep(150);
// 10. drag 10 selected images onto "+" (a new character), make it a LoRA project, open it; then remove all of it
await click('.vl-step', 1);
await waitFor(`document.querySelector('.ce-root .cs-char')`);
await click('.ce-root .cs-char:not(.held)', 7);
await waitFor(visibleLoaded, 20000);
await click('.ce-outfit .ce-grid img', 0);
await click('.ce-outfit .ce-grid img', 9, SHIFT);
const src = await rect('.ce-outfit .ce-grid img.sel', 0), plus = await rect('.cs-drop[title^="ここにドロップ"]');
const countBefore = await js(`document.querySelectorAll('.ce-root .cs-char').length`);
// native drag and drop: CDP mouse events alone never start the OS drag loop, so the drag is intercepted and dropped with
// Input.dispatchDragEvent - the same dragenter/dragover/drop the user's mouse produces
const dragged = new Promise(res => { const prev = ws.onmessage; ws.onmessage = m => { const d = JSON.parse(m.data); if (d.method === 'Input.dragIntercepted') res(d.params.data); prev(m); }; });
await call('Input.setInterceptDrags', {enabled: true});
await mouse('mouseMoved', src.x, src.y, 0, 0); await mouse('mousePressed', src.x, src.y);
for (let k = 1; k <= 6; k++) { await mouse('mouseMoved', src.x + 6 * k, src.y + 6 * k); await sleep(20); }
const data = await Promise.race([dragged, sleep(3000).then(() => null)]);
if (data) {
  data.dragOperationsMask = 1 | 16;  // copy|move, what a real mouse drag of an <img> offers
  for (const type of ['dragEnter', 'dragOver', 'drop']) await call('Input.dispatchDragEvent', {type, x: plus.x, y: plus.y, data});
}
await mouse('mouseReleased', plus.x, plus.y);
await call('Input.setInterceptDrags', {enabled: false});
log('ドラッグ開始を検出', !!data, {types: data?.items?.map(i => i.mimeType)});
ms = await waitFor(`document.querySelectorAll('.ce-root .cs-char').length>${countBefore}`, 8000);
const newName = await js(`[...document.querySelectorAll('.ce-root .cs-char:not(.held) .cs-name')].map(e=>e.textContent).find(n=>n.startsWith('新しいキャラ'))||[...document.querySelectorAll('.ce-root .cs-char:not(.held)')].map(e=>e.querySelector('.cs-name').textContent+'='+e.querySelector('.cs-count').textContent).find(n=>n.endsWith('=10'))`);
log('10枚を＋へドラッグ（新しいキャラ）', ms !== null, {ms, new_character: newName, notice: await js(`document.querySelector('.ce-head').textContent`)});
if (ms !== null) {
  const n = await js(`[...document.querySelectorAll('.ce-root .cs-char:not(.held)')].findIndex(e=>e.querySelector('.cs-count').textContent==='10')`);
  await click('.ce-root .cs-char:not(.held) button[role=checkbox]', n);
  await click('.vl-step', 2);
  await waitFor(`document.querySelector('.vl-lora-go button')&&!document.querySelector('.vl-lora-go button').disabled`);
  await js(`(()=>{const i=document.querySelector('.vl-lora-head input');const set=Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set;set.call(i,'検証用_自動削除');i.dispatchEvent(new Event('input',{bubbles:true}));return 1})()`);
  t = now();
  await click('.vl-lora-go button', 0);
  ms = await waitFor(`document.querySelector('.vl-made-row')`, 60000);
  log('③ LoRAのプロジェクトを作る', ms !== null, {ms, made: await js(`document.querySelector('.vl-made')?.textContent`)});
  await shot('08-made');
  await click('.vl-made-row button', 0);
  ms = await waitFor(`!document.querySelector('.vl-root')&&[...document.querySelectorAll('.wb-project-trigger')].some(t=>t.textContent.startsWith('検証用_自動削除'))`, 15000);
  const pid = await js(`Number(location.pathname.match(/project\\/(\\d+)/)?.[1]||0)`);
  log('開いて学習へ（新しいタブ）', ms !== null, {ms, tab: await js(`document.querySelector('.wb-project-trigger[data-state=active]')?.textContent`), images_shown: await waitFor(`document.querySelectorAll('.wb-project-panel[data-state=active] .df-folder-tile, .wb-project-panel[data-state=active] .df-file').length>0`, 15000) !== null});
  await shot('09-opened');
  // cleanup: delete the test project, close its tab, undo the move
  const made = await js(`(async()=>{const ps=await fetch('/api/projects').then(r=>r.json());const p=ps.find(x=>x.name==='検証用_自動削除');if(!p)return 'none';const r=await fetch('/api/projects/'+p.id,{method:'DELETE'});return r.status})()`);
  await js(`(()=>{const b=[...document.querySelectorAll('.wb-close-tab')].find(b=>b.getAttribute('aria-label').startsWith('検証用_自動削除'));b&&b.click();return 1})()`);
  const set = await js(`(async()=>{const w=await fetch('/api/character-sets/workspaces').then(r=>r.json());return w.workspaces.find(x=>x.name==='アイのシナリオ').set_id})()`);
  await js(`fetch('/api/character-sets/'+encodeURIComponent(${JSON.stringify('')}+${JSON.stringify(set)})+'/undo',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}).then(r=>r.status)`);
  const after = await js(`fetch('/api/character-sets/'+encodeURIComponent(${JSON.stringify(set)})).then(r=>r.json()).then(s=>s.characters.filter(c=>!c.pending).map(c=>c.name+'='+c.count))`);
  log('後片付け', made === 200, {project_deleted: made, mains_after_undo: after});
  await js(`(()=>{try{for(const k of Object.keys(localStorage))if(k.startsWith('videolora:picked:'))localStorage.removeItem(k)}catch{}return 1})()`);
}
fs.writeFileSync(`${shots}/flow.json`, JSON.stringify(out, null, 1));
console.log('DONE', out.filter(r => !r.ok).length, 'failed');
process.exit(0);
