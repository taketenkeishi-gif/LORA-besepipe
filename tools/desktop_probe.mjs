// Inspect the RUNNING LoRA Studio desktop window over its loopback DevTools port, without focus or input.
// node tools/desktop_probe.mjs eval "<js expression>"        -> prints the JSON result
// node tools/desktop_probe.mjs shot  <out.png>               -> saves a screenshot of the real window content
// node tools/desktop_probe.mjs click "<css selector>"        -> element.click() inside the page (no OS mouse)
const port = process.env.LORA_STUDIO_CDP_PORT || '9233';
const [, , cmd, arg] = process.argv;
const targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
const page = targets.find(t => t.type === 'page' && t.url.startsWith('http://127.0.0.1:5175')) || targets.find(t => t.type === 'page');
if (!page) { console.error('no page target'); process.exit(2); }
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((ok, bad) => { ws.onopen = ok; ws.onerror = bad; });
let id = 0; const waiting = new Map();
ws.onmessage = m => { const d = JSON.parse(m.data); if (d.id && waiting.has(d.id)) { waiting.get(d.id)(d); waiting.delete(d.id); } };
const call = (method, params = {}) => new Promise(r => { const i = ++id; waiting.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const evaluate = async expr => { const r = await call('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true }); if (r.result?.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails).slice(0, 400)); return r.result?.result?.value; };
if (cmd === 'eval') console.log(JSON.stringify(await evaluate(arg), null, 1));
else if (cmd === 'click') console.log(JSON.stringify(await evaluate(`(()=>{const e=document.querySelector(${JSON.stringify(arg)});if(!e)return 'not found';e.click();return 'clicked';})()`)));
else if (cmd === 'shot') { const r = await call('Page.captureScreenshot', { format: 'png' }); (await import('fs')).writeFileSync(arg, Buffer.from(r.result.data, 'base64')); console.log('saved', arg, page.url); }
else console.error('usage: eval|shot|click');
ws.close();
