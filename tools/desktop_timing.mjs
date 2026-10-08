// Response time measured in the REAL desktop window (via tools/desktop_probe.mjs' DevTools port), no input to the OS:
// (1) dataset view: click a folder tile -> the new folder's grid rendered and every visible thumbnail decoded
// (2) "動画のキャラ" editor: click another character -> its grid rendered and every visible thumbnail decoded
// node tools/desktop_timing.mjs [projectId] [folderRelative]
import {execFileSync} from 'child_process';
const [, , pid = '14', folder = ''] = process.argv;
const probe = expr => JSON.parse(execFileSync('node', [new URL('./desktop_probe.mjs', import.meta.url).pathname.replace(/^\//, ''), 'eval', expr], {encoding: 'utf8', timeout: 120000}));
const VIS = `(root)=>{if(!root)return {n:0,done:0};const vh=innerHeight;const imgs=[...root.querySelectorAll('img')].filter(i=>{const r=i.getBoundingClientRect();return r.bottom>0&&r.top<vh&&r.width>0});return {n:imgs.length,done:imgs.filter(i=>i.complete&&i.naturalWidth>0).length}}`;
const WAIT = `async(cond,ms=8000)=>{const t=performance.now();while(performance.now()-t<ms){const v=cond();if(v)return v;await new Promise(r=>setTimeout(r,20));}return null}`;
const editorButton = `[...document.querySelectorAll('button')].find(x=>x.textContent.trim()==='動画のキャラ')`;

// close the editor if it is open, then click the sidebar's dataset root (timed too)
const nav = probe(`(async()=>{const vis=${VIS};const wait=${WAIT};if(document.querySelector('.cs-root'))${editorButton}?.click();
 const panel=()=>document.querySelector('.wb-project-panel[data-state=active]');
 const p0=await wait(()=>panel()?.querySelector('.df-workspace')&&panel());if(!p0)return {error:'no dataset view',href:location.href};
 const rootBtn=p0.querySelector('.df-folder-btn[data-drop-folder=""]');const r0=performance.now();rootBtn.click();
 const p=await wait(()=>p0.getAttribute('aria-busy')!=='true'&&!new URLSearchParams(location.search).get('folder')&&p0.querySelector('.df-folder-tile')&&p0);
 if(!p)return {error:'no folder tile',href:location.href};const root_ms=Math.round(performance.now()-r0);
 const tiles=[...p.querySelectorAll('.df-folder-tile')];const tile=tiles.find(t=>t.getAttribute('data-drop-folder')===${JSON.stringify(folder)})||tiles[0];
 const target=tile.getAttribute('data-drop-folder');const label=tile.textContent.trim();const t0=performance.now();tile.click();
 const grid=()=>new URLSearchParams(location.search).get('folder')===target&&p.getAttribute('aria-busy')!=='true'&&p.querySelector('.df-files');
 const rendered=await wait(()=>{const g=grid();return g&&vis(g).n>0?performance.now():null});if(!rendered)return {folder:label,error:'not rendered'};
 const done=await wait(()=>{const v=vis(grid());return v.n&&v.done===v.n?performance.now():null});
 return {root_ms,folder:label,tiles:tiles.length,rendered_ms:Math.round(rendered-t0),visible_images_ms:done?Math.round(done-t0):'timeout',visible:vis(grid()).n};})()`);
console.log('dataset folder', JSON.stringify(nav));

const ed = probe(`(async()=>{const vis=${VIS};const wait=${WAIT};const t=performance.now();if(!document.querySelector('.cs-root'))${editorButton}.click();
 const first=await wait(()=>document.querySelector('.cs-char'));if(!first)return {error:'editor did not open'};
 await wait(()=>{const v=vis(document.querySelector('.cs-grid'));return v.n&&v.done===v.n});const open_ms=Math.round(performance.now()-t);
 const chars=[...document.querySelectorAll('.cs-char')];const res=[];
 for(const c of [chars[1],chars[2],chars[Math.floor(chars.length/2)],chars[chars.length-1]].filter(Boolean)){const name=c.querySelector('.cs-name')?.textContent;const t0=performance.now();c.click();
  const done=await wait(()=>{const h=document.querySelector('.cs-grid-head')?.textContent||'';const v=vis(document.querySelector('.cs-grid'));return h.startsWith(name)&&v.n&&v.done===v.n?performance.now():null});
  res.push({name,images:c.querySelector('.cs-count')?.textContent,visible:vis(document.querySelector('.cs-grid')).n,ms:done?Math.round(done-t0):'timeout'});}
 return {characters:chars.length,open_ms,switch:res};})()`);
console.log('editor', JSON.stringify(ed));
