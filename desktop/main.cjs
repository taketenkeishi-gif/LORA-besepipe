const {app,BrowserWindow,Menu,screen,dialog,ipcMain,shell}=require('electron');
const path=require('path'),fs=require('fs'),{execFile}=require('child_process');
const ROOT=process.env.LORA_STUDIO_ROOT||path.resolve(path.dirname(process.execPath),'../..');
const URL_BASE=process.env.LORA_STUDIO_TEST_URL||'http://127.0.0.1:5175';
if(new URL(URL_BASE).hostname!=='127.0.0.1')throw new Error('Desktop requires a loopback application URL');
if(process.env.LORA_STUDIO_TEST_PROFILE)app.setPath('userData',process.env.LORA_STUDIO_TEST_PROFILE);
app.setName('LoRA Studio');app.setAppUserModelId('Keishi.LoRAStudio');
const WINDOW_TITLE=process.env.LORA_STUDIO_TEST_PROFILE?'LoRA Studio（検証用）':'LoRA Studio';
const runtimeDir=process.env.LORA_STUDIO_TEST_PROFILE||path.join(ROOT,'.runtime');
fs.mkdirSync(runtimeDir,{recursive:true});
const statePath=path.join(runtimeDir,'desktop-window.json');
const statusPath=path.join(runtimeDir,'desktop-status.json');
let win;
function readState(){try{return JSON.parse(fs.readFileSync(statePath,'utf8'))}catch{return {route:'/project/2/dataset/images?folder=references'}}}
function record(extra={}){fs.writeFileSync(statusPath,JSON.stringify({pid:process.pid,root:ROOT,executable:process.execPath,url:win?.webContents.getURL(),visible:win?.isVisible(),bounds:win?.getBounds(),...extra},null,2))}
function save(){if(!win||win.isDestroyed())return;const url=win.webContents.getURL();if(url.startsWith(URL_BASE+'/'))fs.writeFileSync(statePath,JSON.stringify({route:url.slice(URL_BASE.length),bounds:win.getBounds()},null,2))}
function ensureServer(){return new Promise((resolve,reject)=>execFile('powershell.exe',['-NoProfile','-ExecutionPolicy','Bypass','-File',path.join(ROOT,'scripts','Start.ps1')],{windowsHide:true,timeout:60000},(err,out,stderr)=>err?reject(new Error(stderr||err.message)):resolve(out)))}
function trusted(event){if(!event.senderFrame||new URL(event.senderFrame.url).origin!==URL_BASE)throw new Error('Unknown desktop caller');}
function datasetAction(operation,payload){return new Promise((resolve,reject)=>{
 const child=execFile(path.join(ROOT,'backend','.venv','Scripts','python.exe'),['-X','utf8',path.join(ROOT,'backend','desktop_actions.py'),operation],{cwd:path.join(ROOT,'backend'),windowsHide:true,timeout:300000,maxBuffer:16*1024*1024},(error,stdout,stderr)=>{
  try{const reply=JSON.parse(stdout);if(!reply.ok)reject(new Error(reply.error));else resolve(reply.result)}catch(parseError){reject(new Error(stderr||error?.message||String(parseError)))}
 });child.stdin.end(JSON.stringify(payload));
});}
// Every picker opens where it was last used (per kind; else wherever any picker was last used), never at the default root.
const lastPath=path.join(runtimeDir,'last-folders.json');
function lastFolders(){try{return JSON.parse(fs.readFileSync(lastPath,'utf8'))}catch{return {}}}
function startFolder(kind){const l=lastFolders();for(const p of [l[kind],l._any])if(p&&fs.existsSync(p))return p;return undefined;}
function remember(kind,picked,isFile){if(!picked)return;const folder=isFile?path.dirname(picked):picked;const l=lastFolders();l[kind]=folder;l._any=folder;try{fs.writeFileSync(lastPath,JSON.stringify(l,null,2))}catch{}}
async function pick(kind,options,isFile){const start=process.env.LORA_STUDIO_TEST_FOLDER||startFolder(kind);const result=await dialog.showOpenDialog(win,{...options,...(start?{defaultPath:start}:{})});if(result.canceled)return null;remember(kind,result.filePaths[0],isFile);return result.filePaths[0];}
ipcMain.handle('lora:open-dataset-folder',async event=>{trusted(event);const folder=await pick('dataset',{title:'データセットフォルダを開く',buttonLabel:'このフォルダを開く',properties:['openDirectory']},false);return folder?datasetAction('open-folder',{folder}):null;});
ipcMain.handle('lora:dataset-action',(event,operation,payload)=>{trusted(event);if(!['normalize-preview','normalize-apply','naming-preview','naming-apply','naming-undo','preprocess-capabilities','preprocess-start','preprocess-status','preprocess-cancel','web-image','tag-models','tag-start','tag-status','tag-stop','comfy-discover','comfy-inspect','comfy-bind','comfy-checkpoints','comfy-preview-start','comfy-preview-status','comfy-reveal'].includes(operation))throw new Error('Unknown dataset action');return datasetAction(operation,payload);});
ipcMain.handle('lora:choose-video',async event=>{trusted(event);return pick('video',{title:'動画を選ぶ',buttonLabel:'この動画を使う',properties:['openFile'],filters:[{name:'動画',extensions:['mp4','mov','mkv','webm','avi','m4v']}]},true);});
ipcMain.handle('lora:choose-comfy-root',async event=>{trusted(event);return pick('comfy',{title:'ComfyUI本体フォルダを選択',properties:['openDirectory']},false);});
ipcMain.handle('lora:reveal-preset',async(event,id)=>{trusted(event);const folder=path.resolve(process.env.LORA_STUDIO_PRESETS_DIR||path.join(ROOT,'user-presets','training'));if(id==null||id===''){fs.mkdirSync(folder,{recursive:true});const failure=await shell.openPath(folder);if(failure)throw new Error(failure);return folder;}if(typeof id!=='string'||id!==path.basename(id)||!id.toLowerCase().endsWith('.json'))throw new Error('プリセットの指定が不正です');const file=path.join(folder,id);if(path.dirname(path.resolve(file))!==folder)throw new Error('プリセットフォルダの外は開けません');if(!fs.existsSync(file))throw new Error('プリセットのファイルが見つかりません');shell.showItemInFolder(file);return file;});
async function serializeWorkflow(result){
 if(!Array.isArray(result.data.nodes)){const graph=result.data.prompt||result.data;if(!graph||typeof graph!=='object'||!Object.keys(graph).length||!Object.values(graph).every(n=>n&&typeof n.class_type==='string'&&n.inputs))throw new Error('ComfyUIワークフローのJSONを選択してください');return {graph,path:result.path};}
 // Serialize through this installed ComfyUI frontend, in a fresh hidden memory-only session.
 const compiler=new BrowserWindow({show:false,webPreferences:{sandbox:true,contextIsolation:true,nodeIntegration:false,partition:'lora-workflow-'+Date.now(),backgroundThrottling:false}});
 compiler.webContents.setWindowOpenHandler(()=>({action:'deny'}));
 compiler.webContents.session.webRequest.onBeforeRequest((details,callback)=>callback({cancel:!['GET','HEAD'].includes(details.method)}));
 const timer=setTimeout(()=>{if(!compiler.isDestroyed())compiler.destroy();},45000);
 try{
  await compiler.loadURL(result.url);
  const graph=await compiler.webContents.executeJavaScript(`(async()=>{const until=Date.now()+25000;while((!window.comfyAPI?.app?.app?.graph||!window.LiteGraph?.registered_node_types?.EmptyLatentImage)&&Date.now()<until)await new Promise(r=>setTimeout(r,100));const app=window.comfyAPI?.app?.app;if(!app?.graph)throw new Error('ComfyUI画面の準備が完了しませんでした');app.graph.configure(${JSON.stringify(result.data)});const prompt=await app.graphToPrompt();if(!prompt.output||!Object.keys(prompt.output).length||!Object.values(prompt.output).every(n=>typeof n.class_type==='string'&&n.inputs&&!Object.keys(n.inputs).some(k=>k.startsWith('UNKNOWN'))))throw new Error('ワークフローのノードまたは入力を解決できませんでした。接続先のカスタムノードを確認してください');return prompt.output;})()`);
  return {graph,path:result.path};
 }finally{clearTimeout(timer);if(!compiler.isDestroyed())compiler.destroy();}
}
ipcMain.handle('lora:load-comfy-workflow',async(event,payload)=>{trusted(event);return serializeWorkflow(await datasetAction('comfy-workflow',payload));});
ipcMain.handle('lora:compile-comfy-workflow',async(event,payload)=>{trusted(event);if(JSON.stringify(payload.data).length>10*1024*1024)throw new Error('ワークフローが大きすぎます');const binding=await datasetAction('comfy-discover',{});return serializeWorkflow({data:payload.data,path:payload.name,url:binding.url});});
if(!app.requestSingleInstanceLock()){app.quit()}else{
 app.on('second-instance',()=>{if(win){if(win.isMinimized())win.restore();win.showInactive();record({event:'second-instance'})}});
 app.whenReady().then(async()=>{
  Menu.setApplicationMenu(Menu.buildFromTemplate([{label:'編集',submenu:[{role:'undo'},{role:'redo'},{role:'cut'},{role:'copy'},{role:'paste'},{role:'selectAll'}]}]));
  const state=readState();
  const monitor=JSON.parse(fs.readFileSync(path.join(ROOT,'desktop','display1.json'),'utf8'));
  const point=screen.screenToDipPoint({x:monitor.left+20,y:monitor.top+20});
  const area=screen.getDisplayNearestPoint(point).workArea;
  const width=Math.min(1400,area.width-32),height=Math.min(900,area.height-32);
  win=new BrowserWindow({title:WINDOW_TITLE,x:area.x+16,y:area.y+16,width,height,minWidth:Math.min(640,width),minHeight:Math.min(600,height),show:false,autoHideMenuBar:true,icon:path.join(ROOT,'desktop','icon.ico'),backgroundColor:'#151519',webPreferences:{contextIsolation:true,nodeIntegration:false,sandbox:true,backgroundThrottling:process.env.LORA_STUDIO_TEST_HIDE!=='1',preload:path.join(__dirname,'preload.cjs')}});
  win.webContents.setWindowOpenHandler(()=>({action:'deny'}));
  win.webContents.on('will-navigate',(event,url)=>{if(new URL(url).origin!==URL_BASE)event.preventDefault()});
  win.webContents.on('page-title-updated',event=>{event.preventDefault();win.setTitle(WINDOW_TITLE)});
  win.webContents.on('did-navigate-in-page',()=>save());
  win.on('close',save);
  try{
   if(!process.env.LORA_STUDIO_TEST_URL)await ensureServer();
   const response=await fetch(URL_BASE+'/instance');const identity=await response.json();
   if(path.resolve(identity.root).toLowerCase()!==path.resolve(ROOT).toLowerCase())throw new Error('起動先の確認に失敗しました。');
   const route=typeof state.route==='string'&&state.route.startsWith('/')?state.route:'/';
   await win.loadURL(URL_BASE+route);
   if(process.env.LORA_STUDIO_TEST_HIDE!=='1')win.showInactive();
   const pageEvidence=await win.webContents.executeJavaScript(`new Promise(resolve=>{const deadline=Date.now()+15000;function check(){if(document.querySelector('.df-file img')||Date.now()>deadline)resolve();else setTimeout(check,100)}check()}).then(()=>Promise.all(Array.from(document.querySelectorAll('.df-file img')).map(async e=>{try{await Promise.race([e.decode(),new Promise(resolve=>setTimeout(resolve,3000))])}catch{}return {loaded:e.complete&&e.naturalWidth>0}}))).then(images=>({title:document.title,images,desktopBridge:!!window.loraDesktop,taggingBridge:typeof window.loraDesktop?.startTags==="function",comfyBridge:typeof window.loraDesktop?.comfy==="function",scripts:Array.from(document.scripts).map(s=>s.src),folderOpenControl:Array.from(document.querySelectorAll("button")).some(b=>b.textContent.includes("フォルダを開く")),scrollWidth:document.documentElement.scrollWidth,clientWidth:document.documentElement.clientWidth}))`);
   await win.webContents.executeJavaScript(`Promise.race([new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))),new Promise(resolve=>setTimeout(resolve,2000))])`);
   record({event:'ready',title:win.getTitle(),display:area,pageEvidence});
   try{
    const capture=await Promise.race([win.webContents.capturePage(),new Promise((_,reject)=>setTimeout(()=>reject(new Error('Preview capture timed out')),5000))]);
    fs.writeFileSync(path.join(runtimeDir,'desktop-preview.png'),capture.toPNG());
    record({event:'ready',title:win.getTitle(),display:area,pageEvidence,previewCapturedAt:new Date().toISOString()});
   }catch(error){record({event:'ready',title:win.getTitle(),display:area,pageEvidence,previewCaptureError:error.message});}
  }catch(error){record({event:'error',error:error.message});dialog.showErrorBox('LoRA Studio 起動エラー',error.message);app.quit()}
 }).catch(error=>{dialog.showErrorBox('LoRA Studio 起動エラー',error.message);app.quit()});
 app.on('window-all-closed',()=>app.quit());
}
