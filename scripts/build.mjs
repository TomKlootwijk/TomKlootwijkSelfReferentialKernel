import{readFileSync,statSync}from'node:fs';
import{spawnSync}from'node:child_process';
const files=['index.html','style.css','core.js','gpu.js','machine.js','peer.js','app.js','pinion.wgsl','tape.wgsl'];
for(const file of files){const text=readFileSync(new URL('../dist/'+file,import.meta.url),'utf8');if(!text.trim())throw Error('Empty asset '+file);if(file.endsWith('.js')){const r=spawnSync(process.execPath,['--check',new URL('../dist/'+file,import.meta.url).pathname.replace(/^\/([A-Za-z]:)/,'$1')],{stdio:'inherit'});if(r.status)process.exit(r.status);}}
const bytes=files.reduce((n,f)=>n+statSync(new URL('../dist/'+f,import.meta.url)).size,0);
console.log(JSON.stringify({result:'PASS',staticAssets:files.length,totalBytes:bytes,noBundler:true}));
