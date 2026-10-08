import{DEFAULT,Pinion,Tape,packBits,canonical,pairHex,encodeSeed,need,AsyncHopRounds}from'./core.js';
import{Machine}from'./machine.js';import{GPUKernel}from'./gpu.js';import{PeerSession}from'./peer.js';
const $=id=>document.getElementById(id),log=message=>{$('result').textContent=message;$('result').className='';},error=e=>{$('result').textContent=e.message;$('result').className='error';};
let gpu,signalType='offer',busy=false;const session=new PeerSession(new Machine(),null,committed,message=>{$('session').textContent=message;});
const tapeGenes={role:'tape',...new Tape().snapshot()},pinionGenes={role:'pinion',...DEFAULT};
function display(){const m=session.machine,k=m.kernel;$('state').textContent=m.role==='pinion'?`PINION · tick ${k.tick} · epoch ${k.epoch} · pair ${pairHex(k.word)} · packet ${encodeSeed(k).length} bytes`:`TAPE · tick ${k.tick} · q ${k.q}/${k.states} · head ${k.head} · ${k.status}`;$('checkpoint').value=JSON.stringify(m.checkpoint(),null,2);}
function committed(result,round){display();$('trace').textContent=result.trace.slice(-16).map(row=>row.map(x=>x.toString(16).padStart(8,'0')).join(' ')).join('\n')||'Gene / idle commit';log(`ACCEPT round ${round} · ${result.trace.length} transitions · ${session.gpu?'WebGPU verified':'CPU reference'}`);}
async function act(fn){if(busy)return;busy=true;document.querySelectorAll('button').forEach(b=>b.disabled=true);try{await fn();}catch(e){error(e);}finally{busy=false;document.querySelectorAll('button').forEach(b=>b.disabled=false);}}
function backend(){need($('cpu').checked||gpu,'WebGPU is unavailable. Select CPU reference explicitly to run it.');session.gpu=$('cpu').checked?null:gpu;session.passive=$('passive').checked;}
function download(name,value){const a=document.createElement('a'),url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function localOnly(){need(!session.pc,'Close the peer before replacing the local checkpoint');}
$('run').onclick=()=>act(async()=>{backend();await session.stage({kind:'step',...packBits(Array.from({length:128},(_,i)=>i%2))});});
$('verify').onclick=()=>act(async()=>{need(gpu,'WebGPU is unavailable');const machine=new Machine(),rows=await machine.apply([{kind:'step',...packBits(Array.from({length:128},(_,i)=>i%2))}],gpu);need(pairHex(machine.kernel.word)==='11020439010204C7','Retained reference pair differs');log(`PASS · 128 GPU transitions / ${rows.trace.length*8} trace words · retained pair 11020439010204C7`);});
$('apply').onclick=()=>act(async()=>{backend();await session.stage({kind:'genes',genes:JSON.parse($('genes').value)});});
$('pinion').onclick=()=>{$('genes').value=JSON.stringify(pinionGenes,null,2);};$('tape').onclick=()=>{$('genes').value=JSON.stringify(tapeGenes,null,2);};
$('reset').onclick=()=>act(async()=>{localOnly();session.machine=new Machine();session.local();display();log('Default pinion checkpoint restored.');});
$('export').onclick=()=>download('checkpoint.json',session.machine.checkpoint());$('replay').onclick=()=>download('replay.json',session.replay());
$('import').onclick=()=>act(async()=>{localOnly();session.machine=new Machine(JSON.parse($('checkpoint').value));session.local();display();log('Checkpoint imported.');});
$('replayFile').onchange=()=>act(async()=>{localOnly();backend();const replay=JSON.parse(await $('replayFile').files[0].text());need(replay.protocol==='TK-EDGE-1'&&Array.isArray(replay.events)&&replay.events.length<=65536,'Invalid replay');const machine=new Machine(replay.initial),rounds=new AsyncHopRounds(replay.members,(batch)=>machine.apply(batch,session.gpu));for(const event of replay.events)await rounds.receive(event);need(rounds.pending.size===0,'Replay has incomplete rounds');session.machine=machine;session.local();display();log(`Replay verified · ${rounds.round} rounds`);});
$('offer').onclick=()=>act(async()=>{backend();$('signal').value=JSON.stringify(await session.offer(JSON.parse($('ice').value)),null,2);signalType='offer';display();log('Offer ready. Download it for the native host, or paste it on the other device.');});
$('answer').onclick=()=>act(async()=>{backend();$('signal').value=JSON.stringify(await session.answer(JSON.parse($('signal').value)),null,2);signalType='answer';display();log('Answer ready. Return it to the device that created the offer.');});
$('accept').onclick=()=>act(()=>session.accept(JSON.parse($('signal').value)));
$('saveSignal').onclick=()=>download(signalType+'.json',JSON.parse($('signal').value));$('signalFile').onchange=()=>act(async()=>{$('signal').value=await $('signalFile').files[0].text();});
$('disconnect').onclick=()=>{session.local();display();$('session').textContent='Local execution';log('Peer closed. A new local round log begins at the current checkpoint.');};
$('idle').onclick=()=>act(async()=>{backend();await session.stage({kind:'idle'});});$('passive').onchange=()=>{session.passive=$('passive').checked;};
$('genes').value=JSON.stringify({role:'pinion',turns:[11,53,137],gains:[...DEFAULT.gains]},null,2);$('trace').textContent='No transitions emitted.';display();
try{gpu=await GPUKernel.create();session.gpu=gpu;const info=gpu.adapter.info;$('adapter').textContent='WebGPU hardware · '+(info?.description||[info?.vendor,info?.architecture].filter(Boolean).join(' ')||'adapter available');}catch(e){$('adapter').textContent=e.message+' · CPU reference available by explicit selection';}
// Read-only diagnostics for automated conformance checks.
globalThis.kernelDiagnostics={get checkpoint(){return session.machine.checkpoint();},get gpu(){return gpu;},get session(){return session;}};
