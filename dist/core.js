// TK-EDGE-1: exact integer semantics; no random or wall-clock input in execution.
export const DEFAULT = Object.freeze({width:8,height:8,radius:2,center:0,node:0,phase:250,orientation:0,turns:[11,53,137],gains:[1,1,-1,1,-1,-1,1,-1]});
const int=(x,a,b)=>Number.isInteger(x)&&x>=a&&x<=b;
export function need(ok,message){if(!ok)throw Error(message);}
export function popcount(x){x>>>=0;x-=x>>>1&0x55555555;x=(x&0x33333333)+(x>>>2&0x33333333);return Math.imul((x+(x>>>4)&0x0f0f0f0f),0x01010101)>>>24;}
export function pack(r,i,b,a){need(int(r,0,255)&&int(i,0,255)&&int(b,-128,127)&&int(a,0,127),'Invalid RP32 lanes');let v=(r|i<<8|(b&255)<<16|a<<24)>>>0;return(v|((popcount(v)&1)<<31))>>>0;}
export function unpack(w){need(int(w,0,0xffffffff)&&!(popcount(w)&1),'Invalid RP32 parity');const b=w>>>16&255;return[w&255,w>>>8&255,b>127?b-256:b,w>>>24&127];}
export function mirror(w){const[r,i,b,a]=unpack(w);return pack((-r)&255,i,b,a^16);}
export const hex=x=>(x>>>0).toString(16).padStart(8,'0').toUpperCase();
export const pairHex=w=>hex(mirror(w))+hex(w);
export const fieldClass=b=>b<0?0:b===0?1:2;
const mod=(a,b)=>(a%b+b)%b;
const gcd=(a,b)=>{while(b)[a,b]=[b,a%b];return a;};
const lex=(a,b)=>{for(let i=0;i<a.length;i++)if(a[i]!==b[i])return a[i]-b[i];return 0;};
export function world(config=DEFAULT){
 const c={...DEFAULT,...config},W=c.width,H=c.height,N=W*H;
 need(int(W,3,85)&&int(H,3,85)&&N<=256&&int(c.center,0,N-1)&&int(c.radius,1,127),'Invalid world');
 const directions=[[1,0],[-1,0],[0,1],[0,-1]];
 const canonical=(u,v)=>{const q=Math.floor(u/W);return mod(u,W)*H+mod(q&1?-v:v,H);};
 const edges=Array.from({length:N},(_,i)=>directions.map(([du,dv])=>{const u=Math.floor(i/H)+du,v=i%H+dv;return[canonical(u,v),Math.floor(u/W)&1];}));
 const distance=seeds=>{const d=Array(N).fill(-1),queue=[...seeds];for(const i of queue)d[i]=0;for(let p=0;p<queue.length;p++)for(const[j]of edges[queue[p]])if(d[j]<0){d[j]=d[queue[p]]+1;queue.push(j);}return d;};
 const dc=distance([c.center]);need(c.radius<Math.max(...dc),'Radius must leave a positive side');
 const signs=dc.map(d=>Math.sign(d-c.radius)),zero=signs.flatMap((s,i)=>s===0?[i]:[]),bd=distance(zero),phi=bd.map((d,i)=>d*signs[i]);
 const axes=[],factors=[];
 for(let i=0;i<N;i++){const e=edges[i].map(x=>x[0]),g=[phi[e[0]]-phi[e[1]],phi[e[2]]-phi[e[3]]],div=gcd(Math.abs(g[0]),Math.abs(g[1])),psi=div?g.map(x=>x/div):[1,0];axes.push(psi);factors.push(g.map((x,j)=>(1+psi[j]**2)*x));need(Math.abs(phi[i])<=127&&e.every(j=>Math.abs(phi[i]-phi[j])<=1),'Field certificate');}
 const ip=Array(N).fill(0);for(const i of Array.from({length:N},(_,i)=>i).sort((a,b)=>dc[a]-dc[b]||a-b))if(i!==c.center){const parent=Math.min(...edges[i].map(x=>x[0]).filter(j=>dc[j]===dc[i]-1));ip[i]=(ip[parent]+DEFAULT.turns[fieldClass(phi[parent])])&255;}
 const keys=axes.map((p,i)=>[p[0]+2,p[1]+2,Math.floor(Math.log2(dc[i]+1)),ip[i],i]),ordered=keys.map((_,i)=>i).sort((a,b)=>lex(keys[a],keys[b]));
 return{...c,N,edges,phi,factors,keys,ordered,directions};
}
export class Pinion{
 constructor(config=DEFAULT){this.world=world(config);const c=this.world;need(int(c.node,0,c.N-1)&&int(c.phase,0,255)&&int(c.orientation,0,1),'Invalid live state');this.setGenes(c);this.word=pack(c.phase,c.node,c.phi[c.node],1|c.orientation<<4);this.tick=0;this.epoch=0;this.overlay=new Uint32Array(Math.ceil(12*c.N/32));}
 setGenes(g){need(Array.isArray(g.turns)&&g.turns.length===3&&g.turns.every(x=>int(x,0,255))&&Array.isArray(g.gains)&&g.gains.length===8&&g.gains.every(x=>int(x,-4,4)),'Invalid pinion genes');this.turns=[...g.turns];this.gains=[...g.gains];}
 rowWord(row){need(int(row,0,12*this.world.N-1),'Invalid row');const i=Math.floor(row/12),c=Math.floor(row%12/4),s=row%4,[d,t]=this.world.edges[i][s],bit=this.overlay[row>>>5]>>>(row&31)&1;return pack(this.turns[c]^bit,d,this.world.phi[d],1|t<<6);}
 step(j){need(int(j,0,1),'Jitter must be one bit');need(this.tick<0xffffffff&&this.epoch<0xffffffff,'Counter budget exhausted');const w=this.world,[r,i,b,a]=unpack(this.word),eta=a>>>4&1,theta=(eta?-r:r)&255,bank=theta>>>6,q=w.factors[i].map((v,k)=>v*this.gains[2*bank+k]);
  let lo=0,hi=w.N,found=-1;while(lo<hi){const mid=Math.floor((lo+hi-1)/2),id=w.ordered[mid],cmp=lex(w.keys[i],w.keys[id]);if(!cmp){found=id;break;}if(cmp<0)hi=mid;else lo=mid+1;}need(found===i,'f8 lookup failed');
  const scored=w.edges[i].map(([d],s)=>[Math.abs(w.phi[d])-q[0]*w.directions[s][0]-q[1]*w.directions[s][1],d,s]).sort(lex),[,d,s]=scored[0],row=12*i+4*fieldClass(b)+s,old=this.rowWord(row),delta=old&255,tau=w.edges[i][s][1],ne=eta^tau,nt=(theta+delta)&255,nr=(ne?-nt:nt)&255,nw=pack(nr,d,w.phi[d],1|ne<<4);
  this.overlay[row>>>5]=(this.overlay[row>>>5]^(j<<(row&31)))>>>0;this.word=nw;this.tick++;this.epoch+=j;
  return[nw,mirror(nw),row,old,this.rowWord(row),q[0]>>>0,q[1]>>>0,1+Math.abs(w.phi[d])+Math.max(...q.map(Math.abs))-q[0]*w.directions[s][0]-q[1]*w.directions[s][1]];
 }
 clone(){return decodeSeed(encodeSeed(this));}
}
export function encodeSeed(k){const w=k.world,bytes=new Uint8Array(32+(k.overlay.some(Boolean)?k.overlay.length*4:0));bytes.set([84,75,71,49,1,0,w.width,w.height,w.radius,w.center]);
 const[r,i,,a]=unpack(k.word);bytes[10]=i;bytes[11]=r;bytes[12]=a>>>4&1;bytes.set(k.turns,13);bytes.set(k.gains.map(x=>x&255),16);const d=new DataView(bytes.buffer);d.setUint32(24,k.tick,true);d.setUint32(28,k.epoch,true);for(let i=0;32+4*i<bytes.length;i++)d.setUint32(32+4*i,k.overlay[i],true);return bytes;}
export function decodeSeed(bytes){need(bytes instanceof Uint8Array&&bytes.length>=32&&bytes.slice(0,6).every((v,i)=>v===[84,75,71,49,1,0][i]),'Invalid TKG1 pinion packet');const d=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength),k=new Pinion({width:bytes[6],height:bytes[7],radius:bytes[8],center:bytes[9],node:bytes[10],phase:bytes[11],orientation:bytes[12],turns:Array.from(bytes.slice(13,16)),gains:Array.from(bytes.slice(16,24),v=>v>127?v-256:v)});need(bytes.length===32||bytes.length===32+4*k.overlay.length,'Invalid overlay length');k.tick=d.getUint32(24,true);k.epoch=d.getUint32(28,true);need(k.epoch<=k.tick,'Invalid epoch');if(bytes.length>32)for(let i=0;i<k.overlay.length;i++)k.overlay[i]=d.getUint32(32+4*i,true);const used=12*k.world.N%32;if(used)need((k.overlay.at(-1)>>>used)===0,'Invalid padding bits');return k;}
export function base64(bytes){let s='';for(const b of bytes)s+=String.fromCharCode(b);return btoa(s).replaceAll('+','-').replaceAll('/','_').replace(/=+$/,'');}
export function unbase64(s){need(typeof s==='string'&&s.length<=1000000&&/^[\w-]+$/.test(s),'Invalid packet text');return Uint8Array.from(atob(s.replaceAll('-','+').replaceAll('_','/')),c=>c.charCodeAt(0));}
export function packWorld(w){return Uint32Array.from([...w.phi,...w.edges.flat(2),...w.factors.flat(),...w.keys.flat(),...w.ordered],v=>v>>>0);}
export function pinionParams(k,steps,chains=1){return Uint32Array.from([k.world.N,steps,chains,k.turns[0],k.turns[1],k.turns[2],...k.gains.map(x=>x>>>0)]);}
export function packedState(k){return Uint32Array.from([k.word,k.tick,k.epoch,0,...k.overlay]);}
export function installState(k,s){need(s[3]===0,'GPU transition rejected');unpack(s[0]);need(s[1]>=k.tick&&s[2]>=k.epoch,'Invalid GPU counters');k.word=s[0];k.tick=s[1];k.epoch=s[2];k.overlay.set(s.subarray(4));}
// U-TAPE-1: finite-prefix byte-symbol TM interpreter. Growing capacity extends the family.
export class Tape{
 constructor(g={states:2,halt:1,q:0,head:0,cells:[[0,1],[1,1],[2,1]],rules:[[0,1,1,1,0],[0,0,1,0,1]]}){this.setGenes(g);this.q=g.q??0;this.head=g.head??0;need(int(this.q,0,this.states-1)&&Number.isSafeInteger(this.head),'Invalid tape state');need(Array.isArray(g.cells??[]),'Invalid tape cells');this.cells=new Map;const seen=new Set;for(const row of g.cells??[]){need(Array.isArray(row)&&row.length===2,'Invalid tape cell');const[h,b]=row;need(Number.isSafeInteger(h)&&int(b,0,255)&&!seen.has(h),'Invalid/duplicate tape cell');seen.add(h);if(b)this.cells.set(h,b);}this.tick=0;this.status='READY';}
 setGenes(g){need(int(g.states,1,4096)&&int(g.halt,0,g.states-1)&&Array.isArray(g.rules)&&g.rules.length<=g.states*256,'Invalid tape genes');const rules=new Map;for(const row of g.rules){need(Array.isArray(row)&&row.length===5,'Invalid transition');const[q,a,b,d,n]=row;need(int(q,0,g.states-1)&&int(a,0,255)&&int(b,0,255)&&int(d,-1,1)&&int(n,0,g.states-1)&&!rules.has(q*256+a),'Invalid/duplicate transition');rules.set(q*256+a,[b,d,n]);}if(this.q!==undefined)need(this.q<g.states,'Gene change invalidates control');this.states=g.states;this.halt=g.halt;this.rules=rules;}
 step(){if(this.q===this.halt){this.status='HALT';return null;}const a=this.cells.get(this.head)??0,t=this.rules.get(this.q*256+a);if(!t){this.status='INVALID';return null;}const[b,d,q]=t;need(Number.isSafeInteger(this.head+d)&&this.tick<0xffffffff,'Tape budget exhausted');if(b)this.cells.set(this.head,b);else this.cells.delete(this.head);this.head+=d;this.q=q;this.tick++;this.status='ACCEPT';return[this.q,this.head,b,this.tick];}
 genes(){return{states:this.states,halt:this.halt,rules:[...this.rules].map(([key,[b,d,q]])=>[Math.floor(key/256),key%256,b,d,q])};}
 snapshot(){return{...this.genes(),q:this.q,head:this.head,cells:[...this.cells].sort((a,b)=>a[0]-b[0]),tick:this.tick,status:this.status};}
}
// Closed logical rounds: all member slots are needed before any emission.
export class HopRounds{
 constructor(members,apply,maxRounds=4096){need(Array.isArray(members)&&members.length>=1&&members.length<=16&&new Set(members).size===members.length&&members.every(m=>typeof m==='string'&&/^[\w-]{1,48}$/.test(m))&&int(maxRounds,1,65536),'Invalid membership/budget');this.members=[...members].sort();this.apply=apply;this.maxRounds=maxRounds;this.round=0;this.pending=new Map;this.accepted=new Map;this.failed=null;}
 admit(packet){need(!this.failed,this.failed);need(packet&&int(packet.round,0,this.maxRounds-1)&&this.members.includes(packet.member),'Invalid hop envelope');need(['step','genes','idle'].includes(packet.kind),'Invalid hop kind');const text=canonical(packet);need(text.length<=65536,'Hop payload budget exceeded');const key=packet.round+':'+packet.member;if(this.accepted.has(key)){if(this.accepted.get(key)!==text)this.failed='Contradictory duplicate hop';need(!this.failed,this.failed);return false;}need(packet.round>=this.round&&packet.round<=this.round+64,'Hop window exceeded');const slots=this.pending.get(packet.round)??new Map;if(slots.has(packet.member)&&slots.get(packet.member).text!==text)this.failed='Contradictory duplicate hop';need(!this.failed,this.failed);slots.set(packet.member,{packet:JSON.parse(text),text});this.pending.set(packet.round,slots);return true;}
 ready(){return this.members.every(m=>this.pending.get(this.round)?.has(m));}
 batch(){return this.members.map(m=>this.pending.get(this.round).get(m).packet);}
 commit(){const ready=this.pending.get(this.round);for(const m of this.members)this.accepted.set(this.round+':'+m,ready.get(m).text);this.pending.delete(this.round);this.round++;}
 receive(packet){if(!this.admit(packet))return[];const emitted=[];while(this.ready()){const result=this.apply(this.batch(),this.round);need(!result?.then,'Use AsyncHopRounds for GPU execution');this.commit();emitted.push(result);}return emitted;}
}
export function canonical(value){if(Array.isArray(value))return'['+value.map(canonical).join(',')+']';if(value&&typeof value==='object')return'{'+Object.keys(value).sort().map(k=>JSON.stringify(k)+':'+canonical(value[k])).join(',')+'}';need(value!==undefined&&!(typeof value==='number'&&!Number.isFinite(value)),'Noncanonical value');return JSON.stringify(value);}
export class AsyncHopRounds extends HopRounds{
 constructor(...args){super(...args);this.tail=Promise.resolve();}
 receive(packet){const run=async()=>{if(!this.admit(packet))return[];const emitted=[];while(this.ready()){let result;try{result=await this.apply(this.batch(),this.round);}catch(e){this.failed=e.message;throw e;}need(!this.failed,this.failed);this.commit();emitted.push(result);}return emitted;};const result=this.tail.then(run);this.tail=result.catch(()=>{});return result;}
}
export function packBits(bits){need(Array.isArray(bits)&&bits.length>=1&&bits.length<=4096&&bits.every(x=>int(x,0,1)),'Invalid mutation bits');const bytes=new Uint8Array(Math.ceil(bits.length/8));bits.forEach((b,i)=>bytes[i>>>3]|=b<<(i&7));return{count:bits.length,bits:base64(bytes)};}
export function unpackBits(event){need(int(event.count,1,4096),'Invalid tick count');const bytes=unbase64(event.bits);need(bytes.length===Math.ceil(event.count/8),'Invalid packed bits length');if(event.count%8)need((bytes.at(-1)>>>event.count%8)===0,'Invalid packed bits padding');return Array.from({length:event.count},(_,i)=>bytes[i>>>3]>>>(i&7)&1);}
