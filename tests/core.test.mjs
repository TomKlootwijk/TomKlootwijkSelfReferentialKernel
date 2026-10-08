import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {Pinion,Tape,HopRounds,DEFAULT,encodeSeed,decodeSeed,packWorld,pinionParams,packedState,pairHex,hex,mirror,unpack,base64,unbase64} from '../dist/core.js';

const retained=JSON.parse(readFileSync(new URL('./pinion_results_new.json',import.meta.url)));
const defaultKernel=new Pinion(),fixtures=[];
const bits=Array.from({length:128},(_,i)=>i%2),initial=packedState(defaultKernel);
const trace=bits.map(b=>defaultKernel.step(b));
for(let i=0;i<128;i++){
 const row=retained.trace[i],t=trace[i];
 assert.equal(hex(t[1])+hex(t[0]),row.pair);
 assert.equal(t[2],row.row);assert.equal(hex(t[3]),row.before);assert.equal(hex(t[4]),row.after);
 assert.deepEqual(t.slice(5,7),row.q.map(x=>x>>>0));assert.equal(t[7],row.cost);
}
assert.equal(pairHex(defaultKernel.word),'11020439010204C7');
assert.deepEqual(defaultKernel.world.phi,retained.field);
fixtures.push({name:'PDF-retained-default-128',world:[...packWorld(defaultKernel.world)],bits,initial:[...initial],params:[...pinionParams(defaultKernel,128)],state:[...packedState(defaultKernel)],trace:trace.flat()});
// Test entropy belongs to fixtures; the execution kernel consumes only explicit bits.
let random=20261007;const rand=n=>{random=(Math.imul(random,1664525)+1013904223)>>>0;return random%n;};
for(let c=0;c<64;c++){
 const width=3+rand(8),height=3+rand(8),N=width*height;
 const config={width,height,center:rand(N),radius:1,node:rand(N),phase:rand(256),orientation:rand(2)};
 if(c>=32){config.turns=Array.from({length:3},()=>rand(256));config.gains=Array.from({length:8},()=>rand(9)-4);}
 const k=new Pinion(config),before=packedState(k),j=Array.from({length:128},()=>rand(2)),rows=j.map(x=>k.step(x));
 fixtures.push({name:'variant-'+c,config,world:[...packWorld(k.world)],bits:j,initial:[...before],params:[...pinionParams(k,128)],state:[...packedState(k)],trace:rows.flat()});
 assert.deepEqual(packedState(decodeSeed(encodeSeed(k))),packedState(k));
 assert.equal(mirror(mirror(k.word)),k.word);assert.equal(unpack(k.word)[2],k.world.phi[unpack(k.word)[1]]);
}
const blank=new Pinion();assert.equal(encodeSeed(blank).length,32);assert.equal(encodeSeed(defaultKernel).length,128);
assert.deepEqual([...unbase64(base64(encodeSeed(defaultKernel)))],[...encodeSeed(defaultKernel)]);
for(const invalid of [true,2,-1,0.1,undefined])assert.throws(()=>blank.step(invalid));
const corrupt=encodeSeed(defaultKernel);corrupt[12]=2;assert.throws(()=>decodeSeed(corrupt));
assert.throws(()=>blank.setGenes({turns:[1,2,3],gains:[5,1,1,1,1,1,1,1]}));
// All 720 arrival permutations include delivery of round 2 before round 0.
const packets=Array.from({length:3},(_,round)=>['alpha','beta'].map((member,i)=>({round,member,kind:'step',bits:[(round+i)%2]}))).flat();
const permute=(a,fn,p=[])=>{if(!a.length)return fn(p);for(let i=0;i<a.length;i++)permute([...a.slice(0,i),...a.slice(i+1)],fn,[...p,a[i]]);};
let baseline,permutations=0;
permute(packets,p=>{const k=new Pinion(),observed=[],h=new HopRounds(['beta','alpha'],batch=>{const rows=batch.flatMap(x=>x.bits.map(b=>k.step(b)));observed.push(...rows);return rows;});for(const x of p){h.receive(x);h.receive(x);}assert.equal(h.round,3);baseline??=observed;assert.deepEqual(observed,baseline);permutations++;});
const tm=new Tape();const tmTrace=[];for(let i=0;i<5;i++){const t=tm.step();if(t)tmTrace.push(t);}assert.equal(tm.status,'HALT');assert.equal(tm.tick,4);assert.equal(tm.cells.get(3),1);
assert.throws(()=>new Tape({states:2,halt:1,rules:[[0,0,0,1,0],[0,0,1,1,1]]}));
const missing=new Tape({states:2,halt:1,rules:[]});assert.equal(missing.step(),null);assert.equal(missing.status,'INVALID');
writeFileSync(new URL('./fixtures.json',import.meta.url),JSON.stringify(fixtures));
console.log(JSON.stringify({result:'PASS',retainedTicks:128,variants:64,variantTicks:8192,arrivalPermutations:permutations,defaultPacketBytes:32,mutatedPacketBytes:128,finalPair:pairHex(defaultKernel.word)}));
