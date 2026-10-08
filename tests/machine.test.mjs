import assert from 'node:assert/strict';
import{AsyncHopRounds,HopRounds,Pinion,Tape,packBits,unpackBits,canonical,DEFAULT}from'../dist/core.js';
import{Machine}from'../dist/machine.js';
const m=new Machine(),before=m.checkpoint();
await assert.rejects(()=>m.apply([{kind:'step',...packBits([1])},{kind:'genes',genes:{role:'pinion',turns:[999,0,0],gains:[0,0,0,0,0,0,0,0]}}]));
assert.deepEqual(m.checkpoint(),before);
for(let n=1;n<=4096;n++){const bits=Array.from({length:n},(_,i)=>i%2);assert.deepEqual(unpackBits(packBits(bits)),bits);}
assert.throws(()=>unpackBits({count:1,bits:'Aw'}));
const tape={role:'tape',...new Tape().snapshot()};await m.apply([{kind:'genes',genes:tape}]);assert.equal(m.role,'tape');
await m.apply([{kind:'step',...packBits(Array(8).fill(0))}]);assert.equal(m.kernel.status,'HALT');assert.equal(m.kernel.tick,4);
await m.apply([{kind:'genes',genes:{role:'pinion',...DEFAULT}}]);assert.deepEqual(m.checkpoint(),before);
const invalidTape={role:'tape',states:2,halt:1,q:0,head:0,cells:[],rules:[]};await m.apply([{kind:'genes',genes:invalidTape}]);const unchanged=m.checkpoint();await assert.rejects(()=>m.apply([{kind:'step',...packBits([0])}]));assert.deepEqual(m.checkpoint(),unchanged);
const deferred={role:'tape',states:2,halt:1,q:0,head:2047,cells:[],rules:[[0,0,0,1,0]]};await m.apply([{kind:'genes',genes:{role:'pinion',...DEFAULT}}]);await m.apply([{kind:'genes',genes:deferred}]);const boundary=m.checkpoint();await assert.rejects(()=>m.apply([{kind:'step',...packBits([0])}]));assert.deepEqual(m.checkpoint(),boundary);
let unblock;const gate=new Promise(r=>unblock=r);const committed=[],rounds=new AsyncHopRounds(['A','B'],async(batch,r)=>{await gate;committed.push(r);return batch;});
const a={member:'A',round:0,kind:'idle'},b={member:'B',round:0,kind:'idle'};await rounds.receive(a);const queued=rounds.receive(b);await Promise.resolve();assert.equal(rounds.round,0);unblock();await queued;assert.equal(rounds.round,1);assert.deepEqual(committed,[0]);
await rounds.receive({kind:'idle',round:0,member:'A'});assert.equal(rounds.round,1);
await assert.rejects(()=>rounds.receive({...a,kind:'genes',genes:{}}));assert.equal(rounds.round,1);await assert.rejects(()=>rounds.receive({member:'A',round:1,kind:'idle'}));
const budget=new HopRounds(['A'],()=>0,1);budget.receive(a);assert.throws(()=>budget.receive({...a,round:1}));
console.log(JSON.stringify({result:'PASS',bitPackingLengths:4096,atomicRejections:3,asyncCommitFence:true,typedRoleExchange:true,duplicateRejection:true}));
