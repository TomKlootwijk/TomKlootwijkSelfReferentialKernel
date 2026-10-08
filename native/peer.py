"""Headless WebRTC answerer: exchange a signaling JSON file, then GPU hops."""
import argparse
import asyncio
import json
import pathlib
from aiortc import RTCPeerConnection,RTCSessionDescription,RTCConfiguration,RTCIceServer
from backend import CUDA,WebGPU,ROOT
from protocol import Machine,Rounds,canonical

class Peer:
    def __init__(self,checkpoint,backend,passive=True):
        self.initial=checkpoint;self.machine=Machine(checkpoint);self.rounds=Rounds(self.machine,backend);self.passive=passive;self.channel=None;self.ready=False;self.receipts={};self.log=[];self.queue=asyncio.Queue();self.error=None
    def send(self,p):
        if self.channel and self.channel.readyState=='open':self.channel.send(canonical(p))
    def attach(self,channel):
        self.channel=channel
        @channel.on('open')
        def opened():self.send(dict(type='hello',protocol='TK-EDGE-1',checkpoint=self.initial))
        @channel.on('message')
        def received(message):self.queue.put_nowait(message)
        if channel.readyState=='open':opened()
    def verify_receipts(self):
        for r,checkpoint in list(self.receipts.items()):
            if r in self.commits:
                if canonical(checkpoint)!=canonical(self.commits[r]):raise ValueError('Peer checkpoint mismatch')
                del self.receipts[r]
    async def process(self):
        self.commits={}
        while True:
            raw=await self.queue.get()
            try:
                if self.error:continue
                if not isinstance(raw,str) or len(raw)>1000000:raise ValueError('Message budget exceeded')
                p=json.loads(raw)
                if p['type']=='hello':
                    if p.get('protocol')!='TK-EDGE-1' or canonical(p['checkpoint'])!=canonical(self.initial):raise ValueError('Initial checkpoint differs')
                    self.ready=True;print('Peer ready. Both participants have the same checkpoint.',flush=True)
                elif p['type']=='hop':
                    if not self.ready:raise ValueError('Peer is not initialized')
                    e=p['event'];r=e['round'];out=self.rounds.receive(e)
                    self.log.append(e)
                    if self.passive and r>=self.rounds.round and e['member']=='A' and 'B' not in self.rounds.pending.get(r,{}):
                        idle=dict(round=r,member='B',kind='idle');self.log.append(idle);self.send(dict(type='hop',event=idle));out+=self.rounds.receive(idle)
                    for commit in out:
                        self.commits[commit['round']]=commit['checkpoint'];self.send({k:v for k,v in commit.items() if k!='trace'})
                        (ROOT/'results').mkdir(exist_ok=True)
                        (ROOT/'results/peer-checkpoint.json').write_text(json.dumps(self.machine.checkpoint(),indent=2))
                        (ROOT/'results/peer-replay.json').write_text(json.dumps(dict(protocol='TK-EDGE-1',initial=self.initial,members=['A','B'],events=self.log),indent=2))
                        print(f"ACCEPT round={commit['round']} role={self.machine.role} trace_words={sum(map(len,commit['trace']))}",flush=True)
                    self.verify_receipts()
                elif p['type']=='commit':
                    if not 0<=p['round']<4096 or len(self.receipts)>=64:raise ValueError('Receipt budget exceeded')
                    self.receipts[p['round']]=p['checkpoint'];self.verify_receipts()
                elif p['type']=='error':raise ValueError('Peer halted: '+str(p.get('message')))
                else:raise ValueError('Unknown message')
            except Exception as e:
                self.error=str(e);self.send(dict(type='error',message=self.error));print('Session halted: '+self.error,flush=True)

async def run(args):
    offer=json.loads(args.offer.read_text());checkpoint=offer['checkpoint'];backend=CUDA(texture=args.backend=='texture') if args.backend!='wgpu' else WebGPU()
    servers=[RTCIceServer(**s) for s in offer.get('iceServers',[])]
    pc=RTCPeerConnection(RTCConfiguration(iceServers=servers));peer=Peer(checkpoint,backend)
    @pc.on('datachannel')
    def channel(c):peer.attach(c)
    @pc.on('connectionstatechange')
    async def state():print('Connection: '+pc.connectionState,flush=True)
    task=asyncio.create_task(peer.process())
    try:
        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer['sdp'],type='offer'))
        await pc.setLocalDescription(await pc.createAnswer())
        args.answer.parent.mkdir(parents=True,exist_ok=True)
        args.answer.write_text(json.dumps(dict(type='answer',sdp=pc.localDescription.sdp,protocol='TK-EDGE-1')))
        print(f'Answer saved: {args.answer}\nLoad that file into the browser. This process now waits for GPU hops. Ctrl+C closes it.',flush=True)
        await asyncio.Event().wait()
    finally:
        task.cancel();await pc.close();backend.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--offer',type=pathlib.Path,required=True);ap.add_argument('--answer',type=pathlib.Path,default=ROOT/'results/answer.json');ap.add_argument('--backend',choices=['texture','cuda','wgpu'],default='texture');args=ap.parse_args()
    try:asyncio.run(run(args))
    except KeyboardInterrupt:pass
