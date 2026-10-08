"""Loopback-only browser/native integration fixture; never shipped in dist."""
import asyncio
import json
import pathlib
import sys
import threading
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'native'))
from backend import CUDA,ROOT
from peer import Peer
from aiortc import RTCPeerConnection,RTCConfiguration,RTCSessionDescription

async def main():
    loop=asyncio.get_running_loop();gpu=CUDA();state={};pcs=[];tasks=[]
    async def offer(p):
        pc=RTCPeerConnection(RTCConfiguration(iceServers=[]));pcs.append(pc);peer=Peer(p['checkpoint'],gpu);state['peer']=peer
        @pc.on('datachannel')
        def channel(c):peer.attach(c)
        tasks.append(asyncio.create_task(peer.process()))
        await pc.setRemoteDescription(RTCSessionDescription(sdp=p['sdp'],type='offer'));await pc.setLocalDescription(await pc.createAnswer())
        return dict(type='answer',protocol='TK-EDGE-1',sdp=pc.localDescription.sdp)
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*a,**kw):super().__init__(*a,directory=str(ROOT),**kw)
        def reply(self,p):
            b=json.dumps(p).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
        def do_POST(self):
            size=int(self.headers.get('Content-Length','0'))
            if size>1000000:self.send_error(413);return
            p=json.loads(self.rfile.read(size))
            if self.path=='/offer':self.reply(asyncio.run_coroutine_threadsafe(offer(p),loop).result(30))
            elif self.path=='/report':
                p['nativeAdapter']=gpu.name;(ROOT/'results').mkdir(exist_ok=True);(ROOT/'results/browser-peer.json').write_text(json.dumps(p,indent=2));self.reply(dict(saved=True))
            else:self.send_error(404)
        def do_GET(self):
            if self.path=='/result':
                p=state['peer'];self.reply(dict(round=p.rounds.round,error=p.error,checkpoint=p.machine.checkpoint()))
            else:super().do_GET()
    server=ThreadingHTTPServer(('127.0.0.1',8788),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
    print('Browser fixture: http://127.0.0.1:8788/tests/browser.html',flush=True)
    try:await asyncio.Event().wait()
    finally:
        server.shutdown()
        for task in tasks:task.cancel()
        for pc in pcs:await pc.close()
        gpu.close()
if __name__=='__main__':
    try:asyncio.run(main())
    except KeyboardInterrupt:pass
