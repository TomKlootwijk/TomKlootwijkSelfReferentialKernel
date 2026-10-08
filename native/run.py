import argparse
import json
import pathlib
import sys
from backend import CUDA,WebGPU,ROOT
from protocol import Machine,canonical,b64,unb64

def main():
    ap=argparse.ArgumentParser(description='Native integer kernel: CUDA or WebGPU, no renderer.')
    ap.add_argument('--backend',choices=['texture','cuda','wgpu'],default='texture');ap.add_argument('--ticks',type=int,default=128);ap.add_argument('--checkpoint',type=pathlib.Path);ap.add_argument('--out',type=pathlib.Path,default=ROOT/'results/checkpoint.json');ap.add_argument('--tape',action='store_true');args=ap.parse_args()
    if not 1<=args.ticks<=4096:ap.error('ticks must be 1..4096')
    m=Machine(json.loads(args.checkpoint.read_text()) if args.checkpoint else None)
    if args.tape:m=Machine(dict(role='tape',data=dict(states=2,halt=1,q=0,head=0,cells=[[0,1],[1,1],[2,1]],rules=[[0,1,1,1,0],[0,0,1,0,1]])))
    backend=CUDA(texture=args.backend=='texture') if args.backend!='wgpu' else WebGPU()
    try:
        bits=[i%2 for i in range(args.ticks)];packed=bytearray((len(bits)+7)//8)
        for i,b in enumerate(bits):packed[i//8]|=b<<(i%8)
        result=m.apply([dict(kind='step',count=len(bits),bits=b64(packed))],backend)
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result['checkpoint'],indent=2))
        report=dict(backend=args.backend,adapter=backend.name,role=m.role,tick=m.kernel.state[1] if m.role=='pinion' else m.kernel.tick,checkpoint=str(args.out),comparedTraceWords=sum(map(len,result['trace'])))
        if m.role=='pinion':
            from protocol import mirror
            report['pair']=f'{mirror(m.kernel.state[0]):08X}{m.kernel.state[0]:08X}';report['packetBytes']=len(m.kernel.seed())
        print(json.dumps(report,indent=2))
    finally:backend.close()
if __name__=='__main__':main()
