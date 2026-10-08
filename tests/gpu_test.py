import argparse
import json
import pathlib
import sys
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1] / 'native'))
from backend import CUDA,WebGPU,ROOT

def check(eq,message):
    if not eq: raise AssertionError(message)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=['cuda','texture','wgpu'],default='cuda');ap.add_argument('--chains',type=int,default=4096);args=ap.parse_args()
    gpu=CUDA(texture=args.backend=='texture') if args.backend!='wgpu' else WebGPU()
    fixtures=json.loads((ROOT/'tests/fixtures.json').read_text())
    count=0
    try:
        for f in fixtures:
            result=gpu.execute('pinion',f['world'],f['bits'],f['initial'],f['params'],len(f['trace']))
            check(np.array_equal(result['state'],f['state']),f['name']+' state mismatch')
            check(np.array_equal(result['trace'],f['trace']),f['name']+' trace mismatch')
            count+=len(f['bits'])
        f=fixtures[0];chains=args.chains
        check(1<=chains<=32768,'Invalid chain budget')
        params=f['params'][:];params[2]=chains
        result=gpu.execute('pinion',f['world'],np.tile(f['bits'],chains),np.tile(f['initial'],chains),params,len(f['trace'])*chains)
        check(np.array_equal(result['state'],np.tile(f['state'],chains)),'Parallel state mismatch')
        check(np.array_equal(result['trace'],np.tile(f['trace'],chains)),'Parallel trace mismatch')
        # Unary increment, boundary DEFER, and missing transition are distinct.
        capacity=8;program=np.zeros(2*256*3,dtype=np.uint32);program[2::3]=0xffffffff
        program[0:3]=[1,0,1];program[3:6]=[1,1,0]
        state=np.zeros(4+capacity,dtype=np.uint32);state[:4]=[0,2,0,0];state[6:9]=1
        tape=gpu.execute('tape',program,[0],state,[2,16,1,1,capacity],64)
        check(tape['state'][:4].tolist()==[1,5,4,1],'Unary increment halt mismatch')
        check(tape['trace'][:16].tolist()==[0,3,1,1,0,4,1,2,0,5,1,3,1,5,1,4],'Tape trace mismatch')
        loop=program.copy();loop[0:3]=[0,1,0];state[:]=0;state[1]=capacity-1
        deferred=gpu.execute('tape',loop,[0],state,[2,16,1,1,capacity],64)
        check(deferred['state'][:4].tolist()==[0,7,0,3],'Boundary did not defer atomically')
        bad=program.copy();bad[2]=0xffffffff;state[:]=0
        invalid=gpu.execute('tape',bad,[0],state,[2,16,1,1,capacity],64)
        check(invalid['state'][:4].tolist()==[0,0,0,2],'Missing transition not INVALID')
        report=dict(result='PASS',backend=args.backend,adapter=gpu.name,comparedTicks=count,parallelChains=chains,parallelTicks=chains*128,parallelSecondsIncludingTransfers=result['seconds'],tapeChecks=3)
        if 'kernel_seconds' in result:report['launchAndSyncSeconds']=result['kernel_seconds']
        (ROOT/'results').mkdir(exist_ok=True);(ROOT/f'results/{args.backend}.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
    finally:gpu.close()
if __name__=='__main__':main()
