import copy
import itertools
import json
import pathlib
import sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'native'))
from backend import CUDA,ROOT
from protocol import Pinion,Machine,Rounds,b64,require

gpu=CUDA()
try:
    fixtures=json.loads((ROOT/'tests/fixtures.json').read_text())
    for f in fixtures:
        k=Pinion(f.get('config'));require(k.world['packed']==f['world'],'Host world differs: '+f['name'])
        require(k.gpu(f['bits'],gpu)==[f['trace'][i:i+8] for i in range(0,len(f['trace']),8)],'Native protocol trace differs')
        require(k.state==f['state'],'Native protocol state differs')
        require(Pinion.from_seed(k.seed()).state==k.state,'Checkpoint roundtrip differs')
    packets=[dict(round=r,member=m,kind='step',count=1,bits=b64(bytes([(r+i)%2]))) for r in range(2) for i,m in enumerate(('A','B'))]
    baseline=None
    for order in itertools.permutations(packets):
        m=Machine();rounds=Rounds(m,gpu)
        for p in order:rounds.receive(p);rounds.receive(p)
        require(rounds.round==2,'Incomplete rounds')
        baseline=baseline or m.checkpoint();require(m.checkpoint()==baseline,'Arrival-dependent output')
    m=Machine();before=m.checkpoint()
    try:m.apply([dict(kind='step',count=1,bits='AQ'),dict(kind='genes',genes=dict(role='pinion',turns=[-1,0,0],gains=[0]*8))],gpu)
    except ValueError:pass
    else:raise AssertionError('Invalid genes admitted')
    require(m.checkpoint()==before,'Invalid batch mutated state')
    print(json.dumps(dict(result='PASS',nativeProfiles=65,nativeArrivalPermutations=24,atomicRejection=True)))
finally:gpu.close()
