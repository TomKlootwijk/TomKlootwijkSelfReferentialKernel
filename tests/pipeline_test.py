"""Full checkpoint comparisons across asynchronous buffer and slot reuse."""
import json
import pathlib
import sys
import tempfile
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'native'))
from paged import DiskPages, TexturePager, ROOT
from pipelined import PipelinedPager
from protocol import Pinion
from seed_inputs import KEYPHRASES,phrase_digest

def compare(engine,target):
    store=engine.store
    buffer=np.empty(store.page_chains*store.stride,dtype=np.uint32)
    for page in range(store.page_count):
        assert store.read_into(page,buffer)==target
        for index,state in enumerate(buffer[:store.count(page)*store.stride].reshape((-1,store.stride))):
            assert state.tolist()==engine.reference(page*store.page_chains+index,target)

def run(store,pages,target,slots,engine_type,expected_steps=None):
    engine=engine_type(store,slots*store.page_chains*store.stride*4)
    try:
        for index,page in enumerate(pages):
            if isinstance(engine,PipelinedPager) and index+1<len(pages):engine.prefetch(pages[index+1])
            engine.advance(page,target,quantum=8)
        engine.flush_all()
        compare(engine,target)
        if expected_steps is not None:assert engine.steps_executed==expected_steps
        return {k:v['sha256'] for k,v in store.manifest['pages'].items()}
    finally:engine.close()

def main():
    (ROOT/'results').mkdir(exist_ok=True)
    temp=tempfile.TemporaryDirectory(dir=ROOT/'results',prefix='pipeline-test-')
    location=pathlib.Path(temp.name).resolve()
    assert location.is_relative_to((ROOT/'results').resolve())
    compared=0
    try:
        for index,digest in enumerate([None]+[phrase_digest(x) for x in KEYPHRASES]):
            profile=Pinion(dict(width=3,height=3,radius=1,phase=255,orientation=1,
                turns=[255,0,137],gains=[-4,4,0,-1,2,3,-2,-3])) if index%2 else None
            with DiskPages(location/f'a-{index}',79,17,prototype=profile,digest=digest) as serial, DiskPages(location/f'b-{index}',79,17,prototype=profile,digest=digest) as pipeline:
                pages=list(range(serial.page_count))
                assert run(serial,pages,19,1,TexturePager)==run(pipeline,list(reversed(pages)),19,1,PipelinedPager)
                assert run(serial,pages,36,2,TexturePager)==run(pipeline,pages,36,3,PipelinedPager)
                compared+=79*4
            with DiskPages(location/f'b-{index}',79,17) as pipeline:
                run(pipeline,list(reversed(pages)),41,2,PipelinedPager)
                run(pipeline,pages,41,2,PipelinedPager,expected_steps=0)
                compared+=79
        with DiskPages(location/'writer-error',79,17) as store:
            pages=list(range(store.page_count))
            run(store,pages,3,1,TexturePager)
            baseline={k:dict(v) for k,v in store.manifest['pages'].items()}
            commit=store.commit
            def fail(*args,**kwargs):raise OSError('simulated disk writer failure')
            store.commit=fail
            engine=PipelinedPager(store,17*store.stride*4)
            try:
                engine.advance(0,5)
                try:engine.flush_all();raise AssertionError('Writer failure was ignored')
                except OSError:pass
            finally:engine.close()
            assert store.manifest['pages']==baseline
            store.commit=commit
            run(store,pages,5,2,PipelinedPager)
        print(json.dumps(dict(result='PASS',schedules=9,completeChainsCompared=compared,
            checks=['serial/pipeline equality','one-slot reuse','three-slot reuse',
                    'bounded prefetch','remainder quantum','custom genes/geometry',
                    'reverse order','disk resume','idempotent verified target','asynchronous writer failure']),indent=2))
    finally:
        assert pathlib.Path(temp.name).resolve().is_relative_to((ROOT/'results').resolve())
        temp.cleanup()

if __name__=='__main__':main()
