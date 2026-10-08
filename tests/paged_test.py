"""Hardware checks: complete small stores, order independence and durable recovery."""
import hashlib
import json
import pathlib
import sys
import tempfile
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'native'))
from backend import ROOT
from paged import DiskPages, TexturePager, MiB
from protocol import Pinion
from seed_inputs import KEYPHRASES, phrase_digest

def states(store):
    rows=[]
    buffer=np.empty(store.page_chains*store.stride,dtype=np.uint32)
    for page in range(store.page_count):
        store.read_into(page,buffer)
        rows.extend(buffer[:store.count(page)*store.stride].reshape((-1,store.stride)).tolist())
    return rows

def run(store,order,target):
    engine=TexturePager(store,store.page_chains*store.stride*4)
    try:
        for page in order:engine.advance(page,target,quantum=3)
        engine.flush_all()
        assert states(store)==[engine.reference(i,target) for i in range(store.chains)]
        return engine.evictions,engine.steps_executed
    finally:engine.close()

def main():
    (ROOT/'results').mkdir(exist_ok=True)
    temp=tempfile.TemporaryDirectory(prefix='paged-test-',dir=ROOT/'results')
    location=pathlib.Path(temp.name).resolve()
    assert location.is_relative_to((ROOT/'results').resolve())
    try:
        chains,page_chains=79,17
        for index,digest in enumerate([None]+[phrase_digest(x) for x in KEYPHRASES]):
            with DiskPages(location/f'a-{index}',chains,page_chains,digest=digest) as first, DiskPages(location/f'b-{index}',chains,page_chains,digest=digest) as second:
                run(first,range(first.page_count),19)
                run(second,reversed(range(second.page_count)),19)
                assert states(first)==states(second)
            # Reopen with the stored schedule, continue from disk, and execute no duplicates.
            with DiskPages(location/f'a-{index}',chains,page_chains) as first, DiskPages(location/f'b-{index}',chains,page_chains) as second:
                run(first,reversed(range(first.page_count)),31)
                run(second,range(second.page_count),31)
                assert states(first)==states(second)
                assert run(first,range(first.page_count),31)[1]==0
                try:
                    DiskPages(location/f'a-{index}',chains,page_chains)
                    raise AssertionError('Concurrent store writer admitted')
                except RuntimeError:pass
        with DiskPages(location/'crash',chains,page_chains) as store:
            run(store,range(store.page_count),3)
            old=dict(store.entry(0));buffer=np.empty(page_chains*store.stride,dtype=np.uint32)
            store.read_into(0,buffer)
            def crash():raise OSError('simulated termination before manifest replacement')
            try:store.commit(0,buffer,4,before_manifest=crash)
            except OSError:pass
            assert store.entry(0)==old
        with DiskPages(location/'crash',chains,page_chains) as store:
            buffer=np.empty(page_chains*store.stride,dtype=np.uint32)
            assert store.read_into(0,buffer)==3
            # A pass stopped between pages must resume to the same absolute tick.
            engine=TexturePager(store,page_chains*store.stride*4)
            try:engine.advance(2,7);engine.flush_all()
            finally:engine.close()
        with DiskPages(location/'crash',chains,page_chains) as store:
            run(store,reversed(range(store.page_count)),7)
            damaged=store.directory/store.entry(0)['file']
            with damaged.open('r+b') as f:
                byte=f.read(1);f.seek(0);f.write(bytes([byte[0]^1]))
            try:store.read_into(0,buffer);raise AssertionError('Damaged page admitted')
            except ValueError as e:assert 'checksum' in str(e)
        print(json.dumps(dict(result='PASS',schedules=9,completeChainsCompared=chains*9*4,
            checks=['eviction','reverse order','resume',
                    'idempotent absolute target','single writer','interrupted page commit',
                    'interrupted pass','checksum damage rejection']),indent=2))
    finally:
        # Only remove this test-owned directory beneath the verified workspace results root.
        assert pathlib.Path(temp.name).resolve().is_relative_to((ROOT/'results').resolve())
        temp.cleanup()

if __name__=='__main__':main()
