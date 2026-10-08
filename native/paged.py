"""Resident integer texture kernel with bounded, durable disk paging.

Each page commits independently. Reopening and advancing to the same absolute
tick completes an interrupted pass without replaying completed page mutations.
"""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
import math
import os
import pathlib
import shutil
import struct
import sys
import time
from collections import OrderedDict
import numpy as np
from backend import CUDA, ROOT
from protocol import Pinion, rp, unpack
from seed_inputs import mutation_bit, phrase_digest

MiB = 1024 ** 2
LAUNCH_TYPES = [C.c_void_p] + [C.c_uint] * 7 + [C.c_void_p] * 3

def atomic_json(path, data):
    candidate = path.with_suffix('.pending')
    with candidate.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(candidate, path)

class DiskPages:
    """Plain little-endian state; SHA-256 detects damage, never authorizes access."""
    def __init__(self, directory, chains, page_chains, prototype=None, digest=None):
        self.directory = pathlib.Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / 'writer.lock').open('a+b')
        if self.lock.tell() == 0:
            self.lock.write(b'\0')
            self.lock.flush()
        self.lock.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise RuntimeError('Another writer owns this page store') from None
        if sys.byteorder != 'little' or min(chains,page_chains)<1 or chains>=2**32:
            self.close()
            raise ValueError('This device ABI requires little endian and positive u32 chain counts')
        try:
            self.initialize(chains,page_chains,prototype,digest)
        except BaseException:
            self.close()
            raise

    def initialize(self, chains, page_chains, prototype, digest):
        if digest is not None:
            if not isinstance(digest,str) or len(digest)!=64 or len(bytes.fromhex(digest))!=32:
                raise ValueError('Expected a 32-byte SHA-256 digest')
            digest=digest.lower()
        self.path = self.directory / 'manifest.json'
        self.chains, self.page_chains = chains, page_chains
        if self.path.exists():
            self.manifest = json.loads(self.path.read_text(encoding='utf-8'))
            m = self.manifest
            if m.get('format') != 'TK-PAGES-1' or m['chains'] != chains or m['pageChains'] != page_chains:
                raise ValueError('Existing store geometry differs; use its original chain/page counts')
            if digest is not None and digest != m['seedSha256']:
                raise ValueError('Cannot change a running input schedule; create a new store')
            self.prototype = Pinion(m['config'])
            if prototype is not None and dict(prototype.config,turns=prototype.turns,gains=prototype.gains) != m['config']:
                raise ValueError('Existing store profile differs')
        else:
            self.prototype = prototype or Pinion()
            self.manifest = dict(format='TK-PAGES-1', byteOrder='little', chains=chains,
                                 pageChains=page_chains, config=dict(self.prototype.config,
                                 turns=self.prototype.turns, gains=self.prototype.gains),
                                 seedSha256=digest, pages={})
            atomic_json(self.path, self.manifest)
        self.digest = self.manifest['seedSha256']
        self.stride = len(self.prototype.state)
        self.page_count = math.ceil(chains / page_chains)
        self.logical_bytes = chains * self.stride * 4
        # All committed pages plus one candidate must fit with a 1-GiB drive reserve.
        stored = sum(self.count(int(k)) * self.stride * 4 for k in self.manifest['pages'])
        required = self.logical_bytes - stored + page_chains * self.stride * 4 + 1024 * MiB
        if shutil.disk_usage(self.directory).free < required:
            raise ValueError('Insufficient disk space for the working set, candidate and reserve')

    def close(self):
        self.lock.close()  # OS releases the advisory lock, including on process termination.

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def count(self, page):
        if not 0 <= page < self.page_count:
            raise ValueError('Page outside logical working set')
        return min(self.page_chains, self.chains - page * self.page_chains)

    def entry(self, page):
        return self.manifest['pages'].get(str(page))

    def read_into(self, page, array):
        entry = self.entry(page)
        if entry is None:
            return None
        # Reject traversal in an editable manifest.
        name = entry['file']
        if pathlib.Path(name).name != name or not name.startswith(f'page-{page:06d}-') or not name.endswith('.bin'):
            raise ValueError('Invalid page filename')
        view = memoryview(array).cast('B')[:self.count(page) * self.stride * 4]
        with (self.directory / name).open('rb', buffering=0) as f:
            if f.readinto(view) != len(view) or f.read(1):
                raise ValueError('Truncated or oversized page')
        if hashlib.sha256(view).hexdigest() != entry['sha256']:
            raise ValueError('Page checksum mismatch')
        return entry['tick']

    def commit(self, page, array, tick, before_manifest=None):
        old = self.entry(page)
        if old is not None and tick <= old['tick']:
            raise ValueError('Page commit must advance its logical tick')
        size = self.count(page) * self.stride * 4
        if shutil.disk_usage(self.directory).free < size + 1024 * MiB:
            raise ValueError('Drive reserve reached; old page remains committed')
        name = f'page-{page:06d}-tick-{tick:010d}.bin'
        candidate = self.directory / (name + '.pending')
        view = memoryview(array).cast('B')[:size]
        checksum = hashlib.sha256(view).hexdigest()
        with candidate.open('wb', buffering=0) as f:
            if f.write(view) != size:
                raise OSError('Short page write')
            os.fsync(f.fileno())
        os.replace(candidate, self.directory / name)
        if before_manifest:
            before_manifest()  # Fault-injection point: old manifest must survive.
        entry = dict(file=name, tick=tick, sha256=checksum)
        new = dict(self.manifest, pages=dict(self.manifest['pages'], **{str(page): entry}))
        atomic_json(self.path, new)
        self.manifest = new
        if old and old['file'] != name:
            (self.directory / old['file']).unlink(missing_ok=True)

class TexturePager:
    def __init__(self, store, vram_bytes, headroom_bytes=1024*MiB):
        self.store = store
        self.gpu = CUDA(texture=True)
        self.allocations = []
        self.texture = None
        self.stream = C.c_void_p()
        self.host = C.c_void_p()
        self.slots = []
        self.resident = OrderedDict()
        self.read_bytes = self.write_bytes = self.evictions = self.sample_checks = 0
        self.steps_executed = 0
        self.peak_bytes = 0
        try:
            self.gpu.call('cuStreamCreate', [C.POINTER(C.c_void_p), C.c_uint], C.byref(self.stream), 1)
            free, total = self.memory_info()
            self.total_bytes, self.initial_free_bytes = total, free
            self.page_bytes = store.page_chains * store.stride * 4
            if not 1 <= store.page_chains or store.page_chains * store.stride >= 2**32:
                raise ValueError('Page would exceed u32 device addressing')
            if headroom_bytes < 512*MiB or vram_bytes < self.page_bytes:
                raise ValueError('Reserve at least 512 MiB and request at least one page')
            capacity = min(vram_bytes, free-headroom_bytes-self.page_bytes)
            count = min(store.page_count, capacity // self.page_bytes)
            if count < 1:
                raise ValueError('Insufficient free VRAM after driver/display reserve')
            world = np.array(store.prototype.world['packed'], dtype=np.uint32)
            self.world = self.allocate(world.nbytes)
            self.gpu.call('cuMemcpyHtoD_v2', [C.c_uint64, C.c_void_p, C.c_size_t], self.world, world.ctypes.data, world.nbytes)
            self.texture = self.gpu.texture_object(self.world, world.nbytes)
            self.parameters = self.allocate(25*4)
            self.params = np.zeros(25, dtype=np.uint32)
            k = store.prototype
            self.params[:14] = np.array([k.world['n'],0,0]+k.turns+[g&0xffffffff for g in k.gains],dtype=np.uint32)
            self.params[15] = store.digest is not None
            if store.digest:
                self.params[16:24] = struct.unpack('<8I',bytes.fromhex(store.digest))
            self.params[24] = k.state[0]
            for _ in range(count):
                now, _ = self.memory_info()
                if now < self.page_bytes+headroom_bytes:
                    break
                self.slots.append(self.allocate(self.page_bytes))
            if not self.slots:
                raise RuntimeError('VRAM admission failed')
            self.available = list(self.slots)
            self.gpu.call('cuMemHostAlloc', [C.POINTER(C.c_void_p), C.c_size_t, C.c_uint], C.byref(self.host), self.page_bytes, 0)
            self.staging = np.ctypeslib.as_array(C.cast(self.host,C.POINTER(C.c_uint32)), shape=(self.page_bytes//4,))
            self.functions = {}
            for name in ('initialize','pinion_resident'):
                function = C.c_void_p()
                self.gpu.call('cuModuleGetFunction', [C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p], C.byref(function), self.gpu.module, name.encode())
                self.functions[name] = function
            self.allocated_bytes = sum(self.page_bytes for _ in self.slots) + world.nbytes + 100
        except BaseException:
            self.close()
            raise

    def allocate(self, size):
        ptr = C.c_uint64()
        self.gpu.call('cuMemAlloc_v2', [C.POINTER(C.c_uint64),C.c_size_t], C.byref(ptr),size)
        self.allocations.append(ptr)
        return ptr

    def memory_info(self):
        free,total = C.c_size_t(),C.c_size_t()
        self.gpu.call('cuMemGetInfo_v2',[C.POINTER(C.c_size_t),C.POINTER(C.c_size_t)],C.byref(free),C.byref(total))
        return free.value,total.value

    def sync(self):
        self.gpu.call('cuStreamSynchronize',[C.c_void_p],self.stream)

    def launch(self, name, arguments, chains):
        params=(C.c_void_p*len(arguments))(*(C.cast(C.byref(p),C.c_void_p) for p in arguments))
        self.gpu.call('cuLaunchKernel',LAUNCH_TYPES,self.functions[name],(chains+127)//128,1,1,128,1,1,0,self.stream,params,None)

    def set_params(self, page, steps):
        self.params[1:3] = steps,self.store.count(page)
        self.params[14] = page*self.store.page_chains
        # Parameters are small and stream-fenced before this host memory is reused.
        self.sync()
        self.gpu.call('cuMemcpyHtoD_v2',[C.c_uint64,C.c_void_p,C.c_size_t],self.parameters,self.params.ctypes.data,self.params.nbytes)

    def acquire(self, page):
        if page in self.resident:
            self.resident.move_to_end(page)
            return self.resident[page]
        if not self.available:
            oldest = next(iter(self.resident))
            self.flush(oldest)
            slot = self.resident.pop(oldest)['slot']
            self.evictions += 1
        else:
            slot = self.available.pop()
        tick = self.store.read_into(page,self.staging)
        count = self.store.count(page)
        if tick is None:
            self.set_params(page,0)
            self.launch('initialize',[slot,self.parameters],count)
            tick = 0
        else:
            size=count*self.store.stride*4
            self.gpu.call('cuMemcpyHtoDAsync_v2',[C.c_uint64,C.c_void_p,C.c_size_t,C.c_void_p],slot,self.host,size,self.stream)
            self.sync()  # Pinned staging is now safe for the next disk read.
            self.read_bytes += size
        entry=dict(slot=slot,tick=tick,dirty=False)
        self.resident[page]=entry
        self.peak_bytes=max(self.peak_bytes,sum(self.store.count(p)*self.store.stride*4 for p in self.resident))
        return entry

    def advance(self, page, target_tick, quantum=8):
        committed=self.store.entry(page)
        if page not in self.resident and committed and committed['tick']>=target_tick:
            if committed['tick']>target_tick:
                raise ValueError('Target tick precedes the committed state; rollback is not implicit')
            return
        entry=self.acquire(page)
        if not entry['tick'] <= target_tick <= 0xffffffff or not 1 <= quantum <= 128:
            raise ValueError('Invalid target tick or dispatch quantum')
        while entry['tick']<target_tick:
            count=min(quantum,target_tick-entry['tick'])
            self.set_params(page,count)
            self.launch('pinion_resident',[self.texture,entry['slot'],self.parameters],self.store.count(page))
            self.sync()  # Bounded launches keep watchdog risk separate from logical time.
            entry['tick']+=count
            entry['dirty']=True
            self.steps_executed+=count*self.store.count(page)

    def reference(self, chain_id, tick):
        k=Pinion(self.store.manifest['config'])
        r,i,b,a=unpack(k.state[0])
        k.state[0]=rp((r+chain_id)&255,i,b,a^((chain_id&1)<<4))
        k.cpu([int(mutation_bit(self.store.digest,chain_id,t)) for t in range(tick)])
        return k.state

    def flush(self, page):
        entry=self.resident[page]
        if not entry['dirty']:
            return
        count=self.store.count(page);size=count*self.store.stride*4
        self.gpu.call('cuMemcpyDtoHAsync_v2',[C.c_void_p,C.c_uint64,C.c_size_t,C.c_void_p],self.host,entry['slot'],size,self.stream)
        self.sync()
        self.validate(page,self.staging,entry['tick'])
        self.store.commit(page,self.staging,entry['tick'])
        self.write_bytes+=size
        entry['dirty']=False

    def validate(self, page, array, tick):
        count=self.store.count(page)
        state=array[:count*self.store.stride].reshape((count,self.store.stride))
        if np.any(state[:,3]) or np.any(state[:,1]!=tick):
            raise RuntimeError('INVALID/counter mismatch: candidate page was not committed')
        # Every status/tick is checked; exact CPU comparison samples complete overlays.
        if tick<=16384:
            for index in sorted({0,count//2,count-1}):
                if state[index].tolist()!=self.reference(page*self.store.page_chains+index,tick):
                    raise RuntimeError('GPU/reference mismatch: candidate page was not committed')
                self.sample_checks+=1

    def flush_all(self):
        for page in self.resident:
            self.flush(page)

    def close(self):
        # Never implicitly commit on exceptions. Last durable page versions remain valid.
        if getattr(self,'gpu',None) is None:
            return
        try:
            self.sync()
        finally:
            if self.texture is not None:
                self.gpu.call('cuTexObjectDestroy',[C.c_uint64],self.texture)
            for ptr in reversed(self.allocations):
                self.gpu.call('cuMemFree_v2',[C.c_uint64],ptr)
            if self.host.value:
                self.gpu.call('cuMemFreeHost',[C.c_void_p],self.host)
            if self.stream.value:
                self.gpu.call('cuStreamDestroy_v2',[C.c_void_p],self.stream)
            self.gpu.close()
            self.gpu=None

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--store',type=pathlib.Path,default=ROOT/'swap'/'rtx5070ti')
    ap.add_argument('--logical-mib',type=int,default=16384)
    ap.add_argument('--vram-mib',type=int,default=10240)
    ap.add_argument('--headroom-mib',type=int,default=1024)
    ap.add_argument('--page-mib',type=int,default=64)
    ap.add_argument('--ticks',type=int,default=8,help='Absolute accepted tick to reach, also on resume')
    ap.add_argument('--quantum',type=int,default=8)
    ap.add_argument('--mode',choices=['pipeline','serial'],default='pipeline')
    ap.add_argument('--profile',type=pathlib.Path,help='Plain pinion geometry/turns/gains JSON for a new store')
    seeds=ap.add_mutually_exclusive_group()
    seeds.add_argument('--seed-phrase')
    seeds.add_argument('--seed-sha256')
    ap.add_argument('--reverse',action='store_true')
    ap.add_argument('--out',type=pathlib.Path,default=ROOT/'results'/'paged.json')
    args=ap.parse_args()
    if min(args.logical_mib,args.vram_mib,args.page_mib)<1 or not 1<=args.ticks<=0xffffffff or not 1<=args.quantum<=128:
        ap.error('Positive budgets/ticks and a quantum in 1..128 are required')
    k=Pinion(json.loads(args.profile.read_text(encoding='utf-8'))) if args.profile else Pinion()
    stride=len(k.state)*4
    chains=args.logical_mib*MiB//stride
    page_chains=args.page_mib*MiB//stride
    if chains>=2**32:
        ap.error('Global chain IDs must fit u32 in this profile')
    digest=phrase_digest(args.seed_phrase) if args.seed_phrase is not None else args.seed_sha256
    if digest is not None:
        try:
            if len(digest)!=64 or len(bytes.fromhex(digest))!=32:raise ValueError()
        except ValueError:ap.error('Expected a 64-character SHA-256 hex digest')
        digest=digest.lower()
    store=DiskPages(args.store,chains,page_chains,prototype=k if args.profile else None,digest=digest)
    if store.stride*4!=stride:
        store.close()
        ap.error('Resuming a custom world requires the original --profile')
    try:
        if args.mode=='pipeline':
            from pipelined import PipelinedPager
            engine=PipelinedPager(store,args.vram_mib*MiB,args.headroom_mib*MiB)
        else:engine=TexturePager(store,args.vram_mib*MiB,args.headroom_mib*MiB)
    except BaseException:
        store.close()
        raise
    started=time.perf_counter()
    from telemetry import Telemetry
    telemetry=Telemetry()
    telemetry.start()
    try:
        print(json.dumps(dict(event='admitted',adapter=engine.gpu.name,residentMiB=engine.allocated_bytes/MiB,
                             logicalMiB=store.logical_bytes/MiB,slots=len(engine.slots),pages=store.page_count,
                             totalVramMiB=engine.total_bytes/MiB,seedSha256=store.digest,mode=args.mode)),flush=True)
        pages=list(range(store.page_count-1,-1,-1) if args.reverse else range(store.page_count))
        for sequence,page in enumerate(pages):
            if args.mode=='pipeline' and sequence+1<len(pages):engine.prefetch(pages[sequence+1])
            engine.advance(page,args.ticks,args.quantum)
            if sequence%16==0:
                print(json.dumps(dict(event='progress',submittedPages=sequence+1,pages=store.page_count,
                    durableAtTarget=sum(v['tick']==args.ticks for v in store.manifest['pages'].values()),
                    residentWorkingMiB=engine.peak_bytes/MiB,evictions=engine.evictions)),flush=True)
        engine.flush_all()
        free,_=engine.memory_info()
        report=dict(result='PASS',adapter=engine.gpu.name,texture='unsigned u32 point tex1Dfetch',
                    totalVramMiB=engine.total_bytes/MiB,allocatedVramMiB=engine.allocated_bytes/MiB,
                    peakResidentStateMiB=engine.peak_bytes/MiB,remainingFreeVramMiB=free/MiB,
                    logicalStateMiB=store.logical_bytes/MiB,chains=chains,pages=store.page_count,
                    residentSlots=len(engine.slots),targetTick=args.ticks,transitionsExecuted=engine.steps_executed,
                    evictions=engine.evictions,diskReadMiB=engine.read_bytes/MiB,diskWriteMiB=engine.write_bytes/MiB,
                    exactCpuSampleChecks=engine.sample_checks,seconds=time.perf_counter()-started,
                    store=str(store.directory),seedSha256=store.digest,mode=args.mode)
        if args.mode=='pipeline':report['cudaTimeline']=engine.timing_report()
        telemetry.close()
        if telemetry.samples:
            report['nvidiaSmiSamples']=telemetry.samples
            report['peakObservedDeviceUsedMiB']=max(s['usedMiB'] for s in telemetry.samples)
            report['peakObservedGpuUtilizationPercent']=max(s['utilizationPercent'] for s in telemetry.samples)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        atomic_json(args.out,report)
        display=dict(report)
        display.pop('nvidiaSmiSamples',None)
        if 'cudaTimeline' in display:
            display['cudaTimeline']={k:v for k,v in display['cudaTimeline'].items() if k!='intervalsMs'}
        print(json.dumps(display,indent=2),flush=True)
    finally:
        telemetry.close()
        engine.close()
        store.close()

if __name__=='__main__':main()
