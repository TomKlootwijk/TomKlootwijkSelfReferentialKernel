"""Bounded prefetch/writeback with CUDA event ownership for every reused buffer."""
from __future__ import annotations
import ctypes as C
import queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from paged import TexturePager, MiB

class PipelinedPager(TexturePager):
    def __init__(self, store, vram_bytes, headroom_bytes=1024*MiB):
        self.copy_stream=C.c_void_p()
        self.host_buffers=[]
        self.events=[]
        self.read_executor=self.write_executor=None
        self.prefetched={}
        self.writes={}
        self.all_writes=[]
        self.intervals={'compute':[],'upload':[],'download':[]}
        super().__init__(store,vram_bytes,headroom_bytes)
        try:
            self.gpu.call('cuStreamCreate',[C.POINTER(C.c_void_p),C.c_uint],C.byref(self.copy_stream),1)
            self.reader_buffers=queue.Queue()
            self.writer_buffers=queue.Queue()
            for pool in (self.reader_buffers,self.writer_buffers):
                for _ in range(2):pool.put(self.pinned(self.page_bytes))
            self.parameter_host=self.pinned(300*len(self.slots))
            self.parameter_gpu=self.allocate(300*len(self.slots))
            self.allocated_bytes+=300*len(self.slots)
            self.slot_parameters={p.value:index for index,p in enumerate(self.slots)}
            self.slot_uploads={p.value:None for p in self.slots}
            self.origin=self.event(timing=True)
            self.record(self.origin,self.stream)
            self.wait(self.origin)
            self.read_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='tk-prefetch')
            self.write_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='tk-checkpoint')
        except BaseException:
            self.close()
            raise

    def pinned(self, size):
        ptr=C.c_void_p()
        self.gpu.call('cuMemHostAlloc',[C.POINTER(C.c_void_p),C.c_size_t,C.c_uint],C.byref(ptr),size,0)
        self.host_buffers.append(ptr)
        array=np.ctypeslib.as_array(C.cast(ptr,C.POINTER(C.c_uint32)),shape=(size//4,))
        return dict(ptr=ptr,array=array,ready=None)

    def event(self, timing=False):
        event=C.c_void_p()
        self.gpu.call('cuEventCreate',[C.POINTER(C.c_void_p),C.c_uint],C.byref(event),0 if timing else 2)
        self.events.append(event)
        return event

    def record(self, event, stream):
        self.gpu.call('cuEventRecord',[C.c_void_p,C.c_void_p],event,stream)

    def wait(self, event):
        self.gpu.call('cuEventSynchronize',[C.c_void_p],event)

    def depends(self, stream, event):
        self.gpu.call('cuStreamWaitEvent',[C.c_void_p,C.c_void_p,C.c_uint],stream,event,0)

    def measure_start(self, kind, stream):
        start,end=self.event(True),self.event(True)
        self.intervals[kind].append((start,end))
        self.record(start,stream)
        return end

    def prefetch(self, page):
        self.store.count(page)
        if page in self.resident or page in self.prefetched:return
        # Bound look-ahead to two pinned read pages. A future owns its buffer.
        if len(self.prefetched)>=2:
            raise ValueError('Consume a prefetched page before requesting a third')
        if page in self.writes:self.writes[page].result()
        buffer=self.reader_buffers.get()
        def read():
            self.gpu.call('cuCtxSetCurrent',[C.c_void_p],self.gpu.context)
            if buffer['ready'] is not None:self.wait(buffer['ready'])
            tick=self.store.read_into(page,buffer['array'])
            return tick
        self.prefetched[page]=(buffer,self.read_executor.submit(read))

    def acquire(self, page):
        if page in self.resident:
            self.resident.move_to_end(page)
            return self.resident[page]
        self.prefetch(page)
        buffer,future=self.prefetched.pop(page)
        try:
            tick=future.result()
            if not self.available:
                oldest=next(iter(self.resident))
                self.flush(oldest)
                slot=self.resident.pop(oldest)['slot']
                self.evictions+=1
            else:slot=self.available.pop()
            # Host parameter rows cannot change until their previous upload finishes.
            uploaded=self.slot_uploads[slot.value]
            if uploaded is not None:self.wait(uploaded)
            index=self.slot_parameters[slot.value]
            host=self.parameter_host['array'][75*index:75*(index+1)].reshape((3,25))
            host[:]=self.params
            host[:,2]=self.store.count(page)
            host[:,14]=page*self.store.page_chains
            host[:,1]=0
            ready=self.event()
            entry=dict(slot=slot,tick=0 if tick is None else tick,dirty=False,
                       params=C.c_uint64(self.parameter_gpu.value+300*index),
                       hostParams=host,ready=ready,done=None,prepared=None)
            self.gpu.call('cuMemcpyHtoDAsync_v2',[C.c_uint64,C.c_void_p,C.c_size_t,C.c_void_p],
                          entry['params'],host.ctypes.data,300,self.copy_stream)
            if tick is not None:
                size=self.store.count(page)*self.store.stride*4
                end=self.measure_start('upload',self.copy_stream)
                self.gpu.call('cuMemcpyHtoDAsync_v2',[C.c_uint64,C.c_void_p,C.c_size_t,C.c_void_p],
                              slot,buffer['ptr'],size,self.copy_stream)
                self.record(end,self.copy_stream)
                self.read_bytes+=size
                buffer['ready']=ready
            else:buffer['ready']=None
            self.record(ready,self.copy_stream)
            self.depends(self.stream,ready)
            self.slot_uploads[slot.value]=ready
            # Initialization reads a separate immutable row. Updating the two
            # compute rows cannot race a still-running initialization kernel.
            if tick is None:self.launch('initialize',[slot,C.c_uint64(entry['params'].value+200)],self.store.count(page))
            self.resident[page]=entry
            self.peak_bytes=max(self.peak_bytes,sum(self.store.count(p)*self.store.stride*4 for p in self.resident))
            return entry
        finally:
            self.reader_buffers.put(buffer)

    def advance(self, page, target_tick, quantum=8):
        if not 0<=target_tick<=0xffffffff or not 1<=quantum<=128:
            raise ValueError('Invalid target tick or dispatch quantum')
        if page not in self.resident:
            if page in self.writes:self.writes[page].result()
            committed=self.store.entry(page)
            if committed and committed['tick']>=target_tick:
                if committed['tick']>target_tick:raise ValueError('Target tick precedes committed state')
                # Consume look-ahead even when no mutations remain, and verify the
                # durable bytes rather than treating manifest metadata as a proof.
                self.prefetch(page)
                buffer,future=self.prefetched.pop(page)
                try:
                    if future.result()!=target_tick:raise RuntimeError('Committed tick changed during read')
                    self.validate(page,buffer['array'],target_tick)
                    self.read_bytes+=self.store.count(page)*self.store.stride*4
                finally:self.reader_buffers.put(buffer)
                return
        entry=self.acquire(page)
        if entry['tick']>target_tick:raise ValueError('Target tick precedes resident state')
        delta=target_tick-entry['tick']
        if not delta:return
        # Two immutable parameter rows cover full quanta and the final remainder.
        # Fencing only this page leaves other pages' compute/writeback independent.
        if entry['done'] is not None:self.wait(entry['done'])
        self.wait(entry['ready'])
        full=min(quantum,delta);remainder=delta%full
        entry['hostParams'][:2,1]=[full,remainder or full]
        self.gpu.call('cuMemcpyHtoDAsync_v2',[C.c_uint64,C.c_void_p,C.c_size_t,C.c_void_p],
                      entry['params'],entry['hostParams'].ctypes.data,200,self.copy_stream)
        self.record(entry['ready'],self.copy_stream)
        self.depends(self.stream,entry['ready'])
        end=self.measure_start('compute',self.stream)
        while entry['tick']<target_tick:
            count=min(full,target_tick-entry['tick'])
            parameter=entry['params'] if count==full else C.c_uint64(entry['params'].value+100)
            self.launch('pinion_resident',[self.texture,entry['slot'],parameter],self.store.count(page))
            entry['tick']+=count
            self.steps_executed+=count*self.store.count(page)
        self.record(end,self.stream)
        entry['done']=end
        entry['dirty']=True

    def flush(self, page):
        entry=self.resident[page]
        if not entry['dirty']:return
        if page in self.writes:self.writes[page].result()
        buffer=self.writer_buffers.get()
        size=self.store.count(page)*self.store.stride*4
        self.depends(self.copy_stream,entry['done'])
        end=self.measure_start('download',self.copy_stream)
        self.gpu.call('cuMemcpyDtoHAsync_v2',[C.c_void_p,C.c_uint64,C.c_size_t,C.c_void_p],
                      buffer['ptr'],entry['slot'],size,self.copy_stream)
        self.record(end,self.copy_stream)
        tick=entry['tick']
        def commit():
            try:
                self.gpu.call('cuCtxSetCurrent',[C.c_void_p],self.gpu.context)
                self.wait(end)
                self.validate(page,buffer['array'],tick)
                self.store.commit(page,buffer['array'],tick)
                self.write_bytes+=size
            finally:self.writer_buffers.put(buffer)
        future=self.write_executor.submit(commit)
        self.writes[page]=future
        self.all_writes.append(future)
        entry['dirty']=False

    def flush_all(self):
        for page in self.resident:self.flush(page)
        for future in self.all_writes:future.result()
        self.sync()
        self.gpu.call('cuStreamSynchronize',[C.c_void_p],self.copy_stream)

    def timing_report(self):
        def elapsed(event):
            value=C.c_float()
            self.gpu.call('cuEventElapsedTime',[C.POINTER(C.c_float),C.c_void_p,C.c_void_p],
                          C.byref(value),self.origin,event)
            return value.value
        spans={k:[(elapsed(a),elapsed(b)) for a,b in pairs] for k,pairs in self.intervals.items()}
        def union(items):
            merged=[]
            for lo,hi in sorted(items):
                if merged and lo<=merged[-1][1]:merged[-1]=(merged[-1][0],max(hi,merged[-1][1]))
                else:merged.append((lo,hi))
            return merged
        compute=union(spans['compute']);transfers=union(spans['upload']+spans['download'])
        overlap=sum(max(0,min(b,d)-max(a,c)) for a,b in compute for c,d in transfers)
        return dict(computeMs=sum(b-a for a,b in compute),transferMs=sum(b-a for a,b in transfers),
                    computeTransferOverlapMs=overlap,intervalsMs=spans,
                    pinnedHostMiB=(5*self.page_bytes+300*len(self.slots))/MiB)

    def close(self):
        # Complete issued I/O before releasing its pinned buffers. No dirty resident
        # page is implicitly enqueued for publication during failure cleanup.
        try:
            for executor in (self.read_executor,self.write_executor):
                if executor is not None:executor.shutdown(wait=True,cancel_futures=True)
            if getattr(self,'gpu',None) is not None:
                self.sync()
                if self.copy_stream.value:
                    self.gpu.call('cuStreamSynchronize',[C.c_void_p],self.copy_stream)
        finally:
            if getattr(self,'gpu',None) is not None:
                for event in self.events:self.gpu.call('cuEventDestroy_v2',[C.c_void_p],event)
                for host in self.host_buffers:self.gpu.call('cuMemFreeHost',[C.c_void_p],host)
                if self.copy_stream.value:self.gpu.call('cuStreamDestroy_v2',[C.c_void_p],self.copy_stream)
            super().close()
