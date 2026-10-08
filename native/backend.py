"""Direct CUDA Driver/NVRTC and Vulkan/D3D12 WebGPU compute hosts."""
from __future__ import annotations
import ctypes as C
import os
import pathlib
import time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

class CUDA:
    def __init__(self, texture=False):
        self.driver = C.WinDLL('nvcuda.dll') if os.name == 'nt' else C.CDLL('libcuda.so.1')
        self.functions = {}
        self.call('cuInit', [C.c_uint], 0)
        dev = C.c_int()
        self.call('cuDeviceGet', [C.POINTER(C.c_int), C.c_int], C.byref(dev), 0)
        name = C.create_string_buffer(256)
        self.call('cuDeviceGetName', [C.c_void_p, C.c_int, C.c_int], name, 256, dev)
        attrs = []
        for attr in (75, 76):
            value = C.c_int()
            self.call('cuDeviceGetAttribute', [C.POINTER(C.c_int), C.c_int, C.c_int], C.byref(value), attr, dev)
            attrs.append(value.value)
        self.name = name.value.decode()
        self.capability = attrs
        self.texture = texture
        self.device = dev
        self.context = C.c_void_p()
        self.call('cuDevicePrimaryCtxRetain', [C.POINTER(C.c_void_p), C.c_int], C.byref(self.context), dev)
        self.call('cuCtxSetCurrent', [C.c_void_p], self.context)
        from importlib.util import find_spec
        spec = find_spec('nvidia.cuda_nvrtc')
        folder = pathlib.Path(next(iter(spec.submodule_search_locations)))
        if os.name == 'nt':
            self.dll_dir = os.add_dll_directory(str(folder / 'bin'))
            self.builtins = C.CDLL(str(next((folder / 'bin').glob('nvrtc-builtins*.dll'))))
            nvpath = next(p for p in (folder / 'bin').glob('nvrtc64_*.dll') if '.alt.' not in p.name)
        else:
            nvpath = next((folder / 'lib').glob('libnvrtc.so*'))
        self.nvrtc = C.CDLL(str(nvpath))
        source = (ROOT / 'native/kernel.cu').read_text()
        options = []
        if texture:
            runtime = pathlib.Path(next(iter(find_spec('nvidia.cuda_runtime').submodule_search_locations)))
            options = [('--include-path=' + str(runtime / 'include')).encode()]
            source = '#define TK_TEXTURE 1\n' + source
        self.module = self.compile(source, attrs, options)

    def call(self, name, types, *args):
        fn = getattr(self.driver, name)
        fn.argtypes, fn.restype = types, C.c_int
        result = fn(*args)
        if result:
            raise RuntimeError(f'{name}: CUDA error {result}')

    def compile(self, source, capability, extra_options=()):
        lib = self.nvrtc
        def nv(name, types, *args):
            fn = getattr(lib, name)
            fn.argtypes, fn.restype = types, C.c_int
            return fn(*args)
        prog = C.c_void_p()
        result = nv('nvrtcCreateProgram', [C.POINTER(C.c_void_p), C.c_char_p, C.c_char_p, C.c_int, C.c_void_p, C.c_void_p], C.byref(prog), source.encode(), b'kernel.cu', 0, None, None)
        if result: raise RuntimeError(f'nvrtcCreateProgram: {result}')
        opts = [f'--gpu-architecture=compute_{capability[0]}{capability[1]}'.encode(), b'--std=c++11', *extra_options]
        options = (C.c_char_p * len(opts))(*opts)
        result = nv('nvrtcCompileProgram', [C.c_void_p, C.c_int, C.POINTER(C.c_char_p)], prog, len(opts), options)
        size = C.c_size_t()
        nv('nvrtcGetProgramLogSize', [C.c_void_p, C.POINTER(C.c_size_t)], prog, C.byref(size))
        log = C.create_string_buffer(size.value)
        nv('nvrtcGetProgramLog', [C.c_void_p, C.c_void_p], prog, log)
        if result:
            nv('nvrtcDestroyProgram', [C.POINTER(C.c_void_p)], C.byref(prog))
            raise RuntimeError(f'NVRTC {result}: {log.value.decode()}')
        nv('nvrtcGetPTXSize', [C.c_void_p, C.POINTER(C.c_size_t)], prog, C.byref(size))
        ptx = C.create_string_buffer(size.value)
        nv('nvrtcGetPTX', [C.c_void_p, C.c_void_p], prog, ptx)
        nv('nvrtcDestroyProgram', [C.POINTER(C.c_void_p)], C.byref(prog))
        module = C.c_void_p()
        self.call('cuModuleLoadData', [C.POINTER(C.c_void_p), C.c_void_p], C.byref(module), ptx)
        return module

    def texture_object(self, ptr, size):
        from cuda.bindings import driver as d
        resource = d.CUDA_RESOURCE_DESC()
        resource.resType = d.CUresourcetype.CU_RESOURCE_TYPE_LINEAR
        resource.res.linear.devPtr = int(ptr.value)
        resource.res.linear.format = d.CUarray_format.CU_AD_FORMAT_UNSIGNED_INT32
        resource.res.linear.numChannels = 1
        resource.res.linear.sizeInBytes = size
        texture = d.CUDA_TEXTURE_DESC()
        texture.filterMode = d.CUfilter_mode.CU_TR_FILTER_MODE_POINT
        texture.flags = 1
        handle = C.c_uint64()
        self.call('cuTexObjectCreate', [C.POINTER(C.c_uint64), C.c_void_p, C.c_void_p, C.c_void_p],
                  C.byref(handle), resource.getPtr(), texture.getPtr(), None)
        return handle

    def execute(self, role, world, bits, state, params, trace_words):
        self.call('cuCtxSetCurrent', [C.c_void_p], self.context)
        arrays = [np.asarray(a, dtype=np.uint32).copy() for a in (world, bits, state)]
        arrays += [np.zeros(trace_words, dtype=np.uint32), np.asarray(params, dtype=np.uint32).copy()]
        allocations = []
        texture = None
        started = time.perf_counter()
        try:
            for a in arrays:
                ptr = C.c_uint64()
                self.call('cuMemAlloc_v2', [C.POINTER(C.c_uint64), C.c_size_t], C.byref(ptr), max(4, a.nbytes))
                allocations.append(ptr)
                if a.nbytes:
                    self.call('cuMemcpyHtoD_v2', [C.c_uint64, C.c_void_p, C.c_size_t], ptr, a.ctypes.data, a.nbytes)
            fn = C.c_void_p()
            self.call('cuModuleGetFunction', [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p], C.byref(fn), self.module, role.encode())
            arguments = allocations[:]
            if self.texture:
                texture = self.texture_object(allocations[0], arrays[0].nbytes)
                arguments[0] = texture
            args = (C.c_void_p * 5)(*(C.cast(C.byref(p), C.c_void_p) for p in arguments))
            kernel_start = time.perf_counter()
            self.call('cuLaunchKernel', [C.c_void_p, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_void_p, C.c_void_p, C.c_void_p], fn, (int(params[2])+63)//64, 1, 1, 64, 1, 1, 0, None, args, None)
            self.call('cuCtxSynchronize', [], )
            kernel_seconds = time.perf_counter() - kernel_start
            for i in (2, 3):
                self.call('cuMemcpyDtoH_v2', [C.c_void_p, C.c_uint64, C.c_size_t], arrays[i].ctypes.data, allocations[i], arrays[i].nbytes)
            return dict(state=arrays[2], trace=arrays[3], seconds=time.perf_counter()-started, kernel_seconds=kernel_seconds)
        finally:
            if texture is not None:
                self.call('cuTexObjectDestroy', [C.c_uint64], texture)
            for p in allocations:
                self.call('cuMemFree_v2', [C.c_uint64], p)

    def close(self):
        self.call('cuModuleUnload', [C.c_void_p], self.module)
        self.call('cuDevicePrimaryCtxRelease_v2', [C.c_int], self.device)

class WebGPU:
    def __init__(self):
        import wgpu
        self.wgpu = wgpu
        self.adapter = wgpu.gpu.request_adapter_sync(power_preference='high-performance')
        self.device = self.adapter.request_device_sync()
        self.name = dict(self.adapter.info)
        if self.name.get('adapter_type') == 'CPU':
            raise RuntimeError('Software adapter cannot substantiate hardware parity')
        self.pipelines = {}
        entries = [dict(binding=i, visibility=wgpu.ShaderStage.COMPUTE, buffer=dict(type='storage' if i in (2,3) else 'read-only-storage')) for i in range(5)]
        self.layout = self.device.create_bind_group_layout(entries=entries)
        self.pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=[self.layout])

    def execute(self, role, world, bits, state, params, trace_words):
        wgpu, d = self.wgpu, self.device
        if role not in self.pipelines:
            module = d.create_shader_module(code=(ROOT / f'dist/{role}.wgsl').read_text())
            self.pipelines[role] = d.create_compute_pipeline(layout=self.pipeline_layout, compute=dict(module=module, entry_point='main'))
        started = time.perf_counter()
        arrays = [np.asarray(a, dtype=np.uint32) for a in (world,bits,state)] + [np.zeros(trace_words,dtype=np.uint32),np.asarray(params,dtype=np.uint32)]
        buffers = [d.create_buffer_with_data(data=a,usage=wgpu.BufferUsage.STORAGE|wgpu.BufferUsage.COPY_DST|wgpu.BufferUsage.COPY_SRC) for a in arrays]
        try:
            group = d.create_bind_group(layout=self.layout,entries=[dict(binding=i,resource=dict(buffer=b,offset=0,size=b.size)) for i,b in enumerate(buffers)])
            enc = d.create_command_encoder()
            cp = enc.begin_compute_pass()
            cp.set_pipeline(self.pipelines[role]); cp.set_bind_group(0,group); cp.dispatch_workgroups((int(params[2])+63)//64); cp.end()
            d.queue.submit([enc.finish()])
            result = [np.frombuffer(d.queue.read_buffer(buffers[i]),dtype=np.uint32).copy() for i in (2,3)]
            return dict(state=result[0],trace=result[1],seconds=time.perf_counter()-started)
        finally:
            for b in buffers: b.destroy()

    def close(self):
        self.device.destroy()
