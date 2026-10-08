# Klootwijk native operator kernel

TK-EDGE-1 implements your PDF's **TK-PINION-GREEDY-1** integer transition and one-bit operator rewrite on hardware. Native CUDA goes through the CUDA Driver API; native WebGPU uses Vulkan/D3D12 through wgpu. The WGSL device code also runs in a compatible phone browser. There is no pet or game renderer.

This is native GPU compute under the operating system, not OS-free firmware. CUDA targets NVIDIA; the portable shader targets WebGPU implementations. The POCO's GPU, driver and browser must be checked on that device before claiming support or performance.

## Run on this laptop

From this `kernel` directory, PowerShell:

```powershell
.venv\Scripts\python.exe native\run.py
.venv\Scripts\python.exe native\run.py --backend cuda
.venv\Scripts\python.exe native\run.py --backend wgpu
.venv\Scripts\python.exe native\run.py --tape --out results\tape.json
.venv\Scripts\python.exe native\run.py --checkpoint results\checkpoint.json
```

The default texture backend executes 128 transitions with bit 1 on every second accepted tick, compares every emitted trace word and the complete checkpoint with the CPU reference, then saves `results/checkpoint.json`. `--backend cuda` selects the pointer implementation and `--backend wgpu` selects portable compute. The expected default final pair is `11020439010204C7`. No random seed is needed. The one-bit stream is an explicit mutation input, not a complete one-bit encoding of arbitrary state.

Fresh Windows setup, Python 3.12 and a current NVIDIA driver:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
node tests/core.test.mjs
node tests/machine.test.mjs
.venv\Scripts\python.exe tests/gpu_test.py --backend cuda
.venv\Scripts\python.exe tests/gpu_test.py --backend wgpu
.venv\Scripts\python.exe tests/protocol_test.py
```

The task-local environment is already installed here. CUDA needs its driver and the NVRTC wheel, not a full CUDA toolkit. The device source is `native/kernel.cu`; the portable source is `dist/pinion.wgsl` and `dist/tape.wgsl`. Host/runtime libraries are much larger than the packed device state.

## RTX 5070 Ti texture kernel and disk paging

`native/paged.py` executes the same pinion transition through native CUDA **unsigned integer texture-object fetches**. Its immutable 4,352-byte default world/LUT is bound with point sampling and integer reads. Each independent chain owns a 112-byte writable state/overlay in VRAM. Resident allocations persist between dispatches. The texture and pointer backends share one device implementation; the texture backend also executes the U-TAPE-1 program table in the conformance tests.

```powershell
# 16 GiB of logical state, bounded by the currently available VRAM budget.
.venv\Scripts\python.exe native\paged.py --logical-mib 16384 --vram-mib 10240 --ticks 8

# Reload the same plain disk store; advance to absolute tick 64 in reverse page order.
.venv\Scripts\python.exe native\paged.py --logical-mib 16384 --vram-mib 10240 --ticks 64 --reverse

# Optional deterministic input; choose a separate store for a different schedule.
.venv\Scripts\python.exe native\paged.py --store swap\phrase --seed-phrase 'donald trump tower'

.venv\Scripts\python.exe tests\gpu_test.py --backend texture
.venv\Scripts\python.exe tests\paged_test.py
```

On a fresh clone, run `node tests/core.test.mjs` before the GPU fixture tests; it generates the ignored `tests/fixtures.json` locally.

The default reserve is 1,024 MiB; `--headroom-mib 512` is the minimum admitted reserve. The pager caps requests to free VRAM and retains a further page-sized admission margin. It writes completed pages to the drive, verifies their checksums when reloading, and reuses the same GPU slots through least-recently-used eviction. One pinned host page bounds staging RAM. Stream-ordered asynchronous copies are fenced before host reuse and checkpoint publication; disk I/O and page dispatch orchestration are currently serial. There is no automatic performance gain from exceeding VRAM: disk bandwidth and transfer latency can dominate.

`--ticks` means the absolute accepted tick to reach, including on resume. A pass interrupted between pages can be completed with the same command without duplicating mutations. Each page commits independently; a complete pass puts all pages at the target. A candidate file is flushed before an atomically replaced manifest references it. The old file is retained until that replacement succeeds. An OS lock admits one store writer. This is tested process-interruption recovery, not a guarantee against every filesystem/power-loss failure. Keep the manifest with its referenced files when copying a store.

`--page-mib` controls eviction granularity; `--quantum` bounds each launch to 1..128 ticks, default 8. Custom pinion geometry and typed turns/gains can be supplied with `--profile path\profile.json` when creating a store; use that same profile when reopening it. Changing the initial profile or input schedule creates a new store. Live typed role/gene exchange remains available through the existing peer protocol; the disk pager currently stores pinion chains only. It does not page an unbounded universal tape.

The kernel instructions execute on the GPU's compute units. LUT reads use the hardware-managed texture/cache path; neither executable code nor all VRAM can be pinned inside the texture cache. This implementation remains CUDA Driver API compute under Windows/Linux. See [NVIDIA's texture-object API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__TEXOBJECT.html).

Your exact supplied phrase spellings and their SHA-256 digests are in `docs/seed-inputs.json`, together with hashed GPU UUID and active laptop Wi-Fi MAC inputs. The raw local inventory is ignored under `.local/`. Hashes are optional input data, never encryption, signatures, ownership locks or a quantum-resilience claim. Their 256 bits repeat with the global chain ID and logical tick offset; this is a reproducible finite input schedule. Seedless execution keeps the alternating-bit schedule. Stored state remains plain and editable; after an intentional edit, update the relevant checksum consistently.

See `docs/RTX5070TI.md` and `docs/benchmarks/` for verified identifiers and measured working-set runs. The 16 GiB checkpoint store remains local under ignored `swap/`; it is not part of the source distribution.

## Phone / browser and native peer

Open [the diagnostic client](https://klootwijk-kernel.c0mbatduckzz.chatgpt.site) on the POCO, sign in with the owning account, and select **Verify GPU trace**. A passing hardware result is the phone qualification check. The client explicitly identifies WebGPU availability; CPU execution requires selecting **CPU reference**. An Internet connection alone does not provide a GPU API. Chrome documents Android WebGPU support for compatible ARM/Qualcomm devices on Android 12 or later: <https://developer.chrome.com/docs/web-platform/webgpu/overview>.

To synchronize the phone's GPU with native CUDA:

1. On the phone, select **Create offer**, then **Download signaling**. Transfer `offer.json` to this laptop using your preferred file transfer.
2. On the laptop run:

   ```powershell
   .venv\Scripts\python.exe native\peer.py --offer C:\path\to\offer.json
   ```

3. Transfer the generated `results/answer.json` back to the phone. Use **Load signaling**, then **Accept answer**.
4. Wait for **Peer ready**, then execute ticks or commit genes. The headless native participant supplies an idle slot and executes the same committed log on its GPU. It saves `results/peer-checkpoint.json` and `results/peer-replay.json`. Stop it with Ctrl+C.

Two browsers can use the same offer/answer exchange. Some networks need a TURN relay; the ICE configuration accepts your STUN/TURN servers. No relay credentials or service are included, and unrestricted Internet reachability has not been tested. The native CLI is an answerer; use the browser to originate edits. Both-browser sessions allow either member to originate an edit. Keep both pages active; browser suspension defers rounds.

Genes and checkpoints are plain data. There are no signatures, keys, encryption or ownership locks in the gene format. WebRTC still uses its standard encrypted transport. The hosted client is private by default; local copies and exported packets remain editable.

## Determinism and polymorphism

Each chain has its own mutable operator overlay. Device invocation `chain` owns that chain's state and trace; no two invocations mutate one registry concurrently. Packed integer arithmetic, fixed ties, stable node IDs, and logical ticks determine the result. Clock time, GPU scheduling, peer latency and adapter architecture are absent from transition semantics.

For asynchronous peers, a logical round closes only after one slot from each agreed member arrives. Slots execute in sorted member order, including gene changes. Out-of-order delivery within the bounded window and identical duplicates preserve emitted results. Missing slots cause DEFER. Contradictory duplicates halt the session; this protocol does not guarantee convergence against an equivocating participant. It does not infer membership changes from timeouts. CPU/GPU comparison occurs on a candidate copy before local commit, and peers compare committed checkpoints. A disconnect starts no new epoch automatically.

Typed genes can replace phase turns/gain banks or switch between pinion and **U-TAPE-1**, a finite-prefix byte-symbol Turing-machine interpreter. Switching role constructs the new profile from the supplied description. It does not synthesize a program or mutate executable shader bytes. The pinion alone is a bounded finite-state machine. Universality belongs to the extensible tape/program family; this implementation's default tape window, state count, counters and message budgets are finite. Extending them requires an agreed profile revision on all participants.

See `docs/ABI.md` for byte offsets, memory ownership, limits and commit rules. GPU reports are in `results/`; tests retain the unmodified Python reference and JSON extracted from your PDF under `tests/`.

## Hardware verification on 2026-10-07

- NVIDIA RTX 5070 Ti Laptop GPU: CUDA and native Vulkan/WebGPU each matched 65 profiles / 8,320 ticks, plus 4,096 independent chains / 524,288 ticks, including complete traces and mutable overlays.
- Intel Xe-LPG browser WebGPU ↔ NVIDIA native CUDA: seven WebRTC rounds passed, including pinion genes, pinion → tape → pinion changes and equal committed checkpoints.
- 720 JavaScript arrival permutations and 24 native arrival permutations passed. Tape halt, missing transition, boundary DEFER and atomic batch rejection passed.
- The POCO and second laptop have not been tested. These are conformance/finite batch checks, not sustained saturation, datacenter scaling, a formal implementation proof or an availability guarantee.

Optional local preview:

```powershell
.venv\Scripts\python.exe -m http.server 8787 --bind 127.0.0.1 --directory dist
```

Open `http://127.0.0.1:8787` on this laptop. Phone WebGPU should use the hosted HTTPS page; ordinary LAN HTTP is not a secure context. `tests/native_bridge.py` serves a loopback-only integration fixture at port 8788; it is not deployed or intended as a network service.
