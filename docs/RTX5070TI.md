# Verified laptop target

Author: **Tom Klootwijk**. Hardware was queried on 2026-10-08.

| Field | Observed value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| CUDA compute capability | 12.0 |
| Reported VRAM | 12,227 MiB (`cuMemGetInfo`: 12,226.5625 MiB total) |
| Driver | 591.59 |
| GPU UUID | `GPU-072d759c-b0d7-47c0-04e9-48d3e81c9c8d` |
| PCI bus ID | `00000000:01:00.0` |
| L2 cache | 36 MiB (`CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE`) |
| Maximum 1D linear texture width | 268,435,456 elements |
| Network adapter used for the MAC seed | Intel Wi-Fi 6E AX211 160MHz, active Wi-Fi adapter |
| Laptop Wi-Fi MAC | `9c:67:d6:91:38:c7` |

The GPU UUID/PCI bus ID identify the card; this graphics adapter has no network MAC address. The laptop's network MAC is a separate identifier. `provenance.json` records the author-provided name, personal identifier and birth date, together with these verified hardware identifiers, as requested for public inclusion. This records provenance supplied by the author and does not claim identity verification. The seed catalog includes the exact corresponding inputs and their SHA-256 digests. Additional machine inventory remains in ignored `.local/hardware.json`.

The L2 figure is a shared hardware cache capacity, not a reserved texture-cache allocation. The driver/hardware controls texture/cache residency. Device programs execute on SMs, immutable LUT values are fetched through integer texture objects, and mutable chain state occupies allocated VRAM. This is native CUDA compute, not an OS-free firmware replacement.

The texture backend matched all 65 retained profiles (8,320 ticks) and 4,096 independent chains (524,288 ticks), including complete traces and overlays. The tape tests passed for halt, missing transition and boundary DEFER. Nine paging input schedules passed complete small-store state comparisons, reverse page order, disk resume, interrupted page/pass recovery, duplicate target avoidance, single-writer exclusion and corruption rejection. The asynchronous pager additionally matched serial page checksums and all 3,555 compared small-store chain states across nine schedules, including custom geometry/genes, one/three-slot reuse, remainder quanta, repeated targets and writer failure. A fresh 256 MiB run after the initialization-parameter isolation change passed 306,783,360 transitions with two 64 MiB resident slots and 13 complete CPU checkpoints.

The large runs use a 16-GiB logical working set: 153,391,689 independent 112-byte chain states, divided into 257 pages. Residency is bounded below the device/driver budget. Every candidate chain's status and tick is checked before its page commits; first/middle/last complete checkpoints on every page are compared with the CPU reference. The large working set is sampled for full-state semantic comparison, not exhaustively checked chain by chain. Reports distinguish allocated/resident state, total device memory observations, compute transitions, disk bytes and elapsed end-to-end time.

| Run | Resident state | Disk read / write | New transitions | Evictions | Complete CPU samples | Pass time |
|---|---:|---:|---:|---:|---:|---:|
| Initial, ticks 0 → 8 | 9,920 MiB | 0 / 16 GiB | 1,227,133,512 | 102 | 771 | 85.05 s |
| Reload, reverse order, ticks 8 → 64 | 10,432 MiB | 16 / 16 GiB | 8,589,934,584 | 94 | 771 | 80.10 s |
| Pipeline reload, ticks 64 → 128 | 10,432 MiB | 16 / 16 GiB | 9,817,068,096 | 94 | 771 | 33.35 s |
| Pipeline reload, reverse order, ticks 128 → 1,024 | 10,432 MiB | 16 / 16 GiB | 137,438,953,344 | 94 | 771 | 58.54 s |

The reloads left 592 MiB free in the CUDA allocation budget, above their configured 512-MiB reserve. NVIDIA SMI observed total device usage of 10,588 MiB. Peak sampled GPU utilization was 55% initially, 46% for the serial reload, 69% for the short pipeline reload and 100% for the longer pipeline reload. CUDA budget figures and NVIDIA SMI's whole-device accounting differ under this driver/OS; neither should be confused with texture-cache size. These times include disk transfer, checksums, fsync and CPU checkpoint validation; disk pacing leaves idle gaps. The initial run's final flush and the short pipeline run's start overlapped a separate small conformance test process. The serial reload and longer pipeline reload ran without that process. Different tick counts, launch quanta and OS file-cache conditions prevent a controlled speedup claim from this table.

Pass timing begins after CUDA compilation/allocation and ends after page validation and durable writes. The longer pipeline reload used quantum 128. CUDA event intervals measured 21,717.56 ms of compute, 6,489.56 ms of transfer and 908.96 ms of compute/transfer overlap. Its 100% value is the highest sampled device-utilization value, not sustained utilization across the whole run or a measurement of every SM's occupancy. The two large pipeline reports preceded the final initialization-parameter isolation change; they only reloaded existing pages, so they did not execute initialization. The final revision was separately verified by the complete pipeline/serial tests and fresh-page run recorded above.

See `benchmarks/` for sanitized measured reports. This workload fills the admitted resident-state budget; it does not establish datacenter scaling, sustained 100% compute occupancy, phone parity or a formal implementation proof.
