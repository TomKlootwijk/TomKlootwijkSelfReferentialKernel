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

The GPU UUID/PCI bus ID identify the card; this graphics adapter has no network MAC address. The laptop's network MAC is a separate identifier. Its raw current value is retained only in ignored `.local/hardware.json`; the published seed catalog contains its SHA-256 digest. The author name is public. The requested personal ID number and birth date are excluded from this kernel distribution.

The L2 figure is a shared hardware cache capacity, not a reserved texture-cache allocation. The driver/hardware controls texture/cache residency. Device programs execute on SMs, immutable LUT values are fetched through integer texture objects, and mutable chain state occupies allocated VRAM. This is native CUDA compute, not an OS-free firmware replacement.

The texture backend matched all 65 retained profiles (8,320 ticks) and 4,096 independent chains (524,288 ticks), including complete traces and overlays. The tape tests passed for halt, missing transition and boundary DEFER. Nine paging input schedules passed complete small-store state comparisons, reverse page order, disk resume, interrupted page/pass recovery, duplicate target avoidance, single-writer exclusion and corruption rejection.

The large runs use a 16-GiB logical working set: 153,391,689 independent 112-byte chain states, divided into 257 pages. Residency is bounded below the device/driver budget. Every candidate chain's status and tick is checked before its page commits; first/middle/last complete checkpoints on every page are compared with the CPU reference. The large working set is sampled for full-state semantic comparison, not exhaustively checked chain by chain. Reports distinguish allocated/resident state, total device memory observations, compute transitions, disk bytes and elapsed end-to-end time.

| Run | Resident state | Disk read / write | New transitions | Evictions | Complete CPU samples | End-to-end time |
|---|---:|---:|---:|---:|---:|---:|
| Initial, ticks 0 → 8 | 9,920 MiB | 0 / 16 GiB | 1,227,133,512 | 102 | 771 | 85.05 s |
| Reload, reverse order, ticks 8 → 64 | 10,432 MiB | 16 / 16 GiB | 8,589,934,584 | 94 | 771 | 80.10 s |

The reload left 592 MiB free in the CUDA allocation budget, above its configured 512-MiB reserve. NVIDIA SMI observed total device usage of 10,588 MiB, a peak sampled GPU utilization of 46%, and temperatures up to 49°C in that run. CUDA budget figures and NVIDIA SMI's whole-device accounting differ under this driver/OS; neither should be confused with texture-cache size. The initial run's sampled utilization peaked at 55%. These include disk transfer, checksums, fsync and CPU checkpoint validation; disk pacing leaves the GPU idle between bursts. The initial run's final flush overlapped a separate small conformance test process; its timing is an approximate working-set result, not an isolated performance benchmark. The reload ran without that test process.

See `benchmarks/` for sanitized measured reports. This workload fills the admitted resident-state budget; it does not establish datacenter scaling, sustained 100% compute occupancy, phone parity or a formal implementation proof.
