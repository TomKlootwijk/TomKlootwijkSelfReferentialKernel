# TK-EDGE-1 ABI v1

This profile transports explicit state and edits. No entropy source or cryptographic lock participates in the transition. The optional native paged host can derive an explicit mutation input schedule from SHA-256, as specified below.

## RP32

Little-endian u32: phase bits 0..7; node bits 8..15; signed field bits 16..23; metadata bits 24..30; even-parity bit 31. Live metadata is STEP=1 plus orientation at bit 28. Operator metadata is STEP=1 plus seam at bit 30. Mirroring negates the phase modulo 256, toggles orientation, and recomputes parity. Both orientations share one logical operator rewrite.

## TKG1 pinion checkpoint

| Offset | Bytes | Meaning |
|---|---:|---|
| 0 | 4 | ASCII TKG1 |
| 4 | 1 | Version 1 |
| 5 | 1 | Role 0 = pinion |
| 6,7 | 2 | Width, height |
| 8,9 | 2 | Radius, center node |
| 10,11,12 | 3 | Live node, phase, orientation |
| 13 | 3 | Base turns by negative/zero/positive field class |
| 16 | 8 | Signed gain bytes: four 2-component banks |
| 24 | 4 | Accepted tick counter, u32 little-endian |
| 28 | 4 | Cumulative number of bit=1 rewrites, u32 little-endian |
| 32 | optional | `ceil(12*N/32)` little-endian u32 overlay words |

Overlay is omitted iff zero. Each row's single flag changes its current turn by XOR 1. The default N=64 overlay is 96 bytes. Counter exhaustion defers/rejects further batches without wrapping. Node IDs fit one byte. World geometry is reconstructed from width/height/radius/center; the f8 indexing uses the profile's original `[11,53,137]` turns, independent of live gene edits.

Pinion device world is 17*N u32 words: phi[N], edges[8*N] as destination/seam pairs, factors[2*N], f8 keys[5*N], sorted node IDs[N]. Device state stride is `4+ceil(12*N/32)` words: RP32, tick, rewrite counter, status, overlay. Parameters: N, step count, chain count, three turns, eight signed gains reinterpreted as u32. Per-step trace: live word, mirror word, chosen row, old operator, rewritten operator, qx, qy, full cost. All integers are exact within admitted bounds; signed components use two's-complement conversion.

The default device state is 112 bytes per chain, plus 4,352 bytes of shared materialized world data. A dispatch executes sequential steps inside each chain and parallel independent chains across invocations. Pointer/WebGPU backends use integer storage buffers; native texture mode uses `tex1Dfetch<unsigned int>` with `CU_TRSF_READ_AS_INTEGER`, point filtering and unnormalized integer coordinates. The world/program texture remains immutable throughout a launch. Mutable overlays stay in per-chain global memory. No floating-point sampling or interpolation enters semantics. Allocation, readback, traces, runtime libraries and networking consume additional memory.

## TK-PAGES-1 resident profile

Page files contain the same device state stride in little-endian u32 order, without per-step traces. The immutable JSON manifest records geometry, turns, gains, chain count, page chain count, optional seed digest, and each committed page's filename, accepted tick and SHA-256 damage checksum. Checksums are editable integrity metadata, not an authentication boundary. OS locking admits one writer; this store format is a local checkpoint format, not a network consensus protocol.

Global chain ID is `pageId * pageChains + localChainId`. Initial phase is `(configuredPhase + globalChainId) mod 256`; initial orientation is `configuredOrientation XOR (globalChainId mod 2)`. Overlays/counters start zero. Different physical page/slot orders cannot change these initial conditions. IDs must fit u32, each page's word addressing must fit u32, and accepted tick/rewrite counters cannot wrap.

Without a digest, input is 1 on every second accepted tick. With a 32-byte digest, select bit `(acceptedTickBeforeStep + globalChainId) mod 256`, low bit first in each digest byte. Phrase digest is SHA-256 of NFC-normalized UTF-8, preserving case and whitespace. Hardware MAC input uses lowercase colon-separated octets from the active Wi-Fi adapter; the input may change if that adapter's address changes. These digests determine a finite periodic input stream; they do not pack arbitrary state into a bit or provide cryptographic resilience.

Resident parameters extend the original 14 words with global chain base [14], schedule mode [15], eight little-endian digest words [16..23], and initial RP32 [24]. Mutable state belongs exclusively to its chain. Texture reads target immutable world data. A completed dispatch is fenced before eviction/readback; the pinned staging page is fenced before reuse. Disk and transfer failures leave the last manifest-referenced page versions as the recovery source. Completed pages can be ahead of others after an interrupted pass; resume to the same absolute target. There is no whole-working-set transaction or implicit rollback.

## Plain envelopes

Checkpoint JSON: `{ "role": "pinion", "data": "<base64url TKG1 bytes>" }` or `{ "role": "tape", "data": <tape snapshot> }`.

Hop JSON contains `round`, `member`, and `kind`. Kinds:

- `idle`: supplies the required slot without a transition.
- `step`: `count` 1..4096 plus base64url `bits`; bit i is low-bit-first in byte i/8. Unused high bits must be zero. Tape counts steps and ignores mutation bits.
- `genes`: `genes.role` plus typed fields. Pinion requires `turns[3]` (0..255) and `gains[8]` (-4..4). A switch to pinion may provide valid geometry and initial live state. Tape requires `states`, `halt`, `rules`; each rule is `[q,readByte,writeByte,move,nextQ]`, move -1/0/1. Duplicate rule keys are invalid. A switch to tape may provide q/head/cells; updating an existing tape preserves live q/head/cells/tick.

These messages are executable descriptions for fixed interpreters, not native code. Sender intent is explicit in the payload; arbitrary genes do not guarantee useful computation.

## Round contract

Peers agree on initial checkpoint, profile, members and 4,096-round budget before emission. The browser local mode has one member; peer mode fixes A/B. Each member authors at most one immutable slot per round. The generalized host supports up to 16 members, though the shipped WebRTC transport exposes two. Accepted slots replay in lexical member order. Future window is 64 rounds. Identical duplicate payloads compare via canonical JSON with sorted object keys; array order remains meaningful.

An async round commits only after awaited GPU work and checkpoint validation finish. Its candidate state is committed atomically; malformed genes, missing tape transitions, exhausted windows/counters or hardware mismatches leave the previous canonical state intact and fail the session. Earlier successfully closed rounds remain valid. Missing peer slots wait; wall-clock timeout never chooses a winner. Resource failures must be retried from an agreed checkpoint in a new session/profile. This is a replicated log/conformance prototype, not Byzantine consensus or durable crash recovery.

## U-TAPE-1 finite deployment

Device program is dense `(states*256)` rows of three u32 words: write byte, signed move, next state. `0xffffffff` next state denotes a missing rule. Default synchronized deployment uses at most 4,096 control states and tape addresses -2,048..2,047. Cells are byte symbols stored in u32 lanes. State stride is `4+capacity`: q, head offset, tick, status, cells. Parameters: state count, steps, chains, halt state, capacity. Trace: next q, next head offset, written byte, accepted tick.

Device status 0=accepted batch, 1=HALT, 2=INVALID, 3=DEFER. A move outside the allocated window defers before that move writes a cell or advances its counter. Low-level device dispatch can contain an accepted prefix before a later status; synchronized hosts stage and validate the whole round before exposing it. Head offsets convert to signed logical addresses only at the host boundary. Wider tapes/programs are possible by extending capacity and the agreed admission limits; no finite GPU implements infinite memory.
