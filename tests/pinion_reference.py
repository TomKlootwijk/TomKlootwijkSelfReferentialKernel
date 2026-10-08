"""TK-PINION-GREEDY-1: independently written finite CPU reference.

Two implementations of one explicitly new, one-hop pinion profile:
  Dense: full RP32 operator rows, staged products, encoded-phase update.
  Compact: base-plus-bit overlay, cached exact factors, intrinsic-phase update.
Neither implements the historical HP global planner, GPU execution, geometry
mutation, arbitrary program synthesis, crash recovery, or physical adapters.
Python 3.10+, standard library. Run: python pinion_reference.py --out results.json
The actual tests use explicit checks, not assertions disabled by python -O.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Iterable
import argparse
import hashlib
import json
import platform
import random
import sys

DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
GAINS = ((1, 1), (-1, 1), (-1, -1), (1, -1))
TURNS = (11, 53, 137)
STEP = 1


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def integer(value: object, lo: int, hi: int) -> bool:
    return type(value) is int and lo <= value <= hi


def pack(r: int, node: int, field: int, meta: int) -> int:
    if not (integer(r, 0, 255) and integer(node, 0, 255)
            and integer(field, -128, 127) and integer(meta, 0, 127)):
        raise ValueError('invalid RP32 lanes')
    v = r | node << 8 | (field & 255) << 16 | meta << 24
    return v | (v.bit_count() % 2) << 31


def unpack(word: int) -> tuple[int, int, int, int]:
    if not integer(word, 0, 2**32 - 1) or word.bit_count() % 2:
        raise ValueError('invalid RP32 word or parity')
    b = (word >> 16) & 255
    return word & 255, (word >> 8) & 255, b - 256 if b >= 128 else b, (word >> 24) & 127


def mirror(word: int) -> int:
    r, i, b, a = unpack(word)
    return pack(-r % 256, i, b, a ^ 16)


def pair(word: int) -> int:
    return word | mirror(word) << 32


def canonical(u: int, v: int, w: int, h: int) -> int:
    winding, u0 = divmod(u, w)
    return u0 * h + ((-v if winding % 2 else v) % h)


def metric(i: int, j: int, w: int, h: int) -> int:
    u, v = divmod(i, h)
    a, b = divmod(j, h)
    delta = abs(u - a)
    cycle = lambda z: min(z % h, (-z) % h)
    return min(delta + cycle(v - b), w - delta + cycle(v + b))


def bfs(adj: tuple[tuple[int, ...], ...], seeds: Iterable[int]) -> tuple[int, ...]:
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError('distance needs a nonempty boundary')
    d = [-1] * len(adj)
    queue = deque(seeds)
    for i in seeds:
        d[i] = 0
    while queue:
        i = queue.popleft()
        for j in adj[i]:
            if d[j] < 0:
                d[j] = d[i] + 1
                queue.append(j)
    return tuple(d)


def field_class(b: int) -> int:
    return 0 if b < 0 else 1 if b == 0 else 2


@dataclass(frozen=True)
class World:
    width: int = 8
    height: int = 8
    radius: int = 2
    center: int = 0
    epoch: int = 0

    def __post_init__(self) -> None:
        if not (integer(self.width, 3, 85) and integer(self.height, 3, 85)
                and self.width * self.height <= 256
                and integer(self.center, 0, self.width * self.height - 1)
                and integer(self.radius, 1, 127) and integer(self.epoch, 0, 0)):
            raise ValueError('invalid finite world profile')
        n = self.width * self.height
        edge_rows = []
        for i in range(n):
            u, v = divmod(i, self.height)
            edge_rows.append(tuple((canonical(u + du, v + dv, self.width, self.height),
                                    ((u + du) // self.width) & 1)
                                   for du, dv in DIRECTIONS))
        edges = tuple(edge_rows)
        adj = tuple(tuple(j for j, _ in row) for row in edges)
        dc = bfs(adj, (self.center,))
        if self.radius >= max(dc):
            raise ValueError('radius must leave a nonempty positive side')
        signs = tuple(-1 if d < self.radius else 0 if d == self.radius else 1 for d in dc)
        zero = tuple(i for i, s in enumerate(signs) if s == 0)
        bd = bfs(adj, zero)
        phi = tuple(s * d for s, d in zip(signs, bd))
        for name, value in (('n', n), ('edges', edges), ('adj', adj), ('dc', dc),
                            ('signs', signs), ('zero', zero), ('phi', phi)):
            object.__setattr__(self, name, value)
        if not self.certify(phi):
            raise ValueError('field certificate failed')
        factors = []
        axes = []
        for i in range(n):
            g, psi = self.features(i)
            factors.append(tuple((1 + p*p) * v for p, v in zip(psi, g)))
            axes.append(psi)
        object.__setattr__(self, 'factors', tuple(factors))
        index_phase = [0] * n
        for i in sorted(range(n), key=lambda j: (dc[j], j)):
            if i != self.center:
                parent = min(j for j in adj[i] if dc[j] == dc[i] - 1)
                index_phase[i] = (index_phase[parent] + TURNS[field_class(phi[parent])]) % 256
        keys = tuple((axes[i][0]+2, axes[i][1]+2, (dc[i]+1).bit_length()-1,
                      index_phase[i], i) for i in range(n))
        object.__setattr__(self, 'keys', keys)
        object.__setattr__(self, 'ordered', tuple(sorted(keys)))

    def features(self, i: int) -> tuple[tuple[int, int], tuple[int, int]]:
        up, um, vp, vm = self.adj[i]
        g = (self.phi[up] - self.phi[um], self.phi[vp] - self.phi[vm])
        div = gcd(abs(g[0]), abs(g[1]))
        return g, (g[0] // div, g[1] // div) if div else (1, 0)

    def certify(self, values: tuple[int, ...]) -> bool:
        if len(values) != self.n:
            return False
        for i, b in enumerate(values):
            if not integer(b, -127, 127):
                return False
            if (-1 if b < 0 else 0 if b == 0 else 1) != self.signs[i]:
                return False
            if any(abs(b-values[j]) > 1 for j in self.adj[i]):
                return False
            if b != 0 and not any(abs(values[j]) == abs(b)-1 for j in self.adj[i]):
                return False
        return True

    def lookup(self, i: int) -> tuple[int, int]:
        """Actual lower-median BST walk on the sorted f8 keys."""
        target = self.keys[i]
        lo, hi, visits = 0, self.n, 0
        while lo < hi:
            mid = (lo + hi - 1) // 2
            visits += 1
            key = self.ordered[mid]
            if key == target:
                return key[-1], visits
            if target < key:
                hi = mid
            else:
                lo = mid + 1
        raise ValueError('missing f8 key')

    def operator(self, row: int, delta: int) -> int:
        i, rem = divmod(row, 12)
        _, slot = divmod(rem, 4)
        dest, seam = self.edges[i][slot]
        return pack(delta, dest, self.phi[dest], STEP | (seam << 6))


class Engine:
    """Single-thread CPU transaction model; no persistent service claims."""
    def __init__(self, world: World, compact: bool, *, node: int = 0, phase: int = 250,
                 orientation: int = 0, capacity: int = 2, max_ticks: int = 65536,
                 hazards: tuple[int, ...] | None = None):
        if not (integer(node, 0, world.n-1) and integer(phase, 0, 255)
                and integer(orientation, 0, 1) and integer(capacity, 0, 256)
                and integer(max_ticks, 1, 65536) and type(compact) is bool):
            raise ValueError('invalid engine profile')
        hazards = (0,) * world.n if hazards is None else hazards
        if len(hazards) != world.n or any(not integer(x, 0, 255) for x in hazards):
            raise ValueError('invalid fixed hazard field')
        self.world, self.compact, self.capacity, self.max_ticks = world, compact, capacity, max_ticks
        self.hazards = tuple(hazards)
        self.word = pack(phase, node, world.phi[node], STEP | (orientation << 4))
        self.tick = self.program_epoch = self.overlay = 0
        self.queue: tuple[int, ...] = ()
        self.table = None if compact else tuple(world.operator(row, TURNS[(row % 12)//4])
                                               for row in range(12*world.n))

    def row_word(self, row: int) -> int:
        if self.compact:
            c = (row % 12)//4
            return self.world.operator(row, TURNS[c] ^ ((self.overlay >> row) & 1))
        return self.table[row]

    def logical_table(self) -> tuple[int, ...]:
        return tuple(self.row_word(k) for k in range(self.world.n * 12))

    def snapshot(self) -> tuple:
        return (self.tick, self.program_epoch, self.word, self.queue, self.logical_table())

    def step(self, jitter: int, *, budget: int = 1, dependency_available: bool = True,
             expected_epoch: int = 0) -> tuple[str, dict | None]:
        # Total classification for this finite API. Requests never auto-advance time.
        if not (integer(jitter, 0, 1) and integer(budget, 0, 65536)
                and type(dependency_available) is bool and integer(expected_epoch, 0, 65536)):
            return 'INVALID', None
        if expected_epoch != self.world.epoch:
            return 'INVALID', None
        if not dependency_available:
            return 'MISS', None
        if budget < 1 or self.tick >= self.max_ticks:
            return 'DEFER', None
        world = self.world
        r, i, b, meta = unpack(self.word)
        require(b == world.phi[i] and meta in (STEP, STEP | 16), 'live field or role')
        eta = (meta >> 4) & 1
        found, visits = world.lookup(i)
        require(found == i and visits <= world.n.bit_length(), 'f8 search')
        theta = (-r if eta else r) % 256
        gain = GAINS[theta // 64]
        if self.compact:
            q = tuple(a*m for a, m in zip(gain, world.factors[i]))
            # Common additive 1 + max(abs(q)) cancels only for this one-hop choice.
            scored = [(abs(world.phi[d]) + self.hazards[d] - q[0]*e[0] - q[1]*e[1], d, s)
                      for s, (e, (d, _)) in enumerate(zip(DIRECTIONS, world.edges[i]))]
        else:
            g, psi = world.features(i)
            first = tuple(a*(1+p*p) for a, p in zip(gain, psi))
            q = tuple(h*v for h, v in zip(first, g))
            norm = max(abs(q[0]), abs(q[1]))
            scored = [(1 + abs(world.phi[d]) + self.hazards[d] + norm - sum(a*b for a,b in zip(q,e)), d, s)
                      for s, (e, (d, _)) in enumerate(zip(DIRECTIONS, world.edges[i]))]
        _, dest, slot = min(scored)
        c = field_class(b)
        row = 12*i + 4*c + slot
        old = self.row_word(row)
        delta, odest, ob, oa = unpack(old)
        tau = world.edges[i][slot][1]
        require((odest, ob, oa) == (dest, world.phi[dest], STEP | tau << 6), 'operator support/action')
        if self.compact:
            new_theta = (theta + delta) & 255
            new_eta = eta ^ tau
            new_r = (-new_theta if new_eta else new_theta) & 255
        else:
            new_r = ((-1 if tau else 1) * (r + (-1 if eta else 1)*delta)) % 256
            new_eta = eta ^ tau
        new_word = pack(new_r, dest, world.phi[dest], STEP | new_eta << 4)
        new_op = pack(delta ^ jitter, odest, ob, oa)
        new_overlay = self.overlay ^ (jitter << row)
        if self.compact:
            candidate_table = None
            reconstructed = world.operator(row, TURNS[c] ^ ((new_overlay >> row) & 1))
            require(reconstructed == new_op, 'overlay decoding')
        else:
            candidate_table = list(self.table)
            candidate_table[row] = new_op
            candidate_table = tuple(candidate_table)
        emitted = pair(new_word)
        new_queue = (self.queue + (emitted,))[-self.capacity:] if self.capacity else ()
        dropped = max(0, len(self.queue) + 1 - self.capacity)
        record = dict(tick=self.tick+1, source=i, destination=dest, slot=slot,
                      theta=theta, bank=theta//64, q=q, delta=delta, jitter=jitter,
                      seam=tau, phase=new_r, orientation=new_eta, field=world.phi[dest],
                      pair=f'{emitted:016X}', row=row, before=f'{old:08X}', after=f'{new_op:08X}',
                      program_epoch=self.program_epoch+jitter, dropped=dropped,
                      cost=1+abs(world.phi[dest])+self.hazards[dest]+max(map(abs,q))
                           -sum(a*b for a,b in zip(q,DIRECTIONS[slot])))
        require(unpack(new_word)[2] == world.phi[dest], 'candidate field')
        require(mirror(mirror(new_word)) == new_word, 'candidate mirror')
        require(dest in world.adj[i], 'candidate adjacency')
        # One publication point after all checks. Atomicity is logical, single-threaded.
        self.word, self.overlay, self.table, self.queue = new_word, new_overlay, candidate_table, new_queue
        self.tick += 1
        self.program_epoch += jitter
        return 'ACCEPT', record


def run(engine: Engine, jitters: Iterable[int]) -> list[dict]:
    out = []
    for bit in jitters:
        status, record = engine.step(bit)
        if status != 'ACCEPT':
            raise ValueError(f'run not admitted: {status}')
        out.append(record)
    return out


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def tests() -> dict:
    checks = []
    def passed(name: str, cases: int) -> None:
        checks.append(dict(name=name, cases=cases, result='PASS'))
    # Independent quotient metric vs graph traversal over every named vertex pair.
    count = 0
    for w in range(3, 11):
        for h in range(3, 11):
            adj = tuple(tuple(canonical(i//h+du, i%h+dv, w, h) for du,dv in DIRECTIONS)
                        for i in range(w*h))
            for i in range(w*h):
                d = bfs(adj, (i,))
                for j, actual in enumerate(d):
                    require(metric(i,j,w,h) == actual, 'metric vs BFS')
                    count += 1
    passed('Closed-form metric versus BFS (W,H=3..10)', count)
    world = World()
    require(tuple(min(metric(i,z,8,8) for z in world.zero) for i in range(64))
            == tuple(map(abs,world.phi)), 'independent boundary distances')
    broken = list(world.phi); broken[0] -= 1
    require(not world.certify(tuple(broken)), 'forged distance')
    passed('Independent 64-node field values plus forged-field rejection', 65)
    # Exhaustive default-world local decisions: include q ablation witnesses.
    ablation = phase_dependence = local = 0
    witness = None
    for i in range(world.n):
        selections = set()
        for r in range(256):
            for eta in (0,1):
                theta = (-r if eta else r) % 256
                gain = GAINS[theta//64]
                g, psi = world.features(i)
                hvec = tuple(a*(1+p*p) for a,p in zip(gain,psi))
                q = tuple(a*b for a,b in zip(hvec,g))
                cached = tuple(a*b for a,b in zip(gain,world.factors[i]))
                require(q == cached, 'exact factorization')
                full = [(1+abs(world.phi[d])+max(map(abs,q))-sum(a*b for a,b in zip(q,e)), d, s)
                        for s,(e,(d,_)) in enumerate(zip(DIRECTIONS,world.edges[i]))]
                reduced = [(abs(world.phi[d])-sum(a*b for a,b in zip(q,e)),d,s)
                           for s,(e,(d,_)) in enumerate(zip(DIRECTIONS,world.edges[i]))]
                require(min(full)[1:] == min(reduced)[1:], 'argmin common-term cancellation')
                selected = min(full)[1:]
                selections.add(selected)
                neutral = min((abs(world.phi[d]),d,s) for s,(d,_) in enumerate(world.edges[i]))[1:]
                if selected != neutral:
                    ablation += 1
                    if witness is None:
                        witness = dict(source=i,phase=r,orientation=eta,theta=theta,q=q,
                                       selected=selected,without_coupling=neutral)
                mtheta = (-((-r)%256) if eta^1 else (-r)%256)%256
                require(mtheta == theta, 'mirror-invariant selector phase')
                local += 1
        phase_dependence += len(selections)>1
    passed('Staged/cached products, selector equivalence and mirror inputs', local)
    passed('Field-coupling ablation changes a selected action', ablation)
    passed('Nodes with phase-bank-dependent selected actions', phase_dependence)
    # Dense/full versus compact/overlay trajectory and FULL registry after each tick.
    rng = random.Random(20261007)
    equal_ticks = 0
    for case in range(64):
        w,h = rng.randint(3,10), rng.randint(3,10)
        center = rng.randrange(w*h)
        maxd = max(metric(center,j,w,h) for j in range(w*h))
        radius = rng.randint(1,maxd-1)
        world_i = World(w,h,radius,center)
        cfg = dict(node=rng.randrange(w*h),phase=rng.randrange(256),orientation=rng.randrange(2),
                   capacity=rng.choice((0,1,2,8)),hazards=tuple(rng.randrange(8) for _ in range(w*h)))
        dense, compact = Engine(world_i,False,**cfg), Engine(world_i,True,**cfg)
        for _ in range(128):
            bit = rng.randrange(2)
            a,b = dense.step(bit),compact.step(bit)
            require(a == b and dense.snapshot() == compact.snapshot(), 'complete refinement trace')
            equal_ticks += 1
    passed('Dense/compact full state and registry: 64 seeded profiles x 128 ticks', equal_ticks)
    js = [int(t%2 == 0) for t in range(1,129)]
    default = Engine(world,True)
    trace = run(default,js)
    immutable = run(Engine(world,True),[0]*128)
    first = next(t['tick'] for t,u in zip(trace,immutable) if t['pair'] != u['pair'])
    require(first>1, 'operator mutation affects later behavior')
    passed('Integrated pinion + rewrite differs from no-mutation run', 1)
    reproduction = run(Engine(world,True),js)
    require(trace == reproduction, 'original-context regeneration')
    passed('Exact regeneration of complete integrated trace', len(trace))
    for cap in (0,1,2,8):
        e = Engine(world,True,capacity=cap)
        observed = run(e,js)
        strip = lambda rows: [{k:v for k,v in row.items() if k!='dropped'} for row in rows]
        require(strip(observed)==strip(trace) and len(e.queue)<=cap, 'residency independence')
    passed('Capacity-independent semantic trace, capacities 0/1/2/8', 4)
    faults = [(dict(jitter=True),'INVALID'),(dict(jitter=2),'INVALID'),
              (dict(jitter=0,budget=-1),'INVALID'),(dict(jitter=0,expected_epoch=1),'INVALID'),
              (dict(jitter=0,dependency_available=False),'MISS'),(dict(jitter=0,budget=0),'DEFER')]
    faults_count = 0
    for compact in (False,True):
        e = Engine(world,compact,max_ticks=1)
        for args,wanted in faults:
            before = e.snapshot()
            status,_ = e.step(**args)
            require(status==wanted and e.snapshot()==before, 'nonaccepted request atomicity')
            faults_count += 1
        e.step(1)
        before=e.snapshot();status,_=e.step(0)
        require(status=='DEFER' and e.snapshot()==before, 'tick budget atomicity')
        faults_count += 1
    passed('Invalid/missing/deferred requests preserve canonical state', faults_count)
    # Same complete pair relation when the entire initial state is mirrored.
    left = Engine(world,True)
    right = Engine(world,True,phase=6,orientation=1)
    for bit in js:
        a,b=left.step(bit)[1],right.step(bit)[1]
        require(right.word==mirror(left.word) and left.logical_table()==right.logical_table(), 'paired action')
        require((a['row'],a['after'])==(b['row'],b['after']), 'one logical rewrite')
    passed('Complete mirrored integrated execution and shared rewrite',128)
    # Every row mask reconstructs the same restricted RP32 variant.
    for row in range(12*world.n):
        for bit in (0,1):
            c=(row%12)//4
            old=world.operator(row,TURNS[c]);r,g,b,a=unpack(old)
            require(world.operator(row,TURNS[c]^bit)==pack(r^bit,g,b,a), 'overlay row exactness')
    passed('Base-plus-bit overlay decoding of every default row/variant',1536)
    require(len(default.overlay.to_bytes((12*world.n+7)//8,'little'))==96, 'serialized overlay bytes')
    passed('Serialized overlay byte count',1)
    return dict(profile='TK-PINION-GREEDY-1', edition='TK-SROF-2.0', python=platform.python_version(),
                platform=platform.system()+' '+platform.machine(), seed=20261007,
                checks=checks, check_groups=len(checks), total_named_cases=sum(x['cases'] for x in checks),
                first_mutation_behavior_tick=first, ablation_witness=witness,
                default_configuration=dict(width=8,height=8,radius=2,center=0,node=0,phase=250,
                    orientation=0,capacity=2,turns=TURNS,gains=GAINS,zero_hazards=True,
                    jitter='1 on even accepted ticks; 0 otherwise'),
                initial_pair=f'{pair(pack(250,0,world.phi[0],STEP)):016X}',
                final_64_pair=trace[63]['pair'],final_128_pair=trace[-1]['pair'],
                immutable_final_128_pair=immutable[-1]['pair'],
                trace_sha256=digest(trace),trace=trace,field=world.phi,
                storage_design=dict(nodes=64,logical_rows=768,dense_words_bytes=3072,
                    retained_base_turn_bytes=3,compact_overlay_bytes=96,
                    optional_cached_factor_bytes=128,
                    note='Raw exact adapter payload only, not Python heap or full machine state'),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='New finite one-hop profile. No historical HP planner, GPU, physical adapter, or global performance benchmark.')


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path('pinion_results_new.json'))
    args=ap.parse_args()
    result=tests()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('trace','field')},indent=2))


if __name__=='__main__':
    main()
