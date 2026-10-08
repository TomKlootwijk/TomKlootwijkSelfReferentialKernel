"""TK-EDGE-1 plain checkpoints, typed genes and closed logical rounds."""
from __future__ import annotations
import base64
import copy
import json
import math
import struct
from collections import deque
import numpy as np

TURNS=[11,53,137]
GAINS=[1,1,-1,1,-1,-1,1,-1]
def require(ok,message):
    if not ok: raise ValueError(message)
def integer(x,a,b):return type(x) is int and a<=x<=b
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def b64(b):return base64.urlsafe_b64encode(b).decode().rstrip('=')
def unb64(s):
    require(type(s) is str and 0<len(s)<=1000000 and all(c.isalnum() and c.isascii() or c in '-_' for c in s),'Invalid packet text')
    return base64.b64decode(s+'='*((-len(s))%4),altchars=b'-_',validate=True)
def rp(r,i,b,a):
    v=r|(i<<8)|((b&255)<<16)|(a<<24)
    return v|((v.bit_count()&1)<<31)
def unpack(w):
    require(integer(w,0,0xffffffff) and w.bit_count()%2==0,'Invalid parity')
    b=(w>>16)&255
    return w&255,(w>>8)&255,b-256 if b>=128 else b,(w>>24)&127
def mirror(w):
    r,i,b,a=unpack(w);return rp((-r)&255,i,b,a^16)
def fclass(b):return 0 if b<0 else 1 if b==0 else 2
def world(c):
    w,h=c['width'],c['height'];n=w*h
    require(integer(w,3,85) and integer(h,3,85) and n<=256 and integer(c['center'],0,n-1) and integer(c['radius'],1,127),'Invalid world')
    def node(u,v):q,u0=divmod(u,w);return u0*h+((-v if q&1 else v)%h)
    directions=[(1,0),(-1,0),(0,1),(0,-1)]
    edges=[[(node(i//h+du,i%h+dv),((i//h+du)//w)&1) for du,dv in directions] for i in range(n)]
    def distance(seeds):
        d=[-1]*n;queue=deque(seeds)
        for i in queue:d[i]=0
        while queue:
            i=queue.popleft()
            for j,_ in edges[i]:
                if d[j]<0:d[j]=d[i]+1;queue.append(j)
        return d
    dc=distance([c['center']]);require(c['radius']<max(dc),'Radius leaves no positive side')
    signs=[-1 if d<c['radius'] else 0 if d==c['radius'] else 1 for d in dc]
    bd=distance([i for i,s in enumerate(signs) if not s]);phi=[s*d for s,d in zip(signs,bd)]
    axes=[];factors=[]
    for i in range(n):
        e=[j for j,_ in edges[i]];g=[phi[e[0]]-phi[e[1]],phi[e[2]]-phi[e[3]]];div=math.gcd(*g)
        axis=[x//div for x in g] if div else [1,0];axes.append(axis);factors.extend([(1+axis[j]**2)*g[j] for j in range(2)])
    ip=[0]*n
    for i in sorted(range(n),key=lambda i:(dc[i],i)):
        if i!=c['center']:
            parent=min(j for j,_ in edges[i] if dc[j]==dc[i]-1);ip[i]=(ip[parent]+TURNS[fclass(phi[parent])])&255
    keys=[[a[0]+2,a[1]+2,(dc[i]+1).bit_length()-1,ip[i],i] for i,a in enumerate(axes)]
    ordered=sorted(range(n),key=lambda i:keys[i])
    flat=phi+[x for e in edges for pair in e for x in pair]+factors+[x for key in keys for x in key]+ordered
    return dict(n=n,edges=edges,phi=phi,factors=factors,keys=keys,ordered=ordered,packed=[x&0xffffffff for x in flat])

class Pinion:
    def __init__(self,config=None):
        self.config=dict(width=8,height=8,radius=2,center=0,node=0,phase=250,orientation=0)
        if config:self.config.update(config)
        self.world=world(self.config);n=self.world['n'];c=self.config
        require(integer(c['node'],0,n-1) and integer(c['phase'],0,255) and integer(c['orientation'],0,1),'Invalid state')
        self.set_genes(dict(turns=c.get('turns',TURNS),gains=c.get('gains',GAINS)))
        self.state=[rp(c['phase'],c['node'],self.world['phi'][c['node']],1|(c['orientation']<<4)),0,0,0]+[0]*((12*n+31)//32)
    def set_genes(self,g):
        require(type(g.get('turns')) is list and len(g['turns'])==3 and all(integer(x,0,255) for x in g['turns']) and type(g.get('gains')) is list and len(g['gains'])==8 and all(integer(x,-4,4) for x in g['gains']),'Invalid pinion genes')
        self.turns=g['turns'][:];self.gains=g['gains'][:]
    def seed(self):
        r,i,_,a=unpack(self.state[0]);c=self.config
        header=bytes([84,75,71,49,1,0,c['width'],c['height'],c['radius'],c['center'],i,r,(a>>4)&1]+self.turns+[x&255 for x in self.gains])+struct.pack('<II',*self.state[1:3])
        return header+(struct.pack('<'+'I'*(len(self.state)-4),*self.state[4:]) if any(self.state[4:]) else b'')
    @classmethod
    def from_seed(cls,b):
        require(len(b)>=32 and b[:6]==b'TKG1\x01\x00','Invalid TKG1 packet')
        k=cls(dict(width=b[6],height=b[7],radius=b[8],center=b[9],node=b[10],phase=b[11],orientation=b[12],turns=list(b[13:16]),gains=[x-256 if x>127 else x for x in b[16:24]]))
        require(len(b) in (32,32+4*(len(k.state)-4)),'Overlay length mismatch')
        k.state[1:3]=struct.unpack('<II',b[24:32]);require(k.state[2]<=k.state[1],'Invalid epoch')
        if len(b)>32:k.state[4:]=struct.unpack('<'+'I'*(len(k.state)-4),b[32:])
        used=12*k.world['n']%32
        if used:require(k.state[-1]>>used==0,'Nonzero padding')
        return k
    def cpu(self,bits):
        rows=[];w=self.world;s=self.state
        for bit in bits:
            require(integer(bit,0,1) and max(s[1:3])<0xffffffff,'Invalid bit/counter budget')
            r,i,b,a=unpack(s[0]);eta=(a>>4)&1;theta=(-r if eta else r)&255;bank=theta//64
            q=[w['factors'][2*i+j]*self.gains[2*bank+j] for j in range(2)]
            reduced=[abs(w['phi'][d])-dot for (d,_),dot in zip(w['edges'][i],[q[0],-q[0],q[1],-q[1]])]
            slot=min(range(4),key=lambda j:(reduced[j],w['edges'][i][j][0],j));dest,seam=w['edges'][i][slot]
            row=12*i+4*fclass(b)+slot;delta=self.turns[fclass(b)]^((s[4+row//32]>>(row%32))&1)
            old=rp(delta,dest,w['phi'][dest],1|(seam<<6));ne=eta^seam;nt=(theta+delta)&255
            nw=rp((-nt if ne else nt)&255,dest,w['phi'][dest],1|(ne<<4))
            s[4+row//32]^=bit<<(row%32);s[0]=nw;s[1]+=1;s[2]+=bit
            rows.append([nw,mirror(nw),row,old,rp(delta^bit,dest,w['phi'][dest],1|(seam<<6)),q[0]&0xffffffff,q[1]&0xffffffff,reduced[slot]+1+max(map(abs,q))])
        return rows
    def gpu(self,bits,backend):
        expected=copy.deepcopy(self);trace=expected.cpu(bits)
        p=[self.world['n'],len(bits),1]+self.turns+[x&0xffffffff for x in self.gains]
        result=backend.execute('pinion',self.world['packed'],bits,self.state,p,8*len(bits))
        actual=result['trace'].reshape((-1,8)).tolist()
        require(actual==trace and result['state'].tolist()==expected.state,'GPU/reference mismatch')
        self.state=result['state'].tolist();return actual

class Tape:
    def __init__(self,g):
        self.q=g.get('q',0);self.head=g.get('head',0);self.tick=g.get('tick',0);self.status=g.get('status','READY');cells=g.get('cells',[])
        require(type(cells) is list and all(type(r) is list and len(r)==2 for r in cells) and len(set(r[0] for r in cells))==len(cells),'Invalid/duplicate tape cells')
        self.cells={h:b for h,b in cells if b};self.set_genes(g)
        require(self.status in ('READY','ACCEPT','HALT','INVALID','DEFER'),'Invalid tape status')
        require(integer(self.q,0,self.states-1) and integer(self.head,-2048,2047) and integer(self.tick,0,0xffffffff) and all(integer(h,-2048,2047) and integer(b,0,255) for h,b in self.cells.items()),'Invalid tape state/window')
    def set_genes(self,g):
        states,halt,rules=g.get('states'),g.get('halt'),g.get('rules')
        require(integer(states,1,4096) and integer(halt,0,states-1) and type(rules) is list and len(rules)<=states*256,'Invalid tape genes')
        table={}
        for row in rules:
            require(type(row) is list and len(row)==5,'Invalid tape rule')
            q,a,b,d,n=row;require(integer(q,0,states-1) and integer(a,0,255) and integer(b,0,255) and integer(d,-1,1) and integer(n,0,states-1) and q*256+a not in table,'Invalid/duplicate tape transition')
            table[q*256+a]=[b,d,n]
        require(self.q<states,'Gene change invalidates control')
        self.states,self.halt,self.rules=states,halt,table
    def snapshot(self):return dict(states=self.states,halt=self.halt,rules=[[key//256,key%256,*row] for key,row in self.rules.items()],q=self.q,head=self.head,cells=sorted([h,b] for h,b in self.cells.items()),tick=self.tick,status=self.status)
    def cpu(self,count):
        trace=[]
        for _ in range(count):
            if self.q==self.halt:self.status='HALT';break
            row=self.rules.get(self.q*256+self.cells.get(self.head,0));require(row is not None,'Missing tape transition')
            b,d,q=row;require(-2048<=self.head+d<2048 and self.tick<0xffffffff,'DEFER: tape budget exhausted')
            if b:self.cells[self.head]=b
            else:self.cells.pop(self.head,None)
            self.head+=d;self.q=q;self.tick+=1;self.status='ACCEPT';trace.append([q,self.head,b,self.tick])
        if self.q==self.halt:self.status='HALT'
        return trace
    def gpu(self,count,backend):
        expected=copy.deepcopy(self);trace=expected.cpu(count);program=np.zeros(3*256*self.states,dtype=np.uint32);program[2::3]=0xffffffff
        for key,(b,d,q) in self.rules.items():program[3*key:3*key+3]=[b,d&0xffffffff,q]
        state=np.zeros(4100,dtype=np.uint32);state[:4]=[self.q,self.head+2048,self.tick,0]
        for h,b in self.cells.items():state[4+h+2048]=b
        result=backend.execute('tape',program,[0],state,[self.states,count,1,self.halt,4096],count*4);s=result['state']
        require(s[3] in (0,1),'DEFER or INVALID tape transition')
        actual=result['trace'][:4*len(trace)].reshape((-1,4)).astype(np.int64);actual[:,1]-=2048
        require(actual.tolist()==trace,'Tape GPU trace mismatch')
        self.q,self.head,self.tick=int(s[0]),int(s[1])-2048,int(s[2]);self.status='HALT' if self.q==self.halt else 'ACCEPT';self.cells={h-2048:int(b) for h,b in enumerate(s[4:]) if b}
        require(self.snapshot()==expected.snapshot(),'Tape GPU state mismatch');return trace

def unpack_bits(e):
    require(integer(e.get('count'),1,4096),'Invalid count');b=unb64(e.get('bits'));count=e['count']
    require(len(b)==(count+7)//8 and (not count%8 or b[-1]>>(count%8)==0),'Invalid bit packing')
    return [(b[i//8]>>(i%8))&1 for i in range(count)]

class Machine:
    def __init__(self,checkpoint=None):
        if checkpoint is None:self.role='pinion';self.kernel=Pinion()
        else:
            self.role=checkpoint.get('role');require(self.role in ('pinion','tape'),'Unknown role')
            self.kernel=Pinion.from_seed(unb64(checkpoint['data'])) if self.role=='pinion' else Tape(checkpoint['data'])
    def checkpoint(self):return dict(role=self.role,data=b64(self.kernel.seed()) if self.role=='pinion' else self.kernel.snapshot())
    def apply(self,batch,backend):
        candidate=copy.deepcopy(self);trace=[]
        for e in batch:
            if e['kind']=='idle':continue
            if e['kind']=='genes':
                g=e['genes'];role=g.get('role');require(role in ('pinion','tape'),'Unknown gene role')
                if candidate.role!=role:candidate.role=role;candidate.kernel=Pinion(g) if role=='pinion' else Tape(g)
                else:candidate.kernel.set_genes(g)
            else:
                bits=unpack_bits(e)
                trace+=candidate.kernel.gpu(bits,backend) if candidate.role=='pinion' else candidate.kernel.gpu(len(bits),backend)
        self.role,self.kernel=candidate.role,candidate.kernel
        return dict(checkpoint=self.checkpoint(),trace=trace)

class Rounds:
    def __init__(self,machine,backend):self.machine=machine;self.backend=backend;self.round=0;self.pending={};self.accepted={};self.failed=None
    def receive(self,p):
        require(not self.failed,self.failed);require(integer(p.get('round'),0,4095) and p.get('member') in ('A','B') and p.get('kind') in ('idle','genes','step'),'Invalid hop')
        text=canonical(p);require(len(text)<=65536,'Hop budget exceeded');key=(p['round'],p['member'])
        old=self.accepted.get(key) or self.pending.get(p['round'],{}).get(p['member'],('',None))[0]
        if old:
            if old!=text:self.failed='Contradictory duplicate hop'
            require(not self.failed,self.failed);return []
        require(self.round<=p['round']<=self.round+64,'Hop window exceeded')
        self.pending.setdefault(p['round'],{})[p['member']]=(text,copy.deepcopy(p));out=[]
        while all(m in self.pending.get(self.round,{}) for m in ('A','B')):
            slots=self.pending[self.round]
            try:result=self.machine.apply([slots[m][1] for m in ('A','B')],self.backend)
            except Exception as e:self.failed=str(e);raise
            for m in ('A','B'):self.accepted[self.round,m]=slots[m][0]
            del self.pending[self.round];out.append(dict(type='commit',round=self.round,**result));self.round+=1
        return out
