// Exact integer transitions shared by pointer and texture backends.
typedef unsigned int U;
#ifdef TK_TEXTURE
#include <cuda_runtime.h>
typedef cudaTextureObject_t World;
__device__ U read_word(World w,U i){return tex1Dfetch<U>(w,int(i));}
#else
typedef const U* World;
__device__ U read_word(World w,U i){return w[i];}
#endif
__device__ U rp(U r,U i,int b,U a){U v=r|(i<<8)|((U(b)&255)<<16)|(a<<24);return v|((__popc(v)&1)<<31);}
__device__ U mir(U w){return rp((0-(w&255))&255,(w>>8)&255,int(w<<8)>>24,((w>>24)&127)^16);}
__device__ int ab(int v){return v<0?-v:v;}
// Optional SHA-256 input bits repeat with a global-chain offset.
// Page order and physical timing never select the mutation input.
__device__ U input_bit(const U* p,U chain,U tick){
 if(!p[15])return ((tick+1)&1)?0:1;
 U bit=(tick+p[14]+chain)&255;return (p[16+bit/32]>>(bit%32))&1;
}
__device__ void pinion_chain(World world,const U* bits,U* state,U* trace,const U* p,U chain,bool resident){
 U n=p[0],steps=p[1],words=(12*n+31)/32,base=chain*(4+words);
 if(state[base+3])return;
 for(U t=0;t<steps;t++){
  U w=state[base],r=w&255,i=(w>>8)&255,eta=(w>>28)&1;int b=int(w<<8)>>24;
  if(i>=n||(__popc(w)&1)||b!=int(read_word(world,i))||state[base+1]==0xffffffff||state[base+2]==0xffffffff){state[base+3]=2;return;}
  U theta=eta?(0-r)&255:r,bank=theta/64;
  int qx=int(read_word(world,9*n+2*i))*int(p[6+2*bank]),qy=int(read_word(world,9*n+2*i+1))*int(p[7+2*bank]);
  U lo=0,hi=n;bool found=false;
  while(lo<hi){U mid=(lo+hi-1)/2,k=read_word(world,16*n+mid);int cmp=0;
   for(U j=0;j<5;j++){int x=int(read_word(world,11*n+5*i+j)),y=int(read_word(world,11*n+5*k+j));if(x!=y){cmp=x<y?-1:1;break;}}
   if(!cmp){found=true;break;}if(cmp<0)hi=mid;else lo=mid+1;
  }if(!found){state[base+3]=2;return;}
  int score=2147483647;U dest=0xffffffff,slot=0;
  for(U s=0;s<4;s++){U d=read_word(world,n+8*i+2*s);int dot=s==0?qx:s==1?-qx:s==2?qy:-qy;int sc=ab(int(read_word(world,d)))-dot;if(sc<score||(sc==score&&d<dest)){score=sc;dest=d;slot=s;}}
  U cls=b<0?0:b==0?1:2,row=12*i+4*cls+slot,delta=p[3+cls]^((state[base+4+row/32]>>(row%32))&1),seam=read_word(world,n+8*i+2*slot+1),old=rp(delta,dest,int(read_word(world,dest)),1|(seam<<6));
  U j=resident?input_bit(p,chain,state[base+1]):bits[chain*steps+t];if(j>1){state[base+3]=2;return;}
  U nt=(theta+delta)&255,ne=eta^seam,nr=ne?(0-nt)&255:nt,nw=rp(nr,dest,int(read_word(world,dest)),1|(ne<<4));
  state[base+4+row/32]^=j<<(row%32);state[base]=nw;state[base+1]++;state[base+2]+=j;
  if(trace){U off=8*(chain*steps+t);trace[off]=nw;trace[off+1]=mir(nw);trace[off+2]=row;trace[off+3]=old;trace[off+4]=rp(delta^j,dest,int(read_word(world,dest)),1|(seam<<6));trace[off+5]=U(qx);trace[off+6]=U(qy);trace[off+7]=U(score+1+(ab(qx)>ab(qy)?ab(qx):ab(qy)));}
 }
}
extern "C" __global__ void pinion(World world,const U* bits,U* state,U* trace,const U* p){U chain=blockIdx.x*blockDim.x+threadIdx.x;if(chain<p[2])pinion_chain(world,bits,state,trace,p,chain,false);}
extern "C" __global__ void pinion_resident(World world,U* state,const U* p){U chain=blockIdx.x*blockDim.x+threadIdx.x;if(chain<p[2])pinion_chain(world,0,state,0,p,chain,true);}
extern "C" __global__ void initialize(U* state,const U* p){
 U chain=blockIdx.x*blockDim.x+threadIdx.x;if(chain>=p[2])return;U stride=4+(12*p[0]+31)/32,base=chain*stride;
 for(U i=0;i<stride;i++)state[base+i]=0;
 U w=p[24],id=p[14]+chain;state[base]=rp(((w&255)+id)&255,(w>>8)&255,int(w<<8)>>24,((w>>24)&127)^((id&1)<<4));
}
extern "C" __global__ void tape(World program,const U*,U* state,U* trace,const U* p){
 U chain=blockIdx.x*blockDim.x+threadIdx.x;if(chain>=p[2])return;U base=chain*(4+p[4]);
 for(U t=0;t<p[1];t++){
  U q=state[base];int h=int(state[base+1]);if(q==p[3]){state[base+3]=1;return;}
  if(q>=p[0]||h<0||h>=int(p[4])){state[base+3]=3;return;}
  U a=state[base+4+h],row=3*(256*q+a),next=read_word(program,row+2);
  if(next==0xffffffff||next>=p[0]){state[base+3]=2;return;}
  U b=read_word(program,row);int d=int(read_word(program,row+1));if(b>255||d< -1||d>1){state[base+3]=2;return;}
  int nh=h+d;if(nh<0||nh>=int(p[4])||state[base+2]==0xffffffff){state[base+3]=3;return;}
  state[base+4+h]=b;state[base]=next;state[base+1]=U(nh);state[base+2]++;
  U off=4*(chain*p[1]+t);trace[off]=next;trace[off+1]=U(nh);trace[off+2]=b;trace[off+3]=state[base+2];
 }
}
