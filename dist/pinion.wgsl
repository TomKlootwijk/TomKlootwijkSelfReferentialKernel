@group(0) @binding(0) var<storage,read> world: array<u32>;
@group(0) @binding(1) var<storage,read> bits: array<u32>;
@group(0) @binding(2) var<storage,read_write> state: array<u32>;
@group(0) @binding(3) var<storage,read_write> trace: array<u32>;
@group(0) @binding(4) var<storage,read> p: array<u32>;
fn pack(r:u32,i:u32,b:i32,a:u32)->u32 {let v=r|(i<<8u)|((u32(b)&255u)<<16u)|(a<<24u);return v|((countOneBits(v)&1u)<<31u);}
fn mirror(w:u32)->u32 {let b=i32(w<<8u)>>24;return pack((0u-(w&255u))&255u,(w>>8u)&255u,b,((w>>24u)&127u)^16u);}
fn keycmp(a:u32,b:u32,n:u32)->i32 {for(var j=0u;j<5u;j++){let x=i32(world[11u*n+5u*a+j]);let y=i32(world[11u*n+5u*b+j]);if(x<y){return -1;}if(x>y){return 1;}}return 0;}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>){
 let chain=id.x;let n=p[0];let steps=p[1];if(chain>=p[2]){return;}
 let words=(12u*n+31u)/32u;let base=chain*(4u+words);
 for(var t=0u;t<steps;t++){
  let w=state[base];let r=w&255u;let i=(w>>8u)&255u;let b=i32(w<<8u)>>24;let eta=(w>>28u)&1u;
  if(i>=n||countOneBits(w)%2u!=0u||b!=i32(world[i])||state[base+1u]==0xffffffffu||state[base+2u]==0xffffffffu){state[base+3u]=2u;return;}
  let theta=select(r,(0u-r)&255u,eta==1u);let bank=theta/64u;
  let qx=i32(world[9u*n+2u*i])*i32(p[6u+2u*bank]);let qy=i32(world[9u*n+2u*i+1u])*i32(p[7u+2u*bank]);
  var lo=0u;var hi=n;var found=false;loop{if(lo>=hi){break;}let mid=(lo+hi-1u)/2u;let k=world[16u*n+mid];let cmp=keycmp(i,k,n);if(cmp==0){found=true;break;}if(cmp<0){hi=mid;}else{lo=mid+1u;}}
  if(!found){state[base+3u]=2u;return;}
  var score=2147483647;var dest=0xffffffffu;var slot=0u;
  for(var s=0u;s<4u;s++){let d=world[n+8u*i+2u*s];var dot=0;if(s==0u){dot=qx;}if(s==1u){dot=-qx;}if(s==2u){dot=qy;}if(s==3u){dot=-qy;}let sc=abs(i32(world[d]))-dot;if(sc<score||(sc==score&&d<dest)){score=sc;dest=d;slot=s;}}
  var cls=2u;if(b<0){cls=0u;}if(b==0){cls=1u;}
  let row=12u*i+4u*cls+slot;let delta=p[3u+cls]^((state[base+4u+row/32u]>>(row%32u))&1u);
  let seam=world[n+8u*i+2u*slot+1u];let old=pack(delta,dest,i32(world[dest]),1u|(seam<<6u));
  let jitter=bits[chain*steps+t];if(jitter>1u){state[base+3u]=2u;return;}
  let nt=(theta+delta)&255u;let ne=eta^seam;let nr=select(nt,(0u-nt)&255u,ne==1u);let nw=pack(nr,dest,i32(world[dest]),1u|(ne<<4u));
  state[base+4u+row/32u]^=jitter<<(row%32u);state[base]=nw;state[base+1u]++;state[base+2u]+=jitter;
  let off=8u*(chain*steps+t);trace[off]=nw;trace[off+1u]=mirror(nw);trace[off+2u]=row;trace[off+3u]=old;trace[off+4u]=pack(delta^jitter,dest,i32(world[dest]),1u|(seam<<6u));trace[off+5u]=u32(qx);trace[off+6u]=u32(qy);trace[off+7u]=u32(score+1+max(abs(qx),abs(qy)));
 }
}
