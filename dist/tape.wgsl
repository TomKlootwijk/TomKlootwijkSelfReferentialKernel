@group(0) @binding(0) var<storage,read> program: array<u32>;
@group(0) @binding(1) var<storage,read> unused: array<u32>;
@group(0) @binding(2) var<storage,read_write> state: array<u32>;
@group(0) @binding(3) var<storage,read_write> trace: array<u32>;
@group(0) @binding(4) var<storage,read> p: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>){let chain=id.x;if(chain>=p[2]){return;}let base=chain*(4u+p[4]);for(var t=0u;t<p[1];t++){
 let q=state[base];let h=i32(state[base+1u]);if(q==p[3]){state[base+3u]=1u;return;}
 if(q>=p[0]||h<0||h>=i32(p[4])){state[base+3u]=3u;return;}
 let a=state[base+4u+u32(h)];let row=3u*(256u*q+a);let next=program[row+2u];if(next==0xffffffffu||next>=p[0]){state[base+3u]=2u;return;}
 let b=program[row];let d=i32(program[row+1u]);if(b>255u||d< -1||d>1){state[base+3u]=2u;return;}
 let nh=h+d;if(nh<0||nh>=i32(p[4])||state[base+2u]==0xffffffffu){state[base+3u]=3u;return;}
 state[base+4u+u32(h)]=b;state[base]=next;state[base+1u]=u32(nh);state[base+2u]++;
 let off=4u*(chain*p[1]+t);trace[off]=next;trace[off+1u]=u32(nh);trace[off+2u]=b;trace[off+3u]=state[base+2u];
 }}
