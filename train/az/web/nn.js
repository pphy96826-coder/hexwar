// Policy/value network inference in the browser (and Node, for testing). Weights come from export_web.py:
// BatchNorm and input scaling are already folded into the convolutions.
// forward(x, B, H, W): x = Uint8Array [B][87][H][W] -> {v: Float32Array[B], e: Float32Array[B], lg: Float32Array[B][NT][H][W]}
const HexNN=(()=>{
  function parse(buf){
    const dv=new DataView(buf), hl=dv.getUint32(0,true);
    const hdr=JSON.parse(new TextDecoder().decode(new Uint8Array(buf,4,hl)));
    const off=4+hl+((4-(4+hl)%4)%4), all=new Float32Array(buf,off,hdr.n);
    const {f,cin,nt,blocks}=hdr; let o=0;
    const take=n=>{const a=all.subarray(o,o+n);o+=n;return a;};
    const W={hdr,stemW:take(f*cin*9),stemB:take(f),blocks:[]};
    for(let i=0;i<blocks;i++) W.blocks.push({w1:take(f*f*9),b1:take(f),w2:take(f*f*9),b2:take(f)});
    W.p1w=take(32*f); W.p1b=take(32); W.p2w=take(nt*32); W.p2b=take(nt);
    W.v1w=take(128*2*f); W.v1b=take(128); W.v2w=take(128); W.v2b=take(1);
    W.e1w=take(64*2*f); W.e1b=take(64); W.e2w=take(64); W.e2b=take(1);
    if(o!==hdr.n) throw new Error('weight size mismatch');
    return W;
  }
  // ---------------- CPU reference (slow; used for checking and as a last resort) ----------------
  function conv3(inp,cin,w,b,cout,H,W){
    const HW=H*W, out=new Float32Array(cout*HW);
    for(let o=0;o<cout;o++){
      const ob=o*HW; out.fill(b[o],ob,ob+HW);
      for(let c=0;c<cin;c++){
        const ib=c*HW, wb=(o*cin+c)*9;
        for(let dy=-1;dy<=1;dy++) for(let dx=-1;dx<=1;dx++){
          const wv=w[wb+(dy+1)*3+dx+1]; if(!wv) continue;
          for(let y=Math.max(0,-dy);y<Math.min(H,H-dy);y++){
            const ro=ob+y*W, ri=ib+(y+dy)*W+dx;
            for(let x=Math.max(0,-dx);x<Math.min(W,W-dx);x++) out[ro+x]+=wv*inp[ri+x];
          }
        }
      }
    }
    return out;
  }
  function forwardCPU(Wt,x,B,H,W){
    const {f,cin,nt}=Wt.hdr, HW=H*W;
    const v=new Float32Array(B), e=new Float32Array(B), lg=new Float32Array(B*nt*HW);
    for(let bi=0;bi<B;bi++){
      const xi=new Float32Array(cin*HW); for(let i=0;i<cin*HW;i++) xi[i]=x[bi*cin*HW+i];
      const m=xi.subarray(0,HW);
      let h=conv3(xi,cin,Wt.stemW,Wt.stemB,f,H,W);
      for(let c=0;c<f;c++) for(let i=0;i<HW;i++){const j=c*HW+i; h[j]=Math.max(0,h[j])*m[i];}
      for(const bl of Wt.blocks){
        const y=conv3(h,f,bl.w1,bl.b1,f,H,W);
        for(let c=0;c<f;c++) for(let i=0;i<HW;i++){const j=c*HW+i; y[j]=Math.max(0,y[j])*m[i];}
        const z=conv3(y,f,bl.w2,bl.b2,f,H,W);
        for(let c=0;c<f;c++) for(let i=0;i<HW;i++){const j=c*HW+i; h[j]=Math.max(0,h[j]+z[j]*m[i]);}
      }
      // policy head: 1x1 convs
      const t=new Float32Array(32);
      for(let i=0;i<HW;i++){
        for(let o=0;o<32;o++){let s=Wt.p1b[o]; for(let c=0;c<f;c++) s+=Wt.p1w[o*f+c]*h[c*HW+i]; t[o]=Math.max(0,s);}
        for(let o=0;o<nt;o++){let s=Wt.p2b[o]; for(let c=0;c<32;c++) s+=Wt.p2w[o*32+c]*t[c]; lg[(bi*nt+o)*HW+i]=s;}
      }
      // pooled features: masked mean and max
      let cnt=0; for(let i=0;i<HW;i++) cnt+=m[i]; cnt=Math.max(1,cnt);
      const g=new Float32Array(2*f);
      for(let c=0;c<f;c++){let s=0,mx=-1e30; for(let i=0;i<HW;i++){const a=h[c*HW+i]; s+=a; const q=a-(1-m[i])*1e4; if(q>mx)mx=q;} g[c]=s/cnt; g[f+c]=mx;}
      const mlp=(w1,b1,n1,w2,b2)=>{let out=b2[0]; for(let o=0;o<n1;o++){let s=b1[o]; for(let c=0;c<2*f;c++) s+=w1[o*2*f+c]*g[c]; out+=w2[o]*Math.max(0,s);} return out;};
      v[bi]=Math.tanh(mlp(Wt.v1w,Wt.v1b,128,Wt.v2w,Wt.v2b));
      e[bi]=mlp(Wt.e1w,Wt.e1b,64,Wt.e2w,Wt.e2b);
    }
    return {v,e,lg};
  }
  // ---------------- WebGPU ----------------
  // conv3x3 over [B][Cin][H][W]; each thread computes 4 output channels at one cell.
  // mode 0: out = relu(conv)*m     mode 1: out = relu(res + conv*m)
  // conv3x3 as a tiled matrix product: out[o][n] = sum_k Wt[k][o] * col[k][n], k = (cin, tap), n = (position, cell).
  // A workgroup computes a 64 (channels) x 64 (cells) tile, each thread 4x4; weights are stored transposed [k][o].
  // mode 0: out = relu(conv)*m     mode 1: out = relu(res + conv*m)
  const CONV=`
struct P {B:u32, cin:u32, cout:u32, H:u32, W:u32, mode:u32, woff:u32, boff:u32}
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage,read> inp:array<f32>;
@group(0) @binding(2) var<storage,read> wt:array<f32>;
@group(0) @binding(3) var<storage,read> msk:array<f32>;
@group(0) @binding(4) var<storage,read> res:array<f32>;
@group(0) @binding(5) var<storage,read_write> outp:array<f32>;
var<workgroup> Ws:array<f32,1024>;
var<workgroup> Xs:array<f32,1024>;
@compute @workgroup_size(16,16)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) l:vec3<u32>){
  let HW=p.H*p.W; let N=p.B*HW; let K=p.cin*9u;
  let n0=(wg.x+wg.y*65535u)*64u;
  let tid=l.y*16u+l.x; let col=tid%64u; let q=tid/64u;
  // the input column this thread loads never changes: decode it once
  let n=n0+col; let nok=n<N; let nb=n/HW; let cell=n%HW; let cy=i32(cell/p.W); let cx=i32(cell%p.W);
  let base=nb*p.cin*HW;
  let bo=p.boff+l.y*4u;
  let bias=vec4<f32>(wt[bo],wt[bo+1u],wt[bo+2u],wt[bo+3u]);
  var a0=bias; var a1=bias; var a2=bias; var a3=bias;
  for(var k0=0u;k0<K;k0+=16u){
    for(var r=0u;r<4u;r++){
      let kk=q+4u*r; let e=kk*64u+col; let k=k0+kk;
      var wv=0.0; var xv=0.0;
      if(k<K){
        wv=wt[p.woff+k*64u+col];
        if(nok){
          let c=k/9u; let tap=k-c*9u; let ty=tap/3u;
          let y=cy+i32(ty)-1; let x=cx+i32(tap-ty*3u)-1;
          if(y>=0 && y<i32(p.H) && x>=0 && x<i32(p.W)){ xv=inp[base+c*HW+u32(y)*p.W+u32(x)]; }
        }
      }
      Ws[e]=wv; Xs[e]=xv;
    }
    workgroupBarrier();
    for(var kk=0u;kk<16u;kk++){
      let wb=kk*64u+l.y*4u; let xb=kk*64u+l.x;
      let w=vec4<f32>(Ws[wb],Ws[wb+1u],Ws[wb+2u],Ws[wb+3u]);
      a0+=w*Xs[xb]; a1+=w*Xs[xb+16u]; a2+=w*Xs[xb+32u]; a3+=w*Xs[xb+48u];
    }
    workgroupBarrier();
  }
  let accs=array<vec4<f32>,4>(a0,a1,a2,a3);
  for(var j=0u;j<4u;j++){
    let nn=n0+l.x+16u*j; if(nn>=N){continue;}
    let b=nn/HW; let ce=nn%HW; let m=msk[b*HW+ce]; let av=accs[j];
    for(var i=0u;i<4u;i++){
      let oi=(b*p.cout+l.y*4u+i)*HW+ce;
      if(p.mode==0u){outp[oi]=max(av[i],0.0)*m;} else {outp[oi]=max(res[oi]+av[i]*m,0.0);}
    }
  }
}`;
  // policy head (1x1 convs) at every cell: f -> 32 (relu) -> nt
  const POL=`
struct P {B:u32, f:u32, nt:u32, HW:u32, o1w:u32, o1b:u32, o2w:u32, o2b:u32, lgoff:u32}
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage,read> h:array<f32>;
@group(0) @binding(2) var<storage,read> wt:array<f32>;
@group(0) @binding(3) var<storage,read_write> outp:array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
  let id=gid.x; if(id>=p.B*p.HW){return;}
  let b=id/p.HW; let cell=id%p.HW;
  var t:array<f32,32>;
  for(var o=0u;o<32u;o++){
    var s=wt[p.o1b+o];
    for(var c=0u;c<p.f;c++){ s+=wt[p.o1w+o*p.f+c]*h[(b*p.f+c)*p.HW+cell]; }
    t[o]=max(s,0.0);
  }
  for(var o=0u;o<p.nt;o++){
    var s=wt[p.o2b+o];
    for(var c=0u;c<32u;c++){ s+=wt[p.o2w+o*32u+c]*t[c]; }
    outp[p.lgoff+(b*p.nt+o)*p.HW+cell]=s;
  }
}`;
  // pooled features and the value / end heads, one workgroup per position
  const HEAD=`
struct P {B:u32, f:u32, HW:u32, v1w:u32, v1b:u32, v2w:u32, v2b:u32, e1w:u32, e1b:u32, e2w:u32, e2b:u32}
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage,read> h:array<f32>;
@group(0) @binding(2) var<storage,read> wt:array<f32>;
@group(0) @binding(3) var<storage,read> msk:array<f32>;
@group(0) @binding(4) var<storage,read_write> outp:array<f32>;
var<workgroup> g:array<f32,256>;
var<workgroup> hv:array<f32,128>;
var<workgroup> he:array<f32,64>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>){
  let b=wid.x; let i=lid.x;
  if(i<p.f){
    var s=0.0; var mx=-1e30; var cnt=0.0;
    for(var k=0u;k<p.HW;k++){
      let m=msk[b*p.HW+k]; let a=h[(b*p.f+i)*p.HW+k];
      s+=a; cnt+=m; mx=max(mx,a-(1.0-m)*1e4);
    }
    g[i]=s/max(cnt,1.0); g[p.f+i]=mx;
  }
  workgroupBarrier();
  let n=2u*p.f;
  var s=wt[p.v1b+i]; for(var c=0u;c<n;c++){ s+=wt[p.v1w+i*n+c]*g[c]; } hv[i]=max(s,0.0)*wt[p.v2w+i];
  if(i<64u){ var q=wt[p.e1b+i]; for(var c=0u;c<n;c++){ q+=wt[p.e1w+i*n+c]*g[c]; } he[i]=max(q,0.0)*wt[p.e2w+i]; }
  workgroupBarrier();
  if(i==0u){
    var v=wt[p.v2b]; for(var k=0u;k<128u;k++){ v+=hv[k]; }
    var e=wt[p.e2b]; for(var k=0u;k<64u;k++){ e+=he[k]; }
    outp[b]=tanh(v); outp[p.B+b]=e;
  }
}`;
  async function gpu(Wt){
    if(typeof navigator==='undefined'||!navigator.gpu) return null;
    const ad=await navigator.gpu.requestAdapter({powerPreference:'high-performance'}); if(!ad) return null;
    const dev=await ad.requestDevice({requiredLimits:{maxStorageBufferBindingSize:Math.min(ad.limits.maxStorageBufferBindingSize,1<<30),maxBufferSize:Math.min(ad.limits.maxBufferSize,1<<30)}});
    let lost=false; dev.lost.then(()=>{lost=true;});
    const {f,cin,nt,blocks}=Wt.hdr;
    // all weights in one buffer, with offsets (in floats)
    const tr=(w,ci)=>{const o=new Float32Array(w.length),K=ci*9; for(let oc=0;oc<f;oc++) for(let k=0;k<K;k++) o[k*f+oc]=w[oc*K+k]; return o;};
    const parts=[tr(Wt.stemW,cin),Wt.stemB]; for(const b of Wt.blocks) parts.push(tr(b.w1,f),b.b1,tr(b.w2,f),b.b2);
    parts.push(Wt.p1w,Wt.p1b,Wt.p2w,Wt.p2b,Wt.v1w,Wt.v1b,Wt.v2w,Wt.v2b,Wt.e1w,Wt.e1b,Wt.e2w,Wt.e2b);
    const offs=[]; let n=0; for(const a of parts){offs.push(n);n+=a.length;}
    const wbuf=dev.createBuffer({size:n*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    {const all=new Float32Array(n); parts.forEach((a,i)=>all.set(a,offs[i])); dev.queue.writeBuffer(wbuf,0,all);}
    const mk=code=>dev.createComputePipeline({layout:'auto',compute:{module:dev.createShaderModule({code}),entryPoint:'main'}});
    const pConv=mk(CONV), pPol=mk(POL), pHead=mk(HEAD);
    let cap=null;   // buffers sized for (B, H, W)
    function ensure(B,H,W){
      const HW=H*W;
      if(cap&&cap.B>=B&&cap.HW>=HW) return cap;
      if(cap) for(const k of ['x','m','h0','h1','h2','o','rd']) cap[k].destroy();
      const S=GPUBufferUsage.STORAGE, D=GPUBufferUsage.COPY_DST, Sr=GPUBufferUsage.COPY_SRC;
      const big=(fl,u)=>dev.createBuffer({size:Math.max(16,fl*4),usage:u});
      const nout=B*(2+nt*HW);
      cap={B,HW,x:big(B*cin*HW,S|D),m:big(B*HW,S|D),h0:big(B*f*HW,S),h1:big(B*f*HW,S),h2:big(B*f*HW,S),o:big(nout,S|Sr),rd:big(nout,GPUBufferUsage.MAP_READ|D)};
      return cap;
    }
    const uni=vals=>{const b=dev.createBuffer({size:Math.max(16,Math.ceil(vals.length*4/16)*16),usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}); dev.queue.writeBuffer(b,0,new Uint32Array(vals)); return b;};
    let busy=Promise.resolve();
    async function forward(x,B,H,W){
      const run=async()=>{
        const HW=H*W, c=ensure(B,H,W);
        const xf=new Float32Array(B*cin*HW); for(let i=0;i<xf.length;i++) xf[i]=x[i];
        const mf=new Float32Array(B*HW); for(let b=0;b<B;b++) for(let i=0;i<HW;i++) mf[b*HW+i]=x[b*cin*HW+i];
        dev.queue.writeBuffer(c.x,0,xf); dev.queue.writeBuffer(c.m,0,mf);
        const enc=dev.createCommandEncoder(), tmp=[];
        const conv=(inp,ci,wi,res,out,mode)=>{
          const u=uni([B,ci,f,H,W,mode,offs[wi],offs[wi+1]]); tmp.push(u);
          const bg=dev.createBindGroup({layout:pConv.getBindGroupLayout(0),entries:[u,inp,wbuf,c.m,res,out].map((bf,i)=>({binding:i,resource:{buffer:bf}}))});
          const ps=enc.beginComputePass(); ps.setPipeline(pConv); ps.setBindGroup(0,bg);
          const groups=Math.ceil(B*HW/64); ps.dispatchWorkgroups(Math.min(groups,65535),Math.ceil(groups/65535)); ps.end();
        };
        conv(c.x,cin,0,c.h2,c.h0,0);
        let h=c.h0, o1=c.h1, o2=c.h2;
        for(let i=0;i<blocks;i++){
          const wi=2+i*4;
          conv(h,f,wi,h,o1,0);          // y = relu(conv1 h)*m
          conv(o1,f,wi+2,h,o2,1);       // h' = relu(h + conv2(y)*m)
          [h,o2]=[o2,h];
        }
        const pi=2+blocks*4;
        {const u=uni([B,f,nt,HW,offs[pi],offs[pi+1],offs[pi+2],offs[pi+3],2*B]); tmp.push(u);
         const bg=dev.createBindGroup({layout:pPol.getBindGroupLayout(0),entries:[u,h,wbuf,c.o].map((bf,i)=>({binding:i,resource:{buffer:bf}}))});
         const ps=enc.beginComputePass(); ps.setPipeline(pPol); ps.setBindGroup(0,bg); ps.dispatchWorkgroups(Math.ceil(B*HW/64)); ps.end();}
        {const u=uni([B,f,HW,...[4,5,6,7,8,9,10,11].map(j=>offs[pi+j])]); tmp.push(u);
         const bg=dev.createBindGroup({layout:pHead.getBindGroupLayout(0),entries:[u,h,wbuf,c.m,c.o].map((bf,i)=>({binding:i,resource:{buffer:bf}}))});
         const ps=enc.beginComputePass(); ps.setPipeline(pHead); ps.setBindGroup(0,bg); ps.dispatchWorkgroups(B); ps.end();}
        const nout=B*(2+nt*HW);
        enc.copyBufferToBuffer(c.o,0,c.rd,0,nout*4);
        dev.queue.submit([enc.finish()]);
        if(lost) throw new Error('GPU device lost');
        let to; const tmo=new Promise((_,rej)=>{to=setTimeout(()=>rej(new Error('GPU timeout')),10000);});
        try{ await Promise.race([c.rd.mapAsync(GPUMapMode.READ,0,nout*4),tmo]); } finally { clearTimeout(to); }
        const r=new Float32Array(c.rd.getMappedRange(0,nout*4).slice(0)); c.rd.unmap();
        for(const u of tmp) u.destroy();
        return {v:r.subarray(0,B),e:r.subarray(B,2*B),lg:r.subarray(2*B)};
      };
      const p=busy.then(run); busy=p.catch(()=>{}); return p;
    }
    return {forward,adapter:ad.info||{},get lost(){return lost;}};
  }
  return {parse,forwardCPU,gpu};
})();
if(typeof module!=='undefined') module.exports=HexNN;
