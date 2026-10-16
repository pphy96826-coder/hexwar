// PUCT search for the browser: same rules as mcts.js, but collects a batch of leaves per network call
// (virtual loss keeps the batch spread over different lines).
// evalBatch(list of {s, idx}) -> Promise of [{v, logits}] with v for the player to move in s.
function makeMCTSWeb(G,opt={}){
  const cpuct=opt.cpuct||1.6, fpuRed=opt.fpu||0.25, VL=opt.vloss||1, END_IDX=65535;
  const node=s=>({s,p:G.toMove(s),term:!!s.over,exp:false,pend:false,acts:null,idx:null,P:null,N:null,W:null,kids:null,n:0,v:0});
  function prepare(nd){
    if(nd.acts) return;
    nd.acts=G.legal(nd.s); nd.idx=new Uint16Array(nd.acts.length);
    for(let i=0;i<nd.acts.length;i++){const j=G.actIndex(nd.s,nd.acts[i]);nd.idx[i]=j<0?END_IDX:j;}
  }
  function expand(nd,r){
    const L=nd.acts.length, P=new Float32Array(L); let mx=-1e9;
    for(let i=0;i<L;i++) if(r.logits[i]>mx) mx=r.logits[i];
    let sum=0; for(let i=0;i<L;i++){P[i]=Math.exp(r.logits[i]-mx);sum+=P[i];}
    for(let i=0;i<L;i++) P[i]/=sum;
    nd.P=P; nd.N=new Float32Array(L); nd.W=new Float32Array(L); nd.kids=nd.kids||new Array(L); nd.v=r.v; nd.exp=true;
  }
  function select(nd){
    const sq=Math.sqrt(nd.n+1), fpu=nd.v-fpuRed; let best=0,bs=-1e9;
    for(let i=0;i<nd.P.length;i++){
      const q=nd.N[i]?nd.W[i]/nd.N[i]:fpu, u=q+cpuct*nd.P[i]*sq/(1+nd.N[i]);
      if(u>bs){bs=u;best=i;}
    }
    return best;
  }
  // one round: up to `batch` leaves, evaluated together. Returns the number of simulations completed.
  async function step(root,evalBatch,batch){
    if(root.term) return 0;
    if(!root.exp){ prepare(root); const [r]=await evalBatch([{s:root.s,idx:root.idx}]); if(!root.exp) expand(root,r); return 1; }
    const leaves=[]; let done=0, misses=0;
    while(leaves.length<batch&&misses<batch){
      let nd=root; const path=[];
      while(nd.exp&&!nd.term){
        const i=select(nd); path.push([nd,i]);
        if(!nd.kids[i]) nd.kids[i]=node(G.apply(nd.s,nd.acts[i]));
        nd=nd.kids[i];
      }
      if(nd.term){ for(const [par,i] of path){par.N[i]++;par.W[i]+=G.result(nd.s,par.p);par.n++;} done++; continue; }
      if(nd.pend){ misses++; if(!leaves.length) break; else continue; }
      for(const [par,i] of path){par.N[i]+=VL;par.W[i]-=VL;par.n+=VL;}   // virtual loss
      nd.pend=true; prepare(nd); leaves.push({nd,path});
    }
    if(!leaves.length) return done;
    let res;
    try{ res=await evalBatch(leaves.map(l=>({s:l.nd.s,idx:l.nd.idx}))); }
    catch(e){   // undo the virtual losses so the tree stays exact, then report
      for(const {nd,path} of leaves){ nd.pend=false; for(const [par,i] of path){par.N[i]-=VL;par.W[i]+=VL;par.n-=VL;} }
      throw e;
    }
    leaves.forEach(({nd,path},j)=>{
      const r=res[j]; nd.pend=false; if(!nd.exp) expand(nd,r);
      const who=nd.p;
      for(const [par,i] of path){
        const v=par.p===who?r.v:-r.v;
        par.N[i]+=1-VL; par.W[i]+=v+VL; par.n+=1-VL;
      }
    });
    return done+leaves.length;
  }
  function visits(root){ return root.N?Array.from(root.N):[]; }
  function best(root){ let b=0; for(let i=1;i<root.N.length;i++) if(root.N[i]>root.N[b]) b=i; return b; }
  function child(root,i){ prepare(root); if(!root.kids) root.kids=new Array(root.acts.length); if(!root.kids[i]) root.kids[i]=node(G.apply(root.s,root.acts[i])); return root.kids[i]; }
  return {node,step,visits,best,child,prepare,END_IDX};
}
if(typeof module!=='undefined') module.exports=makeMCTSWeb;
