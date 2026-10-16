// PUCT tree search (AlphaZero style) over single actions. Written as generators: whenever a position needs the
// network, the search yields {s, idx} and expects back {v, logits} (v for the player to move in s; one logit per idx).
const END_IDX=65535;
module.exports=function makeMCTS(G,opt={}){
  const cpuct=opt.cpuct||1.6, fpuRed=opt.fpu||0.25;
  const node=s=>({s,p:G.toMove(s),term:!!s.over,exp:false,acts:null,idx:null,P:null,N:null,W:null,kids:null,n:0,v:0});
  function prepare(nd){ // legal actions + their network indices
    nd.acts=G.legal(nd.s);
    nd.idx=new Uint16Array(nd.acts.length);
    for(let i=0;i<nd.acts.length;i++){const j=G.actIndex(nd.s,nd.acts[i]);nd.idx[i]=j<0?END_IDX:j;}
  }
  function expand(nd,r){
    const L=nd.acts.length, P=new Float32Array(L); let mx=-1e9;
    for(let i=0;i<L;i++) if(r.logits[i]>mx) mx=r.logits[i];
    let sum=0; for(let i=0;i<L;i++){P[i]=Math.exp(r.logits[i]-mx);sum+=P[i];}
    for(let i=0;i<L;i++) P[i]/=sum;
    nd.P=P; nd.N=new Uint32Array(L); nd.W=new Float32Array(L); nd.kids=new Array(L); nd.v=r.v; nd.exp=true;
  }
  function noise(nd,alpha=0.3,eps=0.25){
    // Dirichlet noise via normalised Gamma samples (Marsaglia-Tsang for alpha<1 using boost)
    const g=a=>{const d=a+1-1/3,c=1/Math.sqrt(9*d);for(;;){let x,v;do{x=gauss();v=1+c*x;}while(v<=0);v=v*v*v;const u=Math.random();if(Math.log(u)<0.5*x*x+d-d*v+d*Math.log(v))return d*v*Math.pow(Math.random(),1/a);}};
    const L=nd.P.length, a=Math.min(1,alpha*10/L*3), n=new Float32Array(L); let s=0;
    for(let i=0;i<L;i++){n[i]=g(a);s+=n[i];}
    for(let i=0;i<L;i++) nd.P[i]=(1-eps)*nd.P[i]+eps*n[i]/s;
  }
  function gauss(){let u=0,v=0;while(!u)u=Math.random();while(!v)v=Math.random();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
  function select(nd){
    const sq=Math.sqrt(nd.n+1), fpu=nd.v-fpuRed; let best=0,bs=-1e9;
    for(let i=0;i<nd.P.length;i++){
      const q=nd.N[i]?nd.W[i]/nd.N[i]:fpu, u=q+cpuct*nd.P[i]*sq/(1+nd.N[i]);
      if(u>bs){bs=u;best=i;}
    }
    return best;
  }
  function* think(root,sims,addNoise){
    if(root.term) return;
    if(!root.exp){prepare(root);const r=yield {s:root.s,idx:root.idx};expand(root,r);}
    if(addNoise) noise(root);
    for(let it=0;it<sims;it++){
      let nd=root; const path=[];
      while(nd.exp&&!nd.term){
        const i=select(nd); path.push([nd,i]);
        if(!nd.kids[i]) nd.kids[i]=node(G.apply(nd.s,nd.acts[i]));
        nd=nd.kids[i];
      }
      let leafV;   // value for player `who`
      const who=nd.p;
      if(nd.term){ leafV=null; }
      else { prepare(nd); const r=yield {s:nd.s,idx:nd.idx}; expand(nd,r); leafV=r.v; }
      for(const [par,i] of path){
        const v=nd.term?G.result(nd.s,par.p):(par.p===who?leafV:-leafV);
        par.N[i]++; par.W[i]+=v; par.n++;
      }
    }
  }
  function policy(root){ // visit distribution: indices (END=65535) and probabilities, plus all legal indices
    const tot=root.N.reduce((a,b)=>a+b,0)||1;
    return {idx:Array.from(root.idx),p:Array.from(root.N,x=>x/tot)};
  }
  function choose(root,temp){
    if(!temp){let b=0;for(let i=1;i<root.N.length;i++)if(root.N[i]>root.N[b])b=i;return b;}
    let t=0; const w=Array.from(root.N,x=>Math.pow(x,1/temp)); for(const x of w)t+=x;
    let r=Math.random()*t; for(let i=0;i<w.length;i++){r-=w[i];if(r<=0)return i;} return w.length-1;
  }
  function child(root,i){ return root.kids[i]||node(G.apply(root.s,root.acts[i])); }
  return {node,think,policy,choose,child,END_IDX};
};
