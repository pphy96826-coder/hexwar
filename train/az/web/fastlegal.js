// Same result as game.js legal(), but computes the supply network once instead of once per unit type.
function makeFastLegal(E,G){
  const {U,TYPES,BASE,NB1,NB2,CELLS}=E, NT=TYPES.length;
  return function legal(s){
    const p=G.toMove(s), acts=[];
    if(s.placed<8){
      const occ=E.buildOcc(s), conn=E.connect(s,p,occ), TER=E.TER, cache=new Map();
      let enemyCover=null;
      const has=new Set(); for(const u of s.units) if(u.owner===p) has.add(u.type);
      for(let t=0;t<NT;t++){
        const ty=TYPES[t], d=U[ty]; if(d.cost>s.bank[p]) continue;
        if(d.unique&&has.has(ty)) continue;
        const key=(d.dash?1:0)|((d.river||d.fly)?2:0)|(d.drop?4:0)|(d.boom?8:0);
        let out=cache.get(key);
        if(!out){
          out=new Set();
          const rv=!!(d.river||d.fly);
          const ok=k=>!occ.has(k)&&k!==BASE[0]&&k!==BASE[1]&&TER(k)!=='rock'&&(TER(k)!=='river'||rv);
          const R=d.dash?NB2:NB1;
          for(const k of conn){
            for(const n of R.get(k)) if(ok(n)) out.add(n);
            const v=occ.get(k); if(v&&v.type==='banner') for(const n of NB2.get(k)) if(ok(n)) out.add(n);
          }
          for(const n of NB2.get(BASE[p])) if(ok(n)) out.add(n);
          if(d.drop) for(const c of CELLS) if(ok(c.k)&&E.distK(c.k,BASE[1-p])>2) out.add(c.k);
          if(d.boom&&out.size){
            for(const n of NB1.get(BASE[1-p])) out.delete(n);
            if(!enemyCover){enemyCover=new Set(); for(const u of s.units) if(u.owner!==p&&U[u.type].atk>0&&!U[u.type].onlyBase) for(const k of E.cover(u,occ)) enemyCover.add(k);}
            for(const k of enemyCover) out.delete(k);
          }
          cache.set(key,out);
        }
        for(const k of out) acts.push(t*10000+k);
      }
    }
    acts.push(G.END);
    return acts;
  };
}
if(typeof module!=='undefined') module.exports=makeFastLegal;
