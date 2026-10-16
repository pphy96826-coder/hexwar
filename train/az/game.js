// AlphaZero-style interface over the real engine (E = exports of hexwar.html, with a map already loaded).
// One action = place one unit (type index t, cell k) or END the turn. When the second player ends, the round
// resolves with simEndRound (verified identical to the live game).
// State: {round,mover,units,roads,bank,hp,rage,pending,nextId,placed,over,winner}; mover 0/1 = first/second to act.
const END=-1, MAX_PLACE=8, MAX_ROUND=40;
const C=87;   // input planes, see encode()
module.exports=function makeGame(E){
  const U=E.U, TYPES=E.TYPES, NT=TYPES.length, {FINALE}=E.consts;
  const first=s=>(s.round-1)%2;
  const toMove=s=>s.mover?1-first(s):first(s);
  function newState(){
    return {round:1,mover:0,units:[],roads:{},bank:[E.income(1),E.income(1)],hp:[20,20],rage:[0,0],pending:[],nextId:1,placed:0,over:false,winner:null};
  }
  function legal(s){
    const p=toMove(s), acts=[];
    if(s.placed<MAX_PLACE) for(let t=0;t<NT;t++){
      const ty=TYPES[t]; if(U[ty].cost>s.bank[p]) continue;
      for(const k of E.deployable(s,p,ty)) acts.push(t*10000+k);
    }
    acts.push(END);
    return acts;
  }
  function apply(s,a){
    if(a===END){
      if(!s.mover) return Object.assign({},s,{mover:1,placed:0});
      const n=E.simEndRound(s); n.placed=0;
      if(!n.over&&n.round>MAX_ROUND){n.over=true;n.winner=-1;}
      return n;
    }
    const p=toMove(s), t=Math.floor(a/10000), k=a%10000, ty=TYPES[t];
    const n=Object.assign({},s,{units:s.units.slice(),bank:s.bank.slice(),placed:s.placed+1,nextId:s.nextId+1});
    n.bank[p]-=U[ty].cost;
    if(s.roads[k]!==undefined){n.roads=Object.assign({},s.roads);delete n.roads[k];}
    n.units.push({id:s.nextId,type:ty,owner:p,c:E.KC(k),r:E.KR(k),born:s.round});
    return n;
  }
  // ---- encoding: planes [C][H][W] (uint8), board turned so the player to move has its base at the bottom ----
  function dims(){const {cols,rows}=E.dims();return {W:cols,H:(rows+1)/2,cols,rows};}
  function canon(c,r,p,D){ if(p===1){c=D.cols-1-c;r=D.rows+1-r;} return [c,Math.floor(r/2)]; }
  const TERR=['plain','forest','hill','river','bridge','mine','rock'];
  function encode(s){
    const D=dims(), {W,H}=D, HW=H*W, x=new Uint8Array(C*HW), p=toMove(s), e=1-p;
    const set=(ch,c,r,v)=>{const [X,Y]=canon(c,r,p,D);x[ch*HW+Y*W+X]=v;};
    const res=E.resolve(s);
    const depP=E.deployable(s,p,'militia'), depE=E.deployable(s,e,'militia');
    const KB=E.BASE;
    for(const cell of E.CELLS){
      const {c,r,k}=cell;
      set(0,c,r,1); set(1,c,r,c%2);
      set(2+Math.max(0,TERR.indexOf(E.TER(k))),c,r,1);
      if(k===KB[p]) set(9,c,r,1); if(k===KB[e]) set(10,c,r,1);
      if(res.covers[p].has(k)) set(65,c,r,1); if(res.covers[e].has(k)) set(66,c,r,1);
      if(depP.has(k)) set(67,c,r,1); if(depE.has(k)) set(68,c,r,1);
      const rd=s.roads[k]; if(rd===p) set(69,c,r,1); else if(rd===e) set(70,c,r,1);
      set(73,c,r,Math.min(255,E.distK(k,KB[p]))); set(74,c,r,Math.min(255,E.distK(k,KB[e])));
    }
    for(const pr of s.pending) set(pr.owner===p?71:72,pr.c,pr.r,1);
    for(const u of s.units){
      const st=res.stats.get(u.id), d=U[u.type], mine=u.owner===p;
      set((mine?11:34)+E.TIDX[u.type],u.c,u.r,1);
      set(57,u.c,u.r,st.hp); set(58,u.c,u.r,st.atk); set(59,u.c,u.r,st.armor);
      if(st.isolated&&!d.drop) set(60,u.c,u.r,1); if(st.noAtk) set(61,u.c,u.r,1);
      if(d.revive&&!u.revived) set(62,u.c,u.r,1);
      set(63,u.c,u.r,Math.min(20,res.dmg.get(u.id)||0)); if(res.dead.has(u.id)) set(64,u.c,u.r,1);
    }
    const nr=s.round+1, inc=q=>E.income(nr)+E.mines(s,q), uniq=q=>s.units.some(u=>u.owner===q&&U[u.type].unique)?1:0;
    const glob=[[75,s.bank[p]],[76,s.bank[e]],[77,Math.max(0,s.hp[p])],[78,Math.max(0,s.hp[e])],[79,Math.min(40,s.round)],
      [80,s.round>=FINALE?1:0],[81,s.mover?0:1],[82,inc(p)],[83,inc(e)],[84,uniq(p)],[85,uniq(e)],[86,s.placed]];
    for(const cell of E.CELLS) for(const [ch,v] of glob) set(ch,cell.c,cell.r,v);
    return {x,H,W};
  }
  // action <-> index in the policy output [NT][H][W] (+1 for END), in the mover's turned frame
  function actIndex(s,a){
    if(a===END) return -1;
    const D=dims(), t=Math.floor(a/10000), k=a%10000, [X,Y]=canon(E.KC(k),E.KR(k),toMove(s),D);
    return t*D.H*D.W+Y*D.W+X;
  }
  function result(s,p){ return !s.over?0:(s.winner<0?0:(s.winner===p?1:-1)); }
  return {newState,legal,apply,encode,actIndex,toMove,result,dims,END,C,NT};
};
module.exports.C=C; module.exports.END=END;
