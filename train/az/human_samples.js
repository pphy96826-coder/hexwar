// Turns the player's game records (train/az/games/*.json) into training positions, in selfplay.js's 'D' format:
// the winner's and loser's positions get the real result as value target; the human side's positions also get
// the human's move as policy target (policy weight 1); the AI side's only a value target (policy weight 0).
// Records that no longer replay legally under the current rules are skipped.
//   node train/az/human_samples.js <out.bin>
const path=require('path'), fs=require('fs');
const E=require('../engine')(path.join(__dirname,'..','..','hexwar.html'));
const G=require('./game')(E);
const dir=path.join(__dirname,'games'), out=[], meta=[]; let used=0;
for(const f of fs.readdirSync(dir).filter(f=>f.endsWith('.json'))){
  const R=JSON.parse(fs.readFileSync(path.join(dir,f)));
  if(!R.moves||R.winner===undefined||!R.aiSides) { console.error('skip (format)',f); continue; }
  E.loadMap(R.mapDef||R.map||'valley');
  const human=R.aiSides.map(a=>!a);
  let s=G.newState(), ok=true; const rec=[];
  const push=(s,act,isHuman)=>{
    const L=G.legal(s), idx=L.map(x=>{const j=G.actIndex(s,x);return j<0?65535:j;}), e=G.encode(s);
    rec.push({x:e.x,H:e.H,W:e.W,p:G.toMove(s),idx,pi:isHuman?L.map(x=>x===act?1:0):L.map(()=>0),pw:isHuman?1:0});
  };
  for(let round=1;ok&&!s.over&&round<=R.round;round++){
    for(let half=0;half<2&&!s.over;half++){
      const p=G.toMove(s), hm=human[p];
      if(!hm) push(s,null,false);   // AI side: one value-only position per turn
      for(const [r,pl,ty,c,rr] of R.moves.filter(m=>m[0]===round&&m[1]===p)){
        const a=E.TIDX[ty]*10000+c*100+rr;
        if(!E.U[ty]||!G.legal(s).includes(a)){ ok=false; break; }
        if(hm) push(s,a,true);
        s=G.apply(s,a);
      }
      if(!ok) break;
      if(hm) push(s,G.END,true);
      s=G.apply(s,G.END);
    }
  }
  if(!ok||!s.over||s.winner!==R.winner){ console.error('skip (does not replay under current rules)',f); continue; }
  used++;
  for(const sm of rec){ meta.push({H:sm.H,W:sm.W,z:s.winner<0?0:(s.winner===sm.p?1:-1),idx:sm.idx,pi:sm.pi,pw:sm.pw}); out.push(Buffer.from(sm.x)); }
  console.error('use',f,R.map||'valley','rounds',s.round,'positions',rec.length);
}
const js=Buffer.from(JSON.stringify(meta)), h=Buffer.alloc(4); h.writeUInt32LE(js.length,0);
fs.writeFileSync(process.argv[2],Buffer.concat([h,js,...out]));
console.error('games used',used,'positions',meta.length);
