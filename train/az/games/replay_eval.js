// Replays a game record and writes, for every turn start, the encoded position (for value checks in Python).
const path=require('path'), fs=require('fs');
const E=require('../../engine')(path.join(__dirname,'..','..','..','hexwar.html'));
const G=require('../web/gameweb.js')(E);
const R=JSON.parse(fs.readFileSync(process.argv[2])); E.loadMap(R.map);
let s=G.newState(); const out=[], meta=[];
for(let round=1;!s.over&&round<=R.round;round++){
  for(let half=0;half<2&&!s.over;half++){
    const p=G.toMove(s);
    const e=G.encode(s); out.push(Buffer.from(e.x)); 
    const res=E.resolve(s);
    meta.push({round:s.round,p,hp:s.hp.slice(),bank:s.bank.slice(),H:e.H,W:e.W,units:s.units.length});
    for(const [r,pl,ty,c,rr] of R.moves.filter(m=>m[0]===round&&m[1]===p)){
      const a=E.TIDX[ty]*10000+c*100+rr;
      if(!G.legal(s).includes(a)) console.error('illegal',round,pl,ty,c,rr);
      s=G.apply(s,a);
    }
    s=G.apply(s,G.END);
  }
}
console.error('final hp',s.hp,'over',s.over,'winner',s.winner);
fs.writeFileSync(process.argv[3],Buffer.concat(out)); fs.writeFileSync(process.argv[3]+'.json',JSON.stringify(meta));
